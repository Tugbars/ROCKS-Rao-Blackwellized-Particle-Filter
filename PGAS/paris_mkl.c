/**
 * @file paris_mkl.c
 * @brief Standalone MKL-optimized PARIS backward smoother
 *
 * Key optimizations:
 *   1. N_padded stride eliminates SIMD tail masking
 *   2. Pre-allocated per-thread RNG streams (no vslNewStream in hot path)
 *   3. Rank-1 log_trans column access pattern
 *   4. Rank-1 AR(1) optimization: precompute phi*h and mu_k*(1-phi)
 *   5. VML vsExp for batch exponential
 *   6. CBLAS for vectorized max/sum/scale
 *   7. SIMD CDF scan replaces binary search
 *   8. -INFINITY padding for clean exp results
 */

#include "paris_mkl.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

/* MKL headers */
#include <mkl.h>
#include <mkl_vml.h>
#include <mkl_vsl.h>

/* SIMD intrinsics */
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <immintrin.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

/*═══════════════════════════════════════════════════════════════════════════════
 * CONSTANTS
 *═══════════════════════════════════════════════════════════════════════════════*/

#define EPS 1e-10f
#define NEG_INF (-HUGE_VALF)

/* Temporal tiling: process in blocks to fit L2 cache (~256KB)
 * For N=128, each timestep uses ~3KB (h, log_w, regimes)
 * 64 timesteps × 3KB = 192KB fits comfortably in L2 */
#define PARIS_TILE_SIZE 64

/*═══════════════════════════════════════════════════════════════════════════════
 * AVX2 OPTIMIZED INNER KERNEL
 * - Vectorized gather for transition lookups
 * - 4x unrolling for ILP saturation (32 particles per iteration)
 *═══════════════════════════════════════════════════════════════════════════════*/

#if defined(__AVX2__) || defined(_MSC_VER)

/**
 * Compute backward log-weights using AVX2 gather + unrolled FMA
 * Processes 8 particles per vector, 4 vectors per unroll = 32 particles
 */
static inline void compute_log_bw_avx2(
    float *restrict local_log_bw,
    const float *restrict h_t,
    const float *restrict log_w_t,
    const int *restrict regimes_t,
    const float *restrict log_trans_col,
    float mu_shift, float phi, float h_next, float neg_half_inv_var,
    int N, int Np)
{
    const __m256 v_mu_shift = _mm256_set1_ps(mu_shift);
    const __m256 v_phi = _mm256_set1_ps(phi);
    const __m256 v_h_next = _mm256_set1_ps(h_next);
    const __m256 v_neg_half = _mm256_set1_ps(neg_half_inv_var);

    int i = 0;

    /* Main loop: 32 particles per iteration (4 × 8-wide vectors) */
    for (; i + 31 < N; i += 32)
    {
        /* Load 4 blocks of 8 regime indices */
        __m256i v_reg0 = _mm256_loadu_si256((const __m256i *)&regimes_t[i]);
        __m256i v_reg1 = _mm256_loadu_si256((const __m256i *)&regimes_t[i + 8]);
        __m256i v_reg2 = _mm256_loadu_si256((const __m256i *)&regimes_t[i + 16]);
        __m256i v_reg3 = _mm256_loadu_si256((const __m256i *)&regimes_t[i + 24]);

        /* Gather transition probabilities (scale=4 for sizeof(float)) */
        __m256 v_lt0 = _mm256_i32gather_ps(log_trans_col, v_reg0, 4);
        __m256 v_lt1 = _mm256_i32gather_ps(log_trans_col, v_reg1, 4);
        __m256 v_lt2 = _mm256_i32gather_ps(log_trans_col, v_reg2, 4);
        __m256 v_lt3 = _mm256_i32gather_ps(log_trans_col, v_reg3, 4);

        /* Load h values */
        __m256 v_h0 = _mm256_loadu_ps(&h_t[i]);
        __m256 v_h1 = _mm256_loadu_ps(&h_t[i + 8]);
        __m256 v_h2 = _mm256_loadu_ps(&h_t[i + 16]);
        __m256 v_h3 = _mm256_loadu_ps(&h_t[i + 24]);

        /* Load log_w values */
        __m256 v_lw0 = _mm256_loadu_ps(&log_w_t[i]);
        __m256 v_lw1 = _mm256_loadu_ps(&log_w_t[i + 8]);
        __m256 v_lw2 = _mm256_loadu_ps(&log_w_t[i + 16]);
        __m256 v_lw3 = _mm256_loadu_ps(&log_w_t[i + 24]);

        /* Compute mean = mu_shift + phi * h (FMA for ILP) */
        __m256 v_mean0 = _mm256_fmadd_ps(v_phi, v_h0, v_mu_shift);
        __m256 v_mean1 = _mm256_fmadd_ps(v_phi, v_h1, v_mu_shift);
        __m256 v_mean2 = _mm256_fmadd_ps(v_phi, v_h2, v_mu_shift);
        __m256 v_mean3 = _mm256_fmadd_ps(v_phi, v_h3, v_mu_shift);

        /* Compute diff = h_next - mean */
        __m256 v_diff0 = _mm256_sub_ps(v_h_next, v_mean0);
        __m256 v_diff1 = _mm256_sub_ps(v_h_next, v_mean1);
        __m256 v_diff2 = _mm256_sub_ps(v_h_next, v_mean2);
        __m256 v_diff3 = _mm256_sub_ps(v_h_next, v_mean3);

        /* Compute log_h_trans = neg_half_inv_var * diff * diff */
        __m256 v_d2_0 = _mm256_mul_ps(v_diff0, v_diff0);
        __m256 v_d2_1 = _mm256_mul_ps(v_diff1, v_diff1);
        __m256 v_d2_2 = _mm256_mul_ps(v_diff2, v_diff2);
        __m256 v_d2_3 = _mm256_mul_ps(v_diff3, v_diff3);

        __m256 v_lht0 = _mm256_mul_ps(v_neg_half, v_d2_0);
        __m256 v_lht1 = _mm256_mul_ps(v_neg_half, v_d2_1);
        __m256 v_lht2 = _mm256_mul_ps(v_neg_half, v_d2_2);
        __m256 v_lht3 = _mm256_mul_ps(v_neg_half, v_d2_3);

        /* Compute result = log_w + log_trans + log_h_trans */
        __m256 v_sum0 = _mm256_add_ps(v_lw0, v_lt0);
        __m256 v_sum1 = _mm256_add_ps(v_lw1, v_lt1);
        __m256 v_sum2 = _mm256_add_ps(v_lw2, v_lt2);
        __m256 v_sum3 = _mm256_add_ps(v_lw3, v_lt3);

        v_sum0 = _mm256_add_ps(v_sum0, v_lht0);
        v_sum1 = _mm256_add_ps(v_sum1, v_lht1);
        v_sum2 = _mm256_add_ps(v_sum2, v_lht2);
        v_sum3 = _mm256_add_ps(v_sum3, v_lht3);

        /* Store results */
        _mm256_storeu_ps(&local_log_bw[i], v_sum0);
        _mm256_storeu_ps(&local_log_bw[i + 8], v_sum1);
        _mm256_storeu_ps(&local_log_bw[i + 16], v_sum2);
        _mm256_storeu_ps(&local_log_bw[i + 24], v_sum3);
    }

    /* Tail: 8 particles per iteration */
    for (; i + 7 < N; i += 8)
    {
        __m256i v_reg = _mm256_loadu_si256((const __m256i *)&regimes_t[i]);
        __m256 v_lt = _mm256_i32gather_ps(log_trans_col, v_reg, 4);
        __m256 v_h = _mm256_loadu_ps(&h_t[i]);
        __m256 v_lw = _mm256_loadu_ps(&log_w_t[i]);

        __m256 v_mean = _mm256_fmadd_ps(v_phi, v_h, v_mu_shift);
        __m256 v_diff = _mm256_sub_ps(v_h_next, v_mean);
        __m256 v_d2 = _mm256_mul_ps(v_diff, v_diff);
        __m256 v_lht = _mm256_mul_ps(v_neg_half, v_d2);

        __m256 v_sum = _mm256_add_ps(v_lw, v_lt);
        v_sum = _mm256_add_ps(v_sum, v_lht);

        _mm256_storeu_ps(&local_log_bw[i], v_sum);
    }

    /* Scalar tail */
    for (; i < N; i++)
    {
        int regime_i = regimes_t[i];
        float log_trans = log_trans_col[regime_i];
        float mean = mu_shift + phi * h_t[i];
        float diff = h_next - mean;
        float log_h_trans = neg_half_inv_var * diff * diff;
        local_log_bw[i] = log_w_t[i] + log_trans + log_h_trans;
    }

    /* Padding */
    for (; i < Np; i++)
    {
        local_log_bw[i] = NEG_INF;
    }
}

#endif /* __AVX2__ */

/*═══════════════════════════════════════════════════════════════════════════════
 * AVX-512 OPTIMIZED INNER KERNEL (AMD EPYC Zen4+, Intel Skylake-X+)
 * - 16 floats per vector (2x AVX2)
 * - Faster gather (especially on Zen4)
 * - 4x unrolling = 64 particles per iteration
 *═══════════════════════════════════════════════════════════════════════════════*/

#if defined(__AVX512F__) && defined(__AVX512DQ__)

/**
 * Compute backward log-weights using AVX-512 gather + unrolled FMA
 * Processes 16 particles per vector, 4 vectors per unroll = 64 particles
 */
static inline void compute_log_bw_avx512(
    float *restrict local_log_bw,
    const float *restrict h_t,
    const float *restrict log_w_t,
    const int *restrict regimes_t,
    const float *restrict log_trans_col,
    float mu_shift, float phi, float h_next, float neg_half_inv_var,
    int N, int Np)
{
    const __m512 v_mu_shift = _mm512_set1_ps(mu_shift);
    const __m512 v_phi = _mm512_set1_ps(phi);
    const __m512 v_h_next = _mm512_set1_ps(h_next);
    const __m512 v_neg_half = _mm512_set1_ps(neg_half_inv_var);
    const __m512 v_neg_inf = _mm512_set1_ps(-HUGE_VALF);

    int i = 0;

    /* Main loop: 64 particles per iteration (4 × 16-wide vectors) */
    for (; i + 63 < N; i += 64)
    {
        /* Load 4 blocks of 16 regime indices */
        __m512i v_reg0 = _mm512_loadu_si512((const __m512i *)&regimes_t[i]);
        __m512i v_reg1 = _mm512_loadu_si512((const __m512i *)&regimes_t[i + 16]);
        __m512i v_reg2 = _mm512_loadu_si512((const __m512i *)&regimes_t[i + 32]);
        __m512i v_reg3 = _mm512_loadu_si512((const __m512i *)&regimes_t[i + 48]);

        /* Gather transition probabilities (scale=4 for sizeof(float))
         * AVX-512 gather is significantly faster than AVX2, especially on Zen4 */
        __m512 v_lt0 = _mm512_i32gather_ps(v_reg0, log_trans_col, 4);
        __m512 v_lt1 = _mm512_i32gather_ps(v_reg1, log_trans_col, 4);
        __m512 v_lt2 = _mm512_i32gather_ps(v_reg2, log_trans_col, 4);
        __m512 v_lt3 = _mm512_i32gather_ps(v_reg3, log_trans_col, 4);

        /* Load h values */
        __m512 v_h0 = _mm512_loadu_ps(&h_t[i]);
        __m512 v_h1 = _mm512_loadu_ps(&h_t[i + 16]);
        __m512 v_h2 = _mm512_loadu_ps(&h_t[i + 32]);
        __m512 v_h3 = _mm512_loadu_ps(&h_t[i + 48]);

        /* Load log_w values */
        __m512 v_lw0 = _mm512_loadu_ps(&log_w_t[i]);
        __m512 v_lw1 = _mm512_loadu_ps(&log_w_t[i + 16]);
        __m512 v_lw2 = _mm512_loadu_ps(&log_w_t[i + 32]);
        __m512 v_lw3 = _mm512_loadu_ps(&log_w_t[i + 48]);

        /* Compute mean = mu_shift + phi * h (FMA for ILP) */
        __m512 v_mean0 = _mm512_fmadd_ps(v_phi, v_h0, v_mu_shift);
        __m512 v_mean1 = _mm512_fmadd_ps(v_phi, v_h1, v_mu_shift);
        __m512 v_mean2 = _mm512_fmadd_ps(v_phi, v_h2, v_mu_shift);
        __m512 v_mean3 = _mm512_fmadd_ps(v_phi, v_h3, v_mu_shift);

        /* Compute diff = h_next - mean */
        __m512 v_diff0 = _mm512_sub_ps(v_h_next, v_mean0);
        __m512 v_diff1 = _mm512_sub_ps(v_h_next, v_mean1);
        __m512 v_diff2 = _mm512_sub_ps(v_h_next, v_mean2);
        __m512 v_diff3 = _mm512_sub_ps(v_h_next, v_mean3);

        /* Compute log_h_trans = neg_half_inv_var * diff * diff */
        __m512 v_d2_0 = _mm512_mul_ps(v_diff0, v_diff0);
        __m512 v_d2_1 = _mm512_mul_ps(v_diff1, v_diff1);
        __m512 v_d2_2 = _mm512_mul_ps(v_diff2, v_diff2);
        __m512 v_d2_3 = _mm512_mul_ps(v_diff3, v_diff3);

        __m512 v_lht0 = _mm512_mul_ps(v_neg_half, v_d2_0);
        __m512 v_lht1 = _mm512_mul_ps(v_neg_half, v_d2_1);
        __m512 v_lht2 = _mm512_mul_ps(v_neg_half, v_d2_2);
        __m512 v_lht3 = _mm512_mul_ps(v_neg_half, v_d2_3);

        /* Compute result = log_w + log_trans + log_h_trans */
        __m512 v_sum0 = _mm512_add_ps(v_lw0, v_lt0);
        __m512 v_sum1 = _mm512_add_ps(v_lw1, v_lt1);
        __m512 v_sum2 = _mm512_add_ps(v_lw2, v_lt2);
        __m512 v_sum3 = _mm512_add_ps(v_lw3, v_lt3);

        v_sum0 = _mm512_add_ps(v_sum0, v_lht0);
        v_sum1 = _mm512_add_ps(v_sum1, v_lht1);
        v_sum2 = _mm512_add_ps(v_sum2, v_lht2);
        v_sum3 = _mm512_add_ps(v_sum3, v_lht3);

        /* Store results */
        _mm512_storeu_ps(&local_log_bw[i], v_sum0);
        _mm512_storeu_ps(&local_log_bw[i + 16], v_sum1);
        _mm512_storeu_ps(&local_log_bw[i + 32], v_sum2);
        _mm512_storeu_ps(&local_log_bw[i + 48], v_sum3);
    }

    /* Tail: 16 particles per iteration */
    for (; i + 15 < N; i += 16)
    {
        __m512i v_reg = _mm512_loadu_si512((const __m512i *)&regimes_t[i]);
        __m512 v_lt = _mm512_i32gather_ps(v_reg, log_trans_col, 4);
        __m512 v_h = _mm512_loadu_ps(&h_t[i]);
        __m512 v_lw = _mm512_loadu_ps(&log_w_t[i]);

        __m512 v_mean = _mm512_fmadd_ps(v_phi, v_h, v_mu_shift);
        __m512 v_diff = _mm512_sub_ps(v_h_next, v_mean);
        __m512 v_d2 = _mm512_mul_ps(v_diff, v_diff);
        __m512 v_lht = _mm512_mul_ps(v_neg_half, v_d2);

        __m512 v_sum = _mm512_add_ps(v_lw, v_lt);
        v_sum = _mm512_add_ps(v_sum, v_lht);

        _mm512_storeu_ps(&local_log_bw[i], v_sum);
    }

    /* Masked tail for remaining elements (AVX-512 mask registers) */
    if (i < N)
    {
        __mmask16 mask = (__mmask16)((1u << (N - i)) - 1);

        __m512i v_reg = _mm512_maskz_loadu_epi32(mask, &regimes_t[i]);
        __m512 v_lt = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), mask, v_reg, log_trans_col, 4);
        __m512 v_h = _mm512_maskz_loadu_ps(mask, &h_t[i]);
        __m512 v_lw = _mm512_maskz_loadu_ps(mask, &log_w_t[i]);

        __m512 v_mean = _mm512_fmadd_ps(v_phi, v_h, v_mu_shift);
        __m512 v_diff = _mm512_sub_ps(v_h_next, v_mean);
        __m512 v_d2 = _mm512_mul_ps(v_diff, v_diff);
        __m512 v_lht = _mm512_mul_ps(v_neg_half, v_d2);

        __m512 v_sum = _mm512_add_ps(v_lw, v_lt);
        v_sum = _mm512_add_ps(v_sum, v_lht);

        _mm512_mask_storeu_ps(&local_log_bw[i], mask, v_sum);
        i = N;
    }

    /* Padding with masked store */
    if (i < Np)
    {
        int remaining = Np - i;
        while (remaining >= 16)
        {
            _mm512_storeu_ps(&local_log_bw[i], v_neg_inf);
            i += 16;
            remaining -= 16;
        }
        if (remaining > 0)
        {
            __mmask16 mask = (__mmask16)((1u << remaining) - 1);
            _mm512_mask_storeu_ps(&local_log_bw[i], mask, v_neg_inf);
        }
    }
}

#endif /* __AVX512F__ && __AVX512DQ__ */

/*═══════════════════════════════════════════════════════════════════════════════
 * ALLOCATION
 *═══════════════════════════════════════════════════════════════════════════════*/

PARISMKLState *paris_mkl_alloc(int N, int T, int K, uint32_t seed)
{
    PARISMKLState *state = (PARISMKLState *)mkl_calloc(1, sizeof(PARISMKLState), PARIS_MKL_ALIGN);
    if (!state)
        return NULL;

    /* Pad N to multiple of 16 for full SIMD lanes */
    int N_padded = PARIS_MKL_PAD_N(N);

    state->N = N;
    state->N_padded = N_padded;
    state->T = T;
    state->K = K;

    /* Allocate with N_padded stride */
    size_t NT_padded = (size_t)N_padded * T;

    state->regimes = (int *)mkl_malloc(NT_padded * sizeof(int), PARIS_MKL_ALIGN);
    state->h = (float *)mkl_malloc(NT_padded * sizeof(float), PARIS_MKL_ALIGN);
    state->log_weights = (float *)mkl_malloc(NT_padded * sizeof(float), PARIS_MKL_ALIGN);
    state->ancestors = (int *)mkl_malloc(NT_padded * sizeof(int), PARIS_MKL_ALIGN);
    state->smoothed = (int *)mkl_malloc(NT_padded * sizeof(int), PARIS_MKL_ALIGN);

    /* Zero-initialize (padding included) */
    memset(state->regimes, 0, NT_padded * sizeof(int));
    memset(state->h, 0, NT_padded * sizeof(float));
    memset(state->log_weights, 0, NT_padded * sizeof(float));
    memset(state->ancestors, 0, NT_padded * sizeof(int));
    memset(state->smoothed, 0, NT_padded * sizeof(int));

    /* Workspace buffers */
    state->ws_log_bw = (float *)mkl_malloc(N_padded * sizeof(float), PARIS_MKL_ALIGN);
    state->ws_bw = (float *)mkl_malloc(N_padded * sizeof(float), PARIS_MKL_ALIGN);
    state->ws_workspace = (float *)mkl_malloc(N_padded * sizeof(float), PARIS_MKL_ALIGN);
    state->ws_cumsum = (float *)mkl_malloc(N_padded * sizeof(float), PARIS_MKL_ALIGN);
    state->ws_scaled_h = (float *)mkl_malloc(N_padded * sizeof(float), PARIS_MKL_ALIGN);

/* Pre-allocate per-thread workspaces (avoid malloc in hot path + false sharing)
 * Each thread needs: log_bw, bw, workspace, cumsum = 4 * N_padded floats
 * Add 128-byte (32 float) padding between threads to prevent false sharing */
#ifdef _OPENMP
    int max_threads = omp_get_max_threads();
#else
    int max_threads = 1;
#endif
    state->n_thread_streams = (max_threads < PARIS_MKL_MAX_THREADS) ? max_threads : PARIS_MKL_MAX_THREADS;

    /* Stride per thread: 4 buffers + 32-float padding, rounded up to 64-byte alignment */
    state->thread_ws_stride = ((4 * N_padded + 32 + 15) & ~15);
    size_t total_ws_size = (size_t)state->n_thread_streams * state->thread_ws_stride * sizeof(float);
    state->thread_ws = (float *)mkl_malloc(total_ws_size, PARIS_MKL_ALIGN);
    memset(state->thread_ws, 0, total_ws_size);

    /* Main RNG stream */
    vslNewStream((VSLStreamStatePtr *)&state->rng_stream, VSL_BRNG_SFMT19937, seed);

    /* Pre-allocate per-thread RNG streams */
    for (int i = 0; i < state->n_thread_streams; i++)
    {
        vslNewStream((VSLStreamStatePtr *)&state->thread_rng_streams[i],
                     VSL_BRNG_SFMT19937, seed + 1000 * (i + 1));
    }

    /* Initialize model defaults */
    state->model.K = K;
    float unif = 1.0f / K;
    for (int i = 0; i < K; i++)
    {
        for (int j = 0; j < K; j++)
        {
            state->model.trans[i * K + j] = unif;
            state->model.log_trans[i * K + j] = logf(unif);
        }
        state->model.mu_vol[i] = -1.0f + 0.5f * i;
    }
    state->model.phi = 0.97f;
    state->model.sigma_h = 0.15f;
    state->model.inv_sigma_h_sq = 1.0f / (0.15f * 0.15f);

    return state;
}

void paris_mkl_free(PARISMKLState *state)
{
    if (!state)
        return;

    /* Free main RNG */
    if (state->rng_stream)
    {
        vslDeleteStream((VSLStreamStatePtr *)&state->rng_stream);
    }

    /* Free per-thread RNG streams */
    for (int i = 0; i < state->n_thread_streams; i++)
    {
        if (state->thread_rng_streams[i])
        {
            vslDeleteStream((VSLStreamStatePtr *)&state->thread_rng_streams[i]);
        }
    }

    mkl_free(state->regimes);
    mkl_free(state->h);
    mkl_free(state->log_weights);
    mkl_free(state->ancestors);
    mkl_free(state->smoothed);
    mkl_free(state->ws_log_bw);
    mkl_free(state->ws_bw);
    mkl_free(state->ws_workspace);
    mkl_free(state->ws_cumsum);
    mkl_free(state->ws_scaled_h);
    mkl_free(state->thread_ws);
    mkl_free(state);
}

/*═══════════════════════════════════════════════════════════════════════════════
 * CONFIGURATION
 *═══════════════════════════════════════════════════════════════════════════════*/

void paris_mkl_set_model(PARISMKLState *state,
                         const double *trans,
                         const double *mu_vol,
                         double phi,
                         double sigma_h)
{
    if (!state)
        return;

    int K = state->K;
    state->model.K = K;

    for (int i = 0; i < K * K; i++)
    {
        state->model.trans[i] = (float)trans[i];
        state->model.log_trans[i] = logf((float)trans[i] + EPS);
    }

    float phi_f = (float)phi;
    float one_minus_phi = 1.0f - phi_f;

    for (int i = 0; i < K; i++)
    {
        state->model.mu_vol[i] = (float)mu_vol[i];
        /* Rank-1 AR(1) precomputation: mu_k * (1 - phi) */
        state->model.mu_shifts[i] = (float)mu_vol[i] * one_minus_phi;
    }

    state->model.phi = phi_f;
    state->model.sigma_h = (float)sigma_h;
    state->model.inv_sigma_h_sq = 1.0f / ((float)sigma_h * (float)sigma_h);
}

void paris_mkl_load_particles(PARISMKLState *state,
                              const int *regimes,
                              const double *h,
                              const double *weights,
                              const int *ancestors,
                              int T)
{
    if (!state)
        return;

    state->T = T;
    const int N = state->N;
    const int Np = state->N_padded;

    /* Copy with stride conversion: input is [T×N], output is [T×Np] */
    for (int t = 0; t < T; t++)
    {
        /* Copy valid particles */
        for (int n = 0; n < N; n++)
        {
            int src_idx = t * N + n;
            int dst_idx = t * Np + n;

            state->regimes[dst_idx] = regimes[src_idx];
            state->h[dst_idx] = (float)h[src_idx];
            state->log_weights[dst_idx] = logf((float)weights[src_idx] + EPS);
            state->ancestors[dst_idx] = ancestors[src_idx];
        }

        /* Zero padding */
        for (int n = N; n < Np; n++)
        {
            int dst_idx = t * Np + n;
            state->regimes[dst_idx] = 0;
            state->h[dst_idx] = 0.0f;
            state->log_weights[dst_idx] = NEG_INF;
            state->ancestors[dst_idx] = 0;
        }
    }
}

/*═══════════════════════════════════════════════════════════════════════════════
 * SAMPLING UTILITIES
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * SIMD CDF Scan - Replaces binary search with vectorized cumulative sum
 *
 * Instead of O(log N) branchy binary search, we:
 *   1. Compute cumulative sum 8 elements at a time with AVX
 *   2. Compare against uniform random in parallel
 *   3. Find first element where cumsum >= u
 *
 * Benefits:
 *   - No branch mispredictions
 *   - Streaming memory access pattern
 *   - ~2x faster for N < 256
 */
static inline int sample_categorical_simd(const float *weights, int N,
                                          float u)
{
#if defined(__AVX2__) || defined(_MSC_VER)
    __m256 v_u = _mm256_set1_ps(u);
    float running_sum = 0.0f;

    /* Process 8 weights at a time */
    int i = 0;
    for (; i + 8 <= N; i += 8)
    {
        __m256 v_w = _mm256_loadu_ps(&weights[i]);

        /* Compute prefix sum within the 8-element block */
        /* Step 1: shift left by 1 float and add */
        __m256 v_shift1 = _mm256_castsi256_ps(
            _mm256_slli_si256(_mm256_castps_si256(v_w), 4));
        __m256 v_sum1 = _mm256_add_ps(v_w, v_shift1);

        /* Step 2: shift left by 2 floats and add */
        __m256 v_shift2 = _mm256_castsi256_ps(
            _mm256_slli_si256(_mm256_castps_si256(v_sum1), 8));
        __m256 v_sum2 = _mm256_add_ps(v_sum1, v_shift2);

        /* Step 3: Handle cross-lane (AVX lanes are 128-bit) */
        __m256 v_upper = _mm256_permute2f128_ps(v_sum2, v_sum2, 0x00);
        __m256 v_lane_sum = _mm256_shuffle_ps(v_upper, v_upper, 0xFF);
        __m256 v_add_lane = _mm256_blend_ps(_mm256_setzero_ps(), v_lane_sum, 0xF0);
        v_sum2 = _mm256_add_ps(v_sum2, v_add_lane);

        /* Add running sum from previous blocks */
        __m256 v_total = _mm256_add_ps(v_sum2, _mm256_set1_ps(running_sum));

        /* Compare cumsum >= u */
        __m256 v_cmp = _mm256_cmp_ps(v_total, v_u, _CMP_GE_OQ);
        int mask = _mm256_movemask_ps(v_cmp);

        if (mask != 0)
        {
/* Found first element where cumsum >= u */
#ifdef _MSC_VER
            unsigned long idx;
            _BitScanForward(&idx, (unsigned long)mask);
            return i + (int)idx;
#else
            return i + __builtin_ctz(mask);
#endif
        }

        /* Extract last element of v_total for running sum */
        float block_sum[8];
        _mm256_storeu_ps(block_sum, v_total);
        running_sum = block_sum[7];
    }

    /* Handle remaining elements (scalar) */
    for (; i < N; i++)
    {
        running_sum += weights[i];
        if (running_sum >= u)
        {
            return i;
        }
    }

    return N - 1;
#else
    /* Scalar fallback */
    float cumsum = 0.0f;
    for (int i = 0; i < N; i++)
    {
        cumsum += weights[i];
        if (cumsum >= u)
        {
            return i;
        }
    }
    return N - 1;
#endif
}

/**
 * Sample single index from categorical distribution
 * Uses SIMD CDF scan instead of binary search
 */
static inline int sample_categorical(const float *weights, int N,
                                     float *cumsum, VSLStreamStatePtr stream)
{
    /* Generate uniform */
    float u;
    vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, 1, &u, 0.0f, 1.0f);

    /* SIMD CDF scan (cumsum parameter unused but kept for API compatibility) */
    (void)cumsum;
    return sample_categorical_simd(weights, N, u);
}

/*═══════════════════════════════════════════════════════════════════════════════
 * LOG-SUM-EXP NORMALIZATION
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * Compute normalized weights from log-weights
 * Uses N_padded for SIMD, but only sums/normalizes first N
 */
static void logsumexp_normalize(const float *log_w, int N, int Np,
                                float *w, float *workspace)
{
    /* Find max (only valid N) */
    int max_idx = cblas_isamax(N, log_w, 1);
    float max_val = log_w[max_idx];
    float neg_max = -max_val;
    int i; /* MSVC OpenMP requires loop var declared outside */

/* Subtract max (full Np for SIMD) */
#ifndef _MSC_VER
#pragma omp simd
#endif
    for (i = 0; i < Np; i++)
    {
        workspace[i] = log_w[i] + neg_max;
    }

    /* Set padding to -inf */
    for (i = N; i < Np; i++)
    {
        workspace[i] = NEG_INF;
    }

    /* Exp using MKL VML */
    vsExp(Np, workspace, w);

    /* Sum (only valid N) */
    float sum = cblas_sasum(N, w, 1);

    /* Normalize (only valid N) */
    float inv_sum = 1.0f / sum;
    cblas_sscal(N, inv_sum, w, 1);
}

/*═══════════════════════════════════════════════════════════════════════════════
 * BACKWARD SMOOTHING
 *═══════════════════════════════════════════════════════════════════════════════*/

/* Threshold for switching between small/large optimization paths
 * Small path: Skip VML mode switch, simpler loops (less overhead)
 * Large path: Full optimizations (VML_EP, prefetch, etc.)
 *
 * Tuned for HFT SV-HMM: Typical T=128-200, N=64-128, K=4
 */
#define PARIS_SMALL_WORKLOAD_THRESHOLD 8192 /* T*N < 8192 → small path */

void paris_mkl_backward_smooth(PARISMKLState *state)
{
    if (!state || state->T < 2)
        return;

    const int N = state->N;
    const int Np = state->N_padded;
    const int T = state->T;
    const int K = state->model.K;
    const PARISMKLModel *m = &state->model;
    const size_t workload = (size_t)T * N;

    /* Precompute constants (both paths) */
    const float phi = m->phi;
    const float neg_half_inv_var = -0.5f * m->inv_sigma_h_sq;

    /* Pre-transpose log_trans (cheap, always beneficial) */
    float log_trans_T[PARIS_MKL_MAX_REGIMES * PARIS_MKL_MAX_REGIMES];
    for (int i = 0; i < K; i++)
    {
        for (int j = 0; j < K; j++)
        {
            log_trans_T[j * K + i] = m->log_trans[i * K + j];
        }
    }

    /* Initialize at final time */
    for (int n = 0; n < N; n++)
    {
        state->smoothed[(T - 1) * Np + n] = n;
    }

    /* ═══════════════════════════════════════════════════════════════════════
     * PATH SELECTION: Small vs Large workload
     * ═══════════════════════════════════════════════════════════════════════*/
    const int use_large_path = (workload >= PARIS_SMALL_WORKLOAD_THRESHOLD);

#ifdef _OPENMP
    /* ═══════════════════════════════════════════════════════════════════════
     * PARALLEL BACKWARD PASS
     * ═══════════════════════════════════════════════════════════════════════*/

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        VSLStreamStatePtr local_stream = (VSLStreamStatePtr)state->thread_rng_streams[tid];

        /* VML Enhanced Performance mode - set per-thread (thread-local in modern MKL)
         * Only for large workloads; mode switch has fixed overhead (~1µs) */
        if (use_large_path)
        {
            vmlSetMode(VML_EP | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE);
        }

        /* Use pre-allocated workspaces */
        float *thread_base = &state->thread_ws[tid * state->thread_ws_stride];
        float *local_log_bw = thread_base;
        float *local_bw = thread_base + Np;
        float *local_workspace = thread_base + 2 * Np;
        float *local_cumsum = thread_base + 3 * Np;

        int t, n; /* MSVC OpenMP requires loop vars declared outside */

        /* ═══════════════════════════════════════════════════════════════════
         * TEMPORAL TILING: Process in blocks to maximize L2 cache reuse
         * For N=128, each timestep uses ~3KB. 64 timesteps = 192KB (fits L2)
         * ═══════════════════════════════════════════════════════════════════*/
        int tile_size = (use_large_path && T > PARIS_TILE_SIZE) ? PARIS_TILE_SIZE : T;

        for (int tile_end = T - 2; tile_end >= 0; tile_end -= tile_size)
        {
            int tile_start = (tile_end - tile_size + 1 > 0) ? (tile_end - tile_size + 1) : 0;

/* Prefetch entire tile into L2 */
#pragma omp single nowait
            {
                if (use_large_path && tile_start > 0)
                {
                    int prefetch_end = (tile_start - 1 > tile_start - tile_size) ? (tile_start - tile_size) : 0;
                    for (int pt = tile_start - 1; pt >= prefetch_end && pt >= 0; pt--)
                    {
                        _mm_prefetch((const char *)&state->h[pt * Np], _MM_HINT_T1);
                        _mm_prefetch((const char *)&state->log_weights[pt * Np], _MM_HINT_T1);
                    }
                }
            }

            for (t = tile_end; t >= tile_start; t--)
            {
                const float *h_t = &state->h[t * Np];
                const float *log_w_t = &state->log_weights[t * Np];
                const int *regimes_t = &state->regimes[t * Np];

#pragma omp for schedule(dynamic, 8)
                for (n = 0; n < N; n++)
                {
                    int idx_next = state->smoothed[(t + 1) * Np + n];
                    int regime_next = state->regimes[(t + 1) * Np + idx_next];
                    float h_next = state->h[(t + 1) * Np + idx_next];
                    float mu_shift = m->mu_shifts[regime_next];
                    const float *log_trans_col = &log_trans_T[regime_next * K];

#if defined(__AVX512F__) && defined(__AVX512DQ__)
                    /* Use AVX-512 optimized kernel (64 particles/iter) */
                    compute_log_bw_avx512(local_log_bw, h_t, log_w_t, regimes_t,
                                          log_trans_col, mu_shift, phi, h_next,
                                          neg_half_inv_var, N, Np);
#elif defined(__AVX2__) || defined(_MSC_VER)
                    /* Use AVX2 optimized kernel (32 particles/iter) */
                    compute_log_bw_avx2(local_log_bw, h_t, log_w_t, regimes_t,
                                        log_trans_col, mu_shift, phi, h_next,
                                        neg_half_inv_var, N, Np);
#else
                    /* Scalar fallback */
                    int i;
                    for (i = 0; i < N; i++)
                    {
                        int regime_i = regimes_t[i];
                        float log_trans = log_trans_col[regime_i];
                        float mean = mu_shift + phi * h_t[i];
                        float diff = h_next - mean;
                        float log_h_trans = neg_half_inv_var * diff * diff;
                        local_log_bw[i] = log_w_t[i] + log_trans + log_h_trans;
                    }
                    for (i = N; i < Np; i++)
                    {
                        local_log_bw[i] = NEG_INF;
                    }
#endif

                    logsumexp_normalize(local_log_bw, N, Np, local_bw, local_workspace);
                    state->smoothed[t * Np + n] = sample_categorical(local_bw, N, local_cumsum, local_stream);
                }
            }
        }

        /* Restore default VML mode per-thread */
        if (use_large_path)
        {
            vmlSetMode(VML_HA);
        }
    }

#else
    /* ═══════════════════════════════════════════════════════════════════════
     * SEQUENTIAL BACKWARD PASS
     * ═══════════════════════════════════════════════════════════════════════*/

    /* VML Enhanced Performance mode for large workloads */
    if (use_large_path)
    {
        vmlSetMode(VML_EP | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE);
    }

    VSLStreamStatePtr stream = (VSLStreamStatePtr)state->rng_stream;
    float *local_log_bw = state->ws_log_bw;
    float *local_bw = state->ws_bw;
    float *local_workspace = state->ws_workspace;
    float *local_cumsum = state->ws_cumsum;

    int t, n; /* MSVC requires loop vars declared outside */
    for (t = T - 2; t >= 0; t--)
    {
        const float *h_t = &state->h[t * Np];
        const float *log_w_t = &state->log_weights[t * Np];
        const int *regimes_t = &state->regimes[t * Np];

        /* Software prefetch - only for large workloads */
        if (use_large_path && t > 0)
        {
            _mm_prefetch((const char *)&state->h[(t - 1) * Np], _MM_HINT_T1);
            _mm_prefetch((const char *)&state->log_weights[(t - 1) * Np], _MM_HINT_T1);
        }

        for (n = 0; n < N; n++)
        {
            int idx_next = state->smoothed[(t + 1) * Np + n];
            int regime_next = state->regimes[(t + 1) * Np + idx_next];
            float h_next = state->h[(t + 1) * Np + idx_next];
            float mu_shift = m->mu_shifts[regime_next];
            const float *log_trans_col = &log_trans_T[regime_next * K];

#if defined(__AVX512F__) && defined(__AVX512DQ__)
            compute_log_bw_avx512(local_log_bw, h_t, log_w_t, regimes_t,
                                  log_trans_col, mu_shift, phi, h_next,
                                  neg_half_inv_var, N, Np);
#elif defined(__AVX2__) || defined(_MSC_VER)
            compute_log_bw_avx2(local_log_bw, h_t, log_w_t, regimes_t,
                                log_trans_col, mu_shift, phi, h_next,
                                neg_half_inv_var, N, Np);
#else
            int i;
            for (i = 0; i < N; i++)
            {
                int regime_i = regimes_t[i];
                float log_trans = log_trans_col[regime_i];
                float mean = mu_shift + phi * h_t[i];
                float diff = h_next - mean;
                float log_h_trans = neg_half_inv_var * diff * diff;
                local_log_bw[i] = log_w_t[i] + log_trans + log_h_trans;
            }
            for (i = N; i < Np; i++)
            {
                local_log_bw[i] = NEG_INF;
            }
#endif

            logsumexp_normalize(local_log_bw, N, Np, local_bw, local_workspace);
            state->smoothed[t * Np + n] = sample_categorical(local_bw, N, local_cumsum, stream);
        }
    }

    /* Restore default VML mode */
    if (use_large_path)
    {
        vmlSetMode(VML_HA);
    }
#endif
}

/*═══════════════════════════════════════════════════════════════════════════════
 * OUTPUT EXTRACTION
 *═══════════════════════════════════════════════════════════════════════════════*/

void paris_mkl_get_smoothed(const PARISMKLState *state, int t,
                            int *regimes, float *h)
{
    if (!state || t < 0 || t >= state->T)
        return;

    const int N = state->N;
    const int Np = state->N_padded;

    for (int n = 0; n < N; n++)
    {
        int idx = state->smoothed[t * Np + n];
        if (regimes)
            regimes[n] = state->regimes[t * Np + idx];
        if (h)
            h[n] = state->h[t * Np + idx];
    }
}

void paris_mkl_get_trajectory(const PARISMKLState *state, int n,
                              int *regimes, float *h)
{
    if (!state || n < 0 || n >= state->N)
        return;

    const int T = state->T;
    const int Np = state->N_padded;

    for (int t = 0; t < T; t++)
    {
        int idx = state->smoothed[t * Np + n];
        if (regimes)
            regimes[t] = state->regimes[t * Np + idx];
        if (h)
            h[t] = state->h[t * Np + idx];
    }
}

/*═══════════════════════════════════════════════════════════════════════════════
 * DIAGNOSTICS
 *═══════════════════════════════════════════════════════════════════════════════*/

void paris_mkl_print_info(const PARISMKLState *state)
{
    if (!state)
        return;

    printf("═══════════════════════════════════════════════════════════\n");
    printf("PARIS-MKL INFO\n");
    printf("═══════════════════════════════════════════════════════════\n");
    printf("Particles:          %d (padded: %d)\n", state->N, state->N_padded);
    printf("Time steps:         %d\n", state->T);
    printf("Regimes:            %d\n", state->K);
    printf("Thread RNG streams: %d\n", state->n_thread_streams);
    printf("Model:\n");
    printf("  phi:              %.4f\n", state->model.phi);
    printf("  sigma_h:          %.4f\n", state->model.sigma_h);
    printf("  mu_vol[0..%d]:    ", state->K - 1);
    for (int k = 0; k < state->K; k++)
    {
        printf("%.2f ", state->model.mu_vol[k]);
    }
    printf("\n");
    printf("═══════════════════════════════════════════════════════════\n");
}