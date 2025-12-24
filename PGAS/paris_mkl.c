/**
 * @file paris_mkl.c
 * @brief Standalone MKL-optimized PARIS backward smoother
 *
 * ALIGNED WITH RBPF: Uses per-regime sigma_vol[k] for AR process noise.
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

/* Temporal tiling: process in blocks to fit L2 cache (~256KB) */
#define PARIS_TILE_SIZE 64

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

#ifdef _OPENMP
    int max_threads = omp_get_max_threads();
#else
    int max_threads = 1;
#endif
    state->n_thread_streams = (max_threads < PARIS_MKL_MAX_THREADS) ? max_threads : PARIS_MKL_MAX_THREADS;

    /* Stride per thread: 4 buffers + padding */
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

    /* Initialize model defaults with per-regime sigma_vol */
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
        state->model.sigma_vol[i] = 0.15f;
        state->model.inv_sigma_vol_sq[i] = 1.0f / (0.15f * 0.15f);
        state->model.neg_half_inv_sigma_vol_sq[i] = -0.5f / (0.15f * 0.15f);
    }
    state->model.phi = 0.97f;

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
                         const double *sigma_vol,
                         double phi)
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

        /* Per-regime sigma_vol (aligned with RBPF) */
        float sv = (float)sigma_vol[i];
        state->model.sigma_vol[i] = sv;
        state->model.inv_sigma_vol_sq[i] = 1.0f / (sv * sv);
        state->model.neg_half_inv_sigma_vol_sq[i] = -0.5f / (sv * sv);
    }

    state->model.phi = phi_f;
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
 */
static inline int sample_categorical_simd(const float *weights, int N, float u)
{
#if defined(__AVX2__) || defined(_MSC_VER)
    __m256 v_u = _mm256_set1_ps(u);
    float running_sum = 0.0f;

    int i = 0;
    for (; i + 8 <= N; i += 8)
    {
        __m256 v_w = _mm256_loadu_ps(&weights[i]);

        /* Compute prefix sum within the 8-element block */
        __m256 v_shift1 = _mm256_castsi256_ps(
            _mm256_slli_si256(_mm256_castps_si256(v_w), 4));
        __m256 v_sum1 = _mm256_add_ps(v_w, v_shift1);

        __m256 v_shift2 = _mm256_castsi256_ps(
            _mm256_slli_si256(_mm256_castps_si256(v_sum1), 8));
        __m256 v_sum2 = _mm256_add_ps(v_sum1, v_shift2);

        __m256 v_upper = _mm256_permute2f128_ps(v_sum2, v_sum2, 0x00);
        __m256 v_lane_sum = _mm256_shuffle_ps(v_upper, v_upper, 0xFF);
        __m256 v_add_lane = _mm256_blend_ps(_mm256_setzero_ps(), v_lane_sum, 0xF0);
        v_sum2 = _mm256_add_ps(v_sum2, v_add_lane);

        __m256 v_total = _mm256_add_ps(v_sum2, _mm256_set1_ps(running_sum));

        __m256 v_cmp = _mm256_cmp_ps(v_total, v_u, _CMP_GE_OQ);
        int mask = _mm256_movemask_ps(v_cmp);

        if (mask != 0)
        {
#ifdef _MSC_VER
            unsigned long idx;
            _BitScanForward(&idx, (unsigned long)mask);
            return i + (int)idx;
#else
            return i + __builtin_ctz(mask);
#endif
        }

        float block_sum[8];
        _mm256_storeu_ps(block_sum, v_total);
        running_sum = block_sum[7];
    }

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

static inline int sample_categorical(const float *weights, int N,
                                     float *cumsum, VSLStreamStatePtr stream)
{
    float u;
    vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, 1, &u, 0.0f, 1.0f);
    (void)cumsum;
    return sample_categorical_simd(weights, N, u);
}

/*═══════════════════════════════════════════════════════════════════════════════
 * LOG-SUM-EXP NORMALIZATION
 *═══════════════════════════════════════════════════════════════════════════════*/

static void logsumexp_normalize(const float *log_w, int N, int Np,
                                float *w, float *workspace)
{
    int max_idx = cblas_isamax(N, log_w, 1);
    float max_val = log_w[max_idx];

    /* Guard against all -inf weights */
    if (!isfinite(max_val) || max_val < -1e30f)
    {
        float unif = 1.0f / (float)N;
        for (int k = 0; k < N; k++)
        {
            w[k] = unif;
        }
        return;
    }

    float neg_max = -max_val;
    int i;

#ifndef _MSC_VER
#pragma omp simd
#endif
    for (i = 0; i < Np; i++)
    {
        workspace[i] = log_w[i] + neg_max;
    }

    for (i = N; i < Np; i++)
    {
        workspace[i] = NEG_INF;
    }

    vsExp(Np, workspace, w);

    float sum = cblas_sasum(N, w, 1);

    /* Guard against collapsed cloud - fall back to uniform */
    if (sum < 1e-38f || !isfinite(sum))
    {
        float unif = 1.0f / (float)N;
        for (int k = 0; k < N; k++)
        {
            w[k] = unif;
        }
    }
    else
    {
        float inv_sum = 1.0f / sum;
        cblas_sscal(N, inv_sum, w, 1);
    }
}

/*═══════════════════════════════════════════════════════════════════════════════
 * BACKWARD SMOOTHING - SCALAR KERNEL WITH PER-REGIME SIGMA_VOL
 *
 * Uses sigma_vol[regime_next] for each backward transition.
 * This is the RBPF-aligned version where each regime has its own process noise.
 *═══════════════════════════════════════════════════════════════════════════════*/

static inline void compute_log_bw_scalar(
    float *restrict local_log_bw,
    const float *restrict h_t,
    const float *restrict log_w_t,
    const int *restrict regimes_t,
    const float *restrict log_trans_col,
    float mu_shift, float phi, float h_next, float neg_half_inv_var,
    int N, int Np)
{
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
}

#define PARIS_SMALL_WORKLOAD_THRESHOLD 8192

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

    const float phi = m->phi;

    /* Pre-transpose log_trans */
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

    const int use_large_path = (workload >= PARIS_SMALL_WORKLOAD_THRESHOLD);

#ifdef _OPENMP
#pragma omp parallel
    {
        int tid = omp_get_thread_num();

        /* Guard against tid exceeding allocated streams */
        if (tid >= state->n_thread_streams)
            tid = tid % state->n_thread_streams;

        VSLStreamStatePtr local_stream = (VSLStreamStatePtr)state->thread_rng_streams[tid];

        if (use_large_path)
        {
            vmlSetMode(VML_EP | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE);
        }

        float *thread_base = &state->thread_ws[tid * state->thread_ws_stride];
        float *local_log_bw = thread_base;
        float *local_bw = thread_base + Np;
        float *local_workspace = thread_base + 2 * Np;
        float *local_cumsum = thread_base + 3 * Np;

        int t, n;

        for (t = T - 2; t >= 0; t--)
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

                /* Per-regime sigma_vol (aligned with RBPF) */
                float neg_half_inv_var = m->neg_half_inv_sigma_vol_sq[regime_next];

                compute_log_bw_scalar(local_log_bw, h_t, log_w_t, regimes_t,
                                      log_trans_col, mu_shift, phi, h_next,
                                      neg_half_inv_var, N, Np);

                logsumexp_normalize(local_log_bw, N, Np, local_bw, local_workspace);
                state->smoothed[t * Np + n] = sample_categorical(local_bw, N, local_cumsum, local_stream);
            }
        }

        if (use_large_path)
        {
            vmlSetMode(VML_HA);
        }
    }

#else
    /* Sequential backward pass */
    if (use_large_path)
    {
        vmlSetMode(VML_EP | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE);
    }

    VSLStreamStatePtr stream = (VSLStreamStatePtr)state->rng_stream;
    float *local_log_bw = state->ws_log_bw;
    float *local_bw = state->ws_bw;
    float *local_workspace = state->ws_workspace;
    float *local_cumsum = state->ws_cumsum;

    int t, n;
    for (t = T - 2; t >= 0; t--)
    {
        const float *h_t = &state->h[t * Np];
        const float *log_w_t = &state->log_weights[t * Np];
        const int *regimes_t = &state->regimes[t * Np];

        for (n = 0; n < N; n++)
        {
            int idx_next = state->smoothed[(t + 1) * Np + n];
            int regime_next = state->regimes[(t + 1) * Np + idx_next];
            float h_next = state->h[(t + 1) * Np + idx_next];
            float mu_shift = m->mu_shifts[regime_next];
            const float *log_trans_col = &log_trans_T[regime_next * K];

            /* Per-regime sigma_vol (aligned with RBPF) */
            float neg_half_inv_var = m->neg_half_inv_sigma_vol_sq[regime_next];

            compute_log_bw_scalar(local_log_bw, h_t, log_w_t, regimes_t,
                                  log_trans_col, mu_shift, phi, h_next,
                                  neg_half_inv_var, N, Np);

            logsumexp_normalize(local_log_bw, N, Np, local_bw, local_workspace);
            state->smoothed[t * Np + n] = sample_categorical(local_bw, N, local_cumsum, stream);
        }
    }

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
    printf("PARIS-MKL INFO (per-regime sigma_vol)\n");
    printf("═══════════════════════════════════════════════════════════\n");
    printf("Particles:          %d (padded: %d)\n", state->N, state->N_padded);
    printf("Time steps:         %d\n", state->T);
    printf("Regimes:            %d\n", state->K);
    printf("Thread RNG streams: %d\n", state->n_thread_streams);
    printf("Model:\n");
    printf("  phi:              %.4f\n", state->model.phi);
    printf("  mu_vol[0..%d]:    ", state->K - 1);
    for (int k = 0; k < state->K; k++)
    {
        printf("%.2f ", state->model.mu_vol[k]);
    }
    printf("\n");
    printf("  sigma_vol[0..%d]: ", state->K - 1);
    for (int k = 0; k < state->K; k++)
    {
        printf("%.3f ", state->model.sigma_vol[k]);
    }
    printf("\n");
    printf("═══════════════════════════════════════════════════════════\n");
}