/**
 * @file paris_mkl.c
 * @brief Standalone MKL-optimized PARIS backward smoother
 *
 * Key optimizations:
 *   1. N_padded stride eliminates SIMD tail masking
 *   2. Pre-allocated per-thread RNG streams (no vslNewStream in hot path)
 *   3. Rank-1 log_trans column access pattern
 *   4. VML vsExp for batch exponential
 *   5. CBLAS for vectorized max/sum/scale
 *   6. -INFINITY padding for clean exp results
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

#ifdef _OPENMP
#include <omp.h>
#endif

/*═══════════════════════════════════════════════════════════════════════════════
 * CONSTANTS
 *═══════════════════════════════════════════════════════════════════════════════*/

#define EPS 1e-10f
#define NEG_INF (-HUGE_VALF)

/*═══════════════════════════════════════════════════════════════════════════════
 * ALLOCATION
 *═══════════════════════════════════════════════════════════════════════════════*/

PARISMKLState *paris_mkl_alloc(int N, int T, int K, uint32_t seed)
{
    PARISMKLState *state = (PARISMKLState *)mkl_calloc(1, sizeof(PARISMKLState), PARIS_MKL_ALIGN);
    if (!state) return NULL;
    
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
    
    /* Main RNG stream */
    vslNewStream((VSLStreamStatePtr *)&state->rng_stream, VSL_BRNG_SFMT19937, seed);
    
    /* Pre-allocate per-thread RNG streams */
    #ifdef _OPENMP
    int max_threads = omp_get_max_threads();
    #else
    int max_threads = 1;
    #endif
    state->n_thread_streams = (max_threads < PARIS_MKL_MAX_THREADS) ? max_threads : PARIS_MKL_MAX_THREADS;
    
    for (int i = 0; i < state->n_thread_streams; i++) {
        vslNewStream((VSLStreamStatePtr *)&state->thread_rng_streams[i],
                     VSL_BRNG_SFMT19937, seed + 1000 * (i + 1));
    }
    
    /* Initialize model defaults */
    state->model.K = K;
    float unif = 1.0f / K;
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < K; j++) {
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
    if (!state) return;
    
    /* Free main RNG */
    if (state->rng_stream) {
        vslDeleteStream((VSLStreamStatePtr *)&state->rng_stream);
    }
    
    /* Free per-thread RNG streams */
    for (int i = 0; i < state->n_thread_streams; i++) {
        if (state->thread_rng_streams[i]) {
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
    if (!state) return;
    
    int K = state->K;
    state->model.K = K;
    
    for (int i = 0; i < K * K; i++) {
        state->model.trans[i] = (float)trans[i];
        state->model.log_trans[i] = logf((float)trans[i] + EPS);
    }
    
    for (int i = 0; i < K; i++) {
        state->model.mu_vol[i] = (float)mu_vol[i];
    }
    
    state->model.phi = (float)phi;
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
    if (!state) return;
    
    state->T = T;
    const int N = state->N;
    const int Np = state->N_padded;
    
    /* Copy with stride conversion: input is [T×N], output is [T×Np] */
    for (int t = 0; t < T; t++) {
        /* Copy valid particles */
        for (int n = 0; n < N; n++) {
            int src_idx = t * N + n;
            int dst_idx = t * Np + n;
            
            state->regimes[dst_idx] = regimes[src_idx];
            state->h[dst_idx] = (float)h[src_idx];
            state->log_weights[dst_idx] = logf((float)weights[src_idx] + EPS);
            state->ancestors[dst_idx] = ancestors[src_idx];
        }
        
        /* Zero padding */
        for (int n = N; n < Np; n++) {
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
 * Sample single index from categorical distribution using binary search
 */
static inline int sample_categorical(const float *weights, int N,
                                      float *cumsum, VSLStreamStatePtr stream)
{
    /* Build cumulative sum */
    cumsum[0] = weights[0];
    for (int i = 1; i < N; i++) {
        cumsum[i] = cumsum[i-1] + weights[i];
    }
    
    /* Generate uniform */
    float u;
    vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, 1, &u, 0.0f, 1.0f);
    
    /* Binary search */
    int lo = 0, hi = N - 1;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (cumsum[mid] < u) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
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
    
    /* Subtract max (full Np for SIMD) */
    #pragma omp simd
    for (int i = 0; i < Np; i++) {
        workspace[i] = log_w[i] + neg_max;
    }
    
    /* Set padding to -inf */
    for (int i = N; i < Np; i++) {
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

void paris_mkl_backward_smooth(PARISMKLState *state)
{
    if (!state || state->T < 2) return;
    
    const int N = state->N;
    const int Np = state->N_padded;
    const int T = state->T;
    const int K = state->model.K;
    const PARISMKLModel *m = &state->model;
    
    /* Precompute constants */
    const float phi = m->phi;
    const float neg_half_inv_var = -0.5f * m->inv_sigma_h_sq;
    
    /* Initialize at final time */
    for (int n = 0; n < N; n++) {
        state->smoothed[(T-1) * Np + n] = n;
    }
    
    #ifdef _OPENMP
    /* ═══════════════════════════════════════════════════════════════════════
     * PARALLEL BACKWARD PASS
     * ═══════════════════════════════════════════════════════════════════════*/
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        
        /* Use pre-allocated per-thread RNG (critical optimization!) */
        VSLStreamStatePtr local_stream = (VSLStreamStatePtr)state->thread_rng_streams[tid];
        
        /* Thread-local workspace */
        float *local_log_bw = (float *)mkl_malloc(Np * sizeof(float), PARIS_MKL_ALIGN);
        float *local_bw = (float *)mkl_malloc(Np * sizeof(float), PARIS_MKL_ALIGN);
        float *local_workspace = (float *)mkl_malloc(Np * sizeof(float), PARIS_MKL_ALIGN);
        float *local_cumsum = (float *)mkl_malloc(Np * sizeof(float), PARIS_MKL_ALIGN);
        
        for (int t = T - 2; t >= 0; t--) {
            /* Data at time t (stride = Np) */
            const float *h_t = &state->h[t * Np];
            const float *log_w_t = &state->log_weights[t * Np];
            const int *regimes_t = &state->regimes[t * Np];
            
            #pragma omp for schedule(dynamic, 8)
            for (int n = 0; n < N; n++) {
                /* Get smoothed state at t+1 */
                int idx_next = state->smoothed[(t+1) * Np + n];
                int regime_next = state->regimes[(t+1) * Np + idx_next];
                float h_next = state->h[(t+1) * Np + idx_next];
                float mu_k = m->mu_vol[regime_next];
                
                /* ═══════════════════════════════════════════════════════════
                 * RANK-1 OPTIMIZATION: Precompute transition column
                 * log_trans_col[k] = log P(regime_next | k)
                 * ═══════════════════════════════════════════════════════════*/
                float log_trans_col[PARIS_MKL_MAX_REGIMES];
                for (int k = 0; k < K; k++) {
                    log_trans_col[k] = m->log_trans[k * K + regime_next];
                }
                
                /* Compute backward log-weights */
                #pragma omp simd
                for (int i = 0; i < N; i++) {
                    int regime_i = regimes_t[i];
                    float h_i = h_t[i];
                    
                    /* Use precomputed column (no 2D lookup!) */
                    float log_trans = log_trans_col[regime_i];
                    
                    /* AR(1) transition */
                    float mean = mu_k + phi * (h_i - mu_k);
                    float diff = h_next - mean;
                    float log_h_trans = neg_half_inv_var * diff * diff;
                    
                    local_log_bw[i] = log_w_t[i] + log_trans + log_h_trans;
                }
                
                /* Padding to -inf */
                for (int i = N; i < Np; i++) {
                    local_log_bw[i] = NEG_INF;
                }
                
                /* Normalize and sample */
                logsumexp_normalize(local_log_bw, N, Np, local_bw, local_workspace);
                state->smoothed[t * Np + n] = sample_categorical(local_bw, N, local_cumsum, local_stream);
            }
        }
        
        mkl_free(local_log_bw);
        mkl_free(local_bw);
        mkl_free(local_workspace);
        mkl_free(local_cumsum);
    }
    
    #else
    /* ═══════════════════════════════════════════════════════════════════════
     * SEQUENTIAL BACKWARD PASS
     * ═══════════════════════════════════════════════════════════════════════*/
    VSLStreamStatePtr stream = (VSLStreamStatePtr)state->rng_stream;
    float *local_log_bw = state->ws_log_bw;
    float *local_bw = state->ws_bw;
    float *local_workspace = state->ws_workspace;
    float *local_cumsum = state->ws_cumsum;
    
    for (int t = T - 2; t >= 0; t--) {
        const float *h_t = &state->h[t * Np];
        const float *log_w_t = &state->log_weights[t * Np];
        const int *regimes_t = &state->regimes[t * Np];
        
        for (int n = 0; n < N; n++) {
            int idx_next = state->smoothed[(t+1) * Np + n];
            int regime_next = state->regimes[(t+1) * Np + idx_next];
            float h_next = state->h[(t+1) * Np + idx_next];
            float mu_k = m->mu_vol[regime_next];
            
            /* Precompute transition column */
            float log_trans_col[PARIS_MKL_MAX_REGIMES];
            for (int k = 0; k < K; k++) {
                log_trans_col[k] = m->log_trans[k * K + regime_next];
            }
            
            #pragma omp simd
            for (int i = 0; i < N; i++) {
                int regime_i = regimes_t[i];
                float h_i = h_t[i];
                
                float log_trans = log_trans_col[regime_i];
                float mean = mu_k + phi * (h_i - mu_k);
                float diff = h_next - mean;
                float log_h_trans = neg_half_inv_var * diff * diff;
                
                local_log_bw[i] = log_w_t[i] + log_trans + log_h_trans;
            }
            
            for (int i = N; i < Np; i++) {
                local_log_bw[i] = NEG_INF;
            }
            
            logsumexp_normalize(local_log_bw, N, Np, local_bw, local_workspace);
            state->smoothed[t * Np + n] = sample_categorical(local_bw, N, local_cumsum, stream);
        }
    }
    #endif
}

/*═══════════════════════════════════════════════════════════════════════════════
 * OUTPUT EXTRACTION
 *═══════════════════════════════════════════════════════════════════════════════*/

void paris_mkl_get_smoothed(const PARISMKLState *state, int t,
                            int *regimes, float *h)
{
    if (!state || t < 0 || t >= state->T) return;
    
    const int N = state->N;
    const int Np = state->N_padded;
    
    for (int n = 0; n < N; n++) {
        int idx = state->smoothed[t * Np + n];
        if (regimes) regimes[n] = state->regimes[t * Np + idx];
        if (h) h[n] = state->h[t * Np + idx];
    }
}

void paris_mkl_get_trajectory(const PARISMKLState *state, int n,
                              int *regimes, float *h)
{
    if (!state || n < 0 || n >= state->N) return;
    
    const int T = state->T;
    const int Np = state->N_padded;
    
    for (int t = 0; t < T; t++) {
        int idx = state->smoothed[t * Np + n];
        if (regimes) regimes[t] = state->regimes[t * Np + idx];
        if (h) h[t] = state->h[t * Np + idx];
    }
}

/*═══════════════════════════════════════════════════════════════════════════════
 * DIAGNOSTICS
 *═══════════════════════════════════════════════════════════════════════════════*/

void paris_mkl_print_info(const PARISMKLState *state)
{
    if (!state) return;
    
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
    for (int k = 0; k < state->K; k++) {
        printf("%.2f ", state->model.mu_vol[k]);
    }
    printf("\n");
    printf("═══════════════════════════════════════════════════════════\n");
}
