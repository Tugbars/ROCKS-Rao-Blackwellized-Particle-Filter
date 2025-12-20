/**
 * @file pgas_mkl.c
 * @brief MKL-optimized PGAS + PARIS implementation (CORRECTED)
 *
 * Fixes applied:
 *   1. Stride: All accesses use N_padded, not N
 *   2. RNG: Per-thread streams created at allocation, reused in PARIS
 *   3. Inner loop: Use log_trans_col[regime_i] for column access
 *   4. Padding: Use -INFINITY instead of -88.0f
 *   5. Consistency: All arrays allocated with N_padded stride
 */

#include "pgas_mkl.h"
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
#define LOG_EPS -23.0259f
#define NEG_INF (-HUGE_VALF)

/* Adaptive sweep parameters */
#define MIN_SWEEPS 3
#define MAX_SWEEPS 10
#define TARGET_ACCEPTANCE 0.15f
#define ABORT_ACCEPTANCE 0.10f

/*═══════════════════════════════════════════════════════════════════════════════
 * MEMORY ALLOCATION
 *═══════════════════════════════════════════════════════════════════════════════*/

PGASMKLState *pgas_mkl_alloc(int N, int T, int K, uint32_t seed)
{
    PGASMKLState *state = (PGASMKLState *)mkl_calloc(1, sizeof(PGASMKLState), PGAS_MKL_ALIGN);
    if (!state) return NULL;
    
    /* Pad N to multiple of 16 for optimal SIMD */
    int N_padded = ((N + 15) & ~15);
    
    state->N = N;
    state->N_padded = N_padded;
    state->T = T;
    state->K = K;
    state->ref_idx = N - 1;
    
    /* CRITICAL: Use N_padded as stride for all T×N arrays */
    size_t NT_padded = (size_t)N_padded * T;
    
    /* Allocate SoA arrays with MKL alignment - use padded stride! */
    state->regimes = (int *)mkl_malloc(NT_padded * sizeof(int), PGAS_MKL_ALIGN);
    state->h = (float *)mkl_malloc(NT_padded * sizeof(float), PGAS_MKL_ALIGN);
    state->weights = (float *)mkl_malloc(NT_padded * sizeof(float), PGAS_MKL_ALIGN);
    state->log_weights = (float *)mkl_malloc(NT_padded * sizeof(float), PGAS_MKL_ALIGN);
    state->ancestors = (int *)mkl_malloc(NT_padded * sizeof(int), PGAS_MKL_ALIGN);
    state->smoothed = (int *)mkl_malloc(NT_padded * sizeof(int), PGAS_MKL_ALIGN);
    
    /* Zero-initialize to avoid garbage in padding */
    memset(state->regimes, 0, NT_padded * sizeof(int));
    memset(state->h, 0, NT_padded * sizeof(float));
    memset(state->weights, 0, NT_padded * sizeof(float));
    memset(state->log_weights, 0, NT_padded * sizeof(float));
    memset(state->ancestors, 0, NT_padded * sizeof(int));
    memset(state->smoothed, 0, NT_padded * sizeof(int));
    
    state->observations = (float *)mkl_malloc(T * sizeof(float), PGAS_MKL_ALIGN);
    
    state->ref_regimes = (int *)mkl_malloc(T * sizeof(int), PGAS_MKL_ALIGN);
    state->ref_h = (float *)mkl_malloc(T * sizeof(float), PGAS_MKL_ALIGN);
    state->ref_ancestors = (int *)mkl_malloc(T * sizeof(int), PGAS_MKL_ALIGN);
    
    /* Workspace buffers - use N_padded */
    state->ws_log_bw = (float *)mkl_malloc(N_padded * sizeof(float), PGAS_MKL_ALIGN);
    state->ws_bw = (float *)mkl_malloc(N_padded * sizeof(float), PGAS_MKL_ALIGN);
    state->ws_uniform = (float *)mkl_malloc(N_padded * sizeof(float), PGAS_MKL_ALIGN);
    state->ws_normal = (float *)mkl_malloc(N_padded * sizeof(float), PGAS_MKL_ALIGN);
    state->ws_indices = (int *)mkl_malloc(N_padded * sizeof(int), PGAS_MKL_ALIGN);
    state->ws_cumsum = (float *)mkl_malloc(N_padded * sizeof(float), PGAS_MKL_ALIGN);
    
    /* Initialize main MKL RNG (SFMT - fast Mersenne Twister) */
    state->rng.brng = VSL_BRNG_SFMT19937;
    vslNewStream((VSLStreamStatePtr *)&state->rng.stream, state->rng.brng, seed);
    
    /* Pre-create per-thread RNG streams (avoid vslNewStream in hot path!) */
    #ifdef _OPENMP
    int max_threads = omp_get_max_threads();
    #else
    int max_threads = 1;
    #endif
    state->n_thread_streams = (max_threads < PGAS_MKL_MAX_THREADS) ? max_threads : PGAS_MKL_MAX_THREADS;
    
    for (int i = 0; i < state->n_thread_streams; i++) {
        vslNewStream((VSLStreamStatePtr *)&state->thread_rng_streams[i], 
                     VSL_BRNG_SFMT19937, seed + 1000 * (i + 1));
    }
    
    /* Initialize model with uniform transitions */
    state->model.K = K;
    float unif = 1.0f / K;
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < K; j++) {
            state->model.trans[i * K + j] = unif;
            state->model.log_trans[i * K + j] = logf(unif);
        }
        state->model.mu_vol[i] = -1.0f + 0.5f * i;
        state->model.sigma_vol[i] = 0.3f;
    }
    state->model.phi = 0.97f;
    state->model.sigma_h = 0.15f;
    state->model.inv_sigma_h_sq = 1.0f / (0.15f * 0.15f);
    
    return state;
}

void pgas_mkl_free(PGASMKLState *state)
{
    if (!state) return;
    
    /* Free main RNG stream */
    if (state->rng.stream) {
        vslDeleteStream((VSLStreamStatePtr *)&state->rng.stream);
    }
    
    /* Free per-thread RNG streams */
    for (int i = 0; i < state->n_thread_streams; i++) {
        if (state->thread_rng_streams[i]) {
            vslDeleteStream((VSLStreamStatePtr *)&state->thread_rng_streams[i]);
        }
    }
    
    mkl_free(state->regimes);
    mkl_free(state->h);
    mkl_free(state->weights);
    mkl_free(state->log_weights);
    mkl_free(state->ancestors);
    mkl_free(state->smoothed);
    mkl_free(state->observations);
    mkl_free(state->ref_regimes);
    mkl_free(state->ref_h);
    mkl_free(state->ref_ancestors);
    mkl_free(state->ws_log_bw);
    mkl_free(state->ws_bw);
    mkl_free(state->ws_uniform);
    mkl_free(state->ws_normal);
    mkl_free(state->ws_indices);
    mkl_free(state->ws_cumsum);
    mkl_free(state);
}

/*═══════════════════════════════════════════════════════════════════════════════
 * MODEL SETUP
 *═══════════════════════════════════════════════════════════════════════════════*/

void pgas_mkl_set_model(PGASMKLState *state,
                        const double *trans,
                        const double *mu_vol,
                        const double *sigma_vol,
                        double phi,
                        double sigma_h)
{
    if (!state) return;
    
    int K = state->K;
    state->model.K = K;
    
    /* Convert and compute log-trans */
    for (int i = 0; i < K * K; i++) {
        state->model.trans[i] = (float)trans[i];
        state->model.log_trans[i] = logf((float)trans[i] + EPS);
    }
    
    for (int i = 0; i < K; i++) {
        state->model.mu_vol[i] = (float)mu_vol[i];
        state->model.sigma_vol[i] = (float)sigma_vol[i];
    }
    
    state->model.phi = (float)phi;
    state->model.sigma_h = (float)sigma_h;
    state->model.inv_sigma_h_sq = 1.0f / ((float)sigma_h * (float)sigma_h);
}

void pgas_mkl_set_reference(PGASMKLState *state,
                            const int *regimes,
                            const double *h,
                            int T)
{
    if (!state) return;
    
    state->T = T;
    
    for (int t = 0; t < T; t++) {
        state->ref_regimes[t] = regimes[t];
        state->ref_h[t] = (float)h[t];
        state->ref_ancestors[t] = state->ref_idx;
    }
}

void pgas_mkl_load_observations(PGASMKLState *state,
                                const double *observations,
                                int T)
{
    if (!state) return;
    
    state->T = T;
    for (int t = 0; t < T; t++) {
        state->observations[t] = (float)observations[t];
    }
}

/*═══════════════════════════════════════════════════════════════════════════════
 * MKL-ACCELERATED SAMPLING
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * Sample N categorical indices using binary search
 */
static void sample_categorical_mkl(const float *weights, int N,
                                    int *out_indices, int n_samples,
                                    float *ws_uniform, float *ws_cumsum,
                                    VSLStreamStatePtr stream)
{
    /* Compute cumulative sum */
    ws_cumsum[0] = weights[0];
    for (int i = 1; i < N; i++) {
        ws_cumsum[i] = ws_cumsum[i-1] + weights[i];
    }
    
    /* Generate uniform random numbers */
    vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, n_samples, ws_uniform, 0.0f, 1.0f);
    
    /* Binary search for each sample */
    for (int s = 0; s < n_samples; s++) {
        float u = ws_uniform[s];
        int lo = 0, hi = N - 1;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if (ws_cumsum[mid] < u) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        out_indices[s] = lo;
    }
}

/**
 * Sample single categorical index
 */
static int sample_categorical_single(const float *weights, int N,
                                      float *ws_cumsum, VSLStreamStatePtr stream)
{
    ws_cumsum[0] = weights[0];
    for (int i = 1; i < N; i++) {
        ws_cumsum[i] = ws_cumsum[i-1] + weights[i];
    }
    
    float u;
    vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, 1, &u, 0.0f, 1.0f);
    
    int lo = 0, hi = N - 1;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (ws_cumsum[mid] < u) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

/**
 * Sample regime transition
 */
static int sample_regime_mkl(const PGASMKLModel *m, int prev_regime,
                              float *ws_cumsum, VSLStreamStatePtr stream)
{
    return sample_categorical_single(&m->trans[prev_regime * m->K], m->K,
                                      ws_cumsum, stream);
}

/*═══════════════════════════════════════════════════════════════════════════════
 * MKL-ACCELERATED LOG-SUM-EXP
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * Compute normalized weights from log-weights using MKL VML
 *
 * Uses N_padded for SIMD but only sums/normalizes first N elements
 */
static void logsumexp_normalize_mkl(const float *log_weights, int N, int N_padded,
                                     float *weights, float *workspace)
{
    /* Find max using CBLAS (only over valid N elements) */
    int max_idx = cblas_isamax(N, log_weights, 1);
    float max_val = log_weights[max_idx];
    float neg_max = -max_val;
    
    /* Subtract max using full N_padded for SIMD */
    #pragma omp simd
    for (int i = 0; i < N_padded; i++) {
        workspace[i] = log_weights[i] + neg_max;
    }
    
    /* Set padding to -inf so exp produces exact 0 */
    for (int i = N; i < N_padded; i++) {
        workspace[i] = NEG_INF;
    }
    
    /* Exp using MKL VML (full N_padded for SIMD alignment) */
    vsExp(N_padded, workspace, weights);
    
    /* Sum only valid N elements */
    float sum = cblas_sasum(N, weights, 1);
    
    /* Normalize only valid N elements */
    float inv_sum = 1.0f / sum;
    cblas_sscal(N, inv_sum, weights, 1);
}

/*═══════════════════════════════════════════════════════════════════════════════
 * EMISSION PROBABILITY
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * Compute log emission: log P(y | h) where y ~ N(0, exp(h))
 */
static void compute_log_emission_mkl(float y, const float *h, int N, int N_padded,
                                      float *log_lik, float *workspace)
{
    const float log_2pi = 1.8378770664f;
    float y_sq = y * y;
    
    /* workspace = -h */
    #pragma omp simd
    for (int i = 0; i < N_padded; i++) {
        workspace[i] = -h[i];
    }
    
    /* workspace = exp(-h) */
    vsExp(N_padded, workspace, workspace);
    
    /* log_lik = -0.5 * (log_2pi + h + y² * exp(-h)) */
    #pragma omp simd
    for (int i = 0; i < N_padded; i++) {
        log_lik[i] = -0.5f * (log_2pi + h[i] + y_sq * workspace[i]);
    }
    
    /* Set padding to NEG_INF */
    for (int i = N; i < N_padded; i++) {
        log_lik[i] = NEG_INF;
    }
}

/*═══════════════════════════════════════════════════════════════════════════════
 * PGAS CSMC SWEEP
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * Initialize particles at t=0
 */
static void csmc_init_mkl(PGASMKLState *state)
{
    const int N = state->N;
    const int Np = state->N_padded;  /* Use padded stride! */
    const int K = state->K;
    const int ref_idx = state->ref_idx;
    VSLStreamStatePtr stream = (VSLStreamStatePtr)state->rng.stream;
    
    /* Generate random regimes for non-reference particles */
    int *rand_regimes = state->ws_indices;
    viRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, N, rand_regimes, 0, K);
    
    /* Generate random h values */
    vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, stream, N,
                  state->ws_normal, 0.0f, state->model.sigma_h);
    
    /* Initialize particles at t=0 (stride = Np) */
    for (int n = 0; n < N; n++) {
        if (n == ref_idx) {
            state->regimes[n] = state->ref_regimes[0];
            state->h[n] = state->ref_h[0];
        } else {
            int regime = rand_regimes[n];
            state->regimes[n] = regime;
            state->h[n] = state->model.mu_vol[regime] + state->ws_normal[n];
        }
        state->ancestors[n] = n;
    }
    
    /* Zero padding */
    for (int n = N; n < Np; n++) {
        state->regimes[n] = 0;
        state->h[n] = 0.0f;
        state->ancestors[n] = 0;
    }
    
    /* Compute initial weights */
    compute_log_emission_mkl(state->observations[0], state->h, N, Np,
                              state->log_weights, state->ws_bw);
    
    logsumexp_normalize_mkl(state->log_weights, N, Np, state->weights, state->ws_bw);
}

/**
 * Ancestor sampling for reference trajectory
 */
static int ancestor_sample_mkl(PGASMKLState *state, int t)
{
    const int N = state->N;
    const int Np = state->N_padded;
    const int K = state->model.K;
    const PGASMKLModel *m = &state->model;
    VSLStreamStatePtr stream = (VSLStreamStatePtr)state->rng.stream;
    
    int ref_regime = state->ref_regimes[t];
    float ref_h = state->ref_h[t];
    float mu_k = m->mu_vol[ref_regime];
    float phi = m->phi;
    float neg_half_inv_var = -0.5f * m->inv_sigma_h_sq;
    
    float *log_as = state->ws_log_bw;
    
    /* Previous time data (stride = Np) */
    float *prev_log_w = &state->log_weights[(t-1) * Np];
    float *prev_h = &state->h[(t-1) * Np];
    int *prev_regimes = &state->regimes[(t-1) * Np];
    
    /* Precompute log_trans column for ref_regime (Rank-1 optimization!) */
    /* log_trans_col[i] = log P(ref_regime | regime_i) */
    float log_trans_col[PGAS_MKL_MAX_REGIMES];
    for (int k = 0; k < K; k++) {
        log_trans_col[k] = m->log_trans[k * K + ref_regime];
    }
    
    /* Compute AS log-weights */
    #pragma omp simd
    for (int n = 0; n < N; n++) {
        int regime_n = prev_regimes[n];
        float h_n = prev_h[n];
        
        /* Use precomputed column! */
        float log_trans = log_trans_col[regime_n];
        
        /* Log h transition */
        float mean = mu_k + phi * (h_n - mu_k);
        float diff = ref_h - mean;
        float log_h_trans = neg_half_inv_var * diff * diff;
        
        log_as[n] = prev_log_w[n] + log_trans + log_h_trans;
    }
    
    /* Set padding to NEG_INF */
    for (int n = N; n < Np; n++) {
        log_as[n] = NEG_INF;
    }
    
    /* Normalize and sample */
    logsumexp_normalize_mkl(log_as, N, Np, state->ws_bw, state->ws_uniform);
    
    return sample_categorical_single(state->ws_bw, N, state->ws_cumsum, stream);
}

/**
 * One CSMC sweep
 */
float pgas_mkl_csmc_sweep(PGASMKLState *state)
{
    if (!state || state->T < 2) return 0.0f;
    
    const int N = state->N;
    const int Np = state->N_padded;
    const int T = state->T;
    const int K = state->model.K;
    const int ref_idx = state->ref_idx;
    const PGASMKLModel *m = &state->model;
    VSLStreamStatePtr stream = (VSLStreamStatePtr)state->rng.stream;
    
    state->ancestor_proposals = 0;
    state->ancestor_accepts = 0;
    
    /* Initialize at t=0 */
    csmc_init_mkl(state);
    
    /* Forward pass */
    for (int t = 1; t < T; t++) {
        /* Previous time data (stride = Np!) */
        float *prev_weights = &state->weights[(t-1) * Np];
        float *prev_h = &state->h[(t-1) * Np];
        int *prev_regimes = &state->regimes[(t-1) * Np];
        
        /* Current time data (stride = Np!) */
        float *curr_h = &state->h[t * Np];
        int *curr_regimes = &state->regimes[t * Np];
        int *curr_ancestors = &state->ancestors[t * Np];
        
        /* Resample ancestors for non-reference particles */
        sample_categorical_mkl(prev_weights, N, curr_ancestors, N,
                               state->ws_uniform, state->ws_cumsum, stream);
        
        /* Ancestor sampling for reference */
        int old_ref_anc = state->ref_ancestors[t];
        int new_ref_anc = ancestor_sample_mkl(state, t);
        
        state->ancestor_proposals++;
        if (new_ref_anc != old_ref_anc) {
            state->ancestor_accepts++;
            state->ref_ancestors[t] = new_ref_anc;
        }
        curr_ancestors[ref_idx] = state->ref_ancestors[t];
        
        /* Generate random numbers for propagation */
        vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, stream, N,
                      state->ws_normal, 0.0f, m->sigma_h);
        
        /* Propagate particles */
        for (int n = 0; n < N; n++) {
            if (n == ref_idx) {
                curr_regimes[n] = state->ref_regimes[t];
                curr_h[n] = state->ref_h[t];
            } else {
                int anc = curr_ancestors[n];
                int prev_regime = prev_regimes[anc];
                float prev_h_anc = prev_h[anc];
                
                /* Sample regime */
                curr_regimes[n] = sample_regime_mkl(m, prev_regime,
                                                     state->ws_cumsum, stream);
                
                /* Sample h */
                float mu_k = m->mu_vol[curr_regimes[n]];
                float mean = mu_k + m->phi * (prev_h_anc - mu_k);
                curr_h[n] = mean + state->ws_normal[n];
            }
        }
        
        /* Zero padding */
        for (int n = N; n < Np; n++) {
            curr_regimes[n] = 0;
            curr_h[n] = 0.0f;
            curr_ancestors[n] = 0;
        }
        
        /* Compute weights (stride = Np!) */
        compute_log_emission_mkl(state->observations[t], curr_h, N, Np,
                                  &state->log_weights[t * Np], state->ws_bw);
        
        logsumexp_normalize_mkl(&state->log_weights[t * Np], N, Np,
                                 &state->weights[t * Np], state->ws_bw);
    }
    
    /* Sample final trajectory and update reference */
    int final_idx = sample_categorical_single(&state->weights[(T-1) * Np], N,
                                               state->ws_cumsum, stream);
    
    /* Trace back (stride = Np!) */
    int idx = final_idx;
    for (int t = T - 1; t >= 0; t--) {
        state->ref_regimes[t] = state->regimes[t * Np + idx];
        state->ref_h[t] = state->h[t * Np + idx];
        if (t > 0) {
            idx = state->ancestors[t * Np + idx];
        }
    }
    
    state->acceptance_rate = (state->ancestor_proposals > 0) ?
        (float)state->ancestor_accepts / state->ancestor_proposals : 0.0f;
    
    state->current_sweep++;
    state->total_sweeps++;
    
    return state->acceptance_rate;
}

/**
 * Adaptive PGAS
 */
int pgas_mkl_run_adaptive(PGASMKLState *state)
{
    if (!state || state->T < 2) return 2;
    
    state->current_sweep = 0;
    
    /* Minimum sweeps */
    for (int s = 0; s < MIN_SWEEPS; s++) {
        pgas_mkl_csmc_sweep(state);
    }
    
    if (state->acceptance_rate >= TARGET_ACCEPTANCE) {
        return 0;
    }
    
    /* Continue until target or max */
    while (state->current_sweep < MAX_SWEEPS) {
        pgas_mkl_csmc_sweep(state);
        
        if (state->acceptance_rate >= TARGET_ACCEPTANCE) {
            return 0;
        }
    }
    
    if (state->acceptance_rate < ABORT_ACCEPTANCE) {
        return 2;
    }
    
    return 1;
}

/*═══════════════════════════════════════════════════════════════════════════════
 * PARIS BACKWARD SMOOTHING (CORRECTED)
 *═══════════════════════════════════════════════════════════════════════════════*/

void paris_mkl_backward_smooth(PGASMKLState *state)
{
    if (!state || state->T < 2) return;
    
    const int N = state->N;
    const int Np = state->N_padded;
    const int T = state->T;
    const int K = state->model.K;
    const PGASMKLModel *m = &state->model;
    
    /* Precompute constants */
    const float phi = m->phi;
    const float neg_half_inv_var = -0.5f * m->inv_sigma_h_sq;
    
    /* Initialize at final time (stride = Np!) */
    for (int n = 0; n < N; n++) {
        state->smoothed[(T-1) * Np + n] = n;
    }
    
    /* Backward pass with OpenMP */
    #ifdef _OPENMP
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        
        /* Use pre-allocated per-thread RNG (no vslNewStream in hot path!) */
        VSLStreamStatePtr local_stream = (VSLStreamStatePtr)state->thread_rng_streams[tid];
        
        /* Thread-local workspace - use Np for SIMD */
        float *local_log_bw = (float *)mkl_malloc(Np * sizeof(float), PGAS_MKL_ALIGN);
        float *local_bw = (float *)mkl_malloc(Np * sizeof(float), PGAS_MKL_ALIGN);
        float *local_workspace = (float *)mkl_malloc(Np * sizeof(float), PGAS_MKL_ALIGN);
        float *local_cumsum = (float *)mkl_malloc(Np * sizeof(float), PGAS_MKL_ALIGN);
        
        for (int t = T - 2; t >= 0; t--) {
            /* Data at time t (stride = Np!) */
            const float *h_t = &state->h[t * Np];
            const float *log_w_t = &state->log_weights[t * Np];
            const int *regimes_t = &state->regimes[t * Np];
            
            #pragma omp for schedule(dynamic, 8)
            for (int n = 0; n < N; n++) {
                /* Get smoothed state at t+1 (stride = Np!) */
                int idx_next = state->smoothed[(t+1) * Np + n];
                int regime_next = state->regimes[(t+1) * Np + idx_next];
                float h_next = state->h[(t+1) * Np + idx_next];
                float mu_k = m->mu_vol[regime_next];
                
                /* Precompute log_trans column (Rank-1 optimization!) */
                float log_trans_col[PGAS_MKL_MAX_REGIMES];
                for (int k = 0; k < K; k++) {
                    log_trans_col[k] = m->log_trans[k * K + regime_next];
                }
                
                /* Compute backward log-weights */
                #pragma omp simd
                for (int i = 0; i < N; i++) {
                    int regime_i = regimes_t[i];
                    float h_i = h_t[i];
                    
                    /* Use precomputed column! */
                    float log_trans = log_trans_col[regime_i];
                    float mean = mu_k + phi * (h_i - mu_k);
                    float diff = h_next - mean;
                    float log_h_trans = neg_half_inv_var * diff * diff;
                    
                    local_log_bw[i] = log_w_t[i] + log_trans + log_h_trans;
                }
                
                /* Set padding to NEG_INF */
                for (int i = N; i < Np; i++) {
                    local_log_bw[i] = NEG_INF;
                }
                
                /* Normalize */
                logsumexp_normalize_mkl(local_log_bw, N, Np, local_bw, local_workspace);
                
                /* Sample (stride = Np!) */
                state->smoothed[t * Np + n] = sample_categorical_single(
                    local_bw, N, local_cumsum, local_stream);
            }
        }
        
        mkl_free(local_log_bw);
        mkl_free(local_bw);
        mkl_free(local_workspace);
        mkl_free(local_cumsum);
    }
    #else
    /* Sequential version */
    float *local_log_bw = state->ws_log_bw;
    float *local_bw = state->ws_bw;
    float *local_workspace = state->ws_uniform;
    VSLStreamStatePtr stream = (VSLStreamStatePtr)state->rng.stream;
    
    for (int t = T - 2; t >= 0; t--) {
        const float *h_t = &state->h[t * Np];
        const float *log_w_t = &state->log_weights[t * Np];
        const int *regimes_t = &state->regimes[t * Np];
        
        for (int n = 0; n < N; n++) {
            int idx_next = state->smoothed[(t+1) * Np + n];
            int regime_next = state->regimes[(t+1) * Np + idx_next];
            float h_next = state->h[(t+1) * Np + idx_next];
            float mu_k = m->mu_vol[regime_next];
            
            /* Precompute log_trans column */
            float log_trans_col[PGAS_MKL_MAX_REGIMES];
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
            
            logsumexp_normalize_mkl(local_log_bw, N, Np, local_bw, local_workspace);
            
            state->smoothed[t * Np + n] = sample_categorical_single(
                local_bw, N, state->ws_cumsum, stream);
        }
    }
    #endif
}

void paris_mkl_get_smoothed(const PGASMKLState *state, int t,
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

/*═══════════════════════════════════════════════════════════════════════════════
 * LIFEBOAT
 *═══════════════════════════════════════════════════════════════════════════════*/

void pgas_mkl_generate_lifeboat(const PGASMKLState *state, LifeboatPacketMKL *packet)
{
    if (!state || !packet) return;
    
    packet->K = state->K;
    packet->N = state->N;
    packet->T = state->T;
    
    memcpy(packet->trans, state->model.trans, state->K * state->K * sizeof(float));
    memcpy(packet->mu_vol, state->model.mu_vol, state->K * sizeof(float));
    memcpy(packet->sigma_vol, state->model.sigma_vol, state->K * sizeof(float));
    packet->phi = state->model.phi;
    packet->sigma_h = state->model.sigma_h;
    
    if (!packet->final_regimes) {
        packet->final_regimes = (int *)mkl_malloc(state->N * sizeof(int), PGAS_MKL_ALIGN);
        packet->final_h = (float *)mkl_malloc(state->N * sizeof(float), PGAS_MKL_ALIGN);
        packet->final_weights = (float *)mkl_malloc(state->N * sizeof(float), PGAS_MKL_ALIGN);
    }
    
    paris_mkl_get_smoothed(state, state->T - 1, packet->final_regimes, packet->final_h);
    
    float unif_w = 1.0f / state->N;
    for (int n = 0; n < state->N; n++) {
        packet->final_weights[n] = unif_w;
    }
    
    packet->ancestor_acceptance = state->acceptance_rate;
    packet->sweeps_used = state->current_sweep;
}

bool lifeboat_mkl_validate(const LifeboatPacketMKL *packet)
{
    if (!packet) return false;
    
    if (packet->ancestor_acceptance < ABORT_ACCEPTANCE) return false;
    
    for (int i = 0; i < packet->K; i++) {
        float sum = 0;
        for (int j = 0; j < packet->K; j++) {
            float p = packet->trans[i * packet->K + j];
            if (p < 0 || p > 1) return false;
            sum += p;
        }
        if (fabsf(sum - 1.0f) > 1e-4f) return false;
    }
    
    return true;
}

/*═══════════════════════════════════════════════════════════════════════════════
 * DIAGNOSTICS
 *═══════════════════════════════════════════════════════════════════════════════*/

float pgas_mkl_get_acceptance_rate(const PGASMKLState *state)
{
    return state ? state->acceptance_rate : 0.0f;
}

int pgas_mkl_get_sweep_count(const PGASMKLState *state)
{
    return state ? state->current_sweep : 0;
}

float pgas_mkl_get_ess(const PGASMKLState *state, int t)
{
    if (!state || t < 0 || t >= state->T) return 0.0f;
    
    const int N = state->N;
    const int Np = state->N_padded;
    const float *weights = &state->weights[t * Np];
    
    float sum_sq = cblas_sdot(N, weights, 1, weights, 1);
    
    return (sum_sq > EPS) ? 1.0f / sum_sq : 0.0f;
}

void pgas_mkl_print_diagnostics(const PGASMKLState *state)
{
    if (!state) return;
    
    printf("═══════════════════════════════════════════════════════════\n");
    printf("PGAS-MKL DIAGNOSTICS\n");
    printf("═══════════════════════════════════════════════════════════\n");
    printf("Particles:          %d (padded: %d)\n", state->N, state->N_padded);
    printf("Regimes:            %d\n", state->K);
    printf("Buffer length:      %d\n", state->T);
    printf("Thread RNG streams: %d\n", state->n_thread_streams);
    printf("Sweeps completed:   %d\n", state->current_sweep);
    printf("Ancestor accepts:   %d / %d\n", state->ancestor_accepts, state->ancestor_proposals);
    printf("Acceptance rate:    %.3f %s\n", state->acceptance_rate,
           state->acceptance_rate >= TARGET_ACCEPTANCE ? "(CONVERGED)" :
           state->acceptance_rate >= ABORT_ACCEPTANCE ? "(MIXING)" : "(STUCK)");
    printf("Final ESS:          %.1f / %d\n", pgas_mkl_get_ess(state, state->T - 1), state->N);
    printf("═══════════════════════════════════════════════════════════\n");
}
