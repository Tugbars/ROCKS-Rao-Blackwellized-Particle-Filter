/**
 * @file pgas_mkl.c
 * @brief MKL-optimized PGAS + PARIS implementation
 *
 * UPDATED: OCSN 10-component emission + Transition matrix learning
 *
 * Changes from original:
 *   1. Replaced Gaussian emission with OCSN 10-component (Omori et al. 2007)
 *   2. Added transition matrix sampling via Dirichlet posterior
 *   3. Added pgas_mkl_gibbs_sweep() for full Gibbs iteration
 *   4. Matches RBPF and HDP-HMM likelihood exactly for validation
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
 * OCSN 10-COMPONENT MIXTURE (Omori, Chib, Shephard, Nakajima 2007)
 *
 * CRITICAL: This must match RBPF and HDP-HMM exactly for validation!
 *
 * Model: y_t = log(r_t²) = h_t + ε_t, where ε_t ~ log-χ²(1)
 * Approximation: y_t | h_t ~ Σ_{j=1}^{10} q_j × N(h_t + m_j, v_j²)
 *═══════════════════════════════════════════════════════════════════════════════*/

#define OCSN_N_COMPONENTS 10

/* OCSN (2007) mixture weights */
static const float OCSN_PROB[OCSN_N_COMPONENTS] = {
    0.00609f, 0.04775f, 0.13057f, 0.20674f, 0.22715f,
    0.18842f, 0.12047f, 0.05591f, 0.01575f, 0.00115f};

/* OCSN (2007) mixture means */
static const float OCSN_MEAN[OCSN_N_COMPONENTS] = {
    1.92677f, 1.34744f, 0.73504f, 0.02266f, -0.85173f,
    -1.97278f, -3.46788f, -5.55246f, -8.68384f, -14.65000f};

/* OCSN (2007) mixture variances */
static const float OCSN_VAR[OCSN_N_COMPONENTS] = {
    0.11265f, 0.17788f, 0.26768f, 0.40611f, 0.62699f,
    0.98583f, 1.57469f, 2.54498f, 4.16591f, 7.33342f};

/* Precomputed constants for fast likelihood evaluation */
static float OCSN_LOG_CONST[OCSN_N_COMPONENTS]; /* -0.5*log(2π*v²) + log(q) */
static float OCSN_INV_2V[OCSN_N_COMPONENTS];    /* 0.5 / v² */
static int ocsn_initialized = 0;

/**
 * Initialize OCSN precomputed constants (called automatically)
 */
static void init_ocsn_constants(void)
{
    if (ocsn_initialized)
        return;

    const float LOG_2PI = 1.8378770664f;

    for (int j = 0; j < OCSN_N_COMPONENTS; j++)
    {
        /* -0.5 * log(2π * v²) + log(q) */
        OCSN_LOG_CONST[j] = -0.5f * (LOG_2PI + logf(OCSN_VAR[j])) + logf(OCSN_PROB[j]);
        /* 0.5 / v² */
        OCSN_INV_2V[j] = 0.5f / OCSN_VAR[j];
    }
    ocsn_initialized = 1;
}

/**
 * OCSN log-likelihood for single observation
 *
 * @param y  Observation y_t = log(r_t²)
 * @param h  Log-volatility state h_t
 * @return   log P(y | h) under OCSN mixture
 */
static inline float ocsn_log_likelihood_single(float y, float h)
{
    float y_base = y - h;

    float max_log = NEG_INF;
    float log_comps[OCSN_N_COMPONENTS];

    /* Compute log of each mixture component */
    for (int j = 0; j < OCSN_N_COMPONENTS; j++)
    {
        float diff = y_base - OCSN_MEAN[j];
        log_comps[j] = OCSN_LOG_CONST[j] - OCSN_INV_2V[j] * diff * diff;
        if (log_comps[j] > max_log)
        {
            max_log = log_comps[j];
        }
    }

    /* Log-sum-exp for numerical stability */
    float sum = 0.0f;
    for (int j = 0; j < OCSN_N_COMPONENTS; j++)
    {
        sum += expf(log_comps[j] - max_log);
    }

    return max_log + logf(sum);
}

/*═══════════════════════════════════════════════════════════════════════════════
 * MEMORY ALLOCATION
 *═══════════════════════════════════════════════════════════════════════════════*/

PGASMKLState *pgas_mkl_alloc(int N, int T, int K, uint32_t seed)
{
    PGASMKLState *state = (PGASMKLState *)mkl_calloc(1, sizeof(PGASMKLState), PGAS_MKL_ALIGN);
    if (!state)
        return NULL;

    /* Initialize OCSN constants */
    init_ocsn_constants();

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
    /* OCSN workspace: 10 * N_padded for log_comps + N_padded for exp_comps */
    state->ws_ocsn = (float *)mkl_malloc(11 * N_padded * sizeof(float), PGAS_MKL_ALIGN);

    /* Walker's Alias Table workspace */
    state->ws_alias_prob = (float *)mkl_malloc(N_padded * sizeof(float), PGAS_MKL_ALIGN);
    state->ws_alias_idx = (int *)mkl_malloc(N_padded * sizeof(int), PGAS_MKL_ALIGN);
    state->ws_alias_small = (int *)mkl_malloc(N_padded * sizeof(int), PGAS_MKL_ALIGN);
    state->ws_alias_large = (int *)mkl_malloc(N_padded * sizeof(int), PGAS_MKL_ALIGN);

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

    for (int i = 0; i < state->n_thread_streams; i++)
    {
        vslNewStream((VSLStreamStatePtr *)&state->thread_rng_streams[i],
                     VSL_BRNG_SFMT19937, seed + 1000 * (i + 1));

        /* Pre-allocate per-thread workspaces (avoid mkl_malloc in hot path!) */
        state->thread_ws[i].log_bw = (float *)mkl_malloc(N_padded * sizeof(float), PGAS_MKL_ALIGN);
        state->thread_ws[i].bw = (float *)mkl_malloc(N_padded * sizeof(float), PGAS_MKL_ALIGN);
        state->thread_ws[i].workspace = (float *)mkl_malloc(N_padded * sizeof(float), PGAS_MKL_ALIGN);
        state->thread_ws[i].cumsum = (float *)mkl_malloc(N_padded * sizeof(float), PGAS_MKL_ALIGN);
    }

    /* Initialize model with uniform transitions */
    state->model.K = K;
    float unif = 1.0f / K;
    float log_unif = logf(unif);
    for (int i = 0; i < K; i++)
    {
        for (int j = 0; j < K; j++)
        {
            state->model.trans[i * K + j] = unif;
            state->model.log_trans[i * K + j] = log_unif;
            state->model.log_trans_T[j * K + i] = log_unif; /* Transposed */
        }
        state->model.mu_vol[i] = -1.0f + 0.5f * i;
        state->model.sigma_vol[i] = 0.3f;
    }
    state->model.phi = 0.97f;
    state->model.sigma_h = 0.15f;
    state->model.inv_sigma_h_sq = 1.0f / (0.15f * 0.15f);
    state->model.neg_half_inv_sigma_h_sq = -0.5f * state->model.inv_sigma_h_sq;

    /* Precompute mu_shift = mu_vol[k] * (1 - phi) */
    float one_minus_phi = 1.0f - state->model.phi;
    for (int i = 0; i < K; i++)
    {
        state->model.mu_shift[i] = state->model.mu_vol[i] * one_minus_phi;
    }

    /* Initialize transition learning defaults */
    state->prior_alpha = 1.0f;
    state->sticky_kappa = 10.0f;
    memset(state->n_trans, 0, sizeof(state->n_trans));

    /* Initialize adaptive kappa (DISABLED by default) */
    state->adaptive_kappa_enabled = 0;
    state->kappa_min = 20.0f;
    state->kappa_max = 500.0f;
    state->kappa_up_rate = 0.3f;   /* Legacy */
    state->kappa_down_rate = 0.1f; /* Legacy */
    state->last_chatter_ratio = 1.0f;
    state->rls_chatter_estimate = 1.0f; /* Start assuming prior matches data */
    state->rls_variance = 1.0f;         /* High initial uncertainty */
    state->rls_forgetting = 0.97f;      /* ~33 sweep effective window */
    state->last_off_diag_count = 0;
    state->last_total_count = 0;

    return state;
}

void pgas_mkl_free(PGASMKLState *state)
{
    if (!state)
        return;

    /* Free main RNG stream */
    if (state->rng.stream)
    {
        vslDeleteStream((VSLStreamStatePtr *)&state->rng.stream);
    }

    /* Free per-thread RNG streams and workspaces */
    for (int i = 0; i < state->n_thread_streams; i++)
    {
        if (state->thread_rng_streams[i])
        {
            vslDeleteStream((VSLStreamStatePtr *)&state->thread_rng_streams[i]);
        }
        /* Free per-thread workspaces */
        mkl_free(state->thread_ws[i].log_bw);
        mkl_free(state->thread_ws[i].bw);
        mkl_free(state->thread_ws[i].workspace);
        mkl_free(state->thread_ws[i].cumsum);
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
    mkl_free(state->ws_ocsn);
    mkl_free(state->ws_alias_prob);
    mkl_free(state->ws_alias_idx);
    mkl_free(state->ws_alias_small);
    mkl_free(state->ws_alias_large);
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
    if (!state)
        return;

    int K = state->K;
    state->model.K = K;

    /* Convert and compute log-trans + transposed version */
    for (int i = 0; i < K; i++)
    {
        for (int j = 0; j < K; j++)
        {
            float t = (float)trans[i * K + j];
            state->model.trans[i * K + j] = t;
            state->model.log_trans[i * K + j] = logf(t + EPS);
            /* Transposed: log_trans_T[j * K + i] = log_trans[i * K + j] */
            state->model.log_trans_T[j * K + i] = logf(t + EPS);
        }
    }

    /* Store phi and sigma_h first (needed for mu_shift) */
    state->model.phi = (float)phi;
    state->model.sigma_h = (float)sigma_h;
    state->model.inv_sigma_h_sq = 1.0f / ((float)sigma_h * (float)sigma_h);
    state->model.neg_half_inv_sigma_h_sq = -0.5f * state->model.inv_sigma_h_sq;

    /* Compute mu_vol and mu_shift = mu_vol[k] * (1 - phi) */
    float one_minus_phi = 1.0f - (float)phi;
    for (int i = 0; i < K; i++)
    {
        state->model.mu_vol[i] = (float)mu_vol[i];
        state->model.sigma_vol[i] = (float)sigma_vol[i];
        state->model.mu_shift[i] = (float)mu_vol[i] * one_minus_phi;
    }
}

void pgas_mkl_set_reference(PGASMKLState *state,
                            const int *regimes,
                            const double *h,
                            int T)
{
    if (!state)
        return;

    state->T = T;

    for (int t = 0; t < T; t++)
    {
        state->ref_regimes[t] = regimes[t];
        state->ref_h[t] = (float)h[t];
        state->ref_ancestors[t] = state->ref_idx;
    }
}

void pgas_mkl_load_observations(PGASMKLState *state,
                                const double *observations,
                                int T)
{
    if (!state)
        return;

    state->T = T;
    for (int t = 0; t < T; t++)
    {
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
    for (int i = 1; i < N; i++)
    {
        ws_cumsum[i] = ws_cumsum[i - 1] + weights[i];
    }

    /* Generate uniform random numbers */
    vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, n_samples, ws_uniform, 0.0f, 1.0f);

    /* Binary search for each sample */
    for (int s = 0; s < n_samples; s++)
    {
        float u = ws_uniform[s];
        int lo = 0, hi = N - 1;
        while (lo < hi)
        {
            int mid = (lo + hi) >> 1;
            if (ws_cumsum[mid] < u)
            {
                lo = mid + 1;
            }
            else
            {
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
    for (int i = 1; i < N; i++)
    {
        ws_cumsum[i] = ws_cumsum[i - 1] + weights[i];
    }

    float u;
    vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, 1, &u, 0.0f, 1.0f);

    int lo = 0, hi = N - 1;
    while (lo < hi)
    {
        int mid = (lo + hi) >> 1;
        if (ws_cumsum[mid] < u)
        {
            lo = mid + 1;
        }
        else
        {
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
    int i; /* MSVC OpenMP requires loop var declared outside */
#ifndef _MSC_VER
#pragma omp simd
#endif
    for (i = 0; i < N_padded; i++)
    {
        workspace[i] = log_weights[i] + neg_max;
    }

    /* Set padding to -inf so exp produces exact 0 */
    for (i = N; i < N_padded; i++)
    {
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
 * EMISSION PROBABILITY - OCSN 10-COMPONENT (OPTIMIZED)
 *
 * Two optimization strategies:
 *   1. MKL VML batch: Compute all 10×N log-components, single vsExp call
 *   2. AVX2 SIMD: Process 8 particles simultaneously with intrinsics
 *
 * Strategy selection:
 *   - N >= 64: Use MKL VML batch (better for large N due to vsExp efficiency)
 *   - N < 64:  Use AVX2 SIMD (lower overhead for small N)
 *═══════════════════════════════════════════════════════════════════════════════*/

/* Threshold for switching between AVX2 and MKL VML strategies
 * AVX2 is faster for typical particle counts due to better cache locality
 * MKL batch only wins for very large N where vsExp amortization helps
 */
#define OCSN_MKL_THRESHOLD 256

/* Thread-local workspace for MKL VML batch (avoid allocation in hot path) */
#define OCSN_MAX_BATCH_SIZE (PGAS_MKL_MAX_PARTICLES * OCSN_N_COMPONENTS)

/**
 * MKL VML batch strategy: Single vsExp call for all 10×N values
 *
 * Layout: log_comps[j * N_padded + i] = log-component j for particle i
 * This gives contiguous memory access for vsExp
 */
static void compute_log_emission_ocsn_mkl(float y, const float *h, int N, int N_padded,
                                          float *log_lik, float *workspace)
{
    /* workspace needs 10 * N_padded floats for log_comps + N_padded for exp_comps */
    float *log_comps = workspace;                 /* [10 × N_padded] */
    float *exp_comps = workspace + 10 * N_padded; /* [N_padded] temp */

    /* Phase 1: Compute all log-components and track max per particle */
    /* Initialize max to -inf */
    for (int i = 0; i < N_padded; i++)
    {
        log_lik[i] = NEG_INF; /* Reuse log_lik as max tracker temporarily */
    }

    for (int j = 0; j < OCSN_N_COMPONENTS; j++)
    {
        float log_const_j = OCSN_LOG_CONST[j];
        float inv_2v_j = OCSN_INV_2V[j];
        float mean_j = OCSN_MEAN[j];
        float *comp_j = &log_comps[j * N_padded];

        int i;
#ifndef _MSC_VER
#pragma omp simd
#endif
        for (i = 0; i < N; i++)
        {
            float y_base = y - h[i];
            float diff = y_base - mean_j;
            float log_comp = log_const_j - inv_2v_j * diff * diff;
            comp_j[i] = log_comp;

            /* Track max per particle */
            if (log_comp > log_lik[i])
            {
                log_lik[i] = log_comp;
            }
        }

        /* Set padding */
        for (i = N; i < N_padded; i++)
        {
            comp_j[i] = NEG_INF;
        }
    }

    /* Phase 2: Subtract max and prepare for exp */
    for (int j = 0; j < OCSN_N_COMPONENTS; j++)
    {
        float *comp_j = &log_comps[j * N_padded];
        int i;
#ifndef _MSC_VER
#pragma omp simd
#endif
        for (i = 0; i < N_padded; i++)
        {
            comp_j[i] = comp_j[i] - log_lik[i];
        }
    }

    /* Phase 3: Single MKL vsExp call for all 10×N_padded values */
    vsExp(10 * N_padded, log_comps, log_comps);

    /* Phase 4: Sum exp values per particle */
    /* Initialize sums to 0 */
    for (int i = 0; i < N_padded; i++)
    {
        exp_comps[i] = 0.0f;
    }

    for (int j = 0; j < OCSN_N_COMPONENTS; j++)
    {
        float *comp_j = &log_comps[j * N_padded];
        int i;
#ifndef _MSC_VER
#pragma omp simd
#endif
        for (i = 0; i < N_padded; i++)
        {
            exp_comps[i] += comp_j[i];
        }
    }

    /* Phase 5: log(sum) + max = final log-likelihood */
    vsLn(N_padded, exp_comps, exp_comps);

    int i;
#ifndef _MSC_VER
#pragma omp simd
#endif
    for (i = 0; i < N; i++)
    {
        log_lik[i] = log_lik[i] + exp_comps[i];
    }

    /* Set padding to NEG_INF */
    for (i = N; i < N_padded; i++)
    {
        log_lik[i] = NEG_INF;
    }
}

/* Enable AVX2 intrinsics if available
 * GCC/Clang: -mavx2 -mfma sets __AVX2__
 * MSVC: /arch:AVX2 - define PGAS_USE_AVX2 manually or check _MSC_VER + __AVX2__
 */
#if defined(__AVX2__) || defined(PGAS_USE_AVX2)
#define PGAS_HAS_AVX2 1
#include <immintrin.h>
#ifdef _MSC_VER
#include <intrin.h> /* For _BitScanForward */
#endif

/**
 * AVX2 SIMD strategy: Process 8 particles at once
 *
 * Uses polynomial approximation for exp() to avoid MKL call overhead
 * Accuracy: ~1e-5 relative error (sufficient for log-likelihood)
 */

/* Fast exp approximation using AVX2 (Schraudolph-style with refinement) */
static __m256 fast_exp_avx2(__m256 x)
{
    /* Clamp to avoid overflow/underflow */
    const __m256 min_val = _mm256_set1_ps(-87.0f);
    const __m256 max_val = _mm256_set1_ps(88.0f);
    x = _mm256_max_ps(x, min_val);
    x = _mm256_min_ps(x, max_val);

    /* exp(x) = 2^(x * log2(e)) = 2^(n + f) where n=integer, f=fraction */
    const __m256 log2e = _mm256_set1_ps(1.44269504089f);
    const __m256 ln2 = _mm256_set1_ps(0.693147180559945f);

    __m256 t = _mm256_mul_ps(x, log2e);
    __m256 n = _mm256_floor_ps(t);
    __m256 f = _mm256_sub_ps(t, n);

    /* 2^f using polynomial: 1 + f*ln2 + (f*ln2)^2/2 + (f*ln2)^3/6 + ... */
    __m256 f_ln2 = _mm256_mul_ps(f, ln2);
    const __m256 c2 = _mm256_set1_ps(0.5f);
    const __m256 c3 = _mm256_set1_ps(0.166666667f);
    const __m256 c4 = _mm256_set1_ps(0.041666667f);
    const __m256 c5 = _mm256_set1_ps(0.008333333f);

    __m256 f2 = _mm256_mul_ps(f_ln2, f_ln2);
    __m256 f3 = _mm256_mul_ps(f2, f_ln2);
    __m256 f4 = _mm256_mul_ps(f3, f_ln2);
    __m256 f5 = _mm256_mul_ps(f4, f_ln2);

    __m256 exp_f = _mm256_set1_ps(1.0f);
    exp_f = _mm256_add_ps(exp_f, f_ln2);
    exp_f = _mm256_fmadd_ps(f2, c2, exp_f);
    exp_f = _mm256_fmadd_ps(f3, c3, exp_f);
    exp_f = _mm256_fmadd_ps(f4, c4, exp_f);
    exp_f = _mm256_fmadd_ps(f5, c5, exp_f);

    /* Scale by 2^n using integer arithmetic */
    __m256i ni = _mm256_cvtps_epi32(n);
    ni = _mm256_add_epi32(ni, _mm256_set1_epi32(127));
    ni = _mm256_slli_epi32(ni, 23);
    __m256 scale = _mm256_castsi256_ps(ni);

    return _mm256_mul_ps(exp_f, scale);
}

/* Fast log approximation using AVX2 */
static __m256 fast_log_avx2(__m256 x)
{
    /* Extract exponent and mantissa */
    const __m256i exp_mask = _mm256_set1_epi32(0x7F800000);
    const __m256i mant_mask = _mm256_set1_epi32(0x007FFFFF);
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 ln2 = _mm256_set1_ps(0.693147180559945f);

    __m256i xi = _mm256_castps_si256(x);
    __m256i exp_i = _mm256_srli_epi32(_mm256_and_si256(xi, exp_mask), 23);
    exp_i = _mm256_sub_epi32(exp_i, _mm256_set1_epi32(127));
    __m256 exp_f = _mm256_cvtepi32_ps(exp_i);

    /* Normalized mantissa in [1, 2) */
    __m256i mant_i = _mm256_or_si256(_mm256_and_si256(xi, mant_mask),
                                     _mm256_set1_epi32(0x3F800000));
    __m256 mant = _mm256_castsi256_ps(mant_i);

    /* log(1+f) ≈ f - f²/2 + f³/3 for f = mant - 1 */
    __m256 f = _mm256_sub_ps(mant, one);
    __m256 f2 = _mm256_mul_ps(f, f);
    __m256 f3 = _mm256_mul_ps(f2, f);

    const __m256 c2 = _mm256_set1_ps(-0.5f);
    const __m256 c3 = _mm256_set1_ps(0.333333333f);
    const __m256 c4 = _mm256_set1_ps(-0.25f);

    __m256 log_mant = f;
    log_mant = _mm256_fmadd_ps(f2, c2, log_mant);
    log_mant = _mm256_fmadd_ps(f3, c3, log_mant);
    log_mant = _mm256_fmadd_ps(_mm256_mul_ps(f3, f), c4, log_mant);

    /* log(x) = log(2^exp * mant) = exp*ln2 + log(mant) */
    return _mm256_fmadd_ps(exp_f, ln2, log_mant);
}

static void compute_log_emission_ocsn_avx2(float y, const float *h, int N, int N_padded,
                                           float *log_lik)
{
    const __m256 y_vec = _mm256_set1_ps(y);
    const __m256 neg_inf = _mm256_set1_ps(NEG_INF);

    /* Load OCSN constants into registers */
    __m256 log_const[OCSN_N_COMPONENTS];
    __m256 inv_2v[OCSN_N_COMPONENTS];
    __m256 mean[OCSN_N_COMPONENTS];

    for (int j = 0; j < OCSN_N_COMPONENTS; j++)
    {
        log_const[j] = _mm256_set1_ps(OCSN_LOG_CONST[j]);
        inv_2v[j] = _mm256_set1_ps(OCSN_INV_2V[j]);
        mean[j] = _mm256_set1_ps(OCSN_MEAN[j]);
    }

    /* Process 8 particles at a time */
    int i;
    for (i = 0; i + 8 <= N; i += 8)
    {
        __m256 h_vec = _mm256_loadu_ps(&h[i]);
        __m256 y_base = _mm256_sub_ps(y_vec, h_vec);

        /* Compute all 10 log-components and track max */
        __m256 max_log = neg_inf;
        __m256 log_comps[OCSN_N_COMPONENTS];

        for (int j = 0; j < OCSN_N_COMPONENTS; j++)
        {
            __m256 diff = _mm256_sub_ps(y_base, mean[j]);
            __m256 diff_sq = _mm256_mul_ps(diff, diff);
            log_comps[j] = _mm256_fnmadd_ps(inv_2v[j], diff_sq, log_const[j]);
            max_log = _mm256_max_ps(max_log, log_comps[j]);
        }

        /* Subtract max and compute exp */
        __m256 sum = _mm256_setzero_ps();
        for (int j = 0; j < OCSN_N_COMPONENTS; j++)
        {
            __m256 shifted = _mm256_sub_ps(log_comps[j], max_log);
            __m256 exp_val = fast_exp_avx2(shifted);
            sum = _mm256_add_ps(sum, exp_val);
        }

        /* log(sum) + max = final log-likelihood */
        __m256 log_sum = fast_log_avx2(sum);
        __m256 result = _mm256_add_ps(max_log, log_sum);

        _mm256_storeu_ps(&log_lik[i], result);
    }

    /* Handle remainder with scalar */
    for (; i < N; i++)
    {
        log_lik[i] = ocsn_log_likelihood_single(y, h[i]);
    }

    /* Set padding to NEG_INF */
    for (i = N; i < N_padded; i++)
    {
        log_lik[i] = NEG_INF;
    }
}

#endif /* PGAS_HAS_AVX2 */

/**
 * Compute log emission using OCSN 10-component mixture (dispatcher)
 *
 * Automatically selects best strategy based on N and available SIMD
 */
static void compute_log_emission_ocsn(float y, const float *h, int N, int N_padded,
                                      float *log_lik, float *workspace)
{
    /* Ensure OCSN constants are initialized */
    init_ocsn_constants();

#ifdef PGAS_HAS_AVX2
    if (N < OCSN_MKL_THRESHOLD)
    {
        /* Small N: AVX2 has lower overhead */
        compute_log_emission_ocsn_avx2(y, h, N, N_padded, log_lik);
    }
    else
    {
        /* Large N: MKL VML batch is more efficient */
        compute_log_emission_ocsn_mkl(y, h, N, N_padded, log_lik, workspace);
    }
#else
    /* No AVX2: Always use MKL */
    compute_log_emission_ocsn_mkl(y, h, N, N_padded, log_lik, workspace);
#endif
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
    const int Np = state->N_padded; /* Use padded stride! */
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
    for (int n = 0; n < N; n++)
    {
        if (n == ref_idx)
        {
            state->regimes[n] = state->ref_regimes[0];
            state->h[n] = state->ref_h[0];
        }
        else
        {
            int regime = rand_regimes[n];
            state->regimes[n] = regime;
            state->h[n] = state->model.mu_vol[regime] + state->ws_normal[n];
        }
        state->ancestors[n] = n;
    }

    /* Zero padding */
    for (int n = N; n < Np; n++)
    {
        state->regimes[n] = 0;
        state->h[n] = 0.0f;
        state->ancestors[n] = 0;
    }

    /* Compute initial weights using OCSN emission */
    compute_log_emission_ocsn(state->observations[0], state->h, N, Np,
                              state->log_weights, state->ws_ocsn);

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

    /* Use precomputed mu_shift instead of mu_k + phi * (h - mu_k) */
    float mu_shift_k = m->mu_shift[ref_regime];
    float phi = m->phi;
    float neg_half_inv_var = m->neg_half_inv_sigma_h_sq;

    float *log_as = state->ws_log_bw;

    /* Previous time data (stride = Np) */
    float *prev_log_w = &state->log_weights[(t - 1) * Np];
    float *prev_h = &state->h[(t - 1) * Np];
    int *prev_regimes = &state->regimes[(t - 1) * Np];

    /* Use transposed log_trans for contiguous column access */
    const float *log_trans_col = &m->log_trans_T[ref_regime * K];

    /* Compute AS log-weights */
    int n;
#ifndef _MSC_VER
#pragma omp simd
#endif
    for (n = 0; n < N; n++)
    {
        int regime_n = prev_regimes[n];
        float h_n = prev_h[n];

        /* Contiguous access */
        float log_trans = log_trans_col[regime_n];

        /* Rank-1 arithmetic: mean = mu_shift + phi * h_n */
        float mean = mu_shift_k + phi * h_n;
        float diff = ref_h - mean;
        float log_h_trans = neg_half_inv_var * diff * diff;

        log_as[n] = prev_log_w[n] + log_trans + log_h_trans;
    }

    /* Set padding to NEG_INF */
    for (n = N; n < Np; n++)
    {
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
    if (!state || state->T < 2)
        return 0.0f;

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
    for (int t = 1; t < T; t++)
    {
        /* Previous time data (stride = Np!) */
        float *prev_weights = &state->weights[(t - 1) * Np];
        float *prev_h = &state->h[(t - 1) * Np];
        int *prev_regimes = &state->regimes[(t - 1) * Np];

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
        if (new_ref_anc != old_ref_anc)
        {
            state->ancestor_accepts++;
            state->ref_ancestors[t] = new_ref_anc;
        }
        curr_ancestors[ref_idx] = state->ref_ancestors[t];

        /* Generate random numbers for propagation */
        vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, stream, N,
                      state->ws_normal, 0.0f, m->sigma_h);

        /* Propagate particles */
        for (int n = 0; n < N; n++)
        {
            if (n == ref_idx)
            {
                curr_regimes[n] = state->ref_regimes[t];
                curr_h[n] = state->ref_h[t];
            }
            else
            {
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
        for (int n = N; n < Np; n++)
        {
            curr_regimes[n] = 0;
            curr_h[n] = 0.0f;
            curr_ancestors[n] = 0;
        }

        /* Compute weights using OCSN emission (stride = Np!) */
        compute_log_emission_ocsn(state->observations[t], curr_h, N, Np,
                                  &state->log_weights[t * Np], state->ws_ocsn);

        logsumexp_normalize_mkl(&state->log_weights[t * Np], N, Np,
                                &state->weights[t * Np], state->ws_bw);
    }

    /* Sample final trajectory and update reference */
    int final_idx = sample_categorical_single(&state->weights[(T - 1) * Np], N,
                                              state->ws_cumsum, stream);

    /* Trace back (stride = Np!) */
    int idx = final_idx;
    for (int t = T - 1; t >= 0; t--)
    {
        state->ref_regimes[t] = state->regimes[t * Np + idx];
        state->ref_h[t] = state->h[t * Np + idx];
        if (t > 0)
        {
            idx = state->ancestors[t * Np + idx];
        }
    }

    state->acceptance_rate = (state->ancestor_proposals > 0) ? (float)state->ancestor_accepts / state->ancestor_proposals : 0.0f;

    state->current_sweep++;
    state->total_sweeps++;

    return state->acceptance_rate;
}

/**
 * Adaptive PGAS
 */
int pgas_mkl_run_adaptive(PGASMKLState *state)
{
    if (!state || state->T < 2)
        return 2;

    state->current_sweep = 0;

    /* Minimum sweeps */
    for (int s = 0; s < MIN_SWEEPS; s++)
    {
        pgas_mkl_csmc_sweep(state);
    }

    if (state->acceptance_rate >= TARGET_ACCEPTANCE)
    {
        return 0;
    }

    /* Continue until target or max */
    while (state->current_sweep < MAX_SWEEPS)
    {
        pgas_mkl_csmc_sweep(state);

        if (state->acceptance_rate >= TARGET_ACCEPTANCE)
        {
            return 0;
        }
    }

    if (state->acceptance_rate < ABORT_ACCEPTANCE)
    {
        return 2;
    }

    return 1;
}

/*═══════════════════════════════════════════════════════════════════════════════
 * TRANSITION MATRIX LEARNING (NEW)
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * Sample transition matrix from Dirichlet posterior
 *
 * π_i ~ Dirichlet(α + n_{i1}, ..., α + n_{iK} + κ·I(j=i))
 */
void pgas_mkl_sample_transitions(PGASMKLState *state)
{
    if (!state || state->T < 2)
        return;

    const int K = state->K;
    const int T = state->T;
    VSLStreamStatePtr stream = (VSLStreamStatePtr)state->rng.stream;

    /* Count transitions from reference trajectory */
    memset(state->n_trans, 0, K * K * sizeof(int));
    for (int t = 1; t < T; t++)
    {
        int s_prev = state->ref_regimes[t - 1];
        int s_curr = state->ref_regimes[t];
        if (s_prev >= 0 && s_prev < K && s_curr >= 0 && s_curr < K)
        {
            state->n_trans[s_prev * K + s_curr]++;
        }
    }

    /* ═══════════════════════════════════════════════════════════════════
     * CHATTER RATIO COMPUTATION
     *
     * Chatter ratio = observed_off_diagonal / expected_off_diagonal
     *
     * Expected off-diagonal based on current kappa:
     *   expected_diag ≈ (kappa + alpha) / (kappa + K*alpha)
     *   expected_off_diag_rate = 1 - expected_diag
     *   expected_off_diag_count = (T-1) * expected_off_diag_rate
     * ═══════════════════════════════════════════════════════════════════*/

    int off_diag_count = 0;
    int total_count = 0;
    for (int i = 0; i < K; i++)
    {
        for (int j = 0; j < K; j++)
        {
            total_count += state->n_trans[i * K + j];
            if (i != j)
            {
                off_diag_count += state->n_trans[i * K + j];
            }
        }
    }

    /* Store for diagnostics */
    state->last_off_diag_count = off_diag_count;
    state->last_total_count = total_count;

    /* Compute expected off-diagonal rate from prior */
    float expected_diag = (state->sticky_kappa + state->prior_alpha) /
                          (state->sticky_kappa + K * state->prior_alpha);
    float expected_off_diag_count = (float)(T - 1) * (1.0f - expected_diag);

    /* Chatter ratio: >1 means more switching than prior expects */
    if (expected_off_diag_count > 0.1f)
    {
        state->last_chatter_ratio = (float)off_diag_count / expected_off_diag_count;
    }
    else
    {
        state->last_chatter_ratio = 1.0f;
    }

    /* ═══════════════════════════════════════════════════════════════════
     * ADAPTIVE KAPPA UPDATE (if enabled)
     *
     * Method: CHATTER-CORRECTED MOMENT MATCHING with RLS SMOOTHING
     *
     * Key insight: Single-sweep chatter is NOISY (OCSN σ≈2.2).
     * Solution: Use Recursive Least Squares (RLS) with forgetting factor
     *           to estimate the "true" underlying chatter ratio.
     *
     * RLS is a Kalman filter for tracking a slowly-varying parameter:
     *   K = P / (λ + P)                    // Adaptive gain
     *   θ̂ = θ̂ + K * (y - θ̂)              // Update estimate
     *   P = (1/λ) * (P - K*P)              // Update variance
     *
     * λ = forgetting factor (0.97 → ~33 sweep effective window)
     *
     * Benefits over EMA:
     *   - Adaptive gain (trusts estimate more when confident)
     *   - Principled (optimal for tracking non-stationary signals)
     *   - Tracks estimation uncertainty
     *
     * Reference: Ljung & Söderström, "Theory and Practice of RLS"
     * ═══════════════════════════════════════════════════════════════════*/

    if (state->adaptive_kappa_enabled)
    {
        const int K = state->K;
        const float alpha = state->prior_alpha;

        /* 1. Get current diagnostics */
        int total_switches = state->last_off_diag_count;
        int total_obs = state->last_total_count;

        if (total_obs < 10)
            return; /* Need minimum data */

        /* 2. Expected switch rate from current prior */
        float expected_diag = (state->sticky_kappa + alpha) /
                              (state->sticky_kappa + K * alpha);
        float expected_switch_rate = 1.0f - expected_diag;

        /* 3. Observed switch rate */
        float observed_switch_rate = (float)total_switches / (float)total_obs;

        /* 4. Raw Chatter Ratio (with Laplace smoothing) */
        float laplace = 0.5f / (float)total_obs;
        float raw_chatter = (observed_switch_rate + laplace) /
                            (expected_switch_rate + laplace);

        /* 5. RLS Update for chatter estimate
         * This is optimal for tracking a slowly-varying signal */
        float lambda = state->rls_forgetting; /* 0.97 default */
        float P = state->rls_variance;
        float theta = state->rls_chatter_estimate;

        /* Kalman gain */
        float rls_gain = P / (lambda + P);

        /* Update estimate */
        float innovation = raw_chatter - theta;
        theta = theta + rls_gain * innovation;

        /* Update variance (with forgetting) */
        P = (1.0f / lambda) * (P - rls_gain * P);

        /* Prevent variance collapse (maintain adaptivity) */
        if (P < 0.01f)
            P = 0.01f;
        if (P > 10.0f)
            P = 10.0f;

        /* Store RLS state */
        state->rls_chatter_estimate = theta;
        state->rls_variance = P;
        state->last_chatter_ratio = theta; /* For diagnostics */

        /* 6. Clamp smoothed chatter to prevent extreme swings */
        float chatter = theta;
        if (chatter < 0.3f)
            chatter = 0.3f;
        if (chatter > 3.0f)
            chatter = 3.0f;

        /* 7. Calculate Corrected Target Diagonal */
        float target_switch_rate = expected_switch_rate * chatter;
        float target_diag = 1.0f - target_switch_rate;

        /* Clamp to valid range [1/K + ε, 0.999] */
        float min_diag = 1.0f / K + 0.01f;
        if (target_diag < min_diag)
            target_diag = min_diag;
        if (target_diag > 0.999f)
            target_diag = 0.999f;

        /* 8. Map target_diag back to Kappa (The Oracle) */
        float kappa_oracle = alpha * (target_diag * K - 1.0f) / (1.0f - target_diag);

        /* Clamp oracle to bounds before log */
        if (kappa_oracle < state->kappa_min)
            kappa_oracle = state->kappa_min;
        if (kappa_oracle > state->kappa_max)
            kappa_oracle = state->kappa_max;

        /* 9. Log-Space Momentum Update
         * Moderate momentum (0.8) since RLS already smooths */
        float momentum = 0.8f;
        float log_kappa_new = momentum * logf(state->sticky_kappa) +
                              (1.0f - momentum) * logf(kappa_oracle);
        float kappa_new = expf(log_kappa_new);

        /* 10. Clamp to user bounds */
        if (kappa_new < state->kappa_min)
            kappa_new = state->kappa_min;
        if (kappa_new > state->kappa_max)
            kappa_new = state->kappa_max;

        state->sticky_kappa = kappa_new;
    }

    /* Sample each row from Dirichlet (via Gamma samples) */
    float gamma_samples[PGAS_MKL_MAX_K];

    for (int i = 0; i < K; i++)
    {
        float row_sum = 0.0f;

        for (int j = 0; j < K; j++)
        {
            /* Dirichlet parameter = prior + counts + sticky bonus */
            float alpha_j = state->prior_alpha + (float)state->n_trans[i * K + j];
            if (i == j)
            {
                alpha_j += state->sticky_kappa; /* Sticky prior */
            }

            /* Ensure alpha > 0 */
            if (alpha_j < 0.01f)
                alpha_j = 0.01f;

            /* Sample Gamma(alpha, 1) */
            vsRngGamma(VSL_RNG_METHOD_GAMMA_GNORM, stream, 1,
                       &gamma_samples[j], alpha_j, 0.0f, 1.0f);
            row_sum += gamma_samples[j];
        }

        /* Normalize to get Dirichlet sample */
        float inv_sum = 1.0f / (row_sum + EPS);
        for (int j = 0; j < K; j++)
        {
            float p = gamma_samples[j] * inv_sum;
            state->model.trans[i * K + j] = p;
            state->model.log_trans[i * K + j] = logf(p + EPS);
            state->model.log_trans_T[j * K + i] = state->model.log_trans[i * K + j];
        }
    }
}

/**
 * Full Gibbs sweep: CSMC for states + Dirichlet for transitions
 */
float pgas_mkl_gibbs_sweep(PGASMKLState *state)
{
    if (!state || state->T < 2)
        return 0.0f;

    /* 1. Sample states given transitions (CSMC) */
    float accept = pgas_mkl_csmc_sweep(state);

    /* 2. Sample transitions given states (Dirichlet) */
    pgas_mkl_sample_transitions(state);

    return accept;
}

/*═══════════════════════════════════════════════════════════════════════════════
 * PARIS BACKWARD SMOOTHING (PHASE 2 OPTIMIZATIONS)
 *
 * Fixes applied:
 *   1. Inlined AVX2 logsumexp - eliminates 4 MKL calls per particle
 *   2. Walker's Alias Sampling - O(1) branchless sampling
 *   3. Batch RNG per timestep - reduces dispatcher calls
 *   4. schedule(static) - eliminates dynamic work-stealing overhead
 *═══════════════════════════════════════════════════════════════════════════════*/

/* Workload threshold: below this, sequential is faster than parallel overhead */
#define PARIS_PARALLEL_THRESHOLD 4096 /* N * T */

#ifdef PGAS_HAS_AVX2

/**
 * Inlined AVX2 logsumexp + normalize
 *
 * Replaces logsumexp_normalize_mkl to avoid 4 MKL function calls
 * All computation stays in AVX2 registers
 */
static inline void logsumexp_normalize_avx2(const float *log_weights, int N, int N_padded,
                                            float *weights)
{
    /* Phase 1: Find max using AVX2 */
    __m256 max_vec = _mm256_set1_ps(NEG_INF);
    int i;
    for (i = 0; i + 8 <= N; i += 8)
    {
        __m256 v = _mm256_loadu_ps(&log_weights[i]);
        max_vec = _mm256_max_ps(max_vec, v);
    }

    /* Horizontal max reduction */
    __m128 hi = _mm256_extractf128_ps(max_vec, 1);
    __m128 lo = _mm256_castps256_ps128(max_vec);
    __m128 max128 = _mm_max_ps(hi, lo);
    max128 = _mm_max_ps(max128, _mm_shuffle_ps(max128, max128, _MM_SHUFFLE(1, 0, 3, 2)));
    max128 = _mm_max_ps(max128, _mm_shuffle_ps(max128, max128, _MM_SHUFFLE(2, 3, 0, 1)));
    float max_val = _mm_cvtss_f32(max128);

    /* Handle remainder */
    for (; i < N; i++)
    {
        if (log_weights[i] > max_val)
            max_val = log_weights[i];
    }

    /* Phase 2: Subtract max, exp, and accumulate sum */
    __m256 neg_max = _mm256_set1_ps(-max_val);
    __m256 sum_vec = _mm256_setzero_ps();

    for (i = 0; i + 8 <= N; i += 8)
    {
        __m256 v = _mm256_loadu_ps(&log_weights[i]);
        __m256 shifted = _mm256_add_ps(v, neg_max);
        __m256 exp_v = fast_exp_avx2(shifted);
        _mm256_storeu_ps(&weights[i], exp_v);
        sum_vec = _mm256_add_ps(sum_vec, exp_v);
    }

    /* Horizontal sum reduction */
    hi = _mm256_extractf128_ps(sum_vec, 1);
    lo = _mm256_castps256_ps128(sum_vec);
    __m128 sum128 = _mm_add_ps(hi, lo);
    sum128 = _mm_add_ps(sum128, _mm_shuffle_ps(sum128, sum128, _MM_SHUFFLE(1, 0, 3, 2)));
    sum128 = _mm_add_ps(sum128, _mm_shuffle_ps(sum128, sum128, _MM_SHUFFLE(2, 3, 0, 1)));
    float sum = _mm_cvtss_f32(sum128);

    /* Handle remainder */
    for (; i < N; i++)
    {
        float exp_v = expf(log_weights[i] - max_val);
        weights[i] = exp_v;
        sum += exp_v;
    }

    /* Phase 3: Normalize */
    float inv_sum = 1.0f / sum;
    __m256 inv_sum_vec = _mm256_set1_ps(inv_sum);

    for (i = 0; i + 8 <= N; i += 8)
    {
        __m256 v = _mm256_loadu_ps(&weights[i]);
        v = _mm256_mul_ps(v, inv_sum_vec);
        _mm256_storeu_ps(&weights[i], v);
    }

    for (; i < N; i++)
    {
        weights[i] *= inv_sum;
    }

    /* Zero padding */
    for (i = N; i < N_padded; i++)
    {
        weights[i] = 0.0f;
    }
}

/**
 * Walker's Alias Table for O(1) categorical sampling
 *
 * Build phase: O(N) - done once per particle
 * Sample phase: O(1) - branchless table lookup
 */
typedef struct
{
    float *prob; /* [N] probability table */
    int *alias;  /* [N] alias indices */
    int N;
} AliasTable;

/**
 * Build alias table from normalized weights
 * Uses Vose's algorithm (numerically stable)
 */
static void alias_build(AliasTable *table, const float *weights, int N,
                        int *small_stack, int *large_stack)
{
    table->N = N;
    float n_f = (float)N;

    /* Initialize prob = weights * N, classify as small or large */
    int n_small = 0, n_large = 0;

    for (int i = 0; i < N; i++)
    {
        float p = weights[i] * n_f;
        table->prob[i] = p;
        table->alias[i] = i;

        if (p < 1.0f)
        {
            small_stack[n_small++] = i;
        }
        else
        {
            large_stack[n_large++] = i;
        }
    }

    /* Pair small with large until done */
    while (n_small > 0 && n_large > 0)
    {
        int s = small_stack[--n_small];
        int l = large_stack[--n_large];

        table->alias[s] = l;
        table->prob[l] = table->prob[l] + table->prob[s] - 1.0f;

        if (table->prob[l] < 1.0f)
        {
            small_stack[n_small++] = l;
        }
        else
        {
            large_stack[n_large++] = l;
        }
    }

    /* Handle numerical edge cases */
    while (n_large > 0)
    {
        table->prob[large_stack[--n_large]] = 1.0f;
    }
    while (n_small > 0)
    {
        table->prob[small_stack[--n_small]] = 1.0f;
    }
}

/**
 * O(1) branchless sample from alias table
 */
static inline int alias_sample(const AliasTable *table, float u1, float u2)
{
    int idx = (int)(u1 * table->N);
    if (idx >= table->N)
        idx = table->N - 1; /* Safety clamp */

    /* Branchless select: if u2 < prob[idx], return idx, else return alias[idx] */
    int use_alias = (u2 >= table->prob[idx]);
    return use_alias ? table->alias[idx] : idx;
}

#endif /* PGAS_HAS_AVX2 */

void pgas_paris_backward_smooth(PGASMKLState *state)
{
    if (!state || state->T < 2)
        return;

    const int N = state->N;
    const int Np = state->N_padded;
    const int T = state->T;
    const int K = state->model.K;
    const PGASMKLModel *m = &state->model;

    /* Precomputed constants */
    const float phi = m->phi;
    const float neg_half_inv_var = m->neg_half_inv_sigma_h_sq;

    /* Initialize at final time (stride = Np!) */
    for (int n = 0; n < N; n++)
    {
        state->smoothed[(T - 1) * Np + n] = n;
    }

#ifdef PGAS_HAS_AVX2
    /* ═══════════════════════════════════════════════════════════════════
     * OPTIMIZED PATH: AVX2 logsumexp + Walker's alias sampling
     * ═══════════════════════════════════════════════════════════════════*/

    /* Setup alias table (reuses workspace) */
    AliasTable alias_table;
    alias_table.prob = state->ws_alias_prob;
    alias_table.alias = state->ws_alias_idx;

    float *local_log_bw = state->ws_log_bw;
    float *local_bw = state->ws_bw;
    VSLStreamStatePtr stream = (VSLStreamStatePtr)state->rng.stream;

    /* Pre-generate random numbers for this timestep batch */
    float *rand_u1 = state->ws_uniform;
    float *rand_u2 = state->ws_normal;

    int t, n, i;
    for (t = T - 2; t >= 0; t--)
    {
        const float *h_t = &state->h[t * Np];
        const float *log_w_t = &state->log_weights[t * Np];
        const int *regimes_t = &state->regimes[t * Np];

        /* Generate 2*N uniform random numbers for all particles at this timestep */
        vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, N, rand_u1, 0.0f, 1.0f);
        vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, N, rand_u2, 0.0f, 1.0f);

        for (n = 0; n < N; n++)
        {
            int idx_next = state->smoothed[(t + 1) * Np + n];
            int regime_next = state->regimes[(t + 1) * Np + idx_next];
            float h_next = state->h[(t + 1) * Np + idx_next];

            /* Use precomputed mu_shift */
            float mu_shift_k = m->mu_shift[regime_next];

            /* Get log_trans column from TRANSPOSED matrix */
            const float *log_trans_col = &m->log_trans_T[regime_next * K];

            /* Compute backward log-weights with SIMD */
#ifndef _MSC_VER
#pragma omp simd
#endif
            for (i = 0; i < N; i++)
            {
                int regime_i = regimes_t[i];
                float h_i = h_t[i];

                float log_trans = log_trans_col[regime_i];
                float mean = mu_shift_k + phi * h_i;
                float diff = h_next - mean;
                float log_h_trans = neg_half_inv_var * diff * diff;

                local_log_bw[i] = log_w_t[i] + log_trans + log_h_trans;
            }

            for (i = N; i < Np; i++)
            {
                local_log_bw[i] = NEG_INF;
            }

            /* Inlined AVX2 logsumexp (eliminates 4 MKL function calls) */
            logsumexp_normalize_avx2(local_log_bw, N, Np, local_bw);

            /* Build alias table O(N) */
            alias_build(&alias_table, local_bw, N,
                        state->ws_alias_small, state->ws_alias_large);

            /* O(1) branchless sample */
            state->smoothed[t * Np + n] = alias_sample(&alias_table,
                                                       rand_u1[n], rand_u2[n]);
        }
    }

#else
    /* ═══════════════════════════════════════════════════════════════════
     * FALLBACK PATH: MKL-based (when AVX2 not available)
     * ═══════════════════════════════════════════════════════════════════*/

    /* Decide parallel vs sequential based on workload */
    const int use_parallel = (N * T > PARIS_PARALLEL_THRESHOLD);

#ifdef _OPENMP
    if (use_parallel)
    {
#pragma omp parallel
        {
            int tid = omp_get_thread_num();

            vmlSetMode(VML_EP | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE);

            VSLStreamStatePtr local_stream = (VSLStreamStatePtr)state->thread_rng_streams[tid];
            float *local_log_bw = state->thread_ws[tid].log_bw;
            float *local_bw = state->thread_ws[tid].bw;
            float *local_workspace = state->thread_ws[tid].workspace;
            float *local_cumsum = state->thread_ws[tid].cumsum;

            int t, n;
            for (t = T - 2; t >= 0; t--)
            {
                const float *h_t = &state->h[t * Np];
                const float *log_w_t = &state->log_weights[t * Np];
                const int *regimes_t = &state->regimes[t * Np];

#pragma omp for schedule(static)
                for (n = 0; n < N; n++)
                {
                    int idx_next = state->smoothed[(t + 1) * Np + n];
                    int regime_next = state->regimes[(t + 1) * Np + idx_next];
                    float h_next = state->h[(t + 1) * Np + idx_next];

                    float mu_shift_k = m->mu_shift[regime_next];
                    const float *log_trans_col = &m->log_trans_T[regime_next * K];

                    int i;
#ifndef _MSC_VER
#pragma omp simd
#endif
                    for (i = 0; i < N; i++)
                    {
                        int regime_i = regimes_t[i];
                        float h_i = h_t[i];
                        float log_trans = log_trans_col[regime_i];
                        float mean = mu_shift_k + phi * h_i;
                        float diff = h_next - mean;
                        float log_h_trans = neg_half_inv_var * diff * diff;
                        local_log_bw[i] = log_w_t[i] + log_trans + log_h_trans;
                    }

                    for (i = N; i < Np; i++)
                    {
                        local_log_bw[i] = NEG_INF;
                    }

                    logsumexp_normalize_mkl(local_log_bw, N, Np, local_bw, local_workspace);

                    state->smoothed[t * Np + n] = sample_categorical_single(
                        local_bw, N, local_cumsum, local_stream);
                }
            }

            vmlSetMode(VML_HA);
        }
    }
    else
#endif
    {
        vmlSetMode(VML_EP | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE);

        float *local_log_bw = state->ws_log_bw;
        float *local_bw = state->ws_bw;
        float *local_workspace = state->ws_uniform;
        VSLStreamStatePtr stream = (VSLStreamStatePtr)state->rng.stream;

        int t, n, i;
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

                float mu_shift_k = m->mu_shift[regime_next];
                const float *log_trans_col = &m->log_trans_T[regime_next * K];

#ifndef _MSC_VER
#pragma omp simd
#endif
                for (i = 0; i < N; i++)
                {
                    int regime_i = regimes_t[i];
                    float h_i = h_t[i];
                    float log_trans = log_trans_col[regime_i];
                    float mean = mu_shift_k + phi * h_i;
                    float diff = h_next - mean;
                    float log_h_trans = neg_half_inv_var * diff * diff;
                    local_log_bw[i] = log_w_t[i] + log_trans + log_h_trans;
                }

                for (i = N; i < Np; i++)
                {
                    local_log_bw[i] = NEG_INF;
                }

                logsumexp_normalize_mkl(local_log_bw, N, Np, local_bw, local_workspace);

                state->smoothed[t * Np + n] = sample_categorical_single(
                    local_bw, N, state->ws_cumsum, stream);
            }
        }

        vmlSetMode(VML_HA);
    }
#endif /* PGAS_HAS_AVX2 */
}

void pgas_paris_get_smoothed(const PGASMKLState *state, int t,
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

/*═══════════════════════════════════════════════════════════════════════════════
 * LIFEBOAT
 *═══════════════════════════════════════════════════════════════════════════════*/

void pgas_mkl_generate_lifeboat(const PGASMKLState *state, LifeboatPacketMKL *packet)
{
    if (!state || !packet)
        return;

    packet->K = state->K;
    packet->N = state->N;
    packet->T = state->T;

    memcpy(packet->trans, state->model.trans, state->K * state->K * sizeof(float));
    memcpy(packet->mu_vol, state->model.mu_vol, state->K * sizeof(float));
    memcpy(packet->sigma_vol, state->model.sigma_vol, state->K * sizeof(float));
    packet->phi = state->model.phi;
    packet->sigma_h = state->model.sigma_h;

    if (!packet->final_regimes)
    {
        packet->final_regimes = (int *)mkl_malloc(state->N * sizeof(int), PGAS_MKL_ALIGN);
        packet->final_h = (float *)mkl_malloc(state->N * sizeof(float), PGAS_MKL_ALIGN);
        packet->final_weights = (float *)mkl_malloc(state->N * sizeof(float), PGAS_MKL_ALIGN);
    }

    pgas_paris_get_smoothed(state, state->T - 1, packet->final_regimes, packet->final_h);

    float unif_w = 1.0f / state->N;
    for (int n = 0; n < state->N; n++)
    {
        packet->final_weights[n] = unif_w;
    }

    packet->ancestor_acceptance = state->acceptance_rate;
    packet->sweeps_used = state->current_sweep;
}

bool lifeboat_mkl_validate(const LifeboatPacketMKL *packet)
{
    if (!packet)
        return false;

    if (packet->ancestor_acceptance < ABORT_ACCEPTANCE)
        return false;

    for (int i = 0; i < packet->K; i++)
    {
        float sum = 0;
        for (int j = 0; j < packet->K; j++)
        {
            float p = packet->trans[i * packet->K + j];
            if (p < 0 || p > 1)
                return false;
            sum += p;
        }
        if (fabsf(sum - 1.0f) > 1e-4f)
            return false;
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
    if (!state || t < 0 || t >= state->T)
        return 0.0f;

    const int N = state->N;
    const int Np = state->N_padded;
    const float *weights = &state->weights[t * Np];

    float sum_sq = cblas_sdot(N, weights, 1, weights, 1);

    return (sum_sq > EPS) ? 1.0f / sum_sq : 0.0f;
}

void pgas_mkl_print_diagnostics(const PGASMKLState *state)
{
    if (!state)
        return;

    printf("═══════════════════════════════════════════════════════════\n");
    printf("PGAS-MKL DIAGNOSTICS (OCSN 10-component)\n");
    printf("═══════════════════════════════════════════════════════════\n");
    printf("Particles:          %d (padded: %d)\n", state->N, state->N_padded);
    printf("Regimes:            %d\n", state->K);
    printf("Buffer length:      %d\n", state->T);
    printf("Thread RNG streams: %d\n", state->n_thread_streams);
    printf("Sweeps completed:   %d\n", state->current_sweep);
    printf("Ancestor accepts:   %d / %d\n", state->ancestor_accepts, state->ancestor_proposals);
    printf("Acceptance rate:    %.3f %s\n", state->acceptance_rate,
           state->acceptance_rate >= TARGET_ACCEPTANCE ? "(CONVERGED)" : state->acceptance_rate >= ABORT_ACCEPTANCE ? "(MIXING)"
                                                                                                                    : "(STUCK)");
    printf("Final ESS:          %.1f / %d\n", pgas_mkl_get_ess(state, state->T - 1), state->N);
    printf("Transition prior:   α=%.2f, κ=%.2f (sticky)\n", state->prior_alpha, state->sticky_kappa);
    printf("═══════════════════════════════════════════════════════════\n");
}