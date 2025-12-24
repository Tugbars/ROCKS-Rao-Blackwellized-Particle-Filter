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
 *
 * SAFETY: Returns -1e20f instead of -Inf/NaN to prevent weight collapse
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

    float result = max_log + logf(sum);

    /* Guard against NaN/Inf - return large negative instead */
    if (!isfinite(result) || result < -1e20f)
    {
        return -1e20f;
    }

    return result;
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
        state->model.sigma_vol[i] = 0.15f; /* Default AR process noise */

        /* Precompute per-regime AR likelihood constants */
        float sv = state->model.sigma_vol[i];
        state->model.inv_sigma_vol_sq[i] = 1.0f / (sv * sv);
        state->model.neg_half_inv_sigma_vol_sq[i] = -0.5f / (sv * sv);
    }
    state->model.phi = 0.97f;

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

/**
 * Set model parameters
 *
 * ALIGNED WITH RBPF: Uses per-regime sigma_vol[k] for AR process noise.
 *
 * AR dynamics: h_t = μ_k × (1-φ) + φ × h_{t-1} + σ_vol[k] × ε_t
 *
 * The sigma_vol parameter now means AR process noise (NOT emission spread).
 * This matches the RBPF model where crisis regimes have higher process noise.
 *
 * @param state      PGAS state
 * @param trans      Transition matrix [K×K] row-major (double for API compat)
 * @param mu_vol     Regime means [K]
 * @param sigma_vol  Per-regime AR process noise [K]
 * @param phi        AR(1) persistence (shared across regimes)
 */
void pgas_mkl_set_model(PGASMKLState *state,
                        const double *trans,
                        const double *mu_vol,
                        const double *sigma_vol,
                        double phi)
{
    if (!state)
        return;

    int K = state->K;
    state->model.K = K;

    /* Convert and compute log-trans + transposed version
     * Transposed matrix enables contiguous column access in ancestor sampling */
    for (int i = 0; i < K; i++)
    {
        for (int j = 0; j < K; j++)
        {
            float t = (float)trans[i * K + j];
            state->model.trans[i * K + j] = t;
            state->model.log_trans[i * K + j] = logf(t + EPS);
            /* Transposed: log_trans_T[j * K + i] = log_trans[i * K + j]
             * This gives contiguous access when iterating over row i for column j */
            state->model.log_trans_T[j * K + i] = logf(t + EPS);
        }
    }

    /* Store phi (AR persistence - shared across regimes) */
    state->model.phi = (float)phi;

    /* Compute mu_vol, sigma_vol, mu_shift, and per-regime AR constants
     *
     * Rank-1 optimization: Precompute mu_shift = μ_k × (1-φ)
     * This allows AR mean computation as: mean = mu_shift + φ × h
     * instead of: mean = μ_k + φ × (h - μ_k)
     */
    float one_minus_phi = 1.0f - (float)phi;
    for (int i = 0; i < K; i++)
    {
        state->model.mu_vol[i] = (float)mu_vol[i];
        state->model.sigma_vol[i] = (float)sigma_vol[i];
        state->model.mu_shift[i] = (float)mu_vol[i] * one_minus_phi;

        /* Precompute per-regime AR likelihood constants
         * These are used in ancestor sampling and PARIS backward smoothing:
         *   log P(h_next | h, k) = -0.5 × (h_next - mean)² / σ_vol[k]² */
        float sv = state->model.sigma_vol[i];
        state->model.inv_sigma_vol_sq[i] = 1.0f / (sv * sv);
        state->model.neg_half_inv_sigma_vol_sq[i] = -0.5f / (sv * sv);
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
 *
 * SAFETY: If all weights collapse to zero (total particle degeneracy),
 * reset to uniform distribution to keep the filter alive.
 */
static void logsumexp_normalize_mkl(const float *log_weights, int N, int N_padded,
                                    float *weights, float *workspace)
{
    /* Find max using CBLAS (only over valid N elements) */
    MKL_INT max_idx = cblas_isamax(N, log_weights, 1);
    float max_val = log_weights[max_idx];

    /* Guard against all -inf (total collapse) */
    if (max_val < -1e30f || !isfinite(max_val))
    {
        /* Total collapse: Reset to uniform */
        float unif = 1.0f / N;
        for (int i = 0; i < N; i++)
            weights[i] = unif;
        for (int i = N; i < N_padded; i++)
            weights[i] = 0.0f;
        return;
    }

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

    /* Guard against zero sum (shouldn't happen after max check, but be safe) */
    if (sum < 1e-30f || !isfinite(sum))
    {
        /* Total collapse: Reset to uniform */
        float unif = 1.0f / N;
        for (i = 0; i < N; i++)
            weights[i] = unif;
        for (i = N; i < N_padded; i++)
            weights[i] = 0.0f;
        return;
    }

    /* Normalize only valid N elements */
    float inv_sum = 1.0f / sum;
    cblas_sscal(N, inv_sum, weights, 1);
}

/*═══════════════════════════════════════════════════════════════════════════════
 * EMISSION PROBABILITY - OCSN 10-COMPONENT (OPTIMIZED)
 *═══════════════════════════════════════════════════════════════════════════════*/

#define OCSN_MKL_THRESHOLD 256

/**
 * MKL VML batch strategy: Single vsExp call for all 10×N_padded values
 */
static void compute_log_emission_ocsn_mkl(float y, const float *h, int N, int N_padded,
                                          float *log_lik, float *workspace)
{
    float *log_comps = workspace;
    float *exp_comps = workspace + 10 * N_padded;

    /* Phase 1: Compute all log-components and track max per particle */
    for (int i = 0; i < N_padded; i++)
    {
        log_lik[i] = NEG_INF;
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

            if (log_comp > log_lik[i])
            {
                log_lik[i] = log_comp;
            }
        }

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
        float result = log_lik[i] + exp_comps[i];
        if (!isfinite(result) || result < -1e20f)
        {
            result = -1e20f;
        }
        log_lik[i] = result;
    }

    for (i = N; i < N_padded; i++)
    {
        log_lik[i] = NEG_INF;
    }
}

/**
 * Compute log emission using OCSN 10-component mixture (dispatcher)
 */
static void compute_log_emission_ocsn(float y, const float *h, int N, int N_padded,
                                      float *log_lik, float *workspace)
{
    init_ocsn_constants();
    compute_log_emission_ocsn_mkl(y, h, N, N_padded, log_lik, workspace);
}

/*═══════════════════════════════════════════════════════════════════════════════
 * PGAS CSMC SWEEP
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * Initialize particles at t=0
 *
 * ALIGNED WITH RBPF: Uses per-regime sigma_vol[k] for AR process noise.
 * Each regime has its own volatility-of-volatility, which is physically
 * correct (crisis regimes have higher process noise).
 *
 * Initial distribution: h_0^n ~ N(μ_k, σ_vol[k]²) where k = regime^n
 */
static void csmc_init_mkl(PGASMKLState *state)
{
    const int N = state->N;
    const int Np = state->N_padded; /* Use padded stride for SIMD alignment */
    const int K = state->K;
    const int ref_idx = state->ref_idx;
    VSLStreamStatePtr stream = (VSLStreamStatePtr)state->rng.stream;

    /* Generate random regimes for non-reference particles */
    int *rand_regimes = state->ws_indices;
    viRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, N, rand_regimes, 0, K);

    /* Generate standard normal N(0,1) - will scale by per-regime sigma_vol below
     * This is more efficient than generating K different distributions */
    vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, stream, N,
                  state->ws_normal, 0.0f, 1.0f);

    /* Initialize particles at t=0 (stride = Np for SIMD alignment) */
    for (int n = 0; n < N; n++)
    {
        if (n == ref_idx)
        {
            /* Reference particle: use conditioned trajectory */
            state->regimes[n] = state->ref_regimes[0];
            state->h[n] = state->ref_h[0];
        }
        else
        {
            int regime = rand_regimes[n];
            state->regimes[n] = regime;

            /* Per-regime AR process noise (ALIGNED WITH RBPF)
             * h_0 ~ N(μ_k, σ_vol[k]²) - crisis regimes have higher variance */
            float sigma_vol_k = state->model.sigma_vol[regime];
            state->h[n] = state->model.mu_vol[regime] + sigma_vol_k * state->ws_normal[n];
        }
        state->ancestors[n] = n; /* Self-ancestor at t=0 */
    }

    /* Zero padding for SIMD (prevents garbage in vectorized ops) */
    for (int n = N; n < Np; n++)
    {
        state->regimes[n] = 0;
        state->h[n] = 0.0f;
        state->ancestors[n] = 0;
    }

    /* Compute initial weights using OCSN emission likelihood */
    compute_log_emission_ocsn(state->observations[0], state->h, N, Np,
                              state->log_weights, state->ws_ocsn);

    /* Normalize weights (log-sum-exp for numerical stability) */
    logsumexp_normalize_mkl(state->log_weights, N, Np, state->weights, state->ws_bw);
}

/**
 * Ancestor sampling for reference trajectory (PGAS key innovation)
 *
 * ALIGNED WITH RBPF: Uses sigma_vol[ref_regime] for AR likelihood.
 * This is the process noise of the regime we're transitioning INTO.
 *
 * Computes: w̃_n ∝ w_{t-1}^n × P(z_t=k|z_{t-1}=j) × P(h_t|h_{t-1}^n, z_t=k)
 *
 * Optimizations:
 *   - Rank-1 AR(1): Uses precomputed mu_shift = μ_k × (1-φ)
 *   - Contiguous column access via transposed log_trans_T
 *   - SIMD-friendly loop structure
 *
 * @param state  PGAS state
 * @param t      Current time step
 * @return       Sampled ancestor index
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

    /* Use precomputed mu_shift = μ_k × (1-φ) for Rank-1 optimization */
    float mu_shift_k = m->mu_shift[ref_regime];
    float phi = m->phi;

    /* Per-regime AR likelihood (ALIGNED WITH RBPF)
     * Use sigma_vol of the NEXT regime (ref_regime) - the regime we're transitioning INTO
     * This is critical: the process noise depends on the destination regime */
    float neg_half_inv_var = m->neg_half_inv_sigma_vol_sq[ref_regime];

    float *log_as = state->ws_log_bw;

    /* Previous time data (stride = Np for SIMD alignment) */
    float *prev_log_w = &state->log_weights[(t - 1) * Np];
    float *prev_h = &state->h[(t - 1) * Np];
    int *prev_regimes = &state->regimes[(t - 1) * Np];

    /* Use transposed log_trans for contiguous column access (cache-friendly) */
    const float *log_trans_col = &m->log_trans_T[ref_regime * K];

    /* Compute ancestor sampling log-weights */
    int n;
#ifndef _MSC_VER
#pragma omp simd
#endif
    for (n = 0; n < N; n++)
    {
        int regime_n = prev_regimes[n];
        float h_n = prev_h[n];

        /* Contiguous access via transposed matrix */
        float log_trans = log_trans_col[regime_n];

        /* Rank-1 AR(1) arithmetic: mean = mu_shift + φ × h_n
         * Equivalent to: μ_k + φ × (h_n - μ_k) = μ_k × (1-φ) + φ × h_n */
        float mean = mu_shift_k + phi * h_n;
        float diff = ref_h - mean;

        /* log P(h_next | h_n, regime_next) = -0.5 × (h_next - mean)² / σ²
         * Uses precomputed neg_half_inv_var = -0.5 / σ_vol[k]² */
        float log_h_trans = neg_half_inv_var * diff * diff;

        log_as[n] = prev_log_w[n] + log_trans + log_h_trans;
    }

    /* Set padding to NEG_INF (ensures zero weight after exp) */
    for (n = N; n < Np; n++)
    {
        log_as[n] = NEG_INF;
    }

    /* Normalize and sample ancestor */
    logsumexp_normalize_mkl(log_as, N, Np, state->ws_bw, state->ws_uniform);

    return sample_categorical_single(state->ws_bw, N, state->ws_cumsum, stream);
}

/**
 * One CSMC sweep (Conditional Sequential Monte Carlo)
 *
 * PGAS algorithm: Samples a new trajectory conditioned on the reference.
 *
 * ALIGNED WITH RBPF: AR dynamics use per-regime sigma_vol[k]:
 *   h_t = μ_k × (1-φ) + φ × h_{t-1} + σ_vol[k] × ε_t
 *
 * @return Ancestor acceptance rate (mixing diagnostic)
 */
float pgas_mkl_csmc_sweep(PGASMKLState *state)
{
    if (!state || state->T < 2)
        return 0.0f;

    const int N = state->N;
    const int Np = state->N_padded;
    const int T = state->T;
    const int ref_idx = state->ref_idx;
    const PGASMKLModel *m = &state->model;
    VSLStreamStatePtr stream = (VSLStreamStatePtr)state->rng.stream;

    state->ancestor_proposals = 0;
    state->ancestor_accepts = 0;

    /* Initialize particles at t=0 */
    csmc_init_mkl(state);

    /* ═══════════════════════════════════════════════════════════════════
     * FORWARD PASS: Propagate particles with PGAS ancestor sampling
     * ═══════════════════════════════════════════════════════════════════*/
    for (int t = 1; t < T; t++)
    {
        /* Previous time data (stride = Np for SIMD alignment!) */
        float *prev_weights = &state->weights[(t - 1) * Np];
        float *prev_h = &state->h[(t - 1) * Np];
        int *prev_regimes = &state->regimes[(t - 1) * Np];

        /* Current time data (stride = Np!) */
        float *curr_h = &state->h[t * Np];
        int *curr_regimes = &state->regimes[t * Np];
        int *curr_ancestors = &state->ancestors[t * Np];

        /* Resample ancestors for non-reference particles (multinomial) */
        sample_categorical_mkl(prev_weights, N, curr_ancestors, N,
                               state->ws_uniform, state->ws_cumsum, stream);

        /* PGAS ancestor sampling for reference particle */
        int old_ref_anc = state->ref_ancestors[t];
        int new_ref_anc = ancestor_sample_mkl(state, t);

        state->ancestor_proposals++;
        if (new_ref_anc != old_ref_anc)
        {
            state->ancestor_accepts++;
            state->ref_ancestors[t] = new_ref_anc;
        }
        curr_ancestors[ref_idx] = state->ref_ancestors[t];

        /* Generate standard normal N(0,1) - scale by per-regime sigma_vol below
         * More efficient than K separate vsRngGaussian calls */
        vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, stream, N,
                      state->ws_normal, 0.0f, 1.0f);

        /* Propagate particles */
        for (int n = 0; n < N; n++)
        {
            if (n == ref_idx)
            {
                /* Reference particle: use conditioned trajectory */
                curr_regimes[n] = state->ref_regimes[t];
                curr_h[n] = state->ref_h[t];
            }
            else
            {
                int anc = curr_ancestors[n];
                int prev_regime = prev_regimes[anc];
                float prev_h_anc = prev_h[anc];

                /* Sample regime transition */
                curr_regimes[n] = sample_regime_mkl(m, prev_regime,
                                                    state->ws_cumsum, stream);

                /* Sample h from AR(1) with per-regime process noise
                 * ALIGNED WITH RBPF: h_t = μ_k + φ × (h_{t-1} - μ_k) + σ_vol[k] × ε_t
                 * Crisis regimes (higher k) have larger σ_vol for faster adaptation */
                int curr_regime = curr_regimes[n];
                float mu_k = m->mu_vol[curr_regime];
                float sigma_vol_k = m->sigma_vol[curr_regime];
                float mean = mu_k + m->phi * (prev_h_anc - mu_k);
                curr_h[n] = mean + sigma_vol_k * state->ws_normal[n];
            }
        }

        /* Zero padding for SIMD alignment */
        for (int n = N; n < Np; n++)
        {
            curr_regimes[n] = 0;
            curr_h[n] = 0.0f;
            curr_ancestors[n] = 0;
        }

        /* Compute weights using OCSN 10-component emission (stride = Np!) */
        compute_log_emission_ocsn(state->observations[t], curr_h, N, Np,
                                  &state->log_weights[t * Np], state->ws_ocsn);

        /* Normalize weights */
        logsumexp_normalize_mkl(&state->log_weights[t * Np], N, Np,
                                &state->weights[t * Np], state->ws_bw);
    }

    /* ═══════════════════════════════════════════════════════════════════
     * TRAJECTORY SAMPLING: Sample final particle and trace back
     * ═══════════════════════════════════════════════════════════════════*/
    int final_idx = sample_categorical_single(&state->weights[(T - 1) * Np], N,
                                              state->ws_cumsum, stream);

    /* Trace back through ancestors to get full trajectory (stride = Np!) */
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

    /* Compute acceptance rate (mixing diagnostic) */
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

    for (int s = 0; s < MIN_SWEEPS; s++)
    {
        pgas_mkl_csmc_sweep(state);
    }

    if (state->acceptance_rate >= TARGET_ACCEPTANCE)
    {
        return 0;
    }

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
 * TRANSITION MATRIX LEARNING
 *═══════════════════════════════════════════════════════════════════════════════*/

void pgas_mkl_sample_transitions(PGASMKLState *state)
{
    if (!state || state->T < 2)
        return;

    const int K = state->K;
    const int T = state->T;
    VSLStreamStatePtr stream = (VSLStreamStatePtr)state->rng.stream;

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

    state->last_off_diag_count = off_diag_count;
    state->last_total_count = total_count;

    float expected_diag = (state->sticky_kappa + state->prior_alpha) /
                          (state->sticky_kappa + K * state->prior_alpha);
    float expected_off_diag_count = (float)(T - 1) * (1.0f - expected_diag);

    if (expected_off_diag_count > 0.1f)
    {
        state->last_chatter_ratio = (float)off_diag_count / expected_off_diag_count;
    }
    else
    {
        state->last_chatter_ratio = 1.0f;
    }

    if (state->adaptive_kappa_enabled && state->last_total_count >= 10)
    {
        const float alpha = state->prior_alpha;

        int total_switches = state->last_off_diag_count;
        int total_obs = state->last_total_count;

        float prior_diag = (state->sticky_kappa + alpha) /
                           (state->sticky_kappa + K * alpha);
        float expected_switch_rate = 1.0f - prior_diag;

        float observed_switch_rate = (float)total_switches / (float)total_obs;

        float laplace = 0.5f / (float)total_obs;
        float raw_chatter = (observed_switch_rate + laplace) /
                            (expected_switch_rate + laplace);

        float lambda = state->rls_forgetting;
        float P = state->rls_variance;
        float theta = state->rls_chatter_estimate;

        float rls_gain = P / (lambda + P);

        float innovation = raw_chatter - theta;
        theta = theta + rls_gain * innovation;

        P = (1.0f / lambda) * (P - rls_gain * P);

        if (P < 0.01f)
            P = 0.01f;
        if (P > 10.0f)
            P = 10.0f;

        state->rls_chatter_estimate = theta;
        state->rls_variance = P;
        state->last_chatter_ratio = theta;

        float chatter = theta;
        if (chatter < 0.3f)
            chatter = 0.3f;
        if (chatter > 3.0f)
            chatter = 3.0f;

        /*═══════════════════════════════════════════════════════════════════════
         * ADAPTIVE KAPPA: Chatter-based with learned diagonal correction
         *
         * Primary signal: chatter = observed_switches / expected_switches
         *   - chatter > 1 → more switches than prior expects → decrease κ
         *   - chatter < 1 → fewer switches than prior expects → increase κ
         *
         * Secondary signal: learned_diag from Dirichlet posterior
         *   - If learned_diag > prior_diag despite high chatter → κ too weak
         *   - This catches cases where weak κ causes trajectory inflation
         *
         * LIMITATION: When κ is severely wrong, both signals can be corrupted.
         * Adaptive κ works best as fine-tuning around a reasonable starting point.
         *═══════════════════════════════════════════════════════════════════════*/

        /* Compute learned average diagonal from current transition matrix */
        float learned_diag = 0.0f;
        for (int k = 0; k < K; k++)
        {
            learned_diag += state->model.trans[k * K + k];
        }
        learned_diag /= (float)K;

        /* MLE diagonal from raw counts (what trajectory actually shows) */
        float mle_diag = 1.0f - observed_switch_rate;

        float kappa_new = state->sticky_kappa;

        /*═══════════════════════════════════════════════════════════════════════
         * INCREASE κ: Detect when prior is too weak
         *
         * Key insight: If MLE_diag > prior_diag, trajectory is stickier than
         * prior expects, regardless of chatter value. This is a reliable signal
         * that κ should increase.
         *═══════════════════════════════════════════════════════════════════════*/
        float mle_gap = mle_diag - prior_diag;

        if (mle_gap > 0.02f)
        {
            /* Data stickier than prior → increase κ */
            float target_diag = mle_diag;
            if (target_diag > 0.995f)
                target_diag = 0.995f;

            float kappa_target = alpha * (target_diag * K - 1.0f) / (1.0f - target_diag);

            if (kappa_target > state->sticky_kappa)
            {
                /* Smooth increase toward target */
                float increase_rate = 0.15f + 0.3f * mle_gap;
                if (increase_rate > 0.4f)
                    increase_rate = 0.4f;

                kappa_new = state->sticky_kappa + increase_rate * (kappa_target - state->sticky_kappa);
            }
        }
        else if (chatter > 1.05f && mle_gap < 0.01f)
        {
            /*═══════════════════════════════════════════════════════════════════
             * DECREASE κ: More switches than expected AND MLE confirms
             *
             * Both MLE and chatter agree data is less sticky than prior.
             *═══════════════════════════════════════════════════════════════════*/
            float target_switch_rate = expected_switch_rate * chatter;
            float target_diag = 1.0f - target_switch_rate;

            float min_diag = 1.0f / K + 0.01f;
            if (target_diag < min_diag)
                target_diag = min_diag;

            float kappa_target = alpha * (target_diag * K - 1.0f) / (1.0f - target_diag);

            if (kappa_target < 1.0f)
                kappa_target = 1.0f;

            /* Momentum-based decrease */
            float momentum = 0.7f;
            float log_curr = logf(state->sticky_kappa > 1.0f ? state->sticky_kappa : 1.0f);
            float log_targ = logf(kappa_target);
            float log_kappa_new = momentum * log_curr + (1.0f - momentum) * log_targ;
            kappa_new = expf(log_kappa_new);
        }
        else if (chatter < 0.9f)
        {
            /*═══════════════════════════════════════════════════════════════════
             * INCREASE κ: Fewer switches than expected
             *
             * Classic signal that prior is too weak.
             *═══════════════════════════════════════════════════════════════════*/
            kappa_new = state->sticky_kappa * (1.0f + 0.05f * (1.0f - chatter));
        }
        else
        {
            /*═══════════════════════════════════════════════════════════════════
             * STABLE: Chatter near 1.0, small MLE gap
             *
             * Minor drift toward equilibrium.
             *═══════════════════════════════════════════════════════════════════*/
            if (chatter > 1.1f)
            {
                kappa_new = state->sticky_kappa * 0.98f;
            }
            else if (chatter < 0.95f)
            {
                kappa_new = state->sticky_kappa * 1.02f;
            }
        }

        /* Clamp to bounds */
        if (kappa_new < state->kappa_min)
            kappa_new = state->kappa_min;
        if (kappa_new > state->kappa_max)
            kappa_new = state->kappa_max;

        state->sticky_kappa = kappa_new;
    }

    float gamma_samples[PGAS_MKL_MAX_K];

    for (int i = 0; i < K; i++)
    {
        float row_sum = 0.0f;

        for (int j = 0; j < K; j++)
        {
            float alpha_j = state->prior_alpha + (float)state->n_trans[i * K + j];
            if (i == j)
            {
                alpha_j += state->sticky_kappa;
            }

            if (alpha_j < 0.01f)
                alpha_j = 0.01f;

            vsRngGamma(VSL_RNG_METHOD_GAMMA_GNORM, stream, 1,
                       &gamma_samples[j], alpha_j, 0.0f, 1.0f);
            row_sum += gamma_samples[j];
        }

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

float pgas_mkl_gibbs_sweep(PGASMKLState *state)
{
    if (!state || state->T < 2)
        return 0.0f;

    float accept = pgas_mkl_csmc_sweep(state);

    pgas_mkl_sample_transitions(state);

    return accept;
}

/*═══════════════════════════════════════════════════════════════════════════════
 * PARIS BACKWARD SMOOTHING
 *
 * ALIGNED WITH RBPF: Uses per-regime sigma_vol[k] for backward kernel.
 *
 * Algorithm: For t = T-2 down to 0, for each particle n:
 *   1. Get smoothed state (regime_next, h_next) at t+1
 *   2. Compute backward weights:
 *      w̃_i ∝ w_t^i × P(z_{t+1}|z_t^i) × P(h_{t+1}|h_t^i, z_{t+1})
 *   3. Sample ancestor proportional to w̃
 *
 * Critical: The AR likelihood uses sigma_vol[regime_next] - the process noise
 * of the regime we're transitioning INTO, not the regime at time t.
 *═══════════════════════════════════════════════════════════════════════════════*/

/* Workload threshold: below this, sequential is faster than parallel overhead */
#define PARIS_PARALLEL_THRESHOLD 4096 /* N * T */

/**
 * PARIS backward smoothing pass
 *
 * ALIGNED WITH RBPF: Uses per-regime sigma_vol[k] for AR transition likelihood.
 *
 * Backward kernel: P(h_next | h_i, regime_next) uses σ_vol[regime_next]²
 * This is the process noise of the destination regime (regime_next).
 */
void pgas_paris_backward_smooth(PGASMKLState *state)
{
    if (!state || state->T < 2)
        return;

    const int N = state->N;
    const int Np = state->N_padded;
    const int T = state->T;
    const int K = state->model.K;
    const PGASMKLModel *m = &state->model;

    /* Precomputed AR constant */
    const float phi = m->phi;

    /* Initialize at final time (identity mapping) */
    for (int n = 0; n < N; n++)
    {
        state->smoothed[(T - 1) * Np + n] = n;
    }

    /* VML Enhanced Performance mode (trades accuracy for speed) */
    vmlSetMode(VML_EP | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE);

    float *local_log_bw = state->ws_log_bw;
    float *local_bw = state->ws_bw;
    float *local_workspace = state->ws_uniform;
    VSLStreamStatePtr stream = (VSLStreamStatePtr)state->rng.stream;

    /* ═══════════════════════════════════════════════════════════════════
     * BACKWARD PASS: Sample ancestors for each particle trajectory
     * ═══════════════════════════════════════════════════════════════════*/
    int t, n, i;
    for (t = T - 2; t >= 0; t--)
    {
        /* Data at time t (stride = Np for SIMD alignment) */
        const float *h_t = &state->h[t * Np];
        const float *log_w_t = &state->log_weights[t * Np];
        const int *regimes_t = &state->regimes[t * Np];

        for (n = 0; n < N; n++)
        {
            /* Get smoothed state at t+1 for trajectory n */
            int idx_next = state->smoothed[(t + 1) * Np + n];
            int regime_next = state->regimes[(t + 1) * Np + idx_next];
            float h_next = state->h[(t + 1) * Np + idx_next];

            /* Rank-1 precomputed: mu_shift = μ_k × (1-φ) */
            float mu_shift_k = m->mu_shift[regime_next];

            /* Contiguous column access via transposed matrix */
            const float *log_trans_col = &m->log_trans_T[regime_next * K];

            /* Per-regime AR likelihood (ALIGNED WITH RBPF)
             * Use sigma_vol of the NEXT regime (regime_next) - the regime we're
             * transitioning INTO. This is critical for model consistency. */
            float neg_half_inv_var = m->neg_half_inv_sigma_vol_sq[regime_next];

            /* Compute backward log-weights (SIMD-friendly loop) */
#ifndef _MSC_VER
#pragma omp simd
#endif
            for (i = 0; i < N; i++)
            {
                int regime_i = regimes_t[i];
                float h_i = h_t[i];

                /* Transition probability */
                float log_trans = log_trans_col[regime_i];

                /* AR(1) likelihood: log P(h_next | h_i, regime_next)
                 * Rank-1 arithmetic: mean = mu_shift + φ × h_i */
                float mean = mu_shift_k + phi * h_i;
                float diff = h_next - mean;
                float log_h_trans = neg_half_inv_var * diff * diff;

                /* Total backward weight */
                local_log_bw[i] = log_w_t[i] + log_trans + log_h_trans;
            }

            /* Set padding to NEG_INF */
            for (i = N; i < Np; i++)
            {
                local_log_bw[i] = NEG_INF;
            }

            /* Normalize and sample ancestor */
            logsumexp_normalize_mkl(local_log_bw, N, Np, local_bw, local_workspace);

            state->smoothed[t * Np + n] = sample_categorical_single(
                local_bw, N, state->ws_cumsum, stream);
        }
    }

    vmlSetMode(VML_HA);
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