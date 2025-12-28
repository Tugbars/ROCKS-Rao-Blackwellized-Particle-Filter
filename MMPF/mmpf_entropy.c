/**
 * @file mmpf_entropy.c
 * @brief Entropy-Based Stability Detection for Shock Recovery (MKL Optimized)
 *
 * Shannon entropy-based detection of particle filter convergence
 * to determine when to restore sticky transitions after shock.
 *
 * Optimizations:
 * - MKL vdLn for batch log computation
 * - BLAS cblas_ddot for weighted sum
 * - Relative threshold (scales with entropy level)
 */

#include "mmpf_entropy.h"
#include <math.h>
#include <string.h>

#ifdef USE_MKL
#include <mkl.h>
#define HAS_MKL 1
#else
#define HAS_MKL 0
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Floor for dynamic threshold to prevent infinite wait at H→0 */
#define MIN_ENTROPY_THRESHOLD 0.001

/* Maximum weights buffer size (3 models × 1024 particles) */
#define MAX_WEIGHT_BUFFER 4096

/*═══════════════════════════════════════════════════════════════════════════
 * DEFAULTS
 *═══════════════════════════════════════════════════════════════════════════*/

static const MMPF_EntropyConfig DEFAULT_CONFIG = {
    .stability_threshold = 0.01, /* 1% relative change = stable */
    .delta_ema_alpha = 0.3,      /* Fast response */
    .min_shock_duration = 3,     /* At least 3 ticks */
    .max_shock_duration = 50,    /* Force unlock after 50 ticks */
    .use_two_level = 0           /* Single combined entropy by default */
};

/*═══════════════════════════════════════════════════════════════════════════
 * INITIALIZATION
 *═══════════════════════════════════════════════════════════════════════════*/

void mmpf_entropy_init(MMPF_EntropyState *state)
{
    mmpf_entropy_init_config(state, &DEFAULT_CONFIG);
}

void mmpf_entropy_init_config(MMPF_EntropyState *state, const MMPF_EntropyConfig *cfg)
{
    memset(state, 0, sizeof(MMPF_EntropyState));

    if (cfg)
    {
        state->config = *cfg;
    }
    else
    {
        state->config = DEFAULT_CONFIG;
    }

    state->entropy_prev = 1.0; /* Start assuming high entropy */
    state->delta_ema = 1.0;    /* Start high to prevent premature unlock */
    state->is_locked = 0;
    state->ticks_since_shock = 0;

    state->model_entropy = 0.0;
    state->particle_entropy = 0.0;
    state->model_delta_ema = 0.1;
    state->particle_delta_ema = 0.1;
}

/*═══════════════════════════════════════════════════════════════════════════
 * ENTROPY CALCULATION (MKL Optimized)
 *═══════════════════════════════════════════════════════════════════════════*/

double mmpf_calculate_entropy(const MMPF_ROCKS *mmpf)
{
    const double eps = 1e-12;

#if HAS_MKL
    /* MKL vectorized path: gather weights, batch log, dot product */
    static __thread double w_buf[MAX_WEIGHT_BUFFER];
    static __thread double log_w_buf[MAX_WEIGHT_BUFFER];
    int idx = 0;

    /* Gather all weights into contiguous buffer */
    for (int k = 0; k < MMPF_N_MODELS; k++)
    {
        double model_w = mmpf->weights[k];
        if (model_w < eps)
            model_w = eps;

        RBPF_KSC *rbpf = mmpf->ext[k]->rbpf;
        int n_particles = rbpf->n_particles;
        const double *w_norm = (const double *)rbpf->w_norm;

        for (int i = 0; i < n_particles && idx < MAX_WEIGHT_BUFFER; i++)
        {
            double w = model_w * w_norm[i];
            w_buf[idx++] = (w < eps) ? eps : w;
        }
    }

    /* Vectorized log: log_w_buf = ln(w_buf) */
    vdLn(idx, w_buf, log_w_buf);

    /* Dot product: sum(w * log(w)) */
    double dot_prod = cblas_ddot(idx, w_buf, 1, log_w_buf, 1);

    /* Entropy H = -sum(w * log(w)) */
    double H = -dot_prod;

    /* Normalize by log(N) so maximum entropy = 1.0 */
    if (idx > 1)
    {
        H /= log((double)idx);
    }

    return H;

#else
    /* Scalar fallback */
    double H = 0.0;
    int N = 0;

    for (int k = 0; k < MMPF_N_MODELS; k++)
    {
        double model_w = mmpf->weights[k];
        RBPF_KSC *rbpf = mmpf->ext[k]->rbpf;
        int n_particles = rbpf->n_particles;
        const double *w_norm = (const double *)rbpf->w_norm;

        for (int i = 0; i < n_particles; i++)
        {
            double w = model_w * w_norm[i];
            if (w > eps)
            {
                H -= w * log(w);
            }
            N++;
        }
    }

    if (N > 1)
    {
        H /= log((double)N);
    }

    return H;
#endif
}

double mmpf_calculate_model_entropy(const MMPF_ROCKS *mmpf)
{
    double H = 0.0;

    for (int k = 0; k < MMPF_N_MODELS; k++)
    {
        double w = mmpf->weights[k];
        if (w > 1e-12)
        {
            H -= w * log(w);
        }
    }

    /* Normalize by log(n_models) */
    if (MMPF_N_MODELS > 1)
    {
        H /= log((double)MMPF_N_MODELS);
    }

    return H;
}

double mmpf_calculate_particle_entropy(const MMPF_ROCKS *mmpf)
{
    double avg_H = 0.0;
    int count = 0;

    for (int k = 0; k < MMPF_N_MODELS; k++)
    {
        RBPF_KSC *rbpf = mmpf->ext[k]->rbpf;
        int n = rbpf->n_particles;
        const double *w_norm = (const double *)rbpf->w_norm;

        double H_k = 0.0;
        for (int i = 0; i < n; i++)
        {
            double w = w_norm[i];
            if (w > 1e-12)
            {
                H_k -= w * log(w);
            }
        }

        /* Normalize by log(n_particles) */
        if (n > 1)
        {
            H_k /= log((double)n);
        }

        avg_H += H_k;
        count++;
    }

    return (count > 0) ? avg_H / count : 0.0;
}

/*═══════════════════════════════════════════════════════════════════════════
 * SHOCK HANDLING
 *═══════════════════════════════════════════════════════════════════════════*/

void mmpf_entropy_shock_injected(MMPF_EntropyState *state)
{
    state->is_locked = 1;
    state->ticks_since_shock = 0;
    state->total_shocks++;

    /* Reset EMA to high uncertainty - prevents premature unlock */
    state->delta_ema = 0.5;
    state->model_delta_ema = 0.5;
    state->particle_delta_ema = 0.5;
}

/*═══════════════════════════════════════════════════════════════════════════
 * STABILITY CHECK (Relative Threshold)
 *═══════════════════════════════════════════════════════════════════════════*/

int mmpf_entropy_check_stability(MMPF_EntropyState *state, const MMPF_ROCKS *mmpf)
{
    /* If not in shock state, nothing to check */
    if (!state->is_locked)
    {
        return 1; /* Already stable */
    }

    state->ticks_since_shock++;
    state->total_ticks_locked++;

    /*─────────────────────────────────────────────────────────────────────────
     * SAFETY: Minimum duration
     *───────────────────────────────────────────────────────────────────────*/
    if (state->ticks_since_shock < state->config.min_shock_duration)
    {
        return 0; /* Too soon */
    }

    /*─────────────────────────────────────────────────────────────────────────
     * SAFETY: Maximum duration (forced unlock)
     *───────────────────────────────────────────────────────────────────────*/
    if (state->ticks_since_shock >= state->config.max_shock_duration)
    {
        state->is_locked = 0;

        /* Update average lock duration */
        double n = (double)state->total_shocks;
        state->avg_lock_duration = ((n - 1) * state->avg_lock_duration + state->ticks_since_shock) / n;
        return 1; /* Forced unlock */
    }

    /*─────────────────────────────────────────────────────────────────────────
     * ENTROPY COMPUTATION
     *───────────────────────────────────────────────────────────────────────*/
    double alpha = state->config.delta_ema_alpha;
    int stable = 0;

    if (state->config.use_two_level)
    {
        /* Two-level: check both model and particle entropy */
        double H_model = mmpf_calculate_model_entropy(mmpf);
        double H_particle = mmpf_calculate_particle_entropy(mmpf);

        /* Update model delta EMA */
        double delta_model = fabs(H_model - state->model_entropy);
        state->model_delta_ema = alpha * delta_model + (1.0 - alpha) * state->model_delta_ema;
        state->model_entropy = H_model;

        /* Update particle delta EMA */
        double delta_particle = fabs(H_particle - state->particle_entropy);
        state->particle_delta_ema = alpha * delta_particle + (1.0 - alpha) * state->particle_delta_ema;
        state->particle_entropy = H_particle;

        /* Relative thresholds for each level */
        double thresh_model = state->config.stability_threshold * H_model;
        double thresh_particle = state->config.stability_threshold * H_particle;
        if (thresh_model < MIN_ENTROPY_THRESHOLD)
            thresh_model = MIN_ENTROPY_THRESHOLD;
        if (thresh_particle < MIN_ENTROPY_THRESHOLD)
            thresh_particle = MIN_ENTROPY_THRESHOLD;

        /* Stable if BOTH model and particle entropy have stabilized */
        stable = (state->model_delta_ema < thresh_model) &&
                 (state->particle_delta_ema < thresh_particle);

        /* Also update combined for diagnostics */
        state->entropy_prev = 0.5 * H_model + 0.5 * H_particle;
        state->delta_ema = 0.5 * state->model_delta_ema + 0.5 * state->particle_delta_ema;
    }
    else
    {
        /* Single-level: combined entropy over all particles */
        double H = mmpf_calculate_entropy(mmpf);

        /* Update delta EMA */
        double delta = fabs(H - state->entropy_prev);
        state->delta_ema = alpha * delta + (1.0 - alpha) * state->delta_ema;
        state->entropy_prev = H;

        /* Relative threshold: "Is change < X% of current entropy?" */
        double dyn_threshold = state->config.stability_threshold * H;

        /* Floor to prevent infinite wait when H → 0 */
        if (dyn_threshold < MIN_ENTROPY_THRESHOLD)
        {
            dyn_threshold = MIN_ENTROPY_THRESHOLD;
        }

        /* Stable if delta EMA below dynamic threshold */
        stable = (state->delta_ema < dyn_threshold);
    }

    /*─────────────────────────────────────────────────────────────────────────
     * UNLOCK IF STABLE
     *───────────────────────────────────────────────────────────────────────*/
    if (stable)
    {
        state->is_locked = 0;

        /* Update average lock duration */
        double n = (double)state->total_shocks;
        if (n > 0)
        {
            state->avg_lock_duration = ((n - 1) * state->avg_lock_duration + state->ticks_since_shock) / n;
        }
        return 1;
    }

    return 0; /* Still unstable */
}

/*═══════════════════════════════════════════════════════════════════════════
 * QUERY FUNCTIONS
 *═══════════════════════════════════════════════════════════════════════════*/

double mmpf_entropy_get_current(const MMPF_EntropyState *state)
{
    return state->entropy_prev;
}

double mmpf_entropy_get_delta_ema(const MMPF_EntropyState *state)
{
    return state->delta_ema;
}

int mmpf_entropy_is_locked(const MMPF_EntropyState *state)
{
    return state->is_locked;
}

int mmpf_entropy_ticks_since_shock(const MMPF_EntropyState *state)
{
    return state->ticks_since_shock;
}

void mmpf_entropy_force_unlock(MMPF_EntropyState *state)
{
    if (state->is_locked)
    {
        state->is_locked = 0;

        /* Update stats */
        double n = (double)state->total_shocks;
        if (n > 0)
        {
            state->avg_lock_duration = ((n - 1) * state->avg_lock_duration + state->ticks_since_shock) / n;
        }
    }
}

void mmpf_entropy_reset_stats(MMPF_EntropyState *state)
{
    state->total_shocks = 0;
    state->total_ticks_locked = 0;
    state->avg_lock_duration = 0.0;
}