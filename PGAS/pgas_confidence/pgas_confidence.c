/**
 * @file pgas_confidence.c
 * @brief PGAS Confidence Metrics Implementation
 */

/* Include PGAS header FIRST for state access (skip if testing without MKL) */
#ifndef PGAS_CONFIDENCE_NO_PGAS_STATE
#include "pgas_mkl.h"
#endif

#include "pgas_confidence.h"
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

/*═══════════════════════════════════════════════════════════════════════════
 * CROSS-PLATFORM HIGH-RESOLUTION TIMING
 *═══════════════════════════════════════════════════════════════════════════*/

#if defined(_WIN32) || defined(_WIN64)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

static inline int64_t get_time_ns(void)
{
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (int64_t)((double)counter.QuadPart / freq.QuadPart * 1e9);
}
#else
/* POSIX (Linux, macOS) */
static inline int64_t get_time_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}
#endif

/*═══════════════════════════════════════════════════════════════════════════
 * CONFIGURATION DEFAULTS
 *═══════════════════════════════════════════════════════════════════════════*/

PGASConfidenceConfig pgas_confidence_config_defaults(void)
{
    PGASConfidenceConfig cfg;

    /* Score weights (sum to 1.0) */
    cfg.weight_diversity = 0.40f;
    cfg.weight_exploration = 0.30f;
    cfg.weight_innovation = 0.30f;

    /* Gamma mapping for each tier */
    cfg.gamma_very_low = 0.01f;
    cfg.gamma_low = 0.02f;
    cfg.gamma_medium = 0.05f;
    cfg.gamma_high = 0.10f;
    cfg.gamma_very_high = 0.15f;

    /* Regime change threshold */
    cfg.regime_change_threshold = 0.30f;

    return cfg;
}

/*═══════════════════════════════════════════════════════════════════════════
 * HELPER FUNCTIONS
 *═══════════════════════════════════════════════════════════════════════════*/

static float clamp(float x, float lo, float hi)
{
    return (x < lo) ? lo : ((x > hi) ? hi : x);
}

/**
 * Compute diversity score from ESS and unique fraction
 *
 * Score mapping:
 *   ESS < 0.10 OR unique < 0.10  → 0.0 (degeneracy)
 *   ESS > 0.50 AND unique > 0.40 → 1.0 (excellent)
 *   Otherwise → linear interpolation
 */
static float compute_diversity_score(float ess_ratio, float unique_fraction)
{
    /* Degeneracy check */
    if (ess_ratio < PGAS_CONF_ESS_LOW || unique_fraction < PGAS_CONF_UNIQUE_LOW)
    {
        return 0.0f;
    }

    /* Normalize ESS to [0, 1] */
    float ess_score = (ess_ratio - PGAS_CONF_ESS_LOW) /
                      (PGAS_CONF_ESS_HIGH - PGAS_CONF_ESS_LOW);
    ess_score = clamp(ess_score, 0.0f, 1.0f);

    /* Normalize unique fraction to [0, 1] */
    float unique_score = (unique_fraction - PGAS_CONF_UNIQUE_LOW) /
                         (PGAS_CONF_UNIQUE_HIGH - PGAS_CONF_UNIQUE_LOW);
    unique_score = clamp(unique_score, 0.0f, 1.0f);

    /* Geometric mean (penalizes if either is low) */
    return sqrtf(ess_score * unique_score);
}

/**
 * Compute exploration score from acceptance rate
 *
 * Score mapping:
 *   acceptance < 0.05 → 0.0 (stuck on reference)
 *   acceptance > 0.30 → 1.0 (good exploration)
 *   Otherwise → linear interpolation
 */
static float compute_exploration_score(float acceptance_rate)
{
    if (acceptance_rate < PGAS_CONF_ACCEPT_LOW)
    {
        return 0.0f;
    }

    float score = (acceptance_rate - PGAS_CONF_ACCEPT_LOW) /
                  (PGAS_CONF_ACCEPT_HIGH - PGAS_CONF_ACCEPT_LOW);
    return clamp(score, 0.0f, 1.0f);
}

/**
 * Compute innovation score from path divergence
 *
 * This is tricky: some divergence is good (learning), too much is suspicious.
 * Score peaks around 10-20% divergence.
 */
static float compute_innovation_score(float path_divergence)
{
    /* No divergence = no learning (bad) */
    if (path_divergence < 0.02f)
    {
        return 0.2f; /* Some base score - reference might have been correct */
    }

    /* Optimal range: 5-20% divergence */
    if (path_divergence >= 0.05f && path_divergence <= 0.20f)
    {
        return 1.0f;
    }

    /* Low divergence (2-5%): ramping up */
    if (path_divergence < 0.05f)
    {
        return 0.2f + 0.8f * (path_divergence - 0.02f) / 0.03f;
    }

    /* High divergence (>20%): might be noise or regime change */
    /* Still high score but capped */
    return 0.8f;
}

/**
 * Determine confidence level from overall score
 */
static PGASConfidenceLevel score_to_level(float overall_score, bool degeneracy)
{
    if (degeneracy || overall_score < 0.15f)
    {
        return PGAS_CONFIDENCE_VERY_LOW;
    }
    if (overall_score < 0.30f)
    {
        return PGAS_CONFIDENCE_LOW;
    }
    if (overall_score < 0.55f)
    {
        return PGAS_CONFIDENCE_MEDIUM;
    }
    if (overall_score < 0.75f)
    {
        return PGAS_CONFIDENCE_HIGH;
    }
    return PGAS_CONFIDENCE_VERY_HIGH;
}

/**
 * Map confidence level to gamma
 */
static float level_to_gamma(PGASConfidenceLevel level, const PGASConfidenceConfig *cfg)
{
    switch (level)
    {
    case PGAS_CONFIDENCE_VERY_LOW:
        return cfg->gamma_very_low;
    case PGAS_CONFIDENCE_LOW:
        return cfg->gamma_low;
    case PGAS_CONFIDENCE_MEDIUM:
        return cfg->gamma_medium;
    case PGAS_CONFIDENCE_HIGH:
        return cfg->gamma_high;
    case PGAS_CONFIDENCE_VERY_HIGH:
        return cfg->gamma_very_high;
    default:
        return cfg->gamma_medium;
    }
}

/**
 * Count unique values in integer array
 */
static int count_unique(const int *arr, int n)
{
    if (n <= 0)
        return 0;
    if (n == 1)
        return 1;

    /* Simple O(n²) for small arrays */
    int unique = 0;
    for (int i = 0; i < n; i++)
    {
        bool is_unique = true;
        for (int j = 0; j < i; j++)
        {
            if (arr[j] == arr[i])
            {
                is_unique = false;
                break;
            }
        }
        if (is_unique)
            unique++;
    }
    return unique;
}

/*═══════════════════════════════════════════════════════════════════════════
 * MAIN COMPUTATION
 *═══════════════════════════════════════════════════════════════════════════*/

#ifndef PGAS_CONFIDENCE_NO_PGAS_STATE

int pgas_confidence_compute(
    const PGASMKLState *state,
    const int *ref_path_original,
    int T,
    PGASConfidence *conf,
    const PGASConfidenceConfig *config)
{

    if (!state || !conf)
    {
        return -1;
    }

    int64_t start_ns = get_time_ns();

    memset(conf, 0, sizeof(*conf));

    PGASConfidenceConfig cfg = config ? *config : pgas_confidence_config_defaults();

    int N = state->N;
    int K = state->K;
    (void)K; /* Silence unused warning */

    /*───────────────────────────────────────────────────────────────────────
     * 1. ESS RATIO
     *───────────────────────────────────────────────────────────────────────*/

    /* Get ESS at final time step */
    float ess_final = pgas_mkl_get_ess(state, T - 1);
    conf->ess_ratio = ess_final / (float)N;

    /* Find minimum ESS across time steps (worst case) */
    float ess_min = ess_final;
    for (int t = 0; t < T; t++)
    {
        float ess_t = pgas_mkl_get_ess(state, t);
        if (ess_t < ess_min)
        {
            ess_min = ess_t;
        }
    }
    conf->ess_min_ratio = ess_min / (float)N;

    /*───────────────────────────────────────────────────────────────────────
     * 2. ACCEPTANCE RATE
     *───────────────────────────────────────────────────────────────────────*/

    conf->acceptance_rate = state->acceptance_rate;
    conf->ancestor_accepts = state->ancestor_accepts;
    conf->ancestor_proposals = state->ancestor_proposals;

    /*───────────────────────────────────────────────────────────────────────
     * 3. UNIQUE FRACTION (from final particles)
     *───────────────────────────────────────────────────────────────────────*/

    /* Count unique regime values at final time step */
    int t_final = T - 1;
    int *final_regimes = &state->regimes[t_final * state->N_padded];
    int unique_particles = count_unique(final_regimes, N);
    conf->unique_fraction = (float)unique_particles / (float)N;

    /*───────────────────────────────────────────────────────────────────────
     * 4. PATH DIVERGENCE (compare posterior to original reference)
     *───────────────────────────────────────────────────────────────────────*/

    conf->path_length = T;
    conf->path_changes = 0;

    if (ref_path_original && state->ref_regimes)
    {
        for (int t = 0; t < T; t++)
        {
            if (state->ref_regimes[t] != ref_path_original[t])
            {
                conf->path_changes++;
            }
        }
        conf->path_divergence = (float)conf->path_changes / (float)T;
    }
    else
    {
        /* Can't compute divergence without original reference */
        conf->path_divergence = 0.10f; /* Assume moderate divergence */
    }

    /*───────────────────────────────────────────────────────────────────────
     * 5. CONVERGENCE (log-likelihood improvement)
     *───────────────────────────────────────────────────────────────────────*/

    /* TODO: Track log-likelihood across sweeps in PGAS state */
    /* For now, assume convergence if acceptance rate is reasonable */
    conf->converged = (conf->acceptance_rate > 0.10f);
    conf->log_lik_initial = 0.0f;
    conf->log_lik_final = 0.0f;
    conf->log_lik_improvement = 0.0f;

    /*───────────────────────────────────────────────────────────────────────
     * 6. SWEEP STATISTICS
     *───────────────────────────────────────────────────────────────────────*/

    conf->sweeps_run = state->total_sweeps;
    conf->sweep_time_us = 0.0f; /* Would need timing instrumentation */

    /*───────────────────────────────────────────────────────────────────────
     * 7. COMPUTE DERIVED SCORES
     *───────────────────────────────────────────────────────────────────────*/

    conf->diversity_score = compute_diversity_score(conf->ess_ratio, conf->unique_fraction);
    conf->exploration_score = compute_exploration_score(conf->acceptance_rate);
    conf->innovation_score = compute_innovation_score(conf->path_divergence);

    conf->overall_score = cfg.weight_diversity * conf->diversity_score +
                          cfg.weight_exploration * conf->exploration_score +
                          cfg.weight_innovation * conf->innovation_score;

    /*───────────────────────────────────────────────────────────────────────
     * 8. DIAGNOSTIC FLAGS
     *───────────────────────────────────────────────────────────────────────*/

    conf->degeneracy_detected = (conf->ess_min_ratio < PGAS_CONF_ESS_LOW) ||
                                (conf->unique_fraction < PGAS_CONF_UNIQUE_LOW);

    conf->reference_dominated = (conf->acceptance_rate < PGAS_CONF_ACCEPT_LOW) &&
                                (conf->path_divergence < 0.02f);

    conf->regime_change_detected = (conf->path_divergence > cfg.regime_change_threshold);

    /*───────────────────────────────────────────────────────────────────────
     * 9. DETERMINE LEVEL AND GAMMA
     *───────────────────────────────────────────────────────────────────────*/

    conf->level = score_to_level(conf->overall_score, conf->degeneracy_detected);
    conf->suggested_gamma = level_to_gamma(conf->level, &cfg);

    /* Override gamma for regime change */
    if (conf->regime_change_detected && !conf->degeneracy_detected)
    {
        /* Don't go full tier-2 reset automatically, but increase gamma */
        conf->suggested_gamma = fmaxf(conf->suggested_gamma, cfg.gamma_high);
    }

    conf->compute_time_ns = get_time_ns() - start_ns;

    return 0;
}

#endif /* PGAS_CONFIDENCE_NO_PGAS_STATE */

/*═══════════════════════════════════════════════════════════════════════════
 * RAW COMPUTATION (without PGAS state)
 *═══════════════════════════════════════════════════════════════════════════*/

int pgas_confidence_compute_raw(
    float ess_ratio,
    float acceptance_rate,
    float unique_fraction,
    float path_divergence,
    int sweeps_run,
    PGASConfidence *conf,
    const PGASConfidenceConfig *config)
{

    if (!conf)
        return -1;

    memset(conf, 0, sizeof(*conf));

    PGASConfidenceConfig cfg = config ? *config : pgas_confidence_config_defaults();

    /* Store raw metrics */
    conf->ess_ratio = ess_ratio;
    conf->ess_min_ratio = ess_ratio; /* Assume same */
    conf->acceptance_rate = acceptance_rate;
    conf->unique_fraction = unique_fraction;
    conf->path_divergence = path_divergence;
    conf->sweeps_run = sweeps_run;

    /* Compute scores */
    conf->diversity_score = compute_diversity_score(ess_ratio, unique_fraction);
    conf->exploration_score = compute_exploration_score(acceptance_rate);
    conf->innovation_score = compute_innovation_score(path_divergence);

    conf->overall_score = cfg.weight_diversity * conf->diversity_score +
                          cfg.weight_exploration * conf->exploration_score +
                          cfg.weight_innovation * conf->innovation_score;

    /* Flags */
    conf->degeneracy_detected = (ess_ratio < PGAS_CONF_ESS_LOW) ||
                                (unique_fraction < PGAS_CONF_UNIQUE_LOW);
    conf->reference_dominated = (acceptance_rate < PGAS_CONF_ACCEPT_LOW) &&
                                (path_divergence < 0.02f);
    conf->regime_change_detected = (path_divergence > cfg.regime_change_threshold);

    /* Level and gamma */
    conf->level = score_to_level(conf->overall_score, conf->degeneracy_detected);
    conf->suggested_gamma = level_to_gamma(conf->level, &cfg);

    if (conf->regime_change_detected && !conf->degeneracy_detected)
    {
        conf->suggested_gamma = fmaxf(conf->suggested_gamma, cfg.gamma_high);
    }

    return 0;
}

/*═══════════════════════════════════════════════════════════════════════════
 * UTILITY FUNCTIONS
 *═══════════════════════════════════════════════════════════════════════════*/

float pgas_confidence_get_gamma(const PGASConfidence *conf)
{
    return conf ? conf->suggested_gamma : 0.05f;
}

const char *pgas_confidence_level_str(PGASConfidenceLevel level)
{
    switch (level)
    {
    case PGAS_CONFIDENCE_VERY_LOW:
        return "VERY_LOW";
    case PGAS_CONFIDENCE_LOW:
        return "LOW";
    case PGAS_CONFIDENCE_MEDIUM:
        return "MEDIUM";
    case PGAS_CONFIDENCE_HIGH:
        return "HIGH";
    case PGAS_CONFIDENCE_VERY_HIGH:
        return "VERY_HIGH";
    default:
        return "UNKNOWN";
    }
}

void pgas_confidence_print(const PGASConfidence *conf)
{
    if (!conf)
    {
        printf("PGASConfidence: NULL\n");
        return;
    }

    printf("═══════════════════════════════════════════════════════════════\n");
    printf("PGAS CONFIDENCE METRICS\n");
    printf("═══════════════════════════════════════════════════════════════\n");

    printf("\nRAW METRICS:\n");
    printf("  ESS ratio:        %.3f (min: %.3f)\n", conf->ess_ratio, conf->ess_min_ratio);
    printf("  Acceptance rate:  %.3f (%d/%d)\n",
           conf->acceptance_rate, conf->ancestor_accepts, conf->ancestor_proposals);
    printf("  Unique fraction:  %.3f\n", conf->unique_fraction);
    printf("  Path divergence:  %.3f (%d/%d changes)\n",
           conf->path_divergence, conf->path_changes, conf->path_length);
    printf("  Sweeps run:       %d\n", conf->sweeps_run);

    printf("\nSCORES:\n");
    printf("  Diversity:        %.3f\n", conf->diversity_score);
    printf("  Exploration:      %.3f\n", conf->exploration_score);
    printf("  Innovation:       %.3f\n", conf->innovation_score);
    printf("  ─────────────────────────\n");
    printf("  Overall:          %.3f\n", conf->overall_score);

    printf("\nDIAGNOSTICS:\n");
    printf("  Degeneracy:       %s\n", conf->degeneracy_detected ? "YES ⚠️" : "no");
    printf("  Ref dominated:    %s\n", conf->reference_dominated ? "YES ⚠️" : "no");
    printf("  Regime change:    %s\n", conf->regime_change_detected ? "YES 🔄" : "no");

    printf("\nRECOMMENDATION:\n");
    printf("  Confidence:       %s\n", pgas_confidence_level_str(conf->level));
    printf("  Suggested γ:      %.3f\n", conf->suggested_gamma);

    printf("═══════════════════════════════════════════════════════════════\n");
}