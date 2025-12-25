/**
 * @file pgas_confidence.h
 * @brief PGAS Confidence Metrics for Adaptive SAEM γ Selection
 *
 * PROBLEM:
 * SAEM needs to know how much to trust PGAS output. A degeneracy collapse
 * (all particles converged to same path) produces counts, but those counts
 * are garbage. Without confidence feedback, SAEM blindly incorporates them.
 *
 * SOLUTION:
 * PGAS outputs confidence metrics that SAEM uses to adaptively set γ:
 *   - Low confidence (γ ≈ 0.01): "PGAS struggled, mostly ignore"
 *   - Medium confidence (γ ≈ 0.05): "Reasonable output, blend normally"
 *   - High confidence (γ ≈ 0.15): "Strong signal, listen more"
 *   - Tier-2 reset (γ ≈ 0.50): "Regime change detected, pivot hard"
 *
 * METRICS COMPUTED:
 *   1. ESS ratio: Effective Sample Size / N (particle diversity)
 *   2. Acceptance rate: Ancestor sampling acceptance (exploration quality)
 *   3. Unique fraction: Distinct particles after resampling
 *   4. Path divergence: How different is posterior from reference (innovation)
 *   5. Convergence: Did log-likelihood improve across sweeps?
 *
 * USAGE:
 *   // After PGAS run
 *   PGASConfidence conf;
 *   pgas_confidence_compute(pgas_state, &conf);
 *   
 *   // SAEM uses it
 *   float gamma = saem_blender_compute_gamma(&blender, &conf);
 */

#ifndef PGAS_CONFIDENCE_H
#define PGAS_CONFIDENCE_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*═══════════════════════════════════════════════════════════════════════════
 * CONFIGURATION
 *═══════════════════════════════════════════════════════════════════════════*/

/* Thresholds for confidence tiers */
#define PGAS_CONF_ESS_LOW           0.10f   /* Below this = degeneracy */
#define PGAS_CONF_ESS_HIGH          0.50f   /* Above this = good mixing */
#define PGAS_CONF_ACCEPT_LOW        0.05f   /* Below this = stuck on reference */
#define PGAS_CONF_ACCEPT_HIGH       0.30f   /* Above this = good exploration */
#define PGAS_CONF_UNIQUE_LOW        0.10f   /* Below this = collapsed */
#define PGAS_CONF_UNIQUE_HIGH       0.40f   /* Above this = diverse */
#define PGAS_CONF_DIVERGENCE_HIGH   0.20f   /* Above this = significant change */

/*═══════════════════════════════════════════════════════════════════════════
 * DATA STRUCTURES
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Confidence level enum for easy tier selection
 */
typedef enum {
    PGAS_CONFIDENCE_VERY_LOW = 0,   /* Degeneracy detected, γ ≈ 0.01 */
    PGAS_CONFIDENCE_LOW      = 1,   /* Poor mixing, γ ≈ 0.02 */
    PGAS_CONFIDENCE_MEDIUM   = 2,   /* Normal operation, γ ≈ 0.05 */
    PGAS_CONFIDENCE_HIGH     = 3,   /* Strong signal, γ ≈ 0.10 */
    PGAS_CONFIDENCE_VERY_HIGH= 4,   /* Excellent run, γ ≈ 0.15 */
} PGASConfidenceLevel;

/**
 * Comprehensive confidence metrics from PGAS run
 */
typedef struct {
    /*───────────────────────────────────────────────────────────────────────
     * RAW METRICS (computed from PGAS state)
     *───────────────────────────────────────────────────────────────────────*/
    
    /* Particle diversity */
    float ess_ratio;            /* ESS / N, range [0, 1] */
    float ess_min_ratio;        /* Minimum ESS ratio across time steps */
    float unique_fraction;      /* Unique particles / N after final resample */
    
    /* Exploration quality */
    float acceptance_rate;      /* Ancestor sampling acceptance rate */
    int   ancestor_accepts;     /* Raw accept count */
    int   ancestor_proposals;   /* Raw proposal count */
    
    /* Path coherence */
    float path_divergence;      /* Fraction of time steps where posterior ≠ reference */
    int   path_changes;         /* Number of regime changes from reference */
    int   path_length;          /* Total path length T */
    
    /* Convergence */
    float log_lik_initial;      /* Log-likelihood at start */
    float log_lik_final;        /* Log-likelihood at end */
    float log_lik_improvement;  /* Improvement (positive = good) */
    bool  converged;            /* Did log-lik stabilize? */
    
    /* Sweep statistics */
    int   sweeps_run;           /* Number of Gibbs sweeps */
    float sweep_time_us;        /* Average time per sweep (microseconds) */
    
    /*───────────────────────────────────────────────────────────────────────
     * DERIVED SCORES (for easy consumption)
     *───────────────────────────────────────────────────────────────────────*/
    
    float diversity_score;      /* Combined ESS + unique score, [0, 1] */
    float exploration_score;    /* Acceptance-based score, [0, 1] */
    float innovation_score;     /* Path divergence score, [0, 1] */
    float overall_score;        /* Weighted combination, [0, 1] */
    
    PGASConfidenceLevel level;  /* Discretized confidence tier */
    float suggested_gamma;      /* Recommended γ for SAEM */
    
    /*───────────────────────────────────────────────────────────────────────
     * DIAGNOSTIC FLAGS
     *───────────────────────────────────────────────────────────────────────*/
    
    bool degeneracy_detected;   /* ESS collapsed at some point */
    bool reference_dominated;   /* Posterior ≈ reference (no learning) */
    bool regime_change_detected;/* Large innovation suggests market shift */
    
    /* Timestamp */
    int64_t compute_time_ns;    /* Time to compute these metrics */
    
} PGASConfidence;

/**
 * Configuration for confidence computation
 */
typedef struct {
    /* Score weights for overall_score computation */
    float weight_diversity;     /* Default: 0.40 */
    float weight_exploration;   /* Default: 0.30 */
    float weight_innovation;    /* Default: 0.30 */
    
    /* Gamma mapping */
    float gamma_very_low;       /* Default: 0.01 */
    float gamma_low;            /* Default: 0.02 */
    float gamma_medium;         /* Default: 0.05 */
    float gamma_high;           /* Default: 0.10 */
    float gamma_very_high;      /* Default: 0.15 */
    
    /* Regime change detection */
    float regime_change_threshold;  /* Default: 0.30 (30% path divergence) */
    
} PGASConfidenceConfig;

/*═══════════════════════════════════════════════════════════════════════════
 * API
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Get default configuration
 */
PGASConfidenceConfig pgas_confidence_config_defaults(void);

/**
 * Compute confidence metrics from PGAS state
 *
 * Call this after pgas_mkl_run_adaptive() or pgas_mkl_gibbs_sweep().
 *
 * @param state     PGAS state after run
 * @param ref_path  Original reference path (to compute divergence)
 * @param T         Path length
 * @param conf      Output confidence metrics
 * @param config    Configuration (NULL for defaults)
 * @return          0 on success, -1 on error
 */
struct PGASMKLState;  /* Forward declaration */

int pgas_confidence_compute(
    const struct PGASMKLState *state,
    const int *ref_path_original,
    int T,
    PGASConfidence *conf,
    const PGASConfidenceConfig *config);

/**
 * Compute confidence from raw values (when PGAS state not available)
 *
 * Useful for testing or when metrics are collected separately.
 */
int pgas_confidence_compute_raw(
    float ess_ratio,
    float acceptance_rate,
    float unique_fraction,
    float path_divergence,
    int sweeps_run,
    PGASConfidence *conf,
    const PGASConfidenceConfig *config);

/**
 * Get recommended gamma from confidence metrics
 */
float pgas_confidence_get_gamma(const PGASConfidence *conf);

/**
 * Get confidence level as string
 */
const char* pgas_confidence_level_str(PGASConfidenceLevel level);

/**
 * Print confidence metrics
 */
void pgas_confidence_print(const PGASConfidence *conf);

/*═══════════════════════════════════════════════════════════════════════════
 * INTEGRATION WITH SAEM
 *
 * SAEM can use these metrics directly:
 *
 *   PGASConfidence conf;
 *   pgas_confidence_compute(pgas, ref_path, T, &conf, NULL);
 *   
 *   if (conf.regime_change_detected) {
 *       // Tier-2 reset
 *       saem_blender_tier2_reset(&blender);
 *       gamma = 0.50f;
 *   } else if (conf.degeneracy_detected) {
 *       // Don't trust this run
 *       gamma = 0.01f;
 *   } else {
 *       gamma = conf.suggested_gamma;
 *   }
 *   
 *   saem_blender_blend(&blender, S, gamma);
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Quick check: should SAEM use this output at all?
 *
 * Returns false if degeneracy detected or confidence very low.
 */
static inline bool pgas_confidence_usable(const PGASConfidence *conf) {
    if (!conf) return false;
    return !conf->degeneracy_detected && 
           conf->level >= PGAS_CONFIDENCE_LOW;
}

/**
 * Quick check: does this suggest regime change?
 */
static inline bool pgas_confidence_regime_change(const PGASConfidence *conf) {
    return conf && conf->regime_change_detected;
}

#ifdef __cplusplus
}
#endif

#endif /* PGAS_CONFIDENCE_H */
