/**
 * @file saem_blender.h
 * @brief SAEM Parameter Blending for Oracle → RBPF Handoff
 *
 * Stochastic Approximation EM (SAEM) for safe transition matrix updates.
 * When PGAS produces a new Π, blend it into RBPF without destabilizing.
 *
 * Key insight: Update SUFFICIENT STATISTICS, not point estimates.
 *
 * SAEM Update:
 *   Q_t[i,j] = (1 - γ) × Q_{t-1}[i,j] + γ × S_oracle[i,j]
 *   Π[i,j] = Q_t[i,j] / Σ_k Q_t[i,k]   (automatic simplex constraint!)
 *
 * Safety Features:
 *   - Floor enforcement: Π_ij ≥ TRANS_FLOOR (escape hatch)
 *   - Adaptive γ based on acceptance, diversity, surprise
 *   - κ (stickiness) drift control
 *   - Tempered path injection for anti-confirmation bias
 *
 * Reference: ORACLE_INTEGRATION_PLAN.md v1.6
 */

#ifndef SAEM_BLENDER_H
#define SAEM_BLENDER_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*═══════════════════════════════════════════════════════════════════════════
 * COMPILE-TIME CONFIGURATION
 *═══════════════════════════════════════════════════════════════════════════*/

#define SAEM_MAX_REGIMES        8       /* Maximum number of regimes */
#define SAEM_TRANS_FLOOR        1e-5f   /* Minimum transition probability */
#define SAEM_GAMMA_MIN          0.02f   /* Minimum blending weight */
#define SAEM_GAMMA_MAX          0.50f   /* Maximum blending weight */
#define SAEM_GAMMA_DEFAULT      0.15f   /* Default blending weight */

/*═══════════════════════════════════════════════════════════════════════════
 * CONFIGURATION
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Adaptive γ configuration
 */
typedef struct {
    /* Base learning rate */
    float gamma_base;              /* Starting γ (e.g., 0.15) */
    
    /* Robbins-Monro decay */
    bool  use_robbins_monro;       /* Enable 1/√t decay */
    float rm_offset;               /* Offset to prevent early γ → 0 */
    
    /* Acceptance rate adaptation */
    float accept_high;             /* Threshold for "high" acceptance (e.g., 0.4) */
    float accept_low;              /* Threshold for "low" acceptance (e.g., 0.15) */
    float accept_gamma_boost;      /* Multiplier when acceptance high (e.g., 1.5) */
    float accept_gamma_penalty;    /* Multiplier when acceptance low (e.g., 0.5) */
    
    /* Surprise adaptation */
    float surprise_threshold;      /* σ level to boost γ (e.g., 2.0) */
    float surprise_gamma_boost;    /* Multiplier when surprised (e.g., 1.3) */
    
    /* Diversity adaptation */
    float diversity_threshold;     /* ESS fraction for "high diversity" (e.g., 0.5) */
    float diversity_gamma_boost;   /* Multiplier when diverse (e.g., 1.2) */
    
} SAEMGammaConfig;

/**
 * Stickiness (κ) control configuration
 */
typedef struct {
    bool  control_stickiness;      /* Enable κ drift control */
    float target_diag_min;         /* Minimum average diagonal (e.g., 0.90) */
    float target_diag_max;         /* Maximum average diagonal (e.g., 0.98) */
    float stickiness_ema_alpha;    /* EMA for tracking κ drift (e.g., 0.1) */
} SAEMStickinessConfig;

/**
 * Tempered path injection (anti-confirmation bias)
 */
typedef struct {
    bool  enable_tempering;        /* Enable path tempering */
    float flip_probability;        /* Fraction of timesteps to flip (e.g., 0.05) */
    uint64_t seed;                 /* RNG seed for reproducibility */
} SAEMTemperingConfig;

/**
 * Full SAEM Blender configuration
 */
typedef struct {
    int   n_regimes;               /* Number of regimes (K) */
    
    SAEMGammaConfig gamma;         /* Adaptive γ settings */
    SAEMStickinessConfig stickiness; /* κ control settings */
    SAEMTemperingConfig tempering; /* Anti-confirmation bias */
    
    /* Safety bounds */
    float trans_floor;             /* Minimum Π_ij (default: 1e-5) */
    float trans_ceiling;           /* Maximum Π_ij (default: 0.9999) */
    
    /* Diagnostics */
    bool  track_history;           /* Store blend history for debugging */
    int   history_capacity;        /* Max history entries */
    
} SAEMBlenderConfig;

/*═══════════════════════════════════════════════════════════════════════════
 * STATE STRUCTURES
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Blend event record (for diagnostics)
 */
typedef struct {
    int   tick;                    /* When blend occurred */
    float gamma_used;              /* Actual γ after adaptation */
    float acceptance_rate;         /* PGAS acceptance */
    float diversity;               /* ESS / N_particles */
    float surprise_sigma;          /* Hawkes surprise at trigger */
    float pre_diag_avg;            /* Average diagonal before blend */
    float post_diag_avg;           /* Average diagonal after blend */
    float kl_divergence;           /* KL(Π_new || Π_old) */
} SAEMBlendEvent;

/**
 * Main SAEM Blender state
 */
typedef struct {
    /* Configuration */
    SAEMBlenderConfig config;
    
    /* Sufficient statistics Q[i][j] = pseudo-counts */
    float Q[SAEM_MAX_REGIMES][SAEM_MAX_REGIMES];
    
    /* Current transition matrix (derived from Q) */
    float Pi[SAEM_MAX_REGIMES][SAEM_MAX_REGIMES];
    
    /* Running statistics for adaptation */
    int   total_blends;            /* Lifetime blend count */
    float acceptance_ema;          /* EMA of acceptance rates */
    float diag_ema;                /* EMA of average diagonal (κ proxy) */
    float gamma_current;           /* Current (possibly adapted) γ */
    
    /* Tempered path RNG state */
    uint64_t rng_state;
    
    /* History buffer (circular) */
    SAEMBlendEvent *history;       /* Allocated if track_history */
    int   history_head;
    int   history_count;
    
    /* Validation */
    bool  initialized;
    
} SAEMBlender;

/*═══════════════════════════════════════════════════════════════════════════
 * ORACLE OUTPUT STRUCTURE
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * PGAS Oracle output (input to blender)
 * 
 * This is what PGAS produces after a batch run.
 */
typedef struct {
    /* Sufficient statistics: transition counts from PGAS trajectories */
    float S[SAEM_MAX_REGIMES][SAEM_MAX_REGIMES];  /* S[i][j] = count i→j */
    
    /* MCMC diagnostics */
    float acceptance_rate;         /* Fraction of accepted proposals */
    float ess_fraction;            /* ESS / N_particles */
    
    /* Metadata */
    int   n_regimes;               /* Must match blender config */
    int   n_trajectories;          /* Number of PGAS trajectories averaged */
    int   trajectory_length;       /* T for this batch */
    
    /* Trigger context */
    float trigger_surprise;        /* Hawkes surprise that triggered Oracle */
    
} PGASOutput;

/*═══════════════════════════════════════════════════════════════════════════
 * API - LIFECYCLE
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Get default configuration
 */
SAEMBlenderConfig saem_blender_config_defaults(int n_regimes);

/**
 * Initialize blender
 *
 * @param blender   Blender state to initialize
 * @param cfg       Configuration (NULL for defaults with K=4)
 * @param init_Pi   Initial transition matrix [K×K] row-major (NULL for uniform)
 * @return 0 on success, -1 on error
 */
int saem_blender_init(SAEMBlender *blender,
                      const SAEMBlenderConfig *cfg,
                      const float *init_Pi);

/**
 * Reset blender state (keep config)
 */
void saem_blender_reset(SAEMBlender *blender, const float *init_Pi);

/**
 * Free resources
 */
void saem_blender_free(SAEMBlender *blender);

/*═══════════════════════════════════════════════════════════════════════════
 * API - CORE BLENDING
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Result of a blend operation
 */
typedef struct {
    bool  success;                 /* Blend completed without error */
    float gamma_used;              /* Actual γ after adaptation */
    float kl_divergence;           /* KL(Π_new || Π_old) */
    float diag_avg_before;         /* Average diagonal before */
    float diag_avg_after;          /* Average diagonal after */
    int   cells_floored;           /* Number of cells clamped to floor */
    bool  stickiness_adjusted;     /* Whether κ control kicked in */
} SAEMBlendResult;

/**
 * Blend PGAS output into current estimate
 *
 * This is the main entry point. Call this when PGAS completes.
 *
 * @param blender   Blender state
 * @param oracle    PGAS output (sufficient statistics + diagnostics)
 * @return Blend result with diagnostics
 */
SAEMBlendResult saem_blender_blend(SAEMBlender *blender,
                                    const PGASOutput *oracle);

/**
 * Get current transition matrix
 *
 * @param blender   Blender state
 * @param Pi_out    Output buffer [K×K] row-major
 */
void saem_blender_get_Pi(const SAEMBlender *blender, float *Pi_out);

/**
 * Get current sufficient statistics
 *
 * @param blender   Blender state
 * @param Q_out     Output buffer [K×K] row-major
 */
void saem_blender_get_Q(const SAEMBlender *blender, float *Q_out);

/*═══════════════════════════════════════════════════════════════════════════
 * API - TEMPERED PATH GENERATION
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Generate tempered reference path for PGAS
 *
 * Takes RBPF MAP path and injects controlled exploration.
 *
 * @param blender       Blender state (for RNG and config)
 * @param rbpf_path     RBPF MAP path [T] (regime indices)
 * @param T             Path length
 * @param tempered_out  Output tempered path [T]
 * @return Number of flips injected
 */
int saem_blender_temper_path(SAEMBlender *blender,
                              const int *rbpf_path,
                              int T,
                              int *tempered_out);

/*═══════════════════════════════════════════════════════════════════════════
 * API - QUERIES
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Get current γ (learning rate)
 */
float saem_blender_get_gamma(const SAEMBlender *blender);

/**
 * Get average diagonal (κ proxy)
 */
float saem_blender_get_avg_diagonal(const SAEMBlender *blender);

/**
 * Get total blend count
 */
int saem_blender_get_blend_count(const SAEMBlender *blender);

/**
 * Compute KL divergence between two transition matrices
 */
float saem_blender_kl_divergence(const float *P, const float *Q, int K);

/*═══════════════════════════════════════════════════════════════════════════
 * API - MANUAL CONTROL
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Force γ to specific value (overrides adaptation)
 */
void saem_blender_set_gamma(SAEMBlender *blender, float gamma);

/**
 * Reset γ to base value
 */
void saem_blender_reset_gamma(SAEMBlender *blender);

/**
 * Inject transition matrix directly (bypass SAEM)
 * 
 * Use sparingly - this skips all safety checks except floor/normalization.
 */
void saem_blender_inject_Pi(SAEMBlender *blender, const float *Pi);

/*═══════════════════════════════════════════════════════════════════════════
 * API - DIAGNOSTICS
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Get recent blend history
 *
 * @param blender   Blender state
 * @param events    Output buffer
 * @param max_events Buffer capacity
 * @return Number of events written
 */
int saem_blender_get_history(const SAEMBlender *blender,
                              SAEMBlendEvent *events,
                              int max_events);

/**
 * Print current state
 */
void saem_blender_print_state(const SAEMBlender *blender);

/**
 * Print transition matrix
 */
void saem_blender_print_Pi(const SAEMBlender *blender);

/**
 * Print configuration
 */
void saem_blender_print_config(const SAEMBlenderConfig *cfg);

#ifdef __cplusplus
}
#endif

#endif /* SAEM_BLENDER_H */
