/**
 * @file kl_trigger.h
 * @brief KL Divergence Trigger for Oracle Dual-Gate
 *
 * Tracks prediction errors from RBPF and computes KL surprise.
 * Works with Hawkes integrator to form the dual-gate trigger.
 *
 * The idea: RBPF makes one-step-ahead predictions. When actual observations
 * diverge significantly from predictions, the model is "surprised" - indicating
 * a potential regime change that warrants Oracle intervention.
 *
 * Metrics tracked:
 *   1. Prediction error: |y_t - ŷ_t|² or likelihood-based
 *   2. Innovation KL: KL(P_observed || P_predicted)
 *   3. Cumulative surprise: Running sum with decay
 *
 * Reference: ORACLE_INTEGRATION_PLAN.md
 */

#ifndef KL_TRIGGER_H
#define KL_TRIGGER_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*═══════════════════════════════════════════════════════════════════════════
 * CONFIGURATION
 *═══════════════════════════════════════════════════════════════════════════*/

#define KL_TRIGGER_MAX_REGIMES 8
#define KL_TRIGGER_HISTORY_SIZE 256  /* Rolling window for baseline estimation */

/**
 * KL Trigger configuration
 */
typedef struct {
    int n_regimes;                 /* Number of regimes (K) */
    
    /* Baseline estimation */
    float baseline_ema_alpha;      /* EMA decay for baseline (default: 0.05) */
    float variance_ema_alpha;      /* EMA decay for variance (default: 0.05) */
    int   warmup_ticks;            /* Ticks before triggering enabled (default: 100) */
    
    /* Trigger thresholds */
    float trigger_sigma;           /* σ threshold for trigger (default: 2.0) */
    float panic_sigma;             /* σ threshold for panic (default: 5.0) */
    
    /* Hysteresis (prevent rapid re-triggering) */
    float high_water_sigma;        /* Enter triggered state (default: 2.0) */
    float low_water_sigma;         /* Exit triggered state (default: 1.0) */
    int   refractory_ticks;        /* Minimum ticks between triggers (default: 50) */
    
    /* Innovation weighting */
    float regime_weight;           /* Weight for regime prediction error (default: 0.5) */
    float volatility_weight;       /* Weight for volatility prediction error (default: 0.5) */
    
} KLTriggerConfig;

/*═══════════════════════════════════════════════════════════════════════════
 * STATE
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Innovation observation (one-step-ahead prediction vs actual)
 */
typedef struct {
    /* Regime innovation */
    float regime_predicted[KL_TRIGGER_MAX_REGIMES];  /* P(z_t | y_{1:t-1}) */
    int   regime_actual;                              /* Actual regime (MAP or sampled) */
    
    /* Volatility innovation */
    float vol_predicted;           /* E[h_t | y_{1:t-1}] */
    float vol_actual;              /* h_t (from filter) */
    float vol_std;                 /* Std dev of prediction */
    
    /* Observation innovation */
    float obs_predicted;           /* E[y_t | y_{1:t-1}] */
    float obs_actual;              /* y_t */
    float obs_std;                 /* Std dev of prediction */
    
} KLInnovation;

/**
 * Trigger state
 */
typedef enum {
    KL_STATE_CALM = 0,        /* Below low water mark */
    KL_STATE_ELEVATED = 1,    /* Between low and high water */
    KL_STATE_TRIGGERED = 2,   /* Above high water mark */
    KL_STATE_PANIC = 3        /* Extreme surprise */
} KLTriggerState;

/**
 * KL Trigger main state
 */
typedef struct {
    /* Configuration */
    KLTriggerConfig config;
    
    /* Running statistics */
    float innovation_ema;          /* EMA of innovation magnitude */
    float innovation_var_ema;      /* EMA of innovation variance */
    float baseline_mean;           /* Estimated baseline innovation */
    float baseline_std;            /* Estimated baseline std dev */
    
    /* Current state */
    KLTriggerState state;
    float current_surprise;        /* Current innovation in σ units */
    float cumulative_surprise;     /* Cumulative with decay */
    int   ticks_since_trigger;     /* For refractory period */
    int   total_ticks;             /* For warmup */
    
    /* History for robust baseline */
    float history[KL_TRIGGER_HISTORY_SIZE];
    int   history_idx;
    int   history_count;
    
    /* Trigger statistics */
    int   total_triggers;
    int   panic_triggers;
    float max_surprise_seen;
    
    /* Validation */
    bool  initialized;
    
} KLTrigger;

/*═══════════════════════════════════════════════════════════════════════════
 * API - LIFECYCLE
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Get default configuration
 */
KLTriggerConfig kl_trigger_config_defaults(int n_regimes);

/**
 * Initialize KL trigger
 */
int kl_trigger_init(KLTrigger *trigger, const KLTriggerConfig *config);

/**
 * Reset state (keep config)
 */
void kl_trigger_reset(KLTrigger *trigger);

/**
 * Free resources (currently no-op, but for future extensibility)
 */
void kl_trigger_free(KLTrigger *trigger);

/*═══════════════════════════════════════════════════════════════════════════
 * API - CORE OPERATIONS
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Result of update operation
 */
typedef struct {
    KLTriggerState state;          /* Current state */
    float surprise_sigma;          /* Innovation in σ units */
    float cumulative_surprise;     /* Cumulative surprise */
    bool  should_trigger;          /* True if Oracle should be called */
    bool  is_panic;                /* True if panic threshold exceeded */
} KLTriggerResult;

/**
 * Update with new innovation observation
 *
 * Call this each tick with RBPF prediction vs actual.
 *
 * @param trigger     Trigger state
 * @param innovation  One-step-ahead prediction errors
 * @return            Trigger result
 */
KLTriggerResult kl_trigger_update(KLTrigger *trigger, const KLInnovation *innovation);

/**
 * Simplified update with just volatility innovation
 *
 * Use when you only have volatility prediction errors.
 *
 * @param trigger       Trigger state
 * @param vol_predicted Predicted log-volatility
 * @param vol_actual    Actual log-volatility
 * @param vol_std       Prediction std dev (use 0.1 if unknown)
 * @return              Trigger result
 */
KLTriggerResult kl_trigger_update_vol(KLTrigger *trigger,
                                       float vol_predicted,
                                       float vol_actual,
                                       float vol_std);

/**
 * Simplified update with observation innovation
 *
 * Use when you have observation-level prediction errors.
 *
 * @param trigger       Trigger state
 * @param obs_predicted Predicted observation
 * @param obs_actual    Actual observation
 * @param obs_std       Prediction std dev
 * @return              Trigger result
 */
KLTriggerResult kl_trigger_update_obs(KLTrigger *trigger,
                                       float obs_predicted,
                                       float obs_actual,
                                       float obs_std);

/**
 * Acknowledge trigger (reset refractory period)
 *
 * Call after Oracle has run to prevent immediate re-trigger.
 */
void kl_trigger_acknowledge(KLTrigger *trigger);

/*═══════════════════════════════════════════════════════════════════════════
 * API - QUERIES
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Get current state
 */
KLTriggerState kl_trigger_get_state(const KLTrigger *trigger);

/**
 * Get current surprise in σ units
 */
float kl_trigger_get_surprise(const KLTrigger *trigger);

/**
 * Get cumulative surprise
 */
float kl_trigger_get_cumulative(const KLTrigger *trigger);

/**
 * Check if in triggered state
 */
bool kl_trigger_is_triggered(const KLTrigger *trigger);

/**
 * Check if in panic state
 */
bool kl_trigger_is_panic(const KLTrigger *trigger);

/**
 * Get baseline statistics
 */
void kl_trigger_get_baseline(const KLTrigger *trigger,
                              float *mean_out, float *std_out);

/**
 * Get trigger count
 */
int kl_trigger_get_trigger_count(const KLTrigger *trigger);

/*═══════════════════════════════════════════════════════════════════════════
 * API - DIAGNOSTICS
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Print current state
 */
void kl_trigger_print_state(const KLTrigger *trigger);

/**
 * Print configuration
 */
void kl_trigger_print_config(const KLTriggerConfig *config);

#ifdef __cplusplus
}
#endif

#endif /* KL_TRIGGER_H */
