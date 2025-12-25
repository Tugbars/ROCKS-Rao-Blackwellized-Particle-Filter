/**
 * @file kl_trigger.c
 * @brief KL Divergence Trigger for Oracle Dual-Gate
 *
 * Implementation of KL-based surprise detection for RBPF.
 */

#include "kl_trigger.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

/*═══════════════════════════════════════════════════════════════════════════
 * HELPER FUNCTIONS
 *═══════════════════════════════════════════════════════════════════════════*/

static inline float clampf(float x, float lo, float hi) {
    return (x < lo) ? lo : (x > hi) ? hi : x;
}

static inline float maxf(float a, float b) {
    return (a > b) ? a : b;
}

/**
 * Compute KL divergence for categorical distribution
 * KL(actual || predicted) where actual is one-hot
 */
static float compute_regime_kl(const float *predicted, int actual, int K) {
    /* One-hot actual: KL = -log(predicted[actual]) */
    float p = predicted[actual];
    if (p < 1e-10f) p = 1e-10f;
    return -logf(p);
}

/**
 * Compute Gaussian innovation (normalized squared error)
 */
static float compute_gaussian_innovation(float predicted, float actual, float std) {
    if (std < 1e-6f) std = 1e-6f;
    float z = (actual - predicted) / std;
    return z * z;  /* Chi-squared with df=1 */
}

/*═══════════════════════════════════════════════════════════════════════════
 * DEFAULT CONFIGURATION
 *═══════════════════════════════════════════════════════════════════════════*/

KLTriggerConfig kl_trigger_config_defaults(int n_regimes) {
    KLTriggerConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    
    cfg.n_regimes = (n_regimes > 0 && n_regimes <= KL_TRIGGER_MAX_REGIMES) 
                    ? n_regimes : 4;
    
    /* Baseline estimation */
    cfg.baseline_ema_alpha = 0.05f;   /* Slow adaptation */
    cfg.variance_ema_alpha = 0.05f;
    cfg.warmup_ticks = 100;
    
    /* Trigger thresholds */
    cfg.trigger_sigma = 2.0f;         /* 2σ = ~P95 */
    cfg.panic_sigma = 5.0f;           /* 5σ = extreme */
    
    /* Hysteresis */
    cfg.high_water_sigma = 2.0f;
    cfg.low_water_sigma = 1.0f;
    cfg.refractory_ticks = 50;
    
    /* Innovation weighting */
    cfg.regime_weight = 0.5f;
    cfg.volatility_weight = 0.5f;
    
    return cfg;
}

/*═══════════════════════════════════════════════════════════════════════════
 * LIFECYCLE
 *═══════════════════════════════════════════════════════════════════════════*/

int kl_trigger_init(KLTrigger *trigger, const KLTriggerConfig *config) {
    if (!trigger) return -1;
    
    memset(trigger, 0, sizeof(*trigger));
    
    trigger->config = config ? *config : kl_trigger_config_defaults(4);
    
    /* Initialize baseline estimates
     * Start with reasonable defaults for log-volatility innovations */
    trigger->baseline_mean = 1.0f;    /* Expected chi-squared(1) */
    trigger->baseline_std = 1.41f;    /* sqrt(2) for chi-squared(1) */
    trigger->innovation_ema = 1.0f;
    trigger->innovation_var_ema = 2.0f;
    
    trigger->state = KL_STATE_CALM;
    trigger->ticks_since_trigger = trigger->config.refractory_ticks;  /* Allow immediate trigger */
    
    trigger->initialized = true;
    return 0;
}

void kl_trigger_reset(KLTrigger *trigger) {
    if (!trigger || !trigger->initialized) return;
    
    KLTriggerConfig cfg = trigger->config;
    
    trigger->baseline_mean = 1.0f;
    trigger->baseline_std = 1.41f;
    trigger->innovation_ema = 1.0f;
    trigger->innovation_var_ema = 2.0f;
    
    trigger->state = KL_STATE_CALM;
    trigger->current_surprise = 0.0f;
    trigger->cumulative_surprise = 0.0f;
    trigger->ticks_since_trigger = cfg.refractory_ticks;
    trigger->total_ticks = 0;
    
    memset(trigger->history, 0, sizeof(trigger->history));
    trigger->history_idx = 0;
    trigger->history_count = 0;
    
    trigger->total_triggers = 0;
    trigger->panic_triggers = 0;
    trigger->max_surprise_seen = 0.0f;
}

void kl_trigger_free(KLTrigger *trigger) {
    if (trigger) {
        memset(trigger, 0, sizeof(*trigger));
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * CORE INNOVATION COMPUTATION
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Compute combined innovation magnitude from all sources
 */
static float compute_innovation_magnitude(const KLTrigger *trigger,
                                          const KLInnovation *innov) {
    const KLTriggerConfig *cfg = &trigger->config;
    float total = 0.0f;
    float weight_sum = 0.0f;
    
    /* Regime innovation (KL divergence) */
    if (cfg->regime_weight > 0.0f && innov->regime_actual >= 0) {
        float regime_kl = compute_regime_kl(innov->regime_predicted,
                                            innov->regime_actual,
                                            cfg->n_regimes);
        total += cfg->regime_weight * regime_kl;
        weight_sum += cfg->regime_weight;
    }
    
    /* Volatility innovation (normalized squared error) */
    if (cfg->volatility_weight > 0.0f && innov->vol_std > 0.0f) {
        float vol_innov = compute_gaussian_innovation(innov->vol_predicted,
                                                       innov->vol_actual,
                                                       innov->vol_std);
        total += cfg->volatility_weight * vol_innov;
        weight_sum += cfg->volatility_weight;
    }
    
    /* Observation innovation (if provided) */
    if (innov->obs_std > 0.0f) {
        float obs_innov = compute_gaussian_innovation(innov->obs_predicted,
                                                       innov->obs_actual,
                                                       innov->obs_std);
        /* Use remaining weight or equal weight */
        float obs_weight = maxf(0.0f, 1.0f - weight_sum);
        if (obs_weight > 0.0f) {
            total += obs_weight * obs_innov;
            weight_sum += obs_weight;
        }
    }
    
    /* Normalize by total weight */
    if (weight_sum > 1e-6f) {
        total /= weight_sum;
    }
    
    return total;
}

/*═══════════════════════════════════════════════════════════════════════════
 * BASELINE ESTIMATION
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Update baseline statistics with new innovation
 */
static void update_baseline(KLTrigger *trigger, float innovation) {
    const KLTriggerConfig *cfg = &trigger->config;
    
    /* Store in history buffer */
    trigger->history[trigger->history_idx] = innovation;
    trigger->history_idx = (trigger->history_idx + 1) % KL_TRIGGER_HISTORY_SIZE;
    if (trigger->history_count < KL_TRIGGER_HISTORY_SIZE) {
        trigger->history_count++;
    }
    
    /* EMA updates for online estimation */
    float alpha = cfg->baseline_ema_alpha;
    
    /* Mean estimation */
    float delta = innovation - trigger->innovation_ema;
    trigger->innovation_ema += alpha * delta;
    
    /* Variance estimation (Welford's method with EMA) */
    trigger->innovation_var_ema = (1.0f - alpha) * trigger->innovation_var_ema +
                                   alpha * delta * delta;
    
    /* Update baseline from EMA (more stable than raw values) */
    trigger->baseline_mean = trigger->innovation_ema;
    trigger->baseline_std = sqrtf(trigger->innovation_var_ema + 1e-10f);
    
    /* Ensure reasonable bounds */
    trigger->baseline_std = clampf(trigger->baseline_std, 0.1f, 10.0f);
}

/*═══════════════════════════════════════════════════════════════════════════
 * STATE MACHINE
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Update state machine based on surprise level
 */
static void update_state_machine(KLTrigger *trigger, float surprise_sigma) {
    const KLTriggerConfig *cfg = &trigger->config;
    KLTriggerState prev_state = trigger->state;
    
    /* Check panic first */
    if (surprise_sigma >= cfg->panic_sigma) {
        trigger->state = KL_STATE_PANIC;
    }
    /* Hysteresis logic */
    else if (surprise_sigma >= cfg->high_water_sigma) {
        trigger->state = KL_STATE_TRIGGERED;
    }
    else if (surprise_sigma <= cfg->low_water_sigma) {
        trigger->state = KL_STATE_CALM;
    }
    else {
        trigger->state = KL_STATE_ELEVATED;
    }
    
    /* Track state transitions for debugging */
    (void)prev_state;  /* Suppress unused warning */
}

/*═══════════════════════════════════════════════════════════════════════════
 * CORE UPDATE
 *═══════════════════════════════════════════════════════════════════════════*/

KLTriggerResult kl_trigger_update(KLTrigger *trigger, const KLInnovation *innovation) {
    KLTriggerResult result;
    memset(&result, 0, sizeof(result));
    
    if (!trigger || !trigger->initialized || !innovation) {
        result.state = KL_STATE_CALM;
        return result;
    }
    
    trigger->total_ticks++;
    trigger->ticks_since_trigger++;
    
    /* Compute innovation magnitude */
    float innov_mag = compute_innovation_magnitude(trigger, innovation);
    
    /* Update baseline statistics */
    update_baseline(trigger, innov_mag);
    
    /* Compute surprise in σ units */
    float surprise_sigma = 0.0f;
    if (trigger->baseline_std > 1e-6f) {
        surprise_sigma = (innov_mag - trigger->baseline_mean) / trigger->baseline_std;
    }
    trigger->current_surprise = surprise_sigma;
    
    /* Track max surprise */
    if (surprise_sigma > trigger->max_surprise_seen) {
        trigger->max_surprise_seen = surprise_sigma;
    }
    
    /* Update cumulative surprise (decaying sum) */
    float decay = 0.95f;  /* ~20 tick half-life */
    trigger->cumulative_surprise = decay * trigger->cumulative_surprise + 
                                   maxf(0.0f, surprise_sigma);
    
    /* Update state machine */
    update_state_machine(trigger, surprise_sigma);
    
    /* Check if should trigger */
    bool can_trigger = (trigger->total_ticks >= trigger->config.warmup_ticks) &&
                       (trigger->ticks_since_trigger >= trigger->config.refractory_ticks);
    
    bool should_trigger = can_trigger && 
                          (trigger->state == KL_STATE_TRIGGERED || 
                           trigger->state == KL_STATE_PANIC);
    
    bool is_panic = (trigger->state == KL_STATE_PANIC);
    
    /* Record trigger */
    if (should_trigger) {
        trigger->total_triggers++;
        if (is_panic) {
            trigger->panic_triggers++;
        }
        /* Note: Don't reset ticks_since_trigger here - caller must call acknowledge() */
    }
    
    /* Build result */
    result.state = trigger->state;
    result.surprise_sigma = surprise_sigma;
    result.cumulative_surprise = trigger->cumulative_surprise;
    result.should_trigger = should_trigger;
    result.is_panic = is_panic;
    
    return result;
}

/*═══════════════════════════════════════════════════════════════════════════
 * SIMPLIFIED UPDATE FUNCTIONS
 *═══════════════════════════════════════════════════════════════════════════*/

KLTriggerResult kl_trigger_update_vol(KLTrigger *trigger,
                                       float vol_predicted,
                                       float vol_actual,
                                       float vol_std) {
    KLInnovation innov;
    memset(&innov, 0, sizeof(innov));
    
    innov.regime_actual = -1;  /* No regime info */
    innov.vol_predicted = vol_predicted;
    innov.vol_actual = vol_actual;
    innov.vol_std = (vol_std > 0.0f) ? vol_std : 0.1f;
    
    return kl_trigger_update(trigger, &innov);
}

KLTriggerResult kl_trigger_update_obs(KLTrigger *trigger,
                                       float obs_predicted,
                                       float obs_actual,
                                       float obs_std) {
    KLInnovation innov;
    memset(&innov, 0, sizeof(innov));
    
    innov.regime_actual = -1;  /* No regime info */
    innov.obs_predicted = obs_predicted;
    innov.obs_actual = obs_actual;
    innov.obs_std = obs_std;
    
    return kl_trigger_update(trigger, &innov);
}

/*═══════════════════════════════════════════════════════════════════════════
 * ACKNOWLEDGE
 *═══════════════════════════════════════════════════════════════════════════*/

void kl_trigger_acknowledge(KLTrigger *trigger) {
    if (trigger && trigger->initialized) {
        trigger->ticks_since_trigger = 0;
        /* Optionally reset cumulative surprise */
        trigger->cumulative_surprise *= 0.5f;  /* Partial reset */
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * QUERIES
 *═══════════════════════════════════════════════════════════════════════════*/

KLTriggerState kl_trigger_get_state(const KLTrigger *trigger) {
    return trigger ? trigger->state : KL_STATE_CALM;
}

float kl_trigger_get_surprise(const KLTrigger *trigger) {
    return trigger ? trigger->current_surprise : 0.0f;
}

float kl_trigger_get_cumulative(const KLTrigger *trigger) {
    return trigger ? trigger->cumulative_surprise : 0.0f;
}

bool kl_trigger_is_triggered(const KLTrigger *trigger) {
    return trigger && (trigger->state == KL_STATE_TRIGGERED || 
                       trigger->state == KL_STATE_PANIC);
}

bool kl_trigger_is_panic(const KLTrigger *trigger) {
    return trigger && (trigger->state == KL_STATE_PANIC);
}

void kl_trigger_get_baseline(const KLTrigger *trigger,
                              float *mean_out, float *std_out) {
    if (trigger) {
        if (mean_out) *mean_out = trigger->baseline_mean;
        if (std_out) *std_out = trigger->baseline_std;
    } else {
        if (mean_out) *mean_out = 0.0f;
        if (std_out) *std_out = 0.0f;
    }
}

int kl_trigger_get_trigger_count(const KLTrigger *trigger) {
    return trigger ? trigger->total_triggers : 0;
}

/*═══════════════════════════════════════════════════════════════════════════
 * DIAGNOSTICS
 *═══════════════════════════════════════════════════════════════════════════*/

static const char *state_name(KLTriggerState state) {
    switch (state) {
        case KL_STATE_CALM:      return "CALM";
        case KL_STATE_ELEVATED:  return "ELEVATED";
        case KL_STATE_TRIGGERED: return "TRIGGERED";
        case KL_STATE_PANIC:     return "PANIC";
        default:                 return "UNKNOWN";
    }
}

void kl_trigger_print_state(const KLTrigger *trigger) {
    if (!trigger) {
        printf("KLTrigger: NULL\n");
        return;
    }
    
    printf("═══════════════════════════════════════════════════════════\n");
    printf("KL TRIGGER STATE\n");
    printf("═══════════════════════════════════════════════════════════\n");
    printf("State:              %s\n", state_name(trigger->state));
    printf("Current surprise:   %.3f σ\n", trigger->current_surprise);
    printf("Cumulative:         %.3f\n", trigger->cumulative_surprise);
    printf("Baseline mean:      %.4f\n", trigger->baseline_mean);
    printf("Baseline std:       %.4f\n", trigger->baseline_std);
    printf("Total ticks:        %d\n", trigger->total_ticks);
    printf("Ticks since trigger: %d\n", trigger->ticks_since_trigger);
    printf("Total triggers:     %d\n", trigger->total_triggers);
    printf("Panic triggers:     %d\n", trigger->panic_triggers);
    printf("Max surprise seen:  %.3f σ\n", trigger->max_surprise_seen);
    printf("═══════════════════════════════════════════════════════════\n");
}

void kl_trigger_print_config(const KLTriggerConfig *cfg) {
    if (!cfg) {
        printf("KLTriggerConfig: NULL\n");
        return;
    }
    
    printf("═══════════════════════════════════════════════════════════\n");
    printf("KL TRIGGER CONFIG\n");
    printf("═══════════════════════════════════════════════════════════\n");
    printf("n_regimes:          %d\n", cfg->n_regimes);
    printf("baseline_ema_alpha: %.3f\n", cfg->baseline_ema_alpha);
    printf("warmup_ticks:       %d\n", cfg->warmup_ticks);
    printf("trigger_sigma:      %.2f\n", cfg->trigger_sigma);
    printf("panic_sigma:        %.2f\n", cfg->panic_sigma);
    printf("high_water_sigma:   %.2f\n", cfg->high_water_sigma);
    printf("low_water_sigma:    %.2f\n", cfg->low_water_sigma);
    printf("refractory_ticks:   %d\n", cfg->refractory_ticks);
    printf("regime_weight:      %.2f\n", cfg->regime_weight);
    printf("volatility_weight:  %.2f\n", cfg->volatility_weight);
    printf("═══════════════════════════════════════════════════════════\n");
}
