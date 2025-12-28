/**
 * @file injection_decision.h
 * @brief Injection Decision Engine
 *
 * Combines all signals into a single injection decision:
 *   - Π Quality metrics (RMSE, likelihood, lag, D₃)
 *   - Divergence triad (D₁, D₂, D₃)
 *   - Oracle confidence
 *   - Market signals (Hawkes)
 *
 * Output: Should inject? What γ? Use Thompson sample?
 */

#ifndef INJECTION_DECISION_H
#define INJECTION_DECISION_H

#include <stdbool.h>
#include "pi_quality.h"
#include "pi_divergence.h"

#ifdef __cplusplus
extern "C" {
#endif

/*═══════════════════════════════════════════════════════════════════════════
 * INJECTION SOURCE
 *═══════════════════════════════════════════════════════════════════════════*/

typedef enum {
    INJECT_NONE = 0,           /* No injection */
    INJECT_ORACLE,             /* Use Oracle Π */
    INJECT_STORVIK,            /* Self-correct with Storvik */
    INJECT_ORACLE_BOOSTED,     /* Oracle + Storvik agree: high γ */
    INJECT_THOMPSON,           /* Thompson sample (variance shot) */
    INJECT_RESET               /* Full reset from prior */
} InjectionSource;

/*═══════════════════════════════════════════════════════════════════════════
 * URGENCY LEVEL
 *═══════════════════════════════════════════════════════════════════════════*/

typedef enum {
    URGENCY_NONE = 0,          /* No action needed */
    URGENCY_LOW,               /* Can wait for better signal */
    URGENCY_MEDIUM,            /* Should inject soon */
    URGENCY_HIGH,              /* Need to inject */
    URGENCY_EMERGENCY          /* Must inject NOW */
} InjectionUrgency;

/*═══════════════════════════════════════════════════════════════════════════
 * FULL SYSTEM STATE (Input to decision engine)
 *═══════════════════════════════════════════════════════════════════════════*/

typedef struct {
    /* Π Quality (from pi_quality.h) */
    PiQualitySnapshot quality;
    
    /* Divergences (from pi_divergence.h) */
    DivergenceTriad divergence;
    DivergenceScenario scenario;
    
    /* Oracle (from PGAS thread) */
    bool  oracle_available;
    float oracle_confidence;
    
    /* Market activity */
    float hawkes_intensity;        /* >2.0 = hot market */
    float hawkes_surprise_sigma;   /* Surprise level */
    
} SystemState;

/*═══════════════════════════════════════════════════════════════════════════
 * INJECTION DECISION (Output)
 *═══════════════════════════════════════════════════════════════════════════*/

typedef struct {
    /* Action */
    bool              should_inject;
    InjectionSource   source;
    float             gamma;
    bool              use_thompson;   /* Sample from posterior? */
    
    /* Urgency */
    InjectionUrgency  urgency;
    float             urgency_score;  /* 0-1 continuous */
    
    /* Diagnostics */
    const char       *reason;
    
} InjectionDecision;

/*═══════════════════════════════════════════════════════════════════════════
 * CONFIGURATION
 *═══════════════════════════════════════════════════════════════════════════*/

typedef struct {
    /* Urgency thresholds (quality score) */
    float quality_critical;        /* Below this = EMERGENCY (default: 0.25) */
    float quality_high;            /* Below this = HIGH urgency (default: 0.40) */
    float quality_medium;          /* Below this = MEDIUM urgency (default: 0.60) */
    
    /* ESS thresholds */
    float ess_emergency;           /* Below this = EMERGENCY (default: 0.15) */
    float ess_critical;            /* Below this = HIGH (default: 0.25) */
    float ess_degraded;            /* Below this = MEDIUM (default: 0.40) */
    
    /* Oracle confidence thresholds */
    float oracle_conf_high;        /* Above this = confident (default: 0.60) */
    float oracle_conf_medium;      /* Above this = moderate (default: 0.35) */
    
    /* γ parameters */
    float gamma_min;               /* Minimum γ (default: 0.05) */
    float gamma_max;               /* Maximum γ (default: 0.60) */
    float gamma_emergency;         /* γ for emergency (default: 0.40) */
    float gamma_boosted;           /* γ when Oracle+Storvik agree (default: 0.45) */
    
    /* Hawkes influence */
    float hawkes_hot_threshold;    /* Above this = hot market (default: 2.0) */
    float hawkes_urgency_boost;    /* Urgency multiplier when hot (default: 1.3) */
    
    /* Divergence thresholds */
    DivergenceThresholds div_thresh;
    
} InjectionConfig;

/*═══════════════════════════════════════════════════════════════════════════
 * API
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Get default configuration
 */
InjectionConfig injection_config_defaults(void);

/**
 * Compute injection decision from system state
 */
InjectionDecision injection_decide(
    const SystemState *state,
    const InjectionConfig *config);

/**
 * Compute urgency score (0-1)
 */
float injection_compute_urgency_score(
    const SystemState *state,
    const InjectionConfig *config);

/**
 * Compute γ from urgency and confidence
 */
float injection_compute_gamma(
    InjectionUrgency urgency,
    float urgency_score,
    float oracle_confidence,
    DivergenceScenario scenario,
    const InjectionConfig *config);

/**
 * Get urgency level string
 */
const char *injection_urgency_str(InjectionUrgency urgency);

/**
 * Get source string
 */
const char *injection_source_str(InjectionSource source);

#ifdef __cplusplus
}
#endif

#endif /* INJECTION_DECISION_H */
