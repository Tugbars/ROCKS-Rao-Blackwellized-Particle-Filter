/**
 * @file injection_decision.c
 * @brief Injection Decision Engine Implementation
 */

#include "injection_decision.h"
#include <math.h>
#include <string.h>

/*═══════════════════════════════════════════════════════════════════════════
 * DEFAULTS
 *═══════════════════════════════════════════════════════════════════════════*/

InjectionConfig injection_config_defaults(void) {
    InjectionConfig cfg;
    
    /* Quality thresholds */
    cfg.quality_critical = 0.25f;
    cfg.quality_high = 0.40f;
    cfg.quality_medium = 0.60f;
    
    /* ESS thresholds */
    cfg.ess_emergency = 0.15f;
    cfg.ess_critical = 0.25f;
    cfg.ess_degraded = 0.40f;
    
    /* Oracle confidence */
    cfg.oracle_conf_high = 0.60f;
    cfg.oracle_conf_medium = 0.35f;
    
    /* γ parameters */
    cfg.gamma_min = 0.05f;
    cfg.gamma_max = 0.60f;
    cfg.gamma_emergency = 0.40f;
    cfg.gamma_boosted = 0.45f;
    
    /* Hawkes */
    cfg.hawkes_hot_threshold = 2.0f;
    cfg.hawkes_urgency_boost = 1.3f;
    
    /* Divergence */
    cfg.div_thresh = divergence_thresholds_defaults();
    
    return cfg;
}

/*═══════════════════════════════════════════════════════════════════════════
 * URGENCY COMPUTATION
 *═══════════════════════════════════════════════════════════════════════════*/

float injection_compute_urgency_score(
    const SystemState *state,
    const InjectionConfig *config)
{
    if (!state || !config) return 0.0f;
    
    float urgency = 0.0f;
    const PiQualitySnapshot *q = &state->quality;
    
    /*═══════════════════════════════════════════════════════════════════
     * EARLY SIGNALS (High weight)
     *═══════════════════════════════════════════════════════════════════*/
    
    /* RMSE spike */
    if (q->rmse_ratio > 3.0) {
        urgency += 0.40f;
    } else if (q->rmse_ratio > 2.0) {
        urgency += 0.25f;
    } else if (q->rmse_ratio > 1.5) {
        urgency += 0.12f;
    }
    
    /* Likelihood z-score */
    if (q->likelihood_zscore < -4.0) {
        urgency += 0.35f;
    } else if (q->likelihood_zscore < -3.0) {
        urgency += 0.20f;
    } else if (q->likelihood_zscore < -2.0) {
        urgency += 0.10f;
    }
    
    /* D₃ self-contradiction */
    if (q->d3_self_contradiction > 0.25) {
        urgency += 0.35f;
    } else if (q->d3_self_contradiction > 0.15) {
        urgency += 0.20f;
    } else if (q->d3_self_contradiction > 0.10) {
        urgency += 0.10f;
    }
    
    /* Transition lag */
    if (q->avg_transition_lag > 8.0) {
        urgency += 0.25f;
    } else if (q->avg_transition_lag > 5.0) {
        urgency += 0.15f;
    } else if (q->avg_transition_lag > 3.0) {
        urgency += 0.08f;
    }
    
    /*═══════════════════════════════════════════════════════════════════
     * MEDIUM SIGNALS
     *═══════════════════════════════════════════════════════════════════*/
    
    /* Weight concentration */
    if (q->max_weight > 0.7f) {
        urgency += 0.15f;
    } else if (q->max_weight > 0.5f) {
        urgency += 0.08f;
    }
    
    /* Weight variance */
    if (q->weight_variance > 0.15) {
        urgency += 0.10f;
    } else if (q->weight_variance > 0.10) {
        urgency += 0.05f;
    }
    
    /* Hawkes (market activity) */
    if (state->hawkes_intensity > config->hawkes_hot_threshold) {
        urgency *= config->hawkes_urgency_boost;
    }
    
    /*═══════════════════════════════════════════════════════════════════
     * LATE SIGNALS (Emergency)
     *═══════════════════════════════════════════════════════════════════*/
    
    /* ESS */
    if (q->ess_ratio < config->ess_emergency) {
        urgency += 0.40f;  /* Emergency */
    } else if (q->ess_ratio < config->ess_critical) {
        urgency += 0.25f;
    } else if (q->ess_ratio < config->ess_degraded) {
        urgency += 0.12f;
    }
    
    return fminf(1.0f, urgency);
}

static InjectionUrgency urgency_from_score(
    float score,
    const SystemState *state,
    const InjectionConfig *config)
{
    /* ESS emergency overrides */
    if (state->quality.ess_ratio < config->ess_emergency) {
        return URGENCY_EMERGENCY;
    }
    
    /* Quality score emergency */
    if (state->quality.quality_score < config->quality_critical) {
        return URGENCY_EMERGENCY;
    }
    
    /* Score-based levels */
    if (score > 0.75f) return URGENCY_EMERGENCY;
    if (score > 0.55f) return URGENCY_HIGH;
    if (score > 0.35f) return URGENCY_MEDIUM;
    if (score > 0.15f) return URGENCY_LOW;
    
    return URGENCY_NONE;
}

/*═══════════════════════════════════════════════════════════════════════════
 * γ COMPUTATION
 *═══════════════════════════════════════════════════════════════════════════*/

float injection_compute_gamma(
    InjectionUrgency urgency,
    float urgency_score,
    float oracle_confidence,
    DivergenceScenario scenario,
    const InjectionConfig *config)
{
    if (!config) return 0.1f;
    
    float gamma = config->gamma_min;
    
    /*═══════════════════════════════════════════════════════════════════
     * γ DECISION MATRIX
     *
     *                    Oracle Confidence
     *               LOW (<0.35)   MED (0.35-0.6)   HIGH (>0.6)
     *         ┌─────────────┬──────────────┬──────────────┐
     * NONE    │   0.05      │    0.10      │   0.15       │
     *         ├─────────────┼──────────────┼──────────────┤
     * LOW     │   0.08      │    0.15      │   0.25       │
     *         ├─────────────┼──────────────┼──────────────┤
     * MEDIUM  │   0.12      │    0.22      │   0.35       │
     *         ├─────────────┼──────────────┼──────────────┤
     * HIGH    │   0.18      │    0.30      │   0.45       │
     *         ├─────────────┼──────────────┼──────────────┤
     * EMERG   │   0.25      │    0.40      │   0.55       │
     *         └─────────────┴──────────────┴──────────────┘
     *═══════════════════════════════════════════════════════════════════*/
    
    /* Base γ from urgency */
    float base_gamma;
    switch (urgency) {
        case URGENCY_EMERGENCY:
            base_gamma = 0.25f;
            break;
        case URGENCY_HIGH:
            base_gamma = 0.18f;
            break;
        case URGENCY_MEDIUM:
            base_gamma = 0.12f;
            break;
        case URGENCY_LOW:
            base_gamma = 0.08f;
            break;
        default:
            base_gamma = config->gamma_min;
            break;
    }
    
    /* Add confidence component */
    if (oracle_confidence > config->oracle_conf_high) {
        gamma = base_gamma + 0.30f * oracle_confidence;
    } else if (oracle_confidence > config->oracle_conf_medium) {
        gamma = base_gamma + 0.20f * oracle_confidence;
    } else {
        gamma = base_gamma + 0.10f * oracle_confidence;
    }
    
    /* Boost for strong agreement (Oracle + Storvik agree) */
    if (scenario == DIV_SCENARIO_STRONG_AGREEMENT) {
        gamma = fmaxf(gamma, config->gamma_boosted);
    }
    
    /* Emergency cap */
    if (urgency == URGENCY_EMERGENCY) {
        gamma = fmaxf(gamma, config->gamma_emergency);
    }
    
    /* Clamp */
    return fmaxf(config->gamma_min, fminf(config->gamma_max, gamma));
}

/*═══════════════════════════════════════════════════════════════════════════
 * MAIN DECISION LOGIC
 *═══════════════════════════════════════════════════════════════════════════*/

InjectionDecision injection_decide(
    const SystemState *state,
    const InjectionConfig *config)
{
    InjectionDecision dec;
    memset(&dec, 0, sizeof(dec));
    dec.source = INJECT_NONE;
    dec.reason = "No action needed";
    
    if (!state || !config) return dec;
    
    /* Compute urgency */
    dec.urgency_score = injection_compute_urgency_score(state, config);
    dec.urgency = urgency_from_score(dec.urgency_score, state, config);
    
    const PiQualitySnapshot *q = &state->quality;
    DivergenceScenario scenario = state->scenario;
    
    /*═══════════════════════════════════════════════════════════════════
     * PRIORITY 1: ESS EMERGENCY
     * If RBPF is collapsing, we MUST inject something
     *═══════════════════════════════════════════════════════════════════*/
    
    if (q->ess_ratio < config->ess_emergency) {
        dec.should_inject = true;
        dec.urgency = URGENCY_EMERGENCY;
        dec.use_thompson = true;  /* Variance shot */
        
        if (state->oracle_available) {
            dec.source = INJECT_THOMPSON;
            dec.gamma = config->gamma_emergency;
            dec.reason = "ESS emergency, Thompson sample from Oracle";
        } else {
            dec.source = INJECT_STORVIK;
            dec.gamma = 0.25f;
            dec.reason = "ESS emergency, no Oracle, applying Storvik";
        }
        
        return dec;
    }
    
    /*═══════════════════════════════════════════════════════════════════
     * PRIORITY 2: STRONG AGREEMENT (Oracle + Storvik agree)
     * Two independent sources say Operating is wrong → High confidence
     *═══════════════════════════════════════════════════════════════════*/
    
    if (scenario == DIV_SCENARIO_STRONG_AGREEMENT && state->oracle_available) {
        dec.should_inject = true;
        dec.source = INJECT_ORACLE_BOOSTED;
        dec.gamma = injection_compute_gamma(
            dec.urgency, dec.urgency_score,
            state->oracle_confidence, scenario, config);
        dec.use_thompson = false;
        dec.reason = "Oracle + Storvik agree, Operating is stale";
        return dec;
    }
    
    /*═══════════════════════════════════════════════════════════════════
     * PRIORITY 3: SELF-CONTRADICTION (D₃ high, no Oracle or Oracle disagrees)
     * RBPF's learning contradicts its assumption
     *═══════════════════════════════════════════════════════════════════*/
    
    if (scenario == DIV_SCENARIO_SELF_CONTRADICTION || 
        q->d3_self_contradiction > config->div_thresh.d3_danger) {
        
        if (state->oracle_available && 
            state->oracle_confidence > config->oracle_conf_medium) {
            /* Oracle available and confident - use it */
            dec.should_inject = true;
            dec.source = INJECT_ORACLE;
            dec.gamma = injection_compute_gamma(
                dec.urgency, dec.urgency_score,
                state->oracle_confidence, scenario, config);
            dec.use_thompson = (state->oracle_confidence < config->oracle_conf_high);
            dec.reason = "Self-contradiction, Oracle confident";
        } else if (q->d3_self_contradiction > 0.20) {
            /* Strong self-contradiction, no Oracle - self-correct */
            dec.should_inject = true;
            dec.source = INJECT_STORVIK;
            dec.gamma = 0.15f;  /* Conservative */
            dec.use_thompson = false;
            dec.reason = "Self-contradiction, applying Storvik drift";
        }
        
        if (dec.should_inject) return dec;
    }
    
    /*═══════════════════════════════════════════════════════════════════
     * PRIORITY 4: ORACLE INNOVATION (D₂ high, D₃ low)
     * Oracle sees something RBPF doesn't
     *═══════════════════════════════════════════════════════════════════*/
    
    if (scenario == DIV_SCENARIO_ORACLE_INNOVATION && state->oracle_available) {
        /* Check if urgency warrants injection */
        if (dec.urgency >= URGENCY_HIGH) {
            /* High urgency - inject regardless of confidence */
            dec.should_inject = true;
            dec.source = INJECT_ORACLE;
            dec.gamma = injection_compute_gamma(
                dec.urgency, dec.urgency_score,
                state->oracle_confidence, scenario, config);
            dec.use_thompson = (state->oracle_confidence < config->oracle_conf_medium);
            dec.reason = "High urgency, Oracle has innovation";
            return dec;
        }
        
        if (state->oracle_confidence > config->oracle_conf_high) {
            /* Oracle very confident - inject */
            dec.should_inject = true;
            dec.source = INJECT_ORACLE;
            dec.gamma = injection_compute_gamma(
                dec.urgency, dec.urgency_score,
                state->oracle_confidence, scenario, config);
            dec.use_thompson = false;
            dec.reason = "Oracle confident with innovation";
            return dec;
        }
        
        if (dec.urgency >= URGENCY_MEDIUM && 
            state->oracle_confidence > config->oracle_conf_medium) {
            /* Medium urgency + moderate confidence */
            dec.should_inject = true;
            dec.source = INJECT_ORACLE;
            dec.gamma = injection_compute_gamma(
                dec.urgency, dec.urgency_score,
                state->oracle_confidence, scenario, config);
            dec.use_thompson = false;
            dec.reason = "Medium urgency, Oracle moderate confidence";
            return dec;
        }
    }
    
    /*═══════════════════════════════════════════════════════════════════
     * PRIORITY 5: QUALITY DEGRADATION (without specific divergence signal)
     *═══════════════════════════════════════════════════════════════════*/
    
    if (dec.urgency >= URGENCY_HIGH && state->oracle_available) {
        dec.should_inject = true;
        dec.source = INJECT_ORACLE;
        dec.gamma = injection_compute_gamma(
            dec.urgency, dec.urgency_score,
            state->oracle_confidence, scenario, config);
        dec.use_thompson = (state->oracle_confidence < config->oracle_conf_medium);
        dec.reason = "Quality degraded, injecting Oracle";
        return dec;
    }
    
    /*═══════════════════════════════════════════════════════════════════
     * PRIORITY 6: PROACTIVE INJECTION (Healthy but Oracle very confident)
     *═══════════════════════════════════════════════════════════════════*/
    
    if (state->oracle_available && 
        state->oracle_confidence > 0.75f &&
        state->divergence.d2_oracle_vs_operating > 0.08) {
        /* Oracle very confident and different - small proactive injection */
        dec.should_inject = true;
        dec.source = INJECT_ORACLE;
        dec.gamma = state->oracle_confidence * 0.20f;
        dec.use_thompson = false;
        dec.reason = "Proactive: Oracle highly confident";
        return dec;
    }
    
    /*═══════════════════════════════════════════════════════════════════
     * DEFAULT: NO ACTION
     *═══════════════════════════════════════════════════════════════════*/
    
    dec.should_inject = false;
    dec.source = INJECT_NONE;
    dec.reason = "System healthy, no action";
    
    return dec;
}

/*═══════════════════════════════════════════════════════════════════════════
 * STRING HELPERS
 *═══════════════════════════════════════════════════════════════════════════*/

const char *injection_urgency_str(InjectionUrgency urgency) {
    switch (urgency) {
        case URGENCY_NONE:      return "NONE";
        case URGENCY_LOW:       return "LOW";
        case URGENCY_MEDIUM:    return "MEDIUM";
        case URGENCY_HIGH:      return "HIGH";
        case URGENCY_EMERGENCY: return "EMERGENCY";
        default:                return "UNKNOWN";
    }
}

const char *injection_source_str(InjectionSource source) {
    switch (source) {
        case INJECT_NONE:           return "NONE";
        case INJECT_ORACLE:         return "ORACLE";
        case INJECT_STORVIK:        return "STORVIK";
        case INJECT_ORACLE_BOOSTED: return "ORACLE_BOOSTED";
        case INJECT_THOMPSON:       return "THOMPSON";
        case INJECT_RESET:          return "RESET";
        default:                    return "UNKNOWN";
    }
}
