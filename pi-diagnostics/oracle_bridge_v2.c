/**
 * @file oracle_bridge_v2.c
 * @brief Oracle Bridge V2 Implementation
 */

#include "oracle_bridge_v2.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/*═══════════════════════════════════════════════════════════════════════════
 * CONFIGURATION DEFAULTS
 *═══════════════════════════════════════════════════════════════════════════*/

OracleBridgeConfig oracle_bridge_config_defaults(int K) {
    OracleBridgeConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    
    cfg.K = (K > 0 && K <= ORACLE_MAX_K) ? K : 4;
    cfg.observation_window = 500;
    
    cfg.injection = injection_config_defaults();
    cfg.divergence = divergence_thresholds_defaults();
    
    cfg.thompson_prior_alpha = 1.0f;
    cfg.storvik_reset_count = 100.0f;
    cfg.verbose = false;
    
    return cfg;
}

/*═══════════════════════════════════════════════════════════════════════════
 * INITIALIZATION
 *═══════════════════════════════════════════════════════════════════════════*/

static void init_uniform_pi(float *Pi, int K) {
    float p = 1.0f / K;
    for (int i = 0; i < K * K; i++) {
        Pi[i] = p;
    }
    /* Actually make it diagonal-dominant for stability */
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < K; j++) {
            if (i == j) {
                Pi[i * K + j] = 0.7f;
            } else {
                Pi[i * K + j] = 0.3f / (K - 1);
            }
        }
    }
}

static void init_storvik_from_pi(float *Q, const float *Pi, float count, int K) {
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < K; j++) {
            Q[i * K + j] = Pi[i * K + j] * count;
        }
    }
}

int oracle_bridge_init(
    OracleBridgeV2 *bridge,
    const OracleBridgeConfig *config,
    ObservationBuffer *obs_buffer,
    const float *Pi_initial)
{
    if (!bridge || !config || !obs_buffer) return -1;
    if (config->K <= 0 || config->K > ORACLE_MAX_K) return -1;
    
    memset(bridge, 0, sizeof(*bridge));
    
    bridge->config = *config;
    bridge->obs_buffer = obs_buffer;
    
    int K = config->K;
    bridge->rbpf.K = K;
    
    /* Initialize Π */
    if (Pi_initial) {
        memcpy(bridge->rbpf.Pi_operating, Pi_initial, K * K * sizeof(float));
    } else {
        init_uniform_pi(bridge->rbpf.Pi_operating, K);
    }
    
    /* Initialize Storvik */
    init_storvik_from_pi(bridge->rbpf.Q_storvik, 
                         bridge->rbpf.Pi_operating,
                         config->storvik_reset_count, K);
    memcpy(bridge->rbpf.Pi_storvik, bridge->rbpf.Pi_operating, K * K * sizeof(float));
    
    /* Initialize quality monitoring */
    pi_quality_init(&bridge->rbpf.quality);
    
    /* Initialize staging buffer */
    pi_staging_init(&bridge->pi_staging);
    
    /* Initialize RNG */
    thompson_rng_init(&bridge->rbpf.rng, (uint64_t)time(NULL) ^ 0xDEADBEEF);
    
    bridge->initialized = true;
    
    if (config->verbose) {
        printf("[OracleBridge] Initialized with K=%d, window=%d\n",
               K, config->observation_window);
    }
    
    return 0;
}

void oracle_bridge_free(OracleBridgeV2 *bridge) {
    if (!bridge) return;
    bridge->initialized = false;
    /* Nothing dynamically allocated */
}

void oracle_bridge_reset(OracleBridgeV2 *bridge) {
    if (!bridge || !bridge->initialized) return;
    
    OracleBridgeConfig saved_config = bridge->config;
    ObservationBuffer *saved_obs = bridge->obs_buffer;
    
    oracle_bridge_init(bridge, &saved_config, saved_obs, NULL);
}

/*═══════════════════════════════════════════════════════════════════════════
 * RBPF THREAD - TICK UPDATE
 *═══════════════════════════════════════════════════════════════════════════*/

void oracle_bridge_update_hawkes(
    OracleBridgeV2 *bridge,
    float intensity,
    float surprise_sigma)
{
    if (!bridge) return;
    bridge->rbpf.hawkes_intensity = intensity;
    bridge->rbpf.hawkes_surprise_sigma = surprise_sigma;
}

void oracle_bridge_update_storvik(
    OracleBridgeV2 *bridge,
    const float *Q_new)
{
    if (!bridge || !Q_new) return;
    
    int K = bridge->rbpf.K;
    memcpy(bridge->rbpf.Q_storvik, Q_new, K * K * sizeof(float));
    
    /* Update Pi_storvik */
    pi_quality_normalize_counts(bridge->rbpf.Q_storvik, 
                                bridge->rbpf.Pi_storvik, K);
}

InjectionDecision oracle_bridge_rbpf_tick(
    OracleBridgeV2 *bridge,
    double observation,
    double prediction,
    double log_likelihood,
    int map_regime,
    const double *weights,
    int N,
    int64_t tick)
{
    InjectionDecision dec;
    memset(&dec, 0, sizeof(dec));
    dec.source = INJECT_NONE;
    dec.reason = "Not initialized";
    
    if (!bridge || !bridge->initialized) return dec;
    
    int K = bridge->rbpf.K;
    
    /*═══════════════════════════════════════════════════════════════════
     * 1. UPDATE QUALITY MONITORING
     *═══════════════════════════════════════════════════════════════════*/
    
    pi_quality_update(&bridge->rbpf.quality,
                      prediction, observation,
                      log_likelihood, map_regime,
                      weights, N,
                      bridge->rbpf.Pi_storvik,
                      bridge->rbpf.Pi_operating, K);
    
    /*═══════════════════════════════════════════════════════════════════
     * 2. CHECK FOR ORACLE Π
     *═══════════════════════════════════════════════════════════════════*/
    
    bool oracle_available = pi_staging_available(&bridge->pi_staging);
    float oracle_confidence = 0.0f;
    const StagedPi *staged = NULL;
    
    if (oracle_available) {
        staged = pi_staging_peek(&bridge->pi_staging);
        if (staged && staged->valid) {
            oracle_confidence = staged->confidence.overall;
            bridge->last_oracle_confidence = staged->confidence;
        }
    }
    
    /*═══════════════════════════════════════════════════════════════════
     * 3. COMPUTE DIVERGENCES
     *═══════════════════════════════════════════════════════════════════*/
    
    const float *Pi_oracle = (staged && staged->valid) ? staged->Pi : NULL;
    
    DivergenceTriad div = divergence_compute(
        Pi_oracle,
        bridge->rbpf.Pi_storvik,
        bridge->rbpf.Pi_operating, K);
    
    bridge->last_divergence = div;
    
    DivergenceScenario scenario = divergence_classify(&div, &bridge->config.divergence);
    
    /*═══════════════════════════════════════════════════════════════════
     * 4. BUILD SYSTEM STATE
     *═══════════════════════════════════════════════════════════════════*/
    
    SystemState state;
    memset(&state, 0, sizeof(state));
    
    state.quality = pi_quality_get_snapshot(&bridge->rbpf.quality);
    state.divergence = div;
    state.scenario = scenario;
    state.oracle_available = oracle_available;
    state.oracle_confidence = oracle_confidence;
    state.hawkes_intensity = bridge->rbpf.hawkes_intensity;
    state.hawkes_surprise_sigma = bridge->rbpf.hawkes_surprise_sigma;
    
    /*═══════════════════════════════════════════════════════════════════
     * 5. COMPUTE INJECTION DECISION
     *═══════════════════════════════════════════════════════════════════*/
    
    dec = injection_decide(&state, &bridge->config.injection);
    
    /*═══════════════════════════════════════════════════════════════════
     * 6. VERBOSE OUTPUT
     *═══════════════════════════════════════════════════════════════════*/
    
    if (bridge->config.verbose && (dec.should_inject || tick % 100 == 0)) {
        printf("[OracleBridge] t=%lld: quality=%.2f ess=%.2f rmse_ratio=%.2f "
               "d3=%.3f urgency=%s\n",
               (long long)tick,
               state.quality.quality_score,
               state.quality.ess_ratio,
               state.quality.rmse_ratio,
               state.quality.d3_self_contradiction,
               injection_urgency_str(dec.urgency));
        
        if (dec.should_inject) {
            printf("[OracleBridge] → INJECT: source=%s γ=%.3f reason=%s\n",
                   injection_source_str(dec.source), dec.gamma, dec.reason);
        }
    }
    
    return dec;
}

/*═══════════════════════════════════════════════════════════════════════════
 * APPLY INJECTION
 *═══════════════════════════════════════════════════════════════════════════*/

int oracle_bridge_apply_injection(
    OracleBridgeV2 *bridge,
    const InjectionDecision *decision)
{
    if (!bridge || !bridge->initialized || !decision) return -1;
    if (!decision->should_inject) return 0;
    
    int K = bridge->rbpf.K;
    float gamma = decision->gamma;
    float Pi_inject[ORACLE_MAX_K * ORACLE_MAX_K];
    
    /*═══════════════════════════════════════════════════════════════════
     * 1. GET SOURCE Π
     *═══════════════════════════════════════════════════════════════════*/
    
    switch (decision->source) {
        case INJECT_ORACLE:
        case INJECT_ORACLE_BOOSTED:
        case INJECT_THOMPSON: {
            StagedPi staged;
            if (!pi_staging_consume_full(&bridge->pi_staging, &staged)) {
                return -1;  /* No Oracle available */
            }
            
            if (decision->use_thompson || decision->source == INJECT_THOMPSON) {
                /* Thompson sample (variance shot) */
                thompson_sample_with_confidence(
                    staged.Q,
                    bridge->config.thompson_prior_alpha,
                    K,
                    staged.confidence.overall,
                    Pi_inject,
                    &bridge->rbpf.rng);
                bridge->rbpf.thompson_samples++;
            } else {
                /* Use Oracle mean */
                memcpy(Pi_inject, staged.Pi, K * K * sizeof(float));
            }
            break;
        }
        
        case INJECT_STORVIK: {
            /* Self-correct with Storvik */
            memcpy(Pi_inject, bridge->rbpf.Pi_storvik, K * K * sizeof(float));
            bridge->rbpf.storvik_self_corrections++;
            break;
        }
        
        case INJECT_RESET: {
            /* Full reset to diagonal-dominant */
            init_uniform_pi(Pi_inject, K);
            gamma = 1.0f;  /* Full replacement */
            break;
        }
        
        default:
            return -1;
    }
    
    /*═══════════════════════════════════════════════════════════════════
     * 2. BLEND: Π_new = (1-γ)Π_old + γΠ_inject
     *═══════════════════════════════════════════════════════════════════*/
    
    float *Pi_op = bridge->rbpf.Pi_operating;
    
    for (int i = 0; i < K * K; i++) {
        Pi_op[i] = (1.0f - gamma) * Pi_op[i] + gamma * Pi_inject[i];
    }
    
    /* Renormalize rows */
    for (int i = 0; i < K; i++) {
        float row_sum = 0.0f;
        for (int j = 0; j < K; j++) {
            row_sum += Pi_op[i * K + j];
        }
        if (row_sum > 1e-10f) {
            for (int j = 0; j < K; j++) {
                Pi_op[i * K + j] /= row_sum;
            }
        }
    }
    
    /*═══════════════════════════════════════════════════════════════════
     * 3. RESET STORVIK COUNTS ("Lobotomy")
     *
     * Critical: If we only update Π, Storvik's counts will pull it back
     * to the old value in the next tick. We must reset the counts to
     * match the new Π.
     *═══════════════════════════════════════════════════════════════════*/
    
    float effective_count = bridge->config.storvik_reset_count;
    float *Q = bridge->rbpf.Q_storvik;
    
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < K; j++) {
            Q[i * K + j] = Pi_op[i * K + j] * effective_count;
        }
    }
    
    /* Update Pi_storvik to match */
    memcpy(bridge->rbpf.Pi_storvik, Pi_op, K * K * sizeof(float));
    
    /*═══════════════════════════════════════════════════════════════════
     * 4. UPDATE STATISTICS
     *═══════════════════════════════════════════════════════════════════*/
    
    bridge->rbpf.total_injections++;
    bridge->rbpf.cumulative_gamma += gamma;
    bridge->rbpf.last_gamma = gamma;
    bridge->rbpf.last_source = decision->source;
    
    /*═══════════════════════════════════════════════════════════════════
     * 5. RESET QUALITY TRACKING (clean slate after injection)
     *═══════════════════════════════════════════════════════════════════*/
    
    pi_quality_reset(&bridge->rbpf.quality);
    
    if (bridge->config.verbose) {
        printf("[OracleBridge] Injection applied: γ=%.3f source=%s\n",
               gamma, injection_source_str(decision->source));
    }
    
    return 0;
}

/*═══════════════════════════════════════════════════════════════════════════
 * ORACLE THREAD - SUBMIT RESULT
 *═══════════════════════════════════════════════════════════════════════════*/

void oracle_bridge_submit_oracle(
    OracleBridgeV2 *bridge,
    const float *Pi_oracle,
    const float *Q_oracle,
    float acceptance_rate,
    float min_row_count,
    int sweeps_used,
    int64_t window_start,
    int64_t window_end)
{
    if (!bridge || !bridge->initialized || !Pi_oracle) return;
    
    int K = bridge->rbpf.K;
    
    /* Compute Frobenius difference for innovation gate */
    float frobenius = 0.0f;
    const float *Pi_old = bridge->rbpf.Pi_operating;
    for (int i = 0; i < K * K; i++) {
        float d = Pi_oracle[i] - Pi_old[i];
        frobenius += d * d;
    }
    frobenius = sqrtf(frobenius);
    
    /* Compute confidence */
    OracleConfidence conf = pi_staging_compute_confidence(
        acceptance_rate, min_row_count, frobenius, sweeps_used);
    
    /* Get current tick from observation buffer */
    int64_t tick = obs_buffer_get_head(bridge->obs_buffer);
    
    /* Submit to staging */
    pi_staging_submit(&bridge->pi_staging,
                      Pi_oracle, Q_oracle, K,
                      &conf, tick,
                      window_start, window_end);
    
    if (bridge->config.verbose) {
        printf("[OracleBridge] Oracle submitted: conf=%.2f (mix=%.2f info=%.2f innov=%.2f) "
               "accept=%.1f%% window=[%lld,%lld]\n",
               conf.overall, conf.mixing_score, conf.information_score, conf.innovation_score,
               acceptance_rate * 100,
               (long long)window_start, (long long)window_end);
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * DIAGNOSTICS
 *═══════════════════════════════════════════════════════════════════════════*/

const float *oracle_bridge_get_pi(const OracleBridgeV2 *bridge) {
    if (!bridge || !bridge->initialized) return NULL;
    return bridge->rbpf.Pi_operating;
}

const float *oracle_bridge_get_storvik(const OracleBridgeV2 *bridge) {
    if (!bridge || !bridge->initialized) return NULL;
    return bridge->rbpf.Q_storvik;
}

PiQualitySnapshot oracle_bridge_get_quality(const OracleBridgeV2 *bridge) {
    PiQualitySnapshot snap = {0};
    if (!bridge || !bridge->initialized) return snap;
    return pi_quality_get_snapshot(&bridge->rbpf.quality);
}

DivergenceTriad oracle_bridge_get_divergence(const OracleBridgeV2 *bridge) {
    DivergenceTriad div = {0};
    if (!bridge || !bridge->initialized) return div;
    return bridge->last_divergence;
}

OracleConfidence oracle_bridge_get_oracle_confidence(const OracleBridgeV2 *bridge) {
    OracleConfidence conf = {0};
    if (!bridge || !bridge->initialized) return conf;
    return bridge->last_oracle_confidence;
}

OracleBridgeStats oracle_bridge_get_stats(const OracleBridgeV2 *bridge) {
    OracleBridgeStats stats = {0};
    if (!bridge || !bridge->initialized) return stats;
    
    stats.total_injections = bridge->rbpf.total_injections;
    stats.thompson_samples = bridge->rbpf.thompson_samples;
    stats.storvik_self_corrections = bridge->rbpf.storvik_self_corrections;
    
    if (stats.total_injections > 0) {
        stats.avg_gamma = bridge->rbpf.cumulative_gamma / stats.total_injections;
    }
    
    PiQualitySnapshot q = pi_quality_get_snapshot(&bridge->rbpf.quality);
    stats.current_quality_score = (float)q.quality_score;
    stats.current_ess = (float)q.ess_ratio;
    stats.current_rmse_ratio = (float)q.rmse_ratio;
    stats.current_d3 = (float)q.d3_self_contradiction;
    
    return stats;
}

void oracle_bridge_print_state(const OracleBridgeV2 *bridge) {
    if (!bridge || !bridge->initialized) return;
    
    OracleBridgeStats stats = oracle_bridge_get_stats(bridge);
    PiQualitySnapshot q = oracle_bridge_get_quality(bridge);
    DivergenceTriad div = oracle_bridge_get_divergence(bridge);
    
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║               ORACLE BRIDGE V2 STATE                          ║\n");
    printf("╠═══════════════════════════════════════════════════════════════╣\n");
    printf("║ Configuration:                                                ║\n");
    printf("║   K=%d  Window=%d                                             \n",
           bridge->config.K, bridge->config.observation_window);
    printf("╠═══════════════════════════════════════════════════════════════╣\n");
    printf("║ Quality Metrics:                                              ║\n");
    printf("║   Quality Score:  %.3f                                       \n", q.quality_score);
    printf("║   ESS Ratio:      %.3f                                       \n", q.ess_ratio);
    printf("║   RMSE Ratio:     %.3f                                       \n", q.rmse_ratio);
    printf("║   Likelihood Z:   %.2f                                       \n", q.likelihood_zscore);
    printf("║   Transition Lag: %.1f                                       \n", q.avg_transition_lag);
    printf("╠═══════════════════════════════════════════════════════════════╣\n");
    printf("║ Divergences:                                                  ║\n");
    printf("║   D1 (Oracle-Storvik):   %.4f                                \n", div.d1_oracle_vs_storvik);
    printf("║   D2 (Oracle-Operating): %.4f                                \n", div.d2_oracle_vs_operating);
    printf("║   D3 (Self-Contradict):  %.4f                                \n", div.d3_storvik_vs_operating);
    printf("╠═══════════════════════════════════════════════════════════════╣\n");
    printf("║ Statistics:                                                   ║\n");
    printf("║   Total Injections:      %d                                  \n", stats.total_injections);
    printf("║   Thompson Samples:      %d                                  \n", stats.thompson_samples);
    printf("║   Storvik Self-Correct:  %d                                  \n", stats.storvik_self_corrections);
    printf("║   Avg γ:                 %.3f                                \n", stats.avg_gamma);
    printf("╚═══════════════════════════════════════════════════════════════╝\n");
}
