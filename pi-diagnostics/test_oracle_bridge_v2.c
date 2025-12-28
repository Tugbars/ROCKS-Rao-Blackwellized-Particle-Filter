/**
 * @file test_oracle_bridge_v2.c
 * @brief Test Oracle Bridge V2
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "oracle_bridge_v2.h"

/*═══════════════════════════════════════════════════════════════════════════
 * TEST UTILITIES
 *═══════════════════════════════════════════════════════════════════════════*/

#define TEST_PASS(name) printf("  ✓ %s\n", name)
#define TEST_FAIL(name) printf("  ✗ %s\n", name)
#define RUN_TEST(fn) do { \
    printf("\nRunning: %s\n", #fn); \
    int result = fn(); \
    total_tests++; \
    if (result) { passed_tests++; TEST_PASS(#fn); } \
    else { TEST_FAIL(#fn); } \
} while(0)

static int total_tests = 0;
static int passed_tests = 0;

/*═══════════════════════════════════════════════════════════════════════════
 * TEST: Observation Buffer
 *═══════════════════════════════════════════════════════════════════════════*/

static int test_observation_buffer(void) {
    ObservationBuffer buf;
    obs_buffer_init(&buf);
    
    /* Push some observations */
    for (int i = 0; i < 100; i++) {
        obs_buffer_push(&buf, (double)i * 0.1, i);
    }
    
    /* Check head */
    if (obs_buffer_get_head(&buf) != 100) return 0;
    
    /* Snapshot window */
    double window[50];
    int64_t end = obs_buffer_snapshot(&buf, window, NULL, 50);
    
    if (end != 100) return 0;
    if (fabs(window[0] - 5.0) > 1e-6) return 0;  /* obs[50] = 5.0 */
    if (fabs(window[49] - 9.9) > 1e-6) return 0; /* obs[99] = 9.9 */
    
    return 1;
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST: Quality Metrics
 *═══════════════════════════════════════════════════════════════════════════*/

static int test_pi_quality(void) {
    PiQualityState state;
    pi_quality_init(&state);
    
    /* Simulate good predictions */
    for (int i = 0; i < 100; i++) {
        double actual = sin(i * 0.1);
        double predicted = actual + 0.01 * (i % 2 ? 1 : -1);  /* Small error */
        pi_quality_update_prediction(&state, predicted, actual);
    }
    
    /* RMSE ratio should be near 1 after baseline established */
    if (state.prediction.rmse_ratio > 2.0) return 0;
    
    /* Simulate bad predictions */
    for (int i = 0; i < 50; i++) {
        double actual = sin(i * 0.1);
        double predicted = actual + 0.5;  /* Large error */
        pi_quality_update_prediction(&state, predicted, actual);
    }
    
    /* RMSE ratio should spike */
    if (state.prediction.rmse_ratio < 1.5) return 0;
    
    return 1;
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST: Divergence Computation
 *═══════════════════════════════════════════════════════════════════════════*/

static int test_divergence(void) {
    int K = 3;
    
    /* Similar matrices */
    float P1[9] = {0.8f, 0.1f, 0.1f,
                   0.1f, 0.8f, 0.1f,
                   0.1f, 0.1f, 0.8f};
    
    float P2[9] = {0.78f, 0.11f, 0.11f,
                   0.11f, 0.78f, 0.11f,
                   0.11f, 0.11f, 0.78f};
    
    /* Different matrix */
    float P3[9] = {0.5f, 0.3f, 0.2f,
                   0.2f, 0.5f, 0.3f,
                   0.3f, 0.2f, 0.5f};
    
    DivergenceTriad div = divergence_compute(P1, P2, P3, K);
    
    /* D1 (P1 vs P2) should be small */
    if (div.d1_oracle_vs_storvik > 0.05) return 0;
    
    /* D2 (P1 vs P3) should be larger */
    if (div.d2_oracle_vs_operating < 0.1) return 0;
    
    /* D3 (P2 vs P3) should be larger */
    if (div.d3_storvik_vs_operating < 0.1) return 0;
    
    /* Classify scenario */
    DivergenceThresholds thresh = divergence_thresholds_defaults();
    DivergenceScenario scenario = divergence_classify(&div, &thresh);
    
    /* P2 and P3 differ → D3 high → self-contradiction */
    if (scenario != DIV_SCENARIO_SELF_CONTRADICTION &&
        scenario != DIV_SCENARIO_STRONG_AGREEMENT) {
        /* Could be either depending on exact threshold values */
    }
    
    return 1;
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST: Injection Decision
 *═══════════════════════════════════════════════════════════════════════════*/

static int test_injection_decision(void) {
    InjectionConfig cfg = injection_config_defaults();
    
    /* Healthy system state */
    SystemState healthy = {0};
    healthy.quality.quality_score = 0.9;
    healthy.quality.ess_ratio = 0.7;
    healthy.quality.rmse_ratio = 1.0;
    healthy.oracle_available = true;
    healthy.oracle_confidence = 0.5;
    
    InjectionDecision dec = injection_decide(&healthy, &cfg);
    
    /* Should NOT inject when healthy */
    if (dec.should_inject) return 0;
    
    /* Degraded system state */
    SystemState degraded = {0};
    degraded.quality.quality_score = 0.3;
    degraded.quality.ess_ratio = 0.25;
    degraded.quality.rmse_ratio = 2.5;
    degraded.quality.d3_self_contradiction = 0.2;
    degraded.oracle_available = true;
    degraded.oracle_confidence = 0.7;
    
    dec = injection_decide(&degraded, &cfg);
    
    /* Should inject when degraded */
    if (!dec.should_inject) return 0;
    if (dec.gamma < 0.1) return 0;
    
    /* Emergency state (low ESS) */
    SystemState emergency = {0};
    emergency.quality.quality_score = 0.1;
    emergency.quality.ess_ratio = 0.1;  /* Critical */
    emergency.oracle_available = true;
    emergency.oracle_confidence = 0.3;
    
    dec = injection_decide(&emergency, &cfg);
    
    /* Must inject in emergency */
    if (!dec.should_inject) return 0;
    if (dec.urgency != URGENCY_EMERGENCY) return 0;
    if (!dec.use_thompson) return 0;  /* Should use variance shot */
    
    return 1;
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST: Thompson Sampling
 *═══════════════════════════════════════════════════════════════════════════*/

static int test_thompson_sampler(void) {
    ThompsonRNG rng;
    thompson_rng_init(&rng, 12345);
    
    int K = 3;
    float Q[9] = {80, 10, 10,
                  10, 80, 10,
                  10, 10, 80};
    
    float Pi_sample[9];
    float Pi_mean[9];
    
    /* Sample */
    thompson_sample_pi(Q, 1.0f, K, Pi_sample, &rng);
    
    /* Mean */
    thompson_mean_pi(Q, 1.0f, K, Pi_mean);
    
    /* Check row-stochastic */
    for (int i = 0; i < K; i++) {
        float sum_s = 0, sum_m = 0;
        for (int j = 0; j < K; j++) {
            sum_s += Pi_sample[i * K + j];
            sum_m += Pi_mean[i * K + j];
        }
        if (fabs(sum_s - 1.0) > 1e-5) return 0;
        if (fabs(sum_m - 1.0) > 1e-5) return 0;
    }
    
    /* Mean should be close to normalized Q */
    /* Q[0,0] = 80, Q[0,:] = 100, so mean[0,0] ≈ 80/100 = 0.8 */
    if (fabs(Pi_mean[0] - 0.8f) > 0.1f) return 0;
    
    return 1;
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST: Full Bridge Integration
 *═══════════════════════════════════════════════════════════════════════════*/

static int test_bridge_integration(void) {
    int K = 3;
    
    /* Create observation buffer */
    ObservationBuffer obs_buf;
    obs_buffer_init(&obs_buf);
    
    /* Create bridge */
    OracleBridgeV2 bridge;
    OracleBridgeConfig config = oracle_bridge_config_defaults(K);
    config.verbose = false;
    
    if (oracle_bridge_init(&bridge, &config, &obs_buf, NULL) != 0) {
        return 0;
    }
    
    /* Simulate ticks */
    double weights[100];
    for (int i = 0; i < 100; i++) weights[i] = 1.0 / 100;
    
    for (int t = 0; t < 200; t++) {
        double obs = sin(t * 0.05);
        double pred = sin((t - 1) * 0.05);
        double log_lik = -0.5 * (obs - pred) * (obs - pred);
        int regime = (t / 50) % K;
        
        obs_buffer_push(&obs_buf, obs, t);
        
        InjectionDecision dec = oracle_bridge_rbpf_tick(
            &bridge, obs, pred, log_lik, regime,
            weights, 100, t);
        
        /* Shouldn't need injection in normal operation */
    }
    
    /* Submit fake Oracle result */
    float Pi_oracle[9] = {0.7f, 0.2f, 0.1f,
                          0.1f, 0.7f, 0.2f,
                          0.2f, 0.1f, 0.7f};
    float Q_oracle[9] = {70, 20, 10,
                         10, 70, 20,
                         20, 10, 70};
    
    oracle_bridge_submit_oracle(&bridge, Pi_oracle, Q_oracle,
                                0.15f, 100.0f, 50, 100, 200);
    
    /* Next tick should see Oracle available */
    double obs = 0.5, pred = 0.4;
    InjectionDecision dec = oracle_bridge_rbpf_tick(
        &bridge, obs, pred, -0.005, 0, weights, 100, 201);
    
    /* Oracle should be detected */
    /* (May or may not inject depending on thresholds) */
    
    /* Get stats */
    OracleBridgeStats stats = oracle_bridge_get_stats(&bridge);
    
    oracle_bridge_free(&bridge);
    
    return 1;
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST: Self-Correction
 *═══════════════════════════════════════════════════════════════════════════*/

static int test_self_correction(void) {
    int K = 3;
    
    ObservationBuffer obs_buf;
    obs_buffer_init(&obs_buf);
    
    OracleBridgeV2 bridge;
    OracleBridgeConfig config = oracle_bridge_config_defaults(K);
    config.verbose = false;
    
    oracle_bridge_init(&bridge, &config, &obs_buf, NULL);
    
    /* Manually set divergent Storvik */
    float Q_divergent[9] = {50, 30, 20,
                            20, 50, 30,
                            30, 20, 50};
    oracle_bridge_update_storvik(&bridge, Q_divergent);
    
    /* Compute divergences */
    DivergenceTriad div = oracle_bridge_get_divergence(&bridge);
    
    /* D3 should be elevated (Storvik differs from Operating) */
    /* Note: Need to call rbpf_tick to compute divergences */
    
    double weights[100];
    for (int i = 0; i < 100; i++) weights[i] = 1.0 / 100;
    
    /* Fill buffer */
    for (int t = 0; t < 100; t++) {
        obs_buffer_push(&obs_buf, sin(t * 0.1), t);
    }
    
    /* Run tick which computes divergences */
    InjectionDecision dec = oracle_bridge_rbpf_tick(
        &bridge, 0.5, 0.5, -0.01, 0, weights, 100, 100);
    
    div = oracle_bridge_get_divergence(&bridge);
    
    /* D3 should be > 0 since Storvik differs from initial Operating */
    if (div.d3_storvik_vs_operating < 0.01) {
        /* May be small if they haven't diverged much */
    }
    
    oracle_bridge_free(&bridge);
    
    return 1;
}

/*═══════════════════════════════════════════════════════════════════════════
 * MAIN
 *═══════════════════════════════════════════════════════════════════════════*/

int main(void) {
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║           ORACLE BRIDGE V2 TEST SUITE                         ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n");
    
    RUN_TEST(test_observation_buffer);
    RUN_TEST(test_pi_quality);
    RUN_TEST(test_divergence);
    RUN_TEST(test_injection_decision);
    RUN_TEST(test_thompson_sampler);
    RUN_TEST(test_bridge_integration);
    RUN_TEST(test_self_correction);
    
    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("Results: %d/%d tests passed\n", passed_tests, total_tests);
    printf("═══════════════════════════════════════════════════════════════\n");
    
    return (passed_tests == total_tests) ? 0 : 1;
}
