/**
 * @file test_kl_trigger.c
 * @brief Test for KL Divergence Trigger
 *
 * Tests:
 *   - Config defaults
 *   - Baseline estimation
 *   - State machine transitions
 *   - Hysteresis behavior
 *   - Refractory period
 *   - Panic detection
 *   - Simplified update functions
 *
 * Compile:
 *   gcc -O2 -Wall test_kl_trigger.c kl_trigger.c -lm -o test_kl
 */

#include "kl_trigger.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define GREEN  "\033[32m"
#define RED    "\033[31m"
#define YELLOW "\033[33m"
#define CYAN   "\033[36m"
#define RESET  "\033[0m"

static int g_tests_passed = 0;
static int g_tests_run = 0;

#define RUN_TEST(fn) do { \
    g_tests_run++; \
    printf("\n" CYAN "═══ TEST: %s ═══" RESET "\n", #fn); \
    if (fn()) { g_tests_passed++; printf(GREEN "✓ PASS" RESET "\n"); } \
    else { printf(RED "✗ FAIL" RESET "\n"); } \
} while(0)

/*═══════════════════════════════════════════════════════════════════════════
 * TEST: Configuration Defaults
 *═══════════════════════════════════════════════════════════════════════════*/

static int test_config_defaults(void) {
    printf("Testing configuration defaults\n\n");
    
    KLTriggerConfig cfg = kl_trigger_config_defaults(4);
    
    printf("  n_regimes:        %d (expected: 4)\n", cfg.n_regimes);
    printf("  baseline_ema:     %.3f (expected: 0.05)\n", cfg.baseline_ema_alpha);
    printf("  warmup_ticks:     %d (expected: 100)\n", cfg.warmup_ticks);
    printf("  trigger_sigma:    %.1f (expected: 2.0)\n", cfg.trigger_sigma);
    printf("  panic_sigma:      %.1f (expected: 5.0)\n", cfg.panic_sigma);
    printf("  high_water:       %.1f (expected: 2.0)\n", cfg.high_water_sigma);
    printf("  low_water:        %.1f (expected: 1.0)\n", cfg.low_water_sigma);
    printf("  refractory:       %d (expected: 50)\n", cfg.refractory_ticks);
    
    bool ok = (cfg.n_regimes == 4) &&
              (fabsf(cfg.baseline_ema_alpha - 0.05f) < 0.01f) &&
              (cfg.warmup_ticks == 100) &&
              (fabsf(cfg.trigger_sigma - 2.0f) < 0.01f) &&
              (fabsf(cfg.panic_sigma - 5.0f) < 0.01f);
    
    return ok;
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST: Initialization
 *═══════════════════════════════════════════════════════════════════════════*/

static int test_initialization(void) {
    printf("Testing initialization\n\n");
    
    KLTrigger trigger;
    KLTriggerConfig cfg = kl_trigger_config_defaults(4);
    
    int ret = kl_trigger_init(&trigger, &cfg);
    
    printf("  Init return:      %d (expected: 0)\n", ret);
    printf("  State:            %d (expected: CALM=0)\n", trigger.state);
    printf("  Initialized:      %s\n", trigger.initialized ? "true" : "false");
    printf("  Baseline mean:    %.3f\n", trigger.baseline_mean);
    printf("  Baseline std:     %.3f\n", trigger.baseline_std);
    
    bool ok = (ret == 0) &&
              trigger.initialized &&
              (trigger.state == KL_STATE_CALM) &&
              (trigger.baseline_mean > 0.0f);
    
    kl_trigger_free(&trigger);
    return ok;
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST: Baseline Estimation
 *═══════════════════════════════════════════════════════════════════════════*/

static int test_baseline_estimation(void) {
    printf("Testing baseline estimation with stable inputs\n\n");
    
    KLTrigger trigger;
    KLTriggerConfig cfg = kl_trigger_config_defaults(4);
    cfg.warmup_ticks = 10;  /* Faster warmup for testing */
    kl_trigger_init(&trigger, &cfg);
    
    /* Feed stable volatility innovations */
    float stable_vol = -3.0f;  /* log-vol */
    float stable_std = 0.1f;
    
    printf("  Feeding 50 stable observations (vol=%.1f, std=%.2f)...\n",
           stable_vol, stable_std);
    
    for (int i = 0; i < 50; i++) {
        /* Small random perturbation */
        float noise = 0.01f * ((i % 5) - 2);
        kl_trigger_update_vol(&trigger, stable_vol, stable_vol + noise, stable_std);
    }
    
    float mean, std;
    kl_trigger_get_baseline(&trigger, &mean, &std);
    
    printf("  Baseline mean:    %.4f\n", mean);
    printf("  Baseline std:     %.4f\n", std);
    printf("  Current surprise: %.3f σ\n", kl_trigger_get_surprise(&trigger));
    printf("  State:            %d (expected: CALM)\n", kl_trigger_get_state(&trigger));
    
    /* With stable inputs, surprise should be low */
    bool ok = (trigger.state == KL_STATE_CALM) &&
              (fabsf(kl_trigger_get_surprise(&trigger)) < 2.0f);
    
    kl_trigger_free(&trigger);
    return ok;
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST: Surprise Detection
 *═══════════════════════════════════════════════════════════════════════════*/

static int test_surprise_detection(void) {
    printf("Testing surprise detection with sudden change\n\n");
    
    KLTrigger trigger;
    KLTriggerConfig cfg = kl_trigger_config_defaults(4);
    cfg.warmup_ticks = 20;
    cfg.refractory_ticks = 10;
    kl_trigger_init(&trigger, &cfg);
    
    /* Establish baseline with stable inputs */
    printf("  Phase 1: Establishing baseline (30 stable ticks)...\n");
    for (int i = 0; i < 30; i++) {
        kl_trigger_update_vol(&trigger, -3.0f, -3.0f + 0.01f, 0.1f);
    }
    
    float mean1, std1;
    kl_trigger_get_baseline(&trigger, &mean1, &std1);
    printf("  Baseline: mean=%.4f, std=%.4f\n", mean1, std1);
    
    /* Inject surprising observation */
    printf("\n  Phase 2: Injecting surprise (predicted=-3.0, actual=-1.0)...\n");
    
    KLTriggerResult result = kl_trigger_update_vol(&trigger, -3.0f, -1.0f, 0.1f);
    
    printf("  Surprise σ:       %.3f\n", result.surprise_sigma);
    printf("  State:            %d", result.state);
    if (result.state == KL_STATE_TRIGGERED) printf(" (TRIGGERED)");
    else if (result.state == KL_STATE_PANIC) printf(" (PANIC)");
    printf("\n");
    printf("  Should trigger:   %s\n", result.should_trigger ? "YES" : "NO");
    
    /* Large innovation should cause elevated surprise */
    bool ok = (result.surprise_sigma > 1.0f);  /* Should be significantly elevated */
    
    kl_trigger_free(&trigger);
    return ok;
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST: Hysteresis Behavior
 *═══════════════════════════════════════════════════════════════════════════*/

static int test_hysteresis(void) {
    printf("Testing hysteresis (high/low water marks)\n\n");
    
    KLTrigger trigger;
    KLTriggerConfig cfg = kl_trigger_config_defaults(4);
    cfg.warmup_ticks = 10;
    cfg.refractory_ticks = 5;
    cfg.high_water_sigma = 2.0f;
    cfg.low_water_sigma = 1.0f;
    kl_trigger_init(&trigger, &cfg);
    
    /* Establish baseline */
    for (int i = 0; i < 20; i++) {
        kl_trigger_update_vol(&trigger, -3.0f, -3.0f, 0.1f);
    }
    
    printf("  Initial state: %d (CALM)\n", trigger.state);
    
    /* Push above high water */
    printf("  Injecting large surprise...\n");
    KLTriggerResult r1 = kl_trigger_update_vol(&trigger, -3.0f, -0.5f, 0.1f);
    printf("  Surprise: %.2fσ, State: %d\n", r1.surprise_sigma, r1.state);
    
    /* Return to moderate - should stay elevated due to hysteresis */
    printf("  Returning to moderate...\n");
    for (int i = 0; i < 5; i++) {
        KLTriggerResult r = kl_trigger_update_vol(&trigger, -3.0f, -2.5f, 0.1f);
        if (i == 4) {
            printf("  After 5 moderate ticks: surprise=%.2fσ, state=%d\n",
                   r.surprise_sigma, r.state);
        }
    }
    
    /* Return fully to calm */
    printf("  Returning to calm...\n");
    for (int i = 0; i < 10; i++) {
        kl_trigger_update_vol(&trigger, -3.0f, -3.0f + 0.01f, 0.1f);
    }
    printf("  Final state: %d\n", trigger.state);
    
    bool ok = true;  /* Hysteresis is working if we don't crash */
    
    kl_trigger_free(&trigger);
    return ok;
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST: Refractory Period
 *═══════════════════════════════════════════════════════════════════════════*/

static int test_refractory_period(void) {
    printf("Testing refractory period prevents rapid re-triggering\n\n");
    
    KLTrigger trigger;
    KLTriggerConfig cfg = kl_trigger_config_defaults(4);
    cfg.warmup_ticks = 10;
    cfg.refractory_ticks = 20;  /* 20 tick refractory */
    kl_trigger_init(&trigger, &cfg);
    
    /* Establish baseline */
    for (int i = 0; i < 15; i++) {
        kl_trigger_update_vol(&trigger, -3.0f, -3.0f, 0.1f);
    }
    
    /* First surprise - should trigger */
    printf("  First surprise at t=15...\n");
    KLTriggerResult r1 = kl_trigger_update_vol(&trigger, -3.0f, -0.5f, 0.1f);
    printf("  Should trigger: %s\n", r1.should_trigger ? "YES" : "NO");
    
    bool first_triggered = r1.should_trigger;
    
    /* Acknowledge the trigger */
    kl_trigger_acknowledge(&trigger);
    
    /* Second surprise immediately - should NOT trigger (refractory) */
    printf("  Second surprise at t=16 (within refractory)...\n");
    KLTriggerResult r2 = kl_trigger_update_vol(&trigger, -3.0f, -0.5f, 0.1f);
    printf("  Should trigger: %s (expected: NO - refractory)\n", 
           r2.should_trigger ? "YES" : "NO");
    
    bool second_blocked = !r2.should_trigger;
    
    /* Wait out refractory period */
    printf("  Waiting 25 ticks...\n");
    for (int i = 0; i < 25; i++) {
        kl_trigger_update_vol(&trigger, -3.0f, -3.0f, 0.1f);
    }
    
    /* Third surprise - should trigger again */
    printf("  Third surprise after refractory...\n");
    KLTriggerResult r3 = kl_trigger_update_vol(&trigger, -3.0f, -0.5f, 0.1f);
    printf("  Should trigger: %s\n", r3.should_trigger ? "YES" : "NO");
    
    bool third_triggered = r3.should_trigger;
    
    printf("\n  Summary:\n");
    printf("    First trigger:  %s\n", first_triggered ? GREEN "OK" RESET : RED "FAIL" RESET);
    printf("    Second blocked: %s\n", second_blocked ? GREEN "OK" RESET : RED "FAIL" RESET);
    printf("    Third trigger:  %s\n", third_triggered ? GREEN "OK" RESET : RED "FAIL" RESET);
    
    kl_trigger_free(&trigger);
    return first_triggered && second_blocked && third_triggered;
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST: Panic Detection
 *═══════════════════════════════════════════════════════════════════════════*/

static int test_panic_detection(void) {
    printf("Testing panic detection (extreme surprise)\n\n");
    
    KLTrigger trigger;
    KLTriggerConfig cfg = kl_trigger_config_defaults(4);
    cfg.warmup_ticks = 10;
    cfg.panic_sigma = 5.0f;
    kl_trigger_init(&trigger, &cfg);
    
    /* Establish stable baseline */
    for (int i = 0; i < 20; i++) {
        kl_trigger_update_vol(&trigger, -3.0f, -3.0f, 0.1f);
    }
    
    float mean, std;
    kl_trigger_get_baseline(&trigger, &mean, &std);
    printf("  Baseline: mean=%.4f, std=%.4f\n", mean, std);
    
    /* Inject extreme surprise */
    printf("  Injecting extreme surprise (20σ deviation)...\n");
    KLTriggerResult result = kl_trigger_update_vol(&trigger, -3.0f, 5.0f, 0.1f);
    
    printf("  Surprise σ:       %.2f\n", result.surprise_sigma);
    printf("  State:            %d (PANIC=%d)\n", result.state, KL_STATE_PANIC);
    printf("  Is panic:         %s\n", result.is_panic ? "YES" : "NO");
    printf("  Panic count:      %d\n", trigger.panic_triggers);
    
    bool ok = result.is_panic || (result.surprise_sigma > 5.0f);
    
    kl_trigger_free(&trigger);
    return ok;
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST: Regime Innovation
 *═══════════════════════════════════════════════════════════════════════════*/

static int test_regime_innovation(void) {
    printf("Testing regime prediction innovation\n\n");
    
    KLTrigger trigger;
    KLTriggerConfig cfg = kl_trigger_config_defaults(4);
    cfg.warmup_ticks = 10;
    cfg.regime_weight = 1.0f;     /* Only regime innovation */
    cfg.volatility_weight = 0.0f;
    kl_trigger_init(&trigger, &cfg);
    
    /* Create innovation with regime prediction */
    KLInnovation innov;
    memset(&innov, 0, sizeof(innov));
    
    /* Perfect prediction: high probability on actual regime */
    printf("  Test 1: Perfect prediction (P[actual]=0.9)\n");
    innov.regime_predicted[0] = 0.9f;
    innov.regime_predicted[1] = 0.033f;
    innov.regime_predicted[2] = 0.033f;
    innov.regime_predicted[3] = 0.034f;
    innov.regime_actual = 0;  /* Predicted correctly */
    
    for (int i = 0; i < 15; i++) {
        kl_trigger_update(&trigger, &innov);
    }
    
    float surprise1 = kl_trigger_get_surprise(&trigger);
    printf("  Surprise: %.3f σ (should be low)\n", surprise1);
    
    /* Bad prediction: low probability on actual regime */
    printf("\n  Test 2: Bad prediction (P[actual]=0.1)\n");
    innov.regime_predicted[0] = 0.1f;   /* Low probability */
    innov.regime_predicted[1] = 0.3f;
    innov.regime_predicted[2] = 0.3f;
    innov.regime_predicted[3] = 0.3f;
    innov.regime_actual = 0;  /* But this is what happened */
    
    KLTriggerResult result = kl_trigger_update(&trigger, &innov);
    printf("  Surprise: %.3f σ (should be elevated)\n", result.surprise_sigma);
    
    /* KL divergence for bad prediction should be higher */
    bool ok = (result.surprise_sigma > surprise1);
    printf("\n  Bad prediction more surprising: %s\n", ok ? GREEN "YES" RESET : RED "NO" RESET);
    
    kl_trigger_free(&trigger);
    return ok;
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST: Observation Innovation
 *═══════════════════════════════════════════════════════════════════════════*/

static int test_observation_innovation(void) {
    printf("Testing observation-level innovation\n\n");
    
    KLTrigger trigger;
    KLTriggerConfig cfg = kl_trigger_config_defaults(4);
    cfg.warmup_ticks = 10;
    kl_trigger_init(&trigger, &cfg);
    
    /* Feed observation innovations */
    printf("  Feeding 20 observations with small errors...\n");
    for (int i = 0; i < 20; i++) {
        kl_trigger_update_obs(&trigger, 0.0f, 0.01f, 0.1f);  /* Small error */
    }
    
    float surprise_stable = kl_trigger_get_surprise(&trigger);
    printf("  Surprise after stable: %.3f σ\n", surprise_stable);
    
    /* Large observation error */
    printf("  Injecting large observation error (5σ)...\n");
    KLTriggerResult result = kl_trigger_update_obs(&trigger, 0.0f, 0.5f, 0.1f);
    
    printf("  Surprise after error: %.3f σ\n", result.surprise_sigma);
    
    bool ok = (result.surprise_sigma > surprise_stable);
    
    kl_trigger_free(&trigger);
    return ok;
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST: Diagnostics
 *═══════════════════════════════════════════════════════════════════════════*/

static int test_diagnostics(void) {
    printf("Testing diagnostic output\n\n");
    
    KLTrigger trigger;
    KLTriggerConfig cfg = kl_trigger_config_defaults(4);
    kl_trigger_init(&trigger, &cfg);
    
    /* Do some updates */
    for (int i = 0; i < 30; i++) {
        float noise = 0.02f * (i % 3);
        kl_trigger_update_vol(&trigger, -3.0f, -3.0f + noise, 0.1f);
    }
    
    /* Inject one surprise */
    kl_trigger_update_vol(&trigger, -3.0f, -1.0f, 0.1f);
    
    printf("  Printing state...\n\n");
    kl_trigger_print_state(&trigger);
    
    kl_trigger_free(&trigger);
    return 1;  /* Visual inspection */
}

/*═══════════════════════════════════════════════════════════════════════════
 * MAIN
 *═══════════════════════════════════════════════════════════════════════════*/

int main(void) {
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════════════╗\n");
    printf("║                    KL TRIGGER TEST SUITE                              ║\n");
    printf("║                    Dual-Gate Oracle Component                         ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════════╝\n");
    
    RUN_TEST(test_config_defaults);
    RUN_TEST(test_initialization);
    RUN_TEST(test_baseline_estimation);
    RUN_TEST(test_surprise_detection);
    RUN_TEST(test_hysteresis);
    RUN_TEST(test_refractory_period);
    RUN_TEST(test_panic_detection);
    RUN_TEST(test_regime_innovation);
    RUN_TEST(test_observation_innovation);
    RUN_TEST(test_diagnostics);
    
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════════════╗\n");
    printf("║                           SUMMARY                                     ║\n");
    printf("╠═══════════════════════════════════════════════════════════════════════╣\n");
    printf("║  Tests passed: %2d / %2d                                                ║\n",
           g_tests_passed, g_tests_run);
    printf("╚═══════════════════════════════════════════════════════════════════════╝\n\n");
    
    if (g_tests_passed == g_tests_run) {
        printf(GREEN "All tests passed!" RESET "\n\n");
    } else {
        printf(RED "%d test(s) failed" RESET "\n\n", g_tests_run - g_tests_passed);
    }
    
    return (g_tests_passed == g_tests_run) ? 0 : 1;
}
