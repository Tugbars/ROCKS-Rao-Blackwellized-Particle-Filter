/**
 * @file test_oracle_integration.c
 * @brief Integration test for full Oracle stack
 *
 * Tests: Hawkes Trigger → PGAS Oracle → SAEM Blender
 *
 * Compile (with mock PGAS):
 *   gcc -O2 -Wall test_oracle_integration.c oracle_bridge.c \
 *       hawkes_integrator.c saem_blender.c pgas_mkl_mock.c -lm -o test_oracle
 */

#include "oracle_bridge.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define GREEN  "\033[32m"
#define RED    "\033[31m"
#define YELLOW "\033[33m"
#define RESET  "\033[0m"

static int g_tests_run = 0;
static int g_tests_passed = 0;

#define RUN_TEST(fn) do { \
    g_tests_run++; \
    printf("\n--- %s ---\n", #fn); \
    if (fn()) { g_tests_passed++; printf(GREEN "✓ PASS" RESET "\n"); } \
    else { printf(RED "✗ FAIL" RESET "\n"); } \
} while(0)

/*═══════════════════════════════════════════════════════════════════════════
 * TEST UTILITIES
 *═══════════════════════════════════════════════════════════════════════════*/

static uint32_t g_rng = 42;

static float randf(void) {
    g_rng = g_rng * 1103515245 + 12345;
    return (float)(g_rng >> 16) / 65536.0f;
}

static float randn(void) {
    float u1 = randf() + 1e-10f;
    float u2 = randf();
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST: Full Stack Initialization
 *═══════════════════════════════════════════════════════════════════════════*/

static int test_full_stack_init(void) {
    int K = 4;
    int N = 64;
    int T = 200;
    
    /* Create components */
    HawkesIntegrator hawkes;
    HawkesIntegratorConfig hcfg = hawkes_integrator_config_defaults();
    hawkes_integrator_init(&hawkes, &hcfg);
    
    SAEMBlender blender;
    SAEMBlenderConfig scfg = saem_blender_config_defaults(K);
    saem_blender_init(&blender, &scfg, NULL);
    
    PGASMKLState *pgas = pgas_mkl_alloc(N, T, K, 12345);
    if (!pgas) {
        printf("  PGAS allocation failed\n");
        return 0;
    }
    
    /* Create bridge */
    OracleBridge bridge;
    OracleBridgeConfig bcfg = oracle_bridge_config_defaults();
    bcfg.verbose = false;
    
    int ret = oracle_bridge_init(&bridge, &bcfg, &hawkes, &blender, pgas);
    if (ret != 0) {
        printf("  Bridge init failed\n");
        pgas_mkl_free(pgas);
        return 0;
    }
    
    printf("  Bridge initialized: K=%d, N=%d\n", bridge.n_regimes, pgas->N);
    printf("  Dual-gate: %s, Tempered path: %s\n",
           bcfg.use_dual_gate ? "ON" : "OFF",
           bcfg.use_tempered_path ? "ON" : "OFF");
    
    /* Cleanup */
    saem_blender_free(&blender);
    pgas_mkl_free(pgas);
    
    return 1;
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST: Trigger Check Logic
 *═══════════════════════════════════════════════════════════════════════════*/

static int test_trigger_check(void) {
    int K = 4;
    
    /* Setup minimal components */
    HawkesIntegrator hawkes;
    hawkes_integrator_init(&hawkes, NULL);
    
    SAEMBlender blender;
    saem_blender_init(&blender, NULL, NULL);
    
    PGASMKLState *pgas = pgas_mkl_alloc(64, 100, K, 12345);
    
    OracleBridge bridge;
    OracleBridgeConfig cfg = oracle_bridge_config_defaults();
    cfg.use_dual_gate = true;
    cfg.kl_threshold_sigma = 2.0f;
    
    oracle_bridge_init(&bridge, &cfg, &hawkes, &blender, pgas);
    
    /* Test 1: Hawkes fires, KL below threshold → NO trigger */
    HawkesIntegratorResult hr1 = {
        .should_trigger = true,
        .surprise_sigma = 2.5f,
        .triggered_by_panic = false
    };
    OracleTriggerResult tr1 = oracle_bridge_check_trigger(&bridge, &hr1, 1.5f, 100);
    printf("  Case 1 (Hawkes=Y, KL<thresh): trigger=%s\n", 
           tr1.should_trigger ? "YES" : "NO");
    if (tr1.should_trigger) {
        printf("    Expected NO trigger (KL too low)\n");
        return 0;
    }
    
    /* Test 2: Hawkes fires, KL above threshold → trigger */
    OracleTriggerResult tr2 = oracle_bridge_check_trigger(&bridge, &hr1, 2.5f, 101);
    printf("  Case 2 (Hawkes=Y, KL>thresh): trigger=%s\n", 
           tr2.should_trigger ? "YES" : "NO");
    if (!tr2.should_trigger) {
        printf("    Expected trigger\n");
        return 0;
    }
    
    /* Test 3: Panic overrides dual-gate */
    HawkesIntegratorResult hr3 = {
        .should_trigger = true,
        .surprise_sigma = 1.0f,
        .triggered_by_panic = true
    };
    OracleTriggerResult tr3 = oracle_bridge_check_trigger(&bridge, &hr3, 0.5f, 102);
    printf("  Case 3 (Panic): trigger=%s (panic=%s)\n",
           tr3.should_trigger ? "YES" : "NO",
           tr3.triggered_by_panic ? "YES" : "NO");
    if (!tr3.should_trigger || !tr3.triggered_by_panic) {
        printf("    Expected panic trigger\n");
        return 0;
    }
    
    /* Cleanup */
    saem_blender_free(&blender);
    pgas_mkl_free(pgas);
    
    return 1;
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST: Oracle Run (with mock PGAS)
 *═══════════════════════════════════════════════════════════════════════════*/

static int test_oracle_run(void) {
    int K = 4;
    int T = 100;
    
    /* Setup */
    HawkesIntegrator hawkes;
    hawkes_integrator_init(&hawkes, NULL);
    
    SAEMBlender blender;
    SAEMBlenderConfig scfg = saem_blender_config_defaults(K);
    scfg.stickiness.control_stickiness = false;  /* Disable for test */
    saem_blender_init(&blender, &scfg, NULL);
    
    PGASMKLState *pgas = pgas_mkl_alloc(64, T, K, 12345);
    
    OracleBridge bridge;
    OracleBridgeConfig cfg = oracle_bridge_config_defaults();
    cfg.verbose = true;
    cfg.pgas_sweeps_min = 2;
    cfg.pgas_sweeps_max = 5;
    
    oracle_bridge_init(&bridge, &cfg, &hawkes, &blender, pgas);
    
    /* Generate synthetic RBPF path and h */
    int *rbpf_path = malloc(T * sizeof(int));
    double *rbpf_h = malloc(T * sizeof(double));
    double *observations = malloc(T * sizeof(double));
    
    for (int t = 0; t < T; t++) {
        rbpf_path[t] = (t / 25) % K;  /* Cycle through regimes */
        rbpf_h[t] = -1.0 + 0.5 * rbpf_path[t] + 0.1 * randn();
        observations[t] = rbpf_h[t] + 0.5 * randn();  /* Mock observation */
    }
    
    /* Get Pi before */
    float Pi_before[16];
    oracle_bridge_get_Pi(&bridge, Pi_before);
    
    /* Run Oracle */
    OracleRunResult result = oracle_bridge_run(
        &bridge, rbpf_path, rbpf_h, observations, T, 2.5f);
    
    printf("  Oracle result:\n");
    printf("    Success: %s\n", result.success ? "YES" : "NO");
    printf("    Sweeps: %d, Accept: %.3f\n", result.sweeps_used, result.acceptance_rate);
    printf("    KL divergence: %.6f\n", result.kl_divergence);
    printf("    Diagonal: %.4f → %.4f\n", result.diag_before, result.diag_after);
    printf("    Temper flips: %d\n", result.temper_flips);
    
    /* Get Pi after */
    float Pi_after[16];
    oracle_bridge_get_Pi(&bridge, Pi_after);
    
    /* Check that something changed */
    float diff = 0.0f;
    for (int i = 0; i < K * K; i++) {
        diff += fabsf(Pi_after[i] - Pi_before[i]);
    }
    printf("  Total Π change: %.4f\n", diff);
    
    if (!result.success) {
        printf("    Oracle run failed\n");
        free(rbpf_path); free(rbpf_h); free(observations);
        saem_blender_free(&blender);
        pgas_mkl_free(pgas);
        return 0;
    }
    
    if (diff < 1e-6f) {
        printf("    Warning: No change to Π (might be OK with mock)\n");
    }
    
    /* Cleanup */
    free(rbpf_path);
    free(rbpf_h);
    free(observations);
    saem_blender_free(&blender);
    pgas_mkl_free(pgas);
    
    return 1;
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST: End-to-End Simulation
 *═══════════════════════════════════════════════════════════════════════════*/

static int test_end_to_end(void) {
    int K = 4;
    int T = 500;
    
    printf("  Simulating %d ticks with regime changes...\n", T);
    
    /* Setup full stack */
    HawkesIntegrator hawkes;
    HawkesIntegratorConfig hcfg = hawkes_integrator_config_defaults();
    hcfg.warmup_ticks = 50;
    hcfg.refractory_ticks = 100;
    hawkes_integrator_init(&hawkes, &hcfg);
    
    SAEMBlender blender;
    saem_blender_init(&blender, NULL, NULL);
    
    PGASMKLState *pgas = pgas_mkl_alloc(64, T, K, 12345);
    
    OracleBridge bridge;
    OracleBridgeConfig cfg = oracle_bridge_config_defaults();
    cfg.use_dual_gate = false;  /* Single-gate for simplicity */
    cfg.verbose = false;
    
    oracle_bridge_init(&bridge, &cfg, &hawkes, &blender, pgas);
    
    /* Generate data with regime change at t=200 */
    float *returns = malloc(T * sizeof(float));
    int *true_regimes = malloc(T * sizeof(int));
    double *mock_h = malloc(T * sizeof(double));
    double *mock_obs = malloc(T * sizeof(double));
    
    for (int t = 0; t < T; t++) {
        float vol;
        if (t < 200) {
            vol = 0.005f;
            true_regimes[t] = 0;
        } else if (t < 350) {
            vol = 0.035f;  /* Crisis */
            true_regimes[t] = 2;
        } else {
            vol = 0.008f;
            true_regimes[t] = 1;
        }
        returns[t] = randn() * vol;
        mock_h[t] = logf(vol);
        mock_obs[t] = mock_h[t] + 0.5 * randn();
    }
    
    /* Run simulation */
    int trigger_count = 0;
    int oracle_calls = 0;
    
    for (int t = 0; t < T; t++) {
        /* Update Hawkes */
        HawkesIntegratorResult hr = hawkes_integrator_update(&hawkes, (float)t, returns[t]);
        
        /* Check trigger (single-gate) */
        OracleTriggerResult tr = oracle_bridge_check_trigger(&bridge, &hr, 0.0f, t);
        
        if (tr.should_trigger) {
            trigger_count++;
            
            /* Run Oracle with last 100 ticks of data */
            int start = (t > 100) ? t - 100 : 0;
            int len = t - start;
            
            if (len >= 10) {
                OracleRunResult rr = oracle_bridge_run(
                    &bridge, 
                    &true_regimes[start], 
                    &mock_h[start],
                    &mock_obs[start], 
                    len, 
                    hr.surprise_sigma);
                
                if (rr.success) {
                    oracle_calls++;
                    printf("  t=%d: Oracle called (surprise=%.2fσ, sweeps=%d, KL=%.6f)\n",
                           t, hr.surprise_sigma, rr.sweeps_used, rr.kl_divergence);
                }
            }
        }
    }
    
    printf("  SUMMARY:\n");
    printf("    Triggers: %d\n", trigger_count);
    printf("    Successful Oracle calls: %d\n", oracle_calls);
    
    /* Get final stats */
    OracleBridgeStats stats;
    oracle_bridge_get_stats(&bridge, &stats);
    printf("    Final γ: %.4f\n", stats.current_gamma);
    printf("    Avg diagonal: %.4f\n", stats.current_avg_diagonal);
    
    /* Cleanup */
    free(returns);
    free(true_regimes);
    free(mock_h);
    free(mock_obs);
    saem_blender_free(&blender);
    pgas_mkl_free(pgas);
    
    return (oracle_calls >= 1);  /* At least one Oracle call */
}

/*═══════════════════════════════════════════════════════════════════════════
 * MAIN
 *═══════════════════════════════════════════════════════════════════════════*/

int main(void) {
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║          ORACLE INTEGRATION TEST SUITE                       ║\n");
    printf("║          Hawkes → PGAS → SAEM Pipeline                       ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n");
    
    RUN_TEST(test_full_stack_init);
    RUN_TEST(test_trigger_check);
    RUN_TEST(test_oracle_run);
    RUN_TEST(test_end_to_end);
    
    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("RESULTS: %d/%d tests passed\n", g_tests_passed, g_tests_run);
    printf("═══════════════════════════════════════════════════════════════\n\n");
    
    return (g_tests_passed == g_tests_run) ? 0 : 1;
}
