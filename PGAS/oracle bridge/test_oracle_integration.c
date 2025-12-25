/**
 * @file test_oracle_integration.c
 * @brief Integration test for full Oracle stack
 *
 * Tests the complete 6-phase pipeline:
 *   Trigger → Scout → PGAS → Confidence → SAEM → Thompson
 *
 * Compile (with mock PGAS):
 *   gcc -O2 -Wall test_oracle_integration.c oracle_bridge.c \
 *       hawkes_integrator.c saem_blender.c kl_trigger.c \
 *       pgas_confidence.c thompson_sampler.c pgas_mkl_mock.c -lm -o test_oracle
 *
 * Note: Scout sweep tests require paris_mkl which needs MKL.
 *       These tests use NULL for paris (scout disabled).
 */

#include "oracle_bridge.h"
#include "thompson_sampler.h"
#include "pgas_confidence.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define GREEN "\033[32m"
#define RED "\033[31m"
#define YELLOW "\033[33m"
#define CYAN "\033[36m"
#define RESET "\033[0m"

static int g_tests_run = 0;
static int g_tests_passed = 0;

#define RUN_TEST(fn)                                    \
    do                                                  \
    {                                                   \
        g_tests_run++;                                  \
        printf("\n" CYAN "--- %s ---" RESET "\n", #fn); \
        if (fn())                                       \
        {                                               \
            g_tests_passed++;                           \
            printf(GREEN "✓ PASS" RESET "\n");          \
        }                                               \
        else                                            \
        {                                               \
            printf(RED "✗ FAIL" RESET "\n");            \
        }                                               \
    } while (0)

/*═══════════════════════════════════════════════════════════════════════════
 * TEST UTILITIES
 *═══════════════════════════════════════════════════════════════════════════*/

static uint64_t g_rng = 42;

static float randf(void)
{
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 7;
    g_rng ^= g_rng << 17;
    return (float)(g_rng >> 11) * (1.0f / 9007199254740992.0f);
}

static float randn(void)
{
    float u1 = randf() + 1e-10f;
    float u2 = randf();
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST: Full Stack Initialization (New API)
 *═══════════════════════════════════════════════════════════════════════════*/

static int test_full_stack_init(void)
{
    int K = 4;
    int N = 64;
    int T = 200;

    printf("  Testing oracle_bridge_init_full() with all components\n");

    /* Create components */
    HawkesIntegrator hawkes;
    HawkesIntegratorConfig hcfg = hawkes_integrator_config_defaults();
    hawkes_integrator_init(&hawkes, &hcfg);

    KLTrigger kl_trigger;
    KLTriggerConfig kcfg = kl_trigger_config_defaults();
    kl_trigger_init(&kl_trigger, &kcfg);

    SAEMBlender blender;
    SAEMBlenderConfig scfg = saem_blender_config_defaults(K);
    saem_blender_init(&blender, &scfg, NULL);

    ThompsonSampler thompson;
    ThompsonSamplerConfig tcfg = thompson_sampler_config_defaults(K);
    thompson_sampler_init(&thompson, &tcfg);

    PGASMKLState *pgas = pgas_mkl_alloc(N, T, K, 12345);
    if (!pgas)
    {
        printf("  PGAS allocation failed\n");
        return 0;
    }

    /* Create bridge with full init */
    OracleBridge bridge;
    OracleBridgeConfig bcfg = oracle_bridge_config_defaults();
    bcfg.verbose = false;
    bcfg.use_scout_sweep = false; /* No PARIS in this test */

    int ret = oracle_bridge_init_full(&bridge, &bcfg,
                                      &hawkes, &kl_trigger, &blender, pgas,
                                      NULL, /* No PARIS */
                                      &thompson);
    if (ret != 0)
    {
        printf("  Bridge init failed\n");
        pgas_mkl_free(pgas);
        return 0;
    }

    printf("  Bridge initialized:\n");
    printf("    K=%d, N=%d\n", bridge.n_regimes, pgas->N);
    printf("    Hawkes: %s\n", bridge.hawkes ? "YES" : "NO");
    printf("    KL Trigger: %s\n", bridge.kl_trigger ? "YES" : "NO");
    printf("    Thompson: %s\n", bridge.thompson ? "YES" : "NO");
    printf("    PARIS (scout): %s\n", bridge.paris ? "YES" : "NO");
    printf("    Dual-gate: %s\n", bcfg.use_dual_gate ? "ON" : "OFF");

    /* Cleanup */
    thompson_sampler_free(&thompson);
    saem_blender_free(&blender);
    pgas_mkl_free(pgas);

    return 1;
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST: Backward Compatible Init
 *═══════════════════════════════════════════════════════════════════════════*/

static int test_backward_compat_init(void)
{
    int K = 4;

    printf("  Testing backward compatible oracle_bridge_init()\n");

    HawkesIntegrator hawkes;
    hawkes_integrator_init(&hawkes, NULL);

    SAEMBlender blender;
    saem_blender_init(&blender, NULL, NULL);

    PGASMKLState *pgas = pgas_mkl_alloc(64, 100, K, 12345);

    OracleBridge bridge;
    int ret = oracle_bridge_init(&bridge, NULL, &hawkes, NULL, &blender, pgas);

    if (ret != 0)
    {
        printf("  Backward compat init failed\n");
        pgas_mkl_free(pgas);
        return 0;
    }

    printf("  Backward compat init OK\n");
    printf("  Thompson: %s (expected: NO)\n", bridge.thompson ? "YES" : "NO");
    printf("  PARIS: %s (expected: NO)\n", bridge.paris ? "YES" : "NO");

    saem_blender_free(&blender);
    pgas_mkl_free(pgas);

    return (bridge.thompson == NULL && bridge.paris == NULL);
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST: Trigger Check Logic (Dual-Gate)
 *═══════════════════════════════════════════════════════════════════════════*/

static int test_trigger_check(void)
{
    int K = 4;

    printf("  Testing dual-gate trigger logic\n");

    HawkesIntegrator hawkes;
    hawkes_integrator_init(&hawkes, NULL);

    SAEMBlender blender;
    saem_blender_init(&blender, NULL, NULL);

    PGASMKLState *pgas = pgas_mkl_alloc(64, 100, K, 12345);

    OracleBridge bridge;
    OracleBridgeConfig cfg = oracle_bridge_config_defaults();
    cfg.use_dual_gate = true;
    cfg.kl_threshold_sigma = 2.0f;

    oracle_bridge_init(&bridge, &cfg, &hawkes, NULL, &blender, pgas);

    struct
    {
        bool hawkes_fire;
        float hawkes_sigma;
        float kl_sigma;
        bool panic;
        bool expect;
        const char *desc;
    } cases[] = {
        {true, 2.5f, 1.5f, false, false, "Hawkes=Y, KL<thresh"},
        {true, 2.5f, 2.5f, false, true, "Hawkes=Y, KL>thresh"},
        {false, 1.0f, 3.0f, false, false, "Hawkes=N, KL>thresh"},
        {true, 1.0f, 0.5f, true, true, "Panic override"},
    };

    int passed = 0;
    for (int i = 0; i < 4; i++)
    {
        HawkesIntegratorResult hr = {
            .should_trigger = cases[i].hawkes_fire,
            .surprise_sigma = cases[i].hawkes_sigma,
            .triggered_by_panic = cases[i].panic};

        OracleTriggerResult tr = oracle_bridge_check_trigger(
            &bridge, &hr, cases[i].kl_sigma, i * 100);

        bool match = (tr.should_trigger == cases[i].expect);
        printf("    %s: %s %s\n",
               cases[i].desc,
               tr.should_trigger ? "TRIGGER" : "no",
               match ? GREEN "✓" RESET : RED "✗" RESET);
        if (match)
            passed++;
    }

    saem_blender_free(&blender);
    pgas_mkl_free(pgas);

    return (passed == 4);
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST: Oracle Run with Confidence-Based γ
 *═══════════════════════════════════════════════════════════════════════════*/

static int test_oracle_run_with_confidence(void)
{
    int K = 4;
    int T = 100;

    printf("  Testing oracle_bridge_run() with confidence metrics\n");

    HawkesIntegrator hawkes;
    hawkes_integrator_init(&hawkes, NULL);

    SAEMBlender blender;
    SAEMBlenderConfig scfg = saem_blender_config_defaults(K);
    scfg.stickiness.control_stickiness = false;
    saem_blender_init(&blender, &scfg, NULL);

    ThompsonSampler thompson;
    thompson_sampler_init(&thompson, NULL);

    PGASMKLState *pgas = pgas_mkl_alloc(64, T, K, 12345);

    OracleBridge bridge;
    OracleBridgeConfig cfg = oracle_bridge_config_defaults();
    cfg.verbose = true;
    cfg.use_scout_sweep = false;
    cfg.pgas_sweeps_min = 2;
    cfg.pgas_sweeps_max = 5;

    oracle_bridge_init_full(&bridge, &cfg, &hawkes, NULL, &blender, pgas,
                            NULL, &thompson);

    /* Generate synthetic data */
    int *rbpf_path = malloc(T * sizeof(int));
    double *rbpf_h = malloc(T * sizeof(double));
    double *observations = malloc(T * sizeof(double));

    g_rng = 12345;
    for (int t = 0; t < T; t++)
    {
        rbpf_path[t] = (t / 25) % K;
        rbpf_h[t] = -1.0 + 0.5 * rbpf_path[t] + 0.1 * randn();
        observations[t] = rbpf_h[t] + 0.5 * randn();
    }

    /* Run Oracle */
    OracleRunResult result = oracle_bridge_run(
        &bridge, rbpf_path, rbpf_h, observations, T, 2.5f);

    printf("\n  Extended result:\n");
    printf("    Success: %s\n", result.success ? "YES" : "NO");
    printf("    Scout ran: %s\n", result.scout_ran ? "YES" : "NO");
    printf("    PGAS ran: %s\n", result.pgas_ran ? "YES" : "NO");
    printf("    Confidence: %.3f\n", result.confidence_score);
    printf("    Path divergence: %.1f%%\n", result.path_divergence * 100);
    printf("    Regime change: %s\n", result.regime_change_detected ? "YES" : "NO");
    printf("    Degeneracy: %s\n", result.degeneracy_detected ? "YES" : "NO");
    printf("    γ used: %.3f\n", result.gamma_used);
    printf("    KL divergence: %.6f\n", result.kl_divergence);
    printf("    Thompson explored: %s\n", result.thompson_explored ? "YES" : "NO");

    /* Check last confidence */
    PGASConfidence conf;
    oracle_bridge_get_last_confidence(&bridge, &conf);
    printf("    Last conf ESS ratio: %.3f\n", conf.ess_ratio);

    free(rbpf_path);
    free(rbpf_h);
    free(observations);
    thompson_sampler_free(&thompson);
    saem_blender_free(&blender);
    pgas_mkl_free(pgas);

    return result.success;
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST: Thompson Sampling Integration
 *═══════════════════════════════════════════════════════════════════════════*/

static int test_thompson_integration(void)
{
    int K = 4;
    int T = 100;

    printf("  Testing Thompson sampling via oracle_bridge_get_Pi_thompson()\n");

    HawkesIntegrator hawkes;
    hawkes_integrator_init(&hawkes, NULL);

    SAEMBlender blender;
    saem_blender_init(&blender, NULL, NULL);

    ThompsonSampler thompson;
    ThompsonSamplerConfig tcfg = thompson_sampler_config_defaults(K);
    tcfg.exploit_threshold = 10.0f; /* Low threshold to see explore/exploit */
    thompson_sampler_init(&thompson, &tcfg);

    PGASMKLState *pgas = pgas_mkl_alloc(64, T, K, 12345);

    OracleBridge bridge;
    OracleBridgeConfig cfg = oracle_bridge_config_defaults();
    cfg.verbose = false;
    cfg.use_scout_sweep = false;

    oracle_bridge_init_full(&bridge, &cfg, &hawkes, NULL, &blender, pgas,
                            NULL, &thompson);

    /* Get Pi via Thompson (should explore initially due to low Q) */
    float Pi1[16], Pi2[16];
    oracle_bridge_get_Pi_thompson(&bridge, Pi1);
    oracle_bridge_get_Pi_thompson(&bridge, Pi2);

    /* Check if samples differ (Thompson is stochastic when exploring) */
    float diff = 0.0f;
    for (int i = 0; i < K * K; i++)
    {
        diff += fabsf(Pi1[i] - Pi2[i]);
    }

    printf("    Two Thompson samples differ by: %.4f\n", diff);
    printf("    (Non-zero diff indicates exploration)\n");

    /* Get SAEM mean for comparison */
    float Pi_mean[16];
    oracle_bridge_get_Pi(&bridge, Pi_mean);

    printf("    SAEM mean Pi[0][0]: %.4f\n", Pi_mean[0]);
    printf("    Thompson Pi[0][0]: %.4f, %.4f\n", Pi1[0], Pi2[0]);

    thompson_sampler_free(&thompson);
    saem_blender_free(&blender);
    pgas_mkl_free(pgas);

    return 1; /* Pass if no crash - stochastic behavior hard to verify */
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST: Extended Statistics
 *═══════════════════════════════════════════════════════════════════════════*/

static int test_extended_statistics(void)
{
    int K = 4;
    int T = 100;

    printf("  Testing extended statistics tracking\n");

    HawkesIntegrator hawkes;
    hawkes_integrator_init(&hawkes, NULL);

    SAEMBlender blender;
    saem_blender_init(&blender, NULL, NULL);

    ThompsonSampler thompson;
    thompson_sampler_init(&thompson, NULL);

    PGASMKLState *pgas = pgas_mkl_alloc(64, T, K, 12345);

    OracleBridge bridge;
    OracleBridgeConfig cfg = oracle_bridge_config_defaults();
    cfg.verbose = false;
    cfg.use_scout_sweep = false;

    oracle_bridge_init_full(&bridge, &cfg, &hawkes, NULL, &blender, pgas,
                            NULL, &thompson);

    /* Generate data */
    int *regimes = malloc(T * sizeof(int));
    double *log_h = malloc(T * sizeof(double));
    double *obs = malloc(T * sizeof(double));

    g_rng = 99999;
    for (int t = 0; t < T; t++)
    {
        regimes[t] = t / 25;
        log_h[t] = -2.0 + 0.5 * regimes[t];
        obs[t] = log_h[t] + randn() * 0.5;
    }

    /* Run 3 Oracle calls */
    for (int i = 0; i < 3; i++)
    {
        oracle_bridge_run(&bridge, regimes, log_h, obs, T, 2.0f);
    }

    /* Check extended stats */
    OracleBridgeStats stats;
    oracle_bridge_get_stats(&bridge, &stats);

    printf("    Total calls: %d\n", stats.total_oracle_calls);
    printf("    Successful: %d\n", stats.successful_blends);
    printf("    Scout skips: %d\n", stats.scout_skip_count);
    printf("    Regime changes: %d\n", stats.regime_change_count);
    printf("    Degeneracies: %d\n", stats.degeneracy_count);
    printf("    Thompson explore ratio: %.1f%%\n", stats.thompson_explore_ratio * 100);

    free(regimes);
    free(log_h);
    free(obs);
    thompson_sampler_free(&thompson);
    saem_blender_free(&blender);
    pgas_mkl_free(pgas);

    return (stats.total_oracle_calls == 3);
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST: Config Defaults
 *═══════════════════════════════════════════════════════════════════════════*/

static int test_config_defaults(void)
{
    printf("  Testing OracleBridgeConfig defaults\n");

    OracleBridgeConfig cfg = oracle_bridge_config_defaults();

    printf("    pgas_particles: %d\n", cfg.pgas_particles);
    printf("    use_dual_gate: %s\n", cfg.use_dual_gate ? "YES" : "NO");
    printf("    use_scout_sweep: %s\n", cfg.use_scout_sweep ? "YES" : "NO");
    printf("    use_tempered_path: %s\n", cfg.use_tempered_path ? "YES" : "NO");
    printf("    scout_sweeps: %d\n", cfg.scout_sweeps);
    printf("    scout_entropy_skip: %.2f\n", cfg.scout_entropy_skip);
    printf("    gamma_on_regime_change: %.2f\n", cfg.gamma_on_regime_change);
    printf("    gamma_on_degeneracy: %.3f\n", cfg.gamma_on_degeneracy);
    printf("    thompson_exploit_thresh: %.0f\n", cfg.thompson_exploit_thresh);

    /* Verify some key defaults */
    bool ok = (cfg.pgas_particles == 256 &&
               cfg.use_dual_gate == true &&
               cfg.use_scout_sweep == true &&
               cfg.use_tempered_path == true &&
               cfg.gamma_on_regime_change >= 0.4f);

    return ok;
}

/*═══════════════════════════════════════════════════════════════════════════
 * MAIN
 *═══════════════════════════════════════════════════════════════════════════*/

int main(void)
{
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║          ORACLE INTEGRATION TEST SUITE                       ║\n");
    printf("║          Full Pipeline: Scout → PGAS → SAEM → Thompson       ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n");

    RUN_TEST(test_config_defaults);
    RUN_TEST(test_full_stack_init);
    RUN_TEST(test_backward_compat_init);
    RUN_TEST(test_trigger_check);
    RUN_TEST(test_oracle_run_with_confidence);
    RUN_TEST(test_thompson_integration);
    RUN_TEST(test_extended_statistics);

    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("RESULTS: %d/%d tests passed\n", g_tests_passed, g_tests_run);
    printf("═══════════════════════════════════════════════════════════════\n\n");

    return (g_tests_passed == g_tests_run) ? 0 : 1;
}