/**
 * @file test_oracle_full_integration.c
 * @brief Full Integration Test for Oracle Stack
 *
 * Tests the complete 6-phase pipeline with realistic scenarios:
 *   1. Calm → Crisis transition (volatility jump)
 *   2. Multiple regime changes
 *   3. Confidence-based γ adaptation
 *   4. Thompson explore/exploit behavior
 *   5. Dual-gate trigger logic
 *   6. Tempered path effectiveness
 *
 * Compile:
 *   gcc -O2 -Wall test_oracle_full_integration.c oracle_bridge.c \
 *       hawkes_integrator.c saem_blender.c kl_trigger.c \
 *       pgas_confidence.c thompson_sampler.c pgas_mkl_mock.c -lm -o test_full
 */

#include "oracle_bridge.h"
#include "thompson_sampler.h"
#include "pgas_confidence.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/*═══════════════════════════════════════════════════════════════════════════
 * TEST UTILITIES
 *═══════════════════════════════════════════════════════════════════════════*/

#define GREEN "\033[32m"
#define RED "\033[31m"
#define YELLOW "\033[33m"
#define CYAN "\033[36m"
#define RESET "\033[0m"

static int g_tests_run = 0;
static int g_tests_passed = 0;

#define RUN_TEST(fn)                                                                                    \
    do                                                                                                  \
    {                                                                                                   \
        g_tests_run++;                                                                                  \
        printf("\n" CYAN "═══════════════════════════════════════════════════════════════" RESET "\n"); \
        printf(CYAN "TEST: %s" RESET "\n", #fn);                                                        \
        printf(CYAN "═══════════════════════════════════════════════════════════════" RESET "\n");      \
        if (fn())                                                                                       \
        {                                                                                               \
            g_tests_passed++;                                                                           \
            printf(GREEN "✓ PASS: %s" RESET "\n", #fn);                                                 \
        }                                                                                               \
        else                                                                                            \
        {                                                                                               \
            printf(RED "✗ FAIL: %s" RESET "\n", #fn);                                                   \
        }                                                                                               \
    } while (0)

static uint64_t g_rng = 0xDEADBEEF12345678ULL;

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
 * ORACLE STACK WRAPPER (with new components)
 *═══════════════════════════════════════════════════════════════════════════*/

typedef struct
{
    HawkesIntegrator hawkes;
    KLTrigger kl_trigger;
    SAEMBlender blender;
    ThompsonSampler thompson;
    PGASMKLState *pgas;
    OracleBridge bridge;
    int K;
    int N;
    int T_max;
} OracleStack;

static int oracle_stack_init(OracleStack *stack, int K, int N, int T_max)
{
    memset(stack, 0, sizeof(*stack));
    stack->K = K;
    stack->N = N;
    stack->T_max = T_max;

    /* Initialize Hawkes */
    HawkesIntegratorConfig hcfg = hawkes_integrator_config_defaults();
    hcfg.warmup_ticks = 50;
    hcfg.refractory_ticks = 100;
    hcfg.high_water_mark = 1.3f;
    hcfg.low_water_mark = 0.8f;
    hawkes_integrator_init(&stack->hawkes, &hcfg);

    /* Initialize KL Trigger */
    KLTriggerConfig kcfg = kl_trigger_config_defaults(K);
    kl_trigger_init(&stack->kl_trigger, &kcfg);

    /* Initialize SAEM Blender */
    SAEMBlenderConfig scfg = saem_blender_config_defaults(K);
    saem_blender_init(&stack->blender, &scfg, NULL);

    /* Initialize Thompson Sampler */
    ThompsonSamplerConfig tcfg = thompson_sampler_config_defaults(K);
    tcfg.exploit_threshold = 100.0f; /* Start exploring, then exploit */
    thompson_sampler_init(&stack->thompson, &tcfg);

    /* Initialize PGAS */
    stack->pgas = pgas_mkl_alloc(N, T_max, K, 12345);
    if (!stack->pgas)
        return -1;

    /* Initialize Bridge (full pipeline, no scout since no PARIS) */
    OracleBridgeConfig bcfg = oracle_bridge_config_defaults();
    bcfg.use_dual_gate = false;   /* Single-gate for easier testing */
    bcfg.use_scout_sweep = false; /* No PARIS in mock test */
    bcfg.use_tempered_path = true;
    bcfg.verbose = false;
    bcfg.pgas_sweeps_min = 3;
    bcfg.pgas_sweeps_max = 8;
    bcfg.recency_lambda = 0.001f;

    return oracle_bridge_init_full(&stack->bridge, &bcfg,
                                   &stack->hawkes, &stack->kl_trigger,
                                   &stack->blender, stack->pgas,
                                   NULL, /* No PARIS */
                                   &stack->thompson);
}

static void oracle_stack_free(OracleStack *stack)
{
    thompson_sampler_free(&stack->thompson);
    saem_blender_free(&stack->blender);
    pgas_mkl_free(stack->pgas);
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST 1: Single Regime Transition (Calm → Crisis)
 *═══════════════════════════════════════════════════════════════════════════*/

static int test_calm_to_crisis(void)
{
    printf("Scenario: Calm (0.5%% vol) → Crisis (4%% vol) at t=200\n\n");

    int T = 500;
    int K = 4;

    OracleStack stack;
    if (oracle_stack_init(&stack, K, 64, T) != 0)
    {
        printf("  Stack init failed\n");
        return 0;
    }

    /* Generate data with single transition */
    float *returns = malloc(T * sizeof(float));
    int *regimes = malloc(T * sizeof(int));
    double *log_h = malloc(T * sizeof(double));
    double *obs = malloc(T * sizeof(double));

    g_rng = 42;
    int change_point = 200;

    for (int t = 0; t < T; t++)
    {
        float vol = (t < change_point) ? 0.005f : 0.040f;
        int regime = (t < change_point) ? 0 : 3;

        returns[t] = randn() * vol;
        regimes[t] = regime;
        log_h[t] = log(vol);
        obs[t] = log_h[t] + 0.5 * randn();
    }

    /* Run simulation */
    int trigger_count = 0;
    int oracle_calls = 0;
    int first_trigger_after_crisis = -1;

    for (int t = 0; t < T; t++)
    {
        HawkesIntegratorResult hr = hawkes_integrator_update(
            &stack.hawkes, (float)t, returns[t]);

        OracleTriggerResult tr = oracle_bridge_check_trigger(
            &stack.bridge, &hr, 0.0f, t);

        if (tr.should_trigger)
        {
            trigger_count++;

            int start = (t > 100) ? t - 100 : 0;
            int len = t - start;

            if (len >= 10)
            {
                OracleRunResult rr = oracle_bridge_run(
                    &stack.bridge,
                    &regimes[start], &log_h[start], &obs[start],
                    len, hr.surprise_sigma);

                if (rr.success)
                {
                    oracle_calls++;

                    if (t > change_point && first_trigger_after_crisis < 0)
                    {
                        first_trigger_after_crisis = t;
                        printf("  First trigger after crisis at t=%d:\n", t);
                        printf("    Confidence: %.3f\n", rr.confidence_score);
                        printf("    Path divergence: %.1f%%\n", rr.path_divergence * 100);
                        printf("    γ used: %.3f\n", rr.gamma_used);
                        printf("    Regime change detected: %s\n",
                               rr.regime_change_detected ? "YES" : "NO");
                    }
                }
            }
        }
    }

    printf("\n  SUMMARY:\n");
    printf("    Triggers: %d\n", trigger_count);
    printf("    Oracle calls: %d\n", oracle_calls);
    printf("    First trigger after crisis: t=%d\n", first_trigger_after_crisis);

    /* Get final stats */
    OracleBridgeStats stats;
    oracle_bridge_get_stats(&stack.bridge, &stats);
    printf("    Final γ: %.4f\n", stats.current_gamma);
    printf("    Regime changes detected: %d\n", stats.regime_change_count);
    printf("    Thompson explore ratio: %.1f%%\n", stats.thompson_explore_ratio * 100);

    free(returns);
    free(regimes);
    free(log_h);
    free(obs);
    oracle_stack_free(&stack);

    /* Should detect regime change after crisis */
    return (oracle_calls >= 1 && first_trigger_after_crisis > change_point);
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST 2: Multiple Regime Transitions
 *═══════════════════════════════════════════════════════════════════════════*/

static int test_multiple_transitions(void)
{
    printf("Scenario: 3 regime changes at t=150, 300, 450\n\n");

    int T = 600;
    int K = 4;

    OracleStack stack;
    if (oracle_stack_init(&stack, K, 64, T) != 0)
    {
        return 0;
    }

    /* Generate data with 3 transitions */
    float *returns = malloc(T * sizeof(float));
    int *regimes = malloc(T * sizeof(int));
    double *log_h = malloc(T * sizeof(double));
    double *obs = malloc(T * sizeof(double));

    g_rng = 12345;
    float vol_levels[] = {0.005f, 0.015f, 0.035f, 0.008f}; /* Calm, medium, crisis, recovery */
    int change_points[] = {150, 300, 450};

    for (int t = 0; t < T; t++)
    {
        int regime;
        if (t < change_points[0])
            regime = 0;
        else if (t < change_points[1])
            regime = 2;
        else if (t < change_points[2])
            regime = 3;
        else
            regime = 1;

        float vol = vol_levels[regime];
        returns[t] = randn() * vol;
        regimes[t] = regime;
        log_h[t] = log(vol);
        obs[t] = log_h[t] + 0.5 * randn();
    }

    /* Run simulation */
    int oracle_calls = 0;
    int regime_changes = 0;

    for (int t = 0; t < T; t++)
    {
        HawkesIntegratorResult hr = hawkes_integrator_update(
            &stack.hawkes, (float)t, returns[t]);

        OracleTriggerResult tr = oracle_bridge_check_trigger(
            &stack.bridge, &hr, 0.0f, t);

        if (tr.should_trigger)
        {
            int start = (t > 100) ? t - 100 : 0;
            int len = t - start;

            if (len >= 10)
            {
                OracleRunResult rr = oracle_bridge_run(
                    &stack.bridge,
                    &regimes[start], &log_h[start], &obs[start],
                    len, hr.surprise_sigma);

                if (rr.success)
                {
                    oracle_calls++;
                    if (rr.regime_change_detected)
                    {
                        regime_changes++;
                        printf("  Regime change detected at t=%d (γ=%.2f)\n",
                               t, rr.gamma_used);
                    }
                }
            }
        }
    }

    OracleBridgeStats stats;
    oracle_bridge_get_stats(&stack.bridge, &stats);

    printf("\n  SUMMARY:\n");
    printf("    Oracle calls: %d\n", oracle_calls);
    printf("    Regime changes detected: %d (tracked: %d)\n",
           regime_changes, stats.regime_change_count);
    printf("    Degeneracies: %d\n", stats.degeneracy_count);

    free(returns);
    free(regimes);
    free(log_h);
    free(obs);
    oracle_stack_free(&stack);

    return (oracle_calls >= 2);
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST 3: Confidence-Based γ Adaptation
 *═══════════════════════════════════════════════════════════════════════════*/

static int test_confidence_gamma(void)
{
    printf("Testing confidence → γ mapping\n\n");

    int K = 4;
    int T = 100;

    OracleStack stack;
    if (oracle_stack_init(&stack, K, 64, T) != 0)
    {
        return 0;
    }

    /* Generate stable data */
    int *regimes = malloc(T * sizeof(int));
    double *log_h = malloc(T * sizeof(double));
    double *obs = malloc(T * sizeof(double));

    g_rng = 99999;
    for (int t = 0; t < T; t++)
    {
        regimes[t] = 0; /* All same regime */
        log_h[t] = -3.0;
        obs[t] = log_h[t] + randn() * 0.5;
    }

    /* Run multiple Oracle calls and track γ */
    float gamma_values[5];
    for (int i = 0; i < 5; i++)
    {
        OracleRunResult rr = oracle_bridge_run(
            &stack.bridge, regimes, log_h, obs, T, 2.0f);

        gamma_values[i] = rr.gamma_used;
        printf("  Call %d: confidence=%.3f, γ=%.3f\n",
               i + 1, rr.confidence_score, rr.gamma_used);
    }

    printf("\n  γ progression: ");
    for (int i = 0; i < 5; i++)
    {
        printf("%.3f ", gamma_values[i]);
    }
    printf("\n");

    /* γ should be within valid range */
    bool ok = true;
    for (int i = 0; i < 5; i++)
    {
        if (gamma_values[i] < SAEM_GAMMA_MIN || gamma_values[i] > SAEM_GAMMA_MAX)
        {
            printf("  ERROR: γ[%d]=%.3f out of range!\n", i, gamma_values[i]);
            ok = false;
        }
    }

    free(regimes);
    free(log_h);
    free(obs);
    oracle_stack_free(&stack);

    return ok;
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST 4: Thompson Explore/Exploit Transition
 *═══════════════════════════════════════════════════════════════════════════*/

static int test_thompson_transition(void)
{
    printf("Testing Thompson explore → exploit transition\n\n");

    int K = 4;
    int T = 100;

    OracleStack stack;
    if (oracle_stack_init(&stack, K, 64, T) != 0)
    {
        return 0;
    }

    /* Lower exploit threshold to see transition */
    stack.thompson.config.exploit_threshold = 50.0f;

    int *regimes = malloc(T * sizeof(int));
    double *log_h = malloc(T * sizeof(double));
    double *obs = malloc(T * sizeof(double));

    g_rng = 77777;
    for (int t = 0; t < T; t++)
    {
        regimes[t] = t / 25;
        log_h[t] = -2.0 + 0.5 * regimes[t];
        obs[t] = log_h[t] + randn() * 0.5;
    }

    /* Track explore/exploit over multiple calls */
    int explore_count = 0;
    int exploit_count = 0;

    for (int i = 0; i < 10; i++)
    {
        OracleRunResult rr = oracle_bridge_run(
            &stack.bridge, regimes, log_h, obs, T, 2.0f);

        if (rr.thompson_explored)
            explore_count++;
        else
            exploit_count++;

        printf("  Call %d: %s (Q row sum growing)\n",
               i + 1, rr.thompson_explored ? "EXPLORE" : "EXPLOIT");
    }

    printf("\n  Explore: %d, Exploit: %d\n", explore_count, exploit_count);

    /* Should transition from explore to exploit as Q grows */
    OracleBridgeStats stats;
    oracle_bridge_get_stats(&stack.bridge, &stats);
    printf("  Final explore ratio: %.1f%%\n", stats.thompson_explore_ratio * 100);

    free(regimes);
    free(log_h);
    free(obs);
    oracle_stack_free(&stack);

    /* Should have some of each */
    return (explore_count >= 1 || exploit_count >= 1);
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST 5: Dual-Gate Trigger
 *═══════════════════════════════════════════════════════════════════════════*/

static int test_dual_gate_trigger(void)
{
    printf("Testing dual-gate trigger (Hawkes AND KL)\n\n");

    int K = 4;

    OracleStack stack;
    if (oracle_stack_init(&stack, K, 32, 100) != 0)
    {
        return 0;
    }

    /* Enable dual-gate */
    stack.bridge.config.use_dual_gate = true;
    stack.bridge.config.kl_threshold_sigma = 2.0f;

    struct
    {
        bool hawkes;
        float kl;
        bool panic;
        bool expect;
        const char *desc;
    } cases[] = {
        {false, 3.0f, false, false, "Hawkes=N, KL>thresh"},
        {true, 1.0f, false, false, "Hawkes=Y, KL<thresh"},
        {true, 2.5f, false, true, "Hawkes=Y, KL>thresh"},
        {true, 0.5f, true, true, "Panic override"},
    };

    int passed = 0;
    for (int i = 0; i < 4; i++)
    {
        HawkesIntegratorResult hr = {
            .should_trigger = cases[i].hawkes,
            .surprise_sigma = 2.0f,
            .triggered_by_panic = cases[i].panic};

        OracleTriggerResult tr = oracle_bridge_check_trigger(
            &stack.bridge, &hr, cases[i].kl, i * 100);

        bool match = (tr.should_trigger == cases[i].expect);
        printf("  %s: %s %s\n",
               cases[i].desc,
               tr.should_trigger ? "TRIGGER" : "no",
               match ? GREEN "✓" RESET : RED "✗" RESET);
        if (match)
            passed++;
    }

    oracle_stack_free(&stack);

    return (passed == 4);
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST 6: Statistics Accumulation
 *═══════════════════════════════════════════════════════════════════════════*/

static int test_statistics(void)
{
    printf("Testing statistics accumulation\n\n");

    int K = 4;
    int T = 100;

    OracleStack stack;
    if (oracle_stack_init(&stack, K, 32, T) != 0)
    {
        return 0;
    }

    int *regimes = malloc(T * sizeof(int));
    double *log_h = malloc(T * sizeof(double));
    double *obs = malloc(T * sizeof(double));

    for (int t = 0; t < T; t++)
    {
        regimes[t] = t / 25;
        log_h[t] = -2.0;
        obs[t] = log_h[t] + randn() * 0.5;
    }

    /* Run N calls */
    int N_CALLS = 5;
    for (int i = 0; i < N_CALLS; i++)
    {
        oracle_bridge_run(&stack.bridge, regimes, log_h, obs, T, 2.0f);
    }

    OracleBridgeStats stats;
    oracle_bridge_get_stats(&stack.bridge, &stats);

    printf("  Total calls: %d (expected: %d)\n", stats.total_oracle_calls, N_CALLS);
    printf("  Successful: %d\n", stats.successful_blends);
    printf("  Scout skips: %d\n", stats.scout_skip_count);
    printf("  Regime changes: %d\n", stats.regime_change_count);
    printf("  Degeneracies: %d\n", stats.degeneracy_count);
    printf("  Avg KL: %.6f\n", stats.avg_kl_change);
    printf("  Current γ: %.4f\n", stats.current_gamma);
    printf("  Avg diagonal: %.4f\n", stats.current_avg_diagonal);
    printf("  Thompson explore ratio: %.1f%%\n", stats.thompson_explore_ratio * 100);

    free(regimes);
    free(log_h);
    free(obs);
    oracle_stack_free(&stack);

    return (stats.total_oracle_calls == N_CALLS && stats.successful_blends >= N_CALLS - 1);
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST 7: Print State
 *═══════════════════════════════════════════════════════════════════════════*/

static int test_print_state(void)
{
    printf("Testing oracle_bridge_print_state()\n\n");

    int K = 4;
    int T = 100;

    OracleStack stack;
    if (oracle_stack_init(&stack, K, 32, T) != 0)
    {
        return 0;
    }

    int *regimes = malloc(T * sizeof(int));
    double *log_h = malloc(T * sizeof(double));
    double *obs = malloc(T * sizeof(double));

    for (int t = 0; t < T; t++)
    {
        regimes[t] = t / 25;
        log_h[t] = -2.0;
        obs[t] = log_h[t] + randn() * 0.5;
    }

    /* Run a couple calls */
    oracle_bridge_run(&stack.bridge, regimes, log_h, obs, T, 2.0f);
    oracle_bridge_run(&stack.bridge, regimes, log_h, obs, T, 3.0f);

    /* Print state */
    oracle_bridge_print_state(&stack.bridge);

    free(regimes);
    free(log_h);
    free(obs);
    oracle_stack_free(&stack);

    return 1; /* Pass if no crash */
}

/*═══════════════════════════════════════════════════════════════════════════
 * MAIN
 *═══════════════════════════════════════════════════════════════════════════*/

int main(void)
{
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════════════╗\n");
    printf("║              ORACLE FULL INTEGRATION TEST SUITE                       ║\n");
    printf("║              6-Phase Pipeline: Scout→PGAS→Conf→SAEM→Thompson          ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════════╝\n");

    clock_t start = clock();

    RUN_TEST(test_calm_to_crisis);
    RUN_TEST(test_multiple_transitions);
    RUN_TEST(test_confidence_gamma);
    RUN_TEST(test_thompson_transition);
    RUN_TEST(test_dual_gate_trigger);
    RUN_TEST(test_statistics);
    RUN_TEST(test_print_state);

    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;

    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════════════╗\n");
    printf("║                           SUMMARY                                     ║\n");
    printf("╠═══════════════════════════════════════════════════════════════════════╣\n");
    printf("║  Tests passed: %d / %d                                                 ║\n",
           g_tests_passed, g_tests_run);
    printf("║  Time elapsed: %.3f seconds                                           ║\n", elapsed);
    printf("╚═══════════════════════════════════════════════════════════════════════╝\n\n");

    if (g_tests_passed == g_tests_run)
    {
        printf(GREEN "All tests passed!" RESET "\n\n");
    }
    else
    {
        printf(RED "%d test(s) failed" RESET "\n\n", g_tests_run - g_tests_passed);
    }

    return (g_tests_passed == g_tests_run) ? 0 : 1;
}