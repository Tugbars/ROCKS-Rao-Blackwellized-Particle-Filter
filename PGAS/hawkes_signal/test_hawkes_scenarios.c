/**
 * @file test_hawkes_scenarios.c
 * @brief Scenario-Based Tests for Hawkes Oracle Trigger
 *
 * Tests real-world scenarios with KNOWN GROUND TRUTH:
 *   1. Regime transition detection latency
 *   2. False positive rate during calm
 *   3. Spike filtering (don't trigger on transients)
 *   4. Plateau detection (DO trigger on sustained elevation)
 *   5. Crisis clustering detection
 *
 * Success criteria from ORACLE_INTEGRATION_PLAN.md:
 *   - Detect regime changes within reasonable latency
 *   - Don't trigger during stable periods (low FP rate)
 *   - Filter transient spikes via residual check
 *   - Trigger on sustained intensity elevation
 */

#include "hawkes_integrator.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/*═══════════════════════════════════════════════════════════════════════════
 * TEST UTILITIES
 *═══════════════════════════════════════════════════════════════════════════*/

#define GREEN "\033[32m"
#define RED "\033[31m"
#define YELLOW "\033[33m"
#define RESET "\033[0m"

static uint32_t g_rng = 42;

static float randf(void)
{
    g_rng = g_rng * 1103515245 + 12345;
    return (float)(g_rng >> 16) / 65536.0f;
}

static float randn(void)
{
    float u1 = randf() + 1e-10f;
    float u2 = randf();
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
}

/*═══════════════════════════════════════════════════════════════════════════
 * SCENARIO 1: REGIME TRANSITION DETECTION
 *
 * Ground truth: Regime changes at known ticks
 * Success: Trigger within N ticks of each change
 *═══════════════════════════════════════════════════════════════════════════*/

typedef struct
{
    int change_tick;   /* When regime changed */
    int detected_tick; /* When we triggered (-1 if missed) */
    int latency;       /* Detection latency */
    float vol_before;
    float vol_after;
} RegimeChangeEvent;

static void scenario_regime_transitions(void)
{
    printf("\n" YELLOW "═══════════════════════════════════════════════════════════════" RESET "\n");
    printf(YELLOW "SCENARIO 1: REGIME TRANSITION DETECTION" RESET "\n");
    printf(YELLOW "═══════════════════════════════════════════════════════════════" RESET "\n\n");

    HawkesIntegratorConfig cfg = hawkes_integrator_config_defaults();
    cfg.warmup_ticks = 100;
    cfg.refractory_ticks = 100;
    cfg.high_water_mark = 1.5f;
    cfg.low_water_mark = 0.8f;
    cfg.min_ticks_armed = 3;
    cfg.ema_alpha = 0.01f;
    cfg.hawkes.event_threshold = 0.015f;

    HawkesIntegrator integ;
    hawkes_integrator_init(&integ, &cfg);

/* Generate data with known regime transitions */
#define TEST1_N 2000
    float returns[TEST1_N];

    /* Define regime schedule */
    typedef struct
    {
        int start;
        int end;
        float vol;
        const char *name;
    } Regime;
    Regime regimes[] = {
        {0, 300, 0.005f, "Calm"},
        {300, 500, 0.025f, "Mild stress"}, /* Transition 1: t=300 */
        {500, 700, 0.005f, "Calm"},
        {700, 900, 0.040f, "High stress"}, /* Transition 2: t=700 */
        {900, 1200, 0.005f, "Calm"},
        {1200, 1500, 0.060f, "Crisis"}, /* Transition 3: t=1200 */
        {1500, 2000, 0.010f, "Recovery"},
    };
    int n_regimes = sizeof(regimes) / sizeof(regimes[0]);

    /* Generate returns */
    for (int t = 0; t < TEST1_N; t++)
    {
        float vol = 0.005f;
        for (int r = 0; r < n_regimes; r++)
        {
            if (t >= regimes[r].start && t < regimes[r].end)
            {
                vol = regimes[r].vol;
                break;
            }
        }
        returns[t] = randn() * vol;
    }

    /* Track detections */
    int transition_ticks[] = {300, 700, 1200}; /* Known transitions to HIGH vol */
    int n_transitions = 3;
    RegimeChangeEvent detections[3] = {
        {300, -1, -1, 0.005f, 0.025f},
        {700, -1, -1, 0.005f, 0.040f},
        {1200, -1, -1, 0.005f, 0.060f},
    };

    int current_detection_idx = 0;
    int total_triggers = 0;

    /* Run simulation */
    for (int t = 0; t < TEST1_N; t++)
    {
        HawkesIntegratorResult res = hawkes_integrator_update(&integ, (float)t, returns[t]);

        if (res.should_trigger)
        {
            total_triggers++;

            /* Check if this corresponds to a known transition */
            for (int i = 0; i < n_transitions; i++)
            {
                if (detections[i].detected_tick == -1 &&
                    t >= transition_ticks[i] &&
                    t < transition_ticks[i] + 200)
                { /* Within 200 ticks */
                    detections[i].detected_tick = t;
                    detections[i].latency = t - transition_ticks[i];
                    break;
                }
            }

            printf("  Trigger at t=%d (surprise=%.2fσ, λ=%.4f, state=%d)\n",
                   t, res.surprise_sigma, res.integrated_intensity, res.state);
        }
    }

    /* Report results */
    printf("\n  REGIME TRANSITIONS:\n");
    printf("  %-10s %-10s %-10s %-15s\n", "Change@", "Detected@", "Latency", "Vol Change");
    printf("  ──────────────────────────────────────────────────\n");

    int detected_count = 0;
    int total_latency = 0;

    for (int i = 0; i < n_transitions; i++)
    {
        const char *status;
        if (detections[i].detected_tick >= 0)
        {
            status = GREEN "✓" RESET;
            detected_count++;
            total_latency += detections[i].latency;
        }
        else
        {
            status = RED "✗ MISSED" RESET;
        }

        printf("  t=%-7d t=%-7d %-10d %.3f→%.3f %s\n",
               detections[i].change_tick,
               detections[i].detected_tick,
               detections[i].latency,
               detections[i].vol_before,
               detections[i].vol_after,
               status);
    }

    printf("\n  SUMMARY:\n");
    printf("    Transitions detected: %d/%d (%.0f%%)\n",
           detected_count, n_transitions, 100.0f * detected_count / n_transitions);
    printf("    Average latency: %.1f ticks\n",
           detected_count > 0 ? (float)total_latency / detected_count : 0.0f);
    printf("    Total triggers: %d\n", total_triggers);
    printf("    False triggers: %d\n", total_triggers - detected_count);

    if (detected_count >= 2 && total_triggers <= detected_count + 2)
    {
        printf("\n  " GREEN "✓ SCENARIO PASSED" RESET "\n");
    }
    else
    {
        printf("\n  " RED "✗ SCENARIO NEEDS TUNING" RESET "\n");
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * SCENARIO 2: FALSE POSITIVE RATE (CALM MARKET)
 *
 * Ground truth: No regime changes, constant low volatility
 * Success: Zero or very few triggers
 *═══════════════════════════════════════════════════════════════════════════*/

static void scenario_false_positive_rate(void)
{
    printf("\n" YELLOW "═══════════════════════════════════════════════════════════════" RESET "\n");
    printf(YELLOW "SCENARIO 2: FALSE POSITIVE RATE (CALM MARKET)" RESET "\n");
    printf(YELLOW "═══════════════════════════════════════════════════════════════" RESET "\n\n");

    HawkesIntegratorConfig cfg = hawkes_integrator_config_defaults();
    cfg.warmup_ticks = 100;
    cfg.refractory_ticks = 200;

    HawkesIntegrator integ;
    hawkes_integrator_init(&integ, &cfg);

/* Generate 5000 ticks of calm market */
#define TEST2_N 5000
    float vol = 0.005f; /* Low volatility throughout */

    int triggers = 0;
    int trigger_ticks[100];

    for (int t = 0; t < TEST2_N; t++)
    {
        float ret = randn() * vol;
        HawkesIntegratorResult res = hawkes_integrator_update(&integ, (float)t, ret);

        if (res.should_trigger && triggers < 100)
        {
            trigger_ticks[triggers] = t;
            triggers++;
            printf("  Spurious trigger at t=%d (surprise=%.2fσ)\n", t, res.surprise_sigma);
        }
    }

    float fp_rate = (float)triggers / ((TEST2_N - cfg.warmup_ticks) / 1000.0f);

    printf("\n  SUMMARY:\n");
    printf("    Duration: %d ticks (%.1f after warmup)\n", TEST2_N, (float)(TEST2_N - cfg.warmup_ticks));
    printf("    Volatility: constant %.2f%%\n", vol * 100);
    printf("    Triggers: %d\n", triggers);
    printf("    False positive rate: %.2f per 1000 ticks\n", fp_rate);

    if (triggers <= 2)
    {
        printf("\n  " GREEN "✓ SCENARIO PASSED (FP rate acceptable)" RESET "\n");
    }
    else
    {
        printf("\n  " RED "✗ TOO MANY FALSE POSITIVES" RESET "\n");
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * SCENARIO 3: SPIKE FILTERING
 *
 * Ground truth: Single large spike, then calm
 * Success: Should NOT trigger (residual decays, spike is transient)
 *═══════════════════════════════════════════════════════════════════════════*/

static void scenario_spike_filtering(void)
{
    printf("\n" YELLOW "═══════════════════════════════════════════════════════════════" RESET "\n");
    printf(YELLOW "SCENARIO 3: SPIKE FILTERING (TRANSIENT)" RESET "\n");
    printf(YELLOW "═══════════════════════════════════════════════════════════════" RESET "\n\n");

    HawkesIntegratorConfig cfg = hawkes_integrator_config_defaults();
    cfg.warmup_ticks = 100;
    cfg.refractory_ticks = 100;
    cfg.min_ticks_armed = 5;        /* Need sustained elevation */
    cfg.residual_threshold = 0.02f; /* Residual must be positive */
    cfg.ema_alpha = 0.01f;

    HawkesIntegrator integ;
    hawkes_integrator_init(&integ, &cfg);

#define TEST3_N 500
    float returns[TEST3_N];

    /* Calm market with single spike at t=200 */
    for (int t = 0; t < TEST3_N; t++)
    {
        returns[t] = randn() * 0.005f;
    }

    /* Single spike event */
    returns[200] = 0.10f; /* 10% spike */
    returns[201] = 0.05f; /* Follow-through */
    returns[202] = 0.03f; /* Decay */
    /* Then back to calm */

    int triggers = 0;
    float max_surprise = 0;
    float max_residual = 0;

    for (int t = 0; t < TEST3_N; t++)
    {
        HawkesIntegratorResult res = hawkes_integrator_update(&integ, (float)t, returns[t]);

        if (t >= 200 && t < 250)
        {
            if (res.surprise_sigma > max_surprise)
                max_surprise = res.surprise_sigma;
            if (res.residual > max_residual)
                max_residual = res.residual;

            printf("  t=%d: λ=%.4f, surprise=%.2fσ, residual=%.4f, state=%d\n",
                   t, res.integrated_intensity, res.surprise_sigma, res.residual, res.state);
        }

        if (res.should_trigger)
        {
            triggers++;
            printf("  " RED "TRIGGER at t=%d" RESET "\n", t);
        }
    }

    printf("\n  SUMMARY:\n");
    printf("    Spike at t=200 (10%% return)\n");
    printf("    Max surprise: %.2fσ\n", max_surprise);
    printf("    Max residual: %.4f\n", max_residual);
    printf("    Triggers: %d\n", triggers);

    if (triggers == 0)
    {
        printf("\n  " GREEN "✓ SCENARIO PASSED (spike filtered)" RESET "\n");
    }
    else
    {
        printf("\n  " RED "✗ SPIKE SHOULD HAVE BEEN FILTERED" RESET "\n");
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * SCENARIO 4: PLATEAU DETECTION
 *
 * Ground truth: Sustained high volatility period
 * Success: SHOULD trigger during plateau
 *═══════════════════════════════════════════════════════════════════════════*/

static void scenario_plateau_detection(void)
{
    printf("\n" YELLOW "═══════════════════════════════════════════════════════════════" RESET "\n");
    printf(YELLOW "SCENARIO 4: PLATEAU DETECTION (SUSTAINED ELEVATION)" RESET "\n");
    printf(YELLOW "═══════════════════════════════════════════════════════════════" RESET "\n\n");

    HawkesIntegratorConfig cfg = hawkes_integrator_config_defaults();
    cfg.warmup_ticks = 100;
    cfg.refractory_ticks = 100;
    cfg.high_water_mark = 1.5f;
    cfg.low_water_mark = 0.8f;
    cfg.min_ticks_armed = 3;
    cfg.residual_threshold = 0.01f;
    cfg.ema_alpha = 0.01f;
    cfg.hawkes.event_threshold = 0.015f;

    HawkesIntegrator integ;
    hawkes_integrator_init(&integ, &cfg);

#define TEST4_N 600
    float returns[TEST4_N];

    /* Calm → Plateau → Calm */
    for (int t = 0; t < TEST4_N; t++)
    {
        float vol;
        if (t < 200)
        {
            vol = 0.005f; /* Calm */
        }
        else if (t < 400)
        {
            vol = 0.035f; /* PLATEAU - sustained high vol */
        }
        else
        {
            vol = 0.005f; /* Calm again */
        }
        returns[t] = randn() * vol;
    }

    int triggers_during_plateau = 0;
    int triggers_outside = 0;
    int first_trigger_tick = -1;

    for (int t = 0; t < TEST4_N; t++)
    {
        HawkesIntegratorResult res = hawkes_integrator_update(&integ, (float)t, returns[t]);

        if (res.should_trigger)
        {
            if (t >= 200 && t < 400)
            {
                triggers_during_plateau++;
                if (first_trigger_tick < 0)
                    first_trigger_tick = t;
                printf("  " GREEN "Trigger during plateau at t=%d" RESET " (surprise=%.2fσ)\n",
                       t, res.surprise_sigma);
            }
            else
            {
                triggers_outside++;
                printf("  " YELLOW "Trigger outside plateau at t=%d" RESET "\n", t);
            }
        }
    }

    printf("\n  SUMMARY:\n");
    printf("    Plateau period: t=200-400\n");
    printf("    Triggers during plateau: %d\n", triggers_during_plateau);
    printf("    Triggers outside: %d\n", triggers_outside);
    if (first_trigger_tick >= 0)
    {
        printf("    Detection latency: %d ticks\n", first_trigger_tick - 200);
    }

    if (triggers_during_plateau >= 1)
    {
        printf("\n  " GREEN "✓ SCENARIO PASSED (plateau detected)" RESET "\n");
    }
    else
    {
        printf("\n  " RED "✗ PLATEAU SHOULD HAVE TRIGGERED" RESET "\n");
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * SCENARIO 5: CRISIS CLUSTERING
 *
 * Ground truth: Hawkes-like self-exciting process (returns cluster)
 * Success: Should detect the crisis period
 *═══════════════════════════════════════════════════════════════════════════*/

static void scenario_crisis_clustering(void)
{
    printf("\n" YELLOW "═══════════════════════════════════════════════════════════════" RESET "\n");
    printf(YELLOW "SCENARIO 5: CRISIS CLUSTERING (SELF-EXCITING)" RESET "\n");
    printf(YELLOW "═══════════════════════════════════════════════════════════════" RESET "\n\n");

    HawkesIntegratorConfig cfg = hawkes_integrator_config_defaults();
    cfg.warmup_ticks = 100;
    cfg.refractory_ticks = 100;
    cfg.high_water_mark = 1.5f;
    cfg.low_water_mark = 0.8f;
    cfg.ema_alpha = 0.01f;
    cfg.hawkes.event_threshold = 0.015f;

    HawkesIntegrator integ;
    hawkes_integrator_init(&integ, &cfg);

#define TEST5_N 800
    float returns[TEST5_N];

    /* Generate Hawkes-like clustering */
    float vol = 0.005f;
    float base_vol = 0.005f;
    int crisis_start = 300;

    for (int t = 0; t < TEST5_N; t++)
    {
        if (t >= crisis_start && t < crisis_start + 200)
        {
            /* Crisis period: returns cluster */
            float ret = randn() * vol;
            returns[t] = ret;

            /* Self-excitation: large returns boost vol */
            if (fabsf(ret) > 0.015f)
            {
                vol += 0.005f;
            }
            /* Decay toward elevated base */
            vol = 0.015f + (vol - 0.015f) * 0.95f;
        }
        else
        {
            vol = base_vol;
            returns[t] = randn() * vol;
        }
    }

    int triggers_during_crisis = 0;
    int triggers_outside = 0;

    for (int t = 0; t < TEST5_N; t++)
    {
        HawkesIntegratorResult res = hawkes_integrator_update(&integ, (float)t, returns[t]);

        if (res.should_trigger)
        {
            if (t >= crisis_start && t < crisis_start + 200)
            {
                triggers_during_crisis++;
                printf("  " GREEN "Trigger during crisis at t=%d" RESET " (λ=%.4f)\n",
                       t, res.integrated_intensity);
            }
            else
            {
                triggers_outside++;
                printf("  " YELLOW "Trigger outside crisis at t=%d" RESET "\n", t);
            }
        }
    }

    printf("\n  SUMMARY:\n");
    printf("    Crisis period: t=%d-%d (self-exciting)\n", crisis_start, crisis_start + 200);
    printf("    Triggers during crisis: %d\n", triggers_during_crisis);
    printf("    Triggers outside: %d\n", triggers_outside);

    if (triggers_during_crisis >= 1)
    {
        printf("\n  " GREEN "✓ SCENARIO PASSED (crisis detected)" RESET "\n");
    }
    else
    {
        printf("\n  " RED "✗ CRISIS SHOULD HAVE TRIGGERED" RESET "\n");
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * SCENARIO 6: ABSOLUTE PANIC TEST
 *
 * Ground truth: Extremely high intensity (beyond σ-based threshold)
 * Success: Should trigger via panic path regardless of hysteresis
 *═══════════════════════════════════════════════════════════════════════════*/

static void scenario_absolute_panic(void)
{
    printf("\n" YELLOW "═══════════════════════════════════════════════════════════════" RESET "\n");
    printf(YELLOW "SCENARIO 6: ABSOLUTE PANIC (EXTREME INTENSITY)" RESET "\n");
    printf(YELLOW "═══════════════════════════════════════════════════════════════" RESET "\n\n");

    HawkesIntegratorConfig cfg = hawkes_integrator_config_defaults();
    cfg.warmup_ticks = 50;
    cfg.refractory_ticks = 100;
    cfg.high_water_mark = 100.0f; /* Disable normal triggers */
    cfg.low_water_mark = 50.0f;
    cfg.absolute_panic_intensity = 0.15f; /* Panic if avg_λ > 0.15 */
    cfg.instant_spike_multiplier = 1.5f;  /* Or instant_λ > 0.225 */
    cfg.use_absolute_panic = true;
    cfg.hawkes.event_threshold = 0.01f;
    cfg.hawkes.alpha = 0.7f;
    cfg.hawkes.beta = 0.1f;

    HawkesIntegrator integ;
    hawkes_integrator_init(&integ, &cfg);

#define TEST6_N 300
    float returns[TEST6_N];

    /* Calm → EXTREME → Calm */
    for (int t = 0; t < TEST6_N; t++)
    {
        if (t >= 100 && t < 150)
        {
            returns[t] = 0.15f * (randf() > 0.5f ? 1 : -1); /* Extreme returns */
        }
        else
        {
            returns[t] = randn() * 0.003f;
        }
    }

    int panic_triggers = 0;
    int normal_triggers = 0;

    for (int t = 0; t < TEST6_N; t++)
    {
        HawkesIntegratorResult res = hawkes_integrator_update(&integ, (float)t, returns[t]);

        if (t >= 100 && t < 160)
        {
            printf("  t=%d: λ=%.4f, avg_λ=%.4f, panic_thresh=%.4f\n",
                   t, res.instantaneous_intensity, res.integrated_intensity,
                   cfg.absolute_panic_intensity);
        }

        if (res.should_trigger)
        {
            if (res.triggered_by_panic)
            {
                panic_triggers++;
                printf("  " GREEN "PANIC trigger at t=%d" RESET "\n", t);
            }
            else
            {
                normal_triggers++;
                printf("  Normal trigger at t=%d\n", t);
            }
        }
    }

    printf("\n  SUMMARY:\n");
    printf("    Extreme period: t=100-150\n");
    printf("    Panic triggers: %d\n", panic_triggers);
    printf("    Normal triggers: %d\n", normal_triggers);

    if (panic_triggers >= 1)
    {
        printf("\n  " GREEN "✓ SCENARIO PASSED (panic triggered)" RESET "\n");
    }
    else
    {
        printf("\n  " RED "✗ PANIC SHOULD HAVE TRIGGERED" RESET "\n");
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * MAIN
 *═══════════════════════════════════════════════════════════════════════════*/

int main(void)
{
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║     HAWKES ORACLE TRIGGER - SCENARIO-BASED TESTS             ║\n");
    printf("║     Testing according to ORACLE_INTEGRATION_PLAN.md          ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n");

    scenario_regime_transitions();
    scenario_false_positive_rate();
    scenario_spike_filtering();
    scenario_plateau_detection();
    scenario_crisis_clustering();
    scenario_absolute_panic();

    printf("\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("                    ALL SCENARIOS COMPLETE                     \n");
    printf("═══════════════════════════════════════════════════════════════\n\n");

    return 0;
}