/**
 * @file test_rbpf_trajectory.c
 * @brief Tests for RBPF Trajectory Buffer
 *
 * Tests:
 *   - Initialization and configuration
 *   - Circular buffer recording
 *   - Extraction in chronological order
 *   - Tempering (5% flips)
 *   - Buffer wraparound
 *   - Double precision extraction (for PGAS)
 *
 * Compile:
 *   gcc -O2 -Wall test_rbpf_trajectory.c rbpf_trajectory.c -lm -o test_traj
 */

#include "rbpf_trajectory.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define GREEN "\033[32m"
#define RED "\033[31m"
#define YELLOW "\033[33m"
#define CYAN "\033[36m"
#define RESET "\033[0m"

static int g_tests_passed = 0;
static int g_tests_run = 0;

#define RUN_TEST(fn)                                          \
    do                                                        \
    {                                                         \
        g_tests_run++;                                        \
        printf("\n" CYAN "═══ TEST: %s ═══" RESET "\n", #fn); \
        if (fn())                                             \
        {                                                     \
            g_tests_passed++;                                 \
            printf(GREEN "✓ PASS" RESET "\n");                \
        }                                                     \
        else                                                  \
        {                                                     \
            printf(RED "✗ FAIL" RESET "\n");                  \
        }                                                     \
    } while (0)

/*═══════════════════════════════════════════════════════════════════════════
 * TEST: Configuration Defaults
 *═══════════════════════════════════════════════════════════════════════════*/

static int test_config_defaults(void)
{
    printf("Testing configuration defaults\n\n");

    RBPFTrajectoryConfig cfg = rbpf_trajectory_config_defaults(500, 4);

    printf("  T_max:        %d (expected: 500)\n", cfg.T_max);
    printf("  n_regimes:    %d (expected: 4)\n", cfg.n_regimes);
    printf("  temper_prob:  %.2f (expected: 0.05)\n", cfg.temper_prob);

    bool ok = (cfg.T_max == 500) &&
              (cfg.n_regimes == 4) &&
              (fabsf(cfg.temper_prob - 0.05f) < 0.001f);

    return ok;
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST: Initialization
 *═══════════════════════════════════════════════════════════════════════════*/

static int test_initialization(void)
{
    printf("Testing initialization\n\n");

    RBPFTrajectory traj;
    int ret = rbpf_trajectory_init_simple(&traj, 100, 4);

    printf("  Init return:    %d (expected: 0)\n", ret);
    printf("  Initialized:    %s\n", traj.initialized ? "true" : "false");
    printf("  Count:          %d (expected: 0)\n", traj.count);
    printf("  Length:         %d (expected: 0)\n", rbpf_trajectory_length(&traj));
    printf("  Ready(50):      %s (expected: NO)\n",
           rbpf_trajectory_ready(&traj, 50) ? "YES" : "NO");

    bool ok = (ret == 0) &&
              traj.initialized &&
              (traj.count == 0) &&
              !rbpf_trajectory_ready(&traj, 50);

    rbpf_trajectory_free(&traj);
    return ok;
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST: Recording
 *═══════════════════════════════════════════════════════════════════════════*/

static int test_recording(void)
{
    printf("Testing recording\n\n");

    RBPFTrajectory traj;
    rbpf_trajectory_init_simple(&traj, 100, 4);

    /* Record 10 entries */
    for (int t = 0; t < 10; t++)
    {
        int regime = t % 4;
        float h = -3.0f + (float)t * 0.1f;
        rbpf_trajectory_record(&traj, regime, h);
    }

    printf("  After 10 records:\n");
    printf("    Count:        %d (expected: 10)\n", traj.count);
    printf("    Total ticks:  %lld (expected: 10)\n", (long long)traj.total_ticks);
    printf("    Last regime:  %d (expected: 1)\n", rbpf_trajectory_last_regime(&traj));
    printf("    Last h:       %.2f (expected: -2.10)\n", rbpf_trajectory_last_h(&traj));
    printf("    Ready(10):    %s\n", rbpf_trajectory_ready(&traj, 10) ? "YES" : "NO");
    printf("    Ready(50):    %s\n", rbpf_trajectory_ready(&traj, 50) ? "YES" : "NO");

    bool ok = (traj.count == 10) &&
              (traj.total_ticks == 10) &&
              (rbpf_trajectory_last_regime(&traj) == 1) && /* 9 % 4 = 1 */
              (fabsf(rbpf_trajectory_last_h(&traj) - (-2.1f)) < 0.01f) &&
              rbpf_trajectory_ready(&traj, 10) &&
              !rbpf_trajectory_ready(&traj, 50);

    rbpf_trajectory_free(&traj);
    return ok;
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST: Extraction (Chronological Order)
 *═══════════════════════════════════════════════════════════════════════════*/

static int test_extraction(void)
{
    printf("Testing extraction in chronological order\n\n");

    RBPFTrajectory traj;
    rbpf_trajectory_init_simple(&traj, 100, 4);

    /* Record 20 entries with predictable pattern */
    for (int t = 0; t < 20; t++)
    {
        rbpf_trajectory_record(&traj, t % 4, (float)t);
    }

    /* Extract last 10 */
    int regimes[10];
    float h[10];
    RBPFTrajectoryExtractResult result = rbpf_trajectory_extract(&traj, regimes, h, 10);

    printf("  Extracted T:    %d (expected: 10)\n", result.T);
    printf("  Start tick:     %lld (expected: 10)\n", (long long)result.start_tick);
    printf("  End tick:       %lld (expected: 19)\n", (long long)result.end_tick);
    printf("  Fill ratio:     %.2f (expected: 0.20)\n", result.fill_ratio);

    /* Check chronological order: should be ticks 10-19 */
    printf("\n  Extracted values:\n");
    printf("    t=0: regime=%d, h=%.0f (expected: 2, 10)\n", regimes[0], h[0]);
    printf("    t=9: regime=%d, h=%.0f (expected: 3, 19)\n", regimes[9], h[9]);

    bool order_ok = true;
    for (int i = 0; i < 10; i++)
    {
        int expected_regime = (10 + i) % 4;
        float expected_h = 10.0f + (float)i;
        if (regimes[i] != expected_regime || fabsf(h[i] - expected_h) > 0.01f)
        {
            printf("    " RED "MISMATCH at i=%d: got (%d, %.0f), expected (%d, %.0f)" RESET "\n",
                   i, regimes[i], h[i], expected_regime, expected_h);
            order_ok = false;
        }
    }

    bool ok = (result.T == 10) &&
              (result.start_tick == 10) &&
              (result.end_tick == 19) &&
              order_ok;

    rbpf_trajectory_free(&traj);
    return ok;
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST: Circular Buffer Wraparound
 *═══════════════════════════════════════════════════════════════════════════*/

static int test_wraparound(void)
{
    printf("Testing circular buffer wraparound\n\n");

    RBPFTrajectory traj;
    rbpf_trajectory_init_simple(&traj, 20, 4); /* Small buffer */

    /* Record 50 entries (wraps around 2.5 times) */
    for (int t = 0; t < 50; t++)
    {
        rbpf_trajectory_record(&traj, t % 4, (float)t);
    }

    printf("  After 50 records into buffer of 20:\n");
    printf("    Count:        %d (expected: 20 - buffer full)\n", traj.count);
    printf("    Total ticks:  %lld (expected: 50)\n", (long long)traj.total_ticks);
    printf("    Last regime:  %d (expected: 1)\n", rbpf_trajectory_last_regime(&traj));
    printf("    Last h:       %.0f (expected: 49)\n", rbpf_trajectory_last_h(&traj));

    /* Extract all 20 - should be ticks 30-49 */
    int regimes[20];
    float h[20];
    RBPFTrajectoryExtractResult result = rbpf_trajectory_extract(&traj, regimes, h, 20);

    printf("\n  Extracted (should be ticks 30-49):\n");
    printf("    T:            %d\n", result.T);
    printf("    Start tick:   %lld (expected: 30)\n", (long long)result.start_tick);
    printf("    End tick:     %lld (expected: 49)\n", (long long)result.end_tick);
    printf("    First h:      %.0f (expected: 30)\n", h[0]);
    printf("    Last h:       %.0f (expected: 49)\n", h[19]);

    bool order_ok = true;
    for (int i = 0; i < 20; i++)
    {
        float expected_h = 30.0f + (float)i;
        if (fabsf(h[i] - expected_h) > 0.01f)
        {
            printf("    " RED "MISMATCH at i=%d: got h=%.0f, expected h=%.0f" RESET "\n",
                   i, h[i], expected_h);
            order_ok = false;
        }
    }

    bool ok = (traj.count == 20) &&
              (traj.total_ticks == 50) &&
              (result.start_tick == 30) &&
              (result.end_tick == 49) &&
              order_ok;

    rbpf_trajectory_free(&traj);
    return ok;
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST: Tempering
 *═══════════════════════════════════════════════════════════════════════════*/

static int test_tempering(void)
{
    printf("Testing tempering (5%% flips)\n\n");

    RBPFTrajectory traj;
    rbpf_trajectory_init_simple(&traj, 100, 4);

    /* Record 100 entries all in regime 0 */
    for (int t = 0; t < 100; t++)
    {
        rbpf_trajectory_record(&traj, 0, -3.0f);
    }

    /* Extract */
    int regimes[100];
    float h[100];
    rbpf_trajectory_extract(&traj, regimes, h, 100);

    /* Count regime 0 before tempering */
    int before_r0 = 0;
    for (int t = 0; t < 100; t++)
    {
        if (regimes[t] == 0)
            before_r0++;
    }
    printf("  Before tempering: %d/%d in regime 0\n", before_r0, 100);

    /* Temper with 5% flip probability */
    int flips = rbpf_trajectory_temper(&traj, regimes, 100, 0.05f);

    /* Count regime 0 after tempering */
    int after_r0 = 0;
    for (int t = 0; t < 100; t++)
    {
        if (regimes[t] == 0)
            after_r0++;
    }

    printf("  After tempering:  %d/%d in regime 0\n", after_r0, 100);
    printf("  Flips applied:    %d\n", flips);
    printf("  Flip rate:        %.1f%% (expected: ~5%%)\n",
           (float)flips / 100.0f * 100.0f);

    /* With 5% probability and 100 trials, expect 3-10 flips typically */
    bool flip_count_reasonable = (flips >= 1 && flips <= 20);
    bool flip_rate_ok = (flips == (100 - after_r0));

    printf("  Flips match delta: %s\n", flip_rate_ok ? "YES" : "NO");

    rbpf_trajectory_free(&traj);
    return flip_count_reasonable && flip_rate_ok;
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST: Tempering Distribution
 *═══════════════════════════════════════════════════════════════════════════*/

static int test_tempering_distribution(void)
{
    printf("Testing tempering targets different regimes uniformly\n\n");

    RBPFTrajectory traj;
    rbpf_trajectory_init_simple(&traj, 1000, 4);

    /* Record 1000 entries all in regime 0 */
    for (int t = 0; t < 1000; t++)
    {
        rbpf_trajectory_record(&traj, 0, -3.0f);
    }

    /* Extract and temper with 30% (high for testing distribution) */
    int regimes[1000];
    float h[1000];
    rbpf_trajectory_extract(&traj, regimes, h, 1000);

    int flips = rbpf_trajectory_temper(&traj, regimes, 1000, 0.30f);

    /* Count each regime */
    int counts[4] = {0};
    for (int t = 0; t < 1000; t++)
    {
        counts[regimes[t]]++;
    }

    printf("  After 30%% tempering:\n");
    printf("    Flips:      %d\n", flips);
    printf("    Regime 0:   %d (should be ~700)\n", counts[0]);
    printf("    Regime 1:   %d (should be ~100)\n", counts[1]);
    printf("    Regime 2:   %d (should be ~100)\n", counts[2]);
    printf("    Regime 3:   %d (should be ~100)\n", counts[3]);

    /* Flipped entries should be roughly uniform over regimes 1,2,3 */
    /* Each of regimes 1,2,3 should get ~1/3 of the flips */
    int non_zero = counts[1] + counts[2] + counts[3];

    bool distribution_ok = (non_zero >= 200 && non_zero <= 400); /* ~300 expected */

    /* Each of 1,2,3 should have roughly equal counts */
    int min_non = counts[1];
    int max_non = counts[1];
    for (int r = 2; r <= 3; r++)
    {
        if (counts[r] < min_non)
            min_non = counts[r];
        if (counts[r] > max_non)
            max_non = counts[r];
    }

    printf("    Non-zero range: [%d, %d]\n", min_non, max_non);

    /* Allow 3x variation (pretty loose) */
    bool uniform_ok = (max_non < 3 * min_non + 10);

    rbpf_trajectory_free(&traj);
    return distribution_ok && uniform_ok;
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST: Double Precision Extraction
 *═══════════════════════════════════════════════════════════════════════════*/

static int test_double_extraction(void)
{
    printf("Testing double precision extraction (for PGAS)\n\n");

    RBPFTrajectory traj;
    rbpf_trajectory_init_simple(&traj, 100, 4);

    /* Record with values that might lose precision */
    for (int t = 0; t < 20; t++)
    {
        rbpf_trajectory_record(&traj, t % 4, -3.14159265f + (float)t * 0.001f);
    }

    /* Extract to double */
    int regimes[20];
    double h_double[20];
    RBPFTrajectoryExtractResult result = rbpf_trajectory_extract_double(&traj, regimes, h_double, 20);

    printf("  Extracted T:    %d\n", result.T);
    printf("  First h:        %.8f\n", h_double[0]);
    printf("  Last h:         %.8f\n", h_double[19]);

    /* Verify values */
    bool ok = (result.T == 20) &&
              (fabs(h_double[0] - (-3.14159265)) < 0.0001) &&
              (fabs(h_double[19] - (-3.14159265 + 0.019)) < 0.0001);

    rbpf_trajectory_free(&traj);
    return ok;
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST: Regime Distribution
 *═══════════════════════════════════════════════════════════════════════════*/

static int test_regime_distribution(void)
{
    printf("Testing regime distribution computation\n\n");

    RBPFTrajectory traj;
    rbpf_trajectory_init_simple(&traj, 100, 4);

    /* Record 100 entries: 40 in R0, 30 in R1, 20 in R2, 10 in R3 */
    for (int t = 0; t < 40; t++)
        rbpf_trajectory_record(&traj, 0, -3.0f);
    for (int t = 0; t < 30; t++)
        rbpf_trajectory_record(&traj, 1, -3.0f);
    for (int t = 0; t < 20; t++)
        rbpf_trajectory_record(&traj, 2, -3.0f);
    for (int t = 0; t < 10; t++)
        rbpf_trajectory_record(&traj, 3, -3.0f);

    float probs[4];
    rbpf_trajectory_regime_distribution(&traj, probs);

    printf("  Expected: [40%%, 30%%, 20%%, 10%%]\n");
    printf("  Got:      [%.0f%%, %.0f%%, %.0f%%, %.0f%%]\n",
           probs[0] * 100, probs[1] * 100, probs[2] * 100, probs[3] * 100);

    bool ok = (fabsf(probs[0] - 0.40f) < 0.01f) &&
              (fabsf(probs[1] - 0.30f) < 0.01f) &&
              (fabsf(probs[2] - 0.20f) < 0.01f) &&
              (fabsf(probs[3] - 0.10f) < 0.01f);

    rbpf_trajectory_free(&traj);
    return ok;
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST: Reset
 *═══════════════════════════════════════════════════════════════════════════*/

static int test_reset(void)
{
    printf("Testing reset\n\n");

    RBPFTrajectory traj;
    rbpf_trajectory_init_simple(&traj, 100, 4);

    /* Record some data */
    for (int t = 0; t < 50; t++)
    {
        rbpf_trajectory_record(&traj, t % 4, (float)t);
    }

    printf("  Before reset:\n");
    printf("    Count:        %d\n", traj.count);
    printf("    Total ticks:  %lld\n", (long long)traj.total_ticks);

    /* Reset */
    rbpf_trajectory_reset(&traj);

    printf("  After reset:\n");
    printf("    Count:        %d (expected: 0)\n", traj.count);
    printf("    Total ticks:  %lld (expected: 0)\n", (long long)traj.total_ticks);
    printf("    Length:       %d (expected: 0)\n", rbpf_trajectory_length(&traj));

    bool ok = (traj.count == 0) &&
              (traj.total_ticks == 0) &&
              (rbpf_trajectory_length(&traj) == 0);

    rbpf_trajectory_free(&traj);
    return ok;
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST: Diagnostics
 *═══════════════════════════════════════════════════════════════════════════*/

static int test_diagnostics(void)
{
    printf("Testing diagnostic output\n\n");

    RBPFTrajectory traj;
    rbpf_trajectory_init_simple(&traj, 50, 4);

    /* Record some data */
    for (int t = 0; t < 30; t++)
    {
        rbpf_trajectory_record(&traj, t % 4, -3.0f + (float)t * 0.1f);
    }

    rbpf_trajectory_print_state(&traj);
    printf("\n");
    rbpf_trajectory_print_tail(&traj, 5);

    rbpf_trajectory_free(&traj);
    return 1; /* Visual inspection */
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST: Thread-Safe Snapshot
 *═══════════════════════════════════════════════════════════════════════════*/

static int test_snapshot(void)
{
    printf("Testing thread-safe snapshot API\n\n");

    RBPFTrajectory traj;
    rbpf_trajectory_init_simple(&traj, 100, 4);

    /* Record 50 entries */
    for (int t = 0; t < 50; t++)
    {
        rbpf_trajectory_record(&traj, t % 4, (float)t);
    }

    /* Allocate and take snapshot */
    RBPFTrajectorySnapshot snap;
    int alloc_ret = rbpf_trajectory_snapshot_alloc(&snap, 100, 4);
    printf("  Snapshot alloc:   %s\n", alloc_ret == 0 ? "OK" : "FAILED");

    int snap_ret = rbpf_trajectory_snapshot(&traj, &snap, 3);
    printf("  Snapshot take:    %s\n", snap_ret == 0 ? "OK" : "FAILED");
    printf("  Snapshot valid:   %s\n", snap.valid ? "YES" : "NO");
    printf("  Snapshot count:   %d\n", snap.count);
    printf("  Snapshot head:    %d\n", snap.head);

    /* Extract from snapshot */
    int regimes[20];
    float h[20];
    RBPFTrajectoryExtractResult result = rbpf_trajectory_extract_from_snapshot(
        &snap, regimes, h, 20);

    printf("\n  Extract from snapshot:\n");
    printf("    T:            %d (expected: 20)\n", result.T);
    printf("    Start tick:   %lld (expected: 30)\n", (long long)result.start_tick);
    printf("    End tick:     %lld (expected: 49)\n", (long long)result.end_tick);
    printf("    First h:      %.0f (expected: 30)\n", h[0]);
    printf("    Last h:       %.0f (expected: 49)\n", h[19]);

    /* Verify data */
    bool data_ok = true;
    for (int i = 0; i < 20; i++)
    {
        float expected_h = 30.0f + (float)i;
        if (fabsf(h[i] - expected_h) > 0.01f)
        {
            data_ok = false;
            break;
        }
    }
    printf("    Data correct: %s\n", data_ok ? "YES" : "NO");

    bool was_valid = snap.valid;
    rbpf_trajectory_snapshot_free(&snap);
    rbpf_trajectory_free(&traj);

    return (alloc_ret == 0) && (snap_ret == 0) && was_valid &&
           (result.T == 20) && data_ok;
}

/*═══════════════════════════════════════════════════════════════════════════
 * MAIN
 *═══════════════════════════════════════════════════════════════════════════*/

int main(void)
{
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════════════╗\n");
    printf("║              RBPF TRAJECTORY BUFFER TEST SUITE                        ║\n");
    printf("║              Reference Path for PGAS Oracle                           ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════════╝\n");

    RUN_TEST(test_config_defaults);
    RUN_TEST(test_initialization);
    RUN_TEST(test_recording);
    RUN_TEST(test_extraction);
    RUN_TEST(test_wraparound);
    RUN_TEST(test_tempering);
    RUN_TEST(test_tempering_distribution);
    RUN_TEST(test_double_extraction);
    RUN_TEST(test_regime_distribution);
    RUN_TEST(test_reset);
    RUN_TEST(test_diagnostics);
    RUN_TEST(test_snapshot);

    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════════════╗\n");
    printf("║                           SUMMARY                                     ║\n");
    printf("╠═══════════════════════════════════════════════════════════════════════╣\n");
    printf("║  Tests passed: %2d / %2d                                                ║\n",
           g_tests_passed, g_tests_run);
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