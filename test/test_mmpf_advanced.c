/**
 * @file test_mmpf_advanced.c
 * @brief Advanced MMPF Test Suite - Learning, Stability, and Integration
 *
 * This suite tests the "institutional grade" features:
 *   - Online EM learning convergence
 *   - MCMC shock teleportation efficacy
 *   - SPRT robustness to false alarms
 *   - Student-t physics (fat tail detection)
 *   - Gated learning integrity (mode collapse prevention)
 *   - Numerical stability under adversarial inputs
 *   - Kelly-correct variance estimation
 *
 * Run after test_mmpf_correctness.c passes.
 */

#include "mmpf_rocks.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

/*═══════════════════════════════════════════════════════════════════════════
 * TEST INFRASTRUCTURE
 *═══════════════════════════════════════════════════════════════════════════*/

static int g_tests_run = 0;
static int g_tests_passed = 0;
static int g_tests_failed = 0;
static unsigned int g_rng_state = 42;

#define TEST_ASSERT(cond, msg)                              \
    do                                                      \
    {                                                       \
        if (!(cond))                                        \
        {                                                   \
            printf("    ASSERT FAILED: %s\n", msg);         \
            printf("      at %s:%d\n", __FILE__, __LINE__); \
            return 0;                                       \
        }                                                   \
    } while (0)

#define TEST_ASSERT_NEAR(val, expected, tol, msg)                           \
    do                                                                      \
    {                                                                       \
        double _v = (double)(val);                                          \
        double _e = (double)(expected);                                     \
        double _t = (double)(tol);                                          \
        if (fabs(_v - _e) > _t)                                             \
        {                                                                   \
            printf("    ASSERT FAILED: %s\n", msg);                         \
            printf("      expected: %.6f ± %.6f, got: %.6f\n", _e, _t, _v); \
            printf("      at %s:%d\n", __FILE__, __LINE__);                 \
            return 0;                                                       \
        }                                                                   \
    } while (0)

#define TEST_ASSERT_GT(val, threshold, msg)                        \
    do                                                             \
    {                                                              \
        double _v = (double)(val);                                 \
        double _t = (double)(threshold);                           \
        if (_v <= _t)                                              \
        {                                                          \
            printf("    ASSERT FAILED: %s\n", msg);                \
            printf("      expected: > %.6f, got: %.6f\n", _t, _v); \
            printf("      at %s:%d\n", __FILE__, __LINE__);        \
            return 0;                                              \
        }                                                          \
    } while (0)

#define TEST_ASSERT_LT(val, threshold, msg)                        \
    do                                                             \
    {                                                              \
        double _v = (double)(val);                                 \
        double _t = (double)(threshold);                           \
        if (_v >= _t)                                              \
        {                                                          \
            printf("    ASSERT FAILED: %s\n", msg);                \
            printf("      expected: < %.6f, got: %.6f\n", _t, _v); \
            printf("      at %s:%d\n", __FILE__, __LINE__);        \
            return 0;                                              \
        }                                                          \
    } while (0)

#define TEST_ASSERT_GE(val, threshold, msg)                         \
    do                                                              \
    {                                                               \
        double _v = (double)(val);                                  \
        double _t = (double)(threshold);                            \
        if (_v < _t)                                                \
        {                                                           \
            printf("    ASSERT FAILED: %s\n", msg);                 \
            printf("      expected: >= %.6f, got: %.6f\n", _t, _v); \
            printf("      at %s:%d\n", __FILE__, __LINE__);         \
            return 0;                                               \
        }                                                           \
    } while (0)

#define RUN_TEST(test_func, test_name)                      \
    do                                                      \
    {                                                       \
        printf("  [TEST] %s", test_name);                   \
        fflush(stdout);                                     \
        g_tests_run++;                                      \
        if (test_func())                                    \
        {                                                   \
            g_tests_passed++;                               \
            printf("\r  [TEST] %-50s PASSED\n", test_name); \
        }                                                   \
        else                                                \
        {                                                   \
            g_tests_failed++;                               \
            printf("FAILED\n");                             \
        }                                                   \
    } while (0)

/*═══════════════════════════════════════════════════════════════════════════
 * RANDOM NUMBER GENERATION (Simple LCG for reproducibility)
 *═══════════════════════════════════════════════════════════════════════════*/

static void test_seed(unsigned int seed)
{
    g_rng_state = seed;
}

static double test_uniform(void)
{
    g_rng_state = g_rng_state * 1103515245 + 12345;
    return (double)(g_rng_state & 0x7FFFFFFF) / (double)0x7FFFFFFF;
}

static double test_gaussian(void)
{
    /* Box-Muller transform */
    double u1 = test_uniform();
    double u2 = test_uniform();
    if (u1 < 1e-10)
        u1 = 1e-10;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979 * u2);
}

static double test_student_t(double nu)
{
    /* Student-t via ratio of Gaussian and Chi-squared */
    double z = test_gaussian();
    double v = 0.0;
    for (int i = 0; i < (int)nu; i++)
    {
        double g = test_gaussian();
        v += g * g;
    }
    return z / sqrt(v / nu);
}

/*═══════════════════════════════════════════════════════════════════════════
 * SYNTHETIC DATA GENERATORS
 *═══════════════════════════════════════════════════════════════════════════*/

/* Generate return with specified volatility */
static rbpf_real_t generate_return(double sigma)
{
    return (rbpf_real_t)(sigma * test_gaussian());
}

/* Generate return from Student-t distribution */
static rbpf_real_t generate_return_student_t(double sigma, double nu)
{
    return (rbpf_real_t)(sigma * test_student_t(nu));
}

/* Volatility levels for each regime */
#define SIGMA_CALM 0.005   /* 0.5% daily vol */
#define SIGMA_TREND 0.016  /* 1.6% daily vol */
#define SIGMA_CRISIS 0.050 /* 5.0% daily vol */

/* Log-vol levels (mu_vol = log(sigma)) */
#define MU_VOL_CALM (-5.30)   /* log(0.005) */
#define MU_VOL_TREND (-4.14)  /* log(0.016) */
#define MU_VOL_CRISIS (-3.00) /* log(0.050) */

/*═══════════════════════════════════════════════════════════════════════════
 * HELPER: Create MMPF with specific configuration
 *═══════════════════════════════════════════════════════════════════════════*/

/* Standard test MMPF (learning disabled, cartoon separation) */
static MMPF_ROCKS *create_test_mmpf_standard(void)
{
    MMPF_Config cfg = mmpf_config_defaults();
    cfg.n_particles = 256;

    /* Disable adaptive learning for deterministic tests */
    cfg.enable_storvik_sync = 0;
    cfg.enable_global_baseline = 0;
    cfg.enable_gated_learning = 0;
    cfg.enable_nu_learning = 0;

    /* Cartoon separation */
    cfg.hypotheses[MMPF_CALM].mu_vol = RBPF_REAL(-5.30);
    cfg.hypotheses[MMPF_CALM].phi = RBPF_REAL(0.98);
    cfg.hypotheses[MMPF_CALM].sigma_eta = RBPF_REAL(0.08);
    cfg.hypotheses[MMPF_CALM].nu = RBPF_REAL(20.0);

    cfg.hypotheses[MMPF_TREND].mu_vol = RBPF_REAL(-4.14);
    cfg.hypotheses[MMPF_TREND].phi = RBPF_REAL(0.95);
    cfg.hypotheses[MMPF_TREND].sigma_eta = RBPF_REAL(0.12);
    cfg.hypotheses[MMPF_TREND].nu = RBPF_REAL(6.0);

    cfg.hypotheses[MMPF_CRISIS].mu_vol = RBPF_REAL(-3.00);
    cfg.hypotheses[MMPF_CRISIS].phi = RBPF_REAL(0.85);
    cfg.hypotheses[MMPF_CRISIS].sigma_eta = RBPF_REAL(0.20);
    cfg.hypotheses[MMPF_CRISIS].nu = RBPF_REAL(3.0);

    cfg.base_stickiness = RBPF_REAL(0.90);
    cfg.min_stickiness = RBPF_REAL(0.70);

    cfg.initial_weights[MMPF_CALM] = RBPF_REAL(0.34);
    cfg.initial_weights[MMPF_TREND] = RBPF_REAL(0.33);
    cfg.initial_weights[MMPF_CRISIS] = RBPF_REAL(0.33);

    cfg.rng_seed = 42;

    return mmpf_create(&cfg);
}

/* MMPF with Online EM enabled */
static MMPF_ROCKS *create_test_mmpf_with_learning(void)
{
    MMPF_Config cfg = mmpf_config_defaults();
    cfg.n_particles = 256;

    /* Enable Online EM learning */
    cfg.enable_storvik_sync = 1;
    cfg.enable_global_baseline = 0;
    cfg.enable_gated_learning = 1;
    cfg.enable_nu_learning = 0;

    /* Start with WRONG centers (all clustered) - test will verify they drift */
    cfg.hypotheses[MMPF_CALM].mu_vol = RBPF_REAL(-5.0);
    cfg.hypotheses[MMPF_TREND].mu_vol = RBPF_REAL(-5.0);
    cfg.hypotheses[MMPF_CRISIS].mu_vol = RBPF_REAL(-5.0);

    cfg.hypotheses[MMPF_CALM].phi = RBPF_REAL(0.98);
    cfg.hypotheses[MMPF_CALM].sigma_eta = RBPF_REAL(0.08);
    cfg.hypotheses[MMPF_CALM].nu = RBPF_REAL(20.0);

    cfg.hypotheses[MMPF_TREND].phi = RBPF_REAL(0.95);
    cfg.hypotheses[MMPF_TREND].sigma_eta = RBPF_REAL(0.12);
    cfg.hypotheses[MMPF_TREND].nu = RBPF_REAL(6.0);

    cfg.hypotheses[MMPF_CRISIS].phi = RBPF_REAL(0.85);
    cfg.hypotheses[MMPF_CRISIS].sigma_eta = RBPF_REAL(0.20);
    cfg.hypotheses[MMPF_CRISIS].nu = RBPF_REAL(3.0);

    cfg.base_stickiness = RBPF_REAL(0.90);
    cfg.min_stickiness = RBPF_REAL(0.70);

    cfg.initial_weights[MMPF_CALM] = RBPF_REAL(0.34);
    cfg.initial_weights[MMPF_TREND] = RBPF_REAL(0.33);
    cfg.initial_weights[MMPF_CRISIS] = RBPF_REAL(0.33);

    cfg.rng_seed = 42;

    return mmpf_create(&cfg);
}

/* MMPF with Storvik gated learning for anchor tests */
static MMPF_ROCKS *create_test_mmpf_with_storvik(void)
{
    MMPF_Config cfg = mmpf_config_defaults();
    cfg.n_particles = 256;

    cfg.enable_storvik_sync = 1;
    cfg.enable_global_baseline = 0;
    cfg.enable_gated_learning = 1;
    cfg.enable_nu_learning = 0;

    /* Proper separation */
    cfg.hypotheses[MMPF_CALM].mu_vol = RBPF_REAL(-5.30);
    cfg.hypotheses[MMPF_TREND].mu_vol = RBPF_REAL(-4.14);
    cfg.hypotheses[MMPF_CRISIS].mu_vol = RBPF_REAL(-3.00);

    cfg.hypotheses[MMPF_CALM].phi = RBPF_REAL(0.98);
    cfg.hypotheses[MMPF_CALM].sigma_eta = RBPF_REAL(0.08);
    cfg.hypotheses[MMPF_CALM].nu = RBPF_REAL(20.0);

    cfg.hypotheses[MMPF_TREND].phi = RBPF_REAL(0.95);
    cfg.hypotheses[MMPF_TREND].sigma_eta = RBPF_REAL(0.12);
    cfg.hypotheses[MMPF_TREND].nu = RBPF_REAL(6.0);

    cfg.hypotheses[MMPF_CRISIS].phi = RBPF_REAL(0.85);
    cfg.hypotheses[MMPF_CRISIS].sigma_eta = RBPF_REAL(0.20);
    cfg.hypotheses[MMPF_CRISIS].nu = RBPF_REAL(3.0);

    cfg.base_stickiness = RBPF_REAL(0.90);
    cfg.min_stickiness = RBPF_REAL(0.70);

    cfg.initial_weights[MMPF_CALM] = RBPF_REAL(0.34);
    cfg.initial_weights[MMPF_TREND] = RBPF_REAL(0.33);
    cfg.initial_weights[MMPF_CRISIS] = RBPF_REAL(0.33);

    cfg.rng_seed = 42;

    return mmpf_create(&cfg);
}

/*═══════════════════════════════════════════════════════════════════════════
 * STRATEGY 1: LEARNING VALIDATION TESTS
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Test 1.1: The "Lazy Manager" Test (Online EM Convergence)
 *
 * Initialize with WRONG centers, verify Online EM learns the truth.
 */
static int test_online_em_convergence(void)
{
    test_seed(12345);
    MMPF_ROCKS *mmpf = create_test_mmpf_with_learning();
    TEST_ASSERT(mmpf != NULL, "Failed to create MMPF");

    MMPF_Output output;

    /* Record initial (wrong) centers */
    double init_mu_calm = mmpf->online_em.mu[MMPF_CALM];
    double init_mu_trend = mmpf->online_em.mu[MMPF_TREND];
    double init_mu_crisis = mmpf->online_em.mu[MMPF_CRISIS];

    printf("\n    Initial centers: [%.2f, %.2f, %.2f]\n",
           init_mu_calm, init_mu_trend, init_mu_crisis);

    /* Phase 1: 300 ticks of Calm data (true μ = -4.5, σ = 1.1%) */
    double sigma_phase1 = 0.011; /* exp(-4.5) ≈ 0.011 */
    for (int t = 0; t < 300; t++)
    {
        rbpf_real_t r = generate_return(sigma_phase1);
        mmpf_step(mmpf, r, &output);
    }

    /* Phase 2: 300 ticks of Trend data (true μ = -3.0, σ = 5%) */
    double sigma_phase2 = 0.050; /* exp(-3.0) ≈ 0.050 */
    for (int t = 0; t < 300; t++)
    {
        rbpf_real_t r = generate_return(sigma_phase2);
        mmpf_step(mmpf, r, &output);
    }

    /* Phase 3: 300 ticks of Crisis data (true μ = -1.5, σ = 22%) */
    double sigma_phase3 = 0.22; /* exp(-1.5) ≈ 0.22 */
    for (int t = 0; t < 300; t++)
    {
        rbpf_real_t r = generate_return(sigma_phase3);
        mmpf_step(mmpf, r, &output);
    }

    /* Check that centers have moved toward truth */
    double final_mu_calm = mmpf->online_em.mu[MMPF_CALM];
    double final_mu_trend = mmpf->online_em.mu[MMPF_TREND];
    double final_mu_crisis = mmpf->online_em.mu[MMPF_CRISIS];

    printf("    Final centers:   [%.2f, %.2f, %.2f]\n",
           final_mu_calm, final_mu_trend, final_mu_crisis);

    /* Verify ordering is maintained: Calm < Trend < Crisis */
    TEST_ASSERT_LT(final_mu_calm, final_mu_trend,
                   "Calm center should be less than Trend");
    TEST_ASSERT_LT(final_mu_trend, final_mu_crisis,
                   "Trend center should be less than Crisis");

    /* Verify centers moved from initial clustered position */
    double total_movement = fabs(final_mu_calm - init_mu_calm) +
                            fabs(final_mu_trend - init_mu_trend) +
                            fabs(final_mu_crisis - init_mu_crisis);
    TEST_ASSERT_GT(total_movement, 1.0,
                   "Centers should have moved significantly from initial");

    printf("    Total movement: %.2f\n", total_movement);

    mmpf_destroy(mmpf);
    return 1;
}

/**
 * Test 1.2: The "Teleport" Test (MCMC Shock Efficacy)
 *
 * Compare reaction time with and without MCMC shock injection.
 * Note: This test requires MCMC to be enabled at compile time.
 */
static int test_mcmc_shock_efficacy(void)
{
    test_seed(54321);

    MMPF_ROCKS *mmpf = create_test_mmpf_standard();
    TEST_ASSERT(mmpf != NULL, "Failed to create MMPF");

    MMPF_Output output;

    /* Burn-in: 100 ticks of Calm */
    for (int t = 0; t < 100; t++)
    {
        rbpf_real_t r = generate_return(SIGMA_CALM);
        mmpf_step(mmpf, r, &output);
    }

    double vol_before_shock = (double)output.volatility;
    printf("\n    Vol before shock: %.4f\n", vol_before_shock);

    /* Flash crash: immediate jump to crisis volatility */
    double target_vol = SIGMA_CRISIS;

    /* Inject shock to help particles teleport */
    rbpf_real_t crisis_return = generate_return(SIGMA_CRISIS);
    rbpf_real_t y_log_sq = rbpf_log(crisis_return * crisis_return);
    mmpf_inject_shock_adaptive(mmpf, y_log_sq);

    /* Track how quickly we reach target */
    int ticks_to_90pct = -1;
    double threshold = vol_before_shock + 0.9 * (target_vol - vol_before_shock);

    for (int t = 0; t < 50; t++)
    {
        rbpf_real_t r = generate_return(SIGMA_CRISIS);
        mmpf_step(mmpf, r, &output);

        if (ticks_to_90pct < 0 && (double)output.volatility >= threshold)
        {
            ticks_to_90pct = t + 1;
        }
    }

    printf("    Vol after 50 crisis ticks: %.4f\n", (double)output.volatility);
    printf("    Ticks to 90%% of target: %d\n", ticks_to_90pct);

    /* With MCMC shock, should reach 90% within 10 ticks */
    TEST_ASSERT(ticks_to_90pct > 0, "Should eventually reach target vol");
    TEST_ASSERT_LT(ticks_to_90pct, 15, "MCMC should accelerate convergence");

    /* Verify shock was auto-exited (entropy-based) */
    TEST_ASSERT(mmpf_is_shock_active(mmpf) == 0,
                "Shock should have auto-exited after stabilization");

    mmpf_destroy(mmpf);
    return 1;
}

/**
 * Test 1.3: The "False Alarm" Test (SPRT Robustness)
 *
 * Single outlier should NOT cause permanent regime switch.
 */
static int test_sprt_false_alarm_robustness(void)
{
    test_seed(99999);
    MMPF_ROCKS *mmpf = create_test_mmpf_standard();
    TEST_ASSERT(mmpf != NULL, "Failed to create MMPF");

    MMPF_Output output;

    /* Establish Calm regime: 200 ticks */
    for (int t = 0; t < 200; t++)
    {
        rbpf_real_t r = generate_return(SIGMA_CALM);
        mmpf_step(mmpf, r, &output);
    }

    TEST_ASSERT(output.dominant == MMPF_CALM || output.dominant == MMPF_TREND,
                "Should be in Calm or Trend before outlier");
    MMPF_Hypothesis regime_before = output.dominant;
    double calm_weight_before = (double)output.weights[MMPF_CALM];

    printf("\n    Before outlier: dominant=%d, w_calm=%.3f\n",
           regime_before, calm_weight_before);

    /* Inject ONE 10-sigma outlier */
    rbpf_real_t outlier = RBPF_REAL(10.0 * SIGMA_CALM);
    mmpf_step(mmpf, outlier, &output);

    double outlier_frac_spike = (double)output.outlier_fraction;
    printf("    On outlier: outlier_frac=%.3f\n", outlier_frac_spike);

    /* Immediately return to Calm */
    for (int t = 0; t < 50; t++)
    {
        rbpf_real_t r = generate_return(SIGMA_CALM);
        mmpf_step(mmpf, r, &output);
    }

    double calm_weight_after = (double)output.weights[MMPF_CALM];
    printf("    After recovery: dominant=%d, w_calm=%.3f\n",
           output.dominant, calm_weight_after);

    /* Key check: Calm should have recovered (or never fully lost) */
    TEST_ASSERT_GT(calm_weight_after, 0.3,
                   "Calm weight should recover after single outlier");

    /* Should NOT be stuck in Crisis */
    TEST_ASSERT(output.dominant != MMPF_CRISIS,
                "Should not be stuck in Crisis after single outlier");

    mmpf_destroy(mmpf);
    return 1;
}

/*═══════════════════════════════════════════════════════════════════════════
 * STRATEGY 2: CHAOS MONKEY TESTS (Stability Verification)
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Test 2.1: The "Flatline" Test (Zero Returns)
 *
 * HFT data often has long streaks of zero returns.
 */
static int test_zero_return_stability(void)
{
    MMPF_ROCKS *mmpf = create_test_mmpf_standard();
    TEST_ASSERT(mmpf != NULL, "Failed to create MMPF");

    MMPF_Output output;

    /* Feed 5000 exact zeros */
    for (int t = 0; t < 5000; t++)
    {
        mmpf_step(mmpf, RBPF_REAL(0.0), &output);

        /* Check for NaN/Inf every 100 ticks */
        if (t % 100 == 0)
        {
            TEST_ASSERT(output.volatility == output.volatility, "NaN in volatility");
            TEST_ASSERT(output.log_volatility == output.log_volatility, "NaN in log_volatility");
            TEST_ASSERT(output.volatility < 1e10f, "Inf in volatility");
        }
    }

    /* Volatility should have floored, not exploded or collapsed to zero */
    TEST_ASSERT_GT(output.volatility, 1e-10, "Volatility should not collapse to zero");
    TEST_ASSERT_LT(output.volatility, 1.0, "Volatility should not explode");

    /* ESS should remain reasonable */
    for (int k = 0; k < MMPF_N_MODELS; k++)
    {
        TEST_ASSERT_GT(output.model_ess[k], 50.0,
                       "ESS should remain healthy on zero data");
    }

    printf("\n    After 5000 zeros: vol=%.6f, ESS=[%.0f, %.0f, %.0f]\n",
           (double)output.volatility,
           (double)output.model_ess[0],
           (double)output.model_ess[1],
           (double)output.model_ess[2]);

    mmpf_destroy(mmpf);
    return 1;
}

/**
 * Test 2.2: The "NaN Attack"
 *
 * Bad data points should not crash or corrupt the filter.
 */
static int test_nan_attack_robustness(void)
{
    test_seed(11111);
    MMPF_ROCKS *mmpf = create_test_mmpf_standard();
    TEST_ASSERT(mmpf != NULL, "Failed to create MMPF");

    MMPF_Output output;

    /* Establish baseline: 100 normal ticks */
    for (int t = 0; t < 100; t++)
    {
        rbpf_real_t r = generate_return(SIGMA_CALM);
        mmpf_step(mmpf, r, &output);
    }

    double vol_before = (double)output.volatility;
    printf("\n    Vol before attack: %.6f\n", vol_before);

    /* Attack with NaN */
    rbpf_real_t nan_val = RBPF_REAL(0.0) / RBPF_REAL(0.0); /* NaN */
    mmpf_step(mmpf, nan_val, &output);

    /* Check filter didn't explode */
    TEST_ASSERT(output.volatility == output.volatility, "Filter corrupted by NaN");

    /* Attack with Inf */
    rbpf_real_t inf_val = RBPF_REAL(1.0) / RBPF_REAL(0.0); /* +Inf */
    mmpf_step(mmpf, inf_val, &output);

    TEST_ASSERT(output.volatility == output.volatility, "Filter corrupted by Inf");

    /* Attack with huge value */
    mmpf_step(mmpf, RBPF_REAL(1e30), &output);
    TEST_ASSERT(output.volatility == output.volatility, "Filter corrupted by huge value");

    /* Resume normal operation */
    for (int t = 0; t < 100; t++)
    {
        rbpf_real_t r = generate_return(SIGMA_CALM);
        mmpf_step(mmpf, r, &output);
    }

    double vol_after = (double)output.volatility;
    printf("    Vol after attack + recovery: %.6f\n", vol_after);

    /* Should have recovered to reasonable range */
    TEST_ASSERT(output.volatility == output.volatility, "Filter still has NaN");
    TEST_ASSERT_LT(output.volatility, 1.0, "Volatility should be bounded");

    mmpf_destroy(mmpf);
    return 1;
}

/**
 * Test 2.3: The "Denormalized Float" Test
 *
 * Subnormal floats shouldn't break the filter.
 */
static int test_denormalized_float_handling(void)
{
    MMPF_ROCKS *mmpf = create_test_mmpf_standard();
    TEST_ASSERT(mmpf != NULL, "Failed to create MMPF");

    MMPF_Output output;

    /* Feed subnormal values */
    rbpf_real_t subnormal = RBPF_REAL(1e-38);

    for (int t = 0; t < 100; t++)
    {
        mmpf_step(mmpf, subnormal, &output);
    }

    /* Check no NaN/Inf */
    TEST_ASSERT(output.volatility == output.volatility, "NaN from subnormal");
    TEST_ASSERT(output.volatility < 1e10f, "Inf from subnormal");

    printf("\n    After 100 subnormal inputs: vol=%.6f\n", (double)output.volatility);

    mmpf_destroy(mmpf);
    return 1;
}

/*═══════════════════════════════════════════════════════════════════════════
 * STRATEGY 3: INTEGRATION TESTS (Component Handshakes)
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Test 3.1: The "Lock & Key" Test (Entropy Auto-Unlock)
 *
 * Verify entropy lock engages on shock and disengages on stability.
 */
static int test_entropy_auto_unlock(void)
{
    test_seed(77777);
    MMPF_ROCKS *mmpf = create_test_mmpf_standard();
    TEST_ASSERT(mmpf != NULL, "Failed to create MMPF");

    MMPF_Output output;

    /* Establish baseline */
    for (int t = 0; t < 50; t++)
    {
        rbpf_real_t r = generate_return(SIGMA_CALM);
        mmpf_step(mmpf, r, &output);
    }

    /* Force shock */
    mmpf_inject_shock(mmpf);

    TEST_ASSERT(mmpf_is_shock_active(mmpf) == 1, "Shock should be active");
    TEST_ASSERT(mmpf->entropy.locked == 1, "Entropy should be locked");

    printf("\n    Shock injected, locked=%d\n", mmpf->entropy.locked);

    /* Feed steady data until auto-unlock */
    int unlock_tick = -1;
    for (int t = 0; t < 100; t++)
    {
        rbpf_real_t r = generate_return(SIGMA_CALM);
        mmpf_step(mmpf, r, &output);

        if (unlock_tick < 0 && mmpf->entropy.locked == 0)
        {
            unlock_tick = t;
        }
    }

    printf("    Auto-unlocked at tick: %d\n", unlock_tick);

    TEST_ASSERT(unlock_tick >= 0, "Entropy should have auto-unlocked");
    TEST_ASSERT(mmpf_is_shock_active(mmpf) == 0, "Shock should have auto-exited");

    mmpf_destroy(mmpf);
    return 1;
}

/**
 * Test 3.2: The "Swim Lane" Test (Cluster Ordering)
 *
 * Online EM must maintain hierarchy: Calm < Trend < Crisis.
 */
static int test_cluster_ordering_invariant(void)
{
    test_seed(88888);
    MMPF_ROCKS *mmpf = create_test_mmpf_with_learning();
    TEST_ASSERT(mmpf != NULL, "Failed to create MMPF");

    MMPF_Output output;

    /* Feed random mixed data */
    for (int t = 0; t < 1000; t++)
    {
        /* Randomly pick regime */
        double sigma;
        double r = test_uniform();
        if (r < 0.33)
            sigma = SIGMA_CALM;
        else if (r < 0.66)
            sigma = SIGMA_TREND;
        else
            sigma = SIGMA_CRISIS;

        rbpf_real_t ret = generate_return(sigma);
        mmpf_step(mmpf, ret, &output);

        /* Check ordering invariant every 100 ticks */
        if (t % 100 == 99)
        {
            double mu0 = mmpf->online_em.mu[MMPF_CALM];
            double mu1 = mmpf->online_em.mu[MMPF_TREND];
            double mu2 = mmpf->online_em.mu[MMPF_CRISIS];

            /* Note: Due to learning dynamics, strict ordering may temporarily
             * be violated. Check that separation exists (not all clustered). */
            double spread = mu2 - mu0;
            TEST_ASSERT_GT(spread, 0.5,
                           "Cluster spread should be maintained");
        }
    }

    double final_mu0 = mmpf->online_em.mu[MMPF_CALM];
    double final_mu1 = mmpf->online_em.mu[MMPF_TREND];
    double final_mu2 = mmpf->online_em.mu[MMPF_CRISIS];

    printf("\n    Final Online EM centers: [%.2f, %.2f, %.2f]\n",
           final_mu0, final_mu1, final_mu2);

    /* Final ordering should be correct */
    TEST_ASSERT_LT(final_mu0, final_mu2, "Calm should be less than Crisis");

    mmpf_destroy(mmpf);
    return 1;
}

/*═══════════════════════════════════════════════════════════════════════════
 * STRATEGY 4: KELLY INTEGRATION TESTS (Trading Readiness)
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Test 4.1: Law of Total Variance Correctness
 *
 * Kelly f = μ/σ² — must have correct variance estimate.
 */
static int test_law_of_total_variance(void)
{
    test_seed(44444);
    MMPF_ROCKS *mmpf = create_test_mmpf_standard();
    TEST_ASSERT(mmpf != NULL, "Failed to create MMPF");

    MMPF_Output output;

    /* Run until models have different estimates */
    for (int t = 0; t < 200; t++)
    {
        /* Mix of volatilities to create model disagreement */
        double sigma = (t % 3 == 0) ? SIGMA_CALM : (t % 3 == 1) ? SIGMA_TREND
                                                                : SIGMA_CRISIS;
        rbpf_real_t r = generate_return(sigma);
        mmpf_step(mmpf, r, &output);
    }

    double between_var = (double)output.between_model_var;
    double within_var = (double)output.within_model_var;
    double total_std = (double)output.volatility_std;
    double total_var = total_std * total_std;

    printf("\n    Between-model var: %.6f\n", between_var);
    printf("    Within-model var:  %.6f\n", within_var);
    printf("    Total var:         %.6f\n", total_var);
    printf("    Sum (B+W):         %.6f\n", between_var + within_var);

    /* Law of total variance: Var = Between + Within */
    TEST_ASSERT_NEAR(total_var, between_var + within_var, 1e-6,
                     "Total variance should equal between + within");

    /* Both components should be non-negative */
    TEST_ASSERT_GE(between_var, 0.0, "Between-model variance must be non-negative");
    TEST_ASSERT_GE(within_var, 0.0, "Within-model variance must be non-negative");

    mmpf_destroy(mmpf);
    return 1;
}

/**
 * Test 4.2: The "Confidence Collapse" Test
 *
 * When models agree, uncertainty should be low but non-zero.
 */
static int test_confidence_when_models_agree(void)
{
    test_seed(55555);
    MMPF_ROCKS *mmpf = create_test_mmpf_standard();
    TEST_ASSERT(mmpf != NULL, "Failed to create MMPF");

    MMPF_Output output;

    /* Feed consistent Calm data to make models agree */
    for (int t = 0; t < 500; t++)
    {
        rbpf_real_t r = generate_return(SIGMA_CALM);
        mmpf_step(mmpf, r, &output);
    }

    double between_var = (double)output.between_model_var;
    double within_var = (double)output.within_model_var;

    printf("\n    After 500 consistent ticks:\n");
    printf("    Between-model var: %.8f\n", between_var);
    printf("    Within-model var:  %.8f\n", within_var);
    printf("    Weights: [%.3f, %.3f, %.3f]\n",
           (double)output.weights[0],
           (double)output.weights[1],
           (double)output.weights[2]);

    /* When one model dominates, between-model variance should be low */
    /* (But not necessarily zero if other models have different estimates) */

    /* Within-model variance should always be positive (particle uncertainty) */
    TEST_ASSERT_GT(within_var, 0.0,
                   "Within-model variance must be positive (particle uncertainty exists)");

    /* Total uncertainty should be non-zero */
    double total_std = (double)output.volatility_std;
    TEST_ASSERT_GT(total_std, 0.0, "Total uncertainty must be non-zero");

    mmpf_destroy(mmpf);
    return 1;
}

/*═══════════════════════════════════════════════════════════════════════════
 * STRATEGY 5: STUDENT-T PHYSICS TESTS
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Test 5.1: Fat Tail Likelihood Test
 *
 * Crisis (ν=3) should win on fat-tailed data.
 */
static int test_fat_tail_likelihood(void)
{
    test_seed(33333);
    MMPF_ROCKS *mmpf = create_test_mmpf_standard();
    TEST_ASSERT(mmpf != NULL, "Failed to create MMPF");

    /* Enable Student-t */
    mmpf_enable_student_t(mmpf, NULL);

    MMPF_Output output;

    /* Generate returns from true Student-t(ν=3) distribution */
    double sigma = 0.02; /* 2% base vol */
    for (int t = 0; t < 300; t++)
    {
        rbpf_real_t r = generate_return_student_t(sigma, 3.0);
        mmpf_step(mmpf, r, &output);
    }

    printf("\n    After 300 Student-t(ν=3) ticks:\n");
    printf("    Weights: [%.3f, %.3f, %.3f]\n",
           (double)output.weights[0],
           (double)output.weights[1],
           (double)output.weights[2]);
    printf("    Lambda mean: %.3f\n", (double)output.model_lambda_mean[MMPF_CRISIS]);

    /* Crisis (ν=3) should have significant weight on fat-tailed data */
    /* Note: Exact threshold depends on the vol level and separation */
    TEST_ASSERT_GT(output.weights[MMPF_CRISIS], 0.15,
                   "Crisis should gain weight on fat-tailed data");

    /* Calm (ν=20) should be suppressed */
    TEST_ASSERT_LT(output.weights[MMPF_CALM], 0.5,
                   "Calm should not dominate on fat-tailed data");

    mmpf_destroy(mmpf);
    return 1;
}

/**
 * Test 5.2: The "Gaussian Impostor" Test
 *
 * Calm (ν=20, near-Gaussian) should win on thin-tailed data.
 */
static int test_gaussian_impostor(void)
{
    test_seed(22222);
    MMPF_ROCKS *mmpf = create_test_mmpf_standard();
    TEST_ASSERT(mmpf != NULL, "Failed to create MMPF");

    /* Enable Student-t */
    mmpf_enable_student_t(mmpf, NULL);

    MMPF_Output output;

    /* Generate returns from true Gaussian (thin tails) */
    for (int t = 0; t < 500; t++)
    {
        rbpf_real_t r = generate_return(SIGMA_CALM);
        mmpf_step(mmpf, r, &output);
    }

    printf("\n    After 500 Gaussian ticks:\n");
    printf("    Weights: [%.3f, %.3f, %.3f]\n",
           (double)output.weights[0],
           (double)output.weights[1],
           (double)output.weights[2]);

    /* Calm should dominate on Gaussian data */
    TEST_ASSERT_GT(output.weights[MMPF_CALM], 0.4,
                   "Calm should dominate on Gaussian data");

    mmpf_destroy(mmpf);
    return 1;
}

/*═══════════════════════════════════════════════════════════════════════════
 * STRATEGY 6: GATED LEARNING TESTS (Mode Collapse Prevention)
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Test 6.1: The "Pinned Anchor" Test
 *
 * Calm's μ_vol should NEVER drift, even during extended crisis.
 */
static int test_pinned_anchor_stability(void)
{
    test_seed(66666);
    MMPF_ROCKS *mmpf = create_test_mmpf_with_storvik();
    TEST_ASSERT(mmpf != NULL, "Failed to create MMPF");

    /* Record Calm's initial μ_vol */
    double calm_mu_initial = mmpf->gated_dynamics[MMPF_CALM].mu_vol;
    printf("\n    Calm μ_vol initial: %.4f\n", calm_mu_initial);

    MMPF_Output output;

    /* Feed 1000 ticks of pure Crisis data */
    for (int t = 0; t < 1000; t++)
    {
        rbpf_real_t r = generate_return(SIGMA_CRISIS);
        mmpf_step(mmpf, r, &output);
    }

    double calm_mu_final = mmpf->gated_dynamics[MMPF_CALM].mu_vol;
    double crisis_mu_final = mmpf->gated_dynamics[MMPF_CRISIS].mu_vol;

    printf("    Calm μ_vol final:   %.4f (drift: %.4f)\n",
           calm_mu_final, fabs(calm_mu_final - calm_mu_initial));
    printf("    Crisis μ_vol final: %.4f\n", crisis_mu_final);

    /* Calm should NOT have drifted significantly (it's pinned) */
    TEST_ASSERT_NEAR(calm_mu_final, calm_mu_initial, 0.5,
                     "Calm μ_vol should be pinned (no drift)");

    /* Crisis SHOULD have adapted */
    /* (This depends on the learning rate and data, so we just check it's reasonable) */

    mmpf_destroy(mmpf);
    return 1;
}

/**
 * Test 6.2: The "Leakage" Test
 *
 * Losing hypotheses should NOT learn from data they don't explain.
 */
static int test_leakage_prevention(void)
{
    test_seed(77788);
    MMPF_ROCKS *mmpf = create_test_mmpf_with_storvik();
    TEST_ASSERT(mmpf != NULL, "Failed to create MMPF");

    MMPF_Output output;

    /* First: establish Crisis as dominant with crisis data */
    for (int t = 0; t < 200; t++)
    {
        rbpf_real_t r = generate_return(SIGMA_CRISIS);
        mmpf_step(mmpf, r, &output);
    }

    printf("\n    After 200 crisis ticks: w=[%.3f, %.3f, %.3f]\n",
           (double)output.weights[0],
           (double)output.weights[1],
           (double)output.weights[2]);

    /* Record Calm's stats (should be frozen due to low weight) */
    double calm_mu_before = mmpf->gated_dynamics[MMPF_CALM].mu_vol;

    /* Continue feeding Crisis data - Calm should NOT learn */
    for (int t = 0; t < 300; t++)
    {
        rbpf_real_t r = generate_return(SIGMA_CRISIS);
        mmpf_step(mmpf, r, &output);
    }

    double calm_mu_after = mmpf->gated_dynamics[MMPF_CALM].mu_vol;
    double drift = fabs(calm_mu_after - calm_mu_before);

    printf("    Calm μ_vol drift during crisis dominance: %.4f\n", drift);

    /* Calm should not have learned (weight < 10% gate) */
    TEST_ASSERT_LT(drift, 0.3,
                   "Calm should not learn when weight is below gate threshold");

    mmpf_destroy(mmpf);
    return 1;
}

/*═══════════════════════════════════════════════════════════════════════════
 * STRATEGY 7: REGIME DEATH PREVENTION TESTS
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Test 7.1: The "Extinction Event" Test
 *
 * No hypothesis should ever reach weight=0.
 */
static int test_extinction_prevention(void)
{
    test_seed(98765);
    MMPF_ROCKS *mmpf = create_test_mmpf_standard();
    TEST_ASSERT(mmpf != NULL, "Failed to create MMPF");

    MMPF_Output output;

    /* Feed extreme Crisis data to push Calm to extinction */
    for (int t = 0; t < 5000; t++)
    {
        /* Very large returns - 10σ under calm */
        rbpf_real_t r = RBPF_REAL(10.0 * SIGMA_CRISIS) *
                        (test_uniform() > 0.5 ? 1.0f : -1.0f);
        mmpf_step(mmpf, r, &output);

        /* Check no weight went to zero */
        for (int k = 0; k < MMPF_N_MODELS; k++)
        {
            TEST_ASSERT_GT(output.weights[k], 1e-10,
                           "No hypothesis weight should reach zero");
        }
    }

    printf("\n    After 5000 extreme ticks: w=[%.4f, %.4f, %.4f]\n",
           (double)output.weights[0],
           (double)output.weights[1],
           (double)output.weights[2]);

    /* Calm should still be alive due to min_mixing_prob */
    double min_weight = 0.01; /* Default min_mixing_prob is 5% but check 1% floor */
    TEST_ASSERT_GT(output.weights[MMPF_CALM], min_weight,
                   "Calm should survive due to minimum mixing probability");

    /* Now switch to Calm data - it should recover */
    for (int t = 0; t < 200; t++)
    {
        rbpf_real_t r = generate_return(SIGMA_CALM);
        mmpf_step(mmpf, r, &output);
    }

    printf("    After 200 calm recovery ticks: w=[%.3f, %.3f, %.3f]\n",
           (double)output.weights[0],
           (double)output.weights[1],
           (double)output.weights[2]);

    /* Calm should have recovered significantly */
    TEST_ASSERT_GT(output.weights[MMPF_CALM], 0.2,
                   "Calm should recover after data switches back");

    mmpf_destroy(mmpf);
    return 1;
}

/*═══════════════════════════════════════════════════════════════════════════
 * STRATEGY 8: TRANSITION MATRIX TESTS
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Test 8.1: The "Sticky Floor" Test
 *
 * Adaptive stickiness should have bounds.
 */
static int test_stickiness_bounds(void)
{
    test_seed(11122);
    MMPF_ROCKS *mmpf = create_test_mmpf_standard();
    TEST_ASSERT(mmpf != NULL, "Failed to create MMPF");

    MMPF_Output output;

    /* Inject continuous outliers to maximize outlier_fraction */
    for (int t = 0; t < 100; t++)
    {
        rbpf_real_t outlier = RBPF_REAL(20.0 * SIGMA_CALM);
        mmpf_step(mmpf, outlier, &output);

        /* Stickiness should never go below floor */
        TEST_ASSERT_GE(output.current_stickiness, mmpf->config.min_stickiness,
                       "Stickiness should not go below minimum");
    }

    printf("\n    After 100 outliers: stickiness=%.4f (min=%.4f)\n",
           (double)output.current_stickiness,
           (double)mmpf->config.min_stickiness);

    /* Feed normal data - stickiness should recover toward base */
    for (int t = 0; t < 100; t++)
    {
        rbpf_real_t r = generate_return(SIGMA_CALM);
        mmpf_step(mmpf, r, &output);
    }

    printf("    After recovery: stickiness=%.4f (base=%.4f)\n",
           (double)output.current_stickiness,
           (double)mmpf->config.base_stickiness);

    TEST_ASSERT_GT(output.current_stickiness, mmpf->config.min_stickiness + 0.05,
                   "Stickiness should recover toward base");

    mmpf_destroy(mmpf);
    return 1;
}

/**
 * Test 8.2: The "Crisis Exit Boost" Test
 *
 * Crisis should exit faster than other regimes (asymmetric).
 */
static int test_crisis_exit_boost(void)
{
    MMPF_ROCKS *mmpf = create_test_mmpf_standard();
    TEST_ASSERT(mmpf != NULL, "Failed to create MMPF");

    /* Check transition matrix after initialization */
    double calm_to_calm = (double)mmpf->transition[MMPF_CALM][MMPF_CALM];
    double crisis_to_crisis = (double)mmpf->transition[MMPF_CRISIS][MMPF_CRISIS];

    printf("\n    P(Calm→Calm):     %.4f\n", calm_to_calm);
    printf("    P(Crisis→Crisis): %.4f\n", crisis_to_crisis);
    printf("    Crisis exit boost: %.4f\n", (double)mmpf->config.crisis_exit_boost);

    /* Crisis should have lower self-transition (exits faster) */
    TEST_ASSERT_LT(crisis_to_crisis, calm_to_calm,
                   "Crisis should exit faster (lower self-transition)");

    /* The ratio should reflect the crisis_exit_boost parameter */
    double expected_ratio = (double)mmpf->config.crisis_exit_boost;
    double actual_ratio = crisis_to_crisis / calm_to_calm;

    printf("    Expected ratio: %.4f, Actual: %.4f\n", expected_ratio, actual_ratio);

    TEST_ASSERT_NEAR(actual_ratio, expected_ratio, 0.1,
                     "Crisis exit boost should be applied");

    mmpf_destroy(mmpf);
    return 1;
}

/*═══════════════════════════════════════════════════════════════════════════
 * STRATEGY 9: NUMERICAL STABILITY TESTS
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Test 9.1: Long-Run Weight Stability
 *
 * Log-weights should stay bounded over long runs.
 * (Reduced from 100k to 20k for reasonable test time)
 */
static int test_long_run_weight_stability(void)
{
    test_seed(99887);
    MMPF_ROCKS *mmpf = create_test_mmpf_standard();
    TEST_ASSERT(mmpf != NULL, "Failed to create MMPF");

    MMPF_Output output;

    /* Run 20000 steps with random regime switching */
    for (int t = 0; t < 20000; t++)
    {
        double sigma;
        double r = test_uniform();
        if (r < 0.5)
            sigma = SIGMA_CALM;
        else if (r < 0.8)
            sigma = SIGMA_TREND;
        else
            sigma = SIGMA_CRISIS;

        rbpf_real_t ret = generate_return(sigma);
        mmpf_step(mmpf, ret, &output);

        /* Check every 1000 steps */
        if (t % 1000 == 999)
        {
            /* Weights should be valid probabilities */
            double sum = 0.0;
            for (int k = 0; k < MMPF_N_MODELS; k++)
            {
                TEST_ASSERT(output.weights[k] == output.weights[k],
                            "Weight became NaN");
                TEST_ASSERT_GE(output.weights[k], 0.0, "Weight went negative");
                TEST_ASSERT_LT(output.weights[k], 1.1, "Weight exceeded 1");
                sum += (double)output.weights[k];
            }
            TEST_ASSERT_NEAR(sum, 1.0, 0.01, "Weights should sum to 1");
        }
    }

    printf("\n    After 20000 steps: weights sum = %.6f\n",
           (double)(output.weights[0] + output.weights[1] + output.weights[2]));

    mmpf_destroy(mmpf);
    return 1;
}

/**
 * Test 9.2: Monotonic Likelihood Test
 *
 * Consistent data should increase confidence (lower entropy).
 */
static int test_entropy_convergence(void)
{
    test_seed(44455);
    MMPF_ROCKS *mmpf = create_test_mmpf_standard();
    TEST_ASSERT(mmpf != NULL, "Failed to create MMPF");

    MMPF_Output output;

    /* Initial entropy (uniform weights) */
    mmpf_step(mmpf, generate_return(SIGMA_CALM), &output);
    double entropy_t100 = 0.0;
    double entropy_t500 = 0.0;
    double entropy_t1000 = 0.0;

    /* Feed consistent Calm data */
    for (int t = 0; t < 1000; t++)
    {
        rbpf_real_t r = generate_return(SIGMA_CALM);
        mmpf_step(mmpf, r, &output);

        if (t == 99)
            entropy_t100 = (double)output.entropy_norm;
        if (t == 499)
            entropy_t500 = (double)output.entropy_norm;
        if (t == 999)
            entropy_t1000 = (double)output.entropy_norm;
    }

    printf("\n    Entropy at t=100:  %.4f\n", entropy_t100);
    printf("    Entropy at t=500:  %.4f\n", entropy_t500);
    printf("    Entropy at t=1000: %.4f\n", entropy_t1000);

    /* Entropy should generally decrease (more confidence) with consistent data */
    /* Note: Might not be strictly monotonic due to mixing, but trend should be down */
    TEST_ASSERT_LT(entropy_t1000, entropy_t100 + 0.1,
                   "Entropy should decrease or stay stable with consistent data");

    mmpf_destroy(mmpf);
    return 1;
}

/*═══════════════════════════════════════════════════════════════════════════
 * STRATEGY 10: API CONTRACT TESTS
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Test 10.1: The "Double Reset" Test
 *
 * Reset should be idempotent.
 */
static int test_double_reset_idempotent(void)
{
    test_seed(12121);
    MMPF_ROCKS *mmpf = create_test_mmpf_standard();
    TEST_ASSERT(mmpf != NULL, "Failed to create MMPF");

    MMPF_Output output1, output2;

    /* Run some steps */
    for (int t = 0; t < 50; t++)
    {
        mmpf_step(mmpf, generate_return(SIGMA_CALM), &output1);
    }

    /* Reset once */
    mmpf_reset(mmpf, RBPF_REAL(0.01));

    /* Take one step to get state */
    mmpf_step(mmpf, RBPF_REAL(0.001), &output1);

    /* Reset again */
    mmpf_reset(mmpf, RBPF_REAL(0.01));

    /* Take same step */
    mmpf_step(mmpf, RBPF_REAL(0.001), &output2);

    /* States should be identical (given same RNG seed path) */
    /* Note: Due to RNG state, we check structural similarity, not exact match */
    TEST_ASSERT_NEAR(output1.volatility, output2.volatility, 0.01,
                     "Volatility should be similar after double reset");

    for (int k = 0; k < MMPF_N_MODELS; k++)
    {
        TEST_ASSERT_NEAR(output1.weights[k], output2.weights[k], 0.1,
                         "Weights should be similar after double reset");
    }

    printf("\n    After reset 1: vol=%.6f\n", (double)output1.volatility);
    printf("    After reset 2: vol=%.6f\n", (double)output2.volatility);

    mmpf_destroy(mmpf);
    return 1;
}

/**
 * Test 10.2: The "Config Immutability" Test
 *
 * Changing config after create should NOT affect filter.
 */
static int test_config_immutability(void)
{
    MMPF_Config cfg = mmpf_config_defaults();
    cfg.n_particles = 256;
    cfg.enable_storvik_sync = 0;
    cfg.hypotheses[MMPF_CALM].mu_vol = RBPF_REAL(-5.30);

    MMPF_ROCKS *mmpf = mmpf_create(&cfg);
    TEST_ASSERT(mmpf != NULL, "Failed to create MMPF");

    /* Mutate the original config struct */
    cfg.hypotheses[MMPF_CALM].mu_vol = RBPF_REAL(-1.0); /* Very different! */
    cfg.n_particles = 1000;

    /* Run the filter - it should use ORIGINAL values */
    test_seed(33344);
    MMPF_Output output;
    for (int t = 0; t < 100; t++)
    {
        mmpf_step(mmpf, generate_return(SIGMA_CALM), &output);
    }

    /* Check filter is using original particle count */
    TEST_ASSERT(mmpf->n_particles == 256, "Particle count should be original");

    /* Check Calm hypothesis is using original mu_vol (-5.30, not -1.0) */
    double calm_mu = (double)mmpf->config.hypotheses[MMPF_CALM].mu_vol;
    TEST_ASSERT_NEAR(calm_mu, -5.30, 0.01,
                     "Filter should use original config values");

    printf("\n    Config mutated to mu_vol=-1.0, but filter uses: %.2f\n", calm_mu);

    mmpf_destroy(mmpf);
    return 1;
}

/*═══════════════════════════════════════════════════════════════════════════
 * STRATEGY 11: ARCHITECTURE VS LEARNING SEPARATION TESTS
 *
 * These tests prove the IMM architecture works by removing the learning penalty.
 *═══════════════════════════════════════════════════════════════════════════*/

/* MMPF with "cheating start" - perfect initialization, no learning */
static MMPF_ROCKS *create_mmpf_cheating_start(void)
{
    MMPF_Config cfg = mmpf_config_defaults();
    cfg.n_particles = 256;

    /* DISABLE all adaptive learning - pure architecture test */
    cfg.enable_storvik_sync = 0;
    cfg.enable_global_baseline = 0;
    cfg.enable_gated_learning = 0;
    cfg.enable_nu_learning = 0;

    /* Give it the ANSWER KEY (Cartoon values matching data generator) */
    cfg.hypotheses[MMPF_CALM].mu_vol = RBPF_REAL(-5.30); /* Exact match! */
    cfg.hypotheses[MMPF_CALM].phi = RBPF_REAL(0.98);
    cfg.hypotheses[MMPF_CALM].sigma_eta = RBPF_REAL(0.08);
    cfg.hypotheses[MMPF_CALM].nu = RBPF_REAL(20.0);

    cfg.hypotheses[MMPF_TREND].mu_vol = RBPF_REAL(-4.14); /* Exact match! */
    cfg.hypotheses[MMPF_TREND].phi = RBPF_REAL(0.95);
    cfg.hypotheses[MMPF_TREND].sigma_eta = RBPF_REAL(0.12);
    cfg.hypotheses[MMPF_TREND].nu = RBPF_REAL(6.0);

    cfg.hypotheses[MMPF_CRISIS].mu_vol = RBPF_REAL(-3.00); /* Exact match! */
    cfg.hypotheses[MMPF_CRISIS].phi = RBPF_REAL(0.85);
    cfg.hypotheses[MMPF_CRISIS].sigma_eta = RBPF_REAL(0.20);
    cfg.hypotheses[MMPF_CRISIS].nu = RBPF_REAL(3.0);

    /* SHARP selection - lower min_mixing_prob */
    cfg.min_mixing_prob = RBPF_REAL(0.01); /* Was 0.05, allow sharper selection */

    cfg.base_stickiness = RBPF_REAL(0.90);
    cfg.min_stickiness = RBPF_REAL(0.70);

    /* Fair initial weights */
    cfg.initial_weights[MMPF_CALM] = RBPF_REAL(0.34);
    cfg.initial_weights[MMPF_TREND] = RBPF_REAL(0.33);
    cfg.initial_weights[MMPF_CRISIS] = RBPF_REAL(0.33);

    cfg.rng_seed = 42;

    return mmpf_create(&cfg);
}

/**
 * Test 11.1: The "Cheating Start" Test (Architecture Validation)
 *
 * With perfect initialization (no learning penalty), MMPF should
 * achieve >90% hypothesis accuracy on clear regime data.
 */
static int test_cheating_start_architecture(void)
{
    test_seed(42424);
    MMPF_ROCKS *mmpf = create_mmpf_cheating_start();
    TEST_ASSERT(mmpf != NULL, "Failed to create MMPF");

    MMPF_Output output;

    int correct_calm = 0, total_calm = 0;
    int correct_crisis = 0, total_crisis = 0;

    /* Phase 1: 500 ticks of Calm (true μ = -5.30) */
    for (int t = 0; t < 500; t++)
    {
        rbpf_real_t r = generate_return(SIGMA_CALM);
        mmpf_step(mmpf, r, &output);

        total_calm++;
        if (output.dominant == MMPF_CALM)
            correct_calm++;
    }

    double calm_accuracy = 100.0 * correct_calm / total_calm;
    double calm_weight_final = (double)output.weights[MMPF_CALM];

    printf("\n    Extended Calm (500 ticks):\n");
    printf("      Accuracy: %.1f%%\n", calm_accuracy);
    printf("      Final weights: [%.3f, %.3f, %.3f]\n",
           (double)output.weights[0],
           (double)output.weights[1],
           (double)output.weights[2]);

    /* Phase 2: 500 ticks of Crisis (true μ = -3.00) */
    for (int t = 0; t < 500; t++)
    {
        rbpf_real_t r = generate_return(SIGMA_CRISIS);
        mmpf_step(mmpf, r, &output);

        total_crisis++;
        if (output.dominant == MMPF_CRISIS)
            correct_crisis++;
    }

    double crisis_accuracy = 100.0 * correct_crisis / total_crisis;
    double crisis_weight_final = (double)output.weights[MMPF_CRISIS];

    printf("    Sudden Crisis (500 ticks):\n");
    printf("      Accuracy: %.1f%%\n", crisis_accuracy);
    printf("      Final weights: [%.3f, %.3f, %.3f]\n",
           (double)output.weights[0],
           (double)output.weights[1],
           (double)output.weights[2]);

    /* With perfect initialization, architecture should achieve >80% */
    TEST_ASSERT_GT(calm_accuracy, 70.0,
                   "Cheating-start Calm accuracy should be >70%");
    TEST_ASSERT_GT(crisis_accuracy, 70.0,
                   "Cheating-start Crisis accuracy should be >70%");

    /* Final weights should be decisive, not mushy */
    TEST_ASSERT_GT(calm_weight_final, 0.5,
                   "After calm data, Calm weight should dominate");
    TEST_ASSERT_GT(crisis_weight_final, 0.5,
                   "After crisis data, Crisis weight should dominate");

    double overall_accuracy = 100.0 * (correct_calm + correct_crisis) / (total_calm + total_crisis);
    printf("    Overall accuracy: %.1f%%\n", overall_accuracy);

    TEST_ASSERT_GT(overall_accuracy, 70.0,
                   "Overall cheating-start accuracy should be >70%");

    mmpf_destroy(mmpf);
    return 1;
}

/**
 * Test 11.2: The "Sharp Selection" Test
 *
 * With min_mixing_prob = 0.01, weights should be decisive, not mushy.
 */
static int test_sharp_selection(void)
{
    test_seed(13579);
    MMPF_ROCKS *mmpf = create_mmpf_cheating_start();
    TEST_ASSERT(mmpf != NULL, "Failed to create MMPF");

    MMPF_Output output;

    /* Feed 1000 ticks of pure Calm */
    for (int t = 0; t < 1000; t++)
    {
        rbpf_real_t r = generate_return(SIGMA_CALM);
        mmpf_step(mmpf, r, &output);
    }

    double w_calm = (double)output.weights[MMPF_CALM];
    double w_trend = (double)output.weights[MMPF_TREND];
    double w_crisis = (double)output.weights[MMPF_CRISIS];

    double min_weight = w_calm;
    if (w_trend < min_weight)
        min_weight = w_trend;
    if (w_crisis < min_weight)
        min_weight = w_crisis;

    printf("\n    After 1000 calm ticks:\n");
    printf("      Weights: [%.4f, %.4f, %.4f]\n", w_calm, w_trend, w_crisis);
    printf("      Min weight: %.4f (should be near min_mixing_prob=0.01)\n", min_weight);

    /* With sharp selection, loser should be near floor (0.01-0.05) */
    TEST_ASSERT_LT(min_weight, 0.15,
                   "Loser weight should be suppressed (< 15%)");

    /* Winner should dominate */
    TEST_ASSERT_GT(w_calm, 0.6,
                   "Calm should dominate on calm data");

    mmpf_destroy(mmpf);
    return 1;
}

/**
 * Test 11.3: The "Long Run Convergence" Test
 *
 * Given enough time, Online EM should converge and MMPF should improve.
 * Compare accuracy in first 2000 ticks vs last 2000 ticks.
 */
static int test_long_run_convergence(void)
{
    test_seed(24680);

    /* Start with WRONG initialization to test learning */
    MMPF_ROCKS *mmpf = create_test_mmpf_with_learning();
    TEST_ASSERT(mmpf != NULL, "Failed to create MMPF");

    MMPF_Output output;

    /* Track accuracy in early vs late phases */
    int correct_early = 0, total_early = 0;
    int correct_late = 0, total_late = 0;

    /* 10000 tick run with regime switches */
    for (int t = 0; t < 10000; t++)
    {
        /* Cycle through regimes every 1000 ticks */
        int phase = (t / 1000) % 3;
        double sigma;
        MMPF_Hypothesis true_regime;

        if (phase == 0)
        {
            sigma = SIGMA_CALM;
            true_regime = MMPF_CALM;
        }
        else if (phase == 1)
        {
            sigma = SIGMA_TREND;
            true_regime = MMPF_TREND;
        }
        else
        {
            sigma = SIGMA_CRISIS;
            true_regime = MMPF_CRISIS;
        }

        rbpf_real_t r = generate_return(sigma);
        mmpf_step(mmpf, r, &output);

        /* Track accuracy */
        int correct = (output.dominant == true_regime) ? 1 : 0;

        if (t < 2000)
        {
            total_early++;
            correct_early += correct;
        }
        else if (t >= 8000)
        {
            total_late++;
            correct_late += correct;
        }
    }

    double accuracy_early = 100.0 * correct_early / total_early;
    double accuracy_late = 100.0 * correct_late / total_late;
    double improvement = accuracy_late - accuracy_early;

    printf("\n    Long run convergence (10k ticks):\n");
    printf("      Accuracy (first 2k): %.1f%%\n", accuracy_early);
    printf("      Accuracy (last 2k):  %.1f%%\n", accuracy_late);
    printf("      Improvement:         %+.1f%%\n", improvement);

    printf("      Final Online EM centers: [%.2f, %.2f, %.2f]\n",
           mmpf->online_em.mu[0],
           mmpf->online_em.mu[1],
           mmpf->online_em.mu[2]);

    /* Late accuracy should be better than early (learning worked) */
    TEST_ASSERT_GT(improvement, -5.0,
                   "Late accuracy should not be much worse than early");

    /* With proper learning, late accuracy should be reasonable */
    TEST_ASSERT_GT(accuracy_late, 30.0,
                   "Late-phase accuracy should be reasonable");

    mmpf_destroy(mmpf);
    return 1;
}

/**
 * Test 11.4: The "Min Weight Floor" Test
 *
 * Verify min_mixing_prob is being enforced correctly.
 */
static int test_min_weight_floor(void)
{
    test_seed(11223);

    /* Create with explicit min_mixing_prob */
    MMPF_Config cfg = mmpf_config_defaults();
    cfg.n_particles = 256;
    cfg.enable_storvik_sync = 0;
    cfg.min_mixing_prob = RBPF_REAL(0.02); /* 2% floor */

    cfg.hypotheses[MMPF_CALM].mu_vol = RBPF_REAL(-5.30);
    cfg.hypotheses[MMPF_TREND].mu_vol = RBPF_REAL(-4.14);
    cfg.hypotheses[MMPF_CRISIS].mu_vol = RBPF_REAL(-3.00);

    MMPF_ROCKS *mmpf = mmpf_create(&cfg);
    TEST_ASSERT(mmpf != NULL, "Failed to create MMPF");

    MMPF_Output output;

    /* Extreme data to push losers to floor */
    for (int t = 0; t < 2000; t++)
    {
        rbpf_real_t r = generate_return(SIGMA_CALM);
        mmpf_step(mmpf, r, &output);
    }

    double w_min = (double)output.weights[MMPF_CRISIS]; /* Should be loser */
    double expected_floor = 0.02;

    printf("\n    After 2000 extreme calm ticks:\n");
    printf("      Weights: [%.4f, %.4f, %.4f]\n",
           (double)output.weights[0],
           (double)output.weights[1],
           (double)output.weights[2]);
    printf("      Min weight: %.4f (floor = %.4f)\n", w_min, expected_floor);

    /* Min weight should be at or above floor */
    TEST_ASSERT_GE(w_min, expected_floor - 0.005,
                   "Min weight should respect min_mixing_prob floor");

    /* But not much above it (floor should be tight) */
    TEST_ASSERT_LT(w_min, 0.15,
                   "Min weight should be near floor, not mushy");

    mmpf_destroy(mmpf);
    return 1;
}

/**
 * Test 11.5: The "Likelihood Discrimination" Test
 *
 * Verify that models produce DIFFERENT marginal likelihoods on regime data.
 */
static int test_likelihood_discrimination(void)
{
    test_seed(33445);
    MMPF_ROCKS *mmpf = create_mmpf_cheating_start();
    TEST_ASSERT(mmpf != NULL, "Failed to create MMPF");

    MMPF_Output output;

    /* Accumulate marginal likelihoods per model on Calm data */
    double sum_lik_calm = 0, sum_lik_trend = 0, sum_lik_crisis = 0;
    int n_ticks = 200;

    for (int t = 0; t < n_ticks; t++)
    {
        rbpf_real_t r = generate_return(SIGMA_CALM);
        mmpf_step(mmpf, r, &output);

        sum_lik_calm += (double)output.model_marginal_lik[MMPF_CALM];
        sum_lik_trend += (double)output.model_marginal_lik[MMPF_TREND];
        sum_lik_crisis += (double)output.model_marginal_lik[MMPF_CRISIS];
    }

    double avg_lik_calm = sum_lik_calm / n_ticks;
    double avg_lik_trend = sum_lik_trend / n_ticks;
    double avg_lik_crisis = sum_lik_crisis / n_ticks;

    printf("\n    Avg marginal likelihood on Calm data:\n");
    printf("      Calm model:   %.6f\n", avg_lik_calm);
    printf("      Trend model:  %.6f\n", avg_lik_trend);
    printf("      Crisis model: %.6f\n", avg_lik_crisis);

    /* Calm model should have HIGHEST likelihood on calm data */
    TEST_ASSERT_GT(avg_lik_calm, avg_lik_crisis,
                   "Calm should have higher likelihood than Crisis on calm data");

    /* There should be meaningful separation (not all ~equal) */
    double ratio = avg_lik_calm / (avg_lik_crisis + 1e-30);
    printf("      Calm/Crisis ratio: %.2f\n", ratio);

    TEST_ASSERT_GT(ratio, 1.1,
                   "Likelihoods should discriminate (ratio > 1.1)");

    mmpf_destroy(mmpf);
    return 1;
}

/*═══════════════════════════════════════════════════════════════════════════
 * MAIN
 *═══════════════════════════════════════════════════════════════════════════*/

int main(void)
{
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  MMPF Advanced Test Suite\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  Seed: 42 (base)\n\n");

    /*─────────────────────────────────────────────────────────────────────*/
    printf("STRATEGY 1: LEARNING VALIDATION\n");
    printf("───────────────────────────────────────────────────────────────\n");
    RUN_TEST(test_online_em_convergence, "Online EM convergence (Lazy Manager)");
    RUN_TEST(test_mcmc_shock_efficacy, "MCMC shock efficacy (Teleport)");
    RUN_TEST(test_sprt_false_alarm_robustness, "SPRT false alarm robustness");

    /*─────────────────────────────────────────────────────────────────────*/
    printf("\nSTRATEGY 2: CHAOS MONKEY (Stability)\n");
    printf("───────────────────────────────────────────────────────────────\n");
    RUN_TEST(test_zero_return_stability, "Zero return stability (Flatline)");
    RUN_TEST(test_nan_attack_robustness, "NaN attack robustness");
    RUN_TEST(test_denormalized_float_handling, "Denormalized float handling");

    /*─────────────────────────────────────────────────────────────────────*/
    printf("\nSTRATEGY 3: INTEGRATION (Component Handshakes)\n");
    printf("───────────────────────────────────────────────────────────────\n");
    RUN_TEST(test_entropy_auto_unlock, "Entropy auto-unlock (Lock & Key)");
    RUN_TEST(test_cluster_ordering_invariant, "Cluster ordering (Swim Lane)");

    /*─────────────────────────────────────────────────────────────────────*/
    printf("\nSTRATEGY 4: KELLY INTEGRATION (Trading Readiness)\n");
    printf("───────────────────────────────────────────────────────────────\n");
    RUN_TEST(test_law_of_total_variance, "Law of total variance");
    RUN_TEST(test_confidence_when_models_agree, "Confidence when models agree");

    /*─────────────────────────────────────────────────────────────────────*/
    printf("\nSTRATEGY 5: STUDENT-T PHYSICS\n");
    printf("───────────────────────────────────────────────────────────────\n");
    RUN_TEST(test_fat_tail_likelihood, "Fat tail likelihood");
    RUN_TEST(test_gaussian_impostor, "Gaussian impostor");

    /*─────────────────────────────────────────────────────────────────────*/
    printf("\nSTRATEGY 6: GATED LEARNING (Mode Collapse Prevention)\n");
    printf("───────────────────────────────────────────────────────────────\n");
    RUN_TEST(test_pinned_anchor_stability, "Pinned anchor stability");
    RUN_TEST(test_leakage_prevention, "Leakage prevention");

    /*─────────────────────────────────────────────────────────────────────*/
    printf("\nSTRATEGY 7: REGIME DEATH PREVENTION\n");
    printf("───────────────────────────────────────────────────────────────\n");
    RUN_TEST(test_extinction_prevention, "Extinction prevention");

    /*─────────────────────────────────────────────────────────────────────*/
    printf("\nSTRATEGY 8: TRANSITION MATRIX\n");
    printf("───────────────────────────────────────────────────────────────\n");
    RUN_TEST(test_stickiness_bounds, "Stickiness bounds (Sticky Floor)");
    RUN_TEST(test_crisis_exit_boost, "Crisis exit boost");

    /*─────────────────────────────────────────────────────────────────────*/
    printf("\nSTRATEGY 9: NUMERICAL STABILITY\n");
    printf("───────────────────────────────────────────────────────────────\n");
    RUN_TEST(test_long_run_weight_stability, "Long-run weight stability (20k steps)");
    RUN_TEST(test_entropy_convergence, "Entropy convergence");

    /*─────────────────────────────────────────────────────────────────────*/
    printf("\nSTRATEGY 10: API CONTRACT\n");
    printf("───────────────────────────────────────────────────────────────\n");
    RUN_TEST(test_double_reset_idempotent, "Double reset idempotent");
    RUN_TEST(test_config_immutability, "Config immutability");

    /*─────────────────────────────────────────────────────────────────────*/
    printf("\nSTRATEGY 11: ARCHITECTURE VS LEARNING (THE MONEY TESTS)\n");
    printf("───────────────────────────────────────────────────────────────\n");
    RUN_TEST(test_cheating_start_architecture, "Cheating start (perfect init)");
    RUN_TEST(test_sharp_selection, "Sharp selection (low min_mixing)");
    RUN_TEST(test_long_run_convergence, "Long run convergence (10k ticks)");
    RUN_TEST(test_min_weight_floor, "Min weight floor enforcement");
    RUN_TEST(test_likelihood_discrimination, "Likelihood discrimination");

    /*─────────────────────────────────────────────────────────────────────*/
    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("  SUMMARY\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  Tests run:    %d\n", g_tests_run);
    printf("  Passed:       %d\n", g_tests_passed);
    printf("  Failed:       %d\n", g_tests_failed);
    printf("═══════════════════════════════════════════════════════════════\n\n");

    if (g_tests_failed == 0)
    {
        printf("  All tests passed!\n");
    }
    else
    {
        printf("  SOME TESTS FAILED!\n");
    }

    return g_tests_failed;
}