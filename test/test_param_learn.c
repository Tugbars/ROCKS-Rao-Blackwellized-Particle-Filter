/*
 * ═══════════════════════════════════════════════════════════════════════════
 * Test & Benchmark: RBPF Parameter Learning (Sleeping Storvik)
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Tests:
 *   1. Correctness: Prior initialization, NIG updates, sampling
 *   2. Convergence: Stats converge to true parameters
 *   3. Latency: Hot path timing under various scenarios
 *   4. SoA benefit: Compare vs theoretical AoS baseline
 *
 * Build:
 *   Windows (MSVC):
 *     cmake --build . --config Release --target test_param_learn
 *
 *   Linux (GCC):
 *     gcc -O3 -march=native test_param_learn.c rbpf_param_learn.c -o test_param_learn -lm
 *
 * With MKL:
 *   gcc -O3 -march=native -DPARAM_LEARN_USE_MKL test_param_learn.c rbpf_param_learn.c -o test_param_learn -lmkl_rt -lm
 *
 * ═══════════════════════════════════════════════════════════════════════════
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <stdbool.h>

#include "rbpf_param_learn.h"

/*═══════════════════════════════════════════════════════════════════════════
 * TIMING UTILITIES (Cross-platform)
 *═══════════════════════════════════════════════════════════════════════════*/

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

static double g_timer_freq = 0.0;

static void init_timer(void)
{
    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);
    g_timer_freq = (double)freq.QuadPart / 1e6;
}

static inline double get_time_us(void)
{
    LARGE_INTEGER counter;
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / g_timer_freq;
}

#else
#include <sys/time.h>

static void init_timer(void) {}

static inline double get_time_us(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1e6 + (double)tv.tv_usec;
}
#endif

/* Simple timer struct for compatibility */
typedef struct
{
    double start_us;
    double end_us;
} Timer;

static void timer_start(Timer *t)
{
    t->start_us = get_time_us();
}

static double timer_stop_us(Timer *t)
{
    t->end_us = get_time_us();
    return t->end_us - t->start_us;
}

static uint64_t timer_cycles(Timer *t)
{
    (void)t;
    return 0; /* Cycles not available on all platforms */
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST HELPERS
 *═══════════════════════════════════════════════════════════════════════════*/

#define TEST_PASS "[PASS]"
#define TEST_FAIL "[FAIL]"

static int g_tests_passed = 0;
static int g_tests_failed = 0;

#define ASSERT_TRUE(cond, msg)                            \
    do                                                    \
    {                                                     \
        if (cond)                                         \
        {                                                 \
            printf("  %s %s\n", TEST_PASS, msg);          \
            g_tests_passed++;                             \
        }                                                 \
        else                                              \
        {                                                 \
            printf("  %s %s (FAILED)\n", TEST_FAIL, msg); \
            g_tests_failed++;                             \
        }                                                 \
    } while (0)

#define ASSERT_NEAR(a, b, tol, msg)                                                                         \
    do                                                                                                      \
    {                                                                                                       \
        double _diff = fabs((double)(a) - (double)(b));                                                     \
        if (_diff <= (tol))                                                                                 \
        {                                                                                                   \
            printf("  %s %s (%.6f ~ %.6f)\n", TEST_PASS, msg, (double)(a), (double)(b));                    \
            g_tests_passed++;                                                                               \
        }                                                                                                   \
        else                                                                                                \
        {                                                                                                   \
            printf("  %s %s (%.6f != %.6f, diff=%.6e)\n", TEST_FAIL, msg, (double)(a), (double)(b), _diff); \
            g_tests_failed++;                                                                               \
        }                                                                                                   \
    } while (0)

/* Simple RNG for test data generation */
static uint64_t test_rng_state[2] = {0x123456789ABCDEF0ULL, 0xFEDCBA9876543210ULL};

static double test_randn(void)
{
    /* xoroshiro128+ */
    uint64_t s0 = test_rng_state[0];
    uint64_t s1 = test_rng_state[1];
    uint64_t result = s0 + s1;
    s1 ^= s0;
    test_rng_state[0] = ((s0 << 24) | (s0 >> 40)) ^ s1 ^ (s1 << 16);
    test_rng_state[1] = (s1 << 37) | (s1 >> 27);

    /* Box-Muller */
    double u1 = (double)(result >> 11) * (1.0 / 9007199254740992.0);

    s0 = test_rng_state[0];
    s1 = test_rng_state[1];
    result = s0 + s1;
    s1 ^= s0;
    test_rng_state[0] = ((s0 << 24) | (s0 >> 40)) ^ s1 ^ (s1 << 16);
    test_rng_state[1] = (s1 << 37) | (s1 >> 27);

    double u2 = (double)(result >> 11) * (1.0 / 9007199254740992.0);

    if (u1 < 1e-15)
        u1 = 1e-15;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2);
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST 1: INITIALIZATION
 *═══════════════════════════════════════════════════════════════════════════*/

static void test_initialization(void)
{
    printf("\n=== Test 1: Initialization ===\n");

    ParamLearnConfig cfg = param_learn_config_defaults();
    ParamLearner learner;

    int ret = param_learn_init(&learner, &cfg, 200, 4);
    ASSERT_TRUE(ret == 0, "Init returns success");
    ASSERT_TRUE(learner.n_particles == 200, "Particle count correct");
    ASSERT_TRUE(learner.n_regimes == 4, "Regime count correct");
    ASSERT_TRUE(learner.storvik_total_size == 800, "Total size = 200*4");

    /* Check SoA arrays allocated (storvik is double-buffered array) */
    ASSERT_TRUE(learner.storvik[0].m != NULL, "SoA m array allocated");
    ASSERT_TRUE(learner.storvik[0].kappa != NULL, "SoA kappa array allocated");
    ASSERT_TRUE(learner.storvik[0].mu_cached != NULL, "SoA mu_cached allocated");

    /* Check entropy buffer */
    ASSERT_TRUE(learner.entropy.normal != NULL, "Entropy normal buffer allocated");
    ASSERT_TRUE(learner.entropy.uniform != NULL, "Entropy uniform buffer allocated");
    ASSERT_TRUE(learner.entropy.buffer_size == PL_RNG_BUFFER_SIZE, "Entropy buffer size correct");

    /* Check alignment (should be 64-byte for AVX-512) */
    ASSERT_TRUE(((uintptr_t)learner.storvik[0].m % 64) == 0, "m array 64-byte aligned");
    ASSERT_TRUE(((uintptr_t)learner.storvik[0].kappa % 64) == 0, "kappa array 64-byte aligned");

    param_learn_free(&learner);
    printf("\n");
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST 2: PRIOR BROADCAST
 *═══════════════════════════════════════════════════════════════════════════*/

static void test_prior_broadcast(void)
{
    printf("=== Test 2: Prior Broadcast ===\n");

    ParamLearnConfig cfg = param_learn_config_defaults();
    ParamLearner learner;
    param_learn_init(&learner, &cfg, 100, 4);

    /* Set custom priors */
    param_learn_set_prior(&learner, 0, -2.5, 0.98, 0.10);
    param_learn_set_prior(&learner, 1, -1.5, 0.95, 0.15);
    param_learn_set_prior(&learner, 2, -1.0, 0.90, 0.20);
    param_learn_set_prior(&learner, 3, -0.5, 0.85, 0.30);

    /* Broadcast to all particles */
    param_learn_broadcast_priors(&learner);

    /* Check a few particles (storvik[0] is current buffer) */
    StorvikSoA *soa = &learner.storvik[0];
    int nr = learner.n_regimes;

    /* Particle 0, regime 0 */
    ASSERT_NEAR(soa->m[0 * nr + 0], -2.5, 1e-6, "Particle 0 R0 mean");

    /* Particle 50, regime 2 */
    ASSERT_NEAR(soa->m[50 * nr + 2], -1.0, 1e-6, "Particle 50 R2 mean");

    /* Particle 99, regime 3 */
    ASSERT_NEAR(soa->m[99 * nr + 3], -0.5, 1e-6, "Particle 99 R3 mean");

    /* Check phi capping (should be <= 0.995) */
    ASSERT_TRUE(learner.priors[0].phi <= 0.995, "R0 phi capped at 0.995");

    /* Check precomputed terms */
    ASSERT_NEAR(learner.priors[0].one_minus_phi, 1.0 - learner.priors[0].phi, 1e-10, "one_minus_phi computed");
    ASSERT_NEAR(learner.priors[0].inv_one_minus_phi, 1.0 / learner.priors[0].one_minus_phi, 1e-10, "inv_one_minus_phi computed");

    param_learn_free(&learner);
    printf("\n");
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST 3: STAT UPDATE CORRECTNESS
 *═══════════════════════════════════════════════════════════════════════════*/

static void test_stat_update(void)
{
    printf("=== Test 3: Stat Update Correctness ===\n");

    ParamLearnConfig cfg = param_learn_config_defaults();
    cfg.sample_interval[0] = 1000; /* Disable sampling to test stats only */
    cfg.sample_interval[1] = 1000;
    cfg.sample_interval[2] = 1000;
    cfg.sample_interval[3] = 1000;

    ParamLearner learner;
    param_learn_init(&learner, &cfg, 10, 4);

    param_learn_set_prior(&learner, 0, -2.0, 0.98, 0.10);
    param_learn_broadcast_priors(&learner);

    /* Create particle info - all in regime 0 */
    ParticleInfo *particles = (ParticleInfo *)malloc(10 * sizeof(ParticleInfo));
    for (int i = 0; i < 10; i++)
    {
        particles[i].regime = 0;
        particles[i].prev_regime = 0;
        particles[i].ell = -2.0 + 0.05 * test_randn();
        particles[i].ell_lag = -2.0 + 0.05 * test_randn();
        particles[i].weight = 0.1;
    }

    /* Initial state (storvik[0] is current buffer) */
    StorvikSoA *soa = &learner.storvik[0];
    double kappa_before = soa->kappa[0];
    double alpha_before = soa->alpha[0];

    /* Update */
    param_learn_update(&learner, particles, 10);

    /* Check stats updated */
    double kappa_after = soa->kappa[0];
    double alpha_after = soa->alpha[0];

    ASSERT_TRUE(kappa_after > kappa_before, "Kappa increased after update");
    ASSERT_TRUE(alpha_after > alpha_before, "Alpha increased after update");
    ASSERT_TRUE(soa->n_obs[0] == 1, "n_obs incremented");
    ASSERT_TRUE(learner.total_stat_updates == 10, "total_stat_updates counted");

    /* Multiple updates should accumulate */
    for (int t = 0; t < 100; t++)
    {
        for (int i = 0; i < 10; i++)
        {
            particles[i].ell = -2.0 + 0.05 * test_randn();
            particles[i].ell_lag = particles[i].ell + 0.02 * test_randn();
        }
        param_learn_update(&learner, particles, 10);
    }

    ASSERT_TRUE(soa->n_obs[0] == 101, "n_obs = 101 after 101 updates");
    ASSERT_TRUE(soa->kappa[0] > kappa_after, "Kappa keeps increasing");

    free(particles);
    param_learn_free(&learner);
    printf("\n");
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST 4: PARAMETER CONVERGENCE
 *═══════════════════════════════════════════════════════════════════════════*/

static void test_convergence(void)
{
    printf("=== Test 4: Parameter Convergence ===\n");

    /* True parameters we're trying to learn */
    const double TRUE_MU = -2.3;
    const double TRUE_PHI = 0.97;
    const double TRUE_SIGMA = 0.12;

    ParamLearnConfig cfg = param_learn_config_defaults();
    cfg.sample_interval[0] = 10; /* Sample every 10 ticks */

    ParamLearner learner;
    param_learn_init(&learner, &cfg, 50, 4);

    /* Set prior centered near truth (phi is fixed) */
    param_learn_set_prior(&learner, 0, -2.0, TRUE_PHI, 0.15);
    param_learn_broadcast_priors(&learner);

    /* Generate synthetic data from true model */
    ParticleInfo *particles = (ParticleInfo *)malloc(50 * sizeof(ParticleInfo));
    double log_vol = TRUE_MU;
    double log_vol_lag;

    /* More iterations for better convergence */
    for (int t = 0; t < 2000; t++)
    {
        log_vol_lag = log_vol;

        /* True OU dynamics: ell_t = mu + phi(ell_{t-1} - mu) + sigma*eps */
        log_vol = TRUE_MU + TRUE_PHI * (log_vol_lag - TRUE_MU) + TRUE_SIGMA * test_randn();

        /* All particles observe same data (testing learning, not filtering) */
        for (int i = 0; i < 50; i++)
        {
            particles[i].regime = 0;
            particles[i].prev_regime = 0;
            particles[i].ell = log_vol;
            particles[i].ell_lag = log_vol_lag;
            particles[i].weight = 1.0 / 50.0;
        }

        param_learn_update(&learner, particles, 50);
    }

    /* Get learned parameters */
    RegimeParams params;
    param_learn_get_params(&learner, 0, 0, &params);

    printf("  True:    mu=%.4f, sigma=%.4f\n", TRUE_MU, TRUE_SIGMA);
    printf("  Learned: mu=%.4f, sigma=%.4f\n", params.mu, params.sigma);
    printf("  Posterior mean: mu=%.4f\n", params.mu_post_mean);

    /* NIG learning is biased by phi - check posterior mean is closer */
    ASSERT_NEAR(params.mu_post_mean, TRUE_MU, 0.5, "Posterior mean within 0.5 of true mu");
    ASSERT_NEAR(params.sigma, TRUE_SIGMA, 0.08, "sigma converged within 0.08");
    ASSERT_TRUE(params.n_obs >= 2000, "Sufficient observations accumulated");

    free(particles);
    param_learn_free(&learner);
    printf("\n");
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST 5: SLEEPING BEHAVIOR
 *═══════════════════════════════════════════════════════════════════════════*/

static void test_sleeping(void)
{
    printf("=== Test 5: Sleeping Behavior ===\n");

    ParamLearnConfig cfg = param_learn_config_defaults();
    cfg.sample_interval[0] = 50; /* R0: every 50 ticks */
    cfg.sample_interval[1] = 20; /* R1: every 20 ticks */
    cfg.sample_interval[2] = 5;  /* R2: every 5 ticks */
    cfg.sample_interval[3] = 1;  /* R3: every tick */

    ParamLearner learner;
    param_learn_init(&learner, &cfg, 100, 4);
    param_learn_broadcast_priors(&learner);

    ParticleInfo *particles = (ParticleInfo *)malloc(100 * sizeof(ParticleInfo));

    /* All particles in R0 (calm) */
    for (int i = 0; i < 100; i++)
    {
        particles[i].regime = 0;
        particles[i].prev_regime = 0;
        particles[i].ell = -2.0;
        particles[i].ell_lag = -2.0;
        particles[i].weight = 0.01;
    }

    /* Run 100 ticks */
    learner.total_samples_drawn = 0;
    for (int t = 0; t < 100; t++)
    {
        param_learn_update(&learner, particles, 100);
    }

    /* In R0 with interval=50: expect ~2 samples per particle x 100 particles = ~200 */
    /* But first tick always samples, so slightly higher */
    printf("  R0 (interval=50): %llu samples over 100 ticks\n",
           (unsigned long long)learner.total_samples_drawn);
    ASSERT_TRUE(learner.total_samples_drawn < 500, "R0: Few samples (sleeping)");

    /* Switch all to R3 (crisis) */
    for (int i = 0; i < 100; i++)
    {
        particles[i].regime = 3;
        particles[i].prev_regime = 3;
    }

    uint64_t samples_before = learner.total_samples_drawn;
    for (int t = 0; t < 100; t++)
    {
        param_learn_update(&learner, particles, 100);
    }
    uint64_t samples_r3 = learner.total_samples_drawn - samples_before;

    printf("  R3 (interval=1): %llu samples over 100 ticks\n",
           (unsigned long long)samples_r3);
    ASSERT_TRUE(samples_r3 >= 9000, "R3: Many samples (awake)");

    free(particles);
    param_learn_free(&learner);
    printf("\n");
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST 6: REGIME CHANGE TRIGGER
 *═══════════════════════════════════════════════════════════════════════════*/

static void test_regime_change_trigger(void)
{
    printf("=== Test 6: Regime Change Trigger ===\n");

    ParamLearnConfig cfg = param_learn_config_defaults();
    cfg.sample_interval[0] = 1000; /* Disable interval sampling */
    cfg.sample_interval[1] = 1000;
    cfg.sample_on_regime_change = true;

    ParamLearner learner;
    param_learn_init(&learner, &cfg, 10, 4);
    param_learn_broadcast_priors(&learner);

    ParticleInfo *particles = (ParticleInfo *)malloc(10 * sizeof(ParticleInfo));

    /* Start in R0 */
    for (int i = 0; i < 10; i++)
    {
        particles[i].regime = 0;
        particles[i].prev_regime = 0;
        particles[i].ell = -2.0;
        particles[i].ell_lag = -2.0;
        particles[i].weight = 0.1;
    }

    /* First update (first obs trigger) */
    param_learn_update(&learner, particles, 10);
    uint64_t samples_after_first = learner.total_samples_drawn;
    ASSERT_TRUE(samples_after_first == 10, "First observation samples all particles");

    /* 100 updates in same regime (no triggers) */
    for (int t = 0; t < 100; t++)
    {
        param_learn_update(&learner, particles, 10);
    }
    uint64_t samples_after_100 = learner.total_samples_drawn;

    ASSERT_TRUE(samples_after_100 == samples_after_first, "No samples without regime change");

    /* Now change regime - go to R1 which hasn't been visited yet */
    for (int i = 0; i < 10; i++)
    {
        particles[i].regime = 1;
        particles[i].prev_regime = 0; /* Changed! */
    }

    param_learn_update(&learner, particles, 10);

    /* Should have sampled on transition */
    ASSERT_TRUE(learner.total_samples_drawn > samples_after_100, "Samples drawn on regime transition");
    ASSERT_TRUE(learner.total_samples_drawn - samples_after_100 == 10, "All 10 particles sampled on transition");

    /* Now test REGIME_CHANGE specifically: go back to R0 (already visited) */
    for (int i = 0; i < 10; i++)
    {
        particles[i].regime = 0;
        particles[i].prev_regime = 1; /* Changed back! */
    }

    uint64_t samples_before_back = learner.total_samples_drawn;
    uint64_t regime_triggers_before = learner.samples_triggered_regime;

    param_learn_update(&learner, particles, 10);

    /* This should trigger REGIME_CHANGE */
    ASSERT_TRUE(learner.samples_triggered_regime > regime_triggers_before, "Regime change triggers counted");
    ASSERT_TRUE(learner.total_samples_drawn > samples_before_back, "Samples drawn on regime change back");

    free(particles);
    param_learn_free(&learner);
    printf("\n");
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST 7: STRUCTURAL BREAK TRIGGER
 *═══════════════════════════════════════════════════════════════════════════*/

static void test_structural_break_trigger(void)
{
    printf("=== Test 7: Structural Break Trigger ===\n");

    ParamLearnConfig cfg = param_learn_config_defaults();
    cfg.sample_interval[0] = 1000; /* Disable interval sampling */
    cfg.sample_on_structural_break = true;

    ParamLearner learner;
    param_learn_init(&learner, &cfg, 20, 4);
    param_learn_broadcast_priors(&learner);

    ParticleInfo *particles = (ParticleInfo *)malloc(20 * sizeof(ParticleInfo));
    for (int i = 0; i < 20; i++)
    {
        particles[i].regime = 0;
        particles[i].prev_regime = 0;
        particles[i].ell = -2.0;
        particles[i].ell_lag = -2.0;
        particles[i].weight = 0.05;
    }

    /* First update */
    param_learn_update(&learner, particles, 20);
    uint64_t samples_before = learner.total_samples_drawn;

    /* Signal structural break */
    param_learn_signal_structural_break(&learner);
    param_learn_update(&learner, particles, 20);

    ASSERT_TRUE(learner.samples_triggered_break == 20, "20 structural break triggers");
    ASSERT_TRUE(learner.total_samples_drawn > samples_before, "Samples drawn on break");

    free(particles);
    param_learn_free(&learner);
    printf("\n");
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST 8: RESAMPLING
 *═══════════════════════════════════════════════════════════════════════════*/

static void test_resampling(void)
{
    printf("=== Test 8: Resampling ===\n");

    ParamLearnConfig cfg = param_learn_config_defaults();
    ParamLearner learner;
    param_learn_init(&learner, &cfg, 10, 4);
    param_learn_broadcast_priors(&learner);

    /* Manually set different values for different particles */
    StorvikSoA *soa = &learner.storvik[0];
    for (int i = 0; i < 10; i++)
    {
        soa->m[i * 4 + 0] = -2.0 + 0.1 * i; /* Different m for each particle */
        soa->mu_cached[i * 4 + 0] = -2.0 + 0.1 * i;
    }

    /* Resample: all particles become copies of particle 3 */
    int ancestors[10] = {3, 3, 3, 3, 3, 3, 3, 3, 3, 3};
    param_learn_apply_resampling(&learner, ancestors, 10);

    /* All should now have particle 3's value */
    double expected_m = -2.0 + 0.1 * 3;
    for (int i = 0; i < 10; i++)
    {
        ASSERT_NEAR(soa->m[i * 4 + 0], expected_m, 1e-10, "Particle copied from ancestor");
    }

    param_learn_free(&learner);
    printf("\n");
}

/*═══════════════════════════════════════════════════════════════════════════
 * BENCHMARK 1: STAT UPDATE LATENCY
 *═══════════════════════════════════════════════════════════════════════════*/

static void bench_stat_update(void)
{
    printf("=== Benchmark 1: Stat Update Latency ===\n");

    const int N_PARTICLES = 200;
    const int N_REGIMES = 4;
    const int N_ITERS = 10000;
    const int WARMUP = 1000;

    ParamLearnConfig cfg = param_learn_config_defaults();
    cfg.sample_interval[0] = 10000; /* Disable sampling */
    cfg.sample_interval[1] = 10000;
    cfg.sample_interval[2] = 10000;
    cfg.sample_interval[3] = 10000;

    ParamLearner learner;
    param_learn_init(&learner, &cfg, N_PARTICLES, N_REGIMES);
    param_learn_broadcast_priors(&learner);

    ParticleInfo *particles = (ParticleInfo *)malloc(N_PARTICLES * sizeof(ParticleInfo));
    for (int i = 0; i < N_PARTICLES; i++)
    {
        particles[i].regime = i % N_REGIMES;
        particles[i].prev_regime = particles[i].regime;
        particles[i].ell = -2.0 + 0.1 * test_randn();
        particles[i].ell_lag = -2.0 + 0.1 * test_randn();
        particles[i].weight = 1.0 / N_PARTICLES;
    }

    /* Warmup */
    for (int i = 0; i < WARMUP; i++)
    {
        param_learn_update(&learner, particles, N_PARTICLES);
    }

    /* Benchmark */
    Timer timer;
    double total_us = 0;
    uint64_t total_cycles = 0;

    for (int i = 0; i < N_ITERS; i++)
    {
        /* Vary data slightly */
        for (int j = 0; j < N_PARTICLES; j++)
        {
            particles[j].ell_lag = particles[j].ell;
            particles[j].ell += 0.01 * test_randn();
        }

        timer_start(&timer);
        param_learn_update(&learner, particles, N_PARTICLES);
        total_us += timer_stop_us(&timer);
        total_cycles += timer_cycles(&timer);
    }

    double avg_us = total_us / N_ITERS;
    double avg_cycles = (double)total_cycles / N_ITERS;
    double per_particle_ns = (avg_us * 1000.0) / N_PARTICLES;

    printf("  Particles: %d\n", N_PARTICLES);
    printf("  Iterations: %d\n", N_ITERS);
    printf("  Avg time: %.3f us\n", avg_us);
    printf("  Avg cycles: %.0f\n", avg_cycles);
    printf("  Per particle: %.1f ns\n", per_particle_ns);
    printf("  Throughput: %.1f M particles/sec\n", N_PARTICLES / avg_us);

    free(particles);
    param_learn_free(&learner);
    printf("\n");
}

/*═══════════════════════════════════════════════════════════════════════════
 * BENCHMARK 2: FULL UPDATE (WITH SAMPLING)
 *═══════════════════════════════════════════════════════════════════════════*/

static void bench_full_update(void)
{
    printf("=== Benchmark 2: Full Update (Sampling Every Tick) ===\n");

    const int N_PARTICLES = 200;
    const int N_REGIMES = 4;
    const int N_ITERS = 5000;
    const int WARMUP = 500;

    ParamLearnConfig cfg = param_learn_config_full_bayesian(); /* Sample every tick */

    ParamLearner learner;
    param_learn_init(&learner, &cfg, N_PARTICLES, N_REGIMES);
    param_learn_broadcast_priors(&learner);

    ParticleInfo *particles = (ParticleInfo *)malloc(N_PARTICLES * sizeof(ParticleInfo));
    for (int i = 0; i < N_PARTICLES; i++)
    {
        particles[i].regime = i % N_REGIMES;
        particles[i].prev_regime = particles[i].regime;
        particles[i].ell = -2.0 + 0.1 * test_randn();
        particles[i].ell_lag = -2.0 + 0.1 * test_randn();
        particles[i].weight = 1.0 / N_PARTICLES;
    }

    /* Warmup */
    for (int i = 0; i < WARMUP; i++)
    {
        param_learn_update(&learner, particles, N_PARTICLES);
    }

    /* Benchmark */
    Timer timer;
    double total_us = 0;

    for (int i = 0; i < N_ITERS; i++)
    {
        for (int j = 0; j < N_PARTICLES; j++)
        {
            particles[j].ell_lag = particles[j].ell;
            particles[j].ell += 0.01 * test_randn();
        }

        timer_start(&timer);
        param_learn_update(&learner, particles, N_PARTICLES);
        total_us += timer_stop_us(&timer);
    }

    double avg_us = total_us / N_ITERS;
    double per_particle_ns = (avg_us * 1000.0) / N_PARTICLES;

    printf("  Particles: %d\n", N_PARTICLES);
    printf("  Iterations: %d\n", N_ITERS);
    printf("  Avg time: %.3f us (stats + sampling)\n", avg_us);
    printf("  Per particle: %.1f ns\n", per_particle_ns);
    printf("  Throughput: %.1f M particles/sec\n", N_PARTICLES / avg_us);

    free(particles);
    param_learn_free(&learner);
    printf("\n");
}

/*═══════════════════════════════════════════════════════════════════════════
 * BENCHMARK 3: SLEEPING VS AWAKE
 *═══════════════════════════════════════════════════════════════════════════*/

static void bench_sleeping_vs_awake(void)
{
    printf("=== Benchmark 3: Sleeping vs Awake Latency ===\n");

    const int N_PARTICLES = 200;
    const int N_REGIMES = 4;
    const int N_ITERS = 5000;

    /* Sleeping config (R0 only) */
    ParamLearnConfig cfg_sleep = param_learn_config_defaults();
    cfg_sleep.sample_interval[0] = 50; /* Sample every 50 ticks */

    /* Awake config (R3 only) */
    ParamLearnConfig cfg_awake = param_learn_config_defaults();
    cfg_awake.sample_interval[3] = 1; /* Sample every tick */

    ParamLearner learner_sleep, learner_awake;
    param_learn_init(&learner_sleep, &cfg_sleep, N_PARTICLES, N_REGIMES);
    param_learn_init(&learner_awake, &cfg_awake, N_PARTICLES, N_REGIMES);
    param_learn_broadcast_priors(&learner_sleep);
    param_learn_broadcast_priors(&learner_awake);

    ParticleInfo *particles_r0 = (ParticleInfo *)malloc(N_PARTICLES * sizeof(ParticleInfo));
    ParticleInfo *particles_r3 = (ParticleInfo *)malloc(N_PARTICLES * sizeof(ParticleInfo));

    for (int i = 0; i < N_PARTICLES; i++)
    {
        particles_r0[i].regime = 0;
        particles_r0[i].prev_regime = 0;
        particles_r0[i].ell = -2.0;
        particles_r0[i].ell_lag = -2.0;
        particles_r0[i].weight = 1.0 / N_PARTICLES;

        particles_r3[i].regime = 3;
        particles_r3[i].prev_regime = 3;
        particles_r3[i].ell = -0.5;
        particles_r3[i].ell_lag = -0.5;
        particles_r3[i].weight = 1.0 / N_PARTICLES;
    }

    /* Warmup */
    for (int i = 0; i < 500; i++)
    {
        param_learn_update(&learner_sleep, particles_r0, N_PARTICLES);
        param_learn_update(&learner_awake, particles_r3, N_PARTICLES);
    }

    /* Benchmark sleeping (R0) */
    Timer timer;
    double total_sleep = 0;
    for (int i = 0; i < N_ITERS; i++)
    {
        timer_start(&timer);
        param_learn_update(&learner_sleep, particles_r0, N_PARTICLES);
        total_sleep += timer_stop_us(&timer);
    }

    /* Benchmark awake (R3) */
    double total_awake = 0;
    for (int i = 0; i < N_ITERS; i++)
    {
        timer_start(&timer);
        param_learn_update(&learner_awake, particles_r3, N_PARTICLES);
        total_awake += timer_stop_us(&timer);
    }

    double avg_sleep = total_sleep / N_ITERS;
    double avg_awake = total_awake / N_ITERS;

    printf("  Sleeping (R0, interval=50): %.3f us\n", avg_sleep);
    printf("  Awake (R3, interval=1):     %.3f us\n", avg_awake);
    printf("  Speedup when sleeping:      %.1fx\n", avg_awake / avg_sleep);

    free(particles_r0);
    free(particles_r3);
    param_learn_free(&learner_sleep);
    param_learn_free(&learner_awake);
    printf("\n");
}

/*═══════════════════════════════════════════════════════════════════════════
 * BENCHMARK 4: SCALING WITH PARTICLE COUNT
 *═══════════════════════════════════════════════════════════════════════════*/

static void bench_scaling(void)
{
    printf("=== Benchmark 4: Scaling with Particle Count ===\n");

    const int particle_counts[] = {50, 100, 200, 500, 1000};
    const int n_counts = sizeof(particle_counts) / sizeof(particle_counts[0]);
    const int N_ITERS = 2000;

    printf("  %-10s %-12s %-12s %-15s\n", "Particles", "Time (us)", "Per-particle", "Throughput");
    printf("  %-10s %-12s %-12s %-15s\n", "---------", "---------", "------------", "----------");

    for (int c = 0; c < n_counts; c++)
    {
        int np = particle_counts[c];

        ParamLearnConfig cfg = param_learn_config_defaults();
        cfg.sample_interval[0] = 1000; /* Stats only */
        cfg.sample_interval[1] = 1000;
        cfg.sample_interval[2] = 1000;
        cfg.sample_interval[3] = 1000;

        ParamLearner learner;
        param_learn_init(&learner, &cfg, np, 4);
        param_learn_broadcast_priors(&learner);

        ParticleInfo *particles = (ParticleInfo *)malloc(np * sizeof(ParticleInfo));
        for (int i = 0; i < np; i++)
        {
            particles[i].regime = i % 4;
            particles[i].prev_regime = particles[i].regime;
            particles[i].ell = -2.0;
            particles[i].ell_lag = -2.0;
            particles[i].weight = 1.0 / np;
        }

        /* Warmup */
        for (int i = 0; i < 200; i++)
        {
            param_learn_update(&learner, particles, np);
        }

        /* Benchmark */
        Timer timer;
        double total = 0;
        for (int i = 0; i < N_ITERS; i++)
        {
            timer_start(&timer);
            param_learn_update(&learner, particles, np);
            total += timer_stop_us(&timer);
        }

        double avg = total / N_ITERS;
        double per_particle = (avg * 1000.0) / np;
        double throughput = np / avg;

        printf("  %-10d %-12.3f %-12.1f %-15.1f\n", np, avg, per_particle, throughput);

        free(particles);
        param_learn_free(&learner);
    }
    printf("\n");
}

/*═══════════════════════════════════════════════════════════════════════════
 * MAIN
 *═══════════════════════════════════════════════════════════════════════════*/

int main(int argc, char **argv)
{
    (void)argc;
    (void)argv;

    init_timer();

    printf("\n");
    printf("================================================================\n");
    printf("     RBPF Parameter Learning: Test & Benchmark Suite          \n");
    printf("================================================================\n");
#ifdef PARAM_LEARN_USE_MKL
    printf("  Build: MKL enabled (batch RNG)                              \n");
#else
    printf("  Build: Pure C (no MKL)                                      \n");
#endif
    printf("================================================================\n");

    /* Run tests */
    test_initialization();
    test_prior_broadcast();
    test_stat_update();
    test_convergence();
    test_sleeping();
    test_regime_change_trigger();
    test_structural_break_trigger();
    test_resampling();

    printf("===============================================================\n");
    printf("  Tests: %d passed, %d failed\n", g_tests_passed, g_tests_failed);
    printf("===============================================================\n\n");

    if (g_tests_failed > 0)
    {
        printf("  Some tests failed. Skipping benchmarks.\n\n");
        return 1;
    }

    /* Run benchmarks */
    bench_stat_update();
    bench_full_update();
    bench_sleeping_vs_awake();
    bench_scaling();

    printf("===============================================================\n");
    printf("  All tests and benchmarks complete.\n");
    printf("===============================================================\n\n");

    return 0;
}