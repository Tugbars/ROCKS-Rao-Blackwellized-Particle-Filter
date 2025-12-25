/**
 * @file test_thompson_sampler.c
 * @brief Test for Thompson Sampling Handoff
 *
 * Tests:
 *   - Gamma distribution sampling
 *   - Dirichlet distribution sampling
 *   - Explore/exploit decision logic
 *   - Row sum threshold behavior
 *   - Floor probability enforcement
 *
 * Compile:
 *   gcc -O2 -Wall test_thompson_sampler.c thompson_sampler.c -lm -o test_thompson
 */

#include "thompson_sampler.h"
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
    
    ThompsonSamplerConfig cfg = thompson_sampler_config_defaults(4);
    
    printf("  n_regimes:         %d (expected: 4)\n", cfg.n_regimes);
    printf("  exploit_threshold: %.1f (expected: 500.0)\n", cfg.exploit_threshold);
    printf("  min_concentration: %.2f (expected: 0.10)\n", cfg.min_concentration);
    printf("  floor_probability: %.1e (expected: 1e-5)\n", cfg.floor_probability);
    
    bool ok = (cfg.n_regimes == 4) &&
              (fabsf(cfg.exploit_threshold - 500.0f) < 0.1f) &&
              (fabsf(cfg.min_concentration - 0.1f) < 0.01f) &&
              (cfg.floor_probability > 0.0f);
    
    return ok;
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST: Initialization
 *═══════════════════════════════════════════════════════════════════════════*/

static int test_initialization(void) {
    printf("Testing initialization\n\n");
    
    ThompsonSampler sampler;
    ThompsonSamplerConfig cfg = thompson_sampler_config_defaults(4);
    
    int ret = thompson_sampler_init(&sampler, &cfg);
    
    printf("  Init return:     %d (expected: 0)\n", ret);
    printf("  Initialized:     %s\n", sampler.initialized ? "true" : "false");
    printf("  Total samples:   %d (expected: 0)\n", sampler.total_samples);
    
    bool ok = (ret == 0) && sampler.initialized && (sampler.total_samples == 0);
    
    thompson_sampler_free(&sampler);
    return ok;
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST: Gamma Sampling
 *═══════════════════════════════════════════════════════════════════════════*/

static int test_gamma_sampling(void) {
    printf("Testing gamma distribution sampling\n\n");
    
    ThompsonSampler sampler;
    ThompsonSamplerConfig cfg = thompson_sampler_config_defaults(4);
    thompson_sampler_init(&sampler, &cfg);
    
    /* Test shape >= 1 (Marsaglia-Tsang) */
    printf("  Testing Gamma(2.0, 1.0) - 1000 samples...\n");
    float sum = 0.0f, sum_sq = 0.0f;
    int N = 1000;
    
    for (int i = 0; i < N; i++) {
        float x = thompson_sampler_gamma(&sampler, 2.0f, 1.0f);
        sum += x;
        sum_sq += x * x;
    }
    
    float mean = sum / N;
    float var = sum_sq / N - mean * mean;
    
    /* Gamma(k, 1) has mean=k, var=k */
    printf("  Sample mean: %.3f (expected: ~2.0)\n", mean);
    printf("  Sample var:  %.3f (expected: ~2.0)\n", var);
    
    bool shape_ge1_ok = (fabsf(mean - 2.0f) < 0.3f) && (fabsf(var - 2.0f) < 0.5f);
    printf("  Shape >= 1: %s\n", shape_ge1_ok ? GREEN "OK" RESET : RED "FAIL" RESET);
    
    /* Test shape < 1 (Ahrens-Dieter) */
    printf("\n  Testing Gamma(0.5, 1.0) - 1000 samples...\n");
    sum = 0.0f; sum_sq = 0.0f;
    
    for (int i = 0; i < N; i++) {
        float x = thompson_sampler_gamma(&sampler, 0.5f, 1.0f);
        sum += x;
        sum_sq += x * x;
    }
    
    mean = sum / N;
    var = sum_sq / N - mean * mean;
    
    printf("  Sample mean: %.3f (expected: ~0.5)\n", mean);
    printf("  Sample var:  %.3f (expected: ~0.5)\n", var);
    
    bool shape_lt1_ok = (fabsf(mean - 0.5f) < 0.15f) && (fabsf(var - 0.5f) < 0.25f);
    printf("  Shape < 1: %s\n", shape_lt1_ok ? GREEN "OK" RESET : RED "FAIL" RESET);
    
    thompson_sampler_free(&sampler);
    return shape_ge1_ok && shape_lt1_ok;
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST: Dirichlet Sampling
 *═══════════════════════════════════════════════════════════════════════════*/

static int test_dirichlet_sampling(void) {
    printf("Testing Dirichlet distribution sampling\n\n");
    
    ThompsonSampler sampler;
    ThompsonSamplerConfig cfg = thompson_sampler_config_defaults(4);
    thompson_sampler_init(&sampler, &cfg);
    
    int K = 4;
    float alpha[4] = {10.0f, 20.0f, 30.0f, 40.0f};  /* Sum = 100 */
    float out[4];
    
    /* Expected means: alpha[i] / sum(alpha) */
    float expected[4] = {0.1f, 0.2f, 0.3f, 0.4f};
    
    /* Sample many times and compute mean */
    int N = 1000;
    float means[4] = {0};
    
    printf("  Sampling Dir([10, 20, 30, 40]) %d times...\n", N);
    
    for (int i = 0; i < N; i++) {
        thompson_sampler_dirichlet(&sampler, alpha, K, out);
        
        /* Check sum to 1 */
        float sum = 0.0f;
        for (int j = 0; j < K; j++) {
            sum += out[j];
            means[j] += out[j];
        }
        
        if (fabsf(sum - 1.0f) > 1e-5f) {
            printf("  " RED "ERROR: Sample doesn't sum to 1 (sum=%.6f)" RESET "\n", sum);
            thompson_sampler_free(&sampler);
            return 0;
        }
    }
    
    /* Compute sample means */
    for (int j = 0; j < K; j++) {
        means[j] /= N;
    }
    
    printf("  Expected means: [%.2f, %.2f, %.2f, %.2f]\n",
           expected[0], expected[1], expected[2], expected[3]);
    printf("  Sample means:   [%.2f, %.2f, %.2f, %.2f]\n",
           means[0], means[1], means[2], means[3]);
    
    /* Check means are close to expected */
    bool means_ok = true;
    for (int j = 0; j < K; j++) {
        if (fabsf(means[j] - expected[j]) > 0.05f) {
            means_ok = false;
        }
    }
    
    printf("  Means match: %s\n", means_ok ? GREEN "OK" RESET : RED "FAIL" RESET);
    
    thompson_sampler_free(&sampler);
    return means_ok;
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST: Explore/Exploit Decision
 *═══════════════════════════════════════════════════════════════════════════*/

static int test_explore_exploit_decision(void) {
    printf("Testing explore/exploit decision based on row sums\n\n");
    
    ThompsonSampler sampler;
    ThompsonSamplerConfig cfg = thompson_sampler_config_defaults(4);
    cfg.exploit_threshold = 100.0f;  /* Lower for testing */
    thompson_sampler_init(&sampler, &cfg);
    
    int K = 4;
    float Q[THOMPSON_MAX_REGIMES][THOMPSON_MAX_REGIMES];
    float Pi[THOMPSON_MAX_REGIMES][THOMPSON_MAX_REGIMES];
    
    /* Test 1: Low counts - should explore */
    printf("  Test 1: Low counts (row sum ~ 40) → should EXPLORE\n");
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < K; j++) {
            Q[i][j] = (i == j) ? 30.0f : 10.0f / (K - 1);  /* Row sum ≈ 40 */
        }
    }
    
    ThompsonSampleResult r1 = thompson_sampler_sample(&sampler, Q, Pi);
    printf("    Min row sum: %.1f\n", r1.min_row_sum);
    printf("    Explored: %s (expected: YES)\n", r1.explored ? "YES" : "NO");
    
    if (!r1.explored) {
        printf("    " RED "ERROR: Should have explored!" RESET "\n");
        thompson_sampler_free(&sampler);
        return 0;
    }
    
    /* Test 2: High counts - should exploit */
    printf("\n  Test 2: High counts (row sum ~ 200) → should EXPLOIT\n");
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < K; j++) {
            Q[i][j] = (i == j) ? 180.0f : 20.0f / (K - 1);  /* Row sum ≈ 200 */
        }
    }
    
    ThompsonSampleResult r2 = thompson_sampler_sample(&sampler, Q, Pi);
    printf("    Min row sum: %.1f\n", r2.min_row_sum);
    printf("    Explored: %s (expected: NO)\n", r2.explored ? "YES" : "NO");
    
    if (r2.explored) {
        printf("    " RED "ERROR: Should have exploited!" RESET "\n");
        thompson_sampler_free(&sampler);
        return 0;
    }
    
    /* Test 3: Mixed - one low row - should explore */
    printf("\n  Test 3: Mixed counts (one low row) → should EXPLORE\n");
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < K; j++) {
            if (i == 0) {
                Q[i][j] = (i == j) ? 30.0f : 10.0f / (K - 1);  /* Low row */
            } else {
                Q[i][j] = (i == j) ? 180.0f : 20.0f / (K - 1);  /* High rows */
            }
        }
    }
    
    ThompsonSampleResult r3 = thompson_sampler_sample(&sampler, Q, Pi);
    printf("    Min row sum: %.1f\n", r3.min_row_sum);
    printf("    Explored: %s (expected: YES - one row below threshold)\n", 
           r3.explored ? "YES" : "NO");
    
    printf("\n  Explore count: %d\n", sampler.explore_count);
    printf("  Exploit count: %d\n", sampler.exploit_count);
    
    thompson_sampler_free(&sampler);
    return r1.explored && !r2.explored && r3.explored;
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST: Simplex Constraint
 *═══════════════════════════════════════════════════════════════════════════*/

static int test_simplex_constraint(void) {
    printf("Testing that output always sums to 1 (simplex constraint)\n\n");
    
    ThompsonSampler sampler;
    ThompsonSamplerConfig cfg = thompson_sampler_config_defaults(4);
    thompson_sampler_init(&sampler, &cfg);
    
    int K = 4;
    float Q[THOMPSON_MAX_REGIMES][THOMPSON_MAX_REGIMES];
    float Pi[THOMPSON_MAX_REGIMES][THOMPSON_MAX_REGIMES];
    
    /* Various Q configurations */
    float test_cases[][4] = {
        {90.0f, 3.33f, 3.33f, 3.34f},   /* Sticky */
        {25.0f, 25.0f, 25.0f, 25.0f},   /* Uniform */
        {1.0f, 1.0f, 1.0f, 97.0f},      /* Extreme */
        {0.1f, 0.1f, 0.1f, 0.1f},       /* Very low */
    };
    int n_cases = 4;
    
    bool all_ok = true;
    
    for (int c = 0; c < n_cases; c++) {
        /* Set up Q */
        for (int i = 0; i < K; i++) {
            for (int j = 0; j < K; j++) {
                Q[i][j] = test_cases[c][j];
            }
        }
        
        /* Sample (force explore to test Dirichlet) */
        thompson_sampler_explore(&sampler, Q, Pi);
        
        /* Check all rows sum to 1 */
        for (int i = 0; i < K; i++) {
            float sum = 0.0f;
            for (int j = 0; j < K; j++) {
                sum += Pi[i][j];
            }
            
            if (fabsf(sum - 1.0f) > 1e-4f) {
                printf("  Case %d, Row %d: sum=%.6f " RED "FAIL" RESET "\n",
                       c, i, sum);
                all_ok = false;
            }
        }
    }
    
    if (all_ok) {
        printf("  All %d cases × %d rows sum to 1.0 ± 1e-4\n", n_cases, K);
    }
    
    thompson_sampler_free(&sampler);
    return all_ok;
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST: Floor Probability
 *═══════════════════════════════════════════════════════════════════════════*/

static int test_floor_probability(void) {
    printf("Testing floor probability enforcement\n\n");
    
    ThompsonSampler sampler;
    ThompsonSamplerConfig cfg = thompson_sampler_config_defaults(4);
    cfg.floor_probability = 0.01f;  /* 1% floor */
    thompson_sampler_init(&sampler, &cfg);
    
    int K = 4;
    float Q[THOMPSON_MAX_REGIMES][THOMPSON_MAX_REGIMES];
    float Pi[THOMPSON_MAX_REGIMES][THOMPSON_MAX_REGIMES];
    
    /* Extreme Q that would produce near-zero probabilities */
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < K; j++) {
            Q[i][j] = (i == j) ? 1000.0f : 0.001f;
        }
    }
    
    /* Force exploit to get deterministic output */
    thompson_sampler_exploit(&sampler, Q, Pi);
    
    printf("  Floor probability: %.2f%%\n", cfg.floor_probability * 100);
    printf("  Checking all probabilities >= floor...\n");
    
    bool all_ok = true;
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < K; j++) {
            if (Pi[i][j] < cfg.floor_probability - 1e-6f) {
                printf("    Pi[%d][%d] = %.6f < %.4f " RED "FAIL" RESET "\n",
                       i, j, Pi[i][j], cfg.floor_probability);
                all_ok = false;
            }
        }
    }
    
    if (all_ok) {
        printf("    All probabilities >= floor " GREEN "OK" RESET "\n");
    }
    
    /* Print sample row */
    printf("\n  Sample row 0: [");
    for (int j = 0; j < K; j++) {
        printf("%.4f%s", Pi[0][j], j < K-1 ? ", " : "");
    }
    printf("]\n");
    
    thompson_sampler_free(&sampler);
    return all_ok;
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST: Exploration Ratio
 *═══════════════════════════════════════════════════════════════════════════*/

static int test_exploration_ratio(void) {
    printf("Testing exploration ratio tracking\n\n");
    
    ThompsonSampler sampler;
    ThompsonSamplerConfig cfg = thompson_sampler_config_defaults(4);
    cfg.exploit_threshold = 100.0f;
    thompson_sampler_init(&sampler, &cfg);
    
    int K = 4;
    float Q_low[THOMPSON_MAX_REGIMES][THOMPSON_MAX_REGIMES];
    float Q_high[THOMPSON_MAX_REGIMES][THOMPSON_MAX_REGIMES];
    float Pi[THOMPSON_MAX_REGIMES][THOMPSON_MAX_REGIMES];
    
    /* Set up low and high Q matrices */
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < K; j++) {
            Q_low[i][j] = (i == j) ? 30.0f : 10.0f / (K - 1);
            Q_high[i][j] = (i == j) ? 180.0f : 20.0f / (K - 1);
        }
    }
    
    /* Do 10 low (explore) + 10 high (exploit) */
    for (int i = 0; i < 10; i++) {
        thompson_sampler_sample(&sampler, Q_low, Pi);
    }
    for (int i = 0; i < 10; i++) {
        thompson_sampler_sample(&sampler, Q_high, Pi);
    }
    
    float ratio = thompson_sampler_get_explore_ratio(&sampler);
    int total = thompson_sampler_get_total_samples(&sampler);
    
    printf("  Total samples: %d (expected: 20)\n", total);
    printf("  Explore count: %d (expected: 10)\n", sampler.explore_count);
    printf("  Exploit count: %d (expected: 10)\n", sampler.exploit_count);
    printf("  Explore ratio: %.1f%% (expected: 50%%)\n", ratio * 100.0f);
    
    bool ok = (total == 20) && 
              (sampler.explore_count == 10) &&
              (sampler.exploit_count == 10) &&
              (fabsf(ratio - 0.5f) < 0.01f);
    
    thompson_sampler_free(&sampler);
    return ok;
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST: Variance Under Exploration
 *═══════════════════════════════════════════════════════════════════════════*/

static int test_exploration_variance(void) {
    printf("Testing that exploration produces variance in samples\n\n");
    
    ThompsonSampler sampler;
    ThompsonSamplerConfig cfg = thompson_sampler_config_defaults(4);
    thompson_sampler_init(&sampler, &cfg);
    
    int K = 4;
    float Q[THOMPSON_MAX_REGIMES][THOMPSON_MAX_REGIMES];
    float Pi[THOMPSON_MAX_REGIMES][THOMPSON_MAX_REGIMES];
    
    /* Moderate counts - should give some variance */
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < K; j++) {
            Q[i][j] = (i == j) ? 10.0f : 3.0f;  /* Row sum = 19 */
        }
    }
    
    /* Sample multiple times and track variance of Pi[0][0] */
    int N = 100;
    float samples[100];
    float sum = 0.0f;
    
    for (int i = 0; i < N; i++) {
        thompson_sampler_explore(&sampler, Q, Pi);
        samples[i] = Pi[0][0];
        sum += samples[i];
    }
    
    float mean = sum / N;
    float var = 0.0f;
    for (int i = 0; i < N; i++) {
        var += (samples[i] - mean) * (samples[i] - mean);
    }
    var /= N;
    
    printf("  Q[0][0] = %.1f, row_sum = %.1f\n", Q[0][0], 19.0f);
    printf("  Expected mean of Pi[0][0]: %.3f\n", 10.0f / 19.0f);
    printf("  Sample mean: %.3f\n", mean);
    printf("  Sample variance: %.4f\n", var);
    printf("  Sample std dev: %.3f\n", sqrtf(var));
    
    /* With alpha=10, beta=9, the variance should be significant */
    bool has_variance = (var > 0.001f);
    printf("\n  Variance exists: %s\n", has_variance ? GREEN "YES" RESET : RED "NO" RESET);
    
    thompson_sampler_free(&sampler);
    return has_variance;
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST: Diagnostics
 *═══════════════════════════════════════════════════════════════════════════*/

static int test_diagnostics(void) {
    printf("Testing diagnostic output\n\n");
    
    ThompsonSampler sampler;
    ThompsonSamplerConfig cfg = thompson_sampler_config_defaults(4);
    thompson_sampler_init(&sampler, &cfg);
    
    /* Do some samples */
    int K = 4;
    float Q[THOMPSON_MAX_REGIMES][THOMPSON_MAX_REGIMES];
    float Pi[THOMPSON_MAX_REGIMES][THOMPSON_MAX_REGIMES];
    
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < K; j++) {
            Q[i][j] = (i == j) ? 50.0f : 10.0f;
        }
    }
    
    for (int i = 0; i < 5; i++) {
        thompson_sampler_sample(&sampler, Q, Pi);
    }
    
    thompson_sampler_print_state(&sampler);
    
    thompson_sampler_free(&sampler);
    return 1;  /* Visual inspection */
}

/*═══════════════════════════════════════════════════════════════════════════
 * MAIN
 *═══════════════════════════════════════════════════════════════════════════*/

int main(void) {
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════════════╗\n");
    printf("║               THOMPSON SAMPLER TEST SUITE                             ║\n");
    printf("║               Explore/Exploit for Π Handoff                           ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════════╝\n");
    
    RUN_TEST(test_config_defaults);
    RUN_TEST(test_initialization);
    RUN_TEST(test_gamma_sampling);
    RUN_TEST(test_dirichlet_sampling);
    RUN_TEST(test_explore_exploit_decision);
    RUN_TEST(test_simplex_constraint);
    RUN_TEST(test_floor_probability);
    RUN_TEST(test_exploration_ratio);
    RUN_TEST(test_exploration_variance);
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
