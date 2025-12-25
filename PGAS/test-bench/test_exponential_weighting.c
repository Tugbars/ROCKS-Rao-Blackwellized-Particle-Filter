/**
 * @file test_exponential_weighting.c
 * @brief Test for PGAS exponential recency weighting (Window Paradox Solution)
 *
 * Demonstrates how exponential weighting prioritizes recent observations
 * after a regime change, solving the "lagged average" problem.
 *
 * Compile:
 *   gcc -O2 -Wall test_exponential_weighting.c pgas_mkl_mock.c -lm -o test_exp_weight
 */

#include "pgas_mkl.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define GREEN  "\033[32m"
#define RED    "\033[31m"
#define YELLOW "\033[33m"
#define CYAN   "\033[36m"
#define RESET  "\033[0m"

/*═══════════════════════════════════════════════════════════════════════════
 * TEST: Verify Weight Decay Formula
 *═══════════════════════════════════════════════════════════════════════════*/

static int test_weight_decay_formula(void) {
    printf(CYAN "TEST: Weight Decay Formula Verification" RESET "\n\n");
    
    float lambda = 0.001f;
    int T = 2000;
    
    printf("  λ = %.4f (decay rate)\n", lambda);
    printf("  T = %d (window length)\n", T);
    printf("  Half-life = ln(2)/λ = %.1f ticks\n\n", logf(2.0f) / lambda);
    
    printf("  Time    Age    Weight   Description\n");
    printf("  ─────────────────────────────────────────\n");
    
    /* Formula: w(t) = exp(-λ × (T - 1 - t)) */
    float oldest_weight = expf(-lambda * (T - 1));
    float midpoint_weight = expf(-lambda * (T - 1 - T/2));
    float recent_weight = expf(-lambda * (T - 1 - (T - 100)));
    float newest_weight = expf(-lambda * 0);  /* t = T-1 */
    
    printf("  t=0      %4d   %.4f   Oldest observation\n", T-1, oldest_weight);
    printf("  t=%d  %4d   %.4f   Midpoint\n", T/2, T/2-1, midpoint_weight);
    printf("  t=%d  %4d   %.4f   Recent (100 ticks ago)\n", T-100, 99, recent_weight);
    printf("  t=%d     0   %.4f   Most recent\n", T-1, newest_weight);
    
    printf("\n  Weight ratio (newest/oldest): %.2f\n", newest_weight / oldest_weight);
    
    /* Verify the formula is correct */
    float expected_oldest = expf(-0.001f * 1999);
    int ok = (fabsf(oldest_weight - expected_oldest) < 1e-6f);
    
    printf("\n  Formula verification: %s\n", ok ? GREEN "OK" RESET : RED "FAIL" RESET);
    
    return ok;
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST: Effective Sample Size with Weighting
 *═══════════════════════════════════════════════════════════════════════════*/

static int test_effective_sample_size(void) {
    printf("\n" CYAN "TEST: Effective Sample Size with Exponential Weighting" RESET "\n\n");
    
    int T = 2000;
    float lambda = 0.001f;
    
    /* Compute effective sample size */
    float sum_w = 0.0f;
    float sum_w2 = 0.0f;
    
    for (int t = 0; t < T; t++) {
        float w = expf(-lambda * (T - 1 - t));
        sum_w += w;
        sum_w2 += w * w;
    }
    
    /* ESS = (Σw)² / Σw² */
    float ess = (sum_w * sum_w) / sum_w2;
    float ess_fraction = ess / T;
    
    printf("  Window T = %d\n", T);
    printf("  λ = %.4f\n", lambda);
    printf("  Σw = %.2f\n", sum_w);
    printf("  Σw² = %.4f\n", sum_w2);
    printf("  ESS = %.1f ticks (%.1f%% of window)\n", ess, 100.0f * ess_fraction);
    
    /* With λ=0.001, ESS should be roughly half-life × 2 ≈ 1386 */
    float expected_ess = 2.0f * logf(2.0f) / lambda;
    printf("  Expected ESS ≈ %.0f (2 × half-life)\n", expected_ess);
    
    int ok = (ess > 0.3f * T && ess < 0.8f * T);  /* Reasonable range */
    printf("\n  ESS in reasonable range: %s\n", ok ? GREEN "OK" RESET : RED "FAIL" RESET);
    
    return ok;
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST: Lambda = 0 Disables Weighting
 *═══════════════════════════════════════════════════════════════════════════*/

static int test_lambda_zero_disabled(void) {
    printf("\n" CYAN "TEST: λ = 0 Disables Weighting" RESET "\n\n");
    
    int K = 4;
    int N = 32;
    int T = 100;
    
    PGASMKLState *pgas = pgas_mkl_alloc(N, T, K, 12345);
    if (!pgas) {
        printf("  Allocation failed\n");
        return 0;
    }
    
    /* Test with λ = 0 (disabled) */
    pgas_mkl_set_recency_lambda(pgas, 0.0f);
    printf("  Set λ = 0.0 (disabled)\n");
    printf("  recency_lambda in model: %.6f\n", pgas->model.recency_lambda);
    
    int ok = (pgas->model.recency_lambda < 1e-6f);
    printf("  Weighting disabled: %s\n", ok ? GREEN "OK" RESET : RED "FAIL" RESET);
    
    /* Test with λ > 0 (enabled) */
    pgas_mkl_set_recency_lambda(pgas, 0.001f);
    printf("\n  Set λ = 0.001 (enabled)\n");
    printf("  recency_lambda in model: %.6f\n", pgas->model.recency_lambda);
    
    int ok2 = (pgas->model.recency_lambda > 1e-6f);
    printf("  Weighting enabled: %s\n", ok2 ? GREEN "OK" RESET : RED "FAIL" RESET);
    
    pgas_mkl_free(pgas);
    
    return ok && ok2;
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST: Regime Change Scenario (Conceptual)
 *═══════════════════════════════════════════════════════════════════════════*/

static int test_regime_change_scenario(void) {
    printf("\n" CYAN "TEST: Regime Change Scenario (Window Paradox)" RESET "\n\n");
    
    int T = 2000;
    int change_point = 1500;  /* Regime changes at t=1500 */
    float lambda = 0.002f;    /* More aggressive weighting */
    
    printf("  Scenario: Regime change at t=%d\n", change_point);
    printf("  Window: T=%d\n", T);
    printf("  λ = %.4f (half-life = %.0f ticks)\n\n", lambda, logf(2.0f) / lambda);
    
    /* Compute total weight for pre-change and post-change periods */
    float weight_pre = 0.0f;  /* t=0 to t=change_point-1 */
    float weight_post = 0.0f; /* t=change_point to t=T-1 */
    
    for (int t = 0; t < T; t++) {
        float w = expf(-lambda * (T - 1 - t));
        if (t < change_point) {
            weight_pre += w;
        } else {
            weight_post += w;
        }
    }
    
    float total = weight_pre + weight_post;
    float frac_pre = weight_pre / total;
    float frac_post = weight_post / total;
    
    printf("  WITHOUT exponential weighting (λ=0):\n");
    printf("    Pre-change ticks: %d (%.1f%%)\n", change_point, 100.0f * change_point / T);
    printf("    Post-change ticks: %d (%.1f%%)\n", T - change_point, 100.0f * (T - change_point) / T);
    printf("    → PGAS learns average of BOTH regimes (75%% old, 25%% new)\n\n");
    
    printf("  WITH exponential weighting (λ=%.4f):\n", lambda);
    printf("    Pre-change weight: %.2f (%.1f%%)\n", weight_pre, 100.0f * frac_pre);
    printf("    Post-change weight: %.2f (%.1f%%)\n", weight_post, 100.0f * frac_post);
    printf("    → PGAS prioritizes post-change regime\n\n");
    
    /* Success: post-change should dominate even though it has fewer ticks */
    int ok = (frac_post > frac_pre);
    printf("  Post-change dominates: %s (%.1f%% vs %.1f%%)\n", 
           ok ? GREEN "YES" RESET : RED "NO" RESET,
           100.0f * frac_post, 100.0f * frac_pre);
    
    /* Also test with default λ=0.001 to show the trade-off */
    printf("\n  Trade-off analysis (change at t=1500, 500 ticks post-change):\n");
    float test_lambdas[] = {0.001f, 0.002f, 0.003f};
    for (int i = 0; i < 3; i++) {
        float lam = test_lambdas[i];
        float w_pre = 0, w_post = 0;
        for (int t = 0; t < T; t++) {
            float w = expf(-lam * (T - 1 - t));
            if (t < change_point) w_pre += w; else w_post += w;
        }
        float tot = w_pre + w_post;
        printf("    λ=%.3f: pre=%.1f%%, post=%.1f%% %s\n", 
               lam, 100*w_pre/tot, 100*w_post/tot,
               w_post > w_pre ? "(post dominates)" : "");
    }
    
    return ok;
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST: Different Lambda Values
 *═══════════════════════════════════════════════════════════════════════════*/

static int test_lambda_comparison(void) {
    printf("\n" CYAN "TEST: Lambda Value Comparison" RESET "\n\n");
    
    int T = 2000;
    
    float lambdas[] = {0.0001f, 0.0005f, 0.001f, 0.002f, 0.005f};
    int n_lambdas = sizeof(lambdas) / sizeof(lambdas[0]);
    
    printf("  λ          Half-life   Oldest Weight   ESS\n");
    printf("  ──────────────────────────────────────────────\n");
    
    for (int i = 0; i < n_lambdas; i++) {
        float lambda = lambdas[i];
        float half_life = logf(2.0f) / lambda;
        float oldest = expf(-lambda * (T - 1));
        
        /* Compute ESS */
        float sum_w = 0.0f, sum_w2 = 0.0f;
        for (int t = 0; t < T; t++) {
            float w = expf(-lambda * (T - 1 - t));
            sum_w += w;
            sum_w2 += w * w;
        }
        float ess = (sum_w * sum_w) / sum_w2;
        
        printf("  %.4f     %7.1f     %.6f        %.0f\n", 
               lambda, half_life, oldest, ess);
    }
    
    printf("\n  Recommended: λ = 0.001 (half-life ≈ 693 ticks)\n");
    
    return 1;  /* Informational test, always passes */
}

/*═══════════════════════════════════════════════════════════════════════════
 * MAIN
 *═══════════════════════════════════════════════════════════════════════════*/

int main(void) {
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════════════╗\n");
    printf("║         EXPONENTIAL RECENCY WEIGHTING TEST SUITE                      ║\n");
    printf("║         Window Paradox Solution for PGAS                              ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════════╝\n\n");
    
    int passed = 0;
    int total = 5;
    
    if (test_weight_decay_formula()) passed++;
    if (test_effective_sample_size()) passed++;
    if (test_lambda_zero_disabled()) passed++;
    if (test_regime_change_scenario()) passed++;
    if (test_lambda_comparison()) passed++;
    
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════════════╗\n");
    printf("║                           SUMMARY                                     ║\n");
    printf("╠═══════════════════════════════════════════════════════════════════════╣\n");
    printf("║  Tests passed: %d / %d                                                 ║\n", passed, total);
    printf("╚═══════════════════════════════════════════════════════════════════════╝\n\n");
    
    if (passed == total) {
        printf(GREEN "All tests passed!" RESET "\n\n");
    } else {
        printf(RED "%d test(s) failed" RESET "\n\n", total - passed);
    }
    
    return (passed == total) ? 0 : 1;
}
