/**
 * @file test_pgas_confidence.c
 * @brief Tests for PGAS Confidence Metrics
 *
 * Tests the confidence computation and gamma mapping.
 *
 * Compile:
 *   gcc -O2 -Wall test_pgas_confidence.c pgas_confidence.c -lm -o test_conf
 */

#include "pgas_confidence.h"
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
    
    PGASConfidenceConfig cfg = pgas_confidence_config_defaults();
    
    printf("  weight_diversity:   %.2f\n", cfg.weight_diversity);
    printf("  weight_exploration: %.2f\n", cfg.weight_exploration);
    printf("  weight_innovation:  %.2f\n", cfg.weight_innovation);
    printf("  gamma_very_low:     %.2f\n", cfg.gamma_very_low);
    printf("  gamma_medium:       %.2f\n", cfg.gamma_medium);
    printf("  gamma_very_high:    %.2f\n", cfg.gamma_very_high);
    
    float weight_sum = cfg.weight_diversity + cfg.weight_exploration + cfg.weight_innovation;
    printf("\n  Weight sum: %.2f (should be 1.0)\n", weight_sum);
    
    return (fabsf(weight_sum - 1.0f) < 0.01f) &&
           (cfg.gamma_very_low < cfg.gamma_medium) &&
           (cfg.gamma_medium < cfg.gamma_very_high);
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST: Degeneracy Detection
 *═══════════════════════════════════════════════════════════════════════════*/

static int test_degeneracy_detection(void) {
    printf("Testing degeneracy detection\n\n");
    
    PGASConfidence conf;
    
    /* Scenario 1: Degenerate ESS */
    pgas_confidence_compute_raw(
        0.05f,   /* ESS ratio = 5% (below threshold) */
        0.20f,   /* acceptance = 20% */
        0.30f,   /* unique = 30% */
        0.10f,   /* divergence = 10% */
        5,       /* sweeps */
        &conf, NULL);
    
    printf("  Scenario 1: Low ESS (5%%)\n");
    printf("    Degeneracy detected: %s (expected: YES)\n", 
           conf.degeneracy_detected ? "YES" : "NO");
    printf("    Level: %s (expected: VERY_LOW)\n", 
           pgas_confidence_level_str(conf.level));
    printf("    Gamma: %.3f (expected: ~0.01)\n", conf.suggested_gamma);
    
    bool test1 = conf.degeneracy_detected && 
                 (conf.level == PGAS_CONFIDENCE_VERY_LOW) &&
                 (conf.suggested_gamma < 0.02f);
    
    /* Scenario 2: Degenerate unique fraction */
    pgas_confidence_compute_raw(
        0.60f,   /* ESS ratio = 60% (good) */
        0.25f,   /* acceptance = 25% */
        0.05f,   /* unique = 5% (below threshold - collapsed!) */
        0.10f,   /* divergence = 10% */
        5,       /* sweeps */
        &conf, NULL);
    
    printf("\n  Scenario 2: Low unique fraction (5%%)\n");
    printf("    Degeneracy detected: %s (expected: YES)\n", 
           conf.degeneracy_detected ? "YES" : "NO");
    printf("    Level: %s\n", pgas_confidence_level_str(conf.level));
    
    bool test2 = conf.degeneracy_detected;
    
    /* Scenario 3: Healthy run */
    pgas_confidence_compute_raw(
        0.55f,   /* ESS ratio = 55% */
        0.25f,   /* acceptance = 25% */
        0.45f,   /* unique = 45% */
        0.12f,   /* divergence = 12% */
        5,       /* sweeps */
        &conf, NULL);
    
    printf("\n  Scenario 3: Healthy run\n");
    printf("    Degeneracy detected: %s (expected: NO)\n", 
           conf.degeneracy_detected ? "YES" : "NO");
    printf("    Level: %s\n", pgas_confidence_level_str(conf.level));
    
    bool test3 = !conf.degeneracy_detected;
    
    return test1 && test2 && test3;
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST: Reference Dominated Detection
 *═══════════════════════════════════════════════════════════════════════════*/

static int test_reference_dominated(void) {
    printf("Testing reference-dominated detection\n\n");
    
    PGASConfidence conf;
    
    /* Scenario: PGAS stuck on reference path */
    pgas_confidence_compute_raw(
        0.40f,   /* ESS ratio = 40% */
        0.02f,   /* acceptance = 2% (very low!) */
        0.35f,   /* unique = 35% */
        0.01f,   /* divergence = 1% (almost no change) */
        5,       /* sweeps */
        &conf, NULL);
    
    printf("  Low acceptance (2%%) + low divergence (1%%)\n");
    printf("    Reference dominated: %s (expected: YES)\n", 
           conf.reference_dominated ? "YES" : "NO");
    printf("    Exploration score: %.3f\n", conf.exploration_score);
    
    bool test1 = conf.reference_dominated && (conf.exploration_score < 0.1f);
    
    /* Healthy exploration */
    pgas_confidence_compute_raw(
        0.50f,   /* ESS ratio */
        0.25f,   /* acceptance = 25% */
        0.40f,   /* unique */
        0.15f,   /* divergence = 15% */
        5,       /* sweeps */
        &conf, NULL);
    
    printf("\n  Good acceptance (25%%) + divergence (15%%)\n");
    printf("    Reference dominated: %s (expected: NO)\n", 
           conf.reference_dominated ? "YES" : "NO");
    
    bool test2 = !conf.reference_dominated;
    
    return test1 && test2;
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST: Regime Change Detection
 *═══════════════════════════════════════════════════════════════════════════*/

static int test_regime_change_detection(void) {
    printf("Testing regime change detection\n\n");
    
    PGASConfidence conf;
    
    /* Scenario: Large divergence suggests regime change */
    pgas_confidence_compute_raw(
        0.45f,   /* ESS ratio */
        0.30f,   /* acceptance */
        0.40f,   /* unique */
        0.35f,   /* divergence = 35% (above 30% threshold) */
        5,       /* sweeps */
        &conf, NULL);
    
    printf("  High divergence (35%%)\n");
    printf("    Regime change detected: %s (expected: YES)\n", 
           conf.regime_change_detected ? "YES" : "NO");
    printf("    Suggested gamma: %.3f (should be elevated)\n", conf.suggested_gamma);
    
    bool test1 = conf.regime_change_detected && (conf.suggested_gamma >= 0.10f);
    
    /* Normal divergence */
    pgas_confidence_compute_raw(
        0.50f,   /* ESS ratio */
        0.25f,   /* acceptance */
        0.40f,   /* unique */
        0.15f,   /* divergence = 15% */
        5,       /* sweeps */
        &conf, NULL);
    
    printf("\n  Normal divergence (15%%)\n");
    printf("    Regime change detected: %s (expected: NO)\n", 
           conf.regime_change_detected ? "YES" : "NO");
    
    bool test2 = !conf.regime_change_detected;
    
    return test1 && test2;
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST: Gamma Mapping Tiers
 *═══════════════════════════════════════════════════════════════════════════*/

static int test_gamma_tiers(void) {
    printf("Testing gamma mapping across confidence tiers\n\n");
    
    PGASConfidence conf;
    
    /* Tier: VERY_LOW (degeneracy) */
    pgas_confidence_compute_raw(0.05f, 0.20f, 0.08f, 0.10f, 5, &conf, NULL);
    printf("  VERY_LOW (degenerate):\n");
    printf("    Level: %s, Gamma: %.3f\n", 
           pgas_confidence_level_str(conf.level), conf.suggested_gamma);
    bool t1 = (conf.level == PGAS_CONFIDENCE_VERY_LOW) && (conf.suggested_gamma <= 0.02f);
    
    /* Tier: LOW - need really poor metrics but not degenerate */
    pgas_confidence_compute_raw(0.12f, 0.06f, 0.12f, 0.03f, 5, &conf, NULL);
    printf("  LOW:\n");
    printf("    Level: %s, Gamma: %.3f\n", 
           pgas_confidence_level_str(conf.level), conf.suggested_gamma);
    bool t2 = (conf.level == PGAS_CONFIDENCE_LOW);
    
    /* Tier: MEDIUM - modest metrics */
    pgas_confidence_compute_raw(0.20f, 0.10f, 0.20f, 0.08f, 5, &conf, NULL);
    printf("  MEDIUM:\n");
    printf("    Level: %s, Gamma: %.3f\n", 
           pgas_confidence_level_str(conf.level), conf.suggested_gamma);
    bool t3 = (conf.level == PGAS_CONFIDENCE_MEDIUM);
    
    /* Tier: HIGH - good but not excellent */
    pgas_confidence_compute_raw(0.35f, 0.18f, 0.32f, 0.10f, 5, &conf, NULL);
    printf("  HIGH:\n");
    printf("    Level: %s, Gamma: %.3f\n", 
           pgas_confidence_level_str(conf.level), conf.suggested_gamma);
    bool t4 = (conf.level == PGAS_CONFIDENCE_HIGH);
    
    /* Tier: VERY_HIGH - excellent metrics */
    pgas_confidence_compute_raw(0.70f, 0.35f, 0.55f, 0.15f, 5, &conf, NULL);
    printf("  VERY_HIGH:\n");
    printf("    Level: %s, Gamma: %.3f\n", 
           pgas_confidence_level_str(conf.level), conf.suggested_gamma);
    bool t5 = (conf.level == PGAS_CONFIDENCE_VERY_HIGH) && (conf.suggested_gamma >= 0.10f);
    
    return t1 && t2 && t3 && t4 && t5;
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST: Score Computation
 *═══════════════════════════════════════════════════════════════════════════*/

static int test_score_computation(void) {
    printf("Testing score computation\n\n");
    
    PGASConfidence conf;
    
    /* Excellent run */
    pgas_confidence_compute_raw(0.60f, 0.35f, 0.50f, 0.12f, 5, &conf, NULL);
    
    printf("  Excellent metrics:\n");
    printf("    ESS=60%%, Accept=35%%, Unique=50%%, Diverge=12%%\n");
    printf("    Diversity score:   %.3f\n", conf.diversity_score);
    printf("    Exploration score: %.3f\n", conf.exploration_score);
    printf("    Innovation score:  %.3f\n", conf.innovation_score);
    printf("    Overall score:     %.3f\n", conf.overall_score);
    
    bool excellent = (conf.diversity_score > 0.8f) &&
                     (conf.exploration_score > 0.8f) &&
                     (conf.innovation_score > 0.8f) &&
                     (conf.overall_score > 0.8f);
    
    /* Poor run */
    pgas_confidence_compute_raw(0.15f, 0.08f, 0.15f, 0.02f, 5, &conf, NULL);
    
    printf("\n  Poor metrics:\n");
    printf("    ESS=15%%, Accept=8%%, Unique=15%%, Diverge=2%%\n");
    printf("    Diversity score:   %.3f\n", conf.diversity_score);
    printf("    Exploration score: %.3f\n", conf.exploration_score);
    printf("    Innovation score:  %.3f\n", conf.innovation_score);
    printf("    Overall score:     %.3f\n", conf.overall_score);
    
    bool poor = (conf.overall_score < 0.4f);
    
    return excellent && poor;
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST: Usability Check
 *═══════════════════════════════════════════════════════════════════════════*/

static int test_usability_check(void) {
    printf("Testing usability check helper\n\n");
    
    PGASConfidence conf;
    
    /* Degenerate - not usable */
    pgas_confidence_compute_raw(0.05f, 0.20f, 0.05f, 0.10f, 5, &conf, NULL);
    printf("  Degenerate run: usable=%s (expected: NO)\n", 
           pgas_confidence_usable(&conf) ? "YES" : "NO");
    bool t1 = !pgas_confidence_usable(&conf);
    
    /* Good run - usable */
    pgas_confidence_compute_raw(0.50f, 0.25f, 0.40f, 0.15f, 5, &conf, NULL);
    printf("  Good run: usable=%s (expected: YES)\n", 
           pgas_confidence_usable(&conf) ? "YES" : "NO");
    bool t2 = pgas_confidence_usable(&conf);
    
    /* Regime change detected */
    pgas_confidence_compute_raw(0.50f, 0.25f, 0.40f, 0.35f, 5, &conf, NULL);
    printf("  Regime change: detected=%s (expected: YES)\n", 
           pgas_confidence_regime_change(&conf) ? "YES" : "NO");
    bool t3 = pgas_confidence_regime_change(&conf);
    
    return t1 && t2 && t3;
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST: Print Output
 *═══════════════════════════════════════════════════════════════════════════*/

static int test_print_output(void) {
    printf("Testing print output (visual inspection)\n\n");
    
    PGASConfidence conf;
    
    /* Create a medium-confidence scenario */
    pgas_confidence_compute_raw(0.45f, 0.22f, 0.38f, 0.11f, 5, &conf, NULL);
    conf.ancestor_accepts = 110;
    conf.ancestor_proposals = 500;
    conf.path_changes = 55;
    conf.path_length = 500;
    
    pgas_confidence_print(&conf);
    
    return 1;  /* Visual inspection */
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST: Edge Cases
 *═══════════════════════════════════════════════════════════════════════════*/

static int test_edge_cases(void) {
    printf("Testing edge cases\n\n");
    
    PGASConfidence conf;
    
    /* Zero divergence (reference was perfect) */
    pgas_confidence_compute_raw(0.50f, 0.25f, 0.40f, 0.0f, 5, &conf, NULL);
    printf("  Zero divergence:\n");
    printf("    Innovation score: %.3f (should be low but not zero)\n", 
           conf.innovation_score);
    bool t1 = (conf.innovation_score > 0.0f && conf.innovation_score < 0.5f);
    
    /* Perfect metrics */
    pgas_confidence_compute_raw(1.0f, 1.0f, 1.0f, 0.15f, 5, &conf, NULL);
    printf("  Perfect metrics:\n");
    printf("    Overall score: %.3f (should be 1.0)\n", conf.overall_score);
    printf("    Level: %s\n", pgas_confidence_level_str(conf.level));
    bool t2 = (conf.overall_score > 0.95f) && (conf.level == PGAS_CONFIDENCE_VERY_HIGH);
    
    /* NULL checks */
    int ret = pgas_confidence_compute_raw(0.5f, 0.5f, 0.5f, 0.1f, 5, NULL, NULL);
    printf("  NULL output: returns %d (expected: -1)\n", ret);
    bool t3 = (ret == -1);
    
    return t1 && t2 && t3;
}

/*═══════════════════════════════════════════════════════════════════════════
 * MAIN
 *═══════════════════════════════════════════════════════════════════════════*/

int main(void) {
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════════════╗\n");
    printf("║              PGAS CONFIDENCE METRICS TEST SUITE                       ║\n");
    printf("║              Adaptive γ for SAEM Blending                             ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════════╝\n");
    
    RUN_TEST(test_config_defaults);
    RUN_TEST(test_degeneracy_detection);
    RUN_TEST(test_reference_dominated);
    RUN_TEST(test_regime_change_detection);
    RUN_TEST(test_gamma_tiers);
    RUN_TEST(test_score_computation);
    RUN_TEST(test_usability_check);
    RUN_TEST(test_print_output);
    RUN_TEST(test_edge_cases);
    
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
