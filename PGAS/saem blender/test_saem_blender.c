/**
 * @file test_saem_blender.c
 * @brief Tests for SAEM Parameter Blender
 *
 * Compile:
 *   gcc -O2 -Wall test_saem_blender.c saem_blender.c -lm -o test_saem
 */

#include "saem_blender.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define GREEN "\033[32m"
#define RED   "\033[31m"
#define YELLOW "\033[33m"
#define RESET "\033[0m"

static int g_tests_run = 0;
static int g_tests_passed = 0;

#define RUN_TEST(fn) do { \
    g_tests_run++; \
    printf("\n--- %s ---\n", #fn); \
    if (fn()) { g_tests_passed++; printf(GREEN "✓ PASS" RESET "\n"); } \
    else { printf(RED "✗ FAIL" RESET "\n"); } \
} while(0)

#define ASSERT_TRUE(cond, msg) do { \
    if (!(cond)) { printf("  ASSERT FAILED: %s\n", msg); return 0; } \
} while(0)

#define ASSERT_NEAR(a, b, tol, msg) do { \
    if (fabsf((a) - (b)) > (tol)) { \
        printf("  ASSERT FAILED: %s (expected %.4f, got %.4f)\n", msg, (b), (a)); \
        return 0; \
    } \
} while(0)

/*═══════════════════════════════════════════════════════════════════════════
 * TEST: Basic Initialization
 *═══════════════════════════════════════════════════════════════════════════*/

static int test_initialization(void) {
    SAEMBlender blender;
    SAEMBlenderConfig cfg = saem_blender_config_defaults(4);
    
    int ret = saem_blender_init(&blender, &cfg, NULL);
    ASSERT_TRUE(ret == 0, "Init should succeed");
    ASSERT_TRUE(blender.initialized, "Should be marked initialized");
    
    /* Check default Pi is reasonable */
    float avg_diag = saem_blender_get_avg_diagonal(&blender);
    ASSERT_NEAR(avg_diag, 0.9f, 0.01f, "Default diagonal should be ~0.9");
    
    /* Check rows sum to 1 */
    float Pi[16];
    saem_blender_get_Pi(&blender, Pi);
    for (int i = 0; i < 4; i++) {
        float row_sum = 0;
        for (int j = 0; j < 4; j++) row_sum += Pi[i*4 + j];
        ASSERT_NEAR(row_sum, 1.0f, 1e-5f, "Rows should sum to 1");
    }
    
    printf("  Avg diagonal: %.4f\n", avg_diag);
    printf("  Gamma: %.4f\n", saem_blender_get_gamma(&blender));
    
    saem_blender_free(&blender);
    return 1;
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST: SAEM Blending
 *═══════════════════════════════════════════════════════════════════════════*/

static int test_saem_blend(void) {
    SAEMBlender blender;
    SAEMBlenderConfig cfg = saem_blender_config_defaults(4);
    cfg.gamma.use_robbins_monro = false;  /* Fixed gamma for testing */
    cfg.gamma.gamma_base = 0.2f;
    cfg.stickiness.control_stickiness = false;  /* Disable for this test */
    
    saem_blender_init(&blender, &cfg, NULL);
    
    /* Create mock PGAS output with different structure */
    PGASOutput oracle = {
        .n_regimes = 4,
        .acceptance_rate = 0.3f,
        .ess_fraction = 0.5f,
        .trigger_surprise = 2.0f,
    };
    
    /* Oracle says: regime 0 frequently transitions to regime 1 */
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            oracle.S[i][j] = 10.0f;  /* Base counts */
        }
        oracle.S[i][i] = 80.0f;  /* Diagonal */
    }
    oracle.S[0][1] = 50.0f;  /* Strong 0→1 transition */
    oracle.S[0][0] = 40.0f;  /* Reduced 0→0 */
    
    /* Get Pi before */
    float Pi_before[16];
    saem_blender_get_Pi(&blender, Pi_before);
    float p01_before = Pi_before[0*4 + 1];
    
    /* Blend */
    SAEMBlendResult res = saem_blender_blend(&blender, &oracle);
    ASSERT_TRUE(res.success, "Blend should succeed");
    
    /* Get Pi after */
    float Pi_after[16];
    saem_blender_get_Pi(&blender, Pi_after);
    float p01_after = Pi_after[0*4 + 1];
    
    printf("  γ used: %.4f\n", res.gamma_used);
    printf("  Π[0→1] before: %.4f, after: %.4f\n", p01_before, p01_after);
    printf("  KL divergence: %.6f\n", res.kl_divergence);
    printf("  Cells floored: %d\n", res.cells_floored);
    
    /* Π[0→1] should have increased */
    ASSERT_TRUE(p01_after > p01_before, "Π[0→1] should increase toward oracle");
    
    /* Rows should still sum to 1 */
    for (int i = 0; i < 4; i++) {
        float row_sum = 0;
        for (int j = 0; j < 4; j++) row_sum += Pi_after[i*4 + j];
        ASSERT_NEAR(row_sum, 1.0f, 1e-5f, "Rows should sum to 1 after blend");
    }
    
    saem_blender_free(&blender);
    return 1;
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST: Floor Enforcement
 *═══════════════════════════════════════════════════════════════════════════*/

static int test_floor_enforcement(void) {
    SAEMBlender blender;
    SAEMBlenderConfig cfg = saem_blender_config_defaults(4);
    cfg.gamma.use_robbins_monro = false;
    cfg.gamma.gamma_base = 0.5f;  /* High gamma to see effect */
    cfg.stickiness.control_stickiness = false;
    
    saem_blender_init(&blender, &cfg, NULL);
    
    /* Oracle with extreme sparsity */
    PGASOutput oracle = {
        .n_regimes = 4,
        .acceptance_rate = 0.3f,
        .ess_fraction = 0.5f,
    };
    
    /* Nearly deterministic: regime 0 always stays in 0 */
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            oracle.S[i][j] = 0.001f;  /* Tiny counts */
        }
        oracle.S[i][i] = 100.0f;  /* Huge diagonal */
    }
    
    SAEMBlendResult res = saem_blender_blend(&blender, &oracle);
    ASSERT_TRUE(res.success, "Blend should succeed");
    
    /* Check floor enforcement */
    float Pi[16];
    saem_blender_get_Pi(&blender, Pi);
    
    int below_floor = 0;
    for (int i = 0; i < 16; i++) {
        if (Pi[i] < cfg.trans_floor - 1e-10f) {
            below_floor++;
        }
    }
    
    printf("  Cells floored: %d\n", res.cells_floored);
    printf("  Cells below floor after: %d (should be 0)\n", below_floor);
    printf("  Min off-diagonal: %.2e (floor: %.2e)\n", 
           Pi[0*4 + 1], cfg.trans_floor);
    
    ASSERT_TRUE(below_floor == 0, "No cell should be below floor after blend");
    
    saem_blender_free(&blender);
    return 1;
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST: Stickiness Control
 *═══════════════════════════════════════════════════════════════════════════*/

static int test_stickiness_control(void) {
    SAEMBlender blender;
    SAEMBlenderConfig cfg = saem_blender_config_defaults(4);
    cfg.gamma.use_robbins_monro = false;
    cfg.gamma.gamma_base = 0.8f;  /* Very high to force extreme update */
    cfg.stickiness.control_stickiness = true;
    cfg.stickiness.target_diag_min = 0.85f;
    cfg.stickiness.target_diag_max = 0.95f;
    
    saem_blender_init(&blender, &cfg, NULL);
    
    /* Oracle with very low stickiness (would make filter jittery) */
    PGASOutput oracle = {
        .n_regimes = 4,
        .acceptance_rate = 0.3f,
        .ess_fraction = 0.5f,
    };
    
    /* Uniform transitions (no stickiness) */
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            oracle.S[i][j] = 25.0f;  /* All equal */
        }
    }
    
    SAEMBlendResult res = saem_blender_blend(&blender, &oracle);
    ASSERT_TRUE(res.success, "Blend should succeed");
    
    float avg_diag = saem_blender_get_avg_diagonal(&blender);
    
    printf("  Diag before: %.4f, after: %.4f\n", 
           res.diag_avg_before, res.diag_avg_after);
    printf("  Stickiness adjusted: %s\n", 
           res.stickiness_adjusted ? "YES" : "NO");
    printf("  Final avg diagonal: %.4f (target: [%.2f, %.2f])\n",
           avg_diag, cfg.stickiness.target_diag_min, cfg.stickiness.target_diag_max);
    
    /* Should be clamped to target range */
    ASSERT_TRUE(avg_diag >= cfg.stickiness.target_diag_min - 0.01f,
                "Diagonal should be >= target_min");
    
    saem_blender_free(&blender);
    return 1;
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST: Robbins-Monro Decay
 *═══════════════════════════════════════════════════════════════════════════*/

static int test_robbins_monro(void) {
    SAEMBlender blender;
    SAEMBlenderConfig cfg = saem_blender_config_defaults(4);
    cfg.gamma.use_robbins_monro = true;
    cfg.gamma.gamma_base = 0.3f;
    cfg.gamma.rm_offset = 5.0f;
    cfg.stickiness.control_stickiness = false;
    
    saem_blender_init(&blender, &cfg, NULL);
    
    /* Standard oracle */
    PGASOutput oracle = {
        .n_regimes = 4,
        .acceptance_rate = 0.25f,  /* Neutral */
        .ess_fraction = 0.4f,
    };
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) oracle.S[i][j] = 10.0f;
        oracle.S[i][i] = 80.0f;
    }
    
    float gammas[10];
    for (int iter = 0; iter < 10; iter++) {
        SAEMBlendResult res = saem_blender_blend(&blender, &oracle);
        gammas[iter] = res.gamma_used;
    }
    
    printf("  Gamma sequence: ");
    for (int i = 0; i < 10; i++) printf("%.3f ", gammas[i]);
    printf("\n");
    
    /* Gamma should decrease over iterations */
    ASSERT_TRUE(gammas[9] < gammas[0], "Gamma should decay via Robbins-Monro");
    
    saem_blender_free(&blender);
    return 1;
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST: Tempered Path Generation
 *═══════════════════════════════════════════════════════════════════════════*/

static int test_tempered_path(void) {
    SAEMBlender blender;
    SAEMBlenderConfig cfg = saem_blender_config_defaults(4);
    cfg.tempering.enable_tempering = true;
    cfg.tempering.flip_probability = 0.1f;  /* 10% for visible effect */
    cfg.tempering.seed = 12345;
    
    saem_blender_init(&blender, &cfg, NULL);
    
    /* Create a simple path */
    int T = 100;
    int rbpf_path[100];
    int tempered_path[100];
    
    for (int t = 0; t < T; t++) {
        rbpf_path[t] = (t / 25) % 4;  /* 0,0,...,1,1,...,2,2,...,3,3,... */
    }
    
    int flips = saem_blender_temper_path(&blender, rbpf_path, T, tempered_path);
    
    /* Count actual differences */
    int actual_diffs = 0;
    for (int t = 0; t < T; t++) {
        if (rbpf_path[t] != tempered_path[t]) actual_diffs++;
    }
    
    printf("  Path length: %d\n", T);
    printf("  Flips injected: %d\n", flips);
    printf("  Actual differences: %d\n", actual_diffs);
    printf("  Expected: ~%d (%.0f%%)\n", 
           (int)(T * cfg.tempering.flip_probability),
           cfg.tempering.flip_probability * 100);
    
    /* Sample of path */
    printf("  Original:  ");
    for (int t = 0; t < 20; t++) printf("%d", rbpf_path[t]);
    printf("...\n");
    printf("  Tempered:  ");
    for (int t = 0; t < 20; t++) printf("%d", tempered_path[t]);
    printf("...\n");
    
    ASSERT_TRUE(flips > 0, "Should have some flips");
    ASSERT_TRUE(flips == actual_diffs, "Flip count should match differences");
    
    saem_blender_free(&blender);
    return 1;
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST: Convergence Behavior
 *═══════════════════════════════════════════════════════════════════════════*/

static int test_convergence(void) {
    SAEMBlender blender;
    SAEMBlenderConfig cfg = saem_blender_config_defaults(4);
    cfg.gamma.use_robbins_monro = false;
    cfg.gamma.gamma_base = 0.1f;
    cfg.stickiness.control_stickiness = false;
    
    saem_blender_init(&blender, &cfg, NULL);
    
    /* Target matrix */
    float target[4][4] = {
        {0.90, 0.05, 0.03, 0.02},
        {0.04, 0.92, 0.02, 0.02},
        {0.03, 0.02, 0.93, 0.02},
        {0.02, 0.02, 0.02, 0.94},
    };
    
    /* Oracle always reports target (scaled to counts) */
    PGASOutput oracle = {
        .n_regimes = 4,
        .acceptance_rate = 0.3f,
        .ess_fraction = 0.5f,
    };
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            oracle.S[i][j] = target[i][j] * 100.0f;
        }
    }
    
    /* Run many iterations */
    float kl_history[50];
    for (int iter = 0; iter < 50; iter++) {
        SAEMBlendResult res = saem_blender_blend(&blender, &oracle);
        
        /* Compute KL to target */
        float Pi[16];
        saem_blender_get_Pi(&blender, Pi);
        kl_history[iter] = saem_blender_kl_divergence(Pi, &target[0][0], 4);
    }
    
    printf("  KL to target: iter 0=%.6f, iter 25=%.6f, iter 49=%.6f\n",
           kl_history[0], kl_history[25], kl_history[49]);
    
    /* Should converge (KL decreasing) */
    ASSERT_TRUE(kl_history[49] < kl_history[0], "Should converge toward target");
    ASSERT_TRUE(kl_history[49] < 0.01f, "Should get close to target");
    
    /* Print final matrix */
    printf("  Final Π:\n");
    float Pi[16];
    saem_blender_get_Pi(&blender, Pi);
    for (int i = 0; i < 4; i++) {
        printf("    [");
        for (int j = 0; j < 4; j++) printf(" %.3f", Pi[i*4+j]);
        printf(" ]\n");
    }
    
    saem_blender_free(&blender);
    return 1;
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST: Acceptance Rate Adaptation
 *═══════════════════════════════════════════════════════════════════════════*/

static int test_acceptance_adaptation(void) {
    SAEMBlender blender;
    SAEMBlenderConfig cfg = saem_blender_config_defaults(4);
    cfg.gamma.use_robbins_monro = false;
    cfg.gamma.gamma_base = 0.2f;
    cfg.gamma.accept_high = 0.4f;
    cfg.gamma.accept_low = 0.15f;
    cfg.gamma.accept_gamma_boost = 1.5f;
    cfg.gamma.accept_gamma_penalty = 0.5f;
    cfg.stickiness.control_stickiness = false;
    
    saem_blender_init(&blender, &cfg, NULL);
    
    PGASOutput oracle_base = {
        .n_regimes = 4,
        .acceptance_rate = 0.25f,
        .ess_fraction = 0.4f,
    };
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) oracle_base.S[i][j] = 10.0f;
        oracle_base.S[i][i] = 80.0f;
    }
    
    /* Test with different acceptance rates */
    float gamma_low, gamma_mid, gamma_high;
    
    /* Low acceptance */
    PGASOutput oracle_low = oracle_base;
    oracle_low.acceptance_rate = 0.10f;
    saem_blender_blend(&blender, &oracle_low);
    gamma_low = saem_blender_get_gamma(&blender);
    saem_blender_reset(&blender, NULL);
    
    /* Mid acceptance */
    PGASOutput oracle_mid = oracle_base;
    oracle_mid.acceptance_rate = 0.25f;
    saem_blender_blend(&blender, &oracle_mid);
    gamma_mid = saem_blender_get_gamma(&blender);
    saem_blender_reset(&blender, NULL);
    
    /* High acceptance */
    PGASOutput oracle_high = oracle_base;
    oracle_high.acceptance_rate = 0.50f;
    saem_blender_blend(&blender, &oracle_high);
    gamma_high = saem_blender_get_gamma(&blender);
    
    printf("  Gamma with low accept (%.2f): %.4f\n", 
           oracle_low.acceptance_rate, gamma_low);
    printf("  Gamma with mid accept (%.2f): %.4f\n", 
           oracle_mid.acceptance_rate, gamma_mid);
    printf("  Gamma with high accept (%.2f): %.4f\n", 
           oracle_high.acceptance_rate, gamma_high);
    
    ASSERT_TRUE(gamma_high > gamma_mid, "High acceptance should boost gamma");
    ASSERT_TRUE(gamma_low < gamma_mid, "Low acceptance should penalize gamma");
    
    saem_blender_free(&blender);
    return 1;
}

/*═══════════════════════════════════════════════════════════════════════════
 * MAIN
 *═══════════════════════════════════════════════════════════════════════════*/

int main(void) {
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║              SAEM BLENDER TEST SUITE                         ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n");
    
    RUN_TEST(test_initialization);
    RUN_TEST(test_saem_blend);
    RUN_TEST(test_floor_enforcement);
    RUN_TEST(test_stickiness_control);
    RUN_TEST(test_robbins_monro);
    RUN_TEST(test_tempered_path);
    RUN_TEST(test_convergence);
    RUN_TEST(test_acceptance_adaptation);
    
    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("RESULTS: %d/%d tests passed\n", g_tests_passed, g_tests_run);
    printf("═══════════════════════════════════════════════════════════════\n\n");
    
    return (g_tests_passed == g_tests_run) ? 0 : 1;
}
