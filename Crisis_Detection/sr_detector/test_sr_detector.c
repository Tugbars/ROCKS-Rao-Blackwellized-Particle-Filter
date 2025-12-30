/*
 * test_sr_detector.c - Unit tests for Shiryaev-Roberts detector
 * 
 * Compile with MKL:
 *   gcc -O3 -march=native -DMKL_ILP64 -I${MKLROOT}/include \
 *       test_sr_detector.c sr_detector.c \
 *       -L${MKLROOT}/lib/intel64 -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lm \
 *       -o test_sr_detector
 * 
 * Compile without MKL (AVX2):
 *   gcc -O3 -march=native -mavx2 -mfma test_sr_detector.c sr_detector.c -lm -o test_sr_detector
 */

#include "sr_detector.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

/* ═══════════════════════════════════════════════════════════════════════════
 * UTILITIES
 * ═══════════════════════════════════════════════════════════════════════════ */

static uint64_t xorshift64_state = 88172645463325252ULL;

static inline uint64_t xorshift64(void) {
    uint64_t x = xorshift64_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    xorshift64_state = x;
    return x;
}

static inline float randn(void) {
    /* Box-Muller transform */
    float u1 = (xorshift64() >> 11) * (1.0f / 9007199254740992.0f);
    float u2 = (xorshift64() >> 11) * (1.0f / 9007199254740992.0f);
    if (u1 < 1e-10f) u1 = 1e-10f;
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
}

static void seed_rng(uint64_t seed) {
    xorshift64_state = seed;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * TEST 1: Basic Log-LR Computation
 * ═══════════════════════════════════════════════════════════════════════════ */

static int test_log_lr_basic(void) {
    printf("\n=== TEST: Basic Log-LR Computation ===\n");
    
    float sigma_peace = 0.01f;   /* 1% volatility */
    float sigma_crisis = 0.03f;  /* 3% volatility */
    float cap = 10.0f;
    
    /* Test 1: obs = 0 should give log_LR based only on sigma ratio */
    float lr0 = sr_log_lr_gaussian(0.0f, sigma_peace, sigma_crisis, cap);
    float expected0 = logf(sigma_peace / sigma_crisis);  /* = log(1/3) ≈ -1.1 */
    printf("obs=0: log_LR=%.4f (expected≈%.4f)\n", lr0, expected0);
    
    /* Test 2: Large obs should favor crisis (larger sigma) */
    float lr_large = sr_log_lr_gaussian(0.05f, sigma_peace, sigma_crisis, cap);
    printf("obs=0.05 (5σ_peace): log_LR=%.4f (should be positive)\n", lr_large);
    
    /* Test 3: Small obs should favor peace (smaller sigma) */
    float lr_small = sr_log_lr_gaussian(0.005f, sigma_peace, sigma_crisis, cap);
    printf("obs=0.005 (0.5σ_peace): log_LR=%.4f (should be negative)\n", lr_small);
    
    /* Test 4: Winsorization - extreme obs should be capped */
    float lr_extreme = sr_log_lr_gaussian(0.5f, sigma_peace, sigma_crisis, cap);
    float lr_capped = sr_log_lr_gaussian(cap * sigma_peace, sigma_peace, sigma_crisis, cap);
    printf("obs=0.5 (50σ): log_LR=%.4f (should equal capped=%.4f)\n", lr_extreme, lr_capped);
    
    int pass = (lr_large > 0.0f) && (lr_small < 0.0f) && (fabsf(lr0 - expected0) < 0.01f);
    printf("Result: %s\n", pass ? "PASS" : "FAIL");
    
    return pass;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * TEST 2: SR Accumulation
 * ═══════════════════════════════════════════════════════════════════════════ */

static int test_sr_accumulation(void) {
    printf("\n=== TEST: SR Accumulation ===\n");
    
    SRConfig cfg = sr_config_default();
    SRStat sr;
    sr_stat_init(&sr, &cfg);
    
    float sigma_peace = 0.01f;
    
    /* Generate peace data (should keep SR low) */
    printf("Phase 1: Peace regime (100 ticks, σ=0.01)\n");
    seed_rng(12345);
    for (int i = 0; i < 100; i++) {
        float obs = sigma_peace * randn();
        sr_stat_update(&sr, obs, sigma_peace);
    }
    printf("  After peace: log_sr = %.4f\n", sr.log_sr);
    float sr_after_peace = sr.log_sr;
    
    /* Generate crisis data (should make SR explode) */
    printf("Phase 2: Crisis regime (50 ticks, σ=0.03)\n");
    float sigma_crisis_actual = 0.03f;
    for (int i = 0; i < 50; i++) {
        float obs = sigma_crisis_actual * randn();
        sr_stat_update(&sr, obs, sigma_peace);
    }
    printf("  After crisis: log_sr = %.4f\n", sr.log_sr);
    float sr_after_crisis = sr.log_sr;
    
    /* Return to peace (SR should decrease) */
    printf("Phase 3: Return to peace (100 ticks, σ=0.01)\n");
    sr_stat_reset(&sr);
    for (int i = 0; i < 100; i++) {
        float obs = sigma_peace * randn();
        sr_stat_update(&sr, obs, sigma_peace);
    }
    printf("  After return: log_sr = %.4f\n", sr.log_sr);
    
    int pass = (sr_after_crisis > sr_after_peace + 5.0f) && (sr_after_peace < 3.0f);
    printf("Result: %s\n", pass ? "PASS" : "FAIL");
    
    return pass;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * TEST 3: Dual SR (Entry + Exit)
 * ═══════════════════════════════════════════════════════════════════════════ */

static int test_dual_sr(void) {
    printf("\n=== TEST: Dual SR (Entry + Exit) ===\n");
    
    SRConfig cfg = sr_config_default();
    DualSR dsr;
    float sigma_peace = 0.01f;
    dual_sr_init(&dsr, &cfg, sigma_peace);
    
    seed_rng(54321);
    
    /* Phase 1: IDLE - accumulate SR_up during peace (should stay low) */
    printf("Phase 1: IDLE mode, peace data\n");
    for (int i = 0; i < 50; i++) {
        float obs = sigma_peace * randn();
        dual_sr_update_entry(&dsr, obs);
    }
    printf("  SR_up=%.4f, SR_down=%.4f\n", dsr.log_sr_up, dsr.log_sr_down);
    float sr_up_peace = dsr.log_sr_up;
    
    /* Phase 2: Crisis starts - SR_up should spike */
    printf("Phase 2: Crisis data (SR_up should spike)\n");
    float sigma_crisis = 0.04f;
    for (int i = 0; i < 30; i++) {
        float obs = sigma_crisis * randn();
        dual_sr_update_entry(&dsr, obs);
    }
    printf("  SR_up=%.4f (should be >> 5.0)\n", dsr.log_sr_up);
    float sr_up_crisis = dsr.log_sr_up;
    
    /* Phase 3: Simulate CRISIS_ACTIVE - switch to exit mode */
    printf("Phase 3: CRISIS_ACTIVE mode, crisis continues\n");
    dual_sr_reset_up(&dsr);
    for (int i = 0; i < 50; i++) {
        float obs = sigma_crisis * randn();
        dual_sr_update_exit(&dsr, obs);
    }
    printf("  SR_up=%.4f, SR_down=%.4f, σ_crisis_learned=%.4f\n", 
           dsr.log_sr_up, dsr.log_sr_down, dsr.sigma_crisis);
    float sigma_learned = dsr.sigma_crisis;
    
    /* Phase 4: Crisis ends - SR_down should spike */
    printf("Phase 4: Return to peace (SR_down should spike)\n");
    for (int i = 0; i < 50; i++) {
        float obs = sigma_peace * randn();
        dual_sr_update_exit(&dsr, obs);
    }
    printf("  SR_down=%.4f (should be >> 3.0)\n", dsr.log_sr_down);
    float sr_down_exit = dsr.log_sr_down;
    
    int pass = (sr_up_peace < 3.0f) && 
               (sr_up_crisis > 5.0f) && 
               (fabsf(sigma_learned - sigma_crisis) < 0.01f) &&
               (sr_down_exit > 3.0f);
    printf("Result: %s\n", pass ? "PASS" : "FAIL");
    
    return pass;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * TEST 4: Adaptive Threshold
 * ═══════════════════════════════════════════════════════════════════════════ */

static int test_adaptive_threshold(void) {
    printf("\n=== TEST: Adaptive Threshold ===\n");
    
    AdaptiveThresholdConfig cfg = adaptive_threshold_config_default();
    AdaptiveThreshold at;
    adaptive_threshold_init(&at, &cfg);
    
    /* Initial threshold */
    float h0 = adaptive_threshold_compute(&at);
    printf("Initial threshold: %.4f (should be ~5.0)\n", h0);
    
    /* After 1000 ticks of peace */
    for (int i = 0; i < 1000; i++) {
        adaptive_threshold_tick(&at);
    }
    float h1000 = adaptive_threshold_compute(&at);
    printf("After 1000 peace ticks: %.4f (should be ~5.69)\n", h1000);
    
    /* After 5000 ticks of peace */
    for (int i = 0; i < 4000; i++) {
        adaptive_threshold_tick(&at);
    }
    float h5000 = adaptive_threshold_compute(&at);
    printf("After 5000 peace ticks: %.4f (should be ~6.79)\n", h5000);
    
    /* Add wolf penalties */
    adaptive_threshold_false_alarm(&at);
    float h_wolf1 = adaptive_threshold_compute(&at);
    printf("After 1 false alarm: %.4f (should be +2.0)\n", h_wolf1);
    
    adaptive_threshold_false_alarm(&at);
    adaptive_threshold_false_alarm(&at);
    float h_wolf3 = adaptive_threshold_compute(&at);
    printf("After 3 false alarms: %.4f (should be +6.0)\n", h_wolf3);
    
    /* Clean exit resets everything */
    adaptive_threshold_clean_exit(&at);
    float h_reset = adaptive_threshold_compute(&at);
    printf("After clean exit: %.4f (should be ~5.0)\n", h_reset);
    
    int pass = (fabsf(h0 - 5.0f) < 0.1f) && 
               (h1000 > h0) && 
               (h5000 > h1000) &&
               (h_wolf1 > h5000) &&
               (fabsf(h_reset - 5.0f) < 0.1f);
    printf("Result: %s\n", pass ? "PASS" : "FAIL");
    
    return pass;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * TEST 5: Batch Performance
 * ═══════════════════════════════════════════════════════════════════════════ */

static int test_batch_performance(void) {
    printf("\n=== TEST: Batch Performance ===\n");
    
    const int N = 100000;
    float *obs = (float *)aligned_alloc(64, N * sizeof(float));
    float *log_lr = (float *)aligned_alloc(64, N * sizeof(float));
    float *log_sr = (float *)aligned_alloc(64, N * sizeof(float));
    
    seed_rng(99999);
    float sigma = 0.01f;
    for (int i = 0; i < N; i++) {
        obs[i] = sigma * randn();
    }
    
    SRConfig cfg = sr_config_default();
    
    /* Benchmark batch log-LR */
    clock_t start = clock();
    for (int rep = 0; rep < 100; rep++) {
        sr_compute_log_lr_batch(obs, NULL, sigma, log_lr, N, &cfg);
    }
    clock_t end = clock();
    double time_lr = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Log-LR batch: %.2f M obs/sec\n", (100.0 * N) / time_lr / 1e6);
    
    /* Benchmark full update */
    start = clock();
    for (int rep = 0; rep < 100; rep++) {
        sr_update_batch(obs, sigma, log_sr, 0.0f, N, &cfg);
    }
    end = clock();
    double time_full = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Full update: %.2f M obs/sec\n", (100.0 * N) / time_full / 1e6);
    
    /* Verify accumulation is correct */
    SRStat sr;
    sr_stat_init(&sr, &cfg);
    for (int i = 0; i < 1000; i++) {
        sr_stat_update(&sr, obs[i], sigma);
    }
    
    float batch_result = log_sr[999];
    float scalar_result = sr.log_sr;
    printf("Batch vs Scalar (1000 ticks): %.6f vs %.6f (diff=%.2e)\n", 
           batch_result, scalar_result, fabsf(batch_result - scalar_result));
    
    free(obs);
    free(log_lr);
    free(log_sr);
    
    int pass = fabsf(batch_result - scalar_result) < 1e-4f;
    printf("Result: %s\n", pass ? "PASS" : "FAIL");
    
    return pass;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * TEST 6: Slow Bleed Detection
 * ═══════════════════════════════════════════════════════════════════════════ */

static int test_slow_bleed(void) {
    printf("\n=== TEST: Slow Bleed Detection ===\n");
    
    SRConfig cfg = sr_config_default();
    SRStat sr;
    sr_stat_init(&sr, &cfg);
    
    float sigma_peace = 0.01f;
    float threshold = 5.0f;  /* log(148) */
    
    seed_rng(77777);
    
    /* Slow bleed: volatility creeps from 1% to 1.5% over 200 ticks */
    printf("Slow bleed: σ creeps from 0.01 to 0.015 over 200 ticks\n");
    int detection_tick = -1;
    
    for (int i = 0; i < 200; i++) {
        float progress = (float)i / 200.0f;
        float sigma_actual = 0.01f + 0.005f * progress;  /* 1% → 1.5% */
        float obs = sigma_actual * randn();
        
        sr_stat_update(&sr, obs, sigma_peace);
        
        if (sr.log_sr > threshold && detection_tick < 0) {
            detection_tick = i;
            printf("  Detected at tick %d, log_sr=%.4f\n", i, sr.log_sr);
        }
    }
    
    if (detection_tick < 0) {
        printf("  Not detected in 200 ticks, final log_sr=%.4f\n", sr.log_sr);
    }
    
    /* Now test abrupt change for comparison */
    sr_stat_reset(&sr);
    printf("\nAbrupt change: σ jumps from 0.01 to 0.03\n");
    detection_tick = -1;
    
    for (int i = 0; i < 50; i++) {
        float sigma_actual = (i < 10) ? 0.01f : 0.03f;
        float obs = sigma_actual * randn();
        
        sr_stat_update(&sr, obs, sigma_peace);
        
        if (sr.log_sr > threshold && detection_tick < 0) {
            detection_tick = i;
            printf("  Detected at tick %d, log_sr=%.4f\n", i, sr.log_sr);
        }
    }
    
    int pass = (detection_tick >= 0 && detection_tick < 30);
    printf("Result: %s\n", pass ? "PASS" : "FAIL");
    
    return pass;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * TEST 7: Outlier Robustness
 * ═══════════════════════════════════════════════════════════════════════════ */

static int test_outlier_robustness(void) {
    printf("\n=== TEST: Outlier Robustness (Winsorization) ===\n");
    
    float sigma_peace = 0.01f;
    float sigma_crisis = 3.0f * sigma_peace;
    float cap = 10.0f;
    
    /* 
     * Winsorization caps z-scores at ±cap.
     * For z_h0 = obs/sigma_peace, cap kicks in when |obs| > cap * sigma_peace
     * For z_h1 = obs/sigma_crisis, cap kicks in when |obs| > cap * sigma_crisis
     * 
     * With sigma_crisis = 3*sigma_peace:
     * - z_h0 caps at |obs| = 10 * 0.01 = 0.10 (10σ)
     * - z_h1 caps at |obs| = 10 * 0.03 = 0.30 (30σ)
     */
    
    printf("Winsorization behavior (cap=%.0f):\n", cap);
    printf("  z_h0 caps when |obs| > %.3f (%.0fσ_peace)\n", cap * sigma_peace, cap);
    printf("  z_h1 caps when |obs| > %.3f (%.0fσ_peace)\n", cap * sigma_crisis, cap * 3.0f);
    
    /* Test: compare with and without winsorization for various outlier sizes */
    printf("\nlog_LR comparison:\n");
    printf("  Outlier   Winsorized    Raw         Reduction\n");
    
    int all_reductions_positive = 1;
    
    for (int mult = 5; mult <= 25; mult += 5) {
        float obs = mult * sigma_peace;
        float lr_w = sr_log_lr_gaussian(obs, sigma_peace, sigma_crisis, cap);
        float lr_nw = sr_log_lr_gaussian(obs, sigma_peace, sigma_crisis, 1000.0f);
        float reduction = (lr_nw - lr_w) / lr_nw * 100.0f;
        
        printf("  %2dσ       %8.2f      %8.2f    %5.1f%%\n", mult, lr_w, lr_nw, reduction);
        
        /* Beyond the cap, winsorization should reduce the LR */
        if (mult > (int)cap && lr_w >= lr_nw) {
            all_reductions_positive = 0;
        }
    }
    
    /* Key test: 15σ outlier should have significantly reduced impact */
    float obs_15sigma = 15.0f * sigma_peace;
    float lr_winsor = sr_log_lr_gaussian(obs_15sigma, sigma_peace, sigma_crisis, cap);
    float lr_no_winsor = sr_log_lr_gaussian(obs_15sigma, sigma_peace, sigma_crisis, 1000.0f);
    
    printf("\nKey metric: 15σ outlier impact\n");
    printf("  With winsorization:    log_LR = %.2f\n", lr_winsor);
    printf("  Without winsorization: log_LR = %.2f\n", lr_no_winsor);
    printf("  Reduction ratio: %.2fx\n", lr_no_winsor / lr_winsor);
    
    /* Pass if:
     * 1. Outliers beyond cap have reduced log_LR
     * 2. 15σ outlier has at least 2x reduction
     */
    int pass = all_reductions_positive && (lr_no_winsor > lr_winsor * 2.0f);
    printf("Result: %s\n", pass ? "PASS" : "FAIL");
    
    return pass;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * TEST 8: Numerical Stability (Softplus Approximation)
 * ═══════════════════════════════════════════════════════════════════════════ */

static int test_numerical_stability(void) {
    printf("\n=== TEST: Numerical Stability (Softplus) ===\n");
    
    /* Test sr_accumulate_step at extreme values */
    float clamp = 100.0f;  /* High clamp to test softplus, not clamping */
    
    /* Test 1: log_sr_old = 50 (should use approximation: log(1+e^50) ≈ 50) */
    float result_50 = sr_accumulate_step(1.0f, 50.0f, clamp);
    float expected_50 = 1.0f + 50.0f;  /* log_lr + log_sr_old */
    printf("log_sr_old=50: result=%.4f (expected≈%.4f)\n", result_50, expected_50);
    
    /* Test 2: log_sr_old = 90 (would overflow expf without fix) */
    float result_90 = sr_accumulate_step(1.0f, 90.0f, clamp);
    float expected_90 = 1.0f + 90.0f;
    printf("log_sr_old=90: result=%.4f (expected≈%.4f)\n", result_90, expected_90);
    int no_overflow_90 = isfinite(result_90) && fabsf(result_90 - expected_90) < 0.01f;
    printf("  No overflow: %s\n", no_overflow_90 ? "YES" : "NO (CRITICAL BUG!)");
    
    /* Test 3: log_sr_old = -50 (should use approximation: log(1+e^-50) ≈ 0) */
    float result_neg50 = sr_accumulate_step(1.0f, -50.0f, clamp);
    float expected_neg50 = 1.0f + 0.0f;  /* log_lr + 0 */
    printf("log_sr_old=-50: result=%.4f (expected≈%.4f)\n", result_neg50, expected_neg50);
    
    /* Test 4: Transition region (log_sr_old = 15, exact computation) */
    float result_15 = sr_accumulate_step(1.0f, 15.0f, clamp);
    float expected_15 = 1.0f + log1pf(expf(15.0f));
    printf("log_sr_old=15: result=%.4f (expected=%.4f, exact)\n", result_15, expected_15);
    
    /* Test 5: Sustained crisis - accumulate to very high SR without explosion */
    printf("\nSustained crisis test (should not explode):\n");
    SRConfig cfg = sr_config_default();
    cfg.log_sr_clamp = 200.0f;  /* Very high clamp */
    SRStat sr;
    sr_stat_init(&sr, &cfg);
    
    float sigma_peace = 0.01f;
    float sigma_crisis = 0.05f;  /* 5x volatility */
    
    seed_rng(44444);
    int exploded = 0;
    for (int i = 0; i < 10000; i++) {
        float obs = sigma_crisis * randn();
        sr_stat_update(&sr, obs, sigma_peace);
        
        if (!isfinite(sr.log_sr)) {
            printf("  EXPLODED at tick %d!\n", i);
            exploded = 1;
            break;
        }
    }
    
    if (!exploded) {
        printf("  10000 crisis ticks: log_sr=%.2f (finite: YES)\n", sr.log_sr);
    }
    
    int pass = no_overflow_90 && !exploded && 
               fabsf(result_50 - expected_50) < 0.01f &&
               fabsf(result_neg50 - expected_neg50) < 0.01f;
    printf("Result: %s\n", pass ? "PASS" : "FAIL");
    
    return pass;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * TEST 9: Student-t Consistency (Scalar vs Batch)
 * ═══════════════════════════════════════════════════════════════════════════ */

static int test_student_t_consistency(void) {
    printf("\n=== TEST: Student-t Consistency (Scalar vs Batch) ===\n");
    
    const int N = 1000;
    float *obs = (float *)aligned_alloc(64, N * sizeof(float));
    float *log_lr_batch = (float *)aligned_alloc(64, N * sizeof(float));
    float *log_lr_scalar = (float *)aligned_alloc(64, N * sizeof(float));
    
    seed_rng(55555);
    float sigma = 0.01f;
    for (int i = 0; i < N; i++) {
        obs[i] = sigma * randn();
    }
    
    /* Test with Student-t (nu=6) */
    SRConfig cfg = sr_config_student_t(6.0f);
    
    /* Batch computation */
    sr_compute_log_lr_batch(obs, NULL, sigma, log_lr_batch, N, &cfg);
    
    /* Scalar computation */
    float sigma_crisis = cfg.sigma_multiple * sigma;
    for (int i = 0; i < N; i++) {
        log_lr_scalar[i] = sr_log_lr_student_t(obs[i], sigma, sigma_crisis, cfg.student_nu);
    }
    
    /* Compare */
    float max_diff = 0.0f;
    for (int i = 0; i < N; i++) {
        float diff = fabsf(log_lr_batch[i] - log_lr_scalar[i]);
        if (diff > max_diff) max_diff = diff;
    }
    
    printf("Student-t (nu=6) batch vs scalar: max_diff = %.2e\n", max_diff);
    
    /* Also test Gaussian for sanity */
    SRConfig cfg_gauss = sr_config_default();
    sr_compute_log_lr_batch(obs, NULL, sigma, log_lr_batch, N, &cfg_gauss);
    
    float sigma_crisis_g = cfg_gauss.sigma_multiple * sigma;
    for (int i = 0; i < N; i++) {
        log_lr_scalar[i] = sr_log_lr_gaussian(obs[i], sigma, sigma_crisis_g, cfg_gauss.winsorize_cap);
    }
    
    float max_diff_gauss = 0.0f;
    for (int i = 0; i < N; i++) {
        float diff = fabsf(log_lr_batch[i] - log_lr_scalar[i]);
        if (diff > max_diff_gauss) max_diff_gauss = diff;
    }
    
    printf("Gaussian batch vs scalar: max_diff = %.2e\n", max_diff_gauss);
    
    free(obs);
    free(log_lr_batch);
    free(log_lr_scalar);
    
    int pass = (max_diff < 1e-5f) && (max_diff_gauss < 1e-5f);
    printf("Result: %s\n", pass ? "PASS" : "FAIL");
    
    return pass;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * MAIN
 * ═══════════════════════════════════════════════════════════════════════════ */

int main(void) {
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("       SHIRYAEV-ROBERTS DETECTOR TEST SUITE\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    
#if defined(__AVX512F__)
    printf("Backend: AVX-512\n");
#elif defined(__AVX2__)
    printf("Backend: AVX2\n");
#elif defined(__AVX__)
    printf("Backend: AVX\n");
#else
    printf("Backend: Scalar\n");
#endif
    
    int passed = 0;
    int total = 9;
    
    passed += test_log_lr_basic();
    passed += test_sr_accumulation();
    passed += test_dual_sr();
    passed += test_adaptive_threshold();
    passed += test_batch_performance();
    passed += test_slow_bleed();
    passed += test_outlier_robustness();
    passed += test_numerical_stability();
    passed += test_student_t_consistency();
    
    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("       RESULTS: %d/%d tests passed\n", passed, total);
    printf("═══════════════════════════════════════════════════════════════\n");
    
    return (passed == total) ? 0 : 1;
}
