/*
 * test_crisis_detector.c - Integration test for crisis detection system
 * 
 * Tests the full pipeline:
 *   EventDetector → HawkesIntegrator → DualSR → State Machine
 * 
 * Compile:
 *   gcc -O3 -march=native test_crisis_detector.c crisis_detector.c \
 *       event_detector.c sr_detector.c hawkes_integrator.c -lm -o test_crisis
 */

#include "crisis_detector.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

/* ═══════════════════════════════════════════════════════════════════════════
 * RNG
 * ═══════════════════════════════════════════════════════════════════════════ */

static uint64_t rng_state = 88172645463325252ULL;

static inline uint64_t xorshift64(void) {
    uint64_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    rng_state = x;
    return x;
}

static inline float randn(void) {
    float u1 = (xorshift64() >> 11) * (1.0f / 9007199254740992.0f);
    float u2 = (xorshift64() >> 11) * (1.0f / 9007199254740992.0f);
    if (u1 < 1e-10f) u1 = 1e-10f;
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
}

static void seed_rng(uint64_t seed) {
    rng_state = seed;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * TIMING
 * ═══════════════════════════════════════════════════════════════════════════ */

#if defined(__x86_64__) || defined(_M_X64)
static inline uint64_t rdtsc(void) {
    unsigned int lo, hi;
    __asm__ volatile ("rdtsc" : "=a" (lo), "=d" (hi));
    return ((uint64_t)hi << 32) | lo;
}
#else
#include <time.h>
static inline uint64_t rdtsc(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}
#endif

/* ═══════════════════════════════════════════════════════════════════════════
 * TEST 1: Basic Initialization
 * ═══════════════════════════════════════════════════════════════════════════ */

static int test_initialization(void) {
    printf("\n=== TEST: Initialization ===\n");
    
    CrisisDetector cd;
    int ret = crisis_detector_init(&cd, NULL);
    
    printf("  Init returned: %d (expected 0)\n", ret);
    printf("  Initial state: %s (expected IDLE)\n", crisis_state_name(cd.state));
    printf("  Tick count: %ld (expected 0)\n", (long)cd.tick_count);
    
    crisis_detector_free(&cd);
    
    int pass = (ret == 0) && (cd.state == CRISIS_IDLE);
    printf("Result: %s\n", pass ? "PASS" : "FAIL");
    
    return pass;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * TEST 2: Normal Market (Should Stay IDLE)
 * ═══════════════════════════════════════════════════════════════════════════ */

static int test_normal_market(void) {
    printf("\n=== TEST: Normal Market (Stay IDLE) ===\n");
    
    CrisisDetector cd;
    crisis_detector_init(&cd, NULL);
    
    seed_rng(12345);
    
    float sigma = 0.01f;
    int n_ticks = 5000;
    int max_state = CRISIS_IDLE;
    
    for (int t = 0; t < n_ticks; t++) {
        float obs = sigma * randn();
        CrisisState state = crisis_detector_update(&cd, obs, t);
        if (state > max_state) max_state = state;
    }
    
    printf("  Ticks processed: %d\n", n_ticks);
    printf("  Final state: %s\n", crisis_state_name(cd.state));
    printf("  Max state reached: %s\n", crisis_state_name(max_state));
    printf("  Total crises: %d (expected 0)\n", cd.total_crises);
    printf("  False alarms: %d\n", cd.false_alarms);
    
    crisis_detector_free(&cd);
    
    /* In normal market, we may enter ALERT occasionally but shouldn't confirm crisis */
    int pass = (cd.total_crises == 0);
    printf("Result: %s\n", pass ? "PASS" : "FAIL");
    
    return pass;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * TEST 3: Flash Crash Detection
 * ═══════════════════════════════════════════════════════════════════════════ */

static int test_flash_crash(void) {
    printf("\n=== TEST: Flash Crash Detection ===\n");
    
    CrisisDetector cd;
    crisis_detector_init(&cd, NULL);
    
    seed_rng(54321);
    
    float sigma_normal = 0.01f;
    float sigma_crisis = 0.05f;
    
    /* Phase 1: Normal market (warmup + some ticks) */
    printf("  Phase 1: Normal market (500 ticks)...\n");
    for (int t = 0; t < 500; t++) {
        float obs = sigma_normal * randn();
        crisis_detector_update(&cd, obs, t);
    }
    printf("    State after normal: %s\n", crisis_state_name(cd.state));
    
    /* Phase 2: Flash crash (high volatility burst) */
    printf("  Phase 2: Flash crash (100 ticks at 5x volatility)...\n");
    int crisis_detected_at = -1;
    for (int t = 500; t < 600; t++) {
        float obs = sigma_crisis * randn();
        CrisisState state = crisis_detector_update(&cd, obs, t);
        if (state == CRISIS_ACTIVE && crisis_detected_at < 0) {
            crisis_detected_at = t - 500;
            printf("    CRISIS_ACTIVE at relative tick %d\n", crisis_detected_at);
        }
    }
    
    /* Phase 3: Recovery */
    printf("  Phase 3: Recovery (500 ticks)...\n");
    int recovery_at = -1;
    for (int t = 600; t < 1100; t++) {
        float obs = sigma_normal * randn();
        CrisisState state = crisis_detector_update(&cd, obs, t);
        if (state == CRISIS_IDLE && recovery_at < 0 && crisis_detected_at >= 0) {
            recovery_at = t - 600;
            printf("    CRISIS_IDLE (recovered) at relative tick %d\n", recovery_at);
        }
    }
    
    printf("\n  Summary:\n");
    printf("    Crises detected: %d\n", cd.total_crises);
    printf("    Clean exits: %d\n", cd.clean_exits);
    printf("    Re-triggers: %d\n", cd.re_triggers);
    printf("    False alarms: %d\n", cd.false_alarms);
    
    crisis_detector_free(&cd);
    
    int pass = (crisis_detected_at >= 0) && (crisis_detected_at < 50);
    printf("Result: %s\n", pass ? "PASS" : "FAIL");
    
    return pass;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * TEST 4: Sigma Freeze During Crisis
 * ═══════════════════════════════════════════════════════════════════════════ */

static int test_sigma_freeze(void) {
    printf("\n=== TEST: Sigma Freeze During Crisis ===\n");
    
    CrisisDetector cd;
    crisis_detector_init(&cd, NULL);
    
    seed_rng(11111);
    
    float sigma_peace = 0.01f;
    
    /* Set initial sigma */
    crisis_detector_set_sigma_peace(&cd, sigma_peace);
    
    /* Normal market until we can read sigma */
    for (int t = 0; t < 300; t++) {
        crisis_detector_update(&cd, sigma_peace * randn(), t);
    }
    
    float sigma_before = crisis_detector_get_sigma_peace(&cd);
    printf("  Sigma before crisis: %.6f\n", sigma_before);
    
    /* Induce crisis */
    for (int t = 300; t < 400; t++) {
        crisis_detector_update(&cd, 0.05f * randn(), t);
    }
    
    /* Try to update sigma during crisis */
    float new_sigma = 0.02f;
    crisis_detector_set_sigma_peace(&cd, new_sigma);
    
    float sigma_during = crisis_detector_get_sigma_peace(&cd);
    printf("  Sigma during crisis (after set): %.6f (should be frozen)\n", sigma_during);
    printf("  Attempted to set: %.6f\n", new_sigma);
    
    crisis_detector_free(&cd);
    
    /* During crisis, sigma should be frozen */
    int pass = (sigma_during < new_sigma);  /* Should not have updated */
    printf("Result: %s\n", pass ? "PASS" : "FAIL");
    
    return pass;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * TEST 5: Latency
 * ═══════════════════════════════════════════════════════════════════════════ */

static int test_latency(void) {
    printf("\n=== TEST: Latency ===\n");
    
    CrisisDetector cd;
    crisis_detector_init(&cd, NULL);
    
    seed_rng(22222);
    
    const int N = 100000;
    float *obs = malloc(N * sizeof(float));
    
    for (int i = 0; i < N; i++) {
        obs[i] = 0.01f * randn();
    }
    
    /* Warmup */
    for (int t = 0; t < 1000; t++) {
        crisis_detector_update(&cd, obs[t % N], t);
    }
    crisis_detector_reset(&cd);
    
    /* Measure */
    uint64_t start = rdtsc();
    for (int t = 0; t < N; t++) {
        crisis_detector_update(&cd, obs[t], t);
    }
    uint64_t end = rdtsc();
    
    double cycles_per_update = (double)(end - start) / N;
    
    printf("  Updates: %d\n", N);
    printf("  Cycles/update: %.1f\n", cycles_per_update);
    printf("  Target: < 800 cycles (EventDet + Hawkes + SR + StateMachine)\n");
    
    free(obs);
    crisis_detector_free(&cd);
    
    int pass = (cycles_per_update < 800);
    printf("Result: %s\n", pass ? "PASS" : "FAIL");
    
    return pass;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * TEST 6: State Machine Transitions
 * ═══════════════════════════════════════════════════════════════════════════ */

static int test_state_transitions(void) {
    printf("\n=== TEST: State Machine Transitions ===\n");
    
    CrisisDetector cd;
    crisis_detector_init(&cd, NULL);
    
    seed_rng(33333);
    
    int saw_idle = 0;
    int saw_alert = 0;
    int saw_active = 0;
    int saw_recovering = 0;
    
    float sigma = 0.01f;
    
    /* Normal → (should see IDLE) */
    for (int t = 0; t < 300; t++) {
        CrisisState s = crisis_detector_update(&cd, sigma * randn(), t);
        if (s == CRISIS_IDLE) saw_idle = 1;
    }
    
    /* Crisis burst → (should see ALERT, then ACTIVE) */
    for (int t = 300; t < 400; t++) {
        CrisisState s = crisis_detector_update(&cd, 0.05f * randn(), t);
        if (s == CRISIS_ALERT) saw_alert = 1;
        if (s == CRISIS_ACTIVE) saw_active = 1;
    }
    
    /* Recovery → (should see RECOVERING, then back to IDLE) */
    for (int t = 400; t < 1500; t++) {
        CrisisState s = crisis_detector_update(&cd, sigma * randn(), t);
        if (s == CRISIS_RECOVERING) saw_recovering = 1;
        if (s == CRISIS_IDLE && t > 400) break;  /* Clean exit */
    }
    
    printf("  Saw IDLE: %s\n", saw_idle ? "YES" : "NO");
    printf("  Saw ALERT: %s\n", saw_alert ? "YES" : "NO");
    printf("  Saw ACTIVE: %s\n", saw_active ? "YES" : "NO");
    printf("  Saw RECOVERING: %s\n", saw_recovering ? "YES" : "NO");
    
    crisis_detector_free(&cd);
    
    int pass = saw_idle && saw_alert && saw_active && saw_recovering;
    printf("Result: %s\n", pass ? "PASS" : "FAIL");
    
    return pass;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * TEST 7: Event Detector + Hawkes Integration
 * ═══════════════════════════════════════════════════════════════════════════ */

static int test_event_hawkes_integration(void) {
    printf("\n=== TEST: Event Detector + Hawkes Integration ===\n");
    
    CrisisDetector cd;
    crisis_detector_init(&cd, NULL);
    
    seed_rng(44444);
    
    int event_count = 0;
    int hawkes_armed_count = 0;
    
    float sigma = 0.01f;
    
    for (int t = 0; t < 5000; t++) {
        /* Generate occasional large returns to trigger events */
        float obs = sigma * randn();
        if (t % 50 == 25) {
            obs = sigma * 5.0f;  /* Force large return */
        }
        
        crisis_detector_update(&cd, obs, t);
        
        if (cd.last_was_event) event_count++;
        if (cd.last_hawkes_result.state == HAWKES_TRIG_ARMED) hawkes_armed_count++;
    }
    
    printf("  Events detected: %d (expected ~100 forced + ~10%% natural)\n", event_count);
    printf("  Hawkes armed ticks: %d\n", hawkes_armed_count);
    printf("  Event rate: %.2f%%\n", cd.event_det.event_count * 100.0 / cd.event_det.tick_count);
    
    crisis_detector_free(&cd);
    
    int pass = (event_count > 100) && (hawkes_armed_count > 0);
    printf("Result: %s\n", pass ? "PASS" : "FAIL");
    
    return pass;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * TEST 8: Print State (Visual Check)
 * ═══════════════════════════════════════════════════════════════════════════ */

static int test_print_state(void) {
    printf("\n=== TEST: Print State (Visual Check) ===\n");
    
    CrisisDetector cd;
    crisis_detector_init(&cd, NULL);
    
    seed_rng(55555);
    
    /* Run some ticks */
    for (int t = 0; t < 300; t++) {
        crisis_detector_update(&cd, 0.01f * randn(), t);
    }
    
    /* Induce crisis */
    for (int t = 300; t < 350; t++) {
        crisis_detector_update(&cd, 0.05f * randn(), t);
    }
    
    /* Print state */
    crisis_detector_print_state(&cd);
    
    crisis_detector_free(&cd);
    
    printf("Result: PASS (visual)\n");
    return 1;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * MAIN
 * ═══════════════════════════════════════════════════════════════════════════ */

int main(void) {
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("       CRISIS DETECTOR INTEGRATION TEST SUITE\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    
    int passed = 0;
    int total = 8;
    
    passed += test_initialization();
    passed += test_normal_market();
    passed += test_flash_crash();
    passed += test_sigma_freeze();
    passed += test_latency();
    passed += test_state_transitions();
    passed += test_event_hawkes_integration();
    passed += test_print_state();
    
    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("       RESULTS: %d/%d tests passed\n", passed, total);
    printf("═══════════════════════════════════════════════════════════════\n");
    
    return (passed == total) ? 0 : 1;
}
