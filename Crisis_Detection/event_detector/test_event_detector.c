/*
 * test_event_detector.c - Unit tests for event detector
 * 
 * Compile:
 *   gcc -O3 -march=native test_event_detector.c event_detector.c -lm -o test_event_detector
 */

#include "event_detector.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

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

static inline double randn(void) {
    double u1 = (xorshift64() >> 11) * (1.0 / 9007199254740992.0);
    double u2 = (xorshift64() >> 11) * (1.0 / 9007199254740992.0);
    if (u1 < 1e-10) u1 = 1e-10;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979 * u2);
}

static inline double rand_uniform(void) {
    return (xorshift64() >> 11) * (1.0 / 9007199254740992.0);
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
static inline uint64_t rdtsc(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}
#endif

/* ═══════════════════════════════════════════════════════════════════════════
 * TEST 1: Event Rate for Random Data
 * ═══════════════════════════════════════════════════════════════════════════ */

static int test_event_rate_random(void) {
    printf("\n=== TEST: Event Rate for Random Data ===\n");
    
    EventDetector evt;
    event_detector_init(&evt, NULL);
    
    seed_rng(12345);
    
    double sigma = 0.01;
    int n_ticks = 100000;
    
    for (int i = 0; i < n_ticks; i++) {
        double obs = sigma * randn();
        event_detector_update(&evt, obs);
    }
    
    double rate = event_detector_get_rate(&evt);
    double threshold = event_detector_get_threshold(&evt);
    
    printf("  Ticks: %d\n", n_ticks);
    printf("  Events: %ld\n", evt.event_count);
    printf("  Event rate: %.2f%% (expected ~10%%)\n", rate * 100.0);
    printf("  Return threshold: %.6f (90th percentile of |r|)\n", threshold);
    
    /* For 90th percentile, expect ~10% event rate */
    /* Allow some tolerance due to warmup and P² approximation */
    int pass = (rate > 0.08 && rate < 0.12);
    printf("Result: %s\n", pass ? "PASS" : "FAIL");
    
    return pass;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * TEST 2: Warmup Period
 * ═══════════════════════════════════════════════════════════════════════════ */

static int test_warmup_period(void) {
    printf("\n=== TEST: Warmup Period ===\n");
    
    EventDetectorConfig cfg = event_detector_config_default();
    cfg.warmup_ticks = 200;
    
    EventDetector evt;
    event_detector_init(&evt, &cfg);
    
    seed_rng(54321);
    
    double sigma = 0.01;
    int events_during_warmup = 0;
    int events_after_warmup = 0;
    
    for (int i = 0; i < 10000; i++) {
        /* Generate some large returns to ensure events would fire */
        double obs = sigma * randn();
        if (i % 10 == 0) {
            obs = sigma * 5.0;  /* Force large return */
        }
        
        bool is_event = event_detector_update(&evt, obs);
        
        if (i < 200) {
            if (is_event) events_during_warmup++;
        } else {
            if (is_event) events_after_warmup++;
        }
    }
    
    printf("  Warmup ticks: 200\n");
    printf("  Events during warmup: %d (should be 0)\n", events_during_warmup);
    printf("  Events after warmup: %d (should be > 0)\n", events_after_warmup);
    printf("  is_ready at tick 199: %s\n", 
           evt.tick_count >= cfg.warmup_ticks ? "YES" : "NO");
    
    int pass = (events_during_warmup == 0) && (events_after_warmup > 0);
    printf("Result: %s\n", pass ? "PASS" : "FAIL");
    
    return pass;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * TEST 3: Volume Events
 * ═══════════════════════════════════════════════════════════════════════════ */

static int test_volume_events(void) {
    printf("\n=== TEST: Volume Events ===\n");
    
    EventDetector evt;
    event_detector_init(&evt, NULL);
    event_detector_enable_volume(&evt, 1000.0);  /* Initial volume EMA = 1000 */
    
    seed_rng(11111);
    
    double sigma = 0.01;
    int return_events = 0;
    int volume_events = 0;
    
    for (int i = 0; i < 10000; i++) {
        double obs = sigma * randn() * 0.5;  /* Small returns */
        double volume = 1000.0 + 200.0 * randn();  /* Normal volume */
        
        /* Every 100 ticks, spike volume */
        if (i % 100 == 50 && i > 100) {
            volume = 5000.0;  /* 5x normal */
        }
        
        event_detector_update_full(&evt, obs, volume, NAN);
        
        if (evt.last_return_event) return_events++;
        if (evt.last_volume_event) volume_events++;
    }
    
    printf("  Return events: %d\n", return_events);
    printf("  Volume events: %d (expected ~99)\n", volume_events);
    
    int pass = (volume_events > 80 && volume_events < 120);
    printf("Result: %s\n", pass ? "PASS" : "FAIL");
    
    return pass;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * TEST 4: Imbalance Events
 * ═══════════════════════════════════════════════════════════════════════════ */

static int test_imbalance_events(void) {
    printf("\n=== TEST: Imbalance Events ===\n");
    
    EventDetector evt;
    event_detector_init(&evt, NULL);
    event_detector_enable_imbalance(&evt, 0.1);  /* Initial |imbalance| EMA = 0.1 */
    
    seed_rng(22222);
    
    double sigma = 0.01;
    int imbalance_events = 0;
    
    for (int i = 0; i < 10000; i++) {
        double obs = sigma * randn() * 0.5;  /* Small returns */
        double imbalance = 0.1 * randn();     /* Normal imbalance */
        
        /* Every 100 ticks, spike imbalance */
        if (i % 100 == 50 && i > 100) {
            imbalance = 0.8;  /* Strong buy pressure */
        }
        
        event_detector_update_full(&evt, obs, 0.0, imbalance);
        
        if (evt.last_imbalance_event) imbalance_events++;
    }
    
    printf("  Imbalance events: %d (expected ~99)\n", imbalance_events);
    
    int pass = (imbalance_events > 80 && imbalance_events < 120);
    printf("Result: %s\n", pass ? "PASS" : "FAIL");
    
    return pass;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * TEST 5: P² Quantile Accuracy
 * ═══════════════════════════════════════════════════════════════════════════ */

static int test_quantile_accuracy(void) {
    printf("\n=== TEST: P² Quantile Accuracy ===\n");
    
    EventDetector evt;
    event_detector_init(&evt, NULL);
    
    seed_rng(33333);
    
    /* Generate data and also store for verification */
    int n = 10000;
    double *abs_returns = malloc(n * sizeof(double));
    double sigma = 0.01;
    
    for (int i = 0; i < n; i++) {
        double obs = sigma * randn();
        abs_returns[i] = fabs(obs);
        event_detector_update(&evt, obs);
    }
    
    /* Sort to find true 90th percentile */
    for (int i = 0; i < n - 1; i++) {
        for (int j = i + 1; j < n; j++) {
            if (abs_returns[j] < abs_returns[i]) {
                double tmp = abs_returns[i];
                abs_returns[i] = abs_returns[j];
                abs_returns[j] = tmp;
            }
        }
    }
    
    double true_p90 = abs_returns[(int)(n * 0.90)];
    double p2_p90 = event_detector_get_threshold(&evt);
    double error = fabs(p2_p90 - true_p90) / true_p90 * 100.0;
    
    printf("  True 90th percentile: %.6f\n", true_p90);
    printf("  P² estimate: %.6f\n", p2_p90);
    printf("  Error: %.2f%%\n", error);
    
    free(abs_returns);
    
    int pass = (error < 5.0);  /* Less than 5% error */
    printf("Result: %s\n", pass ? "PASS" : "FAIL");
    
    return pass;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * TEST 6: Latency
 * ═══════════════════════════════════════════════════════════════════════════ */

static int test_latency(void) {
    printf("\n=== TEST: Latency ===\n");
    
    EventDetector evt;
    event_detector_init(&evt, NULL);
    
    seed_rng(44444);
    
    /* Generate test data */
    int n = 100000;
    double *obs = malloc(n * sizeof(double));
    double sigma = 0.01;
    
    for (int i = 0; i < n; i++) {
        obs[i] = sigma * randn();
    }
    
    /* Warmup */
    for (int i = 0; i < 1000; i++) {
        event_detector_update(&evt, obs[i % n]);
    }
    event_detector_reset(&evt);
    
    /* Measure */
    uint64_t start = rdtsc();
    
    for (int i = 0; i < n; i++) {
        event_detector_update(&evt, obs[i]);
    }
    
    uint64_t end = rdtsc();
    
    double cycles_per_update = (double)(end - start) / n;
    
    printf("  Updates: %d\n", n);
    printf("  Cycles/update: %.1f\n", cycles_per_update);
    printf("  Target: < 100 cycles\n");
    
    free(obs);
    
    int pass = (cycles_per_update < 100);
    printf("Result: %s\n", pass ? "PASS" : "FAIL");
    
    return pass;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * TEST 7: Combined Event Detection
 * ═══════════════════════════════════════════════════════════════════════════ */

static int test_combined_events(void) {
    printf("\n=== TEST: Combined Event Detection ===\n");
    
    EventDetector evt;
    event_detector_init(&evt, NULL);
    event_detector_enable_volume(&evt, 1000.0);
    event_detector_enable_imbalance(&evt, 0.1);
    
    seed_rng(55555);
    
    double sigma = 0.01;
    int total_events = 0;
    int return_only = 0;
    int volume_only = 0;
    int imbalance_only = 0;
    int multiple_triggers = 0;
    
    for (int i = 0; i < 10000; i++) {
        double obs = sigma * randn();
        double volume = 1000.0 + 200.0 * randn();
        double imbalance = 0.1 * randn();
        
        event_detector_update_full(&evt, obs, volume, imbalance);
        
        if (evt.last_was_event) {
            total_events++;
            
            int triggers = (evt.last_return_event ? 1 : 0) +
                          (evt.last_volume_event ? 1 : 0) +
                          (evt.last_imbalance_event ? 1 : 0);
            
            if (triggers > 1) {
                multiple_triggers++;
            } else if (evt.last_return_event) {
                return_only++;
            } else if (evt.last_volume_event) {
                volume_only++;
            } else if (evt.last_imbalance_event) {
                imbalance_only++;
            }
        }
    }
    
    printf("  Total events: %d\n", total_events);
    printf("  Return-only: %d\n", return_only);
    printf("  Volume-only: %d\n", volume_only);
    printf("  Imbalance-only: %d\n", imbalance_only);
    printf("  Multiple triggers: %d\n", multiple_triggers);
    printf("  Event rate: %.2f%%\n", evt.event_count * 100.0 / evt.tick_count);
    
    /* With all three enabled, event rate should be higher than 10% */
    int pass = (total_events > 1000);  /* > 10% */
    printf("Result: %s\n", pass ? "PASS" : "FAIL");
    
    return pass;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * TEST 8: Different Quantiles
 * ═══════════════════════════════════════════════════════════════════════════ */

static int test_different_quantiles(void) {
    printf("\n=== TEST: Different Quantiles ===\n");
    
    double quantiles[] = {0.80, 0.85, 0.90, 0.95, 0.99};
    double expected_rates[] = {0.20, 0.15, 0.10, 0.05, 0.01};
    int n_quantiles = 5;
    
    seed_rng(66666);
    
    int all_pass = 1;
    
    for (int q = 0; q < n_quantiles; q++) {
        EventDetectorConfig cfg = event_detector_config_default();
        cfg.return_quantile = quantiles[q];
        cfg.warmup_ticks = 500;
        
        EventDetector evt;
        event_detector_init(&evt, &cfg);
        
        double sigma = 0.01;
        for (int i = 0; i < 50000; i++) {
            double obs = sigma * randn();
            event_detector_update(&evt, obs);
        }
        
        double rate = event_detector_get_rate(&evt);
        double expected = expected_rates[q];
        double error = fabs(rate - expected) / expected;
        
        printf("  P%.0f: rate=%.2f%% (expected %.0f%%), error=%.1f%%\n",
               quantiles[q] * 100, rate * 100, expected * 100, error * 100);
        
        if (error > 0.20) {  /* Allow 20% relative error */
            all_pass = 0;
        }
    }
    
    printf("Result: %s\n", all_pass ? "PASS" : "FAIL");
    
    return all_pass;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * MAIN
 * ═══════════════════════════════════════════════════════════════════════════ */

int main(void) {
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("       EVENT DETECTOR TEST SUITE\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    
    int passed = 0;
    int total = 8;
    
    passed += test_event_rate_random();
    passed += test_warmup_period();
    passed += test_volume_events();
    passed += test_imbalance_events();
    passed += test_quantile_accuracy();
    passed += test_latency();
    passed += test_combined_events();
    passed += test_different_quantiles();
    
    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("       RESULTS: %d/%d tests passed\n", passed, total);
    printf("═══════════════════════════════════════════════════════════════\n");
    
    return (passed == total) ? 0 : 1;
}
