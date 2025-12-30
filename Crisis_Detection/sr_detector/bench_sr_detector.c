/*
 * bench_sr_detector.c - Performance benchmark for SR detector
 * 
 * Benchmarks:
 *   1. Log-LR batch computation (vectorizable)
 *   2. Full SR update batch (vectorized LR + sequential accumulation)
 *   3. Single-tick latency (scalar path, HFT-critical)
 *   4. Dual SR update latency
 * 
 * Compile with AVX-512:
 *   gcc -O3 -march=native -DNDEBUG bench_sr_detector.c sr_detector.c -lm -o bench_sr
 * 
 * Compile with MKL:
 *   gcc -O3 -march=native -DNDEBUG -D__INTEL_MKL__ -I${MKLROOT}/include \
 *       bench_sr_detector.c sr_detector.c \
 *       -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lm \
 *       -o bench_sr_mkl
 */

#include "sr_detector.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

/* ═══════════════════════════════════════════════════════════════════════════
 * HIGH-RESOLUTION TIMING
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

static inline double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* Estimate CPU frequency for cycle-to-ns conversion */
static double estimate_cpu_freq_ghz(void) {
    uint64_t start_cycles = rdtsc();
    double start_time = get_time_sec();
    
    /* Busy-wait for ~100ms */
    volatile double x = 1.0;
    for (int i = 0; i < 10000000; i++) {
        x = x * 1.00001 + 0.00001;
    }
    
    uint64_t end_cycles = rdtsc();
    double end_time = get_time_sec();
    
    double elapsed = end_time - start_time;
    uint64_t cycles = end_cycles - start_cycles;
    
    return (double)cycles / elapsed / 1e9;
}

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

/* ═══════════════════════════════════════════════════════════════════════════
 * BENCHMARK 1: Log-LR Batch Throughput
 * ═══════════════════════════════════════════════════════════════════════════ */

static void bench_log_lr_batch(const float *obs, float *log_lr, int n, 
                                const SRConfig *cfg, int reps) {
    printf("\n─── Benchmark: Log-LR Batch Computation ───\n");
    
    float sigma = 0.01f;
    
    /* Warmup */
    for (int r = 0; r < 10; r++) {
        sr_compute_log_lr_batch(obs, NULL, sigma, log_lr, n, cfg);
    }
    
    /* Timed runs */
    double start = get_time_sec();
    uint64_t start_cycles = rdtsc();
    
    for (int r = 0; r < reps; r++) {
        sr_compute_log_lr_batch(obs, NULL, sigma, log_lr, n, cfg);
    }
    
    uint64_t end_cycles = rdtsc();
    double end = get_time_sec();
    
    double elapsed = end - start;
    uint64_t total_cycles = end_cycles - start_cycles;
    int64_t total_obs = (int64_t)n * reps;
    
    printf("  N = %d, reps = %d\n", n, reps);
    printf("  Total time: %.3f sec\n", elapsed);
    printf("  Throughput: %.2f M obs/sec\n", total_obs / elapsed / 1e6);
    printf("  Cycles/obs: %.2f\n", (double)total_cycles / total_obs);
    printf("  ns/obs: %.2f\n", elapsed * 1e9 / total_obs);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * BENCHMARK 2: Full SR Update Batch
 * ═══════════════════════════════════════════════════════════════════════════ */

static void bench_sr_update_batch(const float *obs, float *log_sr, int n,
                                   const SRConfig *cfg, int reps) {
    printf("\n─── Benchmark: Full SR Update Batch ───\n");
    
    float sigma = 0.01f;
    
    /* Warmup */
    for (int r = 0; r < 10; r++) {
        sr_update_batch(obs, sigma, log_sr, 0.0f, n, cfg);
    }
    
    /* Timed runs */
    double start = get_time_sec();
    uint64_t start_cycles = rdtsc();
    
    for (int r = 0; r < reps; r++) {
        sr_update_batch(obs, sigma, log_sr, 0.0f, n, cfg);
    }
    
    uint64_t end_cycles = rdtsc();
    double end = get_time_sec();
    
    double elapsed = end - start;
    uint64_t total_cycles = end_cycles - start_cycles;
    int64_t total_obs = (int64_t)n * reps;
    
    printf("  N = %d, reps = %d\n", n, reps);
    printf("  Total time: %.3f sec\n", elapsed);
    printf("  Throughput: %.2f M obs/sec\n", total_obs / elapsed / 1e6);
    printf("  Cycles/obs: %.2f\n", (double)total_cycles / total_obs);
    printf("  ns/obs: %.2f\n", elapsed * 1e9 / total_obs);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * BENCHMARK 3: Single-Tick Latency (HFT-Critical)
 * ═══════════════════════════════════════════════════════════════════════════ */

static void bench_single_tick_latency(const float *obs, int n, 
                                       const SRConfig *cfg) {
    printf("\n─── Benchmark: Single-Tick Latency ───\n");
    
    SRStat sr;
    sr_stat_init(&sr, cfg);
    float sigma = 0.01f;
    
    /* Warmup */
    for (int i = 0; i < 1000; i++) {
        sr_stat_update(&sr, obs[i % n], sigma);
    }
    sr_stat_reset(&sr);
    
    /* Measure individual tick latencies */
    const int SAMPLES = 100000;
    uint64_t *latencies = (uint64_t *)malloc(SAMPLES * sizeof(uint64_t));
    
    for (int i = 0; i < SAMPLES; i++) {
        uint64_t start = rdtsc();
        sr_stat_update(&sr, obs[i % n], sigma);
        uint64_t end = rdtsc();
        latencies[i] = end - start;
    }
    
    /* Sort for percentiles */
    for (int i = 0; i < SAMPLES - 1; i++) {
        for (int j = i + 1; j < SAMPLES; j++) {
            if (latencies[j] < latencies[i]) {
                uint64_t tmp = latencies[i];
                latencies[i] = latencies[j];
                latencies[j] = tmp;
            }
        }
    }
    
    /* Compute statistics */
    uint64_t sum = 0;
    for (int i = 0; i < SAMPLES; i++) sum += latencies[i];
    double mean = (double)sum / SAMPLES;
    
    printf("  Samples: %d\n", SAMPLES);
    printf("  Mean:    %.1f cycles\n", mean);
    printf("  Median:  %lu cycles\n", latencies[SAMPLES / 2]);
    printf("  P95:     %lu cycles\n", latencies[(int)(SAMPLES * 0.95)]);
    printf("  P99:     %lu cycles\n", latencies[(int)(SAMPLES * 0.99)]);
    printf("  P99.9:   %lu cycles\n", latencies[(int)(SAMPLES * 0.999)]);
    printf("  Max:     %lu cycles\n", latencies[SAMPLES - 1]);
    
    free(latencies);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * BENCHMARK 4: Dual SR Update Latency
 * ═══════════════════════════════════════════════════════════════════════════ */

static void bench_dual_sr_latency(const float *obs, int n, const SRConfig *cfg) {
    printf("\n─── Benchmark: Dual SR Update Latency ───\n");
    
    DualSR dsr;
    dual_sr_init(&dsr, cfg, 0.01f);
    
    /* Warmup */
    for (int i = 0; i < 1000; i++) {
        dual_sr_update_entry(&dsr, obs[i % n]);
    }
    dual_sr_reset(&dsr);
    
    /* Measure entry update latency */
    const int SAMPLES = 100000;
    uint64_t *latencies = (uint64_t *)malloc(SAMPLES * sizeof(uint64_t));
    
    for (int i = 0; i < SAMPLES; i++) {
        uint64_t start = rdtsc();
        dual_sr_update_entry(&dsr, obs[i % n]);
        uint64_t end = rdtsc();
        latencies[i] = end - start;
    }
    
    /* Compute mean */
    uint64_t sum = 0;
    for (int i = 0; i < SAMPLES; i++) sum += latencies[i];
    double mean_entry = (double)sum / SAMPLES;
    
    /* Measure exit update latency */
    dual_sr_reset(&dsr);
    dsr.sigma_crisis = 0.03f;  /* Simulate learned crisis sigma */
    
    for (int i = 0; i < SAMPLES; i++) {
        uint64_t start = rdtsc();
        dual_sr_update_exit(&dsr, obs[i % n]);
        uint64_t end = rdtsc();
        latencies[i] = end - start;
    }
    
    sum = 0;
    for (int i = 0; i < SAMPLES; i++) sum += latencies[i];
    double mean_exit = (double)sum / SAMPLES;
    
    /* Measure both update latency */
    dual_sr_reset(&dsr);
    
    for (int i = 0; i < SAMPLES; i++) {
        uint64_t start = rdtsc();
        dual_sr_update_both(&dsr, obs[i % n]);
        uint64_t end = rdtsc();
        latencies[i] = end - start;
    }
    
    sum = 0;
    for (int i = 0; i < SAMPLES; i++) sum += latencies[i];
    double mean_both = (double)sum / SAMPLES;
    
    printf("  dual_sr_update_entry: %.1f cycles\n", mean_entry);
    printf("  dual_sr_update_exit:  %.1f cycles\n", mean_exit);
    printf("  dual_sr_update_both:  %.1f cycles\n", mean_both);
    
    free(latencies);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * BENCHMARK 5: Adaptive Threshold Latency
 * ═══════════════════════════════════════════════════════════════════════════ */

static void bench_adaptive_threshold(void) {
    printf("\n─── Benchmark: Adaptive Threshold ───\n");
    
    AdaptiveThreshold at;
    adaptive_threshold_init(&at, NULL);
    
    const int SAMPLES = 1000000;
    
    /* Measure compute latency */
    uint64_t start = rdtsc();
    volatile float result = 0;
    for (int i = 0; i < SAMPLES; i++) {
        result += adaptive_threshold_compute(&at);
        adaptive_threshold_tick(&at);
    }
    uint64_t end = rdtsc();
    
    double cycles_per_call = (double)(end - start) / SAMPLES;
    
    printf("  adaptive_threshold_compute + tick: %.1f cycles\n", cycles_per_call);
    (void)result;  /* Prevent optimization */
}

/* ═══════════════════════════════════════════════════════════════════════════
 * BENCHMARK 6: Student-t vs Gaussian
 * ═══════════════════════════════════════════════════════════════════════════ */

static void bench_student_t_vs_gaussian(const float *obs, float *log_lr, int n) {
    printf("\n─── Benchmark: Student-t vs Gaussian ───\n");
    
    float sigma = 0.01f;
    int reps = 100;
    
    /* Gaussian */
    SRConfig cfg_gauss = sr_config_default();
    
    double start = get_time_sec();
    for (int r = 0; r < reps; r++) {
        sr_compute_log_lr_batch(obs, NULL, sigma, log_lr, n, &cfg_gauss);
    }
    double elapsed_gauss = get_time_sec() - start;
    
    /* Student-t (nu=6) */
    SRConfig cfg_student = sr_config_student_t(6.0f);
    
    start = get_time_sec();
    for (int r = 0; r < reps; r++) {
        sr_compute_log_lr_batch(obs, NULL, sigma, log_lr, n, &cfg_student);
    }
    double elapsed_student = get_time_sec() - start;
    
    int64_t total_obs = (int64_t)n * reps;
    
    printf("  Gaussian:  %.2f M obs/sec\n", total_obs / elapsed_gauss / 1e6);
    printf("  Student-t: %.2f M obs/sec\n", total_obs / elapsed_student / 1e6);
    printf("  Slowdown:  %.2fx\n", elapsed_student / elapsed_gauss);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * MAIN
 * ═══════════════════════════════════════════════════════════════════════════ */

int main(void) {
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("       SHIRYAEV-ROBERTS DETECTOR BENCHMARK\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    
    /* Detect backend */
#if defined(__INTEL_MKL__)
    printf("Backend: Intel MKL\n");
#elif defined(__AVX512F__)
    printf("Backend: AVX-512\n");
#elif defined(__AVX2__)
    printf("Backend: AVX2\n");
#elif defined(__AVX__)
    printf("Backend: AVX\n");
#else
    printf("Backend: Scalar\n");
#endif
    
    /* Estimate CPU frequency */
    double freq_ghz = estimate_cpu_freq_ghz();
    printf("CPU Frequency: ~%.2f GHz\n", freq_ghz);
    
    /* Allocate test data */
    const int N = 100000;
    float *obs = (float *)aligned_alloc(64, N * sizeof(float));
    float *log_lr = (float *)aligned_alloc(64, N * sizeof(float));
    float *log_sr = (float *)aligned_alloc(64, N * sizeof(float));
    
    /* Generate random observations */
    rng_state = 12345;
    float sigma = 0.01f;
    for (int i = 0; i < N; i++) {
        obs[i] = sigma * randn();
    }
    
    SRConfig cfg = sr_config_default();
    
    /* Run benchmarks */
    bench_log_lr_batch(obs, log_lr, N, &cfg, 1000);
    bench_sr_update_batch(obs, log_sr, N, &cfg, 1000);
    bench_single_tick_latency(obs, N, &cfg);
    bench_dual_sr_latency(obs, N, &cfg);
    bench_adaptive_threshold();
    bench_student_t_vs_gaussian(obs, log_lr, N);
    
    /* Summary */
    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("       SUMMARY\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("For HFT tick loop (single-tick update):\n");
    printf("  Target: < 100 cycles (~30-50ns at 3GHz)\n");
    printf("  Dual SR update gives you entry + exit detection in one call\n");
    printf("\nFor backtesting (batch processing):\n");
    printf("  Log-LR batch is fully vectorized (16-way AVX-512)\n");
    printf("  Full SR update is bottlenecked by sequential accumulation\n");
    
    free(obs);
    free(log_lr);
    free(log_sr);
    
    return 0;
}
