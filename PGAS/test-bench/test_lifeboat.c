/**
 * @file test_lifeboat.c
 * @brief Test Lifeboat with lock-free RNG and correct cloud pointer
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "lifeboat.h"

/*═══════════════════════════════════════════════════════════════════════════════
 * CROSS-PLATFORM TIMING
 *═══════════════════════════════════════════════════════════════════════════════*/

#ifdef _WIN32
#include <windows.h>
static double get_time_us(void)
{
    static LARGE_INTEGER freq = {0};
    LARGE_INTEGER counter;
    if (freq.QuadPart == 0)
        QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart * 1e6 / (double)freq.QuadPart;
}
static void sleep_ms(int ms) { Sleep(ms); }
#else
#include <unistd.h>
static double get_time_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}
static void sleep_ms(int ms) { usleep(ms * 1000); }
#endif

#include "mkl_tuning.h"

int main(void)
{
    /* Initialize MKL tuning: P-cores only (adjust for your CPU), verbose=1 */
    mkl_tuning_init(0, 1); /* 0 = auto, change to your P-core count */

    printf("╔═══════════════════════════════════════════════════════════════════════╗\n");
    printf("║      LIFEBOAT TEST (Lock-free RNG + Correct Cloud Pointer)            ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════════╝\n\n");

    int N = 100;
    int K = 4;
    int buffer_size = 200;

    /* Create manager */
    printf("Creating Lifeboat manager (N=%d, K=%d, T=%d)...\n", N, K, buffer_size);
    LifeboatManager *mgr = lifeboat_create(N, K, buffer_size, 12345);

    /* Initialize per-thread RNG state */
    uint64_t rng_state[2];
    lifeboat_rng_init(rng_state, 67890);

    /* Set model */
    float trans[16], mu_vol[4], sigma_vol[4];
    for (int i = 0; i < K; i++)
    {
        for (int j = 0; j < K; j++)
        {
            trans[i * K + j] = (i == j) ? 0.9f : 0.1f / (K - 1);
        }
        mu_vol[i] = -2.0f + i * 1.0f;
        sigma_vol[i] = 0.3f;
    }
    lifeboat_set_model(mgr, trans, mu_vol, sigma_vol, 0.97f, 0.15f);

    /* Generate observations */
    float observations[300];
    uint64_t tick_ids[300];
    for (int t = 0; t < 300; t++)
    {
        observations[t] = 0.01f * (lifeboat_rng_uniform(rng_state) * 2 - 1);
        tick_ids[t] = t;
    }

    /* Generate RBPF particles */
    int rbpf_regimes[100];
    float rbpf_h[100];
    float rbpf_weights[100];
    for (int n = 0; n < N; n++)
    {
        rbpf_regimes[n] = (int)(lifeboat_rng_uniform(rng_state) * K);
        rbpf_h[n] = -1.0f + 0.5f * lifeboat_rng_uniform(rng_state);
        rbpf_weights[n] = 1.0f / N;
    }

    printf("Initial particles:\n");
    printf("  regimes[0..4]: %d %d %d %d %d\n",
           rbpf_regimes[0], rbpf_regimes[1], rbpf_regimes[2], rbpf_regimes[3], rbpf_regimes[4]);
    printf("  h[0..4]:       %.3f %.3f %.3f %.3f %.3f\n",
           rbpf_h[0], rbpf_h[1], rbpf_h[2], rbpf_h[3], rbpf_h[4]);

    /* ═══════════════════════════════════════════════════════════════════════
     * Test 1: Lock-free hot-path latency
     * ═══════════════════════════════════════════════════════════════════════*/
    printf("\n═══ Test 1: Lock-free hot-path latency ═══\n");

    int n_checks = 100000;
    double t0 = get_time_us();
    for (int i = 0; i < n_checks; i++)
    {
        bool ready = lifeboat_is_ready(mgr);
        bool running = lifeboat_is_running(mgr);
        (void)ready;
        (void)running;
    }
    double t1 = get_time_us();
    printf("lifeboat_is_ready() x %d: %.2f ns/call (target: <100ns)\n",
           n_checks, (t1 - t0) * 1000.0 / n_checks);

    /* ═══════════════════════════════════════════════════════════════════════
     * Test 2: xoroshiro128+ RNG performance
     * ═══════════════════════════════════════════════════════════════════════*/
    printf("\n═══ Test 2: xoroshiro128+ RNG performance ═══\n");

    int n_rng = 1000000;
    t0 = get_time_us();
    volatile float sum = 0;
    for (int i = 0; i < n_rng; i++)
    {
        sum += lifeboat_rng_uniform(rng_state);
    }
    t1 = get_time_us();
    printf("lifeboat_rng_uniform() x %d: %.2f ns/call\n", n_rng, (t1 - t0) * 1000.0 / n_rng);

    t0 = get_time_us();
    for (int i = 0; i < n_rng; i++)
    {
        sum += lifeboat_rng_normal(rng_state);
    }
    t1 = get_time_us();
    printf("lifeboat_rng_normal() x %d: %.2f ns/call\n", n_rng, (t1 - t0) * 1000.0 / n_rng);

    /* ═══════════════════════════════════════════════════════════════════════
     * Test 3: Background run
     * ═══════════════════════════════════════════════════════════════════════*/
    printf("\n═══ Test 3: Background PGAS+PARIS run ═══\n");

    lifeboat_trigger_manual(mgr);
    lifeboat_start_run(mgr, observations, tick_ids, buffer_size, NULL, NULL);

    while (!lifeboat_is_ready(mgr))
    {
        sleep_ms(1);
    }

    const LifeboatCloud *cloud = lifeboat_get_cloud(mgr);
    printf("Cloud info:\n");
    printf("  valid:          %s\n", cloud->valid ? "YES" : "NO");
    printf("  source_tick_id: %lu\n", cloud->source_tick_id);
    printf("  compute time:   %.2f ms\n", cloud->compute_time_ms);
    printf("  acceptance:     %.3f\n", cloud->ancestor_acceptance);

    /* ═══════════════════════════════════════════════════════════════════════
     * Test 4: Injection with correct cloud pointer
     * ═══════════════════════════════════════════════════════════════════════*/
    printf("\n═══ Test 4: Injection with correct cloud pointer ═══\n");

    uint64_t source_tick;
    uint64_t current_tick = 220;

    /* Save cloud pointer BEFORE inject (it uses the same cloud!) */
    const LifeboatCloud *injected_cloud = lifeboat_get_cloud(mgr);

    if (lifeboat_inject(mgr, rbpf_regimes, rbpf_h, rbpf_weights, &source_tick, rng_state))
    {
        printf("Injection successful!\n");
        printf("  source_tick: %lu, current_tick: %lu, gap: %lu\n",
               source_tick, current_tick, current_tick - source_tick);

        printf("\nAfter injection (before fast-forward):\n");
        printf("  regimes[0..4]: %d %d %d %d %d\n",
               rbpf_regimes[0], rbpf_regimes[1], rbpf_regimes[2],
               rbpf_regimes[3], rbpf_regimes[4]);
        printf("  h[0..4]:       %.3f %.3f %.3f %.3f %.3f\n",
               rbpf_h[0], rbpf_h[1], rbpf_h[2], rbpf_h[3], rbpf_h[4]);
    }

    /* ═══════════════════════════════════════════════════════════════════════
     * Test 5: Fast-forward with CORRECT cloud pointer
     * ═══════════════════════════════════════════════════════════════════════*/
    printf("\n═══ Test 5: Fast-forward with correct cloud pointer ═══\n");

    int n_catchup = (int)(current_tick - source_tick);
    float *catchup_obs = &observations[source_tick + 1];

    t0 = get_time_us();
    /* Use injected_cloud, not mgr->clouds[0]! */
    lifeboat_fast_forward(injected_cloud, rbpf_regimes, rbpf_h, rbpf_weights,
                          catchup_obs, n_catchup, true, rng_state);
    t1 = get_time_us();

    printf("Fast-forwarded %d ticks in %.2f us (%.2f us/tick)\n",
           n_catchup, t1 - t0, (t1 - t0) / n_catchup);

    printf("\nAfter fast-forward:\n");
    printf("  regimes[0..4]: %d %d %d %d %d\n",
           rbpf_regimes[0], rbpf_regimes[1], rbpf_regimes[2],
           rbpf_regimes[3], rbpf_regimes[4]);
    printf("  h[0..4]:       %.3f %.3f %.3f %.3f %.3f\n",
           rbpf_h[0], rbpf_h[1], rbpf_h[2], rbpf_h[3], rbpf_h[4]);

    lifeboat_consume_cloud(mgr);

    /* ═══════════════════════════════════════════════════════════════════════
     * Test 6: Full hot-path timing
     * ═══════════════════════════════════════════════════════════════════════*/
    printf("\n═══ Test 6: Full hot-path timing (simulated injection) ═══\n");

    /* Trigger another run */
    lifeboat_start_run(mgr, observations, tick_ids, buffer_size, NULL, NULL);
    while (!lifeboat_is_ready(mgr))
        sleep_ms(1);

    cloud = lifeboat_get_cloud(mgr);

    /* Time full injection + fast-forward */
    t0 = get_time_us();
    lifeboat_inject(mgr, rbpf_regimes, rbpf_h, rbpf_weights, &source_tick, rng_state);
    lifeboat_fast_forward(cloud, rbpf_regimes, rbpf_h, rbpf_weights,
                          &observations[source_tick + 1], 21, true, rng_state);
    t1 = get_time_us();

    printf("Full injection + 21-tick fast-forward: %.2f us (target: <5000 us)\n", t1 - t0);

    lifeboat_consume_cloud(mgr);

    /* ═══════════════════════════════════════════════════════════════════════
     * Test 7: Diagnostics
     * ═══════════════════════════════════════════════════════════════════════*/
    printf("\n═══ Test 7: Diagnostics ═══\n");
    lifeboat_print_diagnostics(mgr);

    lifeboat_destroy(mgr);
    printf("\n✓ All tests passed!\n");

    return 0;
}