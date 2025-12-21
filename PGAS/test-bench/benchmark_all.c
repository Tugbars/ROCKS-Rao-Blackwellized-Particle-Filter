/**
 * @file benchmark_all.c
 * @brief Comprehensive benchmark for PGAS-MKL + PARIS-MKL + Lifeboat
 *
 * Benchmarks:
 *   1. PGAS-MKL single sweep and adaptive run
 *   2. PARIS-MKL backward smoothing
 *   3. Lifeboat full injection cycle
 *   4. Combined PGAS+PARIS timing
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "pgas_mkl.h"
#include "paris_mkl.h"
#include "lifeboat.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#define TRIALS 5
#define WARMUP 2

/*═══════════════════════════════════════════════════════════════════════════════
 * CROSS-PLATFORM TIMING
 *═══════════════════════════════════════════════════════════════════════════════*/

#ifdef _WIN32
#include <windows.h>
static double get_time_ms(void)
{
    static LARGE_INTEGER freq = {0};
    LARGE_INTEGER counter;
    if (freq.QuadPart == 0)
        QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart * 1000.0 / (double)freq.QuadPart;
}
#else
static double get_time_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}
#endif

/*═══════════════════════════════════════════════════════════════════════════════
 * DATA GENERATION
 *═══════════════════════════════════════════════════════════════════════════════*/

static void generate_model_data(int T, int K, int N,
                                double *trans, double *mu_vol, double *sigma_vol,
                                double *phi, double *sigma_h,
                                double *observations,
                                int *true_regimes, double *true_h)
{
    *phi = 0.97;
    *sigma_h = 0.15;

    /* Sticky transition matrix */
    for (int i = 0; i < K; i++)
    {
        for (int j = 0; j < K; j++)
        {
            trans[i * K + j] = (i == j) ? 0.95 : 0.05 / (K - 1);
        }
        mu_vol[i] = -2.0 + i * 1.5;
        sigma_vol[i] = 0.3;
    }

    /* Generate trajectory */
    true_regimes[0] = 0;
    true_h[0] = mu_vol[0];

    for (int t = 1; t < T; t++)
    {
        double u = (double)rand() / RAND_MAX;
        double cumsum = 0;
        int prev = true_regimes[t - 1];
        true_regimes[t] = prev;
        for (int j = 0; j < K; j++)
        {
            cumsum += trans[prev * K + j];
            if (u < cumsum)
            {
                true_regimes[t] = j;
                break;
            }
        }

        double mu_k = mu_vol[true_regimes[t]];
        double noise = *sigma_h * ((double)rand() / RAND_MAX * 2 - 1) * 1.732;
        true_h[t] = mu_k + (*phi) * (true_h[t - 1] - mu_k) + noise;
    }

    /* Observations */
    for (int t = 0; t < T; t++)
    {
        double vol = exp(true_h[t] / 2);
        observations[t] = vol * ((double)rand() / RAND_MAX * 2 - 1) * 1.732;
    }
}

/*═══════════════════════════════════════════════════════════════════════════════
 * BENCHMARKS
 *═══════════════════════════════════════════════════════════════════════════════*/

static void bench_pgas_mkl(int T, int K, int N, int trials,
                           double *trans, double *mu_vol, double *sigma_vol,
                           double phi, double sigma_h,
                           double *obs, int *true_reg, double *true_h)
{
    double sweep_times[100], adaptive_times[100];
    float acceptance[100];

    for (int i = 0; i < trials + WARMUP; i++)
    {
        PGASMKLState *state = pgas_mkl_alloc(N, T, K, i + 1000);
        pgas_mkl_set_model(state, trans, mu_vol, sigma_vol, phi, sigma_h);
        pgas_mkl_set_reference(state, true_reg, true_h, T);
        pgas_mkl_load_observations(state, obs, T);

        /* Single sweep */
        double t0 = get_time_ms();
        pgas_mkl_csmc_sweep(state);
        double t1 = get_time_ms();

        if (i >= WARMUP)
        {
            sweep_times[i - WARMUP] = t1 - t0;
        }

        /* Adaptive */
        pgas_mkl_set_reference(state, true_reg, true_h, T);
        t0 = get_time_ms();
        pgas_mkl_run_adaptive(state);
        t1 = get_time_ms();

        if (i >= WARMUP)
        {
            adaptive_times[i - WARMUP] = t1 - t0;
            acceptance[i - WARMUP] = state->acceptance_rate;
        }

        pgas_mkl_free(state);
    }

    /* Stats */
    double sweep_avg = 0, adaptive_avg = 0, accept_avg = 0;
    for (int i = 0; i < trials; i++)
    {
        sweep_avg += sweep_times[i];
        adaptive_avg += adaptive_times[i];
        accept_avg += acceptance[i];
    }
    sweep_avg /= trials;
    adaptive_avg /= trials;
    accept_avg /= trials;

    printf("  PGAS-MKL:     sweep=%.2f ms, adaptive=%.2f ms, accept=%.2f\n",
           sweep_avg, adaptive_avg, accept_avg);
}

static void bench_paris_mkl(int T, int K, int N, int trials,
                            double *trans, double *mu_vol,
                            double phi, double sigma_h)
{
    /* Generate particle data */
    int *regimes = malloc(T * N * sizeof(int));
    double *h = malloc(T * N * sizeof(double));
    double *weights = malloc(T * N * sizeof(double));
    int *ancestors = malloc(T * N * sizeof(int));

    for (int t = 0; t < T; t++)
    {
        double wsum = 0;
        for (int n = 0; n < N; n++)
        {
            int idx = t * N + n;
            regimes[idx] = rand() % K;
            h[idx] = mu_vol[regimes[idx]] + 0.2 * ((double)rand() / RAND_MAX - 0.5);
            ancestors[idx] = (t == 0) ? n : (rand() % N);
            weights[idx] = 0.5 + (double)rand() / RAND_MAX;
            wsum += weights[idx];
        }
        for (int n = 0; n < N; n++)
            weights[t * N + n] /= wsum;
    }

    double smooth_times[100];

    for (int i = 0; i < trials + WARMUP; i++)
    {
        PARISMKLState *state = paris_mkl_alloc(N, T, K, i + 2000);
        paris_mkl_set_model(state, trans, mu_vol, phi, sigma_h);
        paris_mkl_load_particles(state, regimes, h, weights, ancestors, T);

        double t0 = get_time_ms();
        paris_mkl_backward_smooth(state);
        double t1 = get_time_ms();

        if (i >= WARMUP)
        {
            smooth_times[i - WARMUP] = t1 - t0;
        }

        paris_mkl_free(state);
    }

    double smooth_avg = 0;
    for (int i = 0; i < trials; i++)
        smooth_avg += smooth_times[i];
    smooth_avg /= trials;

    printf("  PARIS-MKL:    smooth=%.2f ms (%.1f us/tick)\n",
           smooth_avg, smooth_avg * 1000 / T);

    free(regimes);
    free(h);
    free(weights);
    free(ancestors);
}

static void bench_lifeboat(int T, int K, int N, int trials,
                           float *trans_f, float *mu_vol_f, float *sigma_vol_f,
                           float phi_f, float sigma_h_f,
                           float *obs_f, uint64_t *tick_ids)
{
    double total_times[100];
    double inject_times[100];
    float acceptances[100];

    /* RNG state */
    uint64_t rng_state[2];
    lifeboat_rng_init(rng_state, 12345);

    /* RBPF particles */
    int *rbpf_regimes = malloc(N * sizeof(int));
    float *rbpf_h = malloc(N * sizeof(float));
    float *rbpf_weights = malloc(N * sizeof(float));

    for (int i = 0; i < trials + WARMUP; i++)
    {
        LifeboatManager *mgr = lifeboat_create(N, K, T, i + 3000);
        lifeboat_set_model(mgr, trans_f, mu_vol_f, sigma_vol_f, phi_f, sigma_h_f);

        /* Init particles */
        for (int n = 0; n < N; n++)
        {
            rbpf_regimes[n] = rand() % K;
            rbpf_h[n] = mu_vol_f[rbpf_regimes[n]];
            rbpf_weights[n] = 1.0f / N;
        }

        /* Full lifeboat cycle */
        double t0 = get_time_ms();

        lifeboat_trigger_manual(mgr);
        lifeboat_start_run(mgr, obs_f, tick_ids, T, NULL, NULL);

        while (!lifeboat_is_ready(mgr))
        {
            /* Busy wait (in production would do other work) */
        }

        double t1 = get_time_ms();

        /* Injection */
        const LifeboatCloud *cloud = lifeboat_get_cloud(mgr);
        uint64_t source_tick;

        double t2 = get_time_ms();
        lifeboat_inject(mgr, rbpf_regimes, rbpf_h, rbpf_weights, &source_tick, rng_state);

        /* Fast-forward 20 ticks */
        if (cloud && source_tick + 21 < (uint64_t)T)
        {
            lifeboat_fast_forward(cloud, rbpf_regimes, rbpf_h, rbpf_weights,
                                  &obs_f[source_tick + 1], 20, true, rng_state);
        }
        double t3 = get_time_ms();

        if (i >= WARMUP)
        {
            total_times[i - WARMUP] = t1 - t0;
            inject_times[i - WARMUP] = t3 - t2;
            acceptances[i - WARMUP] = cloud ? cloud->ancestor_acceptance : 0;
        }

        lifeboat_consume_cloud(mgr);
        lifeboat_destroy(mgr);
    }

    double total_avg = 0, inject_avg = 0, accept_avg = 0;
    for (int i = 0; i < trials; i++)
    {
        total_avg += total_times[i];
        inject_avg += inject_times[i];
        accept_avg += acceptances[i];
    }
    total_avg /= trials;
    inject_avg /= trials;
    accept_avg /= trials;

    printf("  Lifeboat:     total=%.2f ms, inject+ff=%.3f ms, accept=%.2f\n",
           total_avg, inject_avg, accept_avg);

    free(rbpf_regimes);
    free(rbpf_h);
    free(rbpf_weights);
}

/*═══════════════════════════════════════════════════════════════════════════════
 * MAIN
 *═══════════════════════════════════════════════════════════════════════════════*/

int main(void)
{
    printf("╔═══════════════════════════════════════════════════════════════════════╗\n");
    printf("║                PGAS + PARIS + LIFEBOAT BENCHMARK                      ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════════╝\n\n");

#ifdef _OPENMP
    printf("OpenMP threads: %d\n", omp_get_max_threads());
#endif
    printf("Trials: %d (+ %d warmup)\n\n", TRIALS, WARMUP);

    srand(42);

    /* Test configurations */
    int configs[][3] = {
        /* T, K, N */
        {100, 4, 64},
        {200, 4, 100},
        {300, 4, 100},
        {500, 4, 100},
        {200, 8, 100},
    };
    int n_configs = sizeof(configs) / sizeof(configs[0]);

    for (int c = 0; c < n_configs; c++)
    {
        int T = configs[c][0];
        int K = configs[c][1];
        int N = configs[c][2];

        printf("═══════════════════════════════════════════════════════════════════════\n");
        printf("Configuration: T=%d, K=%d, N=%d\n", T, K, N);
        printf("═══════════════════════════════════════════════════════════════════════\n");

        /* Allocate data */
        double *trans = malloc(K * K * sizeof(double));
        double *mu_vol = malloc(K * sizeof(double));
        double *sigma_vol = malloc(K * sizeof(double));
        double phi, sigma_h;
        double *obs = malloc(T * sizeof(double));
        int *true_reg = malloc(T * sizeof(int));
        double *true_h = malloc(T * sizeof(double));

        generate_model_data(T, K, N, trans, mu_vol, sigma_vol, &phi, &sigma_h,
                            obs, true_reg, true_h);

        /* Convert for lifeboat */
        float *trans_f = malloc(K * K * sizeof(float));
        float *mu_vol_f = malloc(K * sizeof(float));
        float *sigma_vol_f = malloc(K * sizeof(float));
        float *obs_f = malloc(T * sizeof(float));
        uint64_t *tick_ids = malloc(T * sizeof(uint64_t));

        for (int i = 0; i < K * K; i++)
            trans_f[i] = (float)trans[i];
        for (int i = 0; i < K; i++)
        {
            mu_vol_f[i] = (float)mu_vol[i];
            sigma_vol_f[i] = (float)sigma_vol[i];
        }
        for (int t = 0; t < T; t++)
        {
            obs_f[t] = (float)obs[t];
            tick_ids[t] = t;
        }

        /* Run benchmarks */
        bench_pgas_mkl(T, K, N, TRIALS, trans, mu_vol, sigma_vol, phi, sigma_h,
                       obs, true_reg, true_h);

        bench_paris_mkl(T, K, N, TRIALS, trans, mu_vol, phi, sigma_h);

        bench_lifeboat(T, K, N, TRIALS, trans_f, mu_vol_f, sigma_vol_f,
                       (float)phi, (float)sigma_h, obs_f, tick_ids);

        printf("\n");

        /* Cleanup */
        free(trans);
        free(mu_vol);
        free(sigma_vol);
        free(obs);
        free(true_reg);
        free(true_h);
        free(trans_f);
        free(mu_vol_f);
        free(sigma_vol_f);
        free(obs_f);
        free(tick_ids);
    }

    printf("✓ Benchmark complete!\n");
    return 0;
}