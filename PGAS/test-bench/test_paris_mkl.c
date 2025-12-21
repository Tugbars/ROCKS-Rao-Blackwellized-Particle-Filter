/**
 * @file test_paris_mkl.c
 * @brief Benchmark and correctness test for MKL-optimized PARIS backward smoother
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "paris_mkl.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#define TRIALS 10

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
 * TEST DATA GENERATION
 *═══════════════════════════════════════════════════════════════════════════════*/

static void generate_particle_data(int T, int K, int N,
                                   double *trans, double *mu_vol,
                                   double *phi, double *sigma_h,
                                   int *regimes, double *h,
                                   double *weights, int *ancestors)
{
    *phi = 0.97;
    *sigma_h = 0.15;

    /* Sticky transition matrix */
    for (int i = 0; i < K; i++)
    {
        for (int j = 0; j < K; j++)
        {
            trans[i * K + j] = (i == j) ? 0.9 : 0.1 / (K - 1);
        }
        mu_vol[i] = -2.0 + i * 1.0;
    }

    /* Generate particle trajectories (simulate forward filtering) */
    for (int t = 0; t < T; t++)
    {
        double wsum = 0;

        for (int n = 0; n < N; n++)
        {
            int idx = t * N + n;

            if (t == 0)
            {
                regimes[idx] = rand() % K;
                h[idx] = mu_vol[regimes[idx]] + 0.2 * ((double)rand() / RAND_MAX - 0.5);
                ancestors[idx] = n;
            }
            else
            {
                /* Sample ancestor */
                int prev_t = (t - 1) * N;
                int a = rand() % N;
                ancestors[idx] = a;

                /* Sample regime transition */
                int prev_regime = regimes[prev_t + a];
                double u = (double)rand() / RAND_MAX;
                double cumsum = 0;
                regimes[idx] = prev_regime;
                for (int j = 0; j < K; j++)
                {
                    cumsum += trans[prev_regime * K + j];
                    if (u < cumsum)
                    {
                        regimes[idx] = j;
                        break;
                    }
                }

                /* Propagate h */
                double mu_k = mu_vol[regimes[idx]];
                double h_prev = h[prev_t + a];
                double noise = *sigma_h * ((double)rand() / RAND_MAX * 2 - 1) * 1.732;
                h[idx] = mu_k + (*phi) * (h_prev - mu_k) + noise;
            }

            /* Random weights (will normalize) */
            weights[idx] = 0.5 + (double)rand() / RAND_MAX;
            wsum += weights[idx];
        }

        /* Normalize weights at time t */
        for (int n = 0; n < N; n++)
        {
            weights[t * N + n] /= wsum;
        }
    }
}

/*═══════════════════════════════════════════════════════════════════════════════
 * CORRECTNESS TESTS
 *═══════════════════════════════════════════════════════════════════════════════*/

static int test_allocation(void)
{
    printf("  [1] Allocation test... ");

    PARISMKLState *state = paris_mkl_alloc(100, 200, 4, 12345);
    if (!state)
    {
        printf("FAILED (null state)\n");
        return 1;
    }

    if (state->N != 100 || state->T != 200 || state->K != 4)
    {
        printf("FAILED (wrong dimensions)\n");
        paris_mkl_free(state);
        return 1;
    }

    /* Check N_padded is multiple of 16 */
    if (state->N_padded % 16 != 0)
    {
        printf("FAILED (N_padded=%d not multiple of 16)\n", state->N_padded);
        paris_mkl_free(state);
        return 1;
    }

    /* Check thread RNG streams */
    if (state->n_thread_streams < 1)
    {
        printf("FAILED (no thread RNG streams)\n");
        paris_mkl_free(state);
        return 1;
    }

    paris_mkl_free(state);
    printf("OK\n");
    return 0;
}

static int test_model_set(void)
{
    printf("  [2] Model parameter test... ");

    PARISMKLState *state = paris_mkl_alloc(50, 100, 4, 12345);

    double trans[16], mu_vol[4];
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            trans[i * 4 + j] = (i == j) ? 0.85 : 0.15 / 3;
        }
        mu_vol[i] = -1.5 + i * 0.75;
    }

    paris_mkl_set_model(state, trans, mu_vol, 0.95, 0.18);

    /* Verify model stored correctly */
    if (fabsf(state->model.phi - 0.95f) > 1e-5f)
    {
        printf("FAILED (phi mismatch)\n");
        paris_mkl_free(state);
        return 1;
    }

    if (fabsf(state->model.sigma_h - 0.18f) > 1e-5f)
    {
        printf("FAILED (sigma_h mismatch)\n");
        paris_mkl_free(state);
        return 1;
    }

    /* Check log_trans computed */
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            float expected = logf((float)trans[i * 4 + j] + 1e-10f);
            float actual = state->model.log_trans[i * 4 + j];
            if (fabsf(expected - actual) > 1e-4f)
            {
                printf("FAILED (log_trans mismatch at [%d,%d])\n", i, j);
                paris_mkl_free(state);
                return 1;
            }
        }
    }

    paris_mkl_free(state);
    printf("OK\n");
    return 0;
}

static int test_load_particles(void)
{
    printf("  [3] Particle loading test... ");

    int T = 50, N = 32, K = 4;
    PARISMKLState *state = paris_mkl_alloc(N, T, K, 12345);

    double trans[16], mu_vol[4], phi, sigma_h;
    int *regimes = malloc(T * N * sizeof(int));
    double *h = malloc(T * N * sizeof(double));
    double *weights = malloc(T * N * sizeof(double));
    int *ancestors = malloc(T * N * sizeof(int));

    generate_particle_data(T, K, N, trans, mu_vol, &phi, &sigma_h,
                           regimes, h, weights, ancestors);

    paris_mkl_set_model(state, trans, mu_vol, phi, sigma_h);
    paris_mkl_load_particles(state, regimes, h, weights, ancestors, T);

    /* Verify data loaded correctly */
    int Np = state->N_padded;
    for (int t = 0; t < T; t++)
    {
        for (int n = 0; n < N; n++)
        {
            int src = t * N + n;
            int dst = t * Np + n;

            if (state->regimes[dst] != regimes[src])
            {
                printf("FAILED (regime mismatch at t=%d, n=%d)\n", t, n);
                free(regimes);
                free(h);
                free(weights);
                free(ancestors);
                paris_mkl_free(state);
                return 1;
            }

            if (fabsf(state->h[dst] - (float)h[src]) > 1e-5f)
            {
                printf("FAILED (h mismatch at t=%d, n=%d)\n", t, n);
                free(regimes);
                free(h);
                free(weights);
                free(ancestors);
                paris_mkl_free(state);
                return 1;
            }
        }
    }

    free(regimes);
    free(h);
    free(weights);
    free(ancestors);
    paris_mkl_free(state);
    printf("OK\n");
    return 0;
}

static int test_backward_smooth(void)
{
    printf("  [4] Backward smoothing test... ");

    int T = 100, N = 64, K = 4;
    PARISMKLState *state = paris_mkl_alloc(N, T, K, 54321);

    double trans[16], mu_vol[4], phi, sigma_h;
    int *regimes = malloc(T * N * sizeof(int));
    double *h = malloc(T * N * sizeof(double));
    double *weights = malloc(T * N * sizeof(double));
    int *ancestors = malloc(T * N * sizeof(int));

    generate_particle_data(T, K, N, trans, mu_vol, &phi, &sigma_h,
                           regimes, h, weights, ancestors);

    paris_mkl_set_model(state, trans, mu_vol, phi, sigma_h);
    paris_mkl_load_particles(state, regimes, h, weights, ancestors, T);

    /* Run backward smoothing */
    paris_mkl_backward_smooth(state);

    /* Check smoothed indices are valid */
    int Np = state->N_padded;
    for (int t = 0; t < T; t++)
    {
        for (int n = 0; n < N; n++)
        {
            int idx = state->smoothed[t * Np + n];
            if (idx < 0 || idx >= N)
            {
                printf("FAILED (invalid smoothed index at t=%d, n=%d: %d)\n", t, n, idx);
                free(regimes);
                free(h);
                free(weights);
                free(ancestors);
                paris_mkl_free(state);
                return 1;
            }
        }
    }

    /* Check final time is identity (smoothed[T-1][n] == n) */
    for (int n = 0; n < N; n++)
    {
        int idx = state->smoothed[(T - 1) * Np + n];
        if (idx != n)
        {
            printf("FAILED (final time not identity: smoothed[%d]=%d)\n", n, idx);
            free(regimes);
            free(h);
            free(weights);
            free(ancestors);
            paris_mkl_free(state);
            return 1;
        }
    }

    free(regimes);
    free(h);
    free(weights);
    free(ancestors);
    paris_mkl_free(state);
    printf("OK\n");
    return 0;
}

static int test_get_smoothed(void)
{
    printf("  [5] Get smoothed particles test... ");

    int T = 80, N = 50, K = 4;
    PARISMKLState *state = paris_mkl_alloc(N, T, K, 99999);

    double trans[16], mu_vol[4], phi, sigma_h;
    int *regimes = malloc(T * N * sizeof(int));
    double *h = malloc(T * N * sizeof(double));
    double *weights = malloc(T * N * sizeof(double));
    int *ancestors = malloc(T * N * sizeof(int));

    generate_particle_data(T, K, N, trans, mu_vol, &phi, &sigma_h,
                           regimes, h, weights, ancestors);

    paris_mkl_set_model(state, trans, mu_vol, phi, sigma_h);
    paris_mkl_load_particles(state, regimes, h, weights, ancestors, T);
    paris_mkl_backward_smooth(state);

    /* Get smoothed particles at middle time */
    int t_mid = T / 2;
    int *out_regimes = malloc(N * sizeof(int));
    float *out_h = malloc(N * sizeof(float));

    paris_mkl_get_smoothed(state, t_mid, out_regimes, out_h);

    /* Check regimes are valid */
    for (int n = 0; n < N; n++)
    {
        if (out_regimes[n] < 0 || out_regimes[n] >= K)
        {
            printf("FAILED (invalid regime at n=%d: %d)\n", n, out_regimes[n]);
            free(regimes);
            free(h);
            free(weights);
            free(ancestors);
            free(out_regimes);
            free(out_h);
            paris_mkl_free(state);
            return 1;
        }
    }

    /* Check h values are finite */
    for (int n = 0; n < N; n++)
    {
        if (!isfinite(out_h[n]))
        {
            printf("FAILED (non-finite h at n=%d: %f)\n", n, out_h[n]);
            free(regimes);
            free(h);
            free(weights);
            free(ancestors);
            free(out_regimes);
            free(out_h);
            paris_mkl_free(state);
            return 1;
        }
    }

    free(regimes);
    free(h);
    free(weights);
    free(ancestors);
    free(out_regimes);
    free(out_h);
    paris_mkl_free(state);
    printf("OK\n");
    return 0;
}

static int test_trajectory_extraction(void)
{
    printf("  [6] Trajectory extraction test... ");

    int T = 60, N = 40, K = 4;
    PARISMKLState *state = paris_mkl_alloc(N, T, K, 77777);

    double trans[16], mu_vol[4], phi, sigma_h;
    int *regimes = malloc(T * N * sizeof(int));
    double *h = malloc(T * N * sizeof(double));
    double *weights = malloc(T * N * sizeof(double));
    int *ancestors = malloc(T * N * sizeof(int));

    generate_particle_data(T, K, N, trans, mu_vol, &phi, &sigma_h,
                           regimes, h, weights, ancestors);

    paris_mkl_set_model(state, trans, mu_vol, phi, sigma_h);
    paris_mkl_load_particles(state, regimes, h, weights, ancestors, T);
    paris_mkl_backward_smooth(state);

    /* Extract trajectory for particle 0 */
    int *traj_regimes = malloc(T * sizeof(int));
    float *traj_h = malloc(T * sizeof(float));

    paris_mkl_get_trajectory(state, 0, traj_regimes, traj_h);

    /* Check trajectory validity */
    for (int t = 0; t < T; t++)
    {
        if (traj_regimes[t] < 0 || traj_regimes[t] >= K)
        {
            printf("FAILED (invalid regime in trajectory at t=%d)\n", t);
            free(regimes);
            free(h);
            free(weights);
            free(ancestors);
            free(traj_regimes);
            free(traj_h);
            paris_mkl_free(state);
            return 1;
        }

        if (!isfinite(traj_h[t]))
        {
            printf("FAILED (non-finite h in trajectory at t=%d)\n", t);
            free(regimes);
            free(h);
            free(weights);
            free(ancestors);
            free(traj_regimes);
            free(traj_h);
            paris_mkl_free(state);
            return 1;
        }
    }

    free(regimes);
    free(h);
    free(weights);
    free(ancestors);
    free(traj_regimes);
    free(traj_h);
    paris_mkl_free(state);
    printf("OK\n");
    return 0;
}

/*═══════════════════════════════════════════════════════════════════════════════
 * BENCHMARKS
 *═══════════════════════════════════════════════════════════════════════════════*/

static void benchmark_paris_mkl(void)
{
    printf("\n═══════════════════════════════════════════════════════════════════════\n");
    printf("PARIS-MKL BENCHMARK\n");
    printf("═══════════════════════════════════════════════════════════════════════\n\n");

#ifdef _OPENMP
    printf("OpenMP: %d threads\n\n", omp_get_max_threads());
#else
    printf("OpenMP: DISABLED\n\n");
#endif

    int configs[][3] = {
        /* T, K, N */
        {100, 4, 50},
        {100, 4, 100},
        {200, 4, 100},
        {300, 4, 100},
        {500, 4, 100},
        {200, 4, 128},
        {200, 8, 100},
        {1000, 4, 100},
    };
    int n_configs = sizeof(configs) / sizeof(configs[0]);

    printf("%-6s %-4s %-4s %-6s | %-12s %-12s %-12s\n",
           "T", "K", "N", "N_pad", "Load(ms)", "Smooth(ms)", "us/tick");
    printf("-------+----+----+-------|--------------|--------------|------------\n");

    for (int c = 0; c < n_configs; c++)
    {
        int T = configs[c][0];
        int K = configs[c][1];
        int N = configs[c][2];

        double trans[64], mu_vol[8], phi, sigma_h;
        int *regimes = malloc(T * N * sizeof(int));
        double *h = malloc(T * N * sizeof(double));
        double *weights = malloc(T * N * sizeof(double));
        int *ancestors = malloc(T * N * sizeof(int));

        generate_particle_data(T, K, N, trans, mu_vol, &phi, &sigma_h,
                               regimes, h, weights, ancestors);

        double load_total = 0, smooth_total = 0;

        for (int trial = 0; trial < TRIALS; trial++)
        {
            PARISMKLState *state = paris_mkl_alloc(N, T, K, trial + 1000);
            paris_mkl_set_model(state, trans, mu_vol, phi, sigma_h);

            /* Benchmark load */
            double t0 = get_time_ms();
            paris_mkl_load_particles(state, regimes, h, weights, ancestors, T);
            double t1 = get_time_ms();
            load_total += (t1 - t0);

            /* Benchmark smoothing */
            t0 = get_time_ms();
            paris_mkl_backward_smooth(state);
            t1 = get_time_ms();
            smooth_total += (t1 - t0);

            paris_mkl_free(state);
        }

        int N_padded = ((N + 15) & ~15);
        double avg_smooth = smooth_total / TRIALS;
        double us_per_tick = avg_smooth * 1000.0 / T;

        printf("%-6d %-4d %-4d %-6d | %-12.3f %-12.3f %-12.2f\n",
               T, K, N, N_padded,
               load_total / TRIALS,
               avg_smooth,
               us_per_tick);

        free(regimes);
        free(h);
        free(weights);
        free(ancestors);
    }
}

/*═══════════════════════════════════════════════════════════════════════════════
 * MAIN
 *═══════════════════════════════════════════════════════════════════════════════*/

int main(void)
{
    printf("╔═══════════════════════════════════════════════════════════════════════╗\n");
    printf("║                   PARIS-MKL TEST SUITE                                ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════════╝\n\n");

    srand(42);

    printf("═══ CORRECTNESS TESTS ═══\n\n");

    int failures = 0;
    failures += test_allocation();
    failures += test_model_set();
    failures += test_load_particles();
    failures += test_backward_smooth();
    failures += test_get_smoothed();
    failures += test_trajectory_extraction();

    if (failures > 0)
    {
        printf("\n✗ %d test(s) failed!\n", failures);
        return 1;
    }

    printf("\n✓ All correctness tests passed!\n");

    benchmark_paris_mkl();

    printf("\n✓ Benchmark complete!\n");
    return 0;
}