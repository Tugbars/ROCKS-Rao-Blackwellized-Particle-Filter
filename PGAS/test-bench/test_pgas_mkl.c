/**
 * @file test_pgas_mkl.c
 * @brief Benchmark and correctness test for MKL-optimized PGAS
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "pgas_mkl.h"

#define TRIALS 5

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

static void generate_test_data(int T, int K,
                               double *trans, double *mu_vol, double *sigma_vol,
                               double *phi, double *sigma_h,
                               double *observations,
                               int *true_regimes, double *true_h)
{
    /* Model parameters */
    *phi = 0.97;
    *sigma_h = 0.15;

    /* Sticky transition matrix */
    for (int i = 0; i < K; i++)
    {
        for (int j = 0; j < K; j++)
        {
            trans[i * K + j] = (i == j) ? 0.95 : 0.05 / (K - 1);
        }
        mu_vol[i] = -2.0 + i * 1.5; /* Spread regimes */
        sigma_vol[i] = 0.3;
    }

    /* Generate true trajectory */
    true_regimes[0] = 0;
    true_h[0] = mu_vol[0];

    for (int t = 1; t < T; t++)
    {
        /* Regime transition */
        double u = (double)rand() / RAND_MAX;
        double cumsum = 0;
        int prev_regime = true_regimes[t - 1];
        true_regimes[t] = prev_regime;
        for (int j = 0; j < K; j++)
        {
            cumsum += trans[prev_regime * K + j];
            if (u < cumsum)
            {
                true_regimes[t] = j;
                break;
            }
        }

        /* AR(1) h evolution */
        double mu_k = mu_vol[true_regimes[t]];
        double noise = *sigma_h * ((double)rand() / RAND_MAX * 2 - 1) * 1.732;
        true_h[t] = mu_k + (*phi) * (true_h[t - 1] - mu_k) + noise;
    }

    /* Generate observations: y_t ~ N(0, exp(h_t)) */
    for (int t = 0; t < T; t++)
    {
        double vol = exp(true_h[t] / 2);
        double z = ((double)rand() / RAND_MAX * 2 - 1) * 1.732;
        observations[t] = vol * z;
    }
}

/*═══════════════════════════════════════════════════════════════════════════════
 * CORRECTNESS TESTS
 *═══════════════════════════════════════════════════════════════════════════════*/

static int test_allocation(void)
{
    printf("  [1] Allocation test... ");

    PGASMKLState *state = pgas_mkl_alloc(100, 200, 4, 12345);
    if (!state)
    {
        printf("FAILED (null state)\n");
        return 1;
    }

    if (state->N != 100 || state->T != 200 || state->K != 4)
    {
        printf("FAILED (wrong dimensions)\n");
        pgas_mkl_free(state);
        return 1;
    }

    /* Check N_padded is multiple of 16 */
    if (state->N_padded % 16 != 0)
    {
        printf("FAILED (N_padded=%d not multiple of 16)\n", state->N_padded);
        pgas_mkl_free(state);
        return 1;
    }

    pgas_mkl_free(state);
    printf("OK\n");
    return 0;
}

static int test_model_set(void)
{
    printf("  [2] Model parameter test... ");

    PGASMKLState *state = pgas_mkl_alloc(50, 100, 4, 12345);

    double trans[16], mu_vol[4], sigma_vol[4];
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            trans[i * 4 + j] = (i == j) ? 0.9 : 0.1 / 3;
        }
        mu_vol[i] = -1.0 + i * 0.5;
        sigma_vol[i] = 0.25 + i * 0.05;
    }

    pgas_mkl_set_model(state, trans, mu_vol, sigma_vol, 0.95, 0.20);

    /* Verify log_trans computed correctly */
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            float expected = logf((float)trans[i * 4 + j]);
            float actual = state->model.log_trans[i * 4 + j];
            if (fabsf(expected - actual) > 1e-5f)
            {
                printf("FAILED (log_trans mismatch)\n");
                pgas_mkl_free(state);
                return 1;
            }
        }
    }

    pgas_mkl_free(state);
    printf("OK\n");
    return 0;
}

static int test_single_sweep(void)
{
    printf("  [3] Single CSMC sweep test... ");

    int T = 100, N = 50, K = 4;
    PGASMKLState *state = pgas_mkl_alloc(N, T, K, 12345);

    double trans[16], mu_vol[4], sigma_vol[4], phi, sigma_h;
    double *obs = malloc(T * sizeof(double));
    int *true_reg = malloc(T * sizeof(int));
    double *true_h = malloc(T * sizeof(double));

    generate_test_data(T, K, trans, mu_vol, sigma_vol, &phi, &sigma_h,
                       obs, true_reg, true_h);

    pgas_mkl_set_model(state, trans, mu_vol, sigma_vol, phi, sigma_h);
    pgas_mkl_set_reference(state, true_reg, true_h, T);
    pgas_mkl_load_observations(state, obs, T);

    /* Run one sweep */
    pgas_mkl_csmc_sweep(state);

    /* Check that particles exist and weights sum to ~1 */
    float weight_sum = 0;
    for (int n = 0; n < N; n++)
    {
        weight_sum += state->weights[(T - 1) * state->N_padded + n];
    }

    if (fabsf(weight_sum - 1.0f) > 0.01f)
    {
        printf("FAILED (weight sum = %.4f)\n", weight_sum);
        free(obs);
        free(true_reg);
        free(true_h);
        pgas_mkl_free(state);
        return 1;
    }

    /* Check ESS is reasonable */
    float ess = pgas_mkl_get_ess(state, T - 1);
    if (ess < 1 || ess > N)
    {
        printf("FAILED (ESS = %.2f out of range)\n", ess);
        free(obs);
        free(true_reg);
        free(true_h);
        pgas_mkl_free(state);
        return 1;
    }

    free(obs);
    free(true_reg);
    free(true_h);
    pgas_mkl_free(state);
    printf("OK (ESS=%.1f)\n", ess);
    return 0;
}

static int test_adaptive_run(void)
{
    printf("  [4] Adaptive PGAS run test... ");

    int T = 150, N = 64, K = 4;
    PGASMKLState *state = pgas_mkl_alloc(N, T, K, 54321);

    double trans[16], mu_vol[4], sigma_vol[4], phi, sigma_h;
    double *obs = malloc(T * sizeof(double));
    int *true_reg = malloc(T * sizeof(int));
    double *true_h = malloc(T * sizeof(double));

    generate_test_data(T, K, trans, mu_vol, sigma_vol, &phi, &sigma_h,
                       obs, true_reg, true_h);

    pgas_mkl_set_model(state, trans, mu_vol, sigma_vol, phi, sigma_h);
    pgas_mkl_set_reference(state, true_reg, true_h, T);
    pgas_mkl_load_observations(state, obs, T);

    int result = pgas_mkl_run_adaptive(state);

    if (result == 2)
    {
        printf("FAILED (did not converge)\n");
        free(obs);
        free(true_reg);
        free(true_h);
        pgas_mkl_free(state);
        return 1;
    }

    /* Check acceptance rate is reasonable */
    if (state->acceptance_rate < 0.1f || state->acceptance_rate > 1.0f)
    {
        printf("FAILED (acceptance=%.3f)\n", state->acceptance_rate);
        free(obs);
        free(true_reg);
        free(true_h);
        pgas_mkl_free(state);
        return 1;
    }

    free(obs);
    free(true_reg);
    free(true_h);
    pgas_mkl_free(state);
    printf("OK (sweeps=%d, accept=%.2f)\n", state->current_sweep, state->acceptance_rate);
    return 0;
}

static int test_reference_preservation(void)
{
    printf("  [5] Reference trajectory preservation test... ");

    int T = 100, N = 50, K = 4;
    PGASMKLState *state = pgas_mkl_alloc(N, T, K, 99999);

    double trans[16], mu_vol[4], sigma_vol[4], phi, sigma_h;
    double *obs = malloc(T * sizeof(double));
    int *true_reg = malloc(T * sizeof(int));
    double *true_h = malloc(T * sizeof(double));

    generate_test_data(T, K, trans, mu_vol, sigma_vol, &phi, &sigma_h,
                       obs, true_reg, true_h);

    pgas_mkl_set_model(state, trans, mu_vol, sigma_vol, phi, sigma_h);
    pgas_mkl_set_reference(state, true_reg, true_h, T);
    pgas_mkl_load_observations(state, obs, T);

    /* Run sweep */
    pgas_mkl_csmc_sweep(state);

    /* Check particle N-1 matches reference */
    int ref_idx = N - 1;
    int Np = state->N_padded;

    for (int t = 0; t < T; t++)
    {
        int regime = state->regimes[t * Np + ref_idx];
        float h = state->h[t * Np + ref_idx];

        if (regime != true_reg[t])
        {
            printf("FAILED (regime mismatch at t=%d)\n", t);
            free(obs);
            free(true_reg);
            free(true_h);
            pgas_mkl_free(state);
            return 1;
        }

        if (fabsf(h - (float)true_h[t]) > 1e-4f)
        {
            printf("FAILED (h mismatch at t=%d)\n", t);
            free(obs);
            free(true_reg);
            free(true_h);
            pgas_mkl_free(state);
            return 1;
        }
    }

    free(obs);
    free(true_reg);
    free(true_h);
    pgas_mkl_free(state);
    printf("OK\n");
    return 0;
}

/*═══════════════════════════════════════════════════════════════════════════════
 * BENCHMARKS
 *═══════════════════════════════════════════════════════════════════════════════*/

static void benchmark_pgas_mkl(void)
{
    printf("\n═══════════════════════════════════════════════════════════════════════\n");
    printf("PGAS-MKL BENCHMARK\n");
    printf("═══════════════════════════════════════════════════════════════════════\n\n");

    int configs[][3] = {
        /* T, K, N */
        {100, 4, 50},
        {100, 4, 100},
        {200, 4, 100},
        {300, 4, 100},
        {500, 4, 100},
        {200, 4, 128},
        {200, 8, 100},
    };
    int n_configs = sizeof(configs) / sizeof(configs[0]);

    printf("%-6s %-4s %-4s %-6s | %-12s %-12s %-10s | %-8s\n",
           "T", "K", "N", "N_pad", "Sweep(ms)", "Adaptive(ms)", "Accept", "ESS");
    printf("-------+----+----+-------|--------------+--------------+------------|--------\n");

    for (int c = 0; c < n_configs; c++)
    {
        int T = configs[c][0];
        int K = configs[c][1];
        int N = configs[c][2];

        double trans[64], mu_vol[8], sigma_vol[8], phi, sigma_h;
        double *obs = malloc(T * sizeof(double));
        int *true_reg = malloc(T * sizeof(int));
        double *true_h = malloc(T * sizeof(double));

        generate_test_data(T, K, trans, mu_vol, sigma_vol, &phi, &sigma_h,
                           obs, true_reg, true_h);

        double sweep_total = 0, adaptive_total = 0;
        float avg_accept = 0, avg_ess = 0;

        for (int trial = 0; trial < TRIALS; trial++)
        {
            PGASMKLState *state = pgas_mkl_alloc(N, T, K, trial + 1000);
            pgas_mkl_set_model(state, trans, mu_vol, sigma_vol, phi, sigma_h);
            pgas_mkl_set_reference(state, true_reg, true_h, T);
            pgas_mkl_load_observations(state, obs, T);

            /* Benchmark single sweep */
            double t0 = get_time_ms();
            pgas_mkl_csmc_sweep(state);
            double t1 = get_time_ms();
            sweep_total += (t1 - t0);

            /* Reset and benchmark adaptive */
            pgas_mkl_set_reference(state, true_reg, true_h, T);
            t0 = get_time_ms();
            pgas_mkl_run_adaptive(state);
            t1 = get_time_ms();
            adaptive_total += (t1 - t0);

            avg_accept += state->acceptance_rate;
            avg_ess += pgas_mkl_get_ess(state, T - 1);

            pgas_mkl_free(state);
        }

        int N_padded = ((N + 15) & ~15);
        printf("%-6d %-4d %-4d %-6d | %-12.2f %-12.2f %-10.3f | %-8.1f\n",
               T, K, N, N_padded,
               sweep_total / TRIALS,
               adaptive_total / TRIALS,
               avg_accept / TRIALS,
               avg_ess / TRIALS);

        free(obs);
        free(true_reg);
        free(true_h);
    }
}

/*═══════════════════════════════════════════════════════════════════════════════
 * MAIN
 *═══════════════════════════════════════════════════════════════════════════════*/

int main(void)
{
    printf("╔═══════════════════════════════════════════════════════════════════════╗\n");
    printf("║                    PGAS-MKL TEST SUITE                                ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════════╝\n\n");

    srand(42);

    printf("═══ CORRECTNESS TESTS ═══\n\n");

    int failures = 0;
    failures += test_allocation();
    failures += test_model_set();
    failures += test_single_sweep();
    failures += test_adaptive_run();
    failures += test_reference_preservation();

    if (failures > 0)
    {
        printf("\n✗ %d test(s) failed!\n", failures);
        return 1;
    }

    printf("\n✓ All correctness tests passed!\n");

    benchmark_pgas_mkl();

    printf("\n✓ Benchmark complete!\n");
    return 0;
}