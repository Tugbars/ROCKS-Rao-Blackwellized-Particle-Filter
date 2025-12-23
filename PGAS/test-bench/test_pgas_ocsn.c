/**
 * @file test_pgas_ocsn.c
 * @brief Test PGAS with OCSN 10-component emission and transition learning
 *
 * Compile:
 *   gcc -O3 -march=native -o test_pgas_ocsn test_pgas_ocsn.c pgas_mkl.c \
 *       -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lm -lpthread
 *
 * Or with OpenMP:
 *   gcc -O3 -march=native -fopenmp -o test_pgas_ocsn test_pgas_ocsn.c pgas_mkl.c \
 *       -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lm -lpthread
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "pgas_mkl.h"

/*═══════════════════════════════════════════════════════════════════════════════
 * TEST 1: OCSN Likelihood Sanity Check
 *═══════════════════════════════════════════════════════════════════════════════*/

/* Expose the internal OCSN function for testing */
#define OCSN_N_COMPONENTS 10

static const float OCSN_PROB[OCSN_N_COMPONENTS] = {
    0.00609f, 0.04775f, 0.13057f, 0.20674f, 0.22715f,
    0.18842f, 0.12047f, 0.05591f, 0.01575f, 0.00115f
};

static const float OCSN_MEAN[OCSN_N_COMPONENTS] = {
    1.92677f, 1.34744f, 0.73504f, 0.02266f, -0.85173f,
    -1.97278f, -3.46788f, -5.55246f, -8.68384f, -14.65000f
};

static const float OCSN_VAR[OCSN_N_COMPONENTS] = {
    0.11265f, 0.17788f, 0.26768f, 0.40611f, 0.62699f,
    0.98583f, 1.57469f, 2.54498f, 4.16591f, 7.33342f
};

static float test_ocsn_loglik(float y, float h)
{
    const float LOG_2PI = 1.8378770664f;
    float y_base = y - h;
    
    float max_log = -1e30f;
    float log_comps[OCSN_N_COMPONENTS];
    
    for (int j = 0; j < OCSN_N_COMPONENTS; j++) {
        float log_const = -0.5f * (LOG_2PI + logf(OCSN_VAR[j])) + logf(OCSN_PROB[j]);
        float inv_2v = 0.5f / OCSN_VAR[j];
        float diff = y_base - OCSN_MEAN[j];
        log_comps[j] = log_const - inv_2v * diff * diff;
        if (log_comps[j] > max_log) max_log = log_comps[j];
    }
    
    float sum = 0.0f;
    for (int j = 0; j < OCSN_N_COMPONENTS; j++) {
        sum += expf(log_comps[j] - max_log);
    }
    
    return max_log + logf(sum);
}

void test_ocsn_likelihood(void)
{
    printf("\n═══════════════════════════════════════════════════════════\n");
    printf("TEST 1: OCSN 10-Component Likelihood\n");
    printf("═══════════════════════════════════════════════════════════\n\n");
    
    /* Test cases covering different regimes */
    struct { float y; float h; const char *desc; } tests[] = {
        {-4.0f, -3.0f, "Normal vol (h≈-3)"},
        {-2.0f, -2.0f, "Elevated vol (h≈-2)"},
        { 0.0f, -1.0f, "High vol (h≈-1)"},
        { 2.0f,  0.0f, "Crisis (h≈0)"},
        {-8.0f, -5.0f, "Very low vol (h≈-5)"},
        {-6.0f, -3.0f, "Large negative residual"},
        { 0.0f, -3.0f, "Large positive residual"},
    };
    int n_tests = sizeof(tests) / sizeof(tests[0]);
    
    printf("%-30s %10s %10s %12s\n", "Description", "y", "h", "log P(y|h)");
    printf("%-30s %10s %10s %12s\n", "------------------------------", "----------", "----------", "------------");
    
    for (int i = 0; i < n_tests; i++) {
        float ll = test_ocsn_loglik(tests[i].y, tests[i].h);
        printf("%-30s %10.2f %10.2f %12.4f\n", 
               tests[i].desc, tests[i].y, tests[i].h, ll);
    }
    
    /* Verify likelihood is highest when residual matches mixture center */
    printf("\nLikelihood profile (h=-3, varying y):\n");
    float h_fixed = -3.0f;
    for (float y = -8.0f; y <= 2.0f; y += 1.0f) {
        float ll = test_ocsn_loglik(y, h_fixed);
        int bar_len = (int)((ll + 6.0f) * 5);
        if (bar_len < 0) bar_len = 0;
        if (bar_len > 40) bar_len = 40;
        printf("  y=%5.1f: ll=%7.3f |", y, ll);
        for (int j = 0; j < bar_len; j++) printf("█");
        printf("\n");
    }
    
    printf("\n✓ OCSN likelihood test complete\n");
}

/*═══════════════════════════════════════════════════════════════════════════════
 * TEST 2: PGAS with OCSN Emission
 *═══════════════════════════════════════════════════════════════════════════════*/

void test_pgas_csmc(void)
{
    printf("\n═══════════════════════════════════════════════════════════\n");
    printf("TEST 2: PGAS CSMC Sweep with OCSN Emission\n");
    printf("═══════════════════════════════════════════════════════════\n\n");
    
    const int N = 128;
    const int T = 200;
    const int K = 4;
    const uint32_t seed = 42;
    
    /* Allocate PGAS state */
    PGASMKLState *state = pgas_mkl_alloc(N, T, K, seed);
    if (!state) {
        printf("ERROR: Failed to allocate PGAS state\n");
        return;
    }
    
    /* Set up model parameters (typical SV model) */
    double trans[16] = {
        0.92, 0.05, 0.02, 0.01,
        0.05, 0.90, 0.04, 0.01,
        0.02, 0.05, 0.88, 0.05,
        0.01, 0.02, 0.07, 0.90
    };
    double mu_vol[4] = {-5.0, -3.0, -1.5, 0.0};
    double sigma_vol[4] = {0.3, 0.4, 0.5, 0.6};
    double phi = 0.97;
    double sigma_h = 0.15;
    
    pgas_mkl_set_model(state, trans, mu_vol, sigma_vol, phi, sigma_h);
    
    /* Generate synthetic observations (simple random for now) */
    double *obs = (double *)malloc(T * sizeof(double));
    srand(seed);
    for (int t = 0; t < T; t++) {
        /* y = h + noise, h varies slowly */
        double h_true = -3.0 + 2.0 * sin(2.0 * 3.14159 * t / T);
        obs[t] = h_true + 0.5 * ((double)rand() / RAND_MAX - 0.5);
    }
    pgas_mkl_load_observations(state, obs, T);
    
    /* Initialize reference trajectory */
    int *ref_regimes = (int *)malloc(T * sizeof(int));
    double *ref_h = (double *)malloc(T * sizeof(double));
    for (int t = 0; t < T; t++) {
        ref_regimes[t] = 1;  /* Start in regime 1 */
        ref_h[t] = -3.0;
    }
    pgas_mkl_set_reference(state, ref_regimes, ref_h, T);
    
    /* Run CSMC sweeps */
    printf("Running 10 CSMC sweeps...\n\n");
    for (int sweep = 0; sweep < 10; sweep++) {
        float accept = pgas_mkl_csmc_sweep(state);
        float ess = pgas_mkl_get_ess(state, T - 1);
        printf("  Sweep %2d: acceptance=%.3f, ESS=%.1f\n", sweep + 1, accept, ess);
    }
    
    pgas_mkl_print_diagnostics(state);
    
    /* Cleanup */
    free(obs);
    free(ref_regimes);
    free(ref_h);
    pgas_mkl_free(state);
    
    printf("\n✓ PGAS CSMC test complete\n");
}

/*═══════════════════════════════════════════════════════════════════════════════
 * TEST 3: Transition Matrix Learning
 *═══════════════════════════════════════════════════════════════════════════════*/

void test_transition_learning(void)
{
    printf("\n═══════════════════════════════════════════════════════════\n");
    printf("TEST 3: Transition Matrix Learning via Gibbs\n");
    printf("═══════════════════════════════════════════════════════════\n\n");
    
    const int N = 128;
    const int T = 500;
    const int K = 4;
    const uint32_t seed = 123;
    
    /* Allocate PGAS state */
    PGASMKLState *state = pgas_mkl_alloc(N, T, K, seed);
    if (!state) {
        printf("ERROR: Failed to allocate PGAS state\n");
        return;
    }
    
    /* True transition matrix (ground truth) */
    double true_trans[16] = {
        0.90, 0.07, 0.02, 0.01,
        0.05, 0.85, 0.08, 0.02,
        0.02, 0.08, 0.82, 0.08,
        0.01, 0.04, 0.10, 0.85
    };
    double mu_vol[4] = {-5.0, -3.0, -1.5, 0.0};
    double sigma_vol[4] = {0.3, 0.4, 0.5, 0.6};
    double phi = 0.97;
    double sigma_h = 0.15;
    
    /* Start with uniform transitions (to be learned) */
    double init_trans[16];
    for (int i = 0; i < 16; i++) init_trans[i] = 0.25;
    pgas_mkl_set_model(state, init_trans, mu_vol, sigma_vol, phi, sigma_h);
    
    /* Set priors */
    pgas_mkl_set_transition_prior(state, 1.0f, 10.0f);
    
    /* Generate synthetic data from true model */
    double *obs = (double *)malloc(T * sizeof(double));
    int *true_regimes = (int *)malloc(T * sizeof(int));
    double *true_h = (double *)malloc(T * sizeof(double));
    
    srand(seed);
    true_regimes[0] = 0;
    true_h[0] = mu_vol[0];
    obs[0] = true_h[0] + 0.5 * ((double)rand() / RAND_MAX - 0.5);
    
    for (int t = 1; t < T; t++) {
        /* Sample regime transition */
        double u = (double)rand() / RAND_MAX;
        double cumsum = 0;
        int prev_r = true_regimes[t-1];
        true_regimes[t] = K - 1;
        for (int j = 0; j < K; j++) {
            cumsum += true_trans[prev_r * K + j];
            if (u < cumsum) {
                true_regimes[t] = j;
                break;
            }
        }
        
        /* Sample h */
        int r = true_regimes[t];
        double mean = mu_vol[r] + phi * (true_h[t-1] - mu_vol[r]);
        true_h[t] = mean + sigma_h * ((double)rand() / RAND_MAX - 0.5) * 2;
        
        /* Observation */
        obs[t] = true_h[t] + 0.5 * ((double)rand() / RAND_MAX - 0.5);
    }
    
    pgas_mkl_load_observations(state, obs, T);
    pgas_mkl_set_reference(state, true_regimes, true_h, T);
    
    /* Run Gibbs sweeps to learn transitions */
    printf("Running 100 Gibbs sweeps (burn-in)...\n");
    for (int i = 0; i < 100; i++) {
        pgas_mkl_gibbs_sweep(state);
    }
    
    /* Collect samples */
    printf("Collecting 200 samples...\n\n");
    float trans_sum[16] = {0};
    for (int i = 0; i < 200; i++) {
        pgas_mkl_gibbs_sweep(state);
        for (int j = 0; j < 16; j++) {
            trans_sum[j] += state->model.trans[j];
        }
    }
    
    /* Posterior mean */
    printf("Learned vs True Transition Matrix:\n\n");
    printf("             Learned                        True\n");
    printf("  ─────────────────────────    ─────────────────────────\n");
    for (int i = 0; i < K; i++) {
        printf("  ");
        for (int j = 0; j < K; j++) {
            printf("%5.3f ", trans_sum[i * K + j] / 200.0f);
        }
        printf("   ");
        for (int j = 0; j < K; j++) {
            printf("%5.3f ", true_trans[i * K + j]);
        }
        printf("\n");
    }
    
    /* Compute error */
    float max_err = 0;
    for (int i = 0; i < 16; i++) {
        float err = fabsf(trans_sum[i] / 200.0f - (float)true_trans[i]);
        if (err > max_err) max_err = err;
    }
    printf("\nMax absolute error: %.4f\n", max_err);
    
    /* Cleanup */
    free(obs);
    free(true_regimes);
    free(true_h);
    pgas_mkl_free(state);
    
    printf("\n✓ Transition learning test complete\n");
}

/*═══════════════════════════════════════════════════════════════════════════════
 * TEST 4: PARIS Smoothing
 *═══════════════════════════════════════════════════════════════════════════════*/

void test_paris_smoothing(void)
{
    printf("\n═══════════════════════════════════════════════════════════\n");
    printf("TEST 4: PARIS Backward Smoothing\n");
    printf("═══════════════════════════════════════════════════════════\n\n");
    
    const int N = 64;
    const int T = 100;
    const int K = 4;
    const uint32_t seed = 456;
    
    PGASMKLState *state = pgas_mkl_alloc(N, T, K, seed);
    if (!state) {
        printf("ERROR: Failed to allocate PGAS state\n");
        return;
    }
    
    double trans[16] = {
        0.92, 0.05, 0.02, 0.01,
        0.05, 0.90, 0.04, 0.01,
        0.02, 0.05, 0.88, 0.05,
        0.01, 0.02, 0.07, 0.90
    };
    double mu_vol[4] = {-5.0, -3.0, -1.5, 0.0};
    double sigma_vol[4] = {0.3, 0.4, 0.5, 0.6};
    
    pgas_mkl_set_model(state, trans, mu_vol, sigma_vol, 0.97, 0.15);
    
    /* Generate simple observations */
    double *obs = (double *)malloc(T * sizeof(double));
    int *ref_r = (int *)malloc(T * sizeof(int));
    double *ref_h = (double *)malloc(T * sizeof(double));
    
    srand(seed);
    for (int t = 0; t < T; t++) {
        ref_r[t] = t < 50 ? 0 : 2;  /* Regime switch at t=50 */
        ref_h[t] = mu_vol[ref_r[t]];
        obs[t] = ref_h[t] + 0.3 * ((double)rand() / RAND_MAX - 0.5);
    }
    
    pgas_mkl_load_observations(state, obs, T);
    pgas_mkl_set_reference(state, ref_r, ref_h, T);
    
    /* Run CSMC */
    printf("Running 5 CSMC sweeps...\n");
    for (int i = 0; i < 5; i++) {
        pgas_mkl_csmc_sweep(state);
    }
    
    /* Run PARIS smoothing */
    printf("Running PARIS backward smoothing...\n");
    clock_t start = clock();
    pgas_paris_backward_smooth(state);
    clock_t end = clock();
    double elapsed_ms = 1000.0 * (end - start) / CLOCKS_PER_SEC;
    
    printf("PARIS completed in %.2f ms\n\n", elapsed_ms);
    
    /* Check smoothed regimes */
    int *smoothed_regimes = (int *)malloc(N * sizeof(int));
    float *smoothed_h = (float *)malloc(N * sizeof(float));
    
    printf("Smoothed regime distribution at key times:\n");
    int check_times[] = {25, 49, 50, 51, 75};
    for (int i = 0; i < 5; i++) {
        int t = check_times[i];
        pgas_paris_get_smoothed(state, t, smoothed_regimes, smoothed_h);
        
        int counts[4] = {0};
        for (int n = 0; n < N; n++) {
            counts[smoothed_regimes[n]]++;
        }
        
        printf("  t=%2d (true=%d): ", t, ref_r[t]);
        for (int k = 0; k < K; k++) {
            printf("R%d=%3d ", k, counts[k]);
        }
        printf("\n");
    }
    
    free(obs);
    free(ref_r);
    free(ref_h);
    free(smoothed_regimes);
    free(smoothed_h);
    pgas_mkl_free(state);
    
    printf("\n✓ PARIS smoothing test complete\n");
}

/*═══════════════════════════════════════════════════════════════════════════════
 * MAIN
 *═══════════════════════════════════════════════════════════════════════════════*/

int main(void)
{
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════╗\n");
    printf("║     PGAS-MKL Test Suite (OCSN 10-component Edition)       ║\n");
    printf("╚═══════════════════════════════════════════════════════════╝\n");
    
    test_ocsn_likelihood();
    test_pgas_csmc();
    test_transition_learning();
    test_paris_smoothing();
    
    printf("\n═══════════════════════════════════════════════════════════\n");
    printf("ALL TESTS COMPLETE\n");
    printf("═══════════════════════════════════════════════════════════\n\n");
    
    return 0;
}
