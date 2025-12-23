/*═══════════════════════════════════════════════════════════════════════════════
 * PGAS ADAPTIVE KAPPA VALIDATION
 *
 * Tests whether adaptive κ can self-correct when prior mismatches data.
 *
 * Design:
 *   1. Start with WRONG κ (e.g., κ=100 for true diag=0.90)
 *   2. Enable adaptive kappa
 *   3. Run Gibbs sweeps, track κ evolution
 *   4. Verify κ converges toward correct value
 *   5. Verify learned transition matrix improves
 *
 * Success Criteria:
 *   - κ moves in correct direction
 *   - Chatter ratio converges toward 1.0
 *   - Final matrix error < fixed-κ error
 *═══════════════════════════════════════════════════════════════════════════════*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#else
#include <windows.h>
static double omp_get_wtime(void)
{
    LARGE_INTEGER freq, cnt;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&cnt);
    return (double)cnt.QuadPart / (double)freq.QuadPart;
}
#endif

#include "pgas_mkl.h"
#include <mkl.h>
#include <mkl_vsl.h>

/*═══════════════════════════════════════════════════════════════════════════════
 * SYNTHETIC DATA (same as validation_a)
 *═══════════════════════════════════════════════════════════════════════════════*/

static const double OCSN_PROB[10] = {
    0.00609, 0.04775, 0.13057, 0.20674, 0.22715,
    0.18842, 0.12047, 0.05591, 0.01575, 0.00115};

static const double OCSN_MEAN[10] = {
    1.92677, 1.34744, 0.73504, 0.02266, -0.85173,
    -1.97278, -3.46788, -5.55246, -8.68384, -14.65000};

static const double OCSN_VAR[10] = {
    0.11265, 0.17788, 0.26768, 0.40611, 0.62699,
    0.98583, 1.57469, 2.54498, 4.16591, 7.33342};

static int sample_categorical(const double *probs, int K, VSLStreamStatePtr stream)
{
    double u;
    vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, 1, &u, 0.0, 1.0);
    double cumsum = 0.0;
    for (int k = 0; k < K; k++)
    {
        cumsum += probs[k];
        if (u < cumsum)
            return k;
    }
    return K - 1;
}

static double sample_ocsn(VSLStreamStatePtr stream)
{
    int comp = sample_categorical(OCSN_PROB, 10, stream);
    double z;
    vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, stream, 1, &z,
                  OCSN_MEAN[comp], sqrt(OCSN_VAR[comp]));
    return z;
}

typedef struct
{
    int K, T;
    double trans[16], mu_vol[4], sigma_vol[4], phi, sigma_h;
    int *true_regimes;
    float *true_h, *observations;
} SyntheticData;

SyntheticData *generate_synthetic_data(int K, int T, const double *trans,
                                       const double *mu_vol, const double *sigma_vol,
                                       double phi, double sigma_h, uint32_t seed)
{
    SyntheticData *data = (SyntheticData *)calloc(1, sizeof(SyntheticData));
    data->K = K;
    data->T = T;
    data->phi = phi;
    data->sigma_h = sigma_h;
    memcpy(data->trans, trans, K * K * sizeof(double));
    memcpy(data->mu_vol, mu_vol, K * sizeof(double));
    memcpy(data->sigma_vol, sigma_vol, K * sizeof(double));

    data->true_regimes = (int *)malloc(T * sizeof(int));
    data->true_h = (float *)malloc(T * sizeof(float));
    data->observations = (float *)malloc(T * sizeof(float));

    VSLStreamStatePtr stream;
    vslNewStream(&stream, VSL_BRNG_MT19937, seed);

    data->true_regimes[0] = seed % K;
    double h_mean = mu_vol[data->true_regimes[0]];
    double h_var = (sigma_h * sigma_h) / (1.0 - phi * phi);
    double z;
    vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, stream, 1, &z, 0.0, sqrt(h_var));
    data->true_h[0] = (float)(h_mean + z);
    data->observations[0] = data->true_h[0] + (float)sample_ocsn(stream);

    for (int t = 1; t < T; t++)
    {
        int prev = data->true_regimes[t - 1];
        data->true_regimes[t] = sample_categorical(&trans[prev * K], K, stream);
        int curr = data->true_regimes[t];
        double mean_h = mu_vol[curr] * (1.0 - phi) + phi * data->true_h[t - 1];
        vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, stream, 1, &z, 0.0, sigma_h);
        data->true_h[t] = (float)(mean_h + z);
        data->observations[t] = data->true_h[t] + (float)sample_ocsn(stream);
    }
    vslDeleteStream(&stream);
    return data;
}

void free_synthetic_data(SyntheticData *data)
{
    if (data)
    {
        free(data->true_regimes);
        free(data->true_h);
        free(data->observations);
        free(data);
    }
}

/*═══════════════════════════════════════════════════════════════════════════════
 * UTILITIES
 *═══════════════════════════════════════════════════════════════════════════════*/

double frobenius_error(const double *A, const float *B, int K)
{
    double sum = 0.0;
    for (int i = 0; i < K * K; i++)
    {
        double diff = A[i] - (double)B[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

void print_matrix(const char *name, const float *mat, int K)
{
    printf("%s:\n", name);
    for (int i = 0; i < K; i++)
    {
        printf("  [");
        for (int j = 0; j < K; j++)
            printf(" %6.4f", mat[i * K + j]);
        printf(" ]\n");
    }
}

/*═══════════════════════════════════════════════════════════════════════════════
 * ADAPTIVE KAPPA TEST
 *═══════════════════════════════════════════════════════════════════════════════*/

typedef struct
{
    const char *name;
    float initial_kappa;  /* Starting κ (deliberately wrong) */
    float true_diag;      /* True diagonal persistence */
    float expected_kappa; /* Where κ should converge */
    int expect_increase;  /* 1 if κ should go up, 0 if down */
} AdaptiveScenario;

void run_adaptive_test(const AdaptiveScenario *scenario, uint32_t seed)
{
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════════════╗\n");
    printf("║  ADAPTIVE TEST: %-53s ║\n", scenario->name);
    printf("╠═══════════════════════════════════════════════════════════════════════╣\n");
    printf("║  Initial κ=%.0f, True diag=%.2f, Expect κ to %s        ║\n",
           scenario->initial_kappa, scenario->true_diag,
           scenario->expect_increase ? "INCREASE" : "DECREASE");
    printf("╚═══════════════════════════════════════════════════════════════════════╝\n\n");

    const int K = 4;
    const int T = 2000;
    const int N = 128;
    const int N_burnin = 100;
    const int N_adaptive = 500;

    /* TRUE parameters */
    double mu_vol[4] = {-2.25, -0.75, 0.75, 2.25};
    double sigma_vol[4] = {0.2, 0.2, 0.2, 0.2};
    double phi = 0.97;
    double sigma_h = 0.15;

    double diag = scenario->true_diag;
    double off_diag = (1.0 - diag) / (K - 1);
    double trans_true[16];
    for (int i = 0; i < K; i++)
        for (int j = 0; j < K; j++)
            trans_true[i * K + j] = (i == j) ? diag : off_diag;

    /* Generate data */
    printf("Generating synthetic data (diag=%.2f)...\n", diag);
    SyntheticData *data = generate_synthetic_data(K, T, trans_true, mu_vol,
                                                  sigma_vol, phi, sigma_h, seed);

    /* Initialize PGAS with WRONG κ */
    printf("Initializing PGAS with κ=%.0f...\n", scenario->initial_kappa);
    if (scenario->expect_increase)
    {
        printf("  (Prior NOT STICKY ENOUGH for true diag=%.2f → expect κ to INCREASE)\n",
               scenario->true_diag);
    }
    else if (fabsf(scenario->true_diag - 0.95f) < 0.02f)
    {
        printf("  (Prior MATCHES true diag=%.2f → expect κ to stay STABLE)\n",
               scenario->true_diag);
    }
    else
    {
        printf("  (Prior TOO STICKY for true diag=%.2f → expect κ to DECREASE)\n",
               scenario->true_diag);
    }

    PGASMKLState *pgas = pgas_mkl_alloc(N, T, K, seed + 1000);

    double init_trans[16];
    for (int i = 0; i < K * K; i++)
        init_trans[i] = 1.0 / K;

    pgas_mkl_set_model(pgas, init_trans, mu_vol, sigma_vol, phi, sigma_h);
    pgas_mkl_set_transition_prior(pgas, 1.0f, scenario->initial_kappa);

    /* Enable adaptive kappa */
    pgas_mkl_enable_adaptive_kappa(pgas, 1);
    pgas_mkl_configure_adaptive_kappa(pgas,
                                      30.0f,  /* kappa_min */
                                      300.0f, /* kappa_max */
                                      0.3f,   /* up_rate */
                                      0.1f);  /* down_rate */

    /* Load observations */
    double *obs_double = (double *)malloc(T * sizeof(double));
    for (int i = 0; i < T; i++)
        obs_double[i] = (double)data->observations[i];
    pgas_mkl_load_observations(pgas, obs_double, T);
    free(obs_double);

    /* Initialize reference */
    pgas_mkl_csmc_sweep(pgas);

    /* Capture κ BEFORE burn-in */
    float kappa_start = pgas_mkl_get_sticky_kappa(pgas);

    /* Burn-in (WITH adaptation enabled) */
    printf("Burn-in (%d sweeps)...\n", N_burnin);
    for (int i = 0; i < N_burnin; i++)
        pgas_mkl_gibbs_sweep(pgas);

    float kappa_after_burnin = pgas_mkl_get_sticky_kappa(pgas);
    printf("  κ after burn-in: %.1f (started at %.1f)\n", kappa_after_burnin, kappa_start);

    /* Track κ evolution */
    printf("\nAdaptive phase (%d sweeps) - RLS smoothed chatter:\n", N_adaptive);
    printf("─────────────────────────────────────────────────────────\n");
    printf("  Sweep   │    κ     │  Chatter  │  Direction\n");
    printf("─────────────────────────────────────────────────────────\n");

    float kappa_history[11]; /* Track at 0%, 10%, 20%, ..., 100% */
    float chatter_history[11];
    int history_idx = 0;

    /* Accumulate transition matrices */
    float trans_sum[16] = {0};
    float trans_current[16];

    for (int i = 0; i < N_adaptive; i++)
    {
        pgas_mkl_gibbs_sweep(pgas);

        float kappa = pgas_mkl_get_sticky_kappa(pgas);
        float chatter = pgas_mkl_get_chatter_ratio(pgas);

        /* Get current transition matrix */
        pgas_mkl_get_transitions(pgas, trans_current, K);
        for (int j = 0; j < K * K; j++)
            trans_sum[j] += trans_current[j];

        /* Log every 10% */
        if (i % (N_adaptive / 10) == 0)
        {
            const char *dir = "";
            if (i > 0)
            {
                float prev_kappa = kappa_history[history_idx - 1];
                if (kappa > prev_kappa + 1.0f)
                    dir = "↑";
                else if (kappa < prev_kappa - 1.0f)
                    dir = "↓";
                else
                    dir = "─";
            }
            /* Note: chatter shown is now EMA-smoothed */
            printf("  %5d   │  %6.1f  │   %5.2fx   │  %s\n", i, kappa, chatter, dir);

            kappa_history[history_idx] = kappa;
            chatter_history[history_idx] = chatter;
            history_idx++;
        }
    }

    /* Final values */
    float kappa_final = pgas_mkl_get_sticky_kappa(pgas);
    float chatter_final = pgas_mkl_get_chatter_ratio(pgas);
    kappa_history[history_idx] = kappa_final;
    chatter_history[history_idx] = chatter_final;

    printf("  %5d   │  %6.1f  │   %5.2fx   │  FINAL\n", N_adaptive, kappa_final, chatter_final);
    printf("─────────────────────────────────────────────────────────\n");

    /* Compute learned matrix */
    float trans_learned[16];
    for (int j = 0; j < K * K; j++)
        trans_learned[j] = trans_sum[j] / N_adaptive;

    /* ═══════════════════════════════════════════════════════════════════
     * RESULTS
     * ═══════════════════════════════════════════════════════════════════*/
    printf("\n");
    printf("═══════════════════════════════════════════════════════════════════════\n");
    printf("RESULTS\n");
    printf("═══════════════════════════════════════════════════════════════════════\n\n");

    printf("κ EVOLUTION:\n");
    printf("  Initial:     %.1f\n", kappa_start);
    printf("  After burn:  %.1f (change: %+.1f)\n", kappa_after_burnin, kappa_after_burnin - kappa_start);
    printf("  Final:       %.1f\n", kappa_final);
    printf("  Total change: %+.1f (%.0f%%)\n",
           kappa_final - kappa_start,
           100.0f * (kappa_final - kappa_start) / kappa_start);

    int direction_correct = 0;
    float kappa_change = kappa_final - kappa_start;

    if (scenario->expect_increase)
    {
        /* Expected to increase */
        direction_correct = (kappa_change > 5.0f);
    }
    else if (fabsf(scenario->true_diag - 0.95f) < 0.02f)
    {
        /* True diag ≈ 0.95 means κ=100 is correct → should stay stable */
        direction_correct = (fabsf(kappa_change) < 30.0f);
    }
    else
    {
        /* Expected to decrease */
        direction_correct = (kappa_change < -5.0f);
    }

    printf("  Direction: %s\n\n",
           direction_correct ? "✓ CORRECT" : "✗ WRONG");

    printf("CHATTER RATIO:\n");
    printf("  Start:  %.2fx\n", chatter_history[0]);
    printf("  Final:  %.2fx\n", chatter_final);
    printf("  Target: 1.0x\n");

    int chatter_improved = fabsf(chatter_final - 1.0f) < fabsf(chatter_history[0] - 1.0f);
    printf("  Improved: %s\n\n", chatter_improved ? "✓ YES" : "✗ NO");

    /* Matrix comparison */
    double frob_err = frobenius_error(trans_true, trans_learned, K);

    printf("TRANSITION MATRIX:\n");
    printf("  TRUE diagonal:    %.2f\n", scenario->true_diag);
    printf("  Learned diagonal: %.4f, %.4f, %.4f, %.4f\n",
           trans_learned[0], trans_learned[5], trans_learned[10], trans_learned[15]);
    printf("  Frobenius error:  %.4f\n\n", frob_err);

    print_matrix("LEARNED", trans_learned, K);

    /* Overall verdict */
    printf("\n");
    printf("═══════════════════════════════════════════════════════════════════════\n");

    /* Success criteria:
     * 1. κ moved in correct direction (primary)
     * 2. Frobenius error < 0.10 (validation threshold)
     * Chatter improvement is nice-to-have but not required if at minimum
     */
    int pass = direction_correct && (frob_err < 0.10);

    if (pass)
    {
        printf("VERDICT: ✓ PASS - Adaptive κ self-corrected (error=%.4f)\n", frob_err);
    }
    else if (direction_correct)
    {
        printf("VERDICT: ⚠ PARTIAL - Direction correct but error=%.4f > 0.10\n", frob_err);
    }
    else
    {
        printf("VERDICT: ✗ FAIL - Adaptive κ did not converge\n");
    }
    printf("═══════════════════════════════════════════════════════════════════════\n");

    /* Cleanup */
    pgas_mkl_free(pgas);
    free_synthetic_data(data);
}

/*═══════════════════════════════════════════════════════════════════════════════
 * COMPARISON TEST: Fixed vs Adaptive
 *═══════════════════════════════════════════════════════════════════════════════*/

void run_comparison_test(uint32_t seed)
{
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════════════╗\n");
    printf("║           COMPARISON: Fixed κ=100 vs Adaptive κ                      ║\n");
    printf("║           True diagonal = 0.90 (κ=100 is TOO STRONG)                 ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════════╝\n\n");

    const int K = 4;
    const int T = 2000;
    const int N = 128;
    const int N_sweeps = 500;

    /* TRUE parameters with diag=0.90 */
    double mu_vol[4] = {-2.25, -0.75, 0.75, 2.25};
    double sigma_vol[4] = {0.2, 0.2, 0.2, 0.2};
    double phi = 0.97;
    double sigma_h = 0.15;
    double diag = 0.90;
    double off_diag = (1.0 - diag) / (K - 1);

    double trans_true[16];
    for (int i = 0; i < K; i++)
        for (int j = 0; j < K; j++)
            trans_true[i * K + j] = (i == j) ? diag : off_diag;

    /* Generate data ONCE */
    printf("Generating synthetic data (diag=0.90)...\n");
    SyntheticData *data = generate_synthetic_data(K, T, trans_true, mu_vol,
                                                  sigma_vol, phi, sigma_h, seed);

    double *obs_double = (double *)malloc(T * sizeof(double));
    for (int i = 0; i < T; i++)
        obs_double[i] = (double)data->observations[i];

    double init_trans[16];
    for (int i = 0; i < K * K; i++)
        init_trans[i] = 1.0 / K;

    /* ═══════════════════════════════════════════════════════════════════
     * RUN 1: Fixed κ=100
     * ═══════════════════════════════════════════════════════════════════*/
    printf("\n[1] Running with FIXED κ=100...\n");

    PGASMKLState *pgas_fixed = pgas_mkl_alloc(N, T, K, seed + 1000);
    pgas_mkl_set_model(pgas_fixed, init_trans, mu_vol, sigma_vol, phi, sigma_h);
    pgas_mkl_set_transition_prior(pgas_fixed, 1.0f, 100.0f);
    pgas_mkl_load_observations(pgas_fixed, obs_double, T);
    pgas_mkl_csmc_sweep(pgas_fixed);

    float trans_sum_fixed[16] = {0};
    float trans_current[16];

    for (int i = 0; i < N_sweeps; i++)
    {
        pgas_mkl_gibbs_sweep(pgas_fixed);
        pgas_mkl_get_transitions(pgas_fixed, trans_current, K);
        for (int j = 0; j < K * K; j++)
            trans_sum_fixed[j] += trans_current[j];
    }

    float trans_fixed[16];
    for (int j = 0; j < K * K; j++)
        trans_fixed[j] = trans_sum_fixed[j] / N_sweeps;

    float chatter_fixed = pgas_mkl_get_chatter_ratio(pgas_fixed);
    double frob_fixed = frobenius_error(trans_true, trans_fixed, K);

    printf("    Final chatter: %.2fx\n", chatter_fixed);
    printf("    Frobenius error: %.4f\n", frob_fixed);

    pgas_mkl_free(pgas_fixed);

    /* ═══════════════════════════════════════════════════════════════════
     * RUN 2: Adaptive κ starting at 100
     * ═══════════════════════════════════════════════════════════════════*/
    printf("\n[2] Running with ADAPTIVE κ (start=100)...\n");

    PGASMKLState *pgas_adapt = pgas_mkl_alloc(N, T, K, seed + 1000);
    pgas_mkl_set_model(pgas_adapt, init_trans, mu_vol, sigma_vol, phi, sigma_h);
    pgas_mkl_set_transition_prior(pgas_adapt, 1.0f, 100.0f);
    pgas_mkl_enable_adaptive_kappa(pgas_adapt, 1);
    pgas_mkl_configure_adaptive_kappa(pgas_adapt, 30.0f, 300.0f, 0.3f, 0.1f);
    pgas_mkl_load_observations(pgas_adapt, obs_double, T);
    pgas_mkl_csmc_sweep(pgas_adapt);

    float trans_sum_adapt[16] = {0};

    for (int i = 0; i < N_sweeps; i++)
    {
        pgas_mkl_gibbs_sweep(pgas_adapt);
        pgas_mkl_get_transitions(pgas_adapt, trans_current, K);
        for (int j = 0; j < K * K; j++)
            trans_sum_adapt[j] += trans_current[j];

        /* Log κ evolution */
        if (i % 100 == 0)
        {
            printf("    Sweep %d: κ=%.1f, chatter=%.2fx\n",
                   i, pgas_mkl_get_sticky_kappa(pgas_adapt),
                   pgas_mkl_get_chatter_ratio(pgas_adapt));
        }
    }

    float trans_adapt[16];
    for (int j = 0; j < K * K; j++)
        trans_adapt[j] = trans_sum_adapt[j] / N_sweeps;

    float kappa_final = pgas_mkl_get_sticky_kappa(pgas_adapt);
    float chatter_adapt = pgas_mkl_get_chatter_ratio(pgas_adapt);
    double frob_adapt = frobenius_error(trans_true, trans_adapt, K);

    printf("    Final κ: %.1f\n", kappa_final);
    printf("    Final chatter: %.2fx\n", chatter_adapt);
    printf("    Frobenius error: %.4f\n", frob_adapt);

    pgas_mkl_free(pgas_adapt);
    free(obs_double);
    free_synthetic_data(data);

    /* ═══════════════════════════════════════════════════════════════════
     * COMPARISON
     * ═══════════════════════════════════════════════════════════════════*/
    printf("\n");
    printf("═══════════════════════════════════════════════════════════════════════\n");
    printf("COMPARISON\n");
    printf("═══════════════════════════════════════════════════════════════════════\n\n");

    printf("                    │  Fixed κ=100  │  Adaptive κ   │  Winner\n");
    printf("────────────────────┼───────────────┼───────────────┼─────────\n");
    printf("  Final κ           │     100.0     │    %6.1f     │  %s\n",
           kappa_final, kappa_final < 100 ? "Adaptive" : "─");
    printf("  Chatter ratio     │     %5.2fx    │    %5.2fx     │  %s\n",
           chatter_fixed, chatter_adapt,
           fabsf(chatter_adapt - 1.0f) < fabsf(chatter_fixed - 1.0f) ? "Adaptive" : "Fixed");
    printf("  Frobenius error   │     %.4f    │    %.4f     │  %s\n",
           frob_fixed, frob_adapt,
           frob_adapt < frob_fixed ? "Adaptive" : "Fixed");
    printf("────────────────────┴───────────────┴───────────────┴─────────\n");

    float improvement = (float)(frob_fixed - frob_adapt) / (float)frob_fixed * 100.0f;

    printf("\n");
    printf("═══════════════════════════════════════════════════════════════════════\n");
    if (frob_adapt < frob_fixed)
    {
        printf("VERDICT: ✓ ADAPTIVE WINS - %.1f%% improvement in Frobenius error\n", improvement);
    }
    else
    {
        printf("VERDICT: ✗ FIXED WINS - Adaptive did not improve\n");
    }
    printf("═══════════════════════════════════════════════════════════════════════\n");
}

/*═══════════════════════════════════════════════════════════════════════════════
 * MAIN
 *═══════════════════════════════════════════════════════════════════════════════*/

int main(void)
{
    printf("╔═══════════════════════════════════════════════════════════════════════╗\n");
    printf("║           PGAS ADAPTIVE KAPPA VALIDATION                             ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════════╝\n");

    mkl_set_num_threads(8);
    uint32_t base_seed = (uint32_t)time(NULL);

    /* Test scenarios */
    AdaptiveScenario scenarios[] = {
        /* κ too strong for data: chatter > 1 → should DECREASE κ */
        {"κ=100 vs diag=0.90 (too strong)", 100.0f, 0.90f, 50.0f, 0},

        /* κ too weak for data: chatter < 1 → should INCREASE κ */
        {"κ=30 vs diag=0.98 (too weak)", 30.0f, 0.98f, 150.0f, 1},

        /* κ approximately right: chatter ≈ 1 → should stay stable */
        {"κ=100 vs diag=0.95 (correct)", 100.0f, 0.95f, 100.0f, 0},
    };

    int n_scenarios = sizeof(scenarios) / sizeof(scenarios[0]);

    for (int i = 0; i < n_scenarios; i++)
    {
        run_adaptive_test(&scenarios[i], base_seed + i * 10000);
    }

    /* Head-to-head comparison */
    run_comparison_test(base_seed + 99999);

    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════════════╗\n");
    printf("║                    ALL TESTS COMPLETE                                ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════════╝\n");

    return 0;
}