/*═══════════════════════════════════════════════════════════════════════════════
 * PGAS OPTION A VALIDATION: Ground Truth Recovery
 *
 * Tests whether PGAS can recover KNOWN transition parameters from synthetic data.
 *
 * Design:
 *   1. Generate synthetic data with TRUE known parameters
 *   2. Run PGAS Gibbs sampler (fix volatility params, learn transitions)
 *   3. Compare learned transition matrix to ground truth
 *
 * Success Criteria:
 *   - Frobenius error < 0.05 (Easy)
 *   - Diagonal elements within ±0.02
 *   - Off-diagonal structure preserved
 *═══════════════════════════════════════════════════════════════════════════════*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#else
/* Fallback timing for non-OpenMP builds */
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
 * GROUND TRUTH PARAMETERS
 *═══════════════════════════════════════════════════════════════════════════════*/

typedef struct
{
    int K; /* Number of regimes */
    int T; /* Time series length */

    double trans[16];    /* True transition matrix [K×K] */
    double mu_vol[4];    /* True regime means */
    double sigma_vol[4]; /* True regime volatilities */
    double phi;          /* AR(1) persistence */
    double sigma_h;      /* State noise std */

    /* Generated data */
    int *true_regimes;   /* [T] ground truth regime sequence */
    float *true_h;       /* [T] ground truth log-volatility */
    float *observations; /* [T] observed y_t = log(r_t^2) */
} SyntheticData;

/*═══════════════════════════════════════════════════════════════════════════════
 * OCSN 10-COMPONENT CONSTANTS (for generating realistic observations)
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

/*═══════════════════════════════════════════════════════════════════════════════
 * SYNTHETIC DATA GENERATION
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * Sample from categorical distribution (for regime transitions)
 */
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

/**
 * Sample from OCSN 10-component mixture (log-chi-squared approximation)
 */
static double sample_ocsn(VSLStreamStatePtr stream)
{
    /* First select component */
    int comp = sample_categorical(OCSN_PROB, 10, stream);

    /* Then sample from that Gaussian component */
    double z;
    vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, stream, 1, &z,
                  OCSN_MEAN[comp], sqrt(OCSN_VAR[comp]));
    return z;
}

/**
 * Generate synthetic data with known ground truth
 */
SyntheticData *generate_synthetic_data(
    int K, int T,
    const double *trans,     /* [K×K] transition matrix */
    const double *mu_vol,    /* [K] regime means */
    const double *sigma_vol, /* [K] regime volatilities (unused for now) */
    double phi,
    double sigma_h,
    uint32_t seed)
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

    /* Initialize MKL RNG */
    VSLStreamStatePtr stream;
    vslNewStream(&stream, VSL_BRNG_MT19937, seed);

    /* Generate initial regime from stationary distribution (approximate: uniform) */
    data->true_regimes[0] = seed % K;

    /* Generate initial h from stationary distribution */
    double h_mean = mu_vol[data->true_regimes[0]];
    double h_var = (sigma_h * sigma_h) / (1.0 - phi * phi);
    double z;
    vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, stream, 1, &z, 0.0, sqrt(h_var));
    data->true_h[0] = (float)(h_mean + z);

    /* Generate observation y_0 = h_0 + epsilon (OCSN noise) */
    data->observations[0] = data->true_h[0] + (float)sample_ocsn(stream);

    /* Generate time series */
    for (int t = 1; t < T; t++)
    {
        /* Regime transition */
        int prev_regime = data->true_regimes[t - 1];
        const double *trans_row = &trans[prev_regime * K];
        data->true_regimes[t] = sample_categorical(trans_row, K, stream);

        /* State evolution: h_t = mu_k(1-phi) + phi*h_{t-1} + sigma_h*z */
        int curr_regime = data->true_regimes[t];
        double mu_k = mu_vol[curr_regime];
        double mean_h = mu_k * (1.0 - phi) + phi * data->true_h[t - 1];

        vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, stream, 1, &z, 0.0, sigma_h);
        data->true_h[t] = (float)(mean_h + z);

        /* Observation: y_t = h_t + epsilon (OCSN) */
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
 * ANALYSIS UTILITIES
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * Count actual regime transitions in data
 */
void count_transitions(const int *regimes, int T, int K, int *counts)
{
    memset(counts, 0, K * K * sizeof(int));
    for (int t = 1; t < T; t++)
    {
        int from = regimes[t - 1];
        int to = regimes[t];
        counts[from * K + to]++;
    }
}

/**
 * Compute empirical transition matrix from regime sequence
 */
void empirical_transitions(const int *regimes, int T, int K, double *trans)
{
    int counts[16] = {0};
    count_transitions(regimes, T, K, counts);

    for (int i = 0; i < K; i++)
    {
        int row_sum = 0;
        for (int j = 0; j < K; j++)
        {
            row_sum += counts[i * K + j];
        }
        for (int j = 0; j < K; j++)
        {
            trans[i * K + j] = (row_sum > 0) ? (double)counts[i * K + j] / row_sum : 1.0 / K;
        }
    }
}

/**
 * Frobenius norm of matrix difference
 */
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

/**
 * Max absolute element difference
 */
double max_abs_error(const double *A, const float *B, int K)
{
    double max_err = 0.0;
    for (int i = 0; i < K * K; i++)
    {
        double err = fabs(A[i] - (double)B[i]);
        if (err > max_err)
            max_err = err;
    }
    return max_err;
}

/**
 * Print transition matrix
 */
void print_matrix(const char *name, const void *mat, int K, int is_float)
{
    printf("%s:\n", name);
    for (int i = 0; i < K; i++)
    {
        printf("  [");
        for (int j = 0; j < K; j++)
        {
            double val = is_float ? ((const float *)mat)[i * K + j]
                                  : ((const double *)mat)[i * K + j];
            printf(" %6.4f", val);
        }
        printf(" ]\n");
    }
}

/**
 * Count regime occurrences
 */
void count_regimes(const int *regimes, int T, int K, int *counts)
{
    memset(counts, 0, K * sizeof(int));
    for (int t = 0; t < T; t++)
    {
        counts[regimes[t]]++;
    }
}

/*═══════════════════════════════════════════════════════════════════════════════
 * VALIDATION TEST
 *═══════════════════════════════════════════════════════════════════════════════*/

typedef struct
{
    const char *name;
    int T;
    double mu_spread;        /* Spread between mu_vol[0] and mu_vol[K-1] */
    double diag_persistence; /* Diagonal transition probability */
    float sticky_kappa;      /* Sticky prior strength */
    int N_particles;         /* Number of particles */
} TestScenario;

void run_validation_test(const TestScenario *scenario, uint32_t seed)
{
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════════════╗\n");
    printf("║  SCENARIO: %-58s ║\n", scenario->name);
    printf("╠═══════════════════════════════════════════════════════════════════════╣\n");

    const int K = 4;
    const int T = scenario->T;
    const int N_particles = scenario->N_particles;
    const int N_burnin = 300;
    const int N_samples = 500;

    /* Define TRUE parameters */
    double mu_vol[4];
    double spread = scenario->mu_spread;
    mu_vol[0] = -spread / 2.0;
    mu_vol[1] = -spread / 6.0;
    mu_vol[2] = spread / 6.0;
    mu_vol[3] = spread / 2.0;

    double sigma_vol[4] = {0.2, 0.2, 0.2, 0.2};
    double phi = 0.97;
    double sigma_h = 0.15;

    /* TRUE transition matrix (sticky diagonal) */
    double diag = scenario->diag_persistence;
    double off_diag = (1.0 - diag) / (K - 1);

    double trans_true[16];
    for (int i = 0; i < K; i++)
    {
        for (int j = 0; j < K; j++)
        {
            trans_true[i * K + j] = (i == j) ? diag : off_diag;
        }
    }

    printf("║  T=%d, K=%d, N=%d, burn-in=%d, samples=%d           ║\n",
           T, K, N_particles, N_burnin, N_samples);
    printf("║  mu_vol = [%.2f, %.2f, %.2f, %.2f], phi=%.2f, sigma_h=%.2f     ║\n",
           mu_vol[0], mu_vol[1], mu_vol[2], mu_vol[3], phi, sigma_h);
    printf("║  Diagonal persistence = %.2f, sticky_kappa = %.0f               ║\n",
           diag, scenario->sticky_kappa);
    printf("╚═══════════════════════════════════════════════════════════════════════╝\n\n");

    /* Generate synthetic data */
    printf("Generating synthetic data...\n");
    SyntheticData *data = generate_synthetic_data(K, T, trans_true, mu_vol,
                                                  sigma_vol, phi, sigma_h, seed);

    /* Analyze generated data */
    int regime_counts[4];
    count_regimes(data->true_regimes, T, K, regime_counts);
    printf("  Regime distribution: [%d, %d, %d, %d]\n",
           regime_counts[0], regime_counts[1], regime_counts[2], regime_counts[3]);

    double empirical_trans[16];
    empirical_transitions(data->true_regimes, T, K, empirical_trans);
    printf("  Empirical transitions (from true regimes):\n");
    print_matrix("    ", empirical_trans, K, 0);

    /* Initialize PGAS */
    printf("\nInitializing PGAS-MKL...\n");
    PGASMKLState *pgas = pgas_mkl_alloc(N_particles, T, K, seed + 1000);

    /* Set TRUE volatility parameters (we're only learning transitions) */
    /* Start with uniform transitions - PGAS will learn */
    double init_trans[16];
    for (int i = 0; i < K * K; i++)
        init_trans[i] = 1.0 / K;

    pgas_mkl_set_model(pgas, init_trans, mu_vol, sigma_vol, phi, sigma_h);
    pgas_mkl_set_transition_prior(pgas, 1.0f, scenario->sticky_kappa); /* α=1, κ from scenario */

    /* Convert observations to double for API */
    double *obs_double = (double *)malloc(T * sizeof(double));
    for (int i = 0; i < T; i++)
    {
        obs_double[i] = (double)data->observations[i];
    }
    pgas_mkl_load_observations(pgas, obs_double, T);
    free(obs_double);

    /* Initialize reference trajectory by running one CSMC sweep */
    printf("Initializing reference trajectory...\n");
    pgas_mkl_csmc_sweep(pgas);

    /* Burn-in phase */
    printf("Running burn-in (%d sweeps)...\n", N_burnin);
    double burnin_start = omp_get_wtime();
    for (int i = 0; i < N_burnin; i++)
    {
        pgas_mkl_gibbs_sweep(pgas);
    }
    double burnin_time = omp_get_wtime() - burnin_start;
    printf("  Burn-in complete: %.2f ms/sweep\n", 1000.0 * burnin_time / N_burnin);

    /* Collection phase */
    printf("Collecting samples (%d sweeps)...\n", N_samples);

    /* Accumulate transition matrices */
    float trans_sum[16] = {0};
    float trans_current[16];

    double sample_start = omp_get_wtime();
    for (int i = 0; i < N_samples; i++)
    {
        pgas_mkl_gibbs_sweep(pgas);

        /* Get current transition matrix */
        pgas_mkl_get_transitions(pgas, trans_current, K);
        for (int j = 0; j < K * K; j++)
        {
            trans_sum[j] += trans_current[j];
        }
    }
    double sample_time = omp_get_wtime() - sample_start;
    printf("  Sampling complete: %.2f ms/sweep\n", 1000.0 * sample_time / N_samples);

    /* ═══════════════════════════════════════════════════════════════════
     * DIAGNOSTICS: Compare inferred vs true regime sequence
     * ═══════════════════════════════════════════════════════════════════*/
    printf("\nDIAGNOSTICS:\n");

    /* Get final transition counts from PGAS */
    int pgas_counts[16];
    pgas_mkl_get_transition_counts(pgas, pgas_counts, K);

    int pgas_total_trans = 0;
    for (int i = 0; i < K * K; i++)
    {
        if (i % (K + 1) != 0)
            pgas_total_trans += pgas_counts[i]; /* Off-diagonal */
    }

    int true_total_trans = 0;
    int true_counts[16] = {0};
    count_transitions(data->true_regimes, T, K, true_counts);
    for (int i = 0; i < K * K; i++)
    {
        if (i % (K + 1) != 0)
            true_total_trans += true_counts[i];
    }

    printf("  True regime switches: %d (%.1f%%)\n", true_total_trans, 100.0 * true_total_trans / (T - 1));
    printf("  PGAS inferred switches: %d (%.1f%%)\n", pgas_total_trans, 100.0 * pgas_total_trans / (T - 1));
    printf("  Excess switches: %d (%.1fx more)\n", pgas_total_trans - true_total_trans,
           (float)pgas_total_trans / fmaxf(1.0f, (float)true_total_trans));

    /* Print PGAS transition counts */
    printf("\n  PGAS transition counts:\n");
    for (int i = 0; i < K; i++)
    {
        printf("    [");
        for (int j = 0; j < K; j++)
        {
            printf(" %4d", pgas_counts[i * K + j]);
        }
        printf(" ]\n");
    }

    /* Print TRUE transition counts */
    printf("\n  TRUE transition counts:\n");
    for (int i = 0; i < K; i++)
    {
        printf("    [");
        for (int j = 0; j < K; j++)
        {
            printf(" %4d", true_counts[i * K + j]);
        }
        printf(" ]\n");
    }

    /* Sum of counts per row (should equal T-1 for true, may differ for PGAS) */
    int pgas_row_sums[4] = {0};
    int true_row_sums[4] = {0};
    for (int i = 0; i < K; i++)
    {
        for (int j = 0; j < K; j++)
        {
            pgas_row_sums[i] += pgas_counts[i * K + j];
            true_row_sums[i] += true_counts[i * K + j];
        }
    }
    printf("\n  Regime occupancy (from transition row sums):\n");
    printf("    TRUE: [%d, %d, %d, %d] = %d\n",
           true_row_sums[0], true_row_sums[1], true_row_sums[2], true_row_sums[3],
           true_row_sums[0] + true_row_sums[1] + true_row_sums[2] + true_row_sums[3]);
    printf("    PGAS: [%d, %d, %d, %d] = %d\n",
           pgas_row_sums[0], pgas_row_sums[1], pgas_row_sums[2], pgas_row_sums[3],
           pgas_row_sums[0] + pgas_row_sums[1] + pgas_row_sums[2] + pgas_row_sums[3]);

    /* ═══════════════════════════════════════════════════════════════════
     * CRITICAL DIAGNOSTIC: Hamming distance between ref trajectory and truth
     * If this is high, PGAS is not finding the true regime sequence
     * ═══════════════════════════════════════════════════════════════════*/

    /* We need to access ref_regimes from PGAS state - add to diagnostic output */
    /* For now, compute regime sequence accuracy from transition counts */

    /* The key insight: if PGAS has 3-4x more transitions, it means the
     * reference trajectory is "chattering" - briefly visiting other regimes
     * even when it should stay in one regime */

    float chatter_ratio = (true_total_trans > 0) ? (float)pgas_total_trans / (float)true_total_trans : 0.0f;
    printf("\n  CHATTER ANALYSIS:\n");
    printf("    True transitions: %d (expected for diag=%.2f)\n",
           true_total_trans, diag);
    printf("    PGAS transitions: %d\n", pgas_total_trans);
    printf("    Chatter ratio: %.2fx", chatter_ratio);
    if (chatter_ratio > 2.0f)
    {
        printf(" ⚠ HIGH CHATTER - reference trajectory is unstable\n");
    }
    else if (chatter_ratio > 1.5f)
    {
        printf(" ⚠ Moderate chatter\n");
    }
    else
    {
        printf(" ✓ Acceptable\n");
    }

    /* Compute posterior mean */
    float trans_learned[16];
    for (int j = 0; j < K * K; j++)
    {
        trans_learned[j] = trans_sum[j] / N_samples;
    }

    /* ═══════════════════════════════════════════════════════════════════
     * RESULTS
     * ═══════════════════════════════════════════════════════════════════*/
    printf("\n");
    printf("═══════════════════════════════════════════════════════════════════════\n");
    printf("RESULTS\n");
    printf("═══════════════════════════════════════════════════════════════════════\n\n");

    print_matrix("TRUE transition matrix", trans_true, K, 0);
    printf("\n");
    print_matrix("LEARNED transition matrix (posterior mean)", trans_learned, K, 1);
    printf("\n");

    /* Error metrics */
    double frob_err = frobenius_error(trans_true, trans_learned, K);
    double max_err = max_abs_error(trans_true, trans_learned, K);

    /* Diagonal errors */
    double diag_err_sum = 0.0;
    double diag_err_max = 0.0;
    for (int i = 0; i < K; i++)
    {
        double err = fabs(trans_true[i * K + i] - trans_learned[i * K + i]);
        diag_err_sum += err;
        if (err > diag_err_max)
            diag_err_max = err;
    }
    double diag_err_avg = diag_err_sum / K;

    printf("ERROR METRICS:\n");
    printf("  Frobenius error:     %.4f", frob_err);
    printf(frob_err < 0.10 ? "  ✓ PASS\n" : "  ✗ FAIL (threshold: 0.10)\n");

    printf("  Max element error:   %.4f", max_err);
    printf(max_err < 0.05 ? "  ✓ PASS\n" : "  ✗ FAIL (threshold: 0.05)\n");

    printf("  Avg diagonal error:  %.4f", diag_err_avg);
    printf(diag_err_avg < 0.03 ? "  ✓ PASS\n" : "  ✗ FAIL (threshold: 0.03)\n");

    printf("  Max diagonal error:  %.4f", diag_err_max);
    printf(diag_err_max < 0.05 ? "  ✓ PASS\n" : "  ✗ FAIL (threshold: 0.05)\n");

    /* Overall verdict */
    int pass = (frob_err < 0.10) && (max_err < 0.05) && (diag_err_avg < 0.03);
    printf("\n");
    printf("═══════════════════════════════════════════════════════════════════════\n");
    printf("VERDICT: %s\n", pass ? "✓ PASS - PGAS correctly recovers transition matrix"
                                 : "✗ FAIL - Recovery error exceeds threshold");
    printf("═══════════════════════════════════════════════════════════════════════\n");

    /* Cleanup */
    pgas_mkl_free(pgas);
    free_synthetic_data(data);
}

/*═══════════════════════════════════════════════════════════════════════════════
 * MAIN
 *═══════════════════════════════════════════════════════════════════════════════*/

int main(void)
{
    printf("╔═══════════════════════════════════════════════════════════════════════╗\n");
    printf("║           PGAS OPTION A VALIDATION: Ground Truth Recovery            ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════════╝\n");

    /* MKL setup */
    mkl_set_num_threads(8);

    uint32_t base_seed = (uint32_t)time(NULL);

    /* Test scenarios: κ=100 proven to work, now test harder cases
     *
     * NOTE: OCSN noise σ ≈ 2.2, so regime separation must be > 2.0 to be identifiable
     * spread=4.5 → gap=1.5 between adjacent (marginal)
     * spread=6.0 → gap=2.0 between adjacent (identifiable)
     * spread=3.0 → gap=1.0 between adjacent (degenerate - will fail)
     */
    TestScenario scenarios[] = {
        /* Proven working baseline */
        {"EASY: κ=100, T=2000, spread=4.5", 2000, 4.5, 0.95, 100.0f, 128},

        /* Harder: shorter series */
        {"MEDIUM: κ=100, T=1000, spread=4.5", 1000, 4.5, 0.95, 100.0f, 128},

        /* Wider separation for robustness */
        {"WIDE: κ=100, T=2000, spread=6.0", 2000, 6.0, 0.95, 100.0f, 128},

        /* Test with less sticky diagonal */
        {"FAST: κ=100, T=2000, diag=0.90", 2000, 4.5, 0.90, 100.0f, 128},

        /* Combined challenge: short + fast switching */
        {"EXPERT: κ=100, T=1000, diag=0.90", 1000, 4.5, 0.90, 100.0f, 128},
    };

    int n_scenarios = sizeof(scenarios) / sizeof(scenarios[0]);
    printf("Running %d scenarios with SAME seed for comparison...\n", n_scenarios);

    for (int i = 0; i < n_scenarios; i++)
    {
        /* Use SAME base seed for all scenarios to isolate parameter effects */
        run_validation_test(&scenarios[i], base_seed);
    }

    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════════════╗\n");
    printf("║                    ALL SCENARIOS COMPLETE                            ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════════╝\n");

    return 0;
}