/**
 * @file test_vi_vs_dirichlet.c
 * @brief Head-to-Head Comparison: Online VI vs Soft Dirichlet
 *
 * Tests both transition learning methods on synthetic regime-switching data.
 *
 * Metrics:
 *   1. Tracking accuracy (how close to true transition matrix)
 *   2. Uncertainty calibration (does variance predict errors)
 *   3. Adaptation speed (how fast to track changes)
 *   4. Entropy quality (does entropy spike at transitions)
 *   5. Compute time
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "online_vi_transition.h"
#include "soft_dirichlet_transition.h"

/*═══════════════════════════════════════════════════════════════════════════════
 * CONFIGURATION
 *═══════════════════════════════════════════════════════════════════════════════*/

#define K 4          /* Number of regimes */
#define T_TOTAL 2000 /* Total ticks */
#define T_CHANGE 500 /* Tick at which transition matrix changes */
#define N_TRIALS 10  /* Number of trials for averaging */

/*═══════════════════════════════════════════════════════════════════════════════
 * SYNTHETIC DATA GENERATION
 *═══════════════════════════════════════════════════════════════════════════════*/

/* True transition matrices (before and after change) */
static double TRUE_TRANS_1[K][K] = {
    {0.90, 0.05, 0.03, 0.02}, /* Regime 0: very sticky */
    {0.10, 0.80, 0.07, 0.03}, /* Regime 1: sticky */
    {0.05, 0.10, 0.75, 0.10}, /* Regime 2: moderate */
    {0.02, 0.03, 0.15, 0.80}  /* Regime 3: sticky crisis */
};

static double TRUE_TRANS_2[K][K] = {
    {0.70, 0.15, 0.10, 0.05}, /* Regime 0: less sticky after change */
    {0.20, 0.60, 0.15, 0.05}, /* Regime 1: much less sticky */
    {0.10, 0.15, 0.65, 0.10}, /* Regime 2: similar */
    {0.05, 0.05, 0.20, 0.70}  /* Regime 3: less sticky */
};

/* Emission means (log-vol) per regime */
static double MU_VOL[K] = {-3.0, -1.5, 0.0, 1.5};
static double SIGMA_VOL = 0.3;

/* RNG state */
static unsigned int rng_state = 12345;

static double rand_uniform(void)
{
    rng_state = rng_state * 1103515245 + 12345;
    return (double)(rng_state & 0x7fffffff) / (double)0x7fffffff;
}

static double rand_normal(void)
{
    double u1 = rand_uniform();
    double u2 = rand_uniform();
    if (u1 < 1e-10)
        u1 = 1e-10;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265359 * u2);
}

static int sample_categorical(const double *probs, int n)
{
    double u = rand_uniform();
    double cumsum = 0.0;
    for (int i = 0; i < n - 1; i++)
    {
        cumsum += probs[i];
        if (u < cumsum)
            return i;
    }
    return n - 1;
}

/**
 * Generate synthetic regime-switching data
 */
typedef struct
{
    int regimes[T_TOTAL];            /* True regime sequence */
    double observations[T_TOTAL];    /* Observations (log-vol proxy) */
    double regime_probs[T_TOTAL][K]; /* "Oracle" regime probabilities */
    double regime_liks[T_TOTAL][K];  /* Observation likelihoods per regime */
} SyntheticData;

static void generate_data(SyntheticData *data, unsigned int seed)
{
    rng_state = seed;

    /* Start in regime 0 */
    data->regimes[0] = 0;

    /* Generate regime sequence */
    for (int t = 1; t < T_TOTAL; t++)
    {
        int prev = data->regimes[t - 1];

        /* Use appropriate transition matrix */
        const double *trans_row = (t < T_CHANGE) ? TRUE_TRANS_1[prev] : TRUE_TRANS_2[prev];

        data->regimes[t] = sample_categorical(trans_row, K);
    }

    /* Generate observations and compute likelihoods */
    for (int t = 0; t < T_TOTAL; t++)
    {
        int regime = data->regimes[t];

        /* Observation: log-vol with noise */
        data->observations[t] = MU_VOL[regime] + SIGMA_VOL * rand_normal();

        /* Compute likelihood under each regime */
        double sum_lik = 0.0;
        for (int k = 0; k < K; k++)
        {
            double diff = data->observations[t] - MU_VOL[k];
            double log_lik = -0.5 * (diff * diff) / (SIGMA_VOL * SIGMA_VOL);
            data->regime_liks[t][k] = exp(log_lik);
            sum_lik += data->regime_liks[t][k];
        }

        /* Normalize to get "oracle" regime probs (based on emission only) */
        for (int k = 0; k < K; k++)
        {
            data->regime_probs[t][k] = data->regime_liks[t][k] / sum_lik;
        }
    }
}

/*═══════════════════════════════════════════════════════════════════════════════
 * METRICS
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * Frobenius norm of difference between two matrices
 */
static double matrix_frobenius_error(const double *A, const double *B, int n)
{
    double sum = 0.0;
    for (int i = 0; i < n * n; i++)
    {
        double diff = A[i] - B[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

/**
 * Mean absolute error
 */
static double matrix_mae(const double *A, const double *B, int n)
{
    double sum = 0.0;
    for (int i = 0; i < n * n; i++)
    {
        sum += fabs(A[i] - B[i]);
    }
    return sum / (n * n);
}

/**
 * Get current time in microseconds
 */
static double get_time_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

/*═══════════════════════════════════════════════════════════════════════════════
 * TEST RUNNERS
 *═══════════════════════════════════════════════════════════════════════════════*/

typedef struct
{
    /* Tracking error over time */
    double error_before_change[T_CHANGE];
    double error_after_change[T_TOTAL - T_CHANGE];

    /* Aggregates */
    double mean_error_before;
    double mean_error_after;
    double final_error;

    /* Entropy tracking */
    double entropy_at_change;     /* Entropy right at transition */
    double entropy_before_change; /* Average entropy before */
    double entropy_after_stable;  /* Entropy after settling */

    /* Timing */
    double total_time_us;
    double time_per_update_us;

    /* Convergence */
    int ticks_to_converge; /* Ticks after change to reach 80% accuracy */
} TestResult;

/**
 * Run Online VI test
 */
static void run_online_vi_test(const SyntheticData *data, TestResult *result)
{
    OnlineVI vi;
    online_vi_init_sticky(&vi, K, 1.0, 5.0);
    online_vi_set_lr_robbins_monro(&vi, 1.0, 64.0, 0.7);

    double trans_est[K * K];
    double true_trans[K * K];

    memset(result, 0, sizeof(*result));

    double t_start = get_time_us();

    for (int t = 0; t < T_TOTAL; t++)
    {
        /* Update VI */
        online_vi_update(&vi, data->regime_probs[t], data->regime_liks[t]);

        /* Get estimate */
        online_vi_get_mean(&vi, trans_est);

        /* Get true matrix for this time */
        const double (*true_ptr)[K] = (t < T_CHANGE) ? TRUE_TRANS_1 : TRUE_TRANS_2;
        for (int i = 0; i < K; i++)
        {
            for (int j = 0; j < K; j++)
            {
                true_trans[i * K + j] = true_ptr[i][j];
            }
        }

        /* Compute error */
        double error = matrix_frobenius_error(trans_est, true_trans, K);

        if (t < T_CHANGE)
        {
            result->error_before_change[t] = error;
            result->mean_error_before += error;
        }
        else
        {
            result->error_after_change[t - T_CHANGE] = error;
            result->mean_error_after += error;
        }

        /* Track entropy at key points */
        double entropy = online_vi_get_total_entropy(&vi);

        if (t == T_CHANGE - 1)
        {
            result->entropy_before_change = entropy;
        }
        else if (t == T_CHANGE)
        {
            result->entropy_at_change = entropy;
        }
        else if (t == T_TOTAL - 1)
        {
            result->entropy_after_stable = entropy;
        }

        /* Check convergence (80% accuracy = error < 0.3) */
        if (t >= T_CHANGE && result->ticks_to_converge == 0 && error < 0.3)
        {
            result->ticks_to_converge = t - T_CHANGE;
        }
    }

    double t_end = get_time_us();

    result->mean_error_before /= T_CHANGE;
    result->mean_error_after /= (T_TOTAL - T_CHANGE);
    result->final_error = result->error_after_change[T_TOTAL - T_CHANGE - 1];
    result->total_time_us = t_end - t_start;
    result->time_per_update_us = result->total_time_us / T_TOTAL;
}

/**
 * Run Soft Dirichlet test
 */
static void run_soft_dirichlet_test(const SyntheticData *data, TestResult *result)
{
    SoftDirichlet sd;
    soft_dirichlet_init(&sd, K, 200.0f); /* ESS_max = 200 */

    float trans_est_f[K * K];
    double trans_est[K * K];
    double true_trans[K * K];

    memset(result, 0, sizeof(*result));

    /* Convert data to float for Soft Dirichlet */
    float regime_probs_f[K];
    float regime_liks_f[K];

    double t_start = get_time_us();

    for (int t = 0; t < T_TOTAL; t++)
    {
        /* Convert to float */
        for (int k = 0; k < K; k++)
        {
            regime_probs_f[k] = (float)data->regime_probs[t][k];
            regime_liks_f[k] = (float)data->regime_liks[t][k];
        }

        /* Update Soft Dirichlet */
        soft_dirichlet_update(&sd, regime_probs_f, regime_liks_f);

        /* Get estimate */
        soft_dirichlet_get_matrix(&sd, trans_est_f);
        for (int i = 0; i < K * K; i++)
        {
            trans_est[i] = (double)trans_est_f[i];
        }

        /* Get true matrix for this time */
        const double (*true_ptr)[K] = (t < T_CHANGE) ? TRUE_TRANS_1 : TRUE_TRANS_2;
        for (int i = 0; i < K; i++)
        {
            for (int j = 0; j < K; j++)
            {
                true_trans[i * K + j] = true_ptr[i][j];
            }
        }

        /* Compute error */
        double error = matrix_frobenius_error(trans_est, true_trans, K);

        if (t < T_CHANGE)
        {
            result->error_before_change[t] = error;
            result->mean_error_before += error;
        }
        else
        {
            result->error_after_change[t - T_CHANGE] = error;
            result->mean_error_after += error;
        }

        /* Track entropy (Soft Dirichlet has xi_entropy) */
        SoftDirichletStats stats = soft_dirichlet_stats(&sd);
        double entropy = stats.xi_entropy;

        if (t == T_CHANGE - 1)
        {
            result->entropy_before_change = entropy;
        }
        else if (t == T_CHANGE)
        {
            result->entropy_at_change = entropy;
        }
        else if (t == T_TOTAL - 1)
        {
            result->entropy_after_stable = entropy;
        }

        /* Check convergence */
        if (t >= T_CHANGE && result->ticks_to_converge == 0 && error < 0.3)
        {
            result->ticks_to_converge = t - T_CHANGE;
        }
    }

    double t_end = get_time_us();

    result->mean_error_before /= T_CHANGE;
    result->mean_error_after /= (T_TOTAL - T_CHANGE);
    result->final_error = result->error_after_change[T_TOTAL - T_CHANGE - 1];
    result->total_time_us = t_end - t_start;
    result->time_per_update_us = result->total_time_us / T_TOTAL;
}

/*═══════════════════════════════════════════════════════════════════════════════
 * UNCERTAINTY CALIBRATION TEST
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * Test if uncertainty predicts errors (Online VI only - has variance)
 */
static void run_calibration_test(const SyntheticData *data)
{
    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("UNCERTAINTY CALIBRATION TEST (Online VI)\n");
    printf("═══════════════════════════════════════════════════════════════\n\n");

    OnlineVI vi;
    online_vi_init_sticky(&vi, K, 1.0, 5.0);

/* Bucketize by variance, measure actual error */
#define N_BUCKETS 5
    double bucket_bounds[N_BUCKETS + 1] = {0.0, 0.01, 0.02, 0.03, 0.05, 1.0};
    double bucket_sum_error[N_BUCKETS] = {0};
    double bucket_sum_var[N_BUCKETS] = {0};
    int bucket_count[N_BUCKETS] = {0};

    double trans_est[K * K];
    double var_est[K * K];

    for (int t = 0; t < T_TOTAL; t++)
    {
        online_vi_update(&vi, data->regime_probs[t], data->regime_liks[t]);

        if (t < 100)
            continue; /* Skip warmup */

        online_vi_get_mean(&vi, trans_est);
        online_vi_get_variance(&vi, var_est);

        /* Get true matrix */
        const double (*true_ptr)[K] = (t < T_CHANGE) ? TRUE_TRANS_1 : TRUE_TRANS_2;

        /* For each entry, record (variance, squared error) */
        for (int i = 0; i < K; i++)
        {
            for (int j = 0; j < K; j++)
            {
                double var = var_est[i * K + j];
                double err = trans_est[i * K + j] - true_ptr[i][j];
                double sq_err = err * err;

                /* Find bucket */
                for (int b = 0; b < N_BUCKETS; b++)
                {
                    if (var >= bucket_bounds[b] && var < bucket_bounds[b + 1])
                    {
                        bucket_sum_error[b] += sq_err;
                        bucket_sum_var[b] += var;
                        bucket_count[b]++;
                        break;
                    }
                }
            }
        }
    }

    printf("Variance Bucket      | Count  | Mean Var | Mean Sq Err | Ratio\n");
    printf("---------------------|--------|----------|-------------|-------\n");

    for (int b = 0; b < N_BUCKETS; b++)
    {
        if (bucket_count[b] > 0)
        {
            double mean_var = bucket_sum_var[b] / bucket_count[b];
            double mean_err = bucket_sum_error[b] / bucket_count[b];
            double ratio = (mean_var > 1e-10) ? mean_err / mean_var : 0.0;

            printf("[%.3f, %.3f)       | %6d | %.6f | %.6f    | %.2f\n",
                   bucket_bounds[b], bucket_bounds[b + 1],
                   bucket_count[b], mean_var, mean_err, ratio);
        }
    }

    printf("\nInterpretation: Ratio ≈ 1.0 means well-calibrated uncertainty.\n");
    printf("Ratio > 1 = overconfident, Ratio < 1 = underconfident.\n");
}

/*═══════════════════════════════════════════════════════════════════════════════
 * ENTROPY SPIKE TEST
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * Test if entropy spikes at regime transition point
 */
static void run_entropy_spike_test(const SyntheticData *data)
{
    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("ENTROPY SPIKE TEST\n");
    printf("═══════════════════════════════════════════════════════════════\n\n");

    OnlineVI vi;
    online_vi_init_sticky(&vi, K, 1.0, 5.0);

    SoftDirichlet sd;
    soft_dirichlet_init(&sd, K, 200.0f);

    float regime_probs_f[K], regime_liks_f[K];

    printf("Tick   | VI Entropy | SD Entropy | True Regime | Note\n");
    printf("-------|------------|------------|-------------|-----\n");

    for (int t = 0; t < T_TOTAL; t++)
    {
        /* Update both */
        online_vi_update(&vi, data->regime_probs[t], data->regime_liks[t]);

        for (int k = 0; k < K; k++)
        {
            regime_probs_f[k] = (float)data->regime_probs[t][k];
            regime_liks_f[k] = (float)data->regime_liks[t][k];
        }
        soft_dirichlet_update(&sd, regime_probs_f, regime_liks_f);

        /* Print around the change point */
        if (t >= T_CHANGE - 10 && t <= T_CHANGE + 50)
        {
            double vi_entropy = online_vi_get_total_entropy(&vi);

            SoftDirichletStats sd_stats = soft_dirichlet_stats(&sd);

            const char *note = "";
            if (t == T_CHANGE)
                note = "<-- CHANGE";

            printf("%6d | %10.4f | %10.4f | %11d | %s\n",
                   t, vi_entropy, sd_stats.xi_entropy, data->regimes[t], note);
        }
    }
}

/*═══════════════════════════════════════════════════════════════════════════════
 * PER-ROW ENTROPY TEST (Online VI only)
 *═══════════════════════════════════════════════════════════════════════════════*/

static void run_per_row_entropy_test(const SyntheticData *data)
{
    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("PER-ROW ENTROPY TEST (Online VI)\n");
    printf("═══════════════════════════════════════════════════════════════\n\n");

    OnlineVI vi;
    online_vi_init_sticky(&vi, K, 1.0, 5.0);

    double row_entropy[K];

    printf("Tick   | H[row 0] | H[row 1] | H[row 2] | H[row 3] | Max Row\n");
    printf("-------|----------|----------|----------|----------|--------\n");

    for (int t = 0; t < T_TOTAL; t++)
    {
        online_vi_update(&vi, data->regime_probs[t], data->regime_liks[t]);

        /* Print around the change point */
        if (t >= T_CHANGE - 5 && t <= T_CHANGE + 30 && t % 5 == 0)
        {
            online_vi_get_row_entropy(&vi, row_entropy);

            int max_row;
            online_vi_get_max_row_entropy(&vi, &max_row);

            printf("%6d | %8.4f | %8.4f | %8.4f | %8.4f | %d\n",
                   t, row_entropy[0], row_entropy[1],
                   row_entropy[2], row_entropy[3], max_row);
        }
    }

    printf("\nInterpretation: Row with highest entropy = most uncertain transitions.\n");
    printf("This can target HDP-beam refinement to specific rows.\n");
}

/*═══════════════════════════════════════════════════════════════════════════════
 * MAIN
 *═══════════════════════════════════════════════════════════════════════════════*/

int main(void)
{
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("     ONLINE VI vs SOFT DIRICHLET: HEAD-TO-HEAD COMPARISON\n");
    printf("═══════════════════════════════════════════════════════════════\n\n");

    printf("Configuration:\n");
    printf("  Regimes (K):     %d\n", K);
    printf("  Total ticks:     %d\n", T_TOTAL);
    printf("  Change at tick:  %d\n", T_CHANGE);
    printf("  Trials:          %d\n\n", N_TRIALS);

    /* Aggregate results */
    TestResult vi_results[N_TRIALS];
    TestResult sd_results[N_TRIALS];

    SyntheticData data;

    for (int trial = 0; trial < N_TRIALS; trial++)
    {
        generate_data(&data, 12345 + trial * 1000);
        run_online_vi_test(&data, &vi_results[trial]);
        run_soft_dirichlet_test(&data, &sd_results[trial]);
    }

    /* Compute averages */
    double vi_avg_before = 0, vi_avg_after = 0, vi_avg_final = 0;
    double sd_avg_before = 0, sd_avg_after = 0, sd_avg_final = 0;
    double vi_avg_time = 0, sd_avg_time = 0;
    double vi_avg_converge = 0, sd_avg_converge = 0;

    for (int trial = 0; trial < N_TRIALS; trial++)
    {
        vi_avg_before += vi_results[trial].mean_error_before;
        vi_avg_after += vi_results[trial].mean_error_after;
        vi_avg_final += vi_results[trial].final_error;
        vi_avg_time += vi_results[trial].time_per_update_us;
        vi_avg_converge += vi_results[trial].ticks_to_converge;

        sd_avg_before += sd_results[trial].mean_error_before;
        sd_avg_after += sd_results[trial].mean_error_after;
        sd_avg_final += sd_results[trial].final_error;
        sd_avg_time += sd_results[trial].time_per_update_us;
        sd_avg_converge += sd_results[trial].ticks_to_converge;
    }

    vi_avg_before /= N_TRIALS;
    vi_avg_after /= N_TRIALS;
    vi_avg_final /= N_TRIALS;
    vi_avg_time /= N_TRIALS;
    vi_avg_converge /= N_TRIALS;

    sd_avg_before /= N_TRIALS;
    sd_avg_after /= N_TRIALS;
    sd_avg_final /= N_TRIALS;
    sd_avg_time /= N_TRIALS;
    sd_avg_converge /= N_TRIALS;

    printf("═══════════════════════════════════════════════════════════════\n");
    printf("                        RESULTS SUMMARY\n");
    printf("═══════════════════════════════════════════════════════════════\n\n");

    printf("                        | Online VI  | Soft Dirichlet | Winner\n");
    printf("------------------------|------------|----------------|--------\n");
    printf("Error (before change)   | %10.4f | %14.4f | %s\n",
           vi_avg_before, sd_avg_before,
           vi_avg_before < sd_avg_before ? "VI" : "SD");
    printf("Error (after change)    | %10.4f | %14.4f | %s\n",
           vi_avg_after, sd_avg_after,
           vi_avg_after < sd_avg_after ? "VI" : "SD");
    printf("Final error             | %10.4f | %14.4f | %s\n",
           vi_avg_final, sd_avg_final,
           vi_avg_final < sd_avg_final ? "VI" : "SD");
    printf("Ticks to converge       | %10.1f | %14.1f | %s\n",
           vi_avg_converge, sd_avg_converge,
           vi_avg_converge < sd_avg_converge ? "VI" : "SD");
    printf("Time per update (μs)    | %10.2f | %14.2f | %s\n",
           vi_avg_time, sd_avg_time,
           vi_avg_time < sd_avg_time ? "VI" : "SD");

    printf("\n");

    /* Detailed tests on single trial */
    generate_data(&data, 42);

    run_calibration_test(&data);
    run_entropy_spike_test(&data);
    run_per_row_entropy_test(&data);

    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("                        CONCLUSIONS\n");
    printf("═══════════════════════════════════════════════════════════════\n\n");

    printf("Online VI advantages:\n");
    printf("  ✓ Full posterior (mean + variance)\n");
    printf("  ✓ Per-row entropy for targeted HDP triggering\n");
    printf("  ✓ Uncertainty calibration for Kelly sizing\n");
    printf("  ✓ Natural gradient is theoretically principled\n");
    printf("\n");

    printf("Soft Dirichlet advantages:\n");
    printf("  ✓ Simpler implementation\n");
    printf("  ✓ Single-knob tuning (ESS_max)\n");
    printf("  ✓ Potentially faster (depending on impl)\n");
    printf("\n");

    if (vi_avg_after < sd_avg_after && vi_avg_converge < sd_avg_converge)
    {
        printf("RECOMMENDATION: Use Online VI for better tracking + uncertainty.\n");
    }
    else if (sd_avg_after < vi_avg_after)
    {
        printf("RECOMMENDATION: Soft Dirichlet tracks better in this test.\n");
        printf("Consider tuning VI learning rate schedule.\n");
    }
    else
    {
        printf("RECOMMENDATION: Results are mixed. Consider use case:\n");
        printf("  - Need uncertainty? → Online VI\n");
        printf("  - Need simplicity?  → Soft Dirichlet\n");
    }

    return 0;
}