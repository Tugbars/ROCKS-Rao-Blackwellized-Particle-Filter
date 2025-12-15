/**
 * @file test_bocpd.c
 * @brief Comprehensive BOCPD test with synthetic changepoint data
 *
 * Tests:
 * 1. Detection accuracy (true positives vs ground truth)
 * 2. False alarm rate (spurious detections)
 * 3. Detection latency (ticks from true CP to detection)
 * 4. Delta detector calibration
 * 5. Power-law vs constant hazard comparison
 * 6. Performance benchmarks
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "bocpd.h" /* Include path set by CMake target_include_directories */

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/*═══════════════════════════════════════════════════════════════════════════
 * CONFIGURATION
 *═══════════════════════════════════════════════════════════════════════════*/

#define N_TICKS 10000
#define MAX_RUN_LENGTH 1024
#define N_SCENARIOS 6
#define DETECTION_WINDOW 10 /* Ticks after CP to count as "detected" */

/*═══════════════════════════════════════════════════════════════════════════
 * RNG (xoroshiro128+)
 *═══════════════════════════════════════════════════════════════════════════*/

static uint64_t rng_state[2];

static inline uint64_t rotl(const uint64_t x, int k)
{
    return (x << k) | (x >> (64 - k));
}

static inline uint64_t rng_next(void)
{
    const uint64_t s0 = rng_state[0];
    uint64_t s1 = rng_state[1];
    const uint64_t result = s0 + s1;
    s1 ^= s0;
    rng_state[0] = rotl(s0, 24) ^ s1 ^ (s1 << 16);
    rng_state[1] = rotl(s1, 37);
    return result;
}

static void rng_seed(uint64_t seed)
{
    rng_state[0] = seed;
    rng_state[1] = seed ^ 0x123456789ABCDEF0ULL;
    for (int i = 0; i < 20; i++)
        rng_next();
}

static double rng_uniform(void)
{
    return (rng_next() >> 11) * 0x1.0p-53;
}

static double rng_normal(void)
{
    double u1 = rng_uniform();
    double u2 = rng_uniform();
    if (u1 < 1e-10)
        u1 = 1e-10;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

/*═══════════════════════════════════════════════════════════════════════════
 * TIMING (Cross-platform)
 *═══════════════════════════════════════════════════════════════════════════*/

#ifdef _WIN32
#include <windows.h>
static inline double get_time_us(void)
{
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart * 1e6 / (double)freq.QuadPart;
}
#else
static inline double get_time_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1000.0;
}
#endif

/*═══════════════════════════════════════════════════════════════════════════
 * SYNTHETIC DATA GENERATION
 *
 * We generate data with known changepoints where the mean shifts.
 * This allows exact measurement of detection accuracy.
 *═══════════════════════════════════════════════════════════════════════════*/

typedef struct
{
    int tick;         /* Tick number of changepoint */
    double mu_before; /* Mean before CP */
    double mu_after;  /* Mean after CP */
    double sigma;     /* Noise std */
    const char *name; /* Scenario name */
} Changepoint;

typedef struct
{
    double *observations; /* The data */
    int *true_cp;         /* 1 if this tick is a changepoint, 0 otherwise */
    double *true_mean;    /* Ground truth mean at each tick */
    int n_ticks;
    int n_changepoints;
    Changepoint *cps; /* Changepoint definitions */
} SyntheticData;

/* Define test scenarios with various changepoint types */
static Changepoint TEST_CHANGEPOINTS[] = {
    /* Tick    μ_before  μ_after  σ     Name */
    {500, 0.0, 2.0, 1.0, "Mild shift up"},
    {1500, 2.0, 0.0, 1.0, "Return to baseline"},
    {2500, 0.0, 5.0, 1.0, "Large shift up"},
    {3000, 5.0, -2.0, 1.0, "Large shift down"},
    {4500, -2.0, 0.0, 1.5, "Noisy recovery"},
    {6000, 0.0, 1.0, 0.5, "Subtle shift (low noise)"},
    {7000, 1.0, 3.0, 2.0, "Moderate shift (high noise)"},
    {8500, 3.0, 0.0, 1.0, "Final return"},
};
#define N_CHANGEPOINTS (sizeof(TEST_CHANGEPOINTS) / sizeof(TEST_CHANGEPOINTS[0]))

static SyntheticData *generate_data(uint64_t seed)
{
    rng_seed(seed);

    SyntheticData *data = (SyntheticData *)malloc(sizeof(SyntheticData));
    data->observations = (double *)calloc(N_TICKS, sizeof(double));
    data->true_cp = (int *)calloc(N_TICKS, sizeof(int));
    data->true_mean = (double *)calloc(N_TICKS, sizeof(double));
    data->n_ticks = N_TICKS;
    data->n_changepoints = N_CHANGEPOINTS;
    data->cps = TEST_CHANGEPOINTS;

    /* Mark changepoints */
    for (int i = 0; i < (int)N_CHANGEPOINTS; i++)
    {
        int t = TEST_CHANGEPOINTS[i].tick;
        if (t < N_TICKS)
        {
            data->true_cp[t] = 1;
        }
    }

    /* Generate data */
    double current_mu = 0.0;
    double current_sigma = 1.0;
    int next_cp_idx = 0;

    for (int t = 0; t < N_TICKS; t++)
    {
        /* Check if we hit a changepoint */
        if (next_cp_idx < (int)N_CHANGEPOINTS &&
            t == TEST_CHANGEPOINTS[next_cp_idx].tick)
        {
            current_mu = TEST_CHANGEPOINTS[next_cp_idx].mu_after;
            current_sigma = TEST_CHANGEPOINTS[next_cp_idx].sigma;
            next_cp_idx++;
        }

        data->true_mean[t] = current_mu;
        data->observations[t] = current_mu + current_sigma * rng_normal();
    }

    return data;
}

static void free_data(SyntheticData *data)
{
    if (data)
    {
        free(data->observations);
        free(data->true_cp);
        free(data->true_mean);
        free(data);
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * DETECTION METRICS
 *═══════════════════════════════════════════════════════════════════════════*/

typedef struct
{
    int true_positives;       /* Detected within DETECTION_WINDOW of true CP */
    int false_positives;      /* Detection with no nearby true CP */
    int false_negatives;      /* True CP not detected within window */
    int total_detections;     /* Total times detector fired */
    int total_true_cps;       /* Total true changepoints */
    double avg_detection_lag; /* Average ticks from true CP to detection */
    double max_detection_lag; /* Max detection lag */

    /* Per-changepoint details */
    int detected[N_CHANGEPOINTS];
    int detection_lag[N_CHANGEPOINTS];

    /* Latency stats */
    double total_time_us;
    double max_latency_us;
    double *latencies;
    int n_latencies;
} DetectionMetrics;

static int compare_doubles(const void *a, const void *b)
{
    double da = *(const double *)a;
    double db = *(const double *)b;
    return (da > db) - (da < db);
}

static void compute_metrics(DetectionMetrics *m, int *detections,
                            SyntheticData *data, double *latencies)
{
    memset(m, 0, sizeof(*m));
    m->total_true_cps = data->n_changepoints;
    m->latencies = latencies;
    m->n_latencies = data->n_ticks;

    /* For each true changepoint, check if detected nearby */
    for (int i = 0; i < (int)N_CHANGEPOINTS; i++)
    {
        int cp_tick = TEST_CHANGEPOINTS[i].tick;
        if (cp_tick >= data->n_ticks)
            continue;

        int found = 0;
        int lag = -1;

        /* Look in window [cp_tick, cp_tick + DETECTION_WINDOW] */
        for (int t = cp_tick; t < cp_tick + DETECTION_WINDOW && t < data->n_ticks; t++)
        {
            if (detections[t])
            {
                found = 1;
                lag = t - cp_tick;
                break;
            }
        }

        m->detected[i] = found;
        m->detection_lag[i] = lag;

        if (found)
        {
            m->true_positives++;
            m->avg_detection_lag += lag;
            if (lag > m->max_detection_lag)
                m->max_detection_lag = lag;
        }
        else
        {
            m->false_negatives++;
        }
    }

    if (m->true_positives > 0)
    {
        m->avg_detection_lag /= m->true_positives;
    }

    /* Count total detections and false positives */
    for (int t = 0; t < data->n_ticks; t++)
    {
        if (detections[t])
        {
            m->total_detections++;

            /* Check if this is near any true CP */
            int near_true_cp = 0;
            for (int i = 0; i < (int)N_CHANGEPOINTS; i++)
            {
                int cp_tick = TEST_CHANGEPOINTS[i].tick;
                if (t >= cp_tick && t < cp_tick + DETECTION_WINDOW)
                {
                    near_true_cp = 1;
                    break;
                }
            }

            if (!near_true_cp)
            {
                m->false_positives++;
            }
        }
    }

    /* Compute latency stats */
    double sum = 0.0;
    m->max_latency_us = 0.0;
    for (int t = 0; t < data->n_ticks; t++)
    {
        sum += latencies[t];
        if (latencies[t] > m->max_latency_us)
        {
            m->max_latency_us = latencies[t];
        }
    }
    m->total_time_us = sum;
}

/*═══════════════════════════════════════════════════════════════════════════
 * RUN BOCPD TEST
 *═══════════════════════════════════════════════════════════════════════════*/

typedef struct
{
    const char *name;
    double hazard_lambda;   /* For constant hazard */
    double power_law_alpha; /* For power-law hazard (0 = use constant) */
    double z_threshold;     /* Delta detector threshold */
    double prior_kappa0;
    double prior_alpha0;
    double prior_beta0;
} BOCPDConfig;

static void run_bocpd_test(SyntheticData *data, BOCPDConfig *cfg,
                           int *detections, double *latencies,
                           DetectionMetrics *metrics)
{
    /* Initialize BOCPD */
    bocpd_t bocpd;
    bocpd_hazard_t hazard;
    bocpd_delta_detector_t delta;

    bocpd_prior_t prior = {
        .mu0 = 0.0,
        .kappa0 = cfg->prior_kappa0,
        .alpha0 = cfg->prior_alpha0,
        .beta0 = cfg->prior_beta0};

    int use_power_law = (cfg->power_law_alpha > 0.0);

    if (use_power_law)
    {
        bocpd_hazard_init_power_law(&hazard, cfg->power_law_alpha, MAX_RUN_LENGTH);
        bocpd_init_with_hazard(&bocpd, &hazard, prior);
    }
    else
    {
        bocpd_init(&bocpd, cfg->hazard_lambda, prior, MAX_RUN_LENGTH);
    }

    bocpd_delta_init(&delta, 50); /* 50 tick warmup */

    memset(detections, 0, data->n_ticks * sizeof(int));

    /* Run detection */
    for (int t = 0; t < data->n_ticks; t++)
    {
        double x = data->observations[t];

        double t0 = get_time_us();

        bocpd_step(&bocpd, x);

        /* Update delta detector */
        double delta_val = bocpd_delta_update(&delta, bocpd.r,
                                              bocpd.active_len, 0.995);

        /* Check for changepoint */
        double z = bocpd_delta_zscore(&delta, delta_val);
        if (z > cfg->z_threshold && delta.n_observations >= delta.warmup_period)
        {
            detections[t] = 1;
        }

        double t1 = get_time_us();
        latencies[t] = t1 - t0;
    }

    /* Cleanup */
    bocpd_free(&bocpd);
    if (use_power_law)
    {
        bocpd_hazard_free(&hazard);
    }

    /* Compute metrics */
    compute_metrics(metrics, detections, data, latencies);
}

/*═══════════════════════════════════════════════════════════════════════════
 * ALTERNATIVE: Simple p_changepoint threshold
 *═══════════════════════════════════════════════════════════════════════════*/

static void run_bocpd_pcp_test(SyntheticData *data, BOCPDConfig *cfg,
                               double pcp_threshold,
                               int *detections, double *latencies,
                               DetectionMetrics *metrics)
{
    bocpd_t bocpd;
    bocpd_hazard_t hazard;

    bocpd_prior_t prior = {
        .mu0 = 0.0,
        .kappa0 = cfg->prior_kappa0,
        .alpha0 = cfg->prior_alpha0,
        .beta0 = cfg->prior_beta0};

    int use_power_law = (cfg->power_law_alpha > 0.0);

    if (use_power_law)
    {
        bocpd_hazard_init_power_law(&hazard, cfg->power_law_alpha, MAX_RUN_LENGTH);
        bocpd_init_with_hazard(&bocpd, &hazard, prior);
    }
    else
    {
        bocpd_init(&bocpd, cfg->hazard_lambda, prior, MAX_RUN_LENGTH);
    }

    memset(detections, 0, data->n_ticks * sizeof(int));

    for (int t = 0; t < data->n_ticks; t++)
    {
        double x = data->observations[t];

        double t0 = get_time_us();
        bocpd_step(&bocpd, x);
        double t1 = get_time_us();

        latencies[t] = t1 - t0;

        /* Simple threshold on p_changepoint */
        if (bocpd.p_changepoint > pcp_threshold)
        {
            detections[t] = 1;
        }
    }

    bocpd_free(&bocpd);
    if (use_power_law)
    {
        bocpd_hazard_free(&hazard);
    }

    compute_metrics(metrics, detections, data, latencies);
}

/*═══════════════════════════════════════════════════════════════════════════
 * PRINT RESULTS
 *═══════════════════════════════════════════════════════════════════════════*/

static void print_metrics(const char *name, DetectionMetrics *m)
{
    double precision = (m->total_detections > 0) ? (double)m->true_positives / m->total_detections : 0.0;
    double recall = (m->total_true_cps > 0) ? (double)m->true_positives / m->total_true_cps : 0.0;
    double f1 = (precision + recall > 0) ? 2.0 * precision * recall / (precision + recall) : 0.0;

    printf("\n%s\n", name);
    printf("────────────────────────────────────────────────────────────────\n");
    printf("  True Positives:     %d / %d (%.1f%% recall)\n",
           m->true_positives, m->total_true_cps,
           100.0 * recall);
    printf("  False Positives:    %d\n", m->false_positives);
    printf("  False Negatives:    %d\n", m->false_negatives);
    printf("  Total Detections:   %d\n", m->total_detections);
    printf("  Precision:          %.1f%%\n", 100.0 * precision);
    printf("  F1 Score:           %.3f\n", f1);
    printf("  Avg Detection Lag:  %.1f ticks\n", m->avg_detection_lag);
    printf("  Max Detection Lag:  %.0f ticks\n", m->max_detection_lag);

    /* Latency stats */
    qsort(m->latencies, m->n_latencies, sizeof(double), compare_doubles);
    double median = m->latencies[m->n_latencies / 2];
    double p99 = m->latencies[(int)(m->n_latencies * 0.99)];

    printf("  Median Latency:     %.2f us\n", median);
    printf("  P99 Latency:        %.2f us\n", p99);
    printf("  Max Latency:        %.2f us\n", m->max_latency_us);

    /* Per-changepoint details */
    printf("\n  Per-Changepoint Detection:\n");
    for (int i = 0; i < (int)N_CHANGEPOINTS; i++)
    {
        printf("    [%4d] %-25s : %s",
               TEST_CHANGEPOINTS[i].tick,
               TEST_CHANGEPOINTS[i].name,
               m->detected[i] ? "✓" : "✗");
        if (m->detected[i])
        {
            printf(" (lag=%d)", m->detection_lag[i]);
        }
        printf("\n");
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * MAIN
 *═══════════════════════════════════════════════════════════════════════════*/

int main(int argc, char **argv)
{
    uint64_t seed = 42;
    if (argc > 1)
    {
        seed = strtoull(argv[1], NULL, 10);
    }

    printf("══════════════════════════════════════════════════════════════════\n");
    printf("  BOCPD Comprehensive Test (seed=%llu)\n", (unsigned long long)seed);
    printf("══════════════════════════════════════════════════════════════════\n");
    printf("\nTest Data: %d ticks, %d changepoints\n", N_TICKS, (int)N_CHANGEPOINTS);

    /* Generate synthetic data */
    SyntheticData *data = generate_data(seed);

    printf("\nChangepoints:\n");
    for (int i = 0; i < (int)N_CHANGEPOINTS; i++)
    {
        printf("  [%4d] μ: %.1f → %.1f (σ=%.1f) : %s\n",
               TEST_CHANGEPOINTS[i].tick,
               TEST_CHANGEPOINTS[i].mu_before,
               TEST_CHANGEPOINTS[i].mu_after,
               TEST_CHANGEPOINTS[i].sigma,
               TEST_CHANGEPOINTS[i].name);
    }

    /* Allocate buffers */
    int *detections = (int *)calloc(N_TICKS, sizeof(int));
    double *latencies = (double *)calloc(N_TICKS, sizeof(double));
    DetectionMetrics metrics;

    /*═══════════════════════════════════════════════════════════════════════
     * TEST 1: Constant Hazard + Delta Detector
     *═══════════════════════════════════════════════════════════════════════*/

    BOCPDConfig cfg1 = {
        .name = "Constant Hazard (λ=100) + Delta Detector (z=3.0)",
        .hazard_lambda = 100.0,
        .power_law_alpha = 0.0,
        .z_threshold = 3.0,
        .prior_kappa0 = 1.0,
        .prior_alpha0 = 1.0,
        .prior_beta0 = 1.0};

    run_bocpd_test(data, &cfg1, detections, latencies, &metrics);
    print_metrics(cfg1.name, &metrics);

    /*═══════════════════════════════════════════════════════════════════════
     * TEST 2: Power-Law Hazard + Delta Detector
     *═══════════════════════════════════════════════════════════════════════*/

    BOCPDConfig cfg2 = {
        .name = "Power-Law Hazard (α=0.8) + Delta Detector (z=3.0)",
        .hazard_lambda = 0.0,
        .power_law_alpha = 0.8,
        .z_threshold = 3.0,
        .prior_kappa0 = 1.0,
        .prior_alpha0 = 1.0,
        .prior_beta0 = 1.0};

    run_bocpd_test(data, &cfg2, detections, latencies, &metrics);
    print_metrics(cfg2.name, &metrics);

    /*═══════════════════════════════════════════════════════════════════════
     * TEST 3: Simple p_changepoint threshold (baseline)
     *═══════════════════════════════════════════════════════════════════════*/

    BOCPDConfig cfg3 = {
        .name = "Constant Hazard + p_changepoint > 0.5",
        .hazard_lambda = 100.0,
        .power_law_alpha = 0.0,
        .z_threshold = 0.0, /* Not used */
        .prior_kappa0 = 1.0,
        .prior_alpha0 = 1.0,
        .prior_beta0 = 1.0};

    run_bocpd_pcp_test(data, &cfg3, 0.5, detections, latencies, &metrics);
    print_metrics(cfg3.name, &metrics);

    /*═══════════════════════════════════════════════════════════════════════
     * TEST 4: Aggressive z-threshold
     *═══════════════════════════════════════════════════════════════════════*/

    BOCPDConfig cfg4 = {
        .name = "Power-Law Hazard + Delta Detector (z=2.0, aggressive)",
        .hazard_lambda = 0.0,
        .power_law_alpha = 0.8,
        .z_threshold = 2.0,
        .prior_kappa0 = 1.0,
        .prior_alpha0 = 1.0,
        .prior_beta0 = 1.0};

    run_bocpd_test(data, &cfg4, detections, latencies, &metrics);
    print_metrics(cfg4.name, &metrics);

    /*═══════════════════════════════════════════════════════════════════════
     * TEST 5: Conservative z-threshold
     *═══════════════════════════════════════════════════════════════════════*/

    BOCPDConfig cfg5 = {
        .name = "Power-Law Hazard + Delta Detector (z=4.0, conservative)",
        .hazard_lambda = 0.0,
        .power_law_alpha = 0.8,
        .z_threshold = 4.0,
        .prior_kappa0 = 1.0,
        .prior_alpha0 = 1.0,
        .prior_beta0 = 1.0};

    run_bocpd_test(data, &cfg5, detections, latencies, &metrics);
    print_metrics(cfg5.name, &metrics);

    /*═══════════════════════════════════════════════════════════════════════
     * TEST 6: Different prior (more confident)
     *═══════════════════════════════════════════════════════════════════════*/

    BOCPDConfig cfg6 = {
        .name = "Power-Law + Confident Prior (κ=10, α=5)",
        .hazard_lambda = 0.0,
        .power_law_alpha = 0.8,
        .z_threshold = 3.0,
        .prior_kappa0 = 10.0,
        .prior_alpha0 = 5.0,
        .prior_beta0 = 5.0};

    run_bocpd_test(data, &cfg6, detections, latencies, &metrics);
    print_metrics(cfg6.name, &metrics);

    /*═══════════════════════════════════════════════════════════════════════
     * SUMMARY
     *═══════════════════════════════════════════════════════════════════════*/

    printf("\n══════════════════════════════════════════════════════════════════\n");
    printf("  SUMMARY\n");
    printf("══════════════════════════════════════════════════════════════════\n");
    printf("\nRecommended configuration for MMPF integration:\n");
    printf("  - Power-law hazard with α=0.8 (heavy-tailed regime durations)\n");
    printf("  - Delta detector with z_threshold=3.0 (balance precision/recall)\n");
    printf("  - Weak prior (κ=1, α=1, β=1) for fast adaptation\n");
    printf("\nExpected behavior with MMPF:\n");
    printf("  - Normal operation: BOCPD runs, delta stays low, MMPF uses high inertia\n");
    printf("  - Changepoint detected: delta spikes, mmpf_inject_shock() called\n");
    printf("  - MMPF re-evaluates hypotheses with uniform transitions\n");
    printf("  - mmpf_restore_from_shock() returns to normal operation\n");

    /* Write CSV for visualization */
    FILE *csv = fopen("bocpd_test.csv", "w");
    if (csv)
    {
        fprintf(csv, "tick,observation,true_mean,is_cp,detection\n");

        /* Re-run best config for CSV */
        run_bocpd_test(data, &cfg2, detections, latencies, &metrics);

        for (int t = 0; t < data->n_ticks; t++)
        {
            fprintf(csv, "%d,%.6f,%.6f,%d,%d\n",
                    t, data->observations[t], data->true_mean[t],
                    data->true_cp[t], detections[t]);
        }
        fclose(csv);
        printf("\nCSV written: bocpd_test.csv\n");
    }

    /* Cleanup */
    free(detections);
    free(latencies);
    free_data(data);

    return 0;
}