/*=============================================================================
 * PGAS Sliding Window Test
 *
 * Uses the same challenging synthetic dataset from MMPF comparison test.
 * Runs PGAS with sliding window and prints learned Π after each slide.
 *
 * Scenarios (8000 ticks total):
 *   1. Extended Calm (0-1499)        - CALM dominant, 2 outliers
 *   2. Slow Trend (1500-2499)        - CALM → TREND transition
 *   3. Sudden Crisis (2500-2999)     - TREND → CRISIS, 5 fat-tail outliers
 *   4. Crisis Persistence (3000-3999)- CRISIS sustained, 3 extreme outliers
 *   5. Recovery (4000-5199)          - CRISIS → TREND → CALM
 *   6. Flash Crash (5200-5699)       - CALM → CRISIS → CALM (60 ticks)
 *   7. Choppy (5700-7999)            - Mixed regime switching
 *
 * BUILD:
 *   gcc -O3 -march=native -fopenmp test_pgas_sliding.c pgas_sliding.c pgas_mkl.c \
 *       -I${MKLROOT}/include -L${MKLROOT}/lib/intel64 \
 *       -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm \
 *       -o test_pgas_sliding
 *
 *===========================================================================*/

#include "pgas_sliding.h"
#include "pgas_mkl.h"
#include "mkl_tuning.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/*─────────────────────────────────────────────────────────────────────────────
 * TIMING
 *───────────────────────────────────────────────────────────────────────────*/

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
static double g_timer_freq = 0.0;
static void init_timer(void)
{
    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);
    g_timer_freq = (double)freq.QuadPart / 1e6;
}
static inline double get_time_us(void)
{
    LARGE_INTEGER counter;
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / g_timer_freq;
}
#else
#include <sys/time.h>
static void init_timer(void) {}
static inline double get_time_us(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1e6 + tv.tv_usec;
}
#endif

/*─────────────────────────────────────────────────────────────────────────────
 * PCG32 RNG
 *───────────────────────────────────────────────────────────────────────────*/

typedef struct
{
    uint64_t state;
    uint64_t inc;
} pcg32_t;

static uint32_t pcg32_random(pcg32_t *rng)
{
    uint64_t oldstate = rng->state;
    rng->state = oldstate * 6364136223846793005ULL + rng->inc;
    uint32_t xorshifted = (uint32_t)(((oldstate >> 18u) ^ oldstate) >> 27u);
    uint32_t rot = (uint32_t)(oldstate >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

static double pcg32_double(pcg32_t *rng)
{
    return (double)pcg32_random(rng) / 4294967296.0;
}

static double pcg32_gaussian(pcg32_t *rng)
{
    double u1 = pcg32_double(rng);
    double u2 = pcg32_double(rng);
    if (u1 < 1e-10)
        u1 = 1e-10;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979 * u2);
}

/*─────────────────────────────────────────────────────────────────────────────
 * HYPOTHESIS PARAMETERS (Ground Truth)
 *───────────────────────────────────────────────────────────────────────────*/

typedef enum
{
    HYPO_CALM = 0,
    HYPO_TREND = 1,
    HYPO_CRISIS = 2,
    N_HYPOTHESES = 3
} Hypothesis;

static const char *hypothesis_names[] = {"CALM", "TREND", "CRISIS"};

typedef struct
{
    double mu_vol;     /* Long-run mean of log-vol */
    double phi;        /* Persistence */
    double sigma_eta;  /* Vol-of-vol */
    double vol_approx; /* Approximate realized vol */
} HypothesisParams;

static const HypothesisParams TRUE_PARAMS[N_HYPOTHESES] = {
    /* CALM: Low vol, high persistence, smooth */
    {.mu_vol = -5.0, .phi = 0.995, .sigma_eta = 0.08, .vol_approx = 0.007},

    /* TREND: Medium vol, medium persistence */
    {.mu_vol = -3.5, .phi = 0.95, .sigma_eta = 0.20, .vol_approx = 0.030},

    /* CRISIS: High vol, fast mean reversion, explosive */
    {.mu_vol = -1.5, .phi = 0.85, .sigma_eta = 0.50, .vol_approx = 0.220}};

/*─────────────────────────────────────────────────────────────────────────────
 * SYNTHETIC DATA
 *───────────────────────────────────────────────────────────────────────────*/

typedef struct
{
    double *returns;
    double *observations; /* y_t = log(r_t²) for PGAS */
    double *true_log_vol;
    int *true_hypothesis;
    int *is_outlier;
    int n_ticks;
    int scenario_starts[10];
    const char *scenario_names[10];
    int n_scenarios;
    int n_outliers_injected;
} SyntheticData;

/* Inject outlier at tick t */
static void inject_outlier(SyntheticData *data, int t, double target_sigma,
                           double vol, pcg32_t *rng)
{
    double sign = (pcg32_double(rng) < 0.5) ? -1.0 : 1.0;
    data->returns[t] = sign * target_sigma * vol;
    data->is_outlier[t] = 1;
    data->n_outliers_injected++;
}

static SyntheticData *generate_test_data(int seed)
{
    SyntheticData *data = (SyntheticData *)calloc(1, sizeof(SyntheticData));

    int n = 8000;
    data->n_ticks = n;
    data->returns = (double *)malloc(n * sizeof(double));
    data->observations = (double *)malloc(n * sizeof(double));
    data->true_log_vol = (double *)malloc(n * sizeof(double));
    data->true_hypothesis = (int *)malloc(n * sizeof(int));
    data->is_outlier = (int *)calloc(n, sizeof(int));

    pcg32_t rng = {seed * 12345ULL + 1, seed * 67890ULL | 1};

    /* Start in CALM */
    double log_vol = TRUE_PARAMS[HYPO_CALM].mu_vol;
    int t = 0;

/* Helper macro for state evolution */
#define EVOLVE_STATE(H)                                                                       \
    do                                                                                        \
    {                                                                                         \
        const HypothesisParams *p = &TRUE_PARAMS[H];                                          \
        double theta = 1.0 - p->phi;                                                          \
        log_vol = p->phi * log_vol + theta * p->mu_vol + p->sigma_eta * pcg32_gaussian(&rng); \
        double vol = exp(log_vol);                                                            \
        double ret = vol * pcg32_gaussian(&rng);                                              \
        data->returns[t] = ret;                                                               \
        data->true_log_vol[t] = log_vol;                                                      \
        data->true_hypothesis[t] = (H);                                                       \
    } while (0)

    /* Scenario 1: Extended Calm (0-1499) */
    data->scenario_starts[0] = 0;
    data->scenario_names[0] = "Extended Calm";
    data->n_scenarios = 1;

    for (; t < 1500; t++)
    {
        EVOLVE_STATE(HYPO_CALM);
    }
    inject_outlier(data, 500, 6.0, exp(data->true_log_vol[500]), &rng);
    inject_outlier(data, 1200, 8.0, exp(data->true_log_vol[1200]), &rng);

    /* Scenario 2: Slow Trend (1500-2499) */
    data->scenario_starts[1] = 1500;
    data->scenario_names[1] = "Slow Trend";
    data->n_scenarios = 2;

    for (; t < 2500; t++)
    {
        Hypothesis h = (t < 1800) ? HYPO_CALM : HYPO_TREND;
        EVOLVE_STATE(h);
    }

    /* Scenario 3: Sudden Crisis (2500-2999) */
    data->scenario_starts[2] = 2500;
    data->scenario_names[2] = "Sudden Crisis";
    data->n_scenarios = 3;

    for (; t < 3000; t++)
    {
        EVOLVE_STATE(HYPO_CRISIS);
    }
    inject_outlier(data, 2510, 8.0, exp(data->true_log_vol[2510]), &rng);
    inject_outlier(data, 2530, 10.0, exp(data->true_log_vol[2530]), &rng);
    inject_outlier(data, 2560, 12.0, exp(data->true_log_vol[2560]), &rng);
    inject_outlier(data, 2650, 9.0, exp(data->true_log_vol[2650]), &rng);
    inject_outlier(data, 2800, 11.0, exp(data->true_log_vol[2800]), &rng);

    /* Scenario 4: Crisis Persistence (3000-3999) */
    data->scenario_starts[3] = 3000;
    data->scenario_names[3] = "Crisis Persist";
    data->n_scenarios = 4;

    for (; t < 4000; t++)
    {
        EVOLVE_STATE(HYPO_CRISIS);
    }
    inject_outlier(data, 3200, 10.0, exp(data->true_log_vol[3200]), &rng);
    inject_outlier(data, 3500, 15.0, exp(data->true_log_vol[3500]), &rng);
    inject_outlier(data, 3800, 12.0, exp(data->true_log_vol[3800]), &rng);

    /* Scenario 5: Recovery (4000-5199) */
    data->scenario_starts[4] = 4000;
    data->scenario_names[4] = "Recovery";
    data->n_scenarios = 5;

    for (; t < 5200; t++)
    {
        Hypothesis h;
        if (t < 4400)
            h = HYPO_CRISIS;
        else if (t < 4800)
            h = HYPO_TREND;
        else
            h = HYPO_CALM;
        EVOLVE_STATE(h);
    }

    /* Scenario 6: Flash Crash (5200-5699) */
    data->scenario_starts[5] = 5200;
    data->scenario_names[5] = "Flash Crash";
    data->n_scenarios = 6;

    for (; t < 5700; t++)
    {
        Hypothesis h;
        if (t >= 5350 && t < 5410)
            h = HYPO_CRISIS;
        else
            h = HYPO_CALM;
        EVOLVE_STATE(h);
    }
    inject_outlier(data, 5380, 12.0, exp(data->true_log_vol[5380]), &rng);

    /* Scenario 7: Choppy (5700-7999) */
    data->scenario_starts[6] = 5700;
    data->scenario_names[6] = "Choppy";
    data->n_scenarios = 7;

    Hypothesis current_h = HYPO_TREND;
    int next_switch = 5700 + 80 + (int)(pcg32_double(&rng) * 120);

    for (; t < 8000; t++)
    {
        if (t >= next_switch)
        {
            int delta = (pcg32_double(&rng) < 0.5) ? -1 : 1;
            current_h = (Hypothesis)((current_h + delta + N_HYPOTHESES) % N_HYPOTHESES);
            next_switch = t + 80 + (int)(pcg32_double(&rng) * 150);
        }
        EVOLVE_STATE(current_h);
    }

#undef EVOLVE_STATE

    /* Convert returns to PGAS observations: y_t = log(r_t²) */
    for (int i = 0; i < n; i++)
    {
        double r = data->returns[i];
        /* Clamp to avoid log(0) */
        double r2 = r * r;
        if (r2 < 1e-20)
            r2 = 1e-20;
        data->observations[i] = log(r2);
    }

    return data;
}

static void free_synthetic_data(SyntheticData *data)
{
    if (!data)
        return;
    free(data->returns);
    free(data->observations);
    free(data->true_log_vol);
    free(data->true_hypothesis);
    free(data->is_outlier);
    free(data);
}

/*─────────────────────────────────────────────────────────────────────────────
 * GET CURRENT SCENARIO
 *───────────────────────────────────────────────────────────────────────────*/

static const char *get_scenario_at_tick(SyntheticData *data, int tick)
{
    for (int s = data->n_scenarios - 1; s >= 0; s--)
    {
        if (tick >= data->scenario_starts[s])
        {
            return data->scenario_names[s];
        }
    }
    return "Unknown";
}

/*─────────────────────────────────────────────────────────────────────────────
 * COMPUTE GROUND TRUTH Π FOR WINDOW
 *
 * Count actual transitions in ground truth data within window
 *───────────────────────────────────────────────────────────────────────────*/

static void compute_ground_truth_pi(SyntheticData *data, int window_start,
                                    int window_end, int K, float *pi_out)
{
    /* Initialize counts */
    int counts[PGAS_MKL_MAX_K * PGAS_MKL_MAX_K] = {0};

    for (int t = window_start + 1; t < window_end; t++)
    {
        int from = data->true_hypothesis[t - 1];
        int to = data->true_hypothesis[t];
        if (from < K && to < K)
        {
            counts[from * K + to]++;
        }
    }

    /* Normalize to probabilities */
    for (int i = 0; i < K; i++)
    {
        int row_sum = 0;
        for (int j = 0; j < K; j++)
        {
            row_sum += counts[i * K + j];
        }
        for (int j = 0; j < K; j++)
        {
            if (row_sum > 0)
            {
                pi_out[i * K + j] = (float)counts[i * K + j] / row_sum;
            }
            else
            {
                pi_out[i * K + j] = 1.0f / K; /* Uniform if no data */
            }
        }
    }
}

/*─────────────────────────────────────────────────────────────────────────────
 * VALIDATE Π
 *───────────────────────────────────────────────────────────────────────────*/

static int validate_pi(const float *pi, int K)
{
    for (int i = 0; i < K; i++)
    {
        float sum = 0.0f;
        for (int j = 0; j < K; j++)
        {
            float p = pi[i * K + j];
            if (!isfinite(p) || p < 0.0f || p > 1.0f)
            {
                return 0;
            }
            sum += p;
        }
        if (fabsf(sum - 1.0f) > 0.01f)
        {
            return 0;
        }
    }
    return 1;
}

/*─────────────────────────────────────────────────────────────────────────────
 * PRINT Π MATRIX
 *───────────────────────────────────────────────────────────────────────────*/

static void print_pi_matrix(const char *label, const float *pi, int K)
{
    printf("  %s:\n", label);
    for (int i = 0; i < K; i++)
    {
        printf("    [");
        for (int j = 0; j < K; j++)
        {
            printf(" %6.3f", pi[i * K + j]);
        }
        printf(" ]\n");
    }
}

/*─────────────────────────────────────────────────────────────────────────────
 * COMPUTE Π DISTANCE (Frobenius norm)
 *───────────────────────────────────────────────────────────────────────────*/

static float pi_distance(const float *pi1, const float *pi2, int K)
{
    float sum = 0.0f;
    for (int i = 0; i < K * K; i++)
    {
        float d = pi1[i] - pi2[i];
        sum += d * d;
    }
    return sqrtf(sum);
}

/*─────────────────────────────────────────────────────────────────────────────
 * MAIN
 *───────────────────────────────────────────────────────────────────────────*/

int main(int argc, char **argv)
{
    int seed = 42;
    if (argc > 1)
        seed = atoi(argv[1]);

    init_timer();

    /* Initialize MKL tuning: P-cores only, verbose=1 */
    mkl_tuning_init(8, 1); /* 8 P-cores, verbose */

    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  PGAS Sliding Window Test\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  Seed: %d\n\n", seed);

    /* Generate data */
    printf("Generating synthetic data...\n");
    SyntheticData *data = generate_test_data(seed);
    printf("  Ticks: %d\n", data->n_ticks);
    printf("  Scenarios: %d\n", data->n_scenarios);
    printf("  Outliers: %d\n\n", data->n_outliers_injected);

    /* PGAS Configuration */
    const int WINDOW_SIZE = 500; /* Smaller for more frequent updates */
    const int SLIDE_STEP = 100;  /* Slide by 100 ticks */
    const int N_PARTICLES = 128; /* Moderate particle count */
    const int K = 3;             /* Match 3 hypotheses */
    const int N_SWEEPS = 3;      /* Gibbs sweeps per iteration */

    printf("PGAS Configuration:\n");
    printf("  Window size: %d\n", WINDOW_SIZE);
    printf("  Slide step:  %d\n", SLIDE_STEP);
    printf("  Particles:   %d\n", N_PARTICLES);
    printf("  Regimes:     %d\n", K);
    printf("  Sweeps:      %d\n\n", N_SWEEPS);

    /* Allocate PGAS sliding state */
    PGASSlidingState *sliding = pgas_sliding_alloc(WINDOW_SIZE, SLIDE_STEP,
                                                   N_PARTICLES, K, seed);
    if (!sliding)
    {
        fprintf(stderr, "Failed to allocate PGAS sliding state\n");
        free_synthetic_data(data);
        return 1;
    }

    /* Set model parameters (matching ground truth hypotheses) */
    double trans_init[9] = {
        0.95, 0.04, 0.01,
        0.03, 0.94, 0.03,
        0.01, 0.04, 0.95};
    double mu_vol[3] = {-5.0, -3.5, -1.5};    /* CALM, TREND, CRISIS */
    double sigma_vol[3] = {0.08, 0.20, 0.50}; /* Per-regime AR noise */
    double phi = 0.95;

    pgas_sliding_set_model(sliding, trans_init, mu_vol, sigma_vol, phi);
    pgas_sliding_set_prior(sliding, 1.0f, 50.0f); /* Dirichlet prior */
    pgas_sliding_set_recency(sliding, 0.002f);    /* Recency weighting */

    printf("Model Parameters:\n");
    printf("  μ_vol:     [%.1f, %.1f, %.1f]\n", mu_vol[0], mu_vol[1], mu_vol[2]);
    printf("  σ_vol:     [%.2f, %.2f, %.2f]\n", sigma_vol[0], sigma_vol[1], sigma_vol[2]);
    printf("  φ:         %.2f\n", phi);
    printf("  Sticky κ:  50.0\n");
    printf("  Recency λ: 0.002\n\n");

    /* Statistics */
    int windows_processed = 0;
    double total_time_ms = 0.0;
    float total_pi_error = 0.0f;
    int last_print_tick = -1000; /* Force first print */
    const int PRINT_INTERVAL = 1000;

    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  Running PGAS (printing every %d ticks)...\n", PRINT_INTERVAL);
    printf("═══════════════════════════════════════════════════════════════\n\n");

    for (int t = 0; t < data->n_ticks; t++)
    {
        /* Push observation */
        float obs = (float)data->observations[t];
        bool pushed = pgas_sliding_push(sliding, obs, t);

        if (!pushed)
        {
            fprintf(stderr, "WARNING: Failed to push observation at tick %d\n", t);
        }

        /* Check if window is ready */
        if (pgas_sliding_window_ready(sliding))
        {
            double t_start = get_time_us();

            /* Extract window */
            pgas_sliding_extract_window(sliding);

            /* Run Gibbs sweeps */
            float acceptance = pgas_sliding_run_sweeps(sliding, N_SWEEPS);

            /* Get window bounds for ground truth comparison */
            int window_end = t + 1;
            int window_start = window_end - WINDOW_SIZE;
            if (window_start < 0)
                window_start = 0;

            /* Get learned Π */
            float pi_learned[PGAS_MKL_MAX_K * PGAS_MKL_MAX_K];
            pgas_sliding_get_pi(sliding, pi_learned);

            /* Compute ground truth Π for this window */
            float pi_truth[PGAS_MKL_MAX_K * PGAS_MKL_MAX_K];
            compute_ground_truth_pi(data, window_start, window_end, K, pi_truth);

            /* Slide window */
            pgas_sliding_slide(sliding);

            double t_end = get_time_us();
            double iter_time_ms = (t_end - t_start) / 1000.0;
            total_time_ms += iter_time_ms;

            windows_processed++;

            /* Validate and compute error */
            int valid = validate_pi(pi_learned, K);
            float pi_error = pi_distance(pi_learned, pi_truth, K);
            total_pi_error += pi_error;

            /* Get scenario info */
            const char *scenario = get_scenario_at_tick(data, window_end - 1);

            /* Count regime distribution in window */
            int regime_counts[3] = {0};
            for (int i = window_start; i < window_end; i++)
            {
                if (data->true_hypothesis[i] < 3)
                {
                    regime_counts[data->true_hypothesis[i]]++;
                }
            }

            /* Only print every PRINT_INTERVAL ticks */
            if (window_end - last_print_tick >= PRINT_INTERVAL)
            {
                last_print_tick = window_end;

                /* Print results */
                printf("┌─────────────────────────────────────────────────────────────┐\n");
                printf("│ Window %3d: ticks [%5d - %5d]  Scenario: %-14s │\n",
                       windows_processed, window_start, window_end - 1, scenario);
                printf("├─────────────────────────────────────────────────────────────┤\n");
                printf("│ Ground Truth Distribution: CALM=%d TREND=%d CRISIS=%d       \n",
                       regime_counts[0], regime_counts[1], regime_counts[2]);
                printf("├─────────────────────────────────────────────────────────────┤\n");

                print_pi_matrix("Learned Π", pi_learned, K);
                printf("│                                                             │\n");
                print_pi_matrix("Ground Truth Π", pi_truth, K);

                printf("├─────────────────────────────────────────────────────────────┤\n");
                printf("│ Quality Metrics:                                            │\n");
                printf("│   Valid:        %s                                          \n",
                       valid ? "YES ✓" : "NO ✗");
                printf("│   Π Error:      %.4f (Frobenius norm)                       \n", pi_error);
                printf("│   Acceptance:   %.1f%%                                       \n",
                       acceptance * 100.0f);
                printf("│   Time:         %.2f ms                                     \n", iter_time_ms);
                printf("│   Windows done: %d                                          \n",
                       sliding->windows_completed);
                printf("└─────────────────────────────────────────────────────────────┘\n\n");
            }
        }
    }

    /* Final summary */
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  SUMMARY\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  Windows processed: %d\n", windows_processed);
    printf("  Total time:        %.2f ms\n", total_time_ms);
    printf("  Avg time/window:   %.2f ms\n",
           windows_processed > 0 ? total_time_ms / windows_processed : 0.0);
    printf("  Avg Π error:       %.4f\n",
           windows_processed > 0 ? total_pi_error / windows_processed : 0.0);
    printf("═══════════════════════════════════════════════════════════════\n");

    /* Print final sliding state diagnostics */
    printf("\n");
    pgas_sliding_print_diagnostics(sliding);

    /* Cleanup */
    pgas_sliding_free(sliding);
    free_synthetic_data(data);

    return 0;
}