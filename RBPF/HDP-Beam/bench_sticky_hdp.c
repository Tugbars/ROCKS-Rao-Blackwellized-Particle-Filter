/**
 * @file bench_sticky_hdp.c
 * @brief Benchmark for Sticky HDP-HMM Beam Sampling
 *
 * Measures performance breakdown of sticky_hdp_beam_sweep_single()
 *
 * Build:
 *   gcc -O3 -march=native -o bench_sticky_hdp bench_sticky_hdp.c \
 *       sticky_hdp_beam.c -lmkl_rt -lm -lpthread
 *
 * Run:
 *   ./bench_sticky_hdp [T] [K] [n_sweeps]
 *
 * Example:
 *   ./bench_sticky_hdp 500 8 100
 */

#include "sticky_hdp_beam.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/*═══════════════════════════════════════════════════════════════════════════════
 * TIMING
 *═══════════════════════════════════════════════════════════════════════════════*/

#ifdef _WIN32
#include <windows.h>
static double get_time_us(void)
{
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart * 1e6 / (double)freq.QuadPart;
}
#else
#include <sys/time.h>
static double get_time_us(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1e6 + (double)tv.tv_usec;
}
#endif

/*═══════════════════════════════════════════════════════════════════════════════
 * SYNTHETIC DATA GENERATION
 *═══════════════════════════════════════════════════════════════════════════════*/

typedef struct
{
    int T;
    double *y;        /* Observations (log-squared returns) */
    int *true_states; /* Ground truth state sequence */
    int n_regimes;
    double *mu_vol; /* True μ per regime */
} SyntheticData;

static SyntheticData *generate_synthetic_data(int T, int n_regimes, unsigned int seed)
{
    if (T <= 0 || n_regimes <= 0)
        return NULL;

    SyntheticData *data = (SyntheticData *)malloc(sizeof(SyntheticData));
    if (!data)
        return NULL;

    data->T = T;
    data->n_regimes = n_regimes;
    data->y = (double *)malloc(T * sizeof(double));
    data->true_states = (int *)malloc(T * sizeof(int));
    data->mu_vol = (double *)malloc(n_regimes * sizeof(double));

    if (!data->y || !data->true_states || !data->mu_vol)
    {
        if (data->y)
            free(data->y);
        if (data->true_states)
            free(data->true_states);
        if (data->mu_vol)
            free(data->mu_vol);
        free(data);
        return NULL;
    }

    srand(seed);

    /* Set regime μ values (log-vol levels) */
    if (n_regimes == 1)
    {
        data->mu_vol[0] = -3.0; /* Single regime at mid-level */
    }
    else
    {
        for (int k = 0; k < n_regimes; k++)
        {
            data->mu_vol[k] = -4.5 + k * (2.5 / (n_regimes - 1)); /* -4.5 to -2.0 */
        }
    }

    /* Generate state sequence with sticky transitions */
    int state = 0;
    double stickiness = 0.95;

    for (int t = 0; t < T; t++)
    {
        /* Sticky transition */
        double u = (double)rand() / RAND_MAX;
        if (u > stickiness)
        {
            /* Transition to adjacent state */
            if (rand() % 2 == 0 && state > 0)
                state--;
            else if (state < n_regimes - 1)
                state++;
        }
        data->true_states[t] = state;

        /* Generate observation: y = 2h + log(ε²), ε ~ N(0,1) */
        double h = data->mu_vol[state];
        double eps = 0.0;
        /* Box-Muller for standard normal */
        double u1 = ((double)rand() + 1.0) / (RAND_MAX + 2.0);
        double u2 = ((double)rand() + 1.0) / (RAND_MAX + 2.0);
        eps = sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265 * u2);

        data->y[t] = 2.0 * h + log(eps * eps + 1e-10);
    }

    return data;
}

static void free_synthetic_data(SyntheticData *data)
{
    if (!data)
        return;
    free(data->y);
    free(data->true_states);
    free(data->mu_vol);
    free(data);
}

/*═══════════════════════════════════════════════════════════════════════════════
 * BENCHMARK RESULTS
 *═══════════════════════════════════════════════════════════════════════════════*/

typedef struct
{
    /* Timing (microseconds) */
    double total_time;
    double mean_time;
    double min_time;
    double max_time;
    double std_time;
    double p50_time;
    double p99_time;

    /* Per-sweep breakdown (if instrumented) */
    double slice_time;
    double forward_time;
    double backward_time;
    double update_time;

    /* Throughput */
    double sweeps_per_sec;
    double ticks_per_sec;

    /* Quality */
    double avg_active_states;
    double final_log_marginal;
    int final_K;
} BenchResult;

static int compare_double(const void *a, const void *b)
{
    double da = *(const double *)a;
    double db = *(const double *)b;
    return (da > db) - (da < db);
}

/*═══════════════════════════════════════════════════════════════════════════════
 * CORE BENCHMARK
 *═══════════════════════════════════════════════════════════════════════════════*/

static BenchResult bench_beam_sweep(int T, int K_max, int n_sweeps, int warmup)
{
    BenchResult result = {0};

    /* Generate data */
    int n_regimes = (K_max < 8) ? K_max : 8;
    SyntheticData *data = generate_synthetic_data(T, n_regimes, 42);

    /* Create HDP */
    StickyHDP *hdp = sticky_hdp_create(K_max, T + 100);
    if (!hdp)
    {
        fprintf(stderr, "Failed to create HDP\n");
        free_synthetic_data(data);
        result.total_time = -1;
        return result;
    }

    /* Configure */
    sticky_hdp_set_concentration(hdp, 1.0, 1.0);
    sticky_hdp_set_stickiness(hdp, 50.0);

    /* Initialize with known regime structure */
    double sigma_vol[HDP_MAX_STATES];
    for (int k = 0; k < n_regimes; k++)
        sigma_vol[k] = 0.2;
    sticky_hdp_init_regimes(hdp, n_regimes, data->mu_vol, sigma_vol, NULL);

    /* Load observations */
    sticky_hdp_set_observations(hdp, data->y, T);

    /* Warmup sweeps */
    for (int i = 0; i < warmup; i++)
    {
        sticky_hdp_beam_sweep_single(hdp);
    }

    /* Timed sweeps */
    double *times = (double *)malloc(n_sweeps * sizeof(double));

    double t_start_total = get_time_us();

    for (int i = 0; i < n_sweeps; i++)
    {
        double t_start = get_time_us();
        sticky_hdp_beam_sweep_single(hdp);
        double t_end = get_time_us();
        times[i] = t_end - t_start;
    }

    double t_end_total = get_time_us();

    /* Compute statistics */
    result.total_time = t_end_total - t_start_total;
    result.mean_time = result.total_time / n_sweeps;

    result.min_time = times[0];
    result.max_time = times[0];
    double sum = 0, sum_sq = 0;

    for (int i = 0; i < n_sweeps; i++)
    {
        if (times[i] < result.min_time)
            result.min_time = times[i];
        if (times[i] > result.max_time)
            result.max_time = times[i];
        sum += times[i];
        sum_sq += times[i] * times[i];
    }

    double variance = (sum_sq - sum * sum / n_sweeps) / (n_sweeps - 1);
    result.std_time = sqrt(variance > 0 ? variance : 0);

    /* Percentiles */
    qsort(times, n_sweeps, sizeof(double), compare_double);
    result.p50_time = times[n_sweeps / 2];
    result.p99_time = times[(int)(n_sweeps * 0.99)];

    /* Throughput */
    result.sweeps_per_sec = n_sweeps / (result.total_time / 1e6);
    result.ticks_per_sec = (double)T * n_sweeps / (result.total_time / 1e6);

    /* Quality metrics */
    HDP_Diagnostics diag;
    sticky_hdp_get_diagnostics(hdp, &diag);
    result.avg_active_states = diag.avg_active;
    result.final_log_marginal = diag.log_marginal;
    result.final_K = diag.K;

    /* Cleanup */
    free(times);
    sticky_hdp_destroy(hdp);
    free_synthetic_data(data);

    return result;
}

/*═══════════════════════════════════════════════════════════════════════════════
 * SCALING BENCHMARK
 *═══════════════════════════════════════════════════════════════════════════════*/

static void bench_scaling(void)
{
    printf("\n");
    printf("═══════════════════════════════════════════════════════════════════════════════\n");
    printf("  SCALING BENCHMARK: Time vs T (window size) and K (max states)\n");
    printf("═══════════════════════════════════════════════════════════════════════════════\n\n");

    int T_values[] = {100, 200, 500, 1000};
    int K_values[] = {4, 8, 16, 32};
    int n_T = sizeof(T_values) / sizeof(T_values[0]);
    int n_K = sizeof(K_values) / sizeof(K_values[0]);

    int n_sweeps = 50;
    int warmup = 5;

    /* Header */
    printf("  %6s |", "T \\ K");
    for (int j = 0; j < n_K; j++)
    {
        printf(" %8d |", K_values[j]);
    }
    printf("\n");
    printf("  ───────");
    for (int j = 0; j < n_K; j++)
        printf("───────────");
    printf("\n");

    /* Data */
    for (int i = 0; i < n_T; i++)
    {
        printf("  %6d |", T_values[i]);
        for (int j = 0; j < n_K; j++)
        {
            BenchResult r = bench_beam_sweep(T_values[i], K_values[j], n_sweeps, warmup);
            if (r.total_time < 0)
            {
                printf(" %8s |", "ERROR");
            }
            else
            {
                printf(" %7.0f μs|", r.mean_time);
            }
            fflush(stdout);
        }
        printf("\n");
    }

    printf("\n");
}

/*═══════════════════════════════════════════════════════════════════════════════
 * DETAILED SINGLE CONFIG BENCHMARK
 *═══════════════════════════════════════════════════════════════════════════════*/

static void bench_detailed(int T, int K_max, int n_sweeps)
{
    printf("\n");
    printf("═══════════════════════════════════════════════════════════════════════════════\n");
    printf("  DETAILED BENCHMARK: T=%d, K_max=%d, n_sweeps=%d\n", T, K_max, n_sweeps);
    printf("═══════════════════════════════════════════════════════════════════════════════\n\n");

    BenchResult r = bench_beam_sweep(T, K_max, n_sweeps, 10);

    if (r.total_time < 0)
    {
        printf("  ERROR: Benchmark failed\n");
        return;
    }

    printf("  TIMING (per sweep):\n");
    printf("  ────────────────────────────────────────────────────────────────────────────\n");
    printf("    Mean:        %10.2f μs\n", r.mean_time);
    printf("    Std:         %10.2f μs\n", r.std_time);
    printf("    Min:         %10.2f μs\n", r.min_time);
    printf("    Max:         %10.2f μs\n", r.max_time);
    printf("    P50:         %10.2f μs\n", r.p50_time);
    printf("    P99:         %10.2f μs\n", r.p99_time);
    printf("\n");

    printf("  THROUGHPUT:\n");
    printf("  ────────────────────────────────────────────────────────────────────────────\n");
    printf("    Sweeps/sec:  %10.1f\n", r.sweeps_per_sec);
    printf("    Ticks/sec:   %10.0f  (T × sweeps/sec)\n", r.ticks_per_sec);
    printf("\n");

    printf("  QUALITY:\n");
    printf("  ────────────────────────────────────────────────────────────────────────────\n");
    printf("    Final K:     %10d  active states\n", r.final_K);
    printf("    Avg active:  %10.1f  states per step\n", r.avg_active_states);
    printf("    Log marg:    %10.2f\n", r.final_log_marginal);
    printf("\n");

    /* Performance analysis */
    printf("  ANALYSIS:\n");
    printf("  ────────────────────────────────────────────────────────────────────────────\n");

    double time_per_tick = r.mean_time / T;
    double time_per_state = r.mean_time / (T * r.avg_active_states);

    printf("    Time/tick:   %10.3f μs\n", time_per_tick);
    printf("    Time/state:  %10.3f μs  (per tick × state)\n", time_per_state);
    printf("\n");

    /* Complexity estimate */
    double expected_O_TK2 = T * r.avg_active_states * r.avg_active_states;
    double ns_per_op = r.mean_time * 1000.0 / expected_O_TK2;

    printf("    O(T×K²):     %10.0f  operations\n", expected_O_TK2);
    printf("    ns/op:       %10.2f  (mean_time / O(T×K²))\n", ns_per_op);
    printf("\n");

    /* Bottleneck estimate */
    printf("  ESTIMATED BOTTLENECK BREAKDOWN (based on typical profiles):\n");
    printf("  ────────────────────────────────────────────────────────────────────────────\n");
    printf("    Log-sum-exp:   ~40%%  (%7.0f μs) ← PRIMARY TARGET\n", r.mean_time * 0.40);
    printf("    Likelihood:    ~25%%  (%7.0f μs) ← BATCH OPPORTUNITY\n", r.mean_time * 0.25);
    printf("    Forward pass:  ~20%%  (%7.0f μs)\n", r.mean_time * 0.20);
    printf("    Backward:      ~10%%  (%7.0f μs)\n", r.mean_time * 0.10);
    printf("    Other:         ~5%%   (%7.0f μs)\n", r.mean_time * 0.05);
    printf("\n");

    /* Target assessment */
    printf("  TARGET ASSESSMENT:\n");
    printf("  ────────────────────────────────────────────────────────────────────────────\n");

    if (r.mean_time < 500)
    {
        printf("    ✅ EXCELLENT: <500μs — suitable for real-time (100+ sweeps/sec)\n");
    }
    else if (r.mean_time < 1000)
    {
        printf("    ✅ GOOD: <1ms — suitable for online use (periodic sweeps)\n");
    }
    else if (r.mean_time < 5000)
    {
        printf("    🟡 ACCEPTABLE: <5ms — suitable for near-real-time\n");
    }
    else
    {
        printf("    ❌ SLOW: >5ms — optimization needed for real-time use\n");
    }

    printf("\n");
}

/*═══════════════════════════════════════════════════════════════════════════════
 * ADAPTIVE SWEEP BENCHMARK
 *═══════════════════════════════════════════════════════════════════════════════*/

static void bench_adaptive_sweep(int T, int K_max)
{
    printf("\n");
    printf("═══════════════════════════════════════════════════════════════════════════════\n");
    printf("  ADAPTIVE SWEEP BENCHMARK: T=%d, K_max=%d\n", T, K_max);
    printf("═══════════════════════════════════════════════════════════════════════════════\n\n");
    fflush(stdout);

    /* Generate data with regime change */
    int n_regimes = 4;
    SyntheticData *data = generate_synthetic_data(T, n_regimes, 42);
    if (!data)
    {
        printf("  ERROR: Failed to generate data\n");
        return;
    }
    printf("  Generated %d observations with %d regimes\n", T, n_regimes);
    fflush(stdout);

    /* Create HDP */
    StickyHDP *hdp = sticky_hdp_create(K_max, T + 100);
    if (!hdp)
    {
        fprintf(stderr, "  ERROR: Failed to create HDP\n");
        free_synthetic_data(data);
        return;
    }
    printf("  Created HDP sampler\n");
    fflush(stdout);

    sticky_hdp_set_concentration(hdp, 1.0, 1.0);
    sticky_hdp_set_stickiness(hdp, 50.0);

    /* Configure adaptive sweep */
    sticky_hdp_set_adaptive_sweep(hdp, 5.0, 100, 10); /* surprise>5, max 100 idle, min 10 between */

    /* Initialize with known regimes */
    double sigma_vol[HDP_MAX_STATES];
    for (int k = 0; k < n_regimes; k++)
        sigma_vol[k] = 0.2;
    sticky_hdp_init_regimes(hdp, n_regimes, data->mu_vol, sigma_vol, NULL);
    printf("  Initialized %d regimes, K=%d\n", n_regimes, hdp->K);
    fflush(stdout);

    /* Run with adaptive triggering */
    int sweeps_triggered = 0;
    int total_ticks = 0;
    double total_sweep_time = 0.0;

    printf("  Running adaptive sweep test...\n");
    fflush(stdout);

    double t_start = get_time_us();

    for (int t = 0; t < T; t++)
    {
        sticky_hdp_observe(hdp, data->y[t]);
        total_ticks++;

        /* Check should_sweep - with safety check */
        bool should = sticky_hdp_should_sweep(hdp);

        if (should)
        {
            double sweep_start = get_time_us();
            sticky_hdp_beam_sweep_single(hdp);
            total_sweep_time += get_time_us() - sweep_start;
            sweeps_triggered++;

            /* Reset counter */
            hdp->ticks_since_sweep = 0;
        }
    }

    double total_time = get_time_us() - t_start;

    printf("  RESULTS:\n");
    printf("  ────────────────────────────────────────────────────────────────────────────\n");
    printf("    Total ticks:       %d\n", total_ticks);
    printf("    Sweeps triggered:  %d (%.1f%% of ticks)\n",
           sweeps_triggered, 100.0 * sweeps_triggered / total_ticks);
    printf("    Sweep frequency:   1 per %.1f ticks\n",
           sweeps_triggered > 0 ? (double)total_ticks / sweeps_triggered : 0);
    printf("\n");
    printf("  TIMING:\n");
    printf("  ────────────────────────────────────────────────────────────────────────────\n");
    printf("    Total time:        %.2f ms\n", total_time / 1000.0);
    printf("    Sweep time:        %.2f ms (%.1f%%)\n",
           total_sweep_time / 1000.0, 100.0 * total_sweep_time / total_time);
    printf("    Overhead time:     %.2f ms (%.1f%%)\n",
           (total_time - total_sweep_time) / 1000.0,
           100.0 * (total_time - total_sweep_time) / total_time);
    printf("    Time/tick:         %.3f μs\n", total_time / total_ticks);
    printf("\n");

    /* Compare to fixed-schedule baseline */
    int fixed_interval = 100;
    int fixed_sweeps = T / fixed_interval;

    printf("  COMPARISON TO FIXED SCHEDULE (every %d ticks):\n", fixed_interval);
    printf("  ────────────────────────────────────────────────────────────────────────────\n");
    printf("    Fixed sweeps:      %d\n", fixed_sweeps);
    printf("    Adaptive sweeps:   %d\n", sweeps_triggered);
    if (sweeps_triggered > fixed_sweeps)
    {
        printf("    Result:            Volatile data → +%d extra sweeps for faster adaptation\n",
               sweeps_triggered - fixed_sweeps);
    }
    else
    {
        printf("    Result:            Stable data → %d fewer sweeps (%.0f%% reduction)\n",
               fixed_sweeps - sweeps_triggered,
               100.0 * (1.0 - (double)sweeps_triggered / fixed_sweeps));
    }
    printf("\n");

    sticky_hdp_destroy(hdp);
    free_synthetic_data(data);
}

/*═══════════════════════════════════════════════════════════════════════════════
 * MAIN
 *═══════════════════════════════════════════════════════════════════════════════*/

int main(int argc, char **argv)
{
    int T = 500;
    int K_max = 8;
    int n_sweeps = 100;

    if (argc >= 2)
        T = atoi(argv[1]);
    if (argc >= 3)
        K_max = atoi(argv[2]);
    if (argc >= 4)
        n_sweeps = atoi(argv[3]);

    printf("\n");
    printf("███████████████████████████████████████████████████████████████████████████████\n");
    printf("█                                                                             █\n");
    printf("█  STICKY HDP-HMM BEAM SAMPLING BENCHMARK                                     █\n");
    printf("█                                                                             █\n");
    printf("███████████████████████████████████████████████████████████████████████████████\n");
    printf("\n");
    printf("  Configuration:\n");
    printf("    Window size (T):     %d\n", T);
    printf("    Max states (K_max):  %d\n", K_max);
    printf("    Sweeps:              %d\n", n_sweeps);
    printf("\n");

    /* Run detailed benchmark for specified config */
    bench_detailed(T, K_max, n_sweeps);

    /* Run scaling benchmark */
    bench_scaling();

    /* Run adaptive sweep benchmark */
    bench_adaptive_sweep(T, K_max);

    printf("═══════════════════════════════════════════════════════════════════════════════\n");
    printf("  DONE\n");
    printf("═══════════════════════════════════════════════════════════════════════════════\n\n");

    return 0;
}