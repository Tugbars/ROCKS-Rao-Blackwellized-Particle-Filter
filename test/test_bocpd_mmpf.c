/**
 * @file test_bocpd_mmpf.c
 * @brief Integration test: BOCPD shock detection → MMPF regime switching
 *
 * Tests the full sidecar architecture:
 *   1. BOCPD runs on log-returns, detects changepoints via r[0] threshold
 *   2. When r[0] > threshold, inject shock into MMPF
 *   3. MMPF re-evaluates regimes with uniform transitions
 *   4. Restore normal operation
 *
 * Compares:
 *   - MMPF alone (baseline)
 *   - MMPF + BOCPD (integrated)
 *
 * Metrics:
 *   - Regime detection latency (ticks to switch after true CP)
 *   - False regime switches (switches without true CP)
 *   - Vol RMSE (estimation quality)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "mmpf_rocks.h"
#include "bocpd.h"

/*═══════════════════════════════════════════════════════════════════════════
 * TIMING
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
#include <time.h>
static inline double get_time_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1000.0;
}
#endif

/*═══════════════════════════════════════════════════════════════════════════
 * PCG32 RNG
 *═══════════════════════════════════════════════════════════════════════════*/

typedef struct
{
    uint64_t state;
    uint64_t inc;
} pcg32_t;

static inline void pcg32_seed(pcg32_t *rng, uint64_t seed)
{
    rng->state = 0;
    rng->inc = (seed << 1u) | 1u;
    rng->state = rng->state * 6364136223846793005ULL + rng->inc;
    rng->state = rng->state * 6364136223846793005ULL + rng->inc;
}

static inline uint32_t pcg32_random(pcg32_t *rng)
{
    uint64_t oldstate = rng->state;
    rng->state = oldstate * 6364136223846793005ULL + rng->inc;
    uint32_t xorshifted = (uint32_t)(((oldstate >> 18u) ^ oldstate) >> 27u);
    uint32_t rot = (uint32_t)(oldstate >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((32u - rot) & 31u));
}

static inline double pcg32_double(pcg32_t *rng)
{
    return (double)pcg32_random(rng) / 4294967296.0;
}

static double pcg32_gaussian(pcg32_t *rng)
{
    double u1 = pcg32_double(rng);
    double u2 = pcg32_double(rng);
    if (u1 < 1e-10)
        u1 = 1e-10;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2);
}

/*═══════════════════════════════════════════════════════════════════════════
 * SYNTHETIC DATA GENERATION
 *
 * Generate log-returns with known regime changes:
 *   - Calm:   σ = 0.8% daily (μ_vol ≈ -4.83)
 *   - Trend:  σ = 1.5% daily (μ_vol ≈ -4.20)
 *   - Crisis: σ = 3.5% daily (μ_vol ≈ -3.35)
 *═══════════════════════════════════════════════════════════════════════════*/

#define N_TICKS 8000
#define N_CHANGEPOINTS 6

typedef enum
{
    REGIME_CALM = 0,
    REGIME_TREND = 1,
    REGIME_CRISIS = 2
} TrueRegime;

typedef struct
{
    int tick;
    TrueRegime from;
    TrueRegime to;
    const char *description;
} Changepoint;

static const Changepoint TRUE_CHANGEPOINTS[N_CHANGEPOINTS] = {
    {1000, REGIME_CALM, REGIME_TREND, "Calm → Trend"},
    {2000, REGIME_TREND, REGIME_CRISIS, "Trend → Crisis (gradual)"},
    {2500, REGIME_CRISIS, REGIME_TREND, "Crisis → Trend (recovery)"},
    {4000, REGIME_TREND, REGIME_CALM, "Trend → Calm"},
    {5500, REGIME_CALM, REGIME_CRISIS, "Calm → Crisis (sudden)"},
    {6500, REGIME_CRISIS, REGIME_CALM, "Crisis → Calm"}};

static const double REGIME_VOL[3] = {
    0.008, /* Calm:   0.8% daily */
    0.015, /* Trend:  1.5% daily */
    0.035  /* Crisis: 3.5% daily */
};

typedef struct
{
    double *log_returns;
    double *true_vol;
    int *true_regime;
    int *is_changepoint;
    int n_ticks;
} SyntheticData;

static SyntheticData *generate_data(uint64_t seed)
{
    SyntheticData *data = (SyntheticData *)malloc(sizeof(SyntheticData));
    data->log_returns = (double *)malloc(N_TICKS * sizeof(double));
    data->true_vol = (double *)malloc(N_TICKS * sizeof(double));
    data->true_regime = (int *)malloc(N_TICKS * sizeof(int));
    data->is_changepoint = (int *)calloc(N_TICKS, sizeof(int));
    data->n_ticks = N_TICKS;

    pcg32_t rng;
    pcg32_seed(&rng, seed);

    /* Mark changepoints */
    for (int i = 0; i < N_CHANGEPOINTS; i++)
    {
        if (TRUE_CHANGEPOINTS[i].tick < N_TICKS)
        {
            data->is_changepoint[TRUE_CHANGEPOINTS[i].tick] = 1;
        }
    }

    /* Generate data */
    TrueRegime current_regime = REGIME_CALM;
    int cp_idx = 0;

    for (int t = 0; t < N_TICKS; t++)
    {
        /* Check for regime change */
        if (cp_idx < N_CHANGEPOINTS && t == TRUE_CHANGEPOINTS[cp_idx].tick)
        {
            current_regime = TRUE_CHANGEPOINTS[cp_idx].to;
            cp_idx++;
        }

        data->true_regime[t] = current_regime;
        data->true_vol[t] = REGIME_VOL[current_regime];

        /* Generate log-return: r_t = σ_t × ε_t */
        double epsilon = pcg32_gaussian(&rng);
        data->log_returns[t] = data->true_vol[t] * epsilon;
    }

    return data;
}

static void free_data(SyntheticData *data)
{
    free(data->log_returns);
    free(data->true_vol);
    free(data->true_regime);
    free(data->is_changepoint);
    free(data);
}

/*═══════════════════════════════════════════════════════════════════════════
 * METRICS
 *═══════════════════════════════════════════════════════════════════════════*/

typedef struct
{
    /* Regime detection */
    int true_positives;       /* Regime switches near true CPs */
    int false_positives;      /* Regime switches without true CP */
    double avg_detection_lag; /* Ticks from true CP to detected switch */
    double max_detection_lag;

    /* Vol estimation */
    double vol_rmse;
    double log_vol_rmse;

    /* Per-changepoint details */
    int cp_detected[N_CHANGEPOINTS];
    int cp_lag[N_CHANGEPOINTS];

    /* Timing */
    double avg_latency_us;
    double max_latency_us;

    /* BOCPD stats */
    int shock_count;
} IntegrationMetrics;

static void compute_metrics(
    IntegrationMetrics *m,
    const SyntheticData *data,
    const MMPF_Hypothesis *detected_regime,
    const double *estimated_vol,
    const int *shock_fired,
    const double *latencies)
{
    memset(m, 0, sizeof(IntegrationMetrics));

    /* Vol RMSE */
    double sum_sq_err = 0.0;
    double sum_log_sq_err = 0.0;
    int warmup = 100;

    for (int t = warmup; t < data->n_ticks; t++)
    {
        double err = estimated_vol[t] - data->true_vol[t];
        sum_sq_err += err * err;

        double log_err = log(estimated_vol[t]) - log(data->true_vol[t]);
        sum_log_sq_err += log_err * log_err;
    }

    int n_valid = data->n_ticks - warmup;
    m->vol_rmse = sqrt(sum_sq_err / n_valid);
    m->log_vol_rmse = sqrt(sum_log_sq_err / n_valid);

    /* Regime detection: check ±20 ticks around each true CP */
    const int DETECTION_WINDOW = 20;

    for (int i = 0; i < N_CHANGEPOINTS; i++)
    {
        int cp_tick = TRUE_CHANGEPOINTS[i].tick;
        if (cp_tick >= data->n_ticks)
            continue;

        TrueRegime expected = TRUE_CHANGEPOINTS[i].to;
        MMPF_Hypothesis expected_hyp;
        switch (expected)
        {
        case REGIME_CALM:
            expected_hyp = MMPF_CALM;
            break;
        case REGIME_TREND:
            expected_hyp = MMPF_TREND;
            break;
        case REGIME_CRISIS:
            expected_hyp = MMPF_CRISIS;
            break;
        default:
            expected_hyp = MMPF_CALM;
        }

        /* Look for regime switch to expected hypothesis within window */
        m->cp_detected[i] = 0;
        m->cp_lag[i] = -1;

        for (int t = cp_tick; t < cp_tick + DETECTION_WINDOW && t < data->n_ticks; t++)
        {
            if (detected_regime[t] == expected_hyp)
            {
                m->cp_detected[i] = 1;
                m->cp_lag[i] = t - cp_tick;
                m->true_positives++;
                m->avg_detection_lag += m->cp_lag[i];
                if (m->cp_lag[i] > m->max_detection_lag)
                {
                    m->max_detection_lag = m->cp_lag[i];
                }
                break;
            }
        }
    }

    if (m->true_positives > 0)
    {
        m->avg_detection_lag /= m->true_positives;
    }

    /* Count false positives: regime switches outside CP windows */
    for (int t = warmup + 1; t < data->n_ticks; t++)
    {
        if (detected_regime[t] != detected_regime[t - 1])
        {
            /* Check if this is near any true CP */
            int near_cp = 0;
            for (int i = 0; i < N_CHANGEPOINTS; i++)
            {
                int cp_tick = TRUE_CHANGEPOINTS[i].tick;
                if (t >= cp_tick - 5 && t <= cp_tick + DETECTION_WINDOW)
                {
                    near_cp = 1;
                    break;
                }
            }
            if (!near_cp)
            {
                m->false_positives++;
            }
        }
    }

    /* Timing */
    double sum_lat = 0.0;
    m->max_latency_us = 0.0;
    for (int t = 0; t < data->n_ticks; t++)
    {
        sum_lat += latencies[t];
        if (latencies[t] > m->max_latency_us)
        {
            m->max_latency_us = latencies[t];
        }
    }
    m->avg_latency_us = sum_lat / data->n_ticks;

    /* Shock count */
    for (int t = 0; t < data->n_ticks; t++)
    {
        if (shock_fired[t])
            m->shock_count++;
    }
}

static void print_metrics(const char *name, const IntegrationMetrics *m)
{
    printf("\n%s\n", name);
    printf("────────────────────────────────────────────────────────────────\n");
    printf("  Vol RMSE:           %.6f (%.2f%%)\n", m->vol_rmse, m->vol_rmse * 100);
    printf("  Log-Vol RMSE:       %.4f\n", m->log_vol_rmse);
    printf("  True Positives:     %d / %d (%.1f%% recall)\n",
           m->true_positives, N_CHANGEPOINTS,
           100.0 * m->true_positives / N_CHANGEPOINTS);
    printf("  False Positives:    %d\n", m->false_positives);
    printf("  Avg Detection Lag:  %.1f ticks\n", m->avg_detection_lag);
    printf("  Max Detection Lag:  %.0f ticks\n", m->max_detection_lag);
    printf("  Shock Count:        %d\n", m->shock_count);
    printf("  Avg Latency:        %.2f μs\n", m->avg_latency_us);
    printf("  Max Latency:        %.2f μs\n", m->max_latency_us);
    printf("\n  Per-Changepoint:\n");
    for (int i = 0; i < N_CHANGEPOINTS; i++)
    {
        printf("    [%4d] %-25s : %s",
               TRUE_CHANGEPOINTS[i].tick,
               TRUE_CHANGEPOINTS[i].description,
               m->cp_detected[i] ? "✓" : "✗");
        if (m->cp_detected[i])
        {
            printf(" (lag=%d)", m->cp_lag[i]);
        }
        printf("\n");
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST: MMPF ALONE (Baseline)
 *═══════════════════════════════════════════════════════════════════════════*/

static void run_mmpf_alone(
    const SyntheticData *data,
    MMPF_Hypothesis *detected_regime,
    double *estimated_vol,
    int *shock_fired,
    double *latencies,
    IntegrationMetrics *metrics)
{
    MMPF_Config cfg = mmpf_config_defaults();
    cfg.n_particles = 256;
    cfg.enable_student_t = 1;
    cfg.enable_global_baseline = 1;
    cfg.transition_prior_mass = 100.0;
    cfg.crisis_entry_boost = 1.5;

    MMPF_ROCKS *mmpf = mmpf_create(&cfg);
    mmpf_reset(mmpf, 0.01);

    MMPF_Output output;

    for (int t = 0; t < data->n_ticks; t++)
    {
        double t0 = get_time_us();

        mmpf_step(mmpf, (rbpf_real_t)data->log_returns[t], &output);

        double t1 = get_time_us();
        latencies[t] = t1 - t0;

        detected_regime[t] = output.dominant;
        estimated_vol[t] = (double)output.volatility;
        shock_fired[t] = 0; /* No BOCPD */
    }

    compute_metrics(metrics, data, detected_regime, estimated_vol, shock_fired, latencies);

    mmpf_destroy(mmpf);
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST: MMPF + BOCPD (Integrated)
 *═══════════════════════════════════════════════════════════════════════════*/

static void run_mmpf_with_bocpd(
    const SyntheticData *data,
    double r0_threshold,
    int refractory_ticks,
    MMPF_Hypothesis *detected_regime,
    double *estimated_vol,
    int *shock_fired,
    double *latencies,
    IntegrationMetrics *metrics)
{
    /* Initialize MMPF */
    MMPF_Config cfg = mmpf_config_defaults();
    cfg.n_particles = 256;
    cfg.enable_student_t = 1;
    cfg.enable_global_baseline = 1;
    cfg.transition_prior_mass = 100.0;
    cfg.crisis_entry_boost = 1.5;

    MMPF_ROCKS *mmpf = mmpf_create(&cfg);
    mmpf_reset(mmpf, 0.01);

    /* Initialize BOCPD with power-law hazard */
    bocpd_t bocpd;
    bocpd_hazard_t hazard;
    /* Initialize BOCPD with power-law hazard
     * Prior centered on typical log(r²) ≈ -9 (1% daily vol)
     * Weak prior (kappa=1) for fast adaptation */
    bocpd_prior_t prior = {
        .mu0 = -9.0,   /* E[log(r²)] ≈ 2×log(0.01) - 1.27 ≈ -10.5, start middle */
        .kappa0 = 1.0, /* Weak prior on mean */
        .alpha0 = 1.0, /* Weak prior on variance */
        .beta0 = 2.0   /* Var ≈ 2 for log-chi² noise */
    };

    bocpd_hazard_init_power_law(&hazard, 0.8, 512);
    bocpd_init_with_hazard(&bocpd, &hazard, prior);

    MMPF_Output output;
    int refractory_counter = 0;
    int warmup = 50;

    for (int t = 0; t < data->n_ticks; t++)
    {
        double t0 = get_time_us();

        /* Transform observation: y = log(r²)
         * This is what MMPF sees internally.
         * BOCPD needs this to detect VARIANCE changes (not mean changes).
         * Raw log-returns have constant mean=0, so BOCPD sees nothing.
         * log(r²) has mean ≈ 2×log(σ) - 1.27, which shifts with regime. */
        double r = data->log_returns[t];
        double y_log;
        if (fabs(r) < 1e-10)
        {
            y_log = -20.0; /* Floor for zero returns */
        }
        else
        {
            y_log = log(r * r);
        }

        /* Step BOCPD on transformed observation */
        bocpd_step(&bocpd, y_log);

        /* Check for shock (after warmup, respecting refractory) */
        shock_fired[t] = 0;
        if (t >= warmup)
        {
            if (refractory_counter > 0)
            {
                refractory_counter--;
            }
            else if (bocpd.r[0] > r0_threshold)
            {
                shock_fired[t] = 1;
                refractory_counter = refractory_ticks;
            }
        }

        /* Step MMPF with or without shock */
        if (shock_fired[t])
        {
            mmpf_inject_shock(mmpf);
            mmpf_step(mmpf, (rbpf_real_t)data->log_returns[t], &output);
            mmpf_restore_from_shock(mmpf);
        }
        else
        {
            mmpf_step(mmpf, (rbpf_real_t)data->log_returns[t], &output);
        }

        double t1 = get_time_us();
        latencies[t] = t1 - t0;

        detected_regime[t] = output.dominant;
        estimated_vol[t] = (double)output.volatility;
    }

    compute_metrics(metrics, data, detected_regime, estimated_vol, shock_fired, latencies);

    mmpf_destroy(mmpf);
    bocpd_free(&bocpd);
    bocpd_hazard_free(&hazard);
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
    printf("  BOCPD + MMPF Integration Test (seed=%llu)\n", (unsigned long long)seed);
    printf("══════════════════════════════════════════════════════════════════\n");

    /* Generate data */
    SyntheticData *data = generate_data(seed);

    printf("\nTest Data: %d ticks, %d changepoints\n", N_TICKS, N_CHANGEPOINTS);
    printf("\nTrue Changepoints:\n");
    for (int i = 0; i < N_CHANGEPOINTS; i++)
    {
        printf("  [%4d] %s\n", TRUE_CHANGEPOINTS[i].tick, TRUE_CHANGEPOINTS[i].description);
    }

    /* Allocate result arrays */
    MMPF_Hypothesis *detected_regime = (MMPF_Hypothesis *)malloc(N_TICKS * sizeof(MMPF_Hypothesis));
    double *estimated_vol = (double *)malloc(N_TICKS * sizeof(double));
    int *shock_fired = (int *)calloc(N_TICKS, sizeof(int));
    double *latencies = (double *)malloc(N_TICKS * sizeof(double));

    IntegrationMetrics metrics_alone, metrics_integrated;

    /*═══════════════════════════════════════════════════════════════════════
     * TEST 1: MMPF Alone (Baseline)
     *═══════════════════════════════════════════════════════════════════════*/

    run_mmpf_alone(data, detected_regime, estimated_vol, shock_fired, latencies, &metrics_alone);
    print_metrics("MMPF Alone (Baseline)", &metrics_alone);

    /*═══════════════════════════════════════════════════════════════════════
     * TEST 2: MMPF + BOCPD (r[0] > 0.05, refractory=20)
     *═══════════════════════════════════════════════════════════════════════*/

    run_mmpf_with_bocpd(data, 0.05, 20,
                        detected_regime, estimated_vol, shock_fired, latencies,
                        &metrics_integrated);
    print_metrics("MMPF + BOCPD (r[0] > 0.05, refractory=20)", &metrics_integrated);

    /*═══════════════════════════════════════════════════════════════════════
     * TEST 3: MMPF + BOCPD (r[0] > 0.03, refractory=30) - More sensitive
     *═══════════════════════════════════════════════════════════════════════*/

    IntegrationMetrics metrics_sensitive;
    run_mmpf_with_bocpd(data, 0.03, 30,
                        detected_regime, estimated_vol, shock_fired, latencies,
                        &metrics_sensitive);
    print_metrics("MMPF + BOCPD (r[0] > 0.03, refractory=30) - Sensitive", &metrics_sensitive);

    /*═══════════════════════════════════════════════════════════════════════
     * TEST 4: MMPF + BOCPD (r[0] > 0.10, refractory=10) - Conservative
     *═══════════════════════════════════════════════════════════════════════*/

    IntegrationMetrics metrics_conservative;
    run_mmpf_with_bocpd(data, 0.10, 10,
                        detected_regime, estimated_vol, shock_fired, latencies,
                        &metrics_conservative);
    print_metrics("MMPF + BOCPD (r[0] > 0.10, refractory=10) - Conservative", &metrics_conservative);

    /*═══════════════════════════════════════════════════════════════════════
     * SUMMARY
     *═══════════════════════════════════════════════════════════════════════*/

    printf("\n══════════════════════════════════════════════════════════════════\n");
    printf("  SUMMARY\n");
    printf("══════════════════════════════════════════════════════════════════\n");

    printf("\n%-45s %8s %8s %8s %8s\n", "Configuration", "Recall", "FP", "Lag", "Vol RMSE");
    printf("─────────────────────────────────────────────────────────────────────────────\n");
    printf("%-45s %7.1f%% %8d %7.1f %8.6f\n",
           "MMPF Alone",
           100.0 * metrics_alone.true_positives / N_CHANGEPOINTS,
           metrics_alone.false_positives,
           metrics_alone.avg_detection_lag,
           metrics_alone.vol_rmse);
    printf("%-45s %7.1f%% %8d %7.1f %8.6f\n",
           "MMPF + BOCPD (r[0]>0.05, refr=20)",
           100.0 * metrics_integrated.true_positives / N_CHANGEPOINTS,
           metrics_integrated.false_positives,
           metrics_integrated.avg_detection_lag,
           metrics_integrated.vol_rmse);
    printf("%-45s %7.1f%% %8d %7.1f %8.6f\n",
           "MMPF + BOCPD (r[0]>0.03, refr=30) Sensitive",
           100.0 * metrics_sensitive.true_positives / N_CHANGEPOINTS,
           metrics_sensitive.false_positives,
           metrics_sensitive.avg_detection_lag,
           metrics_sensitive.vol_rmse);
    printf("%-45s %7.1f%% %8d %7.1f %8.6f\n",
           "MMPF + BOCPD (r[0]>0.10, refr=10) Conservative",
           100.0 * metrics_conservative.true_positives / N_CHANGEPOINTS,
           metrics_conservative.false_positives,
           metrics_conservative.avg_detection_lag,
           metrics_conservative.vol_rmse);

    printf("\nExpected improvements with BOCPD:\n");
    printf("  - Lower detection lag (shock enables immediate re-evaluation)\n");
    printf("  - Similar or better Vol RMSE\n");
    printf("  - Trade-off: shock_count ≈ FP + TP (each shock is a re-evaluation)\n");

    /* Cleanup */
    free(detected_regime);
    free(estimated_vol);
    free(shock_fired);
    free(latencies);
    free_data(data);

    printf("\n══════════════════════════════════════════════════════════════════\n");

    return 0;
}