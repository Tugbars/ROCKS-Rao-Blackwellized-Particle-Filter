/*=============================================================================
 * Single RBPF Test (Storvik + Adaptive Forgetting + Robust OCSN)
 *
 * Tests the best single-filter configuration across challenging scenarios.
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
 * ═══════════════════════════════════════════════════════════════════════════
 * BUILD
 * ═══════════════════════════════════════════════════════════════════════════
 *
 *   gcc -O3 -march=native -o test_single_rbpf test_single_rbpf.c \
 *       rbpf_ksc.c rbpf_ksc_init.c rbpf_ksc_output.c \
 *       rbpf_ksc_param_integration.c rbpf_param_learn.c \
 *       rbpf_adaptive_forgetting.c rbpf_fixed_lag_smoother.c \
 *       -lmkl_rt -lm -lpthread
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * USAGE
 * ═══════════════════════════════════════════════════════════════════════════
 *
 *   ./test_single_rbpf [seed] [output.csv]
 *
 *===========================================================================*/

#include "rbpf_ksc_param_integration.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/*─────────────────────────────────────────────────────────────────────────────
 * TIMING UTILITIES
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

typedef struct {
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
    if (u1 < 1e-10) u1 = 1e-10;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979 * u2);
}

/*─────────────────────────────────────────────────────────────────────────────
 * HYPOTHESIS PARAMETERS (Ground Truth)
 *───────────────────────────────────────────────────────────────────────────*/

typedef enum {
    HYPO_CALM = 0,
    HYPO_TREND = 1,
    HYPO_CRISIS = 2,
    N_HYPOTHESES = 3
} Hypothesis;

static const char *hypothesis_names[] = {"CALM", "TREND", "CRISIS"};

typedef struct {
    double mu_vol;
    double phi;
    double sigma_eta;
    double vol_approx;
} HypothesisParams;

static const HypothesisParams TRUE_PARAMS[N_HYPOTHESES] = {
    {.mu_vol = -5.0, .phi = 0.995, .sigma_eta = 0.08, .vol_approx = 0.007},
    {.mu_vol = -3.5, .phi = 0.95,  .sigma_eta = 0.20, .vol_approx = 0.030},
    {.mu_vol = -1.5, .phi = 0.85,  .sigma_eta = 0.50, .vol_approx = 0.220}
};

/*─────────────────────────────────────────────────────────────────────────────
 * SYNTHETIC DATA
 *───────────────────────────────────────────────────────────────────────────*/

typedef struct {
    double *returns;
    double *true_log_vol;
    double *true_vol;
    int *true_hypothesis;
    int *is_outlier;
    double *outlier_sigma;
    int n_ticks;
    int scenario_starts[10];
    const char *scenario_names[10];
    int n_scenarios;
    int n_outliers_injected;
} SyntheticData;

static void inject_outlier(SyntheticData *data, int t, double target_sigma, pcg32_t *rng)
{
    double vol = data->true_vol[t];
    double sign = (pcg32_double(rng) < 0.5) ? -1.0 : 1.0;
    data->returns[t] = sign * target_sigma * vol;
    data->is_outlier[t] = 1;
    data->outlier_sigma[t] = target_sigma;
    data->n_outliers_injected++;
}

static SyntheticData *generate_test_data(int seed)
{
    SyntheticData *data = (SyntheticData *)calloc(1, sizeof(SyntheticData));

    int n = 8000;
    data->n_ticks = n;
    data->returns = (double *)malloc(n * sizeof(double));
    data->true_log_vol = (double *)malloc(n * sizeof(double));
    data->true_vol = (double *)malloc(n * sizeof(double));
    data->true_hypothesis = (int *)malloc(n * sizeof(int));
    data->is_outlier = (int *)calloc(n, sizeof(int));
    data->outlier_sigma = (double *)calloc(n, sizeof(double));

    pcg32_t rng = {seed * 12345ULL + 1, seed * 67890ULL | 1};

    double log_vol = TRUE_PARAMS[HYPO_CALM].mu_vol;
    int t = 0;

#define EVOLVE_STATE(H) do { \
    const HypothesisParams *p = &TRUE_PARAMS[H]; \
    double theta = 1.0 - p->phi; \
    log_vol = p->phi * log_vol + theta * p->mu_vol + p->sigma_eta * pcg32_gaussian(&rng); \
    double vol = exp(log_vol); \
    double ret = vol * pcg32_gaussian(&rng); \
    data->returns[t] = ret; \
    data->true_log_vol[t] = log_vol; \
    data->true_vol[t] = vol; \
    data->true_hypothesis[t] = (H); \
} while(0)

    /* Scenario 1: Extended Calm (0-1499) */
    data->scenario_starts[0] = 0;
    data->scenario_names[0] = "Extended Calm";
    data->n_scenarios = 1;
    for (; t < 1500; t++) { EVOLVE_STATE(HYPO_CALM); }
    inject_outlier(data, 500, 6.0, &rng);
    inject_outlier(data, 1200, 8.0, &rng);

    /* Scenario 2: Slow Trend (1500-2499) */
    data->scenario_starts[1] = 1500;
    data->scenario_names[1] = "Slow Trend";
    data->n_scenarios = 2;
    for (; t < 2500; t++) {
        Hypothesis h = (t < 1800) ? HYPO_CALM : HYPO_TREND;
        EVOLVE_STATE(h);
    }

    /* Scenario 3: Sudden Crisis (2500-2999) */
    data->scenario_starts[2] = 2500;
    data->scenario_names[2] = "Sudden Crisis";
    data->n_scenarios = 3;
    for (; t < 3000; t++) { EVOLVE_STATE(HYPO_CRISIS); }
    inject_outlier(data, 2510, 8.0, &rng);
    inject_outlier(data, 2530, 10.0, &rng);
    inject_outlier(data, 2560, 12.0, &rng);
    inject_outlier(data, 2650, 9.0, &rng);
    inject_outlier(data, 2800, 11.0, &rng);

    /* Scenario 4: Crisis Persistence (3000-3999) */
    data->scenario_starts[3] = 3000;
    data->scenario_names[3] = "Crisis Persist";
    data->n_scenarios = 4;
    for (; t < 4000; t++) { EVOLVE_STATE(HYPO_CRISIS); }
    inject_outlier(data, 3200, 10.0, &rng);
    inject_outlier(data, 3500, 15.0, &rng);
    inject_outlier(data, 3800, 12.0, &rng);

    /* Scenario 5: Recovery (4000-5199) */
    data->scenario_starts[4] = 4000;
    data->scenario_names[4] = "Recovery";
    data->n_scenarios = 5;
    for (; t < 5200; t++) {
        Hypothesis h;
        if (t < 4400) h = HYPO_CRISIS;
        else if (t < 4800) h = HYPO_TREND;
        else h = HYPO_CALM;
        EVOLVE_STATE(h);
    }

    /* Scenario 6: Flash Crash (5200-5699) */
    data->scenario_starts[5] = 5200;
    data->scenario_names[5] = "Flash Crash";
    data->n_scenarios = 6;
    for (; t < 5700; t++) {
        Hypothesis h;
        if (t >= 5350 && t < 5410) h = HYPO_CRISIS;
        else h = HYPO_CALM;
        EVOLVE_STATE(h);
    }
    inject_outlier(data, 5380, 12.0, &rng);

    /* Scenario 7: Choppy (5700-7999) */
    data->scenario_starts[6] = 5700;
    data->scenario_names[6] = "Choppy";
    data->n_scenarios = 7;
    Hypothesis current_h = HYPO_TREND;
    int next_switch = 5700 + 80 + (int)(pcg32_double(&rng) * 120);
    for (; t < 8000; t++) {
        if (t >= next_switch) {
            int delta = (pcg32_double(&rng) < 0.5) ? -1 : 1;
            current_h = (Hypothesis)((current_h + delta + N_HYPOTHESES) % N_HYPOTHESES);
            next_switch = t + 80 + (int)(pcg32_double(&rng) * 150);
        }
        EVOLVE_STATE(current_h);
    }

#undef EVOLVE_STATE

    return data;
}

static void free_synthetic_data(SyntheticData *data)
{
    if (!data) return;
    free(data->returns);
    free(data->true_log_vol);
    free(data->true_vol);
    free(data->true_hypothesis);
    free(data->is_outlier);
    free(data->outlier_sigma);
    free(data);
}

/*─────────────────────────────────────────────────────────────────────────────
 * TICK RECORD
 *───────────────────────────────────────────────────────────────────────────*/

typedef struct {
    int tick;

    /* Ground truth */
    double true_log_vol;
    double true_vol;
    int true_hypothesis;
    double return_val;
    int is_outlier;
    double outlier_sigma;

    /* Estimates */
    double est_log_vol;
    double est_vol;
    double vol_std;
    int est_hypothesis;
    int est_regime;
    double regime_probs[4];

    /* Health */
    double ess;
    double outlier_fraction;

    /* Learned params */
    double learned_mu_vol;
    double learned_sigma_vol;

    /* Flags */
    int regime_changed;
    int resampled;

    /* Timing */
    double latency_us;
} TickRecord;

/*─────────────────────────────────────────────────────────────────────────────
 * MAP RBPF REGIME TO HYPOTHESIS
 *───────────────────────────────────────────────────────────────────────────*/

static int rbpf_regime_to_hypothesis(int regime)
{
    if (regime <= 1) return HYPO_CALM;
    if (regime == 2) return HYPO_TREND;
    return HYPO_CRISIS;
}

/*─────────────────────────────────────────────────────────────────────────────
 * RUN SINGLE RBPF TEST
 *───────────────────────────────────────────────────────────────────────────*/

static void run_single_rbpf(SyntheticData *data, TickRecord *records,
                            double *total_time, double *max_latency)
{
    const int N_PARTICLES = 512;
    const int N_REGIMES = 4;

    /* Create extended RBPF with Storvik */
    RBPF_Extended *ext = rbpf_ext_create(N_PARTICLES, N_REGIMES, RBPF_PARAM_STORVIK);
    rbpf_ext_enable_kl_tempering(ext);

    /* Enable PARIS smoothed Storvik (L=50 tick lag) */
    rbpf_ext_enable_smoothed_storvik(ext, 5);

    /* Regime params (θ, μ, σ) */
    rbpf_ext_set_regime_params(ext, 0, 0.0030f, -4.299f, 0.080f);
    rbpf_ext_set_regime_params(ext, 1, 0.0420f, -3.465f, 0.267f);
    rbpf_ext_set_regime_params(ext, 2, 0.0810f, -2.954f, 0.453f);
    rbpf_ext_set_regime_params(ext, 3, 0.1200f, -2.171f, 0.640f);

    /* Transition matrix (stickiness=0.92) */
    rbpf_real_t trans[16] = {
        0.920f, 0.056f, 0.020f, 0.004f,
        0.032f, 0.920f, 0.036f, 0.012f,
        0.012f, 0.036f, 0.920f, 0.032f,
        0.004f, 0.020f, 0.056f, 0.920f
    };
    rbpf_ext_build_transition_lut(ext, trans);

    /* Adaptive forgetting */
    rbpf_ext_enable_adaptive_forgetting_mode(ext, ADAPT_SIGNAL_REGIME);
    rbpf_ext_enable_circuit_breaker(ext, 0.999, 100);

    /* Robust OCSN */
    ext->robust_ocsn.enabled = 1;
    ext->robust_ocsn.regime[0].prob = 0.02f;
    ext->robust_ocsn.regime[0].variance = 100.0f;
    ext->robust_ocsn.regime[1].prob = 0.03f;
    ext->robust_ocsn.regime[1].variance = 120.0f;
    ext->robust_ocsn.regime[2].prob = 0.04f;
    ext->robust_ocsn.regime[2].variance = 140.0f;
    ext->robust_ocsn.regime[3].prob = 0.05f;
    ext->robust_ocsn.regime[3].variance = 160.0f;

    rbpf_ext_init(ext, -4.5f, 0.1f);

    *total_time = 0.0;
    *max_latency = 0.0;

    RBPF_KSC_Output output;
    int n = data->n_ticks;

    for (int t = 0; t < n; t++) {
        memset(&output, 0, sizeof(output));

        double t_start = get_time_us();
        rbpf_ext_step(ext, (rbpf_real_t)data->returns[t], &output);
        double t_end = get_time_us();

        double latency = t_end - t_start;
        *total_time += latency;
        if (latency > *max_latency) *max_latency = latency;

        TickRecord *rec = &records[t];
        rec->tick = t;

        /* Ground truth */
        rec->true_log_vol = data->true_log_vol[t];
        rec->true_vol = data->true_vol[t];
        rec->true_hypothesis = data->true_hypothesis[t];
        rec->return_val = data->returns[t];
        rec->is_outlier = data->is_outlier[t];
        rec->outlier_sigma = data->outlier_sigma[t];

        /* Estimates */
        rec->est_log_vol = output.log_vol_mean;
        rec->est_vol = output.vol_mean;
        rec->vol_std = sqrt(output.log_vol_var);
        rec->est_regime = output.dominant_regime;
        rec->est_hypothesis = rbpf_regime_to_hypothesis(output.dominant_regime);

        for (int r = 0; r < 4; r++) {
            rec->regime_probs[r] = output.regime_probs[r];
        }

        /* Health */
        rec->ess = output.ess;
        rec->outlier_fraction = output.outlier_fraction;

        /* Learned params */
        rec->learned_mu_vol = output.learned_mu_vol[output.dominant_regime];
        rec->learned_sigma_vol = output.learned_sigma_vol[output.dominant_regime];

        /* Flags */
        rec->regime_changed = output.regime_changed;
        rec->resampled = output.resampled;

        rec->latency_us = latency;
    }

    rbpf_ext_print_config(ext);
    rbpf_ext_destroy(ext);
}

/*─────────────────────────────────────────────────────────────────────────────
 * WRITE CSV
 *───────────────────────────────────────────────────────────────────────────*/

static void write_csv(const char *filename, TickRecord *records, int n)
{
    FILE *f = fopen(filename, "w");
    if (!f) {
        fprintf(stderr, "Failed to open %s\n", filename);
        return;
    }

    fprintf(f, "tick,true_log_vol,true_vol,true_hypo,return,is_outlier,outlier_sigma,"
               "est_log_vol,est_vol,vol_std,est_regime,est_hypo,"
               "prob_r0,prob_r1,prob_r2,prob_r3,"
               "ess,outlier_frac,learned_mu,learned_sigma,"
               "regime_changed,resampled,latency_us\n");

    for (int t = 0; t < n; t++) {
        TickRecord *r = &records[t];
        fprintf(f, "%d,%.6f,%.6f,%d,%.8f,%d,%.1f,"
                   "%.6f,%.6f,%.6f,%d,%d,"
                   "%.4f,%.4f,%.4f,%.4f,"
                   "%.2f,%.4f,%.4f,%.4f,"
                   "%d,%d,%.2f\n",
                r->tick, r->true_log_vol, r->true_vol, r->true_hypothesis,
                r->return_val, r->is_outlier, r->outlier_sigma,
                r->est_log_vol, r->est_vol, r->vol_std, r->est_regime, r->est_hypothesis,
                r->regime_probs[0], r->regime_probs[1], r->regime_probs[2], r->regime_probs[3],
                r->ess, r->outlier_fraction, r->learned_mu_vol, r->learned_sigma_vol,
                r->regime_changed, r->resampled, r->latency_us);
    }

    fclose(f);
    printf("  Written: %s\n", filename);
}

/*─────────────────────────────────────────────────────────────────────────────
 * COMPUTE METRICS
 *───────────────────────────────────────────────────────────────────────────*/

typedef struct {
    double log_vol_rmse;
    double log_vol_mae;
    double vol_rmse;
    double hypothesis_accuracy;
    double transition_lag_avg;
    int false_crisis_count;
    int missed_crisis_count;
    int spurious_switches_on_outlier;
    int regime_stable_after_outlier;
    int total_outliers_in_non_crisis;
    double avg_ess;
    double min_ess;
    double log_vol_rmse_outliers;
    double log_vol_rmse_normal;
    double avg_outlier_frac_on_outliers;
    double avg_outlier_frac_on_normal;
    double avg_latency_us;
    double p99_latency_us;
    double max_latency_us;
    int total_resamples;
    int total_regime_changes;
} SummaryMetrics;

static int compare_double(const void *a, const void *b)
{
    double da = *(const double *)a;
    double db = *(const double *)b;
    return (da > db) - (da < db);
}

static void compute_metrics(TickRecord *records, SyntheticData *data, SummaryMetrics *m)
{
    int n = data->n_ticks;
    memset(m, 0, sizeof(SummaryMetrics));

    double sum_log_err2 = 0, sum_log_err = 0, sum_vol_err2 = 0;
    int hypo_correct = 0;
    double sum_ess = 0, min_ess = 1e9;

    double sum_log_err2_outlier = 0, sum_log_err2_normal = 0;
    int n_outlier = 0, n_normal = 0;
    double sum_outlier_frac_outlier = 0, sum_outlier_frac_normal = 0;

    double *latencies = (double *)malloc(n * sizeof(double));
    double max_latency = 0;

    for (int t = 0; t < n; t++) {
        TickRecord *r = &records[t];

        double log_err = r->est_log_vol - r->true_log_vol;
        double vol_err = r->est_vol - r->true_vol;

        sum_log_err += fabs(log_err);
        sum_log_err2 += log_err * log_err;
        sum_vol_err2 += vol_err * vol_err;

        if (r->est_hypothesis == r->true_hypothesis) hypo_correct++;

        if (r->true_hypothesis == HYPO_CRISIS && r->est_hypothesis != HYPO_CRISIS)
            m->missed_crisis_count++;
        if (r->true_hypothesis != HYPO_CRISIS && r->est_hypothesis == HYPO_CRISIS)
            m->false_crisis_count++;

        sum_ess += r->ess;
        if (r->ess < min_ess) min_ess = r->ess;

        if (r->resampled) m->total_resamples++;
        if (r->regime_changed) m->total_regime_changes++;

        if (data->is_outlier[t]) {
            sum_log_err2_outlier += log_err * log_err;
            sum_outlier_frac_outlier += r->outlier_fraction;
            n_outlier++;

            if (data->true_hypothesis[t] != HYPO_CRISIS) {
                m->total_outliers_in_non_crisis++;

                if (t > 0 &&
                    records[t-1].est_hypothesis != HYPO_CRISIS &&
                    r->est_hypothesis == HYPO_CRISIS) {
                    m->spurious_switches_on_outlier++;
                }

                if (t + 5 < n) {
                    int pre_regime = (t > 0) ? records[t-1].est_hypothesis : r->est_hypothesis;
                    int post_regime = records[t + 5].est_hypothesis;
                    if (pre_regime == post_regime) {
                        m->regime_stable_after_outlier++;
                    }
                }
            }
        } else {
            sum_log_err2_normal += log_err * log_err;
            sum_outlier_frac_normal += r->outlier_fraction;
            n_normal++;
        }

        latencies[t] = r->latency_us;
        if (r->latency_us > max_latency) max_latency = r->latency_us;
    }

    m->log_vol_rmse = sqrt(sum_log_err2 / n);
    m->log_vol_mae = sum_log_err / n;
    m->vol_rmse = sqrt(sum_vol_err2 / n);
    m->hypothesis_accuracy = (double)hypo_correct / n;
    m->avg_ess = sum_ess / n;
    m->min_ess = min_ess;

    if (n_outlier > 0) {
        m->log_vol_rmse_outliers = sqrt(sum_log_err2_outlier / n_outlier);
        m->avg_outlier_frac_on_outliers = sum_outlier_frac_outlier / n_outlier;
    }
    if (n_normal > 0) {
        m->log_vol_rmse_normal = sqrt(sum_log_err2_normal / n_normal);
        m->avg_outlier_frac_on_normal = sum_outlier_frac_normal / n_normal;
    }

    qsort(latencies, n, sizeof(double), compare_double);
    m->avg_latency_us = latencies[n / 2];
    m->p99_latency_us = latencies[(int)(0.99 * n)];
    m->max_latency_us = max_latency;

    free(latencies);
}

static double compute_transition_lag(TickRecord *records, SyntheticData *data)
{
    int n = data->n_ticks;
    int total_lag = 0;
    int n_transitions = 0;

    for (int t = 1; t < n; t++) {
        if (data->true_hypothesis[t] != data->true_hypothesis[t-1]) {
            int target = data->true_hypothesis[t];
            int lag = 0;
            for (int s = t; s < n && s < t + 200; s++) {
                if (records[s].est_hypothesis == target) break;
                lag++;
            }
            if (lag < 200) {
                total_lag += lag;
                n_transitions++;
            }
        }
    }

    return (n_transitions > 0) ? (double)total_lag / n_transitions : 0.0;
}

/*─────────────────────────────────────────────────────────────────────────────
 * PRINT RESULTS
 *───────────────────────────────────────────────────────────────────────────*/

static void print_results(SummaryMetrics *m, TickRecord *records, SyntheticData *data)
{
    double trans_lag = compute_transition_lag(records, data);

    printf("\n");
    printf("══════════════════════════════════════════════════════════════════════════════\n");
    printf("  Single RBPF Test Results (%d ticks, %d outliers)\n",
           data->n_ticks, data->n_outliers_injected);
    printf("══════════════════════════════════════════════════════════════════════════════\n");

    printf("\n  VOLATILITY ESTIMATION\n");
    printf("  ────────────────────────────────────────────────────────────────────────────\n");
    printf("    Log-Vol RMSE:           %.4f\n", m->log_vol_rmse);
    printf("    Log-Vol MAE:            %.4f\n", m->log_vol_mae);
    printf("    Vol RMSE:               %.4f\n", m->vol_rmse);
    printf("    RMSE on Outliers:       %.4f\n", m->log_vol_rmse_outliers);
    printf("    RMSE on Normal:         %.4f\n", m->log_vol_rmse_normal);

    printf("\n  REGIME DETECTION\n");
    printf("  ────────────────────────────────────────────────────────────────────────────\n");
    printf("    Hypothesis Accuracy:    %.1f%%\n", 100 * m->hypothesis_accuracy);
    printf("    Avg Transition Lag:     %.1f ticks\n", trans_lag);
    printf("    False Crisis Count:     %d\n", m->false_crisis_count);
    printf("    Missed Crisis Count:    %d\n", m->missed_crisis_count);
    printf("    Total Regime Changes:   %d\n", m->total_regime_changes);

    printf("\n  OUTLIER HANDLING (★ THE MONEY SHOT)\n");
    printf("  ────────────────────────────────────────────────────────────────────────────\n");
    printf("    Spurious Crisis on Outlier:     %d\n", m->spurious_switches_on_outlier);
    printf("    Regime Stable After Outlier:    %d / %d\n",
           m->regime_stable_after_outlier, m->total_outliers_in_non_crisis);
    printf("    Avg Outlier Frac (on outliers): %.2f\n", m->avg_outlier_frac_on_outliers);
    printf("    Avg Outlier Frac (on normal):   %.4f\n", m->avg_outlier_frac_on_normal);

    printf("\n  PARTICLE HEALTH\n");
    printf("  ────────────────────────────────────────────────────────────────────────────\n");
    printf("    Avg ESS:                %.1f\n", m->avg_ess);
    printf("    Min ESS:                %.1f\n", m->min_ess);
    printf("    Total Resamples:        %d\n", m->total_resamples);

    printf("\n  TIMING\n");
    printf("  ────────────────────────────────────────────────────────────────────────────\n");
    printf("    Median Latency:         %.2f us\n", m->avg_latency_us);
    printf("    P99 Latency:            %.2f us\n", m->p99_latency_us);
    printf("    Max Latency:            %.2f us\n", m->max_latency_us);

    printf("\n  PER-SCENARIO ACCURACY\n");
    printf("  ────────────────────────────────────────────────────────────────────────────\n");
    printf("    %-24s %10s\n", "Scenario", "Accuracy");

    for (int s = 0; s < data->n_scenarios; s++) {
        int start = data->scenario_starts[s];
        int end = (s + 1 < data->n_scenarios) ? data->scenario_starts[s+1] : data->n_ticks;
        int count = end - start;
        int correct = 0;

        for (int t = start; t < end; t++) {
            if (records[t].est_hypothesis == data->true_hypothesis[t])
                correct++;
        }

        printf("    %-24s %9.1f%%\n", data->scenario_names[s], 100.0 * correct / count);
    }

    printf("══════════════════════════════════════════════════════════════════════════════\n");
}

/*─────────────────────────────────────────────────────────────────────────────
 * MAIN
 *───────────────────────────────────────────────────────────────────────────*/

int main(int argc, char **argv)
{
    int seed = 42;
    const char *csv_file = "single_rbpf_results.csv";

    if (argc > 1) seed = atoi(argv[1]);
    if (argc > 2) csv_file = argv[2];

    init_timer();

    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  Single RBPF Test (Storvik + Adaptive Forgetting + OCSN)\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  Seed: %d\n", seed);
    printf("  Output: %s\n\n", csv_file);

    /* Generate data */
    printf("Generating synthetic data...\n");
    SyntheticData *data = generate_test_data(seed);
    printf("  Ticks: %d\n", data->n_ticks);
    printf("  Scenarios: %d\n", data->n_scenarios);
    printf("  Outliers: %d\n\n", data->n_outliers_injected);

    /* Run test */
    printf("Running Single RBPF...\n");
    TickRecord *records = (TickRecord *)calloc(data->n_ticks, sizeof(TickRecord));
    double total_time, max_latency;
    run_single_rbpf(data, records, &total_time, &max_latency);
    printf("  Total time: %.2f ms\n", total_time / 1000.0);

    /* Compute metrics */
    SummaryMetrics metrics;
    compute_metrics(records, data, &metrics);

    /* Write CSV */
    printf("\nWriting CSV...\n");
    write_csv(csv_file, records, data->n_ticks);

    /* Print results */
    print_results(&metrics, records, data);

    /* Cleanup */
    free(records);
    free_synthetic_data(data);

    return 0;
}
