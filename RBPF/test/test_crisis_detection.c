/*=============================================================================
 * Crisis Detection Test
 *
 * Tests Hawkes + SR crisis detector against ground truth crisis periods.
 *
 * Ground Truth Crisis Periods:
 *   - Sudden Crisis:    2500-2999 (500 ticks)
 *   - Crisis Persist:   3000-3999 (1000 ticks, continuation)
 *   - Flash Crash:      5350-5410 (60 ticks)
 *
 * Metrics:
 *   - Detection latency (ticks from crisis start to detection)
 *   - False positive rate (detections outside crisis)
 *   - True positive rate (coverage of actual crisis)
 *   - Exit latency (ticks from crisis end to exit detection)
 *
 *===========================================================================*/

#include "rbpf_ksc_param_integration.h"
#include "rbpf_ext_crisis.h"
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
#else
#include <sys/time.h>
static void init_timer(void) {}
#endif

/*─────────────────────────────────────────────────────────────────────────────
 * PCG32 RNG (same as test_single_rbpf.c)
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
    return (xorshifted >> rot) | (xorshifted << ((32u - rot) & 31u));
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
 * HYPOTHESIS PARAMETERS (same as test_single_rbpf.c)
 *───────────────────────────────────────────────────────────────────────────*/

typedef enum
{
    HYPO_CALM = 0,
    HYPO_TREND = 1,
    HYPO_CRISIS = 2,
    N_HYPOTHESES = 3
} Hypothesis;

typedef struct
{
    double mu_vol;
    double phi;
    double sigma_eta;
    double vol_approx;
} HypothesisParams;

static const HypothesisParams TRUE_PARAMS[N_HYPOTHESES] = {
    {.mu_vol = -5.0, .phi = 0.995, .sigma_eta = 0.08, .vol_approx = 0.007},
    {.mu_vol = -3.5, .phi = 0.95, .sigma_eta = 0.20, .vol_approx = 0.030},
    {.mu_vol = -1.5, .phi = 0.85, .sigma_eta = 0.50, .vol_approx = 0.220}};

/*─────────────────────────────────────────────────────────────────────────────
 * SYNTHETIC DATA (same structure as test_single_rbpf.c)
 *───────────────────────────────────────────────────────────────────────────*/

typedef struct
{
    double *returns;
    double *true_log_vol;
    double *true_vol;
    int *true_hypothesis;
    int *is_outlier;
    int n_ticks;
    int n_outliers_injected;
} SyntheticData;

static void inject_outlier(SyntheticData *data, int t, double target_sigma, pcg32_t *rng)
{
    double vol = data->true_vol[t];
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
    data->true_log_vol = (double *)malloc(n * sizeof(double));
    data->true_vol = (double *)malloc(n * sizeof(double));
    data->true_hypothesis = (int *)malloc(n * sizeof(int));
    data->is_outlier = (int *)calloc(n, sizeof(int));

    pcg32_t rng = {seed * 12345ULL + 1, seed * 67890ULL | 1};

    double log_vol = TRUE_PARAMS[HYPO_CALM].mu_vol;
    int t = 0;

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
        data->true_vol[t] = vol;                                                              \
        data->true_hypothesis[t] = (H);                                                       \
    } while (0)

    /* Scenario 1: Extended Calm (0-1499) */
    for (; t < 1500; t++)
    {
        EVOLVE_STATE(HYPO_CALM);
    }
    inject_outlier(data, 500, 6.0, &rng);
    inject_outlier(data, 1200, 8.0, &rng);

    /* Scenario 2: Slow Trend (1500-2499) */
    for (; t < 2500; t++)
    {
        Hypothesis h = (t < 1800) ? HYPO_CALM : HYPO_TREND;
        EVOLVE_STATE(h);
    }

    /* Scenario 3: Sudden Crisis (2500-2999) */
    for (; t < 3000; t++)
    {
        EVOLVE_STATE(HYPO_CRISIS);
    }
    inject_outlier(data, 2510, 8.0, &rng);
    inject_outlier(data, 2530, 10.0, &rng);
    inject_outlier(data, 2560, 12.0, &rng);
    inject_outlier(data, 2650, 9.0, &rng);
    inject_outlier(data, 2800, 11.0, &rng);

    /* Scenario 4: Crisis Persistence (3000-3999) */
    for (; t < 4000; t++)
    {
        EVOLVE_STATE(HYPO_CRISIS);
    }
    inject_outlier(data, 3200, 10.0, &rng);
    inject_outlier(data, 3500, 15.0, &rng);
    inject_outlier(data, 3800, 12.0, &rng);

    /* Scenario 5: Recovery (4000-5199) */
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
    for (; t < 5700; t++)
    {
        Hypothesis h;
        if (t >= 5350 && t < 5410)
            h = HYPO_CRISIS;
        else
            h = HYPO_CALM;
        EVOLVE_STATE(h);
    }
    inject_outlier(data, 5380, 12.0, &rng);

    /* Scenario 7: Choppy (5700-7999) */
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

    return data;
}

static void free_synthetic_data(SyntheticData *data)
{
    if (!data)
        return;
    free(data->returns);
    free(data->true_log_vol);
    free(data->true_vol);
    free(data->true_hypothesis);
    free(data->is_outlier);
    free(data);
}

/*─────────────────────────────────────────────────────────────────────────────
 * GROUND TRUTH CRISIS PERIODS
 *───────────────────────────────────────────────────────────────────────────*/

typedef struct
{
    int start;
    int end;
    const char *name;
} CrisisPeriod;

#define N_CRISIS_PERIODS 2

static const CrisisPeriod GROUND_TRUTH_CRISES[N_CRISIS_PERIODS] = {
    {2500, 4400, "Main Crisis (Sudden + Persist + Early Recovery)"},
    {5350, 5410, "Flash Crash"},
};

static int is_in_crisis_ground_truth(int tick)
{
    for (int i = 0; i < N_CRISIS_PERIODS; i++)
    {
        if (tick >= GROUND_TRUTH_CRISES[i].start && tick < GROUND_TRUTH_CRISES[i].end)
        {
            return 1;
        }
    }
    return 0;
}

/*─────────────────────────────────────────────────────────────────────────────
 * CRISIS DETECTION METRICS
 *───────────────────────────────────────────────────────────────────────────*/

typedef struct
{
    /* Per-tick tracking */
    int *detected_crisis;    /* 1 if we think we're in crisis at tick t */
    float *hawkes_intensity; /* Raw Hawkes intensity */
    float *surprise;         /* RBPF surprise signal */

    /* Summary metrics */
    int true_positives;  /* Detected while actually in crisis */
    int false_positives; /* Detected while NOT in crisis */
    int true_negatives;  /* Not detected while NOT in crisis */
    int false_negatives; /* Not detected while actually in crisis */

    /* Timing metrics */
    int first_detection_tick;                      /* First tick we detected crisis */
    int detection_latency[N_CRISIS_PERIODS];       /* Ticks from start to detection */
    int exit_latency[N_CRISIS_PERIODS];            /* Ticks from end to exit detection */
    int detected_crisis_periods[N_CRISIS_PERIODS]; /* Did we detect each period? */

    /* Hawkes stats */
    float max_intensity;
    float avg_intensity_in_crisis;
    float avg_intensity_out_crisis;
} CrisisMetrics;

/*─────────────────────────────────────────────────────────────────────────────
 * SIMPLE THRESHOLD-BASED CRISIS DETECTOR (Placeholder)
 *
 * This is a simple detector for testing. Replace with your actual
 * Hawkes + SR detector when ready.
 *───────────────────────────────────────────────────────────────────────────*/

typedef struct
{
    /* State */
    int in_crisis;
    int ticks_in_state;

    /* Thresholds (tune these) */
    float hawkes_entry_threshold;   /* Enter crisis if intensity > this */
    float hawkes_exit_threshold;    /* Exit crisis if intensity < this */
    float surprise_entry_threshold; /* Enter if surprise > this */
    int min_confirm_ticks;          /* Min ticks above threshold to confirm */
    int min_exit_ticks;             /* Min ticks below threshold to exit */

    /* Tracking */
    int ticks_above_entry;
    int ticks_below_exit;
} SimpleCrisisDetector;

static void crisis_detector_init(SimpleCrisisDetector *det)
{
    memset(det, 0, sizeof(*det));

    /* Default thresholds - tune based on your Hawkes calibration */
    det->hawkes_entry_threshold = 0.15f;  /* Elevated intensity */
    det->hawkes_exit_threshold = 0.08f;   /* Back to baseline */
    det->surprise_entry_threshold = 3.0f; /* High surprise */
    det->min_confirm_ticks = 5;           /* Debounce */
    det->min_exit_ticks = 20;             /* Don't exit too fast */
}

static int crisis_detector_update(SimpleCrisisDetector *det,
                                  float hawkes_intensity,
                                  float surprise,
                                  RBPF_Extended *ext)
{
    int trigger = 0;

    /* Entry condition: Hawkes elevated OR high surprise */
    if (hawkes_intensity > det->hawkes_entry_threshold ||
        surprise > det->surprise_entry_threshold)
    {
        det->ticks_above_entry++;
        det->ticks_below_exit = 0;
    }
    else
    {
        det->ticks_above_entry = 0;
    }

    /* Exit condition: Hawkes back to normal */
    if (hawkes_intensity < det->hawkes_exit_threshold)
    {
        det->ticks_below_exit++;
    }
    else
    {
        det->ticks_below_exit = 0;
    }

    /* State machine */
    if (!det->in_crisis)
    {
        /* Check for entry */
        if (det->ticks_above_entry >= det->min_confirm_ticks)
        {
            det->in_crisis = 1;
            det->ticks_in_state = 0;
            rbpf_ext_enter_crisis(ext);
            trigger = 1; /* Rising edge */
        }
    }
    else
    {
        /* Check for exit */
        det->ticks_in_state++;
        if (det->ticks_below_exit >= det->min_exit_ticks)
        {
            det->in_crisis = 0;
            det->ticks_in_state = 0;
            rbpf_ext_exit_crisis(ext);
            trigger = -1; /* Falling edge */
        }
    }

    return trigger;
}

/*─────────────────────────────────────────────────────────────────────────────
 * RUN CRISIS DETECTION TEST
 *───────────────────────────────────────────────────────────────────────────*/

static void run_crisis_detection_test(SyntheticData *data, CrisisMetrics *metrics)
{
    const int N_PARTICLES = 512;
    const int N_REGIMES = 4;
    int n = data->n_ticks;

    /* Allocate tracking arrays */
    metrics->detected_crisis = (int *)calloc(n, sizeof(int));
    metrics->hawkes_intensity = (float *)calloc(n, sizeof(float));
    metrics->surprise = (float *)calloc(n, sizeof(float));

    /* Initialize metrics */
    metrics->first_detection_tick = -1;
    for (int i = 0; i < N_CRISIS_PERIODS; i++)
    {
        metrics->detection_latency[i] = -1;
        metrics->exit_latency[i] = -1;
        metrics->detected_crisis_periods[i] = 0;
    }

    /* Create RBPF */
    RBPF_Extended *ext = rbpf_ext_create(N_PARTICLES, N_REGIMES, RBPF_PARAM_STORVIK);
    rbpf_ext_enable_kl_tempering(ext);
    rbpf_ext_enable_smoothed_storvik(ext, 5);

    /* Regime params */
    rbpf_ext_set_regime_params(ext, 0, 0.0030f, -4.299f, 0.080f);
    rbpf_ext_set_regime_params(ext, 1, 0.0420f, -3.465f, 0.267f);
    rbpf_ext_set_regime_params(ext, 2, 0.0810f, -2.954f, 0.453f);
    rbpf_ext_set_regime_params(ext, 3, 0.1200f, -2.171f, 0.640f);

    /* Transition matrix */
    rbpf_real_t trans[16] = {
        0.920f, 0.056f, 0.020f, 0.004f,
        0.032f, 0.920f, 0.036f, 0.012f,
        0.012f, 0.036f, 0.920f, 0.032f,
        0.004f, 0.020f, 0.056f, 0.920f};
    rbpf_ext_build_transition_lut(ext, trans);

    /* Enable Hawkes integrator */
    HawkesIntegratorConfig cfg = hawkes_integrator_config_responsive();
    rbpf_ext_configure_hawkes(ext, &cfg);
    rbpf_ext_enable_apf_kick(ext, 1);
    rbpf_ext_set_apf_surprise_threshold(ext, 1.2f);

    /* Adaptive forgetting */
    rbpf_ext_enable_adaptive_forgetting_mode(ext, ADAPT_SIGNAL_REGIME);

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

    /* Crisis detector */
    SimpleCrisisDetector detector;
    crisis_detector_init(&detector);

    /* Accumulators for Hawkes stats */
    float sum_intensity_in = 0, sum_intensity_out = 0;
    int count_in = 0, count_out = 0;

    RBPF_KSC_Output output;

    printf("\n  Running crisis detection test...\n");
    printf("  ─────────────────────────────────────────────────────────────\n");

    for (int t = 0; t < n; t++)
    {
        memset(&output, 0, sizeof(output));

        /* Step RBPF */
        rbpf_ext_step(ext, (rbpf_real_t)data->returns[t], &output);

        /* Get Hawkes intensity */
        float intensity = rbpf_ext_get_hawkes_intensity(ext);
        float surprise = output.surprise;

        /* Store for analysis */
        metrics->hawkes_intensity[t] = intensity;
        metrics->surprise[t] = surprise;

        /* Update crisis detector */
        crisis_detector_update(&detector, intensity, surprise, ext);

        /* Record detection state */
        metrics->detected_crisis[t] = detector.in_crisis;

        /* Track max intensity */
        if (intensity > metrics->max_intensity)
        {
            metrics->max_intensity = intensity;
        }

        /* Ground truth comparison */
        int actual_crisis = is_in_crisis_ground_truth(t);

        if (actual_crisis && detector.in_crisis)
        {
            metrics->true_positives++;
            sum_intensity_in += intensity;
            count_in++;
        }
        else if (!actual_crisis && detector.in_crisis)
        {
            metrics->false_positives++;
        }
        else if (!actual_crisis && !detector.in_crisis)
        {
            metrics->true_negatives++;
            sum_intensity_out += intensity;
            count_out++;
        }
        else
        {
            metrics->false_negatives++;
            sum_intensity_out += intensity;
            count_out++;
        }

        /* Track first detection */
        if (detector.in_crisis && metrics->first_detection_tick < 0)
        {
            metrics->first_detection_tick = t;
        }

        /* Track detection latency per crisis period */
        for (int i = 0; i < N_CRISIS_PERIODS; i++)
        {
            const CrisisPeriod *cp = &GROUND_TRUTH_CRISES[i];

            /* Detection latency: first tick we're in detected crisis after start */
            if (t >= cp->start && t < cp->end &&
                detector.in_crisis &&
                metrics->detection_latency[i] < 0)
            {
                metrics->detection_latency[i] = t - cp->start;
                metrics->detected_crisis_periods[i] = 1;
            }

            /* Exit latency: first tick we exit after crisis ends */
            if (t >= cp->end && !detector.in_crisis &&
                metrics->exit_latency[i] < 0 &&
                metrics->detected_crisis_periods[i])
            {
                metrics->exit_latency[i] = t - cp->end;
            }
        }

        /* Print key events */
        if (t == 2500 || t == 3000 || t == 4000 || t == 5350 || t == 5410)
        {
            printf("  t=%4d: intensity=%.4f, surprise=%.2f, detected=%d, actual=%d\n",
                   t, intensity, surprise, detector.in_crisis, actual_crisis);
        }
    }

    /* Compute averages */
    metrics->avg_intensity_in_crisis = (count_in > 0) ? sum_intensity_in / count_in : 0;
    metrics->avg_intensity_out_crisis = (count_out > 0) ? sum_intensity_out / count_out : 0;

    /* Print Hawkes state at end */
    printf("\n  Final Hawkes state:\n");
    rbpf_ext_print_hawkes_state(ext);

    /* Print crisis mode state */
    rbpf_ext_print_crisis_state(ext);

    rbpf_ext_destroy(ext);
}

/*─────────────────────────────────────────────────────────────────────────────
 * PRINT CRISIS DETECTION RESULTS
 *───────────────────────────────────────────────────────────────────────────*/

static void print_crisis_results(CrisisMetrics *m, SyntheticData *data)
{
    int total = data->n_ticks;
    int total_crisis = m->true_positives + m->false_negatives;
    int total_non_crisis = m->true_negatives + m->false_positives;

    printf("\n");
    printf("══════════════════════════════════════════════════════════════════════════════\n");
    printf("  CRISIS DETECTION TEST RESULTS\n");
    printf("══════════════════════════════════════════════════════════════════════════════\n");

    printf("\n  CONFUSION MATRIX\n");
    printf("  ────────────────────────────────────────────────────────────────────────────\n");
    printf("                          Actual Crisis    Actual Normal\n");
    printf("    Detected Crisis:      %6d (TP)       %6d (FP)\n",
           m->true_positives, m->false_positives);
    printf("    Detected Normal:      %6d (FN)       %6d (TN)\n",
           m->false_negatives, m->true_negatives);

    printf("\n  RATES\n");
    printf("  ────────────────────────────────────────────────────────────────────────────\n");

    double sensitivity = (total_crisis > 0) ? 100.0 * m->true_positives / total_crisis : 0;
    double specificity = (total_non_crisis > 0) ? 100.0 * m->true_negatives / total_non_crisis : 0;
    double precision = (m->true_positives + m->false_positives > 0) ? 100.0 * m->true_positives / (m->true_positives + m->false_positives) : 0;
    double f1 = (precision + sensitivity > 0) ? 2 * precision * sensitivity / (precision + sensitivity) : 0;

    printf("    Sensitivity (TPR):    %.1f%% (of actual crises detected)\n", sensitivity);
    printf("    Specificity (TNR):    %.1f%% (of normal correctly identified)\n", specificity);
    printf("    Precision (PPV):      %.1f%% (of detections that were real)\n", precision);
    printf("    F1 Score:             %.1f%%\n", f1);

    printf("\n  PER-PERIOD DETECTION\n");
    printf("  ────────────────────────────────────────────────────────────────────────────\n");
    for (int i = 0; i < N_CRISIS_PERIODS; i++)
    {
        const CrisisPeriod *cp = &GROUND_TRUTH_CRISES[i];
        printf("    %s:\n", cp->name);
        printf("      Detected:        %s\n", m->detected_crisis_periods[i] ? "YES" : "NO");
        if (m->detected_crisis_periods[i])
        {
            printf("      Entry latency:   %d ticks (%.1f%% into crisis)\n",
                   m->detection_latency[i],
                   100.0 * m->detection_latency[i] / (cp->end - cp->start));
            if (m->exit_latency[i] >= 0)
            {
                printf("      Exit latency:    %d ticks after crisis ended\n",
                       m->exit_latency[i]);
            }
            else
            {
                printf("      Exit latency:    (still in detected crisis at end)\n");
            }
        }
    }

    printf("\n  HAWKES INTENSITY STATS\n");
    printf("  ────────────────────────────────────────────────────────────────────────────\n");
    printf("    Max intensity:        %.4f\n", m->max_intensity);
    printf("    Avg in crisis:        %.4f\n", m->avg_intensity_in_crisis);
    printf("    Avg out of crisis:    %.4f\n", m->avg_intensity_out_crisis);
    printf("    Ratio (in/out):       %.2fx\n",
           m->avg_intensity_out_crisis > 0 ? m->avg_intensity_in_crisis / m->avg_intensity_out_crisis : 0);

    printf("\n  FIRST DETECTION\n");
    printf("  ────────────────────────────────────────────────────────────────────────────\n");
    if (m->first_detection_tick >= 0)
    {
        printf("    First alarm at tick: %d\n", m->first_detection_tick);
        int first_crisis_start = GROUND_TRUTH_CRISES[0].start;
        if (m->first_detection_tick < first_crisis_start)
        {
            printf("    (FALSE ALARM - %d ticks before first real crisis)\n",
                   first_crisis_start - m->first_detection_tick);
        }
        else
        {
            printf("    (%d ticks into first crisis)\n",
                   m->first_detection_tick - first_crisis_start);
        }
    }
    else
    {
        printf("    No crisis ever detected!\n");
    }

    printf("══════════════════════════════════════════════════════════════════════════════\n");
}

/*─────────────────────────────────────────────────────────────────────────────
 * WRITE CRISIS CSV
 *───────────────────────────────────────────────────────────────────────────*/

static void write_crisis_csv(const char *filename, CrisisMetrics *m,
                             SyntheticData *data)
{
    FILE *f = fopen(filename, "w");
    if (!f)
    {
        fprintf(stderr, "Failed to open %s\n", filename);
        return;
    }

    fprintf(f, "tick,return,true_log_vol,true_hypo,is_outlier,"
               "hawkes_intensity,surprise,detected_crisis,actual_crisis\n");

    for (int t = 0; t < data->n_ticks; t++)
    {
        int actual = is_in_crisis_ground_truth(t);
        fprintf(f, "%d,%.8f,%.6f,%d,%d,%.6f,%.4f,%d,%d\n",
                t, data->returns[t], data->true_log_vol[t],
                data->true_hypothesis[t], data->is_outlier[t],
                m->hawkes_intensity[t], m->surprise[t],
                m->detected_crisis[t], actual);
    }

    fclose(f);
    printf("  Written: %s\n", filename);
}

/*─────────────────────────────────────────────────────────────────────────────
 * MAIN
 *───────────────────────────────────────────────────────────────────────────*/

int main(int argc, char **argv)
{
    int seed = 42;
    const char *csv_file = "crisis_detection_results.csv";

    if (argc > 1)
        seed = atoi(argv[1]);
    if (argc > 2)
        csv_file = argv[2];

    init_timer();

    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  Crisis Detection Test (Hawkes + SR)\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  Seed: %d\n", seed);
    printf("  Output: %s\n", csv_file);

    /* Generate data */
    printf("\n  Generating synthetic data...\n");
    SyntheticData *data = generate_test_data(seed);
    printf("  Ticks: %d\n", data->n_ticks);
    printf("  Outliers: %d\n", data->n_outliers_injected);

    /* Show ground truth */
    printf("\n  Ground Truth Crisis Periods:\n");
    for (int i = 0; i < N_CRISIS_PERIODS; i++)
    {
        printf("    [%d] %s: ticks %d-%d (%d ticks)\n",
               i, GROUND_TRUTH_CRISES[i].name,
               GROUND_TRUTH_CRISES[i].start, GROUND_TRUTH_CRISES[i].end,
               GROUND_TRUTH_CRISES[i].end - GROUND_TRUTH_CRISES[i].start);
    }

    /* Run test */
    CrisisMetrics metrics;
    memset(&metrics, 0, sizeof(metrics));
    run_crisis_detection_test(data, &metrics);

    /* Print results */
    print_crisis_results(&metrics, data);

    /* Write CSV for plotting */
    printf("\n  Writing CSV...\n");
    write_crisis_csv(csv_file, &metrics, data);

    /* Cleanup */
    free(metrics.detected_crisis);
    free(metrics.hawkes_intensity);
    free(metrics.surprise);
    free_synthetic_data(data);

    return 0;
}