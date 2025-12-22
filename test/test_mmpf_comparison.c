/*=============================================================================
 * MMPF vs Single RBPF Comparison Test
 *
 * Compares:
 *   1. Single RBPF (Storvik + Adaptive Forgetting + Robust OCSN) - Best single filter
 *   2. MMPF Full (Student-t + OCSN + Swim Lanes + Dirichlet + Adaptive Forgetting)
 *
 * Goal: Does multi-hypothesis beat a well-tuned single filter?
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
 * Requires: mmpf_core.c, mmpf_api.c, rbpf_ksc.c, rbpf_param_learn.c
 *
 *   cmake --build . --config Release --target test_mmpf_comparison
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * USAGE
 * ═══════════════════════════════════════════════════════════════════════════
 *
 *   test_mmpf_comparison [seed] [output_dir]
 *
 *===========================================================================*/

#include "mmpf_rocks.h"
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
 *
 * These define what each hypothesis "means" in terms of volatility dynamics.
 * MMPF should learn to detect which hypothesis best explains current data.
 *───────────────────────────────────────────────────────────────────────────*/

typedef enum
{
    HYPO_CALM = 0,
    HYPO_TREND = 1,
    HYPO_CRISIS = 2,
    N_HYPOTHESES = 3
} Hypothesis;

static const char *hypothesis_names[] = {"CALM", "TREND", "CRISIS"};

/*
 * Ground truth parameters for data generation
 * These roughly match MMPF swim lane centers
 */
typedef struct
{
    double mu_vol;     /* Long-run mean of log-vol */
    double phi;        /* Persistence */
    double sigma_eta;  /* Vol-of-vol */
    double vol_approx; /* Approximate realized vol (for reference) */
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
    double *true_log_vol;
    double *true_vol;
    int *true_hypothesis; /* Ground truth: CALM/TREND/CRISIS */
    int *is_outlier;
    double *outlier_sigma;
    int n_ticks;
    int scenario_starts[10];
    const char *scenario_names[10];
    int n_scenarios;
    int n_outliers_injected;
} SyntheticData;

/* Inject outlier at tick t */
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
        data->true_vol[t] = vol;                                                              \
        data->true_hypothesis[t] = (H);                                                       \
    } while (0)

    /*═══════════════════════════════════════════════════════════════════════
     * Scenario 1: Extended Calm (0-1499)
     *
     * 1500 ticks of CALM with 2 outliers (6σ, 8σ)
     * Tests: False positive rate, calm dominance, outlier rejection
     *═══════════════════════════════════════════════════════════════════════*/
    data->scenario_starts[0] = 0;
    data->scenario_names[0] = "Extended Calm";
    data->n_scenarios = 1;

    for (; t < 1500; t++)
    {
        EVOLVE_STATE(HYPO_CALM);
    }
    inject_outlier(data, 500, 6.0, &rng);
    inject_outlier(data, 1200, 8.0, &rng);

    /*═══════════════════════════════════════════════════════════════════════
     * Scenario 2: Slow Trend (1500-2499)
     *
     * Gradual transition CALM → TREND over 1000 ticks
     * Tests: Transition detection, how quickly MMPF catches the drift
     *═══════════════════════════════════════════════════════════════════════*/
    data->scenario_starts[1] = 1500;
    data->scenario_names[1] = "Slow Trend";
    data->n_scenarios = 2;

    for (; t < 2500; t++)
    {
        /* Gradual transition: CALM for first 300, then TREND */
        Hypothesis h = (t < 1800) ? HYPO_CALM : HYPO_TREND;
        EVOLVE_STATE(h);
    }

    /*═══════════════════════════════════════════════════════════════════════
     * Scenario 3: Sudden Crisis (2500-2999)
     *
     * Jump from TREND to CRISIS with 5 fat-tail outliers (8-12σ)
     * Tests: Fast crisis detection, OCSN protection
     *═══════════════════════════════════════════════════════════════════════*/
    data->scenario_starts[2] = 2500;
    data->scenario_names[2] = "Sudden Crisis";
    data->n_scenarios = 3;

    for (; t < 3000; t++)
    {
        EVOLVE_STATE(HYPO_CRISIS);
    }
    inject_outlier(data, 2510, 8.0, &rng);
    inject_outlier(data, 2530, 10.0, &rng);
    inject_outlier(data, 2560, 12.0, &rng);
    inject_outlier(data, 2650, 9.0, &rng);
    inject_outlier(data, 2800, 11.0, &rng);

    /*═══════════════════════════════════════════════════════════════════════
     * Scenario 4: Crisis Persistence (3000-3999)
     *
     * Sustained CRISIS with 3 extreme outliers (10-15σ)
     * Tests: No premature exit from crisis, extreme outlier handling
     *═══════════════════════════════════════════════════════════════════════*/
    data->scenario_starts[3] = 3000;
    data->scenario_names[3] = "Crisis Persist";
    data->n_scenarios = 4;

    for (; t < 4000; t++)
    {
        EVOLVE_STATE(HYPO_CRISIS);
    }
    inject_outlier(data, 3200, 10.0, &rng);
    inject_outlier(data, 3500, 15.0, &rng); /* EXTREME */
    inject_outlier(data, 3800, 12.0, &rng);

    /*═══════════════════════════════════════════════════════════════════════
     * Scenario 5: Recovery (4000-5199)
     *
     * CRISIS → TREND → CALM over 1200 ticks
     * Tests: Smooth de-escalation, not getting stuck in crisis
     *═══════════════════════════════════════════════════════════════════════*/
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

    /*═══════════════════════════════════════════════════════════════════════
     * Scenario 6: Flash Crash (5200-5699)
     *
     * CALM → CRISIS (60 ticks) → CALM with 12σ outlier at peak
     * Tests: Fast in, fast out, flash crash detection
     *═══════════════════════════════════════════════════════════════════════*/
    data->scenario_starts[5] = 5200;
    data->scenario_names[5] = "Flash Crash";
    data->n_scenarios = 6;

    for (; t < 5700; t++)
    {
        Hypothesis h;
        if (t >= 5350 && t < 5410)
            h = HYPO_CRISIS; /* 60-tick flash crash */
        else
            h = HYPO_CALM;
        EVOLVE_STATE(h);
    }
    inject_outlier(data, 5380, 12.0, &rng); /* Peak of flash crash */

    /*═══════════════════════════════════════════════════════════════════════
     * Scenario 7: Choppy (5700-7999)
     *
     * Random switching between all hypotheses
     * Tests: General robustness, adaptation speed
     *═══════════════════════════════════════════════════════════════════════*/
    data->scenario_starts[6] = 5700;
    data->scenario_names[6] = "Choppy";
    data->n_scenarios = 7;

    Hypothesis current_h = HYPO_TREND;
    int next_switch = 5700 + 80 + (int)(pcg32_double(&rng) * 120);

    for (; t < 8000; t++)
    {
        if (t >= next_switch)
        {
            /* Random walk through hypotheses */
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
    free(data->outlier_sigma);
    free(data);
}

/*─────────────────────────────────────────────────────────────────────────────
 * TICK RECORD
 *───────────────────────────────────────────────────────────────────────────*/

typedef struct
{
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

    /* Hypothesis/Regime */
    int est_hypothesis;           /* Dominant for MMPF, regime for single RBPF */
    double hypothesis_weights[3]; /* MMPF only */

    /* Health */
    double ess;             /* Single RBPF ESS or MMPF dominant ESS */
    double ess_per_hypo[3]; /* MMPF only */
    double min_weight;      /* MMPF: minimum hypothesis weight (Dirichlet check) */

    /* Robustness */
    double outlier_fraction;

    /* Learned params (for dominant hypothesis/regime) */
    double learned_mu_vol;
    double learned_sigma_vol;

    /* Timing */
    double latency_us;

} TickRecord;

/*─────────────────────────────────────────────────────────────────────────────
 * TEST MODES
 *───────────────────────────────────────────────────────────────────────────*/

typedef enum
{
    MODE_SINGLE_RBPF = 0, /* Best single RBPF: Storvik + Forgetting + OCSN */
    MODE_MMPF_FULL,       /* Full MMPF stack */
    NUM_MODES
} TestMode;

static const char *mode_names[] = {"SingleRBPF", "MMPF_Full"};
static const char *csv_names[] = {"single_rbpf.csv", "mmpf_full.csv"};

/*─────────────────────────────────────────────────────────────────────────────
 * MAP RBPF REGIME TO HYPOTHESIS
 *
 * Single RBPF uses 4 KSC regimes. We map them to 3 hypotheses for comparison.
 *───────────────────────────────────────────────────────────────────────────*/

static int rbpf_regime_to_hypothesis(int regime)
{
    /* R0 → CALM, R1 → CALM/TREND boundary, R2 → TREND, R3 → CRISIS */
    if (regime <= 0)
        return HYPO_CALM;
    if (regime == 1)
        return HYPO_CALM; /* Low-ish vol */
    if (regime == 2)
        return HYPO_TREND;
    return HYPO_CRISIS;
}

/*─────────────────────────────────────────────────────────────────────────────
 * RUN SINGLE RBPF TEST
 *───────────────────────────────────────────────────────────────────────────*/

static void run_single_rbpf(SyntheticData *data, TickRecord *records,
                            double *total_time, double *max_latency)
{

    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  RBPF with Dirichlet Transition Learning Example\n");
    printf("═══════════════════════════════════════════════════════════════\n\n");

    const int N_PARTICLES = 512;
    const int N_REGIMES = 4;

    /* Create extended RBPF with full stack */
    RBPF_Extended *ext = rbpf_ext_create(N_PARTICLES, N_REGIMES, RBPF_PARAM_STORVIK);


    /* Enable PARIS smoothed Storvik (L=50 tick lag) */

    rbpf_ext_enable_smoothed_storvik(ext, 5);


    /* ═══════════════════════════════════════════════════════════════════════════
     * BEST VOL RMSE CONFIG
     *
     * μ_calm=-4.50  μ_crisis=-2.00
     * σ_calm=0.080  σ_ratio=8.0
     * θ_calm=0.0030  θ_ratio=40.0
     * stickiness=0.92  λ_calm=0.9990
     *
     * Results: Vol RMSE=0.136, Hypo Acc=64.2%, Trans Lag=12.6, FC=138
     * ═══════════════════════════════════════════════════════════════════════════*/

    /* Regime params (θ, μ, σ) - linearly interpolated */
    rbpf_ext_set_regime_params(ext, 0, 0.0030f, -4.50f, 0.080f); /* Calm */
    rbpf_ext_set_regime_params(ext, 1, 0.0420f, -3.67f, 0.267f); /* Mild */
    rbpf_ext_set_regime_params(ext, 2, 0.0810f, -2.83f, 0.453f); /* Trend */
    rbpf_ext_set_regime_params(ext, 3, 0.1200f, -2.00f, 0.640f); /* Crisis */

    /* Transition matrix (stickiness=0.92) */
    rbpf_real_t trans[16] = {
        0.920f, 0.056f, 0.020f, 0.004f,
        0.032f, 0.920f, 0.036f, 0.012f,
        0.012f, 0.036f, 0.920f, 0.032f,
        0.004f, 0.020f, 0.056f, 0.920f};
    rbpf_ext_build_transition_lut(ext, trans);

    /* ═══════════════════════════════════════════════════════════════════════
     * ENABLE DIRICHLET TRANSITION LEARNING
     *
     * IMPORTANT: Call set_transition_learning_params() to re-initialize
     * the geometry-aware prior with the CORRECT mu_vol values.
     * ═══════════════════════════════════════════════════════════════════════*/
    /*
    rbpf_ksc_set_transition_learning_params(ext->rbpf,
                                            30.0f,   // stickiness 
                                            1.0f,    // distance_scale 
                                            0.999f); // gamma 
    rbpf_ksc_enable_transition_learning(ext->rbpf, 1);
    */

/* Enable adaptive forgetting in REGIME mode (uses fixed λ, no surprise modulation) */
rbpf_ext_enable_adaptive_forgetting_mode(ext, ADAPT_SIGNAL_REGIME);
//rbpf_ext_enable_adaptive_forgetting(ext); 

/* Set YOUR tuned λ values (not the defaults). Delete these if you want to make it completely adaptive 
rbpf_ext_set_regime_lambda(ext, 0, 0.9990f); 
rbpf_ext_set_regime_lambda(ext, 1, 0.9970f);
rbpf_ext_set_regime_lambda(ext, 2, 0.9950f);
rbpf_ext_set_regime_lambda(ext, 3, 0.9930f);
*/


/* Enable circuit breaker */
rbpf_ext_enable_circuit_breaker(ext, 0.999, 100);

    /* Enable Robust OCSN */
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

    for (int t = 0; t < n; t++)
    {
        memset(&output, 0, sizeof(output));

        double t_start = get_time_us();
        rbpf_ext_step(ext, (rbpf_real_t)data->returns[t], &output);
        double t_end = get_time_us();

        double latency = t_end - t_start;
        *total_time += latency;
        if (latency > *max_latency)
            *max_latency = latency;

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

        /* Map regime to hypothesis */
        rec->est_hypothesis = rbpf_regime_to_hypothesis(output.dominant_regime);
        rec->hypothesis_weights[0] = output.regime_probs[0] + output.regime_probs[1];
        rec->hypothesis_weights[1] = output.regime_probs[2];
        rec->hypothesis_weights[2] = output.regime_probs[3];

        /* Health */
        rec->ess = output.ess;
        rec->min_weight = 0; /* N/A for single RBPF */

        /* Robustness */
        rec->outlier_fraction = output.outlier_fraction;

        /* Learned params */
        rec->learned_mu_vol = output.learned_mu_vol[output.dominant_regime];
        rec->learned_sigma_vol = output.learned_sigma_vol[output.dominant_regime];

        rec->latency_us = latency;
    }

    rbpf_ext_destroy(ext);
}

/*─────────────────────────────────────────────────────────────────────────────
 * RUN MMPF TEST
 *───────────────────────────────────────────────────────────────────────────*/

static void run_mmpf_full(SyntheticData *data, TickRecord *records,
                          double *total_time, double *max_latency)
{
    /* Use default config (includes all our architecture decisions) */
    MMPF_Config cfg = mmpf_config_defaults();

    /* Create MMPF */
    MMPF_ROCKS *mmpf = mmpf_create(&cfg);
    if (!mmpf)
    {
        fprintf(stderr, "Failed to create MMPF\n");
        return;
    }

    *total_time = 0.0;
    *max_latency = 0.0;

    MMPF_Output output;
    int n = data->n_ticks;

    for (int t = 0; t < n; t++)
    {
        memset(&output, 0, sizeof(output));

        double t_start = get_time_us();
        mmpf_step(mmpf, (rbpf_real_t)data->returns[t], &output);
        double t_end = get_time_us();

        double latency = t_end - t_start;
        *total_time += latency;
        if (latency > *max_latency)
            *max_latency = latency;

        TickRecord *rec = &records[t];
        rec->tick = t;

        /* Ground truth */
        rec->true_log_vol = data->true_log_vol[t];
        rec->true_vol = data->true_vol[t];
        rec->true_hypothesis = data->true_hypothesis[t];
        rec->return_val = data->returns[t];
        rec->is_outlier = data->is_outlier[t];
        rec->outlier_sigma = data->outlier_sigma[t];

        /* Estimates (using header field names) */
        rec->est_log_vol = output.log_volatility;
        rec->est_vol = output.volatility;
        rec->vol_std = output.volatility_std;

        /* Hypothesis */
        rec->est_hypothesis = output.dominant;
        for (int h = 0; h < 3; h++)
        {
            rec->hypothesis_weights[h] = output.weights[h];
        }

        /* Health (using header field names) */
        rec->ess = output.model_ess[output.dominant];
        for (int h = 0; h < 3; h++)
        {
            rec->ess_per_hypo[h] = output.model_ess[h];
        }

        /* Minimum weight (Dirichlet prior check) */
        double min_w = output.weights[0];
        for (int h = 1; h < 3; h++)
        {
            if (output.weights[h] < min_w)
                min_w = output.weights[h];
        }
        rec->min_weight = min_w;

        /* Robustness */
        rec->outlier_fraction = output.outlier_fraction;

        /* Learned params from dominant hypothesis */
        rec->learned_mu_vol = output.learned_mu_vol[output.dominant];
        rec->learned_sigma_vol = output.learned_sigma_eta[output.dominant];

        rec->latency_us = latency;
    }

    mmpf_destroy(mmpf);
}

/*─────────────────────────────────────────────────────────────────────────────
 * WRITE CSV
 *───────────────────────────────────────────────────────────────────────────*/

static void write_csv(const char *filename, TickRecord *records, int n, TestMode mode)
{
    FILE *f = fopen(filename, "w");
    if (!f)
    {
        fprintf(stderr, "Failed to open %s\n", filename);
        return;
    }

    /* Header */
    fprintf(f, "tick,true_log_vol,true_vol,true_hypo,return,is_outlier,outlier_sigma,"
               "est_log_vol,est_vol,vol_std,est_hypo,"
               "weight_calm,weight_trend,weight_crisis,"
               "ess,min_weight,outlier_frac,"
               "learned_mu,learned_sigma,latency_us\n");

    for (int t = 0; t < n; t++)
    {
        TickRecord *r = &records[t];
        fprintf(f, "%d,%.6f,%.6f,%d,%.8f,%d,%.1f,"
                   "%.6f,%.6f,%.6f,%d,"
                   "%.4f,%.4f,%.4f,"
                   "%.2f,%.4f,%.4f,"
                   "%.4f,%.4f,%.2f\n",
                r->tick, r->true_log_vol, r->true_vol, r->true_hypothesis,
                r->return_val, r->is_outlier, r->outlier_sigma,
                r->est_log_vol, r->est_vol, r->vol_std, r->est_hypothesis,
                r->hypothesis_weights[0], r->hypothesis_weights[1], r->hypothesis_weights[2],
                r->ess, r->min_weight, r->outlier_fraction,
                r->learned_mu_vol, r->learned_sigma_vol, r->latency_us);
    }

    fclose(f);
    printf("  Written: %s\n", filename);
}

/*─────────────────────────────────────────────────────────────────────────────
 * COMPUTE METRICS
 *───────────────────────────────────────────────────────────────────────────*/

typedef struct
{
    /* Volatility estimation */
    double log_vol_rmse;
    double log_vol_mae;
    double vol_rmse;

    /* Hypothesis detection */
    double hypothesis_accuracy;
    double transition_lag_avg; /* Ticks to detect true transition */
    int false_crisis_count;    /* Crisis dominant when true != CRISIS */
    int missed_crisis_count;   /* Not Crisis dominant when true == CRISIS */

    /* Outlier handling (THE MONEY SHOT) */
    int spurious_switches_on_outlier; /* Switched TO Crisis ON an outlier tick (BAD) */
    int regime_stable_after_outlier;  /* Regime same 5 ticks after outlier (GOOD) */
    int total_outliers_in_non_crisis; /* Outliers injected during CALM/TREND */

    /* Particle health */
    double avg_ess;
    double min_ess;
    double avg_min_weight; /* Dirichlet check: should be > 0 */

    /* Robustness */
    double log_vol_rmse_outliers;
    double log_vol_rmse_normal;
    double avg_outlier_frac_on_outliers;
    double avg_outlier_frac_on_normal;

    /* Timing */
    double avg_latency_us;
    double p99_latency_us;
    double max_latency_us;

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
    double sum_min_weight = 0;

    double sum_log_err2_outlier = 0, sum_log_err2_normal = 0;
    int n_outlier = 0, n_normal = 0;
    double sum_outlier_frac_outlier = 0, sum_outlier_frac_normal = 0;

    double *latencies = (double *)malloc(n * sizeof(double));
    double max_latency = 0;

    for (int t = 0; t < n; t++)
    {
        TickRecord *r = &records[t];

        double log_err = r->est_log_vol - r->true_log_vol;
        double vol_err = r->est_vol - r->true_vol;

        sum_log_err += fabs(log_err);
        sum_log_err2 += log_err * log_err;
        sum_vol_err2 += vol_err * vol_err;

        if (r->est_hypothesis == r->true_hypothesis)
            hypo_correct++;

        /* Crisis detection */
        if (r->true_hypothesis == HYPO_CRISIS && r->est_hypothesis != HYPO_CRISIS)
            m->missed_crisis_count++;
        if (r->true_hypothesis != HYPO_CRISIS && r->est_hypothesis == HYPO_CRISIS)
            m->false_crisis_count++;

        sum_ess += r->ess;
        if (r->ess < min_ess)
            min_ess = r->ess;
        sum_min_weight += r->min_weight;

        if (data->is_outlier[t])
        {
            sum_log_err2_outlier += log_err * log_err;
            sum_outlier_frac_outlier += r->outlier_fraction;
            n_outlier++;

            /*═══════════════════════════════════════════════════════════════
             * THE MONEY SHOT: Spurious Regime Switch on Outlier
             *
             * If an outlier in CALM/TREND causes us to switch TO Crisis,
             * that's a false alarm. The filter should:
             *   1. Absorb it via Student-t (fat tails)
             *   2. Reject it via OCSN (K → 0)
             *   3. NOT change its worldview based on a single spike
             *═══════════════════════════════════════════════════════════════*/
            if (data->true_hypothesis[t] != HYPO_CRISIS)
            {
                m->total_outliers_in_non_crisis++;

                /* Check if we switched TO Crisis on this tick */
                if (t > 0 &&
                    records[t - 1].est_hypothesis != HYPO_CRISIS &&
                    r->est_hypothesis == HYPO_CRISIS)
                {
                    m->spurious_switches_on_outlier++;
                }

                /* Check regime stability: same regime 5 ticks after outlier? */
                if (t + 5 < n)
                {
                    int pre_regime = (t > 0) ? records[t - 1].est_hypothesis : r->est_hypothesis;
                    int post_regime = records[t + 5].est_hypothesis;
                    if (pre_regime == post_regime)
                    {
                        m->regime_stable_after_outlier++;
                    }
                }
            }
        }
        else
        {
            sum_log_err2_normal += log_err * log_err;
            sum_outlier_frac_normal += r->outlier_fraction;
            n_normal++;
        }

        latencies[t] = r->latency_us;
        if (r->latency_us > max_latency)
            max_latency = r->latency_us;
    }

    m->log_vol_rmse = sqrt(sum_log_err2 / n);
    m->log_vol_mae = sum_log_err / n;
    m->vol_rmse = sqrt(sum_vol_err2 / n);
    m->hypothesis_accuracy = (double)hypo_correct / n;
    m->avg_ess = sum_ess / n;
    m->min_ess = min_ess;
    m->avg_min_weight = sum_min_weight / n;

    if (n_outlier > 0)
    {
        m->log_vol_rmse_outliers = sqrt(sum_log_err2_outlier / n_outlier);
        m->avg_outlier_frac_on_outliers = sum_outlier_frac_outlier / n_outlier;
    }
    if (n_normal > 0)
    {
        m->log_vol_rmse_normal = sqrt(sum_log_err2_normal / n_normal);
        m->avg_outlier_frac_on_normal = sum_outlier_frac_normal / n_normal;
    }

    qsort(latencies, n, sizeof(double), compare_double);
    m->avg_latency_us = latencies[n / 2]; /* Median */
    m->p99_latency_us = latencies[(int)(0.99 * n)];
    m->max_latency_us = max_latency;

    free(latencies);
}

/*─────────────────────────────────────────────────────────────────────────────
 * COMPUTE TRANSITION LAG
 *
 * Measures how many ticks it takes to detect a regime transition
 *───────────────────────────────────────────────────────────────────────────*/

static double compute_transition_lag(TickRecord *records, SyntheticData *data)
{
    int n = data->n_ticks;
    int total_lag = 0;
    int n_transitions = 0;

    for (int t = 1; t < n; t++)
    {
        /* Detect ground truth transition */
        if (data->true_hypothesis[t] != data->true_hypothesis[t - 1])
        {
            int target = data->true_hypothesis[t];

            /* Count ticks until filter catches up */
            int lag = 0;
            for (int s = t; s < n && s < t + 200; s++)
            {
                if (records[s].est_hypothesis == target)
                    break;
                lag++;
            }

            if (lag < 200)
            { /* Only count if detected within 200 ticks */
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

static void print_comparison(SummaryMetrics *metrics, TickRecord *records[NUM_MODES],
                             SyntheticData *data)
{
    printf("\n");
    printf("══════════════════════════════════════════════════════════════════════════════\n");
    printf("  MMPF vs Single RBPF Comparison (%d ticks, %d outliers)\n",
           data->n_ticks, data->n_outliers_injected);
    printf("══════════════════════════════════════════════════════════════════════════════\n");

    printf("\n%-32s %15s %15s\n", "Metric", "Single RBPF", "MMPF Full");
    printf("────────────────────────────────────────────────────────────────────────────\n");

    /* Volatility estimation */
    printf("%-32s %15.4f %15.4f\n", "Log-Vol RMSE",
           metrics[0].log_vol_rmse, metrics[1].log_vol_rmse);
    printf("%-32s %15.4f %15.4f\n", "Log-Vol MAE",
           metrics[0].log_vol_mae, metrics[1].log_vol_mae);
    printf("%-32s %15.4f %15.4f\n", "Vol RMSE",
           metrics[0].vol_rmse, metrics[1].vol_rmse);

    printf("────────────────────────────────────────────────────────────────────────────\n");

    /* Hypothesis detection */
    printf("%-32s %14.1f%% %14.1f%%\n", "Hypothesis Accuracy",
           100 * metrics[0].hypothesis_accuracy, 100 * metrics[1].hypothesis_accuracy);

    double lag0 = compute_transition_lag(records[0], data);
    double lag1 = compute_transition_lag(records[1], data);
    printf("%-32s %15.1f %15.1f\n", "Avg Transition Lag (ticks)", lag0, lag1);

    printf("%-32s %15d %15d\n", "False Crisis Count",
           metrics[0].false_crisis_count, metrics[1].false_crisis_count);
    printf("%-32s %15d %15d\n", "Missed Crisis Count",
           metrics[0].missed_crisis_count, metrics[1].missed_crisis_count);

    printf("────────────────────────────────────────────────────────────────────────────\n");

    /* Health */
    printf("%-32s %15.1f %15.1f\n", "Avg ESS",
           metrics[0].avg_ess, metrics[1].avg_ess);
    printf("%-32s %15.1f %15.1f\n", "Min ESS",
           metrics[0].min_ess, metrics[1].min_ess);
    printf("%-32s %15s %14.2f%%\n", "Avg Min Weight (Dirichlet)",
           "N/A", 100 * metrics[1].avg_min_weight);

    printf("────────────────────────────────────────────────────────────────────────────\n");

    /* Robustness */
    printf("%-32s %15.4f %15.4f\n", "RMSE on Outlier Ticks",
           metrics[0].log_vol_rmse_outliers, metrics[1].log_vol_rmse_outliers);
    printf("%-32s %15.4f %15.4f\n", "RMSE on Normal Ticks",
           metrics[0].log_vol_rmse_normal, metrics[1].log_vol_rmse_normal);
    printf("%-32s %15.2f %15.2f\n", "Outlier Frac (on outliers)",
           metrics[0].avg_outlier_frac_on_outliers, metrics[1].avg_outlier_frac_on_outliers);

    printf("────────────────────────────────────────────────────────────────────────────\n");
    printf("  ★ THE MONEY SHOT: Outlier Handling (lower is better)\n");
    printf("────────────────────────────────────────────────────────────────────────────\n");

    /* Spurious switches - THE KEY DIFFERENTIATOR */
    printf("%-32s %15d %15d\n", "Spurious Crisis on Outlier",
           metrics[0].spurious_switches_on_outlier, metrics[1].spurious_switches_on_outlier);
    printf("%-32s %13d/%d %13d/%d\n", "Regime Stable After Outlier",
           metrics[0].regime_stable_after_outlier, metrics[0].total_outliers_in_non_crisis,
           metrics[1].regime_stable_after_outlier, metrics[1].total_outliers_in_non_crisis);

    printf("────────────────────────────────────────────────────────────────────────────\n");

    /* Timing */
    printf("%-32s %15.2f %15.2f\n", "Median Latency (us)",
           metrics[0].avg_latency_us, metrics[1].avg_latency_us);
    printf("%-32s %15.2f %15.2f\n", "P99 Latency (us)",
           metrics[0].p99_latency_us, metrics[1].p99_latency_us);
    printf("%-32s %15.2f %15.2f\n", "Max Latency (us)",
           metrics[0].max_latency_us, metrics[1].max_latency_us);

    printf("══════════════════════════════════════════════════════════════════════════════\n");

    /* Per-scenario breakdown */
    printf("\nPER-SCENARIO HYPOTHESIS ACCURACY\n");
    printf("────────────────────────────────────────────────────────────────────────────\n");
    printf("%-24s %15s %15s\n", "Scenario", "Single RBPF", "MMPF Full");
    printf("────────────────────────────────────────────────────────────────────────────\n");

    for (int s = 0; s < data->n_scenarios; s++)
    {
        int start = data->scenario_starts[s];
        int end = (s + 1 < data->n_scenarios) ? data->scenario_starts[s + 1] : data->n_ticks;
        int count = end - start;

        int correct[NUM_MODES] = {0};
        for (int t = start; t < end; t++)
        {
            for (int m = 0; m < NUM_MODES; m++)
            {
                if (records[m][t].est_hypothesis == data->true_hypothesis[t])
                    correct[m]++;
            }
        }

        printf("%-24s %14.1f%% %14.1f%%\n", data->scenario_names[s],
               100.0 * correct[0] / count,
               100.0 * correct[1] / count);
    }
    printf("══════════════════════════════════════════════════════════════════════════════\n");
}

/*─────────────────────────────────────────────────────────────────────────────
 * MAIN
 *───────────────────────────────────────────────────────────────────────────*/

int main(int argc, char **argv)
{
    int seed = 42;
    const char *output_dir = ".";

    if (argc > 1)
        seed = atoi(argv[1]);
    if (argc > 2)
        output_dir = argv[2];

    init_timer();

    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  MMPF vs Single RBPF Comparison Test\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  Seed: %d\n", seed);
    printf("  Output: %s\n\n", output_dir);

    /* Generate data */
    printf("Generating synthetic data...\n");
    SyntheticData *data = generate_test_data(seed);
    printf("  Ticks: %d\n", data->n_ticks);
    printf("  Scenarios: %d\n", data->n_scenarios);
    printf("  Outliers: %d\n\n", data->n_outliers_injected);

    int n = data->n_ticks;
    TickRecord *records[NUM_MODES];
    SummaryMetrics metrics[NUM_MODES];
    double total_time[NUM_MODES], max_latency[NUM_MODES];

    /* Run Single RBPF */
    printf("Running Single RBPF (Storvik + OCSN)...\n");
    records[MODE_SINGLE_RBPF] = (TickRecord *)calloc(n, sizeof(TickRecord));
    run_single_rbpf(data, records[MODE_SINGLE_RBPF], &total_time[0], &max_latency[0]);
    compute_metrics(records[MODE_SINGLE_RBPF], data, &metrics[MODE_SINGLE_RBPF]);
    printf("  Total: %.2f ms\n\n", total_time[0] / 1000.0);

    /* Run MMPF Full */
    printf("Running MMPF Full...\n");
    records[MODE_MMPF_FULL] = (TickRecord *)calloc(n, sizeof(TickRecord));
    run_mmpf_full(data, records[MODE_MMPF_FULL], &total_time[1], &max_latency[1]);
    compute_metrics(records[MODE_MMPF_FULL], data, &metrics[MODE_MMPF_FULL]);
    printf("  Total: %.2f ms\n\n", total_time[1] / 1000.0);

    /* Write CSVs */
    printf("Writing CSV files...\n");
    for (int m = 0; m < NUM_MODES; m++)
    {
        char path[512];
        snprintf(path, sizeof(path), "%s/%s", output_dir, csv_names[m]);
        write_csv(path, records[m], n, (TestMode)m);
    }

    /* Print comparison */
    print_comparison(metrics, records, data);

    /* Summary */
    printf("\nKEY QUESTIONS ANSWERED:\n");
    printf("────────────────────────────────────────────────────────────────\n");

    double rmse_improvement = (metrics[0].log_vol_rmse - metrics[1].log_vol_rmse) / metrics[0].log_vol_rmse * 100;
    double accuracy_diff = (metrics[1].hypothesis_accuracy - metrics[0].hypothesis_accuracy) * 100;

    printf("1. Does MMPF improve volatility estimation?\n");
    printf("   → RMSE improvement: %.1f%%\n", rmse_improvement);

    printf("2. Does MMPF detect regimes better?\n");
    printf("   → Accuracy difference: %+.1f%%\n", accuracy_diff);

    printf("3. Is the Dirichlet prior working?\n");
    printf("   → Avg min weight: %.2f%% (should be > 0)\n", 100 * metrics[1].avg_min_weight);

    printf("4. What's the latency cost of MMPF?\n");
    printf("   → MMPF median latency: %.1f us (vs %.1f us single)\n",
           metrics[1].avg_latency_us, metrics[0].avg_latency_us);

    printf("\n★ 5. THE MONEY SHOT: Does MMPF resist false alarms on outliers?\n");
    printf("   → Spurious Crisis switches: Single=%d, MMPF=%d\n",
           metrics[0].spurious_switches_on_outlier, metrics[1].spurious_switches_on_outlier);
    printf("   → Regime stability after outlier: Single=%d/%d, MMPF=%d/%d\n",
           metrics[0].regime_stable_after_outlier, metrics[0].total_outliers_in_non_crisis,
           metrics[1].regime_stable_after_outlier, metrics[1].total_outliers_in_non_crisis);
    if (metrics[1].spurious_switches_on_outlier < metrics[0].spurious_switches_on_outlier)
    {
        printf("   ✓ MMPF handles outliers better (fewer false alarms)\n");
    }
    else if (metrics[1].spurious_switches_on_outlier == metrics[0].spurious_switches_on_outlier)
    {
        printf("   = Both handle outliers equally\n");
    }
    else
    {
        printf("   ✗ Single RBPF handles outliers better (investigate!)\n");
    }

    printf("════════════════════════════════════════════════════════════════\n");

    /* Cleanup */
    for (int m = 0; m < NUM_MODES; m++)
        free(records[m]);
    free_synthetic_data(data);

    return 0;
}