/**
 * @file test_rbpf_scenarios.c
 * @brief Realistic scenario tests for RBPF with Storvik + Robust OCSN
 *
 * Tests RBPF's regime detection + volatility tracking against:
 *   1. Flash crash (sudden vol spike, quick recovery)
 *   2. Fed announcement (scheduled event, vol crush after)
 *   3. Earnings gap (overnight gap, elevated vol)
 *   4. Liquidity crisis (persistent high vol, negative drift)
 *   5. Gradual regime shift (slow transition)
 *   6. Overnight gap (gap opening from news)
 *   7. Intraday pattern (lunch lull → power hour)
 *   8. Correlation spike (stress event)
 *   9. Oscillating regimes (low→high→low vol)
 *  10. Fat-tail stress (clustered extreme moves)
 *
 * Compares modes:
 *   - Baseline: True params, no learning
 *   - Storvik: Parameter learning, no forgetting
 *   - Storvik+Full: Learning + Forgetting + Robust OCSN
 *
 * Compile:
 *   icx -O3 -xHost -qopenmp test_rbpf_scenarios.c rbpf_ksc.c \
 *       rbpf_ksc_param_integration.c rbpf_param_learn.c rbpf_ocsn_robust.c \
 *       -o test_scenarios -qmkl -lm
 */

#include "rbpf_ksc.h"
#include "rbpf_ksc_param_integration.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <float.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <time.h>
#endif

/*============================================================================
 * CONFIGURATION
 *============================================================================*/

#define N_MONTE_CARLO 20    /* Simulations per scenario */
#define N_OBSERVATIONS 1500 /* Ticks per simulation */
#define N_PARTICLES 256     /* RBPF particles */
#define N_REGIMES 4         /* Volatility regimes */
#define WARMUP_TICKS 100    /* Warmup period */

/* True regime parameters (ground truth) */
static const rbpf_real_t TRUE_THETA[4] = {0.02f, 0.03f, 0.04f, 0.05f};
static const rbpf_real_t TRUE_MU_VOL[4] = {-4.6f, -3.5f, -2.5f, -1.2f};
static const rbpf_real_t TRUE_SIGMA_VOL[4] = {0.05f, 0.08f, 0.12f, 0.20f};

/*============================================================================
 * TEST MODES
 *============================================================================*/

typedef enum
{
    MODE_BASELINE = 0, /* True params, no learning */
    MODE_STORVIK,      /* Storvik learning, no forgetting, no Robust OCSN */
    MODE_STORVIK_FULL, /* Storvik + Forgetting + Robust OCSN */
    NUM_MODES
} TestMode;

static const char *mode_names[] = {
    "Baseline",
    "Storvik",
    "Storvik+Full"};

/*============================================================================
 * RNG (xorshift128+)
 *============================================================================*/

static uint64_t rng_s[2];

static void rng_seed(uint64_t seed)
{
    rng_s[0] = seed;
    rng_s[1] = seed ^ 0xDEADBEEFCAFEBABEULL;
    for (int i = 0; i < 20; i++)
    {
        rng_s[0] ^= rng_s[0] << 13;
        rng_s[0] ^= rng_s[0] >> 7;
        rng_s[0] ^= rng_s[0] << 17;
    }
}

static inline uint64_t rng_next(void)
{
    uint64_t s1 = rng_s[0];
    const uint64_t s0 = rng_s[1];
    rng_s[0] = s0;
    s1 ^= s1 << 23;
    rng_s[1] = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5);
    return rng_s[1] + s0;
}

static inline double randu(void)
{
    return (rng_next() >> 11) * (1.0 / 9007199254740992.0);
}

static inline double randn(void)
{
    double u1 = randu(), u2 = randu();
    if (u1 < 1e-10)
        u1 = 1e-10;
    return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}

static double rand_student_t(double df)
{
    double z = randn();
    double chi2 = 0.0;
    for (int i = 0; i < (int)df; i++)
    {
        double g = randn();
        chi2 += g * g;
    }
    return z / sqrt(chi2 / df);
}

/*============================================================================
 * TIMING
 *============================================================================*/

#ifdef _WIN32
static LARGE_INTEGER timer_freq;
static void init_timer(void) { QueryPerformanceFrequency(&timer_freq); }
static double get_time_us(void)
{
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart / timer_freq.QuadPart * 1e6;
}
#else
static void init_timer(void) {}
static double get_time_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}
#endif

/*============================================================================
 * STOCHASTIC VOLATILITY MODEL
 *============================================================================*/

typedef struct
{
    double drift;
    double mu_vol;
    double theta_vol;
    double sigma_vol;
    double rho;            /* Vol-return correlation */
    double jump_intensity; /* Jump probability per tick */
    double jump_mean;      /* Jump size mean */
    double jump_std;       /* Jump size std */
    double student_df;     /* 0 = Gaussian, >0 = Student-t df */
    double fat_tail_prob;  /* Probability of fat-tail event */
    double fat_tail_mult;  /* Multiplier for fat-tail (e.g., 8 = 8σ) */
} SVParams;

static SVParams sv_calm(void)
{
    return (SVParams){
        .drift = 0.0,
        .mu_vol = -4.6, /* Regime 0 */
        .theta_vol = 0.02,
        .sigma_vol = 0.05,
        .rho = -0.3,
        .jump_intensity = 0.0,
        .jump_mean = 0.0,
        .jump_std = 0.0,
        .student_df = 0,        /* Gaussian (0 = no student-t) */
        .fat_tail_prob = 0.001, /* 0.1% chance of fat tail */
        .fat_tail_mult = 6.0};
}

static SVParams sv_normal(void)
{
    return (SVParams){
        .drift = 0.0,
        .mu_vol = -3.5, /* Regime 1 */
        .theta_vol = 0.03,
        .sigma_vol = 0.08,
        .rho = -0.5,
        .jump_intensity = 0.0,
        .jump_mean = 0.0,
        .jump_std = 0.0,
        .student_df = 0, /* Gaussian */
        .fat_tail_prob = 0.002,
        .fat_tail_mult = 7.0};
}

static SVParams sv_elevated(void)
{
    return (SVParams){
        .drift = -0.0002,
        .mu_vol = -2.5, /* Regime 2 */
        .theta_vol = 0.04,
        .sigma_vol = 0.12,
        .rho = -0.6,
        .jump_intensity = 0.01,
        .jump_mean = 0.0,
        .jump_std = 0.15,
        .student_df = 5, /* Fat tails */
        .fat_tail_prob = 0.005,
        .fat_tail_mult = 8.0};
}

static SVParams sv_crisis(void)
{
    return (SVParams){
        .drift = -0.0005,
        .mu_vol = -2.3, /* Regime 2-3 boundary (~10% vol, not 30%!) */
        .theta_vol = 0.05,
        .sigma_vol = 0.15,
        .rho = -0.7,
        .jump_intensity = 0.02,
        .jump_mean = -0.01,
        .jump_std = 0.15,
        .student_df = 5,        /* df=5 is fat but not insane */
        .fat_tail_prob = 0.015, /* 1.5% outlier rate */
        .fat_tail_mult = 7.0    /* 7σ not 10σ */
    };
}

static void sv_step(double *log_vol, double *price, const SVParams *p,
                    double *ret_out, int *is_outlier, double *outlier_sigma)
{
    *is_outlier = 0;
    *outlier_sigma = 0.0;

    /* Correlated innovations */
    double z1 = randn();
    double z2 = p->rho * z1 + sqrt(1.0 - p->rho * p->rho) * randn();

    /* Optional Student-t for return innovation (df >= 4 to have finite kurtosis) */
    if (p->student_df >= 4.0)
    {
        double df = p->student_df;
        z1 = rand_student_t(df) / sqrt(df / (df - 2));
    }

    /* Volatility dynamics: mean-reverting log-vol */
    double new_log_vol = (1.0 - p->theta_vol) * (*log_vol) +
                         p->theta_vol * p->mu_vol +
                         p->sigma_vol * z2;

    /* Vol jumps */
    if (p->jump_intensity > 0 && randu() < p->jump_intensity)
    {
        new_log_vol += p->jump_mean + p->jump_std * randn();
    }

    /* Clamp log_vol to reasonable range to prevent filter explosion */
    if (new_log_vol < -6.0)
        new_log_vol = -6.0; /* ~0.25% vol floor */
    if (new_log_vol > -1.5)
        new_log_vol = -1.5; /* ~22% vol ceiling */

    *log_vol = new_log_vol;
    double vol = exp(*log_vol);
    double ret = p->drift + vol * z1;

    /* Fat-tail injection */
    if (p->fat_tail_prob > 0 && randu() < p->fat_tail_prob)
    {
        double sign = (randu() < 0.5) ? -1.0 : 1.0;
        double sigma_mult = p->fat_tail_mult + randn() * 1.0; /* Some randomness */
        if (sigma_mult < 5.0)
            sigma_mult = 5.0;
        if (sigma_mult > 12.0)
            sigma_mult = 12.0; /* Cap extreme outliers */
        ret = sign * vol * sigma_mult;
        *is_outlier = 1;
        *outlier_sigma = sigma_mult;
    }

    *price = (*price) * (1.0 + ret);
    *ret_out = ret;
}

static SVParams sv_interpolate(const SVParams *a, const SVParams *b, double t)
{
    SVParams p;
    p.drift = a->drift + t * (b->drift - a->drift);
    p.mu_vol = a->mu_vol + t * (b->mu_vol - a->mu_vol);
    p.theta_vol = a->theta_vol + t * (b->theta_vol - a->theta_vol);
    p.sigma_vol = a->sigma_vol + t * (b->sigma_vol - a->sigma_vol);
    p.rho = a->rho + t * (b->rho - a->rho);
    p.jump_intensity = a->jump_intensity + t * (b->jump_intensity - a->jump_intensity);
    p.jump_mean = a->jump_mean + t * (b->jump_mean - a->jump_mean);
    p.jump_std = a->jump_std + t * (b->jump_std - a->jump_std);
    p.student_df = a->student_df + t * (b->student_df - a->student_df);
    p.fat_tail_prob = a->fat_tail_prob + t * (b->fat_tail_prob - a->fat_tail_prob);
    p.fat_tail_mult = a->fat_tail_mult + t * (b->fat_tail_mult - a->fat_tail_mult);
    return p;
}

/*============================================================================
 * SCENARIO DEFINITIONS
 *============================================================================*/

typedef struct
{
    const char *name;
    const char *description;
    int changepoint;
    int transition_ticks; /* 0 = instant, >0 = gradual */
    SVParams before;
    SVParams after;
    int has_second_change;
    int second_changepoint;
    SVParams final;
    int expected_regime_before;
    int expected_regime_after;
} Scenario;

/* Scenario 1: Flash Crash */
static Scenario scenario_flash_crash(void)
{
    SVParams normal = sv_normal();
    SVParams crash = sv_crisis();
    crash.drift = -0.002;
    crash.fat_tail_prob = 0.04; /* High outlier rate during crash */
    crash.fat_tail_mult = 9.0;

    SVParams recovery = sv_elevated();
    recovery.mu_vol = -3.0;

    return (Scenario){
        .name = "Flash Crash",
        .description = "Sudden vol spike + outliers, quick recovery",
        .changepoint = 500,
        .transition_ticks = 0,
        .before = normal,
        .after = crash,
        .has_second_change = 1,
        .second_changepoint = 550,
        .final = recovery,
        .expected_regime_before = 1,
        .expected_regime_after = 2};
}

/* Scenario 2: Fed Announcement */
static Scenario scenario_fed_announcement(void)
{
    SVParams pre_fed = sv_normal();
    pre_fed.sigma_vol = 0.10; /* Vol-of-vol rising into event */

    SVParams spike = sv_elevated();
    spike.fat_tail_prob = 0.03;
    spike.fat_tail_mult = 8.0;

    SVParams crush = sv_calm();
    crush.mu_vol = -5.0; /* Vol crush after clarity */
    crush.sigma_vol = 0.03;

    return (Scenario){
        .name = "Fed Announcement",
        .description = "Event spike then vol crush",
        .changepoint = 600,
        .transition_ticks = 0,
        .before = pre_fed,
        .after = spike,
        .has_second_change = 1,
        .second_changepoint = 650,
        .final = crush,
        .expected_regime_before = 1,
        .expected_regime_after = 2};
}

/* Scenario 3: Earnings Surprise */
static Scenario scenario_earnings_gap(void)
{
    SVParams normal = sv_normal();
    SVParams gap = sv_elevated();
    gap.drift = 0.003; /* Gap up */
    gap.fat_tail_prob = 0.02;
    gap.fat_tail_mult = 9.0;

    return (Scenario){
        .name = "Earnings Surprise",
        .description = "Gap up with elevated post-earnings vol",
        .changepoint = 500,
        .transition_ticks = 5,
        .before = normal,
        .after = gap,
        .has_second_change = 0,
        .expected_regime_before = 1,
        .expected_regime_after = 2};
}

/* Scenario 4: Liquidity Crisis */
static Scenario scenario_liquidity_crisis(void)
{
    SVParams normal = sv_normal();
    SVParams crisis = sv_crisis();
    crisis.theta_vol = 0.02; /* Slow mean-reversion (persistent) */
    crisis.fat_tail_prob = 0.025;
    crisis.fat_tail_mult = 8.0;

    return (Scenario){
        .name = "Liquidity Crisis",
        .description = "Persistent elevated vol with clustered outliers",
        .changepoint = 400,
        .transition_ticks = 50,
        .before = normal,
        .after = crisis,
        .has_second_change = 0,
        .expected_regime_before = 1,
        .expected_regime_after = 2};
}

/* Scenario 5: Gradual Regime Shift */
static Scenario scenario_gradual_shift(void)
{
    SVParams calm = sv_calm();
    SVParams elevated = sv_elevated();

    return (Scenario){
        .name = "Gradual Regime Shift",
        .description = "Slow transition from calm to elevated",
        .changepoint = 300,
        .transition_ticks = 400,
        .before = calm,
        .after = elevated,
        .has_second_change = 0,
        .expected_regime_before = 0,
        .expected_regime_after = 2};
}

/* Scenario 6: Overnight Gap */
static Scenario scenario_overnight_gap(void)
{
    SVParams normal = sv_normal();
    SVParams gap = sv_elevated();
    gap.mu_vol = -2.8;
    gap.fat_tail_prob = 0.05; /* High outlier rate at open */
    gap.fat_tail_mult = 8.0;

    SVParams post_gap = sv_normal();
    post_gap.mu_vol = -3.2;

    return (Scenario){
        .name = "Overnight Gap",
        .description = "Gap opening then normalization",
        .changepoint = 500,
        .transition_ticks = 0,
        .before = normal,
        .after = gap,
        .has_second_change = 1,
        .second_changepoint = 530,
        .final = post_gap,
        .expected_regime_before = 1,
        .expected_regime_after = 2};
}

/* Scenario 7: Intraday Pattern */
static Scenario scenario_intraday_pattern(void)
{
    SVParams morning = sv_normal();
    SVParams lunch = sv_calm();
    lunch.mu_vol = -5.0;

    SVParams power_hour = sv_elevated();
    power_hour.fat_tail_prob = 0.01;

    return (Scenario){
        .name = "Intraday Pattern",
        .description = "Lunch lull then power hour",
        .changepoint = 400,
        .transition_ticks = 30,
        .before = morning,
        .after = lunch,
        .has_second_change = 1,
        .second_changepoint = 800,
        .final = power_hour,
        .expected_regime_before = 1,
        .expected_regime_after = 0};
}

/* Scenario 8: Correlation Spike (Stress Event) */
static Scenario scenario_correlation_spike(void)
{
    SVParams normal = sv_normal();
    normal.rho = -0.3;

    SVParams stress = sv_crisis();
    stress.rho = -0.8;
    stress.fat_tail_prob = 0.03;
    stress.fat_tail_mult = 8.0;

    return (Scenario){
        .name = "Correlation Spike",
        .description = "Stress event with high vol-return correlation",
        .changepoint = 500,
        .transition_ticks = 10,
        .before = normal,
        .after = stress,
        .has_second_change = 0,
        .expected_regime_before = 1,
        .expected_regime_after = 2};
}

/* Scenario 9: Oscillating Regimes */
static Scenario scenario_oscillating(void)
{
    SVParams low = sv_calm();
    SVParams high = sv_elevated();
    high.fat_tail_prob = 0.015;

    return (Scenario){
        .name = "Oscillating Regimes",
        .description = "Low→High→Low vol oscillation",
        .changepoint = 400,
        .transition_ticks = 0,
        .before = low,
        .after = high,
        .has_second_change = 1,
        .second_changepoint = 800,
        .final = low,
        .expected_regime_before = 0,
        .expected_regime_after = 2};
}

/* Scenario 10: Fat-Tail Stress Test */
static Scenario scenario_fat_tail_stress(void)
{
    SVParams normal = sv_normal();
    normal.fat_tail_prob = 0.002;

    SVParams stress = sv_elevated();
    stress.fat_tail_prob = 0.08; /* 8% outlier rate - challenging but survivable */
    stress.fat_tail_mult = 8.0;
    stress.student_df = 5;

    SVParams recovery = sv_normal();
    recovery.fat_tail_prob = 0.005;

    return (Scenario){
        .name = "Fat-Tail Stress",
        .description = "Clustered extreme moves (tests Robust OCSN)",
        .changepoint = 500,
        .transition_ticks = 0,
        .before = normal,
        .after = stress,
        .has_second_change = 1,
        .second_changepoint = 700,
        .final = recovery,
        .expected_regime_before = 1,
        .expected_regime_after = 2};
}

/*============================================================================
 * DATA GENERATION
 *============================================================================*/

typedef struct
{
    double *returns;
    double *true_vol;
    double *true_log_vol;
    int *true_regime;
    int *is_outlier;
    double *outlier_sigma;
    int n;
    int n_outliers;
} ScenarioData;

static int classify_log_vol_to_regime(double log_vol)
{
    /* Map log-vol to regime based on TRUE_MU_VOL boundaries:
     *   R0: μ=-4.6, boundary <= -4.05
     *   R1: μ=-3.5, -4.05 to -3.0
     *   R2: μ=-2.5, -3.0 to -1.85
     *   R3: μ=-1.2, > -1.85
     */
    if (log_vol <= -4.05)
        return 0;
    if (log_vol < -3.0)
        return 1;
    if (log_vol < -1.85)
        return 2;
    return 3;
}

static ScenarioData *generate_scenario_data(const Scenario *s, int n)
{
    ScenarioData *data = (ScenarioData *)malloc(sizeof(ScenarioData));
    data->n = n;
    data->returns = (double *)malloc(n * sizeof(double));
    data->true_vol = (double *)malloc(n * sizeof(double));
    data->true_log_vol = (double *)malloc(n * sizeof(double));
    data->true_regime = (int *)malloc(n * sizeof(int));
    data->is_outlier = (int *)malloc(n * sizeof(int));
    data->outlier_sigma = (double *)malloc(n * sizeof(double));
    data->n_outliers = 0;

    double price = 100.0;
    double log_vol = s->before.mu_vol;

    for (int t = 0; t < n; t++)
    {
        SVParams params;

        if (t < s->changepoint)
        {
            params = s->before;
        }
        else if (s->transition_ticks > 0 && t < s->changepoint + s->transition_ticks)
        {
            double frac = (double)(t - s->changepoint) / s->transition_ticks;
            params = sv_interpolate(&s->before, &s->after, frac);
        }
        else if (s->has_second_change && t >= s->second_changepoint)
        {
            params = s->final;
        }
        else
        {
            params = s->after;
        }

        double ret;
        int is_out;
        double out_sigma;
        sv_step(&log_vol, &price, &params, &ret, &is_out, &out_sigma);

        data->returns[t] = ret;
        data->true_vol[t] = exp(log_vol);
        data->true_log_vol[t] = log_vol;
        data->true_regime[t] = classify_log_vol_to_regime(log_vol);
        data->is_outlier[t] = is_out;
        data->outlier_sigma[t] = out_sigma;
        if (is_out)
            data->n_outliers++;
    }

    return data;
}

static void free_scenario_data(ScenarioData *data)
{
    if (data)
    {
        free(data->returns);
        free(data->true_vol);
        free(data->true_log_vol);
        free(data->true_regime);
        free(data->is_outlier);
        free(data->outlier_sigma);
        free(data);
    }
}

/*============================================================================
 * STATISTICS
 *============================================================================*/

typedef struct
{
    /* State estimation */
    double log_vol_rmse;
    double log_vol_rmse_outliers;
    double log_vol_rmse_normal;

    /* Regime tracking */
    double regime_accuracy;
    double regime_accuracy_post; /* After changepoint */

    /* ESS health */
    double avg_ess;
    double min_ess;
    int ess_collapse_count;

    /* Outlier handling */
    double avg_outlier_frac_on_outliers;
    int outlier_detections;

    /* Change detection */
    double mean_detection_delay;
    double detection_rate;

    /* Timing */
    double avg_latency_us;

    int n_runs;
    int n_outliers_total;
} ScenarioStats;

/*============================================================================
 * FILTER CREATION
 *============================================================================*/

static rbpf_real_t TRANS_MATRIX[16] = {
    0.95f, 0.04f, 0.01f, 0.00f,
    0.03f, 0.92f, 0.04f, 0.01f,
    0.01f, 0.04f, 0.92f, 0.03f,
    0.00f, 0.01f, 0.04f, 0.95f};

static RBPF_KSC *create_baseline_rbpf(void)
{
    RBPF_KSC *rbpf = rbpf_ksc_create(N_PARTICLES, N_REGIMES);
    for (int r = 0; r < N_REGIMES; r++)
    {
        rbpf_ksc_set_regime_params(rbpf, r, TRUE_THETA[r], TRUE_MU_VOL[r], TRUE_SIGMA_VOL[r]);
    }
    rbpf_ksc_build_transition_lut(rbpf, TRANS_MATRIX);
    rbpf_ksc_init(rbpf, TRUE_MU_VOL[0], 0.1f);
    return rbpf;
}

static RBPF_Extended *create_storvik_rbpf(int with_forgetting, int with_robust_ocsn)
{
    RBPF_Extended *ext = rbpf_ext_create(N_PARTICLES, N_REGIMES, RBPF_PARAM_STORVIK);

    for (int r = 0; r < N_REGIMES; r++)
    {
        rbpf_ext_set_regime_params(ext, r, TRUE_THETA[r], TRUE_MU_VOL[r], TRUE_SIGMA_VOL[r]);
    }
    rbpf_ext_build_transition_lut(ext, TRANS_MATRIX);

    if (with_forgetting)
    {
        param_learn_set_forgetting(&ext->storvik, 1, 0.997f);
        param_learn_set_regime_forgetting(&ext->storvik, 0, 0.999f);
        param_learn_set_regime_forgetting(&ext->storvik, 1, 0.998f);
        param_learn_set_regime_forgetting(&ext->storvik, 2, 0.996f);
        param_learn_set_regime_forgetting(&ext->storvik, 3, 0.993f);
    }
    else
    {
        param_learn_set_forgetting(&ext->storvik, 0, 1.0f);
    }

    if (with_robust_ocsn)
    {
        ext->robust_ocsn.enabled = 1;
        ext->robust_ocsn.regime[0].prob = 0.010f;
        ext->robust_ocsn.regime[0].variance = 18.0f;
        ext->robust_ocsn.regime[1].prob = 0.015f;
        ext->robust_ocsn.regime[1].variance = 22.0f;
        ext->robust_ocsn.regime[2].prob = 0.020f;
        ext->robust_ocsn.regime[2].variance = 26.0f;
        ext->robust_ocsn.regime[3].prob = 0.025f;
        ext->robust_ocsn.regime[3].variance = 30.0f;
    }
    else
    {
        ext->robust_ocsn.enabled = 0;
    }

    rbpf_ext_init(ext, TRUE_MU_VOL[0], 0.1f);
    return ext;
}

/*============================================================================
 * SCENARIO EVALUATION
 *============================================================================*/

static void run_scenario_mode(const Scenario *s, TestMode mode, int n_runs, ScenarioStats *stats)
{
    memset(stats, 0, sizeof(ScenarioStats));
    stats->n_runs = n_runs;

    double sum_log_vol_se = 0, sum_log_vol_se_out = 0, sum_log_vol_se_norm = 0;
    int count_all = 0, count_out = 0, count_norm = 0;
    double sum_regime_acc = 0, sum_regime_acc_post = 0;
    double sum_ess = 0;
    double min_ess_global = 1e9;
    int ess_collapse_total = 0;
    double sum_outlier_frac = 0;
    int outlier_frac_count = 0;
    int outlier_detect_total = 0;
    double sum_latency = 0;
    int tick_total = 0;
    int outliers_total = 0;

    int detection_window = 80;
    double sum_delay = 0;
    int n_detected = 0;

    for (int run = 0; run < n_runs; run++)
    {
        rng_seed(42 + run * 7919 + (int)mode * 1000);

        ScenarioData *data = generate_scenario_data(s, N_OBSERVATIONS);
        outliers_total += data->n_outliers;

        /* Create filter based on mode */
        RBPF_KSC *rbpf_raw = NULL;
        RBPF_Extended *ext = NULL;

        switch (mode)
        {
        case MODE_BASELINE:
            rbpf_raw = create_baseline_rbpf();
            break;
        case MODE_STORVIK:
            ext = create_storvik_rbpf(0, 0);
            break;
        case MODE_STORVIK_FULL:
            ext = create_storvik_rbpf(1, 1);
            break;
        default:
            break;
        }

        int regime_correct = 0, regime_count = 0;
        int regime_correct_post = 0, regime_count_post = 0;
        int detection_tick = -1;

        for (int t = 0; t < data->n; t++)
        {
            RBPF_KSC_Output output;
            memset(&output, 0, sizeof(output));

            double t_start = get_time_us();

            rbpf_real_t ret = (rbpf_real_t)data->returns[t];

            if (mode == MODE_BASELINE)
            {
                rbpf_ksc_step(rbpf_raw, ret, &output);
            }
            else
            {
                rbpf_ext_step(ext, ret, &output);
            }

            double t_end = get_time_us();
            sum_latency += (t_end - t_start);
            tick_total++;

            /* Skip warmup */
            if (t < WARMUP_TICKS)
                continue;

            /* Log-vol RMSE */
            double err = output.log_vol_mean - data->true_log_vol[t];
            sum_log_vol_se += err * err;
            count_all++;

            if (data->is_outlier[t])
            {
                sum_log_vol_se_out += err * err;
                count_out++;
            }
            else
            {
                sum_log_vol_se_norm += err * err;
                count_norm++;
            }

            /* Regime accuracy */
            if (output.dominant_regime == data->true_regime[t])
            {
                regime_correct++;
            }
            regime_count++;

            if (t > s->changepoint + 50)
            {
                if (output.dominant_regime == data->true_regime[t])
                {
                    regime_correct_post++;
                }
                regime_count_post++;
            }

            /* ESS */
            sum_ess += output.ess;
            if (output.ess < min_ess_global)
            {
                min_ess_global = output.ess;
            }
            if (output.ess < N_PARTICLES * 0.1)
            {
                ess_collapse_total++;
            }

            /* Outlier detection */
            if (data->is_outlier[t])
            {
                sum_outlier_frac += output.outlier_fraction;
                outlier_frac_count++;
                if (output.outlier_fraction > 0.5)
                {
                    outlier_detect_total++;
                }
            }

            /* Change detection (use surprise signal) */
            if (detection_tick < 0 && output.surprise > 5.0)
            {
                if (t >= s->changepoint - 10 && t <= s->changepoint + detection_window)
                {
                    detection_tick = t;
                }
            }
        }

        /* Aggregate run */
        if (regime_count > 0)
        {
            sum_regime_acc += (double)regime_correct / regime_count;
        }
        if (regime_count_post > 0)
        {
            sum_regime_acc_post += (double)regime_correct_post / regime_count_post;
        }

        if (detection_tick >= 0)
        {
            sum_delay += (detection_tick - s->changepoint);
            n_detected++;
        }

        /* Cleanup */
        if (rbpf_raw)
            rbpf_ksc_destroy(rbpf_raw);
        if (ext)
            rbpf_ext_destroy(ext);
        free_scenario_data(data);
    }

    /* Compute statistics */
    stats->log_vol_rmse = count_all > 0 ? sqrt(sum_log_vol_se / count_all) : 0;
    stats->log_vol_rmse_outliers = count_out > 0 ? sqrt(sum_log_vol_se_out / count_out) : 0;
    stats->log_vol_rmse_normal = count_norm > 0 ? sqrt(sum_log_vol_se_norm / count_norm) : 0;

    stats->regime_accuracy = sum_regime_acc / n_runs * 100.0;
    stats->regime_accuracy_post = sum_regime_acc_post / n_runs * 100.0;

    stats->avg_ess = sum_ess / (tick_total - n_runs * WARMUP_TICKS);
    stats->min_ess = min_ess_global;
    stats->ess_collapse_count = ess_collapse_total;

    stats->avg_outlier_frac_on_outliers = outlier_frac_count > 0 ? sum_outlier_frac / outlier_frac_count : 0;
    stats->outlier_detections = outlier_detect_total;
    stats->n_outliers_total = outliers_total;

    stats->mean_detection_delay = n_detected > 0 ? sum_delay / n_detected : -1;
    stats->detection_rate = (double)n_detected / n_runs * 100.0;

    stats->avg_latency_us = sum_latency / tick_total;
}

/*============================================================================
 * REPORTING
 *============================================================================*/

static void print_scenario_header(const Scenario *s)
{
    printf("\n┌────────────────────────────────────────────────────────────────────────────┐\n");
    printf("│ %-74s │\n", s->name);
    printf("├────────────────────────────────────────────────────────────────────────────┤\n");
    printf("│ %-74s │\n", s->description);
    printf("│ Changepoint: t=%d", s->changepoint);
    if (s->transition_ticks > 0)
        printf(" (gradual, %d ticks)", s->transition_ticks);
    printf("%*s │\n", s->transition_ticks > 0 ? 36 : 53, "");
    printf("│ μ_vol: %.1f → %.1f (regime %d→%d)%*s │\n",
           s->before.mu_vol, s->after.mu_vol,
           s->expected_regime_before, s->expected_regime_after, 38, "");
    printf("└────────────────────────────────────────────────────────────────────────────┘\n");
}

static void print_scenario_results(ScenarioStats stats[NUM_MODES])
{
    printf("┌──────────────────────────┬────────────┬────────────┬────────────┐\n");
    printf("│ Metric                   │   Baseline │    Storvik │ Storvik+F  │\n");
    printf("├──────────────────────────┼────────────┼────────────┼────────────┤\n");

    printf("│ Log-Vol RMSE             │ %10.4f │ %10.4f │ %10.4f │\n",
           stats[0].log_vol_rmse, stats[1].log_vol_rmse, stats[2].log_vol_rmse);
    printf("│   on Outliers            │ %10.4f │ %10.4f │ %10.4f │\n",
           stats[0].log_vol_rmse_outliers, stats[1].log_vol_rmse_outliers, stats[2].log_vol_rmse_outliers);
    printf("│   on Normal              │ %10.4f │ %10.4f │ %10.4f │\n",
           stats[0].log_vol_rmse_normal, stats[1].log_vol_rmse_normal, stats[2].log_vol_rmse_normal);
    printf("├──────────────────────────┼────────────┼────────────┼────────────┤\n");

    printf("│ Regime Accuracy (post)   │ %9.1f%% │ %9.1f%% │ %9.1f%% │\n",
           stats[0].regime_accuracy_post, stats[1].regime_accuracy_post, stats[2].regime_accuracy_post);
    printf("├──────────────────────────┼────────────┼────────────┼────────────┤\n");

    printf("│ Avg ESS                  │ %10.1f │ %10.1f │ %10.1f │\n",
           stats[0].avg_ess, stats[1].avg_ess, stats[2].avg_ess);
    printf("│ Min ESS                  │ %10.1f │ %10.1f │ %10.1f │\n",
           stats[0].min_ess, stats[1].min_ess, stats[2].min_ess);
    printf("│ ESS Collapse Count       │ %10d │ %10d │ %10d │\n",
           stats[0].ess_collapse_count, stats[1].ess_collapse_count, stats[2].ess_collapse_count);
    printf("├──────────────────────────┼────────────┼────────────┼────────────┤\n");

    printf("│ Outlier Frac (outliers)  │ %10.2f │ %10.2f │ %10.2f │\n",
           stats[0].avg_outlier_frac_on_outliers,
           stats[1].avg_outlier_frac_on_outliers,
           stats[2].avg_outlier_frac_on_outliers);
    printf("│ Outlier Detections       │ %10d │ %10d │ %10d │\n",
           stats[0].outlier_detections, stats[1].outlier_detections, stats[2].outlier_detections);
    printf("│   (of %3d total)         │            │            │            │\n", stats[2].n_outliers_total);
    printf("├──────────────────────────┼────────────┼────────────┼────────────┤\n");

    printf("│ Detection Rate           │ %9.1f%% │ %9.1f%% │ %9.1f%% │\n",
           stats[0].detection_rate, stats[1].detection_rate, stats[2].detection_rate);
    printf("│ Detection Delay          │ %+9.1f │ %+9.1f │ %+9.1f │\n",
           stats[0].mean_detection_delay, stats[1].mean_detection_delay, stats[2].mean_detection_delay);
    printf("├──────────────────────────┼────────────┼────────────┼────────────┤\n");

    printf("│ Latency (μs)             │ %10.2f │ %10.2f │ %10.2f │\n",
           stats[0].avg_latency_us, stats[1].avg_latency_us, stats[2].avg_latency_us);
    printf("└──────────────────────────┴────────────┴────────────┴────────────┘\n");
}

static void print_summary_table(Scenario *scenarios, ScenarioStats all_stats[][NUM_MODES], int n_scenarios)
{
    printf("\n╔════════════════════════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║                                    SUMMARY BY SCENARIO                                         ║\n");
    printf("╠═════════════════════╦═════════════════════════════╦═════════════════════════════════════════════╣\n");
    printf("║                     ║    Log-Vol RMSE (Outliers)  ║           ESS Collapse Count                ║\n");
    printf("║ Scenario            ║ Baseline  Storvik  Storv+F  ║ Baseline    Storvik    Storv+F   (#outliers)║\n");
    printf("╠═════════════════════╬═════════════════════════════╬═════════════════════════════════════════════╣\n");

    for (int i = 0; i < n_scenarios; i++)
    {
        printf("║ %-19s ║ %7.3f  %7.3f  %7.3f   ║ %8d  %9d  %9d   (%4d)     ║\n",
               scenarios[i].name,
               all_stats[i][0].log_vol_rmse_outliers,
               all_stats[i][1].log_vol_rmse_outliers,
               all_stats[i][2].log_vol_rmse_outliers,
               all_stats[i][0].ess_collapse_count,
               all_stats[i][1].ess_collapse_count,
               all_stats[i][2].ess_collapse_count,
               all_stats[i][2].n_outliers_total);
    }

    printf("╚═════════════════════╩═════════════════════════════╩═════════════════════════════════════════════╝\n");
}

/*============================================================================
 * MAIN
 *============================================================================*/

int main(int argc, char **argv)
{
    int seed = 42;
    if (argc > 1)
        seed = atoi(argv[1]);

    init_timer();

    printf("╔════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║           RBPF REALISTIC SCENARIO TESTS (with Robust OCSN)                 ║\n");
    printf("╠════════════════════════════════════════════════════════════════════════════╣\n");
    printf("║  Modes: Baseline (true params) │ Storvik │ Storvik+Full (Fgt+RobustOCSN)   ║\n");
    printf("║  Monte Carlo: %2d runs × %4d ticks, Particles: %3d                         ║\n",
           N_MONTE_CARLO, N_OBSERVATIONS, N_PARTICLES);
    printf("║  Seed: %d                                                                   ║\n", seed);
    printf("╚════════════════════════════════════════════════════════════════════════════╝\n");

    Scenario scenarios[] = {
        scenario_flash_crash(),
        scenario_fed_announcement(),
        scenario_earnings_gap(),
        scenario_liquidity_crisis(),
        scenario_gradual_shift(),
        scenario_overnight_gap(),
        scenario_intraday_pattern(),
        scenario_correlation_spike(),
        scenario_oscillating(),
        scenario_fat_tail_stress()};
    int n_scenarios = sizeof(scenarios) / sizeof(scenarios[0]);

    ScenarioStats all_stats[10][NUM_MODES];

    for (int i = 0; i < n_scenarios; i++)
    {
        printf("\nRunning: %s ", scenarios[i].name);
        fflush(stdout);

        print_scenario_header(&scenarios[i]);

        for (int m = 0; m < NUM_MODES; m++)
        {
            printf("  [%s] ", mode_names[m]);
            fflush(stdout);
            run_scenario_mode(&scenarios[i], (TestMode)m, N_MONTE_CARLO, &all_stats[i][m]);
            printf("✓");
            fflush(stdout);
        }
        printf("\n");

        print_scenario_results(all_stats[i]);
    }

    print_summary_table(scenarios, all_stats, n_scenarios);

    /* Overall assessment */
    printf("\n════════════════════════════════════════════════════════════════════════════\n");
    printf("ASSESSMENT:\n\n");

    double sum_outlier_rmse[NUM_MODES] = {0};
    int sum_ess_collapse[NUM_MODES] = {0};
    double sum_regime_acc[NUM_MODES] = {0};

    for (int i = 0; i < n_scenarios; i++)
    {
        for (int m = 0; m < NUM_MODES; m++)
        {
            sum_outlier_rmse[m] += all_stats[i][m].log_vol_rmse_outliers;
            sum_ess_collapse[m] += all_stats[i][m].ess_collapse_count;
            sum_regime_acc[m] += all_stats[i][m].regime_accuracy_post;
        }
    }

    printf("  Avg Outlier RMSE:  Baseline=%.3f  Storvik=%.3f  Storvik+Full=%.3f\n",
           sum_outlier_rmse[0] / n_scenarios,
           sum_outlier_rmse[1] / n_scenarios,
           sum_outlier_rmse[2] / n_scenarios);
    printf("  Total ESS Collapse: Baseline=%d  Storvik=%d  Storvik+Full=%d\n",
           sum_ess_collapse[0], sum_ess_collapse[1], sum_ess_collapse[2]);
    printf("  Avg Regime Acc:    Baseline=%.1f%%  Storvik=%.1f%%  Storvik+Full=%.1f%%\n",
           sum_regime_acc[0] / n_scenarios,
           sum_regime_acc[1] / n_scenarios,
           sum_regime_acc[2] / n_scenarios);

    /* Check if Robust OCSN helps */
    double outlier_improvement = (sum_outlier_rmse[1] - sum_outlier_rmse[2]) / sum_outlier_rmse[1] * 100;
    int ess_improvement = sum_ess_collapse[1] - sum_ess_collapse[2];

    printf("\n  Robust OCSN Impact:\n");
    printf("    Outlier RMSE improvement: %.1f%%\n", outlier_improvement);
    printf("    ESS collapses prevented:  %d\n", ess_improvement);

    if (outlier_improvement > 20 && ess_improvement > 0)
    {
        printf("\n  ✓ Robust OCSN provides meaningful protection during fat-tail events\n");
    }
    else if (outlier_improvement > 10)
    {
        printf("\n  ~ Robust OCSN provides moderate improvement\n");
    }
    else
    {
        printf("\n  ✗ Robust OCSN not showing clear benefit (check config)\n");
    }

    printf("════════════════════════════════════════════════════════════════════════════\n");

    return 0;
}