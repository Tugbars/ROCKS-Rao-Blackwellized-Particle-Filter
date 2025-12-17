/**
 * @file rbpf_tuner.c
 * @brief RBPF Parameter Calibration via Grid Search
 *
 * Smart parameterization: 8 knobs → 32 derived params
 * Optimizes for: Vol RMSE, Regime Accuracy, Transition Lag
 *
 * Usage:
 *   ./rbpf_tuner [--quick]
 *
 * Output:
 *   - Best configurations per metric
 *   - CSV: rbpf_tuning_results.csv
 *   - Copy-paste C code
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <omp.h>

#include "rbpf_ksc.h"
#include "rbpf_ksc_param_integration.h"

/*═══════════════════════════════════════════════════════════════════════════
 * PARAMETER GRID (8 knobs)
 *═══════════════════════════════════════════════════════════════════════════*/

static const float MU_CALM[] = {-5.5f, -5.0f, -4.5f};
static const float MU_CRISIS[] = {-2.0f, -1.5f, -1.0f};
static const float SIGMA_CALM[] = {0.06f, 0.08f, 0.10f};
static const float SIGMA_RATIO[] = {4.0f, 6.0f, 8.0f};
static const float THETA_CALM[] = {0.003f, 0.005f, 0.008f};
static const float THETA_RATIO[] = {15.0f, 25.0f, 40.0f};
static const float STICKINESS[] = {0.92f, 0.95f, 0.98f};
static const float LAMBDA_CALM[] = {0.998f, 0.999f, 0.9995f};

#define N_MU_CALM 3
#define N_MU_CRISIS 3
#define N_SIGMA_CALM 3
#define N_SIGMA_RATIO 3
#define N_THETA_CALM 3
#define N_THETA_RATIO 3
#define N_STICKINESS 3
#define N_LAMBDA_CALM 3

/*═══════════════════════════════════════════════════════════════════════════
 * CONFIG STRUCT
 *═══════════════════════════════════════════════════════════════════════════*/

typedef struct
{
    float mu_calm;
    float mu_crisis;
    float sigma_calm;
    float sigma_ratio;
    float theta_calm;
    float theta_ratio;
    float stickiness;
    float lambda_calm;
} TunerConfig;

typedef struct
{
    double vol_rmse;
    double log_vol_rmse;
    double regime_accuracy;
    double transition_lag;
    double false_crisis;
    double min_ess;
} TunerMetrics;

typedef struct
{
    TunerConfig cfg;
    TunerMetrics metrics;
    double score;
} TunerResult;

/*═══════════════════════════════════════════════════════════════════════════
 * SYNTHETIC DATA GENERATION - 7 REALISTIC SCENARIOS
 *
 * EXACT MATCH to test_mmpf_comparison.c scenarios:
 *   1. Extended Calm (0-1499)        - CALM dominant, 2 outliers
 *   2. Slow Trend (1500-2499)        - CALM → TREND transition
 *   3. Sudden Crisis (2500-2999)     - TREND → CRISIS, 5 fat-tail outliers
 *   4. Crisis Persistence (3000-3999)- CRISIS sustained, 3 extreme outliers
 *   5. Recovery (4000-5199)          - CRISIS → TREND → CALM
 *   6. Flash Crash (5200-5699)       - CALM → CRISIS → CALM (60 ticks)
 *   7. Choppy (5700-7999)            - Mixed regime switching
 *
 * Uses 3 hypotheses (CALM/TREND/CRISIS) mapped to 4 RBPF regimes:
 *   R0, R1 → CALM,  R2 → TREND,  R3 → CRISIS
 *═══════════════════════════════════════════════════════════════════════════*/

typedef enum
{
    HYPO_CALM = 0,
    HYPO_TREND = 1,
    HYPO_CRISIS = 2,
    N_HYPOTHESES = 3
} Hypothesis;

/* PCG32 RNG - matches test_mmpf_comparison.c */
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

/* Ground truth parameters - EXACT match to test_mmpf_comparison.c */
typedef struct
{
    double mu_vol;    /* Long-run mean of log-vol */
    double phi;       /* Persistence (1 - theta) */
    double sigma_eta; /* Vol-of-vol */
} HypothesisParams;

static const HypothesisParams TRUE_PARAMS[N_HYPOTHESES] = {
    /* CALM: Low vol, high persistence, smooth */
    {.mu_vol = -5.0, .phi = 0.995, .sigma_eta = 0.08},
    /* TREND: Medium vol, medium persistence */
    {.mu_vol = -3.5, .phi = 0.95, .sigma_eta = 0.20},
    /* CRISIS: High vol, fast mean reversion, explosive */
    {.mu_vol = -1.5, .phi = 0.85, .sigma_eta = 0.50}};

typedef struct
{
    float *returns;
    float *true_vol;
    int *true_regime; /* 4-regime for RBPF comparison */
    int *true_hypo;   /* 3-hypothesis ground truth */
    int *is_outlier;
    int n_ticks;
    int n_outliers;
} SyntheticData;

/* Map 3-hypothesis to 4-regime (for ground truth comparison) */
static int hypothesis_to_regime(Hypothesis h)
{
    switch (h)
    {
    case HYPO_CALM:
        return 0;
    case HYPO_TREND:
        return 2;
    case HYPO_CRISIS:
        return 3;
    default:
        return 0;
    }
}

/* Map 4-regime RBPF output to 3-hypothesis */
static int regime_to_hypothesis(int regime)
{
    if (regime <= 1)
        return HYPO_CALM;
    if (regime == 2)
        return HYPO_TREND;
    return HYPO_CRISIS;
}

/* Inject outlier at tick t */
static void inject_outlier(SyntheticData *data, int t, double target_sigma, pcg32_t *rng)
{
    double vol = data->true_vol[t];
    double sign = (pcg32_double(rng) < 0.5) ? -1.0 : 1.0;
    data->returns[t] = (float)(sign * target_sigma * vol);
    data->is_outlier[t] = 1;
    data->n_outliers++;
}

static SyntheticData *generate_data(int seed, int n_ticks_unused)
{
    (void)n_ticks_unused; /* Always 8000 for scenario structure */

    int n = 8000;
    SyntheticData *data = malloc(sizeof(SyntheticData));
    data->returns = malloc(n * sizeof(float));
    data->true_vol = malloc(n * sizeof(float));
    data->true_regime = malloc(n * sizeof(int));
    data->true_hypo = malloc(n * sizeof(int));
    data->is_outlier = calloc(n, sizeof(int));
    data->n_ticks = n;
    data->n_outliers = 0;

    /* PCG32 seeding - EXACT match to test_mmpf_comparison.c */
    pcg32_t rng = {seed * 12345ULL + 1, seed * 67890ULL | 1};

    /* Start in CALM */
    double log_vol = TRUE_PARAMS[HYPO_CALM].mu_vol;
    int t = 0;

/* Helper macro for state evolution - EXACT match */
#define EVOLVE_STATE(H)                                                                       \
    do                                                                                        \
    {                                                                                         \
        const HypothesisParams *p = &TRUE_PARAMS[H];                                          \
        double theta = 1.0 - p->phi;                                                          \
        log_vol = p->phi * log_vol + theta * p->mu_vol + p->sigma_eta * pcg32_gaussian(&rng); \
        double vol = exp(log_vol);                                                            \
        double ret = vol * pcg32_gaussian(&rng);                                              \
        data->returns[t] = (float)ret;                                                        \
        data->true_vol[t] = (float)vol;                                                       \
        data->true_hypo[t] = (H);                                                             \
        data->true_regime[t] = hypothesis_to_regime(H);                                       \
    } while (0)

    /*═══════════════════════════════════════════════════════════════════════
     * Scenario 1: Extended Calm (0-1499)
     * 1500 ticks of CALM with 2 outliers (6σ, 8σ)
     *═══════════════════════════════════════════════════════════════════════*/
    for (; t < 1500; t++)
    {
        EVOLVE_STATE(HYPO_CALM);
    }
    inject_outlier(data, 500, 6.0, &rng);
    inject_outlier(data, 1200, 8.0, &rng);

    /*═══════════════════════════════════════════════════════════════════════
     * Scenario 2: Slow Trend (1500-2499)
     * Gradual transition CALM → TREND over 1000 ticks
     *═══════════════════════════════════════════════════════════════════════*/
    for (; t < 2500; t++)
    {
        Hypothesis h = (t < 1800) ? HYPO_CALM : HYPO_TREND;
        EVOLVE_STATE(h);
    }

    /*═══════════════════════════════════════════════════════════════════════
     * Scenario 3: Sudden Crisis (2500-2999)
     * Jump from TREND to CRISIS with 5 fat-tail outliers (8-12σ)
     *═══════════════════════════════════════════════════════════════════════*/
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
     * Sustained CRISIS with 3 extreme outliers (10-15σ)
     *═══════════════════════════════════════════════════════════════════════*/
    for (; t < 4000; t++)
    {
        EVOLVE_STATE(HYPO_CRISIS);
    }
    inject_outlier(data, 3200, 10.0, &rng);
    inject_outlier(data, 3500, 15.0, &rng); /* EXTREME */
    inject_outlier(data, 3800, 12.0, &rng);

    /*═══════════════════════════════════════════════════════════════════════
     * Scenario 5: Recovery (4000-5199)
     * CRISIS → TREND → CALM over 1200 ticks
     *═══════════════════════════════════════════════════════════════════════*/
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
     * CALM → CRISIS (60 ticks) → CALM with 12σ outlier at peak
     *═══════════════════════════════════════════════════════════════════════*/
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
     * Random switching between all hypotheses
     *═══════════════════════════════════════════════════════════════════════*/
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

static void free_data(SyntheticData *data)
{
    if (data)
    {
        free(data->returns);
        free(data->true_vol);
        free(data->true_regime);
        free(data->true_hypo);
        free(data->is_outlier);
        free(data);
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * APPLY CONFIG TO RBPF
 *═══════════════════════════════════════════════════════════════════════════*/

static void apply_config(RBPF_Extended *ext, const TunerConfig *cfg)
{
    /* Interpolate μ_vol across regimes */
    float mu[4];
    mu[0] = cfg->mu_calm;
    mu[3] = cfg->mu_crisis;
    mu[1] = cfg->mu_calm + (cfg->mu_crisis - cfg->mu_calm) * 0.33f;
    mu[2] = cfg->mu_calm + (cfg->mu_crisis - cfg->mu_calm) * 0.67f;

    /* Interpolate σ_vol */
    float sigma[4];
    sigma[0] = cfg->sigma_calm;
    sigma[3] = cfg->sigma_calm * cfg->sigma_ratio;
    sigma[1] = sigma[0] + (sigma[3] - sigma[0]) * 0.33f;
    sigma[2] = sigma[0] + (sigma[3] - sigma[0]) * 0.67f;

    /* Interpolate θ */
    float theta[4];
    theta[0] = cfg->theta_calm;
    theta[3] = cfg->theta_calm * cfg->theta_ratio;
    theta[1] = theta[0] + (theta[3] - theta[0]) * 0.33f;
    theta[2] = theta[0] + (theta[3] - theta[0]) * 0.67f;

    /* Apply regime params */
    for (int r = 0; r < 4; r++)
    {
        rbpf_ext_set_regime_params(ext, r, theta[r], mu[r], sigma[r]);
    }

    /* Build transition matrix from stickiness */
    float s = cfg->stickiness;
    float leak = 1.0f - s;

    rbpf_real_t trans[16] = {
        s, leak * 0.7f, leak * 0.25f, leak * 0.05f,
        leak * 0.4f, s, leak * 0.45f, leak * 0.15f,
        leak * 0.15f, leak * 0.45f, s, leak * 0.4f,
        leak * 0.05f, leak * 0.25f, leak * 0.7f, s};

    /* Normalize rows */
    for (int i = 0; i < 4; i++)
    {
        float sum = 0;
        for (int j = 0; j < 4; j++)
            sum += trans[i * 4 + j];
        for (int j = 0; j < 4; j++)
            trans[i * 4 + j] /= sum;
    }

    rbpf_ext_build_transition_lut(ext, trans);

    /* Interpolate forgetting λ */
    float lambda_crisis = cfg->lambda_calm - 0.006f; /* ~6 points lower */
    if (lambda_crisis < 0.990f)
        lambda_crisis = 0.990f;

    float lambda[4];
    lambda[0] = cfg->lambda_calm;
    lambda[3] = lambda_crisis;
    lambda[1] = lambda[0] - (lambda[0] - lambda[3]) * 0.33f;
    lambda[2] = lambda[0] - (lambda[0] - lambda[3]) * 0.67f;

    for (int r = 0; r < 4; r++)
    {
        param_learn_set_regime_forgetting(&ext->storvik, r, lambda[r]);
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * RUN SINGLE CONFIG
 *═══════════════════════════════════════════════════════════════════════════*/

static TunerMetrics run_config(const SyntheticData *data, const TunerConfig *cfg)
{
    TunerMetrics m = {0};

    const int N_PARTICLES = 256;
    const int N_REGIMES = 4;

    /* Create RBPF */
    RBPF_Extended *ext = rbpf_ext_create(N_PARTICLES, N_REGIMES, RBPF_PARAM_STORVIK);
    if (!ext)
    {
        m.vol_rmse = 1e10;
        return m;
    }

    /* Apply tuning config */
    apply_config(ext, cfg);

    /* Set full update mode - maximum accuracy */
    rbpf_ext_set_full_update_mode(ext);

    /* Initialize */
    rbpf_ext_init(ext, -4.0f, 0.1f);

    /* Accumulators */
    double sum_sq_vol_err = 0.0;
    double sum_sq_log_err = 0.0;
    int hypo_correct = 0;
    int false_crisis = 0;
    int missed_crisis = 0;
    double min_ess = 1e10;

    /* Transition lag tracking */
    int prev_true_hypo = data->true_hypo[0];
    int transition_count = 0;
    int total_lag = 0;
    int looking_for = 0;
    int target_hypo = 0;
    int ticks_since = 0;

    /* Run */
    RBPF_KSC_Output output;

    for (int t = 0; t < data->n_ticks; t++)
    {
        rbpf_ext_step(ext, data->returns[t], &output);

        /* Vol error */
        double true_vol = data->true_vol[t];
        double est_vol = output.vol_mean;
        double vol_err = est_vol - true_vol;
        sum_sq_vol_err += vol_err * vol_err;

        double true_log_vol = log(true_vol);
        double log_err = output.log_vol_mean - true_log_vol;
        sum_sq_log_err += log_err * log_err;

        /* Map RBPF regime to hypothesis for fair comparison */
        int true_hypo = data->true_hypo[t];
        int est_hypo = regime_to_hypothesis(output.dominant_regime);

        /* Hypothesis accuracy */
        if (est_hypo == true_hypo)
        {
            hypo_correct++;
        }

        /* False crisis: estimated CRISIS when true != CRISIS */
        if (est_hypo == HYPO_CRISIS && true_hypo != HYPO_CRISIS)
        {
            false_crisis++;
        }

        /* Missed crisis: not CRISIS when true == CRISIS */
        if (est_hypo != HYPO_CRISIS && true_hypo == HYPO_CRISIS)
        {
            missed_crisis++;
        }

        /* Min ESS */
        if (output.ess < min_ess)
        {
            min_ess = output.ess;
        }

        /* Transition lag (hypothesis-based) */
        if (true_hypo != prev_true_hypo)
        {
            looking_for = 1;
            target_hypo = true_hypo;
            ticks_since = 0;
        }

        if (looking_for)
        {
            ticks_since++;
            if (est_hypo == target_hypo)
            {
                total_lag += ticks_since;
                transition_count++;
                looking_for = 0;
            }
            if (ticks_since > 100)
            {
                total_lag += 100;
                transition_count++;
                looking_for = 0;
            }
        }

        prev_true_hypo = true_hypo;
    }

    /* Compute metrics */
    m.vol_rmse = sqrt(sum_sq_vol_err / data->n_ticks);
    m.log_vol_rmse = sqrt(sum_sq_log_err / data->n_ticks);
    m.regime_accuracy = 100.0 * hypo_correct / data->n_ticks;
    m.transition_lag = (transition_count > 0) ? (double)total_lag / transition_count : 0.0;
    m.false_crisis = (double)false_crisis;
    m.min_ess = min_ess;

    rbpf_ext_destroy(ext);

    return m;
}

/*═══════════════════════════════════════════════════════════════════════════
 * SCORING
 *═══════════════════════════════════════════════════════════════════════════*/

static double compute_score(const TunerMetrics *m)
{
    /* Lower is better */
    double norm_rmse = m->vol_rmse / 0.02;                 /* 2% vol = 1.0 */
    double norm_acc = (100.0 - m->regime_accuracy) / 50.0; /* 50% miss = 1.0 */
    double norm_lag = m->transition_lag / 20.0;            /* 20 ticks = 1.0 */
    double norm_fc = m->false_crisis / 200.0;              /* 200 false = 1.0 */

    return 0.35 * norm_rmse + 0.30 * norm_acc + 0.20 * norm_lag + 0.15 * norm_fc;
}

/*═══════════════════════════════════════════════════════════════════════════
 * PRINT HELPERS
 *═══════════════════════════════════════════════════════════════════════════*/

static void print_result(const TunerResult *r)
{
    printf("  Config:\n");
    printf("    μ_calm=%.2f  μ_crisis=%.2f\n", r->cfg.mu_calm, r->cfg.mu_crisis);
    printf("    σ_calm=%.3f  σ_ratio=%.1f\n", r->cfg.sigma_calm, r->cfg.sigma_ratio);
    printf("    θ_calm=%.4f  θ_ratio=%.1f\n", r->cfg.theta_calm, r->cfg.theta_ratio);
    printf("    stickiness=%.2f  λ_calm=%.4f\n", r->cfg.stickiness, r->cfg.lambda_calm);
    printf("  ─────────────────────────────\n");
    printf("    Vol RMSE:     %.5f\n", r->metrics.vol_rmse);
    printf("    Log-Vol RMSE: %.4f\n", r->metrics.log_vol_rmse);
    printf("    Hypo Acc:     %.1f%%\n", r->metrics.regime_accuracy);
    printf("    Trans Lag:    %.1f ticks\n", r->metrics.transition_lag);
    printf("    False Crisis: %.0f\n", r->metrics.false_crisis);
    printf("    Min ESS:      %.1f\n", r->metrics.min_ess);
    printf("    Score:        %.4f\n", r->score);
}

static void print_c_code(const TunerResult *r)
{
    TunerConfig cfg = r->cfg;

    /* Compute derived values */
    float mu[4], sigma[4], theta[4], lambda[4];

    mu[0] = cfg.mu_calm;
    mu[3] = cfg.mu_crisis;
    mu[1] = cfg.mu_calm + (cfg.mu_crisis - cfg.mu_calm) * 0.33f;
    mu[2] = cfg.mu_calm + (cfg.mu_crisis - cfg.mu_calm) * 0.67f;

    sigma[0] = cfg.sigma_calm;
    sigma[3] = cfg.sigma_calm * cfg.sigma_ratio;
    sigma[1] = sigma[0] + (sigma[3] - sigma[0]) * 0.33f;
    sigma[2] = sigma[0] + (sigma[3] - sigma[0]) * 0.67f;

    theta[0] = cfg.theta_calm;
    theta[3] = cfg.theta_calm * cfg.theta_ratio;
    theta[1] = theta[0] + (theta[3] - theta[0]) * 0.33f;
    theta[2] = theta[0] + (theta[3] - theta[0]) * 0.67f;

    float lambda_crisis = cfg.lambda_calm - 0.006f;
    if (lambda_crisis < 0.990f)
        lambda_crisis = 0.990f;
    lambda[0] = cfg.lambda_calm;
    lambda[3] = lambda_crisis;
    lambda[1] = lambda[0] - (lambda[0] - lambda[3]) * 0.33f;
    lambda[2] = lambda[0] - (lambda[0] - lambda[3]) * 0.67f;

    printf("/* Regime params (θ, μ, σ) */\n");
    printf("rbpf_ext_set_regime_params(ext, 0, %.4ff, %.2ff, %.3ff);  /* Calm */\n",
           theta[0], mu[0], sigma[0]);
    printf("rbpf_ext_set_regime_params(ext, 1, %.4ff, %.2ff, %.3ff);  /* Mild */\n",
           theta[1], mu[1], sigma[1]);
    printf("rbpf_ext_set_regime_params(ext, 2, %.4ff, %.2ff, %.3ff);  /* Trend */\n",
           theta[2], mu[2], sigma[2]);
    printf("rbpf_ext_set_regime_params(ext, 3, %.4ff, %.2ff, %.3ff);  /* Crisis */\n",
           theta[3], mu[3], sigma[3]);

    printf("\n/* Transition matrix (stickiness=%.2f) */\n", cfg.stickiness);
    float s = cfg.stickiness;
    float leak = 1.0f - s;
    printf("rbpf_real_t trans[16] = {\n");
    printf("    %.3ff, %.3ff, %.3ff, %.3ff,\n", s, leak * 0.7f, leak * 0.25f, leak * 0.05f);
    printf("    %.3ff, %.3ff, %.3ff, %.3ff,\n", leak * 0.4f, s, leak * 0.45f, leak * 0.15f);
    printf("    %.3ff, %.3ff, %.3ff, %.3ff,\n", leak * 0.15f, leak * 0.45f, s, leak * 0.4f);
    printf("    %.3ff, %.3ff, %.3ff, %.3ff};\n", leak * 0.05f, leak * 0.25f, leak * 0.7f, s);
    printf("rbpf_ext_build_transition_lut(ext, trans);\n");

    printf("\n/* Forgetting λ per regime */\n");
    printf("param_learn_set_regime_forgetting(&ext->storvik, 0, %.4ff);\n", lambda[0]);
    printf("param_learn_set_regime_forgetting(&ext->storvik, 1, %.4ff);\n", lambda[1]);
    printf("param_learn_set_regime_forgetting(&ext->storvik, 2, %.4ff);\n", lambda[2]);
    printf("param_learn_set_regime_forgetting(&ext->storvik, 3, %.4ff);\n", lambda[3]);
}

/*═══════════════════════════════════════════════════════════════════════════
 * MAIN
 *═══════════════════════════════════════════════════════════════════════════*/

int main(int argc, char **argv)
{
    int quick_mode = 0;

    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "--quick") == 0)
            quick_mode = 1;
    }

    /* Grid sizes */
    int n_mu_calm = quick_mode ? 2 : N_MU_CALM;
    int n_mu_crisis = quick_mode ? 2 : N_MU_CRISIS;
    int n_sigma_calm = quick_mode ? 2 : N_SIGMA_CALM;
    int n_sigma_ratio = quick_mode ? 2 : N_SIGMA_RATIO;
    int n_theta_calm = quick_mode ? 2 : N_THETA_CALM;
    int n_theta_ratio = quick_mode ? 2 : N_THETA_RATIO;
    int n_stickiness = quick_mode ? 2 : N_STICKINESS;
    int n_lambda_calm = quick_mode ? 2 : N_LAMBDA_CALM;

    int total_configs = n_mu_calm * n_mu_crisis * n_sigma_calm * n_sigma_ratio *
                        n_theta_calm * n_theta_ratio * n_stickiness * n_lambda_calm;

    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  RBPF Parameter Tuner\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  Mode: %s\n", quick_mode ? "Quick (2^8=256)" : "Full (3^8=6561)");
    printf("  Configs: %d\n", total_configs);
    printf("  Particles: 256\n\n");

    /* Generate data */
    int n_ticks = quick_mode ? 4000 : 8000;
    SyntheticData *data = generate_data(42, n_ticks);
    printf("  Data: %d ticks\n\n", n_ticks);

    /* Results storage */
    TunerResult *results = malloc(total_configs * sizeof(TunerResult));
    int n_results = 0;

    /* Best trackers */
    TunerResult best_rmse = {.metrics.vol_rmse = 1e10};
    TunerResult best_acc = {.metrics.regime_accuracy = 0};
    TunerResult best_lag = {.metrics.transition_lag = 1e10};
    TunerResult best_balanced = {.score = 1e10};

    /* Build flat config array for parallel iteration */
    TunerConfig *configs = malloc(total_configs * sizeof(TunerConfig));
    int cfg_idx = 0;

    for (int i0 = 0; i0 < n_mu_calm; i0++)
    {
        for (int i1 = 0; i1 < n_mu_crisis; i1++)
        {
            for (int i2 = 0; i2 < n_sigma_calm; i2++)
            {
                for (int i3 = 0; i3 < n_sigma_ratio; i3++)
                {
                    for (int i4 = 0; i4 < n_theta_calm; i4++)
                    {
                        for (int i5 = 0; i5 < n_theta_ratio; i5++)
                        {
                            for (int i6 = 0; i6 < n_stickiness; i6++)
                            {
                                for (int i7 = 0; i7 < n_lambda_calm; i7++)
                                {
                                    configs[cfg_idx++] = (TunerConfig){
                                        .mu_calm = MU_CALM[i0],
                                        .mu_crisis = MU_CRISIS[i1],
                                        .sigma_calm = SIGMA_CALM[i2],
                                        .sigma_ratio = SIGMA_RATIO[i3],
                                        .theta_calm = THETA_CALM[i4],
                                        .theta_ratio = THETA_RATIO[i5],
                                        .stickiness = STICKINESS[i6],
                                        .lambda_calm = LAMBDA_CALM[i7]};
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /* Grid search - PARALLEL */
    double start = omp_get_wtime();
    int progress = 0;

    int n_threads = omp_get_max_threads();
    printf("  Threads: %d\n\n", n_threads);

    int i;
#pragma omp parallel for schedule(dynamic, 16) private(i)
    for (i = 0; i < total_configs; i++)
    {
        TunerConfig cfg = configs[i];
        TunerMetrics m = run_config(data, &cfg);
        double score = compute_score(&m);

        results[i] = (TunerResult){.cfg = cfg, .metrics = m, .score = score};

/* Progress (atomic update) */
#pragma omp atomic
        progress++;

        if (progress % 100 == 0)
        {
#pragma omp critical
            {
                printf("  Progress: %d/%d (%.0f%%)\r", progress, total_configs,
                       100.0 * progress / total_configs);
                fflush(stdout);
            }
        }
    }

    n_results = total_configs;

    /* Find bests (sequential, fast) */
    for (int i = 0; i < n_results; i++)
    {
        TunerResult *r = &results[i];
        if (r->metrics.vol_rmse < best_rmse.metrics.vol_rmse)
            best_rmse = *r;
        if (r->metrics.regime_accuracy > best_acc.metrics.regime_accuracy)
            best_acc = *r;
        if (r->metrics.transition_lag < best_lag.metrics.transition_lag && r->metrics.transition_lag > 0)
            best_lag = *r;
        if (r->score < best_balanced.score)
            best_balanced = *r;
    }

    free(configs);

    double elapsed = omp_get_wtime() - start;

    printf("\n\nCompleted in %.1f sec (%.1f ms/config)\n\n", elapsed, 1000.0 * elapsed / n_results);

    /* Print results */
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  BEST FOR VOL RMSE\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    print_result(&best_rmse);

    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("  BEST FOR HYPOTHESIS ACCURACY\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    print_result(&best_acc);

    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("  BEST FOR TRANSITION LAG\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    print_result(&best_lag);

    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("  BEST BALANCED (0.35×RMSE + 0.30×Acc + 0.20×Lag + 0.15×FC)\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    print_result(&best_balanced);

    /* Write CSV */
    FILE *csv = fopen("rbpf_tuning_results.csv", "w");
    if (csv)
    {
        fprintf(csv, "mu_calm,mu_crisis,sigma_calm,sigma_ratio,theta_calm,theta_ratio,"
                     "stickiness,lambda_calm,vol_rmse,log_vol_rmse,hypo_acc,"
                     "transition_lag,false_crisis,min_ess,score\n");

        for (int i = 0; i < n_results; i++)
        {
            TunerResult *r = &results[i];
            fprintf(csv, "%.2f,%.2f,%.3f,%.1f,%.4f,%.1f,%.2f,%.4f,"
                         "%.6f,%.4f,%.2f,%.2f,%.0f,%.1f,%.6f\n",
                    r->cfg.mu_calm, r->cfg.mu_crisis,
                    r->cfg.sigma_calm, r->cfg.sigma_ratio,
                    r->cfg.theta_calm, r->cfg.theta_ratio,
                    r->cfg.stickiness, r->cfg.lambda_calm,
                    r->metrics.vol_rmse, r->metrics.log_vol_rmse,
                    r->metrics.regime_accuracy, r->metrics.transition_lag,
                    r->metrics.false_crisis, r->metrics.min_ess, r->score);
        }
        fclose(csv);
        printf("\n  Results saved to: rbpf_tuning_results.csv\n");
    }

    /* Print recommended C code */
    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("  RECOMMENDED C CODE (Balanced Best)\n");
    printf("═══════════════════════════════════════════════════════════════\n\n");
    print_c_code(&best_balanced);

    /* Cleanup */
    free(results);
    free_data(data);

    return 0;
}