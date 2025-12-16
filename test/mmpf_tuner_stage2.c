/**
 * @file mmpf_tuner_stage2.c
 * @brief Stage 2: BOCPD + MMPF Integration Tuner
 *
 * Assumes Stage 1 has found optimal static MMPF parameters.
 * This tuner optimizes the "handshake" between BOCPD and MMPF:
 *   - BOCPD trigger sensitivity (r0_threshold)
 *   - BOCPD refractory period
 *   - Shock injection magnitude
 *   - Post-shock lockout duration
 *   - Crisis dynamics (φ)
 *
 * Usage:
 *   ./mmpf_tuner_stage2 [--quick]
 *
 * Prerequisite: Run mmpf_tuner first to get optimal static params.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <float.h>

#include "mmpf_rocks.h"
#include "bocpd.h"  /* Your BOCPD implementation */

/*═══════════════════════════════════════════════════════════════════════════
 * STAGE 1 LOCKED PARAMETERS (From mmpf_tuner results)
 * 
 * Replace these with your actual Stage 1 optimal values.
 *═══════════════════════════════════════════════════════════════════════════*/

static void apply_stage1_params(MMPF_Config *cfg)
{
    /* TODO: Replace with your Stage 1 optimal values */
    
    /* Swim Lanes - Calm */
    cfg->swim_lanes[MMPF_CALM].mu_vol_min = RBPF_REAL(-6.0);
    cfg->swim_lanes[MMPF_CALM].mu_vol_max = RBPF_REAL(-4.5);
    
    /* Swim Lanes - Crisis */
    cfg->swim_lanes[MMPF_CRISIS].mu_vol_min = RBPF_REAL(-3.0);
    cfg->swim_lanes[MMPF_CRISIS].mu_vol_max = RBPF_REAL(-0.5);
    
    /* Dirichlet Prior */
    cfg->transition_prior_alpha = RBPF_REAL(1.0);
    cfg->transition_prior_mass = RBPF_REAL(100.0);
    
    /* Student-t */
    cfg->hypothesis_nu[MMPF_CALM] = RBPF_REAL(20.0);
    cfg->hypothesis_nu[MMPF_CRISIS] = RBPF_REAL(3.0);
    cfg->hypotheses[MMPF_CALM].nu = RBPF_REAL(20.0);
    cfg->hypotheses[MMPF_CRISIS].nu = RBPF_REAL(3.0);
    
    /* Base Stickiness (may be overridden - with BOCPD we want HIGHER) */
    cfg->base_stickiness = RBPF_REAL(0.98);
}

/*═══════════════════════════════════════════════════════════════════════════
 * STAGE 2 PARAMETER GRID
 *═══════════════════════════════════════════════════════════════════════════*/

/* BOCPD Detonator: When to fire */
static const double BOCPD_R0_THRESH[] = {0.03, 0.05, 0.08, 0.10};
#define N_BOCPD_THRESH 4

static const int BOCPD_REFRACTORY[] = {5, 10, 15, 20};
#define N_BOCPD_REFRACTORY 4

/* Shock Handshake: How MMPF responds */
static const double SHOCK_MULTIPLIER[] = {20.0, 35.0, 50.0, 75.0, 100.0};
#define N_SHOCK_MULT 5

static const int POST_SHOCK_LOCKOUT[] = {3, 5, 10, 15, 20};
#define N_LOCKOUT 5

/* Crisis Dynamics: Internal physics */
static const double CRISIS_PHI[] = {0.75, 0.80, 0.85, 0.90};
#define N_CRISIS_PHI 4

/* Stickiness (re-tuned with BOCPD - expect higher optimal) */
static const double STICKINESS_WITH_BOCPD[] = {0.97, 0.98, 0.99, 0.995};
#define N_STICKINESS 4

/* Total configurations */
#define N_CONFIGS (N_BOCPD_THRESH * N_BOCPD_REFRACTORY * N_SHOCK_MULT * N_LOCKOUT * N_CRISIS_PHI * N_STICKINESS)

/*═══════════════════════════════════════════════════════════════════════════
 * BENCHMARK DATA GENERATION
 *═══════════════════════════════════════════════════════════════════════════*/

typedef enum {
    SCENARIO_CALM = 0,
    SCENARIO_TREND,
    SCENARIO_CRISIS,
    SCENARIO_CRISIS_PERSIST,
    SCENARIO_RECOVERY,
    SCENARIO_FLASH,
    SCENARIO_CHOPPY,
    SCENARIO_COUNT
} Scenario;

typedef struct {
    double *returns;
    double *true_vol;
    int *true_regime;
    int *scenario_id;
    int *is_shock_tick;  /* Ground truth: should BOCPD fire here? */
    int n_ticks;
    int n_outliers;
    int n_true_shocks;
} BenchmarkData;

static unsigned int g_seed = 42;

static double rand_uniform(void) {
    g_seed = g_seed * 1103515245 + 12345;
    return (double)(g_seed % 1000000) / 1000000.0;
}

static double rand_normal(void) {
    double u1 = rand_uniform();
    double u2 = rand_uniform();
    if (u1 < 1e-10) u1 = 1e-10;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979 * u2);
}

static BenchmarkData* generate_benchmark(int seed, int ticks_per_scenario) {
    g_seed = (unsigned int)seed;
    
    BenchmarkData *data = (BenchmarkData*)malloc(sizeof(BenchmarkData));
    int total = ticks_per_scenario * SCENARIO_COUNT;
    
    data->returns = (double*)malloc(total * sizeof(double));
    data->true_vol = (double*)malloc(total * sizeof(double));
    data->true_regime = (int*)malloc(total * sizeof(int));
    data->scenario_id = (int*)malloc(total * sizeof(int));
    data->is_shock_tick = (int*)calloc(total, sizeof(int));
    data->n_ticks = total;
    data->n_outliers = 0;
    data->n_true_shocks = 0;
    
    int idx = 0;
    int prev_regime = 0;
    
    for (int s = 0; s < SCENARIO_COUNT; s++) {
        double vol, vol_drift;
        int regime;
        
        switch (s) {
            case SCENARIO_CALM:
                vol = 0.008;
                vol_drift = 0.0;
                regime = 0;
                break;
            case SCENARIO_TREND:
                vol = 0.012;
                vol_drift = 0.00002;
                regime = 1;
                break;
            case SCENARIO_CRISIS:
                vol = 0.035;
                vol_drift = 0.0;
                regime = 2;
                break;
            case SCENARIO_CRISIS_PERSIST:
                vol = 0.040;
                vol_drift = 0.0;
                regime = 2;
                break;
            case SCENARIO_RECOVERY:
                vol = 0.025;
                vol_drift = -0.00015;
                regime = 1;
                break;
            case SCENARIO_FLASH:
                vol = 0.010;
                vol_drift = 0.0;
                regime = 0;
                break;
            case SCENARIO_CHOPPY:
                vol = 0.018;
                vol_drift = 0.0;
                regime = 1;
                break;
            default:
                vol = 0.01;
                vol_drift = 0.0;
                regime = 0;
        }
        
        for (int t = 0; t < ticks_per_scenario; t++) {
            double current_vol = vol + vol_drift * t;
            if (current_vol < 0.005) current_vol = 0.005;
            if (current_vol > 0.10) current_vol = 0.10;
            
            int current_regime = regime;
            
            /* Flash crash in middle of flash scenario */
            if (s == SCENARIO_FLASH && t >= ticks_per_scenario/2 - 5 && t < ticks_per_scenario/2 + 5) {
                current_vol = 0.08;
                current_regime = 2;
            }
            
            /* Mark shock ticks (regime transitions) */
            if (idx > 0 && current_regime != prev_regime) {
                data->is_shock_tick[idx] = 1;
                data->n_true_shocks++;
            }
            
            double r = current_vol * rand_normal();
            
            if (rand_uniform() < 0.001) {
                r *= 5.0;
                data->n_outliers++;
            }
            
            data->returns[idx] = r;
            data->true_vol[idx] = current_vol;
            data->true_regime[idx] = current_regime;
            data->scenario_id[idx] = s;
            
            prev_regime = current_regime;
            idx++;
        }
    }
    
    return data;
}

static void free_benchmark(BenchmarkData *data) {
    if (data) {
        free(data->returns);
        free(data->true_vol);
        free(data->true_regime);
        free(data->scenario_id);
        free(data->is_shock_tick);
        free(data);
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * METRICS (Extended for BOCPD)
 *═══════════════════════════════════════════════════════════════════════════*/

typedef struct {
    /* Estimation quality */
    double vol_rmse;
    double log_vol_rmse;
    
    /* Regime detection */
    double hypothesis_accuracy;
    double avg_transition_lag;
    double per_scenario_accuracy[SCENARIO_COUNT];
    
    /* BOCPD specific */
    int true_positives;     /* BOCPD fired near actual regime change */
    int false_positives;    /* BOCPD fired during stable period */
    int false_negatives;    /* BOCPD missed actual regime change */
    double precision;       /* TP / (TP + FP) */
    double recall;          /* TP / (TP + FN) */
    double f1_score;        /* Harmonic mean of precision and recall */
    double avg_detection_lag;  /* Ticks from true shock to BOCPD fire */
    
    /* Stability */
    int shock_count;        /* Total BOCPD fires */
    int flicker_count;      /* Rapid back-to-back fires (bad) */
    
} TuningMetrics;

/*═══════════════════════════════════════════════════════════════════════════
 * SHOCK INJECTION: Uses mmpf_inject_shock() and mmpf_restore_from_shock() 
 * from mmpf_rocks.h
 *═══════════════════════════════════════════════════════════════════════════*/

/*═══════════════════════════════════════════════════════════════════════════
 * RUN INTEGRATED CONFIGURATION
 *═══════════════════════════════════════════════════════════════════════════*/

static TuningMetrics run_config(
    const BenchmarkData *data,
    double bocpd_r0_thresh,
    int bocpd_refractory,
    double shock_mult,
    int post_shock_lockout,
    double crisis_phi,
    double stickiness)
{
    TuningMetrics metrics = {0};
    
    /* Create MMPF with Stage 1 params */
    MMPF_Config cfg = mmpf_config_defaults();
    apply_stage1_params(&cfg);
    
    /* Apply Stage 2 params */
    cfg.base_stickiness = (rbpf_real_t)stickiness;
    cfg.hypotheses[MMPF_CRISIS].phi = (rbpf_real_t)crisis_phi;
    
    MMPF_ROCKS *mmpf = mmpf_create(&cfg);
    if (!mmpf) {
        metrics.vol_rmse = 1e10;
        return metrics;
    }
    
    /* Initialize BOCPD with power-law hazard
     * Prior centered on typical log(r²) ≈ -9 (1% daily vol) */
    bocpd_t bocpd;
    bocpd_hazard_t hazard;
    bocpd_prior_t prior;
    prior.mu0 = -9.0;     /* E[log(r²)] for ~1% vol */
    prior.kappa0 = 1.0;   /* Weak prior on mean */
    prior.alpha0 = 1.0;   /* Weak prior on variance */
    prior.beta0 = 2.0;    /* Var ≈ 2 for log-chi² noise */
    
    bocpd_hazard_init_power_law(&hazard, 0.8, 512);
    bocpd_init_with_hazard(&bocpd, &hazard, prior);
    
    /* Accumulators */
    double sum_sq_err = 0.0;
    double sum_sq_log_err = 0.0;
    int correct_regime = 0;
    int total_ticks = data->n_ticks;
    int scenario_correct[SCENARIO_COUNT] = {0};
    int scenario_count[SCENARIO_COUNT] = {0};
    
    /* Transition tracking */
    int prev_true_regime = data->true_regime[0];
    int transition_count = 0;
    int total_lag = 0;
    int ticks_since_transition = 0;
    int looking_for_detection = 0;
    
    /* BOCPD tracking */
    int refractory_counter = 0;
    int lockout_counter = 0;
    int shock_count = 0;
    int last_shock_tick = -100;
    int flicker_count = 0;
    
    /* Detection quality tracking */
    int true_positives = 0;
    int false_positives = 0;
    int detection_lags[1000];
    int n_detections = 0;
    
    /* Window for matching BOCPD fires to true shocks */
    const int MATCH_WINDOW = 5;
    const int WARMUP = 50;  /* Skip first ticks for BOCPD to stabilize */
    
    MMPF_Output output;
    
    for (int t = 0; t < total_ticks; t++) {
        double r = data->returns[t];
        double true_vol = data->true_vol[t];
        int true_regime = data->true_regime[t];
        int scenario = data->scenario_id[t];
        
        /* Transform to log(r²) for BOCPD */
        double y_log = (fabs(r) > 1e-10) ? log(r * r) : -20.0;
        
        /* Step BOCPD */
        bocpd_step(&bocpd, y_log);
        
        /* Check for shock detection (after warmup, respecting refractory) */
        double r0_prob = bocpd.r[0];
        
        int shock_fired = 0;
        if (t >= WARMUP) {
            if (refractory_counter > 0) {
                refractory_counter--;
            } else if (r0_prob > bocpd_r0_thresh) {
                shock_fired = 1;
                shock_count++;
                refractory_counter = bocpd_refractory;
                
                /* Check for flicker (rapid successive fires) */
                if (t - last_shock_tick < 20) {
                    flicker_count++;
                }
                last_shock_tick = t;
                
                /* Classify as TP or FP */
                /* Look backward and forward for true shock */
                int matched = 0;
                for (int dt = -MATCH_WINDOW; dt <= MATCH_WINDOW; dt++) {
                    int check_t = t + dt;
                    if (check_t >= 0 && check_t < total_ticks && data->is_shock_tick[check_t]) {
                        matched = 1;
                        if (n_detections < 1000) {
                            detection_lags[n_detections++] = (dt >= 0) ? dt : -dt;
                        }
                        break;
                    }
                }
                
                if (matched) {
                    true_positives++;
                } else {
                    false_positives++;
                }
            }
        }
        
        /* Update lockout */
        if (lockout_counter > 0) {
            lockout_counter--;
        }
        
        /* Step MMPF with optional shock injection */
        if (shock_fired && lockout_counter == 0) {
            mmpf_inject_shock(mmpf);
            mmpf_step(mmpf, (rbpf_real_t)r, &output);
            mmpf_restore_from_shock(mmpf);
            lockout_counter = post_shock_lockout;
        } else {
            mmpf_step(mmpf, (rbpf_real_t)r, &output);
        }
        
        /* Vol error */
        double vol_err = output.volatility - true_vol;
        sum_sq_err += vol_err * vol_err;
        
        double log_err = output.log_volatility - log(true_vol);
        sum_sq_log_err += log_err * log_err;
        
        /* Regime accuracy */
        int detected_regime = (int)output.dominant;
        if (detected_regime == true_regime) {
            correct_regime++;
            scenario_correct[scenario]++;
        }
        scenario_count[scenario]++;
        
        /* Transition lag tracking */
        if (true_regime != prev_true_regime) {
            looking_for_detection = 1;
            ticks_since_transition = 0;
        }
        
        if (looking_for_detection) {
            ticks_since_transition++;
            if (detected_regime == true_regime) {
                total_lag += ticks_since_transition;
                transition_count++;
                looking_for_detection = 0;
            }
            if (ticks_since_transition > 100) {
                total_lag += 100;
                transition_count++;
                looking_for_detection = 0;
            }
        }
        
        prev_true_regime = true_regime;
    }
    
    /* Compute metrics */
    metrics.vol_rmse = sqrt(sum_sq_err / total_ticks);
    metrics.log_vol_rmse = sqrt(sum_sq_log_err / total_ticks);
    metrics.hypothesis_accuracy = 100.0 * correct_regime / total_ticks;
    metrics.avg_transition_lag = (transition_count > 0) ? 
        (double)total_lag / transition_count : 0.0;
    
    for (int s = 0; s < SCENARIO_COUNT; s++) {
        metrics.per_scenario_accuracy[s] = (scenario_count[s] > 0) ?
            100.0 * scenario_correct[s] / scenario_count[s] : 0.0;
    }
    
    /* BOCPD metrics */
    metrics.true_positives = true_positives;
    metrics.false_positives = false_positives;
    metrics.false_negatives = data->n_true_shocks - true_positives;
    if (metrics.false_negatives < 0) metrics.false_negatives = 0;
    
    metrics.precision = (true_positives + false_positives > 0) ?
        (double)true_positives / (true_positives + false_positives) : 0.0;
    metrics.recall = (data->n_true_shocks > 0) ?
        (double)true_positives / data->n_true_shocks : 0.0;
    metrics.f1_score = (metrics.precision + metrics.recall > 0) ?
        2.0 * metrics.precision * metrics.recall / (metrics.precision + metrics.recall) : 0.0;
    
    double lag_sum = 0.0;
    for (int i = 0; i < n_detections; i++) {
        lag_sum += detection_lags[i];
    }
    metrics.avg_detection_lag = (n_detections > 0) ? lag_sum / n_detections : 0.0;
    
    metrics.shock_count = shock_count;
    metrics.flicker_count = flicker_count;
    
    /* Cleanup */
    mmpf_destroy(mmpf);
    bocpd_free(&bocpd);
    bocpd_hazard_free(&hazard);
    
    return metrics;
}

/*═══════════════════════════════════════════════════════════════════════════
 * COMPOSITE SCORE (Stage 2 emphasis on detection quality)
 *═══════════════════════════════════════════════════════════════════════════*/

static double compute_score(const TuningMetrics *m)
{
    /* Stage 2 priorities:
     * 1. Detection quality (F1 score) — want high
     * 2. Vol RMSE — want low
     * 3. Detection lag — want low
     * 4. Flicker — want low (penalize instability)
     */
    
    double norm_f1 = 1.0 - m->f1_score;  /* 0 = perfect, 1 = worst */
    double norm_rmse = m->vol_rmse / 0.05;
    double norm_lag = m->avg_detection_lag / 10.0;
    double norm_flicker = (double)m->flicker_count / 20.0;
    
    /* Weighted sum (lower is better) */
    return 0.35 * norm_f1 + 
           0.30 * norm_rmse + 
           0.20 * norm_lag +
           0.15 * norm_flicker;
}

/*═══════════════════════════════════════════════════════════════════════════
 * RESULTS STRUCTURE
 *═══════════════════════════════════════════════════════════════════════════*/

typedef struct {
    double bocpd_r0_thresh;
    int bocpd_refractory;
    double shock_mult;
    int post_shock_lockout;
    double crisis_phi;
    double stickiness;
    TuningMetrics metrics;
    double score;
} ConfigResult;

static void print_config(const ConfigResult *r) {
    printf("  BOCPD r0 threshold: %.3f\n", r->bocpd_r0_thresh);
    printf("  BOCPD refractory:   %d ticks\n", r->bocpd_refractory);
    printf("  Shock multiplier:   %.1f\n", r->shock_mult);
    printf("  Post-shock lockout: %d ticks\n", r->post_shock_lockout);
    printf("  Crisis φ:           %.2f\n", r->crisis_phi);
    printf("  Stickiness:         %.3f\n", r->stickiness);
    printf("  ─────────────────────────────\n");
    printf("  Vol RMSE:           %.4f\n", r->metrics.vol_rmse);
    printf("  Hypothesis Acc:     %.1f%%\n", r->metrics.hypothesis_accuracy);
    printf("  Transition Lag:     %.1f ticks\n", r->metrics.avg_transition_lag);
    printf("  ─────────────────────────────\n");
    printf("  BOCPD Precision:    %.1f%%\n", r->metrics.precision * 100);
    printf("  BOCPD Recall:       %.1f%%\n", r->metrics.recall * 100);
    printf("  BOCPD F1:           %.3f\n", r->metrics.f1_score);
    printf("  Detection Lag:      %.1f ticks\n", r->metrics.avg_detection_lag);
    printf("  Shock Count:        %d\n", r->metrics.shock_count);
    printf("  Flicker Count:      %d\n", r->metrics.flicker_count);
    printf("  ─────────────────────────────\n");
    printf("  Score:              %.4f\n", r->score);
}

/*═══════════════════════════════════════════════════════════════════════════
 * MAIN
 *═══════════════════════════════════════════════════════════════════════════*/

int main(int argc, char **argv) {
    int quick_mode = 0;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--quick") == 0) {
            quick_mode = 1;
        }
    }
    
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  MMPF Stage 2 Tuner: BOCPD Integration\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  Mode: %s\n", quick_mode ? "Quick" : "Full");
    printf("  Total configurations: %d\n", N_CONFIGS);
    printf("\n");
    
    /* Generate benchmark data */
    int ticks_per_scenario = quick_mode ? 500 : 1000;
    BenchmarkData *data = generate_benchmark(42, ticks_per_scenario);
    
    printf("  Benchmark: %d ticks, %d scenarios\n", data->n_ticks, SCENARIO_COUNT);
    printf("  True shocks: %d, Outliers: %d\n\n", data->n_true_shocks, data->n_outliers);
    
    /* Storage */
    ConfigResult *results = (ConfigResult*)malloc(N_CONFIGS * sizeof(ConfigResult));
    int n_results = 0;
    
    /* Best trackers */
    ConfigResult best_f1 = {.metrics.f1_score = 0};
    ConfigResult best_rmse = {.metrics.vol_rmse = 1e10};
    ConfigResult best_balanced = {.score = 1e10};
    
    /* Grid dimensions */
    int n_thresh = quick_mode ? 2 : N_BOCPD_THRESH;
    int n_refrac = quick_mode ? 2 : N_BOCPD_REFRACTORY;
    int n_shock = quick_mode ? 2 : N_SHOCK_MULT;
    int n_lock = quick_mode ? 2 : N_LOCKOUT;
    int n_phi = quick_mode ? 2 : N_CRISIS_PHI;
    int n_stick = quick_mode ? 2 : N_STICKINESS;
    
    int total_configs = n_thresh * n_refrac * n_shock * n_lock * n_phi * n_stick;
    printf("Running %d configurations...\n", total_configs);
    
    clock_t start = clock();
    int config_idx = 0;
    
    for (int i_th = 0; i_th < n_thresh; i_th++) {
        for (int i_rf = 0; i_rf < n_refrac; i_rf++) {
            for (int i_sh = 0; i_sh < n_shock; i_sh++) {
                for (int i_lo = 0; i_lo < n_lock; i_lo++) {
                    for (int i_ph = 0; i_ph < n_phi; i_ph++) {
                        for (int i_st = 0; i_st < n_stick; i_st++) {
                            
                            ConfigResult r;
                            r.bocpd_r0_thresh = BOCPD_R0_THRESH[i_th];
                            r.bocpd_refractory = BOCPD_REFRACTORY[i_rf];
                            r.shock_mult = SHOCK_MULTIPLIER[i_sh];
                            r.post_shock_lockout = POST_SHOCK_LOCKOUT[i_lo];
                            r.crisis_phi = CRISIS_PHI[i_ph];
                            r.stickiness = STICKINESS_WITH_BOCPD[i_st];
                            
                            r.metrics = run_config(data,
                                r.bocpd_r0_thresh,
                                r.bocpd_refractory,
                                r.shock_mult,
                                r.post_shock_lockout,
                                r.crisis_phi,
                                r.stickiness);
                            
                            r.score = compute_score(&r.metrics);
                            results[n_results++] = r;
                            
                            /* Track bests */
                            if (r.metrics.f1_score > best_f1.metrics.f1_score) {
                                best_f1 = r;
                            }
                            if (r.metrics.vol_rmse < best_rmse.metrics.vol_rmse) {
                                best_rmse = r;
                            }
                            if (r.score < best_balanced.score) {
                                best_balanced = r;
                            }
                            
                            config_idx++;
                            if (config_idx % 100 == 0) {
                                printf("  Progress: %d/%d (%.0f%%)\r",
                                       config_idx, total_configs,
                                       100.0 * config_idx / total_configs);
                                fflush(stdout);
                            }
                        }
                    }
                }
            }
        }
    }
    
    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("\n\nCompleted in %.1f seconds (%.1f ms/config)\n\n",
           elapsed, 1000.0 * elapsed / config_idx);
    
    /* Print results */
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  BEST FOR F1 SCORE (Detection Quality)\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    print_config(&best_f1);
    
    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("  BEST FOR VOL RMSE (Estimation Quality)\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    print_config(&best_rmse);
    
    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("  BEST BALANCED\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    print_config(&best_balanced);
    
    /* Write CSV */
    FILE *csv = fopen("mmpf_tuning_stage2_results.csv", "w");
    if (csv) {
        fprintf(csv, "r0_thresh,refractory,shock_mult,lockout,crisis_phi,stickiness,"
                     "vol_rmse,accuracy,transition_lag,"
                     "precision,recall,f1,detection_lag,shock_count,flicker,score\n");
        
        for (int i = 0; i < n_results; i++) {
            ConfigResult *r = &results[i];
            fprintf(csv, "%.3f,%d,%.1f,%d,%.2f,%.3f,"
                        "%.6f,%.2f,%.2f,"
                        "%.3f,%.3f,%.3f,%.2f,%d,%d,%.6f\n",
                    r->bocpd_r0_thresh, r->bocpd_refractory,
                    r->shock_mult, r->post_shock_lockout,
                    r->crisis_phi, r->stickiness,
                    r->metrics.vol_rmse, r->metrics.hypothesis_accuracy,
                    r->metrics.avg_transition_lag,
                    r->metrics.precision, r->metrics.recall,
                    r->metrics.f1_score, r->metrics.avg_detection_lag,
                    r->metrics.shock_count, r->metrics.flicker_count,
                    r->score);
        }
        fclose(csv);
        printf("\n  Results written to: mmpf_tuning_stage2_results.csv\n");
    }
    
    /* Recommended config */
    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("  RECOMMENDED INTEGRATION CONFIG\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("\n/* BOCPD Trigger */\n");
    printf("#define BOCPD_R0_THRESHOLD  %.3f\n", best_balanced.bocpd_r0_thresh);
    printf("#define BOCPD_REFRACTORY    %d\n", best_balanced.bocpd_refractory);
    printf("\n/* Shock Injection */\n");
    printf("#define SHOCK_MULTIPLIER    %.1f\n", best_balanced.shock_mult);
    printf("#define POST_SHOCK_LOCKOUT  %d\n", best_balanced.post_shock_lockout);
    printf("\n/* MMPF Config */\n");
    printf("cfg.hypotheses[MMPF_CRISIS].phi = RBPF_REAL(%.2f);\n", best_balanced.crisis_phi);
    printf("cfg.base_stickiness = RBPF_REAL(%.3f);\n", best_balanced.stickiness);
    
    /* Cleanup */
    free(results);
    free_benchmark(data);
    
    return 0;
}