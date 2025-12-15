/**
 * @file mmpf_tuner.c
 * @brief MMPF Parameter Calibration Harness
 *
 * Grid search over key parameters to find optimal configuration.
 * Optimizes for: Vol RMSE, Hypothesis Accuracy, Transition Lag
 *
 * Usage:
 *   ./mmpf_tuner [--quick]
 *
 * Output:
 *   - Console: Best configurations per metric
 *   - CSV: Full grid search results (mmpf_tuning_results.csv)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <float.h>

/* Include your MMPF headers */
#include "mmpf_rocks.h"

/*═══════════════════════════════════════════════════════════════════════════
 * PARAMETER SPACE DEFINITION
 *═══════════════════════════════════════════════════════════════════════════*/

/* Swim Lane: Calm μ_vol bounds */
static const double CALM_MU_MIN[] = {-6.5, -6.0, -5.5};
static const double CALM_MU_MAX[] = {-4.5, -4.0, -3.5};
#define N_CALM_MU 3

/* Swim Lane: Crisis μ_vol bounds */
static const double CRISIS_MU_MIN[] = {-3.5, -3.0, -2.5};
static const double CRISIS_MU_MAX[] = {-1.0, -0.5, 0.0};
#define N_CRISIS_MU 3

/* Dirichlet prior */
static const double DIRICHLET_ALPHA[] = {0.5, 1.0, 2.0};
static const double DIRICHLET_MASS[] = {50.0, 100.0, 200.0};
#define N_DIRICHLET_ALPHA 3
#define N_DIRICHLET_MASS 3

/* Student-t ν for Calm */
static const double CALM_NU[] = {10.0, 20.0, 30.0};
#define N_CALM_NU 3

/* Student-t ν for Crisis */
static const double CRISIS_NU[] = {3.0, 5.0, 8.0};
#define N_CRISIS_NU 3

/* Stickiness */
static const double BASE_STICKINESS[] = {0.95, 0.98, 0.99};
#define N_STICKINESS 3

/* Total configurations */
#define N_CONFIGS (N_CALM_MU * N_CRISIS_MU * N_DIRICHLET_ALPHA * N_DIRICHLET_MASS * N_CALM_NU * N_CRISIS_NU * N_STICKINESS)

/*═══════════════════════════════════════════════════════════════════════════
 * BENCHMARK DATA GENERATION (Same as your test)
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
    int *true_regime;      /* 0=Calm, 1=Trend, 2=Crisis */
    int *scenario_id;
    int n_ticks;
    int n_outliers;
} BenchmarkData;

/* Simple RNG for reproducibility */
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
    data->n_ticks = total;
    data->n_outliers = 0;
    
    int idx = 0;
    
    for (int s = 0; s < SCENARIO_COUNT; s++) {
        double vol, vol_drift;
        int regime;
        
        switch (s) {
            case SCENARIO_CALM:
                vol = 0.008;  /* 0.8% daily vol */
                vol_drift = 0.0;
                regime = 0;
                break;
            case SCENARIO_TREND:
                vol = 0.012;  /* 1.2% vol, drifting up */
                vol_drift = 0.00002;
                regime = 1;
                break;
            case SCENARIO_CRISIS:
                vol = 0.035;  /* 3.5% vol, sudden jump */
                vol_drift = 0.0;
                regime = 2;
                break;
            case SCENARIO_CRISIS_PERSIST:
                vol = 0.040;  /* 4% vol, persistent */
                vol_drift = 0.0;
                regime = 2;
                break;
            case SCENARIO_RECOVERY:
                vol = 0.025;  /* 2.5% declining to 1% */
                vol_drift = -0.00015;
                regime = 1;
                break;
            case SCENARIO_FLASH:
                vol = 0.010;  /* 1% with flash spike */
                vol_drift = 0.0;
                regime = 0;
                break;
            case SCENARIO_CHOPPY:
                vol = 0.018;  /* 1.8% choppy */
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
            
            /* Flash crash in middle of flash scenario */
            if (s == SCENARIO_FLASH && t >= ticks_per_scenario/2 - 5 && t < ticks_per_scenario/2 + 5) {
                current_vol = 0.08;
                regime = 2;
            } else if (s == SCENARIO_FLASH) {
                regime = 0;
            }
            
            /* Generate return */
            double r = current_vol * rand_normal();
            
            /* Occasional outlier */
            if (rand_uniform() < 0.001) {
                r *= 5.0;
                data->n_outliers++;
            }
            
            data->returns[idx] = r;
            data->true_vol[idx] = current_vol;
            data->true_regime[idx] = regime;
            data->scenario_id[idx] = s;
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
        free(data);
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * METRICS
 *═══════════════════════════════════════════════════════════════════════════*/

typedef struct {
    double vol_rmse;
    double log_vol_rmse;
    double hypothesis_accuracy;
    double avg_transition_lag;
    double false_crisis_count;
    double per_scenario_accuracy[SCENARIO_COUNT];
    double latency_median_us;
    double latency_p99_us;
} TuningMetrics;

/*═══════════════════════════════════════════════════════════════════════════
 * RUN SINGLE CONFIGURATION
 *═══════════════════════════════════════════════════════════════════════════*/

static TuningMetrics run_config(
    const BenchmarkData *data,
    double calm_mu_min, double calm_mu_max,
    double crisis_mu_min, double crisis_mu_max,
    double dirichlet_alpha, double dirichlet_mass,
    double calm_nu,
    double crisis_nu,
    double base_stickiness)
{
    TuningMetrics metrics = {0};
    
    /* Create config */
    MMPF_Config cfg = mmpf_config_defaults();
    
    /* Apply tuning parameters */
    
    /* Swim lanes - Calm */
    cfg.swim_lanes[MMPF_CALM].mu_vol_min = (rbpf_real_t)calm_mu_min;
    cfg.swim_lanes[MMPF_CALM].mu_vol_max = (rbpf_real_t)calm_mu_max;
    cfg.hypotheses[MMPF_CALM].mu_vol = (rbpf_real_t)((calm_mu_min + calm_mu_max) / 2.0);
    
    /* Swim lanes - Crisis */
    cfg.swim_lanes[MMPF_CRISIS].mu_vol_min = (rbpf_real_t)crisis_mu_min;
    cfg.swim_lanes[MMPF_CRISIS].mu_vol_max = (rbpf_real_t)crisis_mu_max;
    cfg.hypotheses[MMPF_CRISIS].mu_vol = (rbpf_real_t)((crisis_mu_min + crisis_mu_max) / 2.0);
    
    /* Dirichlet */
    cfg.transition_prior_alpha = (rbpf_real_t)dirichlet_alpha;
    cfg.transition_prior_mass = (rbpf_real_t)dirichlet_mass;
    
    /* Student-t ν - both hypotheses */
    cfg.hypothesis_nu[MMPF_CALM] = (rbpf_real_t)calm_nu;
    cfg.hypotheses[MMPF_CALM].nu = (rbpf_real_t)calm_nu;
    cfg.hypothesis_nu[MMPF_CRISIS] = (rbpf_real_t)crisis_nu;
    cfg.hypotheses[MMPF_CRISIS].nu = (rbpf_real_t)crisis_nu;
    
    /* Stickiness */
    cfg.base_stickiness = (rbpf_real_t)base_stickiness;
    
    /* Create MMPF */
    MMPF_ROCKS *mmpf = mmpf_create(&cfg);
    if (!mmpf) {
        metrics.vol_rmse = 1e10;  /* Invalid config */
        return metrics;
    }
    
    /* Accumulators */
    double sum_sq_err = 0.0;
    double sum_sq_log_err = 0.0;
    int correct_regime = 0;
    int total_ticks = data->n_ticks;
    int scenario_correct[SCENARIO_COUNT] = {0};
    int scenario_count[SCENARIO_COUNT] = {0};
    int false_crisis = 0;
    
    /* Track transitions for lag calculation */
    int prev_true_regime = data->true_regime[0];
    int transition_count = 0;
    int total_lag = 0;
    int ticks_since_transition = 0;
    int looking_for_detection = 0;
    int target_regime = 0;
    
    /* Run */
    MMPF_Output output;
    
    for (int t = 0; t < total_ticks; t++) {
        double r = data->returns[t];
        double true_vol = data->true_vol[t];
        int true_regime = data->true_regime[t];
        int scenario = data->scenario_id[t];
        
        mmpf_step(mmpf, (rbpf_real_t)r, &output);
        
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
        
        /* False crisis */
        if (detected_regime == 2 && true_regime != 2) {
            false_crisis++;
        }
        
        /* Transition lag tracking */
        if (true_regime != prev_true_regime) {
            /* True regime changed */
            looking_for_detection = 1;
            target_regime = true_regime;
            ticks_since_transition = 0;
        }
        
        if (looking_for_detection) {
            ticks_since_transition++;
            if (detected_regime == target_regime) {
                total_lag += ticks_since_transition;
                transition_count++;
                looking_for_detection = 0;
            }
            if (ticks_since_transition > 100) {
                /* Give up, count as max lag */
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
    metrics.false_crisis_count = (double)false_crisis;
    
    for (int s = 0; s < SCENARIO_COUNT; s++) {
        metrics.per_scenario_accuracy[s] = (scenario_count[s] > 0) ?
            100.0 * scenario_correct[s] / scenario_count[s] : 0.0;
    }
    
    mmpf_destroy(mmpf);
    
    return metrics;
}

/*═══════════════════════════════════════════════════════════════════════════
 * COMPOSITE SCORE
 *═══════════════════════════════════════════════════════════════════════════*/

static double compute_score(const TuningMetrics *m, 
                           double w_vol_rmse,
                           double w_accuracy,
                           double w_lag,
                           double w_false_crisis)
{
    /* Lower is better for all metrics in this formulation */
    /* Normalize each metric to ~[0,1] range */
    
    double norm_rmse = m->vol_rmse / 0.05;  /* 5% vol = score 1.0 */
    double norm_acc = (100.0 - m->hypothesis_accuracy) / 100.0;  /* 100% acc = 0 */
    double norm_lag = m->avg_transition_lag / 20.0;  /* 20 ticks = score 1.0 */
    double norm_fc = m->false_crisis_count / 500.0;  /* 500 false = score 1.0 */
    
    return w_vol_rmse * norm_rmse + 
           w_accuracy * norm_acc + 
           w_lag * norm_lag +
           w_false_crisis * norm_fc;
}

/*═══════════════════════════════════════════════════════════════════════════
 * GRID SEARCH
 *═══════════════════════════════════════════════════════════════════════════*/

typedef struct {
    double calm_mu_min, calm_mu_max;
    double crisis_mu_min, crisis_mu_max;
    double dirichlet_alpha, dirichlet_mass;
    double calm_nu;
    double crisis_nu;
    double base_stickiness;
    TuningMetrics metrics;
    double score;
} ConfigResult;

static void print_scenario_name(int s) {
    const char* names[] = {"Calm", "Trend", "Crisis", "CrisisPersist", 
                          "Recovery", "Flash", "Choppy"};
    printf("%s", names[s]);
}

static void print_config(const ConfigResult *r) {
    printf("  Calm μ:     [%.1f, %.1f]\n", r->calm_mu_min, r->calm_mu_max);
    printf("  Crisis μ:   [%.1f, %.1f]\n", r->crisis_mu_min, r->crisis_mu_max);
    printf("  Dirichlet:  α=%.1f, N=%.0f\n", r->dirichlet_alpha, r->dirichlet_mass);
    printf("  Calm ν:     %.0f\n", r->calm_nu);
    printf("  Crisis ν:   %.0f\n", r->crisis_nu);
    printf("  ν ratio:    %.1f (Calm/Crisis)\n", r->calm_nu / r->crisis_nu);
    printf("  Stickiness: %.2f\n", r->base_stickiness);
    printf("  ─────────────────────────────\n");
    printf("  Vol RMSE:       %.4f\n", r->metrics.vol_rmse);
    printf("  Hypothesis Acc: %.1f%%\n", r->metrics.hypothesis_accuracy);
    printf("  Transition Lag: %.1f ticks\n", r->metrics.avg_transition_lag);
    printf("  False Crisis:   %.0f\n", r->metrics.false_crisis_count);
    printf("  Score:          %.4f\n", r->score);
}

int main(int argc, char **argv) {
    int quick_mode = 0;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--quick") == 0) {
            quick_mode = 1;
        }
    }
    
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  MMPF Parameter Tuner\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  Mode: %s\n", quick_mode ? "Quick (reduced grid)" : "Full");
    printf("  Total configurations: %d\n", N_CONFIGS);
    printf("\n");
    
    /* Generate benchmark data */
    int ticks_per_scenario = quick_mode ? 500 : 1000;
    BenchmarkData *data = generate_benchmark(42, ticks_per_scenario);
    
    printf("  Benchmark: %d ticks, %d scenarios, %d outliers\n\n", 
           data->n_ticks, SCENARIO_COUNT, data->n_outliers);
    
    /* Storage for results */
    ConfigResult *results = (ConfigResult*)malloc(N_CONFIGS * sizeof(ConfigResult));
    int n_results = 0;
    
    /* Best trackers */
    ConfigResult best_vol_rmse = {.metrics.vol_rmse = 1e10};
    ConfigResult best_accuracy = {.metrics.hypothesis_accuracy = 0};
    ConfigResult best_lag = {.metrics.avg_transition_lag = 1e10};
    ConfigResult best_balanced = {.score = 1e10};
    
    /* Grid search */
    int config_idx = 0;
    int total_configs = N_CONFIGS;
    
    /* Reduce grid in quick mode */
    int n_calm_mu = quick_mode ? 2 : N_CALM_MU;
    int n_crisis_mu = quick_mode ? 2 : N_CRISIS_MU;
    int n_dirichlet_alpha = quick_mode ? 2 : N_DIRICHLET_ALPHA;
    int n_dirichlet_mass = quick_mode ? 2 : N_DIRICHLET_MASS;
    int n_calm_nu = quick_mode ? 2 : N_CALM_NU;
    int n_crisis_nu = quick_mode ? 2 : N_CRISIS_NU;
    int n_stickiness = quick_mode ? 2 : N_STICKINESS;
    
    if (quick_mode) {
        total_configs = n_calm_mu * n_crisis_mu * n_dirichlet_alpha * 
                       n_dirichlet_mass * n_calm_nu * n_crisis_nu * n_stickiness;
    }
    
    printf("Running %d configurations...\n", total_configs);
    
    clock_t start = clock();
    
    for (int i_cm = 0; i_cm < n_calm_mu; i_cm++) {
        for (int i_cr = 0; i_cr < n_crisis_mu; i_cr++) {
            for (int i_da = 0; i_da < n_dirichlet_alpha; i_da++) {
                for (int i_dm = 0; i_dm < n_dirichlet_mass; i_dm++) {
                    for (int i_nu = 0; i_nu < n_calm_nu; i_nu++) {
                        for (int i_cnu = 0; i_cnu < n_crisis_nu; i_cnu++) {
                            for (int i_st = 0; i_st < n_stickiness; i_st++) {
                            
                            ConfigResult r;
                            r.calm_mu_min = CALM_MU_MIN[i_cm];
                            r.calm_mu_max = CALM_MU_MAX[i_cm];
                            r.crisis_mu_min = CRISIS_MU_MIN[i_cr];
                            r.crisis_mu_max = CRISIS_MU_MAX[i_cr];
                            r.dirichlet_alpha = DIRICHLET_ALPHA[i_da];
                            r.dirichlet_mass = DIRICHLET_MASS[i_dm];
                            r.calm_nu = CALM_NU[i_nu];
                            r.crisis_nu = CRISIS_NU[i_cnu];
                            r.base_stickiness = BASE_STICKINESS[i_st];
                            
                            r.metrics = run_config(data,
                                r.calm_mu_min, r.calm_mu_max,
                                r.crisis_mu_min, r.crisis_mu_max,
                                r.dirichlet_alpha, r.dirichlet_mass,
                                r.calm_nu, r.crisis_nu, r.base_stickiness);
                            
                            /* Balanced score: equal weights */
                            r.score = compute_score(&r.metrics, 0.3, 0.3, 0.2, 0.2);
                            
                            results[n_results++] = r;
                            
                            /* Track bests */
                            if (r.metrics.vol_rmse < best_vol_rmse.metrics.vol_rmse) {
                                best_vol_rmse = r;
                            }
                            if (r.metrics.hypothesis_accuracy > best_accuracy.metrics.hypothesis_accuracy) {
                                best_accuracy = r;
                            }
                            if (r.metrics.avg_transition_lag < best_lag.metrics.avg_transition_lag) {
                                best_lag = r;
                            }
                            if (r.score < best_balanced.score) {
                                best_balanced = r;
                            }
                            
                            config_idx++;
                            if (config_idx % 50 == 0) {
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
    }
    
    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("\n\nCompleted in %.1f seconds (%.1f ms/config)\n\n", 
           elapsed, 1000.0 * elapsed / config_idx);
    
    /* Print results */
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  BEST FOR VOL RMSE\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    print_config(&best_vol_rmse);
    
    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("  BEST FOR HYPOTHESIS ACCURACY\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    print_config(&best_accuracy);
    
    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("  BEST FOR TRANSITION LAG\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    print_config(&best_lag);
    
    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("  BEST BALANCED (0.3×RMSE + 0.3×Acc + 0.2×Lag + 0.2×FC)\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    print_config(&best_balanced);
    
    /* Per-scenario breakdown for balanced best */
    printf("\n  Per-Scenario Accuracy:\n");
    for (int s = 0; s < SCENARIO_COUNT; s++) {
        printf("    ");
        print_scenario_name(s);
        printf(": %.1f%%\n", best_balanced.metrics.per_scenario_accuracy[s]);
    }
    
    /* Write CSV */
    FILE *csv = fopen("mmpf_tuning_results.csv", "w");
    if (csv) {
        fprintf(csv, "calm_mu_min,calm_mu_max,crisis_mu_min,crisis_mu_max,"
                     "dirichlet_alpha,dirichlet_mass,calm_nu,crisis_nu,nu_ratio,stickiness,"
                     "vol_rmse,log_vol_rmse,accuracy,transition_lag,false_crisis,score,"
                     "acc_calm,acc_trend,acc_crisis,acc_crisis_persist,"
                     "acc_recovery,acc_flash,acc_choppy\n");
        
        for (int i = 0; i < n_results; i++) {
            ConfigResult *r = &results[i];
            fprintf(csv, "%.1f,%.1f,%.1f,%.1f,%.1f,%.0f,%.0f,%.0f,%.2f,%.2f,"
                        "%.6f,%.6f,%.2f,%.2f,%.0f,%.6f,"
                        "%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n",
                    r->calm_mu_min, r->calm_mu_max,
                    r->crisis_mu_min, r->crisis_mu_max,
                    r->dirichlet_alpha, r->dirichlet_mass,
                    r->calm_nu, r->crisis_nu, r->calm_nu / r->crisis_nu,
                    r->base_stickiness,
                    r->metrics.vol_rmse, r->metrics.log_vol_rmse,
                    r->metrics.hypothesis_accuracy,
                    r->metrics.avg_transition_lag,
                    r->metrics.false_crisis_count,
                    r->score,
                    r->metrics.per_scenario_accuracy[0],
                    r->metrics.per_scenario_accuracy[1],
                    r->metrics.per_scenario_accuracy[2],
                    r->metrics.per_scenario_accuracy[3],
                    r->metrics.per_scenario_accuracy[4],
                    r->metrics.per_scenario_accuracy[5],
                    r->metrics.per_scenario_accuracy[6]);
        }
        fclose(csv);
        printf("\n  Results written to: mmpf_tuning_results.csv\n");
    }
    
    /* Recommended config as C code */
    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("  RECOMMENDED CONFIG (Copy to your code)\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("\nMMPF_Config cfg = mmpf_config_defaults();\n\n");
    printf("/* Swim Lanes - Calm */\n");
    printf("cfg.swim_lanes[MMPF_CALM].mu_vol_min = RBPF_REAL(%.1f);\n", best_balanced.calm_mu_min);
    printf("cfg.swim_lanes[MMPF_CALM].mu_vol_max = RBPF_REAL(%.1f);\n", best_balanced.calm_mu_max);
    printf("\n/* Swim Lanes - Crisis */\n");
    printf("cfg.swim_lanes[MMPF_CRISIS].mu_vol_min = RBPF_REAL(%.1f);\n", best_balanced.crisis_mu_min);
    printf("cfg.swim_lanes[MMPF_CRISIS].mu_vol_max = RBPF_REAL(%.1f);\n", best_balanced.crisis_mu_max);
    printf("\n/* Dirichlet Prior */\n");
    printf("cfg.transition_prior_alpha = RBPF_REAL(%.1f);\n", best_balanced.dirichlet_alpha);
    printf("cfg.transition_prior_mass = RBPF_REAL(%.0f);\n", best_balanced.dirichlet_mass);
    printf("\n/* Student-t (ν ratio = %.1f) */\n", best_balanced.calm_nu / best_balanced.crisis_nu);
    printf("cfg.hypothesis_nu[MMPF_CALM] = RBPF_REAL(%.0f);\n", best_balanced.calm_nu);
    printf("cfg.hypotheses[MMPF_CALM].nu = RBPF_REAL(%.0f);\n", best_balanced.calm_nu);
    printf("cfg.hypothesis_nu[MMPF_CRISIS] = RBPF_REAL(%.0f);\n", best_balanced.crisis_nu);
    printf("cfg.hypotheses[MMPF_CRISIS].nu = RBPF_REAL(%.0f);\n", best_balanced.crisis_nu);
    printf("\n/* Stickiness */\n");
    printf("cfg.base_stickiness = RBPF_REAL(%.2f);\n", best_balanced.base_stickiness);
    
    /* Cleanup */
    free(results);
    free_benchmark(data);
    
    return 0;
}
