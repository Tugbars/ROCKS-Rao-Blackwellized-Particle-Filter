/**
 * @file rbpf_viz_export.c
 * @brief Export RBPF performance data as CSV for visualization
 *
 * Runs the Best Vol RMSE config on 7-scenario synthetic data
 * and exports tick-by-tick data for plotting.
 *
 * Output: rbpf_viz_data.csv
 *
 * Usage:
 *   ./rbpf_viz_export [seed]
 *
 * Default seed: 42 (matches test_mmpf_comparison.c)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "rbpf_ksc.h"
#include "rbpf_ksc_param_integration.h"

/*═══════════════════════════════════════════════════════════════════════════
 * PCG32 RNG (matches test_mmpf_comparison.c exactly)
 *═══════════════════════════════════════════════════════════════════════════*/

typedef struct {
    uint64_t state;
    uint64_t inc;
} pcg32_t;

static uint32_t pcg32_random(pcg32_t *rng) {
    uint64_t oldstate = rng->state;
    rng->state = oldstate * 6364136223846793005ULL + rng->inc;
    uint32_t xorshifted = (uint32_t)(((oldstate >> 18u) ^ oldstate) >> 27u);
    uint32_t rot = (uint32_t)(oldstate >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

static double pcg32_double(pcg32_t *rng) {
    return (double)pcg32_random(rng) / 4294967296.0;
}

static double pcg32_gaussian(pcg32_t *rng) {
    double u1 = pcg32_double(rng);
    double u2 = pcg32_double(rng);
    if (u1 < 1e-10) u1 = 1e-10;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979 * u2);
}

/*═══════════════════════════════════════════════════════════════════════════
 * HYPOTHESIS DEFINITIONS
 *═══════════════════════════════════════════════════════════════════════════*/

typedef enum {
    HYPO_CALM = 0,
    HYPO_TREND = 1,
    HYPO_CRISIS = 2,
    N_HYPOTHESES = 3
} Hypothesis;

static const char *hypo_names[] = {"CALM", "TREND", "CRISIS"};

typedef struct {
    double mu_vol;
    double phi;
    double sigma_eta;
} HypothesisParams;

static const HypothesisParams TRUE_PARAMS[N_HYPOTHESES] = {
    {.mu_vol = -5.0, .phi = 0.995, .sigma_eta = 0.08},
    {.mu_vol = -3.5, .phi = 0.95,  .sigma_eta = 0.20},
    {.mu_vol = -1.5, .phi = 0.85,  .sigma_eta = 0.50}
};

/*═══════════════════════════════════════════════════════════════════════════
 * SCENARIO DEFINITIONS
 *═══════════════════════════════════════════════════════════════════════════*/

typedef struct {
    int start;
    int end;
    const char *name;
} Scenario;

static const Scenario SCENARIOS[] = {
    {0,    1500, "Extended_Calm"},
    {1500, 2500, "Slow_Trend"},
    {2500, 3000, "Sudden_Crisis"},
    {3000, 4000, "Crisis_Persist"},
    {4000, 5200, "Recovery"},
    {5200, 5700, "Flash_Crash"},
    {5700, 8000, "Choppy"}
};
#define N_SCENARIOS 7

static const char *get_scenario(int t) {
    for (int i = 0; i < N_SCENARIOS; i++) {
        if (t >= SCENARIOS[i].start && t < SCENARIOS[i].end) {
            return SCENARIOS[i].name;
        }
    }
    return "Unknown";
}

/*═══════════════════════════════════════════════════════════════════════════
 * SYNTHETIC DATA GENERATION (exact match to test_mmpf_comparison.c)
 *═══════════════════════════════════════════════════════════════════════════*/

typedef struct {
    double *returns;
    double *true_log_vol;
    double *true_vol;
    int *true_hypo;
    int *is_outlier;
    double *outlier_sigma;
    int n_ticks;
    int n_outliers;
} SyntheticData;

static void inject_outlier(SyntheticData *data, int t, double sigma, pcg32_t *rng) {
    double sign = (pcg32_double(rng) < 0.5) ? -1.0 : 1.0;
    data->returns[t] = sign * sigma * data->true_vol[t];
    data->is_outlier[t] = 1;
    data->outlier_sigma[t] = sigma;
    data->n_outliers++;
}

static SyntheticData *generate_data(int seed) {
    int n = 8000;
    SyntheticData *data = calloc(1, sizeof(SyntheticData));
    data->returns = malloc(n * sizeof(double));
    data->true_log_vol = malloc(n * sizeof(double));
    data->true_vol = malloc(n * sizeof(double));
    data->true_hypo = malloc(n * sizeof(int));
    data->is_outlier = calloc(n, sizeof(int));
    data->outlier_sigma = calloc(n, sizeof(double));
    data->n_ticks = n;
    data->n_outliers = 0;
    
    pcg32_t rng = {seed * 12345ULL + 1, seed * 67890ULL | 1};
    
    double log_vol = TRUE_PARAMS[HYPO_CALM].mu_vol;
    int t = 0;

#define EVOLVE(H) do { \
    const HypothesisParams *p = &TRUE_PARAMS[H]; \
    double theta = 1.0 - p->phi; \
    log_vol = p->phi * log_vol + theta * p->mu_vol + p->sigma_eta * pcg32_gaussian(&rng); \
    data->true_log_vol[t] = log_vol; \
    data->true_vol[t] = exp(log_vol); \
    data->returns[t] = data->true_vol[t] * pcg32_gaussian(&rng); \
    data->true_hypo[t] = (H); \
} while(0)

    /* Scenario 1: Extended Calm (0-1499) */
    for (; t < 1500; t++) EVOLVE(HYPO_CALM);
    inject_outlier(data, 500, 6.0, &rng);
    inject_outlier(data, 1200, 8.0, &rng);

    /* Scenario 2: Slow Trend (1500-2499) */
    for (; t < 2500; t++) {
        Hypothesis h = (t < 1800) ? HYPO_CALM : HYPO_TREND;
        EVOLVE(h);
    }

    /* Scenario 3: Sudden Crisis (2500-2999) */
    for (; t < 3000; t++) EVOLVE(HYPO_CRISIS);
    inject_outlier(data, 2510, 8.0, &rng);
    inject_outlier(data, 2530, 10.0, &rng);
    inject_outlier(data, 2560, 12.0, &rng);
    inject_outlier(data, 2650, 9.0, &rng);
    inject_outlier(data, 2800, 11.0, &rng);

    /* Scenario 4: Crisis Persistence (3000-3999) */
    for (; t < 4000; t++) EVOLVE(HYPO_CRISIS);
    inject_outlier(data, 3200, 10.0, &rng);
    inject_outlier(data, 3500, 15.0, &rng);
    inject_outlier(data, 3800, 12.0, &rng);

    /* Scenario 5: Recovery (4000-5199) */
    for (; t < 5200; t++) {
        Hypothesis h = (t < 4400) ? HYPO_CRISIS : (t < 4800) ? HYPO_TREND : HYPO_CALM;
        EVOLVE(h);
    }

    /* Scenario 6: Flash Crash (5200-5699) */
    for (; t < 5700; t++) {
        Hypothesis h = (t >= 5350 && t < 5410) ? HYPO_CRISIS : HYPO_CALM;
        EVOLVE(h);
    }
    inject_outlier(data, 5380, 12.0, &rng);

    /* Scenario 7: Choppy (5700-7999) */
    Hypothesis current_h = HYPO_TREND;
    int next_switch = 5700 + 80 + (int)(pcg32_double(&rng) * 120);
    for (; t < 8000; t++) {
        if (t >= next_switch) {
            int delta = (pcg32_double(&rng) < 0.5) ? -1 : 1;
            current_h = (Hypothesis)((current_h + delta + N_HYPOTHESES) % N_HYPOTHESES);
            next_switch = t + 80 + (int)(pcg32_double(&rng) * 150);
        }
        EVOLVE(current_h);
    }

#undef EVOLVE
    return data;
}

static void free_data(SyntheticData *data) {
    if (!data) return;
    free(data->returns);
    free(data->true_log_vol);
    free(data->true_vol);
    free(data->true_hypo);
    free(data->is_outlier);
    free(data->outlier_sigma);
    free(data);
}

/*═══════════════════════════════════════════════════════════════════════════
 * REGIME TO HYPOTHESIS MAPPING
 *═══════════════════════════════════════════════════════════════════════════*/

static int regime_to_hypo(int regime) {
    if (regime <= 1) return HYPO_CALM;
    if (regime == 2) return HYPO_TREND;
    return HYPO_CRISIS;
}

/*═══════════════════════════════════════════════════════════════════════════
 * MAIN
 *═══════════════════════════════════════════════════════════════════════════*/

int main(int argc, char **argv) {
    int seed = 42;
    if (argc > 1) seed = atoi(argv[1]);
    
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  RBPF Visualization Export\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  Seed: %d\n", seed);
    printf("  Config: Best Vol RMSE\n\n");
    
    /* Generate data */
    printf("Generating synthetic data...\n");
    SyntheticData *data = generate_data(seed);
    printf("  Ticks: %d\n", data->n_ticks);
    printf("  Outliers: %d\n\n", data->n_outliers);
    
    /* Create RBPF with BEST VOL RMSE config */
    const int N_PARTICLES = 512;
    const int N_REGIMES = 4;
    
    RBPF_Extended *ext = rbpf_ext_create(N_PARTICLES, N_REGIMES, RBPF_PARAM_STORVIK);
    rbpf_ext_enable_kl_tempering(ext);

    /* Enable PARIS smoothed Storvik (L=50 tick lag) */
    rbpf_ext_enable_smoothed_storvik(ext, 5);
    
    /* ═══════════════════════════════════════════════════════════════════════
     * BEST VOL RMSE CONFIG
     * μ_calm=-4.50  μ_crisis=-2.00
     * σ_calm=0.080  σ_ratio=8.0
     * θ_calm=0.0030  θ_ratio=40.0
     * stickiness=0.92  λ_calm=0.9990
     * ═══════════════════════════════════════════════════════════════════════*/
    
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
    
    /* Enable adaptive forgetting in REGIME mode (uses fixed λ, no surprise modulation) */
    rbpf_ext_enable_adaptive_forgetting_mode(ext, ADAPT_SIGNAL_REGIME);
    // rbpf_ext_enable_adaptive_forgetting(ext);

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
    
    /* Open CSV */
    FILE *csv = fopen("rbpf_viz_data.csv", "w");
    if (!csv) {
        fprintf(stderr, "Failed to open output file\n");
        return 1;
    }
    
    /* Header */
    fprintf(csv, "tick,scenario,true_vol,est_vol,true_log_vol,est_log_vol,est_log_vol_std,"
                 "true_hypo,est_hypo,hypo_correct,return,is_outlier,outlier_sigma,"
                 "ess,regime_prob_0,regime_prob_1,regime_prob_2,regime_prob_3,"
                 "surprise,outlier_frac\n");
    
    /* Run and export */
    printf("Running RBPF...\n");
    RBPF_KSC_Output output;
    
    int correct_count = 0;
    double sum_sq_err = 0.0;
    
    for (int t = 0; t < data->n_ticks; t++) {
        rbpf_ext_step(ext, (rbpf_real_t)data->returns[t], &output);
        
        int true_hypo = data->true_hypo[t];
        int est_hypo = regime_to_hypo(output.dominant_regime);
        int hypo_correct = (true_hypo == est_hypo) ? 1 : 0;
        correct_count += hypo_correct;
        
        double log_err = output.log_vol_mean - data->true_log_vol[t];
        sum_sq_err += log_err * log_err;
        
        fprintf(csv, "%d,%s,%.8f,%.8f,%.6f,%.6f,%.6f,"
                     "%d,%d,%d,%.10f,%d,%.1f,"
                     "%.2f,%.4f,%.4f,%.4f,%.4f,"
                     "%.4f,%.4f\n",
                t,
                get_scenario(t),
                data->true_vol[t],
                output.vol_mean,
                data->true_log_vol[t],
                output.log_vol_mean,
                sqrt(output.log_vol_var),
                true_hypo,
                est_hypo,
                hypo_correct,
                data->returns[t],
                data->is_outlier[t],
                data->outlier_sigma[t],
                output.ess,
                output.regime_probs[0],
                output.regime_probs[1],
                output.regime_probs[2],
                output.regime_probs[3],
                output.surprise,
                output.outlier_fraction);
    }
    
    fclose(csv);
    
    /* Summary stats */
    double accuracy = 100.0 * correct_count / data->n_ticks;
    double rmse = sqrt(sum_sq_err / data->n_ticks);
    
    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("  SUMMARY\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  Log-Vol RMSE:      %.4f\n", rmse);
    printf("  Hypothesis Acc:    %.1f%%\n", accuracy);
    printf("  Output:            rbpf_viz_data.csv\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    
    /* Cleanup */
    rbpf_ext_destroy(ext);
    free_data(data);
    
    printf("\nDone! Import rbpf_viz_data.csv into Python/matplotlib for plots.\n");
    
    return 0;
}
