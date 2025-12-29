/**
 * @file test_hawkes_timing.c
 * @brief Test Hawkes Integrator timing on synthetic regime-switching data
 *
 * Workflow:
 *   1. Generate synthetic data
 *   2. Run Hawkes MLE to learn optimal α, β, λ₀ from data
 *   3. Test Hawkes integrator with learned parameters
 *   4. Compare against default/responsive/conservative configs
 *
 * Verifies that Hawkes:
 *   1. Fires during actual crisis scenarios
 *   2. Does NOT fire during calm periods
 *   3. Fires BEFORE or AT regime transitions (not after)
 *
 * Uses the same synthetic data as test_pgas_injection.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

/* Include your Hawkes headers */
#include "hawkes_integrator.h"
#include "smc2_hawkes_mle.h"

/*─────────────────────────────────────────────────────────────────────────────
 * PCG32 RNG (same as test_pgas_injection)
 *───────────────────────────────────────────────────────────────────────────*/

typedef struct {
    uint64_t state;
    uint64_t inc;
} pcg32_t;

static uint32_t pcg32_random(pcg32_t* rng) {
    uint64_t oldstate = rng->state;
    rng->state = oldstate * 6364136223846793005ULL + rng->inc;
    uint32_t xorshifted = (uint32_t)(((oldstate >> 18u) ^ oldstate) >> 27u);
    uint32_t rot = (uint32_t)(oldstate >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

static double pcg32_double(pcg32_t* rng) {
    return (double)pcg32_random(rng) / 4294967296.0;
}

static double pcg32_gaussian(pcg32_t* rng) {
    double u1 = pcg32_double(rng);
    double u2 = pcg32_double(rng);
    if (u1 < 1e-10) u1 = 1e-10;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979 * u2);
}

/*─────────────────────────────────────────────────────────────────────────────
 * HYPOTHESIS / REGIME DEFINITIONS
 *───────────────────────────────────────────────────────────────────────────*/

typedef enum {
    HYPO_CALM = 0,
    HYPO_TREND = 1,
    HYPO_CRISIS = 2,
    N_HYPOTHESES = 3
} Hypothesis;

static const char* hypothesis_names[] = {"CALM", "TREND", "CRISIS"};

typedef struct {
    double mu_vol;
    double phi;
    double sigma_eta;
    double vol_approx;
} HypothesisParams;

static const HypothesisParams TRUE_PARAMS[N_HYPOTHESES] = {
    {.mu_vol = -5.0,  .phi = 0.995, .sigma_eta = 0.08, .vol_approx = 0.007},
    {.mu_vol = -3.5,  .phi = 0.95,  .sigma_eta = 0.20, .vol_approx = 0.030},
    {.mu_vol = -1.5,  .phi = 0.85,  .sigma_eta = 0.50, .vol_approx = 0.220}
};

/*─────────────────────────────────────────────────────────────────────────────
 * SYNTHETIC DATA (Same as test_pgas_injection)
 *───────────────────────────────────────────────────────────────────────────*/

typedef struct {
    double* returns;
    double* true_log_vol;
    double* true_vol;
    int* true_hypothesis;
    int* is_outlier;
    int n_ticks;
    
    /* Scenario tracking */
    int scenario_starts[10];
    const char* scenario_names[10];
    int n_scenarios;
    int n_outliers_injected;
} SyntheticData;

static void inject_outlier(SyntheticData* data, int t, double target_sigma, pcg32_t* rng) {
    double vol = data->true_vol[t];
    double sign = (pcg32_double(rng) < 0.5) ? -1.0 : 1.0;
    data->returns[t] = sign * target_sigma * vol;
    data->is_outlier[t] = 1;
    data->n_outliers_injected++;
}

static SyntheticData* generate_test_data(int seed) {
    SyntheticData* data = (SyntheticData*)calloc(1, sizeof(SyntheticData));
    
    int n = 8000;
    data->n_ticks = n;
    data->returns = (double*)malloc(n * sizeof(double));
    data->true_log_vol = (double*)malloc(n * sizeof(double));
    data->true_vol = (double*)malloc(n * sizeof(double));
    data->true_hypothesis = (int*)malloc(n * sizeof(int));
    data->is_outlier = (int*)calloc(n, sizeof(int));
    
    pcg32_t rng = {seed * 12345ULL + 1, seed * 67890ULL | 1};
    
    double log_vol = TRUE_PARAMS[HYPO_CALM].mu_vol;
    int t = 0;
    
#define EVOLVE_STATE(H) do { \
    const HypothesisParams* p = &TRUE_PARAMS[H]; \
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
    for (; t < 1500; t++) {
        EVOLVE_STATE(HYPO_CALM);
    }
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
    for (; t < 3000; t++) {
        EVOLVE_STATE(HYPO_CRISIS);
    }
    inject_outlier(data, 2510, 8.0, &rng);
    inject_outlier(data, 2530, 10.0, &rng);
    inject_outlier(data, 2560, 12.0, &rng);
    inject_outlier(data, 2650, 9.0, &rng);
    inject_outlier(data, 2800, 11.0, &rng);
    
    /* Scenario 4: Crisis Persistence (3000-3999) */
    data->scenario_starts[3] = 3000;
    data->scenario_names[3] = "Crisis Persist";
    data->n_scenarios = 4;
    for (; t < 4000; t++) {
        EVOLVE_STATE(HYPO_CRISIS);
    }
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

static void free_synthetic_data(SyntheticData* data) {
    if (!data) return;
    free(data->returns);
    free(data->true_log_vol);
    free(data->true_vol);
    free(data->true_hypothesis);
    free(data->is_outlier);
    free(data);
}

/*─────────────────────────────────────────────────────────────────────────────
 * HAWKES TIMING ANALYSIS STRUCTURES
 *───────────────────────────────────────────────────────────────────────────*/

typedef struct {
    int tick;
    int true_hypothesis;
    int prev_hypothesis;
    float intensity;
    float surprise;
    HawkesTriggerState state;
    bool triggered;
    bool is_transition;  /* True regime changed this tick */
    bool is_outlier;
} HawkesTickRecord;

typedef struct {
    /* Trigger stats */
    int total_triggers;
    int triggers_in_calm;
    int triggers_in_trend;
    int triggers_in_crisis;
    
    /* Detection timing */
    int crisis_entries;         /* Number of times we entered crisis */
    int crisis_detected;        /* Number of times Hawkes was active during entry */
    int sum_detection_lag;      /* Sum of ticks from crisis start to Hawkes trigger */
    
    /* False positives */
    int false_triggers_calm;    /* Triggers during extended calm (not near transitions) */
    
    /* State coverage */
    int ticks_idle;
    int ticks_armed;
    int ticks_fired;
    int ticks_refractory;
    
    /* Per-scenario */
    int triggers_per_scenario[10];
    int ticks_active_per_scenario[10];  /* Ticks in non-IDLE state */
} HawkesStats;

/* Forward declarations */
static void compute_hawkes_stats(HawkesTickRecord* records, SyntheticData* data, HawkesStats* stats);
static void print_hawkes_stats(HawkesStats* stats, SyntheticData* data);
static void print_trigger_events(HawkesTickRecord* records, SyntheticData* data);
static void print_transitions_with_hawkes(HawkesTickRecord* records, SyntheticData* data);

/*─────────────────────────────────────────────────────────────────────────────
 * HAWKES MLE CALIBRATION
 *───────────────────────────────────────────────────────────────────────────*/

typedef struct {
    double* timestamps;
    float* returns;
    int n_events;
    float threshold_used;
} ExtractedEvents;

/**
 * Extract events from return series
 * An "event" is when |return| > threshold
 */
static ExtractedEvents extract_events(SyntheticData* data, float threshold) {
    ExtractedEvents events;
    events.threshold_used = threshold;
    
    /* First pass: count events */
    int count = 0;
    for (int t = 0; t < data->n_ticks; t++) {
        if (fabsf((float)data->returns[t]) > threshold) {
            count++;
        }
    }
    
    events.n_events = count;
    events.timestamps = (double*)malloc(count * sizeof(double));
    events.returns = (float*)malloc(count * sizeof(float));
    
    /* Second pass: extract */
    int idx = 0;
    for (int t = 0; t < data->n_ticks; t++) {
        float ret = (float)data->returns[t];
        if (fabsf(ret) > threshold) {
            events.timestamps[idx] = (double)t;
            events.returns[idx] = ret;
            idx++;
        }
    }
    
    return events;
}

static void free_events(ExtractedEvents* events) {
    free(events->timestamps);
    free(events->returns);
    events->n_events = 0;
}

/**
 * Run MLE calibration and print results
 */
static HawkesMLE calibrate_hawkes_mle(SyntheticData* data) {
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════════════╗\n");
    printf("║                    HAWKES MLE CALIBRATION                             ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════════╝\n");
    
    /* Try different thresholds to find good event count */
    float thresholds[] = {0.01f, 0.02f, 0.03f, 0.05f};
    int n_thresholds = sizeof(thresholds) / sizeof(thresholds[0]);
    
    printf("\n  EVENT EXTRACTION (trying different thresholds)\n");
    printf("  ────────────────────────────────────────────────────────────────────────\n");
    printf("  %10s  %10s  %12s\n", "Threshold", "Events", "Rate");
    
    ExtractedEvents best_events = {0};
    float best_threshold = 0.02f;
    
    for (int i = 0; i < n_thresholds; i++) {
        ExtractedEvents events = extract_events(data, thresholds[i]);
        float rate = (float)events.n_events / data->n_ticks;
        
        printf("  %10.3f  %10d  %11.3f%%\n", 
               thresholds[i], events.n_events, 100.0f * rate);
        
        /* Target: 200-2000 events for good MLE fit */
        if (events.n_events >= 200 && events.n_events <= 2000) {
            if (best_events.n_events == 0 || events.n_events > best_events.n_events) {
                if (best_events.n_events > 0) {
                    free_events(&best_events);
                }
                best_events = events;
                best_threshold = thresholds[i];
            } else {
                free_events(&events);
            }
        } else {
            free_events(&events);
        }
    }
    
    /* If no good threshold found, use 0.02 */
    if (best_events.n_events == 0) {
        best_events = extract_events(data, 0.02f);
        best_threshold = 0.02f;
    }
    
    printf("\n  Selected threshold: %.3f (%d events)\n", 
           best_threshold, best_events.n_events);
    
    /* Event distribution by regime */
    printf("\n  EVENT DISTRIBUTION BY TRUE REGIME\n");
    printf("  ────────────────────────────────────────────────────────────────────────\n");
    
    int events_per_regime[N_HYPOTHESES] = {0};
    for (int i = 0; i < best_events.n_events; i++) {
        int t = (int)best_events.timestamps[i];
        if (t >= 0 && t < data->n_ticks) {
            events_per_regime[data->true_hypothesis[t]]++;
        }
    }
    
    for (int h = 0; h < N_HYPOTHESES; h++) {
        printf("    %-8s: %4d events (%.1f%%)\n", 
               hypothesis_names[h], 
               events_per_regime[h],
               100.0f * events_per_regime[h] / best_events.n_events);
    }
    
    /* Run MLE */
    printf("\n  RUNNING MLE FIT...\n");
    printf("  ────────────────────────────────────────────────────────────────────────\n");
    
    HawkesMLEConfig mle_cfg = hawkes_mle_default_config();
    HawkesMLE mle = hawkes_mle_fit(best_events.timestamps, best_events.n_events, &mle_cfg);
    
    hawkes_mle_print(&mle);
    
    /* Estimate gamma (volatility coupling) */
    float gamma = hawkes_mle_estimate_gamma(best_events.timestamps, best_events.returns,
                                            best_events.n_events,
                                            mle.alpha, mle.beta, mle.lambda0);
    printf("  Estimated γ (vol coupling): %.4f\n\n", gamma);
    
    /* Validate fit */
    printf("  FIT QUALITY ASSESSMENT\n");
    printf("  ────────────────────────────────────────────────────────────────────────\n");
    
    if (!mle.converged) {
        printf("    ⚠ WARNING: MLE did not converge!\n");
    }
    
    if (mle.branching_ratio > 0.9f) {
        printf("    ⚠ WARNING: Branching ratio %.2f is near critical (unstable)\n", 
               mle.branching_ratio);
    } else if (mle.branching_ratio > 0.7f) {
        printf("    ✓ Branching ratio %.2f indicates moderate clustering\n",
               mle.branching_ratio);
    } else {
        printf("    ✓ Branching ratio %.2f indicates mild clustering\n",
               mle.branching_ratio);
    }
    
    float half_life = 0.693147f / mle.beta;
    printf("    ✓ Half-life: %.1f ticks (decay time)\n", half_life);
    printf("    ✓ Mean cluster size: %.1f events\n", mle.mean_cluster_size);
    
    free_events(&best_events);
    
    return mle;
}

/**
 * Create Hawkes integrator config from MLE results
 */
static HawkesIntegratorConfig config_from_mle(const HawkesMLE* mle, float event_threshold) {
    HawkesIntegratorConfig cfg = hawkes_integrator_config_defaults();
    
    /* Use MLE-learned parameters */
    cfg.hawkes.alpha = mle->alpha;
    cfg.hawkes.beta = mle->beta;
    cfg.hawkes.mu = mle->lambda0;
    cfg.hawkes.event_threshold = event_threshold;
    
    /* Adjust integration window based on half-life */
    float half_life = 0.693147f / mle->beta;
    cfg.window_size = (int)(half_life * 4);  /* ~4 half-lives */
    if (cfg.window_size < 16) cfg.window_size = 16;
    if (cfg.window_size > 256) cfg.window_size = 256;
    
    /* Adjust thresholds based on expected clustering */
    if (mle->branching_ratio > 0.7f) {
        /* High clustering: be more sensitive */
        cfg.high_water_mark = 1.0f;
        cfg.low_water_mark = 0.5f;
        cfg.min_ticks_armed = 2;
    } else if (mle->branching_ratio > 0.4f) {
        /* Moderate clustering */
        cfg.high_water_mark = 1.3f;
        cfg.low_water_mark = 0.7f;
        cfg.min_ticks_armed = 3;
    } else {
        /* Low clustering: less sensitive */
        cfg.high_water_mark = 1.5f;
        cfg.low_water_mark = 0.8f;
        cfg.min_ticks_armed = 4;
    }
    
    /* Refractory based on mean cluster duration */
    cfg.refractory_ticks = (int)(half_life * mle->mean_cluster_size);
    if (cfg.refractory_ticks < 100) cfg.refractory_ticks = 100;
    if (cfg.refractory_ticks > 1000) cfg.refractory_ticks = 1000;
    
    return cfg;
}

/*─────────────────────────────────────────────────────────────────────────────
 * RUN HAWKES TEST WITH CONFIG
 *───────────────────────────────────────────────────────────────────────────*/

static void run_hawkes_test(const char* test_name,
                            HawkesIntegratorConfig* cfg,
                            SyntheticData* data,
                            HawkesTickRecord* records,
                            bool print_details) {
    printf("\n═══════════════════════════════════════════════════════════════════════════════\n");
    printf("%s\n", test_name);
    printf("═══════════════════════════════════════════════════════════════════════════════\n");
    
    HawkesIntegrator hawkes;
    hawkes_integrator_init(&hawkes, cfg);
    
    if (print_details) {
        hawkes_integrator_print_config(cfg);
    }
    
    /* Run through data */
    for (int t = 0; t < data->n_ticks; t++) {
        float obs = (float)data->returns[t];
        
        HawkesIntegratorResult result = hawkes_integrator_update(&hawkes, (float)t, obs);
        
        records[t].tick = t;
        records[t].true_hypothesis = data->true_hypothesis[t];
        records[t].prev_hypothesis = (t > 0) ? data->true_hypothesis[t - 1] : data->true_hypothesis[0];
        records[t].intensity = result.integrated_intensity;
        records[t].surprise = result.surprise_sigma;
        records[t].state = result.state;
        records[t].triggered = result.should_trigger;
        records[t].is_transition = (t > 0 && data->true_hypothesis[t] != data->true_hypothesis[t - 1]);
        records[t].is_outlier = data->is_outlier[t];
    }
    
    /* Compute and print stats */
    HawkesStats stats;
    compute_hawkes_stats(records, data, &stats);
    print_hawkes_stats(&stats, data);
    
    if (print_details) {
        print_trigger_events(records, data);
        print_transitions_with_hawkes(records, data);
        hawkes_integrator_print_state(&hawkes);
    }
}

/*─────────────────────────────────────────────────────────────────────────────
 * COMPUTE AND PRINT STATS
 *───────────────────────────────────────────────────────────────────────────*/

static void compute_hawkes_stats(HawkesTickRecord* records, SyntheticData* data,
                                  HawkesStats* stats) {
    memset(stats, 0, sizeof(*stats));
    
    int n = data->n_ticks;
    
    /* Track crisis entries for detection lag */
    bool in_crisis = false;
    int crisis_start_tick = -1;
    bool crisis_detected_this_entry = false;
    
    for (int t = 0; t < n; t++) {
        HawkesTickRecord* r = &records[t];
        
        /* State coverage */
        switch (r->state) {
            case HAWKES_TRIG_IDLE: stats->ticks_idle++; break;
            case HAWKES_TRIG_ARMED: stats->ticks_armed++; break;
            case HAWKES_TRIG_FIRED: stats->ticks_fired++; break;
            case HAWKES_TRIG_REFRACTORY: stats->ticks_refractory++; break;
        }
        
        /* Track triggers by regime */
        if (r->triggered) {
            stats->total_triggers++;
            
            switch (r->true_hypothesis) {
                case HYPO_CALM: stats->triggers_in_calm++; break;
                case HYPO_TREND: stats->triggers_in_trend++; break;
                case HYPO_CRISIS: stats->triggers_in_crisis++; break;
            }
            
            /* Check if this is a false positive in calm
             * (not within 50 ticks of a transition or outlier) */
            if (r->true_hypothesis == HYPO_CALM) {
                bool near_transition = false;
                for (int dt = -50; dt <= 50; dt++) {
                    int check_t = t + dt;
                    if (check_t >= 0 && check_t < n) {
                        if (records[check_t].is_transition || 
                            data->is_outlier[check_t]) {
                            near_transition = true;
                            break;
                        }
                    }
                }
                if (!near_transition) {
                    stats->false_triggers_calm++;
                }
            }
        }
        
        /* Track crisis detection */
        if (r->true_hypothesis == HYPO_CRISIS && !in_crisis) {
            /* Just entered crisis */
            in_crisis = true;
            crisis_start_tick = t;
            crisis_detected_this_entry = false;
            stats->crisis_entries++;
        }
        
        if (in_crisis && !crisis_detected_this_entry) {
            /* Check if Hawkes is active (ARMED, FIRED, or REFRACTORY) */
            if (r->state != HAWKES_TRIG_IDLE) {
                crisis_detected_this_entry = true;
                stats->crisis_detected++;
                stats->sum_detection_lag += (t - crisis_start_tick);
            }
        }
        
        if (r->true_hypothesis != HYPO_CRISIS && in_crisis) {
            /* Exited crisis */
            in_crisis = false;
        }
        
        /* Per-scenario stats */
        for (int s = 0; s < data->n_scenarios; s++) {
            int start = data->scenario_starts[s];
            int end = (s + 1 < data->n_scenarios) ? data->scenario_starts[s + 1] : n;
            
            if (t >= start && t < end) {
                if (r->triggered) {
                    stats->triggers_per_scenario[s]++;
                }
                if (r->state != HAWKES_TRIG_IDLE) {
                    stats->ticks_active_per_scenario[s]++;
                }
                break;
            }
        }
    }
}

static void print_hawkes_stats(HawkesStats* stats, SyntheticData* data) {
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════════════╗\n");
    printf("║                    HAWKES TIMING ANALYSIS                             ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════════╝\n");
    
    printf("\n  TRIGGER DISTRIBUTION BY TRUE REGIME\n");
    printf("  ────────────────────────────────────────────────────────────────────────\n");
    printf("    Total triggers:       %d\n", stats->total_triggers);
    printf("    In CALM:              %d\n", stats->triggers_in_calm);
    printf("    In TREND:             %d\n", stats->triggers_in_trend);
    printf("    In CRISIS:            %d\n", stats->triggers_in_crisis);
    printf("    False positives:      %d (calm, not near transition/outlier)\n", 
           stats->false_triggers_calm);
    
    printf("\n  CRISIS DETECTION\n");
    printf("  ────────────────────────────────────────────────────────────────────────\n");
    printf("    Crisis entries:       %d\n", stats->crisis_entries);
    printf("    Detected by Hawkes:   %d (%.1f%%)\n", 
           stats->crisis_detected,
           stats->crisis_entries > 0 ? 100.0 * stats->crisis_detected / stats->crisis_entries : 0.0);
    printf("    Avg detection lag:    %.1f ticks\n",
           stats->crisis_detected > 0 ? (float)stats->sum_detection_lag / stats->crisis_detected : 0.0);
    
    printf("\n  STATE COVERAGE\n");
    printf("  ────────────────────────────────────────────────────────────────────────\n");
    printf("    IDLE:                 %d ticks (%.1f%%)\n", 
           stats->ticks_idle, 100.0 * stats->ticks_idle / data->n_ticks);
    printf("    ARMED:                %d ticks (%.1f%%)\n",
           stats->ticks_armed, 100.0 * stats->ticks_armed / data->n_ticks);
    printf("    FIRED:                %d ticks (%.1f%%)\n",
           stats->ticks_fired, 100.0 * stats->ticks_fired / data->n_ticks);
    printf("    REFRACTORY:           %d ticks (%.1f%%)\n",
           stats->ticks_refractory, 100.0 * stats->ticks_refractory / data->n_ticks);
    
    printf("\n  PER-SCENARIO ANALYSIS\n");
    printf("  ────────────────────────────────────────────────────────────────────────\n");
    printf("    %-20s %10s %10s %12s\n", "Scenario", "Triggers", "Active", "Active %");
    
    for (int s = 0; s < data->n_scenarios; s++) {
        int start = data->scenario_starts[s];
        int end = (s + 1 < data->n_scenarios) ? data->scenario_starts[s + 1] : data->n_ticks;
        int duration = end - start;
        
        printf("    %-20s %10d %10d %11.1f%%\n",
               data->scenario_names[s],
               stats->triggers_per_scenario[s],
               stats->ticks_active_per_scenario[s],
               100.0 * stats->ticks_active_per_scenario[s] / duration);
    }
    
    printf("\n");
}

/*─────────────────────────────────────────────────────────────────────────────
 * PRINT TRIGGER EVENTS
 *───────────────────────────────────────────────────────────────────────────*/

static void print_trigger_events(HawkesTickRecord* records, SyntheticData* data) {
    printf("\n  TRIGGER EVENTS (Detailed)\n");
    printf("  ────────────────────────────────────────────────────────────────────────\n");
    printf("  %6s  %8s  %10s  %10s  %8s  %s\n", 
           "Tick", "Regime", "Intensity", "Surprise", "Outlier", "Scenario");
    
    for (int t = 0; t < data->n_ticks; t++) {
        if (records[t].triggered) {
            /* Find scenario */
            const char* scenario = "Unknown";
            for (int s = 0; s < data->n_scenarios; s++) {
                int start = data->scenario_starts[s];
                int end = (s + 1 < data->n_scenarios) ? data->scenario_starts[s + 1] : data->n_ticks;
                if (t >= start && t < end) {
                    scenario = data->scenario_names[s];
                    break;
                }
            }
            
            printf("  %6d  %8s  %10.4f  %10.2f  %8s  %s\n",
                   t,
                   hypothesis_names[records[t].true_hypothesis],
                   records[t].intensity,
                   records[t].surprise,
                   records[t].is_outlier ? "YES" : "no",
                   scenario);
        }
    }
    printf("\n");
}

/*─────────────────────────────────────────────────────────────────────────────
 * PRINT REGIME TRANSITIONS WITH HAWKES STATE
 *───────────────────────────────────────────────────────────────────────────*/

static const char* state_name(HawkesTriggerState s) {
    switch (s) {
        case HAWKES_TRIG_IDLE: return "IDLE";
        case HAWKES_TRIG_ARMED: return "ARMED";
        case HAWKES_TRIG_FIRED: return "FIRED";
        case HAWKES_TRIG_REFRACTORY: return "REFRACT";
        default: return "???";
    }
}

static void print_transitions_with_hawkes(HawkesTickRecord* records, SyntheticData* data) {
    printf("\n  REGIME TRANSITIONS vs HAWKES STATE\n");
    printf("  ────────────────────────────────────────────────────────────────────────\n");
    printf("  %6s  %8s → %-8s  %10s  %10s  %10s\n", 
           "Tick", "From", "To", "Hawkes", "Intensity", "Surprise");
    
    for (int t = 1; t < data->n_ticks; t++) {
        if (records[t].is_transition) {
            printf("  %6d  %8s → %-8s  %10s  %10.4f  %10.2f\n",
                   t,
                   hypothesis_names[records[t].prev_hypothesis],
                   hypothesis_names[records[t].true_hypothesis],
                   state_name(records[t].state),
                   records[t].intensity,
                   records[t].surprise);
        }
    }
    printf("\n");
}

/*─────────────────────────────────────────────────────────────────────────────
 * MAIN
 *───────────────────────────────────────────────────────────────────────────*/

int main(int argc, char** argv) {
    int seed = 42;
    if (argc > 1) seed = atoi(argv[1]);
    
    printf("╔═══════════════════════════════════════════════════════════════════════╗\n");
    printf("║                    HAWKES INTEGRATOR TIMING TEST                      ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════════╝\n");
    printf("  Seed: %d\n\n", seed);
    
    /* Generate data */
    printf("Generating synthetic data...\n");
    SyntheticData* data = generate_test_data(seed);
    printf("  Ticks: %d\n", data->n_ticks);
    printf("  Scenarios: %d\n", data->n_scenarios);
    printf("  Outliers: %d\n\n", data->n_outliers_injected);
    
    /* Print scenario timeline */
    printf("  SCENARIO TIMELINE\n");
    printf("  ────────────────────────────────────────────────────────────────────────\n");
    for (int s = 0; s < data->n_scenarios; s++) {
        int start = data->scenario_starts[s];
        int end = (s + 1 < data->n_scenarios) ? data->scenario_starts[s + 1] : data->n_ticks;
        
        /* Count crisis ticks in this scenario */
        int crisis_ticks = 0;
        for (int t = start; t < end; t++) {
            if (data->true_hypothesis[t] == HYPO_CRISIS) crisis_ticks++;
        }
        
        printf("    [%4d - %4d] %-16s", start, end - 1, data->scenario_names[s]);
        if (crisis_ticks > 0) {
            printf(" (CRISIS: %d ticks)", crisis_ticks);
        }
        printf("\n");
    }
    printf("\n");
    
    /* Allocate records */
    HawkesTickRecord* records = (HawkesTickRecord*)calloc(data->n_ticks, sizeof(HawkesTickRecord));
    
    /* ═══════════════════════════════════════════════════════════════════════
     * STEP 1: MLE CALIBRATION
     * ═══════════════════════════════════════════════════════════════════════ */
    HawkesMLE mle = calibrate_hawkes_mle(data);
    
    /* ═══════════════════════════════════════════════════════════════════════
     * STEP 2: TEST WITH MLE-CALIBRATED CONFIG
     * ═══════════════════════════════════════════════════════════════════════ */
    HawkesIntegratorConfig cfg_mle = config_from_mle(&mle, 0.02f);
    run_hawkes_test("TEST 1: MLE-CALIBRATED Hawkes Configuration", 
                    &cfg_mle, data, records, true);
    
    /* ═══════════════════════════════════════════════════════════════════════
     * STEP 3: TEST WITH DEFAULT CONFIG (for comparison)
     * ═══════════════════════════════════════════════════════════════════════ */
    HawkesIntegratorConfig cfg_default = hawkes_integrator_config_defaults();
    run_hawkes_test("TEST 2: DEFAULT Hawkes Configuration",
                    &cfg_default, data, records, true);
    
    /* ═══════════════════════════════════════════════════════════════════════
     * STEP 4: TEST WITH RESPONSIVE CONFIG
     * ═══════════════════════════════════════════════════════════════════════ */
    HawkesIntegratorConfig cfg_responsive = hawkes_integrator_config_responsive();
    run_hawkes_test("TEST 3: RESPONSIVE Hawkes Configuration",
                    &cfg_responsive, data, records, false);
    
    /* ═══════════════════════════════════════════════════════════════════════
     * STEP 5: TEST WITH CONSERVATIVE CONFIG
     * ═══════════════════════════════════════════════════════════════════════ */
    HawkesIntegratorConfig cfg_conservative = hawkes_integrator_config_conservative();
    run_hawkes_test("TEST 4: CONSERVATIVE Hawkes Configuration",
                    &cfg_conservative, data, records, false);
    
    /* ═══════════════════════════════════════════════════════════════════════
     * COMPARISON SUMMARY
     * ═══════════════════════════════════════════════════════════════════════ */
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════════════╗\n");
    printf("║                         COMPARISON SUMMARY                            ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════════╝\n");
    printf("\n");
    
    /* Run all configs and collect stats */
    HawkesIntegratorConfig configs[] = {cfg_mle, cfg_default, cfg_responsive, cfg_conservative};
    const char* config_names[] = {"MLE-Calibrated", "Default", "Responsive", "Conservative"};
    int n_configs = 4;
    
    printf("  %-16s %8s %8s %8s %12s %10s\n", 
           "Config", "Triggers", "Crisis%", "Calm FP", "Detect Lag", "Detect %");
    printf("  ────────────────────────────────────────────────────────────────────────\n");
    
    for (int c = 0; c < n_configs; c++) {
        HawkesIntegrator hawkes;
        hawkes_integrator_init(&hawkes, &configs[c]);
        
        for (int t = 0; t < data->n_ticks; t++) {
            float obs = (float)data->returns[t];
            HawkesIntegratorResult result = hawkes_integrator_update(&hawkes, (float)t, obs);
            
            records[t].tick = t;
            records[t].true_hypothesis = data->true_hypothesis[t];
            records[t].prev_hypothesis = (t > 0) ? data->true_hypothesis[t - 1] : data->true_hypothesis[0];
            records[t].intensity = result.integrated_intensity;
            records[t].surprise = result.surprise_sigma;
            records[t].state = result.state;
            records[t].triggered = result.should_trigger;
            records[t].is_transition = (t > 0 && data->true_hypothesis[t] != data->true_hypothesis[t - 1]);
            records[t].is_outlier = data->is_outlier[t];
        }
        
        HawkesStats stats;
        compute_hawkes_stats(records, data, &stats);
        
        float crisis_pct = (stats.total_triggers > 0) ? 
            100.0f * stats.triggers_in_crisis / stats.total_triggers : 0.0f;
        float detect_pct = (stats.crisis_entries > 0) ?
            100.0f * stats.crisis_detected / stats.crisis_entries : 0.0f;
        float avg_lag = (stats.crisis_detected > 0) ?
            (float)stats.sum_detection_lag / stats.crisis_detected : 0.0f;
        
        printf("  %-16s %8d %7.1f%% %8d %11.1f %9.1f%%\n",
               config_names[c],
               stats.total_triggers,
               crisis_pct,
               stats.false_triggers_calm,
               avg_lag,
               detect_pct);
    }
    
    printf("\n");
    
    /* ═══════════════════════════════════════════════════════════════════════
     * EVALUATION CRITERIA
     * ═══════════════════════════════════════════════════════════════════════ */
    printf("╔═══════════════════════════════════════════════════════════════════════╗\n");
    printf("║                         EVALUATION CRITERIA                           ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════════╝\n");
    printf("\n");
    printf("  GOOD Hawkes behavior:\n");
    printf("  ┌────────────────────────────────────────────────────────────────────┐\n");
    printf("  │ ✓ Crisis%%: >50%% of triggers should be in CRISIS regime            │\n");
    printf("  │ ✓ Calm FP: <5 false positives in Extended Calm                     │\n");
    printf("  │ ✓ Detect Lag: <20 ticks from crisis start                          │\n");
    printf("  │ ✓ Detect %%: >80%% of crisis entries detected                        │\n");
    printf("  └────────────────────────────────────────────────────────────────────┘\n");
    printf("\n");
    printf("  For PGAS Veto Integration:\n");
    printf("  ┌────────────────────────────────────────────────────────────────────┐\n");
    printf("  │ Use: bool crisis = (hawkes_state != HAWKES_TRIG_IDLE)              │\n");
    printf("  │ This vetoes PGAS injection during ARMED, FIRED, and REFRACTORY    │\n");
    printf("  │ states, protecting RBPF from stale PGAS data during crises.       │\n");
    printf("  └────────────────────────────────────────────────────────────────────┘\n");
    printf("\n");
    
    /* Cleanup */
    free(records);
    free_synthetic_data(data);
    
    return 0;
}
