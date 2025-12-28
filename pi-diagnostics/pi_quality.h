/**
 * @file pi_quality.h
 * @brief Direct Π Quality Metrics
 *
 * Monitors transition matrix accuracy through its downstream effects:
 *   - Prediction RMSE (if Π wrong → bad predictions)
 *   - Transition detection lag (if Π sticky → slow regime detection)
 *   - Marginal likelihood (if Π wrong → data looks "impossible")
 *   - Weight variance (if Π wrong → particles disagree)
 *
 * These are EARLY signals. ESS drop is a LATE symptom.
 *
 * The causal chain:
 *   Π wrong → Bad predictions → Weight skew → ESS drops → Collapse
 *   ↑ ROOT CAUSE                                         ↑ TOO LATE
 */

#ifndef PI_QUALITY_H
#define PI_QUALITY_H

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/*═══════════════════════════════════════════════════════════════════════════
 * CONFIGURATION
 *═══════════════════════════════════════════════════════════════════════════*/

#ifndef PI_QUALITY_RMSE_WINDOW
#define PI_QUALITY_RMSE_WINDOW 50  /* Rolling window for RMSE */
#endif

#ifndef PI_QUALITY_LAG_HISTORY
#define PI_QUALITY_LAG_HISTORY 20  /* Recent transitions to track */
#endif

/*═══════════════════════════════════════════════════════════════════════════
 * PREDICTION ERROR TRACKER
 *═══════════════════════════════════════════════════════════════════════════*/

typedef struct {
    /* Rolling RMSE */
    double error_sq_sum;           /* Sum of squared errors */
    double error_buffer[PI_QUALITY_RMSE_WINDOW];
    int    buffer_idx;
    int    buffer_count;
    
    /* Baseline (learned during stable periods) */
    double baseline_rmse;
    double baseline_ema;           /* EMA of RMSE for baseline */
    int    baseline_samples;
    bool   baseline_initialized;
    
    /* Current metrics */
    double current_rmse;
    double rmse_ratio;             /* current / baseline (>1 = degraded) */
    
} PredictionErrorTracker;

/*═══════════════════════════════════════════════════════════════════════════
 * TRANSITION LAG TRACKER
 *
 * Measures how quickly RBPF detects regime changes.
 * If Π diagonal is too sticky, transitions are detected LATE.
 *═══════════════════════════════════════════════════════════════════════════*/

typedef struct {
    /* Likelihood drop detection (signals regime change in data) */
    double likelihood_ema;
    double likelihood_std_ema;
    double drop_threshold_sigma;   /* How many σ for "drop" */
    int    ticks_since_drop;
    bool   drop_detected;
    
    /* Regime tracking */
    int    last_regime;
    int    ticks_in_regime;
    
    /* Lag statistics */
    int    lag_history[PI_QUALITY_LAG_HISTORY];
    int    lag_history_idx;
    int    lag_history_count;
    
    double avg_lag;
    int    max_lag;
    int    lag_violations;         /* Times lag > threshold */
    int    lag_threshold;          /* What's "too slow" (default: 5 ticks) */
    
} TransitionLagTracker;

/*═══════════════════════════════════════════════════════════════════════════
 * LIKELIHOOD TRACKER
 *═══════════════════════════════════════════════════════════════════════════*/

typedef struct {
    /* Running statistics */
    double log_likelihood_ema;
    double log_likelihood_sq_ema;
    double log_likelihood_std;
    int    sample_count;
    
    /* Current */
    double current_ll;
    double ll_zscore;              /* How unusual (< -2 = very unlikely) */
    
} LikelihoodTracker;

/*═══════════════════════════════════════════════════════════════════════════
 * WEIGHT DISTRIBUTION TRACKER
 *═══════════════════════════════════════════════════════════════════════════*/

typedef struct {
    double variance;               /* Var(w) - high = particles disagree */
    double max_weight;             /* Particle monopoly check */
    double ess_ratio;              /* ESS / N */
    double entropy;                /* -sum(w * log(w)) */
    
} WeightStats;

/*═══════════════════════════════════════════════════════════════════════════
 * COMPLETE Π QUALITY STATE
 *═══════════════════════════════════════════════════════════════════════════*/

typedef struct {
    PredictionErrorTracker prediction;
    TransitionLagTracker   transition;
    LikelihoodTracker      likelihood;
    WeightStats            weights;
    
    /* Self-consistency (D₃) */
    double d3_storvik_vs_operating;
    
    /* Composite score: 0 = garbage, 1 = perfect */
    double quality_score;
    
    /* Flags */
    bool is_initialized;
    
} PiQualityState;

/*═══════════════════════════════════════════════════════════════════════════
 * SNAPSHOT (For injection decision)
 *═══════════════════════════════════════════════════════════════════════════*/

typedef struct {
    /* Early signals (act immediately) */
    double rmse_ratio;             /* >1.5 = problem */
    double likelihood_zscore;      /* <-2 = problem */
    double d3_self_contradiction;  /* >0.1 = problem */
    double avg_transition_lag;     /* >3 = problem */
    
    /* Medium signals (prepare for action) */
    double weight_variance;
    double max_weight;
    
    /* Late signals (damage done) */
    double ess_ratio;
    
    /* Composite */
    double quality_score;
    
} PiQualitySnapshot;

/*═══════════════════════════════════════════════════════════════════════════
 * API - INITIALIZATION
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Initialize quality state
 */
void pi_quality_init(PiQualityState *state);

/**
 * Reset (after injection)
 */
void pi_quality_reset(PiQualityState *state);

/*═══════════════════════════════════════════════════════════════════════════
 * API - UPDATE (Called every tick by RBPF)
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Update prediction error tracker
 * 
 * @param state       Quality state
 * @param predicted   E[y_t | y_{1:t-1}]
 * @param actual      y_t
 */
void pi_quality_update_prediction(
    PiQualityState *state,
    double predicted,
    double actual);

/**
 * Update transition lag tracker
 * 
 * @param state       Quality state
 * @param log_lik     Log marginal likelihood P(y_t | y_{1:t-1})
 * @param map_regime  Current MAP regime estimate
 */
void pi_quality_update_transition(
    PiQualityState *state,
    double log_lik,
    int map_regime);

/**
 * Update likelihood tracker
 */
void pi_quality_update_likelihood(
    PiQualityState *state,
    double log_lik);

/**
 * Update weight statistics
 * 
 * @param state    Quality state
 * @param weights  Particle weights (normalized)
 * @param N        Number of particles
 */
void pi_quality_update_weights(
    PiQualityState *state,
    const double *weights,
    int N);

/**
 * Update self-consistency (D₃)
 * 
 * @param state         Quality state
 * @param Pi_storvik    Π derived from Storvik counts (row-stochastic)
 * @param Pi_operating  Currently used Π (row-stochastic)
 * @param K             Number of regimes
 */
void pi_quality_update_d3(
    PiQualityState *state,
    const float *Pi_storvik,
    const float *Pi_operating,
    int K);

/**
 * Full update (convenience function)
 */
void pi_quality_update(
    PiQualityState *state,
    double predicted,
    double actual,
    double log_lik,
    int map_regime,
    const double *weights,
    int N,
    const float *Pi_storvik,
    const float *Pi_operating,
    int K);

/*═══════════════════════════════════════════════════════════════════════════
 * API - QUERY
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Get quality snapshot for injection decision
 */
PiQualitySnapshot pi_quality_get_snapshot(const PiQualityState *state);

/**
 * Compute composite quality score
 */
double pi_quality_compute_score(const PiQualityState *state);

/**
 * Check if injection is urgently needed (based on early signals)
 */
bool pi_quality_needs_injection(const PiQualityState *state);

/*═══════════════════════════════════════════════════════════════════════════
 * API - UTILITIES
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Compute KL divergence between two row-stochastic matrices
 * Returns average KL per row
 */
double pi_quality_kl_divergence(
    const float *P,
    const float *Q,
    int K);

/**
 * Normalize Storvik counts to get Π
 * Q_ij → Π_ij = Q_ij / sum_j(Q_ij)
 */
void pi_quality_normalize_counts(
    const float *Q,
    float *Pi_out,
    int K);

#ifdef __cplusplus
}
#endif

#endif /* PI_QUALITY_H */
