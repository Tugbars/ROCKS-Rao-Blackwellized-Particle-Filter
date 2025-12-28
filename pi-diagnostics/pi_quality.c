/**
 * @file pi_quality.c
 * @brief Direct Π Quality Metrics Implementation
 */

#include "pi_quality.h"
#include <math.h>
#include <string.h>
#include <float.h>

/*═══════════════════════════════════════════════════════════════════════════
 * CONSTANTS
 *═══════════════════════════════════════════════════════════════════════════*/

#define EMA_ALPHA_FAST  0.1
#define EMA_ALPHA_SLOW  0.02
#define MIN_PROB        1e-10
#define KL_CAP          10.0   /* Cap KL when q ≈ 0 */

/*═══════════════════════════════════════════════════════════════════════════
 * INITIALIZATION
 *═══════════════════════════════════════════════════════════════════════════*/

void pi_quality_init(PiQualityState *state) {
    if (!state) return;
    
    memset(state, 0, sizeof(*state));
    
    /* Prediction tracker */
    state->prediction.baseline_rmse = 1.0;  /* Default until learned */
    state->prediction.baseline_initialized = false;
    
    /* Transition tracker */
    state->transition.drop_threshold_sigma = 2.0;
    state->transition.lag_threshold = 5;
    state->transition.last_regime = -1;
    
    /* Likelihood tracker */
    state->likelihood.sample_count = 0;
    
    /* Quality score starts at 1.0 (assume good until proven otherwise) */
    state->quality_score = 1.0;
    state->is_initialized = true;
}

void pi_quality_reset(PiQualityState *state) {
    if (!state) return;
    
    /* Keep baseline, reset current tracking */
    double saved_baseline = state->prediction.baseline_rmse;
    bool saved_init = state->prediction.baseline_initialized;
    
    pi_quality_init(state);
    
    state->prediction.baseline_rmse = saved_baseline;
    state->prediction.baseline_initialized = saved_init;
}

/*═══════════════════════════════════════════════════════════════════════════
 * PREDICTION ERROR TRACKING
 *═══════════════════════════════════════════════════════════════════════════*/

void pi_quality_update_prediction(
    PiQualityState *state,
    double predicted,
    double actual)
{
    if (!state) return;
    
    PredictionErrorTracker *p = &state->prediction;
    
    double error = actual - predicted;
    double error_sq = error * error;
    
    /* Update rolling window */
    if (p->buffer_count < PI_QUALITY_RMSE_WINDOW) {
        /* Still filling buffer */
        p->error_buffer[p->buffer_idx] = error_sq;
        p->error_sq_sum += error_sq;
        p->buffer_count++;
    } else {
        /* Replace oldest */
        p->error_sq_sum -= p->error_buffer[p->buffer_idx];
        p->error_buffer[p->buffer_idx] = error_sq;
        p->error_sq_sum += error_sq;
    }
    
    p->buffer_idx = (p->buffer_idx + 1) % PI_QUALITY_RMSE_WINDOW;
    
    /* Compute current RMSE */
    if (p->buffer_count > 0) {
        p->current_rmse = sqrt(p->error_sq_sum / p->buffer_count);
    }
    
    /* Update baseline EMA (slow adaptation) */
    if (!p->baseline_initialized) {
        if (p->buffer_count >= PI_QUALITY_RMSE_WINDOW) {
            p->baseline_rmse = p->current_rmse;
            p->baseline_ema = p->current_rmse;
            p->baseline_initialized = true;
        }
    } else {
        /* Only update baseline when quality is good (not during degradation) */
        if (p->rmse_ratio < 1.3) {
            p->baseline_ema = EMA_ALPHA_SLOW * p->current_rmse + 
                              (1.0 - EMA_ALPHA_SLOW) * p->baseline_ema;
            p->baseline_rmse = p->baseline_ema;
        }
    }
    
    /* Compute ratio */
    if (p->baseline_rmse > MIN_PROB) {
        p->rmse_ratio = p->current_rmse / p->baseline_rmse;
    } else {
        p->rmse_ratio = 1.0;
    }
    
    p->baseline_samples++;
}

/*═══════════════════════════════════════════════════════════════════════════
 * TRANSITION LAG TRACKING
 *═══════════════════════════════════════════════════════════════════════════*/

void pi_quality_update_transition(
    PiQualityState *state,
    double log_lik,
    int map_regime)
{
    if (!state) return;
    
    TransitionLagTracker *t = &state->transition;
    
    /* Update likelihood statistics */
    if (t->ticks_in_regime == 0) {
        t->likelihood_ema = log_lik;
        t->likelihood_std_ema = 0.1;  /* Initial estimate */
    } else {
        /* EMA of likelihood */
        double old_ema = t->likelihood_ema;
        t->likelihood_ema = EMA_ALPHA_FAST * log_lik + 
                            (1.0 - EMA_ALPHA_FAST) * t->likelihood_ema;
        
        /* EMA of squared deviation (for std) */
        double dev = log_lik - old_ema;
        t->likelihood_std_ema = EMA_ALPHA_SLOW * (dev * dev) + 
                                 (1.0 - EMA_ALPHA_SLOW) * t->likelihood_std_ema;
    }
    
    double lik_std = sqrt(t->likelihood_std_ema + MIN_PROB);
    
    /* Detect likelihood drop (regime change in data) */
    double surprise = t->likelihood_ema - log_lik;
    double surprise_sigma = surprise / lik_std;
    
    if (surprise_sigma > t->drop_threshold_sigma && !t->drop_detected) {
        t->drop_detected = true;
        t->ticks_since_drop = 0;
    }
    
    if (t->drop_detected) {
        t->ticks_since_drop++;
    }
    
    /* Detect regime change in RBPF */
    if (t->last_regime >= 0 && map_regime != t->last_regime) {
        /* Regime changed - record lag if we were tracking a drop */
        if (t->drop_detected) {
            int lag = t->ticks_since_drop;
            
            /* Update history buffer */
            t->lag_history[t->lag_history_idx] = lag;
            t->lag_history_idx = (t->lag_history_idx + 1) % PI_QUALITY_LAG_HISTORY;
            if (t->lag_history_count < PI_QUALITY_LAG_HISTORY) {
                t->lag_history_count++;
            }
            
            /* Update statistics */
            if (lag > t->max_lag) t->max_lag = lag;
            if (lag > t->lag_threshold) t->lag_violations++;
            
            /* Update average */
            double sum = 0;
            for (int i = 0; i < t->lag_history_count; i++) {
                sum += t->lag_history[i];
            }
            t->avg_lag = sum / t->lag_history_count;
            
            /* Reset drop detector */
            t->drop_detected = false;
        }
        
        t->ticks_in_regime = 0;
    } else {
        t->ticks_in_regime++;
    }
    
    t->last_regime = map_regime;
    
    /* Timeout: if we've been waiting too long, reset drop detector */
    if (t->drop_detected && t->ticks_since_drop > 20) {
        t->drop_detected = false;  /* False alarm */
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * LIKELIHOOD TRACKING
 *═══════════════════════════════════════════════════════════════════════════*/

void pi_quality_update_likelihood(
    PiQualityState *state,
    double log_lik)
{
    if (!state) return;
    
    LikelihoodTracker *l = &state->likelihood;
    
    l->current_ll = log_lik;
    
    if (l->sample_count == 0) {
        l->log_likelihood_ema = log_lik;
        l->log_likelihood_sq_ema = log_lik * log_lik;
        l->ll_zscore = 0.0;
    } else {
        /* Update EMAs */
        double old_ema = l->log_likelihood_ema;
        l->log_likelihood_ema = EMA_ALPHA_FAST * log_lik + 
                                 (1.0 - EMA_ALPHA_FAST) * l->log_likelihood_ema;
        l->log_likelihood_sq_ema = EMA_ALPHA_FAST * (log_lik * log_lik) + 
                                    (1.0 - EMA_ALPHA_FAST) * l->log_likelihood_sq_ema;
        
        /* Compute std */
        double var = l->log_likelihood_sq_ema - 
                     l->log_likelihood_ema * l->log_likelihood_ema;
        l->log_likelihood_std = sqrt(fmax(var, MIN_PROB));
        
        /* Compute z-score */
        l->ll_zscore = (log_lik - old_ema) / (l->log_likelihood_std + MIN_PROB);
    }
    
    l->sample_count++;
}

/*═══════════════════════════════════════════════════════════════════════════
 * WEIGHT STATISTICS
 *═══════════════════════════════════════════════════════════════════════════*/

void pi_quality_update_weights(
    PiQualityState *state,
    const double *weights,
    int N)
{
    if (!state || !weights || N <= 0) return;
    
    WeightStats *w = &state->weights;
    
    /* Compute statistics */
    double sum = 0.0;
    double sum_sq = 0.0;
    double max_w = 0.0;
    double entropy = 0.0;
    
    for (int i = 0; i < N; i++) {
        double wi = weights[i];
        sum += wi;
        sum_sq += wi * wi;
        if (wi > max_w) max_w = wi;
        if (wi > MIN_PROB) {
            entropy -= wi * log(wi);
        }
    }
    
    /* Normalize if needed */
    if (fabs(sum - 1.0) > 1e-6 && sum > MIN_PROB) {
        sum_sq /= (sum * sum);
        max_w /= sum;
    }
    
    /* Variance of normalized weights */
    double mean_w = 1.0 / N;
    w->variance = sum_sq - mean_w * mean_w;
    w->max_weight = max_w;
    
    /* ESS */
    if (sum_sq > MIN_PROB) {
        w->ess_ratio = 1.0 / (N * sum_sq);
    } else {
        w->ess_ratio = 1.0;
    }
    
    w->entropy = entropy;
}

/*═══════════════════════════════════════════════════════════════════════════
 * SELF-CONSISTENCY (D₃)
 *═══════════════════════════════════════════════════════════════════════════*/

double pi_quality_kl_divergence(
    const float *P,
    const float *Q,
    int K)
{
    if (!P || !Q || K <= 0) return 0.0;
    
    double total_kl = 0.0;
    
    for (int i = 0; i < K; i++) {
        double row_kl = 0.0;
        for (int j = 0; j < K; j++) {
            float p = P[i * K + j];
            float q = Q[i * K + j];
            
            if (p > MIN_PROB && q > MIN_PROB) {
                row_kl += p * log(p / q);
            } else if (p > MIN_PROB) {
                /* p > 0 but q ≈ 0: infinite divergence, cap it */
                row_kl += KL_CAP;
            }
        }
        total_kl += row_kl;
    }
    
    return total_kl / K;  /* Average per row */
}

void pi_quality_normalize_counts(
    const float *Q,
    float *Pi_out,
    int K)
{
    if (!Q || !Pi_out || K <= 0) return;
    
    for (int i = 0; i < K; i++) {
        float row_sum = 0.0f;
        for (int j = 0; j < K; j++) {
            row_sum += Q[i * K + j];
        }
        
        if (row_sum > MIN_PROB) {
            for (int j = 0; j < K; j++) {
                Pi_out[i * K + j] = Q[i * K + j] / row_sum;
            }
        } else {
            /* Uniform if no counts */
            for (int j = 0; j < K; j++) {
                Pi_out[i * K + j] = 1.0f / K;
            }
        }
    }
}

void pi_quality_update_d3(
    PiQualityState *state,
    const float *Pi_storvik,
    const float *Pi_operating,
    int K)
{
    if (!state || !Pi_storvik || !Pi_operating) return;
    
    state->d3_storvik_vs_operating = pi_quality_kl_divergence(
        Pi_storvik, Pi_operating, K);
}

/*═══════════════════════════════════════════════════════════════════════════
 * FULL UPDATE
 *═══════════════════════════════════════════════════════════════════════════*/

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
    int K)
{
    if (!state) return;
    
    pi_quality_update_prediction(state, predicted, actual);
    pi_quality_update_transition(state, log_lik, map_regime);
    pi_quality_update_likelihood(state, log_lik);
    pi_quality_update_weights(state, weights, N);
    pi_quality_update_d3(state, Pi_storvik, Pi_operating, K);
    
    state->quality_score = pi_quality_compute_score(state);
}

/*═══════════════════════════════════════════════════════════════════════════
 * QUALITY SCORE COMPUTATION
 *═══════════════════════════════════════════════════════════════════════════*/

double pi_quality_compute_score(const PiQualityState *state) {
    if (!state) return 0.0;
    
    double score = 1.0;
    
    /* ═══════════════════════════════════════════════════════════════════
     * EARLY SIGNALS (High weight)
     * ═══════════════════════════════════════════════════════════════════*/
    
    /* RMSE penalty */
    double rmse_ratio = state->prediction.rmse_ratio;
    if (rmse_ratio > 3.0) {
        score *= 0.1;   /* Predictions 3x worse */
    } else if (rmse_ratio > 2.0) {
        score *= 0.3;   /* Predictions 2x worse */
    } else if (rmse_ratio > 1.5) {
        score *= 0.6;   /* Predictions 50% worse */
    }
    
    /* Likelihood z-score penalty */
    double ll_z = state->likelihood.ll_zscore;
    if (ll_z < -4.0) {
        score *= 0.2;   /* Extremely unlikely data */
    } else if (ll_z < -3.0) {
        score *= 0.4;   /* Very unlikely data */
    } else if (ll_z < -2.0) {
        score *= 0.7;   /* Unlikely data */
    }
    
    /* D₃ self-contradiction penalty */
    double d3 = state->d3_storvik_vs_operating;
    if (d3 > 0.3) {
        score *= 0.3;   /* Severe self-contradiction */
    } else if (d3 > 0.2) {
        score *= 0.5;   /* Moderate self-contradiction */
    } else if (d3 > 0.1) {
        score *= 0.7;   /* Mild self-contradiction */
    }
    
    /* Transition lag penalty */
    double avg_lag = state->transition.avg_lag;
    if (avg_lag > 8.0) {
        score *= 0.4;   /* Very slow detection */
    } else if (avg_lag > 5.0) {
        score *= 0.6;   /* Slow detection */
    } else if (avg_lag > 3.0) {
        score *= 0.8;   /* Slightly slow */
    }
    
    /* ═══════════════════════════════════════════════════════════════════
     * MEDIUM SIGNALS (Medium weight)
     * ═══════════════════════════════════════════════════════════════════*/
    
    /* Weight concentration penalty */
    double max_w = state->weights.max_weight;
    if (max_w > 0.7) {
        score *= 0.5;   /* One particle dominates */
    } else if (max_w > 0.5) {
        score *= 0.7;   /* High concentration */
    }
    
    /* Weight variance penalty */
    double w_var = state->weights.variance;
    if (w_var > 0.1) {
        score *= 0.8;   /* Particles disagree */
    }
    
    /* ═══════════════════════════════════════════════════════════════════
     * LATE SIGNALS (Lower weight, but still relevant)
     * ═══════════════════════════════════════════════════════════════════*/
    
    double ess = state->weights.ess_ratio;
    if (ess < 0.15) {
        score *= 0.3;   /* Critical ESS */
    } else if (ess < 0.25) {
        score *= 0.5;   /* Low ESS */
    } else if (ess < 0.4) {
        score *= 0.7;   /* Degraded ESS */
    }
    
    return fmax(0.0, fmin(1.0, score));
}

/*═══════════════════════════════════════════════════════════════════════════
 * SNAPSHOT
 *═══════════════════════════════════════════════════════════════════════════*/

PiQualitySnapshot pi_quality_get_snapshot(const PiQualityState *state) {
    PiQualitySnapshot snap = {0};
    
    if (!state) return snap;
    
    /* Early signals */
    snap.rmse_ratio = state->prediction.rmse_ratio;
    snap.likelihood_zscore = state->likelihood.ll_zscore;
    snap.d3_self_contradiction = state->d3_storvik_vs_operating;
    snap.avg_transition_lag = state->transition.avg_lag;
    
    /* Medium signals */
    snap.weight_variance = state->weights.variance;
    snap.max_weight = state->weights.max_weight;
    
    /* Late signals */
    snap.ess_ratio = state->weights.ess_ratio;
    
    /* Composite */
    snap.quality_score = state->quality_score;
    
    return snap;
}

/*═══════════════════════════════════════════════════════════════════════════
 * INJECTION NEED CHECK
 *═══════════════════════════════════════════════════════════════════════════*/

bool pi_quality_needs_injection(const PiQualityState *state) {
    if (!state) return false;
    
    /* Check early signals */
    if (state->prediction.rmse_ratio > 2.0) return true;
    if (state->likelihood.ll_zscore < -3.0) return true;
    if (state->d3_storvik_vs_operating > 0.15) return true;
    if (state->transition.avg_lag > 5.0) return true;
    
    /* Check medium signals */
    if (state->weights.max_weight > 0.6) return true;
    
    /* Check late signals (emergency) */
    if (state->weights.ess_ratio < 0.2) return true;
    
    /* Overall score check */
    if (state->quality_score < 0.4) return true;
    
    return false;
}
