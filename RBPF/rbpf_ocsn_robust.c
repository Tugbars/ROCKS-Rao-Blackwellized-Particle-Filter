/**
 * @file rbpf_ocsn_robust.c
 * @brief Robust OCSN Implementation - 11th Outlier Component
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * THEORY
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Standard OCSN (Omori et al., 2007) uses 10-component mixture to approximate
 * log(χ²₁) for the observation equation:
 *
 *   y_t = exp(h_t/2) * ε_t,  ε_t ~ N(0,1)
 *   log(y_t²) = h_t + log(ε_t²)
 *             = h_t + ξ_t,   ξ_t ~ log(χ²₁)
 *
 * The 10-component mixture works well for normal returns (|ε| < 4σ), but
 * fat-tail events (8-15σ) cause particle collapse because:
 *   - All particles assign near-zero likelihood
 *   - ESS crashes to single digits
 *   - Resampling produces degenerate ensemble
 *
 * ROBUST OCSN adds an 11th "outlier" component:
 *
 *   P(obs | h, regime) = (1 - π_out) × P_OCSN(obs | h)
 *                      + π_out × N(obs | h, σ²_out)
 *
 * Where:
 *   π_out    = outlier probability (regime-dependent, typically 1-2.5%)
 *   σ²_out   = outlier variance (regime-dependent, typically 18-30)
 *
 * The outlier component is a broad Gaussian that provides "support" during
 * extreme moves, preventing complete particle collapse.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * VARIANCE BOUNDS (Critical for Signal Preservation)
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Max OCSN variance = 7.33 (component 10)
 *
 * If outlier variance too LOW (<15):
 *   - Outlier component competes with OCSN on normal observations
 *   - Distorts Kalman updates, biases state estimates
 *
 * If outlier variance too HIGH (>50):
 *   - Kalman gain K → 0 during outliers
 *   - Filter ignores extreme moves entirely ("signal suppression")
 *   - Defeats purpose of robust handling
 *
 * Sweet spot: 2-4× max OCSN variance ≈ 15-30
 *   - R0 (calm):   var=18, prob=1.0%   (rare outliers, modest variance)
 *   - R1:          var=22, prob=1.5%
 *   - R2:          var=26, prob=2.0%
 *   - R3 (crisis): var=30, prob=2.5%   (more outliers, wider variance)
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * IMPLEMENTATION NOTES
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * 1. Likelihood computation uses log-sum-exp for numerical stability
 * 2. Kalman update blends OCSN and outlier posteriors by mixture weight
 * 3. outlier_fraction output enables monitoring and diagnostics
 * 4. Per-regime parameters allow crisis-adaptive robustness
 *
 * ═══════════════════════════════════════════════════════════════════════════
 */

#include "rbpf_ksc.h"
#include <math.h>
#include <float.h>

/*═══════════════════════════════════════════════════════════════════════════
 * OCSN MIXTURE PARAMETERS (Kim, Shephard, Chib 1998)
 *
 * 10-component mixture approximation to log(χ²₁)
 *═══════════════════════════════════════════════════════════════════════════*/

static const rbpf_real_t OCSN_PROB[10] = {
    0.00609f, 0.04775f, 0.13057f, 0.20674f, 0.22715f,
    0.18842f, 0.12047f, 0.05591f, 0.01575f, 0.00115f
};

static const rbpf_real_t OCSN_MEAN[10] = {
    1.92677f, 1.34744f, 0.73504f, 0.02266f, -0.85173f,
    -1.97278f, -3.46788f, -5.55246f, -8.68384f, -14.65000f
};

static const rbpf_real_t OCSN_VAR[10] = {
    0.11265f, 0.17788f, 0.26768f, 0.40611f, 0.62699f,
    0.98583f, 1.57469f, 2.54498f, 4.16591f, 7.33342f
};

/* Precomputed log(2π) for Gaussian PDF */
#define LOG_2PI 1.8378770664093453f

/*═══════════════════════════════════════════════════════════════════════════
 * HELPER: Log-Sum-Exp for numerical stability
 *═══════════════════════════════════════════════════════════════════════════*/

static inline rbpf_real_t log_sum_exp_2(rbpf_real_t log_a, rbpf_real_t log_b)
{
    if (log_a > log_b)
    {
        return log_a + rbpf_log(1.0f + rbpf_exp(log_b - log_a));
    }
    else
    {
        return log_b + rbpf_log(1.0f + rbpf_exp(log_a - log_b));
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * HELPER: Gaussian log-PDF
 *═══════════════════════════════════════════════════════════════════════════*/

static inline rbpf_real_t gaussian_log_pdf(rbpf_real_t x, rbpf_real_t mean, rbpf_real_t var)
{
    rbpf_real_t diff = x - mean;
    return -0.5f * (LOG_2PI + rbpf_log(var) + diff * diff / var);
}

/*═══════════════════════════════════════════════════════════════════════════
 * COMPUTE OCSN LOG-LIKELIHOOD (standard 10-component)
 *═══════════════════════════════════════════════════════════════════════════*/

static rbpf_real_t compute_ocsn_log_likelihood(rbpf_real_t obs, rbpf_real_t h)
{
    /* obs = log(y²), expected = h under each mixture component */
    rbpf_real_t max_log_lik = -1e30f;
    rbpf_real_t log_liks[10];
    
    /* Compute log-likelihood under each component */
    for (int k = 0; k < 10; k++)
    {
        /* Component k: obs ~ N(h + m_k, v_k) */
        rbpf_real_t mean_k = h + OCSN_MEAN[k];
        rbpf_real_t var_k = OCSN_VAR[k];
        rbpf_real_t log_prob_k = rbpf_log(OCSN_PROB[k]);
        
        log_liks[k] = log_prob_k + gaussian_log_pdf(obs, mean_k, var_k);
        
        if (log_liks[k] > max_log_lik)
            max_log_lik = log_liks[k];
    }
    
    /* Log-sum-exp for numerical stability */
    rbpf_real_t sum = 0.0f;
    for (int k = 0; k < 10; k++)
    {
        sum += rbpf_exp(log_liks[k] - max_log_lik);
    }
    
    return max_log_lik + rbpf_log(sum);
}

/*═══════════════════════════════════════════════════════════════════════════
 * COMPUTE OCSN KALMAN UPDATE (standard - used for blending)
 *
 * Returns posterior mean and variance after OCSN update
 *═══════════════════════════════════════════════════════════════════════════*/

static void compute_ocsn_kalman_update(
    rbpf_real_t obs,
    rbpf_real_t h_prior,
    rbpf_real_t P_prior,
    rbpf_real_t *h_post,
    rbpf_real_t *P_post)
{
    /* Mixture-weighted Kalman update
     * Weight each component by its posterior probability */
    
    rbpf_real_t log_weights[10];
    rbpf_real_t max_log_w = -1e30f;
    
    /* Compute posterior weights for each component */
    for (int k = 0; k < 10; k++)
    {
        rbpf_real_t mean_k = h_prior + OCSN_MEAN[k];
        rbpf_real_t var_k = OCSN_VAR[k];
        rbpf_real_t log_prob_k = rbpf_log(OCSN_PROB[k]);
        
        log_weights[k] = log_prob_k + gaussian_log_pdf(obs, mean_k, var_k);
        
        if (log_weights[k] > max_log_w)
            max_log_w = log_weights[k];
    }
    
    /* Normalize weights */
    rbpf_real_t weights[10];
    rbpf_real_t sum_w = 0.0f;
    for (int k = 0; k < 10; k++)
    {
        weights[k] = rbpf_exp(log_weights[k] - max_log_w);
        sum_w += weights[k];
    }
    for (int k = 0; k < 10; k++)
    {
        weights[k] /= sum_w;
    }
    
    /* Compute weighted posterior mean and variance */
    rbpf_real_t h_sum = 0.0f;
    rbpf_real_t P_sum = 0.0f;
    
    for (int k = 0; k < 10; k++)
    {
        rbpf_real_t var_k = OCSN_VAR[k];
        
        /* Kalman gain for component k */
        rbpf_real_t S_k = P_prior + var_k;  /* Innovation variance */
        rbpf_real_t K_k = P_prior / S_k;    /* Kalman gain */
        
        /* Innovation */
        rbpf_real_t innovation = obs - (h_prior + OCSN_MEAN[k]);
        
        /* Posterior for component k */
        rbpf_real_t h_k = h_prior + K_k * innovation;
        rbpf_real_t P_k = P_prior * (1.0f - K_k);
        
        /* Accumulate weighted by posterior probability */
        h_sum += weights[k] * h_k;
        P_sum += weights[k] * (P_k + (h_k - h_sum) * (h_k - h_sum));
    }
    
    /* Recompute variance with correct mean */
    P_sum = 0.0f;
    for (int k = 0; k < 10; k++)
    {
        rbpf_real_t var_k = OCSN_VAR[k];
        rbpf_real_t S_k = P_prior + var_k;
        rbpf_real_t K_k = P_prior / S_k;
        rbpf_real_t innovation = obs - (h_prior + OCSN_MEAN[k]);
        rbpf_real_t h_k = h_prior + K_k * innovation;
        rbpf_real_t P_k = P_prior * (1.0f - K_k);
        
        /* Variance = E[P] + E[(h - E[h])²] */
        P_sum += weights[k] * (P_k + (h_k - h_sum) * (h_k - h_sum));
    }
    
    *h_post = h_sum;
    *P_post = P_sum;
}

/*═══════════════════════════════════════════════════════════════════════════
 * rbpf_ksc_update_robust
 *
 * Performs RBPF update with 11-component Robust OCSN likelihood.
 * Blends standard OCSN with broad outlier component.
 *
 * @param rbpf          RBPF filter state
 * @param y             Raw return observation
 * @param robust_ocsn   Robust OCSN configuration (per-regime outlier params)
 * @return              Weighted average log-likelihood
 *═══════════════════════════════════════════════════════════════════════════*/

rbpf_real_t rbpf_ksc_update_robust(
    RBPF_KSC *rbpf,
    rbpf_real_t y,
    const RBPF_RobustOCSN *robust_ocsn)
{
    if (!rbpf || !robust_ocsn || !robust_ocsn->enabled)
    {
        /* Fall back to standard update if robust not enabled */
        return rbpf_ksc_update(rbpf, y);
    }
    
    const int n = rbpf->n_particles;
    
    /* Transform observation: obs = log(y²) */
    rbpf_real_t y_sq = y * y;
    if (y_sq < 1e-30f) y_sq = 1e-30f;  /* Prevent log(0) */
    rbpf_real_t obs = rbpf_log(y_sq);
    
    rbpf_real_t max_log_w = -1e30f;
    
    /* Update each particle */
    for (int i = 0; i < n; i++)
    {
        int regime = rbpf->regime[i];
        rbpf_real_t h_prior = rbpf->mu[i];
        rbpf_real_t P_prior = rbpf->var[i];
        
        /* Get regime-specific outlier parameters */
        rbpf_real_t pi_out = robust_ocsn->regime[regime].prob;
        rbpf_real_t var_out = robust_ocsn->regime[regime].variance;
        
        /* Clamp variance to safe bounds */
        if (var_out < RBPF_OUTLIER_VAR_MIN) var_out = RBPF_OUTLIER_VAR_MIN;
        if (var_out > RBPF_OUTLIER_VAR_MAX) var_out = RBPF_OUTLIER_VAR_MAX;
        
        /*───────────────────────────────────────────────────────────────────
         * Compute log-likelihoods
         *─────────────────────────────────────────────────────────────────*/
        
        /* Standard OCSN log-likelihood */
        rbpf_real_t log_lik_ocsn = compute_ocsn_log_likelihood(obs, h_prior);
        
        /* Outlier component log-likelihood: obs ~ N(h, var_out) */
        rbpf_real_t log_lik_out = gaussian_log_pdf(obs, h_prior, var_out);
        
        /* Combined log-likelihood (log-sum-exp) */
        rbpf_real_t log_1_minus_pi = rbpf_log(1.0f - pi_out);
        rbpf_real_t log_pi = rbpf_log(pi_out);
        
        rbpf_real_t log_lik_combined = log_sum_exp_2(
            log_1_minus_pi + log_lik_ocsn,
            log_pi + log_lik_out
        );
        
        /*───────────────────────────────────────────────────────────────────
         * Compute posterior outlier probability
         *─────────────────────────────────────────────────────────────────*/
        
        /* P(outlier | obs) = π × P(obs|outlier) / P(obs) */
        rbpf_real_t log_post_out = log_pi + log_lik_out - log_lik_combined;
        rbpf_real_t post_out = rbpf_exp(log_post_out);
        
        /* Clamp to [0, 1] */
        if (post_out < 0.0f) post_out = 0.0f;
        if (post_out > 1.0f) post_out = 1.0f;
        
        /*───────────────────────────────────────────────────────────────────
         * Blended Kalman update
         *
         * h_post = (1 - post_out) × h_ocsn + post_out × h_outlier
         * P_post = blended variance (more complex)
         *─────────────────────────────────────────────────────────────────*/
        
        /* OCSN posterior */
        rbpf_real_t h_ocsn, P_ocsn;
        compute_ocsn_kalman_update(obs, h_prior, P_prior, &h_ocsn, &P_ocsn);
        
        /* Outlier posterior (simple Kalman with broad variance) */
        rbpf_real_t S_out = P_prior + var_out;
        rbpf_real_t K_out = P_prior / S_out;
        rbpf_real_t innovation_out = obs - h_prior;
        rbpf_real_t h_out = h_prior + K_out * innovation_out;
        rbpf_real_t P_out = P_prior * (1.0f - K_out);
        
        /* Blend posteriors */
        rbpf_real_t w_ocsn = 1.0f - post_out;
        rbpf_real_t w_out = post_out;
        
        rbpf_real_t h_post = w_ocsn * h_ocsn + w_out * h_out;
        
        /* Blended variance = E[P] + E[(h - E[h])²] */
        rbpf_real_t P_post = w_ocsn * P_ocsn + w_out * P_out
                           + w_ocsn * (h_ocsn - h_post) * (h_ocsn - h_post)
                           + w_out * (h_out - h_post) * (h_out - h_post);
        
        /* Store updated state */
        rbpf->mu[i] = h_post;
        rbpf->var[i] = P_post;
        
        /* Update log-weight */
        rbpf->log_weight[i] += log_lik_combined;
        
        if (rbpf->log_weight[i] > max_log_w)
            max_log_w = rbpf->log_weight[i];
    }
    
    /* Normalize weights (log-sum-exp) */
    rbpf_real_t sum_w = 0.0f;
    for (int i = 0; i < n; i++)
    {
        rbpf->w_norm[i] = rbpf_exp(rbpf->log_weight[i] - max_log_w);
        sum_w += rbpf->w_norm[i];
    }
    
    rbpf_real_t inv_sum = 1.0f / sum_w;
    for (int i = 0; i < n; i++)
    {
        rbpf->w_norm[i] *= inv_sum;
    }
    
    /* Compute weighted average log-likelihood */
    rbpf_real_t avg_log_lik = max_log_w + rbpf_log(sum_w / n);
    
    return avg_log_lik;
}

/*═══════════════════════════════════════════════════════════════════════════
 * rbpf_ksc_compute_outlier_fraction
 *
 * Computes the weighted average posterior probability that the observation
 * came from the outlier component (across all particles).
 *
 * Use for:
 *   - Monitoring: High values indicate fat-tail event
 *   - Diagnostics: Should spike on injected outliers
 *   - Alerting: Threshold crossings indicate market stress
 *
 * @param rbpf          RBPF filter state
 * @param y             Raw return observation
 * @param robust_ocsn   Robust OCSN configuration
 * @return              Weighted average P(outlier | obs) in [0, 1]
 *═══════════════════════════════════════════════════════════════════════════*/

rbpf_real_t rbpf_ksc_compute_outlier_fraction(
    const RBPF_KSC *rbpf,
    rbpf_real_t y,
    const RBPF_RobustOCSN *robust_ocsn)
{
    if (!rbpf || !robust_ocsn || !robust_ocsn->enabled)
    {
        return 0.0f;
    }
    
    const int n = rbpf->n_particles;
    
    /* Transform observation */
    rbpf_real_t y_sq = y * y;
    if (y_sq < 1e-30f) y_sq = 1e-30f;
    rbpf_real_t obs = rbpf_log(y_sq);
    
    rbpf_real_t weighted_sum = 0.0f;
    
    for (int i = 0; i < n; i++)
    {
        int regime = rbpf->regime[i];
        rbpf_real_t h = rbpf->mu[i];
        rbpf_real_t weight = rbpf->w_norm[i];
        
        /* Get regime-specific parameters */
        rbpf_real_t pi_out = robust_ocsn->regime[regime].prob;
        rbpf_real_t var_out = robust_ocsn->regime[regime].variance;
        
        /* Clamp variance */
        if (var_out < RBPF_OUTLIER_VAR_MIN) var_out = RBPF_OUTLIER_VAR_MIN;
        if (var_out > RBPF_OUTLIER_VAR_MAX) var_out = RBPF_OUTLIER_VAR_MAX;
        
        /* Compute likelihoods */
        rbpf_real_t log_lik_ocsn = compute_ocsn_log_likelihood(obs, h);
        rbpf_real_t log_lik_out = gaussian_log_pdf(obs, h, var_out);
        
        /* Combined likelihood */
        rbpf_real_t log_1_minus_pi = rbpf_log(1.0f - pi_out);
        rbpf_real_t log_pi = rbpf_log(pi_out);
        
        rbpf_real_t log_lik_combined = log_sum_exp_2(
            log_1_minus_pi + log_lik_ocsn,
            log_pi + log_lik_out
        );
        
        /* Posterior outlier probability */
        rbpf_real_t log_post_out = log_pi + log_lik_out - log_lik_combined;
        rbpf_real_t post_out = rbpf_exp(log_post_out);
        
        /* Clamp */
        if (post_out < 0.0f) post_out = 0.0f;
        if (post_out > 1.0f) post_out = 1.0f;
        
        /* Accumulate weighted by particle weight */
        weighted_sum += weight * post_out;
    }
    
    return weighted_sum;
}
