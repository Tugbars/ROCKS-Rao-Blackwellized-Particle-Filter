/**
 * @file rbpf_sprt.c
 * @brief Sequential Probability Ratio Test (SPRT) for Regime Detection
 *
 * Implementation of Wald's SPRT for statistically principled regime switching.
 */

#include "rbpf_sprt.h"
#include <math.h>
#include <string.h>
#include <float.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/*═══════════════════════════════════════════════════════════════════════════
 * BINARY SPRT IMPLEMENTATION
 *═══════════════════════════════════════════════════════════════════════════*/

void sprt_binary_init(SPRT_Binary *sprt, double alpha, double beta) {
    sprt->alpha = alpha;
    sprt->beta = beta;
    
    /* Wald's thresholds */
    sprt->threshold_high = log((1.0 - beta) / alpha);
    sprt->threshold_low = log(beta / (1.0 - alpha));
    
    sprt->log_ratio = 0.0;
    sprt->current_hypothesis = 0;  /* Start at H₀ */
    sprt->samples_since_reset = 0;
    
    sprt->total_h0_accepts = 0;
    sprt->total_h1_accepts = 0;
    sprt->total_samples = 0;
}

void sprt_binary_reset(SPRT_Binary *sprt) {
    sprt->log_ratio = 0.0;
    sprt->samples_since_reset = 0;
}

SPRT_Decision sprt_binary_update(SPRT_Binary *sprt, double ll_h1, double ll_h0) {
    /* Accumulate log-likelihood ratio */
    double delta = ll_h1 - ll_h0;
    
    /* Clamp extreme values for numerical stability */
    if (delta > 20.0) delta = 20.0;
    if (delta < -20.0) delta = -20.0;
    
    sprt->log_ratio += delta;
    sprt->samples_since_reset++;
    sprt->total_samples++;
    
    /* Check decision boundaries */
    if (sprt->log_ratio > sprt->threshold_high) {
        /* Strong evidence for H₁ */
        sprt->current_hypothesis = 1;
        sprt->total_h1_accepts++;
        sprt_binary_reset(sprt);
        return SPRT_ACCEPT_H1;
    }
    
    if (sprt->log_ratio < sprt->threshold_low) {
        /* Strong evidence for H₀ */
        sprt->current_hypothesis = 0;
        sprt->total_h0_accepts++;
        sprt_binary_reset(sprt);
        return SPRT_ACCEPT_H0;
    }
    
    /* Not enough evidence yet */
    return SPRT_CONTINUE;
}

double sprt_binary_get_ratio(const SPRT_Binary *sprt) {
    return sprt->log_ratio;
}

double sprt_binary_get_evidence(const SPRT_Binary *sprt) {
    /* Map log_ratio from [threshold_low, threshold_high] to [0, 1] */
    double range = sprt->threshold_high - sprt->threshold_low;
    if (range < 1e-10) return 0.5;
    
    double normalized = (sprt->log_ratio - sprt->threshold_low) / range;
    
    /* Clamp to [0, 1] */
    if (normalized < 0.0) normalized = 0.0;
    if (normalized > 1.0) normalized = 1.0;
    
    return normalized;
}

/*═══════════════════════════════════════════════════════════════════════════
 * MULTI-REGIME SPRT IMPLEMENTATION
 *═══════════════════════════════════════════════════════════════════════════*/

void sprt_multi_init(SPRT_Multi *sprt, int n_regimes,
                     double alpha, double beta, int min_dwell) {
    if (n_regimes < 2) n_regimes = 2;
    if (n_regimes > SPRT_MAX_REGIMES) n_regimes = SPRT_MAX_REGIMES;
    
    sprt->n_regimes = n_regimes;
    sprt->current_regime = 0;
    sprt->alpha = alpha;
    sprt->beta = beta;
    sprt->min_dwell_time = min_dwell;
    sprt->ticks_in_current = 0;
    
    /* Initialize all pairwise tests */
    for (int i = 0; i < n_regimes; i++) {
        sprt->regime_evidence[i] = 0.0;
        for (int j = i + 1; j < n_regimes; j++) {
            sprt_binary_init(&sprt->tests[i][j], alpha, beta);
        }
    }
}

/**
 * @brief Aggregate pairwise evidence for each regime
 *
 * For regime k to be favored, it should beat all other regimes in pairwise tests.
 * Evidence = average of pairwise evidences involving k.
 */
static void aggregate_evidence(SPRT_Multi *sprt) {
    int n = sprt->n_regimes;
    
    for (int k = 0; k < n; k++) {
        double sum = 0.0;
        int count = 0;
        
        for (int j = 0; j < n; j++) {
            if (j == k) continue;
            
            /* Get pairwise test (ordered: smaller index first) */
            int i_lo = (k < j) ? k : j;
            int i_hi = (k < j) ? j : k;
            SPRT_Binary *test = &sprt->tests[i_lo][i_hi];
            
            /* Evidence for k in this pair */
            double ev = sprt_binary_get_evidence(test);
            
            /* If k is the higher index, we need to flip (test favors i_lo at 0, i_hi at 1) */
            if (k == i_hi) {
                sum += ev;
            } else {
                sum += (1.0 - ev);
            }
            count++;
        }
        
        sprt->regime_evidence[k] = (count > 0) ? sum / count : 0.5;
    }
}

int sprt_multi_update(SPRT_Multi *sprt, const double *log_liks) {
    int n = sprt->n_regimes;
    int current = sprt->current_regime;
    
    sprt->ticks_in_current++;
    
    /* Update all pairwise tests */
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            /* Test i vs j: H₀ = regime i, H₁ = regime j */
            sprt_binary_update(&sprt->tests[i][j], log_liks[j], log_liks[i]);
        }
    }
    
    /* Aggregate evidence */
    aggregate_evidence(sprt);
    
    /* Check if we should switch (respecting dwell time) */
    if (sprt->ticks_in_current < sprt->min_dwell_time) {
        return current;  /* Too soon to switch */
    }
    
    /* Find regime with highest evidence */
    int best = current;
    double best_evidence = sprt->regime_evidence[current];
    
    for (int k = 0; k < n; k++) {
        if (k != current && sprt->regime_evidence[k] > best_evidence) {
            best = k;
            best_evidence = sprt->regime_evidence[k];
        }
    }
    
    /* Only switch if evidence is strongly in favor */
    /* Threshold: need >0.7 evidence for new regime (clear winner) */
    if (best != current && best_evidence > 0.7) {
        /* Reset all pairwise tests involving old and new regime */
        for (int j = 0; j < n; j++) {
            if (j == current || j == best) continue;
            
            int i_lo = (current < j) ? current : j;
            int i_hi = (current < j) ? j : current;
            sprt_binary_reset(&sprt->tests[i_lo][i_hi]);
            
            i_lo = (best < j) ? best : j;
            i_hi = (best < j) ? j : best;
            sprt_binary_reset(&sprt->tests[i_lo][i_hi]);
        }
        
        sprt->current_regime = best;
        sprt->ticks_in_current = 0;
        return best;
    }
    
    return current;
}

void sprt_multi_get_evidence(const SPRT_Multi *sprt, double *evidence) {
    memcpy(evidence, sprt->regime_evidence, sprt->n_regimes * sizeof(double));
}

void sprt_multi_force_regime(SPRT_Multi *sprt, int regime) {
    if (regime < 0 || regime >= sprt->n_regimes) return;
    
    /* Reset all pairwise tests */
    for (int i = 0; i < sprt->n_regimes; i++) {
        for (int j = i + 1; j < sprt->n_regimes; j++) {
            sprt_binary_reset(&sprt->tests[i][j]);
        }
    }
    
    sprt->current_regime = regime;
    sprt->ticks_in_current = 0;
    
    /* Bias evidence toward forced regime */
    for (int k = 0; k < sprt->n_regimes; k++) {
        sprt->regime_evidence[k] = (k == regime) ? 0.8 : 0.2 / (sprt->n_regimes - 1);
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * LIKELIHOOD HELPERS
 *═══════════════════════════════════════════════════════════════════════════*/

double sprt_gaussian_loglik(double y, double mu, double var) {
    if (var < 1e-10) var = 1e-10;
    
    double diff = y - mu;
    double log_lik = -0.5 * (log(2.0 * M_PI * var) + diff * diff / var);
    
    return log_lik;
}

/* Log-gamma function (Stirling approximation for large x) */
static double log_gamma(double x) {
    if (x <= 0) return 0.0;
    
    /* Use lgamma from math.h if available */
#ifdef _GNU_SOURCE
    return lgamma(x);
#else
    /* Stirling approximation */
    if (x > 10) {
        return (x - 0.5) * log(x) - x + 0.5 * log(2.0 * M_PI)
               + 1.0 / (12.0 * x) - 1.0 / (360.0 * x * x * x);
    }
    
    /* Recursion for small x: Γ(x) = Γ(x+1) / x */
    double result = 0.0;
    while (x < 10) {
        result -= log(x);
        x += 1.0;
    }
    return result + (x - 0.5) * log(x) - x + 0.5 * log(2.0 * M_PI);
#endif
}

double sprt_student_t_loglik(double y, double mu, double var, double nu) {
    if (var < 1e-10) var = 1e-10;
    if (nu < 1.0) nu = 1.0;
    
    double diff = y - mu;
    double scale_sq = var;  /* Assuming var is σ² */
    
    /* Student-t log-PDF:
     * log Γ((ν+1)/2) - log Γ(ν/2) - 0.5*log(ν*π*σ²)
     * - ((ν+1)/2) * log(1 + (y-μ)²/(ν*σ²))
     */
    double log_lik = log_gamma((nu + 1.0) / 2.0)
                   - log_gamma(nu / 2.0)
                   - 0.5 * log(nu * M_PI * scale_sq)
                   - ((nu + 1.0) / 2.0) * log(1.0 + diff * diff / (nu * scale_sq));
    
    return log_lik;
}

/*═══════════════════════════════════════════════════════════════════════════
 * LOG-CHI-SQUARED LIKELIHOOD (KSC Model)
 *
 * Omori et al. (2007) 10-component Gaussian mixture approximation.
 *═══════════════════════════════════════════════════════════════════════════*/

/* OCSN (2007) 10-component mixture parameters for log(χ²(1)) */
static const double OCSN_PI[10] = {
    0.00609, 0.04775, 0.13057, 0.20674, 0.22715,
    0.18842, 0.12047, 0.05591, 0.01575, 0.00115
};

static const double OCSN_MU[10] = {
    1.92677, 1.34744, 0.73504, 0.02266, -0.85173,
    -1.97278, -3.46788, -5.55246, -8.68384, -14.65000
};

static const double OCSN_VAR[10] = {
    0.11265, 0.17788, 0.26768, 0.40611, 0.62699,
    0.98583, 1.57469, 2.54498, 4.16591, 7.33342
};

double sprt_logchisq_loglik(double y_log_sq, double h) {
    /* y = log(r²) = 2h + log(ε²), where log(ε²) ~ log-χ²(1)
     * So: y - 2h ~ log-χ²(1)
     *
     * P(y | h) = Σ_k π_k × N(y | 2h + μ_k, σ_k²)
     */
    double y_shifted = y_log_sq - 2.0 * h;
    
    /* Log-sum-exp for numerical stability */
    double max_log_lik = -1e30;
    double log_liks[10];
    
    for (int k = 0; k < 10; k++) {
        double diff = y_shifted - OCSN_MU[k];
        double log_lik = log(OCSN_PI[k])
                       - 0.5 * log(2.0 * M_PI * OCSN_VAR[k])
                       - 0.5 * diff * diff / OCSN_VAR[k];
        log_liks[k] = log_lik;
        if (log_lik > max_log_lik) max_log_lik = log_lik;
    }
    
    /* Compute log(Σ exp(log_lik - max)) + max */
    double sum = 0.0;
    for (int k = 0; k < 10; k++) {
        sum += exp(log_liks[k] - max_log_lik);
    }
    
    return max_log_lik + log(sum);
}

/*═══════════════════════════════════════════════════════════════════════════
 * REGIME-SPECIFIC LIKELIHOOD FOR MMPF INTEGRATION
 *
 * Computes per-regime observation log-likelihoods using the KSC model.
 * No artificial boosting - trust the math.
 *═══════════════════════════════════════════════════════════════════════════*/

void sprt_compute_regime_logliks(double y_log_sq,
                                  const double *regime_mu,
                                  const double *regime_sigma,
                                  const double *regime_nu,
                                  int n_regimes,
                                  double *log_liks) {
    (void)regime_sigma;  /* Currently unused - could add vol-of-vol weighting */
    (void)regime_nu;     /* Currently unused - Student-t handled by RBPF obs model */
    
    for (int k = 0; k < n_regimes; k++) {
        /* Use regime center h = μ_k for likelihood computation
         * The OCSN mixture handles the observation model correctly */
        double h = regime_mu[k];
        log_liks[k] = sprt_logchisq_loglik(y_log_sq, h);
    }
}