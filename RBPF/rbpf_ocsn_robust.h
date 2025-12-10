/**
 * @file rbpf_ocsn_robust.h
 * @brief Robust OCSN Likelihood with 11th "Outlier" Component
 *
 * Problem: Standard 10-component OCSN mixture approximates log(χ²(1)) well,
 * but real market returns have heavier tails. During crashes, the likelihood
 * for extreme observations goes to zero, killing particles.
 *
 * Solution: Add an 11th "outlier" component - a wide Gaussian that acts as
 * a safety valve for tail events. This preserves OCSN precision for normal
 * trading while keeping particles alive during 8σ moves.
 *
 * Math:
 *   P(obs) = (1 - π_outlier) × P_OCSN(obs) + π_outlier × P_broad(obs)
 *
 * Reference: This is standard "robust SV" engineering practice.
 */

#ifndef RBPF_OCSN_ROBUST_H
#define RBPF_OCSN_ROBUST_H

#include "rbpf_ksc.h"

#ifdef __cplusplus
extern "C" {
#endif

/*═══════════════════════════════════════════════════════════════════════════
 * CONFIGURATION
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Outlier component parameters
 * 
 * These can be regime-dependent if desired.
 */
typedef struct {
    float prob;           /* π_outlier: probability of outlier (default: 0.01 = 1%) */
    float mean_shift;     /* Offset from h_t for outlier mean (default: 0.0) */
    float variance;       /* Outlier variance (default: 20.0, very wide) */
} OutlierConfig;

/**
 * Per-regime outlier configuration
 * 
 * Crisis regimes may have higher outlier probability and wider variance.
 */
typedef struct {
    int n_regimes;
    OutlierConfig regime[8];
    int enabled;          /* 0 = use standard OCSN, 1 = use robust OCSN */
} RobustOCSNConfig;

/*═══════════════════════════════════════════════════════════════════════════
 * DEFAULT CONFIGURATIONS
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Get default robust OCSN configuration
 * Same outlier params for all regimes
 */
static inline RobustOCSNConfig robust_ocsn_config_defaults(void)
{
    RobustOCSNConfig cfg;
    cfg.n_regimes = 4;
    cfg.enabled = 1;
    
    for (int r = 0; r < 8; r++) {
        cfg.regime[r].prob = 0.01f;        /* 1% outlier probability */
        cfg.regime[r].mean_shift = 0.0f;   /* Centered on h_t */
        cfg.regime[r].variance = 20.0f;    /* Wide (OCSN vars are ~0.1 to ~7.3) */
    }
    
    return cfg;
}

/**
 * Get regime-scaled configuration
 * Higher outlier probability in crisis regimes
 */
static inline RobustOCSNConfig robust_ocsn_config_regime_scaled(void)
{
    RobustOCSNConfig cfg;
    cfg.n_regimes = 4;
    cfg.enabled = 1;
    
    /* R0: Calm - rare outliers */
    cfg.regime[0].prob = 0.005f;
    cfg.regime[0].mean_shift = 0.0f;
    cfg.regime[0].variance = 15.0f;
    
    /* R1: Mild - occasional outliers */
    cfg.regime[1].prob = 0.01f;
    cfg.regime[1].mean_shift = 0.0f;
    cfg.regime[1].variance = 18.0f;
    
    /* R2: Elevated - more outliers */
    cfg.regime[2].prob = 0.02f;
    cfg.regime[2].mean_shift = 0.0f;
    cfg.regime[2].variance = 22.0f;
    
    /* R3: Crisis - frequent outliers, very wide */
    cfg.regime[3].prob = 0.03f;
    cfg.regime[3].mean_shift = 0.0f;
    cfg.regime[3].variance = 30.0f;
    
    /* Copy for regimes 4-7 if needed */
    for (int r = 4; r < 8; r++) {
        cfg.regime[r] = cfg.regime[3];
    }
    
    return cfg;
}

/*═══════════════════════════════════════════════════════════════════════════
 * CORE LIKELIHOOD FUNCTIONS
 *═══════════════════════════════════════════════════════════════════════════*/

/* Precomputed constants */
#define OCSN_INV_SQRT_2PI  0.3989422804f   /* 1/sqrt(2π) */
#define OCSN_LOG_2PI_HALF  0.9189385332f   /* 0.5 * log(2π) */

/**
 * Standard OCSN log-likelihood (unchanged from your current implementation)
 * 
 * @param log_y_sq  Observed log(y²) where y is the return
 * @param h_t       Current log-volatility state
 * @return Log-likelihood
 */
static inline float ocsn_loglik_standard(float log_y_sq, float h_t,
                                         const rbpf_real_t *prob,
                                         const rbpf_real_t *mean,
                                         const rbpf_real_t *var,
                                         int n_components)
{
    float lik = 0.0f;
    
    for (int k = 0; k < n_components; k++) {
        float m = h_t + (float)mean[k];
        float v = (float)var[k];
        float diff = log_y_sq - m;
        
        float dens = OCSN_INV_SQRT_2PI / sqrtf(v) * expf(-0.5f * diff * diff / v);
        lik += (float)prob[k] * dens;
    }
    
    return logf(lik + 1e-30f);
}

/**
 * Robust OCSN log-likelihood with 11th outlier component
 * 
 * This is the main function to use instead of the standard likelihood.
 * 
 * @param log_y_sq  Observed log(y²)
 * @param h_t       Current log-volatility state
 * @param regime    Current regime (for regime-specific outlier params)
 * @param cfg       Robust OCSN configuration
 * @param prob      OCSN mixture probabilities [n_components]
 * @param mean      OCSN mixture means [n_components]
 * @param var       OCSN mixture variances [n_components]
 * @param n_components  Number of OCSN components (10)
 * @return Log-likelihood
 */
static inline float ocsn_loglik_robust(float log_y_sq, float h_t, int regime,
                                       const RobustOCSNConfig *cfg,
                                       const rbpf_real_t *prob,
                                       const rbpf_real_t *mean,
                                       const rbpf_real_t *var,
                                       int n_components)
{
    /* 1. Standard OCSN likelihood (the "body") */
    float lik_ocsn = 0.0f;
    
    for (int k = 0; k < n_components; k++) {
        float m = h_t + (float)mean[k];
        float v = (float)var[k];
        float diff = log_y_sq - m;
        
        float dens = OCSN_INV_SQRT_2PI / sqrtf(v) * expf(-0.5f * diff * diff / v);
        lik_ocsn += (float)prob[k] * dens;
    }
    
    /* 2. Outlier likelihood (the "tail") - single wide Gaussian */
    const OutlierConfig *out = &cfg->regime[regime];
    float diff_out = log_y_sq - (h_t + out->mean_shift);
    float lik_outlier = OCSN_INV_SQRT_2PI / sqrtf(out->variance) * 
                        expf(-0.5f * diff_out * diff_out / out->variance);
    
    /* 3. Mix them */
    float total_lik = (1.0f - out->prob) * lik_ocsn + out->prob * lik_outlier;
    
    return logf(total_lik + 1e-30f);
}

/**
 * Simplified robust OCSN (regime-independent)
 * 
 * Use this if you don't need per-regime outlier parameters.
 * 
 * @param log_y_sq      Observed log(y²)
 * @param h_t           Current log-volatility state
 * @param outlier_prob  Outlier probability (e.g., 0.01)
 * @param outlier_var   Outlier variance (e.g., 20.0)
 * @param prob          OCSN mixture probabilities
 * @param mean          OCSN mixture means
 * @param var           OCSN mixture variances
 * @param n_components  Number of OCSN components
 * @return Log-likelihood
 */
static inline float ocsn_loglik_robust_simple(float log_y_sq, float h_t,
                                              float outlier_prob, float outlier_var,
                                              const rbpf_real_t *prob,
                                              const rbpf_real_t *mean,
                                              const rbpf_real_t *var,
                                              int n_components)
{
    /* 1. Standard OCSN */
    float lik_ocsn = 0.0f;
    for (int k = 0; k < n_components; k++) {
        float m = h_t + (float)mean[k];
        float v = (float)var[k];
        float diff = log_y_sq - m;
        float dens = OCSN_INV_SQRT_2PI / sqrtf(v) * expf(-0.5f * diff * diff / v);
        lik_ocsn += (float)prob[k] * dens;
    }
    
    /* 2. Single outlier component */
    float diff_out = log_y_sq - h_t;
    float lik_outlier = OCSN_INV_SQRT_2PI / sqrtf(outlier_var) * 
                        expf(-0.5f * diff_out * diff_out / outlier_var);
    
    /* 3. Mix */
    float total_lik = (1.0f - outlier_prob) * lik_ocsn + outlier_prob * lik_outlier;
    
    return logf(total_lik + 1e-30f);
}

/*═══════════════════════════════════════════════════════════════════════════
 * VECTORIZED VERSION (for batch processing)
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Compute robust OCSN log-likelihood for multiple particles
 * 
 * @param log_y_sq      Single observation log(y²)
 * @param h_t           Array of log-volatilities [n_particles]
 * @param regimes       Array of regimes [n_particles]
 * @param cfg           Robust OCSN configuration
 * @param out_loglik    Output log-likelihoods [n_particles]
 * @param n_particles   Number of particles
 * @param prob          OCSN mixture probabilities
 * @param mean          OCSN mixture means
 * @param var           OCSN mixture variances
 * @param n_components  Number of OCSN components
 */
static inline void ocsn_loglik_robust_batch(float log_y_sq,
                                            const float *h_t,
                                            const int *regimes,
                                            const RobustOCSNConfig *cfg,
                                            float *out_loglik,
                                            int n_particles,
                                            const rbpf_real_t *prob,
                                            const rbpf_real_t *mean,
                                            const rbpf_real_t *var,
                                            int n_components)
{
    for (int i = 0; i < n_particles; i++) {
        out_loglik[i] = ocsn_loglik_robust(log_y_sq, h_t[i], regimes[i], cfg,
                                           prob, mean, var, n_components);
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * DIAGNOSTICS
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Compute breakdown of likelihood between OCSN and outlier components
 * Useful for monitoring how often the outlier component is being used.
 * 
 * @param log_y_sq      Observed log(y²)
 * @param h_t           Current log-volatility state
 * @param regime        Current regime
 * @param cfg           Robust OCSN configuration
 * @param out_ocsn_frac Output: fraction of likelihood from OCSN component
 * @param out_outlier_frac Output: fraction from outlier component
 */
static inline void ocsn_likelihood_breakdown(float log_y_sq, float h_t, int regime,
                                             const RobustOCSNConfig *cfg,
                                             const rbpf_real_t *prob,
                                             const rbpf_real_t *mean,
                                             const rbpf_real_t *var,
                                             int n_components,
                                             float *out_ocsn_frac,
                                             float *out_outlier_frac)
{
    /* OCSN likelihood */
    float lik_ocsn = 0.0f;
    for (int k = 0; k < n_components; k++) {
        float m = h_t + (float)mean[k];
        float v = (float)var[k];
        float diff = log_y_sq - m;
        float dens = OCSN_INV_SQRT_2PI / sqrtf(v) * expf(-0.5f * diff * diff / v);
        lik_ocsn += (float)prob[k] * dens;
    }
    
    /* Outlier likelihood */
    const OutlierConfig *out = &cfg->regime[regime];
    float diff_out = log_y_sq - (h_t + out->mean_shift);
    float lik_outlier = OCSN_INV_SQRT_2PI / sqrtf(out->variance) * 
                        expf(-0.5f * diff_out * diff_out / out->variance);
    
    /* Weighted contributions */
    float contrib_ocsn = (1.0f - out->prob) * lik_ocsn;
    float contrib_outlier = out->prob * lik_outlier;
    float total = contrib_ocsn + contrib_outlier + 1e-30f;
    
    if (out_ocsn_frac) *out_ocsn_frac = contrib_ocsn / total;
    if (out_outlier_frac) *out_outlier_frac = contrib_outlier / total;
}

/**
 * Check if observation is being explained primarily by outlier component
 * 
 * @return 1 if outlier fraction > threshold, 0 otherwise
 */
static inline int ocsn_is_outlier(float log_y_sq, float h_t, int regime,
                                  const RobustOCSNConfig *cfg,
                                  const rbpf_real_t *prob,
                                  const rbpf_real_t *mean,
                                  const rbpf_real_t *var,
                                  int n_components,
                                  float threshold)
{
    float ocsn_frac, outlier_frac;
    ocsn_likelihood_breakdown(log_y_sq, h_t, regime, cfg, prob, mean, var,
                              n_components, &ocsn_frac, &outlier_frac);
    return outlier_frac > threshold ? 1 : 0;
}

#ifdef __cplusplus
}
#endif

#endif /* RBPF_OCSN_ROBUST_H */
