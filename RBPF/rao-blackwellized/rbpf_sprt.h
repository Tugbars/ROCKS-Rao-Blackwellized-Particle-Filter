/**
 * @file rbpf_sprt.h
 * @brief Sequential Probability Ratio Test (SPRT) for Regime Detection
 *
 * Wald's SPRT is the optimal sequential test - it minimizes expected samples
 * to reach a decision at given error rates (α, β).
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * THEORY
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Setup:
 *   H₀: We're in regime A (e.g., Calm)
 *   H₁: We're in regime B (e.g., Crisis)
 *
 * At each observation, compute log-likelihood ratio:
 *   Λ_t = Λ_{t-1} + log(P(y_t | H₁) / P(y_t | H₀))
 *
 * Decision boundaries (from error rates α, β):
 *   A = log((1-β)/α)    Upper threshold: Accept H₁
 *   B = log(β/(1-α))    Lower threshold: Accept H₀
 *
 * Decision rule:
 *   If Λ_t > A → Accept H₁ (switch to Crisis)
 *   If Λ_t < B → Accept H₀ (stay in Calm)
 *   Otherwise  → Continue sampling
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * USAGE IN MMPF
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * SPRT serves a different purpose than BOCPD:
 *
 *   BOCPD: "What's the probability a changepoint just occurred?"
 *          → Continuous probability output
 *          → Good for triggering MCMC shock
 *
 *   SPRT:  "Should I definitively switch regime labels NOW?"
 *          → Binary decision with controlled error rates
 *          → Good for IMM hypothesis switching
 *
 * Recommended: Use BOCPD for shock detection, SPRT for regime labeling.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * MULTI-REGIME EXTENSION
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * For K regimes, maintain K(K-1)/2 pairwise SPRT tests:
 *   - Calm vs Trend
 *   - Calm vs Crisis
 *   - Trend vs Crisis
 *
 * Or use "One vs Rest" approach with K tests:
 *   - Is it Calm? (Calm vs {Trend, Crisis})
 *   - Is it Trend? (Trend vs {Calm, Crisis})
 *   - Is it Crisis? (Crisis vs {Calm, Trend})
 */

#ifndef RBPF_SPRT_H
#define RBPF_SPRT_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*═══════════════════════════════════════════════════════════════════════════
 * BINARY SPRT (Two Hypotheses)
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * @brief SPRT decision result
 */
typedef enum {
    SPRT_CONTINUE = 0,    /**< Not enough evidence, keep sampling */
    SPRT_ACCEPT_H0 = -1,  /**< Accept null hypothesis (stay in current regime) */
    SPRT_ACCEPT_H1 = 1    /**< Accept alternative hypothesis (switch regime) */
} SPRT_Decision;

/**
 * @brief Binary SPRT detector state
 */
typedef struct {
    double log_ratio;        /**< Cumulative log-likelihood ratio Λ */
    double threshold_high;   /**< A: Accept H₁ if Λ > A */
    double threshold_low;    /**< B: Accept H₀ if Λ < B */
    int current_hypothesis;  /**< 0 = H₀, 1 = H₁ */
    int samples_since_reset; /**< Ticks since last decision */
    
    /* Configuration */
    double alpha;            /**< Type I error rate (false positive) */
    double beta;             /**< Type II error rate (false negative) */
    
    /* Statistics */
    int total_h0_accepts;
    int total_h1_accepts;
    int total_samples;
} SPRT_Binary;

/**
 * @brief Initialize binary SPRT with error rates
 *
 * @param sprt   Detector state
 * @param alpha  Type I error rate (false positive), typically 0.01-0.05
 * @param beta   Type II error rate (false negative), typically 0.01-0.05
 *
 * Lower α, β → higher thresholds → more samples needed → fewer false alarms
 * Higher α, β → lower thresholds → faster decisions → more errors
 */
void sprt_binary_init(SPRT_Binary *sprt, double alpha, double beta);

/**
 * @brief Reset SPRT after a decision
 *
 * Call this after SPRT_ACCEPT_H0 or SPRT_ACCEPT_H1 to start new test.
 */
void sprt_binary_reset(SPRT_Binary *sprt);

/**
 * @brief Update SPRT with new observation
 *
 * @param sprt   Detector state
 * @param ll_h1  Log-likelihood under H₁ (alternative)
 * @param ll_h0  Log-likelihood under H₀ (null)
 * @return       Decision: CONTINUE, ACCEPT_H0, or ACCEPT_H1
 *
 * The caller computes log-likelihoods based on their model.
 * For regime detection, this might be:
 *   ll_h1 = log P(y_t | Crisis model)
 *   ll_h0 = log P(y_t | Calm model)
 */
SPRT_Decision sprt_binary_update(SPRT_Binary *sprt, double ll_h1, double ll_h0);

/**
 * @brief Get current log-likelihood ratio
 */
double sprt_binary_get_ratio(const SPRT_Binary *sprt);

/**
 * @brief Get normalized evidence (0 to 1 scale for visualization)
 *
 * Maps Λ from [threshold_low, threshold_high] to [0, 1]
 * Values outside range clamp to 0 or 1.
 */
double sprt_binary_get_evidence(const SPRT_Binary *sprt);

/*═══════════════════════════════════════════════════════════════════════════
 * MULTI-REGIME SPRT (K Hypotheses)
 *═══════════════════════════════════════════════════════════════════════════*/

#define SPRT_MAX_REGIMES 8

/**
 * @brief Multi-regime SPRT using pairwise tests
 *
 * For K regimes, maintains K(K-1)/2 pairwise SPRT detectors.
 * Decision rule: Switch to regime j if all tests involving j favor j.
 */
typedef struct {
    int n_regimes;
    int current_regime;
    
    /* Pairwise tests: tests[i][j] compares regime i vs regime j (i < j) */
    SPRT_Binary tests[SPRT_MAX_REGIMES][SPRT_MAX_REGIMES];
    
    /* Per-regime evidence (aggregated from pairwise tests) */
    double regime_evidence[SPRT_MAX_REGIMES];
    
    /* Configuration */
    double alpha;
    double beta;
    
    /* Minimum ticks before allowing switch (prevents rapid flipping) */
    int min_dwell_time;
    int ticks_in_current;
} SPRT_Multi;

/**
 * @brief Initialize multi-regime SPRT
 *
 * @param sprt         Detector state
 * @param n_regimes    Number of regimes (2-8)
 * @param alpha        Type I error rate
 * @param beta         Type II error rate
 * @param min_dwell    Minimum ticks in regime before switch allowed
 */
void sprt_multi_init(SPRT_Multi *sprt, int n_regimes,
                     double alpha, double beta, int min_dwell);

/**
 * @brief Update multi-regime SPRT
 *
 * @param sprt         Detector state
 * @param log_liks     Log-likelihoods under each regime [n_regimes]
 * @return             New regime index (may be same as current)
 *
 * Internally updates all pairwise tests and determines if regime should switch.
 */
int sprt_multi_update(SPRT_Multi *sprt, const double *log_liks);

/**
 * @brief Get evidence for each regime (for visualization/debugging)
 *
 * @param sprt         Detector state
 * @param evidence     Output array [n_regimes], values in [0, 1]
 */
void sprt_multi_get_evidence(const SPRT_Multi *sprt, double *evidence);

/**
 * @brief Force regime (external override)
 *
 * Use when shock detector (BOCPD) fires - force SPRT to a specific regime.
 */
void sprt_multi_force_regime(SPRT_Multi *sprt, int regime);

/*═══════════════════════════════════════════════════════════════════════════
 * LIKELIHOOD HELPERS
 *
 * Common likelihood functions for regime comparison.
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Gaussian log-likelihood
 *
 * log P(y | μ, σ²) = -0.5 * [log(2πσ²) + (y-μ)²/σ²]
 */
double sprt_gaussian_loglik(double y, double mu, double var);

/**
 * @brief Student-t log-likelihood
 *
 * log P(y | μ, σ², ν) = log Γ((ν+1)/2) - log Γ(ν/2) - 0.5*log(νπσ²)
 *                       - ((ν+1)/2) * log(1 + (y-μ)²/(νσ²))
 */
double sprt_student_t_loglik(double y, double mu, double var, double nu);

/**
 * @brief Log-chi-squared log-likelihood (KSC observation model)
 *
 * For y = log(r²) where r ~ N(0, exp(h)):
 * Uses 10-component Gaussian mixture approximation.
 *
 * @param y_log_sq   Observed log(return²)
 * @param h          Log-volatility state
 * @return           Log-likelihood
 */
double sprt_logchisq_loglik(double y_log_sq, double h);

#ifdef __cplusplus
}
#endif

#endif /* RBPF_SPRT_H */
