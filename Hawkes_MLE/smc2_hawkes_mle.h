/**
 * @file smc2_hawkes_mle.h
 * @brief Maximum Likelihood Estimation for Hawkes process parameters
 *
 * Fits (α, β, λ₀) from observed timestamps using MLE.
 * Intended to run on slow path (periodic calibration).
 *
 * Log-likelihood for univariate Hawkes:
 *   L(α, β, λ₀) = Σᵢ log(λ(tᵢ)) - ∫₀ᵀ λ(t)dt
 *
 * where:
 *   λ(t) = λ₀ + Σ_{tⱼ<t} α·exp(-β(t - tⱼ))
 *
 * Reference: Ogata (1981), "On Lewis' simulation method for point processes"
 */

#ifndef SMC2_HAWKES_MLE_H
#define SMC2_HAWKES_MLE_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C"
{
#endif

    /* ═══════════════════════════════════════════════════════════════════════════
     * Types
     * ═══════════════════════════════════════════════════════════════════════════ */

    /**
     * Hawkes MLE result
     */
    typedef struct HawkesMLE
    {
        /* Fitted parameters */
        float alpha;   /* Jump size per event */
        float beta;    /* Decay rate (1/mean_decay_time) */
        float lambda0; /* Baseline intensity */

        /* Derived quantities */
        float branching_ratio;   /* α/β - must be < 1 for stationarity */
        float mean_intensity;    /* λ₀ / (1 - α/β) */
        float mean_cluster_size; /* 1 / (1 - α/β) */

        /* Fit quality */
        float log_likelihood; /* Log-likelihood at fitted params */
        float aic;            /* Akaike Information Criterion */
        float bic;            /* Bayesian Information Criterion */

        /* Diagnostics */
        int n_events;    /* Number of events used */
        double duration; /* T - t₀ */
        int iterations;  /* Optimizer iterations */
        bool converged;  /* Did optimizer converge? */

    } HawkesMLE;

    /**
     * Hawkes MLE configuration
     */
    typedef struct HawkesMLEConfig
    {
        /* Parameter bounds */
        float alpha_min, alpha_max;     /* [0.01, 10.0] typical */
        float beta_min, beta_max;       /* [0.01, 10.0] typical */
        float lambda0_min, lambda0_max; /* [0.01, 10.0] typical */

        /* Optimizer settings */
        int max_iterations;  /* Max gradient descent steps */
        float tolerance;     /* Convergence tolerance */
        float learning_rate; /* Initial step size */

        /* Grid search for initialization */
        int grid_points; /* Points per dimension (e.g., 5 → 125 evals) */

    } HawkesMLEConfig;

    /* ═══════════════════════════════════════════════════════════════════════════
     * API
     * ═══════════════════════════════════════════════════════════════════════════ */

    /**
     * Get default MLE configuration
     */
    HawkesMLEConfig hawkes_mle_default_config(void);

    /**
     * Fit Hawkes parameters from timestamps
     *
     * @param timestamps  Array of event times (must be sorted ascending)
     * @param n           Number of events
     * @param config      MLE configuration (NULL for defaults)
     * @return            Fitted parameters and diagnostics
     */
    HawkesMLE hawkes_mle_fit(const double *timestamps, int n,
                             const HawkesMLEConfig *config);

    /**
     * Compute log-likelihood for given parameters
     *
     * Useful for comparing models or validating fits.
     *
     * @param timestamps  Array of event times
     * @param n           Number of events
     * @param alpha       Jump size
     * @param beta        Decay rate
     * @param lambda0     Baseline intensity
     * @return            Log-likelihood value
     */
    double hawkes_mle_loglik(const double *timestamps, int n,
                             float alpha, float beta, float lambda0);

    /**
     * Compute intensity at each event time
     *
     * @param timestamps  Array of event times
     * @param n           Number of events
     * @param alpha       Jump size
     * @param beta        Decay rate
     * @param lambda0     Baseline intensity
     * @param lambda_out  Output array [n] for intensities
     */
    void hawkes_mle_intensity(const double *timestamps, int n,
                              float alpha, float beta, float lambda0,
                              float *lambda_out);

    /**
     * Print MLE results
     */
    void hawkes_mle_print(const HawkesMLE *result);

    /**
     * Estimate gamma (h coupling) from returns and timestamps
     *
     * Fits: log|y_t| = a + γ·log(λ_t/λ₀) + ε
     *
     * This gives the coupling strength between intensity and volatility.
     *
     * @param timestamps  Event times
     * @param returns     Return values at each event
     * @param n           Number of events
     * @param alpha       Hawkes alpha (from MLE)
     * @param beta        Hawkes beta (from MLE)
     * @param lambda0     Hawkes lambda0 (from MLE)
     * @return            Estimated gamma (coupling strength)
     */
    float hawkes_mle_estimate_gamma(const double *timestamps,
                                    const float *returns, int n,
                                    float alpha, float beta, float lambda0);

#ifdef __cplusplus
}
#endif

#endif /* SMC2_HAWKES_MLE_H */