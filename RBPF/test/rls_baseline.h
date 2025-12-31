/**
 * @file rls_baseline.h
 * @brief RLS (Recursive Least Squares) Baseline Tracker
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * Tracks baseline intensity using RLS/Kalman filtering instead of EMA.
 *
 * Advantages over EMA:
 *   - Adaptive gain: responds faster during high uncertainty
 *   - Optimal in MSE sense for linear Gaussian systems
 *   - Uncertainty tracking (P) for confidence-aware thresholds
 *   - Innovation (surprise) as natural byproduct
 *
 * Model:
 *   x_t = x_{t-1} + w_t     (random walk baseline)
 *   y_t = x_t + v_t         (noisy observation)
 *
 * This reduces to a scalar Kalman filter.
 *
 * Usage:
 *   RLS_Baseline rls;
 *   rls_baseline_init(&rls, 0.05f);  // Initial baseline guess
 *
 *   for each tick:
 *       float innovation = rls_baseline_update(&rls, observed_intensity);
 *       float entry_thresh = rls_baseline_get_threshold(&rls, 2.5f);  // 2.5σ
 *       float exit_thresh  = rls_baseline_get_threshold(&rls, 1.0f);  // 1σ
 *
 * ═══════════════════════════════════════════════════════════════════════════
 */

#ifndef RLS_BASELINE_H
#define RLS_BASELINE_H

#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/*═══════════════════════════════════════════════════════════════════════════
 * CONFIGURATION
 *═══════════════════════════════════════════════════════════════════════════*/

#define RLS_DEFAULT_P0     1.0f      /* Initial uncertainty */
#define RLS_DEFAULT_Q      1e-5f    /* Process noise (baseline drift rate) */
#define RLS_DEFAULT_R      0.01f    /* Observation noise */
#define RLS_MIN_P          1e-6f    /* Floor to prevent numerical issues */
#define RLS_MAX_P          10.0f    /* Ceiling to prevent divergence */

/*═══════════════════════════════════════════════════════════════════════════
 * RLS BASELINE TRACKER
 *═══════════════════════════════════════════════════════════════════════════*/

typedef struct {
    /* State */
    float mu;           /**< Estimated baseline (state) */
    float P;            /**< State variance (uncertainty) */
    
    /* Noise parameters */
    float Q;            /**< Process noise variance */
    float R;            /**< Observation noise variance */
    
    /* Observation noise estimation (online) */
    float R_ema;        /**< EMA of squared innovations */
    float R_alpha;      /**< EMA coefficient for R estimation */
    int   adapt_R;      /**< Whether to adapt R online */
    
    /* Running variance for σ estimation */
    float var_ema;      /**< EMA of (x - μ)² */
    float sigma;        /**< sqrt(var_ema) - for threshold scaling */
    float var_alpha;    /**< EMA coefficient for variance */
    
    /* Last innovation for diagnostics */
    float last_innovation;
    float last_K;       /**< Last Kalman gain */
    
    /* Statistics */
    int ticks;
    int warmup_ticks;
    int warmed_up;
} RLS_Baseline;

/*═══════════════════════════════════════════════════════════════════════════
 * API - LIFECYCLE
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Initialize RLS baseline tracker
 *
 * @param rls      Tracker instance
 * @param mu0      Initial baseline estimate (e.g., 0.05 for Hawkes)
 */
static inline void rls_baseline_init(RLS_Baseline *rls, float mu0)
{
    rls->mu = mu0;
    rls->P = RLS_DEFAULT_P0;
    rls->Q = RLS_DEFAULT_Q;
    rls->R = RLS_DEFAULT_R;
    
    rls->R_ema = RLS_DEFAULT_R;
    rls->R_alpha = 0.01f;      /* Slow adaptation of R */
    rls->adapt_R = 0;          /* Disabled by default - manual R is better for crisis detection */
    
    rls->var_ema = 0.0025f;    /* Initial variance estimate (σ ≈ 0.05) */
    rls->sigma = 0.05f;        /* Initial σ - conservative */
    rls->var_alpha = 0.01f;    /* ~100 tick memory for variance */
    
    rls->last_innovation = 0.0f;
    rls->last_K = 0.0f;
    
    rls->ticks = 0;
    rls->warmup_ticks = 100;
    rls->warmed_up = 0;
}

/**
 * @brief Initialize with custom noise parameters
 */
static inline void rls_baseline_init_custom(RLS_Baseline *rls, 
                                             float mu0, float P0,
                                             float Q, float R)
{
    rls_baseline_init(rls, mu0);
    rls->P = P0;
    rls->Q = Q;
    rls->R = R;
    rls->R_ema = R;
}

/*═══════════════════════════════════════════════════════════════════════════
 * API - CORE UPDATE
 *═══════════════════════════════════════════════════════════════════════════*/

/* Forward declaration */
static inline float rls_baseline_update_conditional(RLS_Baseline *rls, float x, int do_update);

/**
 * @brief Update baseline estimate with new observation
 *
 * Runs one step of scalar Kalman filter:
 *   1. Predict: P_pred = P + Q
 *   2. Update:  K = P_pred / (P_pred + R)
 *               μ = μ + K * innovation
 *               P = (1 - K) * P_pred
 *
 * @param rls   Tracker instance
 * @param x     New observation (e.g., Hawkes intensity)
 * @return      Innovation (surprise) = x - μ_predicted
 */
static inline float rls_baseline_update(RLS_Baseline *rls, float x)
{
    return rls_baseline_update_conditional(rls, x, 1);  /* Always update */
}

/**
 * @brief Conditional update - can freeze learning
 *
 * @param rls           Tracker instance
 * @param x             New observation
 * @param do_update     If 0, only compute innovation but don't update state
 * @return              Innovation
 */
static inline float rls_baseline_update_conditional(RLS_Baseline *rls, float x, int do_update)
{
    rls->ticks++;
    
    /*═══════════════════════════════════════════════════════════════════════
     * PREDICT STEP
     *═══════════════════════════════════════════════════════════════════════*/
    float P_pred = rls->P + rls->Q;
    
    /* Clamp to prevent numerical issues */
    if (P_pred > RLS_MAX_P) P_pred = RLS_MAX_P;
    
    /*═══════════════════════════════════════════════════════════════════════
     * COMPUTE INNOVATION (always, for diagnostics)
     *═══════════════════════════════════════════════════════════════════════*/
    float innovation = x - rls->mu;
    rls->last_innovation = innovation;
    
    /* Kalman gain */
    float K = P_pred / (P_pred + rls->R);
    rls->last_K = K;
    
    /* Early exit if frozen */
    if (!do_update) {
        /* Still check warmup */
        if (!rls->warmed_up && rls->ticks >= rls->warmup_ticks) {
            rls->warmed_up = 1;
        }
        return innovation;
    }
    
    /*═══════════════════════════════════════════════════════════════════════
     * UPDATE STEP (only if not frozen)
     *═══════════════════════════════════════════════════════════════════════*/
    
    /* State update */
    rls->mu = rls->mu + K * innovation;
    
    /* Covariance update (Joseph form for stability) */
    rls->P = (1.0f - K) * P_pred;
    if (rls->P < RLS_MIN_P) rls->P = RLS_MIN_P;
    
    /*═══════════════════════════════════════════════════════════════════════
     * ADAPT OBSERVATION NOISE (Optional)
     *
     * Estimate R from innovation variance:
     *   E[innovation²] = P_pred + R
     *   R_est = innovation² - P_pred  (clamped)
     *═══════════════════════════════════════════════════════════════════════*/
    if (rls->adapt_R && rls->ticks > rls->warmup_ticks)
    {
        float innov_sq = innovation * innovation;
        rls->R_ema = rls->R_alpha * innov_sq + (1.0f - rls->R_alpha) * rls->R_ema;
        
        /* R should be positive and bounded */
        float R_est = rls->R_ema;
        if (R_est < 1e-6f) R_est = 1e-6f;
        if (R_est > 1.0f) R_est = 1.0f;
        rls->R = R_est;
    }
    
    /*═══════════════════════════════════════════════════════════════════════
     * UPDATE VARIANCE ESTIMATE (for σ thresholds)
     *
     * Track variance of observations around baseline
     *═══════════════════════════════════════════════════════════════════════*/
    float deviation_sq = innovation * innovation;
    rls->var_ema = rls->var_alpha * deviation_sq + (1.0f - rls->var_alpha) * rls->var_ema;
    rls->sigma = sqrtf(rls->var_ema);
    
    /* Clamp σ for stability - allow larger range for crisis detection */
    if (rls->sigma < 0.005f) rls->sigma = 0.005f;  /* Floor: don't go too tight */
    if (rls->sigma > 0.5f) rls->sigma = 0.5f;      /* Ceiling: reasonable max */
    
    /*═══════════════════════════════════════════════════════════════════════
     * WARMUP CHECK
     *═══════════════════════════════════════════════════════════════════════*/
    if (!rls->warmed_up && rls->ticks >= rls->warmup_ticks)
    {
        rls->warmed_up = 1;
    }
    
    return innovation;
}

/*═══════════════════════════════════════════════════════════════════════════
 * API - QUERIES
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Get current baseline estimate
 */
static inline float rls_baseline_get_mu(const RLS_Baseline *rls)
{
    return rls->mu;
}

/**
 * @brief Get current standard deviation estimate
 */
static inline float rls_baseline_get_sigma(const RLS_Baseline *rls)
{
    return rls->sigma;
}

/**
 * @brief Get adaptive threshold at n standard deviations above baseline
 *
 * @param rls   Tracker instance
 * @param n_sigma  Number of standard deviations (e.g., 2.5 for entry)
 * @return      Threshold = μ + n_sigma * σ
 */
static inline float rls_baseline_get_threshold(const RLS_Baseline *rls, float n_sigma)
{
    return rls->mu + n_sigma * rls->sigma;
}

/**
 * @brief Get last innovation (surprise)
 */
static inline float rls_baseline_get_innovation(const RLS_Baseline *rls)
{
    return rls->last_innovation;
}

/**
 * @brief Get surprise in σ units
 */
static inline float rls_baseline_get_surprise_sigma(const RLS_Baseline *rls)
{
    if (rls->sigma < 0.001f) return 0.0f;
    return rls->last_innovation / rls->sigma;
}

/**
 * @brief Get current Kalman gain (for diagnostics)
 *
 * K ≈ 0.5 → balanced between prior and observation
 * K → 1.0 → trusting observations (high uncertainty)
 * K → 0.0 → trusting prior (low uncertainty)
 */
static inline float rls_baseline_get_kalman_gain(const RLS_Baseline *rls)
{
    return rls->last_K;
}

/**
 * @brief Get uncertainty (P)
 */
static inline float rls_baseline_get_uncertainty(const RLS_Baseline *rls)
{
    return rls->P;
}

/**
 * @brief Check if warmed up
 */
static inline int rls_baseline_is_ready(const RLS_Baseline *rls)
{
    return rls->warmed_up;
}

/*═══════════════════════════════════════════════════════════════════════════
 * API - CONFIGURATION
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Set process noise (baseline drift rate)
 *
 * Higher Q → baseline can change faster → more responsive
 * Lower Q → baseline is stable → smoother
 */
static inline void rls_baseline_set_Q(RLS_Baseline *rls, float Q)
{
    rls->Q = Q;
}

/**
 * @brief Set observation noise
 *
 * Higher R → observations are noisy → trust prior more
 * Lower R → observations are accurate → trust observations more
 */
static inline void rls_baseline_set_R(RLS_Baseline *rls, float R)
{
    rls->R = R;
    rls->R_ema = R;
}

/**
 * @brief Enable/disable online R adaptation
 */
static inline void rls_baseline_set_adapt_R(RLS_Baseline *rls, int enable)
{
    rls->adapt_R = enable;
}

/**
 * @brief Boost uncertainty (after detected regime change)
 *
 * Call this when you detect a structural break to make the
 * filter respond faster temporarily.
 */
static inline void rls_baseline_boost_uncertainty(RLS_Baseline *rls, float factor)
{
    rls->P *= factor;
    if (rls->P > RLS_MAX_P) rls->P = RLS_MAX_P;
}

/**
 * @brief Reset to initial state
 */
static inline void rls_baseline_reset(RLS_Baseline *rls, float mu0)
{
    float Q = rls->Q;
    float R = rls->R;
    rls_baseline_init(rls, mu0);
    rls->Q = Q;
    rls->R = R;
}

/*═══════════════════════════════════════════════════════════════════════════
 * API - DIAGNOSTICS
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Print current state
 */
static inline void rls_baseline_print(const RLS_Baseline *rls)
{
    printf("RLS Baseline:\n");
    printf("  μ = %.4f (baseline)\n", rls->mu);
    printf("  σ = %.4f (std dev)\n", rls->sigma);
    printf("  P = %.6f (uncertainty)\n", rls->P);
    printf("  K = %.4f (Kalman gain)\n", rls->last_K);
    printf("  Q = %.2e, R = %.4f\n", rls->Q, rls->R);
    printf("  Ticks: %d, Warmed up: %s\n", 
           rls->ticks, rls->warmed_up ? "YES" : "NO");
}

#ifdef __cplusplus
}
#endif

#endif /* RLS_BASELINE_H */