/**
 * @file rbpf_ext_types.h
 * @brief RBPF-KSC Extended: Type Definitions
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * This file contains:
 *   - Constants and limits
 *   - Enumerations (param mode, asset presets, signal sources)
 *   - RBPF_AdaptiveForgetting struct
 *   - RBPF_Extended struct (main integration layer)
 *
 * For function declarations, see rbpf_ext_api.h
 * For backward compatibility, include rbpf_ksc_param_integration.h
 * ═══════════════════════════════════════════════════════════════════════════
 */

#ifndef RBPF_EXT_TYPES_H
#define RBPF_EXT_TYPES_H

#include "rbpf_ksc.h"
#include "rbpf_param_learn.h"
#include "hawkes_integrator.h"
#include "p2_quantile.h"

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C"
{
#endif

    /*═══════════════════════════════════════════════════════════════════════════
     * CONSTANTS
     *═══════════════════════════════════════════════════════════════════════════*/

/** Maximum number of regimes supported */
#define RBPF_MAX_REGIMES 8

    /*═══════════════════════════════════════════════════════════════════════════
     * ENUMERATIONS
     *═══════════════════════════════════════════════════════════════════════════*/

    /**
     * @brief Parameter learning mode
     *
     * Determines how regime parameters (μ_vol, σ_vol) are learned/updated.
     */
    typedef enum
    {
        RBPF_PARAM_DISABLED = 0, /**< Fixed parameters, no learning */
        RBPF_PARAM_LIU_WEST,     /**< Liu-West kernel smoothing (legacy) */
        RBPF_PARAM_STORVIK,      /**< Storvik sufficient statistics (recommended) */
        RBPF_PARAM_HYBRID        /**< Storvik + Liu-West fallback */
    } RBPF_ParamMode;

    /**
     * @brief Asset class presets
     *
     * Pre-configured parameter sets optimized for different asset classes.
     * Use rbpf_ext_apply_preset() to apply.
     */
    typedef enum
    {
        RBPF_PRESET_CUSTOM = 0,  /**< User-defined parameters */
        RBPF_PRESET_EQUITY_INDEX, /**< S&P 500, NASDAQ, etc. */
        RBPF_PRESET_SINGLE_STOCK, /**< Individual equities */
        RBPF_PRESET_FX_G10,       /**< Major currency pairs */
        RBPF_PRESET_FX_EM,        /**< Emerging market FX */
        RBPF_PRESET_CRYPTO,       /**< BTC, ETH, etc. */
        RBPF_PRESET_COMMODITIES,  /**< Gold, oil, etc. */
        RBPF_PRESET_BONDS         /**< Fixed income */
    } RBPF_AssetPreset;

    /**
     * @brief Adaptive forgetting signal source
     *
     * Determines what drives the adaptive forgetting factor λ.
     */
    typedef enum
    {
        ADAPT_SIGNAL_REGIME = 0,          /**< Regime-only (use baseline λ per regime) */
        ADAPT_SIGNAL_OUTLIER_FRAC,        /**< Scale λ by OCSN outlier fraction */
        ADAPT_SIGNAL_PREDICTIVE_SURPRISE, /**< Scale λ by predictive surprise z-score */
        ADAPT_SIGNAL_COMBINED             /**< Max of outlier + surprise (recommended) */
    } RBPF_AdaptSignal;

    /*═══════════════════════════════════════════════════════════════════════════
     * FORWARD DECLARATIONS
     *═══════════════════════════════════════════════════════════════════════════*/

    /** Fixed-lag smoother for PARIS algorithm (see rbpf_fixed_lag_smoother.h) */
    struct RBPF_FixedLagSmoother;

    /** KL tempering state (see rbpf_kl_tempering.h) */
    struct RBPF_KL_State;

    /*═══════════════════════════════════════════════════════════════════════════
     * ADAPTIVE FORGETTING STATE
     *
     * Manages dynamic adjustment of Storvik forgetting factor λ based on
     * market regime and tail events. Includes P² circuit breaker for
     * structural break detection.
     *═══════════════════════════════════════════════════════════════════════════*/

    /**
     * @brief Adaptive forgetting controller state
     *
     * The forgetting factor λ controls how quickly Storvik "forgets" old
     * observations. This struct manages dynamic λ adjustment:
     *
     * - Per-regime baseline λ (calm regimes forget slower)
     * - Surprise-based scaling (high surprise → faster forgetting)
     * - P² circuit breaker (tail events trigger emergency λ)
     *
     * Typical values:
     *   - λ = 0.999: ~1000 tick memory (calm markets)
     *   - λ = 0.99:  ~100 tick memory (volatile markets)
     *   - λ = 0.95:  ~20 tick memory (crisis/structural break)
     */
    typedef struct
    {
        /*─────────────────────────────────────────────────────────────────────
         * ENABLE/MODE
         *─────────────────────────────────────────────────────────────────────*/
        int enabled;                 /**< 0 = use fixed λ, 1 = adaptive */
        RBPF_AdaptSignal signal_source; /**< What drives adaptation */

        /*─────────────────────────────────────────────────────────────────────
         * PER-REGIME BASELINES
         *
         * Each regime has its own baseline λ. High-vol regimes typically
         * use lower λ (faster forgetting) since dynamics change quickly.
         *─────────────────────────────────────────────────────────────────────*/
        rbpf_real_t lambda_per_regime[RBPF_MAX_REGIMES];

        /*─────────────────────────────────────────────────────────────────────
         * SURPRISE TRACKING
         *
         * EMA of predictive surprise for z-score computation.
         * Z-score drives sigmoid response.
         *─────────────────────────────────────────────────────────────────────*/
        rbpf_real_t surprise_baseline;  /**< EMA of |surprise| */
        rbpf_real_t surprise_var;       /**< EMA of surprise² for variance */
        rbpf_real_t surprise_ema_alpha; /**< EMA decay (default: 0.01) */
        rbpf_real_t signal_ema;         /**< Smoothed adaptation signal */
        rbpf_real_t signal_ema_alpha;   /**< Signal smoothing (default: 0.1) */

        /*─────────────────────────────────────────────────────────────────────
         * SIGMOID RESPONSE
         *
         * Maps z-score to λ discount: λ_eff = λ_base * (1 - discount)
         * discount = max_discount * sigmoid((z - center) * steepness)
         *─────────────────────────────────────────────────────────────────────*/
        rbpf_real_t sigmoid_center;    /**< Z-score threshold (default: 2.0) */
        rbpf_real_t sigmoid_steepness; /**< Sigmoid slope (default: 1.0) */
        rbpf_real_t max_discount;      /**< Max λ reduction (default: 0.05) */

        /*─────────────────────────────────────────────────────────────────────
         * BOUNDS
         *─────────────────────────────────────────────────────────────────────*/
        rbpf_real_t lambda_floor;   /**< Minimum λ (default: 0.95) */
        rbpf_real_t lambda_ceiling; /**< Maximum λ (default: 0.9999) */

        /*─────────────────────────────────────────────────────────────────────
         * COOLDOWN
         *
         * Prevents λ oscillation after intervention.
         *─────────────────────────────────────────────────────────────────────*/
        int cooldown_ticks;     /**< Ticks to wait after intervention */
        int cooldown_remaining; /**< Current cooldown counter */

        /*─────────────────────────────────────────────────────────────────────
         * P² CIRCUIT BREAKER
         *
         * Uses P² algorithm to track surprise quantiles online.
         * When surprise exceeds trigger_percentile, signals structural break
         * and applies emergency_lambda.
         *─────────────────────────────────────────────────────────────────────*/
        int enable_circuit_breaker;            /**< 0 = disabled */
        double trigger_percentile;             /**< e.g., 0.999 for 99.9th */
        int min_ticks_for_lambda;              /**< Min ticks before triggering */
        int warmup_ticks;                      /**< Warmup period for P² */
        uint64_t ticks_since_last_break;       /**< Ticks since last trigger */
        P2Quantile surprise_quantile;          /**< P² quantile tracker */
        uint64_t circuit_breaker_trips;        /**< Total trip count */
        int structural_break_detected;         /**< Flag for current tick */
        rbpf_real_t last_trigger_percentile_value; /**< Threshold that triggered */
        rbpf_real_t emergency_lambda_used;     /**< λ applied during break */

        /*─────────────────────────────────────────────────────────────────────
         * OUTPUT (computed each tick)
         *─────────────────────────────────────────────────────────────────────*/
        rbpf_real_t lambda_current;   /**< Current effective λ */
        rbpf_real_t surprise_current; /**< Raw surprise this tick */
        rbpf_real_t surprise_zscore;  /**< Z-score this tick */
        rbpf_real_t discount_applied; /**< Discount applied this tick */

        /*─────────────────────────────────────────────────────────────────────
         * LAMBDA OVERRIDE/RESTORE STATE
         *
         * When circuit breaker fires, all λ sources are overridden with
         * emergency_lambda. These fields track original values for
         * gradual restoration after crisis passes.
         *─────────────────────────────────────────────────────────────────────*/
        int lambda_override_active;                    /**< Override in effect */
        rbpf_real_t saved_lambda_global;               /**< Saved global λ */
        rbpf_real_t saved_lambda_regime[RBPF_MAX_REGIMES]; /**< Saved per-regime */
        int restore_blend_ticks;                       /**< Blend duration */
        int restore_ticks_elapsed;                     /**< Blend progress */

        /*─────────────────────────────────────────────────────────────────────
         * STATISTICS
         *─────────────────────────────────────────────────────────────────────*/
        uint64_t interventions;      /**< Total λ adjustments */
        rbpf_real_t max_surprise_seen; /**< Largest surprise observed */

    } RBPF_AdaptiveForgetting;

    /*═══════════════════════════════════════════════════════════════════════════
     * RBPF_Extended: MAIN INTEGRATION STRUCTURE
     *
     * Wraps RBPF_KSC with:
     *   1. Storvik online parameter learning (μ_vol, σ_vol per regime)
     *   2. Hawkes self-excitation (jump-sensitive transitions via APF kick)
     *   3. Robust OCSN (11th mixture component for outliers)
     *   4. PARIS smoothed Storvik (fixed-lag backward smoother)
     *   5. Adaptive forgetting (regime-aware λ with P² circuit breaker)
     *   6. Transition learning (online Dirichlet updates)
     *   7. KL tempering (information-geometric weight normalization)
     *═══════════════════════════════════════════════════════════════════════════*/

    typedef struct RBPF_Extended
    {
        /*─────────────────────────────────────────────────────────────────────
         * CORE RBPF
         *─────────────────────────────────────────────────────────────────────*/
        RBPF_KSC *rbpf;          /**< Core particle filter */
        RBPF_ParamMode param_mode; /**< Parameter learning mode */

        /*─────────────────────────────────────────────────────────────────────
         * STORVIK PARAMETER LEARNING
         *
         * Learns regime parameters (μ_vol, σ_vol) online using sufficient
         * statistics. Updates are per-particle, per-regime.
         *─────────────────────────────────────────────────────────────────────*/
        ParamLearner storvik;      /**< Storvik learner state */
        int storvik_initialized;   /**< 1 if storvik is ready */

        /*─────────────────────────────────────────────────────────────────────
         * PARTICLE INFO WORKSPACE
         *
         * Scratch space for extracting particle state for Storvik updates.
         *─────────────────────────────────────────────────────────────────────*/
        ParticleInfo *particle_info; /**< [N] current tick info */
        rbpf_real_t *ell_lag_buffer; /**< [N] previous tick ℓ values */
        int *prev_regime;            /**< [N] previous tick regimes */

        /*─────────────────────────────────────────────────────────────────────
         * HAWKES / APF KICK
         *
         * Hawkes process detects sustained volatility clustering.
         * APF kick uses observation likelihood to weight regime transitions
         * for faster crisis detection (auxiliary particle filter style).
         *─────────────────────────────────────────────────────────────────────*/
        HawkesIntegrator hawkes_integrator; /**< Hawkes intensity tracker */
        int apf_kick_enabled;               /**< 1 = use APF during elevated intensity */
        float apf_surprise_threshold;       /**< Surprise σ threshold for APF */
        rbpf_real_t base_trans_matrix[RBPF_MAX_REGIMES * RBPF_MAX_REGIMES]; /**< Base Π */
        rbpf_real_t last_hawkes_intensity;  /**< Last computed intensity */

        /*─────────────────────────────────────────────────────────────────────
         * ROBUST OCSN (11th Component)
         *
         * Adds heavy-tailed outlier component to KSC mixture for robustness.
         *─────────────────────────────────────────────────────────────────────*/
        RBPF_RobustOCSN robust_ocsn;        /**< Outlier component config */
        rbpf_real_t last_outlier_fraction;  /**< Fraction assigned to outlier */

        /*─────────────────────────────────────────────────────────────────────
         * ADAPTIVE FORGETTING
         *─────────────────────────────────────────────────────────────────────*/
        RBPF_AdaptiveForgetting adaptive_forgetting;

        /*─────────────────────────────────────────────────────────────────────
         * TRANSITION LEARNING (Online Dirichlet)
         *
         * Learns transition matrix Π online from observed regime switches.
         *─────────────────────────────────────────────────────────────────────*/
        int trans_learn_enabled;    /**< 1 = learning enabled */
        double trans_counts[RBPF_MAX_REGIMES][RBPF_MAX_REGIMES]; /**< Pseudo-counts */
        double trans_forgetting;    /**< Count decay factor */
        double trans_prior_diag;    /**< Prior for staying (diagonal) */
        double trans_prior_off;     /**< Prior for switching (off-diagonal) */
        int trans_update_interval;  /**< Ticks between LUT rebuilds */
        int trans_ticks_since_update; /**< Counter */

        /*─────────────────────────────────────────────────────────────────────
         * SMOOTHED STORVIK (PARIS Fixed-Lag)
         *
         * When enabled, Storvik receives smoothed (ℓ̃, ℓ̃_lag) from PARIS
         * backward pass instead of filtered values. Reduces parameter
         * oscillation while maintaining immediate vol_mean for trading.
         *─────────────────────────────────────────────────────────────────────*/
        int smoothed_storvik_enabled;           /**< 0 = filtered, 1 = smoothed */
        int smoothed_storvik_lag;               /**< L = smoothing lag */
        struct RBPF_FixedLagSmoother *smoother; /**< PARIS smoother */
        int cooldown_remaining;                 /**< Flush cooldown */
        int min_buffer_for_flush;               /**< Min ticks before flush */
        float ess_collapse_threshold;           /**< ESS threshold for reset */
        uint64_t flush_count;                   /**< Emergency flush count */
        uint64_t reset_count;                   /**< ESS-collapse resets */

        /*─────────────────────────────────────────────────────────────────────
         * POLICY ENGINE STATE
         *
         * Tracks regime change detection state for SPRT and P² detectors.
         *─────────────────────────────────────────────────────────────────────*/
        int prev_sprt_regime; /**< Previous SPRT-confirmed regime */

        /*─────────────────────────────────────────────────────────────────────
         * KL TEMPERING
         *
         * Prevents "numerical genocide" of particles by limiting how much
         * a single observation can change the weight distribution.
         * KL divergence is clamped to log(N) nats per tick.
         *─────────────────────────────────────────────────────────────────────*/
        struct RBPF_KL_State *kl_state; /**< KL tempering state */
        int kl_tempering_enabled;       /**< 1 = use KL tempering */
        int last_resampled;             /**< Did last tick resample? */

        /*─────────────────────────────────────────────────────────────────────
         * MISC STATE
         *─────────────────────────────────────────────────────────────────────*/
        int structural_break_signaled;  /**< Pending structural break */
        RBPF_AssetPreset current_preset; /**< Active preset */
        uint64_t tick_count;            /**< Total ticks processed */

    } RBPF_Extended;

#ifdef __cplusplus
}
#endif

#endif /* RBPF_EXT_TYPES_H */
