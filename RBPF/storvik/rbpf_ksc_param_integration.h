/**
 * @file rbpf_ksc_param_integration.h
 * @brief RBPF-KSC Extended: Full Integration Layer
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * ARCHITECTURE
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * RBPF_Extended wraps RBPF_KSC with:
 *   1. Storvik online parameter learning (μ_vol, σ_vol per regime)
 *   2. Hawkes self-excitation (jump-sensitive transitions)
 *   3. Robust OCSN (11th mixture component for outliers)
 *   4. PARIS smoothed Storvik (fixed-lag backward smoother)
 *   5. Adaptive forgetting (regime-aware λ with circuit breaker)
 *   6. Transition learning (online Dirichlet updates)
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * FILE ORGANIZATION
 * ═══════════════════════════════════════════════════════════════════════════
 *
 *   rbpf_ksc_param_integration.c      Core lifecycle + step function
 *   rbpf_ext_hawkes.c                 Hawkes + Robust OCSN + Presets
 *   rbpf_ext_smoothed_storvik.c       PARIS fixed-lag smoother
 *
 * ═══════════════════════════════════════════════════════════════════════════
 */

#ifndef RBPF_KSC_PARAM_INTEGRATION_H
#define RBPF_KSC_PARAM_INTEGRATION_H

#include "rbpf_ksc.h"
#include "rbpf_param_learn.h"
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

#define RBPF_MAX_REGIMES 8

    /*═══════════════════════════════════════════════════════════════════════════
     * ENUMS
     *═══════════════════════════════════════════════════════════════════════════*/

    typedef enum
    {
        RBPF_PARAM_DISABLED = 0,
        RBPF_PARAM_LIU_WEST,
        RBPF_PARAM_STORVIK,
        RBPF_PARAM_HYBRID
    } RBPF_ParamMode;

    typedef enum
    {
        RBPF_PRESET_CUSTOM = 0,
        RBPF_PRESET_EQUITY_INDEX,
        RBPF_PRESET_SINGLE_STOCK,
        RBPF_PRESET_FX_G10,
        RBPF_PRESET_FX_EM,
        RBPF_PRESET_CRYPTO,
        RBPF_PRESET_COMMODITIES,
        RBPF_PRESET_BONDS
    } RBPF_AssetPreset;

    /*═══════════════════════════════════════════════════════════════════════════
     * HAWKES STATE
     *═══════════════════════════════════════════════════════════════════════════*/

    typedef struct
    {
        int enabled;

        /* Core parameters */
        rbpf_real_t mu;        /* Baseline intensity */
        rbpf_real_t alpha;     /* Jump magnitude */
        rbpf_real_t beta;      /* Decay rate */
        rbpf_real_t threshold; /* |return| threshold for excitation */

        /* State */
        rbpf_real_t intensity;      /* Current λ(t) */
        rbpf_real_t intensity_prev; /* Previous for hysteresis */

        /* Transition modification */
        rbpf_real_t boost_scale; /* How much to boost transitions */
        rbpf_real_t boost_cap;   /* Maximum boost */
        int lut_dirty;           /* 1 if LUT was modified */

        /* Adaptive decay (regime-dependent β) */
        int adaptive_beta_enabled;
        rbpf_real_t beta_regime_scale[RBPF_MAX_REGIMES];

    } RBPF_HawkesState;

    /*═══════════════════════════════════════════════════════════════════════════
     * NOTE: RBPF_OutlierParams and RBPF_RobustOCSN are defined in rbpf_ksc.h
     *═══════════════════════════════════════════════════════════════════════════*/

    /*═══════════════════════════════════════════════════════════════════════════
     * FORWARD DECLARATION (full struct in rbpf_fixed_lag_smoother.h)
     *═══════════════════════════════════════════════════════════════════════════*/

    struct RBPF_FixedLagSmoother;

    /*═══════════════════════════════════════════════════════════════════════════
     * ADAPTIVE SIGNAL SOURCE
     *═══════════════════════════════════════════════════════════════════════════*/

    typedef enum
    {
        ADAPT_SIGNAL_REGIME = 0,          /* Regime-only (baseline λ) */
        ADAPT_SIGNAL_OUTLIER_FRAC,        /* Outlier fraction only */
        ADAPT_SIGNAL_PREDICTIVE_SURPRISE, /* Z-score only */
        ADAPT_SIGNAL_COMBINED             /* Max of outlier + surprise (recommended) */
    } RBPF_AdaptSignal;

    /*═══════════════════════════════════════════════════════════════════════════
     * ADAPTIVE FORGETTING STATE
     *═══════════════════════════════════════════════════════════════════════════*/

    typedef struct
    {
        int enabled;
        RBPF_AdaptSignal signal_source;

        /* Regime baselines */
        rbpf_real_t lambda_per_regime[RBPF_MAX_REGIMES];

        /* Surprise tracking */
        rbpf_real_t surprise_baseline;
        rbpf_real_t surprise_var;
        rbpf_real_t surprise_ema_alpha;
        rbpf_real_t signal_ema;
        rbpf_real_t signal_ema_alpha;

        /* Sigmoid response */
        rbpf_real_t sigmoid_center;
        rbpf_real_t sigmoid_steepness;
        rbpf_real_t max_discount;

        /* Bounds */
        rbpf_real_t lambda_floor;
        rbpf_real_t lambda_ceiling;

        /* Cooldown */
        int cooldown_ticks;
        int cooldown_remaining;

        /* Circuit breaker (P² quantile) */
        int enable_circuit_breaker;
        double trigger_percentile;
        int min_ticks_for_lambda;
        int warmup_ticks;
        uint64_t ticks_since_last_break;
        P2Quantile surprise_quantile;
        uint64_t circuit_breaker_trips;
        int structural_break_detected;
        rbpf_real_t last_trigger_percentile_value;
        rbpf_real_t emergency_lambda_used;

        /* Output */
        rbpf_real_t lambda_current;
        rbpf_real_t surprise_current;
        rbpf_real_t surprise_zscore;
        rbpf_real_t discount_applied;

        /* Statistics */
        uint64_t interventions;
        rbpf_real_t max_surprise_seen;

    } RBPF_AdaptiveForgetting;

    /*═══════════════════════════════════════════════════════════════════════════
     * RBPF_Extended: MAIN STRUCTURE
     *═══════════════════════════════════════════════════════════════════════════*/

    typedef struct RBPF_Extended
    {

        /*───────────────────────────────────────────────────────────────────────
         * CORE RBPF
         *───────────────────────────────────────────────────────────────────────*/
        RBPF_KSC *rbpf;
        RBPF_ParamMode param_mode;

        /*───────────────────────────────────────────────────────────────────────
         * STORVIK PARAMETER LEARNING
         *───────────────────────────────────────────────────────────────────────*/
        ParamLearner storvik;
        int storvik_initialized;

        /*───────────────────────────────────────────────────────────────────────
         * PARTICLE INFO WORKSPACE
         *───────────────────────────────────────────────────────────────────────*/
        ParticleInfo *particle_info; /* [N] current tick info */
        rbpf_real_t *ell_lag_buffer; /* [N] previous tick ℓ values */
        int *prev_regime;            /* [N] previous tick regimes */

        /*───────────────────────────────────────────────────────────────────────
         * HAWKES SELF-EXCITATION
         *───────────────────────────────────────────────────────────────────────*/
        RBPF_HawkesState hawkes;
        rbpf_real_t base_trans_matrix[RBPF_MAX_REGIMES * RBPF_MAX_REGIMES];
        rbpf_real_t last_hawkes_intensity;

        /*───────────────────────────────────────────────────────────────────────
         * ROBUST OCSN (11th Component)
         *───────────────────────────────────────────────────────────────────────*/
        RBPF_RobustOCSN robust_ocsn;
        rbpf_real_t last_outlier_fraction;

        /*───────────────────────────────────────────────────────────────────────
         * ADAPTIVE FORGETTING
         *───────────────────────────────────────────────────────────────────────*/
        RBPF_AdaptiveForgetting adaptive_forgetting;

        /*───────────────────────────────────────────────────────────────────────
         * TRANSITION LEARNING (Online Dirichlet)
         *───────────────────────────────────────────────────────────────────────*/
        int trans_learn_enabled;
        double trans_counts[RBPF_MAX_REGIMES][RBPF_MAX_REGIMES];
        double trans_forgetting;
        double trans_prior_diag;
        double trans_prior_off;
        int trans_update_interval;
        int trans_ticks_since_update;

        /*───────────────────────────────────────────────────────────────────────
         * SMOOTHED STORVIK (PARIS Fixed-Lag)
         *
         * When enabled, Storvik receives smoothed (ℓ̃, ℓ̃_lag) from PARIS
         * instead of filtered values. Reduces parameter oscillation.
         *
         * Architecture:
         *   - Forward pass: RBPF gives IMMEDIATE vol_mean for trading
         *   - Backward pass: PARIS smooths L-tick window for Storvik
         *───────────────────────────────────────────────────────────────────────*/
        int smoothed_storvik_enabled;           /* 0 = filtered, 1 = smoothed */
        int smoothed_storvik_lag;               /* L = smoothing lag (default: 50) */
        struct RBPF_FixedLagSmoother *smoother; /* NULL when disabled */

        /* Cooldown state (prevents flush cascade during volatility waterfall) */
        int cooldown_remaining;   /* Ticks until next flush allowed */
        int min_buffer_for_flush; /* Min ticks before flush (default: 10) */

        /* ESS collapse threshold for buffer reset */
        float ess_collapse_threshold; /* Default: N/20 */

        /* Diagnostics */
        uint64_t flush_count; /* Emergency flushes (P² triggered) */
        uint64_t reset_count; /* ESS-collapse resets */

        /*───────────────────────────────────────────────────────────────────────
         * POLICY ENGINE STATE
         *
         * The Extended layer owns the decision logic for regime change detection.
         * Core layer (KSC) only computes raw signals; this layer interprets them.
         *───────────────────────────────────────────────────────────────────────*/
        int prev_sprt_regime; /* Previous SPRT-confirmed regime */

        /*───────────────────────────────────────────────────────────────────────
         * MISC STATE
         *───────────────────────────────────────────────────────────────────────*/
        int structural_break_signaled;
        RBPF_AssetPreset current_preset;
        uint64_t tick_count;

    } RBPF_Extended;

    /*═══════════════════════════════════════════════════════════════════════════
     * SECTION 1: CORE LIFECYCLE (rbpf_ksc_param_integration.c)
     *═══════════════════════════════════════════════════════════════════════════*/

    /**
     * Create RBPF_Extended instance
     */
    RBPF_Extended *rbpf_ext_create(int n_particles, int n_regimes, RBPF_ParamMode mode);

    /**
     * Destroy RBPF_Extended instance
     */
    void rbpf_ext_destroy(RBPF_Extended *ext);

    /**
     * Initialize filter state
     */
    void rbpf_ext_init(RBPF_Extended *ext, rbpf_real_t mu0, rbpf_real_t var0);

    /**
     * Main step function - processes one observation
     */
    void rbpf_ext_step(RBPF_Extended *ext, rbpf_real_t obs, RBPF_KSC_Output *output);

    /**
     * APF step (lookahead) - currently disabled, falls back to standard
     */
    void rbpf_ext_step_apf(RBPF_Extended *ext, rbpf_real_t obs_current,
                           rbpf_real_t obs_next, RBPF_KSC_Output *output);

    /*═══════════════════════════════════════════════════════════════════════════
     * SECTION 2: BASIC CONFIGURATION (rbpf_ksc_param_integration.c)
     *═══════════════════════════════════════════════════════════════════════════*/

    /**
     * Set regime parameters (θ, μ_vol, σ_vol)
     */
    void rbpf_ext_set_regime_params(RBPF_Extended *ext, int regime,
                                    rbpf_real_t theta, rbpf_real_t mu_vol,
                                    rbpf_real_t sigma_vol);

    /**
     * Build transition LUT from probability matrix
     */
    void rbpf_ext_build_transition_lut(RBPF_Extended *ext, const rbpf_real_t *trans_matrix);

    /**
     * Set Storvik sampling interval for regime
     */
    void rbpf_ext_set_storvik_interval(RBPF_Extended *ext, int regime, int interval);

    /**
     * Enable HFT mode (regime-adaptive sampling)
     */
    void rbpf_ext_set_hft_mode(RBPF_Extended *ext, int enable);

    /**
     * Enable full update mode (every tick, all regimes)
     */
    void rbpf_ext_set_full_update_mode(RBPF_Extended *ext);

    /**
     * Signal structural break (triggers circuit breaker)
     */
    void rbpf_ext_signal_structural_break(RBPF_Extended *ext);

    /**
     * Check if structural break was detected (circuit breaker tripped)
     * Implementation in rbpf_adaptive_forgetting.c
     */
    int rbpf_ext_structural_break_detected(const RBPF_Extended *ext);

    /*═══════════════════════════════════════════════════════════════════════════
     * SECTION 3: TRANSITION LEARNING (rbpf_ksc_param_integration.c)
     *═══════════════════════════════════════════════════════════════════════════*/

    void rbpf_ext_enable_transition_learning(RBPF_Extended *ext, int enable);
    void rbpf_ext_configure_transition_learning(RBPF_Extended *ext,
                                                double forgetting,
                                                double prior_diag,
                                                double prior_off,
                                                int update_interval);
    void rbpf_ext_reset_transition_counts(RBPF_Extended *ext);
    double rbpf_ext_get_transition_prob(const RBPF_Extended *ext, int from, int to);

    /*═══════════════════════════════════════════════════════════════════════════
     * SECTION 4: PARAMETER ACCESS (rbpf_ksc_param_integration.c)
     *═══════════════════════════════════════════════════════════════════════════*/

    void rbpf_ext_get_learned_params(const RBPF_Extended *ext, int regime,
                                     rbpf_real_t *mu_vol, rbpf_real_t *sigma_vol);
    void rbpf_ext_get_storvik_summary(const RBPF_Extended *ext, int regime,
                                      RegimeParams *summary);
    void rbpf_ext_get_learning_stats(const RBPF_Extended *ext,
                                     uint64_t *stat_updates,
                                     uint64_t *samples_drawn,
                                     uint64_t *samples_skipped);

    /*═══════════════════════════════════════════════════════════════════════════
     * SECTION 5: DIAGNOSTICS (rbpf_ksc_param_integration.c)
     *═══════════════════════════════════════════════════════════════════════════*/

    void rbpf_ext_print_config(const RBPF_Extended *ext);
    void rbpf_ext_print_storvik_stats(const RBPF_Extended *ext, int regime);

    /*═══════════════════════════════════════════════════════════════════════════
     * SECTION 6: HAWKES SELF-EXCITATION (rbpf_ext_hawkes.c)
     *═══════════════════════════════════════════════════════════════════════════*/

    void rbpf_ext_enable_hawkes(RBPF_Extended *ext,
                                rbpf_real_t mu, rbpf_real_t alpha,
                                rbpf_real_t beta, rbpf_real_t threshold);
    void rbpf_ext_disable_hawkes(RBPF_Extended *ext);
    void rbpf_ext_set_hawkes_boost(RBPF_Extended *ext,
                                   rbpf_real_t boost_scale, rbpf_real_t boost_cap);
    void rbpf_ext_enable_adaptive_hawkes(RBPF_Extended *ext, int enable);
    void rbpf_ext_set_hawkes_regime_scale(RBPF_Extended *ext, int regime, rbpf_real_t scale);
    rbpf_real_t rbpf_ext_get_hawkes_intensity(const RBPF_Extended *ext);

    /* Internal Hawkes functions (called from rbpf_ext_step) */
    void rbpf_ext_hawkes_apply_to_transitions(RBPF_Extended *ext);
    void rbpf_ext_hawkes_update_intensity(RBPF_Extended *ext, rbpf_real_t obs);
    void rbpf_ext_hawkes_restore_base_transitions(RBPF_Extended *ext);

    /*═══════════════════════════════════════════════════════════════════════════
     * SECTION 7: ROBUST OCSN (rbpf_ext_hawkes.c)
     *═══════════════════════════════════════════════════════════════════════════*/

    void rbpf_ext_enable_robust_ocsn(RBPF_Extended *ext);
    void rbpf_ext_enable_robust_ocsn_simple(RBPF_Extended *ext,
                                            rbpf_real_t prob, rbpf_real_t variance);
    void rbpf_ext_set_outlier_params(RBPF_Extended *ext, int regime,
                                     rbpf_real_t prob, rbpf_real_t variance);
    void rbpf_ext_disable_robust_ocsn(RBPF_Extended *ext);
    rbpf_real_t rbpf_ext_get_outlier_fraction(const RBPF_Extended *ext);

    /*═══════════════════════════════════════════════════════════════════════════
     * SECTION 8: ASSET PRESETS (rbpf_ext_hawkes.c)
     *═══════════════════════════════════════════════════════════════════════════*/

    void rbpf_ext_apply_preset(RBPF_Extended *ext, RBPF_AssetPreset preset);
    RBPF_AssetPreset rbpf_ext_get_preset(const RBPF_Extended *ext);

    /*═══════════════════════════════════════════════════════════════════════════
     * SECTION 9: SMOOTHED STORVIK (rbpf_ext_smoothed_storvik.c)
     *═══════════════════════════════════════════════════════════════════════════*/

    /**
     * Enable PARIS-smoothed Storvik parameter learning
     *
     * Replaces filtered (ℓ, ℓ_lag) with smoothed (ℓ̃, ℓ̃_lag) in Storvik updates.
     * Trading signal (vol_mean) remains IMMEDIATE - no delay.
     * Only parameter learning gets L-tick smoothed values.
     *
     * @param ext   RBPF_Extended handle
     * @param lag   Smoothing lag L (recommended: 50 for HFT)
     * @return      0 on success, -1 on failure
     */
    int rbpf_ext_enable_smoothed_storvik(RBPF_Extended *ext, int lag);

    /**
     * Disable PARIS-smoothed Storvik (revert to filtered baseline)
     */
    void rbpf_ext_disable_smoothed_storvik(RBPF_Extended *ext);

    /**
     * Check if smoothed Storvik is enabled
     */
    int rbpf_ext_is_smoothed_storvik_enabled(const RBPF_Extended *ext);

    /**
     * Get smoothed Storvik diagnostics
     */
    void rbpf_ext_get_smoother_stats(const RBPF_Extended *ext,
                                     uint64_t *flush_count,
                                     uint64_t *reset_count,
                                     double *avg_smooth_us,
                                     int *buffer_fill);

    /**
     * Configure smoothed Storvik parameters
     */
    void rbpf_ext_configure_smoother(RBPF_Extended *ext,
                                     int min_buffer_for_flush,
                                     float ess_collapse_thresh);

    /**
     * Internal: Process smoother step (called from rbpf_ext_step)
     */
    void rbpf_ext_smoother_step(RBPF_Extended *ext, const RBPF_KSC_Output *output);

    /**
     * Print smoother configuration
     */
    void rbpf_ext_print_smoother_config(const RBPF_Extended *ext);

    /*═══════════════════════════════════════════════════════════════════════════
     * SECTION 10: ADAPTIVE FORGETTING (rbpf_adaptive_forgetting.c)
     *═══════════════════════════════════════════════════════════════════════════*/

    void rbpf_adaptive_forgetting_init(RBPF_AdaptiveForgetting *af);
    void rbpf_adaptive_forgetting_update(RBPF_Extended *ext, rbpf_real_t marginal_lik, int dominant_regime);

    void rbpf_ext_enable_adaptive_forgetting(RBPF_Extended *ext);
    void rbpf_ext_enable_adaptive_forgetting_mode(RBPF_Extended *ext, RBPF_AdaptSignal signal);
    void rbpf_ext_disable_adaptive_forgetting(RBPF_Extended *ext);
    void rbpf_ext_set_regime_lambda(RBPF_Extended *ext, int regime, rbpf_real_t lambda);
    void rbpf_ext_set_adaptive_sigmoid(RBPF_Extended *ext,
                                       rbpf_real_t center,
                                       rbpf_real_t steepness,
                                       rbpf_real_t max_discount);
    void rbpf_ext_set_adaptive_bounds(RBPF_Extended *ext,
                                      rbpf_real_t floor,
                                      rbpf_real_t ceiling);
    void rbpf_ext_set_adaptive_smoothing(RBPF_Extended *ext,
                                         rbpf_real_t baseline_alpha,
                                         rbpf_real_t signal_alpha);
    void rbpf_ext_set_adaptive_cooldown(RBPF_Extended *ext, int ticks);

    void rbpf_ext_enable_circuit_breaker(RBPF_Extended *ext, double quantile, int window);
    void rbpf_ext_disable_circuit_breaker(RBPF_Extended *ext);
    void rbpf_ext_set_circuit_breaker_min_memory(RBPF_Extended *ext, int min_ticks);
    uint64_t rbpf_ext_get_circuit_breaker_trips(const RBPF_Extended *ext);
    rbpf_real_t rbpf_ext_get_circuit_breaker_threshold(const RBPF_Extended *ext);
    rbpf_real_t rbpf_ext_get_last_emergency_lambda(const RBPF_Extended *ext);

    rbpf_real_t rbpf_ext_get_current_lambda(const RBPF_Extended *ext);
    rbpf_real_t rbpf_ext_get_surprise_zscore(const RBPF_Extended *ext);
    void rbpf_ext_get_adaptive_stats(const RBPF_Extended *ext,
                                     uint64_t *interventions,
                                     rbpf_real_t *current_lambda,
                                     rbpf_real_t *max_surprise);
    void rbpf_ext_print_adaptive_config(const RBPF_Extended *ext);

#ifdef __cplusplus
}
#endif

#endif /* RBPF_KSC_PARAM_INTEGRATION_H */