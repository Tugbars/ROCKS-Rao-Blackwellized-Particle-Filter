/**
 * @file rbpf_ext_api.h
 * @brief RBPF-KSC Extended: Function Declarations
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * API SECTIONS:
 *   1. Core Lifecycle      - create, destroy, init, step
 *   2. Basic Configuration - regime params, transition LUT, Storvik intervals
 *   3. Transition Learning - online Dirichlet updates
 *   4. Parameter Access    - get learned params, Storvik summary
 *   5. Hawkes / APF Kick   - self-excitation and auxiliary particle filter
 *   6. Robust OCSN         - outlier component configuration
 *   7. Asset Presets       - pre-configured parameter sets
 *   8. Smoothed Storvik    - PARIS fixed-lag smoother
 *   9. Adaptive Forgetting - dynamic λ with P² circuit breaker
 *  10. KL Tempering        - information-geometric weight normalization
 *  11. Diagnostics         - print functions
 *
 * For type definitions, see rbpf_ext_types.h
 * For backward compatibility, include rbpf_ksc_param_integration.h
 * ═══════════════════════════════════════════════════════════════════════════
 */

#ifndef RBPF_EXT_API_H
#define RBPF_EXT_API_H

#include "rbpf_ext_types.h"

#ifdef __cplusplus
extern "C"
{
#endif

    /*═══════════════════════════════════════════════════════════════════════════
     * SECTION 1: CORE LIFECYCLE
     *
     * Implementation: rbpf_ksc_param_integration.c
     *═══════════════════════════════════════════════════════════════════════════*/

    /**
     * @brief Create RBPF_Extended instance
     *
     * Allocates and initializes the extended particle filter with optional
     * Storvik parameter learning.
     *
     * @param n_particles  Number of particles (typically 256-2048)
     * @param n_regimes    Number of volatility regimes (typically 3-4)
     * @param mode         Parameter learning mode (RBPF_PARAM_STORVIK recommended)
     * @return             Allocated instance, or NULL on failure
     *
     * @note Caller must call rbpf_ext_destroy() to free resources.
     */
    RBPF_Extended *rbpf_ext_create(int n_particles, int n_regimes, RBPF_ParamMode mode);

    /**
     * @brief Destroy RBPF_Extended instance
     *
     * Frees all allocated memory including internal RBPF_KSC, Storvik learner,
     * smoother, and workspace buffers.
     *
     * @param ext  Instance to destroy (safe to pass NULL)
     */
    void rbpf_ext_destroy(RBPF_Extended *ext);

    /**
     * @brief Initialize filter state
     *
     * Resets particles to initial log-volatility distribution and broadcasts
     * regime priors to Storvik learner.
     *
     * @param ext   RBPF_Extended instance
     * @param mu0   Initial log-volatility mean (e.g., -4.6 ≈ 1% daily vol)
     * @param var0  Initial log-volatility variance (e.g., 0.5)
     */
    void rbpf_ext_init(RBPF_Extended *ext, rbpf_real_t mu0, rbpf_real_t var0);

    /**
     * @brief Process one observation (main step function)
     *
     * This is the hot path. Performs:
     *   1. Observation transform (y = log(r²))
     *   2. Hawkes update + regime transition (APF kick if enabled)
     *   3. Kalman predict + update
     *   4. KL tempering (if enabled)
     *   5. Compute outputs + resample
     *   6. Adaptive forgetting + policy engine
     *   7. Storvik parameter learning
     *   8. Transition learning
     *
     * @param ext     RBPF_Extended instance
     * @param obs     Raw return observation (not log-squared)
     * @param output  Output struct to populate (caller allocates)
     */
    void rbpf_ext_step(RBPF_Extended *ext, rbpf_real_t obs, RBPF_KSC_Output *output);

    /**
     * @brief APF step with lookahead (currently falls back to standard)
     * @deprecated Use rbpf_ext_step() instead
     */
    void rbpf_ext_step_apf(RBPF_Extended *ext, rbpf_real_t obs_current,
                           rbpf_real_t obs_next, RBPF_KSC_Output *output);

    /*═══════════════════════════════════════════════════════════════════════════
     * SECTION 2: BASIC CONFIGURATION
     *
     * Implementation: rbpf_ext_config.c
     *═══════════════════════════════════════════════════════════════════════════*/

    /**
     * @brief Set regime parameters
     *
     * Configures θ (mean reversion), μ_vol (mean log-vol), σ_vol (vol-of-vol)
     * for a specific regime. Also updates Storvik prior if initialized.
     *
     * @param ext       RBPF_Extended instance
     * @param regime    Regime index [0, n_regimes)
     * @param theta     Mean reversion speed (0 = random walk, 1 = white noise)
     * @param mu_vol    Mean log-volatility (e.g., -9 for calm, -4 for crisis)
     * @param sigma_vol Volatility of volatility (e.g., 0.1 for calm, 0.5 for crisis)
     */
    void rbpf_ext_set_regime_params(RBPF_Extended *ext, int regime,
                                    rbpf_real_t theta, rbpf_real_t mu_vol,
                                    rbpf_real_t sigma_vol);

    /**
     * @brief Build transition LUT from probability matrix
     *
     * Converts n×n transition probability matrix to fast lookup table.
     * Also stores copy in base_trans_matrix for Hawkes restoration.
     *
     * @param ext          RBPF_Extended instance
     * @param trans_matrix Row-major n×n transition probabilities (rows sum to 1)
     */
    void rbpf_ext_build_transition_lut(RBPF_Extended *ext, const rbpf_real_t *trans_matrix);

    /**
     * @brief Set Storvik sampling interval for regime
     *
     * Controls how often Storvik samples new parameters for a regime.
     * Lower intervals = more frequent sampling = faster adaptation.
     *
     * @param ext      RBPF_Extended instance
     * @param regime   Regime index
     * @param interval Ticks between samples (1 = every tick)
     */
    void rbpf_ext_set_storvik_interval(RBPF_Extended *ext, int regime, int interval);

    /**
     * @brief Enable HFT mode
     *
     * Sets regime-adaptive sampling intervals optimized for high-frequency:
     *   R0 (calm): every 100 ticks
     *   R1: every 50 ticks
     *   R2: every 20 ticks
     *   R3 (crisis): every 5 ticks
     *
     * @param ext     RBPF_Extended instance
     * @param enable  1 = HFT intervals, 0 = every tick
     */
    void rbpf_ext_set_hft_mode(RBPF_Extended *ext, int enable);

    /**
     * @brief Enable full update mode
     *
     * Configures Storvik for maximum accuracy (every tick, all regimes).
     * Higher CPU cost but best for research/backtesting.
     */
    void rbpf_ext_set_full_update_mode(RBPF_Extended *ext);

    /**
     * @brief Signal structural break
     *
     * Triggers circuit breaker behavior: resets Storvik to priors and
     * applies emergency forgetting factor.
     *
     * @param ext  RBPF_Extended instance
     */
    void rbpf_ext_signal_structural_break(RBPF_Extended *ext);

    /**
     * @brief Check if structural break was detected
     *
     * Returns 1 if P² circuit breaker fired this tick.
     *
     * @param ext  RBPF_Extended instance
     * @return     1 if break detected, 0 otherwise
     */
    int rbpf_ext_structural_break_detected(const RBPF_Extended *ext);

    /*═══════════════════════════════════════════════════════════════════════════
     * SECTION 3: TRANSITION LEARNING
     *
     * Implementation: rbpf_ext_config.c
     *═══════════════════════════════════════════════════════════════════════════*/

    /**
     * @brief Enable/disable online transition learning
     *
     * When enabled, transition matrix is learned from observed regime switches
     * using Dirichlet-multinomial model with forgetting.
     */
    void rbpf_ext_enable_transition_learning(RBPF_Extended *ext, int enable);

    /**
     * @brief Configure transition learning parameters
     *
     * @param ext             RBPF_Extended instance
     * @param forgetting      Count decay factor (0.995 typical)
     * @param prior_diag      Prior pseudo-count for staying (50.0 typical)
     * @param prior_off       Prior pseudo-count for switching (1.0 typical)
     * @param update_interval Ticks between LUT rebuilds (100 typical)
     */
    void rbpf_ext_configure_transition_learning(RBPF_Extended *ext,
                                                double forgetting,
                                                double prior_diag,
                                                double prior_off,
                                                int update_interval);

    /** @brief Reset transition counts to zero */
    void rbpf_ext_reset_transition_counts(RBPF_Extended *ext);

    /**
     * @brief Get learned transition probability
     *
     * @param ext   RBPF_Extended instance
     * @param from  Source regime
     * @param to    Destination regime
     * @return      Posterior mean P(to | from)
     */
    double rbpf_ext_get_transition_prob(const RBPF_Extended *ext, int from, int to);

    /*═══════════════════════════════════════════════════════════════════════════
     * SECTION 4: PARAMETER ACCESS
     *
     * Implementation: rbpf_ext_diagnostics.c
     *═══════════════════════════════════════════════════════════════════════════*/

    /**
     * @brief Get learned regime parameters
     *
     * Returns current Storvik posterior mean for μ_vol and σ_vol.
     * Falls back to fixed params if Storvik not initialized.
     *
     * @param ext       RBPF_Extended instance
     * @param regime    Regime index
     * @param mu_vol    Output: mean log-volatility (can be NULL)
     * @param sigma_vol Output: vol-of-vol (can be NULL)
     */
    void rbpf_ext_get_learned_params(const RBPF_Extended *ext, int regime,
                                     rbpf_real_t *mu_vol, rbpf_real_t *sigma_vol);

    /**
     * @brief Get full Storvik summary for regime
     *
     * @param ext     RBPF_Extended instance
     * @param regime  Regime index
     * @param summary Output struct (caller allocates)
     */
    void rbpf_ext_get_storvik_summary(const RBPF_Extended *ext, int regime,
                                      RegimeParams *summary);

    /**
     * @brief Get Storvik learning statistics
     *
     * @param ext             RBPF_Extended instance
     * @param stat_updates    Output: total sufficient stat updates
     * @param samples_drawn   Output: total parameter samples drawn
     * @param samples_skipped Output: samples skipped due to load
     */
    void rbpf_ext_get_learning_stats(const RBPF_Extended *ext,
                                     uint64_t *stat_updates,
                                     uint64_t *samples_drawn,
                                     uint64_t *samples_skipped);

    /*═══════════════════════════════════════════════════════════════════════════
     * SECTION 5: HAWKES / APF KICK
     *
     * Implementation: rbpf_ext_config.c (config), rbpf_ext_diagnostics.c (query)
     *═══════════════════════════════════════════════════════════════════════════*/

    /**
     * @brief Enable/disable APF kick
     *
     * When enabled, uses Hawkes intensity to trigger auxiliary particle filter
     * style transition weighting during elevated volatility.
     */
    void rbpf_ext_enable_apf_kick(RBPF_Extended *ext, int enable);

    /** @brief Check if APF kick is enabled */
    int rbpf_ext_apf_kick_enabled(const RBPF_Extended *ext);

    /**
     * @brief Set APF surprise threshold
     *
     * APF kick activates when Hawkes surprise_sigma exceeds this threshold.
     *
     * @param ext       RBPF_Extended instance
     * @param threshold Surprise σ threshold (default: 0.5)
     */
    void rbpf_ext_set_apf_surprise_threshold(RBPF_Extended *ext, float threshold);

    /** @brief Get current APF surprise threshold */
    float rbpf_ext_get_apf_surprise_threshold(const RBPF_Extended *ext);

    /**
     * @brief Configure Hawkes integrator with full config
     *
     * @param ext  RBPF_Extended instance
     * @param cfg  Hawkes configuration struct
     */
    void rbpf_ext_configure_hawkes(RBPF_Extended *ext, const HawkesIntegratorConfig *cfg);

    /**
     * @brief Configure Hawkes parameters directly
     *
     * @param ext             RBPF_Extended instance
     * @param mu              Baseline intensity
     * @param alpha           Excitation magnitude
     * @param beta            Decay rate
     * @param event_threshold Return threshold for event detection
     */
    void rbpf_ext_configure_hawkes_params(RBPF_Extended *ext,
                                          float mu, float alpha, float beta,
                                          float event_threshold);

    /** @brief Get current Hawkes intensity */
    float rbpf_ext_get_hawkes_intensity(const RBPF_Extended *ext);

    /** @brief Get current Hawkes surprise (σ above baseline) */
    float rbpf_ext_get_hawkes_surprise(const RBPF_Extended *ext);

    /** @brief Check if Hawkes integrator has warmed up */
    int rbpf_ext_hawkes_is_ready(const RBPF_Extended *ext);

    /** @brief Print Hawkes integrator state */
    void rbpf_ext_print_hawkes_state(const RBPF_Extended *ext);

    /*═══════════════════════════════════════════════════════════════════════════
     * SECTION 6: ROBUST OCSN
     *
     * Implementation: rbpf_ext_hawkes.c
     *═══════════════════════════════════════════════════════════════════════════*/

    /** @brief Enable robust OCSN with default per-regime parameters */
    void rbpf_ext_enable_robust_ocsn(RBPF_Extended *ext);

    /**
     * @brief Enable robust OCSN with simple uniform parameters
     *
     * @param ext      RBPF_Extended instance
     * @param prob     Outlier probability (e.g., 0.01)
     * @param variance Outlier variance (e.g., 20.0)
     */
    void rbpf_ext_enable_robust_ocsn_simple(RBPF_Extended *ext,
                                            rbpf_real_t prob, rbpf_real_t variance);

    /**
     * @brief Set outlier parameters for specific regime
     *
     * @param ext      RBPF_Extended instance
     * @param regime   Regime index
     * @param prob     Outlier probability
     * @param variance Outlier variance
     */
    void rbpf_ext_set_outlier_params(RBPF_Extended *ext, int regime,
                                     rbpf_real_t prob, rbpf_real_t variance);

    /** @brief Disable robust OCSN */
    void rbpf_ext_disable_robust_ocsn(RBPF_Extended *ext);

    /** @brief Get fraction of likelihood assigned to outlier component */
    rbpf_real_t rbpf_ext_get_outlier_fraction(const RBPF_Extended *ext);

    /*═══════════════════════════════════════════════════════════════════════════
     * SECTION 7: ASSET PRESETS
     *
     * Implementation: rbpf_ext_hawkes.c
     *═══════════════════════════════════════════════════════════════════════════*/

    /**
     * @brief Apply asset class preset
     *
     * Configures regime params, transition matrix, Hawkes, and OCSN for
     * the specified asset class.
     */
    void rbpf_ext_apply_preset(RBPF_Extended *ext, RBPF_AssetPreset preset);

    /** @brief Get currently active preset */
    RBPF_AssetPreset rbpf_ext_get_preset(const RBPF_Extended *ext);

    /*═══════════════════════════════════════════════════════════════════════════
     * SECTION 8: SMOOTHED STORVIK (PARIS Fixed-Lag)
     *
     * Implementation: rbpf_ext_smoothed_storvik.c
     *═══════════════════════════════════════════════════════════════════════════*/

    /**
     * @brief Enable PARIS-smoothed Storvik
     *
     * Replaces filtered (ℓ, ℓ_lag) with smoothed values in Storvik updates.
     * Trading signal (vol_mean) remains immediate.
     *
     * @param ext  RBPF_Extended instance
     * @param lag  Smoothing lag L (50 recommended for HFT)
     * @return     0 on success, -1 on failure
     */
    int rbpf_ext_enable_smoothed_storvik(RBPF_Extended *ext, int lag);

    /** @brief Disable smoothed Storvik (use filtered values) */
    void rbpf_ext_disable_smoothed_storvik(RBPF_Extended *ext);

    /** @brief Check if smoothed Storvik is enabled */
    int rbpf_ext_is_smoothed_storvik_enabled(const RBPF_Extended *ext);

    /**
     * @brief Get smoother statistics
     *
     * @param ext         RBPF_Extended instance
     * @param flush_count Output: emergency flushes
     * @param reset_count Output: ESS-collapse resets
     * @param avg_smooth_us Output: average smoothing time (μs)
     * @param buffer_fill Output: current buffer fill level
     */
    void rbpf_ext_get_smoother_stats(const RBPF_Extended *ext,
                                     uint64_t *flush_count,
                                     uint64_t *reset_count,
                                     double *avg_smooth_us,
                                     int *buffer_fill);

    /**
     * @brief Configure smoother parameters
     *
     * @param ext                  RBPF_Extended instance
     * @param min_buffer_for_flush Min ticks before flush allowed
     * @param ess_collapse_thresh  ESS threshold for reset (N/20 default)
     */
    void rbpf_ext_configure_smoother(RBPF_Extended *ext,
                                     int min_buffer_for_flush,
                                     float ess_collapse_thresh);

    /** @brief Print smoother configuration */
    void rbpf_ext_print_smoother_config(const RBPF_Extended *ext);

    /*═══════════════════════════════════════════════════════════════════════════
     * SECTION 9: ADAPTIVE FORGETTING
     *
     * Implementation: rbpf_adaptive_forgetting.c
     *═══════════════════════════════════════════════════════════════════════════*/

    /** @brief Initialize adaptive forgetting state */
    void rbpf_adaptive_forgetting_init(RBPF_AdaptiveForgetting *af);

    /**
     * @brief Update adaptive forgetting (called from rbpf_ext_step)
     *
     * @param ext            RBPF_Extended instance
     * @param marginal_lik   Marginal likelihood from Kalman update
     * @param dominant_regime Most common regime among particles
     */
    void rbpf_adaptive_forgetting_update(RBPF_Extended *ext,
                                         rbpf_real_t marginal_lik,
                                         int dominant_regime);

    /** @brief Enable adaptive forgetting with combined signal */
    void rbpf_ext_enable_adaptive_forgetting(RBPF_Extended *ext);

    /** @brief Enable adaptive forgetting with specific signal source */
    void rbpf_ext_enable_adaptive_forgetting_mode(RBPF_Extended *ext, RBPF_AdaptSignal signal);

    /** @brief Disable adaptive forgetting (use fixed λ) */
    void rbpf_ext_disable_adaptive_forgetting(RBPF_Extended *ext);

    /** @brief Set baseline λ for specific regime */
    void rbpf_ext_set_regime_lambda(RBPF_Extended *ext, int regime, rbpf_real_t lambda);

    /**
     * @brief Configure sigmoid response parameters
     *
     * @param ext          RBPF_Extended instance
     * @param center       Z-score threshold (default: 2.0)
     * @param steepness    Sigmoid slope (default: 1.0)
     * @param max_discount Maximum λ reduction (default: 0.05)
     */
    void rbpf_ext_set_adaptive_sigmoid(RBPF_Extended *ext,
                                       rbpf_real_t center,
                                       rbpf_real_t steepness,
                                       rbpf_real_t max_discount);

    /** @brief Set λ bounds */
    void rbpf_ext_set_adaptive_bounds(RBPF_Extended *ext,
                                      rbpf_real_t floor,
                                      rbpf_real_t ceiling);

    /** @brief Set EMA smoothing parameters */
    void rbpf_ext_set_adaptive_smoothing(RBPF_Extended *ext,
                                         rbpf_real_t baseline_alpha,
                                         rbpf_real_t signal_alpha);

    /** @brief Set cooldown ticks after intervention */
    void rbpf_ext_set_adaptive_cooldown(RBPF_Extended *ext, int ticks);

    /**
     * @brief Enable P² circuit breaker
     *
     * @param ext      RBPF_Extended instance
     * @param quantile Trigger percentile (e.g., 0.999)
     * @param window   Warmup window in ticks
     */
    void rbpf_ext_enable_circuit_breaker(RBPF_Extended *ext, double quantile, int window);

    /** @brief Disable P² circuit breaker */
    void rbpf_ext_disable_circuit_breaker(RBPF_Extended *ext);

    /** @brief Set minimum memory before circuit breaker can fire */
    void rbpf_ext_set_circuit_breaker_min_memory(RBPF_Extended *ext, int min_ticks);

    /** @brief Set restoration blend duration after circuit breaker */
    void rbpf_ext_set_circuit_breaker_restore_ticks(RBPF_Extended *ext, int ticks);

    /** @brief Get circuit breaker trip count */
    uint64_t rbpf_ext_get_circuit_breaker_trips(const RBPF_Extended *ext);

    /** @brief Get current circuit breaker threshold value */
    rbpf_real_t rbpf_ext_get_circuit_breaker_threshold(const RBPF_Extended *ext);

    /** @brief Get last emergency λ used */
    rbpf_real_t rbpf_ext_get_last_emergency_lambda(const RBPF_Extended *ext);

    /** @brief Check if λ override is active */
    int rbpf_ext_lambda_override_active(const RBPF_Extended *ext);

    /** @brief Get restoration progress (0.0 to 1.0) */
    rbpf_real_t rbpf_ext_get_restore_progress(const RBPF_Extended *ext);

    /** @brief Get current effective λ */
    rbpf_real_t rbpf_ext_get_current_lambda(const RBPF_Extended *ext);

    /** @brief Get current surprise z-score */
    rbpf_real_t rbpf_ext_get_surprise_zscore(const RBPF_Extended *ext);

    /**
     * @brief Get adaptive forgetting statistics
     *
     * @param ext            RBPF_Extended instance
     * @param interventions  Output: total λ adjustments
     * @param current_lambda Output: current effective λ
     * @param max_surprise   Output: maximum surprise seen
     */
    void rbpf_ext_get_adaptive_stats(const RBPF_Extended *ext,
                                     uint64_t *interventions,
                                     rbpf_real_t *current_lambda,
                                     rbpf_real_t *max_surprise);

    /** @brief Print adaptive forgetting configuration */
    void rbpf_ext_print_adaptive_config(const RBPF_Extended *ext);

    /*═══════════════════════════════════════════════════════════════════════════
     * SECTION 10: KL TEMPERING
     *
     * Implementation: rbpf_ext_config.c (enable/disable), 
     *                 rbpf_ext_diagnostics.c (queries)
     *═══════════════════════════════════════════════════════════════════════════*/

    /**
     * @brief Enable KL tempering
     *
     * When enabled, weight updates are tempered based on KL divergence
     * to prevent particle collapse on extreme observations.
     */
    void rbpf_ext_enable_kl_tempering(RBPF_Extended *ext);

    /** @brief Disable KL tempering */
    void rbpf_ext_disable_kl_tempering(RBPF_Extended *ext);

    /** @brief Check if KL tempering is enabled */
    int rbpf_ext_kl_tempering_enabled(const RBPF_Extended *ext);

    /**
     * @brief Get last tempering factor β
     * @return β ∈ [0.1, 1.0], where 1.0 = full update, <1.0 = tempered
     */
    float rbpf_ext_get_last_beta(const RBPF_Extended *ext);

    /**
     * @brief Get last KL divergence
     * @return KL(proposed || old) in nats
     */
    float rbpf_ext_get_last_kl(const RBPF_Extended *ext);

    /**
     * @brief Get KL ceiling (log N)
     * @return Hard limit on per-tick information in nats
     */
    rbpf_real_t rbpf_ext_get_kl_ceiling(const RBPF_Extended *ext);

    /** @brief Get zombie reset count */
    uint64_t rbpf_ext_get_zombie_resets(const RBPF_Extended *ext);

    /** @brief Print KL tempering diagnostics */
    void rbpf_ext_print_kl_diagnostics(const RBPF_Extended *ext);

    /*═══════════════════════════════════════════════════════════════════════════
     * SECTION 11: DIAGNOSTICS
     *
     * Implementation: rbpf_ext_diagnostics.c
     *═══════════════════════════════════════════════════════════════════════════*/

    /** @brief Print full configuration summary */
    void rbpf_ext_print_config(const RBPF_Extended *ext);

    /** @brief Print Storvik statistics for regime */
    void rbpf_ext_print_storvik_stats(const RBPF_Extended *ext, int regime);

    /*═══════════════════════════════════════════════════════════════════════════
     * SECTION 12: FORGETTING HELPERS
     *
     * Implementation: rbpf_ext_config.c
     *═══════════════════════════════════════════════════════════════════════════*/

    /**
     * @brief Derive per-regime λ from transition matrix
     *
     * Sets λ such that effective memory = α × expected dwell time.
     *
     * @param ext   RBPF_Extended instance
     * @param alpha Memory multiplier (3.0 typical)
     */
    void rbpf_ext_compute_forgetting_from_transitions(RBPF_Extended *ext, float alpha);

    /** @brief Auto-configure forgetting with default α=3.0 */
    void rbpf_ext_auto_configure_forgetting(RBPF_Extended *ext);

#ifdef __cplusplus
}
#endif

#endif /* RBPF_EXT_API_H */
