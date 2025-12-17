/**
 * @file sticky_hdp_beam.h
 * @brief Sticky HDP-HMM with Beam Sampling (MKL Accelerated)
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * STICKY HDP-HMM (Fox et al., 2011)
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * Nonparametric Bayesian hidden Markov model with:
 *   - Infinite state capacity (Dirichlet Process prior)
 *   - Hierarchical sharing of transition structure
 *   - Sticky transitions (self-transition bias via κ)
 *
 * MODEL:
 *   β ~ GEM(γ)                              Global state distribution
 *   π_k ~ DP(α + κ, (α·β + κ·δ_k)/(α + κ))  Transition from state k
 *   θ_k ~ H                                  Emission parameters
 *   s_t | s_{t-1} ~ π_{s_{t-1}}             State sequence
 *   y_t | s_t ~ F(θ_{s_t})                  Observations
 *
 * PARAMETERS:
 *   γ  - DP concentration for global β (controls # states)
 *   α  - DP concentration for transitions (controls sharing)
 *   κ  - Stickiness (self-transition bonus)
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * BEAM SAMPLING (Van Gael et al., 2008)
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * Efficient inference by introducing auxiliary slice variables:
 *
 *   u_t | s_{t-1}, s_t ~ Uniform(0, π_{s_{t-1}, s_t})
 *
 * This restricts active states to: A_t = {k : π_{s_{t-1}, k} > u_t}
 * Typically |A_t| ≈ 3-8, making forward-backward tractable.
 *
 * ALGORITHM:
 *   1. Sample slice variables u_t (determines active states)
 *   2. Forward filter over active states only
 *   3. Backward sample state sequence
 *   4. Update β, π, θ given new states
 *
 * COMPLEXITY: O(T × K_active²) vs O(T × K_trunc²) for standard truncation
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * MKL ACCELERATION
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * Bottleneck          | MKL Solution           | Speedup
 * --------------------|------------------------|--------
 * Likelihood batch    | vdExp, vdLn, vdMul    | 4-8×
 * Forward filtering   | cblas_dgemv           | 2-4×
 * Log-sum-exp         | vdExp + cblas_dasum   | 3-5×
 * Stick-breaking      | vdRngBeta             | 2-3×
 * Slice sampling      | vdRngUniform          | 2×
 *
 * Target: ~1ms per beam sweep (T=100, K_active≈6)
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * USAGE
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * // Create sampler
 * StickyHDP *hdp = sticky_hdp_create(K_MAX, T_MAX);
 *
 * // Configure priors
 * sticky_hdp_set_concentration(hdp, gamma, alpha);
 * sticky_hdp_set_stickiness(hdp, kappa);
 *
 * // Online: process observations
 * for (int t = 0; t < T; t++) {
 *     sticky_hdp_observe(hdp, y[t]);
 *
 *     // Periodically run beam sampling (every N ticks or on trigger)
 *     if (t % 100 == 0 || regime_uncertain) {
 *         sticky_hdp_beam_sweep(hdp, n_sweeps);
 *     }
 *
 *     // Get current state estimate
 *     int state = sticky_hdp_map_state(hdp);
 *     double* probs = sticky_hdp_state_probs(hdp);
 * }
 *
 * // Get learned transition matrix (for export to RBPF)
 * sticky_hdp_get_transitions(hdp, trans_matrix, n_states);
 *
 * sticky_hdp_destroy(hdp);
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 */

#ifndef STICKY_HDP_BEAM_H
#define STICKY_HDP_BEAM_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C"
{
#endif

    /*═══════════════════════════════════════════════════════════════════════════════
     * CONFIGURATION
     *═══════════════════════════════════════════════════════════════════════════════*/

#define HDP_MAX_STATES 32   /* Maximum states (truncation) */
#define HDP_MAX_WINDOW 2000 /* Maximum observation window */
#define HDP_CACHE_LINE 64   /* Cache alignment */

/* Numerical stability */
#define HDP_LOG_ZERO -1e10 /* Log of "zero" probability */
#define HDP_MIN_PROB 1e-10 /* Minimum probability */
#define HDP_EPS 1e-12      /* Numerical epsilon */

    /*═══════════════════════════════════════════════════════════════════════════════
     * EMISSION MODEL
     *═══════════════════════════════════════════════════════════════════════════════*/

    /**
     * Emission parameters for log-volatility (OU process)
     *
     * State k has: ℓ_t = (1-θ_k)ℓ_{t-1} + θ_k·μ_k + η_t,  η_t ~ N(0, σ_k²)
     */
    typedef struct
    {
        double mu;    /**< Long-run mean log-vol */
        double theta; /**< Mean reversion speed */
        double sigma; /**< Volatility of volatility */

        /* Sufficient statistics for conjugate updates */
        double sum_x;  /**< Σ x_t (observations in this state) */
        double sum_x2; /**< Σ x_t² */
        double sum_xy; /**< Σ x_t × x_{t-1} */
        double sum_y;  /**< Σ x_{t-1} */
        double sum_y2; /**< Σ x_{t-1}² */
        int n;         /**< Count of observations */
    } HDP_EmissionParams;

    /**
     * Prior for emission parameters (Normal-Inverse-Gamma)
     */
    typedef struct
    {
        double mu0;       /**< Prior mean for μ */
        double kappa0;    /**< Prior precision weight for μ */
        double alpha0;    /**< Prior shape for σ² */
        double beta0;     /**< Prior rate for σ² */
        double theta_min; /**< Min mean reversion (stability) */
        double theta_max; /**< Max mean reversion */
    } HDP_EmissionPrior;

    /*═══════════════════════════════════════════════════════════════════════════════
     * CORE STRUCTURES
     *═══════════════════════════════════════════════════════════════════════════════*/

    /**
     * Sticky HDP-HMM state
     */
    typedef struct
    {
        /*─────────────────────────────────────────────────────────────────────────
         * HYPERPARAMETERS
         *───────────────────────────────────────────────────────────────────────*/
        double gamma; /**< Global DP concentration */
        double alpha; /**< Transition DP concentration */
        double kappa; /**< Stickiness parameter */

        /* Hyperparameter priors (for learning) */
        double gamma_a, gamma_b; /**< Gamma prior on γ */
        double alpha_a, alpha_b; /**< Gamma prior on α */
        double kappa_a, kappa_b; /**< Gamma prior on κ */
        bool learn_hyperparams;  /**< Update γ, α, κ? */

        /*─────────────────────────────────────────────────────────────────────────
         * GLOBAL MEASURE β (Stick-Breaking)
         *───────────────────────────────────────────────────────────────────────*/
        int K;                /**< Current number of active states */
        int K_max;            /**< Maximum states (truncation) */
        double *beta;         /**< Global state weights [K_max] */
        double *log_beta;     /**< Log of beta [K_max] */
        double *stick_remain; /**< Remaining stick mass [K_max] */

        /*─────────────────────────────────────────────────────────────────────────
         * TRANSITION DISTRIBUTIONS π
         *───────────────────────────────────────────────────────────────────────*/
        double *pi;     /**< Transition probs [K_max × K_max] row-major */
        double *log_pi; /**< Log transition probs [K_max × K_max] */
        int *n_trans;   /**< Transition counts [K_max × K_max] */

        /*─────────────────────────────────────────────────────────────────────────
         * EMISSION PARAMETERS
         *───────────────────────────────────────────────────────────────────────*/
        HDP_EmissionParams *emit;     /**< Per-state emission params [K_max] */
        HDP_EmissionPrior emit_prior; /**< Shared emission prior */

        /*─────────────────────────────────────────────────────────────────────────
         * OBSERVATION WINDOW
         *───────────────────────────────────────────────────────────────────────*/
        int T;     /**< Current window length */
        int T_max; /**< Maximum window */
        double *y; /**< Observations [T_max] */
        int *s;    /**< State assignments [T_max] */
        double *u; /**< Slice variables [T_max] */

        /*─────────────────────────────────────────────────────────────────────────
         * BEAM SAMPLING WORKSPACE
         *───────────────────────────────────────────────────────────────────────*/

        /* Active state tracking */
        int *active;     /**< Active state indices [K_max] */
        int n_active;    /**< Number of active states */
        bool *is_active; /**< Fast lookup [K_max] */

        /* Forward messages (log-space) */
        double *log_alpha;     /**< Forward messages [T_max × K_max] */
        double *log_alpha_sum; /**< Normalizers [T_max] */

        /* Likelihood cache */
        double *log_lik; /**< Log-likelihoods [T_max × K_max] */
        bool *lik_valid; /**< Validity flags [T_max × K_max] */

        /* Scratch buffers (cache-aligned for MKL) */
        double *scratch1; /**< [K_max] */
        double *scratch2; /**< [K_max] */
        double *scratch3; /**< [K_max] */

        /*─────────────────────────────────────────────────────────────────────────
         * MKL RNG
         *───────────────────────────────────────────────────────────────────────*/
        void *mkl_stream;  /**< VSLStreamStatePtr */
        uint64_t rng_seed; /**< RNG seed for reproducibility */

        /*─────────────────────────────────────────────────────────────────────────
         * DIAGNOSTICS
         *───────────────────────────────────────────────────────────────────────*/
        uint64_t total_sweeps;     /**< Total beam sweeps performed */
        uint64_t total_new_states; /**< Times a new state was created */
        double avg_active_states;  /**< Running average of |A_t| */
        double last_log_marginal;  /**< Log marginal likelihood */

        /*─────────────────────────────────────────────────────────────────────────
         * ADAPTIVE SWEEP TRIGGERING (v2)
         *───────────────────────────────────────────────────────────────────────*/
        double last_surprise;      /**< Most recent observation surprise */
        double surprise_ema;       /**< Exponential moving average of surprise */
        int ticks_since_sweep;     /**< Ticks since last sweep */
        double surprise_threshold; /**< Trigger sweep if surprise > threshold */
        int max_idle_ticks;        /**< Max ticks without sweep */
        int min_sweep_interval;    /**< Min ticks between sweeps */

    } StickyHDP;

    /*═══════════════════════════════════════════════════════════════════════════════
     * LIFECYCLE
     *═══════════════════════════════════════════════════════════════════════════════*/

    /**
     * @brief Create Sticky HDP-HMM sampler
     *
     * @param K_max     Maximum number of states (truncation level)
     * @param T_max     Maximum observation window length
     * @return          Allocated sampler or NULL on failure
     */
    StickyHDP *sticky_hdp_create(int K_max, int T_max);

    /**
     * @brief Destroy sampler and free resources
     */
    void sticky_hdp_destroy(StickyHDP *hdp);

    /**
     * @brief Reset sampler to initial state (keep configuration)
     */
    void sticky_hdp_reset(StickyHDP *hdp);

    /*═══════════════════════════════════════════════════════════════════════════════
     * CONFIGURATION
     *═══════════════════════════════════════════════════════════════════════════════*/

    /**
     * @brief Set DP concentration parameters
     *
     * @param hdp       Sampler
     * @param gamma     Global concentration (controls expected # states)
     *                  E[K] ≈ γ × log(T) for large T
     * @param alpha     Transition concentration (controls sharing)
     *                  Higher α → transitions closer to global β
     */
    void sticky_hdp_set_concentration(StickyHDP *hdp, double gamma, double alpha);

    /**
     * @brief Set stickiness parameter
     *
     * @param hdp       Sampler
     * @param kappa     Self-transition bonus
     *                  κ = 0: standard HDP-HMM (rapid switching)
     *                  κ = 10-50: moderate stickiness
     *                  κ = 100+: very sticky (regime-like)
     *
     * INTERPRETATION:
     *   E[π_{kk}] ≈ (α·β_k + κ) / (α + κ)
     *   For κ >> α: E[π_{kk}] ≈ 1 (very sticky)
     *   For κ = 0:  E[π_{kk}] ≈ β_k (follow global)
     */
    void sticky_hdp_set_stickiness(StickyHDP *hdp, double kappa);

    /**
     * @brief Enable hyperparameter learning
     *
     * When enabled, γ, α, κ are sampled during beam sweeps.
     *
     * @param hdp       Sampler
     * @param gamma_a   Shape for Gamma prior on γ
     * @param gamma_b   Rate for Gamma prior on γ
     * @param alpha_a   Shape for Gamma prior on α
     * @param alpha_b   Rate for Gamma prior on α
     * @param kappa_a   Shape for Gamma prior on κ
     * @param kappa_b   Rate for Gamma prior on κ
     */
    void sticky_hdp_enable_hyperparam_learning(StickyHDP *hdp,
                                               double gamma_a, double gamma_b,
                                               double alpha_a, double alpha_b,
                                               double kappa_a, double kappa_b);

    /**
     * @brief Disable hyperparameter learning (use fixed values)
     */
    void sticky_hdp_disable_hyperparam_learning(StickyHDP *hdp);

    /**
     * @brief Set emission prior
     *
     * Normal-Inverse-Gamma prior on (μ, σ²):
     *   μ | σ² ~ N(μ₀, σ²/κ₀)
     *   σ² ~ Inv-Gamma(α₀, β₀)
     */
    void sticky_hdp_set_emission_prior(StickyHDP *hdp,
                                       double mu0, double kappa0,
                                       double alpha0, double beta0,
                                       double theta_min, double theta_max);

    /**
     * @brief Initialize with known regime structure
     *
     * Useful for warm-starting from RBPF regime estimates.
     *
     * @param hdp           Sampler
     * @param n_regimes     Number of initial regimes
     * @param mu_vol        Mean log-vol for each regime [n_regimes]
     * @param sigma_vol     Vol-of-vol for each regime [n_regimes]
     * @param theta         Mean reversion for each regime [n_regimes]
     */
    void sticky_hdp_init_regimes(StickyHDP *hdp, int n_regimes,
                                 const double *mu_vol,
                                 const double *sigma_vol,
                                 const double *theta);

    /*═══════════════════════════════════════════════════════════════════════════════
     * OBSERVATION MANAGEMENT
     *═══════════════════════════════════════════════════════════════════════════════*/

    /**
     * @brief Add observation to window
     *
     * For online use, call this each tick. The window is managed internally.
     *
     * @param hdp   Sampler
     * @param y     Observation (log-squared return: y = log(r²))
     */
    void sticky_hdp_observe(StickyHDP *hdp, double y);

    /**
     * @brief Set full observation sequence (batch mode)
     *
     * @param hdp   Sampler
     * @param y     Observation array
     * @param T     Length
     */
    void sticky_hdp_set_observations(StickyHDP *hdp, const double *y, int T);

    /**
     * @brief Clear observation window
     */
    void sticky_hdp_clear_observations(StickyHDP *hdp);

    /**
     * @brief Slide window (remove oldest, keep capacity for new)
     *
     * @param hdp       Sampler
     * @param n_remove  Number of oldest observations to remove
     */
    void sticky_hdp_slide_window(StickyHDP *hdp, int n_remove);

    /*═══════════════════════════════════════════════════════════════════════════════
     * INFERENCE
     *═══════════════════════════════════════════════════════════════════════════════*/

    /**
     * @brief Run beam sampling sweeps
     *
     * Each sweep:
     *   1. Sample slice variables
     *   2. Forward filter (active states only)
     *   3. Backward sample state sequence
     *   4. Update transitions and emissions
     *   5. Optionally update hyperparameters
     *
     * @param hdp       Sampler
     * @param n_sweeps  Number of sweeps (1-10 typical for online)
     * @return          Log marginal likelihood estimate
     */
    double sticky_hdp_beam_sweep(StickyHDP *hdp, int n_sweeps);

    /**
     * @brief Run single beam sweep (for fine-grained control)
     */
    double sticky_hdp_beam_sweep_single(StickyHDP *hdp);

    /*═══════════════════════════════════════════════════════════════════════════════
     * BLOCKED GIBBS (FFBS) SAMPLING
     *═══════════════════════════════════════════════════════════════════════════════
     *
     * Blocked Forward-Filtering Backward-Sampling for improved mixing.
     *
     * Standard beam sampling updates states sequentially and can get stuck in
     * local modes. Blocked Gibbs samples entire blocks jointly, providing:
     *
     *   - 3-5× faster convergence (2-3 sweeps vs 10)
     *   - Better exploration of state space
     *   - More robust regime detection
     *
     * USE WHEN:
     *   - Need high-quality posterior samples (not just point estimates)
     *   - Model is getting stuck in suboptimal configurations
     *   - Running batch analysis (not real-time)
     *
     * BLOCK SIZE GUIDELINES:
     *   - block_size = 50:  Good balance of mixing and speed
     *   - block_size = 100: Better mixing, slightly slower
     *   - block_size = T:   Full FFBS (best mixing, O(T×K²))
     */

    /**
     * @brief Run blocked Gibbs (FFBS) sweeps
     *
     * @param hdp         Sampler
     * @param n_sweeps    Number of sweeps (2-5 typical - converges faster than beam)
     * @param block_size  Block size (50-100 recommended, 0 for default=50)
     * @return            Log marginal likelihood estimate
     */
    double sticky_hdp_blocked_gibbs(StickyHDP *hdp, int n_sweeps, int block_size);

    /**
     * @brief Run single blocked Gibbs (FFBS) sweep
     *
     * @param hdp         Sampler
     * @param block_size  Block size (50-100 recommended, 0 for default=50)
     * @return            Log marginal likelihood estimate
     */
    double sticky_hdp_blocked_gibbs_single(StickyHDP *hdp, int block_size);

    /**
     * @brief Update only hyperparameters (given current state sequence)
     *
     * Useful for periodic recalibration without full inference.
     */
    void sticky_hdp_update_hyperparams(StickyHDP *hdp);

    /**
     * @brief Update only emission parameters (given current state sequence)
     */
    void sticky_hdp_update_emissions(StickyHDP *hdp);

    /*═══════════════════════════════════════════════════════════════════════════════
     * STATE QUERIES
     *═══════════════════════════════════════════════════════════════════════════════*/

    /**
     * @brief Get MAP state estimate for current time
     */
    int sticky_hdp_map_state(const StickyHDP *hdp);

    /**
     * @brief Get state probabilities for current time
     *
     * @param hdp       Sampler
     * @param probs     Output array [K_max] (will sum to 1)
     * @return          Number of active states
     */
    int sticky_hdp_state_probs(const StickyHDP *hdp, double *probs);

    /**
     * @brief Get full state sequence
     *
     * @param hdp       Sampler
     * @param states    Output array [T]
     * @return          Sequence length T
     */
    int sticky_hdp_get_states(const StickyHDP *hdp, int *states);

    /**
     * @brief Get number of active states
     */
    int sticky_hdp_num_states(const StickyHDP *hdp);

    /*═══════════════════════════════════════════════════════════════════════════════
     * PARAMETER QUERIES
     *═══════════════════════════════════════════════════════════════════════════════*/

    /**
     * @brief Get learned transition matrix
     *
     * Extracts π into a dense matrix for use with RBPF.
     *
     * @param hdp       Sampler
     * @param trans     Output matrix [K × K] row-major
     * @param K         Number of states to extract
     */
    void sticky_hdp_get_transitions(const StickyHDP *hdp, double *trans, int K);

    /**
     * @brief Get global state distribution β
     */
    void sticky_hdp_get_beta(const StickyHDP *hdp, double *beta, int K);

    /**
     * @brief Get emission parameters for state k
     */
    void sticky_hdp_get_emission(const StickyHDP *hdp, int k, HDP_EmissionParams *params);

    /**
     * @brief Get current hyperparameter values
     */
    void sticky_hdp_get_hyperparams(const StickyHDP *hdp,
                                    double *gamma, double *alpha, double *kappa);

    /**
     * @brief Get stickiness (self-transition probability) for state k
     *
     * Returns E[π_{kk}] under current parameters.
     */
    double sticky_hdp_get_stickiness_prob(const StickyHDP *hdp, int k);

    /*═══════════════════════════════════════════════════════════════════════════════
     * DIAGNOSTICS
     *═══════════════════════════════════════════════════════════════════════════════*/

    /**
     * @brief Print sampler state summary
     */
    void sticky_hdp_print_summary(const StickyHDP *hdp);

    /**
     * @brief Print transition matrix
     */
    void sticky_hdp_print_transitions(const StickyHDP *hdp);

    /**
     * @brief Print emission parameters
     */
    void sticky_hdp_print_emissions(const StickyHDP *hdp);

    /**
     * @brief Get diagnostic statistics
     */
    typedef struct
    {
        int K;                      /**< Current number of states */
        double avg_active;          /**< Average active states per step */
        double log_marginal;        /**< Log marginal likelihood */
        double gamma, alpha, kappa; /**< Current hyperparameters */
        uint64_t total_sweeps;      /**< Total sweeps performed */
        double last_sweep_time_us;  /**< Last sweep wall time */
    } HDP_Diagnostics;

    void sticky_hdp_get_diagnostics(const StickyHDP *hdp, HDP_Diagnostics *diag);

    /*═══════════════════════════════════════════════════════════════════════════════
     * INTEGRATION WITH RBPF
     *═══════════════════════════════════════════════════════════════════════════════*/

    /**
     * @brief Export to RBPF-compatible format
     *
     * Converts HDP states to fixed regime structure for RBPF.
     * Maps K_hdp states to K_rbpf regimes by μ ordering.
     *
     * @param hdp           HDP sampler
     * @param K_rbpf        Target number of regimes (e.g., 4)
     * @param trans         Output transition matrix [K_rbpf × K_rbpf]
     * @param mu_vol        Output mean log-vol [K_rbpf]
     * @param sigma_vol     Output vol-of-vol [K_rbpf]
     * @param theta         Output mean reversion [K_rbpf]
     */
    void sticky_hdp_export_to_rbpf(const StickyHDP *hdp, int K_rbpf,
                                   double *trans,
                                   double *mu_vol,
                                   double *sigma_vol,
                                   double *theta);

    /**
     * @brief Import RBPF regime estimates to warm-start HDP
     *
     * @param hdp           HDP sampler
     * @param K_rbpf        Number of RBPF regimes
     * @param regime_seq    RBPF regime sequence [T]
     * @param T             Sequence length
     * @param mu_vol        RBPF mean log-vol [K_rbpf]
     * @param sigma_vol     RBPF vol-of-vol [K_rbpf]
     */
    void sticky_hdp_import_from_rbpf(StickyHDP *hdp, int K_rbpf,
                                     const int *regime_seq, int T,
                                     const double *mu_vol,
                                     const double *sigma_vol);

    /*═══════════════════════════════════════════════════════════════════════════════
     * ADAPTIVE SWEEP TRIGGERING (v2)
     *═══════════════════════════════════════════════════════════════════════════════
     *
     * Instead of sweeping on a fixed schedule, trigger sweeps when:
     *   1. High surprise (potential new regime)
     *   2. Too many ticks without a sweep
     *
     * This can reduce sweep frequency by 5-10× during stable periods.
     */

    /**
     * @brief Configure adaptive sweep triggering
     *
     * @param hdp                   Sampler
     * @param surprise_threshold    Trigger if surprise > threshold (default: 5.0)
     * @param max_idle_ticks        Max ticks without sweep (default: 100)
     * @param min_sweep_interval    Min ticks between sweeps (default: 10)
     */
    void sticky_hdp_set_adaptive_sweep(StickyHDP *hdp,
                                       double surprise_threshold,
                                       int max_idle_ticks,
                                       int min_sweep_interval);

    /**
     * @brief Check if sweep should be triggered
     *
     * Call after each observation. Returns true if sweep recommended.
     *
     * @param hdp       Sampler
     * @return          true if sweep should be run
     */
    bool sticky_hdp_should_sweep(StickyHDP *hdp);

    /**
     * @brief Get current surprise level
     *
     * Surprise = -log P(y_t | current model)
     * High surprise indicates observation doesn't fit current states.
     */
    double sticky_hdp_get_surprise(const StickyHDP *hdp);

    /*═══════════════════════════════════════════════════════════════════════════════
     * BIRTH/MERGE PROPOSALS (v2)
     *═══════════════════════════════════════════════════════════════════════════════
     *
     * Fast regime discovery without waiting for slow DP diffusion:
     *   - BIRTH: Create new state when surprise is high
     *   - MERGE: Combine redundant states with similar parameters
     */

    /**
     * @brief Propose birth of new state based on recent high-surprise observations
     *
     * Creates a new state fitted to observations that don't match existing states.
     *
     * @param hdp               Sampler
     * @param surprise_window   Look back this many ticks for high-surprise obs
     * @param min_surprise      Only consider obs with surprise > this
     * @return                  Index of new state, or -1 if none created
     */
    int sticky_hdp_propose_birth(StickyHDP *hdp, int surprise_window, double min_surprise);

    /**
     * @brief Propose merge of two similar states
     *
     * Merges states i and j if their emission parameters are similar.
     *
     * @param hdp               Sampler
     * @param mu_threshold      Merge if |μ_i - μ_j| < threshold
     * @return                  Number of merges performed
     */
    int sticky_hdp_propose_merge(StickyHDP *hdp, double mu_threshold);

    /**
     * @brief Run birth/merge proposals
     *
     * Convenience function: runs birth if high surprise, merge if too many states.
     *
     * @param hdp       Sampler
     * @return          Net change in number of states
     */
    int sticky_hdp_birth_merge(StickyHDP *hdp);

#ifdef __cplusplus
}
#endif

#endif /* STICKY_HDP_BEAM_H */