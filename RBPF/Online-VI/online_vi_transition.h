/**
 * @file online_vi_transition.h
 * @brief Online Variational Inference for Transition Matrix Learning
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * OVERVIEW
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * Maintains a full posterior q(π) over transition probabilities using online
 * variational inference. Unlike point-estimate methods, this preserves
 * uncertainty information for downstream decision-making.
 *
 * KEY INSIGHT:
 *   Instead of just tracking E[π_ij], we maintain the full Dirichlet posterior:
 *
 *     q(π_i) = Dirichlet(α̃_i1, ..., α̃_iK)
 *
 *   This gives us:
 *     - E[π_ij] = α̃_ij / Σ_k α̃_ik           (mean)
 *     - Var[π_ij] = ...                       (uncertainty)
 *     - H[q(π_i)] = ...                       (entropy per row)
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * THE MATH: Natural Gradient VI
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * For Dirichlet-Multinomial, the natural gradient update is:
 *
 *   α̃_ij ← (1 - ρ_t) × α̃_ij + ρ_t × (α_prior_ij + N × ξ_ij)
 *
 * Where:
 *   ρ_t     = learning rate at time t
 *   N       = effective sample size (typically 1 for single-tick updates)
 *   ξ_ij    = soft transition responsibility P(s_{t-1}=i, s_t=j | y_{1:t})
 *
 * This differs from additive Soft Dirichlet:
 *   - Soft Dirichlet: α += ξ, then decay
 *   - Online VI: blend toward posterior with learning rate
 *
 * The learning rate can follow:
 *   - Fixed: ρ_t = ρ_0 (simple but suboptimal)
 *   - Robbins-Monro: ρ_t = (τ + t)^{-κ} where κ ∈ (0.5, 1]
 *   - Adaptive: based on HDP-beam feedback
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * UNCERTAINTY QUANTIFICATION
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * The Dirichlet posterior gives closed-form uncertainty:
 *
 *   E[π_ij] = α̃_ij / α̃_i0        where α̃_i0 = Σ_k α̃_ik
 *
 *   Var[π_ij] = α̃_ij(α̃_i0 - α̃_ij) / (α̃_i0² × (α̃_i0 + 1))
 *
 *   H[q(π_i)] = log B(α̃_i) + (α̃_i0 - K)ψ(α̃_i0) - Σ_j(α̃_ij - 1)ψ(α̃_ij)
 *
 * Where ψ is the digamma function and B is the multivariate beta.
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * COMPARISON WITH SOFT DIRICHLET
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * | Aspect          | Soft Dirichlet      | Online VI               |
 * |-----------------|---------------------|-------------------------|
 * | State           | Point counts α      | Full posterior q(π)     |
 * | Update          | Additive + decay    | Natural gradient blend  |
 * | Output          | E[π] only           | E[π], Var[π], H[π]      |
 * | Uncertainty     | Lost                | Preserved               |
 * | Kelly use       | Entropy discount    | Direct variance prop    |
 * | Trigger signal  | Global entropy      | Per-row entropy         |
 * | Theory          | Heuristic           | Variational Bayes       |
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * USAGE
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 *   // Initialize with 4 regimes
 *   OnlineVI vi;
 *   online_vi_init(&vi, 4);
 *
 *   // Set learning rate schedule (Robbins-Monro)
 *   online_vi_set_learning_rate(&vi, 1.0, 64.0, 0.7);  // ρ_t = (64+t)^{-0.7}
 *
 *   // In RBPF hot loop:
 *   online_vi_update(&vi, prev_regime_probs, curr_regime_probs, regime_liks);
 *
 *   // Get transition probabilities for RBPF
 *   double trans[16];
 *   online_vi_get_mean(&vi, trans);
 *
 *   // Get uncertainty for Kelly sizing
 *   double var[16];
 *   online_vi_get_variance(&vi, var);
 *
 *   // Get per-row entropy for triggering
 *   double H[4];
 *   online_vi_get_row_entropy(&vi, H);
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 */

#ifndef ONLINE_VI_TRANSITION_H
#define ONLINE_VI_TRANSITION_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*═══════════════════════════════════════════════════════════════════════════════
 * CONFIGURATION
 *═══════════════════════════════════════════════════════════════════════════════*/

#ifndef ONLINE_VI_MAX_REGIMES
#define ONLINE_VI_MAX_REGIMES 8
#endif

/*═══════════════════════════════════════════════════════════════════════════════
 * LEARNING RATE SCHEDULES
 *═══════════════════════════════════════════════════════════════════════════════*/

typedef enum {
    VI_LR_FIXED,           /**< ρ_t = ρ_0 (constant) */
    VI_LR_ROBBINS_MONRO,   /**< ρ_t = ρ_0 × (τ + t)^{-κ} */
    VI_LR_ADAPTIVE         /**< Adjusted by external signal (HDP-beam) */
} VI_LearningRateSchedule;

/*═══════════════════════════════════════════════════════════════════════════════
 * CORE STRUCTURE
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * Online Variational Inference for Transition Matrix
 *
 * Maintains full Dirichlet posterior over each row of the transition matrix.
 */
typedef struct {
    int K;  /**< Number of regimes */
    
    /*───────────────────────────────────────────────────────────────────────────
     * Variational Parameters
     *───────────────────────────────────────────────────────────────────────────*/
    
    /** Dirichlet parameters: q(π_i) = Dir(α̃_i1, ..., α̃_iK) */
    double alpha[ONLINE_VI_MAX_REGIMES][ONLINE_VI_MAX_REGIMES];
    
    /** Prior parameters (for KL computation and reset) */
    double alpha_prior[ONLINE_VI_MAX_REGIMES][ONLINE_VI_MAX_REGIMES];
    
    /** Row sums: α̃_i0 = Σ_k α̃_ik (cached for efficiency) */
    double alpha_sum[ONLINE_VI_MAX_REGIMES];
    
    /*───────────────────────────────────────────────────────────────────────────
     * Posterior Statistics (computed on demand or cached)
     *───────────────────────────────────────────────────────────────────────────*/
    
    /** Posterior mean: E[π_ij] = α̃_ij / α̃_i0 */
    double mean[ONLINE_VI_MAX_REGIMES][ONLINE_VI_MAX_REGIMES];
    
    /** Posterior variance: Var[π_ij] */
    double var[ONLINE_VI_MAX_REGIMES][ONLINE_VI_MAX_REGIMES];
    
    /** Per-row entropy: H[q(π_i)] */
    double row_entropy[ONLINE_VI_MAX_REGIMES];
    
    /** Total entropy (sum of row entropies) */
    double total_entropy;
    
    /** Flag: do we need to recompute statistics? */
    bool stats_dirty;
    
    /*───────────────────────────────────────────────────────────────────────────
     * Learning Rate
     *───────────────────────────────────────────────────────────────────────────*/
    
    VI_LearningRateSchedule lr_schedule;
    
    double rho;           /**< Current learning rate */
    double rho_base;      /**< Base learning rate (ρ_0) */
    double rho_tau;       /**< Delay parameter (τ) for Robbins-Monro */
    double rho_kappa;     /**< Decay exponent (κ) for Robbins-Monro */
    double rho_min;       /**< Minimum learning rate floor */
    double rho_max;       /**< Maximum learning rate ceiling */
    
    /*───────────────────────────────────────────────────────────────────────────
     * State
     *───────────────────────────────────────────────────────────────────────────*/
    
    /** Previous regime probabilities (for ξ computation) */
    double prev_probs[ONLINE_VI_MAX_REGIMES];
    
    /** Has the filter been initialized? */
    bool initialized;
    
    /** Update count (for learning rate schedule) */
    uint64_t t;
    
    /*───────────────────────────────────────────────────────────────────────────
     * Diagnostics
     *───────────────────────────────────────────────────────────────────────────*/
    
    /** Evidence Lower Bound (ELBO) - tracks inference quality */
    double elbo;
    
    /** KL divergence from prior: KL(q || prior) */
    double kl_from_prior;
    
    /** Last ξ matrix (for debugging) */
    double last_xi[ONLINE_VI_MAX_REGIMES][ONLINE_VI_MAX_REGIMES];
    
    /** Total updates performed */
    uint64_t total_updates;
    
} OnlineVI;

/**
 * Diagnostic statistics snapshot
 */
typedef struct {
    double mean_entropy;         /**< Average row entropy */
    double max_entropy;          /**< Maximum row entropy (most uncertain row) */
    double min_entropy;          /**< Minimum row entropy (most certain row) */
    double total_entropy;        /**< Sum of row entropies */
    double mean_stickiness;      /**< Average self-transition probability */
    double kl_from_prior;        /**< KL(q || prior) */
    double current_rho;          /**< Current learning rate */
    uint64_t total_updates;      /**< Number of updates */
} OnlineVI_Stats;

/*═══════════════════════════════════════════════════════════════════════════════
 * LIFECYCLE
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Initialize with uniform prior
 *
 * Sets up uniform transition prior Dir(1, 1, ..., 1).
 * Default learning rate: Robbins-Monro with τ=64, κ=0.7
 *
 * @param vi         Online VI structure
 * @param K          Number of regimes (1-8)
 */
void online_vi_init(OnlineVI *vi, int K);

/**
 * @brief Initialize with sticky prior
 *
 * Sets up sticky transition prior with higher self-transition counts.
 * α_ii = α_base + stickiness, α_ij = α_base for i≠j
 *
 * @param vi          Online VI structure
 * @param K           Number of regimes
 * @param alpha_base  Base concentration (typically 1.0)
 * @param stickiness  Extra self-transition mass (typically 5-20)
 */
void online_vi_init_sticky(OnlineVI *vi, int K, double alpha_base, double stickiness);

/**
 * @brief Initialize from existing transition matrix
 *
 * Warm-starts the posterior from a known transition matrix.
 * Useful for starting from HDP-beam output or tuned parameters.
 *
 * @param vi           Online VI structure
 * @param K            Number of regimes
 * @param trans        Initial transition matrix [K × K] row-major
 * @param confidence   Effective sample size (higher = more confident prior)
 */
void online_vi_init_from_matrix(OnlineVI *vi, int K, 
                                 const double *trans, double confidence);

/**
 * @brief Reset to prior (keep configuration)
 */
void online_vi_reset(OnlineVI *vi);

/*═══════════════════════════════════════════════════════════════════════════════
 * LEARNING RATE CONFIGURATION
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Set fixed learning rate
 *
 * @param vi    Online VI structure
 * @param rho   Learning rate (0 < rho ≤ 1)
 */
void online_vi_set_lr_fixed(OnlineVI *vi, double rho);

/**
 * @brief Set Robbins-Monro learning rate schedule
 *
 * ρ_t = ρ_0 × (τ + t)^{-κ}
 *
 * @param vi       Online VI structure
 * @param rho_0    Base learning rate (typically 1.0)
 * @param tau      Delay parameter (higher = slower initial decay)
 * @param kappa    Decay exponent (0.5 < κ ≤ 1.0, typically 0.7)
 */
void online_vi_set_lr_robbins_monro(OnlineVI *vi, double rho_0, 
                                     double tau, double kappa);

/**
 * @brief Set adaptive learning rate mode
 *
 * Learning rate is controlled externally (e.g., by HDP-beam feedback).
 * Use online_vi_set_rho() to update.
 *
 * @param vi        Online VI structure
 * @param rho_init  Initial learning rate
 */
void online_vi_set_lr_adaptive(OnlineVI *vi, double rho_init);

/**
 * @brief Directly set current learning rate (for adaptive mode)
 *
 * @param vi    Online VI structure
 * @param rho   New learning rate (clamped to [rho_min, rho_max])
 */
void online_vi_set_rho(OnlineVI *vi, double rho);

/**
 * @brief Set learning rate bounds
 *
 * @param vi       Online VI structure
 * @param rho_min  Minimum learning rate (default 0.001)
 * @param rho_max  Maximum learning rate (default 1.0)
 */
void online_vi_set_rho_bounds(OnlineVI *vi, double rho_min, double rho_max);

/*═══════════════════════════════════════════════════════════════════════════════
 * CORE UPDATE
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Update with soft ξ responsibility (MAIN FUNCTION)
 *
 * Computes ξ_ij = P(s_{t-1}=i, s_t=j | y_{1:t}) and performs natural gradient
 * update on the Dirichlet posterior.
 *
 * @param vi             Online VI structure
 * @param regime_probs   Current regime probabilities p_t(j) [K]
 *                       (from RBPF output)
 * @param regime_liks    Per-regime observation likelihoods ℓ_t(j) [K]
 *                       (NOT log-likelihoods)
 *
 * NOTE: On first call, regime_probs is stored. Update starts on 2nd call.
 */
void online_vi_update(OnlineVI *vi,
                       const double *regime_probs,
                       const double *regime_liks);

/**
 * @brief Update with pre-computed ξ matrix
 *
 * Use this if you've already computed the soft transition responsibilities.
 *
 * @param vi    Online VI structure
 * @param xi    Soft transition matrix [K × K], xi[i][j] = P(s_{t-1}=i, s_t=j)
 */
void online_vi_update_with_xi(OnlineVI *vi, const double *xi);

/**
 * @brief Update with log-likelihoods (convenience wrapper)
 *
 * @param vi              Online VI structure
 * @param regime_probs    Current regime probabilities [K]
 * @param log_regime_liks Log-likelihoods per regime [K]
 */
void online_vi_update_log(OnlineVI *vi,
                           const double *regime_probs,
                           const double *log_regime_liks);

/*═══════════════════════════════════════════════════════════════════════════════
 * POSTERIOR QUERIES
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Get posterior mean transition matrix
 *
 * E[π_ij] = α̃_ij / Σ_k α̃_ik
 *
 * @param vi     Online VI structure
 * @param trans  Output: [K × K] row-major transition matrix
 */
void online_vi_get_mean(OnlineVI *vi, double *trans);

/**
 * @brief Get posterior variance matrix
 *
 * Var[π_ij] = α̃_ij(α̃_i0 - α̃_ij) / (α̃_i0² × (α̃_i0 + 1))
 *
 * @param vi    Online VI structure
 * @param var   Output: [K × K] row-major variance matrix
 */
void online_vi_get_variance(OnlineVI *vi, double *var);

/**
 * @brief Get posterior standard deviation matrix
 *
 * @param vi     Online VI structure
 * @param std    Output: [K × K] row-major std matrix
 */
void online_vi_get_std(OnlineVI *vi, double *std);

/**
 * @brief Get single transition probability
 *
 * @param vi    Online VI structure
 * @param from  Source regime
 * @param to    Target regime
 * @return      E[π_{from,to}]
 */
double online_vi_get_prob(OnlineVI *vi, int from, int to);

/**
 * @brief Get single transition variance
 *
 * @param vi    Online VI structure
 * @param from  Source regime
 * @param to    Target regime
 * @return      Var[π_{from,to}]
 */
double online_vi_get_prob_var(OnlineVI *vi, int from, int to);

/**
 * @brief Get transition row
 *
 * @param vi    Online VI structure
 * @param from  Source regime
 * @param row   Output: [K] transition probabilities from this regime
 */
void online_vi_get_row(OnlineVI *vi, int from, double *row);

/**
 * @brief Get self-transition probability (stickiness)
 *
 * @param vi      Online VI structure
 * @param regime  Regime index
 * @return        E[π_{regime,regime}]
 */
double online_vi_get_stickiness(OnlineVI *vi, int regime);

/*═══════════════════════════════════════════════════════════════════════════════
 * ENTROPY & UNCERTAINTY
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Get per-row entropy
 *
 * H[q(π_i)] = entropy of Dirichlet posterior over transitions from regime i
 *
 * Higher entropy = more uncertainty about transitions from this regime.
 * Useful for triggering HDP-beam: if specific row entropy spikes, that row
 * needs refinement.
 *
 * @param vi    Online VI structure
 * @param H     Output: [K] per-row entropies
 */
void online_vi_get_row_entropy(OnlineVI *vi, double *H);

/**
 * @brief Get total entropy
 *
 * Sum of per-row entropies. Global measure of transition uncertainty.
 *
 * @param vi    Online VI structure
 * @return      Σ_i H[q(π_i)]
 */
double online_vi_get_total_entropy(OnlineVI *vi);

/**
 * @brief Get maximum row entropy
 *
 * Identifies the most uncertain transition row.
 *
 * @param vi          Online VI structure
 * @param max_row     Output: index of row with highest entropy (can be NULL)
 * @return            Maximum row entropy
 */
double online_vi_get_max_row_entropy(OnlineVI *vi, int *max_row);

/**
 * @brief Get confidence score
 *
 * Returns 1 - (H / H_max) where H_max is entropy of uniform Dirichlet.
 * Useful for Kelly sizing: kelly_fraction × confidence.
 *
 * @param vi    Online VI structure
 * @return      Confidence in [0, 1]
 */
double online_vi_get_confidence(OnlineVI *vi);

/*═══════════════════════════════════════════════════════════════════════════════
 * DIAGNOSTICS
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Get KL divergence from prior
 *
 * KL(q(π) || p(π)) - measures how far posterior has moved from prior.
 *
 * @param vi    Online VI structure
 * @return      KL divergence (nats)
 */
double online_vi_get_kl_from_prior(OnlineVI *vi);

/**
 * @brief Get diagnostic statistics
 *
 * @param vi     Online VI structure
 * @param stats  Output: diagnostic snapshot
 */
void online_vi_get_stats(OnlineVI *vi, OnlineVI_Stats *stats);

/**
 * @brief Get current learning rate
 *
 * @param vi    Online VI structure
 * @return      Current ρ_t
 */
double online_vi_get_rho(const OnlineVI *vi);

/**
 * @brief Get effective sample size per row
 *
 * ESS_i = Σ_j α̃_ij - K (concentration minus prior)
 *
 * @param vi    Online VI structure
 * @param ess   Output: [K] effective sample sizes
 */
void online_vi_get_row_ess(const OnlineVI *vi, double *ess);

/*═══════════════════════════════════════════════════════════════════════════════
 * INTEGRATION WITH HDP-BEAM
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Reset from HDP-beam posterior
 *
 * Called after HDP-beam sweep to re-anchor the VI posterior.
 *
 * @param vi           Online VI structure
 * @param hdp_trans    HDP transition matrix [K × K]
 * @param confidence   How much to trust HDP (higher = stronger prior)
 */
void online_vi_reset_from_hdp(OnlineVI *vi, const double *hdp_trans, 
                               double confidence);

/**
 * @brief Adjust learning rate based on HDP correction magnitude
 *
 * If HDP made large corrections, increase ρ (world changed).
 * If HDP made small corrections, decrease ρ (world is stable).
 *
 * @param vi              Online VI structure
 * @param correction_kl   KL divergence between old and new transition matrices
 * @param sensitivity     How much to adjust (0.1 = 10% change per unit KL)
 */
void online_vi_adapt_rho_from_hdp(OnlineVI *vi, double correction_kl, 
                                   double sensitivity);

/*═══════════════════════════════════════════════════════════════════════════════
 * UTILITY
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Print current state (for debugging)
 */
void online_vi_print(const OnlineVI *vi);

/**
 * @brief Print transition matrix with uncertainties
 */
void online_vi_print_matrix(OnlineVI *vi);

#ifdef __cplusplus
}
#endif

#endif /* ONLINE_VI_TRANSITION_H */
