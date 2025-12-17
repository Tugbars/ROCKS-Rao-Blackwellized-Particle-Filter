/**
 * @file soft_dirichlet_transition.h
 * @brief Soft Dirichlet Transition Learning with ξ Updates
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * OVERVIEW
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * Bayesian online learning of transition matrix A using the exact posterior
 * transition responsibility ξ_t(i,j) every tick — NOT discrete events.
 *
 * KEY INSIGHT:
 *   Instead of waiting for "confirmed" regime changes (which requires heuristics),
 *   we update with the Bayes-consistent joint posterior over transitions:
 *
 *     ξ_t(i,j) = P(r_{t-1} = i, r_t = j | y_{1:t})
 *
 *   This is not a heuristic — it's the exact Bayesian responsibility.
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * THE MATH
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * STEP 1: Compute transition responsibility
 *
 *   ξ̃_t(i,j) = p_{t-1}(i) × A[i][j] × ℓ_t(j)
 *   ξ_t(i,j) = ξ̃_t(i,j) / Σ_{i',j'} ξ̃_t(i',j')
 *
 *   Where:
 *     p_{t-1}(i) = P(r_{t-1} = i | y_{1:t-1})  — regime prob at t-1
 *     A[i][j]    = current transition matrix
 *     ℓ_t(j)     = P(y_t | r_t = j)            — observation likelihood
 *
 * STEP 2: Update Dirichlet pseudo-counts
 *
 *   α[i][j] ← γ · α[i][j] + κ · ξ_t(i,j)
 *
 *   Where:
 *     γ = discount factor (memory decay)
 *     κ = learning rate (typically 1.0)
 *
 * STEP 3: Row ESS capping (controls effective memory)
 *
 *   ESS_i = Σ_j α[i][j]
 *   if ESS_i > ESS_max:
 *       α[i,:] *= ESS_max / ESS_i
 *
 * STEP 4: Compute posterior mean
 *
 *   A[i][j] = α[i][j] / Σ_k α[i][k]
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * WHY THIS ELIMINATES HEURISTICS
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * Problem                    | Event-Based Dirichlet | Soft ξ Dirichlet
 * ---------------------------|----------------------|------------------
 * "What is a transition?"    | SPRT threshold       | ELIMINATED
 * Choppy periods             | Forced hard switch   | Fractional updates
 * Tuning knobs               | 3 (interact badly)   | 1 (ESS_max)
 * SPRT dependency            | Required             | NONE
 * Theoretical basis          | Heuristic            | Bayes-consistent
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * SINGLE-KNOB CONFIGURATION
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * The only parameter you need to set is ESS_max:
 *
 *   ESS_max = 200   → ~200 ticks of effective memory
 *   ESS_max = 500   → ~500 ticks of effective memory (more stable)
 *   ESS_max = 100   → ~100 ticks of effective memory (more reactive)
 *
 * With ESS capping, you can set γ = 1.0 (no decay) and κ = 1.0 (full ξ).
 * The ESS cap alone controls adaptivity.
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * USAGE
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 *   // Initialize with 4 regimes, ESS_max = 200
 *   SoftDirichlet sd;
 *   soft_dirichlet_init(&sd, 4, 200.0f);
 *
 *   // Optional: warm-start from existing transition matrix
 *   soft_dirichlet_init_from_matrix(&sd, 4, initial_trans, 200.0f);
 *
 *   // In RBPF hot loop (after computing regime_probs and likelihoods):
 *   soft_dirichlet_update(&sd, prev_regime_probs, regime_liks);
 *
 *   // Rebuild RBPF transition LUT
 *   soft_dirichlet_rebuild_lut(&sd, rbpf->trans_lut, 4);
 *
 *   // Query current probabilities
 *   float p = soft_dirichlet_prob(&sd, from, to);
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 */

#ifndef SOFT_DIRICHLET_TRANSITION_H
#define SOFT_DIRICHLET_TRANSITION_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*═══════════════════════════════════════════════════════════════════════════════
 * CONFIGURATION
 *═══════════════════════════════════════════════════════════════════════════════*/

#ifndef SOFT_DIRICHLET_MAX_REGIMES
#define SOFT_DIRICHLET_MAX_REGIMES 8
#endif

/*═══════════════════════════════════════════════════════════════════════════════
 * CORE STRUCTURE
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * Soft Dirichlet Transition Learner
 *
 * Learns transition matrix online using Bayes-consistent ξ updates.
 */
typedef struct {
    int n_regimes;                                          /**< Number of regimes */
    
    /* Dirichlet parameters */
    float alpha[SOFT_DIRICHLET_MAX_REGIMES][SOFT_DIRICHLET_MAX_REGIMES];
                                                            /**< Pseudo-counts */
    float prob[SOFT_DIRICHLET_MAX_REGIMES][SOFT_DIRICHLET_MAX_REGIMES];
                                                            /**< Posterior mean E[A] */
    
    /* Configuration */
    float gamma;                                            /**< Discount factor */
    float kappa;                                            /**< Learning rate */
    float ess_max;                                          /**< Row ESS cap */
    float alpha_floor;                                      /**< Min pseudo-count */
    
    /* State from previous tick */
    float prev_probs[SOFT_DIRICHLET_MAX_REGIMES];          /**< p_{t-1}(i) */
    int initialized;                                        /**< Have we seen a tick? */
    
    /* Diagnostics */
    uint64_t total_updates;                                 /**< Number of updates */
    float row_ess[SOFT_DIRICHLET_MAX_REGIMES];             /**< Current row ESS */
    float last_xi_entropy;                                  /**< Entropy of last ξ */
    
} SoftDirichlet;

/**
 * Diagnostic statistics
 */
typedef struct {
    float avg_stickiness;           /**< Average self-transition prob */
    float avg_row_ess;              /**< Average row ESS */
    float min_row_ess;              /**< Minimum row ESS */
    float max_row_ess;              /**< Maximum row ESS */
    float xi_entropy;               /**< Entropy of last ξ (uncertainty) */
    uint64_t total_updates;         /**< Total updates performed */
} SoftDirichletStats;

/*═══════════════════════════════════════════════════════════════════════════════
 * LIFECYCLE
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Initialize with uniform prior
 *
 * Sets up a uniform transition matrix with specified ESS_max.
 * Uses single-knob configuration: γ=1.0, κ=1.0, ESS capping handles adaptivity.
 *
 * @param sd         Soft Dirichlet structure
 * @param n_regimes  Number of regimes (1-8)
 * @param ess_max    Row ESS cap (effective memory in ticks)
 */
void soft_dirichlet_init(SoftDirichlet *sd, int n_regimes, float ess_max);

/**
 * @brief Initialize from existing transition matrix
 *
 * Warm-starts the Dirichlet from a known transition matrix.
 * Useful for starting from a hand-tuned or previously learned matrix.
 *
 * @param sd          Soft Dirichlet structure
 * @param n_regimes   Number of regimes
 * @param trans       Initial transition matrix [n_regimes × n_regimes] row-major
 * @param ess_max     Row ESS cap
 */
void soft_dirichlet_init_from_matrix(SoftDirichlet *sd, int n_regimes,
                                      const float *trans, float ess_max);

/**
 * @brief Reset to initial state (keep configuration)
 */
void soft_dirichlet_reset(SoftDirichlet *sd);

/*═══════════════════════════════════════════════════════════════════════════════
 * CONFIGURATION
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Set ESS cap (the main tuning knob)
 *
 * @param sd        Soft Dirichlet structure
 * @param ess_max   Row ESS cap
 *                  - 100: reactive (~100 tick memory)
 *                  - 200: balanced (~200 tick memory)
 *                  - 500: stable (~500 tick memory)
 */
void soft_dirichlet_set_ess_max(SoftDirichlet *sd, float ess_max);

/**
 * @brief Set advanced parameters (usually not needed)
 *
 * @param sd      Soft Dirichlet structure
 * @param gamma   Discount factor (default 1.0 with ESS capping)
 * @param kappa   Learning rate (default 1.0)
 */
void soft_dirichlet_set_params(SoftDirichlet *sd, float gamma, float kappa);

/**
 * @brief Set minimum pseudo-count floor
 *
 * Prevents any α[i][j] from going to zero, ensuring all transitions remain possible.
 *
 * @param sd          Soft Dirichlet structure
 * @param alpha_floor Minimum pseudo-count (default 0.01)
 */
void soft_dirichlet_set_floor(SoftDirichlet *sd, float alpha_floor);

/*═══════════════════════════════════════════════════════════════════════════════
 * CORE UPDATE
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Update with soft ξ responsibility (MAIN FUNCTION)
 *
 * Call this every tick after computing regime probabilities and likelihoods.
 *
 * @param sd            Soft Dirichlet structure
 * @param regime_probs  Current regime probabilities p_t(j) [n_regimes]
 *                      (from RBPF output: out->regime_probs)
 * @param regime_liks   Per-regime observation likelihoods ℓ_t(j) [n_regimes]
 *                      (NOT log-likelihoods — actual probabilities)
 *
 * NOTE: On first call, regime_probs is stored for next tick. Update starts on 2nd call.
 */
void soft_dirichlet_update(SoftDirichlet *sd,
                           const float *regime_probs,
                           const float *regime_liks);

/**
 * @brief Update with log-likelihoods (convenience wrapper)
 *
 * Same as soft_dirichlet_update but takes log-likelihoods and converts internally.
 *
 * @param sd             Soft Dirichlet structure
 * @param regime_probs   Current regime probabilities [n_regimes]
 * @param log_regime_liks Log-likelihoods log(ℓ_t(j)) [n_regimes]
 */
void soft_dirichlet_update_log(SoftDirichlet *sd,
                                const float *regime_probs,
                                const float *log_regime_liks);

/*═══════════════════════════════════════════════════════════════════════════════
 * QUERIES
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Get transition probability P(to | from)
 */
float soft_dirichlet_prob(const SoftDirichlet *sd, int from, int to);

/**
 * @brief Get full transition matrix
 *
 * @param sd    Soft Dirichlet structure
 * @param trans Output matrix [n_regimes × n_regimes] row-major
 */
void soft_dirichlet_get_matrix(const SoftDirichlet *sd, float *trans);

/**
 * @brief Get row of transition matrix (transitions from regime 'from')
 *
 * @param sd    Soft Dirichlet structure
 * @param from  Source regime
 * @param row   Output array [n_regimes]
 */
void soft_dirichlet_get_row(const SoftDirichlet *sd, int from, float *row);

/**
 * @brief Get self-transition (stickiness) probability for regime
 */
float soft_dirichlet_stickiness(const SoftDirichlet *sd, int regime);

/**
 * @brief Get diagnostic statistics
 */
SoftDirichletStats soft_dirichlet_stats(const SoftDirichlet *sd);

/*═══════════════════════════════════════════════════════════════════════════════
 * RBPF INTEGRATION
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Rebuild RBPF transition LUT from learned probabilities
 *
 * The RBPF uses a uint8_t[regime][1024] lookup table for fast transition sampling.
 * This function rebuilds it from the current Dirichlet posterior.
 *
 * @param sd        Soft Dirichlet structure
 * @param trans_lut RBPF transition LUT [n_regimes][1024]
 * @param n_regimes Number of regimes
 */
void soft_dirichlet_rebuild_lut(const SoftDirichlet *sd,
                                 uint8_t trans_lut[][1024],
                                 int n_regimes);

/**
 * @brief Export to RBPF-compatible float matrix
 *
 * @param sd        Soft Dirichlet structure
 * @param trans     Output matrix [n_regimes × n_regimes] row-major (float)
 * @param n_regimes Number of regimes
 */
void soft_dirichlet_export_matrix(const SoftDirichlet *sd,
                                   float *trans, int n_regimes);

/*═══════════════════════════════════════════════════════════════════════════════
 * DIAGNOSTICS
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Print current state
 */
void soft_dirichlet_print(const SoftDirichlet *sd);

/**
 * @brief Print transition matrix
 */
void soft_dirichlet_print_matrix(const SoftDirichlet *sd);

/**
 * @brief Print pseudo-counts (α)
 */
void soft_dirichlet_print_alpha(const SoftDirichlet *sd);

#ifdef __cplusplus
}
#endif

#endif /* SOFT_DIRICHLET_TRANSITION_H */
