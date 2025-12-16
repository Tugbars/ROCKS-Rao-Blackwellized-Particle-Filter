/**
 * @file mmpf_entropy.h
 * @brief Entropy-Based Stability Detection for Shock Recovery
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * THE PROBLEM: FIXED RECOVERY TIMER
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * After shock injection, we wait a fixed number of ticks:
 *
 *   if (ticks_since_shock > 20) restore_transitions();
 *
 * Problems:
 *   - Flash crashes settle in 2-3 ticks → waiting 20 wastes accuracy
 *   - Regime shifts need 50+ ticks → unlocking at 20 causes flicker
 *   - Time ≠ Information
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * THE SOLUTION: SHANNON ENTROPY
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Shannon entropy measures disorder in particle distribution:
 *
 *   H = -Σ wᵢ × log(wᵢ)
 *
 * High entropy (H → log(N)): Particles disagree; filter uncertain
 *   → Keep transitions uniform (explore all regimes)
 *
 * Low entropy (H → 0): Particles converged; filter confident
 *   → Restore sticky transitions (exploit knowledge)
 *
 * The key insight: track CHANGE in entropy, not absolute value.
 * When entropy fluctuation dies down, we've reached equilibrium.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * TWO-LEVEL ENTROPY
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Optionally track entropy at two levels:
 *
 * 1. Model entropy: Uncertainty about which regime
 *    H_model = -Σ model_weight[k] × log(model_weight[k])
 *
 * 2. Particle entropy: Uncertainty within each model
 *    H_particle[k] = -Σ w_norm[i] × log(w_norm[i])
 *
 * Unlock when BOTH stabilize. Prevents edge cases where models
 * are certain but particles are scattered (or vice versa).
 */

#ifndef MMPF_ENTROPY_H
#define MMPF_ENTROPY_H

#include "mmpf_rocks.h"

#ifdef __cplusplus
extern "C" {
#endif

/*═══════════════════════════════════════════════════════════════════════════
 * CONFIGURATION
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Entropy-based stability detector configuration
 */
typedef struct {
    /* Stability threshold: unlock when delta_H_ema < threshold */
    double stability_threshold;
    
    /* EMA smoothing for delta computation */
    double delta_ema_alpha;     /**< EMA weight (default: 0.3 for fast response) */
    
    /* Safety bounds */
    int min_shock_duration;     /**< Minimum ticks before unlock allowed */
    int max_shock_duration;     /**< Maximum ticks before forced unlock */
    
    /* Two-level entropy (optional) */
    int use_two_level;          /**< 1 = track model + particle entropy separately */
} MMPF_EntropyConfig;

/**
 * @brief Entropy detector state
 */
typedef struct {
    /* Configuration */
    MMPF_EntropyConfig config;
    
    /* State */
    double entropy_prev;        /**< Previous tick's entropy */
    double delta_ema;           /**< EMA of |H_t - H_{t-1}| */
    int ticks_since_shock;      /**< Ticks since shock injection */
    int is_locked;              /**< 1 = in shock state (transitions uniform) */
    
    /* Two-level state */
    double model_entropy;       /**< H over model weights */
    double particle_entropy;    /**< Average H over particles within models */
    double model_delta_ema;     /**< EMA of model entropy change */
    double particle_delta_ema;  /**< EMA of particle entropy change */
    
    /* Statistics */
    int total_shocks;
    int total_ticks_locked;
    double avg_lock_duration;
} MMPF_EntropyState;

/**
 * @brief Initialize entropy detector with defaults
 */
void mmpf_entropy_init(MMPF_EntropyState *state);

/**
 * @brief Initialize with custom configuration
 */
void mmpf_entropy_init_config(MMPF_EntropyState *state, const MMPF_EntropyConfig *cfg);

/**
 * @brief Notify detector that shock was injected
 *
 * Call this when MCMC shock is applied.
 */
void mmpf_entropy_shock_injected(MMPF_EntropyState *state);

/**
 * @brief Update entropy and check stability
 *
 * @param state   Entropy state
 * @param mmpf    MMPF instance (to compute entropy from weights)
 * @return        1 if stable (unlock), 0 if unstable (keep locked)
 *
 * Call this every tick after shock injection.
 */
int mmpf_entropy_check_stability(MMPF_EntropyState *state, const MMPF_ROCKS *mmpf);

/**
 * @brief Calculate current entropy
 *
 * @param mmpf   MMPF instance
 * @return       Normalized entropy [0, 1]
 *
 * Computes entropy over all particles across all models.
 * Normalized by log(N) so maximum is 1.0.
 */
double mmpf_calculate_entropy(const MMPF_ROCKS *mmpf);

/**
 * @brief Calculate model-level entropy
 *
 * @param mmpf   MMPF instance
 * @return       Entropy over model weights [0, log(n_models)]
 */
double mmpf_calculate_model_entropy(const MMPF_ROCKS *mmpf);

/**
 * @brief Calculate average particle entropy
 *
 * @param mmpf   MMPF instance
 * @return       Average entropy within each model
 */
double mmpf_calculate_particle_entropy(const MMPF_ROCKS *mmpf);

/**
 * @brief Get current entropy value
 */
double mmpf_entropy_get_current(const MMPF_EntropyState *state);

/**
 * @brief Get entropy delta EMA
 */
double mmpf_entropy_get_delta_ema(const MMPF_EntropyState *state);

/**
 * @brief Check if currently in locked (shock) state
 */
int mmpf_entropy_is_locked(const MMPF_EntropyState *state);

/**
 * @brief Get ticks since shock
 */
int mmpf_entropy_ticks_since_shock(const MMPF_EntropyState *state);

/**
 * @brief Force unlock (external override)
 */
void mmpf_entropy_force_unlock(MMPF_EntropyState *state);

/**
 * @brief Reset statistics
 */
void mmpf_entropy_reset_stats(MMPF_EntropyState *state);

#ifdef __cplusplus
}
#endif

#endif /* MMPF_ENTROPY_H */
