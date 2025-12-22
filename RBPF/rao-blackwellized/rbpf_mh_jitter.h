/**
 * @file rbpf_mh_jitter.h
 * @brief Metropolis-Hastings Jittering for Particle Diversity
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * THE PROBLEM: PARTICLE WASTE
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * After resampling in stable markets:
 *   - 500 particles, 450 are copies of same ancestor
 *   - Silverman noise adds blind Gaussian jitter
 *   - Particles cluster instead of exploring likelihood surface
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * THE SOLUTION: INFORMED JITTERING
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * After Silverman, apply MH step:
 *   1. Propose: h' = h + ε, where ε ~ N(0, σ²)
 *   2. If h' crosses regime boundary → reject (preserve Storvik stats)
 *   3. Accept with prob: min(1, p(y|h') / p(y|h))
 *
 * Result: Particles spread along likelihood ridge, not randomly.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * STORVIK SAFETY
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * By rejecting cross-regime proposals, we ensure:
 *   - Particle stays in same regime
 *   - Sufficient statistics remain valid
 *   - No collision with Storvik learning
 */

#ifndef RBPF_MH_JITTER_H
#define RBPF_MH_JITTER_H

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief MH Jittering configuration
 */
typedef struct {
    float proposal_scale;      /**< Scale of proposal distribution (relative to Silverman) */
    int max_attempts;          /**< Max proposals per particle before giving up */
    bool enabled;              /**< Master enable switch */
    
    /* Diagnostics */
    int total_proposals;       /**< Total proposals made */
    int total_accepts;         /**< Total accepts */
    int total_boundary_rejects;/**< Rejected due to regime crossing */
    int total_likelihood_rejects; /**< Rejected due to likelihood */
    float avg_accept_ratio;    /**< Running average of acceptance probability */
} MH_Jitter_Config;

/**
 * @brief Initialize MH jittering with defaults
 *
 * Defaults:
 *   - proposal_scale = 1.0 (same as Silverman bandwidth)
 *   - max_attempts = 3
 *   - enabled = true
 */
void mh_jitter_init(MH_Jitter_Config *cfg);

/**
 * @brief Reset diagnostics counters
 */
void mh_jitter_reset_stats(MH_Jitter_Config *cfg);

/**
 * @brief Apply MH jittering to all particles
 *
 * Call this AFTER Silverman noise, BEFORE weight computation.
 *
 * @param cfg           Configuration and diagnostics
 * @param mu            Particle states [n] (modified in place)
 * @param regime        Particle regimes [n] (read only)
 * @param n             Number of particles
 * @param y_obs         Current observation (for likelihood)
 * @param silverman_h   Silverman bandwidth (proposal scale base)
 * @param regime_bounds Regime boundaries [n_regimes + 1] (e.g., [-6, -4.5, -3.5, -2.5, 0])
 * @param n_regimes     Number of regimes
 * @param rng_stream    MKL random stream (for proposals)
 */
void mh_jitter_apply(MH_Jitter_Config *cfg,
                     float *mu,
                     const int *regime,
                     int n,
                     float y_obs,
                     float silverman_h,
                     const float *regime_bounds,
                     int n_regimes,
                     void *rng_stream);

/**
 * @brief Print MH jittering statistics
 */
void mh_jitter_print_stats(const MH_Jitter_Config *cfg);

/**
 * @brief Get acceptance rate (0 to 1)
 */
float mh_jitter_acceptance_rate(const MH_Jitter_Config *cfg);

#ifdef __cplusplus
}
#endif

#endif /* RBPF_MH_JITTER_H */
