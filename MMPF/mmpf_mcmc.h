/**
 * @file mmpf_mcmc.h
 * @brief MCMC Move Step for Shock Response - Particle Teleportation
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * THE PROBLEM: SHARK FIN LAG
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * When a flash crash occurs, true volatility teleports from 1% to 80% instantly.
 * But particles "walk" via state transitions:
 *
 *   h_{t+1} = μ + φ(h_t - μ) + η
 *
 * Even with φ=0.99, walking from h=-5 to h=0 takes ~50 ticks.
 * This creates the "shark fin" ramp-up artifact - massive RMSE.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * THE SOLUTION: MCMC TELEPORTATION
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * When BOCPD fires (changepoint detected):
 *
 * 1. PAUSE the filter
 * 2. Run Metropolis-Hastings on each particle:
 *    - Propose new position
 *    - Accept if likelihood improves
 *    - Repeat for K steps
 * 3. Reset variance to high uncertainty (var = 1.0)
 * 4. Flatten transition matrix (uniform)
 * 5. RESUME normal filtering
 *
 * Particles CLIMB the likelihood gradient and land at the new truth.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * KEY DESIGN DECISIONS
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Q: Won't all models converge to the same location?
 * A: Yes, temporarily. But the PREDICT step immediately separates them:
 *    - Calm (φ=0.9, μ=-5): Pulls hard toward low vol → h ≈ 4.0
 *    - Crisis (φ=0.99, μ=-2): Stays near shock → h ≈ 4.93
 *    The model matching subsequent data wins.
 *
 * Q: Isn't using y_t twice "data incest"?
 * A: In structural breaks, the prior P(x_t|x_{t-1}) is INVALID.
 *    MCMC re-initializes based on likelihood alone. The var=1.0 reset
 *    prevents overconfidence.
 *
 * Q: What if BOCPD false positive?
 * A: High variance gives high Kalman gain → filter snaps back on next tick.
 *    Cost: ~3-5 ticks of wide confidence intervals.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * PERFORMANCE
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Optimizations:
 * - MKL batch RNG: One call for all random numbers
 * - Pre-computed log(uniform) for acceptance
 * - AVX2 vectorization: 4 particles per iteration
 * - Branchless accept/reject via blendv
 *
 * Budget: <500μs for 768 particles × 5 MCMC steps
 */

#ifndef MMPF_MCMC_H
#define MMPF_MCMC_H

#include "mmpf_rocks.h"

/* Forward declaration for MKL VSL stream */
#ifdef USE_MKL
#include <mkl_vsl_types.h>
#else
typedef void *VSLStreamStatePtr; /* Stub for non-MKL builds */
#endif

#ifdef __cplusplus
extern "C"
{
#endif

    /*═══════════════════════════════════════════════════════════════════════════
     * CONFIGURATION
     *═══════════════════════════════════════════════════════════════════════════*/

    /**
     * @brief MCMC shock configuration
     */
    typedef struct
    {
        int n_steps;             /**< Number of MH iterations per particle (default: 5) */
        double proposal_sigma;   /**< Random walk step size (default: 1.5) */
        double var_reset;        /**< Variance reset value (default: 1.0) */
        double min_log_vol;      /**< Floor for log-vol (default: -10) */
        double max_log_vol;      /**< Ceiling for log-vol (default: 2) */
        int flatten_transitions; /**< 1 = set transitions to uniform */
    } MMPF_MCMC_Config;

    /**
     * @brief Initialize MCMC config with defaults
     */
    void mmpf_mcmc_config_defaults(MMPF_MCMC_Config *cfg);

    /*═══════════════════════════════════════════════════════════════════════════
     * CORE API
     *═══════════════════════════════════════════════════════════════════════════*/

    /**
     * @brief Inject shock with MCMC move step
     *
     * This is the main entry point. Call when BOCPD detects changepoint.
     *
     * @param mmpf       MMPF instance
     * @param y_log_sq   Observed log(return²) - the shock observation
     * @param cfg        Configuration (NULL for defaults)
     *
     * Effects:
     * - All particles across all models teleport to likelihood peak
     * - Kalman variance reset to high uncertainty
     * - Transition matrix flattened (if configured)
     * - shock_active flag set
     *
     * Call sequence:
     *   1. BOCPD detects shock
     *   2. mmpf_inject_shock_mcmc(mmpf, y_log_sq, NULL)
     *   3. mmpf_step(mmpf, y, &output)  ← normal step, predict will differentiate
     *   4. mmpf_restore_from_shock(mmpf) when entropy stabilizes
     */
    void mmpf_inject_shock_mcmc(MMPF_ROCKS *mmpf, double y_log_sq,
                                const MMPF_MCMC_Config *cfg);

    /**
     * @brief AVX2+MKL optimized version
     *
     * Uses:
     * - MKL vdRngGaussian for batch random numbers
     * - Pre-computed log(uniform) for acceptance
     * - AVX2 vectorized likelihood and accept/reject
     *
     * Falls back to scalar version if AVX2 not available.
     */
    void mmpf_inject_shock_mcmc_avx(MMPF_ROCKS *mmpf, double y_log_sq,
                                    const MMPF_MCMC_Config *cfg);

    /*═══════════════════════════════════════════════════════════════════════════
     * SCRATCH BUFFER MANAGEMENT
     *
     * INTEGRATION REQUIREMENT:
     * Add these fields to your MMPF_ROCKS struct:
     *
     *   typedef struct MMPF_ROCKS {
     *       // ... existing fields ...
     *
     *       MMPF_MCMC_Scratch mcmc_scratch;      // Pre-allocated RNG buffers
     *       VSLStreamStatePtr mcmc_vsl_stream;   // MKL RNG stream (NULL OK, will create temp)
     *       MMPF_MCMC_Stats   mcmc_stats;        // Diagnostic statistics
     *   } MMPF_ROCKS;
     *
     * At MMPF initialization, call:
     *   mmpf_mcmc_init(mmpf, n_particles, MMPF_N_MODELS, 5);  // 5 MCMC steps
     *
     * At MMPF destruction, call:
     *   mmpf_mcmc_destroy(mmpf);
     *═══════════════════════════════════════════════════════════════════════════*/

    /**
     * @brief Initialize MCMC subsystem for an MMPF instance
     *
     * Call once at MMPF creation. Pre-allocates scratch buffers.
     *
     * @param mmpf      MMPF instance
     * @param n_part    Number of particles per model
     * @param n_models  Number of models
     * @param n_steps   MCMC steps per shock (typically 5)
     */
    void mmpf_mcmc_init(MMPF_ROCKS *mmpf, int n_part, int n_models, int n_steps);

    /**
     * @brief Destroy MCMC subsystem, free scratch buffers
     *
     * Call at MMPF destruction.
     */
    void mmpf_mcmc_destroy(MMPF_ROCKS *mmpf);

    /*═══════════════════════════════════════════════════════════════════════════
     * LIKELIHOOD EVALUATION
     *═══════════════════════════════════════════════════════════════════════════*/

    /**
     * @brief Log-likelihood for KSC observation model
     *
     * P(y | h) where y = log(r²), h = log-volatility
     * Uses log-χ² approximation (single dominant component for speed).
     *
     * @param y_log_sq   Observed log(return²)
     * @param h          Log-volatility state
     * @return           Log-likelihood (unnormalized, for ratios)
     */
    double mmpf_mcmc_loglik(double y_log_sq, double h);

/**
 * @brief AVX2 vectorized likelihood for 4 particles
 *
 * @param h_vec      AVX2 vector of 4 log-vol states
 * @param y_log_sq   Observed log(return²) (broadcast)
 * @return           AVX2 vector of 4 log-likelihoods
 */
#ifdef __AVX2__
#include <immintrin.h>
    __m256d mmpf_mcmc_loglik_avx(__m256d h_vec, double y_log_sq);
#endif

    /*═══════════════════════════════════════════════════════════════════════════
     * DIAGNOSTICS
     *═══════════════════════════════════════════════════════════════════════════*/

    /**
     * @brief MCMC shock statistics
     */
    typedef struct
    {
        int total_shocks;      /**< Number of shocks injected */
        int total_proposals;   /**< Total proposals across all shocks */
        int total_accepts;     /**< Total accepts */
        double avg_acceptance; /**< Running average acceptance rate */
        double last_pre_mean;  /**< Mean log-vol before last shock */
        double last_post_mean; /**< Mean log-vol after last shock */
        double last_teleport;  /**< Distance teleported (post - pre) */
    } MMPF_MCMC_Stats;

    /**
     * @brief Get MCMC statistics
     */
    void mmpf_mcmc_get_stats(const MMPF_ROCKS *mmpf, MMPF_MCMC_Stats *stats);

    /**
     * @brief Reset MCMC statistics
     */
    void mmpf_mcmc_reset_stats(MMPF_ROCKS *mmpf);

#ifdef __cplusplus
}
#endif

#endif /* MMPF_MCMC_H */