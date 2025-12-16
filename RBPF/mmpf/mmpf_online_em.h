/**
 * @file mmpf_online_em.h
 * @brief Structural Hierarchical Estimation for Regime Discovery
 *
 * Instead of learning 3 independent μ values (which collapse),
 * we learn 2 structural parameters:
 *
 *   B = Base Level (the "tide" that lifts all boats)
 *   S = Spread (the distance between regimes)
 *
 * Model positions are DERIVED from structure:
 *   μ_calm   = B + coeff[0] × S  = B - 1.0×S
 *   μ_trend  = B + coeff[1] × S  = B + 0.0×S = B
 *   μ_crisis = B + coeff[2] × S  = B + 1.5×S
 *
 * WHY THIS IS BULLETPROOF:
 *   - Zero collapse risk: S_min enforces minimum separation
 *   - Coupled learning: All data informs B, fleet stays in formation
 *   - Noise immunity: Even mushy weights find the centroid
 *   - Adaptive: B and S can both shift as market evolves
 */

#ifndef MMPF_ONLINE_EM_H
#define MMPF_ONLINE_EM_H

#include "rbpf_ksc.h"

#ifdef __cplusplus
extern "C"
{
#endif

    /*═══════════════════════════════════════════════════════════════════════════
     * CONSTANTS
     *═══════════════════════════════════════════════════════════════════════════*/

#define MMPF_EM_MAX_REGIMES 8

    /*═══════════════════════════════════════════════════════════════════════════
     * STRUCTURAL HIERARCHICAL ESTIMATION STATE
     *═══════════════════════════════════════════════════════════════════════════*/

    typedef struct
    {
        /*───────────────────────────────────────────────────────────────────────
         * THE PHYSICAL STATE
         * We only learn these two numbers. Everything else is derived.
         *───────────────────────────────────────────────────────────────────────*/
        double base_level; /**< B: The "Center of Gravity" of market vol */
        double spread;     /**< S: The "Distance" between regimes */

        /*───────────────────────────────────────────────────────────────────────
         * THE GEOMETRY (Fixed Design)
         * Defines the shape: Calm=-1.0, Trend=0.0, Crisis=1.5
         * This is structural: Crisis is ALWAYS higher than Calm.
         *───────────────────────────────────────────────────────────────────────*/
        int n_regimes;                            /**< Number of regimes (typically 3) */
        double coefficients[MMPF_EM_MAX_REGIMES]; /**< Position multipliers */

        /*───────────────────────────────────────────────────────────────────────
         * HYPERPARAMETERS (Physics, not Heuristics)
         *───────────────────────────────────────────────────────────────────────*/
        double lr_base;    /**< Speed of secular drift (e.g. 0.005) */
        double lr_spread;  /**< Speed of structural change (e.g. 0.001) */
        double min_spread; /**< Physical floor - PREVENTS COLLAPSE */
        double max_spread; /**< Physical ceiling - PREVENTS EXPLOSION */
        double min_base;   /**< Safety bound on base level */
        double max_base;   /**< Safety bound on base level */

        /*───────────────────────────────────────────────────────────────────────
         * OUTPUT
         * The projected means used by RBPF: mu[k] = B + coeff[k]*S
         *───────────────────────────────────────────────────────────────────────*/
        double mu[MMPF_EM_MAX_REGIMES];    /**< Projected regime centers */
        double sigma[MMPF_EM_MAX_REGIMES]; /**< Per-regime spreads (optional) */

        /*───────────────────────────────────────────────────────────────────────
         * DIAGNOSTICS
         *───────────────────────────────────────────────────────────────────────*/
        int tick_count;
        int warmup_ticks;
        double last_y;           /**< Last observation processed */
        double last_grad_base;   /**< Last gradient for base level */
        double last_grad_spread; /**< Last gradient for spread */

    } MMPF_OnlineEM;

    /*═══════════════════════════════════════════════════════════════════════════
     * API FUNCTIONS
     *═══════════════════════════════════════════════════════════════════════════*/

    /**
     * @brief Initialize structural estimator with default 3-regime geometry
     */
    void mmpf_online_em_init(MMPF_OnlineEM *em);

    /**
     * @brief Initialize with custom number of regimes and coefficients
     */
    void mmpf_online_em_init_custom(MMPF_OnlineEM *em, int n_regimes,
                                    const double *coefficients);

    /**
     * @brief Update structural parameters using SGD
     */
    void mmpf_online_em_update(MMPF_OnlineEM *em, double y_log_vol,
                               const rbpf_real_t *weights);

    /**
     * @brief Get projected regime centers
     */
    void mmpf_online_em_get_centers(const MMPF_OnlineEM *em, double *mu_out);

    /**
     * @brief Set learning rates
     */
    void mmpf_online_em_set_learning_rates(MMPF_OnlineEM *em,
                                           double lr_base, double lr_spread);

    /**
     * @brief Set structural constraints
     */
    void mmpf_online_em_set_constraints(MMPF_OnlineEM *em,
                                        double min_spread, double max_spread);

    /**
     * @brief Reset estimator to initial state
     */
    void mmpf_online_em_reset(MMPF_OnlineEM *em);

    /**
     * @brief Print estimator state (for debugging)
     */
    void mmpf_online_em_dump(const MMPF_OnlineEM *em);

#ifdef __cplusplus
}
#endif

#endif /* MMPF_ONLINE_EM_H */