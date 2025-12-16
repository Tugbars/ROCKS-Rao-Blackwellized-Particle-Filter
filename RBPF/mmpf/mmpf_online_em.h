/**
 * @file mmpf_online_em.h
 * @brief Robust Structural Hierarchical Estimation (M-Estimation)
 *
 * PHILOSOPHY:
 * 1. Geometry: We solve for Base (B) and Spread (S).
 *    μ_calm   = B - 1.0×S
 *    μ_trend  = B
 *    μ_crisis = B + 1.5×S
 *
 * 2. Robustness: Huber Loss caps influence of structural outliers.
 *    This mathematically prevents "Calm Model" from shrinking spread
 *    during crisis, without using any `if` heuristics.
 *
 * 3. Physics: Asymmetric spread dynamics (Volatility Clustering).
 *    Markets explode fast (fear) and decay slow (memory).
 */

#ifndef MMPF_ONLINE_EM_H
#define MMPF_ONLINE_EM_H

#include "rbpf_ksc.h"

#ifdef __cplusplus
extern "C"
{
#endif

#define MMPF_EM_MAX_REGIMES 8

    typedef struct
    {
        /*───────────────────────────────────────────────────────────────────────
         * THE PHYSICAL STATE (Only these are learned)
         *───────────────────────────────────────────────────────────────────────*/
        double base_level; /**< B: Center of gravity of market vol */
        double spread;     /**< S: Distance between regimes */

        /*───────────────────────────────────────────────────────────────────────
         * THE GEOMETRY (Fixed Design)
         *───────────────────────────────────────────────────────────────────────*/
        int n_regimes;
        double coefficients[MMPF_EM_MAX_REGIMES];

        /*───────────────────────────────────────────────────────────────────────
         * PHYSICS & STATISTICS (Tunable)
         *───────────────────────────────────────────────────────────────────────*/
        double lr_base;       /**< Base level drift rate */
        double lr_spread_pos; /**< Spread expansion rate (Fast Fear) */
        double lr_spread_neg; /**< Spread contraction rate (Slow Decay) */
        double robust_scale;  /**< Huber scale (~3σ of vol-of-vol) */
        double min_spread;    /**< Physical floor */

        /*───────────────────────────────────────────────────────────────────────
         * OUTPUT
         *───────────────────────────────────────────────────────────────────────*/
        double mu[MMPF_EM_MAX_REGIMES];

        /*───────────────────────────────────────────────────────────────────────
         * DIAGNOSTICS
         *───────────────────────────────────────────────────────────────────────*/
        int tick_count;
        int warmup_ticks;
        double last_grad_base;
        double last_grad_spread;

    } MMPF_OnlineEM;

    void mmpf_online_em_init(MMPF_OnlineEM *em);
    void mmpf_online_em_update(MMPF_OnlineEM *em, double y_log_sq, const rbpf_real_t *weights);
    void mmpf_online_em_get_centers(const MMPF_OnlineEM *em, double *mu_out);
    void mmpf_online_em_dump(const MMPF_OnlineEM *em);

#ifdef __cplusplus
}
#endif

#endif /* MMPF_ONLINE_EM_H */