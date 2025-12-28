/**
 * @file mmpf_online_em.c
 * @brief Robust Structural Hierarchical Estimation (M-Estimation)
 *
 * PHILOSOPHY:
 * 1. Geometry: We solve for Base (B) and Spread (S).
 * 2. Robustness: We use Huber Loss to cap the influence of structural outliers.
 *    This mathematically prevents the "Calm Model" from shrinking the spread
 *    during a crisis, without using any `if` heuristics.
 * 3. Physics: We allow asymmetric spread dynamics (Vol clustering).
 */

#include "mmpf_online_em.h"
#include <math.h>
#include <string.h>
#include <stdio.h>

/*═══════════════════════════════════════════════════════════════════════════
 * ROBUST STATISTICS CORE
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Huber Influence Function (psi)
 * Minimizes L_robust(e) instead of L2(e).
 * Gradient is linear for small e, constant for large e.
 */
static double get_robust_gradient(double raw_error, double scale)
{
    if (raw_error > scale)
        return scale;
    if (raw_error < -scale)
        return -scale;
    return raw_error;
}

/*═══════════════════════════════════════════════════════════════════════════
 * INITIALIZATION
 *═══════════════════════════════════════════════════════════════════════════*/

void mmpf_online_em_init(MMPF_OnlineEM *em)
{
    memset(em, 0, sizeof(MMPF_OnlineEM));

    /* GEOMETRY: The Invariant Shape */
    em->n_regimes = 3;
    em->coefficients[0] = -1.0; /* Calm */
    em->coefficients[1] = 0.0;  /* Trend */
    em->coefficients[2] = 1.5;  /* Crisis (Asymmetric upside) */

    /* INITIAL STATE */
    em->base_level = -4.5;
    em->spread = 1.0;

    /* PHYSICS & STATISTICS */
    /* Base Level: Symmetric drift */
    em->lr_base = 0.01;

    /* Spread: Asymmetric Physics (Volatility Clustering)
     * Markets explode fast (fear) and decay slow (memory).
     * This is a property of the asset class, not a heuristic hack. */
    em->lr_spread_pos = 0.050; /* Fast Expansion */
    em->lr_spread_neg = 0.001; /* Slow Contraction */

    /* Robust Scale: 1.0 log-units (approx 3 sigma of vol-of-vol)
     * Errors larger than this are structural breaks, not noise. */
    em->robust_scale = 1.0;

    /* Hard Physical Constraints */
    em->min_spread = 0.5;
    em->warmup_ticks = 50;

    /* Project Initial */
    for (int k = 0; k < em->n_regimes; k++)
    {
        em->mu[k] = em->base_level + em->coefficients[k] * em->spread;
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * THE ROBUST STRUCTURAL SOLVER (SGD on Huber Loss)
 *═══════════════════════════════════════════════════════════════════════════*/

void mmpf_online_em_update(MMPF_OnlineEM *em, double y_log_sq, const rbpf_real_t *weights)
{
    em->tick_count++;
    if (em->tick_count < em->warmup_ticks)
        return;
    if (y_log_sq != y_log_sq)
        return; /* NaN guard */

    double grad_base = 0.0;
    double grad_spread = 0.0;
    double total_weight = 0.0;

    for (int k = 0; k < em->n_regimes; k++)
    {
        double w = (double)weights[k];
        if (w < 1e-6)
            continue;

        double c = em->coefficients[k];
        double pred = em->base_level + c * em->spread;

        /* 1. Raw Prediction Error */
        double raw_error = y_log_sq - pred;

        /* 2. Robust Gradient Calculation (The "Magic")
         *
         * If Calm model has error +2.0 (Crisis), standard L2 gives gradient -2.0
         * Huber clips this to -1.0.
         * Crisis model error +0.5 is uncapped.
         * Result: Crisis vote (+0.75) > Calm vote (-1.0 * small_weight).
         * The spread expands NATURALLY. */
        double robust_grad = get_robust_gradient(raw_error, em->robust_scale);

        /* Accumulate */
        grad_base += w * robust_grad;
        grad_spread += w * robust_grad * c;

        total_weight += w;
    }

    if (total_weight > 1e-6)
    {
        grad_base /= total_weight;
        grad_spread /= total_weight;
    }

    /* Diagnostics */
    em->last_grad_base = grad_base;
    em->last_grad_spread = grad_spread;

    /* 3. Update Base Level (Drift) */
    em->base_level += em->lr_base * grad_base;

    /* 4. Update Spread (Asymmetric Physics) */
    double lr_s = (grad_spread > 0.0) ? em->lr_spread_pos : em->lr_spread_neg;
    em->spread += lr_s * grad_spread;

    /* 5. Enforce Constraints */
    if (em->spread < em->min_spread)
        em->spread = em->min_spread;
    if (em->spread > 4.0)
        em->spread = 4.0;

    if (em->base_level < -9.0)
        em->base_level = -9.0;
    if (em->base_level > 2.0)
        em->base_level = 2.0;

    /* 6. Project to Models */
    for (int k = 0; k < em->n_regimes; k++)
    {
        em->mu[k] = em->base_level + em->coefficients[k] * em->spread;
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * QUERY FUNCTIONS
 *═══════════════════════════════════════════════════════════════════════════*/

void mmpf_online_em_get_centers(const MMPF_OnlineEM *em, double *mu_out)
{
    memcpy(mu_out, em->mu, em->n_regimes * sizeof(double));
}

void mmpf_online_em_dump(const MMPF_OnlineEM *em)
{
    printf("=== Robust Structural Estimation ===\n");
    printf("Base Level (B): %.4f\n", em->base_level);
    printf("Spread (S):     %.4f\n", em->spread);
    printf("Ticks:          %d\n", em->tick_count);
    printf("\nProjected Means:\n");
    for (int k = 0; k < em->n_regimes; k++)
    {
        printf("  [%d] coeff=%.2f -> mu=%.4f\n",
               k, em->coefficients[k], em->mu[k]);
    }
    printf("\nLast Gradients:\n");
    printf("  grad_B: %.6f\n", em->last_grad_base);
    printf("  grad_S: %.6f\n", em->last_grad_spread);
    printf("====================================\n");
}