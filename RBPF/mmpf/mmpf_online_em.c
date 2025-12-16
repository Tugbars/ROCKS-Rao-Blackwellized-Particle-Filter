/**
 * @file mmpf_online_em.c
 * @brief Structural Hierarchical Estimation with Robust M-Estimation
 *
 * This uses Robust SGD (Huber influence) to find the Base and Spread
 * that minimize a robust loss between the model and reality.
 *
 * GEOMETRY (Fixed by Design):
 *   μ_calm   = B - 1.0×S  (One spread below base)
 *   μ_trend  = B          (At base level)
 *   μ_crisis = B + 1.5×S  (1.5 spreads above - asymmetric)
 *
 * WHY ROBUST M-ESTIMATION:
 *   L2 loss desperately wants to "save" the wrong model during crisis
 *   by shrinking spread. Robust clipping caps this influence, allowing
 *   the correct model to dominate when it has weight.
 */

#include "mmpf_online_em.h"
#include <math.h>
#include <string.h>
#include <stdio.h>

/*═══════════════════════════════════════════════════════════════════════════
 * ROBUST INFLUENCE FUNCTION (Huber)
 *═══════════════════════════════════════════════════════════════════════════*/

static double get_robust_error(double raw_error, double scale)
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

    /* GEOMETRY */
    em->n_regimes = 3;
    em->coefficients[0] = -1.0; /* Calm: B - S */
    em->coefficients[1] = 0.0;  /* Trend: B */
    em->coefficients[2] = 1.5;  /* Crisis: B + 1.5S (asymmetric) */

    /* INITIAL STATE */
    em->base_level = -4.5;
    em->spread = 1.0;

    /* PHYSICS */
    em->lr_base = 0.01;   /* Drift tracking */
    em->lr_spread = 0.0;  /* Not used - asymmetric in update */
    em->min_spread = 0.5; /* Minimum separation */
    em->max_spread = 4.0; /* Maximum separation */
    em->min_base = -9.0;
    em->max_base = 2.0;
    em->warmup_ticks = 50;

    /* Project initial μ values */
    for (int k = 0; k < em->n_regimes; k++)
    {
        em->mu[k] = em->base_level + em->coefficients[k] * em->spread;
        em->sigma[k] = 0.5;
    }
}

void mmpf_online_em_init_custom(MMPF_OnlineEM *em, int n_regimes,
                                const double *coefficients)
{
    mmpf_online_em_init(em);

    if (n_regimes < 2)
        n_regimes = 2;
    if (n_regimes > MMPF_EM_MAX_REGIMES)
        n_regimes = MMPF_EM_MAX_REGIMES;

    em->n_regimes = n_regimes;

    if (coefficients)
    {
        memcpy(em->coefficients, coefficients, n_regimes * sizeof(double));
    }

    for (int k = 0; k < em->n_regimes; k++)
    {
        em->mu[k] = em->base_level + em->coefficients[k] * em->spread;
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * THE ROBUST STRUCTURAL SOLVER
 *═══════════════════════════════════════════════════════════════════════════*/

void mmpf_online_em_update(MMPF_OnlineEM *em, double y_log_vol,
                           const rbpf_real_t *weights)
{
    em->tick_count++;
    em->last_y = y_log_vol;

    if (em->tick_count < em->warmup_ticks)
        return;
    if (y_log_vol != y_log_vol)
        return; /* NaN guard */

    double grad_base = 0.0;
    double grad_spread = 0.0;
    double total_weight = 0.0;

    /* TUNING: Scale of "normal" deviations
     * Vol-of-vol is roughly 0.5. 1.0 is ~2 sigma. */
    const double ROBUST_SCALE = 1.0;

    for (int k = 0; k < em->n_regimes; k++)
    {
        double w = (double)weights[k];
        if (w < 1e-6)
            continue;

        double c = em->coefficients[k];
        double pred = em->base_level + c * em->spread;

        /* 1. Calculate Raw Error */
        double raw_error = y_log_vol - pred;

        /* 2. Apply Robust Statistics (M-Estimation)
         * If error is huge (e.g. Calm model during Crisis), clip it.
         * This prevents the "Wrong Model" from hijacking the structure. */
        double robust_err = get_robust_error(raw_error, ROBUST_SCALE);

        /* 3. Accumulate Gradients */
        grad_base += w * robust_err;
        grad_spread += w * robust_err * c;

        total_weight += w;
    }

    if (total_weight > 1e-6)
    {
        grad_base /= total_weight;
        grad_spread /= total_weight;
    }

    em->last_grad_base = grad_base;
    em->last_grad_spread = grad_spread;

    /* Update Base Level */
    em->base_level += em->lr_base * grad_base;

    /* Asymmetric Spread Dynamics ("Fast to Fear, Slow to Calm")
     * Now that robust gradients fix the sign, this works correctly:
     *   - Positive gradient (crisis) → fast expansion
     *   - Negative gradient (recovery) → slow contraction */
    double lr_s = (grad_spread > 0.0) ? 0.050  /* Expand Fast */
                                      : 0.001; /* Contract Slow */

    em->spread += lr_s * grad_spread;

    /* Constraints */
    if (em->spread < em->min_spread)
        em->spread = em->min_spread;
    if (em->spread > em->max_spread)
        em->spread = em->max_spread;

    if (em->base_level < em->min_base)
        em->base_level = em->min_base;
    if (em->base_level > em->max_base)
        em->base_level = em->max_base;

    /* Project to model means */
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

void mmpf_online_em_set_learning_rates(MMPF_OnlineEM *em,
                                       double lr_base, double lr_spread)
{
    if (lr_base < 0.0001)
        lr_base = 0.0001;
    if (lr_base > 0.1)
        lr_base = 0.1;
    em->lr_base = lr_base;
    em->lr_spread = lr_spread; /* Not used in asymmetric mode */
}

void mmpf_online_em_set_constraints(MMPF_OnlineEM *em,
                                    double min_spread, double max_spread)
{
    if (min_spread < 0.1)
        min_spread = 0.1;
    if (max_spread < min_spread)
        max_spread = min_spread + 1.0;

    em->min_spread = min_spread;
    em->max_spread = max_spread;

    if (em->spread < em->min_spread)
        em->spread = em->min_spread;
    if (em->spread > em->max_spread)
        em->spread = em->max_spread;

    for (int k = 0; k < em->n_regimes; k++)
    {
        em->mu[k] = em->base_level + em->coefficients[k] * em->spread;
    }
}

void mmpf_online_em_reset(MMPF_OnlineEM *em)
{
    int n = em->n_regimes;
    double coeffs[MMPF_EM_MAX_REGIMES];
    memcpy(coeffs, em->coefficients, n * sizeof(double));

    mmpf_online_em_init(em);

    em->n_regimes = n;
    memcpy(em->coefficients, coeffs, n * sizeof(double));

    for (int k = 0; k < em->n_regimes; k++)
    {
        em->mu[k] = em->base_level + em->coefficients[k] * em->spread;
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * DIAGNOSTICS
 *═══════════════════════════════════════════════════════════════════════════*/

void mmpf_online_em_dump(const MMPF_OnlineEM *em)
{
    printf("=== Structural Hierarchical Estimation (Robust) ===\n");
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
    printf("================================================\n");
}