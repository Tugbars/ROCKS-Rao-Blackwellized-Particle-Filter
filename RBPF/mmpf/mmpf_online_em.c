/**
 * @file mmpf_online_em.c
 * @brief Structural Hierarchical Estimation for Regime Discovery
 *
 * This uses Stochastic Gradient Descent (SGD) to find the Base and Spread
 * that minimize the error between the model and reality.
 *
 * GEOMETRY (Fixed by Design):
 *   μ_calm   = B - 1.0×S  (One spread below base)
 *   μ_trend  = B          (At base level)
 *   μ_crisis = B + 1.5×S  (1.5 spreads above - asymmetric)
 *
 * LEARNING:
 *   Loss: L = 0.5 × Σ w_k × (y - (B + c_k × S))²
 *
 *   Gradients:
 *     ∂L/∂B = -Σ w_k × error_k
 *     ∂L/∂S = -Σ w_k × error_k × c_k
 *
 * WHY THIS KILLS MODE COLLAPSE:
 *   - min_spread enforces S ≥ 0.3, so models CANNOT touch
 *   - All data informs B, fleet stays in formation
 *   - Even mushy weights find the correct centroid
 */

#include "mmpf_online_em.h"
#include <math.h>
#include <string.h>
#include <stdio.h>

/*═══════════════════════════════════════════════════════════════════════════
 * INITIALIZATION
 *═══════════════════════════════════════════════════════════════════════════*/

void mmpf_online_em_init(MMPF_OnlineEM *em)
{
    memset(em, 0, sizeof(MMPF_OnlineEM));

    /*───────────────────────────────────────────────────────────────────────
     * GEOMETRY: The "Shape" of Volatility
     * Calm is 1 unit below base, Crisis is 1.5 units above (asymmetric)
     *───────────────────────────────────────────────────────────────────────*/
    em->n_regimes = 3;
    em->coefficients[0] = -1.0;                                /* Calm: B - S */
    em->coefficients[1] = 0.0;                                 /* Trend: B (the base) */
    em->coefficients[2] = 1.0; /* Crisis: B + S (symmetric) */ /* Crisis: B + 1.5S (asymmetric) */

    /*───────────────────────────────────────────────────────────────────────
     * INITIALIZATION
     * Start somewhere reasonable; the solver will calibrate quickly
     *───────────────────────────────────────────────────────────────────────*/
    em->base_level = -4.0; /* Trend/base starts at -4.0 */
    em->spread = 0.5;      /* Initial separation */

    /*───────────────────────────────────────────────────────────────────────
     * PHYSICS (Not Heuristics)
     *───────────────────────────────────────────────────────────────────────*/
    em->lr_base = 0.005;                      /* Base moves moderately fast (drift) */
    em->lr_spread = 0.0; /* FIXED GEOMETRY */ /* Structure changes very slowly */
    em->min_spread = 0.5;                     /* Hard physical limit - PREVENTS COLLAPSE */
    em->max_spread = 5.0;                     /* Sanity cap */
    em->min_base = -9.0;                      /* Safety bound */
    em->max_base = 2.0;                       /* Safety bound */
    em->warmup_ticks = 50;

    /* Project initial μ values */
    for (int k = 0; k < em->n_regimes; k++)
    {
        em->mu[k] = em->base_level + em->coefficients[k] * em->spread;
        em->sigma[k] = 0.5; /* Default per-regime spread */
    }
}

void mmpf_online_em_init_custom(MMPF_OnlineEM *em, int n_regimes,
                                const double *coefficients)
{
    /* Start with defaults */
    mmpf_online_em_init(em);

    /* Override with custom geometry */
    if (n_regimes < 2)
        n_regimes = 2;
    if (n_regimes > MMPF_EM_MAX_REGIMES)
        n_regimes = MMPF_EM_MAX_REGIMES;

    em->n_regimes = n_regimes;

    if (coefficients)
    {
        memcpy(em->coefficients, coefficients, n_regimes * sizeof(double));
    }

    /* Re-project μ values with new geometry */
    for (int k = 0; k < em->n_regimes; k++)
    {
        em->mu[k] = em->base_level + em->coefficients[k] * em->spread;
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * THE STRUCTURAL SOLVER (SGD)
 *═══════════════════════════════════════════════════════════════════════════*/

void mmpf_online_em_update(MMPF_OnlineEM *em, double y_log_vol,
                           const rbpf_real_t *weights)
{
    em->tick_count++;
    em->last_y = y_log_vol;

    /* Skip during warmup - let RBPF stabilize first */
    if (em->tick_count < em->warmup_ticks)
    {
        return;
    }

    /* NaN guard */
    if (y_log_vol != y_log_vol)
    {
        return;
    }

    /*───────────────────────────────────────────────────────────────────────
     * COMPUTE GRADIENTS
     *
     * We want to minimize: L = 0.5 × Σ w_k × (y - (B + c_k×S))²
     *
     * Gradients (negative for ascent, but we use error directly):
     *   ∂L/∂B = Σ w_k × error_k
     *   ∂L/∂S = Σ w_k × error_k × c_k
     *───────────────────────────────────────────────────────────────────────*/
    double grad_base = 0.0;
    double grad_spread = 0.0;
    double total_weight = 0.0;

    for (int k = 0; k < em->n_regimes; k++)
    {
        double w = (double)weights[k];
        if (w < 1e-6)
            continue;

        double c = em->coefficients[k];

        /* Prediction: Where this model thinks vol should be */
        double predicted_mu = em->base_level + c * em->spread;

        /* Error: How wrong was this model? */
        /* Positive error = vol higher than expected → move UP */
        double error = y_log_vol - predicted_mu;

        /* Accumulate gradients */
        grad_base += w * error;       /* If error > 0, raise Base */
        grad_spread += w * error * c; /* If error > 0 AND c > 0, increase Spread */

        total_weight += w;
    }

    /* Normalize gradients (proper expectation) */
    if (total_weight > 1e-6)
    {
        grad_base /= total_weight;
        grad_spread /= total_weight;
    }

    /* Store for diagnostics */
    em->last_grad_base = grad_base;
    em->last_grad_spread = grad_spread;

    /*───────────────────────────────────────────────────────────────────────
     * UPDATE STATE (SGD Step)
     *───────────────────────────────────────────────────────────────────────*/
    em->base_level += em->lr_base * grad_base;
    em->spread += em->lr_spread * grad_spread;

    /*───────────────────────────────────────────────────────────────────────
     * ENFORCE PHYSICAL CONSTRAINTS
     * This is what makes mode collapse IMPOSSIBLE
     *───────────────────────────────────────────────────────────────────────*/
    if (em->spread < em->min_spread)
    {
        em->spread = em->min_spread;
    }
    if (em->spread > em->max_spread)
    {
        em->spread = em->max_spread;
    }

    /* Safety bounds on base level */
    if (em->base_level < em->min_base)
    {
        em->base_level = em->min_base;
    }
    if (em->base_level > em->max_base)
    {
        em->base_level = em->max_base;
    }

    /*───────────────────────────────────────────────────────────────────────
     * PROJECT TO MODEL MEANS
     * This GUARANTEES: μ_calm < μ_trend < μ_crisis (forever)
     *───────────────────────────────────────────────────────────────────────*/
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
    if (lr_spread < 0.00001)
        lr_spread = 0.00001;
    if (lr_spread > 0.01)
        lr_spread = 0.01;

    em->lr_base = lr_base;
    em->lr_spread = lr_spread;
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

    /* Enforce constraints immediately */
    if (em->spread < em->min_spread)
    {
        em->spread = em->min_spread;
    }
    if (em->spread > em->max_spread)
    {
        em->spread = em->max_spread;
    }

    /* Re-project */
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

    /* Re-project */
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
    printf("=== Structural Hierarchical Estimation ===\n");
    printf("Base Level (B): %.4f\n", em->base_level);
    printf("Spread (S):     %.4f\n", em->spread);
    printf("Ticks:          %d\n", em->tick_count);
    printf("\nProjected Means:\n");
    for (int k = 0; k < em->n_regimes; k++)
    {
        printf("  [%d] coeff=%.2f → μ=%.4f\n",
               k, em->coefficients[k], em->mu[k]);
    }
    printf("\nLast Gradients:\n");
    printf("  grad_B: %.6f\n", em->last_grad_base);
    printf("  grad_S: %.6f\n", em->last_grad_spread);
    printf("==========================================\n");
}