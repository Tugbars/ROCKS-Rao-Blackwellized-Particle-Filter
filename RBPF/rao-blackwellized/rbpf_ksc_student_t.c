/**
 * @file rbpf_ksc_student_t.c
 * @brief Student-t observation model for RBPF-KSC
 *
 * Replaces Gaussian innovations with Student-t for fat-tail robustness.
 * Uses data augmentation: ε | λ ~ N(0, 1/λ),  λ ~ Gamma(ν/2, ν/2)
 */

#include "rbpf_ksc.h"

/* Only compile this file when Student-t is enabled.
 * When disabled, stubs in rbpf_ksc.c provide no-op implementations. */
#if RBPF_ENABLE_STUDENT_T

#include <string.h>
#include <stdio.h>

/*═══════════════════════════════════════════════════════════════════════════
 * OMORI 10-COMPONENT MIXTURE (must match rbpf_ksc.c exactly)
 *═══════════════════════════════════════════════════════════════════════════*/

static const rbpf_real_t KSC_PROB[KSC_N_COMPONENTS] = {
    RBPF_REAL(0.00609), RBPF_REAL(0.04775), RBPF_REAL(0.13057),
    RBPF_REAL(0.20674), RBPF_REAL(0.22715), RBPF_REAL(0.18842),
    RBPF_REAL(0.12047), RBPF_REAL(0.05591), RBPF_REAL(0.01575),
    RBPF_REAL(0.00115)};

static const rbpf_real_t KSC_MEAN[KSC_N_COMPONENTS] = {
    RBPF_REAL(1.92677), RBPF_REAL(1.34744), RBPF_REAL(0.73504),
    RBPF_REAL(0.02266), RBPF_REAL(-0.85173), RBPF_REAL(-1.97278),
    RBPF_REAL(-3.46788), RBPF_REAL(-5.55246), RBPF_REAL(-8.68384),
    RBPF_REAL(-14.65000)};

static const rbpf_real_t KSC_VAR[KSC_N_COMPONENTS] = {
    RBPF_REAL(0.11265), RBPF_REAL(0.17788), RBPF_REAL(0.26768),
    RBPF_REAL(0.40611), RBPF_REAL(0.62699), RBPF_REAL(0.98583),
    RBPF_REAL(1.57469), RBPF_REAL(2.54498), RBPF_REAL(4.16591),
    RBPF_REAL(7.33342)};

/* Precomputed log(prob) for each component */
static const rbpf_real_t KSC_LOG_PROB[KSC_N_COMPONENTS] = {
    RBPF_REAL(-5.10168), RBPF_REAL(-3.04155), RBPF_REAL(-2.03518),
    RBPF_REAL(-1.57656), RBPF_REAL(-1.48204), RBPF_REAL(-1.66940),
    RBPF_REAL(-2.11649), RBPF_REAL(-2.88404), RBPF_REAL(-4.15078),
    RBPF_REAL(-6.76773)};

/*═══════════════════════════════════════════════════════════════════════════
 * SPECIAL FUNCTIONS
 *═══════════════════════════════════════════════════════════════════════════*/

static rbpf_real_t digamma(rbpf_real_t x)
{
    rbpf_real_t result = RBPF_REAL(0.0);
    while (x < RBPF_REAL(6.0))
    {
        result -= RBPF_REAL(1.0) / x;
        x += RBPF_REAL(1.0);
    }
    rbpf_real_t inv_x = RBPF_REAL(1.0) / x;
    rbpf_real_t inv_x2 = inv_x * inv_x;
    result += rbpf_log(x) - RBPF_REAL(0.5) * inv_x;
    result -= inv_x2 * (RBPF_REAL(0.0833333333333333)                           /* 1/12 */
                        - inv_x2 * (RBPF_REAL(0.0083333333333333)               /* 1/120 */
                                    - inv_x2 * RBPF_REAL(0.0039682539682540))); /* 1/252 */
    return result;
}

static rbpf_real_t trigamma(rbpf_real_t x)
{
    rbpf_real_t result = RBPF_REAL(0.0);
    while (x < RBPF_REAL(6.0))
    {
        result += RBPF_REAL(1.0) / (x * x);
        x += RBPF_REAL(1.0);
    }
    rbpf_real_t inv_x = RBPF_REAL(1.0) / x;
    rbpf_real_t inv_x2 = inv_x * inv_x;
    result += inv_x + RBPF_REAL(0.5) * inv_x2;
    result += inv_x2 * inv_x * (RBPF_REAL(0.1666666666666667)                           /* 1/6 */
                                - inv_x2 * (RBPF_REAL(0.0333333333333333)               /* 1/30 */
                                            - inv_x2 * RBPF_REAL(0.0238095238095238))); /* 1/42 */
    return result;
}

/*═══════════════════════════════════════════════════════════════════════════
 * STUDENT-T CONFIGURATION API
 *═══════════════════════════════════════════════════════════════════════════*/

void rbpf_ksc_enable_student_t(RBPF_KSC *rbpf, rbpf_real_t nu)
{
    if (!rbpf)
        return;

    if (nu < RBPF_NU_FLOOR)
        nu = RBPF_NU_FLOOR;
    if (nu > RBPF_NU_CEIL)
        nu = RBPF_NU_CEIL;

    rbpf->student_t_enabled = 1;

    for (int r = 0; r < rbpf->n_regimes; r++)
    {
        rbpf->student_t[r].enabled = 1;
        rbpf->student_t[r].nu = nu;
        rbpf->student_t[r].nu_floor = RBPF_NU_FLOOR;
        rbpf->student_t[r].nu_ceil = RBPF_NU_CEIL;
        rbpf->student_t[r].learn_nu = 0;
        rbpf->student_t[r].nu_learning_rate = RBPF_REAL(0.99);

        rbpf->student_t_stats[r].sum_lambda = RBPF_REAL(0.0);
        rbpf->student_t_stats[r].sum_lambda_sq = RBPF_REAL(0.0);
        rbpf->student_t_stats[r].sum_log_lambda = RBPF_REAL(0.0);
        rbpf->student_t_stats[r].n_eff = RBPF_REAL(0.0);
        rbpf->student_t_stats[r].nu_estimate = nu;
    }

    if (rbpf->lambda == NULL)
    {
        fprintf(stderr, "Warning: lambda arrays not allocated. "
                        "Student-t may not work correctly.\n");
    }
}

void rbpf_ksc_disable_student_t(RBPF_KSC *rbpf)
{
    if (!rbpf)
        return;
    rbpf->student_t_enabled = 0;
    for (int r = 0; r < rbpf->n_regimes; r++)
    {
        rbpf->student_t[r].enabled = 0;
    }
}

void rbpf_ksc_set_student_t_nu(RBPF_KSC *rbpf, int regime, rbpf_real_t nu)
{
    if (!rbpf || regime < 0 || regime >= rbpf->n_regimes)
        return;
    if (nu < RBPF_NU_FLOOR)
        nu = RBPF_NU_FLOOR;
    if (nu > RBPF_NU_CEIL)
        nu = RBPF_NU_CEIL;
    rbpf->student_t[regime].nu = nu;
    rbpf->student_t_stats[regime].nu_estimate = nu;
}

void rbpf_ksc_enable_nu_learning(RBPF_KSC *rbpf, int regime, rbpf_real_t learning_rate)
{
    if (!rbpf || regime < 0 || regime >= rbpf->n_regimes)
        return;
    if (learning_rate < RBPF_REAL(0.9))
        learning_rate = RBPF_REAL(0.9);
    if (learning_rate > RBPF_REAL(0.999))
        learning_rate = RBPF_REAL(0.999);
    rbpf->student_t[regime].learn_nu = 1;
    rbpf->student_t[regime].nu_learning_rate = learning_rate;
}

void rbpf_ksc_disable_nu_learning(RBPF_KSC *rbpf, int regime)
{
    if (!rbpf || regime < 0 || regime >= rbpf->n_regimes)
        return;
    rbpf->student_t[regime].learn_nu = 0;
}

rbpf_real_t rbpf_ksc_get_nu(const RBPF_KSC *rbpf, int regime)
{
    if (!rbpf || regime < 0 || regime >= rbpf->n_regimes)
        return RBPF_NU_DEFAULT;
    if (rbpf->student_t[regime].learn_nu)
    {
        return rbpf->student_t_stats[regime].nu_estimate;
    }
    return rbpf->student_t[regime].nu;
}

void rbpf_ksc_get_lambda_stats(const RBPF_KSC *rbpf, int regime,
                               rbpf_real_t *mean_out, rbpf_real_t *var_out,
                               rbpf_real_t *n_eff_out)
{
    if (!rbpf || regime < 0 || regime >= rbpf->n_regimes)
    {
        if (mean_out)
            *mean_out = RBPF_REAL(1.0);
        if (var_out)
            *var_out = RBPF_REAL(0.0);
        if (n_eff_out)
            *n_eff_out = RBPF_REAL(0.0);
        return;
    }
    const RBPF_StudentT_Stats *stats = &rbpf->student_t_stats[regime];
    if (stats->n_eff < RBPF_REAL(1.0))
    {
        if (mean_out)
            *mean_out = RBPF_REAL(1.0);
        if (var_out)
            *var_out = RBPF_REAL(0.0);
        if (n_eff_out)
            *n_eff_out = RBPF_REAL(0.0);
        return;
    }
    rbpf_real_t mean = stats->sum_lambda / stats->n_eff;
    rbpf_real_t var = stats->sum_lambda_sq / stats->n_eff - mean * mean;
    if (var < RBPF_REAL(0.0))
        var = RBPF_REAL(0.0);
    if (mean_out)
        *mean_out = mean;
    if (var_out)
        *var_out = var;
    if (n_eff_out)
        *n_eff_out = stats->n_eff;
}

void rbpf_ksc_reset_nu_learning(RBPF_KSC *rbpf, int regime)
{
    if (!rbpf || regime < 0 || regime >= rbpf->n_regimes)
        return;
    RBPF_StudentT_Stats *stats = &rbpf->student_t_stats[regime];
    stats->sum_lambda = RBPF_REAL(0.0);
    stats->sum_lambda_sq = RBPF_REAL(0.0);
    stats->sum_log_lambda = RBPF_REAL(0.0);
    stats->n_eff = RBPF_REAL(0.0);
    stats->nu_estimate = rbpf->student_t[regime].nu;
}

/*═══════════════════════════════════════════════════════════════════════════
 * ν ESTIMATION
 *═══════════════════════════════════════════════════════════════════════════*/

static rbpf_real_t estimate_nu_digamma(const RBPF_StudentT_Stats *stats,
                                       rbpf_real_t nu_floor, rbpf_real_t nu_ceil)
{
    if (stats->n_eff < RBPF_REAL(20.0))
    {
        return (nu_floor + nu_ceil) / RBPF_REAL(2.0);
    }

    rbpf_real_t mean_log_lambda = stats->sum_log_lambda / stats->n_eff;
    rbpf_real_t nu = stats->nu_estimate;
    if (nu < nu_floor)
        nu = nu_floor;
    if (nu > nu_ceil)
        nu = nu_ceil;

    for (int iter = 0; iter < 15; iter++)
    {
        rbpf_real_t half_nu = nu / RBPF_REAL(2.0);
        rbpf_real_t psi = digamma(half_nu);
        rbpf_real_t target = psi - rbpf_log(half_nu);
        rbpf_real_t error = mean_log_lambda - target;

        if (rbpf_fabs(error) < RBPF_REAL(1e-6))
            break;

        rbpf_real_t grad = trigamma(half_nu) / RBPF_REAL(2.0) - RBPF_REAL(1.0) / nu;
        if (rbpf_fabs(grad) < RBPF_REAL(1e-10))
            break;

        nu -= error / grad;
        if (nu < nu_floor)
            nu = nu_floor;
        if (nu > nu_ceil)
            nu = nu_ceil;
    }

    return nu;
}

/*═══════════════════════════════════════════════════════════════════════════
 * CORE STUDENT-T UPDATE (NUMERICALLY STABLE)
 *
 * Key fix: GPB1 collapse uses log-space arithmetic throughout to prevent
 * underflow when likelihoods are very small.
 *═══════════════════════════════════════════════════════════════════════════*/

rbpf_real_t rbpf_ksc_update_student_t(RBPF_KSC *rbpf, rbpf_real_t y)
{
    if (!rbpf)
        return RBPF_REAL(0.0);
    if (!rbpf->student_t_enabled || !rbpf->lambda)
    {
        return rbpf_ksc_update(rbpf, y);
    }

    const int n = rbpf->n_particles;
    const rbpf_real_t H = RBPF_REAL(2.0);

    rbpf_real_t *mu_pred = rbpf->mu_pred;
    rbpf_real_t *var_pred = rbpf->var_pred;
    rbpf_real_t *mu = rbpf->mu;
    rbpf_real_t *var = rbpf->var;
    rbpf_real_t *log_weight = rbpf->log_weight;
    rbpf_real_t *lambda = rbpf->lambda;
    rbpf_real_t *log_lambda = rbpf->log_lambda;
    const int *regime = rbpf->regime;

    /* DEFENSIVE: Check if predict was called and produced valid results */
    if (mu_pred == NULL || var_pred == NULL)
    {
        return rbpf_ksc_update(rbpf, y);
    }

    /* DEFENSIVE: Check first particle for NaN (indicates upstream problem) */
    if (mu_pred[0] != mu_pred[0] || var_pred[0] != var_pred[0])
    {
        return rbpf_ksc_update(rbpf, y);
    }

    /* DEFENSIVE: Check observation */
    if (y != y)
    {
        return rbpf_ksc_update(rbpf, y);
    }

    /* Per-regime λ accumulators */
    rbpf_real_t regime_sum_lambda[RBPF_MAX_REGIMES] = {0};
    rbpf_real_t regime_sum_lambda_sq[RBPF_MAX_REGIMES] = {0};
    rbpf_real_t regime_sum_log_lambda[RBPF_MAX_REGIMES] = {0};
    rbpf_real_t regime_sum_weight[RBPF_MAX_REGIMES] = {0};

    /* Precompute effective KSC stats (weighted averages) */
    rbpf_real_t ksc_var_eff = RBPF_REAL(0.0);
    rbpf_real_t ksc_mean_eff = RBPF_REAL(0.0);
    for (int k = 0; k < KSC_N_COMPONENTS; k++)
    {
        ksc_var_eff += KSC_PROB[k] * KSC_VAR[k];
        ksc_mean_eff += KSC_PROB[k] * KSC_MEAN[k];
    }

    rbpf_real_t max_log_weight = -RBPF_REAL(1e30);

    /*═══════════════════════════════════════════════════════════════════════
     * MAIN LOOP: For each particle
     *═══════════════════════════════════════════════════════════════════════*/

    for (int i = 0; i < n; i++)
    {
        int r = regime[i];
        rbpf_real_t nu = rbpf->student_t[r].nu;

        /* Predicted observation variance */
        rbpf_real_t v2_eff = H * H * var_pred[i] + ksc_var_eff;

        /* DEFENSIVE: Ensure v2_eff is positive and finite */
        if (v2_eff < RBPF_REAL(0.1))
            v2_eff = RBPF_REAL(0.1);
        if (v2_eff != v2_eff)
            v2_eff = ksc_var_eff; /* NaN fallback */

        /* Innovation at λ=1 */
        rbpf_real_t y_pred = H * mu_pred[i] + ksc_mean_eff;
        rbpf_real_t innov = y - y_pred;

        /* DEFENSIVE: Clamp extreme innovations */
        if (innov > RBPF_REAL(50.0))
            innov = RBPF_REAL(50.0);
        if (innov < RBPF_REAL(-50.0))
            innov = RBPF_REAL(-50.0);
        if (innov != innov)
            innov = RBPF_REAL(0.0); /* NaN fallback */

        /* Sample λ from Gamma posterior */
        rbpf_real_t alpha_post = (nu + RBPF_REAL(1.0)) / RBPF_REAL(2.0);
        rbpf_real_t beta_post = (nu + innov * innov / v2_eff) / RBPF_REAL(2.0);

        /* Clamp parameters to prevent numerical issues */
        if (alpha_post < RBPF_REAL(0.5))
            alpha_post = RBPF_REAL(0.5);
        if (alpha_post > RBPF_REAL(50.0))
            alpha_post = RBPF_REAL(50.0);
        if (beta_post < RBPF_REAL(0.1))
            beta_post = RBPF_REAL(0.1);
        if (beta_post > RBPF_REAL(100.0))
            beta_post = RBPF_REAL(100.0);

        /* DEFENSIVE: Check for NaN in parameters */
        if (alpha_post != alpha_post || beta_post != beta_post)
        {
            alpha_post = RBPF_REAL(2.5); /* Fallback: ν=4 equivalent */
            beta_post = RBPF_REAL(2.5);
        }

        rbpf_real_t lam = rbpf_pcg32_gamma(&rbpf->pcg[0], alpha_post, beta_post);

        /* DEFENSIVE: Check gamma output */
        if (lam != lam || lam <= RBPF_REAL(0.0))
        {                         /* NaN or non-positive */
            lam = RBPF_REAL(1.0); /* Fallback to Gaussian equivalent */
        }

        /* Clamp λ */
        if (lam < RBPF_LAMBDA_FLOOR)
            lam = RBPF_LAMBDA_FLOOR;
        if (lam > RBPF_LAMBDA_CEIL)
            lam = RBPF_LAMBDA_CEIL;

        lambda[i] = lam;
        log_lambda[i] = rbpf_log(lam);

        /* Accumulate for ν learning */
        regime_sum_lambda[r] += lam;
        regime_sum_lambda_sq[r] += lam * lam;
        regime_sum_log_lambda[r] += log_lambda[i];
        regime_sum_weight[r] += RBPF_REAL(1.0);

        /* Shifted observation */
        rbpf_real_t y_shifted = y + log_lambda[i];

        /* DEFENSIVE: Check y_shifted */
        if (y_shifted != y_shifted)
        {                  /* NaN check */
            y_shifted = y; /* Fallback to unshifted (Gaussian equivalent) */
        }

        /*═══════════════════════════════════════════════════════════════════
         * GPB1 COLLAPSE (NUMERICALLY STABLE)
         *
         * Key: Store log-likelihoods, find max, then normalize.
         *═══════════════════════════════════════════════════════════════════*/

        rbpf_real_t mu_p = mu_pred[i];
        rbpf_real_t var_p = var_pred[i];

        /* Store per-component results */
        rbpf_real_t log_lik[KSC_N_COMPONENTS];
        rbpf_real_t mu_upd[KSC_N_COMPONENTS];
        rbpf_real_t var_upd[KSC_N_COMPONENTS];

        rbpf_real_t max_log_lik = -RBPF_REAL(1e30);

        /* Compute Kalman update for each KSC component */
        for (int k = 0; k < KSC_N_COMPONENTS; k++)
        {
            rbpf_real_t mean_k = KSC_MEAN[k];
            rbpf_real_t var_k = KSC_VAR[k];

            /* Innovation */
            rbpf_real_t y_pred_k = H * mu_p + mean_k;
            rbpf_real_t innov_k = y_shifted - y_pred_k;

            /* Innovation variance: S = H² P + v² */
            rbpf_real_t S_k = H * H * var_p + var_k;
            rbpf_real_t S_inv = RBPF_REAL(1.0) / S_k;

            /* Kalman gain: K = P H / S */
            rbpf_real_t K_k = var_p * H * S_inv;

            /* Updated mean and variance */
            mu_upd[k] = mu_p + K_k * innov_k;
            var_upd[k] = var_p - K_k * H * var_p;
            if (var_upd[k] < RBPF_REAL(1e-10))
                var_upd[k] = RBPF_REAL(1e-10);

            /* Log-likelihood: log(p_k) - 0.5 * (log(S) + innov²/S) */
            log_lik[k] = KSC_LOG_PROB[k] - RBPF_REAL(0.5) * rbpf_log(S_k) - RBPF_REAL(0.5) * innov_k * innov_k * S_inv;

            if (log_lik[k] > max_log_lik)
            {
                max_log_lik = log_lik[k];
            }
        }

        /* Compute normalized weights and GPB1 collapse */
        rbpf_real_t sum_w = RBPF_REAL(0.0);
        rbpf_real_t w[KSC_N_COMPONENTS];

        /* DEFENSIVE: Check max_log_lik is valid */
        if (max_log_lik < RBPF_REAL(-500.0))
            max_log_lik = RBPF_REAL(-500.0);

        for (int k = 0; k < KSC_N_COMPONENTS; k++)
        {
            w[k] = rbpf_exp(log_lik[k] - max_log_lik);
            sum_w += w[k];
        }

        /* DEFENSIVE: Ensure sum_w is positive */
        if (sum_w < RBPF_REAL(1e-30))
            sum_w = RBPF_REAL(1e-30);

        /* Normalize weights */
        rbpf_real_t inv_sum_w = RBPF_REAL(1.0) / sum_w;

        /* GPB1 moment matching */
        rbpf_real_t mu_new = RBPF_REAL(0.0);
        rbpf_real_t E_mu_sq = RBPF_REAL(0.0); /* E[μ²] for variance calculation */
        rbpf_real_t E_var = RBPF_REAL(0.0);   /* E[var] */

        for (int k = 0; k < KSC_N_COMPONENTS; k++)
        {
            rbpf_real_t wk = w[k] * inv_sum_w;
            mu_new += wk * mu_upd[k];
            E_mu_sq += wk * mu_upd[k] * mu_upd[k];
            E_var += wk * var_upd[k];
        }

        /* Var[μ] = E[μ²] - E[μ]² (between-component variance) */
        rbpf_real_t var_between = E_mu_sq - mu_new * mu_new;
        if (var_between < RBPF_REAL(0.0))
            var_between = RBPF_REAL(0.0);

        /* Total variance = E[var] + Var[μ] (law of total variance) */
        rbpf_real_t var_new = E_var + var_between;
        if (var_new < RBPF_REAL(1e-10))
            var_new = RBPF_REAL(1e-10);

        /* DEFENSIVE: Check for NaN before storing */
        if (mu_new != mu_new)
        {                  /* NaN check */
            mu_new = mu_p; /* Keep previous value */
        }
        if (var_new != var_new)
        { /* NaN check */
            var_new = var_p > RBPF_REAL(0.0) ? var_p : RBPF_REAL(0.01);
        }

        /* Store updated state */
        mu[i] = mu_new;
        var[i] = var_new;

        /* Update particle log-weight */
        rbpf_real_t log_lik_total = max_log_lik + rbpf_log(sum_w);

        /* [SAFETY FIX 2] Cap extremely negative likelihoods to prevent immediate kill.
         * Also handle NaN from numerical edge cases. */
        if (log_lik_total != log_lik_total)
        { /* NaN check */
            log_lik_total = RBPF_REAL(-700.0);
        }
        else if (log_lik_total < RBPF_REAL(-700.0))
        {
            log_lik_total = RBPF_REAL(-700.0);
        }

        log_weight[i] += log_lik_total;

        if (log_weight[i] > max_log_weight)
        {
            max_log_weight = log_weight[i];
        }
    }

    /*═══════════════════════════════════════════════════════════════════════
     * NORMALIZE WEIGHTS
     *═══════════════════════════════════════════════════════════════════════*/

    rbpf_real_t sum_weight = RBPF_REAL(0.0);
    for (int i = 0; i < n; i++)
    {
        log_weight[i] -= max_log_weight;
        rbpf_real_t w = rbpf_exp(log_weight[i]);
        rbpf->w_norm[i] = w;
        sum_weight += w;
    }

    /* DEFENSIVE: Ensure sum_weight is positive */
    if (sum_weight < RBPF_REAL(1e-30))
        sum_weight = RBPF_REAL(1e-30);

    rbpf_real_t inv_sum = RBPF_REAL(1.0) / sum_weight;
    for (int i = 0; i < n; i++)
    {
        rbpf->w_norm[i] *= inv_sum;
    }

    /* NOTE: log_weight re-centering is handled in rbpf_ksc_resample()
     * to prevent accumulated underflow. See Fix 1 in rbpf_ksc.c. */

    /* Marginal likelihood (in log space for internal computation) */
    rbpf_real_t log_marginal = max_log_weight + rbpf_log(sum_weight) - rbpf_log((rbpf_real_t)n);

    /* DEFENSIVE: Ensure log_marginal is valid */
    if (log_marginal != log_marginal)
    {                                    /* NaN check */
        log_marginal = RBPF_REAL(-10.0); /* Reasonable fallback */
    }
    if (log_marginal < RBPF_REAL(-500.0))
        log_marginal = RBPF_REAL(-500.0);
    if (log_marginal > RBPF_REAL(100.0))
        log_marginal = RBPF_REAL(100.0);

    /* CRITICAL: Final NaN check and recovery
     * If any particle has NaN state, the entire filter output becomes NaN.
     * This catches any edge cases we missed above. */
    for (int i = 0; i < n; i++)
    {
        if (mu[i] != mu[i])
        {                       /* NaN check */
            mu[i] = mu_pred[i]; /* Fallback to predicted state */
        }
        if (var[i] != var[i] || var[i] <= RBPF_REAL(0.0))
        {
            var[i] = var_pred[i] > RBPF_REAL(0.0) ? var_pred[i] : RBPF_REAL(0.1);
        }
    }

    /* ══════════════════════════════════════════════════════════════════════
     * CRITICAL FIX: Return LINEAR likelihood (positive), not log-likelihood!
     *
     * MMPF expects: model_likelihood > 0, then computes log(model_likelihood)
     * We were returning: log_marginal (negative), causing log(negative) = NaN
     *
     * Convert log-likelihood to linear likelihood before returning.
     * ══════════════════════════════════════════════════════════════════════ */
    rbpf_real_t marginal_lik = rbpf_exp(log_marginal);

    /* Clamp to reasonable positive range (float-safe) */
    if (marginal_lik < RBPF_EPS)
        marginal_lik = RBPF_EPS;
    if (marginal_lik > RBPF_REAL(1e30))
        marginal_lik = RBPF_REAL(1e30);

    /*═══════════════════════════════════════════════════════════════════════
     * UPDATE ν LEARNING STATISTICS
     *═══════════════════════════════════════════════════════════════════════*/

    for (int r = 0; r < rbpf->n_regimes; r++)
    {
        if (!rbpf->student_t[r].learn_nu)
            continue;
        if (regime_sum_weight[r] < RBPF_REAL(1.0))
            continue;

        RBPF_StudentT_Stats *stats = &rbpf->student_t_stats[r];
        RBPF_StudentT_Config *cfg = &rbpf->student_t[r];
        rbpf_real_t lr = cfg->nu_learning_rate;

        rbpf_real_t new_mean_lambda = regime_sum_lambda[r] / regime_sum_weight[r];
        rbpf_real_t new_mean_lambda_sq = regime_sum_lambda_sq[r] / regime_sum_weight[r];
        rbpf_real_t new_mean_log_lambda = regime_sum_log_lambda[r] / regime_sum_weight[r];

        stats->sum_lambda = lr * stats->sum_lambda + (RBPF_REAL(1.0) - lr) * new_mean_lambda;
        stats->sum_lambda_sq = lr * stats->sum_lambda_sq + (RBPF_REAL(1.0) - lr) * new_mean_lambda_sq;
        stats->sum_log_lambda = lr * stats->sum_log_lambda + (RBPF_REAL(1.0) - lr) * new_mean_log_lambda;
        stats->n_eff = lr * stats->n_eff + regime_sum_weight[r];

        stats->nu_estimate = estimate_nu_digamma(stats, cfg->nu_floor, cfg->nu_ceil);
    }

    return marginal_lik;
}

/*═══════════════════════════════════════════════════════════════════════════
 * CONVENIENCE WRAPPERS
 *═══════════════════════════════════════════════════════════════════════════*/

rbpf_real_t rbpf_ksc_update_student_t_nu(RBPF_KSC *rbpf, rbpf_real_t y, rbpf_real_t nu)
{
    if (!rbpf)
        return RBPF_REAL(0.0);

    rbpf_real_t saved_nu[RBPF_MAX_REGIMES];
    for (int r = 0; r < rbpf->n_regimes; r++)
    {
        saved_nu[r] = rbpf->student_t[r].nu;
        rbpf->student_t[r].nu = nu;
    }

    int was_enabled = rbpf->student_t_enabled;
    rbpf->student_t_enabled = 1;

    rbpf_real_t marginal = rbpf_ksc_update_student_t(rbpf, y);

    rbpf->student_t_enabled = was_enabled;
    for (int r = 0; r < rbpf->n_regimes; r++)
    {
        rbpf->student_t[r].nu = saved_nu[r];
    }

    return marginal;
}

void rbpf_ksc_step_student_t(RBPF_KSC *rbpf, rbpf_real_t obs, RBPF_KSC_Output *output)
{
    if (!rbpf)
        return;

    rbpf_real_t y;
    if (obs == RBPF_REAL(0.0))
    {
        y = RBPF_REAL(-18.0);
    }
    else
    {
        y = rbpf_log(obs * obs);
    }

    rbpf_ksc_transition(rbpf);
    rbpf_ksc_predict(rbpf);

    rbpf_real_t marginal = rbpf_ksc_update_student_t(rbpf, y);

    int resampled = rbpf_ksc_resample(rbpf);

    rbpf_ksc_compute_outputs(rbpf, marginal, output);
    output->resampled = resampled;
    output->student_t_active = 1;

    /* Student-t diagnostics */
    if (rbpf->lambda != NULL)
    {
        rbpf_real_t sum_lam = RBPF_REAL(0.0);
        rbpf_real_t sum_lam_sq = RBPF_REAL(0.0);
        const rbpf_real_t *w = rbpf->w_norm;
        const rbpf_real_t *lam = rbpf->lambda;

        for (int i = 0; i < rbpf->n_particles; i++)
        {
            sum_lam += w[i] * lam[i];
            sum_lam_sq += w[i] * lam[i] * lam[i];
        }

        output->lambda_mean = sum_lam;
        output->lambda_var = sum_lam_sq - sum_lam * sum_lam;
        if (output->lambda_var < RBPF_REAL(0.0))
            output->lambda_var = RBPF_REAL(0.0);

        output->nu_effective = (output->lambda_var > RBPF_REAL(0.01))
                                   ? RBPF_REAL(2.0) / output->lambda_var
                                   : RBPF_NU_CEIL;

        for (int r = 0; r < rbpf->n_regimes; r++)
        {
            output->learned_nu[r] = rbpf_ksc_get_nu(rbpf, r);
        }
    }
}

void rbpf_ksc_step_student_t_nu(RBPF_KSC *rbpf, rbpf_real_t obs, rbpf_real_t nu,
                                RBPF_KSC_Output *output)
{
    if (!rbpf)
        return;

    rbpf_real_t y;
    if (obs == RBPF_REAL(0.0))
    {
        y = RBPF_REAL(-18.0);
    }
    else
    {
        y = rbpf_log(obs * obs);
    }

    rbpf_ksc_transition(rbpf);
    rbpf_ksc_predict(rbpf);

    rbpf_real_t marginal = rbpf_ksc_update_student_t_nu(rbpf, y, nu);

    int resampled = rbpf_ksc_resample(rbpf);

    rbpf_ksc_compute_outputs(rbpf, marginal, output);
    output->resampled = resampled;
    output->student_t_active = 1;

    if (rbpf->lambda != NULL)
    {
        rbpf_real_t sum_lam = RBPF_REAL(0.0);
        rbpf_real_t sum_lam_sq = RBPF_REAL(0.0);
        const rbpf_real_t *w = rbpf->w_norm;
        const rbpf_real_t *lam = rbpf->lambda;

        for (int i = 0; i < rbpf->n_particles; i++)
        {
            sum_lam += w[i] * lam[i];
            sum_lam_sq += w[i] * lam[i] * lam[i];
        }

        output->lambda_mean = sum_lam;
        output->lambda_var = sum_lam_sq - sum_lam * sum_lam;
        output->nu_effective = (output->lambda_var > RBPF_REAL(0.01))
                                   ? RBPF_REAL(2.0) / output->lambda_var
                                   : RBPF_NU_CEIL;
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * MEMORY MANAGEMENT
 *═══════════════════════════════════════════════════════════════════════════*/

void rbpf_ksc_free_student_t(RBPF_KSC *rbpf)
{
    if (!rbpf)
        return;

#if defined(_MSC_VER)
    if (rbpf->lambda)
        _aligned_free(rbpf->lambda);
    if (rbpf->lambda_tmp)
        _aligned_free(rbpf->lambda_tmp);
    if (rbpf->log_lambda)
        _aligned_free(rbpf->log_lambda);
#else
    if (rbpf->lambda)
        free(rbpf->lambda);
    if (rbpf->lambda_tmp)
        free(rbpf->lambda_tmp);
    if (rbpf->log_lambda)
        free(rbpf->log_lambda);
#endif

    rbpf->lambda = NULL;
    rbpf->lambda_tmp = NULL;
    rbpf->log_lambda = NULL;
}

int rbpf_ksc_alloc_student_t(RBPF_KSC *rbpf)
{
    if (!rbpf)
        return -1;

    int n = rbpf->n_particles;
    size_t size = n * sizeof(rbpf_real_t);

#if defined(_MSC_VER)
    rbpf->lambda = (rbpf_real_t *)_aligned_malloc(size, RBPF_ALIGN);
    rbpf->lambda_tmp = (rbpf_real_t *)_aligned_malloc(size, RBPF_ALIGN);
    rbpf->log_lambda = (rbpf_real_t *)_aligned_malloc(size, RBPF_ALIGN);
#else
    rbpf->lambda = (rbpf_real_t *)aligned_alloc(RBPF_ALIGN, size);
    rbpf->lambda_tmp = (rbpf_real_t *)aligned_alloc(RBPF_ALIGN, size);
    rbpf->log_lambda = (rbpf_real_t *)aligned_alloc(RBPF_ALIGN, size);
#endif

    if (!rbpf->lambda || !rbpf->lambda_tmp || !rbpf->log_lambda)
    {
        rbpf_ksc_free_student_t(rbpf);
        return -1;
    }

    /* Initialize to λ=1 */
    for (int i = 0; i < n; i++)
    {
        rbpf->lambda[i] = RBPF_REAL(1.0);
        rbpf->lambda_tmp[i] = RBPF_REAL(1.0);
        rbpf->log_lambda[i] = RBPF_REAL(0.0);
    }

    /* Initialize Student-t config (disabled by default) */
    rbpf->student_t_enabled = 0;
    for (int r = 0; r < RBPF_MAX_REGIMES; r++)
    {
        rbpf->student_t[r].enabled = 0;
        rbpf->student_t[r].nu = RBPF_NU_DEFAULT;
        rbpf->student_t[r].nu_floor = RBPF_NU_FLOOR;
        rbpf->student_t[r].nu_ceil = RBPF_NU_CEIL;
        rbpf->student_t[r].learn_nu = 0;
        rbpf->student_t[r].nu_learning_rate = RBPF_REAL(0.99);

        rbpf->student_t_stats[r].sum_lambda = RBPF_REAL(0.0);
        rbpf->student_t_stats[r].sum_lambda_sq = RBPF_REAL(0.0);
        rbpf->student_t_stats[r].sum_log_lambda = RBPF_REAL(0.0);
        rbpf->student_t_stats[r].n_eff = RBPF_REAL(0.0);
        rbpf->student_t_stats[r].nu_estimate = RBPF_NU_DEFAULT;
    }

    return 0;
}

void rbpf_ksc_resample_student_t(RBPF_KSC *rbpf, const int *indices)
{
    if (!rbpf || !indices || !rbpf->lambda)
        return;

    int n = rbpf->n_particles;

    for (int i = 0; i < n; i++)
    {
        rbpf->lambda_tmp[i] = rbpf->lambda[indices[i]];
    }

    rbpf_real_t *tmp = rbpf->lambda;
    rbpf->lambda = rbpf->lambda_tmp;
    rbpf->lambda_tmp = tmp;
}

/*═══════════════════════════════════════════════════════════════════════════
 * DEBUG
 *═══════════════════════════════════════════════════════════════════════════*/

void rbpf_ksc_print_student_t_config(const RBPF_KSC *rbpf)
{
    if (!rbpf)
        return;

    printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    printf("║           Student-t Observation Model Configuration          ║\n");
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    printf("║ Enabled: %s                                                  ║\n",
           rbpf->student_t_enabled ? "YES" : "NO ");

    if (rbpf->student_t_enabled)
    {
        printf("╠══════════════════════════════════════════════════════════════╣\n");
        printf("║ Per-regime configuration:                                    ║\n");

        for (int r = 0; r < rbpf->n_regimes; r++)
        {
            const RBPF_StudentT_Config *cfg = &rbpf->student_t[r];
            const RBPF_StudentT_Stats *stats = &rbpf->student_t_stats[r];

            printf("║   Regime %d: ν=%.2f", r, (double)cfg->nu);
            if (cfg->learn_nu)
            {
                printf(" (learning: ν_est=%.2f, n_eff=%.0f)",
                       (double)stats->nu_estimate, (double)stats->n_eff);
            }
            printf("\n");
        }

        printf("╠══════════════════════════════════════════════════════════════╣\n");
        printf("║ Bounds: ν ∈ [%.1f, %.1f]                                      ║\n",
               (double)RBPF_NU_FLOOR, (double)RBPF_NU_CEIL);
        printf("║ λ bounds: [%.2f, %.1f]                                        ║\n",
               (double)RBPF_LAMBDA_FLOOR, (double)RBPF_LAMBDA_CEIL);
    }

    printf("╚══════════════════════════════════════════════════════════════╝\n");
}

#endif /* RBPF_ENABLE_STUDENT_T */