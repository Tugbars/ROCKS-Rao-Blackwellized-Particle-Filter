/**
 * @file rbpf_ksc_student_t.c
 * @brief Complete Student-t implementation for RBPF-KSC
 *
 * Contains:
 *   - Student-t API functions (enable, disable, set_nu, etc.)
 *   - Memory allocation/free for lambda arrays
 *   - Scalar update functions
 *   - MKL-optimized update function
 *
 * MSVC-compatible (C89 loop declarations for OpenMP)
 */

#include "rbpf_ksc.h"

#if RBPF_ENABLE_STUDENT_T

#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <mkl.h>
#include <mkl_vml.h>
#include <mkl_cblas.h>

/*═══════════════════════════════════════════════════════════════════════════
 * CONFIGURATION
 *═══════════════════════════════════════════════════════════════════════════*/

#ifndef RBPF_MKL_BLOCK_SIZE
#define RBPF_MKL_BLOCK_SIZE 2048
#endif

#define N_COMPONENTS_TOTAL 11

/*═══════════════════════════════════════════════════════════════════════════
 * KSC CONSTANTS (Omori et al. 2007)
 *═══════════════════════════════════════════════════════════════════════════*/

static const double KSC_PROB_ST[10] = {
    0.00609, 0.04775, 0.13057, 0.20674, 0.22715,
    0.18842, 0.12047, 0.05591, 0.01575, 0.00115};

static const double KSC_MEAN_ST[10] = {
    1.92677, 1.34744, 0.73504, 0.02266, -0.85173,
    -1.97278, -3.46788, -5.55246, -8.68384, -14.65000};

static const double KSC_VAR_ST[10] = {
    0.11265, 0.17788, 0.26768, 0.40611, 0.62699,
    0.98583, 1.57469, 2.54498, 4.16591, 7.33342};

static const double KSC_LOG_PROB_ST[10] = {
    -5.10168, -3.04155, -2.03518, -1.57656, -1.48204,
    -1.66940, -2.11649, -2.88404, -4.15078, -6.76773};

/*═══════════════════════════════════════════════════════════════════════════
 * FORWARD DECLARATIONS
 *═══════════════════════════════════════════════════════════════════════════*/

void rbpf_ksc_free_student_t(RBPF_KSC *rbpf);

/*═══════════════════════════════════════════════════════════════════════════
 * MEMORY ALLOCATION
 *═══════════════════════════════════════════════════════════════════════════*/

int rbpf_ksc_alloc_student_t(RBPF_KSC *rbpf)
{
    int i, n;

    if (!rbpf)
        return -1;

    n = rbpf->n_particles;

    rbpf->lambda = (rbpf_real_t *)mkl_malloc(n * sizeof(rbpf_real_t), RBPF_ALIGN);
    rbpf->lambda_tmp = (rbpf_real_t *)mkl_malloc(n * sizeof(rbpf_real_t), RBPF_ALIGN);
    rbpf->log_lambda = (rbpf_real_t *)mkl_malloc(n * sizeof(rbpf_real_t), RBPF_ALIGN);

    if (!rbpf->lambda || !rbpf->lambda_tmp || !rbpf->log_lambda)
    {
        rbpf_ksc_free_student_t(rbpf);
        return -1;
    }

    /* Initialize to 1.0 (no scaling) */
    for (i = 0; i < n; i++)
    {
        rbpf->lambda[i] = RBPF_REAL(1.0);
        rbpf->lambda_tmp[i] = RBPF_REAL(1.0);
        rbpf->log_lambda[i] = RBPF_REAL(0.0);
    }

    return 0;
}

void rbpf_ksc_free_student_t(RBPF_KSC *rbpf)
{
    if (!rbpf)
        return;

    if (rbpf->lambda)
    {
        mkl_free(rbpf->lambda);
        rbpf->lambda = NULL;
    }
    if (rbpf->lambda_tmp)
    {
        mkl_free(rbpf->lambda_tmp);
        rbpf->lambda_tmp = NULL;
    }
    if (rbpf->log_lambda)
    {
        mkl_free(rbpf->log_lambda);
        rbpf->log_lambda = NULL;
    }
}

void rbpf_ksc_resample_student_t(RBPF_KSC *rbpf, const int *indices)
{
    int i, n;
    rbpf_real_t *tmp;

    if (!rbpf || !rbpf->lambda || !indices)
        return;

    n = rbpf->n_particles;

    /* Gather lambda values according to resample indices */
    for (i = 0; i < n; i++)
    {
        rbpf->lambda_tmp[i] = rbpf->lambda[indices[i]];
    }

    /* Swap buffers */
    tmp = rbpf->lambda;
    rbpf->lambda = rbpf->lambda_tmp;
    rbpf->lambda_tmp = tmp;

    /* Update log_lambda */
    for (i = 0; i < n; i++)
    {
        rbpf->log_lambda[i] = rbpf_log(rbpf->lambda[i]);
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * STUDENT-T API FUNCTIONS
 *═══════════════════════════════════════════════════════════════════════════*/

void rbpf_ksc_enable_student_t(RBPF_KSC *rbpf, rbpf_real_t nu)
{
    int r;

    if (!rbpf)
        return;

    rbpf->student_t_enabled = 1;

    /* Set default ν for all regimes */
    for (r = 0; r < rbpf->n_regimes; r++)
    {
        rbpf->student_t[r].enabled = 1;
        rbpf->student_t[r].nu = nu;
        rbpf->student_t[r].nu_floor = RBPF_NU_FLOOR;
        rbpf->student_t[r].nu_ceil = RBPF_NU_CEIL;
        rbpf->student_t[r].learn_nu = 0;
        rbpf->student_t[r].nu_learning_rate = RBPF_REAL(0.99);
    }
}

void rbpf_ksc_disable_student_t(RBPF_KSC *rbpf)
{
    int r;

    if (!rbpf)
        return;

    rbpf->student_t_enabled = 0;

    for (r = 0; r < rbpf->n_regimes; r++)
    {
        rbpf->student_t[r].enabled = 0;
    }
}

void rbpf_ksc_set_student_t_nu(RBPF_KSC *rbpf, int regime, rbpf_real_t nu)
{
    if (!rbpf || regime < 0 || regime >= rbpf->n_regimes)
        return;

    /* Clamp ν to valid range */
    if (nu < RBPF_NU_FLOOR)
        nu = RBPF_NU_FLOOR;
    if (nu > RBPF_NU_CEIL)
        nu = RBPF_NU_CEIL;

    rbpf->student_t[regime].nu = nu;
}

void rbpf_ksc_enable_nu_learning(RBPF_KSC *rbpf, int regime, rbpf_real_t learning_rate)
{
    if (!rbpf || regime < 0 || regime >= rbpf->n_regimes)
        return;

    rbpf->student_t[regime].learn_nu = 1;
    rbpf->student_t[regime].nu_learning_rate = learning_rate;

    /* Reset statistics */
    rbpf_ksc_reset_nu_learning(rbpf, regime);
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
    {
        return RBPF_NU_DEFAULT;
    }

    return rbpf->student_t[regime].nu;
}

void rbpf_ksc_get_lambda_stats(const RBPF_KSC *rbpf, int regime,
                               rbpf_real_t *mean_out, rbpf_real_t *var_out,
                               rbpf_real_t *n_eff_out)
{
    const RBPF_StudentT_Stats *stats;

    if (!rbpf || regime < 0 || regime >= rbpf->n_regimes)
        return;

    stats = &rbpf->student_t_stats[regime];

    if (mean_out)
        *mean_out = (stats->n_eff > 0) ? stats->sum_lambda / stats->n_eff : RBPF_REAL(1.0);
    if (var_out)
    {
        if (stats->n_eff > 1)
        {
            rbpf_real_t mean = stats->sum_lambda / stats->n_eff;
            *var_out = (stats->sum_lambda_sq / stats->n_eff) - mean * mean;
        }
        else
        {
            *var_out = RBPF_REAL(0.0);
        }
    }
    if (n_eff_out)
        *n_eff_out = stats->n_eff;
}

void rbpf_ksc_reset_nu_learning(RBPF_KSC *rbpf, int regime)
{
    RBPF_StudentT_Stats *stats;

    if (!rbpf || regime < 0 || regime >= rbpf->n_regimes)
        return;

    stats = &rbpf->student_t_stats[regime];
    stats->sum_lambda = RBPF_REAL(0.0);
    stats->sum_lambda_sq = RBPF_REAL(0.0);
    stats->sum_log_lambda = RBPF_REAL(0.0);
    stats->n_eff = RBPF_REAL(0.0);
    stats->nu_estimate = rbpf->student_t[regime].nu;
}

/*═══════════════════════════════════════════════════════════════════════════
 * SCALAR UPDATE FUNCTIONS
 *═══════════════════════════════════════════════════════════════════════════*/

rbpf_real_t rbpf_ksc_update_student_t(RBPF_KSC *rbpf, rbpf_real_t y)
{
    /* Use regime 0's ν as default */
    rbpf_real_t nu = rbpf->student_t[0].nu;
    return rbpf_ksc_update_student_t_nu(rbpf, y, nu);
}

rbpf_real_t rbpf_ksc_update_student_t_nu(RBPF_KSC *rbpf, rbpf_real_t y, rbpf_real_t nu)
{
    const int n = rbpf->n_particles;
    const rbpf_real_t H = RBPF_REAL(2.0);
    const rbpf_real_t H2 = RBPF_REAL(4.0);
    const rbpf_real_t nu_half = nu * RBPF_REAL(0.5);
    const rbpf_real_t nu_plus_1_half = (nu + RBPF_REAL(1.0)) * RBPF_REAL(0.5);

    rbpf_real_t max_log_weight = RBPF_REAL(-1e30);
    rbpf_real_t sum_weight, inv_sum;
    int i, k;

    /* Phase 1: Sample λ and compute Kalman updates */
    for (i = 0; i < n; i++)
    {
        const rbpf_real_t mp = rbpf->mu_pred[i];
        const rbpf_real_t vp = rbpf->var_pred[i];
        rbpf_real_t innov_dom, var_eff, beta, lam, inv_lam;
        rbpf_real_t max_ll, lik_total, mu_acc, mu2_acc;
        rbpf_real_t inv_lik, mu_new, var_new, ll;

        /* Sample λ from conditional posterior using dominant component */
        innov_dom = (y - (rbpf_real_t)KSC_MEAN_ST[4]) - H * mp;
        if (innov_dom > RBPF_REAL(50.0))
            innov_dom = RBPF_REAL(50.0);
        else if (innov_dom < RBPF_REAL(-50.0))
            innov_dom = RBPF_REAL(-50.0);

        var_eff = H2 * vp + (rbpf_real_t)KSC_VAR_ST[4];
        beta = nu_half + RBPF_REAL(0.5) * innov_dom * innov_dom / var_eff;
        if (beta < RBPF_REAL(0.1))
            beta = RBPF_REAL(0.1);
        else if (beta > RBPF_REAL(100.0))
            beta = RBPF_REAL(100.0);

        lam = rbpf_pcg32_gamma(&rbpf->pcg[0], nu_plus_1_half, beta);

        if (lam != lam || lam <= RBPF_REAL(0.0))
            lam = RBPF_REAL(1.0);
        else if (lam < RBPF_LAMBDA_FLOOR)
            lam = RBPF_LAMBDA_FLOOR;
        else if (lam > RBPF_LAMBDA_CEIL)
            lam = RBPF_LAMBDA_CEIL;

        rbpf->lambda[i] = lam;
        rbpf->log_lambda[i] = rbpf_log(lam);

        inv_lam = RBPF_REAL(1.0) / lam;

        /* 10-component mixture update with λ-scaled variance */
        max_ll = RBPF_REAL(-1e30);
        lik_total = RBPF_REAL(0.0);
        mu_acc = RBPF_REAL(0.0);
        mu2_acc = RBPF_REAL(0.0);

        for (k = 0; k < 10; k++)
        {
            rbpf_real_t S = H2 * vp + (rbpf_real_t)KSC_VAR_ST[k] * inv_lam;
            rbpf_real_t innov = y - (rbpf_real_t)KSC_MEAN_ST[k] - H * mp;
            rbpf_real_t inv_S = RBPF_REAL(1.0) / S;
            rbpf_real_t log_lik, K, mu_post, var_post, w;

            log_lik = (rbpf_real_t)KSC_LOG_PROB_ST[k] - RBPF_REAL(0.5) * (rbpf_log(S) + innov * innov * inv_S);

            if (log_lik > max_ll)
                max_ll = log_lik;

            K = H * vp * inv_S;
            mu_post = mp + K * innov;
            var_post = (RBPF_REAL(1.0) - K * H) * vp;

            w = rbpf_exp(log_lik - max_ll + RBPF_REAL(700.0));
            lik_total += w;
            mu_acc += w * mu_post;
            mu2_acc += w * (var_post + mu_post * mu_post);
        }

        /* GPB1 collapse */
        inv_lik = RBPF_REAL(1.0) / (lik_total + RBPF_EPS);
        mu_new = mu_acc * inv_lik;
        var_new = mu2_acc * inv_lik - mu_new * mu_new;

        if (var_new < RBPF_REAL(1e-6))
            var_new = RBPF_REAL(1e-6);

        rbpf->mu[i] = mu_new;
        rbpf->var[i] = var_new;

        /* Update log-weight */
        ll = max_ll - RBPF_REAL(700.0) + rbpf_log(lik_total + RBPF_EPS);
        rbpf->log_weight[i] += ll;

        if (rbpf->log_weight[i] > max_log_weight)
        {
            max_log_weight = rbpf->log_weight[i];
        }
    }

    /* Normalize weights */
    sum_weight = RBPF_REAL(0.0);
    for (i = 0; i < n; i++)
    {
        rbpf_real_t w;
        rbpf->log_weight[i] -= max_log_weight;
        w = rbpf_exp(rbpf->log_weight[i]);
        rbpf->w_norm[i] = w;
        sum_weight += w;
    }

    if (sum_weight < RBPF_EPS)
        sum_weight = RBPF_EPS;
    inv_sum = RBPF_REAL(1.0) / sum_weight;

    for (i = 0; i < n; i++)
    {
        rbpf->w_norm[i] *= inv_sum;
    }

    return rbpf_exp(max_log_weight + rbpf_log(sum_weight) - rbpf_log((rbpf_real_t)n));
}

rbpf_real_t rbpf_ksc_update_student_t_robust(RBPF_KSC *rbpf, rbpf_real_t y,
                                             rbpf_real_t nu,
                                             const RBPF_RobustOCSN *ocsn)
{
    const int n = rbpf->n_particles;
    const rbpf_real_t H = RBPF_REAL(2.0);
    const rbpf_real_t H2 = RBPF_REAL(4.0);
    const rbpf_real_t nu_half = nu * RBPF_REAL(0.5);
    const rbpf_real_t nu_plus_1_half = (nu + RBPF_REAL(1.0)) * RBPF_REAL(0.5);

    rbpf_real_t max_log_weight = RBPF_REAL(-1e30);
    rbpf_real_t sum_weight, inv_sum;
    int i, k;

    if (!ocsn || !ocsn->enabled)
    {
        return rbpf_ksc_update_student_t_nu(rbpf, y, nu);
    }

    for (i = 0; i < n; i++)
    {
        const int r = rbpf->regime[i];
        const rbpf_real_t mp = rbpf->mu_pred[i];
        const rbpf_real_t vp = rbpf->var_pred[i];
        rbpf_real_t ocsn_var, ocsn_pi;
        rbpf_real_t innov_dom, var_eff, beta, lam, inv_lam;
        rbpf_real_t log_1_minus_pi, log_pi;
        rbpf_real_t max_ll;
        rbpf_real_t log_liks[11], mu_posts[11], var_posts[11];
        rbpf_real_t lik_total, mu_acc, mu2_acc;
        rbpf_real_t inv_lik, mu_new, var_new, ll;

        /* OCSN params */
        ocsn_var = ocsn->regime[r].variance;
        ocsn_pi = ocsn->regime[r].prob;
        if (ocsn_var < RBPF_OUTLIER_VAR_MIN)
            ocsn_var = RBPF_OUTLIER_VAR_MIN;
        if (ocsn_var > RBPF_OUTLIER_VAR_MAX)
            ocsn_var = RBPF_OUTLIER_VAR_MAX;

        /* Sample λ */
        innov_dom = (y - (rbpf_real_t)KSC_MEAN_ST[4]) - H * mp;
        if (innov_dom > RBPF_REAL(50.0))
            innov_dom = RBPF_REAL(50.0);
        else if (innov_dom < RBPF_REAL(-50.0))
            innov_dom = RBPF_REAL(-50.0);

        var_eff = H2 * vp + (rbpf_real_t)KSC_VAR_ST[4];
        beta = nu_half + RBPF_REAL(0.5) * innov_dom * innov_dom / var_eff;
        if (beta < RBPF_REAL(0.1))
            beta = RBPF_REAL(0.1);
        else if (beta > RBPF_REAL(100.0))
            beta = RBPF_REAL(100.0);

        lam = rbpf_pcg32_gamma(&rbpf->pcg[0], nu_plus_1_half, beta);

        if (lam != lam || lam <= RBPF_REAL(0.0))
            lam = RBPF_REAL(1.0);
        else if (lam < RBPF_LAMBDA_FLOOR)
            lam = RBPF_LAMBDA_FLOOR;
        else if (lam > RBPF_LAMBDA_CEIL)
            lam = RBPF_LAMBDA_CEIL;

        rbpf->lambda[i] = lam;
        rbpf->log_lambda[i] = rbpf_log(lam);

        inv_lam = RBPF_REAL(1.0) / lam;
        log_1_minus_pi = rbpf_log(RBPF_REAL(1.0) - ocsn_pi);
        log_pi = rbpf_log(ocsn_pi);

        /* 11-component mixture (10 KSC + 1 outlier) */
        max_ll = RBPF_REAL(-1e30);

        /* 10 KSC components */
        for (k = 0; k < 10; k++)
        {
            rbpf_real_t S = H2 * vp + (rbpf_real_t)KSC_VAR_ST[k] * inv_lam;
            rbpf_real_t innov = y - (rbpf_real_t)KSC_MEAN_ST[k] - H * mp;
            rbpf_real_t inv_S = RBPF_REAL(1.0) / S;
            rbpf_real_t K;

            log_liks[k] = log_1_minus_pi + (rbpf_real_t)KSC_LOG_PROB_ST[k] - RBPF_REAL(0.5) * (rbpf_log(S) + innov * innov * inv_S);

            if (log_liks[k] > max_ll)
                max_ll = log_liks[k];

            K = H * vp * inv_S;
            mu_posts[k] = mp + K * innov;
            var_posts[k] = (RBPF_REAL(1.0) - K * H) * vp;
        }

        /* Outlier component (NOT λ-scaled) */
        {
            rbpf_real_t S = H2 * vp + ocsn_var;
            rbpf_real_t innov = y - H * mp;
            rbpf_real_t inv_S = RBPF_REAL(1.0) / S;
            rbpf_real_t K;

            log_liks[10] = log_pi - RBPF_REAL(0.5) * (rbpf_log(S) + innov * innov * inv_S);

            if (log_liks[10] > max_ll)
                max_ll = log_liks[10];

            K = H * vp * inv_S;
            mu_posts[10] = mp + K * innov;
            var_posts[10] = (RBPF_REAL(1.0) - K * H) * vp;
        }

        /* GPB1 collapse */
        lik_total = RBPF_REAL(0.0);
        mu_acc = RBPF_REAL(0.0);
        mu2_acc = RBPF_REAL(0.0);

        for (k = 0; k < 11; k++)
        {
            rbpf_real_t w = rbpf_exp(log_liks[k] - max_ll);
            lik_total += w;
            mu_acc += w * mu_posts[k];
            mu2_acc += w * (var_posts[k] + mu_posts[k] * mu_posts[k]);
        }

        inv_lik = RBPF_REAL(1.0) / (lik_total + RBPF_EPS);
        mu_new = mu_acc * inv_lik;
        var_new = mu2_acc * inv_lik - mu_new * mu_new;

        if (var_new < RBPF_REAL(1e-6))
            var_new = RBPF_REAL(1e-6);

        rbpf->mu[i] = mu_new;
        rbpf->var[i] = var_new;

        /* Update log-weight */
        ll = max_ll + rbpf_log(lik_total + RBPF_EPS);
        rbpf->log_weight[i] += ll;

        if (rbpf->log_weight[i] > max_log_weight)
        {
            max_log_weight = rbpf->log_weight[i];
        }
    }

    /* Normalize weights */
    sum_weight = RBPF_REAL(0.0);
    for (i = 0; i < n; i++)
    {
        rbpf_real_t w;
        rbpf->log_weight[i] -= max_log_weight;
        w = rbpf_exp(rbpf->log_weight[i]);
        rbpf->w_norm[i] = w;
        sum_weight += w;
    }

    if (sum_weight < RBPF_EPS)
        sum_weight = RBPF_EPS;
    inv_sum = RBPF_REAL(1.0) / sum_weight;

    for (i = 0; i < n; i++)
    {
        rbpf->w_norm[i] *= inv_sum;
    }

    return rbpf_exp(max_log_weight + rbpf_log(sum_weight) - rbpf_log((rbpf_real_t)n));
}

/*═══════════════════════════════════════════════════════════════════════════
 * MKL WORKSPACE MANAGEMENT
 *═══════════════════════════════════════════════════════════════════════════*/

static int ensure_mkl_workspace(RBPF_KSC *rbpf)
{
    const int n_block = RBPF_MKL_BLOCK_SIZE;
    const int n_total_elems = n_block * N_COMPONENTS_TOTAL;
    const size_t sz = sizeof(double);
    size_t total_bytes;
    double *base;

    total_bytes = (7 * n_total_elems * sz) + (n_block * sz) + 1024;

    if (rbpf->mkl_workspace.capacity_bytes >= total_bytes)
    {
        return 0;
    }

    if (rbpf->mkl_workspace.ptr_block)
    {
        mkl_free(rbpf->mkl_workspace.ptr_block);
    }

    rbpf->mkl_workspace.ptr_block = mkl_malloc(total_bytes, 64);
    if (!rbpf->mkl_workspace.ptr_block)
    {
        rbpf->mkl_workspace.capacity_bytes = 0;
        return -1;
    }

    rbpf->mkl_workspace.capacity_bytes = total_bytes;

    base = (double *)rbpf->mkl_workspace.ptr_block;
    rbpf->mkl_workspace.S_all = base;
    base += n_total_elems;
    rbpf->mkl_workspace.log_S_all = base;
    base += n_total_elems;
    rbpf->mkl_workspace.innov_all = base;
    base += n_total_elems;
    rbpf->mkl_workspace.log_lik_all = base;
    base += n_total_elems;
    rbpf->mkl_workspace.lik_all = base;
    base += n_total_elems;
    rbpf->mkl_workspace.mu_post_all = base;
    base += n_total_elems;
    rbpf->mkl_workspace.var_post_all = base;
    base += n_total_elems;
    rbpf->mkl_workspace.max_log_lik = base;

    return 0;
}

void rbpf_ksc_mkl_free(RBPF_KSC *rbpf)
{
    if (!rbpf)
        return;

    if (rbpf->mkl_workspace.ptr_block)
    {
        mkl_free(rbpf->mkl_workspace.ptr_block);
        rbpf->mkl_workspace.ptr_block = NULL;
        rbpf->mkl_workspace.capacity_bytes = 0;
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * MKL-OPTIMIZED UPDATE FUNCTION
 *═══════════════════════════════════════════════════════════════════════════*/

rbpf_real_t rbpf_ksc_update_student_t_robust_mkl(
    RBPF_KSC *rbpf,
    rbpf_real_t y,
    rbpf_real_t nu,
    const RBPF_RobustOCSN *ocsn)
{
    /* MSVC-compatible loop variable declarations */
    int i, k, gi, b_start;
    int base, idx;
    int n_total;
    double H, H2, nu_d, nu_half, nu_plus_1_half, y_d;
    double *S_all, *log_S_all, *innov_all, *log_lik_all;
    double *lik_all, *mu_post_all, *var_post_all, *max_ll_buf;
    double global_max_log_weight;

    /* Fallback if OCSN disabled */
    if (!ocsn || !ocsn->enabled)
    {
        return rbpf_ksc_update_student_t_nu(rbpf, y, nu);
    }

    if (ensure_mkl_workspace(rbpf) != 0)
    {
        return rbpf_ksc_update_student_t_robust(rbpf, y, nu, ocsn);
    }

    n_total = rbpf->n_particles;
    H = 2.0;
    H2 = 4.0;
    nu_d = (double)nu;
    nu_half = nu_d * 0.5;
    nu_plus_1_half = (nu_d + 1.0) * 0.5;
    y_d = (double)y;

    S_all = rbpf->mkl_workspace.S_all;
    log_S_all = rbpf->mkl_workspace.log_S_all;
    innov_all = rbpf->mkl_workspace.innov_all;
    log_lik_all = rbpf->mkl_workspace.log_lik_all;
    lik_all = rbpf->mkl_workspace.lik_all;
    mu_post_all = rbpf->mkl_workspace.mu_post_all;
    var_post_all = rbpf->mkl_workspace.var_post_all;
    max_ll_buf = rbpf->mkl_workspace.max_log_lik;

    global_max_log_weight = -1e30;

    /* Blocked loop */
    for (b_start = 0; b_start < n_total; b_start += RBPF_MKL_BLOCK_SIZE)
    {
        int b_end = (b_start + RBPF_MKL_BLOCK_SIZE > n_total)
                        ? n_total
                        : b_start + RBPF_MKL_BLOCK_SIZE;
        int n_batch = b_end - b_start;
        int n_batch_total = n_batch * N_COMPONENTS_TOTAL;

        /* Phase 1: λ sampling + fill buffers */
        for (i = 0; i < n_batch; i++)
        {
            int r;
            double mp, vp, innov_dom, var_eff, beta, lam, inv_lam;
            double ocsn_var, ocsn_pi, log_1_minus_pi, log_pi;

            gi = b_start + i;
            r = rbpf->regime[gi];
            mp = (double)rbpf->mu_pred[gi];
            vp = (double)rbpf->var_pred[gi];

            innov_dom = (y_d - KSC_MEAN_ST[4]) - H * mp;
            if (innov_dom > 50.0)
                innov_dom = 50.0;
            else if (innov_dom < -50.0)
                innov_dom = -50.0;

            var_eff = H2 * vp + KSC_VAR_ST[4];
            beta = nu_half + 0.5 * innov_dom * innov_dom / var_eff;
            if (beta < 0.1)
                beta = 0.1;
            else if (beta > 100.0)
                beta = 100.0;

            lam = (double)rbpf_pcg32_gamma(&rbpf->pcg[0],
                                           (rbpf_real_t)nu_plus_1_half,
                                           (rbpf_real_t)beta);

            if (lam != lam || lam <= 0.0)
                lam = 1.0;
            else if (lam < RBPF_LAMBDA_FLOOR)
                lam = RBPF_LAMBDA_FLOOR;
            else if (lam > RBPF_LAMBDA_CEIL)
                lam = RBPF_LAMBDA_CEIL;

            rbpf->lambda[gi] = (rbpf_real_t)lam;
            rbpf->log_lambda[gi] = (rbpf_real_t)log(lam);

            inv_lam = 1.0 / lam;

            ocsn_var = (double)ocsn->regime[r].variance;
            ocsn_pi = (double)ocsn->regime[r].prob;
            if (ocsn_var < RBPF_OUTLIER_VAR_MIN)
                ocsn_var = RBPF_OUTLIER_VAR_MIN;
            if (ocsn_var > RBPF_OUTLIER_VAR_MAX)
                ocsn_var = RBPF_OUTLIER_VAR_MAX;

            log_1_minus_pi = log(1.0 - ocsn_pi);
            log_pi = log(ocsn_pi);

            base = i * N_COMPONENTS_TOTAL;

            for (k = 0; k < 10; k++)
            {
                S_all[base + k] = H2 * vp + KSC_VAR_ST[k] * inv_lam;
                innov_all[base + k] = y_d - KSC_MEAN_ST[k] - H * mp;
                log_lik_all[base + k] = log_1_minus_pi + KSC_LOG_PROB_ST[k];
            }

            S_all[base + 10] = H2 * vp + ocsn_var;
            innov_all[base + 10] = y_d - H * mp;
            log_lik_all[base + 10] = log_pi;
        }

        /* Phase 2: Batched log(S) */
        vdLn(n_batch_total, S_all, log_S_all);

/* Phase 3: Complete log-likelihoods + Kalman updates */
#pragma omp parallel for if (n_batch > 256) private(i, gi, base, k, idx)
        for (i = 0; i < n_batch; i++)
        {
            double mp_loc, vp_loc, H_vp, max_ll;
            double S_loc, inv_S, innov_loc, ll, K_loc;

            gi = b_start + i;
            mp_loc = (double)rbpf->mu_pred[gi];
            vp_loc = (double)rbpf->var_pred[gi];
            H_vp = H * vp_loc;

            base = i * N_COMPONENTS_TOTAL;
            max_ll = -1e30;

            for (k = 0; k < N_COMPONENTS_TOTAL; k++)
            {
                idx = base + k;
                S_loc = S_all[idx];
                inv_S = 1.0 / S_loc;
                innov_loc = innov_all[idx];

                ll = log_lik_all[idx] - 0.5 * (log_S_all[idx] + innov_loc * innov_loc * inv_S);
                log_lik_all[idx] = ll;

                if (ll > max_ll)
                    max_ll = ll;

                K_loc = H_vp * inv_S;
                mu_post_all[idx] = mp_loc + K_loc * innov_loc;
                var_post_all[idx] = (1.0 - K_loc * H) * vp_loc;
            }

            max_ll_buf[i] = max_ll;
        }

/* Phase 4: Subtract max + batched exp */
#pragma omp parallel for if (n_batch > 256) private(i, base, k)
        for (i = 0; i < n_batch; i++)
        {
            double m;
            base = i * N_COMPONENTS_TOTAL;
            m = max_ll_buf[i];
            for (k = 0; k < N_COMPONENTS_TOTAL; k++)
            {
                log_lik_all[base + k] -= m;
            }
        }

        vdExp(n_batch_total, log_lik_all, lik_all);

/* Phase 5: GPB1 collapse */
#pragma omp parallel for if (n_batch > 256) private(i, gi, base, k)
        for (i = 0; i < n_batch; i++)
        {
            double sum_lik, mu_acc, mu2_acc;
            double w, mk, vk;
            double inv_sum, mu_new, var_new, ll_total;

            gi = b_start + i;
            base = i * N_COMPONENTS_TOTAL;

            sum_lik = 0.0;
            mu_acc = 0.0;
            mu2_acc = 0.0;

            for (k = 0; k < N_COMPONENTS_TOTAL; k++)
            {
                w = lik_all[base + k];
                mk = mu_post_all[base + k];
                vk = var_post_all[base + k];

                sum_lik += w;
                mu_acc += w * mk;
                mu2_acc += w * (vk + mk * mk);
            }

            inv_sum = 1.0 / (sum_lik + 1e-30);
            mu_new = mu_acc * inv_sum;
            var_new = (mu2_acc * inv_sum) - (mu_new * mu_new);

            if (var_new < 1e-6)
                var_new = 1e-6;
            if (var_new != var_new)
                var_new = (double)rbpf->var_pred[gi];
            if (mu_new != mu_new)
                mu_new = (double)rbpf->mu_pred[gi];

            rbpf->mu[gi] = (rbpf_real_t)mu_new;
            rbpf->var[gi] = (rbpf_real_t)var_new;

            ll_total = max_ll_buf[i] + log(sum_lik + 1e-30);
            if (ll_total < -700.0)
                ll_total = -700.0;

            rbpf->log_weight[gi] += (rbpf_real_t)ll_total;
        }
    }

    /* Phase 6: Global normalization */
    for (i = 0; i < n_total; i++)
    {
        double lw = (double)rbpf->log_weight[i];
        if (lw > global_max_log_weight)
        {
            global_max_log_weight = lw;
        }
    }

    {
        double sum_weight = 0.0;
        double inv_sum, log_marginal;

#pragma omp parallel for reduction(+ : sum_weight) if (n_total > 256) private(i)
        for (i = 0; i < n_total; i++)
        {
            double lw, w;
            lw = (double)rbpf->log_weight[i] - global_max_log_weight;
            rbpf->log_weight[i] = (rbpf_real_t)lw;
            w = exp(lw);
            rbpf->w_norm[i] = (rbpf_real_t)w;
            sum_weight += w;
        }

        if (sum_weight < 1e-30)
            sum_weight = 1e-30;
        inv_sum = 1.0 / sum_weight;

        for (i = 0; i < n_total; i++)
        {
            rbpf->w_norm[i] = (rbpf_real_t)((double)rbpf->w_norm[i] * inv_sum);
        }

        log_marginal = global_max_log_weight + log(sum_weight) - log((double)n_total);
        return (rbpf_real_t)exp(log_marginal);
    }
}

#endif /* RBPF_ENABLE_STUDENT_T */