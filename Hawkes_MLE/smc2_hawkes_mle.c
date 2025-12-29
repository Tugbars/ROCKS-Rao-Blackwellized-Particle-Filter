/**
 * @file smc2_hawkes_mle.c
 * @brief Maximum Likelihood Estimation for Hawkes process parameters
 */

#include "smc2_hawkes_mle.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

/* ═══════════════════════════════════════════════════════════════════════════
 * Configuration
 * ═══════════════════════════════════════════════════════════════════════════ */

HawkesMLEConfig hawkes_mle_default_config(void)
{
    HawkesMLEConfig config;

    /* Parameter bounds (these are fallbacks, fit() uses adaptive bounds) */
    config.alpha_min = 0.01f;
    config.alpha_max = 5.0f;
    config.beta_min = 0.01f;
    config.beta_max = 5.0f;
    config.lambda0_min = 0.01f;
    config.lambda0_max = 10.0f;

    /* Optimizer settings */
    config.max_iterations = 300;
    config.tolerance = 1e-6f;
    config.learning_rate = 0.01f;

    /* Grid search - 7³ = 343 points for better initialization */
    config.grid_points = 7;

    return config;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Core Computations
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * Compute A(i) = Σ_{j<i} exp(-β(tᵢ - tⱼ)) recursively in O(n)
 *
 * Recurrence: A(i) = exp(-β·Δt) · (1 + A(i-1))
 * where Δt = t_i - t_{i-1}
 */
static void compute_A_recursive(const double *timestamps, int n,
                                float beta, double *A_out)
{
    if (n == 0)
        return;

    A_out[0] = 0.0; /* No previous events */

    for (int i = 1; i < n; i++)
    {
        double dt = timestamps[i] - timestamps[i - 1];
        double decay = exp(-beta * dt);
        A_out[i] = decay * (1.0 + A_out[i - 1]);
    }
}

/**
 * Compute log-likelihood and optionally gradients
 *
 * L = Σᵢ log(λ(tᵢ)) - ∫₀ᵀ λ(t)dt
 *
 * where:
 *   λ(tᵢ) = λ₀ + α·A(i)
 *   ∫₀ᵀ λ(t)dt = λ₀·T + (α/β)·Σᵢ(1 - exp(-β(T - tᵢ)))
 */
static double compute_loglik_and_grad(const double *timestamps, int n,
                                      float alpha, float beta, float lambda0,
                                      const double *A, /* Precomputed A(i) */
                                      float *grad_alpha,
                                      float *grad_beta,
                                      float *grad_lambda0)
{
    if (n < 2)
        return -INFINITY;

    double T = timestamps[n - 1] - timestamps[0];
    if (T <= 0.0)
        return -INFINITY;

    /* Sum terms */
    double sum_log_lambda = 0.0;
    double sum_integral = 0.0;

    /* Gradient accumulators */
    double dL_dalpha = 0.0;
    double dL_dbeta = 0.0;
    double dL_dlambda0 = 0.0;

    for (int i = 0; i < n; i++)
    {
        /* Intensity at event i */
        double lambda_i = lambda0 + alpha * A[i];

        /* Guard against log(0) */
        if (lambda_i < 1e-10)
            lambda_i = 1e-10;

        sum_log_lambda += log(lambda_i);

        /* Integral contribution: (α/β)·(1 - exp(-β(T - tᵢ))) */
        double dt_to_end = timestamps[n - 1] - timestamps[i];
        double exp_term = exp(-beta * dt_to_end);
        sum_integral += (1.0 - exp_term);

        /* Gradients if requested */
        if (grad_alpha || grad_beta || grad_lambda0)
        {
            double inv_lambda = 1.0 / lambda_i;

            if (grad_alpha)
            {
                /* ∂log(λᵢ)/∂α = A(i)/λᵢ */
                dL_dalpha += A[i] * inv_lambda;
            }

            if (grad_lambda0)
            {
                /* ∂log(λᵢ)/∂λ₀ = 1/λᵢ */
                dL_dlambda0 += inv_lambda;
            }

            if (grad_beta)
            {
                /* ∂log(λᵢ)/∂β needs ∂A(i)/∂β
                 * This is complex - approximate numerically for now */
            }
        }
    }

    /* Complete log-likelihood */
    double loglik = sum_log_lambda - lambda0 * T - (alpha / beta) * sum_integral;

    /* Complete gradients */
    if (grad_alpha)
    {
        /* ∂L/∂α = Σᵢ A(i)/λᵢ - (1/β)·Σᵢ(1 - exp(-β(T-tᵢ))) */
        *grad_alpha = (float)(dL_dalpha - sum_integral / beta);
    }

    if (grad_lambda0)
    {
        /* ∂L/∂λ₀ = Σᵢ 1/λᵢ - T */
        *grad_lambda0 = (float)(dL_dlambda0 - T);
    }

    if (grad_beta)
    {
        /* Numerical gradient for beta (complex analytical form) */
        double eps = 1e-5;
        double *A_plus = (double *)malloc(n * sizeof(double));
        compute_A_recursive(timestamps, n, beta + eps, A_plus);

        double sum_log_plus = 0.0;
        double sum_int_plus = 0.0;
        for (int i = 0; i < n; i++)
        {
            double lambda_i = lambda0 + alpha * A_plus[i];
            if (lambda_i < 1e-10)
                lambda_i = 1e-10;
            sum_log_plus += log(lambda_i);

            double dt_to_end = timestamps[n - 1] - timestamps[i];
            sum_int_plus += (1.0 - exp(-(beta + eps) * dt_to_end));
        }
        double loglik_plus = sum_log_plus - lambda0 * T - (alpha / (beta + eps)) * sum_int_plus;

        *grad_beta = (float)((loglik_plus - loglik) / eps);
        free(A_plus);
    }

    return loglik;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Public API
 * ═══════════════════════════════════════════════════════════════════════════ */

double hawkes_mle_loglik(const double *timestamps, int n,
                         float alpha, float beta, float lambda0)
{
    if (n < 2)
        return -INFINITY;

    /* Compute A recursively */
    double *A = (double *)malloc(n * sizeof(double));
    compute_A_recursive(timestamps, n, beta, A);

    double loglik = compute_loglik_and_grad(timestamps, n,
                                            alpha, beta, lambda0,
                                            A, NULL, NULL, NULL);
    free(A);
    return loglik;
}

void hawkes_mle_intensity(const double *timestamps, int n,
                          float alpha, float beta, float lambda0,
                          float *lambda_out)
{
    if (n == 0)
        return;

    double *A = (double *)malloc(n * sizeof(double));
    compute_A_recursive(timestamps, n, beta, A);

    for (int i = 0; i < n; i++)
    {
        lambda_out[i] = lambda0 + alpha * (float)A[i];
    }

    free(A);
}

HawkesMLE hawkes_mle_fit(const double *timestamps, int n,
                         const HawkesMLEConfig *config)
{
    HawkesMLE result;
    memset(&result, 0, sizeof(result));

    /* Use defaults if no config provided */
    HawkesMLEConfig cfg;
    if (config)
    {
        cfg = *config;
    }
    else
    {
        cfg = hawkes_mle_default_config();
    }

    if (n < 10)
    {
        /* Too few events for reliable estimation */
        result.converged = false;
        result.alpha = 0.3f;
        result.beta = 1.0f;
        result.lambda0 = (float)n / (float)(timestamps[n - 1] - timestamps[0]);
        return result;
    }

    result.n_events = n;
    result.duration = timestamps[n - 1] - timestamps[0];

    /* ═══════════════════════════════════════════════════════════════════════
     * Moment-based Initialization
     *
     * For stationary Hawkes:
     *   E[λ] = λ₀ / (1 - α/β)   (mean intensity)
     *   Var[Δt] relates to clustering
     * ═══════════════════════════════════════════════════════════════════════ */

    /* Compute empirical statistics */
    float emp_rate = (float)n / (float)result.duration;

    /* Compute inter-arrival time statistics */
    double sum_dt = 0.0, sum_dt2 = 0.0;
    for (int i = 1; i < n; i++)
    {
        double dt = timestamps[i] - timestamps[i - 1];
        sum_dt += dt;
        sum_dt2 += dt * dt;
    }
    double mean_dt = sum_dt / (n - 1);
    double var_dt = sum_dt2 / (n - 1) - mean_dt * mean_dt;
    double cv_dt = sqrt(var_dt) / mean_dt; /* Coefficient of variation */

    /* For Poisson (no clustering): CV = 1
     * For Hawkes with clustering: CV > 1
     * Use CV to estimate branching ratio roughly */
    float est_branching = (cv_dt > 1.0) ? fminf(0.8f, (float)(cv_dt - 1.0) * 0.5f) : 0.2f;

    /* Initial estimates */
    float init_lambda0 = emp_rate * (1.0f - est_branching);
    float init_beta = 1.0f; /* 1 second time constant as default */
    float init_alpha = est_branching * init_beta;

    /* Clamp to valid range */
    if (init_lambda0 < 0.1f)
        init_lambda0 = 0.1f;
    if (init_alpha < 0.05f)
        init_alpha = 0.05f;

    /* ═══════════════════════════════════════════════════════════════════════
     * Adaptive bounds based on data
     * ═══════════════════════════════════════════════════════════════════════ */

    float alpha_min = 0.01f;
    float alpha_max = fmaxf(2.0f, init_alpha * 3.0f);
    float beta_min = 0.1f;
    float beta_max = fmaxf(5.0f, init_beta * 5.0f);
    float lambda0_min = 0.01f;
    float lambda0_max = fmaxf(emp_rate * 2.0f, 5.0f);

    /* ═══════════════════════════════════════════════════════════════════════
     * Phase 1: Grid Search centered on moment estimates
     * ═══════════════════════════════════════════════════════════════════════ */

    float best_alpha = init_alpha;
    float best_beta = init_beta;
    float best_lambda0 = init_lambda0;
    double best_loglik = -INFINITY;

    int G = cfg.grid_points;

    /* Grid centered on initial estimates */
    float alpha_lo = fmaxf(alpha_min, init_alpha * 0.2f);
    float alpha_hi = fminf(alpha_max, init_alpha * 3.0f);
    float beta_lo = fmaxf(beta_min, init_beta * 0.2f);
    float beta_hi = fminf(beta_max, init_beta * 5.0f);
    float lambda0_lo = fmaxf(lambda0_min, init_lambda0 * 0.3f);
    float lambda0_hi = fminf(lambda0_max, init_lambda0 * 3.0f);

    for (int ia = 0; ia < G; ia++)
    {
        float alpha = alpha_lo + (alpha_hi - alpha_lo) * ia / (G - 1);

        for (int ib = 0; ib < G; ib++)
        {
            float beta = beta_lo + (beta_hi - beta_lo) * ib / (G - 1);

            /* Skip if branching ratio >= 0.95 (near non-stationary) */
            if (alpha / beta >= 0.95f)
                continue;

            for (int il = 0; il < G; il++)
            {
                float lambda0 = lambda0_lo + (lambda0_hi - lambda0_lo) * il / (G - 1);

                double loglik = hawkes_mle_loglik(timestamps, n, alpha, beta, lambda0);

                if (loglik > best_loglik)
                {
                    best_loglik = loglik;
                    best_alpha = alpha;
                    best_beta = beta;
                    best_lambda0 = lambda0;
                }
            }
        }
    }

    /* Also try the moment-based initial estimate directly */
    double init_loglik = hawkes_mle_loglik(timestamps, n, init_alpha, init_beta, init_lambda0);
    if (init_loglik > best_loglik)
    {
        best_alpha = init_alpha;
        best_beta = init_beta;
        best_lambda0 = init_lambda0;
        best_loglik = init_loglik;
    }

    /* ═══════════════════════════════════════════════════════════════════════
     * Phase 2: Gradient Descent Refinement
     * ═══════════════════════════════════════════════════════════════════════ */

    float alpha = best_alpha;
    float beta = best_beta;
    float lambda0 = best_lambda0;

    double *A = (double *)malloc(n * sizeof(double));

    float lr = cfg.learning_rate;
    double prev_loglik = best_loglik;
    int iter;

    for (iter = 0; iter < cfg.max_iterations; iter++)
    {
        /* Compute A for current beta */
        compute_A_recursive(timestamps, n, beta, A);

        /* Compute gradients */
        float grad_alpha, grad_beta, grad_lambda0;
        double loglik = compute_loglik_and_grad(timestamps, n,
                                                alpha, beta, lambda0, A,
                                                &grad_alpha, &grad_beta, &grad_lambda0);

        /* Gradient ascent (maximizing likelihood) */
        float new_alpha = alpha + lr * grad_alpha;
        float new_beta = beta + lr * grad_beta;
        float new_lambda0 = lambda0 + lr * grad_lambda0;

        /* Enforce adaptive bounds */
        if (new_alpha < alpha_min)
            new_alpha = alpha_min;
        if (new_alpha > alpha_max)
            new_alpha = alpha_max;
        if (new_beta < beta_min)
            new_beta = beta_min;
        if (new_beta > beta_max)
            new_beta = beta_max;
        if (new_lambda0 < lambda0_min)
            new_lambda0 = lambda0_min;
        if (new_lambda0 > lambda0_max)
            new_lambda0 = lambda0_max;

        /* Enforce stationarity: α/β < 0.95 */
        if (new_alpha / new_beta >= 0.95f)
        {
            new_alpha = new_beta * 0.90f;
        }

        /* Check new likelihood */
        double new_loglik = hawkes_mle_loglik(timestamps, n, new_alpha, new_beta, new_lambda0);

        if (new_loglik > loglik)
        {
            /* Accept step */
            alpha = new_alpha;
            beta = new_beta;
            lambda0 = new_lambda0;

            /* Increase learning rate */
            lr *= 1.1f;
            if (lr > 0.5f)
                lr = 0.5f;
        }
        else
        {
            /* Reject step, decrease learning rate */
            lr *= 0.5f;
            if (lr < 1e-6f)
                break;
        }

        /* Check convergence */
        if (fabs(new_loglik - prev_loglik) < cfg.tolerance)
        {
            result.converged = true;
            break;
        }
        prev_loglik = new_loglik;
    }

    free(A);

    /* ═══════════════════════════════════════════════════════════════════════
     * Finalize Results
     * ═══════════════════════════════════════════════════════════════════════ */

    result.alpha = alpha;
    result.beta = beta;
    result.lambda0 = lambda0;
    result.iterations = iter;

    /* Derived quantities */
    result.branching_ratio = alpha / beta;
    if (result.branching_ratio < 1.0f)
    {
        result.mean_intensity = lambda0 / (1.0f - result.branching_ratio);
        result.mean_cluster_size = 1.0f / (1.0f - result.branching_ratio);
    }
    else
    {
        result.mean_intensity = INFINITY;
        result.mean_cluster_size = INFINITY;
    }

    /* Fit quality */
    result.log_likelihood = (float)hawkes_mle_loglik(timestamps, n, alpha, beta, lambda0);
    result.aic = -2.0f * result.log_likelihood + 2.0f * 3; /* 3 parameters */
    result.bic = -2.0f * result.log_likelihood + logf((float)n) * 3;

    if (!result.converged && iter >= cfg.max_iterations)
    {
        result.converged = (lr > 1e-6f); /* Converged if lr didn't collapse */
    }

    return result;
}

float hawkes_mle_estimate_gamma(const double *timestamps,
                                const float *returns, int n,
                                float alpha, float beta, float lambda0)
{
    if (n < 20)
        return 0.1f; /* Default */

    /* Compute intensity at each event */
    float *lambda = (float *)malloc(n * sizeof(float));
    hawkes_mle_intensity(timestamps, n, alpha, beta, lambda0, lambda);

    /* Fit: log|y| = a + γ·log(λ/λ₀) + ε via OLS
     *
     * Let x = log(λ/λ₀), y_obs = log|return|
     * γ = Cov(x, y_obs) / Var(x)
     */

    double sum_x = 0.0, sum_y = 0.0;
    double sum_xx = 0.0, sum_xy = 0.0;
    int valid = 0;

    for (int i = 0; i < n; i++)
    {
        float r = returns[i];
        if (fabsf(r) < 1e-10f)
            continue; /* Skip near-zero returns */

        float x = logf(lambda[i] / lambda0);
        float y = logf(fabsf(r));

        sum_x += x;
        sum_y += y;
        sum_xx += x * x;
        sum_xy += x * y;
        valid++;
    }

    free(lambda);

    if (valid < 10)
        return 0.1f;

    double mean_x = sum_x / valid;
    double mean_y = sum_y / valid;
    double var_x = sum_xx / valid - mean_x * mean_x;
    double cov_xy = sum_xy / valid - mean_x * mean_y;

    if (var_x < 1e-10)
        return 0.1f;

    float gamma = (float)(cov_xy / var_x);

    /* The relationship is: log|y| ≈ 0.5·h + noise
     * and h += γ·log(λ/λ₀)
     * So our regression gives ~0.5·γ
     * Adjust: γ_actual = 2 * regression_coef
     */
    gamma *= 2.0f;

    /* Clamp to reasonable range */
    if (gamma < 0.0f)
        gamma = 0.0f;
    if (gamma > 0.5f)
        gamma = 0.5f;

    return gamma;
}

void hawkes_mle_print(const HawkesMLE *result)
{
    printf("\n  Hawkes MLE Results:\n");
    printf("  ┌────────────────────────┬────────────────┐\n");
    printf("  │ Parameter              │ Value          │\n");
    printf("  ├────────────────────────┼────────────────┤\n");
    printf("  │ α (jump size)          │ %14.4f │\n", result->alpha);
    printf("  │ β (decay rate)         │ %14.4f │\n", result->beta);
    printf("  │ λ₀ (baseline)          │ %14.4f │\n", result->lambda0);
    printf("  ├────────────────────────┼────────────────┤\n");
    printf("  │ Branching ratio (α/β)  │ %14.4f │\n", result->branching_ratio);
    printf("  │ Mean intensity         │ %14.4f │\n", result->mean_intensity);
    printf("  │ Mean cluster size      │ %14.4f │\n", result->mean_cluster_size);
    printf("  ├────────────────────────┼────────────────┤\n");
    printf("  │ Log-likelihood         │ %14.2f │\n", result->log_likelihood);
    printf("  │ AIC                    │ %14.2f │\n", result->aic);
    printf("  │ BIC                    │ %14.2f │\n", result->bic);
    printf("  ├────────────────────────┼────────────────┤\n");
    printf("  │ Events                 │ %14d │\n", result->n_events);
    printf("  │ Duration (sec)         │ %14.1f │\n", result->duration);
    printf("  │ Iterations             │ %14d │\n", result->iterations);
    printf("  │ Converged              │ %14s │\n", result->converged ? "yes" : "no");
    printf("  └────────────────────────┴────────────────┘\n\n");
}