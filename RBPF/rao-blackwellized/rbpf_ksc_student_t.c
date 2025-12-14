/**
 * @file rbpf_ksc_student_t.c
 * @brief Student-t observation model for RBPF-KSC
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * STUDENT-T OBSERVATION MODEL
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Replaces Gaussian innovations with Student-t for fat-tail robustness:
 *
 *   Gaussian: r_t = σ_t × ε_t,    ε_t ~ N(0,1)
 *   Student:  r_t = σ_t × ε_t,    ε_t ~ t_ν(0,1)
 *
 * Key insight: t_ν is a scale mixture of Gaussians:
 *   ε | λ ~ N(0, 1/λ),  λ ~ Gamma(ν/2, ν/2)
 *
 * This preserves the KSC machinery — we just:
 *   1. Sample λ from conditional posterior
 *   2. Shift observation by log(λ)
 *   3. Run standard 10-component KSC update
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * REGIME-DEPENDENT ν FOR MMPF
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Different hypotheses get different tail thickness:
 *
 *   Calm:   ν=10 → P(5σ) ≈ 10⁻⁵  (near-Gaussian)
 *   Trend:  ν=5  → P(5σ) ≈ 10⁻³  (moderate tails)
 *   Crisis: ν=3  → P(5σ) ≈ 10⁻²  (heavy tails)
 *
 * Crisis naturally wins Bayesian model comparison during extreme events
 * because it EXPECTED fat tails — no hacks needed.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * REFERENCES
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * - Geweke (1993): Bayesian treatment of SV with t-errors
 * - Chib, Nardari & Shephard (2002): MCMC for SV-t
 * - Jacquier, Polson & Rossi (2004): Bayesian analysis of SV models
 */

#include "rbpf_ksc.h"
#include <string.h>
#include <stdio.h>

/*═══════════════════════════════════════════════════════════════════════════
 * OMORI 10-COMPONENT MIXTURE (from rbpf_ksc.c)
 *
 * These must match the main implementation exactly.
 *═══════════════════════════════════════════════════════════════════════════*/

static const rbpf_real_t KSC_PROB[KSC_N_COMPONENTS] = {
    RBPF_REAL(0.00609), RBPF_REAL(0.04775), RBPF_REAL(0.13057),
    RBPF_REAL(0.20674), RBPF_REAL(0.22715), RBPF_REAL(0.18842),
    RBPF_REAL(0.12047), RBPF_REAL(0.05591), RBPF_REAL(0.01575),
    RBPF_REAL(0.00115)
};

static const rbpf_real_t KSC_MEAN[KSC_N_COMPONENTS] = {
    RBPF_REAL(1.92677), RBPF_REAL(1.34744), RBPF_REAL(0.73504),
    RBPF_REAL(0.02266), RBPF_REAL(-0.85173), RBPF_REAL(-1.97278),
    RBPF_REAL(-3.46788), RBPF_REAL(-5.55246), RBPF_REAL(-8.68384),
    RBPF_REAL(-14.65000)
};

static const rbpf_real_t KSC_VAR[KSC_N_COMPONENTS] = {
    RBPF_REAL(0.11265), RBPF_REAL(0.17788), RBPF_REAL(0.26768),
    RBPF_REAL(0.40611), RBPF_REAL(0.62699), RBPF_REAL(0.98583),
    RBPF_REAL(1.57469), RBPF_REAL(2.54498), RBPF_REAL(4.16591),
    RBPF_REAL(7.33342)
};

/* Precomputed: -0.5 * log(2π * var) */
static const rbpf_real_t KSC_LOG_NORM[KSC_N_COMPONENTS] = {
    RBPF_REAL(-0.97538), RBPF_REAL(-1.20349), RBPF_REAL(-1.41047),
    RBPF_REAL(-1.62152), RBPF_REAL(-1.84068), RBPF_REAL(-2.06824),
    RBPF_REAL(-2.30457), RBPF_REAL(-2.54977), RBPF_REAL(-2.80378),
    RBPF_REAL(-3.09153)
};

/*═══════════════════════════════════════════════════════════════════════════
 * SPECIAL FUNCTIONS
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Digamma function ψ(x) = d/dx log(Γ(x))
 * 
 * Asymptotic expansion for x > 6, recursion for smaller x.
 * Accuracy: ~10⁻⁸ relative error.
 */
static rbpf_real_t digamma(rbpf_real_t x)
{
    rbpf_real_t result = RBPF_REAL(0.0);
    
    /* Use recursion ψ(x) = ψ(x+1) - 1/x to shift x > 6 */
    while (x < RBPF_REAL(6.0)) {
        result -= RBPF_REAL(1.0) / x;
        x += RBPF_REAL(1.0);
    }
    
    /* Asymptotic expansion for large x */
    rbpf_real_t inv_x = RBPF_REAL(1.0) / x;
    rbpf_real_t inv_x2 = inv_x * inv_x;
    
    result += rbpf_log(x) - RBPF_REAL(0.5) * inv_x;
    result -= inv_x2 * (RBPF_REAL(1.0/12.0) 
                      - inv_x2 * (RBPF_REAL(1.0/120.0) 
                      - inv_x2 * RBPF_REAL(1.0/252.0)));
    
    return result;
}

/**
 * Trigamma function ψ'(x) = d²/dx² log(Γ(x))
 * 
 * Used for Newton-Raphson in ν estimation.
 */
static rbpf_real_t trigamma(rbpf_real_t x)
{
    rbpf_real_t result = RBPF_REAL(0.0);
    
    /* Use recursion ψ'(x) = ψ'(x+1) + 1/x² to shift x > 6 */
    while (x < RBPF_REAL(6.0)) {
        result += RBPF_REAL(1.0) / (x * x);
        x += RBPF_REAL(1.0);
    }
    
    /* Asymptotic expansion */
    rbpf_real_t inv_x = RBPF_REAL(1.0) / x;
    rbpf_real_t inv_x2 = inv_x * inv_x;
    
    result += inv_x + RBPF_REAL(0.5) * inv_x2;
    result += inv_x2 * inv_x * (RBPF_REAL(1.0/6.0) 
                              - inv_x2 * (RBPF_REAL(1.0/30.0) 
                              - inv_x2 * RBPF_REAL(1.0/42.0)));
    
    return result;
}

/*═══════════════════════════════════════════════════════════════════════════
 * STUDENT-T CONFIGURATION API
 *═══════════════════════════════════════════════════════════════════════════*/

void rbpf_ksc_enable_student_t(RBPF_KSC *rbpf, rbpf_real_t nu)
{
    if (!rbpf) return;
    
    /* Clamp ν to valid range */
    if (nu < RBPF_NU_FLOOR) nu = RBPF_NU_FLOOR;
    if (nu > RBPF_NU_CEIL) nu = RBPF_NU_CEIL;
    
    rbpf->student_t_enabled = 1;
    
    /* Set all regimes to default ν */
    for (int r = 0; r < rbpf->n_regimes; r++) {
        rbpf->student_t[r].enabled = 1;
        rbpf->student_t[r].nu = nu;
        rbpf->student_t[r].nu_floor = RBPF_NU_FLOOR;
        rbpf->student_t[r].nu_ceil = RBPF_NU_CEIL;
        rbpf->student_t[r].learn_nu = 0;
        rbpf->student_t[r].nu_learning_rate = RBPF_REAL(0.99);
        
        /* Reset learning stats */
        rbpf->student_t_stats[r].sum_lambda = RBPF_REAL(0.0);
        rbpf->student_t_stats[r].sum_lambda_sq = RBPF_REAL(0.0);
        rbpf->student_t_stats[r].sum_log_lambda = RBPF_REAL(0.0);
        rbpf->student_t_stats[r].n_eff = RBPF_REAL(0.0);
        rbpf->student_t_stats[r].nu_estimate = nu;
    }
    
    /* Initialize λ arrays if not already done */
    if (rbpf->lambda == NULL) {
        /* These should be allocated in rbpf_ksc_create() */
        /* For safety, we check here */
        fprintf(stderr, "Warning: lambda arrays not allocated. "
                        "Student-t may not work correctly.\n");
    }
}

void rbpf_ksc_disable_student_t(RBPF_KSC *rbpf)
{
    if (!rbpf) return;
    
    rbpf->student_t_enabled = 0;
    
    for (int r = 0; r < rbpf->n_regimes; r++) {
        rbpf->student_t[r].enabled = 0;
    }
}

void rbpf_ksc_set_student_t_nu(RBPF_KSC *rbpf, int regime, rbpf_real_t nu)
{
    if (!rbpf) return;
    if (regime < 0 || regime >= rbpf->n_regimes) return;
    
    /* Clamp ν */
    if (nu < RBPF_NU_FLOOR) nu = RBPF_NU_FLOOR;
    if (nu > RBPF_NU_CEIL) nu = RBPF_NU_CEIL;
    
    rbpf->student_t[regime].nu = nu;
    rbpf->student_t_stats[regime].nu_estimate = nu;
}

void rbpf_ksc_enable_nu_learning(RBPF_KSC *rbpf, int regime, rbpf_real_t learning_rate)
{
    if (!rbpf) return;
    if (regime < 0 || regime >= rbpf->n_regimes) return;
    
    /* Clamp learning rate */
    if (learning_rate < RBPF_REAL(0.9)) learning_rate = RBPF_REAL(0.9);
    if (learning_rate > RBPF_REAL(0.999)) learning_rate = RBPF_REAL(0.999);
    
    rbpf->student_t[regime].learn_nu = 1;
    rbpf->student_t[regime].nu_learning_rate = learning_rate;
}

void rbpf_ksc_disable_nu_learning(RBPF_KSC *rbpf, int regime)
{
    if (!rbpf) return;
    if (regime < 0 || regime >= rbpf->n_regimes) return;
    
    rbpf->student_t[regime].learn_nu = 0;
}

rbpf_real_t rbpf_ksc_get_nu(const RBPF_KSC *rbpf, int regime)
{
    if (!rbpf) return RBPF_NU_DEFAULT;
    if (regime < 0 || regime >= rbpf->n_regimes) return RBPF_NU_DEFAULT;
    
    if (rbpf->student_t[regime].learn_nu) {
        return rbpf->student_t_stats[regime].nu_estimate;
    }
    return rbpf->student_t[regime].nu;
}

void rbpf_ksc_get_lambda_stats(const RBPF_KSC *rbpf, int regime,
                                rbpf_real_t *mean_out, rbpf_real_t *var_out,
                                rbpf_real_t *n_eff_out)
{
    if (!rbpf || regime < 0 || regime >= rbpf->n_regimes) {
        if (mean_out) *mean_out = RBPF_REAL(1.0);
        if (var_out) *var_out = RBPF_REAL(0.0);
        if (n_eff_out) *n_eff_out = RBPF_REAL(0.0);
        return;
    }
    
    const RBPF_StudentT_Stats *stats = &rbpf->student_t_stats[regime];
    
    if (stats->n_eff < RBPF_REAL(1.0)) {
        if (mean_out) *mean_out = RBPF_REAL(1.0);
        if (var_out) *var_out = RBPF_REAL(0.0);
        if (n_eff_out) *n_eff_out = RBPF_REAL(0.0);
        return;
    }
    
    rbpf_real_t mean = stats->sum_lambda / stats->n_eff;
    rbpf_real_t var = stats->sum_lambda_sq / stats->n_eff - mean * mean;
    if (var < RBPF_REAL(0.0)) var = RBPF_REAL(0.0);
    
    if (mean_out) *mean_out = mean;
    if (var_out) *var_out = var;
    if (n_eff_out) *n_eff_out = stats->n_eff;
}

void rbpf_ksc_reset_nu_learning(RBPF_KSC *rbpf, int regime)
{
    if (!rbpf) return;
    if (regime < 0 || regime >= rbpf->n_regimes) return;
    
    RBPF_StudentT_Stats *stats = &rbpf->student_t_stats[regime];
    stats->sum_lambda = RBPF_REAL(0.0);
    stats->sum_lambda_sq = RBPF_REAL(0.0);
    stats->sum_log_lambda = RBPF_REAL(0.0);
    stats->n_eff = RBPF_REAL(0.0);
    stats->nu_estimate = rbpf->student_t[regime].nu;
}

/*═══════════════════════════════════════════════════════════════════════════
 * ν ESTIMATION FROM λ STATISTICS
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Estimate ν from accumulated λ statistics using digamma method
 * 
 * For λ ~ Gamma(ν/2, ν/2):
 *   E[log(λ)] = ψ(ν/2) - log(ν/2)
 * 
 * Solve via Newton-Raphson.
 */
static rbpf_real_t estimate_nu_digamma(const RBPF_StudentT_Stats *stats,
                                        rbpf_real_t nu_floor, rbpf_real_t nu_ceil)
{
    if (stats->n_eff < RBPF_REAL(20.0)) {
        /* Not enough data, return midpoint of range */
        return (nu_floor + nu_ceil) / RBPF_REAL(2.0);
    }
    
    rbpf_real_t mean_log_lambda = stats->sum_log_lambda / stats->n_eff;
    
    /* Newton-Raphson: find ν such that ψ(ν/2) - log(ν/2) = mean_log_lambda */
    rbpf_real_t nu = stats->nu_estimate;  /* Start from current estimate */
    if (nu < nu_floor) nu = nu_floor;
    if (nu > nu_ceil) nu = nu_ceil;
    
    for (int iter = 0; iter < 15; iter++) {
        rbpf_real_t half_nu = nu / RBPF_REAL(2.0);
        rbpf_real_t psi = digamma(half_nu);
        rbpf_real_t target = psi - rbpf_log(half_nu);
        rbpf_real_t error = mean_log_lambda - target;
        
        if (rbpf_fabs(error) < RBPF_REAL(1e-6)) break;
        
        /* Gradient: d/dν [ψ(ν/2) - log(ν/2)] = ψ'(ν/2)/2 - 1/ν */
        rbpf_real_t grad = trigamma(half_nu) / RBPF_REAL(2.0) - RBPF_REAL(1.0) / nu;
        
        /* Avoid division by zero */
        if (rbpf_fabs(grad) < RBPF_REAL(1e-10)) break;
        
        nu -= error / grad;
        
        /* Clamp to valid range */
        if (nu < nu_floor) nu = nu_floor;
        if (nu > nu_ceil) nu = nu_ceil;
    }
    
    return nu;
}

/**
 * Estimate ν using method of moments on Var[λ]
 * 
 * For λ ~ Gamma(ν/2, ν/2):
 *   E[λ] = 1,  Var[λ] = 2/ν
 *   
 * Therefore: ν = 2 / Var[λ]
 * 
 * Simpler than digamma but less accurate for small ν.
 */
static rbpf_real_t estimate_nu_moments(const RBPF_StudentT_Stats *stats,
                                        rbpf_real_t nu_floor, rbpf_real_t nu_ceil)
{
    if (stats->n_eff < RBPF_REAL(20.0)) {
        return (nu_floor + nu_ceil) / RBPF_REAL(2.0);
    }
    
    rbpf_real_t mean = stats->sum_lambda / stats->n_eff;
    rbpf_real_t var = stats->sum_lambda_sq / stats->n_eff - mean * mean;
    
    /* Floor variance to prevent ν → ∞ */
    if (var < RBPF_REAL(0.01)) var = RBPF_REAL(0.01);
    
    rbpf_real_t nu = RBPF_REAL(2.0) / var;
    
    if (nu < nu_floor) nu = nu_floor;
    if (nu > nu_ceil) nu = nu_ceil;
    
    return nu;
}

/*═══════════════════════════════════════════════════════════════════════════
 * CORE STUDENT-T UPDATE
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Student-t Kalman update via auxiliary variable λ
 * 
 * Algorithm:
 *   For each particle i:
 *     1. Get ν for particle's regime
 *     2. Compute innovation at λ=1 (prior mean)
 *     3. Sample λ from conditional posterior Gamma((ν+1)/2, (ν + innov²/v²)/2)
 *     4. Shift observation: y_shifted = y + log(λ)
 *     5. Run 10-component KSC update with y_shifted
 *     6. Accumulate λ statistics for ν learning
 * 
 * Observation model:
 *   y = log(r²) = 2ℓ + log(ε²)
 *   ε | λ ~ N(0, 1/λ)
 *   log(ε²) | λ = log(χ²/λ) = log(χ²) - log(λ)
 *   
 *   Therefore: y + log(λ) = 2ℓ + log(χ²) ← standard KSC form
 */
rbpf_real_t rbpf_ksc_update_student_t(RBPF_KSC *rbpf, rbpf_real_t y)
{
    if (!rbpf) return RBPF_REAL(0.0);
    if (!rbpf->student_t_enabled) {
        /* Fall back to Gaussian update */
        return rbpf_ksc_update(rbpf, y);
    }
    
    const int n = rbpf->n_particles;
    const rbpf_real_t H = RBPF_REAL(2.0);  /* Observation matrix: y = 2ℓ + noise */
    
    /* Workspace pointers */
    rbpf_real_t *mu_pred = rbpf->mu_pred;
    rbpf_real_t *var_pred = rbpf->var_pred;
    rbpf_real_t *mu = rbpf->mu;
    rbpf_real_t *var = rbpf->var;
    rbpf_real_t *log_weight = rbpf->log_weight;
    rbpf_real_t *lambda = rbpf->lambda;
    rbpf_real_t *log_lambda = rbpf->log_lambda;
    const int *regime = rbpf->regime;
    
    /* Per-regime λ accumulators for ν learning */
    rbpf_real_t regime_sum_lambda[RBPF_MAX_REGIMES] = {0};
    rbpf_real_t regime_sum_lambda_sq[RBPF_MAX_REGIMES] = {0};
    rbpf_real_t regime_sum_log_lambda[RBPF_MAX_REGIMES] = {0};
    rbpf_real_t regime_sum_weight[RBPF_MAX_REGIMES] = {0};
    
    /* Marginal likelihood accumulator (log-sum-exp) */
    rbpf_real_t max_log_weight = -RBPF_REAL(1e30);
    
    /*═══════════════════════════════════════════════════════════════════════
     * PHASE 1: Sample λ and compute shifted observations
     *═══════════════════════════════════════════════════════════════════════*/
    
    for (int i = 0; i < n; i++) {
        int r = regime[i];
        rbpf_real_t nu = rbpf->student_t[r].nu;
        
        /* Predicted observation variance (average across KSC components) */
        /* H² × var_pred + E[v²_KSC] ≈ 4 × var_pred + 1.5 */
        rbpf_real_t v2_eff = H * H * var_pred[i] + RBPF_REAL(1.5);
        
        /* Innovation at λ=1 */
        rbpf_real_t y_pred = H * mu_pred[i] + RBPF_REAL(-1.27);  /* -1.27 = E[log(χ²(1))] */
        rbpf_real_t innov = y - y_pred;
        
        /* Conditional posterior for λ:
         * p(λ | y, ℓ) ∝ Gamma(λ | ν/2, ν/2) × N(y | 2ℓ + ..., v²/λ)
         * 
         * After conjugacy:
         *   λ | y, ℓ ~ Gamma((ν+1)/2, (ν + (y-ŷ)²/v²)/2)
         */
        rbpf_real_t alpha_post = (nu + RBPF_REAL(1.0)) / RBPF_REAL(2.0);
        rbpf_real_t beta_post = (nu + innov * innov / v2_eff) / RBPF_REAL(2.0);
        
        /* Sample λ from Gamma posterior */
        rbpf_real_t lam = rbpf_pcg32_gamma(&rbpf->pcg[0], alpha_post, beta_post);
        
        /* Clamp λ for numerical stability */
        if (lam < RBPF_LAMBDA_FLOOR) lam = RBPF_LAMBDA_FLOOR;
        if (lam > RBPF_LAMBDA_CEIL) lam = RBPF_LAMBDA_CEIL;
        
        lambda[i] = lam;
        log_lambda[i] = rbpf_log(lam);
        
        /* Accumulate for ν learning */
        regime_sum_lambda[r] += lam;
        regime_sum_lambda_sq[r] += lam * lam;
        regime_sum_log_lambda[r] += log_lambda[i];
        regime_sum_weight[r] += RBPF_REAL(1.0);
    }
    
    /*═══════════════════════════════════════════════════════════════════════
     * PHASE 2: KSC 10-component mixture update with shifted observations
     *═══════════════════════════════════════════════════════════════════════*/
    
    /* For each particle, run GPB1 collapse across 10 KSC components */
    for (int i = 0; i < n; i++) {
        /* Shifted observation: undoes the λ scaling in log-space */
        rbpf_real_t y_shifted = y + log_lambda[i];
        
        rbpf_real_t mu_accum = RBPF_REAL(0.0);
        rbpf_real_t var_accum = RBPF_REAL(0.0);
        rbpf_real_t log_lik_total = -RBPF_REAL(1e30);  /* For log-sum-exp */
        
        /* Predicted state */
        rbpf_real_t mu_p = mu_pred[i];
        rbpf_real_t var_p = var_pred[i];
        
        /* Loop over 10 KSC mixture components */
        for (int k = 0; k < KSC_N_COMPONENTS; k++) {
            rbpf_real_t prob_k = KSC_PROB[k];
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
            rbpf_real_t mu_k = mu_p + K_k * innov_k;
            rbpf_real_t var_k_post = var_p - K_k * H * var_p;
            if (var_k_post < RBPF_REAL(1e-10)) var_k_post = RBPF_REAL(1e-10);
            
            /* Log-likelihood for this component */
            rbpf_real_t log_lik_k = KSC_LOG_NORM[k] 
                                  - RBPF_REAL(0.5) * rbpf_log(S_k)
                                  - RBPF_REAL(0.5) * innov_k * innov_k * S_inv
                                  + rbpf_log(prob_k);
            
            /* Log-sum-exp accumulation for total likelihood */
            if (log_lik_k > log_lik_total) {
                rbpf_real_t delta = log_lik_total - log_lik_k;
                log_lik_total = log_lik_k + rbpf_log(RBPF_REAL(1.0) + rbpf_exp(delta));
            } else {
                log_lik_total = log_lik_total + rbpf_log(RBPF_REAL(1.0) + rbpf_exp(log_lik_k - log_lik_total));
            }
            
            /* Weight for GPB1 collapse (unnormalized) */
            rbpf_real_t lik_k = rbpf_exp(log_lik_k);
            
            /* Accumulate for moment matching (GPB1) */
            mu_accum += lik_k * mu_k;
            var_accum += lik_k * (var_k_post + mu_k * mu_k);  /* E[X²] for total variance */
        }
        
        /* Normalize and store */
        rbpf_real_t lik_total = rbpf_exp(log_lik_total);
        if (lik_total < RBPF_EPS) lik_total = RBPF_EPS;
        
        rbpf_real_t lik_inv = RBPF_REAL(1.0) / lik_total;
        rbpf_real_t mu_new = mu_accum * lik_inv;
        rbpf_real_t var_new = var_accum * lik_inv - mu_new * mu_new;  /* Var = E[X²] - E[X]² */
        if (var_new < RBPF_REAL(1e-10)) var_new = RBPF_REAL(1e-10);
        
        mu[i] = mu_new;
        var[i] = var_new;
        
        /* Update log-weight */
        log_weight[i] += log_lik_total;
        
        if (log_weight[i] > max_log_weight) {
            max_log_weight = log_weight[i];
        }
    }
    
    /*═══════════════════════════════════════════════════════════════════════
     * PHASE 3: Normalize weights and compute marginal likelihood
     *═══════════════════════════════════════════════════════════════════════*/
    
    /* Subtract max for numerical stability */
    rbpf_real_t sum_weight = RBPF_REAL(0.0);
    for (int i = 0; i < n; i++) {
        log_weight[i] -= max_log_weight;
        rbpf_real_t w = rbpf_exp(log_weight[i]);
        rbpf->w_norm[i] = w;
        sum_weight += w;
    }
    
    /* Normalize */
    rbpf_real_t inv_sum = RBPF_REAL(1.0) / sum_weight;
    for (int i = 0; i < n; i++) {
        rbpf->w_norm[i] *= inv_sum;
    }
    
    /* Marginal likelihood: p(y_t | y_{1:t-1}) */
    rbpf_real_t marginal_lik = max_log_weight + rbpf_log(sum_weight) - rbpf_log((rbpf_real_t)n);
    
    /*═══════════════════════════════════════════════════════════════════════
     * PHASE 4: Update ν learning statistics
     *═══════════════════════════════════════════════════════════════════════*/
    
    for (int r = 0; r < rbpf->n_regimes; r++) {
        if (!rbpf->student_t[r].learn_nu) continue;
        if (regime_sum_weight[r] < RBPF_REAL(1.0)) continue;
        
        RBPF_StudentT_Stats *stats = &rbpf->student_t_stats[r];
        RBPF_StudentT_Config *cfg = &rbpf->student_t[r];
        rbpf_real_t lr = cfg->nu_learning_rate;
        
        /* Exponentially weighted update */
        rbpf_real_t new_sum_lambda = regime_sum_lambda[r] / regime_sum_weight[r];
        rbpf_real_t new_sum_lambda_sq = regime_sum_lambda_sq[r] / regime_sum_weight[r];
        rbpf_real_t new_sum_log_lambda = regime_sum_log_lambda[r] / regime_sum_weight[r];
        
        stats->sum_lambda = lr * stats->sum_lambda + (RBPF_REAL(1.0) - lr) * new_sum_lambda * stats->n_eff
                          + new_sum_lambda;
        stats->sum_lambda_sq = lr * stats->sum_lambda_sq + (RBPF_REAL(1.0) - lr) * new_sum_lambda_sq * stats->n_eff
                             + new_sum_lambda_sq;
        stats->sum_log_lambda = lr * stats->sum_log_lambda + (RBPF_REAL(1.0) - lr) * new_sum_log_lambda * stats->n_eff
                              + new_sum_log_lambda;
        stats->n_eff = lr * stats->n_eff + regime_sum_weight[r];
        
        /* Update ν estimate */
        stats->nu_estimate = estimate_nu_digamma(stats, cfg->nu_floor, cfg->nu_ceil);
    }
    
    return marginal_lik;
}

/**
 * Student-t update with explicit ν (ignores per-regime settings)
 */
rbpf_real_t rbpf_ksc_update_student_t_nu(RBPF_KSC *rbpf, rbpf_real_t y, rbpf_real_t nu)
{
    if (!rbpf) return RBPF_REAL(0.0);
    
    /* Temporarily override all regimes to use explicit ν */
    rbpf_real_t saved_nu[RBPF_MAX_REGIMES];
    for (int r = 0; r < rbpf->n_regimes; r++) {
        saved_nu[r] = rbpf->student_t[r].nu;
        rbpf->student_t[r].nu = nu;
    }
    
    /* Ensure Student-t is enabled */
    int was_enabled = rbpf->student_t_enabled;
    rbpf->student_t_enabled = 1;
    
    /* Run update */
    rbpf_real_t marginal = rbpf_ksc_update_student_t(rbpf, y);
    
    /* Restore */
    rbpf->student_t_enabled = was_enabled;
    for (int r = 0; r < rbpf->n_regimes; r++) {
        rbpf->student_t[r].nu = saved_nu[r];
    }
    
    return marginal;
}

/*═══════════════════════════════════════════════════════════════════════════
 * COMBINED STEP FUNCTIONS
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Combined step with Student-t observations
 * 
 * Sequence: transition → predict → Student-t update → resample → output
 */
void rbpf_ksc_step_student_t(RBPF_KSC *rbpf, rbpf_real_t obs, RBPF_KSC_Output *output)
{
    if (!rbpf) return;
    
    /* Transform observation: r → y = log(r²) */
    rbpf_real_t y;
    if (obs == RBPF_REAL(0.0)) {
        /* Zero return: use floor */
        y = RBPF_REAL(-18.0);  /* ≈ log((1bp)²) */
    } else {
        y = rbpf_log(obs * obs);
    }
    
    /* Transition (sample new regimes) */
    rbpf_ksc_transition(rbpf);
    
    /* Predict */
    rbpf_ksc_predict(rbpf);
    
    /* Student-t update */
    rbpf_real_t marginal = rbpf_ksc_update_student_t(rbpf, y);
    
    /* Resample if needed */
    int resampled = rbpf_ksc_resample(rbpf);
    
    /* Compute outputs */
    rbpf_ksc_compute_outputs(rbpf, marginal, output);
    output->resampled = resampled;
    output->student_t_active = 1;
    
    /* Student-t specific diagnostics */
    if (rbpf->lambda != NULL) {
        rbpf_real_t sum_lam = RBPF_REAL(0.0);
        rbpf_real_t sum_lam_sq = RBPF_REAL(0.0);
        const rbpf_real_t *w = rbpf->w_norm;
        const rbpf_real_t *lam = rbpf->lambda;
        
        for (int i = 0; i < rbpf->n_particles; i++) {
            sum_lam += w[i] * lam[i];
            sum_lam_sq += w[i] * lam[i] * lam[i];
        }
        
        output->lambda_mean = sum_lam;
        output->lambda_var = sum_lam_sq - sum_lam * sum_lam;
        if (output->lambda_var < RBPF_REAL(0.0)) output->lambda_var = RBPF_REAL(0.0);
        
        /* Implied ν from observed λ variance */
        if (output->lambda_var > RBPF_REAL(0.01)) {
            output->nu_effective = RBPF_REAL(2.0) / output->lambda_var;
        } else {
            output->nu_effective = RBPF_NU_CEIL;  /* Near-Gaussian */
        }
        
        /* Copy learned ν estimates */
        for (int r = 0; r < rbpf->n_regimes; r++) {
            output->learned_nu[r] = rbpf_ksc_get_nu(rbpf, r);
        }
    }
}

/**
 * Combined step with explicit ν
 */
void rbpf_ksc_step_student_t_nu(RBPF_KSC *rbpf, rbpf_real_t obs, rbpf_real_t nu,
                                 RBPF_KSC_Output *output)
{
    if (!rbpf) return;
    
    /* Transform observation */
    rbpf_real_t y;
    if (obs == RBPF_REAL(0.0)) {
        y = RBPF_REAL(-18.0);
    } else {
        y = rbpf_log(obs * obs);
    }
    
    /* Transition */
    rbpf_ksc_transition(rbpf);
    
    /* Predict */
    rbpf_ksc_predict(rbpf);
    
    /* Student-t update with explicit ν */
    rbpf_real_t marginal = rbpf_ksc_update_student_t_nu(rbpf, y, nu);
    
    /* Resample if needed */
    int resampled = rbpf_ksc_resample(rbpf);
    
    /* Compute outputs */
    rbpf_ksc_compute_outputs(rbpf, marginal, output);
    output->resampled = resampled;
    output->student_t_active = 1;
    
    /* Diagnostics */
    if (rbpf->lambda != NULL) {
        rbpf_real_t sum_lam = RBPF_REAL(0.0);
        rbpf_real_t sum_lam_sq = RBPF_REAL(0.0);
        const rbpf_real_t *w = rbpf->w_norm;
        const rbpf_real_t *lam = rbpf->lambda;
        
        for (int i = 0; i < rbpf->n_particles; i++) {
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
 * MEMORY ALLOCATION HELPERS
 * 
 * These should be called from rbpf_ksc_create() to allocate Student-t arrays.
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Allocate Student-t auxiliary arrays
 * 
 * Call this from rbpf_ksc_create() after allocating main particle arrays.
 */
int rbpf_ksc_alloc_student_t(RBPF_KSC *rbpf)
{
    if (!rbpf) return -1;
    
    int n = rbpf->n_particles;
    size_t size = n * sizeof(rbpf_real_t);
    
    /* Allocate λ arrays */
#if defined(_MSC_VER)
    rbpf->lambda = (rbpf_real_t *)_aligned_malloc(size, RBPF_ALIGN);
    rbpf->lambda_tmp = (rbpf_real_t *)_aligned_malloc(size, RBPF_ALIGN);
    rbpf->log_lambda = (rbpf_real_t *)_aligned_malloc(size, RBPF_ALIGN);
#else
    rbpf->lambda = (rbpf_real_t *)aligned_alloc(RBPF_ALIGN, size);
    rbpf->lambda_tmp = (rbpf_real_t *)aligned_alloc(RBPF_ALIGN, size);
    rbpf->log_lambda = (rbpf_real_t *)aligned_alloc(RBPF_ALIGN, size);
#endif
    
    if (!rbpf->lambda || !rbpf->lambda_tmp || !rbpf->log_lambda) {
        return -1;
    }
    
    /* Initialize to λ=1 (Gaussian equivalent) */
    for (int i = 0; i < n; i++) {
        rbpf->lambda[i] = RBPF_REAL(1.0);
        rbpf->lambda_tmp[i] = RBPF_REAL(1.0);
        rbpf->log_lambda[i] = RBPF_REAL(0.0);
    }
    
    /* Initialize Student-t config to disabled */
    rbpf->student_t_enabled = 0;
    for (int r = 0; r < RBPF_MAX_REGIMES; r++) {
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

/**
 * Free Student-t auxiliary arrays
 * 
 * Call this from rbpf_ksc_destroy().
 */
void rbpf_ksc_free_student_t(RBPF_KSC *rbpf)
{
    if (!rbpf) return;
    
#if defined(_MSC_VER)
    if (rbpf->lambda) _aligned_free(rbpf->lambda);
    if (rbpf->lambda_tmp) _aligned_free(rbpf->lambda_tmp);
    if (rbpf->log_lambda) _aligned_free(rbpf->log_lambda);
#else
    if (rbpf->lambda) free(rbpf->lambda);
    if (rbpf->lambda_tmp) free(rbpf->lambda_tmp);
    if (rbpf->log_lambda) free(rbpf->log_lambda);
#endif
    
    rbpf->lambda = NULL;
    rbpf->lambda_tmp = NULL;
    rbpf->log_lambda = NULL;
}

/**
 * Apply resampling to λ arrays (call after main particle resampling)
 */
void rbpf_ksc_resample_student_t(RBPF_KSC *rbpf, const int *indices)
{
    if (!rbpf || !indices || !rbpf->lambda) return;
    
    int n = rbpf->n_particles;
    
    /* Gather λ values according to resample indices */
    for (int i = 0; i < n; i++) {
        rbpf->lambda_tmp[i] = rbpf->lambda[indices[i]];
    }
    
    /* Swap buffers */
    rbpf_real_t *tmp = rbpf->lambda;
    rbpf->lambda = rbpf->lambda_tmp;
    rbpf->lambda_tmp = tmp;
}

/*═══════════════════════════════════════════════════════════════════════════
 * PRINT DIAGNOSTICS
 *═══════════════════════════════════════════════════════════════════════════*/

void rbpf_ksc_print_student_t_config(const RBPF_KSC *rbpf)
{
    if (!rbpf) return;
    
    printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    printf("║           Student-t Observation Model Configuration          ║\n");
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    printf("║ Enabled: %s                                                  ║\n",
           rbpf->student_t_enabled ? "YES" : "NO ");
    
    if (rbpf->student_t_enabled) {
        printf("╠══════════════════════════════════════════════════════════════╣\n");
        printf("║ Per-regime configuration:                                    ║\n");
        
        for (int r = 0; r < rbpf->n_regimes; r++) {
            const RBPF_StudentT_Config *cfg = &rbpf->student_t[r];
            const RBPF_StudentT_Stats *stats = &rbpf->student_t_stats[r];
            
            printf("║   Regime %d: ν=%.2f", r, cfg->nu);
            if (cfg->learn_nu) {
                printf(" (learning: ν_est=%.2f, n_eff=%.0f)", 
                       stats->nu_estimate, stats->n_eff);
            }
            printf("\n");
        }
        
        printf("╠══════════════════════════════════════════════════════════════╣\n");
        printf("║ Bounds: ν ∈ [%.1f, %.1f]                                      ║\n",
               RBPF_NU_FLOOR, RBPF_NU_CEIL);
        printf("║ λ bounds: [%.2f, %.1f]                                        ║\n",
               RBPF_LAMBDA_FLOOR, RBPF_LAMBDA_CEIL);
    }
    
    printf("╚══════════════════════════════════════════════════════════════╝\n");
}
