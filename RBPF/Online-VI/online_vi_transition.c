/**
 * @file online_vi_transition.c
 * @brief Online Variational Inference for Transition Matrix Learning
 *
 * Implementation of natural gradient VI for Dirichlet-Multinomial transition model.
 */

#include "online_vi_transition.h"
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <float.h>

/*═══════════════════════════════════════════════════════════════════════════════
 * CONSTANTS AND HELPERS
 *═══════════════════════════════════════════════════════════════════════════════*/

#define VI_EPS 1e-10
#define VI_LOG_EPS -23.0259 /* log(1e-10) */

static inline double maxd(double a, double b) { return a > b ? a : b; }
static inline double mind(double a, double b) { return a < b ? a : b; }
static inline double clampd(double x, double lo, double hi)
{
    return x < lo ? lo : (x > hi ? hi : x);
}

/**
 * Digamma function approximation (Bernardo 1976)
 * ψ(x) = d/dx ln Γ(x)
 */
static double digamma(double x)
{
    if (x < VI_EPS)
        return VI_LOG_EPS;

    double result = 0.0;

    /* Use recurrence ψ(x+1) = ψ(x) + 1/x to shift to x > 6 */
    while (x < 6.0)
    {
        result -= 1.0 / x;
        x += 1.0;
    }

    /* Asymptotic expansion for large x */
    double x2 = 1.0 / (x * x);
    result += log(x) - 0.5 / x - x2 * (1.0 / 12.0 - x2 * (1.0 / 120.0 - x2 / 252.0));

    return result;
}

/**
 * Log of multivariate beta function
 * ln B(α) = Σ ln Γ(α_i) - ln Γ(Σ α_i)
 */
static double log_multivariate_beta(const double *alpha, int K)
{
    double sum_alpha = 0.0;
    double sum_log_gamma = 0.0;

    for (int i = 0; i < K; i++)
    {
        sum_alpha += alpha[i];
        sum_log_gamma += lgamma(alpha[i]);
    }

    return sum_log_gamma - lgamma(sum_alpha);
}

/**
 * Entropy of Dirichlet distribution
 * H[Dir(α)] = ln B(α) + (α_0 - K)ψ(α_0) - Σ(α_i - 1)ψ(α_i)
 */
static double dirichlet_entropy(const double *alpha, int K)
{
    double alpha_0 = 0.0;
    for (int i = 0; i < K; i++)
    {
        alpha_0 += alpha[i];
    }

    double log_B = log_multivariate_beta(alpha, K);
    double psi_alpha_0 = digamma(alpha_0);

    double sum_term = 0.0;
    for (int i = 0; i < K; i++)
    {
        sum_term += (alpha[i] - 1.0) * digamma(alpha[i]);
    }

    return log_B + (alpha_0 - K) * psi_alpha_0 - sum_term;
}

/**
 * KL divergence between two Dirichlet distributions
 * KL(Dir(α) || Dir(β))
 */
static double dirichlet_kl(const double *alpha, const double *beta, int K)
{
    double alpha_0 = 0.0, beta_0 = 0.0;
    for (int i = 0; i < K; i++)
    {
        alpha_0 += alpha[i];
        beta_0 += beta[i];
    }

    double kl = lgamma(alpha_0) - lgamma(beta_0);

    for (int i = 0; i < K; i++)
    {
        kl -= lgamma(alpha[i]) - lgamma(beta[i]);
        kl += (alpha[i] - beta[i]) * (digamma(alpha[i]) - digamma(alpha_0));
    }

    return maxd(0.0, kl); /* Ensure non-negative (numerical) */
}

/**
 * Recompute cached statistics (mean, variance, entropy)
 */
static void recompute_stats(OnlineVI *vi)
{
    if (!vi->stats_dirty)
        return;

    const int K = vi->K;

    vi->total_entropy = 0.0;
    vi->kl_from_prior = 0.0;

    for (int i = 0; i < K; i++)
    {
        /* Row sum */
        double alpha_0 = 0.0;
        for (int j = 0; j < K; j++)
        {
            alpha_0 += vi->alpha[i][j];
        }
        vi->alpha_sum[i] = alpha_0;

        /* Mean and variance */
        double alpha_0_sq = alpha_0 * alpha_0;
        double denom = alpha_0_sq * (alpha_0 + 1.0);

        for (int j = 0; j < K; j++)
        {
            double a_ij = vi->alpha[i][j];
            vi->mean[i][j] = a_ij / alpha_0;
            vi->var[i][j] = a_ij * (alpha_0 - a_ij) / denom;
        }

        /* Per-row entropy */
        vi->row_entropy[i] = dirichlet_entropy(vi->alpha[i], K);
        vi->total_entropy += vi->row_entropy[i];

        /* KL from prior for this row */
        vi->kl_from_prior += dirichlet_kl(vi->alpha[i], vi->alpha_prior[i], K);
    }

    vi->stats_dirty = false;
}

/**
 * Update learning rate based on schedule
 */
static void update_learning_rate(OnlineVI *vi)
{
    switch (vi->lr_schedule)
    {
    case VI_LR_FIXED:
        /* No change */
        break;

    case VI_LR_ROBBINS_MONRO:
        /* ρ_t = ρ_0 × (τ + t)^{-κ} */
        vi->rho = vi->rho_base * pow(vi->rho_tau + (double)vi->t, -vi->rho_kappa);
        vi->rho = clampd(vi->rho, vi->rho_min, vi->rho_max);
        break;

    case VI_LR_ADAPTIVE:
        /* External control - just clamp */
        vi->rho = clampd(vi->rho, vi->rho_min, vi->rho_max);
        break;
    }
}

/*═══════════════════════════════════════════════════════════════════════════════
 * LIFECYCLE
 *═══════════════════════════════════════════════════════════════════════════════*/

void online_vi_init(OnlineVI *vi, int K)
{
    if (!vi)
        return;

    K = (K < 1) ? 1 : (K > ONLINE_VI_MAX_REGIMES ? ONLINE_VI_MAX_REGIMES : K);

    memset(vi, 0, sizeof(*vi));
    vi->K = K;

    /* Uniform prior: Dir(1, 1, ..., 1) */
    for (int i = 0; i < K; i++)
    {
        for (int j = 0; j < K; j++)
        {
            vi->alpha[i][j] = 1.0;
            vi->alpha_prior[i][j] = 1.0;
        }
        vi->alpha_sum[i] = (double)K;
    }

    /* Default learning rate: Robbins-Monro */
    vi->lr_schedule = VI_LR_ROBBINS_MONRO;
    vi->rho_base = 1.0;
    vi->rho_tau = 64.0;
    vi->rho_kappa = 0.7;
    vi->rho_min = 0.001;
    vi->rho_max = 1.0;
    vi->rho = vi->rho_base; /* Initial rate */

    vi->initialized = false;
    vi->t = 0;
    vi->stats_dirty = true;
    vi->total_updates = 0;

    recompute_stats(vi);
}

void online_vi_init_sticky(OnlineVI *vi, int K, double alpha_base, double stickiness)
{
    if (!vi)
        return;

    /* First init with uniform */
    online_vi_init(vi, K);

    /* Then add stickiness to diagonal */
    alpha_base = maxd(0.1, alpha_base);
    stickiness = maxd(0.0, stickiness);

    for (int i = 0; i < K; i++)
    {
        for (int j = 0; j < K; j++)
        {
            double val = (i == j) ? (alpha_base + stickiness) : alpha_base;
            vi->alpha[i][j] = val;
            vi->alpha_prior[i][j] = val;
        }
        vi->alpha_sum[i] = alpha_base * K + stickiness;
    }

    vi->stats_dirty = true;
    recompute_stats(vi);
}

void online_vi_init_from_matrix(OnlineVI *vi, int K,
                                const double *trans, double confidence)
{
    if (!vi || !trans)
        return;

    /* First init with uniform */
    online_vi_init(vi, K);

    confidence = maxd(1.0, confidence);

    /* Set alpha = confidence × trans + 1 (ensures α > 0) */
    for (int i = 0; i < K; i++)
    {
        double row_sum = 0.0;

        for (int j = 0; j < K; j++)
        {
            double p_ij = trans[i * K + j];
            p_ij = clampd(p_ij, VI_EPS, 1.0 - VI_EPS);

            double alpha_ij = 1.0 + confidence * p_ij;
            vi->alpha[i][j] = alpha_ij;
            vi->alpha_prior[i][j] = alpha_ij;
            row_sum += alpha_ij;
        }
        vi->alpha_sum[i] = row_sum;
    }

    vi->stats_dirty = true;
    recompute_stats(vi);
}

void online_vi_reset(OnlineVI *vi)
{
    if (!vi)
        return;

    const int K = vi->K;

    /* Reset alpha to prior */
    for (int i = 0; i < K; i++)
    {
        for (int j = 0; j < K; j++)
        {
            vi->alpha[i][j] = vi->alpha_prior[i][j];
        }
    }

    /* Reset state */
    vi->initialized = false;
    vi->t = 0;
    vi->rho = vi->rho_base;
    vi->stats_dirty = true;
    vi->total_updates = 0;

    memset(vi->prev_probs, 0, sizeof(vi->prev_probs));
    memset(vi->last_xi, 0, sizeof(vi->last_xi));

    recompute_stats(vi);
}

/*═══════════════════════════════════════════════════════════════════════════════
 * LEARNING RATE CONFIGURATION
 *═══════════════════════════════════════════════════════════════════════════════*/

void online_vi_set_lr_fixed(OnlineVI *vi, double rho)
{
    if (!vi)
        return;
    vi->lr_schedule = VI_LR_FIXED;
    vi->rho = clampd(rho, vi->rho_min, vi->rho_max);
    vi->rho_base = vi->rho;
}

void online_vi_set_lr_robbins_monro(OnlineVI *vi, double rho_0,
                                    double tau, double kappa)
{
    if (!vi)
        return;
    vi->lr_schedule = VI_LR_ROBBINS_MONRO;
    vi->rho_base = maxd(0.01, rho_0);
    vi->rho_tau = maxd(1.0, tau);
    vi->rho_kappa = clampd(kappa, 0.5, 1.0);
    vi->rho = vi->rho_base;
}

void online_vi_set_lr_adaptive(OnlineVI *vi, double rho_init)
{
    if (!vi)
        return;
    vi->lr_schedule = VI_LR_ADAPTIVE;
    vi->rho = clampd(rho_init, vi->rho_min, vi->rho_max);
    vi->rho_base = vi->rho;
}

void online_vi_set_rho(OnlineVI *vi, double rho)
{
    if (!vi)
        return;
    vi->rho = clampd(rho, vi->rho_min, vi->rho_max);
}

void online_vi_set_rho_bounds(OnlineVI *vi, double rho_min, double rho_max)
{
    if (!vi)
        return;
    vi->rho_min = maxd(1e-6, rho_min);
    vi->rho_max = mind(1.0, rho_max);
    if (vi->rho_min > vi->rho_max)
        vi->rho_min = vi->rho_max;
}

/*═══════════════════════════════════════════════════════════════════════════════
 * CORE UPDATE
 *═══════════════════════════════════════════════════════════════════════════════*/

void online_vi_update(OnlineVI *vi,
                      const double *regime_probs,
                      const double *regime_liks)
{
    if (!vi || !regime_probs || !regime_liks)
        return;

    const int K = vi->K;

    /*───────────────────────────────────────────────────────────────────────────
     * First tick: just store regime probs
     *───────────────────────────────────────────────────────────────────────────*/
    if (!vi->initialized)
    {
        for (int i = 0; i < K; i++)
        {
            vi->prev_probs[i] = regime_probs[i];
        }
        vi->initialized = true;
        return;
    }

    /*───────────────────────────────────────────────────────────────────────────
     * Compute ξ_ij = P(s_{t-1}=i, s_t=j | y_{1:t})
     *
     * ξ̃_ij = p_{t-1}(i) × π_ij × ℓ_t(j)
     * ξ_ij = ξ̃_ij / Z
     *───────────────────────────────────────────────────────────────────────────*/
    double xi[ONLINE_VI_MAX_REGIMES][ONLINE_VI_MAX_REGIMES];
    double Z = 0.0;

    /* Ensure we have current mean transition matrix */
    recompute_stats(vi);

    for (int i = 0; i < K; i++)
    {
        for (int j = 0; j < K; j++)
        {
            double xi_ij = vi->prev_probs[i] * vi->mean[i][j] * regime_liks[j];
            xi[i][j] = xi_ij;
            Z += xi_ij;
        }
    }

    /* Normalize */
    if (Z > VI_EPS)
    {
        double inv_Z = 1.0 / Z;
        for (int i = 0; i < K; i++)
        {
            for (int j = 0; j < K; j++)
            {
                xi[i][j] *= inv_Z;
                vi->last_xi[i][j] = xi[i][j];
            }
        }
    }
    else
    {
        /* Fallback: uniform */
        double uniform = 1.0 / (K * K);
        for (int i = 0; i < K; i++)
        {
            for (int j = 0; j < K; j++)
            {
                xi[i][j] = uniform;
                vi->last_xi[i][j] = uniform;
            }
        }
    }

    /*───────────────────────────────────────────────────────────────────────────
     * Natural gradient update
     *
     * α̃_ij ← (1 - ρ) × α̃_ij + ρ × (α_prior_ij + N × ξ_ij)
     *
     * Where N = 1 for single observation (can scale for batches)
     *───────────────────────────────────────────────────────────────────────────*/

    vi->t++;
    update_learning_rate(vi);

    double rho = vi->rho;
    double one_minus_rho = 1.0 - rho;
    double N = 1.0; /* Effective sample size for this update */

    for (int i = 0; i < K; i++)
    {
        double row_sum = 0.0;

        for (int j = 0; j < K; j++)
        {
            /* Natural gradient step */
            double target = vi->alpha_prior[i][j] + N * xi[i][j];
            vi->alpha[i][j] = one_minus_rho * vi->alpha[i][j] + rho * target;

            /* Ensure positivity */
            vi->alpha[i][j] = maxd(VI_EPS, vi->alpha[i][j]);
            row_sum += vi->alpha[i][j];
        }

        vi->alpha_sum[i] = row_sum;
    }

    /*───────────────────────────────────────────────────────────────────────────
     * Update state for next iteration
     *───────────────────────────────────────────────────────────────────────────*/
    for (int i = 0; i < K; i++)
    {
        vi->prev_probs[i] = regime_probs[i];
    }

    vi->stats_dirty = true;
    vi->total_updates++;
}

void online_vi_update_with_xi(OnlineVI *vi, const double *xi)
{
    if (!vi || !xi)
        return;

    const int K = vi->K;

    /* Store xi for diagnostics */
    for (int i = 0; i < K; i++)
    {
        for (int j = 0; j < K; j++)
        {
            vi->last_xi[i][j] = xi[i * K + j];
        }
    }

    /* Update learning rate */
    vi->t++;
    update_learning_rate(vi);

    double rho = vi->rho;
    double one_minus_rho = 1.0 - rho;
    double N = 1.0;

    /* Natural gradient update */
    for (int i = 0; i < K; i++)
    {
        double row_sum = 0.0;

        for (int j = 0; j < K; j++)
        {
            double target = vi->alpha_prior[i][j] + N * xi[i * K + j];
            vi->alpha[i][j] = one_minus_rho * vi->alpha[i][j] + rho * target;
            vi->alpha[i][j] = maxd(VI_EPS, vi->alpha[i][j]);
            row_sum += vi->alpha[i][j];
        }

        vi->alpha_sum[i] = row_sum;
    }

    vi->stats_dirty = true;
    vi->total_updates++;
    vi->initialized = true;
}

void online_vi_update_log(OnlineVI *vi,
                          const double *regime_probs,
                          const double *log_regime_liks)
{
    if (!vi || !regime_probs || !log_regime_liks)
        return;

    const int K = vi->K;

    /* Convert log-likelihoods to likelihoods (with numerical care) */
    double liks[ONLINE_VI_MAX_REGIMES];
    double max_log = log_regime_liks[0];

    for (int j = 1; j < K; j++)
    {
        if (log_regime_liks[j] > max_log)
            max_log = log_regime_liks[j];
    }

    for (int j = 0; j < K; j++)
    {
        liks[j] = exp(log_regime_liks[j] - max_log);
    }

    online_vi_update(vi, regime_probs, liks);
}

/*═══════════════════════════════════════════════════════════════════════════════
 * POSTERIOR QUERIES
 *═══════════════════════════════════════════════════════════════════════════════*/

void online_vi_get_mean(OnlineVI *vi, double *trans)
{
    if (!vi || !trans)
        return;

    recompute_stats(vi);

    const int K = vi->K;
    for (int i = 0; i < K; i++)
    {
        for (int j = 0; j < K; j++)
        {
            trans[i * K + j] = vi->mean[i][j];
        }
    }
}

void online_vi_get_variance(OnlineVI *vi, double *var)
{
    if (!vi || !var)
        return;

    recompute_stats(vi);

    const int K = vi->K;
    for (int i = 0; i < K; i++)
    {
        for (int j = 0; j < K; j++)
        {
            var[i * K + j] = vi->var[i][j];
        }
    }
}

void online_vi_get_std(OnlineVI *vi, double *std)
{
    if (!vi || !std)
        return;

    recompute_stats(vi);

    const int K = vi->K;
    for (int i = 0; i < K; i++)
    {
        for (int j = 0; j < K; j++)
        {
            std[i * K + j] = sqrt(vi->var[i][j]);
        }
    }
}

double online_vi_get_prob(OnlineVI *vi, int from, int to)
{
    if (!vi || from < 0 || from >= vi->K || to < 0 || to >= vi->K)
        return 0.0;

    recompute_stats(vi);
    return vi->mean[from][to];
}

double online_vi_get_prob_var(OnlineVI *vi, int from, int to)
{
    if (!vi || from < 0 || from >= vi->K || to < 0 || to >= vi->K)
        return 0.0;

    recompute_stats(vi);
    return vi->var[from][to];
}

void online_vi_get_row(OnlineVI *vi, int from, double *row)
{
    if (!vi || !row || from < 0 || from >= vi->K)
        return;

    recompute_stats(vi);

    for (int j = 0; j < vi->K; j++)
    {
        row[j] = vi->mean[from][j];
    }
}

double online_vi_get_stickiness(OnlineVI *vi, int regime)
{
    return online_vi_get_prob(vi, regime, regime);
}

/*═══════════════════════════════════════════════════════════════════════════════
 * ENTROPY & UNCERTAINTY
 *═══════════════════════════════════════════════════════════════════════════════*/

void online_vi_get_row_entropy(OnlineVI *vi, double *H)
{
    if (!vi || !H)
        return;

    recompute_stats(vi);

    for (int i = 0; i < vi->K; i++)
    {
        H[i] = vi->row_entropy[i];
    }
}

double online_vi_get_total_entropy(OnlineVI *vi)
{
    if (!vi)
        return 0.0;

    recompute_stats(vi);
    return vi->total_entropy;
}

double online_vi_get_max_row_entropy(OnlineVI *vi, int *max_row)
{
    if (!vi)
        return 0.0;

    recompute_stats(vi);

    double max_H = vi->row_entropy[0];
    int max_i = 0;

    for (int i = 1; i < vi->K; i++)
    {
        if (vi->row_entropy[i] > max_H)
        {
            max_H = vi->row_entropy[i];
            max_i = i;
        }
    }

    if (max_row)
        *max_row = max_i;
    return max_H;
}

double online_vi_get_confidence(OnlineVI *vi)
{
    if (!vi)
        return 0.0;

    recompute_stats(vi);

    /* Maximum entropy for K-dimensional Dirichlet with α = 1 (uniform)
     * For a single row, H_max = log(K) when using categorical entropy
     * But Dirichlet entropy can be negative, so we use a different approach:
     *
     * Confidence = 1 - (variance_sum / max_variance_sum)
     * where max variance is at uniform Dirichlet(1,1,...,1)
     */
    const int K = vi->K;

    /* Sum of variances (uncertainty measure) */
    double var_sum = 0.0;
    for (int i = 0; i < K; i++)
    {
        for (int j = 0; j < K; j++)
        {
            var_sum += vi->var[i][j];
        }
    }

    /* Max variance at uniform: Var = (K-1)/(K^2 * (K+1)) per entry, K^2 entries */
    double max_var_per_entry = (double)(K - 1) / (double)(K * K * (K + 1));
    double max_var_sum = max_var_per_entry * K * K * K; /* K rows */

    if (max_var_sum < VI_EPS)
        return 1.0;

    double confidence = 1.0 - var_sum / max_var_sum;
    return clampd(confidence, 0.0, 1.0);
}

/**
 * Get normalized entropy (always positive, 0 = certain, 1 = maximum uncertainty)
 */
double online_vi_get_normalized_entropy(OnlineVI *vi)
{
    if (!vi)
        return 0.0;

    recompute_stats(vi);

    const int K = vi->K;

    /* Use mean posterior variance as uncertainty proxy (always positive) */
    double var_sum = 0.0;
    for (int i = 0; i < K; i++)
    {
        for (int j = 0; j < K; j++)
        {
            var_sum += vi->var[i][j];
        }
    }

    /* Normalize by max possible variance */
    double max_var_per_entry = (double)(K - 1) / (double)(K * K * (K + 1));
    double max_var_sum = max_var_per_entry * K * K * K;

    if (max_var_sum < VI_EPS)
        return 0.0;

    return clampd(var_sum / max_var_sum, 0.0, 1.0);
}

/*═══════════════════════════════════════════════════════════════════════════════
 * DIAGNOSTICS
 *═══════════════════════════════════════════════════════════════════════════════*/

double online_vi_get_kl_from_prior(OnlineVI *vi)
{
    if (!vi)
        return 0.0;

    recompute_stats(vi);
    return vi->kl_from_prior;
}

void online_vi_get_stats(OnlineVI *vi, OnlineVI_Stats *stats)
{
    if (!vi || !stats)
        return;

    recompute_stats(vi);

    const int K = vi->K;

    /* Entropy stats */
    stats->total_entropy = vi->total_entropy;
    stats->mean_entropy = vi->total_entropy / K;
    stats->max_entropy = vi->row_entropy[0];
    stats->min_entropy = vi->row_entropy[0];

    for (int i = 1; i < K; i++)
    {
        if (vi->row_entropy[i] > stats->max_entropy)
            stats->max_entropy = vi->row_entropy[i];
        if (vi->row_entropy[i] < stats->min_entropy)
            stats->min_entropy = vi->row_entropy[i];
    }

    /* Stickiness */
    stats->mean_stickiness = 0.0;
    for (int i = 0; i < K; i++)
    {
        stats->mean_stickiness += vi->mean[i][i];
    }
    stats->mean_stickiness /= K;

    /* Other */
    stats->kl_from_prior = vi->kl_from_prior;
    stats->current_rho = vi->rho;
    stats->total_updates = vi->total_updates;
}

double online_vi_get_rho(const OnlineVI *vi)
{
    return vi ? vi->rho : 0.0;
}

void online_vi_get_row_ess(const OnlineVI *vi, double *ess)
{
    if (!vi || !ess)
        return;

    const int K = vi->K;
    for (int i = 0; i < K; i++)
    {
        /* ESS = α_0 - K (effective counts beyond uniform prior) */
        double alpha_0 = 0.0;
        double prior_0 = 0.0;
        for (int j = 0; j < K; j++)
        {
            alpha_0 += vi->alpha[i][j];
            prior_0 += vi->alpha_prior[i][j];
        }
        ess[i] = maxd(0.0, alpha_0 - prior_0);
    }
}

/*═══════════════════════════════════════════════════════════════════════════════
 * INTEGRATION WITH HDP-BEAM
 *═══════════════════════════════════════════════════════════════════════════════*/

void online_vi_reset_from_hdp(OnlineVI *vi, const double *hdp_trans,
                              double confidence)
{
    if (!vi || !hdp_trans)
        return;

    const int K = vi->K;
    confidence = maxd(1.0, mind(100.0, confidence));

    /* Set alpha from HDP posterior */
    for (int i = 0; i < K; i++)
    {
        double row_sum = 0.0;

        for (int j = 0; j < K; j++)
        {
            double p_ij = hdp_trans[i * K + j];
            p_ij = clampd(p_ij, VI_EPS, 1.0 - VI_EPS);

            /* α = prior + confidence × HDP_prob */
            vi->alpha[i][j] = vi->alpha_prior[i][j] + confidence * p_ij;
            row_sum += vi->alpha[i][j];
        }
        vi->alpha_sum[i] = row_sum;
    }

    vi->stats_dirty = true;
    vi->initialized = true;
}

void online_vi_adapt_rho_from_hdp(OnlineVI *vi, double correction_kl,
                                  double sensitivity)
{
    if (!vi || vi->lr_schedule != VI_LR_ADAPTIVE)
        return;

    sensitivity = clampd(sensitivity, 0.01, 1.0);

    /* Large correction KL → increase ρ (need to adapt faster)
     * Small correction KL → decrease ρ (stable, can learn slowly) */

    double log_kl = log(maxd(VI_EPS, correction_kl));
    double adjustment = sensitivity * log_kl;

    /* Multiplicative adjustment */
    double new_rho = vi->rho * exp(adjustment);
    vi->rho = clampd(new_rho, vi->rho_min, vi->rho_max);
}

/*═══════════════════════════════════════════════════════════════════════════════
 * UTILITY
 *═══════════════════════════════════════════════════════════════════════════════*/

void online_vi_print(const OnlineVI *vi)
{
    if (!vi)
    {
        printf("OnlineVI: NULL\n");
        return;
    }

    printf("OnlineVI State:\n");
    printf("  K = %d\n", vi->K);
    printf("  t = %lu updates\n", (unsigned long)vi->t);
    printf("  ρ = %.4f (%s)\n", vi->rho,
           vi->lr_schedule == VI_LR_FIXED ? "fixed" : vi->lr_schedule == VI_LR_ROBBINS_MONRO ? "Robbins-Monro"
                                                                                             : "adaptive");
    printf("  Total entropy = %.4f\n", vi->total_entropy);
    printf("  KL from prior = %.4f\n", vi->kl_from_prior);
}

void online_vi_print_matrix(OnlineVI *vi)
{
    if (!vi)
        return;

    recompute_stats(vi);

    const int K = vi->K;

    printf("Transition Matrix E[π] ± std:\n");
    for (int i = 0; i < K; i++)
    {
        printf("  [");
        for (int j = 0; j < K; j++)
        {
            printf(" %.3f±%.3f", vi->mean[i][j], sqrt(vi->var[i][j]));
        }
        printf(" ]  H=%.3f\n", vi->row_entropy[i]);
    }
}