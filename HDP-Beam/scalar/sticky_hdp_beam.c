/**
 * @file sticky_hdp_beam.c
 * @brief Sticky HDP-HMM with Beam Sampling - MKL Accelerated Implementation
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * PERFORMANCE TARGETS
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * T=100, K_active≈6:  ~500μs per sweep
 * T=500, K_active≈8:  ~2ms per sweep  
 * T=1000, K_active≈8: ~4ms per sweep
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 */

#include "sticky_hdp_beam.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

/* MKL includes */
#include <mkl.h>
#include <mkl_vsl.h>
#include <mkl_vml.h>
#include <mkl_cblas.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


/*═══════════════════════════════════════════════════════════════════════════════
 * COMPILER HINTS
 *═══════════════════════════════════════════════════════════════════════════════*/

#if defined(__GNUC__) || defined(__clang__)
#define LIKELY(x)       __builtin_expect(!!(x), 1)
#define UNLIKELY(x)     __builtin_expect(!!(x), 0)
#define RESTRICT        __restrict__
#define FORCE_INLINE    __attribute__((always_inline)) inline
#else
#define LIKELY(x)       (x)
#define UNLIKELY(x)     (x)
#define RESTRICT
#define FORCE_INLINE    inline
#endif

/*═══════════════════════════════════════════════════════════════════════════════
 * MEMORY HELPERS
 *═══════════════════════════════════════════════════════════════════════════════*/

static FORCE_INLINE double *aligned_alloc_double(int n)
{
    return (double *)mkl_malloc(n * sizeof(double), HDP_CACHE_LINE);
}

static FORCE_INLINE int *aligned_alloc_int(int n)
{
    return (int *)mkl_malloc(n * sizeof(int), HDP_CACHE_LINE);
}

static FORCE_INLINE bool *aligned_alloc_bool(int n)
{
    return (bool *)mkl_malloc(n * sizeof(bool), HDP_CACHE_LINE);
}

/*═══════════════════════════════════════════════════════════════════════════════
 * LOG-SPACE UTILITIES (MKL Accelerated)
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * Log-sum-exp with numerical stability
 * 
 * log(Σ exp(x_i)) = max(x) + log(Σ exp(x_i - max(x)))
 */
static double log_sum_exp(const double *RESTRICT x, int n)
{
    if (n == 0) return HDP_LOG_ZERO;
    if (n == 1) return x[0];
    
    /* Find max using BLAS */
    int max_idx = cblas_idamax(n, x, 1);
    double max_val = x[max_idx];
    
    if (max_val <= HDP_LOG_ZERO) return HDP_LOG_ZERO;
    
    /* Compute sum of exp(x - max) */
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        if (x[i] > HDP_LOG_ZERO) {
            sum += exp(x[i] - max_val);
        }
    }
    
    return max_val + log(sum);
}

/**
 * Vectorized log-sum-exp for multiple rows
 * Uses MKL vdExp for batch exponentiation
 */
static void log_sum_exp_batch(const double *RESTRICT x, int n_rows, int n_cols,
                               double *RESTRICT result, double *RESTRICT scratch)
{
    for (int i = 0; i < n_rows; i++) {
        const double *row = x + i * n_cols;
        
        /* Find max */
        int max_idx = cblas_idamax(n_cols, row, 1);
        double max_val = row[max_idx];
        
        if (max_val <= HDP_LOG_ZERO) {
            result[i] = HDP_LOG_ZERO;
            continue;
        }
        
        /* Compute x - max */
        for (int j = 0; j < n_cols; j++) {
            scratch[j] = row[j] - max_val;
        }
        
        /* Batch exp */
        vdExp(n_cols, scratch, scratch);
        
        /* Sum */
        result[i] = max_val + log(cblas_dasum(n_cols, scratch, 1));
    }
}

/**
 * Log of matrix-vector product (for forward filtering)
 * 
 * result[j] = log(Σ_i exp(log_M[i,j] + log_v[i]))
 */
static void log_matvec(const double *RESTRICT log_M, int M_rows, int M_cols,
                       const double *RESTRICT log_v,
                       double *RESTRICT result, double *RESTRICT scratch)
{
    for (int j = 0; j < M_cols; j++) {
        /* Collect log_M[i,j] + log_v[i] for all i */
        for (int i = 0; i < M_rows; i++) {
            scratch[i] = log_M[i * M_cols + j] + log_v[i];
        }
        result[j] = log_sum_exp(scratch, M_rows);
    }
}

/*═══════════════════════════════════════════════════════════════════════════════
 * SAMPLING UTILITIES
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * Sample from categorical distribution given log-probabilities
 */
static int sample_categorical_log(VSLStreamStatePtr stream,
                                   const double *RESTRICT log_probs, int n,
                                   double *RESTRICT scratch)
{
    /* Convert to probabilities */
    double max_lp = log_probs[cblas_idamax(n, log_probs, 1)];
    for (int i = 0; i < n; i++) {
        scratch[i] = log_probs[i] - max_lp;
    }
    vdExp(n, scratch, scratch);
    
    /* Normalize */
    double sum = cblas_dasum(n, scratch, 1);
    cblas_dscal(n, 1.0 / sum, scratch, 1);
    
    /* Sample uniform */
    double u;
    vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, 1, &u, 0.0, 1.0);
    
    /* Inverse CDF */
    double cumsum = 0.0;
    for (int i = 0; i < n; i++) {
        cumsum += scratch[i];
        if (u <= cumsum) return i;
    }
    return n - 1;
}

/**
 * Sample from Beta(a, b)
 */
static double sample_beta(VSLStreamStatePtr stream, double a, double b)
{
    double g1, g2;
    vdRngGamma(VSL_RNG_METHOD_GAMMA_GNORM, stream, 1, &g1, a, 0.0, 1.0);
    vdRngGamma(VSL_RNG_METHOD_GAMMA_GNORM, stream, 1, &g2, b, 0.0, 1.0);
    return g1 / (g1 + g2 + HDP_EPS);
}

/**
 * Sample from Gamma(a, b) where b is rate
 */
static double sample_gamma(VSLStreamStatePtr stream, double shape, double rate)
{
    double x;
    vdRngGamma(VSL_RNG_METHOD_GAMMA_GNORM, stream, 1, &x, shape, 0.0, 1.0 / rate);
    return x;
}

/*═══════════════════════════════════════════════════════════════════════════════
 * STICK-BREAKING
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * Sample global measure β using stick-breaking construction
 * 
 * β_k = V_k × Π_{j<k} (1 - V_j)
 * V_k ~ Beta(1, γ)
 */
static void sample_beta_stick_breaking(StickyHDP *hdp)
{
    VSLStreamStatePtr stream = (VSLStreamStatePtr)hdp->mkl_stream;
    double remain = 1.0;
    
    for (int k = 0; k < hdp->K_max - 1; k++) {
        /* V_k ~ Beta(1 + n_k, γ + n_{>k}) 
         * where n_k = number in state k, n_{>k} = number in states > k */
        
        /* Count transitions TO state k (across all source states) */
        int n_to_k = 0;
        int n_after_k = 0;
        for (int j = 0; j < hdp->K_max; j++) {
            for (int l = 0; l < hdp->K_max; l++) {
                if (l == k) n_to_k += hdp->n_trans[j * hdp->K_max + l];
                if (l > k) n_after_k += hdp->n_trans[j * hdp->K_max + l];
            }
        }
        
        double a = 1.0 + n_to_k;
        double b = hdp->gamma + n_after_k;
        
        double V = sample_beta(stream, a, b);
        hdp->beta[k] = remain * V;
        hdp->stick_remain[k] = remain;
        remain *= (1.0 - V);
        
        if (remain < HDP_MIN_PROB) {
            /* Remaining stick is negligible */
            for (int l = k + 1; l < hdp->K_max; l++) {
                hdp->beta[l] = 0.0;
                hdp->stick_remain[l] = 0.0;
            }
            break;
        }
    }
    
    /* Last stick gets remainder */
    hdp->beta[hdp->K_max - 1] = remain;
    hdp->stick_remain[hdp->K_max - 1] = remain;
    
    /* Compute log-beta */
    for (int k = 0; k < hdp->K_max; k++) {
        hdp->log_beta[k] = (hdp->beta[k] > HDP_MIN_PROB) 
                          ? log(hdp->beta[k]) 
                          : HDP_LOG_ZERO;
    }
    
    /* Update K (number of active states) */
    hdp->K = 0;
    for (int k = 0; k < hdp->K_max; k++) {
        if (hdp->beta[k] > HDP_MIN_PROB) hdp->K = k + 1;
    }
}

/*═══════════════════════════════════════════════════════════════════════════════
 * TRANSITION SAMPLING
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * Sample transition distributions π_k given counts and β
 * 
 * π_k ~ Dir(α·β_1 + n_{k1}, ..., α·β_K + n_{kK} + κ·I(j=k))
 */
static void sample_transitions(StickyHDP *hdp)
{
    VSLStreamStatePtr stream = (VSLStreamStatePtr)hdp->mkl_stream;
    const int K = hdp->K_max;
    const double alpha = hdp->alpha;
    const double kappa = hdp->kappa;
    const double *beta = hdp->beta;
    
    double dir_params[HDP_MAX_STATES];
    double dir_sample[HDP_MAX_STATES];
    
    for (int k = 0; k < K; k++) {
        /* Build Dirichlet parameters */
        double sum = 0.0;
        for (int j = 0; j < K; j++) {
            dir_params[j] = alpha * beta[j] + hdp->n_trans[k * K + j];
            if (j == k) dir_params[j] += kappa;  /* Stickiness */
            if (dir_params[j] < HDP_EPS) dir_params[j] = HDP_EPS;
            sum += dir_params[j];
        }
        
        /* Sample from Dirichlet via Gamma */
        for (int j = 0; j < K; j++) {
            vdRngGamma(VSL_RNG_METHOD_GAMMA_GNORM, stream, 1, 
                       &dir_sample[j], dir_params[j], 0.0, 1.0);
        }
        
        /* Normalize */
        double total = cblas_dasum(K, dir_sample, 1);
        if (total < HDP_EPS) total = 1.0;
        
        for (int j = 0; j < K; j++) {
            hdp->pi[k * K + j] = dir_sample[j] / total;
            hdp->log_pi[k * K + j] = (hdp->pi[k * K + j] > HDP_MIN_PROB)
                                    ? log(hdp->pi[k * K + j])
                                    : HDP_LOG_ZERO;
        }
    }
}

/*═══════════════════════════════════════════════════════════════════════════════
 * EMISSION LIKELIHOOD
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * Compute log-likelihood of observation y under state k
 * 
 * Using log-chi-squared mixture (OCSN 2007) approximation:
 * y = 2h + log(ε²), ε ~ N(0,1)
 * log(ε²) ≈ Σ π_j × N(m_j, v_j²)
 */

/* OCSN mixture parameters (10 components) */
static const double OCSN_PROB[10] = {
    0.00609, 0.04775, 0.13057, 0.20674, 0.22715,
    0.18842, 0.12047, 0.05591, 0.01575, 0.00115
};
static const double OCSN_MEAN[10] = {
    1.92677, 1.34744, 0.73504, 0.02266, -0.85173,
    -1.97278, -3.46788, -5.55246, -8.68384, -14.65000
};
static const double OCSN_VAR[10] = {
    0.11265, 0.17788, 0.26768, 0.40611, 0.62699,
    0.98583, 1.57469, 2.54498, 4.16591, 7.33342
};

static double emission_log_likelihood(double y, const HDP_EmissionParams *emit)
{
    /* y = 2h + log(ε²), h = emit->mu (log-vol) */
    double h = emit->mu;
    
    /* Mixture likelihood */
    double log_lik = HDP_LOG_ZERO;
    
    for (int j = 0; j < 10; j++) {
        double y_adj = y - 2.0 * h - OCSN_MEAN[j];
        double var = OCSN_VAR[j];
        double log_comp = -0.5 * log(2.0 * M_PI * var) 
                         - 0.5 * y_adj * y_adj / var
                         + log(OCSN_PROB[j]);
        
        /* Log-add */
        if (log_comp > log_lik) {
            log_lik = log_comp + log1p(exp(log_lik - log_comp));
        } else if (log_lik > HDP_LOG_ZERO) {
            log_lik = log_lik + log1p(exp(log_comp - log_lik));
        }
    }
    
    return log_lik;
}

/**
 * Batch compute likelihoods for all states at time t
 * Uses MKL where possible
 */
static void compute_likelihoods_batch(StickyHDP *hdp, int t)
{
    double y = hdp->y[t];
    double *log_lik = hdp->log_lik + t * hdp->K_max;
    
    for (int k = 0; k < hdp->K; k++) {
        log_lik[k] = emission_log_likelihood(y, &hdp->emit[k]);
    }
    
    /* Zero out inactive states */
    for (int k = hdp->K; k < hdp->K_max; k++) {
        log_lik[k] = HDP_LOG_ZERO;
    }
}

/*═══════════════════════════════════════════════════════════════════════════════
 * BEAM SAMPLING CORE
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * Sample slice variables u_t
 * 
 * u_t ~ Uniform(0, π_{s_{t-1}, s_t})
 * 
 * The slice restricts active states to those with π_{s_{t-1}, k} > u_t
 */
static void sample_slice_variables(StickyHDP *hdp)
{
    VSLStreamStatePtr stream = (VSLStreamStatePtr)hdp->mkl_stream;
    const int T = hdp->T;
    const int K = hdp->K_max;
    
    /* Generate uniform randoms in batch */
    vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, T, hdp->u, 0.0, 1.0);
    
    /* Scale by transition probability */
    for (int t = 1; t < T; t++) {
        int s_prev = hdp->s[t - 1];
        int s_curr = hdp->s[t];
        double pi_trans = hdp->pi[s_prev * K + s_curr];
        hdp->u[t] *= pi_trans;
    }
    hdp->u[0] = 0.0;  /* No slice at t=0 */
}

/**
 * Determine active states given slice variables
 * 
 * A_t = {k : π_{s_{t-1}, k} > u_t}
 */
static void determine_active_states(StickyHDP *hdp, int t, int s_prev)
{
    const int K = hdp->K_max;
    double u = hdp->u[t];
    
    hdp->n_active = 0;
    memset(hdp->is_active, 0, K * sizeof(bool));
    
    /* States reachable from s_prev with prob > u */
    for (int k = 0; k < hdp->K; k++) {
        if (hdp->pi[s_prev * K + k] > u) {
            hdp->active[hdp->n_active++] = k;
            hdp->is_active[k] = true;
        }
    }
    
    /* Always include current state (numerical stability) */
    int s_curr = (t < hdp->T) ? hdp->s[t] : 0;
    if (!hdp->is_active[s_curr] && s_curr < hdp->K) {
        hdp->active[hdp->n_active++] = s_curr;
        hdp->is_active[s_curr] = true;
    }
    
    /* Must have at least one active state */
    if (hdp->n_active == 0) {
        hdp->active[0] = 0;
        hdp->n_active = 1;
        hdp->is_active[0] = true;
    }
}

/**
 * Forward filtering over active states only
 * 
 * α_t(k) = P(y_{1:t}, s_t = k)
 * α_t(k) ∝ P(y_t | k) × Σ_{j ∈ A_t} α_{t-1}(j) × π_{jk}
 */
static void forward_filter(StickyHDP *hdp)
{
    const int T = hdp->T;
    const int K = hdp->K_max;
    
    /* Initialize: α_0(k) ∝ β_k × P(y_0 | k) */
    compute_likelihoods_batch(hdp, 0);
    
    double *log_alpha_0 = hdp->log_alpha;
    for (int k = 0; k < hdp->K; k++) {
        log_alpha_0[k] = hdp->log_beta[k] + hdp->log_lik[k];
    }
    for (int k = hdp->K; k < K; k++) {
        log_alpha_0[k] = HDP_LOG_ZERO;
    }
    hdp->log_alpha_sum[0] = log_sum_exp(log_alpha_0, hdp->K);
    
    /* Forward pass */
    for (int t = 1; t < T; t++) {
        double *log_alpha_prev = hdp->log_alpha + (t - 1) * K;
        double *log_alpha_curr = hdp->log_alpha + t * K;
        double *log_lik_t = hdp->log_lik + t * K;
        
        /* Compute likelihoods */
        compute_likelihoods_batch(hdp, t);
        
        /* Determine active states based on previous state */
        int s_prev = hdp->s[t - 1];
        determine_active_states(hdp, t, s_prev);
        
        /* Initialize all to -inf */
        for (int k = 0; k < K; k++) {
            log_alpha_curr[k] = HDP_LOG_ZERO;
        }
        
        /* Compute forward message for active states only */
        for (int ai = 0; ai < hdp->n_active; ai++) {
            int k = hdp->active[ai];
            
            /* Sum over active predecessors */
            double log_sum = HDP_LOG_ZERO;
            for (int aj = 0; aj < hdp->n_active; aj++) {
                int j = hdp->active[aj];
                double log_contrib = log_alpha_prev[j] + hdp->log_pi[j * K + k];
                
                if (log_contrib > log_sum) {
                    log_sum = log_contrib + log1p(exp(log_sum - log_contrib));
                } else if (log_sum > HDP_LOG_ZERO) {
                    log_sum = log_sum + log1p(exp(log_contrib - log_sum));
                }
            }
            
            log_alpha_curr[k] = log_sum + log_lik_t[k];
        }
        
        /* Normalize */
        hdp->log_alpha_sum[t] = log_sum_exp(log_alpha_curr, K);
    }
    
    /* Update diagnostics */
    hdp->avg_active_states = 0.9 * hdp->avg_active_states + 0.1 * hdp->n_active;
}

/**
 * Backward sampling
 * 
 * Sample s_T from α_T(·), then s_{t-1} | s_t from backward kernel
 */
static void backward_sample(StickyHDP *hdp)
{
    VSLStreamStatePtr stream = (VSLStreamStatePtr)hdp->mkl_stream;
    const int T = hdp->T;
    const int K = hdp->K_max;
    
    /* Sample s_{T-1} from final forward message */
    double *log_alpha_T = hdp->log_alpha + (T - 1) * K;
    hdp->s[T - 1] = sample_categorical_log(stream, log_alpha_T, hdp->K, hdp->scratch1);
    
    /* Backward pass */
    for (int t = T - 2; t >= 0; t--) {
        int s_next = hdp->s[t + 1];
        double *log_alpha_t = hdp->log_alpha + t * K;
        
        /* Backward kernel: P(s_t | s_{t+1}, y_{1:t}) ∝ α_t(s_t) × π_{s_t, s_{t+1}} */
        for (int k = 0; k < hdp->K; k++) {
            hdp->scratch1[k] = log_alpha_t[k] + hdp->log_pi[k * K + s_next];
        }
        
        hdp->s[t] = sample_categorical_log(stream, hdp->scratch1, hdp->K, hdp->scratch2);
    }
}

/**
 * Update transition counts from sampled state sequence
 */
static void update_transition_counts(StickyHDP *hdp)
{
    const int T = hdp->T;
    const int K = hdp->K_max;
    
    /* Zero counts */
    memset(hdp->n_trans, 0, K * K * sizeof(int));
    
    /* Accumulate */
    for (int t = 1; t < T; t++) {
        int s_prev = hdp->s[t - 1];
        int s_curr = hdp->s[t];
        hdp->n_trans[s_prev * K + s_curr]++;
    }
}

/**
 * Update emission parameters given state assignments
 * 
 * Uses conjugate Normal-Inverse-Gamma update for (μ, σ²)
 */
static void update_emissions(StickyHDP *hdp)
{
    VSLStreamStatePtr stream = (VSLStreamStatePtr)hdp->mkl_stream;
    const int T = hdp->T;
    
    /* Reset sufficient statistics */
    for (int k = 0; k < hdp->K_max; k++) {
        hdp->emit[k].sum_x = 0.0;
        hdp->emit[k].sum_x2 = 0.0;
        hdp->emit[k].n = 0;
    }
    
    /* Accumulate */
    for (int t = 0; t < T; t++) {
        int k = hdp->s[t];
        double y = hdp->y[t];
        
        /* For log-chi-sq, the "effective observation" is (y - E[log ε²])/2 ≈ y/2 + 0.635 */
        double x = y / 2.0 + 0.635;  /* Adjust for E[log χ²₁] ≈ -1.27 */
        
        hdp->emit[k].sum_x += x;
        hdp->emit[k].sum_x2 += x * x;
        hdp->emit[k].n++;
    }
    
    /* Posterior update for each state */
    const HDP_EmissionPrior *prior = &hdp->emit_prior;
    
    for (int k = 0; k < hdp->K; k++) {
        HDP_EmissionParams *e = &hdp->emit[k];
        if (e->n == 0) continue;
        
        double n = (double)e->n;
        double x_bar = e->sum_x / n;
        double ss = e->sum_x2 - n * x_bar * x_bar;  /* Sum of squares */
        
        /* Normal-Inverse-Gamma posterior */
        double kappa_n = prior->kappa0 + n;
        double mu_n = (prior->kappa0 * prior->mu0 + n * x_bar) / kappa_n;
        double alpha_n = prior->alpha0 + n / 2.0;
        double beta_n = prior->beta0 + 0.5 * ss 
                       + 0.5 * prior->kappa0 * n * (x_bar - prior->mu0) * (x_bar - prior->mu0) / kappa_n;
        
        /* Sample σ² ~ Inv-Gamma(alpha_n, beta_n) */
        double sigma2 = 1.0 / sample_gamma(stream, alpha_n, beta_n);
        
        /* Sample μ ~ N(mu_n, σ²/kappa_n) */
        double mu;
        vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, 1, &mu, 
                      mu_n, sqrt(sigma2 / kappa_n));
        
        e->mu = mu;
        e->sigma = sqrt(sigma2);
    }
}

/*═══════════════════════════════════════════════════════════════════════════════
 * HYPERPARAMETER UPDATES
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * Sample concentration parameters γ, α using auxiliary variable method
 * (Escobar & West, 1995; Teh et al., 2006)
 */
static void sample_hyperparameters(StickyHDP *hdp)
{
    if (!hdp->learn_hyperparams) return;
    
    VSLStreamStatePtr stream = (VSLStreamStatePtr)hdp->mkl_stream;
    const int K = hdp->K;
    const int T = hdp->T;
    
    /* Sample γ given K (number of states) using auxiliary variable */
    /* γ | K, T ~ Gamma(γ_a + K - 1, γ_b - log(η)) where η ~ Beta(γ+1, T) */
    double eta = sample_beta(stream, hdp->gamma + 1.0, (double)T);
    double pi_eta = (hdp->gamma_a + K - 1.0) / 
                    (hdp->gamma_a + K - 1.0 + T * (hdp->gamma_b - log(eta)));
    
    double u;
    vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, 1, &u, 0.0, 1.0);
    
    if (u < pi_eta) {
        hdp->gamma = sample_gamma(stream, hdp->gamma_a + K, hdp->gamma_b - log(eta));
    } else {
        hdp->gamma = sample_gamma(stream, hdp->gamma_a + K - 1, hdp->gamma_b - log(eta));
    }
    
    /* Sample α and κ (simplified - could use more sophisticated sampler) */
    /* For now, use MH with Gamma proposal */
    /* TODO: Implement proper auxiliary variable sampler for sticky parameters */
}

/*═══════════════════════════════════════════════════════════════════════════════
 * LIFECYCLE
 *═══════════════════════════════════════════════════════════════════════════════*/

StickyHDP *sticky_hdp_create(int K_max, int T_max)
{
    if (K_max > HDP_MAX_STATES) K_max = HDP_MAX_STATES;
    if (T_max > HDP_MAX_WINDOW) T_max = HDP_MAX_WINDOW;
    
    StickyHDP *hdp = (StickyHDP *)mkl_calloc(1, sizeof(StickyHDP), HDP_CACHE_LINE);
    if (!hdp) return NULL;
    
    hdp->K_max = K_max;
    hdp->T_max = T_max;
    hdp->K = 1;
    hdp->T = 0;
    
    /* Default hyperparameters */
    hdp->gamma = 1.0;
    hdp->alpha = 1.0;
    hdp->kappa = 10.0;
    hdp->learn_hyperparams = false;
    
    /* Allocate arrays */
    hdp->beta = aligned_alloc_double(K_max);
    hdp->log_beta = aligned_alloc_double(K_max);
    hdp->stick_remain = aligned_alloc_double(K_max);
    
    hdp->pi = aligned_alloc_double(K_max * K_max);
    hdp->log_pi = aligned_alloc_double(K_max * K_max);
    hdp->n_trans = aligned_alloc_int(K_max * K_max);
    
    hdp->emit = (HDP_EmissionParams *)mkl_calloc(K_max, sizeof(HDP_EmissionParams), HDP_CACHE_LINE);
    
    hdp->y = aligned_alloc_double(T_max);
    hdp->s = aligned_alloc_int(T_max);
    hdp->u = aligned_alloc_double(T_max);
    
    hdp->active = aligned_alloc_int(K_max);
    hdp->is_active = aligned_alloc_bool(K_max);
    
    hdp->log_alpha = aligned_alloc_double(T_max * K_max);
    hdp->log_alpha_sum = aligned_alloc_double(T_max);
    
    hdp->log_lik = aligned_alloc_double(T_max * K_max);
    hdp->lik_valid = aligned_alloc_bool(T_max * K_max);
    
    hdp->scratch1 = aligned_alloc_double(K_max);
    hdp->scratch2 = aligned_alloc_double(K_max);
    hdp->scratch3 = aligned_alloc_double(K_max);
    
    /* Check allocations */
    if (!hdp->beta || !hdp->log_beta || !hdp->stick_remain ||
        !hdp->pi || !hdp->log_pi || !hdp->n_trans ||
        !hdp->emit || !hdp->y || !hdp->s || !hdp->u ||
        !hdp->active || !hdp->is_active ||
        !hdp->log_alpha || !hdp->log_alpha_sum ||
        !hdp->log_lik || !hdp->lik_valid ||
        !hdp->scratch1 || !hdp->scratch2 || !hdp->scratch3) {
        sticky_hdp_destroy(hdp);
        return NULL;
    }
    
    /* Initialize MKL RNG */
    hdp->rng_seed = 42;
    vslNewStream((VSLStreamStatePtr *)&hdp->mkl_stream, VSL_BRNG_SFMT19937, hdp->rng_seed);
    
    /* Initialize β to uniform */
    for (int k = 0; k < K_max; k++) {
        hdp->beta[k] = 1.0 / K_max;
        hdp->log_beta[k] = log(hdp->beta[k]);
        hdp->stick_remain[k] = 1.0;
    }
    
    /* Initialize π to uniform with stickiness */
    double self_prob = 0.9;
    double other_prob = 0.1 / (K_max - 1);
    for (int i = 0; i < K_max; i++) {
        for (int j = 0; j < K_max; j++) {
            hdp->pi[i * K_max + j] = (i == j) ? self_prob : other_prob;
            hdp->log_pi[i * K_max + j] = log(hdp->pi[i * K_max + j]);
        }
    }
    
    /* Default emission prior */
    hdp->emit_prior.mu0 = -3.0;
    hdp->emit_prior.kappa0 = 0.1;
    hdp->emit_prior.alpha0 = 2.0;
    hdp->emit_prior.beta0 = 0.5;
    hdp->emit_prior.theta_min = 0.001;
    hdp->emit_prior.theta_max = 0.5;
    
    /* Initialize emissions */
    for (int k = 0; k < K_max; k++) {
        hdp->emit[k].mu = -4.5 + k * 0.75;  /* Spread across range */
        hdp->emit[k].sigma = 0.2;
        hdp->emit[k].theta = 0.05;
    }
    
    return hdp;
}

void sticky_hdp_destroy(StickyHDP *hdp)
{
    if (!hdp) return;
    
    if (hdp->mkl_stream) {
        vslDeleteStream((VSLStreamStatePtr *)&hdp->mkl_stream);
    }
    
    mkl_free(hdp->beta);
    mkl_free(hdp->log_beta);
    mkl_free(hdp->stick_remain);
    mkl_free(hdp->pi);
    mkl_free(hdp->log_pi);
    mkl_free(hdp->n_trans);
    mkl_free(hdp->emit);
    mkl_free(hdp->y);
    mkl_free(hdp->s);
    mkl_free(hdp->u);
    mkl_free(hdp->active);
    mkl_free(hdp->is_active);
    mkl_free(hdp->log_alpha);
    mkl_free(hdp->log_alpha_sum);
    mkl_free(hdp->log_lik);
    mkl_free(hdp->lik_valid);
    mkl_free(hdp->scratch1);
    mkl_free(hdp->scratch2);
    mkl_free(hdp->scratch3);
    
    mkl_free(hdp);
}

void sticky_hdp_reset(StickyHDP *hdp)
{
    if (!hdp) return;
    
    hdp->T = 0;
    hdp->K = 1;
    
    memset(hdp->n_trans, 0, hdp->K_max * hdp->K_max * sizeof(int));
    memset(hdp->s, 0, hdp->T_max * sizeof(int));
    
    hdp->total_sweeps = 0;
    hdp->total_new_states = 0;
    hdp->avg_active_states = 1.0;
    hdp->last_log_marginal = HDP_LOG_ZERO;
}

/*═══════════════════════════════════════════════════════════════════════════════
 * CONFIGURATION
 *═══════════════════════════════════════════════════════════════════════════════*/

void sticky_hdp_set_concentration(StickyHDP *hdp, double gamma, double alpha)
{
    if (!hdp) return;
    hdp->gamma = gamma > 0.01 ? gamma : 0.01;
    hdp->alpha = alpha > 0.01 ? alpha : 0.01;
}

void sticky_hdp_set_stickiness(StickyHDP *hdp, double kappa)
{
    if (!hdp) return;
    hdp->kappa = kappa > 0.0 ? kappa : 0.0;
}

void sticky_hdp_enable_hyperparam_learning(StickyHDP *hdp,
                                            double gamma_a, double gamma_b,
                                            double alpha_a, double alpha_b,
                                            double kappa_a, double kappa_b)
{
    if (!hdp) return;
    hdp->learn_hyperparams = true;
    hdp->gamma_a = gamma_a;
    hdp->gamma_b = gamma_b;
    hdp->alpha_a = alpha_a;
    hdp->alpha_b = alpha_b;
    hdp->kappa_a = kappa_a;
    hdp->kappa_b = kappa_b;
}

void sticky_hdp_disable_hyperparam_learning(StickyHDP *hdp)
{
    if (!hdp) return;
    hdp->learn_hyperparams = false;
}

void sticky_hdp_set_emission_prior(StickyHDP *hdp,
                                    double mu0, double kappa0,
                                    double alpha0, double beta0,
                                    double theta_min, double theta_max)
{
    if (!hdp) return;
    hdp->emit_prior.mu0 = mu0;
    hdp->emit_prior.kappa0 = kappa0;
    hdp->emit_prior.alpha0 = alpha0;
    hdp->emit_prior.beta0 = beta0;
    hdp->emit_prior.theta_min = theta_min;
    hdp->emit_prior.theta_max = theta_max;
}

void sticky_hdp_init_regimes(StickyHDP *hdp, int n_regimes,
                              const double *mu_vol,
                              const double *sigma_vol,
                              const double *theta)
{
    if (!hdp || n_regimes <= 0) return;
    if (n_regimes > hdp->K_max) n_regimes = hdp->K_max;
    
    hdp->K = n_regimes;
    
    /* Set emission parameters */
    for (int k = 0; k < n_regimes; k++) {
        hdp->emit[k].mu = mu_vol[k];
        hdp->emit[k].sigma = sigma_vol[k];
        hdp->emit[k].theta = theta ? theta[k] : 0.05;
    }
    
    /* Initialize β uniformly over active states */
    double beta_k = 1.0 / n_regimes;
    for (int k = 0; k < n_regimes; k++) {
        hdp->beta[k] = beta_k;
        hdp->log_beta[k] = log(beta_k);
    }
    for (int k = n_regimes; k < hdp->K_max; k++) {
        hdp->beta[k] = 0.0;
        hdp->log_beta[k] = HDP_LOG_ZERO;
    }
}

/*═══════════════════════════════════════════════════════════════════════════════
 * OBSERVATION MANAGEMENT
 *═══════════════════════════════════════════════════════════════════════════════*/

void sticky_hdp_observe(StickyHDP *hdp, double y)
{
    if (!hdp) return;
    if (hdp->T >= hdp->T_max) {
        /* Window full - slide */
        sticky_hdp_slide_window(hdp, hdp->T_max / 4);
    }
    
    hdp->y[hdp->T] = y;
    hdp->s[hdp->T] = hdp->K > 0 ? hdp->s[hdp->T > 0 ? hdp->T - 1 : 0] : 0;
    hdp->T++;
}

void sticky_hdp_set_observations(StickyHDP *hdp, const double *y, int T)
{
    if (!hdp || !y || T <= 0) return;
    if (T > hdp->T_max) T = hdp->T_max;
    
    memcpy(hdp->y, y, T * sizeof(double));
    hdp->T = T;
    
    /* Initialize state sequence to 0 */
    memset(hdp->s, 0, T * sizeof(int));
}

void sticky_hdp_clear_observations(StickyHDP *hdp)
{
    if (!hdp) return;
    hdp->T = 0;
}

void sticky_hdp_slide_window(StickyHDP *hdp, int n_remove)
{
    if (!hdp || n_remove <= 0) return;
    if (n_remove >= hdp->T) {
        hdp->T = 0;
        return;
    }
    
    int new_T = hdp->T - n_remove;
    memmove(hdp->y, hdp->y + n_remove, new_T * sizeof(double));
    memmove(hdp->s, hdp->s + n_remove, new_T * sizeof(int));
    hdp->T = new_T;
}

/*═══════════════════════════════════════════════════════════════════════════════
 * INFERENCE
 *═══════════════════════════════════════════════════════════════════════════════*/

double sticky_hdp_beam_sweep_single(StickyHDP *hdp)
{
    if (!hdp || hdp->T == 0) return HDP_LOG_ZERO;
    
    /* 1. Sample slice variables */
    sample_slice_variables(hdp);
    
    /* 2. Forward filter */
    forward_filter(hdp);
    
    /* 3. Backward sample */
    backward_sample(hdp);
    
    /* 4. Update counts */
    update_transition_counts(hdp);
    
    /* 5. Sample β */
    sample_beta_stick_breaking(hdp);
    
    /* 6. Sample π */
    sample_transitions(hdp);
    
    /* 7. Update emissions */
    update_emissions(hdp);
    
    /* 8. Optionally update hyperparameters */
    if (hdp->learn_hyperparams) {
        sample_hyperparameters(hdp);
    }
    
    hdp->total_sweeps++;
    
    /* Return log marginal likelihood */
    hdp->last_log_marginal = 0.0;
    for (int t = 0; t < hdp->T; t++) {
        hdp->last_log_marginal += hdp->log_alpha_sum[t];
    }
    
    return hdp->last_log_marginal;
}

double sticky_hdp_beam_sweep(StickyHDP *hdp, int n_sweeps)
{
    double log_marg = HDP_LOG_ZERO;
    for (int i = 0; i < n_sweeps; i++) {
        log_marg = sticky_hdp_beam_sweep_single(hdp);
    }
    return log_marg;
}

void sticky_hdp_update_hyperparams(StickyHDP *hdp)
{
    if (!hdp) return;
    sample_hyperparameters(hdp);
}

void sticky_hdp_update_emissions(StickyHDP *hdp)
{
    if (!hdp) return;
    update_emissions(hdp);
}

/*═══════════════════════════════════════════════════════════════════════════════
 * STATE QUERIES
 *═══════════════════════════════════════════════════════════════════════════════*/

int sticky_hdp_map_state(const StickyHDP *hdp)
{
    if (!hdp || hdp->T == 0) return 0;
    return hdp->s[hdp->T - 1];
}

int sticky_hdp_state_probs(const StickyHDP *hdp, double *probs)
{
    if (!hdp || !probs || hdp->T == 0) return 0;
    
    const double *log_alpha = hdp->log_alpha + (hdp->T - 1) * hdp->K_max;
    double max_la = log_alpha[cblas_idamax(hdp->K, log_alpha, 1)];
    
    double sum = 0.0;
    for (int k = 0; k < hdp->K_max; k++) {
        if (k < hdp->K && log_alpha[k] > HDP_LOG_ZERO) {
            probs[k] = exp(log_alpha[k] - max_la);
            sum += probs[k];
        } else {
            probs[k] = 0.0;
        }
    }
    
    if (sum > 0) {
        for (int k = 0; k < hdp->K_max; k++) {
            probs[k] /= sum;
        }
    }
    
    return hdp->K;
}

int sticky_hdp_get_states(const StickyHDP *hdp, int *states)
{
    if (!hdp || !states) return 0;
    memcpy(states, hdp->s, hdp->T * sizeof(int));
    return hdp->T;
}

int sticky_hdp_num_states(const StickyHDP *hdp)
{
    return hdp ? hdp->K : 0;
}

/*═══════════════════════════════════════════════════════════════════════════════
 * PARAMETER QUERIES
 *═══════════════════════════════════════════════════════════════════════════════*/

void sticky_hdp_get_transitions(const StickyHDP *hdp, double *trans, int K)
{
    if (!hdp || !trans) return;
    if (K > hdp->K) K = hdp->K;
    
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < K; j++) {
            trans[i * K + j] = hdp->pi[i * hdp->K_max + j];
        }
    }
}

void sticky_hdp_get_beta(const StickyHDP *hdp, double *beta, int K)
{
    if (!hdp || !beta) return;
    if (K > hdp->K_max) K = hdp->K_max;
    memcpy(beta, hdp->beta, K * sizeof(double));
}

void sticky_hdp_get_emission(const StickyHDP *hdp, int k, HDP_EmissionParams *params)
{
    if (!hdp || !params || k < 0 || k >= hdp->K_max) return;
    *params = hdp->emit[k];
}

void sticky_hdp_get_hyperparams(const StickyHDP *hdp,
                                 double *gamma, double *alpha, double *kappa)
{
    if (!hdp) return;
    if (gamma) *gamma = hdp->gamma;
    if (alpha) *alpha = hdp->alpha;
    if (kappa) *kappa = hdp->kappa;
}

double sticky_hdp_get_stickiness_prob(const StickyHDP *hdp, int k)
{
    if (!hdp || k < 0 || k >= hdp->K) return 0.0;
    return hdp->pi[k * hdp->K_max + k];
}

/*═══════════════════════════════════════════════════════════════════════════════
 * DIAGNOSTICS
 *═══════════════════════════════════════════════════════════════════════════════*/

void sticky_hdp_print_summary(const StickyHDP *hdp)
{
    if (!hdp) return;
    
    printf("\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  Sticky HDP-HMM Summary\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  States:      K=%d (max=%d)\n", hdp->K, hdp->K_max);
    printf("  Window:      T=%d (max=%d)\n", hdp->T, hdp->T_max);
    printf("  Hyperparams: γ=%.3f, α=%.3f, κ=%.3f\n", 
           hdp->gamma, hdp->alpha, hdp->kappa);
    printf("  Learning:    %s\n", hdp->learn_hyperparams ? "ON" : "OFF");
    printf("  Sweeps:      %lu\n", (unsigned long)hdp->total_sweeps);
    printf("  Avg active:  %.1f states\n", hdp->avg_active_states);
    printf("  Log marg:    %.2f\n", hdp->last_log_marginal);
    printf("═══════════════════════════════════════════════════════════════\n");
}

void sticky_hdp_print_transitions(const StickyHDP *hdp)
{
    if (!hdp) return;
    
    printf("\n  Transition Matrix (π):\n");
    printf("       ");
    for (int j = 0; j < hdp->K; j++) printf("   S%d   ", j);
    printf("\n");
    
    for (int i = 0; i < hdp->K; i++) {
        printf("  S%d: ", i);
        for (int j = 0; j < hdp->K; j++) {
            double p = hdp->pi[i * hdp->K_max + j];
            if (i == j)
                printf(" [%5.1f%%]", p * 100);
            else
                printf("  %5.1f%% ", p * 100);
        }
        printf("\n");
    }
}

void sticky_hdp_print_emissions(const StickyHDP *hdp)
{
    if (!hdp) return;
    
    printf("\n  Emission Parameters:\n");
    printf("  %-6s %10s %10s %10s\n", "State", "μ", "σ", "n");
    for (int k = 0; k < hdp->K; k++) {
        printf("  S%-5d %10.3f %10.3f %10d\n",
               k, hdp->emit[k].mu, hdp->emit[k].sigma, hdp->emit[k].n);
    }
}

void sticky_hdp_get_diagnostics(const StickyHDP *hdp, HDP_Diagnostics *diag)
{
    if (!hdp || !diag) return;
    
    diag->K = hdp->K;
    diag->avg_active = hdp->avg_active_states;
    diag->log_marginal = hdp->last_log_marginal;
    diag->gamma = hdp->gamma;
    diag->alpha = hdp->alpha;
    diag->kappa = hdp->kappa;
    diag->total_sweeps = hdp->total_sweeps;
    diag->last_sweep_time_us = 0.0;  /* Would need timing instrumentation */
}

/*═══════════════════════════════════════════════════════════════════════════════
 * RBPF INTEGRATION
 *═══════════════════════════════════════════════════════════════════════════════*/

void sticky_hdp_export_to_rbpf(const StickyHDP *hdp, int K_rbpf,
                                double *trans,
                                double *mu_vol,
                                double *sigma_vol,
                                double *theta)
{
    if (!hdp || K_rbpf <= 0) return;
    
    int K_src = hdp->K < K_rbpf ? hdp->K : K_rbpf;
    
    /* Sort states by μ for consistent ordering */
    int order[HDP_MAX_STATES];
    for (int k = 0; k < K_src; k++) order[k] = k;
    
    /* Simple insertion sort by mu */
    for (int i = 1; i < K_src; i++) {
        int j = i;
        while (j > 0 && hdp->emit[order[j-1]].mu > hdp->emit[order[j]].mu) {
            int tmp = order[j];
            order[j] = order[j-1];
            order[j-1] = tmp;
            j--;
        }
    }
    
    /* Export sorted parameters */
    for (int i = 0; i < K_rbpf; i++) {
        int k = (i < K_src) ? order[i] : order[K_src - 1];
        
        if (mu_vol) mu_vol[i] = hdp->emit[k].mu;
        if (sigma_vol) sigma_vol[i] = hdp->emit[k].sigma;
        if (theta) theta[i] = hdp->emit[k].theta;
    }
    
    /* Export transition matrix (reordered) */
    if (trans) {
        for (int i = 0; i < K_rbpf; i++) {
            int ki = (i < K_src) ? order[i] : order[K_src - 1];
            for (int j = 0; j < K_rbpf; j++) {
                int kj = (j < K_src) ? order[j] : order[K_src - 1];
                trans[i * K_rbpf + j] = hdp->pi[ki * hdp->K_max + kj];
            }
        }
        
        /* Renormalize rows */
        for (int i = 0; i < K_rbpf; i++) {
            double sum = 0.0;
            for (int j = 0; j < K_rbpf; j++) sum += trans[i * K_rbpf + j];
            if (sum > 0) {
                for (int j = 0; j < K_rbpf; j++) trans[i * K_rbpf + j] /= sum;
            }
        }
    }
}

void sticky_hdp_import_from_rbpf(StickyHDP *hdp, int K_rbpf,
                                  const int *regime_seq, int T,
                                  const double *mu_vol,
                                  const double *sigma_vol)
{
    if (!hdp || K_rbpf <= 0) return;
    
    /* Initialize HDP states from RBPF regimes */
    sticky_hdp_init_regimes(hdp, K_rbpf, mu_vol, sigma_vol, NULL);
    
    /* Copy state sequence */
    if (regime_seq && T > 0) {
        int T_copy = T < hdp->T_max ? T : hdp->T_max;
        memcpy(hdp->s, regime_seq, T_copy * sizeof(int));
        
        /* Count transitions */
        memset(hdp->n_trans, 0, hdp->K_max * hdp->K_max * sizeof(int));
        for (int t = 1; t < T_copy; t++) {
            int s_prev = regime_seq[t-1];
            int s_curr = regime_seq[t];
            if (s_prev >= 0 && s_prev < K_rbpf && s_curr >= 0 && s_curr < K_rbpf) {
                hdp->n_trans[s_prev * hdp->K_max + s_curr]++;
            }
        }
        
        /* Sample transitions given counts */
        sample_transitions(hdp);
    }
}
