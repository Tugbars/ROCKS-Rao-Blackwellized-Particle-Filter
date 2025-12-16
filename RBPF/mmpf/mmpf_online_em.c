/**
 * @file mmpf_online_em.c
 * @brief Online Expectation-Maximization for Regime Discovery
 *
 * Stepwise EM algorithm for learning regime centers from streaming data.
 */

#include "mmpf_online_em.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/*═══════════════════════════════════════════════════════════════════════════
 * HELPERS
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Gaussian PDF (not log, for responsibility computation)
 */
static inline double gaussian_pdf(double x, double mu, double var) {
    if (var < 1e-10) var = 1e-10;
    double d = x - mu;
    return exp(-0.5 * d * d / var) / sqrt(2.0 * M_PI * var);
}

/**
 * @brief Bubble sort clusters by mu (enforce ordering constraint)
 */
static void sort_clusters(MMPF_OnlineEM *em) {
    int n = em->n_clusters;
    
    /* Simple bubble sort (n is small, typically 3) */
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - 1 - i; j++) {
            if (em->mu[j] > em->mu[j + 1]) {
                /* Swap all parameters */
                double tmp;
                
                tmp = em->mu[j];
                em->mu[j] = em->mu[j + 1];
                em->mu[j + 1] = tmp;
                
                tmp = em->var[j];
                em->var[j] = em->var[j + 1];
                em->var[j + 1] = tmp;
                
                tmp = em->pi[j];
                em->pi[j] = em->pi[j + 1];
                em->pi[j + 1] = tmp;
            }
        }
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * INITIALIZATION
 *═══════════════════════════════════════════════════════════════════════════*/

void mmpf_online_em_init(MMPF_OnlineEM *em, int n_clusters) {
    if (n_clusters < 2) n_clusters = 2;
    if (n_clusters > MMPF_EM_MAX_CLUSTERS) n_clusters = MMPF_EM_MAX_CLUSTERS;
    
    memset(em, 0, sizeof(MMPF_OnlineEM));
    em->n_clusters = n_clusters;
    
    /* Default initialization from Stage 1 tuner results */
    if (n_clusters == 3) {
        /* Calm */
        em->mu[0] = -4.5;   /* Midpoint of [-5.5, -3.5] */
        em->var[0] = 1.0;
        em->pi[0] = 0.5;    /* Most time is calm */
        
        /* Trend */
        em->mu[1] = -3.0;   /* Between calm and crisis */
        em->var[1] = 1.0;
        em->pi[1] = 0.3;
        
        /* Crisis */
        em->mu[2] = -1.25;  /* Midpoint of [-2.5, 0.0] */
        em->var[2] = 1.5;   /* Higher variance for crisis */
        em->pi[2] = 0.2;    /* Rare but important */
    } else {
        /* Generic initialization: spread evenly from -5 to 0 */
        double step = 4.0 / (n_clusters - 1);
        for (int k = 0; k < n_clusters; k++) {
            em->mu[k] = -5.0 + k * step;
            em->var[k] = 1.0;
            em->pi[k] = 1.0 / n_clusters;
        }
    }
    
    /* Bounds */
    em->eta = 0.001;      /* ~1000 tick memory */
    em->min_var = 0.1;    /* Prevent singularity */
    em->min_pi = 0.05;    /* Prevent cluster death */
    em->min_mu = -10.0;
    em->max_mu = 2.0;
    
    em->tick_count = 0;
}

void mmpf_online_em_init_custom(MMPF_OnlineEM *em, int n_clusters,
                                 const double *mu, const double *var,
                                 const double *pi) {
    mmpf_online_em_init(em, n_clusters);
    
    /* Override with custom values */
    if (mu) {
        memcpy(em->mu, mu, n_clusters * sizeof(double));
    }
    
    if (var) {
        memcpy(em->var, var, n_clusters * sizeof(double));
    }
    
    if (pi) {
        memcpy(em->pi, pi, n_clusters * sizeof(double));
        
        /* Normalize */
        double sum = 0.0;
        for (int k = 0; k < n_clusters; k++) sum += em->pi[k];
        if (sum > 1e-10) {
            for (int k = 0; k < n_clusters; k++) em->pi[k] /= sum;
        }
    }
    
    /* Ensure ordering */
    sort_clusters(em);
}

/*═══════════════════════════════════════════════════════════════════════════
 * ONLINE EM UPDATE
 *═══════════════════════════════════════════════════════════════════════════*/

void mmpf_online_em_update(MMPF_OnlineEM *em, double y_log_vol) {
    int n = em->n_clusters;
    double gamma[MMPF_EM_MAX_CLUSTERS];
    double sum_gamma = 0.0;
    
    /*─────────────────────────────────────────────────────────────────────────
     * E-STEP: Compute responsibilities
     *───────────────────────────────────────────────────────────────────────*/
    for (int k = 0; k < n; k++) {
        double prob = gaussian_pdf(y_log_vol, em->mu[k], em->var[k]);
        gamma[k] = em->pi[k] * prob;
        sum_gamma += gamma[k];
    }
    
    /* Normalize responsibilities */
    if (sum_gamma < 1e-10) {
        /* Observation far from all clusters: uniform assignment */
        for (int k = 0; k < n; k++) {
            gamma[k] = 1.0 / n;
        }
    } else {
        for (int k = 0; k < n; k++) {
            gamma[k] /= sum_gamma;
        }
    }
    
    /* Store for diagnostics */
    memcpy(em->last_responsibility, gamma, n * sizeof(double));
    
    /*─────────────────────────────────────────────────────────────────────────
     * M-STEP: Update parameters via stochastic approximation
     *───────────────────────────────────────────────────────────────────────*/
    double eta = em->eta;
    
    for (int k = 0; k < n; k++) {
        /* Update weight (π_k)
         * π_new = (1-η)×π_old + η×γ_k
         */
        em->pi[k] = (1.0 - eta) * em->pi[k] + eta * gamma[k];
        
        /* Floor to prevent cluster death */
        if (em->pi[k] < em->min_pi) {
            em->pi[k] = em->min_pi;
        }
        
        /* Effective learning rate for this cluster
         * If cluster is rare (low π), it learns slower for stability
         * η_k = η × γ_k / π_k
         */
        double eta_k = eta * gamma[k] / (em->pi[k] + 1e-10);
        
        /* Update mean (μ_k)
         * μ_new = μ_old + η_k × (y - μ_old)
         */
        double delta = y_log_vol - em->mu[k];
        em->mu[k] += eta_k * delta;
        
        /* Clamp to bounds */
        if (em->mu[k] < em->min_mu) em->mu[k] = em->min_mu;
        if (em->mu[k] > em->max_mu) em->mu[k] = em->max_mu;
        
        /* Update variance (σ²_k)
         * σ²_new = (1-η_k)×σ²_old + η_k×(y - μ_new)²
         *
         * Note: We use the NEW mean for computing residual.
         * This is the online Welford-style update.
         */
        double residual = y_log_vol - em->mu[k];
        double new_var = (1.0 - eta_k) * em->var[k] + eta_k * residual * residual;
        
        /* Floor to prevent singularity */
        if (new_var < em->min_var) {
            new_var = em->min_var;
        }
        
        em->var[k] = new_var;
    }
    
    /*─────────────────────────────────────────────────────────────────────────
     * ENFORCE ORDERING CONSTRAINT
     * Ensure μ₀ < μ₁ < ... < μ_{K-1} to prevent label switching
     *───────────────────────────────────────────────────────────────────────*/
    sort_clusters(em);
    
    /*─────────────────────────────────────────────────────────────────────────
     * RE-NORMALIZE WEIGHTS
     * After flooring, weights might not sum to 1
     *───────────────────────────────────────────────────────────────────────*/
    double pi_sum = 0.0;
    for (int k = 0; k < n; k++) {
        pi_sum += em->pi[k];
    }
    if (pi_sum > 1e-10) {
        for (int k = 0; k < n; k++) {
            em->pi[k] /= pi_sum;
        }
    }
    
    em->tick_count++;
}

/*═══════════════════════════════════════════════════════════════════════════
 * QUERY FUNCTIONS
 *═══════════════════════════════════════════════════════════════════════════*/

int mmpf_online_em_classify(const MMPF_OnlineEM *em, double y_log_vol) {
    int n = em->n_clusters;
    int best_k = 0;
    double best_score = -1e30;
    
    for (int k = 0; k < n; k++) {
        double prob = gaussian_pdf(y_log_vol, em->mu[k], em->var[k]);
        double score = em->pi[k] * prob;
        if (score > best_score) {
            best_score = score;
            best_k = k;
        }
    }
    
    return best_k;
}

void mmpf_online_em_responsibilities(const MMPF_OnlineEM *em, double y_log_vol,
                                      double *gamma) {
    int n = em->n_clusters;
    double sum = 0.0;
    
    for (int k = 0; k < n; k++) {
        double prob = gaussian_pdf(y_log_vol, em->mu[k], em->var[k]);
        gamma[k] = em->pi[k] * prob;
        sum += gamma[k];
    }
    
    if (sum > 1e-10) {
        for (int k = 0; k < n; k++) {
            gamma[k] /= sum;
        }
    } else {
        for (int k = 0; k < n; k++) {
            gamma[k] = 1.0 / n;
        }
    }
}

void mmpf_online_em_set_learning_rate(MMPF_OnlineEM *em, double eta) {
    if (eta < 0.0001) eta = 0.0001;
    if (eta > 0.1) eta = 0.1;
    em->eta = eta;
}

void mmpf_online_em_get_centers(const MMPF_OnlineEM *em, double *mu_out) {
    memcpy(mu_out, em->mu, em->n_clusters * sizeof(double));
}

int mmpf_online_em_converged(const MMPF_OnlineEM *em, double var_threshold) {
    /* Simple heuristic: check if all variances are below threshold */
    /* This indicates clusters have stabilized */
    for (int k = 0; k < em->n_clusters; k++) {
        if (em->var[k] > var_threshold) {
            return 0;
        }
    }
    return 1;
}

void mmpf_online_em_reset(MMPF_OnlineEM *em) {
    int n = em->n_clusters;
    mmpf_online_em_init(em, n);
}

/*═══════════════════════════════════════════════════════════════════════════
 * DIAGNOSTICS
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Print EM state (for debugging)
 */
void mmpf_online_em_dump(const MMPF_OnlineEM *em) {
    /* For debugging, implement as needed */
    (void)em;
}
