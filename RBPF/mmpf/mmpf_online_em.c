/**
 * @file mmpf_online_em.c
 * @brief Online Expectation-Maximization for Regime Discovery
 */
#include <math.h>
#include <stdlib.h>
#include "mmpf_internal.h"

#define EM_LEARNING_RATE 0.001  /* Slow adaptation (approx 1000 ticks memory) */
#define MIN_SIGMA        0.1    /* Prevent collapse to singularity */

/* Gaussian PDF helper */
static double gaussian_pdf(double x, double mu, double var) {
    double d = x - mu;
    return exp(-0.5 * d * d / var) / sqrt(2.0 * M_PI * var);
}

void mmpf_online_em_init(MMPF_OnlineEM *em, double initial_vol) {
    /* Initialize 3 clusters slightly separated around initial vol */
    /* Cluster 0: Calm (Lower) */
    em->mu[0] = initial_vol - 1.0; 
    em->var[0] = 0.5;
    em->pi[0] = 0.5; /* Prior: Mostly calm */

    /* Cluster 1: Trend (Middle) */
    em->mu[1] = initial_vol;
    em->var[1] = 0.5;
    em->pi[1] = 0.3;

    /* Cluster 2: Crisis (Higher) */
    em->mu[2] = initial_vol + 1.5;
    em->var[2] = 1.0;
    em->pi[2] = 0.2;
}

void mmpf_online_em_update(MMPF_OnlineEM *em, double y_log_vol) {
    double gamma[3];
    double sum_gamma = 0.0;
    int k;

    /* --- E-STEP: Calculate Responsibilities --- */
    for (k = 0; k < 3; k++) {
        double prob = gaussian_pdf(y_log_vol, em->mu[k], em->var[k]);
        gamma[k] = em->pi[k] * prob;
        sum_gamma += gamma[k];
    }

    /* Normalize responsibilities */
    if (sum_gamma < 1e-10) {
        /* Outlier far from all clusters: assign to closest mu */
        /* (Simplified fallback: uniform assignment) */
        for (k = 0; k < 3; k++) gamma[k] = 1.0/3.0; 
    } else {
        for (k = 0; k < 3; k++) gamma[k] /= sum_gamma;
    }

    /* --- M-STEP: Update Parameters directly (Stochastic Approximation) --- */
    /* Note: We update parameters directly rather than sufficient stats 
     * to avoid numerical drift in long-running C code. 
     * Formula: mu_new = mu_old + eta * gamma * (y - mu_old) / pi_old 
     * This is mathematically equivalent to the sufficient stat update.
     */
    
    double eta = EM_LEARNING_RATE;

    for (k = 0; k < 3; k++) {
        /* Update Weight (Pi) */
        /* pi_new = (1-eta)*pi_old + eta*gamma */
        em->pi[k] = (1.0 - eta) * em->pi[k] + eta * gamma[k];

        /* Effective learning rate for this cluster */
        /* If cluster is rare (low pi), it learns slower to stay stable */
        double eta_k = eta * gamma[k] / (em->pi[k] + 1e-10);

        /* Update Mean (Mu) */
        double delta = y_log_vol - em->mu[k];
        em->mu[k] += eta_k * delta;

        /* Update Variance (Sigma^2) */
        /* var_new = (1-eta_k)*var_old + eta_k * (y - mu_new)(y - mu_old) */
        /* We use simplified approximation: error^2 */
        double error_sq = delta * delta;
        em->var[k] = (1.0 - eta_k) * em->var[k] + eta_k * error_sq;

        /* Safety floor for variance */
        if (em->var[k] < MIN_SIGMA) em->var[k] = MIN_SIGMA;
    }

    /* --- ENFORCE ORDERING (Swim Lanes Logic) --- */
    /* Ensure Mu[0] < Mu[1] < Mu[2] so "Calm" stays "Calm" */
    /* Simple bubble sort of the clusters */
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2 - i; j++) {
            if (em->mu[j] > em->mu[j+1]) {
                /* Swap K and K+1 */
                double t_mu = em->mu[j]; em->mu[j] = em->mu[j+1]; em->mu[j+1] = t_mu;
                double t_var = em->var[j]; em->var[j] = em->var[j+1]; em->var[j+1] = t_var;
                double t_pi = em->pi[j]; em->pi[j] = em->pi[j+1]; em->pi[j+1] = t_pi;
            }
        }
    }
}