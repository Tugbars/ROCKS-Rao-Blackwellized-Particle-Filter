/**
 * @file mmpf_entropy.c
 * @brief Thermodynamic Stability Detection
 */
#include <mkl.h>
#include <math.h>
#include "mmpf_internal.h"

/* * Calculates Shannon Entropy of the global particle swarm
 * H = - Sum(w_i * log(w_i))
 * Normalized H = H / log(N) (Range 0.0 to 1.0)
 */
double mmpf_calculate_entropy(MMPF_ROCKS *mmpf)
{
    const int n_total = mmpf->n_particles * MMPF_N_MODELS;
    /* We need a flattened view of all weights across all models */
    /* Assuming we can access weights linearly or gather them */
    
    /* Using stack buffer for simplicity, use heap/scratch in production */
    double weights[1024]; 
    double log_weights[1024];
    double h_val = 0.0;
    int idx = 0;
    int k, i;

    /* Gather normalized weights */
    double sum_w = 0.0;
    
    /* 1. Gather global weights */
    for (k = 0; k < MMPF_N_MODELS; k++) {
        RBPF_KSC *rbpf = mmpf->ext[k]->rbpf;
        double model_w = mmpf->weights[k];
        
        for (i = 0; i < rbpf->n_particles; i++) {
            /* Global weight = Model_Prob * Particle_Weight */
            /* Note: rbpf->w_norm is normalized within the model */
            weights[idx] = model_w * rbpf->w_norm[i];
            sum_w += weights[idx];
            idx++;
        }
    }

    /* 2. Compute Entropy using MKL */
    /* y = ln(x) */
    vdLn(idx, weights, log_weights); // Vectorized Log

    /* H = -Sum(w * ln(w)) */
    for (i = 0; i < idx; i++) {
        /* Handle 0.0 * -inf = 0.0 */
        if (weights[i] > 1e-12) {
            h_val -= weights[i] * log_weights[i];
        }
    }

    /* Normalize by log(N) so max entropy is 1.0 */
    return h_val / log(idx);
}

/* * Updates the stability metric and decides whether to unlock
 * Returns: 1 if stable (Unlock), 0 if unstable (Keep Lock)
 */
int mmpf_check_stability(MMPF_ROCKS *mmpf)
{
    double h_current = mmpf_calculate_entropy(mmpf);
    double h_prev = mmpf->current_entropy;
    
    /* Calculate absolute change */
    double delta = fabs(h_current - h_prev);
    
    /* Update EMA (Fast alpha = 0.3 for quick reaction) */
    mmpf->entropy_change_ema = 0.3 * delta + 0.7 * mmpf->entropy_change_ema;
    
    mmpf->current_entropy = h_current;
    mmpf->ticks_since_shock++;

    /* Safety Gates */
    if (mmpf->ticks_since_shock < mmpf->min_shock_duration) return 0;
    if (mmpf->ticks_since_shock > mmpf->max_shock_duration) return 1; /* Forced unlock */

    /* Thermodynamic Condition: Change in Information is negligible */
    if (mmpf->entropy_change_ema < mmpf->stability_threshold) {
        return 1; /* STABLE */
    }
    
    return 0; /* UNSTABLE */
}