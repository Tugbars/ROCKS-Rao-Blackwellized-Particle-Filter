/**
 * @file rbpf_mh_jitter.c
 * @brief Metropolis-Hastings Jittering for Particle Diversity
 *
 * Turns blind Silverman noise into informed exploration.
 */

#include "rbpf_mh_jitter.h"
#include "rbpf_sprt.h"  /* For sprt_logchisq_loglik */
#include <mkl.h>
#include <mkl_vsl.h>
#include <stdio.h>
#include <math.h>

void mh_jitter_init(MH_Jitter_Config *cfg)
{
    cfg->proposal_scale = 1.0f;
    cfg->max_attempts = 3;
    cfg->enabled = true;
    
    cfg->total_proposals = 0;
    cfg->total_accepts = 0;
    cfg->total_boundary_rejects = 0;
    cfg->total_likelihood_rejects = 0;
    cfg->avg_accept_ratio = 0.5f;
}

void mh_jitter_reset_stats(MH_Jitter_Config *cfg)
{
    cfg->total_proposals = 0;
    cfg->total_accepts = 0;
    cfg->total_boundary_rejects = 0;
    cfg->total_likelihood_rejects = 0;
    cfg->avg_accept_ratio = 0.5f;
}

/**
 * @brief Check if value is within regime bounds
 */
static inline bool in_regime(float h, int regime, const float *bounds, int n_regimes)
{
    if (regime < 0 || regime >= n_regimes) return false;
    return (h >= bounds[regime] && h < bounds[regime + 1]);
}

void mh_jitter_apply(MH_Jitter_Config *cfg,
                     float *mu,
                     const int *regime,
                     int n,
                     float y_obs,
                     float silverman_h,
                     const float *regime_bounds,
                     int n_regimes,
                     void *rng_stream)
{
    if (!cfg->enabled) return;
    if (silverman_h < 1e-10f) return;
    
    VSLStreamStatePtr stream = (VSLStreamStatePtr)rng_stream;
    
    /* Proposal standard deviation */
    float proposal_std = cfg->proposal_scale * silverman_h;
    
    double y_obs_d = (double)y_obs;
    
    int accepts = 0;
    int boundary_rejects = 0;
    int likelihood_rejects = 0;
    
    for (int i = 0; i < n; i++) {
        float h_old = mu[i];
        int r = regime[i];
        double log_lik_old = sprt_logchisq_loglik(y_obs_d, (double)h_old);
        
        /* ─────────────────────────────────────────────────────────────────
         * MULTI-ATTEMPT REJECTION LOOP
         *
         * Try up to max_attempts to find a valid move.
         * This increases diversity without wasting particles.
         * ───────────────────────────────────────────────────────────────── */
        bool accepted = false;
        
        for (int attempt = 0; attempt < cfg->max_attempts && !accepted; attempt++) {
            /* Generate fresh proposal for each attempt */
            float proposal_offset;
            vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, 1, &proposal_offset, 0.0f, proposal_std);
            float h_new = h_old + proposal_offset;
            
            cfg->total_proposals++;
            
            /* ─────────────────────────────────────────────────────────────
             * BOUNDARY CHECK: Stay within regime
             *
             * If proposal crosses regime boundary, reject immediately.
             * This preserves Storvik statistics validity.
             * ───────────────────────────────────────────────────────────── */
            if (!in_regime(h_new, r, regime_bounds, n_regimes)) {
                boundary_rejects++;
                continue;  /* Try again */
            }
            
            /* ─────────────────────────────────────────────────────────────
             * LIKELIHOOD RATIO: Accept with probability min(1, L_new/L_old)
             *
             * Using OCSN log-likelihood for observation model.
             * Log acceptance ratio = log_lik(h_new) - log_lik(h_old)
             * ───────────────────────────────────────────────────────────── */
            double log_lik_new = sprt_logchisq_loglik(y_obs_d, (double)h_new);
            double log_alpha = log_lik_new - log_lik_old;
            
            /* Generate uniform for this attempt */
            float u;
            vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, 1, &u, 0.0f, 1.0f);
            
            /* Accept if log(U) < log_alpha, i.e., U < exp(log_alpha) */
            if (log((double)u) < log_alpha) {
                mu[i] = h_new;
                accepts++;
                accepted = true;
            } else {
                likelihood_rejects++;
                /* Continue to next attempt */
            }
        }
    }
    
    /* Update diagnostics */
    cfg->total_accepts += accepts;
    cfg->total_boundary_rejects += boundary_rejects;
    cfg->total_likelihood_rejects += likelihood_rejects;
    
    /* Exponential moving average of acceptance rate */
    float batch_rate = (n > 0) ? (float)accepts / (float)n : 0.0f;
    cfg->avg_accept_ratio = 0.9f * cfg->avg_accept_ratio + 0.1f * batch_rate;
    
    /* ═══════════════════════════════════════════════════════════════════════
     * ADAPTIVE PROPOSAL SCALING
     *
     * Target acceptance rate: ~50% for 1D state (Roberts et al. 1997)
     *   - Too high (>60%): proposals too timid, not exploring
     *   - Too low (<40%): proposals too aggressive, wasting computation
     *
     * Adjust every call to converge toward optimal.
     * ═══════════════════════════════════════════════════════════════════════ */
    if (cfg->avg_accept_ratio > 0.60f) {
        cfg->proposal_scale *= 1.05f;  /* Too easy → be bolder */
    } else if (cfg->avg_accept_ratio < 0.40f) {
        cfg->proposal_scale *= 0.95f;  /* Too hard → be more conservative */
    }
    
    /* Clamp to reasonable range */
    if (cfg->proposal_scale < 0.1f) cfg->proposal_scale = 0.1f;
    if (cfg->proposal_scale > 5.0f) cfg->proposal_scale = 5.0f;
}

void mh_jitter_print_stats(const MH_Jitter_Config *cfg)
{
    printf("\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  MH Jittering Statistics\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  Enabled:                %s\n", cfg->enabled ? "YES" : "NO");
    printf("  Proposal scale:         %.2f × Silverman\n", cfg->proposal_scale);
    printf("  Max attempts:           %d\n", cfg->max_attempts);
    printf("───────────────────────────────────────────────────────────────\n");
    printf("  Total proposals:        %d\n", cfg->total_proposals);
    printf("  Accepts:                %d (%.1f%%)\n", 
           cfg->total_accepts, 
           cfg->total_proposals > 0 ? 100.0f * cfg->total_accepts / cfg->total_proposals : 0.0f);
    printf("  Boundary rejects:       %d (%.1f%%)\n",
           cfg->total_boundary_rejects,
           cfg->total_proposals > 0 ? 100.0f * cfg->total_boundary_rejects / cfg->total_proposals : 0.0f);
    printf("  Likelihood rejects:     %d (%.1f%%)\n",
           cfg->total_likelihood_rejects,
           cfg->total_proposals > 0 ? 100.0f * cfg->total_likelihood_rejects / cfg->total_proposals : 0.0f);
    printf("  Avg acceptance rate:    %.1f%%\n", 100.0f * cfg->avg_accept_ratio);
    printf("═══════════════════════════════════════════════════════════════\n");
}

float mh_jitter_acceptance_rate(const MH_Jitter_Config *cfg)
{
    if (cfg->total_proposals == 0) return 0.0f;
    return (float)cfg->total_accepts / (float)cfg->total_proposals;
}
