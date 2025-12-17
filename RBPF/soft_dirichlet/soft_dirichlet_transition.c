/**
 * @file soft_dirichlet_transition.c
 * @brief Soft Dirichlet Transition Learning - Implementation
 *
 * Bayes-consistent transition matrix learning using ξ updates.
 * Single tuning knob: ESS_max
 */

#include "soft_dirichlet_transition.h"
#include <string.h>
#include <stdio.h>
#include <math.h>

/*═══════════════════════════════════════════════════════════════════════════════
 * HELPERS
 *═══════════════════════════════════════════════════════════════════════════════*/

static inline float maxf(float a, float b) { return a > b ? a : b; }
static inline float minf(float a, float b) { return a < b ? a : b; }

/**
 * Compute entropy of distribution (for diagnostics)
 */
static float compute_entropy(const float *p, int n)
{
    float H = 0.0f;
    for (int i = 0; i < n; i++) {
        if (p[i] > 1e-10f) {
            H -= p[i] * logf(p[i]);
        }
    }
    return H;
}

/*═══════════════════════════════════════════════════════════════════════════════
 * LIFECYCLE
 *═══════════════════════════════════════════════════════════════════════════════*/

void soft_dirichlet_init(SoftDirichlet *sd, int n_regimes, float ess_max)
{
    if (!sd) return;
    if (n_regimes < 1) n_regimes = 1;
    if (n_regimes > SOFT_DIRICHLET_MAX_REGIMES) n_regimes = SOFT_DIRICHLET_MAX_REGIMES;
    
    memset(sd, 0, sizeof(SoftDirichlet));
    
    sd->n_regimes = n_regimes;
    sd->ess_max = ess_max > 1.0f ? ess_max : 100.0f;
    sd->gamma = 1.0f;       /* No decay — ESS capping handles adaptivity */
    sd->kappa = 1.0f;       /* Full ξ contribution */
    sd->alpha_floor = 0.01f;
    sd->initialized = 0;
    
    /* Initialize with uniform prior, scaled to ESS_max */
    float alpha_init = sd->ess_max / (float)n_regimes;
    float prob_init = 1.0f / (float)n_regimes;
    
    for (int i = 0; i < n_regimes; i++) {
        for (int j = 0; j < n_regimes; j++) {
            sd->alpha[i][j] = alpha_init;
            sd->prob[i][j] = prob_init;
        }
        sd->prev_probs[i] = prob_init;
        sd->row_ess[i] = sd->ess_max;
    }
}

void soft_dirichlet_init_from_matrix(SoftDirichlet *sd, int n_regimes,
                                      const float *trans, float ess_max)
{
    if (!sd || !trans) return;
    
    /* First do basic init */
    soft_dirichlet_init(sd, n_regimes, ess_max);
    
    /* Set alpha proportional to trans, scaled to ESS_max per row */
    for (int i = 0; i < n_regimes; i++) {
        float row_sum = 0.0f;
        
        /* Copy probabilities */
        for (int j = 0; j < n_regimes; j++) {
            sd->prob[i][j] = trans[i * n_regimes + j];
            row_sum += sd->prob[i][j];
        }
        
        /* Normalize if needed */
        if (row_sum > 1e-10f && fabsf(row_sum - 1.0f) > 1e-6f) {
            for (int j = 0; j < n_regimes; j++) {
                sd->prob[i][j] /= row_sum;
            }
        }
        
        /* Set alpha = prob * ESS_max */
        for (int j = 0; j < n_regimes; j++) {
            sd->alpha[i][j] = sd->prob[i][j] * ess_max;
            if (sd->alpha[i][j] < sd->alpha_floor) {
                sd->alpha[i][j] = sd->alpha_floor;
            }
        }
        
        sd->row_ess[i] = ess_max;
    }
}

void soft_dirichlet_reset(SoftDirichlet *sd)
{
    if (!sd) return;
    
    int n = sd->n_regimes;
    float ess = sd->ess_max;
    
    soft_dirichlet_init(sd, n, ess);
}

/*═══════════════════════════════════════════════════════════════════════════════
 * CONFIGURATION
 *═══════════════════════════════════════════════════════════════════════════════*/

void soft_dirichlet_set_ess_max(SoftDirichlet *sd, float ess_max)
{
    if (!sd) return;
    sd->ess_max = ess_max > 1.0f ? ess_max : 1.0f;
}

void soft_dirichlet_set_params(SoftDirichlet *sd, float gamma, float kappa)
{
    if (!sd) return;
    sd->gamma = maxf(0.0f, minf(1.0f, gamma));
    sd->kappa = maxf(0.0f, kappa);
}

void soft_dirichlet_set_floor(SoftDirichlet *sd, float alpha_floor)
{
    if (!sd) return;
    sd->alpha_floor = maxf(1e-6f, alpha_floor);
}

/*═══════════════════════════════════════════════════════════════════════════════
 * CORE UPDATE
 *═══════════════════════════════════════════════════════════════════════════════*/

void soft_dirichlet_update(SoftDirichlet *sd,
                           const float *regime_probs,
                           const float *regime_liks)
{
    if (!sd || !regime_probs || !regime_liks) return;
    
    const int R = sd->n_regimes;
    
    /* ═══════════════════════════════════════════════════════════════════════
     * FIRST TICK: Just store regime probs for next iteration
     * ═══════════════════════════════════════════════════════════════════════*/
    if (!sd->initialized) {
        for (int i = 0; i < R; i++) {
            sd->prev_probs[i] = regime_probs[i];
        }
        sd->initialized = 1;
        return;
    }
    
    /* ═══════════════════════════════════════════════════════════════════════
     * STEP 1: Compute ξ_t(i,j) = p_{t-1}(i) × A[i][j] × ℓ_t(j)
     *
     * This is the Bayes-consistent joint posterior over transitions.
     * ═══════════════════════════════════════════════════════════════════════*/
    float xi[SOFT_DIRICHLET_MAX_REGIMES][SOFT_DIRICHLET_MAX_REGIMES];
    float Z = 0.0f;
    
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < R; j++) {
            xi[i][j] = sd->prev_probs[i] * sd->prob[i][j] * regime_liks[j];
            Z += xi[i][j];
        }
    }
    
    /* Normalize ξ */
    if (Z > 1e-30f) {
        float inv_Z = 1.0f / Z;
        for (int i = 0; i < R; i++) {
            for (int j = 0; j < R; j++) {
                xi[i][j] *= inv_Z;
            }
        }
    } else {
        /* Fallback: uniform ξ (shouldn't happen with valid inputs) */
        float uniform = 1.0f / (float)(R * R);
        for (int i = 0; i < R; i++) {
            for (int j = 0; j < R; j++) {
                xi[i][j] = uniform;
            }
        }
    }
    
    /* Compute ξ entropy for diagnostics */
    float xi_flat[SOFT_DIRICHLET_MAX_REGIMES * SOFT_DIRICHLET_MAX_REGIMES];
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < R; j++) {
            xi_flat[i * R + j] = xi[i][j];
        }
    }
    sd->last_xi_entropy = compute_entropy(xi_flat, R * R);
    
    /* ═══════════════════════════════════════════════════════════════════════
     * STEP 2: Update α with discount + soft counts
     *
     *   α[i][j] ← γ · α[i][j] + κ · ξ_t(i,j)
     * ═══════════════════════════════════════════════════════════════════════*/
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < R; j++) {
            sd->alpha[i][j] = sd->gamma * sd->alpha[i][j] + sd->kappa * xi[i][j];
            
            /* Enforce floor */
            if (sd->alpha[i][j] < sd->alpha_floor) {
                sd->alpha[i][j] = sd->alpha_floor;
            }
        }
    }
    
    /* ═══════════════════════════════════════════════════════════════════════
     * STEP 3: Row ESS capping
     *
     *   If ESS_i > ESS_max, scale row down.
     *   This replaces "tuning γ" with an intuitive memory control.
     * ═══════════════════════════════════════════════════════════════════════*/
    for (int i = 0; i < R; i++) {
        float row_ess = 0.0f;
        for (int j = 0; j < R; j++) {
            row_ess += sd->alpha[i][j];
        }
        sd->row_ess[i] = row_ess;
        
        if (row_ess > sd->ess_max) {
            float scale = sd->ess_max / row_ess;
            for (int j = 0; j < R; j++) {
                sd->alpha[i][j] *= scale;
                /* Re-enforce floor after scaling */
                if (sd->alpha[i][j] < sd->alpha_floor) {
                    sd->alpha[i][j] = sd->alpha_floor;
                }
            }
            sd->row_ess[i] = sd->ess_max;
        }
    }
    
    /* ═══════════════════════════════════════════════════════════════════════
     * STEP 4: Rebuild probabilities (posterior mean)
     *
     *   E[A[i][j]] = α[i][j] / Σ_k α[i][k]
     * ═══════════════════════════════════════════════════════════════════════*/
    for (int i = 0; i < R; i++) {
        float row_sum = 0.0f;
        for (int j = 0; j < R; j++) {
            row_sum += sd->alpha[i][j];
        }
        
        float inv_sum = (row_sum > 1e-10f) ? 1.0f / row_sum : 1.0f;
        for (int j = 0; j < R; j++) {
            sd->prob[i][j] = sd->alpha[i][j] * inv_sum;
        }
    }
    
    /* ═══════════════════════════════════════════════════════════════════════
     * Store current regime probs for next tick
     * ═══════════════════════════════════════════════════════════════════════*/
    for (int i = 0; i < R; i++) {
        sd->prev_probs[i] = regime_probs[i];
    }
    
    sd->total_updates++;
}

void soft_dirichlet_update_log(SoftDirichlet *sd,
                                const float *regime_probs,
                                const float *log_regime_liks)
{
    if (!sd || !regime_probs || !log_regime_liks) return;
    
    const int R = sd->n_regimes;
    
    /* Convert log-likelihoods to likelihoods with numerical stability */
    float liks[SOFT_DIRICHLET_MAX_REGIMES];
    
    /* Find max for log-sum-exp style normalization */
    float max_ll = log_regime_liks[0];
    for (int j = 1; j < R; j++) {
        if (log_regime_liks[j] > max_ll) max_ll = log_regime_liks[j];
    }
    
    /* Convert with shift */
    for (int j = 0; j < R; j++) {
        liks[j] = expf(log_regime_liks[j] - max_ll);
    }
    
    /* Call main update */
    soft_dirichlet_update(sd, regime_probs, liks);
}

/*═══════════════════════════════════════════════════════════════════════════════
 * QUERIES
 *═══════════════════════════════════════════════════════════════════════════════*/

float soft_dirichlet_prob(const SoftDirichlet *sd, int from, int to)
{
    if (!sd) return 0.0f;
    if (from < 0 || from >= sd->n_regimes) return 0.0f;
    if (to < 0 || to >= sd->n_regimes) return 0.0f;
    
    return sd->prob[from][to];
}

void soft_dirichlet_get_matrix(const SoftDirichlet *sd, float *trans)
{
    if (!sd || !trans) return;
    
    const int R = sd->n_regimes;
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < R; j++) {
            trans[i * R + j] = sd->prob[i][j];
        }
    }
}

void soft_dirichlet_get_row(const SoftDirichlet *sd, int from, float *row)
{
    if (!sd || !row) return;
    if (from < 0 || from >= sd->n_regimes) return;
    
    for (int j = 0; j < sd->n_regimes; j++) {
        row[j] = sd->prob[from][j];
    }
}

float soft_dirichlet_stickiness(const SoftDirichlet *sd, int regime)
{
    if (!sd) return 0.0f;
    if (regime < 0 || regime >= sd->n_regimes) return 0.0f;
    
    return sd->prob[regime][regime];
}

SoftDirichletStats soft_dirichlet_stats(const SoftDirichlet *sd)
{
    SoftDirichletStats stats = {0};
    if (!sd) return stats;
    
    const int R = sd->n_regimes;
    
    /* Compute average stickiness */
    float sum_sticky = 0.0f;
    for (int i = 0; i < R; i++) {
        sum_sticky += sd->prob[i][i];
    }
    stats.avg_stickiness = sum_sticky / (float)R;
    
    /* Compute ESS statistics */
    float sum_ess = 0.0f;
    stats.min_row_ess = sd->row_ess[0];
    stats.max_row_ess = sd->row_ess[0];
    
    for (int i = 0; i < R; i++) {
        sum_ess += sd->row_ess[i];
        if (sd->row_ess[i] < stats.min_row_ess) stats.min_row_ess = sd->row_ess[i];
        if (sd->row_ess[i] > stats.max_row_ess) stats.max_row_ess = sd->row_ess[i];
    }
    stats.avg_row_ess = sum_ess / (float)R;
    
    stats.xi_entropy = sd->last_xi_entropy;
    stats.total_updates = sd->total_updates;
    
    return stats;
}

/*═══════════════════════════════════════════════════════════════════════════════
 * RBPF INTEGRATION
 *═══════════════════════════════════════════════════════════════════════════════*/

void soft_dirichlet_rebuild_lut(const SoftDirichlet *sd,
                                 uint8_t trans_lut[][1024],
                                 int n_regimes)
{
    if (!sd || !trans_lut) return;
    if (n_regimes > sd->n_regimes) n_regimes = sd->n_regimes;
    
    for (int r = 0; r < n_regimes; r++) {
        /* Build cumulative distribution */
        float cumsum[SOFT_DIRICHLET_MAX_REGIMES];
        cumsum[0] = sd->prob[r][0];
        for (int j = 1; j < n_regimes; j++) {
            cumsum[j] = cumsum[j - 1] + sd->prob[r][j];
        }
        
        /* Fill LUT: for each u ∈ [0, 1024), find smallest j where cumsum[j] > u/1024 */
        for (int i = 0; i < 1024; i++) {
            float u = (float)i / 1024.0f;
            int next = n_regimes - 1;
            for (int j = 0; j < n_regimes - 1; j++) {
                if (u < cumsum[j]) {
                    next = j;
                    break;
                }
            }
            trans_lut[r][i] = (uint8_t)next;
        }
    }
}

void soft_dirichlet_export_matrix(const SoftDirichlet *sd,
                                   float *trans, int n_regimes)
{
    if (!sd || !trans) return;
    if (n_regimes > sd->n_regimes) n_regimes = sd->n_regimes;
    
    for (int i = 0; i < n_regimes; i++) {
        for (int j = 0; j < n_regimes; j++) {
            trans[i * n_regimes + j] = sd->prob[i][j];
        }
    }
}

/*═══════════════════════════════════════════════════════════════════════════════
 * DIAGNOSTICS
 *═══════════════════════════════════════════════════════════════════════════════*/

void soft_dirichlet_print(const SoftDirichlet *sd)
{
    if (!sd) return;
    
    SoftDirichletStats stats = soft_dirichlet_stats(sd);
    
    printf("\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  Soft Dirichlet Transition Learner\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  Regimes:        %d\n", sd->n_regimes);
    printf("  ESS max:        %.1f\n", sd->ess_max);
    printf("  γ (discount):   %.4f\n", sd->gamma);
    printf("  κ (learn rate): %.4f\n", sd->kappa);
    printf("  α floor:        %.4f\n", sd->alpha_floor);
    printf("  Updates:        %lu\n", (unsigned long)sd->total_updates);
    printf("───────────────────────────────────────────────────────────────\n");
    printf("  Avg stickiness: %.1f%%\n", stats.avg_stickiness * 100.0f);
    printf("  Row ESS:        %.1f avg, [%.1f, %.1f] range\n",
           stats.avg_row_ess, stats.min_row_ess, stats.max_row_ess);
    printf("  ξ entropy:      %.3f (max=%.3f)\n", 
           stats.xi_entropy, logf((float)(sd->n_regimes * sd->n_regimes)));
    printf("═══════════════════════════════════════════════════════════════\n");
}

void soft_dirichlet_print_matrix(const SoftDirichlet *sd)
{
    if (!sd) return;
    
    const int R = sd->n_regimes;
    
    printf("\n  Transition Matrix E[A]:\n");
    printf("       ");
    for (int j = 0; j < R; j++) {
        printf("    R%d   ", j);
    }
    printf("\n");
    
    for (int i = 0; i < R; i++) {
        printf("  R%d: ", i);
        for (int j = 0; j < R; j++) {
            float p = sd->prob[i][j];
            if (i == j) {
                printf(" [%5.1f%%]", p * 100.0f);
            } else {
                printf("  %5.1f%% ", p * 100.0f);
            }
        }
        printf("  (ESS=%.0f)\n", sd->row_ess[i]);
    }
}

void soft_dirichlet_print_alpha(const SoftDirichlet *sd)
{
    if (!sd) return;
    
    const int R = sd->n_regimes;
    
    printf("\n  Pseudo-counts α:\n");
    printf("       ");
    for (int j = 0; j < R; j++) {
        printf("    R%d   ", j);
    }
    printf("\n");
    
    for (int i = 0; i < R; i++) {
        printf("  R%d: ", i);
        for (int j = 0; j < R; j++) {
            printf(" %7.2f ", sd->alpha[i][j]);
        }
        printf("\n");
    }
}
