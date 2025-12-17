/**
 * @file rbpf_dirichlet_transition.h
 * @brief Online Transition Matrix Learning via Discounted Dirichlet
 *
 * The transition matrix P[i][j] = P(regime_t = j | regime_{t-1} = i) is learned
 * online using a Dirichlet-Multinomial conjugate model with exponential forgetting.
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * THEORY
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * Each row of the transition matrix is modeled as:
 *
 *   P[i,:] ~ Dirichlet(α[i,0], α[i,1], ..., α[i,K-1])
 *
 * The posterior mean is:
 *
 *   E[P[i,j]] = α[i,j] / Σⱼ α[i,j]
 *
 * Standard Dirichlet update:
 *   α[i,j] ← α[i,j] + 𝟙[transition from i to j]
 *
 * Problem: As counts grow, the model freezes and stops adapting.
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * DISCOUNTED DIRICHLET (Exponential Forgetting)
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * Before each update, decay all counts:
 *
 *   α[i,j] ← γ × α[i,j]     for all j
 *   α[i,j] ← α[i,j] + 𝟙[transition to j]
 *
 * Where γ ∈ (0, 1) is the forgetting factor.
 *
 * Effective memory window: ~1/(1-γ) ticks
 *   γ = 0.999 → ~1000 tick memory
 *   γ = 0.995 → ~200 tick memory
 *   γ = 0.990 → ~100 tick memory
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * GEOMETRY-AWARE INITIALIZATION
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * Instead of uniform prior, we initialize based on regime distance:
 *
 *   α[i,j] ∝ exp(-|μ_i - μ_j| / scale)   for i ≠ j
 *   α[i,i] = stickiness                   for self-transition
 *
 * This encodes: "Nearby regimes are more likely transition targets"
 *
 * Reference:
 *   - Bishop (2006), Pattern Recognition and Machine Learning, §2.4
 *   - West & Harrison (1997), Bayesian Forecasting and Dynamic Models
 */

#ifndef RBPF_DIRICHLET_TRANSITION_H
#define RBPF_DIRICHLET_TRANSITION_H

#include <math.h>
#include <string.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef RBPF_MAX_REGIMES
#define RBPF_MAX_REGIMES 8
#endif

/*═══════════════════════════════════════════════════════════════════════════════
 * RECOMMENDED WORKFLOW
 *═══════════════════════════════════════════════════════════════════════════════
 *
 * 1. Initialize once with geometry-aware prior:
 *
 *    float mu_vol[4] = {-4.5f, -3.5f, -2.5f, -1.5f};
 *    dirichlet_transition_init_geometric(&dt, 4, mu_vol,
 *        30.0f,   // stickiness (moderate)
 *        1.0f,    // distance_scale
 *        0.999f); // gamma (slow forgetting)
 *
 * 2. Only update on SPRT-confirmed transitions:
 *
 *    if (sprt_decision == SPRT_ACCEPT_H1 && new_regime != old_regime) {
 *        dirichlet_transition_update(&dt, old_regime, new_regime);
 *        dirichlet_transition_build_lut(&dt, rbpf->trans_lut, 4);
 *    }
 *
 * 3. Do NOT call _stay() every tick. The prior handles stickiness.
 *
 * 4. Periodically log statistics for diagnostics:
 *
 *    DirichletTransitionStats stats = dirichlet_transition_stats(&dt);
 *    printf("Avg stickiness: %.1f%%, transitions: %d\n",
 *           stats.avg_stickiness * 100, stats.total_transitions);
 *
 *═══════════════════════════════════════════════════════════════════════════════*/

/*═══════════════════════════════════════════════════════════════════════════════
 * TYPES
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Discounted Dirichlet prior for transition matrix
 */
typedef struct {
    /* Pseudo-counts (Dirichlet parameters) */
    float alpha[RBPF_MAX_REGIMES][RBPF_MAX_REGIMES];
    
    /* Current MAP estimate of transition probabilities */
    float prob[RBPF_MAX_REGIMES][RBPF_MAX_REGIMES];
    
    /* Configuration */
    float gamma;           /**< Forgetting factor ∈ (0,1), e.g., 0.999 */
    float alpha_floor;     /**< Minimum pseudo-count to prevent P=0 */
    float stickiness;      /**< Prior strength for self-transitions */
    float distance_scale;  /**< Scale for distance-based prior */
    
    /* State */
    int n_regimes;
    int total_transitions; /**< Total observed transitions (for diagnostics) */
    
    /* Per-regime transition counts (for diagnostics) */
    int observed[RBPF_MAX_REGIMES][RBPF_MAX_REGIMES];
} DirichletTransition;

/*═══════════════════════════════════════════════════════════════════════════════
 * INITIALIZATION
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Initialize with uniform prior
 *
 * @param dt           Dirichlet transition struct
 * @param n_regimes    Number of regimes
 * @param stickiness   Prior pseudo-count for self-transitions (e.g., 10.0)
 * @param off_diag     Prior pseudo-count for off-diagonal (e.g., 1.0)
 * @param gamma        Forgetting factor (e.g., 0.999)
 */
static inline void dirichlet_transition_init_uniform(
    DirichletTransition *dt,
    int n_regimes,
    float stickiness,
    float off_diag,
    float gamma)
{
    memset(dt, 0, sizeof(DirichletTransition));
    
    dt->n_regimes = n_regimes;
    dt->gamma = gamma;
    dt->alpha_floor = 0.01f;
    dt->stickiness = stickiness;
    dt->distance_scale = 1.0f;
    dt->total_transitions = 0;
    
    for (int i = 0; i < n_regimes; i++) {
        for (int j = 0; j < n_regimes; j++) {
            if (i == j) {
                dt->alpha[i][j] = stickiness;
            } else {
                dt->alpha[i][j] = off_diag;
            }
            dt->observed[i][j] = 0;
        }
    }
    
    /* Compute initial probabilities */
    for (int i = 0; i < n_regimes; i++) {
        float row_sum = 0.0f;
        for (int j = 0; j < n_regimes; j++) {
            row_sum += dt->alpha[i][j];
        }
        for (int j = 0; j < n_regimes; j++) {
            dt->prob[i][j] = dt->alpha[i][j] / row_sum;
        }
    }
}

/**
 * @brief Initialize with geometry-aware prior based on regime distances
 *
 * PARAMETER SELECTION GUIDE:
 * ┌────────────────┬─────────────────────────────────────────────────────────┐
 * │ Parameter      │ Effect                                                  │
 * ├────────────────┼─────────────────────────────────────────────────────────┤
 * │ stickiness     │ Higher = harder to leave current regime                 │
 * │   10-20        │ Reactive (regime changes frequently)                    │
 * │   30-50        │ Moderate (balanced)                                     │
 * │   100+         │ Very sticky (regime rarely changes)                     │
 * ├────────────────┼─────────────────────────────────────────────────────────┤
 * │ distance_scale │ How much geometry matters                               │
 * │   0.5          │ Strong: only adjacent regimes likely                    │
 * │   1.0          │ Moderate: distance matters but not dominant             │
 * │   2.0          │ Weak: nearly uniform off-diagonal                       │
 * ├────────────────┼─────────────────────────────────────────────────────────┤
 * │ gamma          │ How fast to forget old data                             │
 * │   0.990        │ ~100 tick memory (very reactive)                        │
 * │   0.995        │ ~200 tick memory (reactive)                             │
 * │   0.999        │ ~1000 tick memory (stable, recommended)                 │
 * │   0.9995       │ ~2000 tick memory (very stable)                         │
 * └────────────────┴─────────────────────────────────────────────────────────┘
 *
 * RECOMMENDED DEFAULTS:
 *   stickiness = 30.0f, distance_scale = 1.0f, gamma = 0.999f
 *
 * @param dt             Dirichlet transition struct
 * @param n_regimes      Number of regimes
 * @param mu_vol         Array of regime log-vol means [n_regimes]
 * @param stickiness     Prior pseudo-count for self-transitions
 * @param distance_scale Scale for exponential distance decay
 * @param gamma          Forgetting factor
 */
static inline void dirichlet_transition_init_geometric(
    DirichletTransition *dt,
    int n_regimes,
    const float *mu_vol,
    float stickiness,
    float distance_scale,
    float gamma)
{
    memset(dt, 0, sizeof(DirichletTransition));
    
    dt->n_regimes = n_regimes;
    dt->gamma = gamma;
    dt->alpha_floor = 0.01f;
    dt->stickiness = stickiness;
    dt->distance_scale = distance_scale;
    dt->total_transitions = 0;
    
    for (int i = 0; i < n_regimes; i++) {
        float off_diag_sum = 0.0f;
        
        /* First pass: compute distance-based weights */
        for (int j = 0; j < n_regimes; j++) {
            if (i == j) {
                dt->alpha[i][j] = stickiness;
            } else {
                /* Exponential decay with distance */
                float dist = fabsf(mu_vol[i] - mu_vol[j]);
                float weight = expf(-dist / distance_scale);
                dt->alpha[i][j] = weight;
                off_diag_sum += weight;
            }
            dt->observed[i][j] = 0;
        }
        
        /* Normalize off-diagonal to sum to (1 - stickiness ratio) */
        /* This ensures the prior encodes the desired stickiness */
        float target_off_diag_total = stickiness * 0.1f; /* 10% of stickiness goes to transitions */
        if (off_diag_sum > 0.0f) {
            for (int j = 0; j < n_regimes; j++) {
                if (i != j) {
                    dt->alpha[i][j] *= target_off_diag_total / off_diag_sum;
                    if (dt->alpha[i][j] < dt->alpha_floor) {
                        dt->alpha[i][j] = dt->alpha_floor;
                    }
                }
            }
        }
    }
    
    /* Compute initial probabilities */
    for (int i = 0; i < n_regimes; i++) {
        float row_sum = 0.0f;
        for (int j = 0; j < n_regimes; j++) {
            row_sum += dt->alpha[i][j];
        }
        for (int j = 0; j < n_regimes; j++) {
            dt->prob[i][j] = dt->alpha[i][j] / row_sum;
        }
    }
}

/*═══════════════════════════════════════════════════════════════════════════════
 * ONLINE UPDATE
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Update after observing a transition (with decay)
 *
 * @param dt    Dirichlet transition struct
 * @param from  Source regime
 * @param to    Destination regime
 */
static inline void dirichlet_transition_update(
    DirichletTransition *dt,
    int from,
    int to)
{
    int n = dt->n_regimes;
    
    /* Decay all counts in this row */
    for (int j = 0; j < n; j++) {
        dt->alpha[from][j] *= dt->gamma;
        if (dt->alpha[from][j] < dt->alpha_floor) {
            dt->alpha[from][j] = dt->alpha_floor;
        }
    }
    
    /* Increment observed transition */
    dt->alpha[from][to] += 1.0f;
    dt->observed[from][to]++;
    dt->total_transitions++;
    
    /* Recompute probabilities for this row */
    float row_sum = 0.0f;
    for (int j = 0; j < n; j++) {
        row_sum += dt->alpha[from][j];
    }
    for (int j = 0; j < n; j++) {
        dt->prob[from][j] = dt->alpha[from][j] / row_sum;
    }
}

/**
 * @brief Update for a "stay" event (no regime change)
 *
 * ╔═══════════════════════════════════════════════════════════════════════════╗
 * ║  ⚠️  WARNING: USE WITH EXTREME CAUTION                                    ║
 * ╠═══════════════════════════════════════════════════════════════════════════╣
 * ║  Calling this every tick will cause stickiness to EXPLODE.                ║
 * ║  The model will freeze and never transition.                              ║
 * ║                                                                           ║
 * ║  RECOMMENDED: Do NOT use this function.                                   ║
 * ║  Instead, only call dirichlet_transition_update() when SPRT               ║
 * ║  confirms an actual regime transition. The prior already                  ║
 * ║  encodes stickiness — you don't need to reinforce stays.                  ║
 * ║                                                                           ║
 * ║  If you MUST use this, use weight ≤ 0.001                                 ║
 * ╚═══════════════════════════════════════════════════════════════════════════╝
 *
 * @param dt      Dirichlet transition struct
 * @param regime  Current regime (self-transition)
 * @param weight  How much to count this stay (MUST be tiny, e.g., 0.001)
 */
static inline void dirichlet_transition_stay_DANGEROUS(
    DirichletTransition *dt,
    int regime,
    float weight)
{
    int n = dt->n_regimes;
    
    /* Decay all counts in this row */
    for (int j = 0; j < n; j++) {
        dt->alpha[regime][j] *= dt->gamma;
        if (dt->alpha[regime][j] < dt->alpha_floor) {
            dt->alpha[regime][j] = dt->alpha_floor;
        }
    }
    
    /* Increment self-transition with given weight */
    dt->alpha[regime][regime] += weight;
    
    /* Recompute probabilities for this row */
    float row_sum = 0.0f;
    for (int j = 0; j < n; j++) {
        row_sum += dt->alpha[regime][j];
    }
    for (int j = 0; j < n; j++) {
        dt->prob[regime][j] = dt->alpha[regime][j] / row_sum;
    }
}

/*═══════════════════════════════════════════════════════════════════════════════
 * ACCESSORS
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Get transition probability P(to | from)
 */
static inline float dirichlet_transition_prob(
    const DirichletTransition *dt,
    int from,
    int to)
{
    return dt->prob[from][to];
}

/**
 * @brief Get entire transition matrix (row-major)
 */
static inline void dirichlet_transition_get_matrix(
    const DirichletTransition *dt,
    float *matrix_out)
{
    for (int i = 0; i < dt->n_regimes; i++) {
        for (int j = 0; j < dt->n_regimes; j++) {
            matrix_out[i * dt->n_regimes + j] = dt->prob[i][j];
        }
    }
}

/**
 * @brief Get effective sample size for a row (sum of alphas)
 *
 * Higher = more confident estimate
 */
static inline float dirichlet_transition_row_ess(
    const DirichletTransition *dt,
    int from)
{
    float sum = 0.0f;
    for (int j = 0; j < dt->n_regimes; j++) {
        sum += dt->alpha[from][j];
    }
    return sum;
}

/**
 * @brief Get current stickiness (self-transition probability) per regime
 */
static inline float dirichlet_transition_stickiness(
    const DirichletTransition *dt,
    int regime)
{
    return dt->prob[regime][regime];
}

/*═══════════════════════════════════════════════════════════════════════════════
 * INTEGRATION WITH RBPF
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Rebuild RBPF transition LUT from Dirichlet posterior
 *
 * Call this after updates to sync the RBPF's internal LUT.
 *
 * @param dt          Dirichlet transition struct
 * @param trans_lut   RBPF's transition LUT (cumulative probabilities)
 * @param n_regimes   Number of regimes
 *
 * The LUT format is: trans_lut[i * n_regimes + j] = cumsum P(i → 0..j)
 */
static inline void dirichlet_transition_build_lut(
    const DirichletTransition *dt,
    float *trans_lut,
    int n_regimes)
{
    for (int i = 0; i < n_regimes; i++) {
        float cumsum = 0.0f;
        for (int j = 0; j < n_regimes; j++) {
            cumsum += dt->prob[i][j];
            trans_lut[i * n_regimes + j] = cumsum;
        }
        /* Ensure last element is exactly 1.0 */
        trans_lut[i * n_regimes + (n_regimes - 1)] = 1.0f;
    }
}

/*═══════════════════════════════════════════════════════════════════════════════
 * DIAGNOSTICS
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Print current state for debugging
 */
static inline void dirichlet_transition_print(const DirichletTransition *dt)
{
    int n = dt->n_regimes;
    
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  Discounted Dirichlet Transition Prior\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  Regimes: %d, γ=%.4f, Total transitions: %d\n\n",
           n, dt->gamma, dt->total_transitions);
    
    printf("  Pseudo-counts (α):\n");
    printf("       ");
    for (int j = 0; j < n; j++) printf("    R%d   ", j);
    printf("\n");
    
    for (int i = 0; i < n; i++) {
        printf("  R%d: ", i);
        for (int j = 0; j < n; j++) {
            printf(" %7.2f", dt->alpha[i][j]);
        }
        printf("  (ESS=%.1f)\n", dirichlet_transition_row_ess(dt, i));
    }
    
    printf("\n  Transition Probabilities P(row → col):\n");
    printf("       ");
    for (int j = 0; j < n; j++) printf("    R%d   ", j);
    printf("\n");
    
    for (int i = 0; i < n; i++) {
        printf("  R%d: ", i);
        for (int j = 0; j < n; j++) {
            if (i == j) {
                printf(" [%5.1f%%]", dt->prob[i][j] * 100.0f);
            } else {
                printf("  %5.1f%% ", dt->prob[i][j] * 100.0f);
            }
        }
        printf("\n");
    }
    
    printf("\n  Observed Transitions:\n");
    printf("       ");
    for (int j = 0; j < n; j++) printf("   R%d  ", j);
    printf("\n");
    
    for (int i = 0; i < n; i++) {
        printf("  R%d: ", i);
        for (int j = 0; j < n; j++) {
            printf(" %5d ", dt->observed[i][j]);
        }
        printf("\n");
    }
    
    printf("═══════════════════════════════════════════════════════════════\n");
}

/**
 * @brief Get summary statistics
 */
typedef struct {
    float avg_stickiness;      /**< Average self-transition probability */
    float min_stickiness;      /**< Minimum self-transition probability */
    float max_stickiness;      /**< Maximum self-transition probability */
    float avg_row_ess;         /**< Average effective sample size per row */
    int total_transitions;     /**< Total observed transitions */
} DirichletTransitionStats;

static inline DirichletTransitionStats dirichlet_transition_stats(
    const DirichletTransition *dt)
{
    DirichletTransitionStats stats = {0};
    int n = dt->n_regimes;
    
    stats.min_stickiness = 1.0f;
    stats.max_stickiness = 0.0f;
    
    for (int i = 0; i < n; i++) {
        float s = dt->prob[i][i];
        stats.avg_stickiness += s;
        if (s < stats.min_stickiness) stats.min_stickiness = s;
        if (s > stats.max_stickiness) stats.max_stickiness = s;
        stats.avg_row_ess += dirichlet_transition_row_ess(dt, i);
    }
    
    stats.avg_stickiness /= n;
    stats.avg_row_ess /= n;
    stats.total_transitions = dt->total_transitions;
    
    return stats;
}

#ifdef __cplusplus
}
#endif

#endif /* RBPF_DIRICHLET_TRANSITION_H */
