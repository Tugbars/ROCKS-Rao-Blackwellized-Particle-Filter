/**
 * @file lifeboat_validation.h
 * @brief Semantic alignment and divergence checks for Lifeboat injection.
 *
 * Solves the Label Switching Problem:
 *   PGAS can return mathematically equivalent solutions with permuted
 *   regime labels. This module aligns labels by μ (physical meaning)
 *   before computing divergence metrics.
 *
 * Usage:
 *   LifeboatDivergence div;
 *   LifeboatDecision decision = lifeboat_validate(live, lifeboat, &div);
 *   
 *   if (decision == LIFEBOAT_MIX || decision == LIFEBOAT_REPLACE) {
 *       lifeboat_apply_permutation(lifeboat, div.perm);
 *       lifeboat_inject(...);
 *   }
 */

#ifndef LIFEBOAT_VALIDATION_H
#define LIFEBOAT_VALIDATION_H

#include <math.h>
#include <float.h>
#include <string.h>
#include <stdio.h>

#ifdef _MSC_VER
#define LV_INLINE __inline
#else
#define LV_INLINE static inline
#endif

/*═══════════════════════════════════════════════════════════════════════════════
 * CONFIGURATION THRESHOLDS
 *═══════════════════════════════════════════════════════════════════════════════*/

/* Decision thresholds - tune these based on your market */
#define LV_THRESH_DISCARD_KL        0.02f   /* KL below this → models identical */
#define LV_THRESH_DISCARD_DELTA_MU  0.02f   /* μ drift below this → identical */
#define LV_THRESH_DISCARD_DELTA_PHI 0.01f   /* φ drift below this → identical */
#define LV_THRESH_DISCARD_DELTA_S2  0.05f   /* σ² relative drift threshold */

#define LV_THRESH_REPLACE_KL        0.50f   /* KL above this → major shift */
#define LV_THRESH_REPLACE_DELTA_H   0.50f   /* h level shift threshold */
#define LV_THRESH_REPLACE_DELTA_MU  0.30f   /* μ drift above this → major shift */

#define LV_THRESH_MU_COLLISION      0.05f   /* Minimum μ separation */
#define LV_THRESH_REGIME_EMPTY      0.01f   /* Regime with < 1% particles */
#define LV_THRESH_PI_ENTROPY_LOW    0.1f    /* Transition matrix too deterministic */
#define LV_THRESH_PI_ENTROPY_HIGH   0.95f   /* Transition matrix too random */
#define LV_THRESH_ESS_MIN           0.10f   /* Minimum ESS ratio (ESS/N) */

#define LV_MAX_CONSECUTIVE_ABORTS   5       /* Circuit breaker threshold */

/*═══════════════════════════════════════════════════════════════════════════════
 * DATA STRUCTURES
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * Divergence metrics between live RBPF and lifeboat
 * All metrics computed AFTER label alignment
 */
typedef struct {
    /* Parameter drift (aligned) */
    float delta_mu;          /* Average |μ_lb[k] - μ_live[perm[k]]| */
    float delta_phi;         /* |φ_lb - φ_live| */
    float delta_sigma2;      /* |σ²_lb - σ²_live| / σ²_live (relative) */
    float delta_Pi;          /* ||Π_lb - Π_live||_F / K (Frobenius) */
    
    /* Distribution divergence (aligned) */
    float kl_regime;         /* KL(P_lb || Q_live) on aligned regime counts */
    float delta_h_mean;      /* |E[h]_lb - E[h]_live| */
    float delta_h_var;       /* |Var[h]_lb - Var[h]_live| / Var[h]_live */
    
    /* Health metrics */
    float ess_live;          /* Effective sample size ratio of live RBPF */
    float ess_lifeboat;      /* Effective sample size ratio of lifeboat */
    float pi_entropy;        /* Normalized entropy of Π (0=deterministic, 1=uniform) */
    float mu_min_separation; /* Minimum |μ_i - μ_j| after sorting */
    
    /* Alignment info */
    int perm[8];             /* Permutation map: lb_regime[k] → live_regime[perm[k]] */
    int regime_empty_flags;  /* Bitmask: bit k set if regime k has < 1% particles */
    int alignment_reliable;  /* 1 if μ-sorting is stable, 0 if collision detected */
    
    /* Diagnostics */
    int abort_reason;        /* If ABORT, why? See LV_ABORT_* codes */
} LifeboatDivergence;

/**
 * Decision outcome
 */
typedef enum {
    LIFEBOAT_DISCARD = 0,    /* Models too similar; skip injection */
    LIFEBOAT_MIX     = 1,    /* Moderate difference; blend particles */
    LIFEBOAT_REPLACE = 2,    /* Major shift; full replacement */
    LIFEBOAT_ABORT   = 3     /* Invalid state; log and investigate */
} LifeboatDecision;

/**
 * Abort reason codes
 */
#define LV_ABORT_NONE               0
#define LV_ABORT_LIVE_DEGENERATE    1   /* Live RBPF ESS too low */
#define LV_ABORT_LB_DEGENERATE      2   /* Lifeboat ESS too low */
#define LV_ABORT_MU_COLLISION       3   /* Cannot reliably align labels */
#define LV_ABORT_PI_DEGENERATE      4   /* Transition matrix is garbage */
#define LV_ABORT_REGIME_EMPTY       5   /* One or more regimes are empty */
#define LV_ABORT_NAN_DETECTED       6   /* NaN in parameters */

/**
 * Model parameters (matches your existing struct)
 */
typedef struct {
    int K;
    float mu[8];
    float phi;
    float sigma2;
    float Pi[64];  /* K×K, row-major */
} LVModelParams;

/*═══════════════════════════════════════════════════════════════════════════════
 * HELPER FUNCTIONS
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * Argsort: returns indices that would sort arr in ascending order
 * Uses insertion sort (fast for K ≤ 8)
 */
LV_INLINE void lv_argsort_float(const float *arr, int n, int *indices)
{
    for (int i = 0; i < n; i++) indices[i] = i;
    
    for (int i = 1; i < n; i++) {
        int key_idx = indices[i];
        float key_val = arr[key_idx];
        int j = i - 1;
        
        while (j >= 0 && arr[indices[j]] > key_val) {
            indices[j + 1] = indices[j];
            j--;
        }
        indices[j + 1] = key_idx;
    }
}

/**
 * Check for NaN in float array
 */
LV_INLINE int lv_has_nan(const float *arr, int n)
{
    for (int i = 0; i < n; i++) {
        if (arr[i] != arr[i]) return 1;  /* NaN != NaN */
    }
    return 0;
}

/**
 * Compute effective sample size ratio from log-weights
 * Returns ESS/N in range [1/N, 1]
 */
LV_INLINE float lv_compute_ess_ratio(const float *log_weights, int N)
{
    /* Find max for numerical stability */
    float max_lw = -FLT_MAX;
    for (int i = 0; i < N; i++) {
        if (log_weights[i] > max_lw) max_lw = log_weights[i];
    }
    
    /* Compute normalized weights and ESS */
    float sum_w = 0.0f, sum_w2 = 0.0f;
    for (int i = 0; i < N; i++) {
        float w = expf(log_weights[i] - max_lw);
        sum_w += w;
        sum_w2 += w * w;
    }
    
    if (sum_w < 1e-10f) return 0.0f;
    
    float ess = (sum_w * sum_w) / sum_w2;
    return ess / N;
}

/**
 * Compute normalized entropy of transition matrix row
 * Returns value in [0, 1]: 0=deterministic, 1=uniform
 */
LV_INLINE float lv_row_entropy(const float *row, int K)
{
    float entropy = 0.0f;
    float log_K = logf((float)K);
    
    for (int j = 0; j < K; j++) {
        if (row[j] > 1e-10f) {
            entropy -= row[j] * logf(row[j]);
        }
    }
    
    return entropy / log_K;  /* Normalize to [0, 1] */
}

/**
 * Average entropy of transition matrix
 */
LV_INLINE float lv_transition_entropy(const float *Pi, int K)
{
    float total = 0.0f;
    for (int i = 0; i < K; i++) {
        total += lv_row_entropy(&Pi[i * K], K);
    }
    return total / K;
}

/*═══════════════════════════════════════════════════════════════════════════════
 * ALIGNMENT
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * Compute label alignment based on μ values
 * Returns 1 if alignment is reliable, 0 if μ-collision detected
 */
LV_INLINE int lv_compute_alignment(
    const float *mu_live, 
    const float *mu_lb, 
    int K,
    int *perm,           /* Output: lb_index → live_index */
    float *min_sep)      /* Output: minimum μ separation */
{
    int live_order[8], lb_order[8];
    
    /* Sort both by μ ascending */
    lv_argsort_float(mu_live, K, live_order);
    lv_argsort_float(mu_lb, K, lb_order);
    
    /* Create permutation: lb_regime[lb_order[i]] maps to live_regime[live_order[i]] */
    for (int i = 0; i < K; i++) {
        perm[lb_order[i]] = live_order[i];
    }
    
    /* Check for μ-collisions in live model */
    *min_sep = FLT_MAX;
    for (int i = 1; i < K; i++) {
        float sep = mu_live[live_order[i]] - mu_live[live_order[i-1]];
        if (sep < *min_sep) *min_sep = sep;
    }
    
    return (*min_sep >= LV_THRESH_MU_COLLISION) ? 1 : 0;
}

/*═══════════════════════════════════════════════════════════════════════════════
 * DIVERGENCE COMPUTATION
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * Compute regime distribution from particle regime assignments
 */
LV_INLINE void lv_compute_regime_dist(
    const int *regimes, 
    int N, 
    int K, 
    float *dist,
    int *empty_flags)
{
    memset(dist, 0, K * sizeof(float));
    *empty_flags = 0;
    
    float inv_N = 1.0f / N;
    for (int n = 0; n < N; n++) {
        dist[regimes[n]] += inv_N;
    }
    
    for (int k = 0; k < K; k++) {
        if (dist[k] < LV_THRESH_REGIME_EMPTY) {
            *empty_flags |= (1 << k);
        }
    }
}

/**
 * Compute KL divergence: KL(P || Q)
 * P = lifeboat distribution (aligned)
 * Q = live distribution
 */
LV_INLINE float lv_compute_kl(const float *P, const float *Q, int K)
{
    float kl = 0.0f;
    const float eps = 1e-6f;
    
    for (int k = 0; k < K; k++) {
        float p = P[k] + eps;
        float q = Q[k] + eps;
        kl += p * logf(p / q);
    }
    
    return fmaxf(0.0f, kl);
}

/**
 * Main validation function
 */
LV_INLINE LifeboatDecision lifeboat_validate(
    /* Live RBPF state */
    const LVModelParams *live_params,
    const int *live_regimes,
    const float *live_h,
    const float *live_log_weights,
    int N_live,
    
    /* Lifeboat state */
    const LVModelParams *lb_params,
    const int *lb_regimes,
    const float *lb_h,
    const float *lb_log_weights,
    int N_lb,
    
    /* Output */
    LifeboatDivergence *div)
{
    int K = live_params->K;
    memset(div, 0, sizeof(LifeboatDivergence));
    
    /* ═══════════════════════════════════════════════════════════════════════
     * SANITY CHECKS
     * ═══════════════════════════════════════════════════════════════════════*/
    
    /* Check for NaN */
    if (lv_has_nan(live_params->mu, K) || lv_has_nan(lb_params->mu, K) ||
        lv_has_nan(&live_params->phi, 1) || lv_has_nan(&lb_params->phi, 1)) {
        div->abort_reason = LV_ABORT_NAN_DETECTED;
        return LIFEBOAT_ABORT;
    }
    
    /* Check ESS */
    div->ess_live = lv_compute_ess_ratio(live_log_weights, N_live);
    div->ess_lifeboat = lv_compute_ess_ratio(lb_log_weights, N_lb);
    
    if (div->ess_live < LV_THRESH_ESS_MIN) {
        div->abort_reason = LV_ABORT_LIVE_DEGENERATE;
        return LIFEBOAT_ABORT;
    }
    
    if (div->ess_lifeboat < LV_THRESH_ESS_MIN) {
        div->abort_reason = LV_ABORT_LB_DEGENERATE;
        return LIFEBOAT_ABORT;
    }
    
    /* ═══════════════════════════════════════════════════════════════════════
     * STEP 1: LABEL ALIGNMENT BY μ
     * ═══════════════════════════════════════════════════════════════════════*/
    
    div->alignment_reliable = lv_compute_alignment(
        live_params->mu, lb_params->mu, K, 
        div->perm, &div->mu_min_separation);
    
    if (!div->alignment_reliable) {
        /* μ-collision detected - alignment may be unstable */
        div->abort_reason = LV_ABORT_MU_COLLISION;
        /* Don't abort yet - might still be useful info, just flag it */
    }
    
    /* ═══════════════════════════════════════════════════════════════════════
     * STEP 2: PARAMETER DRIFT (ALIGNED)
     * ═══════════════════════════════════════════════════════════════════════*/
    
    /* Δμ: average drift in regime means */
    div->delta_mu = 0.0f;
    for (int k = 0; k < K; k++) {
        int live_k = div->perm[k];
        div->delta_mu += fabsf(lb_params->mu[k] - live_params->mu[live_k]);
    }
    div->delta_mu /= K;
    
    /* Δφ: AR persistence drift */
    div->delta_phi = fabsf(lb_params->phi - live_params->phi);
    
    /* Δσ²: relative innovation variance drift */
    div->delta_sigma2 = fabsf(lb_params->sigma2 - live_params->sigma2) /
                        fmaxf(live_params->sigma2, 1e-6f);
    
    /* ΔΠ: transition matrix drift (Frobenius norm, aligned) */
    div->delta_Pi = 0.0f;
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < K; j++) {
            /* Align both indices */
            int live_i = div->perm[i];
            int live_j = div->perm[j];
            float diff = lb_params->Pi[i * K + j] - 
                        live_params->Pi[live_i * K + live_j];
            div->delta_Pi += diff * diff;
        }
    }
    div->delta_Pi = sqrtf(div->delta_Pi) / K;
    
    /* ═══════════════════════════════════════════════════════════════════════
     * STEP 3: TRANSITION MATRIX HEALTH
     * ═══════════════════════════════════════════════════════════════════════*/
    
    div->pi_entropy = lv_transition_entropy(lb_params->Pi, K);
    
    if (div->pi_entropy < LV_THRESH_PI_ENTROPY_LOW ||
        div->pi_entropy > LV_THRESH_PI_ENTROPY_HIGH) {
        div->abort_reason = LV_ABORT_PI_DEGENERATE;
        /* Continue computing metrics but flag the issue */
    }
    
    /* ═══════════════════════════════════════════════════════════════════════
     * STEP 4: REGIME DISTRIBUTION KL (ALIGNED)
     * ═══════════════════════════════════════════════════════════════════════*/
    
    float dist_live[8], dist_lb[8], dist_lb_aligned[8];
    int empty_live, empty_lb;
    
    lv_compute_regime_dist(live_regimes, N_live, K, dist_live, &empty_live);
    lv_compute_regime_dist(lb_regimes, N_lb, K, dist_lb, &empty_lb);
    
    /* Align lifeboat distribution to live labels */
    for (int k = 0; k < K; k++) {
        dist_lb_aligned[div->perm[k]] = dist_lb[k];
    }
    
    div->kl_regime = lv_compute_kl(dist_lb_aligned, dist_live, K);
    div->regime_empty_flags = empty_live | empty_lb;
    
    if (div->regime_empty_flags && div->abort_reason == LV_ABORT_NONE) {
        div->abort_reason = LV_ABORT_REGIME_EMPTY;
    }
    
    /* ═══════════════════════════════════════════════════════════════════════
     * STEP 5: VOLATILITY LEVEL CONSISTENCY
     * ═══════════════════════════════════════════════════════════════════════*/
    
    float h_mean_live = 0.0f, h_mean_lb = 0.0f;
    float h_var_live = 0.0f, h_var_lb = 0.0f;
    
    for (int n = 0; n < N_live; n++) h_mean_live += live_h[n];
    for (int n = 0; n < N_lb; n++) h_mean_lb += lb_h[n];
    h_mean_live /= N_live;
    h_mean_lb /= N_lb;
    
    for (int n = 0; n < N_live; n++) {
        float d = live_h[n] - h_mean_live;
        h_var_live += d * d;
    }
    for (int n = 0; n < N_lb; n++) {
        float d = lb_h[n] - h_mean_lb;
        h_var_lb += d * d;
    }
    h_var_live /= N_live;
    h_var_lb /= N_lb;
    
    div->delta_h_mean = fabsf(h_mean_lb - h_mean_live);
    div->delta_h_var = fabsf(h_var_lb - h_var_live) / fmaxf(h_var_live, 1e-6f);
    
    /* ═══════════════════════════════════════════════════════════════════════
     * STEP 6: DECISION LOGIC
     * ═══════════════════════════════════════════════════════════════════════*/
    
    /* Check for ABORT conditions (but not μ-collision alone) */
    if (div->abort_reason == LV_ABORT_LIVE_DEGENERATE ||
        div->abort_reason == LV_ABORT_LB_DEGENERATE ||
        div->abort_reason == LV_ABORT_NAN_DETECTED) {
        return LIFEBOAT_ABORT;
    }
    
    /* DISCARD: parameters nearly identical */
    if (div->delta_mu < LV_THRESH_DISCARD_DELTA_MU &&
        div->delta_phi < LV_THRESH_DISCARD_DELTA_PHI &&
        div->delta_sigma2 < LV_THRESH_DISCARD_DELTA_S2 &&
        div->kl_regime < LV_THRESH_DISCARD_KL) {
        return LIFEBOAT_DISCARD;
    }
    
    /* REPLACE: major structural shift */
    if (div->kl_regime > LV_THRESH_REPLACE_KL ||
        div->delta_h_mean > LV_THRESH_REPLACE_DELTA_H ||
        div->delta_mu > LV_THRESH_REPLACE_DELTA_MU) {
        return LIFEBOAT_REPLACE;
    }
    
    /* MIX: moderate refinement */
    return LIFEBOAT_MIX;
}

/*═══════════════════════════════════════════════════════════════════════════════
 * APPLY PERMUTATION
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * Apply permutation to lifeboat parameters and regimes
 * Call this BEFORE injection if decision is MIX or REPLACE
 */
LV_INLINE void lifeboat_apply_permutation(
    LVModelParams *lb_params,
    int *lb_regimes,
    int N,
    const int *perm)
{
    int K = lb_params->K;
    
    /* Permute μ */
    float mu_new[8];
    for (int k = 0; k < K; k++) {
        mu_new[perm[k]] = lb_params->mu[k];
    }
    memcpy(lb_params->mu, mu_new, K * sizeof(float));
    
    /* Permute Π (both rows and columns) */
    float Pi_new[64];
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < K; j++) {
            Pi_new[perm[i] * K + perm[j]] = lb_params->Pi[i * K + j];
        }
    }
    memcpy(lb_params->Pi, Pi_new, K * K * sizeof(float));
    
    /* Permute regime assignments */
    for (int n = 0; n < N; n++) {
        lb_regimes[n] = perm[lb_regimes[n]];
    }
}

/*═══════════════════════════════════════════════════════════════════════════════
 * LOGGING / DIAGNOSTICS
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * Get human-readable abort reason
 */
LV_INLINE const char* lv_abort_reason_str(int reason)
{
    switch (reason) {
        case LV_ABORT_NONE:            return "none";
        case LV_ABORT_LIVE_DEGENERATE: return "live_rbpf_degenerate";
        case LV_ABORT_LB_DEGENERATE:   return "lifeboat_degenerate";
        case LV_ABORT_MU_COLLISION:    return "mu_collision";
        case LV_ABORT_PI_DEGENERATE:   return "pi_degenerate";
        case LV_ABORT_REGIME_EMPTY:    return "regime_empty";
        case LV_ABORT_NAN_DETECTED:    return "nan_detected";
        default:                       return "unknown";
    }
}

/**
 * Get human-readable decision
 */
LV_INLINE const char* lv_decision_str(LifeboatDecision d)
{
    switch (d) {
        case LIFEBOAT_DISCARD: return "DISCARD";
        case LIFEBOAT_MIX:     return "MIX";
        case LIFEBOAT_REPLACE: return "REPLACE";
        case LIFEBOAT_ABORT:   return "ABORT";
        default:               return "UNKNOWN";
    }
}

/**
 * Print divergence report (for logging/debugging)
 */
LV_INLINE void lv_print_divergence(const LifeboatDivergence *div, LifeboatDecision decision)
{
    printf("┌─────────────────────────────────────────────────────────┐\n");
    printf("│           LIFEBOAT VALIDATION REPORT                    │\n");
    printf("├─────────────────────────────────────────────────────────┤\n");
    printf("│  Decision:         %-10s                            │\n", lv_decision_str(decision));
    if (div->abort_reason != LV_ABORT_NONE) {
        printf("│  Abort Reason:     %-25s        │\n", lv_abort_reason_str(div->abort_reason));
    }
    printf("├─────────────────────────────────────────────────────────┤\n");
    printf("│  Parameter Drift (aligned):                             │\n");
    printf("│    Δμ:       %.4f  (thresh: %.2f/%.2f)                 │\n", 
           div->delta_mu, LV_THRESH_DISCARD_DELTA_MU, LV_THRESH_REPLACE_DELTA_MU);
    printf("│    Δφ:       %.4f  (thresh: %.2f)                      │\n", 
           div->delta_phi, LV_THRESH_DISCARD_DELTA_PHI);
    printf("│    Δσ²:      %.4f  (thresh: %.2f)                      │\n", 
           div->delta_sigma2, LV_THRESH_DISCARD_DELTA_S2);
    printf("│    ΔΠ:       %.4f                                      │\n", div->delta_Pi);
    printf("├─────────────────────────────────────────────────────────┤\n");
    printf("│  Distribution Divergence:                               │\n");
    printf("│    KL:       %.4f  (thresh: %.2f/%.2f)                 │\n", 
           div->kl_regime, LV_THRESH_DISCARD_KL, LV_THRESH_REPLACE_KL);
    printf("│    Δh_mean:  %.4f  (thresh: %.2f)                      │\n", 
           div->delta_h_mean, LV_THRESH_REPLACE_DELTA_H);
    printf("│    Δh_var:   %.4f                                      │\n", div->delta_h_var);
    printf("├─────────────────────────────────────────────────────────┤\n");
    printf("│  Health Metrics:                                        │\n");
    printf("│    ESS_live:     %.2f  (min: %.2f)                     │\n", 
           div->ess_live, LV_THRESH_ESS_MIN);
    printf("│    ESS_lifeboat: %.2f  (min: %.2f)                     │\n", 
           div->ess_lifeboat, LV_THRESH_ESS_MIN);
    printf("│    Π_entropy:    %.2f  (range: %.2f-%.2f)              │\n", 
           div->pi_entropy, LV_THRESH_PI_ENTROPY_LOW, LV_THRESH_PI_ENTROPY_HIGH);
    printf("│    μ_min_sep:    %.4f  (min: %.2f)                     │\n", 
           div->mu_min_separation, LV_THRESH_MU_COLLISION);
    printf("│    Alignment:    %s                               │\n", 
           div->alignment_reliable ? "RELIABLE" : "UNSTABLE");
    printf("└─────────────────────────────────────────────────────────┘\n");
}

/*═══════════════════════════════════════════════════════════════════════════════
 * CIRCUIT BREAKER
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * Track consecutive aborts for circuit breaker
 */
typedef struct {
    int consecutive_aborts;
    int total_aborts;
    int total_discards;
    int total_mixes;
    int total_replaces;
    int circuit_breaker_tripped;
} LifeboatStats;

LV_INLINE void lv_stats_init(LifeboatStats *stats)
{
    memset(stats, 0, sizeof(LifeboatStats));
}

LV_INLINE void lv_stats_update(LifeboatStats *stats, LifeboatDecision decision)
{
    switch (decision) {
        case LIFEBOAT_ABORT:
            stats->consecutive_aborts++;
            stats->total_aborts++;
            if (stats->consecutive_aborts >= LV_MAX_CONSECUTIVE_ABORTS) {
                stats->circuit_breaker_tripped = 1;
            }
            break;
        case LIFEBOAT_DISCARD:
            stats->consecutive_aborts = 0;
            stats->total_discards++;
            break;
        case LIFEBOAT_MIX:
            stats->consecutive_aborts = 0;
            stats->total_mixes++;
            break;
        case LIFEBOAT_REPLACE:
            stats->consecutive_aborts = 0;
            stats->total_replaces++;
            break;
    }
}

#endif /* LIFEBOAT_VALIDATION_H */
