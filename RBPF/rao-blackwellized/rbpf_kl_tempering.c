/**
 * @file rbpf_kl_tempering.c
 * @brief Information-Geometric Weight Normalization - Implementation
 *
 * This file contains:
 *   - Scalar implementations (portable)
 *   - AVX-512 optimized kernels (Intel)
 *   - State management and diagnostics
 *
 * Performance targets (512 particles, 14900KF @ 5GHz):
 *   - KL computation: < 0.1 μs
 *   - Weight application: < 0.05 μs
 *   - Total overhead: < 0.2 μs per tick
 */

#include "rbpf_kl_tempering.h"
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

#if RBPF_KL_USE_AVX512
#include <immintrin.h>
#endif

/*═══════════════════════════════════════════════════════════════════════════
 * LIFECYCLE
 *═══════════════════════════════════════════════════════════════════════════*/

void rbpf_kl_init(RBPF_KL_State *state, int n)
{
    memset(state, 0, sizeof(*state));
    
    /* Initialize P² for 95th percentile */
    p2_init(&state->kl_quantile, RBPF_KL_DEFAULT_QUANTILE);
    
    /* Configuration */
    state->enabled = 1;
    state->beta_floor = RBPF_KL_BETA_FLOOR;
    state->damped_threshold = RBPF_KL_DAMPED_THRESHOLD;
    state->emergency_lambda = RBPF_KL_EMERGENCY_LAMBDA;
    state->max_damped_before_reset = RBPF_KL_MAX_DAMPED_TICKS;
    
    /* Computed values */
    state->kl_ceiling = logf((float)n);
    state->log_Z_old = 0.0f;  /* Uniform weights after init */
    
    /* Diagnostics */
    state->min_beta_seen = 1.0f;
    state->max_kl_seen = 0.0f;
}

void rbpf_kl_reset(RBPF_KL_State *state)
{
    /* Reset operational state but preserve learned quantile */
    state->log_Z_old = 0.0f;
    state->consecutive_damped_ticks = 0;
    state->last_beta = 1.0f;
    state->last_kl = 0.0f;
}

void rbpf_kl_reset_full(RBPF_KL_State *state, int n)
{
    rbpf_kl_init(state, n);
}

/*═══════════════════════════════════════════════════════════════════════════
 * SCALAR IMPLEMENTATIONS
 *═══════════════════════════════════════════════════════════════════════════*/

float rbpf_kl_compute(
    const float *log_weight,
    const float *log_lik_increment,
    int n,
    float log_Z_old)
{
#if RBPF_KL_USE_AVX512
    /* Use AVX-512 if available and n is suitable */
    if ((n % 16) == 0 && n >= 64) {
        return rbpf_kl_compute_avx512(log_weight, log_lik_increment, n, log_Z_old);
    }
#endif

    /* Scalar fallback */
    
    /* 1. Compute proposed log-weights and find max */
    float max_lw_prop = -FLT_MAX;
    for (int i = 0; i < n; i++) {
        float lw_prop = log_weight[i] + log_lik_increment[i];
        if (lw_prop > max_lw_prop) {
            max_lw_prop = lw_prop;
        }
    }
    
    /* 2. Compute sum and dot product in one pass */
    float sum_exp = 0.0f;
    float dot_product = 0.0f;
    
    for (int i = 0; i < n; i++) {
        float lw_prop = log_weight[i] + log_lik_increment[i];
        float w_unnorm = expf(lw_prop - max_lw_prop);
        sum_exp += w_unnorm;
        dot_product += w_unnorm * log_lik_increment[i];
    }
    
    /* 3. Compute KL */
    float log_Z_prop = max_lw_prop + logf(sum_exp);
    float kl = (dot_product / sum_exp) - log_Z_prop + log_Z_old;
    
    /* Clamp numerical errors */
    if (kl < 0.0f) kl = 0.0f;
    
    return kl;
}

float rbpf_kl_compute_beta(RBPF_KL_State *state, float proposed_kl)
{
    float beta = 1.0f;
    
    /* Get current P² estimate */
    float kl_p95 = (float)p2_get_quantile(&state->kl_quantile);
    state->kl_p95 = kl_p95;
    
    /* Check against hard ceiling first */
    if (proposed_kl > state->kl_ceiling) {
        /* Hard clamp - always active */
        beta = state->kl_ceiling / proposed_kl;
        state->hard_clamp_count++;
    }
    else if (state->warmup_complete && proposed_kl > kl_p95 && kl_p95 > 0.0f) {
        /* Soft continuous dampening: β = expected / actual */
        beta = kl_p95 / proposed_kl;
        
        /* But never below hard ceiling ratio */
        float beta_floor_hard = state->kl_ceiling / proposed_kl;
        if (beta < beta_floor_hard) {
            beta = beta_floor_hard;
            state->hard_clamp_count++;
        } else {
            state->soft_damp_count++;
        }
    }
    
    /* Apply absolute floor */
    if (beta < state->beta_floor) {
        beta = state->beta_floor;
    }
    
    /* Update min tracking */
    if (beta < state->min_beta_seen) {
        state->min_beta_seen = beta;
    }
    
    return beta;
}

void rbpf_kl_apply_tempered(
    float *log_weight,
    const float *log_lik_increment,
    float beta,
    int n)
{
#if RBPF_KL_USE_AVX512
    if ((n % 16) == 0 && n >= 64) {
        rbpf_kl_apply_tempered_avx512(log_weight, log_lik_increment, beta, n);
        return;
    }
#endif

    /* Scalar fallback */
    for (int i = 0; i < n; i++) {
        log_weight[i] += beta * log_lik_increment[i];
    }
}

float rbpf_kl_compute_log_Z(const float *log_weight, int n)
{
#if RBPF_KL_USE_AVX512
    if ((n % 16) == 0 && n >= 64) {
        return rbpf_kl_compute_log_Z_avx512(log_weight, n);
    }
#endif

    /* Scalar fallback with log-sum-exp trick */
    float max_lw = log_weight[0];
    for (int i = 1; i < n; i++) {
        if (log_weight[i] > max_lw) {
            max_lw = log_weight[i];
        }
    }
    
    float sum_exp = 0.0f;
    for (int i = 0; i < n; i++) {
        sum_exp += expf(log_weight[i] - max_lw);
    }
    
    return max_lw + logf(sum_exp);
}

/*═══════════════════════════════════════════════════════════════════════════
 * FULL UPDATE STEP
 *═══════════════════════════════════════════════════════════════════════════*/

RBPF_KL_Result rbpf_kl_step(
    RBPF_KL_State *state,
    float *log_weight,
    const float *log_lik_increment,
    int n,
    int resampled)
{
    RBPF_KL_Result result = {0};
    
    if (!state->enabled) {
        /* Disabled: apply full update */
        for (int i = 0; i < n; i++) {
            log_weight[i] += log_lik_increment[i];
        }
        result.beta = 1.0f;
        return result;
    }
    
    state->ticks_processed++;
    
    /* 1. Handle post-resampling state */
    if (resampled) {
        /* After resampling, weights are uniform → log_Z = 0 */
        state->log_Z_old = 0.0f;
    }
    
    /* 2. Compute proposed KL divergence */
    float proposed_kl = rbpf_kl_compute(
        log_weight, log_lik_increment, n, state->log_Z_old);
    
    state->last_kl = proposed_kl;
    result.kl = proposed_kl;
    
    /* Track max */
    if (proposed_kl > state->max_kl_seen) {
        state->max_kl_seen = proposed_kl;
    }
    
    /* 3. Update P² quantile estimator */
    p2_update(&state->kl_quantile, (double)proposed_kl);
    
    /* Check warmup */
    if (!state->warmup_complete && 
        p2_get_count(&state->kl_quantile) >= RBPF_KL_WARMUP_TICKS) {
        state->warmup_complete = 1;
    }
    
    /* 4. Compute tempering factor β */
    float beta = rbpf_kl_compute_beta(state, proposed_kl);
    state->last_beta = beta;
    result.beta = beta;
    
    /* 5. Zombie detection */
    result.heavily_damped = (beta < state->damped_threshold) ? 1 : 0;
    
    if (result.heavily_damped) {
        state->consecutive_damped_ticks++;
        
        if (state->consecutive_damped_ticks > state->max_damped_before_reset) {
            /* Zombie detected - signal reset */
            result.zombie_detected = 1;
            state->zombie_resets++;
            state->consecutive_damped_ticks = 0;
        }
    } else {
        state->consecutive_damped_ticks = 0;
    }
    
    /* 6. Apply tempered weights */
    rbpf_kl_apply_tempered(log_weight, log_lik_increment, beta, n);
    
    /* 7. Update log_Z_old for next tick */
    state->log_Z_old = rbpf_kl_compute_log_Z(log_weight, n);
    
    return result;
}

/*═══════════════════════════════════════════════════════════════════════════
 * AVX-512 IMPLEMENTATIONS
 *═══════════════════════════════════════════════════════════════════════════*/

#if RBPF_KL_USE_AVX512

float rbpf_kl_compute_avx512(
    const float *log_weight,
    const float *log_lik_increment,
    int n,
    float log_Z_old)
{
    /* 1. Compute proposed log-weights and find max */
    __m512 max_vec = _mm512_set1_ps(-FLT_MAX);
    
    for (int i = 0; i < n; i += 16) {
        __m512 lw = _mm512_load_ps(log_weight + i);
        __m512 ll = _mm512_load_ps(log_lik_increment + i);
        __m512 lw_prop = _mm512_add_ps(lw, ll);
        max_vec = _mm512_max_ps(max_vec, lw_prop);
    }
    
    /* Horizontal max reduction */
    float max_lw_prop = _mm512_reduce_max_ps(max_vec);
    __m512 max_broadcast = _mm512_set1_ps(max_lw_prop);
    
    /* 2. Compute sum and dot product */
    __m512 sum_vec = _mm512_setzero_ps();
    __m512 dot_vec = _mm512_setzero_ps();
    
    for (int i = 0; i < n; i += 16) {
        __m512 lw = _mm512_load_ps(log_weight + i);
        __m512 ll = _mm512_load_ps(log_lik_increment + i);
        __m512 lw_prop = _mm512_add_ps(lw, ll);
        
        /* Shifted for numerical stability */
        __m512 lw_shifted = _mm512_sub_ps(lw_prop, max_broadcast);
        
        /* exp - Intel SVML via intrinsic */
        __m512 w_unnorm = _mm512_exp_ps(lw_shifted);
        
        /* Accumulate sum */
        sum_vec = _mm512_add_ps(sum_vec, w_unnorm);
        
        /* Accumulate dot product: w_unnorm * log_lik */
        dot_vec = _mm512_fmadd_ps(w_unnorm, ll, dot_vec);
    }
    
    /* Horizontal reductions */
    float sum_exp = _mm512_reduce_add_ps(sum_vec);
    float dot_product = _mm512_reduce_add_ps(dot_vec);
    
    /* 3. Compute KL */
    float log_Z_prop = max_lw_prop + logf(sum_exp);
    float kl = (dot_product / sum_exp) - log_Z_prop + log_Z_old;
    
    /* Clamp numerical errors */
    if (kl < 0.0f) kl = 0.0f;
    
    return kl;
}

void rbpf_kl_apply_tempered_avx512(
    float *log_weight,
    const float *log_lik_increment,
    float beta,
    int n)
{
    __m512 beta_vec = _mm512_set1_ps(beta);
    
    for (int i = 0; i < n; i += 16) {
        __m512 lw = _mm512_load_ps(log_weight + i);
        __m512 ll = _mm512_load_ps(log_lik_increment + i);
        __m512 scaled_ll = _mm512_mul_ps(ll, beta_vec);
        __m512 lw_new = _mm512_add_ps(lw, scaled_ll);
        _mm512_store_ps(log_weight + i, lw_new);
    }
}

float rbpf_kl_compute_log_Z_avx512(const float *log_weight, int n)
{
    /* Find max */
    __m512 max_vec = _mm512_set1_ps(-FLT_MAX);
    
    for (int i = 0; i < n; i += 16) {
        __m512 lw = _mm512_load_ps(log_weight + i);
        max_vec = _mm512_max_ps(max_vec, lw);
    }
    
    float max_lw = _mm512_reduce_max_ps(max_vec);
    __m512 max_broadcast = _mm512_set1_ps(max_lw);
    
    /* Sum exp(log_weight - max) */
    __m512 sum_vec = _mm512_setzero_ps();
    
    for (int i = 0; i < n; i += 16) {
        __m512 lw = _mm512_load_ps(log_weight + i);
        __m512 lw_shifted = _mm512_sub_ps(lw, max_broadcast);
        __m512 w = _mm512_exp_ps(lw_shifted);
        sum_vec = _mm512_add_ps(sum_vec, w);
    }
    
    float sum_exp = _mm512_reduce_add_ps(sum_vec);
    
    return max_lw + logf(sum_exp);
}

#endif /* RBPF_KL_USE_AVX512 */

/*═══════════════════════════════════════════════════════════════════════════
 * DIAGNOSTICS
 *═══════════════════════════════════════════════════════════════════════════*/

void rbpf_kl_print_diagnostics(const RBPF_KL_State *state)
{
    printf("\n");
    printf("════════════════════════════════════════════════════════════════\n");
    printf("  KL TEMPERING DIAGNOSTICS\n");
    printf("════════════════════════════════════════════════════════════════\n");
    printf("\n");
    
    printf("Configuration:\n");
    printf("  Enabled:              %s\n", state->enabled ? "YES" : "NO");
    printf("  KL Ceiling (log N):   %.3f nats\n", state->kl_ceiling);
    printf("  Beta Floor:           %.2f\n", state->beta_floor);
    printf("  Damped Threshold:     %.2f\n", state->damped_threshold);
    printf("  Max Damped Ticks:     %d\n", state->max_damped_before_reset);
    printf("  Warmup Complete:      %s\n", state->warmup_complete ? "YES" : "NO");
    printf("\n");
    
    printf("Current State:\n");
    printf("  Last KL:              %.4f nats\n", state->last_kl);
    printf("  Last Beta:            %.4f\n", state->last_beta);
    printf("  KL P95 (learned):     %.4f nats\n", state->kl_p95);
    printf("  Consecutive Damped:   %d\n", state->consecutive_damped_ticks);
    printf("  log_Z_old:            %.4f\n", state->log_Z_old);
    printf("\n");
    
    printf("Statistics:\n");
    printf("  Ticks Processed:      %llu\n", (unsigned long long)state->ticks_processed);
    printf("  Soft Damp Events:     %llu (%.2f%%)\n", 
           (unsigned long long)state->soft_damp_count,
           state->ticks_processed > 0 ? 
               100.0 * state->soft_damp_count / state->ticks_processed : 0.0);
    printf("  Hard Clamp Events:    %llu (%.2f%%)\n",
           (unsigned long long)state->hard_clamp_count,
           state->ticks_processed > 0 ?
               100.0 * state->hard_clamp_count / state->ticks_processed : 0.0);
    printf("  Zombie Resets:        %llu\n", (unsigned long long)state->zombie_resets);
    printf("  Min Beta Seen:        %.4f\n", state->min_beta_seen);
    printf("  Max KL Seen:          %.4f nats\n", state->max_kl_seen);
    printf("\n");
    
    printf("Health Check:\n");
    if (state->zombie_resets > 0) {
        printf("  ⚠️  Zombie resets detected - model may be misspecified\n");
    }
    if (state->min_beta_seen < 0.3f) {
        printf("  ⚠️  Severe tempering occurred (β < 0.3)\n");
    }
    if (state->hard_clamp_count > state->ticks_processed / 100) {
        printf("  ⚠️  >1%% of ticks hit hard ceiling - check regime params\n");
    }
    if (state->soft_damp_count < state->ticks_processed / 1000 && 
        state->ticks_processed > 1000) {
        printf("  ✓  Healthy: minimal dampening required\n");
    }
    
    printf("════════════════════════════════════════════════════════════════\n\n");
}
