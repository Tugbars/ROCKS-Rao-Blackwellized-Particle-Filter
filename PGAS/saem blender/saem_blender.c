/**
 * @file saem_blender.c
 * @brief SAEM Parameter Blending Implementation
 *
 * Safe Oracle → RBPF handoff via Stochastic Approximation EM.
 */

#include "saem_blender.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

/*═══════════════════════════════════════════════════════════════════════════
 * INTERNAL HELPERS
 *═══════════════════════════════════════════════════════════════════════════*/

static inline float maxf(float a, float b) { return a > b ? a : b; }
static inline float minf(float a, float b) { return a < b ? a : b; }
static inline float clampf(float x, float lo, float hi) { 
    return x < lo ? lo : (x > hi ? hi : x); 
}

/* Fast xorshift64 RNG */
static inline uint64_t xorshift64(uint64_t *state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

static inline float rand_uniform(uint64_t *state) {
    return (float)(xorshift64(state) >> 11) / (float)(1ULL << 53);
}

static inline int rand_int(uint64_t *state, int max) {
    return (int)(rand_uniform(state) * max);
}

/*═══════════════════════════════════════════════════════════════════════════
 * MATRIX UTILITIES
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Normalize rows of matrix to sum to 1
 * NOTE: Uses SAEM_MAX_REGIMES as stride for 2D array compatibility
 */
static void normalize_rows_2d(float M[SAEM_MAX_REGIMES][SAEM_MAX_REGIMES], 
                               int K, float floor_val) {
    for (int i = 0; i < K; i++) {
        float row_sum = 0.0f;
        
        /* First pass: apply floor and compute sum */
        for (int j = 0; j < K; j++) {
            if (M[i][j] < floor_val) {
                M[i][j] = floor_val;
            }
            row_sum += M[i][j];
        }
        
        /* Second pass: normalize */
        if (row_sum > 0.0f) {
            float inv_sum = 1.0f / row_sum;
            for (int j = 0; j < K; j++) {
                M[i][j] *= inv_sum;
            }
        } else {
            /* Degenerate case: uniform */
            float uniform = 1.0f / K;
            for (int j = 0; j < K; j++) {
                M[i][j] = uniform;
            }
        }
    }
}

/**
 * Compute average diagonal of transition matrix
 */
static float compute_avg_diagonal_2d(const float M[SAEM_MAX_REGIMES][SAEM_MAX_REGIMES], int K) {
    float sum = 0.0f;
    for (int i = 0; i < K; i++) {
        sum += M[i][i];
    }
    return sum / K;
}

/**
 * Compute KL divergence: KL(P || Q) = Σ P(x) log(P(x)/Q(x))
 * For flat arrays with stride K
 */
static float compute_kl_divergence(const float *P, const float *Q, int K) {
    float kl = 0.0f;
    const float eps = 1e-10f;
    
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < K; j++) {
            float p = P[i * K + j];
            float q = Q[i * K + j];
            if (p > eps && q > eps) {
                kl += p * logf(p / q);
            }
        }
    }
    
    return kl;
}

/**
 * Compute KL divergence for 2D arrays
 */
static float compute_kl_divergence_2d(const float P[SAEM_MAX_REGIMES][SAEM_MAX_REGIMES],
                                       const float Q_flat[SAEM_MAX_REGIMES * SAEM_MAX_REGIMES],
                                       int K) {
    float kl = 0.0f;
    const float eps = 1e-10f;
    
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < K; j++) {
            float p = P[i][j];
            float q = Q_flat[i * K + j];
            if (p > eps && q > eps) {
                kl += p * logf(p / q);
            }
        }
    }
    
    return kl;
}

/**
 * Copy 2D matrix to flat array
 */
static void copy_2d_to_flat(const float src[SAEM_MAX_REGIMES][SAEM_MAX_REGIMES], 
                             float *dst, int K) {
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < K; j++) {
            dst[i * K + j] = src[i][j];
        }
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * DEFAULT CONFIGURATION
 *═══════════════════════════════════════════════════════════════════════════*/

SAEMBlenderConfig saem_blender_config_defaults(int n_regimes) {
    SAEMBlenderConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    
    cfg.n_regimes = n_regimes > 0 ? n_regimes : 4;
    
    /* Gamma (learning rate) adaptation */
    cfg.gamma.gamma_base = SAEM_GAMMA_DEFAULT;
    cfg.gamma.use_robbins_monro = true;
    cfg.gamma.rm_offset = 10.0f;   /* γ = base / sqrt(t + offset) */
    
    cfg.gamma.accept_high = 0.35f;
    cfg.gamma.accept_low = 0.15f;
    cfg.gamma.accept_gamma_boost = 1.4f;
    cfg.gamma.accept_gamma_penalty = 0.6f;
    
    cfg.gamma.surprise_threshold = 2.0f;
    cfg.gamma.surprise_gamma_boost = 1.3f;
    
    cfg.gamma.diversity_threshold = 0.4f;
    cfg.gamma.diversity_gamma_boost = 1.2f;
    
    /* Stickiness control */
    cfg.stickiness.control_stickiness = true;
    cfg.stickiness.target_diag_min = 0.85f;
    cfg.stickiness.target_diag_max = 0.98f;
    cfg.stickiness.stickiness_ema_alpha = 0.1f;
    
    /* Tempered path injection */
    cfg.tempering.enable_tempering = true;
    cfg.tempering.flip_probability = 0.05f;  /* 5% of timesteps */
    cfg.tempering.seed = 0xDEADBEEF;
    
    /* Safety bounds */
    cfg.trans_floor = SAEM_TRANS_FLOOR;
    cfg.trans_ceiling = 1.0f - SAEM_TRANS_FLOOR * (n_regimes - 1);
    
    /* Diagnostics */
    cfg.track_history = false;
    cfg.history_capacity = 100;
    
    return cfg;
}

/*═══════════════════════════════════════════════════════════════════════════
 * LIFECYCLE
 *═══════════════════════════════════════════════════════════════════════════*/

int saem_blender_init(SAEMBlender *blender,
                      const SAEMBlenderConfig *cfg,
                      const float *init_Pi) {
    if (!blender) return -1;
    
    memset(blender, 0, sizeof(*blender));
    
    /* Set config */
    blender->config = cfg ? *cfg : saem_blender_config_defaults(4);
    int K = blender->config.n_regimes;
    
    if (K < 2 || K > SAEM_MAX_REGIMES) {
        fprintf(stderr, "SAEM: Invalid n_regimes %d\n", K);
        return -1;
    }
    
    /* Initialize transition matrix */
    if (init_Pi) {
        for (int i = 0; i < K; i++) {
            for (int j = 0; j < K; j++) {
                blender->Pi[i][j] = init_Pi[i * K + j];
            }
        }
    } else {
        /* Default: high diagonal (sticky), uniform off-diagonal */
        float diag = 0.9f;
        float off_diag = (1.0f - diag) / (K - 1);
        for (int i = 0; i < K; i++) {
            for (int j = 0; j < K; j++) {
                blender->Pi[i][j] = (i == j) ? diag : off_diag;
            }
        }
    }
    
    /* Initialize Q from Pi (treat Pi as normalized counts) */
    /* Start with pseudo-count of 100 per row for stability */
    float pseudo_count = 100.0f;
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < K; j++) {
            blender->Q[i][j] = pseudo_count * blender->Pi[i][j];
        }
    }
    
    /* Initialize running stats */
    blender->gamma_current = blender->config.gamma.gamma_base;
    blender->diag_ema = compute_avg_diagonal_2d(blender->Pi, K);
    blender->acceptance_ema = 0.25f;  /* Neutral starting point */
    
    /* Initialize RNG */
    blender->rng_state = blender->config.tempering.seed;
    if (blender->rng_state == 0) blender->rng_state = 0x12345678;
    
    /* Allocate history if requested */
    if (blender->config.track_history && blender->config.history_capacity > 0) {
        blender->history = calloc(blender->config.history_capacity, 
                                   sizeof(SAEMBlendEvent));
    }
    
    blender->initialized = true;
    return 0;
}

void saem_blender_reset(SAEMBlender *blender, const float *init_Pi) {
    if (!blender || !blender->initialized) return;
    
    SAEMBlenderConfig cfg = blender->config;
    SAEMBlendEvent *hist = blender->history;
    
    /* Reset state but preserve config and history buffer */
    int K = cfg.n_regimes;
    
    /* Re-init Pi */
    if (init_Pi) {
        for (int i = 0; i < K; i++) {
            for (int j = 0; j < K; j++) {
                blender->Pi[i][j] = init_Pi[i * K + j];
            }
        }
    } else {
        float diag = 0.9f;
        float off_diag = (1.0f - diag) / (K - 1);
        for (int i = 0; i < K; i++) {
            for (int j = 0; j < K; j++) {
                blender->Pi[i][j] = (i == j) ? diag : off_diag;
            }
        }
    }
    
    /* Re-init Q */
    float pseudo_count = 100.0f;
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < K; j++) {
            blender->Q[i][j] = pseudo_count * blender->Pi[i][j];
        }
    }
    
    /* Reset stats */
    blender->total_blends = 0;
    blender->gamma_current = cfg.gamma.gamma_base;
    blender->diag_ema = compute_avg_diagonal_2d(blender->Pi, K);
    blender->acceptance_ema = 0.25f;
    blender->rng_state = cfg.tempering.seed;
    
    /* Clear history */
    blender->history = hist;
    blender->history_head = 0;
    blender->history_count = 0;
}

void saem_blender_free(SAEMBlender *blender) {
    if (blender) {
        if (blender->history) {
            free(blender->history);
            blender->history = NULL;
        }
        memset(blender, 0, sizeof(*blender));
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * ADAPTIVE GAMMA COMPUTATION
 *═══════════════════════════════════════════════════════════════════════════*/

static float compute_adaptive_gamma(SAEMBlender *blender, const PGASOutput *oracle) {
    const SAEMGammaConfig *gc = &blender->config.gamma;
    float gamma = gc->gamma_base;
    
    /* Robbins-Monro decay: γ_t = γ_0 / sqrt(t + offset) */
    if (gc->use_robbins_monro && blender->total_blends > 0) {
        gamma = gc->gamma_base / sqrtf((float)blender->total_blends + gc->rm_offset);
    }
    
    /* Acceptance rate adaptation */
    if (oracle->acceptance_rate > gc->accept_high) {
        gamma *= gc->accept_gamma_boost;
    } else if (oracle->acceptance_rate < gc->accept_low) {
        gamma *= gc->accept_gamma_penalty;
    }
    
    /* Surprise adaptation */
    if (oracle->trigger_surprise > gc->surprise_threshold) {
        gamma *= gc->surprise_gamma_boost;
    }
    
    /* Diversity adaptation */
    if (oracle->ess_fraction > gc->diversity_threshold) {
        gamma *= gc->diversity_gamma_boost;
    }
    
    /* Clamp to bounds */
    gamma = clampf(gamma, SAEM_GAMMA_MIN, SAEM_GAMMA_MAX);
    
    return gamma;
}

/*═══════════════════════════════════════════════════════════════════════════
 * STICKINESS CONTROL
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Adjust transition matrix to control average diagonal (stickiness)
 *
 * If κ drifts too high (filter fossilized) or too low (jittery),
 * we extract the relative structure and re-impose target stickiness.
 */
static bool apply_stickiness_control(float Pi[SAEM_MAX_REGIMES][SAEM_MAX_REGIMES],
                                      int K,
                                      const SAEMStickinessConfig *sc,
                                      float *diag_ema) {
    if (!sc->control_stickiness) return false;
    
    float current_diag = compute_avg_diagonal_2d(Pi, K);
    
    /* Update EMA */
    *diag_ema = (1.0f - sc->stickiness_ema_alpha) * (*diag_ema) + 
                sc->stickiness_ema_alpha * current_diag;
    
    /* Check if out of bounds */
    if (current_diag >= sc->target_diag_min && 
        current_diag <= sc->target_diag_max) {
        return false;  /* No adjustment needed */
    }
    
    /* Compute target diagonal */
    float target_diag;
    if (current_diag < sc->target_diag_min) {
        target_diag = sc->target_diag_min;
    } else {
        target_diag = sc->target_diag_max;
    }
    
    /* Extract relative off-diagonal structure and re-impose stickiness */
    for (int i = 0; i < K; i++) {
        /* Get current row structure */
        float diag_i = Pi[i][i];
        float off_diag_sum = 1.0f - diag_i;
        
        if (off_diag_sum < 1e-6f) {
            /* Row is essentially all diagonal, set uniform off-diagonal */
            float new_off = (1.0f - target_diag) / (K - 1);
            for (int j = 0; j < K; j++) {
                Pi[i][j] = (i == j) ? target_diag : new_off;
            }
        } else {
            /* Scale off-diagonal to achieve target */
            float scale = (1.0f - target_diag) / off_diag_sum;
            for (int j = 0; j < K; j++) {
                if (i == j) {
                    Pi[i][j] = target_diag;
                } else {
                    Pi[i][j] *= scale;
                }
            }
        }
    }
    
    return true;
}

/*═══════════════════════════════════════════════════════════════════════════
 * CORE BLENDING
 *═══════════════════════════════════════════════════════════════════════════*/

SAEMBlendResult saem_blender_blend(SAEMBlender *blender, const PGASOutput *oracle) {
    SAEMBlendResult result;
    memset(&result, 0, sizeof(result));
    
    if (!blender || !blender->initialized || !oracle) {
        result.success = false;
        return result;
    }
    
    int K = blender->config.n_regimes;
    
    if (oracle->n_regimes != K) {
        fprintf(stderr, "SAEM: Regime mismatch: blender=%d, oracle=%d\n",
                K, oracle->n_regimes);
        result.success = false;
        return result;
    }
    
    /* Save old Pi for KL computation */
    float Pi_old[SAEM_MAX_REGIMES * SAEM_MAX_REGIMES];
    copy_2d_to_flat(blender->Pi, Pi_old, K);
    
    result.diag_avg_before = compute_avg_diagonal_2d(blender->Pi, K);
    
    /* Compute adaptive γ */
    float gamma = compute_adaptive_gamma(blender, oracle);
    blender->gamma_current = gamma;
    result.gamma_used = gamma;
    
    /* Update acceptance EMA */
    blender->acceptance_ema = 0.9f * blender->acceptance_ema + 
                              0.1f * oracle->acceptance_rate;
    
    /* ═══════════════════════════════════════════════════════════════════
     * SAEM UPDATE: Blend sufficient statistics
     *
     *   Q_new[i,j] = (1 - γ) × Q_old[i,j] + γ × S_oracle[i,j]
     *   Π[i,j] = Q[i,j] / Σ_k Q[i,k]
     * ═══════════════════════════════════════════════════════════════════ */
    
    for (int i = 0; i < K; i++) {
        float row_sum = 0.0f;
        
        for (int j = 0; j < K; j++) {
            /* Blend counts */
            blender->Q[i][j] = (1.0f - gamma) * blender->Q[i][j] + 
                               gamma * oracle->S[i][j];
            
            /* Ensure positive */
            if (blender->Q[i][j] < 1e-10f) {
                blender->Q[i][j] = 1e-10f;
            }
            
            row_sum += blender->Q[i][j];
        }
        
        /* Normalize row to get Π */
        for (int j = 0; j < K; j++) {
            blender->Pi[i][j] = blender->Q[i][j] / row_sum;
        }
    }
    
    /* ═══════════════════════════════════════════════════════════════════
     * SAFETY: Floor enforcement
     * ═══════════════════════════════════════════════════════════════════ */
    
    int floored = 0;
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < K; j++) {
            if (blender->Pi[i][j] < blender->config.trans_floor) {
                blender->Pi[i][j] = blender->config.trans_floor;
                floored++;
            }
            if (blender->Pi[i][j] > blender->config.trans_ceiling) {
                blender->Pi[i][j] = blender->config.trans_ceiling;
            }
        }
    }
    result.cells_floored = floored;
    
    /* Re-normalize after floor enforcement */
    normalize_rows_2d(blender->Pi, K, blender->config.trans_floor);
    
    /* ═══════════════════════════════════════════════════════════════════
     * SAFETY: Stickiness control
     * ═══════════════════════════════════════════════════════════════════ */
    
    result.stickiness_adjusted = apply_stickiness_control(
        blender->Pi, K, &blender->config.stickiness, &blender->diag_ema);
    
    /* ═══════════════════════════════════════════════════════════════════
     * DIAGNOSTICS
     * ═══════════════════════════════════════════════════════════════════ */
    
    result.diag_avg_after = compute_avg_diagonal_2d(blender->Pi, K);
    result.kl_divergence = compute_kl_divergence_2d(blender->Pi, Pi_old, K);
    
    /* Update Q to match normalized Pi (keep counts consistent) */
    float row_sums[SAEM_MAX_REGIMES];
    for (int i = 0; i < K; i++) {
        row_sums[i] = 0.0f;
        for (int j = 0; j < K; j++) {
            row_sums[i] += blender->Q[i][j];
        }
    }
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < K; j++) {
            blender->Q[i][j] = blender->Pi[i][j] * row_sums[i];
        }
    }
    
    /* Record history */
    if (blender->config.track_history && blender->history) {
        SAEMBlendEvent evt = {
            .tick = blender->total_blends,
            .gamma_used = gamma,
            .acceptance_rate = oracle->acceptance_rate,
            .diversity = oracle->ess_fraction,
            .surprise_sigma = oracle->trigger_surprise,
            .pre_diag_avg = result.diag_avg_before,
            .post_diag_avg = result.diag_avg_after,
            .kl_divergence = result.kl_divergence,
        };
        
        int idx = blender->history_head;
        blender->history[idx] = evt;
        blender->history_head = (idx + 1) % blender->config.history_capacity;
        if (blender->history_count < blender->config.history_capacity) {
            blender->history_count++;
        }
    }
    
    blender->total_blends++;
    result.success = true;
    
    return result;
}

/*═══════════════════════════════════════════════════════════════════════════
 * GETTERS
 *═══════════════════════════════════════════════════════════════════════════*/

void saem_blender_get_Pi(const SAEMBlender *blender, float *Pi_out) {
    if (!blender || !Pi_out) return;
    int K = blender->config.n_regimes;
    copy_2d_to_flat(blender->Pi, Pi_out, K);
}

void saem_blender_get_Q(const SAEMBlender *blender, float *Q_out) {
    if (!blender || !Q_out) return;
    int K = blender->config.n_regimes;
    copy_2d_to_flat(blender->Q, Q_out, K);
}

float saem_blender_get_gamma(const SAEMBlender *blender) {
    return blender ? blender->gamma_current : 0.0f;
}

float saem_blender_get_avg_diagonal(const SAEMBlender *blender) {
    if (!blender) return 0.0f;
    return compute_avg_diagonal_2d(blender->Pi, blender->config.n_regimes);
}

int saem_blender_get_blend_count(const SAEMBlender *blender) {
    return blender ? blender->total_blends : 0;
}

float saem_blender_kl_divergence(const float *P, const float *Q, int K) {
    return compute_kl_divergence(P, Q, K);
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEMPERED PATH GENERATION
 *═══════════════════════════════════════════════════════════════════════════*/

int saem_blender_temper_path(SAEMBlender *blender,
                              const int *rbpf_path,
                              int T,
                              int *tempered_out) {
    if (!blender || !rbpf_path || !tempered_out || T <= 0) return 0;
    
    const SAEMTemperingConfig *tc = &blender->config.tempering;
    int K = blender->config.n_regimes;
    
    /* Copy path */
    memcpy(tempered_out, rbpf_path, T * sizeof(int));
    
    if (!tc->enable_tempering) {
        return 0;
    }
    
    /* Inject random flips */
    int flips = 0;
    for (int t = 0; t < T; t++) {
        if (rand_uniform(&blender->rng_state) < tc->flip_probability) {
            /* Pick random regime different from current */
            int current = tempered_out[t];
            int new_regime;
            do {
                new_regime = rand_int(&blender->rng_state, K);
            } while (new_regime == current && K > 1);
            
            tempered_out[t] = new_regime;
            flips++;
        }
    }
    
    return flips;
}

/*═══════════════════════════════════════════════════════════════════════════
 * MANUAL CONTROL
 *═══════════════════════════════════════════════════════════════════════════*/

void saem_blender_set_gamma(SAEMBlender *blender, float gamma) {
    if (blender) {
        blender->gamma_current = clampf(gamma, SAEM_GAMMA_MIN, SAEM_GAMMA_MAX);
    }
}

void saem_blender_reset_gamma(SAEMBlender *blender) {
    if (blender) {
        blender->gamma_current = blender->config.gamma.gamma_base;
    }
}

void saem_blender_inject_Pi(SAEMBlender *blender, const float *Pi) {
    if (!blender || !Pi) return;
    
    int K = blender->config.n_regimes;
    
    /* Copy and normalize */
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < K; j++) {
            blender->Pi[i][j] = Pi[i * K + j];
        }
    }
    
    normalize_rows_2d(blender->Pi, K, blender->config.trans_floor);
    
    /* Update Q to match */
    float pseudo_count = 100.0f;
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < K; j++) {
            blender->Q[i][j] = pseudo_count * blender->Pi[i][j];
        }
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * DIAGNOSTICS
 *═══════════════════════════════════════════════════════════════════════════*/

int saem_blender_get_history(const SAEMBlender *blender,
                              SAEMBlendEvent *events,
                              int max_events) {
    if (!blender || !events || !blender->history || max_events <= 0) return 0;
    
    int count = blender->history_count;
    if (count > max_events) count = max_events;
    
    /* Copy from circular buffer (oldest first) */
    int start = (blender->history_head - blender->history_count + 
                 blender->config.history_capacity) % blender->config.history_capacity;
    
    for (int i = 0; i < count; i++) {
        int idx = (start + i) % blender->config.history_capacity;
        events[i] = blender->history[idx];
    }
    
    return count;
}

void saem_blender_print_Pi(const SAEMBlender *blender) {
    if (!blender) return;
    
    int K = blender->config.n_regimes;
    
    printf("\nTransition Matrix Π [%d×%d]:\n", K, K);
    printf("       ");
    for (int j = 0; j < K; j++) printf("   R%d   ", j);
    printf("\n");
    
    for (int i = 0; i < K; i++) {
        printf("  R%d: [", i);
        for (int j = 0; j < K; j++) {
            printf(" %6.4f", blender->Pi[i][j]);
        }
        printf(" ]\n");
    }
    printf("  Avg diagonal: %.4f\n", compute_avg_diagonal_2d(blender->Pi, K));
}

void saem_blender_print_state(const SAEMBlender *blender) {
    if (!blender) return;
    
    int K = blender->config.n_regimes;
    
    printf("\n");
    printf("+===========================================================+\n");
    printf("|                  SAEM BLENDER STATE                       |\n");
    printf("+===========================================================+\n");
    printf("| Regimes: %d   Blends: %d   γ_current: %.4f               \n",
           K, blender->total_blends, blender->gamma_current);
    printf("| Acceptance EMA: %.3f   Diagonal EMA: %.4f               \n",
           blender->acceptance_ema, blender->diag_ema);
    printf("+-----------------------------------------------------------+\n");
    
    saem_blender_print_Pi(blender);
    
    printf("+===========================================================+\n");
}

void saem_blender_print_config(const SAEMBlenderConfig *cfg) {
    if (!cfg) return;
    
    printf("\n");
    printf("+===========================================================+\n");
    printf("|                 SAEM BLENDER CONFIG                       |\n");
    printf("+===========================================================+\n");
    printf("| Regimes: %d                                               \n", cfg->n_regimes);
    printf("+-----------------------------------------------------------+\n");
    printf("| Gamma (learning rate):                                    |\n");
    printf("|   Base: %.3f   Min: %.3f   Max: %.3f                     \n",
           cfg->gamma.gamma_base, SAEM_GAMMA_MIN, SAEM_GAMMA_MAX);
    printf("|   Robbins-Monro: %s (offset=%.1f)                        \n",
           cfg->gamma.use_robbins_monro ? "ON" : "OFF", cfg->gamma.rm_offset);
    printf("|   Accept boost: %.2f (>%.2f)  penalty: %.2f (<%.2f)      \n",
           cfg->gamma.accept_gamma_boost, cfg->gamma.accept_high,
           cfg->gamma.accept_gamma_penalty, cfg->gamma.accept_low);
    printf("+-----------------------------------------------------------+\n");
    printf("| Stickiness control: %s                                   \n",
           cfg->stickiness.control_stickiness ? "ON" : "OFF");
    printf("|   Target diagonal: [%.2f, %.2f]                          \n",
           cfg->stickiness.target_diag_min, cfg->stickiness.target_diag_max);
    printf("+-----------------------------------------------------------+\n");
    printf("| Tempering: %s (flip prob=%.2f)                           \n",
           cfg->tempering.enable_tempering ? "ON" : "OFF",
           cfg->tempering.flip_probability);
    printf("+-----------------------------------------------------------+\n");
    printf("| Safety: floor=%.1e  ceiling=%.4f                         \n",
           cfg->trans_floor, cfg->trans_ceiling);
    printf("+===========================================================+\n");
}
