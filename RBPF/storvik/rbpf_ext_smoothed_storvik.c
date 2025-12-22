/**
 * @file rbpf_ext_smoothed_storvik.c
 * @brief PARIS Fixed-Lag Smoother Integration for Storvik Parameter Learning
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * ARCHITECTURE
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Standard Storvik uses FILTERED values (ℓ_t, ℓ_{t-1}) from forward pass.
 * This introduces bias because particles haven't yet seen future observations.
 *
 * PARIS Smoothed Storvik uses BACKWARD-SMOOTHED values (ℓ̃_t, ℓ̃_{t-1}) 
 * computed by running PARIS on an L-tick window.
 *
 * Result: Lower variance parameter estimates, faster convergence.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * STATE MACHINE
 * ═══════════════════════════════════════════════════════════════════════════
 *
 *   NORMAL ──[P² fire]──→ FLUSH ──→ COOLDOWN ──[buffer full]──→ NORMAL
 *                              │
 *                              └──[P² fire]──→ (ignore, stay in COOLDOWN)
 *                              │
 *                              └──[ESS < N/20]──→ RESET (wipe, stay in COOLDOWN)
 *
 * - P² = Circuit breaker (99.9th percentile surprise)
 * - FLUSH = Smooth partial buffer, update Storvik, reset
 * - RESET = Wipe buffer without smoothing (ESS collapse)
 * - COOLDOWN = Wait for buffer to refill before allowing next flush
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * Related files:
 *   - rbpf_ksc_param_integration.c  Core lifecycle + step function
 *   - rbpf_ext_hawkes.c             Hawkes + Robust OCSN
 *   - rbpf_fixed_lag_smoother.h     PARIS smoother wrapper
 * ═══════════════════════════════════════════════════════════════════════════
 */

#include "rbpf_ksc_param_integration.h"
#include "rbpf_fixed_lag_smoother.h"
#include <stdio.h>
#include <stdlib.h>

/*═══════════════════════════════════════════════════════════════════════════
 * BACKWARD COMPATIBILITY: fls_batch_update_storvik
 *
 * If your rbpf_fixed_lag_smoother.c doesn't have this function, we provide
 * it here. This extracts ALL smoothed (ℓ̃_t, ℓ̃_{t-1}) pairs from the buffer
 * and updates Storvik with each pair.
 *
 * Used during emergency flush when circuit breaker fires.
 *
 * NOTE: If you add this function to rbpf_fixed_lag_smoother.c later,
 *       remove this implementation to avoid duplicate symbol errors.
 *═══════════════════════════════════════════════════════════════════════════*/

int fls_batch_update_storvik(RBPF_FixedLagSmoother *fls, ParamLearner *learner)
{
    if (!fls || !learner || fls->count < 2) {
        return 0;
    }
    
    int updates = 0;
    int N = fls->n_particles;
    int N_pad = fls->n_particles_padded;
    
    /* Stack buffer for particle info - 1024 max particles */
    ParticleInfo info_stack[1024];
    ParticleInfo *info = (N <= 1024) ? info_stack : (ParticleInfo *)malloc(N * sizeof(ParticleInfo));
    if (!info) return 0;
    
    /* 
     * Iterate through buffer extracting consecutive pairs.
     * We need (ℓ̃_t, ℓ̃_{t-1}) for each t in the buffer.
     */
    for (int t = 1; t < fls->count; t++) {
        /* Get buffer indices for t and t-1 */
        int idx_curr = (fls->head - fls->count + t + fls->buffer_size) % fls->buffer_size;
        int idx_prev = (fls->head - fls->count + t - 1 + fls->buffer_size) % fls->buffer_size;
        
        for (int i = 0; i < N; i++) {
            int buf_curr = idx_curr * N_pad + i;
            int buf_prev = idx_prev * N_pad + i;
            
            info[i].regime = fls->regime_buffer[buf_curr];
            info[i].prev_regime = fls->regime_buffer[buf_prev];
            info[i].ell = (param_real)fls->h_buffer[buf_curr];      /* Smoothed ℓ̃_t */
            info[i].ell_lag = (param_real)fls->h_buffer[buf_prev];  /* Smoothed ℓ̃_{t-1} */
            
            /* Uniform weights for smoothed update */
            info[i].weight = 1.0 / N;
        }
        
        /* Update Storvik sufficient statistics */
        param_learn_update(learner, info, N);
        updates++;
    }
    
    if (info != info_stack) {
        free(info);
    }
    
    return updates;
}

/*═══════════════════════════════════════════════════════════════════════════
 * INTERNAL: BUFFER RESET (ESS Collapse)
 *
 * Wipes buffer without smoothing. Used when particle cloud degenerates
 * to near-singularity (ESS < N/20).
 *═══════════════════════════════════════════════════════════════════════════*/

static inline void smoother_reset_buffer(RBPF_Extended *ext)
{
    if (ext->smoother) {
        fls_reset(ext->smoother);
    }
    ext->reset_count++;
}

/*═══════════════════════════════════════════════════════════════════════════
 * INTERNAL: EMERGENCY FLUSH (Structural Break)
 *
 * When P² circuit breaker fires:
 *   1. Run PARIS on partial buffer (even if < L ticks)
 *   2. Update Storvik with all available smoothed pairs
 *   3. Reset buffer for new epoch
 *   4. Signal structural break to Storvik (slam λ to floor)
 *═══════════════════════════════════════════════════════════════════════════*/

static void smoother_emergency_flush(RBPF_Extended *ext)
{
    RBPF_FixedLagSmoother *s = ext->smoother;
    
    if (!s || s->count < 2) {
        /* Not enough data to smooth - just reset */
        smoother_reset_buffer(ext);
        return;
    }
    
    /*───────────────────────────────────────────────────────────────────────
     * 1. PARTIAL SMOOTHING PASS
     *
     * PARIS handles T < lag gracefully. Even 5-10 ticks gives useful
     * backward information to correct the initial crash detection.
     *───────────────────────────────────────────────────────────────────────*/
    fls_smooth(s);
    
    /*───────────────────────────────────────────────────────────────────────
     * 2. BATCH UPDATE STORVIK
     *
     * Extract ALL smoothed (ℓ̃, ℓ̃_lag) pairs from partial buffer.
     * This is the "last look" before resetting for new epoch.
     *───────────────────────────────────────────────────────────────────────*/
    fls_batch_update_storvik(s, &ext->storvik);
    
    /*───────────────────────────────────────────────────────────────────────
     * 3. RESET BUFFER
     *
     * Clear for new epoch. Next L ticks will accumulate fresh data.
     *───────────────────────────────────────────────────────────────────────*/
    fls_reset(s);
    
    /*───────────────────────────────────────────────────────────────────────
     * 4. SIGNAL STRUCTURAL BREAK
     *
     * Slam Storvik's λ to floor (0.95). New data gets maximum weight.
     *───────────────────────────────────────────────────────────────────────*/
    param_learn_signal_structural_break(&ext->storvik);
    
    ext->flush_count++;
}

/*═══════════════════════════════════════════════════════════════════════════
 * SMOOTHER STEP (Called from rbpf_ext_step)
 *
 * Implements the state machine:
 *   - NORMAL: Push data, run incremental smoothed update
 *   - FLUSH: On P² fire (if not in cooldown)
 *   - RESET: On ESS collapse
 *   - COOLDOWN: Decrement counter, ignore P² until buffer refills
 *═══════════════════════════════════════════════════════════════════════════*/

void rbpf_ext_smoother_step(RBPF_Extended *ext, const RBPF_KSC_Output *output)
{
    if (!ext || !ext->smoothed_storvik_enabled || !ext->smoother) {
        return;
    }
    
    RBPF_FixedLagSmoother *s = ext->smoother;
    RBPF_KSC *rbpf = ext->rbpf;
    
    /*═══════════════════════════════════════════════════════════════════════
     * 1. PUSH: Record current particle state (always cheap)
     *═══════════════════════════════════════════════════════════════════════*/
    fls_push(s,
             rbpf->mu,          /* h = log-volatility */
             rbpf->regime,      /* r = regime index */
             rbpf->log_weight,  /* log(w) */
             output->resampled ? rbpf->indices : NULL);
    
    /*═══════════════════════════════════════════════════════════════════════
     * 2. STATE MACHINE: Determine action
     *═══════════════════════════════════════════════════════════════════════*/
    
    int p2_triggered = ext->structural_break_signaled;
    int ess_collapsed = (output->ess < ext->ess_collapse_threshold);
    int buffer_ready = (s->count >= ext->min_buffer_for_flush);
    int in_cooldown = (ext->cooldown_remaining > 0);
    
    if (p2_triggered && !in_cooldown && buffer_ready) {
        /*═══════════════════════════════════════════════════════════════════
         * PATH A: EMERGENCY FLUSH (Structural Break)
         *
         * P² detected 99.9th percentile surprise. Current regime is dead.
         * Smooth partial buffer, archive to Storvik, reset for new epoch.
         *═══════════════════════════════════════════════════════════════════*/
        smoother_emergency_flush(ext);
        ext->cooldown_remaining = s->lag;  /* Start cooldown */
    }
    else if (ess_collapsed) {
        /*═══════════════════════════════════════════════════════════════════
         * PATH B: BUFFER RESET (Mathematical Degeneracy)
         *
         * ESS < N/20 means particle cloud collapsed to near-singularity.
         * History is biased beyond repair. Wipe without smoothing.
         *═══════════════════════════════════════════════════════════════════*/
        smoother_reset_buffer(ext);
    }
    /* 
     * PATH C: NORMAL - Use FILTERED Storvik (no PARIS overhead)
     *
     * The smoother buffer collects data. PARIS only runs on:
     *   - Emergency flush (structural break)
     *   - Periodic batch update (configurable, e.g., every L ticks)
     *
     * This is the HYBRID approach:
     *   - Real-time: Filtered Storvik (fast, slight bias)
     *   - Background: Smoothed corrections on flush
     *
     * The filtered path in rbpf_ext_step handles Storvik updates normally.
     * We only ADD smoothing value during structural breaks.
     */
    
    /*═══════════════════════════════════════════════════════════════════════
     * 3. COOLDOWN DECAY
     *
     * Prevents "flush cascade" during volatility waterfall.
     * Must wait for buffer to refill before allowing next flush.
     *═══════════════════════════════════════════════════════════════════════*/
    if (ext->cooldown_remaining > 0) {
        ext->cooldown_remaining--;
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * API: ENABLE SMOOTHED STORVIK
 *═══════════════════════════════════════════════════════════════════════════*/

int rbpf_ext_enable_smoothed_storvik(RBPF_Extended *ext, int lag)
{
    if (!ext) return -1;
    if (!ext->storvik_initialized) {
        /* Smoothed Storvik requires Storvik to be enabled */
        return -1;
    }
    if (lag < 5 || lag > 500) {
        /* Sanity bounds: [5, 500] ticks */
        return -1;
    }
    
    /* Destroy existing smoother if present */
    if (ext->smoother) {
        fls_destroy(ext->smoother);
        ext->smoother = NULL;
    }
    
    /* Create new smoother */
    ext->smoother = fls_create(
        ext->rbpf->n_particles,
        ext->rbpf->n_regimes,
        lag,
        (uint32_t)(ext->tick_count + 12345)  /* Seed from tick count */
    );
    
    if (!ext->smoother) {
        return -1;
    }
    
    /*───────────────────────────────────────────────────────────────────────
     * Sync model parameters from RBPF to smoother
     *
     * PARIS needs: transition matrix, μ_vol per regime, φ, σ_h
     *───────────────────────────────────────────────────────────────────────*/
    double mu_vol[RBPF_MAX_REGIMES];
    double trans_d[RBPF_MAX_REGIMES * RBPF_MAX_REGIMES];
    double phi = 0.0, sigma_h = 0.0;
    
    const int nr = ext->rbpf->n_regimes;
    
    for (int r = 0; r < nr; r++) {
        mu_vol[r] = (double)ext->rbpf->params[r].mu_vol;
        /* Average phi and sigma_h across regimes (they should be similar) */
        phi += (double)(1.0f - ext->rbpf->params[r].theta);
        sigma_h += (double)ext->rbpf->params[r].sigma_vol;
    }
    phi /= nr;
    sigma_h /= nr;
    
    /* Copy base transition matrix */
    for (int i = 0; i < nr * nr; i++) {
        trans_d[i] = (double)ext->base_trans_matrix[i];
    }
    
    fls_set_model(ext->smoother, trans_d, mu_vol, phi, sigma_h);
    
    /* Enable */
    ext->smoothed_storvik_enabled = 1;
    ext->smoothed_storvik_lag = lag;
    ext->cooldown_remaining = 0;
    
    return 0;
}

/*═══════════════════════════════════════════════════════════════════════════
 * API: DISABLE SMOOTHED STORVIK
 *═══════════════════════════════════════════════════════════════════════════*/

void rbpf_ext_disable_smoothed_storvik(RBPF_Extended *ext)
{
    if (!ext) return;
    
    if (ext->smoother) {
        fls_destroy(ext->smoother);
        ext->smoother = NULL;
    }
    
    ext->smoothed_storvik_enabled = 0;
    ext->cooldown_remaining = 0;
}

/*═══════════════════════════════════════════════════════════════════════════
 * API: CHECK IF ENABLED
 *═══════════════════════════════════════════════════════════════════════════*/

int rbpf_ext_is_smoothed_storvik_enabled(const RBPF_Extended *ext)
{
    if (!ext) return 0;
    return ext->smoothed_storvik_enabled;
}

/*═══════════════════════════════════════════════════════════════════════════
 * API: GET SMOOTHER STATS
 *═══════════════════════════════════════════════════════════════════════════*/

void rbpf_ext_get_smoother_stats(const RBPF_Extended *ext,
                                  uint64_t *flush_count,
                                  uint64_t *reset_count,
                                  double *avg_smooth_us,
                                  int *buffer_fill)
{
    if (!ext) {
        if (flush_count) *flush_count = 0;
        if (reset_count) *reset_count = 0;
        if (avg_smooth_us) *avg_smooth_us = 0.0;
        if (buffer_fill) *buffer_fill = 0;
        return;
    }
    
    if (flush_count) *flush_count = ext->flush_count;
    if (reset_count) *reset_count = ext->reset_count;
    
    if (avg_smooth_us) {
        if (ext->smoother && ext->smoother->smoothing_calls > 0) {
            *avg_smooth_us = ext->smoother->total_smooth_time_us / 
                             ext->smoother->smoothing_calls;
        } else {
            *avg_smooth_us = 0.0;
        }
    }
    
    if (buffer_fill) {
        if (ext->smoother) {
            *buffer_fill = ext->smoother->count;
        } else {
            *buffer_fill = 0;
        }
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * API: CONFIGURE SMOOTHER
 *═══════════════════════════════════════════════════════════════════════════*/

void rbpf_ext_configure_smoother(RBPF_Extended *ext,
                                  int min_buffer_for_flush,
                                  float ess_collapse_thresh)
{
    if (!ext) return;
    
    if (min_buffer_for_flush >= 2) {
        ext->min_buffer_for_flush = min_buffer_for_flush;
    }
    
    if (ess_collapse_thresh > 0) {
        ext->ess_collapse_threshold = ess_collapse_thresh;
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * DIAGNOSTICS: PRINT SMOOTHER CONFIG
 *═══════════════════════════════════════════════════════════════════════════*/

void rbpf_ext_print_smoother_config(const RBPF_Extended *ext)
{
    if (!ext) return;
    
    printf("\n  PARIS Fixed-Lag Smoother:\n");
    if (ext->smoothed_storvik_enabled && ext->smoother) {
        printf("    Enabled:         YES\n");
        printf("    Lag:             %d ticks\n", ext->smoothed_storvik_lag);
        printf("    Buffer fill:     %d / %d\n", 
               ext->smoother->count, ext->smoother->buffer_size);
        printf("    Min flush buf:   %d ticks\n", ext->min_buffer_for_flush);
        printf("    ESS collapse:    < %.1f\n", ext->ess_collapse_threshold);
        printf("    Cooldown:        %d ticks remaining\n", ext->cooldown_remaining);
        printf("    Flush count:     %llu\n", (unsigned long long)ext->flush_count);
        printf("    Reset count:     %llu\n", (unsigned long long)ext->reset_count);
        
        if (ext->smoother->smoothing_calls > 0) {
            double avg_us = ext->smoother->total_smooth_time_us / 
                            ext->smoother->smoothing_calls;
            printf("    Avg smooth time: %.2f μs\n", avg_us);
        }
    } else {
        printf("    Enabled:         NO (using filtered Storvik baseline)\n");
    }
}