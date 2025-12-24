/**
 * @file rbpf_fixed_lag_smoother.h
 * @brief Fixed-Lag PARIS Smoother for Storvik Parameter Learning
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * MOTIVATION: Filtering Bias in Storvik Updates
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Standard Storvik uses filtered h_t ~ p(h_t | y_{1:t}):
 *   z_t = h_t - φ·h_{t-1}
 *   S_z += λ·z_t,  S_zz += λ·z_t²
 *
 * Problem: Filtered h_t is noisy (doesn't see y_{t+1:T})
 * Result: Storvik "chases noise", parameters oscillate
 *
 * Solution: Use smoothed h̃_t ~ p(h_t | y_{1:T}):
 *   z̃_t = h̃_t - φ·h̃_{t-1}
 *   S_z += λ·z̃_t,  S_zz += λ·z̃_t²
 *
 * Smoothed h̃ has lower variance → stable parameter estimates
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * FIXED-LAG APPROXIMATION
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Full smoothing needs complete trajectory (offline).
 * Fixed-lag uses p(h_t | y_{1:t+L}) where L is small (50-100 ticks).
 *
 * For AR(1) with φ ≈ 0.97, information decays as φ^k:
 *   k=50:  φ^50 ≈ 0.22  (78% of smoothing benefit captured)
 *   k=100: φ^100 ≈ 0.05 (95% of smoothing benefit captured)
 *
 * Trade-off:
 *   - Larger L → better smoothing → more memory, latency
 *   - Smaller L → less smoothing → faster, less memory
 *
 * Recommended: L=50 for HFT (1 tick/sec → 50 sec delay, acceptable)
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * ARCHITECTURE
 * ═══════════════════════════════════════════════════════════════════════════
 *
 *   RBPF Forward Pass              Ring Buffer                PARIS
 *   ─────────────────              ───────────                ─────
 *   t=0: filter → push(h,r,w,a) → [0]
 *   t=1: filter → push(h,r,w,a) → [0][1]
 *   ...
 *   t=L: filter → push(h,r,w,a) → [0][1]...[L]
 *                                      ↓
 *                              Load to PARIS
 *                                      ↓
 *                              PARIS backward smooth
 *                                      ↓
 *                              Extract h̃_0, h̃_1
 *                                      ↓
 *                              Storvik update with smoothed z̃
 *                                      ↓
 *                              Pop [0], shift buffer
 *
 * ═══════════════════════════════════════════════════════════════════════════
 */

#ifndef RBPF_FIXED_LAG_SMOOTHER_H
#define RBPF_FIXED_LAG_SMOOTHER_H

#include "paris_mkl.h"
#include "rbpf_param_learn.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

    /*═══════════════════════════════════════════════════════════════════════════
     * CONFIGURATION
     *═══════════════════════════════════════════════════════════════════════════*/

#define FLS_MAX_LAG 200    /* Maximum smoothing lag */
#define FLS_DEFAULT_LAG 50 /* Default lag (good for φ=0.97) */
#define FLS_ALIGN 64       /* Memory alignment */

    /*═══════════════════════════════════════════════════════════════════════════
     * STRUCTURES
     *═══════════════════════════════════════════════════════════════════════════*/

    /**
     * Fixed-Lag Smoother State
     *
     * Maintains a sliding window of particle history for PARIS smoothing.
     */
    typedef struct RBPF_FixedLagSmoother
    {
        int enabled;            /* 0 = disabled, 1 = active */
        int lag;                /* L = smoothing lag */
        int n_particles;        /* N = number of particles */
        int n_particles_padded; /* N_padded for SIMD */
        int n_regimes;          /* K = number of regimes */

        /*───────────────────────────────────────────────────────────────────────
         * RING BUFFER
         *
         * Stores last (lag+1) ticks of particle state.
         * +1 because we need both h̃_{t-L} and h̃_{t-L-1} for z computation.
         *
         * Layout: [buffer_size × n_particles_padded]
         * Access: buffer[t * n_particles_padded + i]
         *───────────────────────────────────────────────────────────────────────*/
        int buffer_size; /* lag + 1 */
        int head;        /* Next write position (circular) */
        int count;       /* Number of valid entries */

        float *h_buffer;          /* [buffer_size × N_pad] log-volatility */
        int *regime_buffer;       /* [buffer_size × N_pad] regime indices */
        float *log_weight_buffer; /* [buffer_size × N_pad] log-weights */
        int *ancestor_buffer;     /* [buffer_size × N_pad] ancestor indices */

        /*───────────────────────────────────────────────────────────────────────
         * PARIS SMOOTHER
         *───────────────────────────────────────────────────────────────────────*/
        PARISMKLState *paris; /* PARIS-MKL state (reused) */

        /*───────────────────────────────────────────────────────────────────────
         * REORDER WORKSPACE (per-instance for thread safety)
         *
         * CRITICAL: These must be per-instance, not static!
         * MMPF runs multiple smoothers in parallel - shared buffers = race condition.
         *
         * Layout: [buffer_size × N] for PARIS (which uses unpadded N)
         *───────────────────────────────────────────────────────────────────────*/
        double *h_ordered; /* [buffer_size × N] reordered h */
        double *w_ordered; /* [buffer_size × N] reordered weights */
        int *r_ordered;    /* [buffer_size × N] reordered regimes */
        int *a_ordered;    /* [buffer_size × N] reordered ancestors */

        /*───────────────────────────────────────────────────────────────────────
         * SMOOTHED OUTPUT (for Storvik)
         *
         * After PARIS backward pass, these contain:
         *   smoothed_h[i]     = h̃_{t-L} for particle i
         *   smoothed_h_lag[i] = h̃_{t-L-1} for particle i
         *   smoothed_regime[i] = r_{t-L} for particle i
         *───────────────────────────────────────────────────────────────────────*/
        float *smoothed_h;      /* [N] h̃ at t-L */
        float *smoothed_h_lag;  /* [N] h̃ at t-L-1 */
        int *smoothed_regime;   /* [N] regime at t-L */
        float *smoothed_weight; /* [N] normalized weight at t-L */

        /*───────────────────────────────────────────────────────────────────────
         * PARTICLE INFO BRIDGE (for Storvik API)
         *───────────────────────────────────────────────────────────────────────*/
        ParticleInfo *particle_info; /* [N] reusable ParticleInfo array */

        /*───────────────────────────────────────────────────────────────────────
         * DIAGNOSTICS
         *───────────────────────────────────────────────────────────────────────*/
        uint64_t smoothing_calls;    /* Number of PARIS backward passes */
        uint64_t emergency_flushes;  /* Number of circuit breaker flushes */
        uint64_t ticks_processed;    /* Total ticks pushed */
        double total_smooth_time_us; /* Cumulative smoothing time */
        double last_smooth_time_us;  /* Last smoothing latency */

    } RBPF_FixedLagSmoother;

    /*═══════════════════════════════════════════════════════════════════════════
     * LIFECYCLE
     *═══════════════════════════════════════════════════════════════════════════*/

    /**
     * Create fixed-lag smoother
     *
     * @param n_particles   Number of particles
     * @param n_regimes     Number of regimes
     * @param lag           Smoothing lag L (50-100 recommended)
     * @param rng_seed      RNG seed for PARIS
     * @return              Smoother handle, or NULL on failure
     */
    RBPF_FixedLagSmoother *fls_create(int n_particles, int n_regimes,
                                      int lag, uint32_t rng_seed);

    /**
     * Destroy fixed-lag smoother
     */
    void fls_destroy(RBPF_FixedLagSmoother *fls);

    /**
     * Reset smoother state (clear buffer, keep configuration)
     */
    void fls_reset(RBPF_FixedLagSmoother *fls);

    /*═══════════════════════════════════════════════════════════════════════════
     * CONFIGURATION
     *═══════════════════════════════════════════════════════════════════════════*/
    /**
     * Set model parameters
     *
     * @param fls        Fixed-lag smoother
     * @param trans      Transition matrix [K×K] row-major
     * @param mu_vol     Per-regime log-vol means [K]
     * @param sigma_vol  Per-regime AR process noise [K]
     * @param phi        AR(1) persistence
     */
    void fls_set_model(RBPF_FixedLagSmoother *fls,
                       const double *trans,
                       const double *mu_vol,
                       const double *sigma_vol,
                       double phi);

    /**
     * Enable/disable smoother
     *
     * When disabled, push() is a no-op and ready() always returns 0.
     */
    void fls_enable(RBPF_FixedLagSmoother *fls, int enable);

    /*═══════════════════════════════════════════════════════════════════════════
     * MAIN API
     *═══════════════════════════════════════════════════════════════════════════*/

    /**
     * Push current tick's particle state to buffer
     *
     * Call this AFTER rbpf_ksc_step() to record particle state.
     *
     * @param fls           Smoother handle
     * @param h             Log-volatility means [N] (rbpf->mu)
     * @param regimes       Regime indices [N] (rbpf->regime)
     * @param log_weights   Log-weights [N] (rbpf->log_weight)
     * @param ancestors     Ancestor indices [N] (from resampling, or identity)
     */
    void fls_push(RBPF_FixedLagSmoother *fls,
                  const float *h,
                  const int *regimes,
                  const float *log_weights,
                  const int *ancestors);

    /**
     * Check if smoother has enough data
     *
     * @return 1 if buffer is full (can call fls_smooth), 0 otherwise
     */
    int fls_ready(const RBPF_FixedLagSmoother *fls);

    /**
     * Run PARIS backward smoothing and extract smoothed values
     *
     * Only call when fls_ready() returns 1.
     *
     * After this call:
     *   - fls->smoothed_h contains h̃_{t-L}
     *   - fls->smoothed_h_lag contains h̃_{t-L-1}
     *   - fls->smoothed_regime contains r_{t-L}
     *   - fls->particle_info is populated for Storvik
     *
     * @param fls   Smoother handle
     * @return      Smoothing time in microseconds
     */
    double fls_smooth(RBPF_FixedLagSmoother *fls);

    /**
     * Get smoothed ParticleInfo array for Storvik update
     *
     * Only valid after fls_smooth().
     * The returned array has:
     *   - ell = smoothed h̃_{t-L}
     *   - ell_lag = smoothed h̃_{t-L-1}
     *   - regime = r_{t-L}
     *   - weight = normalized weight
     *
     * @param fls   Smoother handle
     * @return      ParticleInfo array [N], or NULL if not ready
     */
    const ParticleInfo *fls_get_particle_info(const RBPF_FixedLagSmoother *fls);

    /**
     * Pop oldest entry from buffer
     *
     * Call this AFTER processing the smoothed output to advance the window.
     */
    void fls_pop(RBPF_FixedLagSmoother *fls);

    /*═══════════════════════════════════════════════════════════════════════════
     * CONVENIENCE: Combined smooth + update
     *═══════════════════════════════════════════════════════════════════════════*/

    /**
     * Run smoothed Storvik update (if ready)
     *
     * Combines fls_smooth() + param_learn_update() + fls_pop().
     * Call this after fls_push() each tick.
     *
     * @param fls       Smoother handle
     * @param learner   Storvik learner
     * @return          1 if update was performed, 0 if buffer not full yet
     */
    int fls_update_storvik(RBPF_FixedLagSmoother *fls, ParamLearner *learner);

    /*═══════════════════════════════════════════════════════════════════════════
     * EMERGENCY FLUSH (Circuit Breaker Integration)
     *═══════════════════════════════════════════════════════════════════════════*/

    /**
     * Batch update Storvik with ALL smoothed pairs from buffer
     *
     * After PARIS backward pass, extracts all available (ℓ̃_t, ℓ̃_{t-1}) pairs
     * and updates Storvik with each one. Used during emergency flush.
     *
     * @param fls       Smoother handle (must have run fls_smooth first)
     * @param learner   Storvik learner
     * @return          Number of updates performed
     */
    int fls_batch_update_storvik(RBPF_FixedLagSmoother *fls, ParamLearner *learner);

    /**
     * Emergency flush on circuit breaker trigger
     *
     * Called when P² detects structural break. Runs partial smoothing
     * over available buffer (count < lag), updates Storvik with ALL
     * available smoothed pairs, then resets buffer to start fresh epoch.
     *
     * Rationale: During flash crash, we want Storvik to see the break
     * ASAP, not L ticks later when the buffer fills again.
     *
     * @param fls       Smoother handle
     * @param learner   Storvik learner
     * @return          Number of Storvik updates performed
     */
    int fls_emergency_flush(RBPF_FixedLagSmoother *fls, ParamLearner *learner);

    /**
     * Check if emergency flush is recommended
     *
     * Returns 1 if buffer has enough data for a meaningful partial smooth.
     * Typically requires at least 5-10 ticks.
     *
     * @param fls   Smoother handle
     * @return      1 if flush would be useful, 0 otherwise
     */
    int fls_should_flush(const RBPF_FixedLagSmoother *fls);

    /*═══════════════════════════════════════════════════════════════════════════
     * DIAGNOSTICS
     *═══════════════════════════════════════════════════════════════════════════*/

    /**
     * Get smoothing statistics
     */
    void fls_get_stats(const RBPF_FixedLagSmoother *fls,
                       uint64_t *smoothing_calls,
                       uint64_t *ticks_processed,
                       double *avg_smooth_time_us);

    /**
     * Print smoother info
     */
    void fls_print_info(const RBPF_FixedLagSmoother *fls);

#ifdef __cplusplus
}
#endif

#endif /* RBPF_FIXED_LAG_SMOOTHER_H */