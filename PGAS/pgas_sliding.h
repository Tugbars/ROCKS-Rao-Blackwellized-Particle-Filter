/**
 * @file pgas_sliding.h
 * @brief Sliding window wrapper for PGAS-MKL
 *
 * Architecture:
 *   - Tick loop (Core 0-1): Calls pgas_sliding_push() at each tick
 *   - PGAS loop (Cores 2-31): Runs sweeps, publishes Π, slides window
 *
 * The wrapper manages:
 *   - Ring buffer for continuous observation ingestion
 *   - Window extraction into underlying PGASMKLState
 *   - Reference trajectory sliding (warm start)
 *   - Lock-free communication between tick and PGAS loops
 *
 * Usage:
 *   1. Allocate: pgas_sliding_alloc(window=2000, slide=500, N=256, K=4, seed)
 *   2. Configure: pgas_sliding_set_model(...)
 *   3. Tick loop: pgas_sliding_push(obs, tick) for each observation
 *   4. PGAS loop:
 *        while (running) {
 *            if (pgas_sliding_window_ready(state)) {
 *                pgas_sliding_iterate(state, 5, pi_out);
 *                // Publish pi_out to channel
 *            }
 *        }
 */

#ifndef PGAS_SLIDING_H
#define PGAS_SLIDING_H

#include <stdint.h>
#include <stdbool.h>
#include "pgas_compat.h" /* Cross-platform atomics */
#include "pgas_mkl.h"    /* For PGASMKLState */

#ifdef __cplusplus
extern "C"
{
#endif

    /*═══════════════════════════════════════════════════════════════════════════════
     * CONFIGURATION
     *═══════════════════════════════════════════════════════════════════════════════*/

#define PGAS_SLIDING_MAX_K 8
#define PGAS_SLIDING_RING_MULT 4 /* Ring capacity = window_size * RING_MULT */

    /*═══════════════════════════════════════════════════════════════════════════════
     * SLIDING WINDOW STATE
     *═══════════════════════════════════════════════════════════════════════════════*/

    typedef struct PGASSlidingState
    {
        /* ═══════════════════════════════════════════════════════════════════════
         * OWNED PGAS STATE
         * ═══════════════════════════════════════════════════════════════════════*/
        PGASMKLState *pgas;

        /* ═══════════════════════════════════════════════════════════════════════
         * RING BUFFER (Lock-free, single producer / single consumer)
         *
         * Tick loop writes at ring_write_idx (atomic increment)
         * PGAS loop reads from ring_read_idx up to ring_write_idx
         * ═══════════════════════════════════════════════════════════════════════*/
        float *obs_ring;    /* Circular buffer for observations */
        int64_t *tick_ring; /* Corresponding tick numbers */
        int ring_capacity;  /* Power of 2 for fast modulo */
        int ring_mask;      /* ring_capacity - 1 */

        pgas_atomic_int64 ring_write_idx; /* Next write position (monotonic) */
        int64_t ring_read_idx;            /* Next read position (PGAS loop only) */

        /* ═══════════════════════════════════════════════════════════════════════
         * WINDOW CONFIGURATION
         * ═══════════════════════════════════════════════════════════════════════*/
        int window_size; /* PGAS window length T (e.g., 2000) */
        int slide_step;  /* Slide amount S (e.g., 500) */
        int warmup_size; /* Min observations before first run */

        /* ═══════════════════════════════════════════════════════════════════════
         * WINDOW STATE
         * ═══════════════════════════════════════════════════════════════════════*/
        int64_t window_start_tick; /* Global tick at window[0] */
        int64_t window_end_tick;   /* Global tick at window[T-1] */
        int windows_completed;     /* Number of windows processed */

        /* ═══════════════════════════════════════════════════════════════════════
         * ADAPTIVE SLIDE CONTROL
         *
         * Instead of fixed slide_step, we adapt based on SR statistic:
         *   - SR low  → slide = slide_normal (save compute)
         *   - SR high → slide = slide_fast (faster Π updates)
         *
         * This is safer than dynamic plasticity because false alarms
         * just mean extra computation, not memory destruction.
         * ═══════════════════════════════════════════════════════════════════════*/
        int adaptive_slide_enabled; /**< 1 = enabled (default), 0 = use fixed slide_step */
        int slide_normal;           /**< Slide for SR < threshold_elevated (default: slide_step) */
        int slide_elevated;         /**< Slide for SR in [elevated, fast) (default: slide_step/2) */
        int slide_fast;             /**< Slide for SR >= threshold_fast (default: slide_step/3) */
        int effective_slide_step;   /**< Current slide being used */

        float sr_threshold_elevated; /**< SR threshold for elevated mode (default: 1.0) */
        float sr_threshold_fast;     /**< SR threshold for fast mode (default: 3.0) */
        float last_sr_stat;          /**< Last SR value received */

        /* Hysteresis */
        int ticks_in_current_mode;   /**< How long we've been in current mode */
        int min_ticks_before_loosen; /**< Minimum ticks before allowing loosening */

        /* Statistics */
        int adaptive_mode_switches; /**< Total mode switch count */
        int ticks_in_fast_mode;     /**< Cumulative time in fast mode */
        int ticks_in_elevated_mode; /**< Cumulative time in elevated mode */
        int ticks_in_normal_mode;   /**< Cumulative time in normal mode */

        /* ═══════════════════════════════════════════════════════════════════════
         * REFERENCE TRAJECTORY (Warm Start)
         *
         * We work directly on pgas->ref_* buffers (in-place).
         * When sliding, we shift left and extend with propagated states.
         * ═══════════════════════════════════════════════════════════════════════*/
        bool has_reference; /* Valid reference from previous sweep? */

        /* ═══════════════════════════════════════════════════════════════════════
         * OPTIMIZATION: Workspaces for vectorized tail propagation
         * Pre-allocated to avoid allocs in hot path
         * ═══════════════════════════════════════════════════════════════════════*/
        float *ws_rng_uniform; /* Size: slide_step (regime transitions) */
        float *ws_rng_normal;  /* Size: slide_step (AR noise) */

        /* ═══════════════════════════════════════════════════════════════════════
         * OUTPUT (Last completed sweep)
         * ═══════════════════════════════════════════════════════════════════════*/
        float pi[PGAS_SLIDING_MAX_K * PGAS_SLIDING_MAX_K];
        float last_acceptance_rate;
        int last_sweeps_used;
        int64_t last_window_end_tick;

        /* ═══════════════════════════════════════════════════════════════════════
         * DIAGNOSTICS
         * ═══════════════════════════════════════════════════════════════════════*/
        int64_t total_observations;
        int64_t dropped_observations; /* Ring overflow */
        float avg_acceptance_rate;

    } PGASSlidingState;

    /*═══════════════════════════════════════════════════════════════════════════════
     * LIFECYCLE
     *═══════════════════════════════════════════════════════════════════════════════*/

    /**
     * Allocate sliding window PGAS state
     *
     * @param window_size  PGAS window length T (e.g., 2000)
     * @param slide_step   Observations to slide by after each sweep (e.g., 500)
     * @param N            Number of particles
     * @param K            Number of regimes
     * @param seed         RNG seed
     * @return             Allocated state, or NULL on failure
     */
    PGASSlidingState *pgas_sliding_alloc(int window_size, int slide_step,
                                         int N, int K, uint32_t seed);

    /**
     * Free sliding window state and owned PGAS state
     */
    void pgas_sliding_free(PGASSlidingState *state);

    /*═══════════════════════════════════════════════════════════════════════════════
     * MODEL CONFIGURATION
     *═══════════════════════════════════════════════════════════════════════════════*/

    /**
     * Set model parameters (delegates to pgas_mkl_set_model)
     */
    void pgas_sliding_set_model(PGASSlidingState *state,
                                const double *trans,
                                const double *mu_vol,
                                const double *sigma_vol,
                                double phi);

    /**
     * Set sticky prior parameters
     */
    void pgas_sliding_set_prior(PGASSlidingState *state,
                                float alpha, float kappa);

    /**
     * Set recency weighting (0 = disabled, 0.001 = half-life ~693 ticks)
     */
    void pgas_sliding_set_recency(PGASSlidingState *state, float lambda);

    /*═══════════════════════════════════════════════════════════════════════════════
     * OBSERVATION INPUT (Tick Loop - Lock-Free)
     *═══════════════════════════════════════════════════════════════════════════════*/

    /**
     * Push observation into ring buffer (called from tick loop)
     *
     * Thread-safe: Single producer (tick loop), single consumer (PGAS loop).
     * Lock-free: Uses atomic write index.
     *
     * @param state  Sliding state
     * @param obs    Observation value (log-return squared, etc.)
     * @param tick   Global tick number (must be monotonically increasing)
     * @return       true if accepted, false if ring overflow
     */
    bool pgas_sliding_push(PGASSlidingState *state, float obs, int64_t tick);

    /*═══════════════════════════════════════════════════════════════════════════════
     * PGAS LOOP OPERATIONS
     *═══════════════════════════════════════════════════════════════════════════════*/

    /**
     * Check if enough observations for next window
     *
     * First window: needs window_size observations
     * Subsequent: needs slide_step new observations since last window
     */
    bool pgas_sliding_window_ready(const PGASSlidingState *state);

    /**
     * Extract observations from ring buffer into PGAS state
     *
     * Call this when pgas_sliding_window_ready() returns true.
     * Also sets up reference trajectory from previous sweep (warm start).
     */
    void pgas_sliding_extract_window(PGASSlidingState *state);

    /**
     * Run PGAS Gibbs sweeps on current window
     *
     * @param state     Sliding state
     * @param n_sweeps  Number of Gibbs sweeps to run
     * @return          Final acceptance rate
     */
    float pgas_sliding_run_sweeps(PGASSlidingState *state, int n_sweeps);

    /**
     * Get estimated transition matrix from last sweep
     *
     * @param state   Sliding state
     * @param pi_out  Output buffer [K*K], row-major
     */
    void pgas_sliding_get_pi(const PGASSlidingState *state, float *pi_out);

    /**
     * Slide window forward after sweep completion
     *
     * This:
     *   1. Saves current reference trajectory
     *   2. Advances ring_read_idx by slide_step
     *   3. Shifts saved reference for warm start
     */
    void pgas_sliding_slide(PGASSlidingState *state);

    /*═══════════════════════════════════════════════════════════════════════════════
     * COMBINED OPERATIONS
     *═══════════════════════════════════════════════════════════════════════════════*/

    /**
     * Complete PGAS iteration: extract → run sweeps → get Π → slide
     *
     * Convenience function for typical PGAS loop:
     *
     *   while (running) {
     *       if (pgas_sliding_window_ready(state)) {
     *           pgas_sliding_iterate(state, 5, pi_out);
     *           // Publish pi_out to channel
     *       }
     *   }
     *
     * @param state     Sliding state
     * @param n_sweeps  Number of Gibbs sweeps
     * @param pi_out    Output transition matrix [K*K], or NULL to use internal
     * @return          Acceptance rate, or -1.0f if window not ready
     */
    float pgas_sliding_iterate(PGASSlidingState *state, int n_sweeps, float *pi_out);

    /*═══════════════════════════════════════════════════════════════════════════════
     * DIAGNOSTICS
     *═══════════════════════════════════════════════════════════════════════════════*/

    /**
     * Get number of observations currently in ring buffer
     */
    int64_t pgas_sliding_ring_count(const PGASSlidingState *state);

    /**
     * Get number of observations available for next window
     */
    int64_t pgas_sliding_available(const PGASSlidingState *state);

    /**
     * Print diagnostic information
     */
    void pgas_sliding_print_diagnostics(const PGASSlidingState *state);

    /*═══════════════════════════════════════════════════════════════════════════════
     * ADAPTIVE SLIDE CONTROL
     *═══════════════════════════════════════════════════════════════════════════════*/

    /**
     * @brief Enable/disable adaptive slide
     *
     * When disabled, uses fixed slide_step.
     * When enabled, adapts slide based on SR statistic.
     */
    void pgas_sliding_set_adaptive_slide(PGASSlidingState *state, int enabled);

    /**
     * @brief Set SR thresholds for adaptive slide
     *
     * @param sr_elevated  SR threshold for elevated mode (default: 1.0)
     * @param sr_fast      SR threshold for fast mode (default: 3.0)
     */
    void pgas_sliding_set_sr_thresholds(PGASSlidingState *state,
                                        float sr_elevated, float sr_fast);

    /**
     * @brief Update adaptive slide based on current SR statistic
     *
     * Call this from the main loop whenever you have a new SR value.
     * The effective slide will be adjusted based on SR level.
     *
     * @param sr_stat  Current SR statistic from Hawkes integrator
     */
    void pgas_sliding_update_sr(PGASSlidingState *state, float sr_stat);

    /**
     * @brief Get current effective slide interval
     */
    int pgas_sliding_get_effective_slide(const PGASSlidingState *state);

    /**
     * @brief Print adaptive slide statistics
     */
    void pgas_sliding_print_adaptive_slide(const PGASSlidingState *state);

    /*═══════════════════════════════════════════════════════════════════════════════
     * BOUNDED-MEMORY BAYESIAN ESTIMATION
     *═══════════════════════════════════════════════════════════════════════════════*/

    /**
     * @brief Set memory decay for accumulated counts (forwards to pgas_mkl)
     *
     * @param decay  Per-window decay factor (0.99 default)
     * @param floor  Minimum count floor (0.5 default)
     */
    void pgas_sliding_set_memory_decay(PGASSlidingState *state, float decay, float floor);

    /**
     * @brief Set maximum inertia for accumulated counts (forwards to pgas_mkl)
     *
     * Controls agility: lower = more responsive to new data
     *   - 300 = Agile (20 ticks = 6.7% impact) [RECOMMENDED]
     */
    void pgas_sliding_set_max_inertia(PGASSlidingState *state, float max_inertia);

    /**
     * @brief Enable/disable adaptive kappa (anti-chattering)
     *
     * When enabled, sticky_kappa is dynamically adjusted based on observed
     * chatter rate using RLS estimation.
     *
     * @param enabled  1 = enabled, 0 = disabled (default)
     */
    void pgas_sliding_set_adaptive_kappa(PGASSlidingState *state, int enabled);

#ifdef __cplusplus
}
#endif

#endif /* PGAS_SLIDING_H */