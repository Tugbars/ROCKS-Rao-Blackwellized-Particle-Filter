/**
 * @file hawkes_integrator.h
 * @brief Hawkes Process Oracle Trigger - Self-Contained MKL Implementation
 *
 * Self-exciting point process for Oracle triggering.
 * Detects sustained intensity elevation (plateaus) vs transient spikes.
 *
 * Model:
 *   λ(t) = μ + Σᵢ α·exp(-β·(t - tᵢ))·|rᵢ|
 *
 * Features:
 *   - MKL-accelerated kernel computation (vmsExp)
 *   - Integrated intensity over sliding window
 *   - Self-calibrating variance-based thresholds (z × σ)
 *   - Spike vs plateau detection via cumulative residual
 *   - Hysteresis state machine (high/low water marks)
 *   - Refractory period to prevent trigger spam
 *
 * This is the "Lead" signal in the dual-trigger Oracle system.
 * Outputs surprise in σ units for the dual-gate trigger.
 *
 * Thread Safety:
 *   Each HawkesIntegrator instance is NOT thread-safe.
 *   Use one instance per thread, or synchronize access externally.
 *
 * Reference: ORACLE_INTEGRATION_PLAN.md v1.6
 */

#ifndef HAWKES_INTEGRATOR_H
#define HAWKES_INTEGRATOR_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

/*═══════════════════════════════════════════════════════════════════════════
 * CROSS-PLATFORM ALIGNMENT MACROS
 *═══════════════════════════════════════════════════════════════════════════*/

/* MKL alignment */
#define HAWKES_ALIGN 64

/*
 * Cross-platform alignment for struct members.
 *
 * MSVC:  __declspec(align(x)) must come BEFORE the type
 * GCC:   __attribute__((aligned(x))) can come after the variable
 * C11:   alignas(x) works on both (preferred if available)
 *
 * For struct members, we use a typedef wrapper approach for portability.
 */
#if defined(_MSC_VER)
/* MSVC: Use __declspec(align) with typedef wrapper */
#define HAWKES_ALIGNAS(x) __declspec(align(x))
#define HAWKES_ALIGNED_ARRAY(type, name, size) \
    HAWKES_ALIGNAS(HAWKES_ALIGN)               \
    type name[size]
#elif defined(__GNUC__) || defined(__clang__)
/* GCC/Clang: Use __attribute__ */
#define HAWKES_ALIGNAS(x) __attribute__((aligned(x)))
#define HAWKES_ALIGNED_ARRAY(type, name, size) \
    type name[size] HAWKES_ALIGNAS(HAWKES_ALIGN)
#else
/* Fallback: No alignment (works but may be slower) */
#define HAWKES_ALIGNAS(x)
#define HAWKES_ALIGNED_ARRAY(type, name, size) type name[size]
#endif

    /*═══════════════════════════════════════════════════════════════════════════
     * COMPILE-TIME CONFIGURATION
     *═══════════════════════════════════════════════════════════════════════════*/

#define HAWKES_MAX_EVENTS 128       /* Event ring buffer size */
#define HAWKES_INTEG_MAX_WINDOW 256 /* Integration window max size */

/* Numerical stability constants */
#define HAWKES_EPSILON 1e-10f
#define HAWKES_MIN_SIGMA 1e-6f
#define HAWKES_MIN_INTENSITY 1e-4f
#define HAWKES_MAX_INTENSITY 10.0f
#define HAWKES_KERNEL_CUTOFF 1e-6f /* Prune events below this kernel value */

    /*═══════════════════════════════════════════════════════════════════════════
     * HAWKES ENGINE CONFIGURATION
     *═══════════════════════════════════════════════════════════════════════════*/

    /**
     * Core Hawkes parameters
     *
     * Branching ratio n = α/β must be < 1 for stationarity.
     * Half-life of excitation = ln(2)/β ≈ 0.693/β ticks.
     */
    typedef struct
    {
        float mu;              /* Baseline intensity */
        float alpha;           /* Excitation strength */
        float beta;            /* Decay rate */
        float event_threshold; /* |return| threshold to register event */
    } HawkesParams;

    /*═══════════════════════════════════════════════════════════════════════════
     * INTEGRATOR CONFIGURATION
     *═══════════════════════════════════════════════════════════════════════════*/

    /**
     * Full integrator configuration
     */
    typedef struct
    {

        /* ═══════════════════════════════════════════════════════════════════
         * HAWKES ENGINE
         * ═══════════════════════════════════════════════════════════════════ */
        HawkesParams hawkes;

        /* ═══════════════════════════════════════════════════════════════════
         * INTEGRATION WINDOW
         * ═══════════════════════════════════════════════════════════════════ */
        int window_size; /* Ticks to integrate over (50-100) */

        /* ═══════════════════════════════════════════════════════════════════
         * VARIANCE TRACKING (self-calibrating threshold)
         * ═══════════════════════════════════════════════════════════════════ */
        float ema_alpha;   /* EMA decay for μ and σ² (e.g., 0.01) */
        float sigma_floor; /* Minimum σ to prevent division issues */
        float sigma_cap;   /* Maximum σ to prevent blindness (Final Tier) */

        /* ═══════════════════════════════════════════════════════════════════
         * CUMULATIVE RESIDUAL (spike vs plateau)
         * ═══════════════════════════════════════════════════════════════════ */
        float residual_decay;     /* Per-tick decay of residual integral */
        float residual_threshold; /* Min residual to confirm plateau */

        /* ═══════════════════════════════════════════════════════════════════
         * HYSTERESIS THRESHOLDS
         * ═══════════════════════════════════════════════════════════════════ */
        float high_water_mark; /* σ level to arm trigger (e.g., 2.5) */
        float low_water_mark;  /* σ level to disarm (e.g., 1.5) */
        int min_ticks_armed;   /* Minimum ticks armed before trigger */

        /* ═══════════════════════════════════════════════════════════════════
         * ABSOLUTE PANIC (Final Tier - prevents blindness)
         * ═══════════════════════════════════════════════════════════════════ */
        float absolute_panic_intensity; /* Trigger regardless of σ if avg λ exceeds */
        float instant_spike_multiplier; /* Multiplier for instantaneous spike (e.g., 1.5) */
        bool use_absolute_panic;        /* Enable absolute panic gate */

        /* ═══════════════════════════════════════════════════════════════════
         * REFRACTORY PERIOD
         * ═══════════════════════════════════════════════════════════════════ */
        int refractory_ticks; /* Minimum ticks between triggers */

        /* ═══════════════════════════════════════════════════════════════════
         * WARMUP
         * ═══════════════════════════════════════════════════════════════════ */
        int warmup_ticks; /* Ticks before triggering enabled */

    } HawkesIntegratorConfig;

    /*═══════════════════════════════════════════════════════════════════════════
     * STATE STRUCTURES
     *═══════════════════════════════════════════════════════════════════════════*/

    /**
     * Hysteresis state machine states
     */
    typedef enum
    {
        HAWKES_TRIG_IDLE,       /* Below high water, not watching */
        HAWKES_TRIG_ARMED,      /* Above high water, waiting for plateau */
        HAWKES_TRIG_FIRED,      /* Just triggered (transient) */
        HAWKES_TRIG_REFRACTORY, /* In refractory period */
    } HawkesTriggerState;

    /**
     * Main integrator state
     *
     * Memory layout optimized for cache efficiency:
     *   - Hot path data first (current values, counters)
     *   - Config and event buffer after
     */
    typedef struct
    {

        /* ═══════════════════════════════════════════════════════════════════
         * HOT PATH - Current tick data (fits in ~2 cache lines)
         * ═══════════════════════════════════════════════════════════════════ */
        float current_time;           /* Current tick */
        float current_intensity;      /* λ(t) instantaneous */
        float current_avg_intensity;  /* ∫λ/window (smoothed) */
        float current_surprise_sigma; /* (avg_λ - μ_λ) / σ_λ */
        float current_residual;       /* Cumulative residual */
        float last_predicted;         /* λ_pred from previous tick */
        HawkesTriggerState trigger_state;
        bool trigger_fired_this_tick;
        int ticks_armed;
        int ticks_since_trigger;
        int total_ticks;
        bool warmed_up;

        /* ═══════════════════════════════════════════════════════════════════
         * VARIANCE TRACKING
         * ═══════════════════════════════════════════════════════════════════ */
        float lambda_ema;     /* μ_λ: baseline reference */
        float lambda_var_ema; /* σ²_λ: variance estimate */
        float lambda_sigma;   /* σ_λ: sqrt of variance (cached) */

        /* ═══════════════════════════════════════════════════════════════════
         * CUMULATIVE RESIDUAL (spike vs plateau detection)
         * ═══════════════════════════════════════════════════════════════════ */
        float residual_integral; /* ∫(λ_obs - λ_pred)dt */

        /* ═══════════════════════════════════════════════════════════════════
         * INTEGRATION WINDOW (ring buffer)
         * ═══════════════════════════════════════════════════════════════════ */
        float intensity_sum; /* Running sum for O(1) average */
        int window_head;
        int window_count;

        /* ═══════════════════════════════════════════════════════════════════
         * EVENT BUFFER (ring buffer) - Separate arrays for MKL vectorization
         * ═══════════════════════════════════════════════════════════════════ */
        int event_head;   /* Next write position */
        int event_count;  /* Number of valid events */
        int total_events; /* Lifetime event count */

        /* ═══════════════════════════════════════════════════════════════════
         * STATISTICS
         * ═══════════════════════════════════════════════════════════════════ */
        int total_triggers;
        int triggers_by_hysteresis;
        int triggers_by_panic;
        int disarms_by_low_water;
        float sum_armed_duration; /* For average calculation */

        /* ═══════════════════════════════════════════════════════════════════
         * CONFIGURATION (read-only after init)
         * ═══════════════════════════════════════════════════════════════════ */
        HawkesIntegratorConfig config;

        /* ═══════════════════════════════════════════════════════════════════
         * ALIGNED ARRAYS (at end for alignment, accessed via MKL)
         * ═══════════════════════════════════════════════════════════════════ */
        HAWKES_ALIGNED_ARRAY(float, intensity_window, HAWKES_INTEG_MAX_WINDOW);
        HAWKES_ALIGNED_ARRAY(float, event_times, HAWKES_MAX_EVENTS);
        HAWKES_ALIGNED_ARRAY(float, event_marks, HAWKES_MAX_EVENTS);
        HAWKES_ALIGNED_ARRAY(float, scratch_dt, HAWKES_MAX_EVENTS);
        HAWKES_ALIGNED_ARRAY(float, scratch_kernel, HAWKES_MAX_EVENTS);

    } HawkesIntegrator;

    /*═══════════════════════════════════════════════════════════════════════════
     * RESULT STRUCTURE
     *═══════════════════════════════════════════════════════════════════════════*/

    /**
     * Result from each update tick
     */
    typedef struct
    {
        /* Primary output for dual-gate */
        bool should_trigger;  /* Fire the Oracle? */
        float surprise_sigma; /* Hawkes surprise in σ units */

        /* Secondary signals */
        float integrated_intensity;    /* Smoothed λ over window */
        float instantaneous_intensity; /* Raw λ(t) */
        float residual;                /* Spike vs plateau signal */

        /* State info */
        HawkesTriggerState state; /* Current FSM state */
        bool is_valid;            /* False if in refractory/warmup */
        bool triggered_by_panic;  /* True if absolute panic fired */

        /* Diagnostics */
        float baseline_ema; /* μ_λ */
        float sigma;        /* σ_λ (possibly capped) */
        int ticks_armed;    /* How long armed */
        int active_events;  /* Events contributing to intensity */

    } HawkesIntegratorResult;

    /*═══════════════════════════════════════════════════════════════════════════
     * API - LIFECYCLE
     *═══════════════════════════════════════════════════════════════════════════*/

    /**
     * Get default configuration
     */
    HawkesIntegratorConfig hawkes_integrator_config_defaults(void);

    /**
     * Get configuration tuned for fast response
     */
    HawkesIntegratorConfig hawkes_integrator_config_responsive(void);

    /**
     * Get configuration tuned for conservative triggering
     */
    HawkesIntegratorConfig hawkes_integrator_config_conservative(void);

    /**
     * Initialize integrator
     *
     * @param integ     Integrator state to initialize
     * @param cfg       Configuration (NULL for defaults)
     * @return 0 on success, -1 on error
     */
    int hawkes_integrator_init(HawkesIntegrator *integ,
                               const HawkesIntegratorConfig *cfg);

    /**
     * Reset state (keep config)
     */
    void hawkes_integrator_reset(HawkesIntegrator *integ);

    /**
     * Free resources (currently no-op, but future-proof)
     */
    void hawkes_integrator_free(HawkesIntegrator *integ);

    /*═══════════════════════════════════════════════════════════════════════════
     * API - CORE UPDATE
     *═══════════════════════════════════════════════════════════════════════════*/

    /**
     * Process one tick and check for trigger
     *
     * @param integ         Integrator state
     * @param time          Current time (tick index)
     * @param obs_return    Observed return (raw, not absolute)
     * @return Result with trigger decision and diagnostics
     */
    HawkesIntegratorResult hawkes_integrator_update(HawkesIntegrator *integ,
                                                    float time,
                                                    float obs_return);

    /**
     * Batch update (more efficient for backtesting)
     *
     * @param integ         Integrator state
     * @param returns       Array of returns [n]
     * @param n             Number of observations
     * @param out_results   Output results array [n] (can be NULL)
     * @return Number of triggers fired
     */
    int hawkes_integrator_update_batch(HawkesIntegrator *integ,
                                       const float *returns,
                                       int n,
                                       HawkesIntegratorResult *out_results);

    /*═══════════════════════════════════════════════════════════════════════════
     * API - QUERIES (no state mutation)
     *═══════════════════════════════════════════════════════════════════════════*/

    /**
     * Get current surprise in σ units
     */
    float hawkes_integrator_get_surprise(const HawkesIntegrator *integ);

    /**
     * Get current integrated intensity
     */
    float hawkes_integrator_get_intensity(const HawkesIntegrator *integ);

    /**
     * Get instantaneous intensity
     */
    float hawkes_integrator_get_instant_intensity(const HawkesIntegrator *integ);

    /**
     * Get current trigger state
     */
    HawkesTriggerState hawkes_integrator_get_state(const HawkesIntegrator *integ);

    /**
     * Check if integrator is ready (past warmup, not in refractory)
     */
    bool hawkes_integrator_is_ready(const HawkesIntegrator *integ);

    /**
     * Get branching ratio n = α/β
     * Must be < 1 for stationary process
     */
    float hawkes_integrator_get_branching_ratio(const HawkesIntegrator *integ);

    /**
     * Get excitation half-life in ticks
     */
    float hawkes_integrator_get_half_life(const HawkesIntegrator *integ);

    /**
     * Predict intensity at future time (assuming no new events)
     */
    float hawkes_integrator_predict(const HawkesIntegrator *integ, float future_time);

    /*═══════════════════════════════════════════════════════════════════════════
     * API - MANUAL CONTROL
     *═══════════════════════════════════════════════════════════════════════════*/

    /**
     * Force trigger (for external override)
     */
    void hawkes_integrator_force_trigger(HawkesIntegrator *integ);

    /**
     * Clear refractory period (for emergency re-trigger)
     */
    void hawkes_integrator_clear_refractory(HawkesIntegrator *integ);

    /**
     * Adjust hysteresis thresholds dynamically
     */
    void hawkes_integrator_set_thresholds(HawkesIntegrator *integ,
                                          float high_water,
                                          float low_water);

    /**
     * Adjust Hawkes parameters dynamically
     */
    void hawkes_integrator_set_params(HawkesIntegrator *integ,
                                      float mu,
                                      float alpha,
                                      float beta);

    /*═══════════════════════════════════════════════════════════════════════════
     * API - DIAGNOSTICS
     *═══════════════════════════════════════════════════════════════════════════*/

    /**
     * Trigger statistics
     */
    typedef struct
    {
        int total_triggers;
        int triggers_hysteresis;
        int triggers_panic;
        int disarms_low_water;
        int total_events;
        float trigger_rate; /* Triggers per 1000 ticks */
        float event_rate;   /* Events per 1000 ticks */
        float avg_armed_duration;
    } HawkesIntegratorStats;

    /**
     * Get statistics
     */
    void hawkes_integrator_get_stats(const HawkesIntegrator *integ,
                                     HawkesIntegratorStats *stats);

    /**
     * Print current state
     */
    void hawkes_integrator_print_state(const HawkesIntegrator *integ);

    /**
     * Print configuration
     */
    void hawkes_integrator_print_config(const HawkesIntegratorConfig *cfg);

#ifdef __cplusplus
}
#endif

#endif /* HAWKES_INTEGRATOR_H */