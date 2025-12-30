/*
 * crisis_detector.h - Integration Layer for Crisis Detection
 *
 * Combines:
 *   - EventDetector (P² quantile-based event detection)
 *   - HawkesIntegrator (existing self-exciting point process)
 *   - DualSR (Shiryaev-Roberts change-point detection)
 *   - AdaptiveThreshold (breathing threshold with wolf penalty)
 *
 * This is the v3.1 "two-stage detonator" from PGAS_RBPF_INTEGRATION_PLAN.
 *
 * Architecture:
 *   EventDetector → HawkesIntegrator → DualSR → State Machine
 *                                         ↑
 *                              AdaptiveThreshold
 *
 * Key insight: EventDetector gates what Hawkes sees as "events".
 * Instead of using Hawkes' fixed event_threshold, we use P² quantile
 * to dynamically determine what constitutes a meaningful return.
 *
 * Usage:
 *   CrisisDetector cd;
 *   crisis_detector_init(&cd, NULL);
 *
 *   for each tick:
 *       CrisisState state = crisis_detector_update(&cd, obs, tick);
 *
 *       switch (state) {
 *           case CRISIS_IDLE:       // Normal operation
 *           case CRISIS_ALERT:      // Hawkes armed, SR accumulating
 *           case CRISIS_ACTIVE:     // Crisis confirmed
 *           case CRISIS_RECOVERING: // Exiting crisis
 *       }
 */

#ifndef CRISIS_DETECTOR_H
#define CRISIS_DETECTOR_H

#include "event_detector.h"
#include "hawkes_integrator.h"
#include "sr_detector.h"
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

    /* ═══════════════════════════════════════════════════════════════════════════
     * CRISIS STATE
     * ═══════════════════════════════════════════════════════════════════════════ */

    typedef enum
    {
        CRISIS_IDLE = 0,  /* Normal market, monitoring */
        CRISIS_ALERT,     /* Hawkes armed OR SR elevated, watching closely */
        CRISIS_ACTIVE,    /* Crisis confirmed, regime change in progress */
        CRISIS_RECOVERING /* Exiting crisis, both SR active for hysteresis */
    } CrisisState;

    static inline const char *crisis_state_name(CrisisState state)
    {
        switch (state)
        {
        case CRISIS_IDLE:
            return "IDLE";
        case CRISIS_ALERT:
            return "ALERT";
        case CRISIS_ACTIVE:
            return "ACTIVE";
        case CRISIS_RECOVERING:
            return "RECOVERING";
        default:
            return "UNKNOWN";
        }
    }

    /* ═══════════════════════════════════════════════════════════════════════════
     * CONFIGURATION
     * ═══════════════════════════════════════════════════════════════════════════ */

    typedef struct
    {
        /* Event detector config */
        EventDetectorConfig event_cfg;

        /* Hawkes config (uses existing hawkes_integrator_config_*) */
        HawkesIntegratorConfig hawkes_cfg;

        /* SR config */
        SRConfig sr_cfg;

        /* Adaptive threshold config */
        AdaptiveThresholdConfig threshold_cfg;

        /* State machine parameters */
        float sr_alert_fraction;   /* Enter ALERT when SR > fraction × threshold (default: 0.5) */
        float sr_exit_threshold;   /* log(H) for SR_down exit detection (default: 3.0 = 20:1 odds) */
        float sr_reentry_fraction; /* Re-trigger if SR_up > fraction × threshold in RECOVERING (default: 0.7) */

        /* ═══════════════════════════════════════════════════════════════════
         * FSM REFINEMENTS (v2)
         * ═══════════════════════════════════════════════════════════════════ */

        /* Confirmation window: require N consecutive ticks above threshold */
        int confirmation_ticks; /* Ticks SR must stay > H to confirm ACTIVE (default: 3) */

        /* Minimum hold time: force ACTIVE for at least T ticks */
        int min_hold_active; /* Min ticks in ACTIVE before exit allowed (default: 50) */

        /* Cooldown after false alarm */
        int cooldown_ticks;     /* Ticks to ignore hawkes after false alarm (default: 100) */
        float nuclear_override; /* SR multiple to override cooldown (default: 2.0 = 2×H) */

        /* Crisis sigma learning */
        float sigma_crisis_ema_alpha; /* EMA alpha for learning sigma during crisis (default: 0.01) */

        /* Initial sigma (before PGAS provides estimate) */
        float initial_sigma_peace; /* Default: 0.01 (1%) */

        /* Warmup */
        int warmup_ticks; /* Ticks before detection enabled (default: 200) */

        /* ═══════════════════════════════════════════════════════════════════
         * ROBUST WARMUP (v3) - Handles cold start during crisis
         * ═══════════════════════════════════════════════════════════════════ */
        bool use_robust_warmup;   /* Enable outlier rejection during warmup (default: true) */
        float warmup_outlier_k;   /* Reject |obs| > k × MAD during warmup (default: 3.0) */
        float warmup_winsorize_k; /* Winsorize at k × MAD during warmup (default: 4.0) */
        int warmup_min_clean;     /* Min clean observations before baseline set (default: 50) */

        /* ═══════════════════════════════════════════════════════════════════
         * SANITY ANCHOR (v3.1) - Fundamental prior / circuit breaker
         * ═══════════════════════════════════════════════════════════════════ */
        float max_peace_sigma; /* Hard limit: σ above this is NEVER peace (default: 0.002 = 20bps) */
                               /* Derived from asset fundamentals, not learned */

    } CrisisDetectorConfig;

    CrisisDetectorConfig crisis_detector_config_default(void);
    CrisisDetectorConfig crisis_detector_config_sensitive(void);
    CrisisDetectorConfig crisis_detector_config_conservative(void);

    /* ═══════════════════════════════════════════════════════════════════════════
     * CRISIS DETECTOR
     * ═══════════════════════════════════════════════════════════════════════════ */

    typedef struct
    {
        /* Components */
        EventDetector event_det;
        HawkesIntegrator hawkes;
        DualSR dual_sr;
        AdaptiveThreshold adaptive_th;

        /* State */
        CrisisState state;
        int64_t tick_count;
        int64_t ticks_in_state;
        int64_t last_state_change_tick;

        /* ═══════════════════════════════════════════════════════════════════
         * FSM REFINEMENT STATE (v2)
         * ═══════════════════════════════════════════════════════════════════ */

        /* Confirmation tracking: consecutive ticks above threshold */
        int confirmation_count; /* Consecutive ticks with SR > H */

        /* Cooldown tracking: ticks remaining in cooldown */
        int cooldown_remaining; /* Decrements each tick, 0 = not in cooldown */

        /* ═══════════════════════════════════════════════════════════════════
         * ROBUST WARMUP STATE (v3)
         * ═══════════════════════════════════════════════════════════════════ */
        float warmup_obs_buffer[256]; /* Ring buffer for warmup observations */
        int warmup_obs_count;         /* Total observations seen */
        int warmup_clean_count;       /* Observations that passed outlier filter */
        float warmup_median;          /* Running median estimate */
        float warmup_mad;             /* Running MAD estimate */
        float warmup_sum_clean;       /* Sum of clean (winsorized) observations² */
        bool warmup_baseline_set;     /* Has baseline been established? */
        bool warmup_audit_done;       /* Has post-warmup sanity check been performed? */
        bool cold_start_crisis;       /* Did we detect cold start during crisis? */

        /* Frozen baseline (during crisis) */
        float sigma_peace_frozen;
        bool baseline_frozen;

        /* Crisis sigma (learned during ACTIVE) */
        float sigma_crisis;

        /* Statistics */
        int total_crises;
        int false_alarms;      /* ALERT → IDLE transitions */
        int clean_exits;       /* RECOVERING → IDLE transitions */
        int re_triggers;       /* RECOVERING → ACTIVE transitions */
        int cooldown_blocks;   /* Times cooldown prevented ALERT entry */
        int nuclear_overrides; /* Times nuclear override triggered */

        /* Config */
        CrisisDetectorConfig cfg;

        /* Last update info (for diagnostics) */
        bool last_was_event;
        float last_return_threshold;
        float last_log_sr_up;
        float last_log_sr_down;
        float last_log_H;
        float last_hawkes_surprise;
        HawkesIntegratorResult last_hawkes_result;

    } CrisisDetector;

    /* ═══════════════════════════════════════════════════════════════════════════
     * CORE API
     * ═══════════════════════════════════════════════════════════════════════════ */

    /**
     * Initialize crisis detector
     *
     * @param cd   Crisis detector structure
     * @param cfg  Configuration (NULL for defaults)
     * @return     0 on success, -1 on error
     */
    int crisis_detector_init(CrisisDetector *cd, const CrisisDetectorConfig *cfg);

    /**
     * Reset to initial state (keeps config)
     */
    void crisis_detector_reset(CrisisDetector *cd);

    /**
     * Free any allocated resources
     */
    void crisis_detector_free(CrisisDetector *cd);

    /**
     * Main update function (returns only)
     *
     * @param cd    Crisis detector
     * @param obs   Observation (return)
     * @param tick  Current tick number (used for Hawkes time)
     * @return      Current crisis state
     */
    CrisisState crisis_detector_update(CrisisDetector *cd, float obs, int64_t tick);

    /**
     * Update with full market data
     *
     * @param cd         Crisis detector
     * @param obs        Observation (return)
     * @param volume     Trade volume (0 to skip)
     * @param imbalance  Order book imbalance (NAN to skip)
     * @param tick       Current tick number
     * @return           Current crisis state
     */
    CrisisState crisis_detector_update_full(CrisisDetector *cd, float obs,
                                            float volume, float imbalance,
                                            int64_t tick);

    /* ═══════════════════════════════════════════════════════════════════════════
     * STATE QUERIES
     * ═══════════════════════════════════════════════════════════════════════════ */

    static inline CrisisState crisis_detector_get_state(const CrisisDetector *cd)
    {
        return cd->state;
    }

    static inline bool crisis_detector_is_crisis(const CrisisDetector *cd)
    {
        return cd->state == CRISIS_ACTIVE || cd->state == CRISIS_RECOVERING;
    }

    static inline bool crisis_detector_is_alert(const CrisisDetector *cd)
    {
        return cd->state >= CRISIS_ALERT;
    }

    static inline bool crisis_detector_is_ready(const CrisisDetector *cd)
    {
        return cd->tick_count >= cd->cfg.warmup_ticks;
    }

    /* ═══════════════════════════════════════════════════════════════════════════
     * SIGMA ACCESS (for PGAS/RBPF integration)
     * ═══════════════════════════════════════════════════════════════════════════ */

    /**
     * Get current peace sigma (for SR baseline)
     * During crisis, returns frozen value
     */
    float crisis_detector_get_sigma_peace(const CrisisDetector *cd);

    /**
     * Set peace sigma from PGAS
     * Ignored during frozen period (ACTIVE/RECOVERING)
     */
    void crisis_detector_set_sigma_peace(CrisisDetector *cd, float sigma);

    /**
     * Get learned crisis sigma
     */
    static inline float crisis_detector_get_sigma_crisis(const CrisisDetector *cd)
    {
        return cd->sigma_crisis;
    }

    /* ═══════════════════════════════════════════════════════════════════════════
     * HAWKES ACCESS (for advanced users)
     * ═══════════════════════════════════════════════════════════════════════════ */

    /**
     * Get pointer to underlying Hawkes integrator
     */
    static inline HawkesIntegrator *crisis_detector_get_hawkes(CrisisDetector *cd)
    {
        return &cd->hawkes;
    }

    /**
     * Get last Hawkes result
     */
    static inline const HawkesIntegratorResult *crisis_detector_get_last_hawkes_result(
        const CrisisDetector *cd)
    {
        return &cd->last_hawkes_result;
    }

    /* ═══════════════════════════════════════════════════════════════════════════
     * DIAGNOSTICS
     * ═══════════════════════════════════════════════════════════════════════════ */

    typedef struct
    {
        CrisisState state;
        int64_t ticks_in_state;

        /* Event detector */
        bool is_event;
        double return_threshold;
        double event_rate;

        /* Hawkes (from your existing implementation) */
        float hawkes_intensity;
        float hawkes_surprise;
        HawkesTriggerState hawkes_state;
        bool hawkes_should_trigger;

        /* SR */
        float log_sr_up;
        float log_sr_down;
        float log_H;
        float sigma_peace;
        float sigma_crisis;

        /* FSM refinement state */
        int confirmation_count;
        int cooldown_remaining;

        /* Statistics */
        int total_crises;
        int false_alarms;
        int clean_exits;
        int re_triggers;
        int cooldown_blocks;
        int nuclear_overrides;

    } CrisisDetectorInfo;

    void crisis_detector_get_info(const CrisisDetector *cd, CrisisDetectorInfo *info);
    void crisis_detector_print_state(const CrisisDetector *cd);

#ifdef __cplusplus
}
#endif

#endif /* CRISIS_DETECTOR_H */