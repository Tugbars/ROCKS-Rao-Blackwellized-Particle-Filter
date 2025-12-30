/*
 * sr_detector.h - Shiryaev-Roberts Change-Point Detector
 *
 * Log-space implementation with winsorization for numerical stability.
 * Supports both scalar (tick-by-tick) and vectorized (batch/backtest) operations.
 *
 * Theory:
 *   SR statistic: R_t = LR_t × (1 + R_{t-1})
 *   Log-space:    log(R_t) = log(LR_t) + log(1 + exp(log(R_{t-1})))
 *
 *   LR = p(obs|H1) / p(obs|H0) where:
 *     H0: obs ~ N(0, σ_h0)  [peace]
 *     H1: obs ~ N(0, σ_h1)  [crisis, typically 3×σ_h0]
 *
 *   log(LR) = 0.5×(z_h0² - z_h1²) + log(σ_h0/σ_h1)
 *
 * Winsorization: cap |z| to prevent single outliers from dominating.
 */

#ifndef SR_DETECTOR_H
#define SR_DETECTOR_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C"
{
#endif

    /* ═══════════════════════════════════════════════════════════════════════════
     * CONFIGURATION
     * ═══════════════════════════════════════════════════════════════════════════ */

    typedef struct
    {
        float sigma_multiple; /* Crisis σ = multiple × peace σ (default: 3.0) */
        float winsorize_cap;  /* Cap |z| at this value (default: 10.0) */
        float log_sr_clamp;   /* Clamp |log_sr| to prevent overflow (default: 20.0) */
        float student_nu;     /* Degrees of freedom for Student-t (0 = use Gaussian) */
    } SRConfig;

    /* Default configuration */
    static inline SRConfig sr_config_default(void)
    {
        return (SRConfig){
            .sigma_multiple = 3.0f,
            .winsorize_cap = 10.0f,
            .log_sr_clamp = 20.0f,
            .student_nu = 0.0f /* Gaussian by default */
        };
    }

    /* Student-t configuration (more robust to heavy tails) */
    static inline SRConfig sr_config_student_t(float nu)
    {
        return (SRConfig){
            .sigma_multiple = 3.0f,
            .winsorize_cap = 10.0f,
            .log_sr_clamp = 20.0f,
            .student_nu = nu /* Typical: 5-8 */
        };
    }

    /* ═══════════════════════════════════════════════════════════════════════════
     * SINGLE SR STATISTIC
     * ═══════════════════════════════════════════════════════════════════════════ */

    typedef struct
    {
        float log_sr; /* Current log(R_t) */
        SRConfig cfg;
    } SRStat;

    /* Initialize SR statistic */
    void sr_stat_init(SRStat *sr, const SRConfig *cfg);

    /* Reset to initial state (log_sr = 0, i.e., R = 1) */
    void sr_stat_reset(SRStat *sr);

    /*
     * Update SR with new observation.
     * Returns: new log_sr value
     */
    float sr_stat_update(SRStat *sr, float obs, float sigma_peace);

    /*
     * Check if SR crossed threshold.
     * log_threshold: e.g., 5.0 for ~148:1 odds
     */
    static inline bool sr_stat_triggered(const SRStat *sr, float log_threshold)
    {
        return sr->log_sr > log_threshold;
    }

    /* ═══════════════════════════════════════════════════════════════════════════
     * DUAL SR STATISTICS (for entry and exit detection)
     * ═══════════════════════════════════════════════════════════════════════════ */

    typedef struct
    {
        float log_sr_up;              /* Peace → Crisis (entry detection) */
        float log_sr_down;            /* Crisis → Peace (exit detection) */
        float sigma_peace;            /* Baseline volatility (from PGAS calm regime) */
        float sigma_crisis;           /* Learned crisis volatility (EMA during crisis) */
        float sigma_crisis_ema_alpha; /* EMA decay for sigma_crisis learning */
        SRConfig cfg;
    } DualSR;

    /* Initialize dual SR */
    void dual_sr_init(DualSR *dsr, const SRConfig *cfg, float initial_sigma_peace);

    /* Reset both statistics */
    void dual_sr_reset(DualSR *dsr);

    /* Reset only SR_up (on crisis confirmation) */
    void dual_sr_reset_up(DualSR *dsr);

    /* Reset only SR_down (on exit confirmation or false exit) */
    void dual_sr_reset_down(DualSR *dsr);

    /*
     * Update for IDLE/ALERT states: only SR_up active
     * Returns: log_sr_up
     */
    float dual_sr_update_entry(DualSR *dsr, float obs);

    /*
     * Update for CRISIS_ACTIVE state: only SR_down active, learn sigma_crisis
     * Returns: log_sr_down
     */
    float dual_sr_update_exit(DualSR *dsr, float obs);

    /*
     * Update for RECOVERING state: both active
     * Returns: log_sr_up (also updates log_sr_down internally)
     */
    float dual_sr_update_both(DualSR *dsr, float obs);

    /* Update sigma_peace (called when PGAS publishes new estimate) */
    void dual_sr_set_sigma_peace(DualSR *dsr, float sigma_peace);

    /* ═══════════════════════════════════════════════════════════════════════════
     * VECTORIZED OPERATIONS (for batch processing / backtesting)
     * Uses MKL VML when available, falls back to SIMD
     * ═══════════════════════════════════════════════════════════════════════════ */

    /*
     * Compute log-likelihood ratios for batch of observations.
     *
     * obs[n]:         Input observations
     * sigma_peace[n]: Per-observation sigma (or NULL for constant)
     * sigma_peace_const: Used if sigma_peace is NULL
     * log_lr[n]:      Output log-likelihood ratios
     * n:              Number of observations
     * cfg:            Configuration
     */
    void sr_compute_log_lr_batch(
        const float *obs,
        const float *sigma_peace, /* NULL for constant sigma */
        float sigma_peace_const,
        float *log_lr,
        int n,
        const SRConfig *cfg);

    /*
     * Compute cumulative SR statistics for batch.
     * This is the full SR recursion: log_sr[t] = log_lr[t] + log1p(exp(log_sr[t-1]))
     *
     * log_lr[n]:      Input log-likelihood ratios
     * log_sr[n]:      Output cumulative log-SR values
     * log_sr_init:    Initial log_sr (typically 0)
     * n:              Number of observations
     * clamp:          Clamp value for |log_sr|
     *
     * Returns: final log_sr value
     */
    float sr_accumulate_batch(
        const float *log_lr,
        float *log_sr,
        float log_sr_init,
        int n,
        float clamp);

    /*
     * Combined: compute log_lr and accumulate in one pass.
     * More cache-friendly for large batches.
     *
     * Returns: final log_sr value
     */
    float sr_update_batch(
        const float *obs,
        float sigma_peace,
        float *log_sr, /* Output: cumulative log_sr at each tick */
        float log_sr_init,
        int n,
        const SRConfig *cfg);

    /* ═══════════════════════════════════════════════════════════════════════════
     * ADAPTIVE THRESHOLD
     * ═══════════════════════════════════════════════════════════════════════════ */

    typedef struct
    {
        float log_H_base;    /* Base threshold (default: 5.0 = ~148:1) */
        float inertia_scale; /* Ticks divisor for inertia (default: 1000.0) */
        float wolf_penalty;  /* Penalty per false alarm (default: 2.0) */
        int max_wolf_count;  /* Cap on wolf penalty count (default: 3) */
    } AdaptiveThresholdConfig;

    static inline AdaptiveThresholdConfig adaptive_threshold_config_default(void)
    {
        return (AdaptiveThresholdConfig){
            .log_H_base = 5.0f,
            .inertia_scale = 1000.0f,
            .wolf_penalty = 2.0f,
            .max_wolf_count = 3};
    }

    typedef struct
    {
        int64_t ticks_since_crisis;
        int recent_false_alarms;
        AdaptiveThresholdConfig cfg;
    } AdaptiveThreshold;

    void adaptive_threshold_init(AdaptiveThreshold *at, const AdaptiveThresholdConfig *cfg);
    void adaptive_threshold_reset(AdaptiveThreshold *at);

    /* Increment tick counter (call every tick in IDLE) */
    void adaptive_threshold_tick(AdaptiveThreshold *at);

    /* Record false alarm (call when SR rejects in ALERT) */
    void adaptive_threshold_false_alarm(AdaptiveThreshold *at);

    /* Reset on clean exit (call when exiting RECOVERING cleanly) */
    void adaptive_threshold_clean_exit(AdaptiveThreshold *at);

    /* Compute current threshold */
    float adaptive_threshold_compute(const AdaptiveThreshold *at);

    /* ═══════════════════════════════════════════════════════════════════════════
     * LOW-LEVEL PRIMITIVES (for custom integrations)
     * ═══════════════════════════════════════════════════════════════════════════ */

    /*
     * Compute single log-LR (Gaussian, winsorized)
     * This is inlined for maximum performance in tick loops.
     */
    float sr_log_lr_gaussian(float obs, float sigma_h0, float sigma_h1, float winsorize_cap);

    /*
     * Compute single log-LR (Student-t)
     */
    float sr_log_lr_student_t(float obs, float sigma_h0, float sigma_h1, float nu);

    /*
     * SR accumulation step: log_sr_new = log_lr + log1p(exp(log_sr_old))
     */
    float sr_accumulate_step(float log_lr, float log_sr_old, float clamp);

#ifdef __cplusplus
}
#endif

#endif /* SR_DETECTOR_H */