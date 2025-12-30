/*
 * crisis_detector.c - Crisis Detection Integration Layer
 *
 * Implements the v3.1 two-stage detonator:
 *   Stage 1: EventDetector + Hawkes (fast, early warning)
 *   Stage 2: DualSR (principled hypothesis testing)
 *
 * Uses the existing HawkesIntegrator implementation.
 */

#include "crisis_detector.h"
#include <string.h>
#include <stdio.h>
#include <math.h>

/* ═══════════════════════════════════════════════════════════════════════════
 * CONFIGURATION DEFAULTS
 * ═══════════════════════════════════════════════════════════════════════════ */

CrisisDetectorConfig crisis_detector_config_default(void)
{
    CrisisDetectorConfig cfg;
    memset(&cfg, 0, sizeof(cfg));

    /* Event detector: P90 quantile */
    cfg.event_cfg = event_detector_config_default();

    /* Hawkes: use existing defaults, but set event_threshold = 0
     * so we control events via EventDetector */
    cfg.hawkes_cfg = hawkes_integrator_config_defaults();
    cfg.hawkes_cfg.hawkes.event_threshold = 0.0f;

    /* SR: Gaussian with winsorization */
    cfg.sr_cfg = sr_config_default();

    /* Adaptive threshold */
    cfg.threshold_cfg = adaptive_threshold_config_default();

    /* State machine */
    cfg.sr_alert_fraction = 0.5f;   /* ALERT when SR > 50% of threshold */
    cfg.sr_exit_threshold = 3.0f;   /* log(20) ≈ 20:1 odds for exit */
    cfg.sr_reentry_fraction = 0.7f; /* Re-trigger at 70% of threshold */

    /* FSM refinements (v2) */
    cfg.confirmation_ticks = 3;  /* 3 consecutive ticks above H to confirm */
    cfg.min_hold_active = 50;    /* Stay in ACTIVE for at least 50 ticks */
    cfg.cooldown_ticks = 100;    /* 100 tick cooldown after false alarm */
    cfg.nuclear_override = 2.0f; /* Override cooldown if SR > 2×H */

    /* Crisis sigma learning */
    cfg.sigma_crisis_ema_alpha = 0.01f;

    /* Initial sigma */
    cfg.initial_sigma_peace = 0.01f; /* 1% */

    /* Warmup */
    cfg.warmup_ticks = 200;

    /* Robust warmup (v3) */
    cfg.use_robust_warmup = true;
    cfg.warmup_outlier_k = 3.0f;   /* Reject > 3 MAD during warmup */
    cfg.warmup_winsorize_k = 4.0f; /* Winsorize at 4 MAD */
    cfg.warmup_min_clean = 50;     /* Need 50 clean obs before baseline */

    /* Sanity anchor (v3.1) - THE CIRCUIT BREAKER */
    /* This is a HARD LIMIT derived from asset fundamentals.
     * For 1-minute equity bars: ~0.015 per bar ≈ 15% annualized vol is high but "peace"
     * Anything above ~2-3% per bar is definitely crisis territory.
     *
     * IMPORTANT: Adjust this based on your asset class and bar frequency!
     * - For tick data: lower (e.g., 0.001)
     * - For 1-min bars: ~0.015
     * - For daily bars: ~0.03
     */
    cfg.max_peace_sigma = 0.015f; /* 1.5% per bar - conservative anchor */

    return cfg;
}

CrisisDetectorConfig crisis_detector_config_sensitive(void)
{
    CrisisDetectorConfig cfg = crisis_detector_config_default();

    cfg.event_cfg = event_detector_config_sensitive();
    cfg.hawkes_cfg = hawkes_integrator_config_responsive();
    cfg.hawkes_cfg.hawkes.event_threshold = 0.0f;

    cfg.threshold_cfg.log_H_base = 4.0f; /* Lower threshold */
    cfg.sr_alert_fraction = 0.4f;

    /* Faster FSM (but still principled) */
    cfg.confirmation_ticks = 2;
    cfg.min_hold_active = 30;
    cfg.cooldown_ticks = 50;

    cfg.warmup_ticks = 100;

    return cfg;
}

CrisisDetectorConfig crisis_detector_config_conservative(void)
{
    CrisisDetectorConfig cfg = crisis_detector_config_default();

    cfg.hawkes_cfg = hawkes_integrator_config_conservative();
    cfg.hawkes_cfg.hawkes.event_threshold = 0.0f;

    cfg.threshold_cfg.log_H_base = 6.0f; /* Higher threshold */
    cfg.threshold_cfg.wolf_penalty = 3.0f;
    cfg.sr_alert_fraction = 0.6f;

    /* Slower FSM (more confirmation required) */
    cfg.confirmation_ticks = 5;
    cfg.min_hold_active = 100;
    cfg.cooldown_ticks = 200;

    cfg.warmup_ticks = 500;

    return cfg;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * INITIALIZATION
 * ═══════════════════════════════════════════════════════════════════════════ */

int crisis_detector_init(CrisisDetector *cd, const CrisisDetectorConfig *cfg)
{
    if (!cd)
        return -1;

    memset(cd, 0, sizeof(*cd));

    if (cfg)
    {
        cd->cfg = *cfg;
    }
    else
    {
        cd->cfg = crisis_detector_config_default();
    }

    /* Initialize event detector */
    event_detector_init(&cd->event_det, &cd->cfg.event_cfg);

    /* Initialize Hawkes using the existing API */
    if (hawkes_integrator_init(&cd->hawkes, &cd->cfg.hawkes_cfg) != 0)
    {
        return -1;
    }

    /* Initialize dual SR */
    dual_sr_init(&cd->dual_sr, &cd->cfg.sr_cfg, cd->cfg.initial_sigma_peace);

    /* Initialize adaptive threshold */
    adaptive_threshold_init(&cd->adaptive_th, &cd->cfg.threshold_cfg);

    /* State */
    cd->state = CRISIS_IDLE;
    cd->sigma_crisis = cd->cfg.sr_cfg.sigma_multiple * cd->cfg.initial_sigma_peace;

    return 0;
}

void crisis_detector_reset(CrisisDetector *cd)
{
    if (!cd)
        return;

    CrisisDetectorConfig cfg = cd->cfg;
    crisis_detector_init(cd, &cfg);
}

void crisis_detector_free(CrisisDetector *cd)
{
    if (!cd)
        return;

    hawkes_integrator_free(&cd->hawkes);
    /* Other components don't need explicit free */
}

/* ═══════════════════════════════════════════════════════════════════════════
 * SIGMA ACCESS
 * ═══════════════════════════════════════════════════════════════════════════ */

float crisis_detector_get_sigma_peace(const CrisisDetector *cd)
{
    if (!cd)
        return 0.01f;

    if (cd->baseline_frozen)
    {
        return cd->sigma_peace_frozen;
    }
    return cd->dual_sr.sigma_peace;
}

void crisis_detector_set_sigma_peace(CrisisDetector *cd, float sigma)
{
    if (!cd)
        return;

    if (cd->baseline_frozen)
    {
        return; /* Ignore during crisis */
    }
    dual_sr_set_sigma_peace(&cd->dual_sr, sigma);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * STATE MACHINE HELPERS
 * ═══════════════════════════════════════════════════════════════════════════ */

static void enter_state(CrisisDetector *cd, CrisisState new_state, int64_t tick)
{
    if (cd->state != new_state)
    {
        cd->state = new_state;
        cd->last_state_change_tick = tick;
        cd->ticks_in_state = 0;
    }
}

static void freeze_baseline(CrisisDetector *cd)
{
    if (!cd->baseline_frozen)
    {
        cd->sigma_peace_frozen = cd->dual_sr.sigma_peace;
        cd->baseline_frozen = true;
    }
}

static void unfreeze_baseline(CrisisDetector *cd)
{
    cd->baseline_frozen = false;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * CORE UPDATE
 * ═══════════════════════════════════════════════════════════════════════════ */

CrisisState crisis_detector_update(CrisisDetector *cd, float obs, int64_t tick)
{
    return crisis_detector_update_full(cd, obs, 0.0f, NAN, tick);
}

CrisisState crisis_detector_update_full(CrisisDetector *cd, float obs,
                                        float volume, float imbalance,
                                        int64_t tick)
{
    if (!cd)
        return CRISIS_IDLE;

    cd->tick_count++;
    cd->ticks_in_state++;

    /* Decrement cooldown timer */
    if (cd->cooldown_remaining > 0)
    {
        cd->cooldown_remaining--;
    }

    /* ═══════════════════════════════════════════════════════════════════
     * STEP 0: Robust Warmup (v3) - Handle cold start during crisis
     * ═══════════════════════════════════════════════════════════════════ */
    if (cd->cfg.use_robust_warmup && !cd->warmup_baseline_set)
    {
        float abs_obs = fabsf(obs);
        int idx = cd->warmup_obs_count % 256;

        /* First few observations: bootstrap median/MAD estimates */
        if (cd->warmup_obs_count < 10)
        {
            cd->warmup_obs_buffer[idx] = abs_obs;
            cd->warmup_obs_count++;

            /* Simple initial estimate from first 10 obs */
            if (cd->warmup_obs_count == 10)
            {
                /* Sort to find median */
                float sorted[10];
                for (int i = 0; i < 10; i++)
                    sorted[i] = cd->warmup_obs_buffer[i];
                for (int i = 0; i < 9; i++)
                {
                    for (int j = i + 1; j < 10; j++)
                    {
                        if (sorted[j] < sorted[i])
                        {
                            float tmp = sorted[i];
                            sorted[i] = sorted[j];
                            sorted[j] = tmp;
                        }
                    }
                }
                cd->warmup_median = (sorted[4] + sorted[5]) / 2.0f;

                /* MAD = median of |x - median| */
                float deviations[10];
                for (int i = 0; i < 10; i++)
                {
                    deviations[i] = fabsf(sorted[i] - cd->warmup_median);
                }
                for (int i = 0; i < 9; i++)
                {
                    for (int j = i + 1; j < 10; j++)
                    {
                        if (deviations[j] < deviations[i])
                        {
                            float tmp = deviations[i];
                            deviations[i] = deviations[j];
                            deviations[j] = tmp;
                        }
                    }
                }
                cd->warmup_mad = (deviations[4] + deviations[5]) / 2.0f;
                if (cd->warmup_mad < 1e-8f)
                    cd->warmup_mad = cd->warmup_median * 0.5f;
            }
            return cd->state; /* Stay in IDLE during bootstrap */
        }

        /* After bootstrap: apply outlier rejection */
        float threshold = cd->cfg.warmup_outlier_k * cd->warmup_mad * 1.4826f; /* MAD→σ */
        bool is_outlier = (abs_obs > cd->warmup_median + threshold);

        if (is_outlier)
        {
            /* Winsorize instead of reject completely */
            float winsor_limit = cd->cfg.warmup_winsorize_k * cd->warmup_mad * 1.4826f;
            abs_obs = fminf(abs_obs, cd->warmup_median + winsor_limit);
        }
        else
        {
            cd->warmup_clean_count++;
        }

        /* Update running estimates with exponential smoothing */
        float alpha = 0.05f;
        cd->warmup_median = (1.0f - alpha) * cd->warmup_median + alpha * abs_obs;
        float dev = fabsf(abs_obs - cd->warmup_median);
        cd->warmup_mad = (1.0f - alpha) * cd->warmup_mad + alpha * dev;
        if (cd->warmup_mad < 1e-8f)
            cd->warmup_mad = 1e-6f;

        /* Accumulate for σ estimate */
        cd->warmup_sum_clean += abs_obs * abs_obs;
        cd->warmup_obs_buffer[idx] = abs_obs;
        cd->warmup_obs_count++;

        /* Check if we have enough clean observations to set baseline */
        if (cd->warmup_clean_count >= cd->cfg.warmup_min_clean &&
            cd->warmup_obs_count >= cd->cfg.warmup_ticks / 2)
        {
            /* Set σ_peace from clean observations */
            float sigma_est = sqrtf(cd->warmup_sum_clean / cd->warmup_obs_count);
            dual_sr_set_sigma_peace(&cd->dual_sr, sigma_est);
            cd->warmup_baseline_set = true;

            /* Also update event detector baseline */
            /* (P² will have adapted, but at least SR is clean now) */
        }
    }

    /* ═══════════════════════════════════════════════════════════════════
     * STEP 1: Event Detection (P² quantile-based)
     * ═══════════════════════════════════════════════════════════════════ */
    bool is_event = event_detector_update_full(&cd->event_det, obs, volume, imbalance);
    cd->last_was_event = is_event;
    cd->last_return_threshold = (float)event_detector_get_threshold(&cd->event_det);

    /* ═══════════════════════════════════════════════════════════════════
     * STEP 2: Hawkes Update
     * ═══════════════════════════════════════════════════════════════════ */
    float hawkes_return = is_event ? obs : 0.0f;
    HawkesIntegratorResult hawkes_result = hawkes_integrator_update(
        &cd->hawkes, (float)tick, hawkes_return);

    cd->last_hawkes_result = hawkes_result;
    cd->last_hawkes_surprise = hawkes_result.surprise_sigma;

    bool hawkes_armed = (hawkes_result.state == HAWKES_TRIG_ARMED) ||
                        hawkes_result.should_trigger;

    /* ═══════════════════════════════════════════════════════════════════
     * STEP 3: Dual SR Update (state-dependent)
     * ═══════════════════════════════════════════════════════════════════ */
    switch (cd->state)
    {
    case CRISIS_IDLE:
    case CRISIS_ALERT:
        dual_sr_update_entry(&cd->dual_sr, obs);
        break;

    case CRISIS_ACTIVE:
        dual_sr_update_exit(&cd->dual_sr, obs);
        cd->sigma_crisis = (1.0f - cd->cfg.sigma_crisis_ema_alpha) * cd->sigma_crisis + cd->cfg.sigma_crisis_ema_alpha * fabsf(obs);
        break;

    case CRISIS_RECOVERING:
        dual_sr_update_both(&cd->dual_sr, obs);
        break;
    }

    cd->last_log_sr_up = cd->dual_sr.log_sr_up;
    cd->last_log_sr_down = cd->dual_sr.log_sr_down;

    /* ═══════════════════════════════════════════════════════════════════
     * STEP 4: Adaptive Threshold
     * ═══════════════════════════════════════════════════════════════════ */
    float log_H = adaptive_threshold_compute(&cd->adaptive_th);
    cd->last_log_H = log_H;

    /* ═══════════════════════════════════════════════════════════════════
     * STEP 5: State Machine (v2 - with refinements)
     * ═══════════════════════════════════════════════════════════════════ */

    /* Skip state transitions during warmup */
    if (cd->tick_count < cd->cfg.warmup_ticks)
    {
        return cd->state;
    }

    /* ═══════════════════════════════════════════════════════════════════
     * STEP 5.1: POST-WARMUP SANITY AUDIT (v3.1 - The Anchor)
     *
     * At the end of warmup, verify that learned baseline is physically
     * consistent with a "peace" regime. If not, we have a COLD START
     * DURING CRISIS - reject the learned baseline and force to ACTIVE.
     * ═══════════════════════════════════════════════════════════════════ */
    if (!cd->warmup_audit_done)
    {
        cd->warmup_audit_done = true;

        /* Get what we learned during warmup */
        float learned_sigma = cd->dual_sr.sigma_peace;

        /* THE AUDIT: Is this physically consistent with "peace"? */
        if (learned_sigma > cd->cfg.max_peace_sigma)
        {

            /* ═══════════════════════════════════════════════════════════
             * VIOLATION: Cold Start During Crisis
             * The learned baseline is too high to be "peace".
             * We cannot trust anything we learned.
             * ═══════════════════════════════════════════════════════════ */
            cd->cold_start_crisis = true;

            /* A. Override σ_peace with the Sanity Anchor */
            dual_sr_set_sigma_peace(&cd->dual_sr, cd->cfg.max_peace_sigma);

            /* B. Set σ_crisis to what we actually observed (high vol) */
            /*    This ensures SR_down (exit) works correctly */
            cd->sigma_crisis = learned_sigma;

            /* C. Wipe the toxic P² memory - replace with theoretical dist */
            event_detector_force_theoretical_distribution(
                &cd->event_det,
                (double)cd->cfg.max_peace_sigma);

            /* D. Force immediate state transition to CRISIS_ACTIVE */
            enter_state(cd, CRISIS_ACTIVE, tick);
            cd->total_crises++;

            /* E. Freeze the (now safe) baseline */
            freeze_baseline(cd);

            /* F. Reset confirmation/cooldown state */
            cd->confirmation_count = 0;
            cd->cooldown_remaining = 0;
            cd->ticks_in_state = 0;

            /* Return immediately - we're now in crisis mode */
            return cd->state;
        }
    }

    float sr_alert_level = cd->cfg.sr_alert_fraction * log_H;
    float sr_reentry_level = cd->cfg.sr_reentry_fraction * log_H;
    float sr_exit_threshold = cd->cfg.sr_exit_threshold;
    float nuclear_level = cd->cfg.nuclear_override * log_H;

    /* Check if SR is above threshold for confirmation tracking */
    bool sr_above_threshold = (cd->dual_sr.log_sr_up > log_H);

    switch (cd->state)
    {

    case CRISIS_IDLE:
        adaptive_threshold_tick(&cd->adaptive_th);
        cd->confirmation_count = 0; /* Reset confirmation counter */

        /* Check for nuclear override (ignores cooldown) */
        if (cd->dual_sr.log_sr_up > nuclear_level)
        {
            enter_state(cd, CRISIS_ALERT, tick);
            cd->nuclear_overrides++;
            break;
        }

        /* During cooldown, only nuclear can trigger */
        if (cd->cooldown_remaining > 0)
        {
            cd->cooldown_blocks++;
            break;
        }

        /* Normal entry: Hawkes armed OR SR elevated */
        if (hawkes_armed || cd->dual_sr.log_sr_up > sr_alert_level)
        {
            enter_state(cd, CRISIS_ALERT, tick);
        }
        break;

    case CRISIS_ALERT:
        /* Track consecutive ticks above threshold */
        if (sr_above_threshold)
        {
            cd->confirmation_count++;
        }
        else
        {
            cd->confirmation_count = 0;
        }

        /* Confirm crisis only after N consecutive ticks above H */
        if (cd->confirmation_count >= cd->cfg.confirmation_ticks)
        {
            enter_state(cd, CRISIS_ACTIVE, tick);
            freeze_baseline(cd);
            dual_sr_reset_up(&cd->dual_sr);
            cd->total_crises++;
            cd->confirmation_count = 0;
        }
        /* Abort if SR drops significantly */
        else if (cd->dual_sr.log_sr_up < 0.0f && cd->ticks_in_state > 10)
        {
            enter_state(cd, CRISIS_IDLE, tick);
            adaptive_threshold_false_alarm(&cd->adaptive_th);
            cd->false_alarms++;
            cd->confirmation_count = 0;
            /* Start cooldown */
            cd->cooldown_remaining = cd->cfg.cooldown_ticks;
        }
        break;

    case CRISIS_ACTIVE:
        /* Enforce minimum hold time - cannot exit until ticks_in_state >= min_hold_active */
        if (cd->ticks_in_state < cd->cfg.min_hold_active)
        {
            break; /* Stay in ACTIVE, ignore exit conditions */
        }

        /* Check exit condition via SR_down */
        if (cd->dual_sr.log_sr_down > sr_exit_threshold)
        {
            enter_state(cd, CRISIS_RECOVERING, tick);
        }
        break;

    case CRISIS_RECOVERING:
        /* Clean exit if SR_down confirms AND SR_up is low */
        if (cd->dual_sr.log_sr_down > sr_exit_threshold &&
            cd->dual_sr.log_sr_up < sr_alert_level)
        {
            enter_state(cd, CRISIS_IDLE, tick);
            unfreeze_baseline(cd);
            dual_sr_reset(&cd->dual_sr);
            adaptive_threshold_clean_exit(&cd->adaptive_th);
            cd->clean_exits++;
        }
        /* False exit - crisis re-triggering */
        else if (cd->dual_sr.log_sr_up > sr_reentry_level)
        {
            enter_state(cd, CRISIS_ACTIVE, tick);
            dual_sr_reset_down(&cd->dual_sr);
            cd->re_triggers++;
        }
        break;
    }

    return cd->state;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * DIAGNOSTICS
 * ═══════════════════════════════════════════════════════════════════════════ */

void crisis_detector_get_info(const CrisisDetector *cd, CrisisDetectorInfo *info)
{
    if (!cd || !info)
        return;

    memset(info, 0, sizeof(*info));

    info->state = cd->state;
    info->ticks_in_state = cd->ticks_in_state;

    /* Event detector */
    info->is_event = cd->last_was_event;
    info->return_threshold = cd->last_return_threshold;
    info->event_rate = event_detector_get_rate(&cd->event_det);

    /* Hawkes (from existing implementation) */
    info->hawkes_intensity = hawkes_integrator_get_intensity(&cd->hawkes);
    info->hawkes_surprise = cd->last_hawkes_surprise;
    info->hawkes_state = hawkes_integrator_get_state(&cd->hawkes);
    info->hawkes_should_trigger = cd->last_hawkes_result.should_trigger;

    /* SR */
    info->log_sr_up = cd->last_log_sr_up;
    info->log_sr_down = cd->last_log_sr_down;
    info->log_H = cd->last_log_H;
    info->sigma_peace = crisis_detector_get_sigma_peace(cd);
    info->sigma_crisis = cd->sigma_crisis;

    /* FSM refinement state */
    info->confirmation_count = cd->confirmation_count;
    info->cooldown_remaining = cd->cooldown_remaining;

    /* Statistics */
    info->total_crises = cd->total_crises;
    info->false_alarms = cd->false_alarms;
    info->clean_exits = cd->clean_exits;
    info->re_triggers = cd->re_triggers;
    info->cooldown_blocks = cd->cooldown_blocks;
    info->nuclear_overrides = cd->nuclear_overrides;
}

void crisis_detector_print_state(const CrisisDetector *cd)
{
    if (!cd)
        return;

    CrisisDetectorInfo info;
    crisis_detector_get_info(cd, &info);

    printf("\n");
    printf("+===========================================================+\n");
    printf("|              CRISIS DETECTOR STATE (v2)                   |\n");
    printf("+===========================================================+\n");
    printf("| State: %-12s  Ticks in state: %ld\n",
           crisis_state_name(info.state), (long)info.ticks_in_state);
    printf("| Confirmation: %d/%d  Cooldown: %d remaining\n",
           info.confirmation_count, cd->cfg.confirmation_ticks,
           info.cooldown_remaining);
    printf("+-----------------------------------------------------------+\n");
    printf("| Event Detector:\n");
    printf("|   Last event: %s  Threshold: %.6f\n",
           info.is_event ? "YES" : "NO", info.return_threshold);
    printf("|   Event rate: %.2f%%\n", info.event_rate * 100.0);
    printf("+-----------------------------------------------------------+\n");
    printf("| Hawkes:\n");
    printf("|   Intensity: %.4f  Surprise: %.2f σ\n",
           info.hawkes_intensity, info.hawkes_surprise);
    printf("|   State: %s  Should trigger: %s\n",
           info.hawkes_state == HAWKES_TRIG_ARMED ? "ARMED" : info.hawkes_state == HAWKES_TRIG_IDLE     ? "IDLE"
                                                          : info.hawkes_state == HAWKES_TRIG_REFRACTORY ? "REFRACTORY"
                                                                                                        : "FIRED",
           info.hawkes_should_trigger ? "YES" : "NO");
    printf("+-----------------------------------------------------------+\n");
    printf("| Shiryaev-Roberts:\n");
    printf("|   log_SR_up: %.2f  log_SR_down: %.2f\n",
           info.log_sr_up, info.log_sr_down);
    printf("|   log_H (threshold): %.2f  Nuclear: %.2f\n",
           info.log_H, cd->cfg.nuclear_override * info.log_H);
    printf("|   σ_peace: %.6f  σ_crisis: %.6f\n",
           info.sigma_peace, info.sigma_crisis);
    printf("+-----------------------------------------------------------+\n");
    printf("| Statistics:\n");
    printf("|   Crises: %d  False alarms: %d\n",
           info.total_crises, info.false_alarms);
    printf("|   Clean exits: %d  Re-triggers: %d\n",
           info.clean_exits, info.re_triggers);
    printf("|   Cooldown blocks: %d  Nuclear overrides: %d\n",
           info.cooldown_blocks, info.nuclear_overrides);
    printf("+===========================================================+\n");
}