/*
 * event_detector.c - Meaningful Event Detection Implementation
 */

#include "event_detector.h"
#include <math.h>
#include <string.h>

/* ═══════════════════════════════════════════════════════════════════════════
 * INITIALIZATION
 * ═══════════════════════════════════════════════════════════════════════════ */

void event_detector_init(EventDetector *evt, const EventDetectorConfig *cfg) {
    memset(evt, 0, sizeof(*evt));
    
    if (cfg) {
        evt->cfg = *cfg;
    } else {
        evt->cfg = event_detector_config_default();
    }
    
    /* Initialize P² for tracking quantile of |returns| */
    p2_init(&evt->return_quantile, evt->cfg.return_quantile);
    
    /* Volume/imbalance disabled by default */
    evt->has_volume = false;
    evt->has_imbalance = false;
    evt->volume_ema = 0.0;
    evt->imbalance_ema = 0.0;
}

void event_detector_reset(EventDetector *evt) {
    EventDetectorConfig cfg = evt->cfg;
    event_detector_init(evt, &cfg);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * OPTIONAL DATA SOURCES
 * ═══════════════════════════════════════════════════════════════════════════ */

void event_detector_enable_volume(EventDetector *evt, double initial_volume) {
    evt->has_volume = true;
    evt->volume_ema = initial_volume;
}

void event_detector_enable_imbalance(EventDetector *evt, double initial_imbalance) {
    evt->has_imbalance = true;
    evt->imbalance_ema = fabs(initial_imbalance);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * CORE UPDATE
 * ═══════════════════════════════════════════════════════════════════════════ */

bool event_detector_update(EventDetector *evt, double obs) {
    return event_detector_update_full(evt, obs, 0.0, NAN);
}

bool event_detector_update_full(EventDetector *evt, double obs,
                                 double volume, double imbalance) {
    evt->tick_count++;
    
    double abs_return = fabs(obs);
    
    /* Update P² quantile with |return| */
    p2_update(&evt->return_quantile, abs_return);
    
    /* Update volume EMA if enabled */
    if (evt->has_volume && volume > 0.0) {
        evt->volume_ema = (1.0 - evt->cfg.ema_alpha) * evt->volume_ema 
                        + evt->cfg.ema_alpha * volume;
    }
    
    /* Update imbalance EMA if enabled */
    if (evt->has_imbalance && !isnan(imbalance)) {
        double abs_imb = fabs(imbalance);
        evt->imbalance_ema = (1.0 - evt->cfg.ema_alpha) * evt->imbalance_ema 
                           + evt->cfg.ema_alpha * abs_imb;
    }
    
    /* Reset event flags */
    evt->last_return_event = false;
    evt->last_volume_event = false;
    evt->last_imbalance_event = false;
    evt->last_was_event = false;
    
    /* Don't fire events during warmup */
    if (evt->tick_count < evt->cfg.warmup_ticks) {
        return false;
    }
    
    /* Get current threshold */
    double return_threshold = p2_get_quantile(&evt->return_quantile);
    evt->last_return_threshold = return_threshold;
    
    /* Check return event */
    if (abs_return > return_threshold) {
        evt->last_return_event = true;
    }
    
    /* Check volume event (if enabled) */
    if (evt->has_volume && volume > 0.0) {
        double volume_threshold = evt->cfg.volume_multiple * evt->volume_ema;
        if (volume > volume_threshold) {
            evt->last_volume_event = true;
        }
    }
    
    /* Check imbalance event (if enabled) */
    if (evt->has_imbalance && !isnan(imbalance)) {
        double imbalance_threshold = evt->cfg.imbalance_multiple * evt->imbalance_ema;
        if (fabs(imbalance) > imbalance_threshold) {
            evt->last_imbalance_event = true;
        }
    }
    
    /* Any trigger counts as an event */
    evt->last_was_event = evt->last_return_event || 
                          evt->last_volume_event || 
                          evt->last_imbalance_event;
    
    if (evt->last_was_event) {
        evt->event_count++;
    }
    
    return evt->last_was_event;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * DIAGNOSTICS
 * ═══════════════════════════════════════════════════════════════════════════ */

void event_detector_get_last_info(const EventDetector *evt, EventInfo *info) {
    info->is_event = evt->last_was_event;
    info->return_triggered = evt->last_return_event;
    info->volume_triggered = evt->last_volume_event;
    info->imbalance_triggered = evt->last_imbalance_event;
    
    info->return_threshold = evt->last_return_threshold;
    info->volume_threshold = evt->cfg.volume_multiple * evt->volume_ema;
    info->imbalance_threshold = evt->cfg.imbalance_multiple * evt->imbalance_ema;
    
    /* Note: actual values from last tick not stored, would need separate tracking */
    info->return_value = 0.0;
    info->volume_value = 0.0;
    info->imbalance_value = 0.0;
}
