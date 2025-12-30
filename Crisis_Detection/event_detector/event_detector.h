/*
 * event_detector.h - Meaningful Event Detection for Hawkes Process
 * 
 * Detects "events" that Hawkes should respond to, rather than every tick.
 * Uses P² algorithm for O(1) rolling quantile estimation.
 * 
 * Event types:
 *   - Return spike: |r| > 90th percentile of recent |returns|
 *   - Volume spike: volume > 2× EMA (optional, when data available)
 *   - Imbalance spike: |imbalance| > 2× EMA (optional, when data available)
 * 
 * Usage:
 *   EventDetector evt;
 *   event_detector_init(&evt, NULL);
 *   
 *   for each tick:
 *       bool is_event = event_detector_update(&evt, obs);
 *       if (is_event) {
 *           hawkes_add_event(...);
 *       }
 */

#ifndef EVENT_DETECTOR_H
#define EVENT_DETECTOR_H

#include "p2_quantile.h"
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ═══════════════════════════════════════════════════════════════════════════
 * CONFIGURATION
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct {
    double return_quantile;      /* Quantile for return events (default: 0.90) */
    double volume_multiple;      /* Volume spike threshold (default: 2.0) */
    double imbalance_multiple;   /* Imbalance spike threshold (default: 2.0) */
    double ema_alpha;            /* EMA decay for volume/imbalance (default: 0.01) */
    int    warmup_ticks;         /* Ticks before events can fire (default: 100) */
} EventDetectorConfig;

static inline EventDetectorConfig event_detector_config_default(void) {
    return (EventDetectorConfig){
        .return_quantile    = 0.90,
        .volume_multiple    = 2.0,
        .imbalance_multiple = 2.0,
        .ema_alpha          = 0.01,
        .warmup_ticks       = 100
    };
}

/* More sensitive config for fast detection */
static inline EventDetectorConfig event_detector_config_sensitive(void) {
    return (EventDetectorConfig){
        .return_quantile    = 0.85,   /* Lower threshold */
        .volume_multiple    = 1.5,
        .imbalance_multiple = 1.5,
        .ema_alpha          = 0.02,   /* Faster adaptation */
        .warmup_ticks       = 50
    };
}

/* ═══════════════════════════════════════════════════════════════════════════
 * EVENT DETECTOR
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct {
    /* P² quantile estimator for |returns| */
    P2Quantile return_quantile;
    
    /* Optional: volume EMA */
    double volume_ema;
    bool   has_volume;
    
    /* Optional: imbalance EMA */
    double imbalance_ema;
    bool   has_imbalance;
    
    /* State */
    int64_t tick_count;
    int64_t event_count;
    
    /* Last event info (for diagnostics) */
    bool   last_was_event;
    bool   last_return_event;
    bool   last_volume_event;
    bool   last_imbalance_event;
    double last_return_threshold;
    
    /* Config */
    EventDetectorConfig cfg;
    
} EventDetector;

/* ═══════════════════════════════════════════════════════════════════════════
 * CORE API
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * Initialize event detector
 * 
 * @param evt  Event detector structure
 * @param cfg  Configuration (NULL for defaults)
 */
void event_detector_init(EventDetector *evt, const EventDetectorConfig *cfg);

/**
 * Reset to initial state (keeps config)
 */
void event_detector_reset(EventDetector *evt);

/**
 * Update with new observation (returns only)
 * 
 * @param evt  Event detector
 * @param obs  Observation (return)
 * @return     true if this tick is an event
 */
bool event_detector_update(EventDetector *evt, double obs);

/**
 * Update with full market data (returns + volume + imbalance)
 * 
 * @param evt        Event detector
 * @param obs        Observation (return)
 * @param volume     Trade volume (0 to skip)
 * @param imbalance  Order book imbalance [-1, +1] (NAN to skip)
 * @return           true if this tick is an event
 */
bool event_detector_update_full(EventDetector *evt, double obs,
                                 double volume, double imbalance);

/**
 * Check if detector is warmed up
 */
static inline bool event_detector_is_ready(const EventDetector *evt) {
    return evt->tick_count >= evt->cfg.warmup_ticks;
}

/**
 * Get current return threshold (90th percentile of |returns|)
 */
static inline double event_detector_get_threshold(const EventDetector *evt) {
    return p2_get_quantile(&evt->return_quantile);
}

/**
 * Get event rate (events / total ticks)
 */
static inline double event_detector_get_rate(const EventDetector *evt) {
    if (evt->tick_count == 0) return 0.0;
    return (double)evt->event_count / (double)evt->tick_count;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * OPTIONAL DATA SOURCES
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * Enable volume-based events
 * Call once when volume data becomes available
 * 
 * @param evt             Event detector
 * @param initial_volume  Initial volume estimate for EMA
 */
void event_detector_enable_volume(EventDetector *evt, double initial_volume);

/**
 * Enable imbalance-based events
 * Call once when order book data becomes available
 * 
 * @param evt                Event detector
 * @param initial_imbalance  Initial |imbalance| estimate for EMA
 */
void event_detector_enable_imbalance(EventDetector *evt, double initial_imbalance);

/* ═══════════════════════════════════════════════════════════════════════════
 * DIAGNOSTICS
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * Get detailed event info from last update
 */
typedef struct {
    bool   is_event;
    bool   return_triggered;
    bool   volume_triggered;
    bool   imbalance_triggered;
    double return_value;
    double return_threshold;
    double volume_value;
    double volume_threshold;
    double imbalance_value;
    double imbalance_threshold;
} EventInfo;

void event_detector_get_last_info(const EventDetector *evt, EventInfo *info);

#ifdef __cplusplus
}
#endif

#endif /* EVENT_DETECTOR_H */
