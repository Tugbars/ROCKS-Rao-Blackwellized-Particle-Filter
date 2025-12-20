/**
 * @file circular_buffer.h
 * @brief Lock-free circular buffer for observation buffering during PGAS
 *
 * Fixed capacity (512 ticks) with power-of-2 indexing for fast bitwise ops.
 * Designed for single-producer (main thread) single-consumer (background thread).
 *
 * Overflow policy: Drop oldest (staleness cutoff).
 * Rationale: A Lifeboat based on 500-tick-old data is useful;
 *            one based on 5000-tick-old data describes an ended regime.
 */

#ifndef CIRCULAR_BUFFER_H
#define CIRCULAR_BUFFER_H

#include <stdint.h>
#include <stdbool.h>
#include <stdatomic.h>

#ifdef __cplusplus
extern "C" {
#endif

/*═══════════════════════════════════════════════════════════════════════════════
 * CONFIGURATION
 *═══════════════════════════════════════════════════════════════════════════════*/

/** Buffer capacity - MUST be power of 2 for fast modulo via bitmask */
#define CBUF_CAPACITY 512
#define CBUF_MASK     (CBUF_CAPACITY - 1)

/*═══════════════════════════════════════════════════════════════════════════════
 * DATA STRUCTURES
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * Observation entry with timestamp
 */
typedef struct {
    double y;           /**< Observation value (e.g., log-return) */
    uint64_t tick_id;   /**< Monotonic tick counter */
} ObservationEntry;

/**
 * Circular buffer for observations
 * 
 * Single-producer single-consumer (SPSC) design.
 * Main thread pushes, background thread bulk-reads.
 */
typedef struct {
    ObservationEntry data[CBUF_CAPACITY] __attribute__((aligned(64)));
    
    /** Write position (main thread only) */
    _Atomic uint32_t head;
    
    /** Read position (background thread only) */
    _Atomic uint32_t tail;
    
    /** Statistics */
    uint64_t total_pushed;
    uint64_t total_dropped;
    uint64_t total_consumed;
    
} CircularBuffer;

/**
 * Snapshot of buffer for background thread consumption
 */
typedef struct {
    double observations[CBUF_CAPACITY];
    uint64_t tick_ids[CBUF_CAPACITY];
    int count;
    uint64_t first_tick;
    uint64_t last_tick;
} BufferSnapshot;

/*═══════════════════════════════════════════════════════════════════════════════
 * INITIALIZATION
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Initialize circular buffer
 * @param buf Buffer to initialize
 */
void cbuf_init(CircularBuffer *buf);

/**
 * @brief Reset buffer to empty state
 * @param buf Buffer to reset
 */
void cbuf_reset(CircularBuffer *buf);

/*═══════════════════════════════════════════════════════════════════════════════
 * MAIN THREAD OPERATIONS (Producer)
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Push observation to buffer (main thread)
 *
 * If buffer is full, drops oldest entry (staleness cutoff).
 * This is O(1) with fast bitwise indexing.
 *
 * @param buf     Circular buffer
 * @param y       Observation value
 * @param tick_id Monotonic tick counter
 * @return        true if pushed, false if dropped oldest to make room
 */
bool cbuf_push(CircularBuffer *buf, double y, uint64_t tick_id);

/**
 * @brief Get current buffer occupancy
 * @param buf Circular buffer
 * @return Number of entries in buffer
 */
uint32_t cbuf_count(const CircularBuffer *buf);

/**
 * @brief Check if buffer is full
 * @param buf Circular buffer
 * @return true if at capacity
 */
bool cbuf_is_full(const CircularBuffer *buf);

/**
 * @brief Check if buffer is empty
 * @param buf Circular buffer
 * @return true if empty
 */
bool cbuf_is_empty(const CircularBuffer *buf);

/*═══════════════════════════════════════════════════════════════════════════════
 * BACKGROUND THREAD OPERATIONS (Consumer)
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Take snapshot of current buffer contents (background thread)
 *
 * Copies all buffered observations to snapshot struct.
 * Does NOT clear the buffer - call cbuf_consume() after processing.
 *
 * @param buf      Circular buffer
 * @param snapshot Output: copied observations
 * @return         Number of observations in snapshot
 */
int cbuf_snapshot(const CircularBuffer *buf, BufferSnapshot *snapshot);

/**
 * @brief Mark entries as consumed (background thread)
 *
 * Advances tail pointer by count entries.
 * Call after successfully processing a snapshot.
 *
 * @param buf   Circular buffer
 * @param count Number of entries to consume
 */
void cbuf_consume(CircularBuffer *buf, int count);

/**
 * @brief Peek at oldest entry without consuming
 * @param buf   Circular buffer
 * @param entry Output: oldest entry (if exists)
 * @return      true if entry exists, false if empty
 */
bool cbuf_peek_oldest(const CircularBuffer *buf, ObservationEntry *entry);

/**
 * @brief Peek at newest entry without consuming
 * @param buf   Circular buffer
 * @param entry Output: newest entry (if exists)
 * @return      true if entry exists, false if empty
 */
bool cbuf_peek_newest(const CircularBuffer *buf, ObservationEntry *entry);

/*═══════════════════════════════════════════════════════════════════════════════
 * DIAGNOSTICS
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Get buffer statistics
 * @param buf      Circular buffer
 * @param pushed   Output: total observations pushed
 * @param dropped  Output: total observations dropped (overflow)
 * @param consumed Output: total observations consumed
 */
void cbuf_get_stats(const CircularBuffer *buf,
                    uint64_t *pushed,
                    uint64_t *dropped,
                    uint64_t *consumed);

/**
 * @brief Get staleness (age of oldest entry in ticks)
 * @param buf         Circular buffer
 * @param current_tick Current tick counter
 * @return            Age in ticks, or 0 if empty
 */
uint64_t cbuf_get_staleness(const CircularBuffer *buf, uint64_t current_tick);

#ifdef __cplusplus
}
#endif

#endif /* CIRCULAR_BUFFER_H */
