/**
 * @file observation_buffer.h
 * @brief Lock-Free Circular Buffer for Observations
 *
 * Shared between RBPF (writer) and Oracle (reader).
 * RBPF pushes observations every tick.
 * Oracle snapshots a window for PGAS processing.
 */

#ifndef OBSERVATION_BUFFER_H
#define OBSERVATION_BUFFER_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/*═══════════════════════════════════════════════════════════════════════════
 * CONFIGURATION
 *═══════════════════════════════════════════════════════════════════════════*/

#ifndef OBS_BUFFER_SIZE
#define OBS_BUFFER_SIZE 8192  /* Power of 2 for fast modulo */
#endif

#define OBS_BUFFER_MASK (OBS_BUFFER_SIZE - 1)

/*═══════════════════════════════════════════════════════════════════════════
 * PLATFORM-SPECIFIC ALIGNMENT
 *═══════════════════════════════════════════════════════════════════════════*/

#if defined(_MSC_VER)
    #define OBS_ALIGNED64 __declspec(align(64))
    #include <intrin.h>
    #define OBS_MEMORY_FENCE() _mm_mfence()
#elif defined(__GNUC__) || defined(__clang__)
    #define OBS_ALIGNED64 __attribute__((aligned(64)))
    #define OBS_MEMORY_FENCE() __sync_synchronize()
#else
    #define OBS_ALIGNED64
    #define OBS_MEMORY_FENCE()
#endif

/*═══════════════════════════════════════════════════════════════════════════
 * OBSERVATION BUFFER
 *═══════════════════════════════════════════════════════════════════════════*/

typedef struct {
    /* Observation data */
    double observations[OBS_BUFFER_SIZE];
    
    /* Timestamps (tick indices) */
    int64_t timestamps[OBS_BUFFER_SIZE];
    
    /* Write head - only RBPF writes, cache-line isolated */
    OBS_ALIGNED64 volatile int64_t write_head;
    
    /* Padding to prevent false sharing */
    OBS_ALIGNED64 char _padding[64];
    
    /* Total observations pushed (monotonic) */
    volatile int64_t total_count;
    
} ObservationBuffer;

/*═══════════════════════════════════════════════════════════════════════════
 * API
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Initialize buffer
 */
void obs_buffer_init(ObservationBuffer *buf);

/**
 * Push observation (RBPF thread only)
 * 
 * @param buf    Buffer
 * @param obs    Observation value
 * @param tick   Tick index
 */
void obs_buffer_push(ObservationBuffer *buf, double obs, int64_t tick);

/**
 * Get current write head position
 */
int64_t obs_buffer_get_head(const ObservationBuffer *buf);

/**
 * Get total observations pushed
 */
int64_t obs_buffer_get_count(const ObservationBuffer *buf);

/**
 * Snapshot a window of observations (Oracle thread)
 * 
 * @param buf       Buffer
 * @param out_obs   Output array for observations (size T)
 * @param out_tick  Output array for timestamps (size T, can be NULL)
 * @param T         Window size
 * @return          End tick index, or -1 if not enough data
 */
int64_t obs_buffer_snapshot(
    const ObservationBuffer *buf,
    double *out_obs,
    int64_t *out_tick,
    int T);

/**
 * Snapshot window ending at specific tick
 * 
 * @param buf       Buffer
 * @param end_tick  End tick (inclusive)
 * @param out_obs   Output array
 * @param T         Window size
 * @return          0 on success, -1 if data not available
 */
int obs_buffer_snapshot_at(
    const ObservationBuffer *buf,
    int64_t end_tick,
    double *out_obs,
    int T);

#ifdef __cplusplus
}
#endif

#endif /* OBSERVATION_BUFFER_H */
