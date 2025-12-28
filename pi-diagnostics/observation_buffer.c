/**
 * @file observation_buffer.c
 * @brief Lock-Free Circular Buffer Implementation
 */

#include "observation_buffer.h"
#include <string.h>

/*═══════════════════════════════════════════════════════════════════════════
 * INITIALIZATION
 *═══════════════════════════════════════════════════════════════════════════*/

void obs_buffer_init(ObservationBuffer *buf) {
    if (!buf) return;
    
    memset(buf->observations, 0, sizeof(buf->observations));
    memset(buf->timestamps, 0, sizeof(buf->timestamps));
    buf->write_head = 0;
    buf->total_count = 0;
}

/*═══════════════════════════════════════════════════════════════════════════
 * WRITER (RBPF Thread)
 *═══════════════════════════════════════════════════════════════════════════*/

void obs_buffer_push(ObservationBuffer *buf, double obs, int64_t tick) {
    if (!buf) return;
    
    int64_t head = buf->write_head;
    int64_t idx = head & OBS_BUFFER_MASK;
    
    /* Write data */
    buf->observations[idx] = obs;
    buf->timestamps[idx] = tick;
    
    /* Memory fence before updating head */
    OBS_MEMORY_FENCE();
    
    /* Update head (atomic on aligned 64-bit) */
    buf->write_head = head + 1;
    buf->total_count++;
}

/*═══════════════════════════════════════════════════════════════════════════
 * READER (Oracle Thread)
 *═══════════════════════════════════════════════════════════════════════════*/

int64_t obs_buffer_get_head(const ObservationBuffer *buf) {
    if (!buf) return 0;
    return buf->write_head;
}

int64_t obs_buffer_get_count(const ObservationBuffer *buf) {
    if (!buf) return 0;
    return buf->total_count;
}

int64_t obs_buffer_snapshot(
    const ObservationBuffer *buf,
    double *out_obs,
    int64_t *out_tick,
    int T)
{
    if (!buf || !out_obs || T <= 0) return -1;
    if (T > OBS_BUFFER_SIZE) return -1;
    
    /* Read head */
    int64_t head = buf->write_head;
    
    /* Check if we have enough data */
    if (head < T) return -1;
    
    /* Compute start position */
    int64_t start = head - T;
    
    /* Copy data */
    for (int i = 0; i < T; i++) {
        int64_t idx = (start + i) & OBS_BUFFER_MASK;
        out_obs[i] = buf->observations[idx];
        if (out_tick) {
            out_tick[i] = buf->timestamps[idx];
        }
    }
    
    /* Verify head didn't move too much during copy (torn read check) */
    OBS_MEMORY_FENCE();
    int64_t new_head = buf->write_head;
    
    if (new_head - head > OBS_BUFFER_SIZE / 2) {
        /* Buffer wrapped during read - data may be corrupted */
        return -1;
    }
    
    return head;  /* Return end tick */
}

int obs_buffer_snapshot_at(
    const ObservationBuffer *buf,
    int64_t end_tick,
    double *out_obs,
    int T)
{
    if (!buf || !out_obs || T <= 0) return -1;
    
    int64_t head = buf->write_head;
    
    /* Check if requested data is still in buffer */
    if (end_tick > head) return -1;  /* Future data */
    if (head - end_tick > OBS_BUFFER_SIZE - T) return -1;  /* Too old */
    
    /* Compute start position */
    int64_t start = end_tick - T;
    
    /* Copy data */
    for (int i = 0; i < T; i++) {
        int64_t idx = (start + i) & OBS_BUFFER_MASK;
        out_obs[i] = buf->observations[idx];
    }
    
    return 0;
}
