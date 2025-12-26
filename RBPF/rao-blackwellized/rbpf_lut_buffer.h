/**
 * @file rbpf_lut_buffer.h
 * @brief Thread-Safe Double-Buffered Transition LUT
 *
 * Solves the Oracle→RBPF race condition where Oracle writes ~16KB
 * while RBPF reads. Uses atomic index swap for lock-free safety.
 *
 * USAGE:
 *   Writer (Oracle thread):
 *     rbpf_lut_begin_write(&buf)   → get shadow buffer pointer
 *     ... fill shadow buffer ...
 *     rbpf_lut_commit_write(&buf)  → atomic swap
 *
 *   Reader (RBPF hot path):
 *     const uint8_t (*lut)[N] = rbpf_lut_acquire_read(&buf);
 *     ... use lut for entire particle loop ...
 *     (no release needed - pointer stays valid until next acquire)
 *
 * GUARANTEES:
 *   - Reader never sees partial writes
 *   - Writer never blocks reader
 *   - Reader never blocks writer
 *   - Total memory: 2 × K × LUT_SIZE bytes
 */

#ifndef RBPF_LUT_BUFFER_H
#define RBPF_LUT_BUFFER_H

#include <stdint.h>
#include <string.h>

#ifdef _WIN32
#include <intrin.h>
#pragma intrinsic(_InterlockedExchange, _ReadWriteBarrier)
#else
/* GCC/Clang atomics */
#endif

/*═══════════════════════════════════════════════════════════════════════════
 * CONFIGURATION
 *═══════════════════════════════════════════════════════════════════════════*/

#ifndef RBPF_LUT_MAX_REGIMES
#define RBPF_LUT_MAX_REGIMES 8
#endif

#ifndef RBPF_LUT_SIZE
#define RBPF_LUT_SIZE 2048
#endif

#define RBPF_LUT_ALIGN 64

/*═══════════════════════════════════════════════════════════════════════════
 * DOUBLE BUFFER STRUCTURE
 *═══════════════════════════════════════════════════════════════════════════*/

typedef struct RBPF_LUT_Buffer
{
    /* Two complete LUT copies - 64-byte aligned for cache line efficiency */
    __declspec(align(64)) uint8_t buffers[2][RBPF_LUT_MAX_REGIMES][RBPF_LUT_SIZE];

    /* Which buffer is currently active for readers (0 or 1) */
    volatile long active_idx;

    /* Diagnostics */
    volatile long write_count;    /* Total writes committed */
    volatile long swap_count;     /* Should equal write_count */

} RBPF_LUT_Buffer;

/*═══════════════════════════════════════════════════════════════════════════
 * LIFECYCLE
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Initialize double buffer (call once at startup)
 */
static inline void rbpf_lut_init(RBPF_LUT_Buffer *buf)
{
    memset(buf->buffers, 0, sizeof(buf->buffers));
    buf->active_idx = 0;
    buf->write_count = 0;
    buf->swap_count = 0;
}

/*═══════════════════════════════════════════════════════════════════════════
 * WRITER API (Oracle Thread)
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Get shadow buffer for writing
 *
 * Returns pointer to the buffer that readers are NOT using.
 * Safe to write any amount of data - readers won't see it until commit.
 *
 * @param buf       The double buffer
 * @return          Pointer to shadow buffer [K][LUT_SIZE]
 */
static inline uint8_t (*rbpf_lut_begin_write(RBPF_LUT_Buffer *buf))[RBPF_LUT_SIZE]
{
    long shadow_idx = 1 - buf->active_idx;
    return buf->buffers[shadow_idx];
}

/**
 * @brief Commit shadow buffer (atomic swap)
 *
 * Makes the shadow buffer active. Readers will see new data on next acquire.
 * This is a single atomic instruction - no blocking possible.
 *
 * @param buf       The double buffer
 */
static inline void rbpf_lut_commit_write(RBPF_LUT_Buffer *buf)
{
    long shadow_idx = 1 - buf->active_idx;

#ifdef _WIN32
    /* Full memory barrier + atomic write */
    _InterlockedExchange(&buf->active_idx, shadow_idx);
#else
    __atomic_store_n(&buf->active_idx, shadow_idx, __ATOMIC_RELEASE);
#endif

    buf->write_count++;
    buf->swap_count++;
}

/*═══════════════════════════════════════════════════════════════════════════
 * READER API (RBPF Hot Path)
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Acquire active buffer for reading
 *
 * Returns pointer to the currently active buffer. This pointer remains
 * valid even if a write commits during your read loop - you'll just
 * see the old data until next acquire.
 *
 * CRITICAL: Call once per step, cache the pointer, use for entire loop.
 *           Do NOT call inside particle loop.
 *
 * @param buf       The double buffer
 * @return          Pointer to active buffer [K][LUT_SIZE] (read-only)
 */
static inline const uint8_t (*rbpf_lut_acquire_read(RBPF_LUT_Buffer *buf))[RBPF_LUT_SIZE]
{
#ifdef _WIN32
    _ReadWriteBarrier();
    long idx = buf->active_idx;
    _ReadWriteBarrier();
#else
    long idx = __atomic_load_n(&buf->active_idx, __ATOMIC_ACQUIRE);
#endif
    return (const uint8_t (*)[RBPF_LUT_SIZE])buf->buffers[idx];
}

/**
 * @brief Direct buffer access (NOT thread-safe - for init/debug only)
 */
static inline uint8_t (*rbpf_lut_get_buffer(RBPF_LUT_Buffer *buf, int idx))[RBPF_LUT_SIZE]
{
    return buf->buffers[idx & 1];
}

/*═══════════════════════════════════════════════════════════════════════════
 * DIAGNOSTICS
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Check buffer health
 *
 * @return 1 if healthy (write_count == swap_count), 0 if corrupted
 */
static inline int rbpf_lut_check_health(const RBPF_LUT_Buffer *buf)
{
    return (buf->write_count == buf->swap_count) ? 1 : 0;
}

/**
 * @brief Get current active index (for debugging)
 */
static inline int rbpf_lut_active_index(const RBPF_LUT_Buffer *buf)
{
    return (int)buf->active_idx;
}

/**
 * @brief Get write count (for debugging)
 */
static inline long rbpf_lut_write_count(const RBPF_LUT_Buffer *buf)
{
    return buf->write_count;
}

#endif /* RBPF_LUT_BUFFER_H */