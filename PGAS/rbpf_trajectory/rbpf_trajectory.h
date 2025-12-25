/**
 * @file rbpf_trajectory.h
 * @brief Trajectory Buffer for RBPF → PGAS Reference Path
 *
 * PGAS requires a reference trajectory [regimes[T], h[T]] to condition on.
 * RBPF doesn't store history by default - this module adds a circular buffer
 * to capture the marginal MAP trajectory for Oracle handoff.
 *
 * THREAD SAFETY:
 * This buffer supports single-writer (RBPF) / single-reader (Oracle) access.
 * Two strategies are provided:
 *
 * 1. SNAPSHOT MODE (recommended for HFT):
 *    - Call rbpf_trajectory_snapshot() to atomically copy buffer state
 *    - Extract from the snapshot (zero contention with writer)
 *    - ~2x memory but zero latency impact on RBPF
 *
 * 2. DIRECT EXTRACTION with seqlock:
 *    - Uses sequence lock to detect concurrent writes
 *    - Retries extraction if write detected during read
 *    - Less memory but potential retry overhead
 *
 * Usage:
 *   RBPFTrajectory traj;
 *   rbpf_trajectory_init(&traj, 500, 4);  // 500 ticks, 4 regimes
 *
 *   // RBPF thread (each tick after rbpf_ksc_step()):
 *   rbpf_trajectory_record(&traj, output.dominant_regime, output.log_vol_mean);
 *
 *   // Oracle thread (when triggered):
 *   RBPFTrajectorySnapshot snap;
 *   rbpf_trajectory_snapshot(&traj, &snap);  // Atomic copy
 *
 *   int regimes[500];
 *   float h[500];
 *   int T = rbpf_trajectory_extract_from_snapshot(&snap, regimes, h, 500);
 *
 *   // Temper and feed to PGAS:
 *   rbpf_trajectory_temper(&traj, regimes, T, 0.05f);
 *   pgas_mkl_set_reference(pgas, regimes, h_double, T);
 */

#ifndef RBPF_TRAJECTORY_H
#define RBPF_TRAJECTORY_H

#include <stdint.h>
#include <stdbool.h>

/*═══════════════════════════════════════════════════════════════════════════
 * CROSS-PLATFORM ATOMIC TYPES
 *
 * MSVC doesn't support C11 <stdatomic.h> in C mode, so we use Windows
 * Interlocked functions on MSVC and C11 atomics on GCC/Clang.
 *
 * On x86/x64, aligned 32/64-bit loads are naturally atomic.
 * We use volatile to prevent compiler reordering, and explicit barriers
 * where memory ordering is needed (seqlock pattern requires acquire/release).
 *═══════════════════════════════════════════════════════════════════════════*/

#if defined(_MSC_VER)
/* MSVC: Use Windows Interlocked functions + volatile for aligned loads */
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <intrin.h>

typedef volatile LONG rbpf_atomic_int;
typedef volatile LONG64 rbpf_atomic_int64;
typedef volatile LONG rbpf_atomic_uint32;

/*─────────────────────────────────────────────────────────────────────────
 * LOADS: On x86/x64, aligned volatile reads are atomic.
 * We add _ReadBarrier() for acquire semantics where needed.
 *─────────────────────────────────────────────────────────────────────────*/

/* Relaxed load - just atomicity, no ordering guarantee */
#define RBPF_ATOMIC_LOAD_INT_RELAXED(ptr) (*(volatile LONG *)(ptr))
#define RBPF_ATOMIC_LOAD_INT64_RELAXED(ptr) (*(volatile LONG64 *)(ptr))
#define RBPF_ATOMIC_LOAD_UINT32_RELAXED(ptr) ((uint32_t)(*(volatile LONG *)(ptr)))

/* Acquire load - ensures subsequent reads see prior writes */
static __forceinline LONG rbpf_atomic_load_int_acquire(volatile LONG const *ptr)
{
    LONG val = *ptr;
    _ReadBarrier();
    return val;
}
static __forceinline LONG64 rbpf_atomic_load_int64_acquire(volatile LONG64 const *ptr)
{
    LONG64 val = *ptr;
    _ReadBarrier();
    return val;
}
static __forceinline uint32_t rbpf_atomic_load_uint32_acquire(volatile LONG const *ptr)
{
    LONG val = *ptr;
    _ReadBarrier();
    return (uint32_t)val;
}

/* Default loads use acquire semantics (safe for seqlock) */
#define RBPF_ATOMIC_LOAD_INT(ptr) rbpf_atomic_load_int_acquire((volatile LONG const *)(ptr))
#define RBPF_ATOMIC_LOAD_INT64(ptr) rbpf_atomic_load_int64_acquire((volatile LONG64 const *)(ptr))
#define RBPF_ATOMIC_LOAD_UINT32(ptr) rbpf_atomic_load_uint32_acquire((volatile LONG const *)(ptr))

/*─────────────────────────────────────────────────────────────────────────
 * STORES: Use InterlockedExchange for release semantics.
 * This ensures prior writes are visible before the store completes.
 *─────────────────────────────────────────────────────────────────────────*/
#define RBPF_ATOMIC_STORE_INT(ptr, val) InterlockedExchange((volatile LONG *)(ptr), (val))
#define RBPF_ATOMIC_STORE_INT64(ptr, val) InterlockedExchange64((volatile LONG64 *)(ptr), (val))
#define RBPF_ATOMIC_STORE_UINT32(ptr, val) InterlockedExchange((volatile LONG *)(ptr), (LONG)(val))

/*─────────────────────────────────────────────────────────────────────────
 * READ-MODIFY-WRITE: Full atomic operations
 *─────────────────────────────────────────────────────────────────────────*/
#define RBPF_ATOMIC_FETCH_ADD_INT(ptr, val) InterlockedExchangeAdd((volatile LONG *)(ptr), (val))
#define RBPF_ATOMIC_FETCH_ADD_INT64(ptr, val) InterlockedExchangeAdd64((volatile LONG64 *)(ptr), (val))
#define RBPF_ATOMIC_FETCH_ADD_UINT32(ptr, val) ((uint32_t)InterlockedExchangeAdd((volatile LONG *)(ptr), (LONG)(val)))

#else
/* GCC/Clang: Use C11 atomics */
#include <stdatomic.h>

typedef _Atomic int rbpf_atomic_int;
typedef _Atomic int64_t rbpf_atomic_int64;
typedef _Atomic uint32_t rbpf_atomic_uint32;

/* Relaxed loads */
#define RBPF_ATOMIC_LOAD_INT_RELAXED(ptr) atomic_load_explicit((ptr), memory_order_relaxed)
#define RBPF_ATOMIC_LOAD_INT64_RELAXED(ptr) atomic_load_explicit((ptr), memory_order_relaxed)
#define RBPF_ATOMIC_LOAD_UINT32_RELAXED(ptr) atomic_load_explicit((ptr), memory_order_relaxed)

/* Acquire loads (default) */
#define RBPF_ATOMIC_LOAD_INT(ptr) atomic_load_explicit((ptr), memory_order_acquire)
#define RBPF_ATOMIC_LOAD_INT64(ptr) atomic_load_explicit((ptr), memory_order_acquire)
#define RBPF_ATOMIC_LOAD_UINT32(ptr) atomic_load_explicit((ptr), memory_order_acquire)

/* Release stores */
#define RBPF_ATOMIC_STORE_INT(ptr, val) atomic_store_explicit((ptr), (val), memory_order_release)
#define RBPF_ATOMIC_STORE_INT64(ptr, val) atomic_store_explicit((ptr), (val), memory_order_release)
#define RBPF_ATOMIC_STORE_UINT32(ptr, val) atomic_store_explicit((ptr), (val), memory_order_release)

/* Fetch-add with acquire-release semantics */
#define RBPF_ATOMIC_FETCH_ADD_INT(ptr, val) atomic_fetch_add_explicit((ptr), (val), memory_order_acq_rel)
#define RBPF_ATOMIC_FETCH_ADD_INT64(ptr, val) atomic_fetch_add_explicit((ptr), (val), memory_order_acq_rel)
#define RBPF_ATOMIC_FETCH_ADD_UINT32(ptr, val) atomic_fetch_add_explicit((ptr), (val), memory_order_acq_rel)
#endif

#ifdef __cplusplus
extern "C"
{
#endif

    /*═══════════════════════════════════════════════════════════════════════════
     * CONFIGURATION
     *═══════════════════════════════════════════════════════════════════════════*/

#ifndef RBPF_TRAJ_MAX_T
#define RBPF_TRAJ_MAX_T 2048 /* Maximum trajectory length */
#endif

#ifndef RBPF_TRAJ_MAX_REGIMES
#define RBPF_TRAJ_MAX_REGIMES 8
#endif

    /*═══════════════════════════════════════════════════════════════════════════
     * DATA STRUCTURES
     *═══════════════════════════════════════════════════════════════════════════*/

    /**
     * Trajectory buffer configuration
     */
    typedef struct
    {
        int T_max;         /* Maximum trajectory length */
        int n_regimes;     /* Number of regimes (for tempering) */
        float temper_prob; /* Default tempering probability (0.05) */
        uint64_t seed;     /* RNG seed for tempering */
    } RBPFTrajectoryConfig;

    /**
     * Trajectory buffer state
     *
     * THREAD SAFETY: Uses seqlock pattern for lock-free single-writer/single-reader.
     * - Writer (RBPF): increments seq before/after write (always odd during write)
     * - Reader (Oracle): checks seq before/after read, retries if changed or odd
     */
    typedef struct
    {
        RBPFTrajectoryConfig config;

        /* Circular buffer storage */
        int *regimes; /* [T_max] regime at each tick */
        float *h;     /* [T_max] log-vol at each tick */

        /* Buffer state - atomic for thread safety */
        rbpf_atomic_int head;          /* Next write position */
        rbpf_atomic_int count;         /* Number of valid entries (up to T_max) */
        rbpf_atomic_int64 total_ticks; /* Total ticks recorded */

        /* Seqlock for consistent snapshots */
        rbpf_atomic_uint32 seq; /* Sequence number (odd = write in progress) */

        /* RNG state for tempering (xoroshiro128+) */
        uint64_t rng_state[2];

        /* Timestamp of oldest entry (for age tracking) */
        int64_t oldest_tick;

        bool initialized;

    } RBPFTrajectory;

    /**
     * Snapshot for thread-safe extraction
     *
     * Oracle thread takes a snapshot, then extracts from it.
     * Zero contention with RBPF writer during extraction.
     */
    typedef struct
    {
        int T_max;
        int n_regimes;

        int *regimes; /* [T_max] copied regime data */
        float *h;     /* [T_max] copied log-vol data */

        int head;            /* Snapshot of head at copy time */
        int count;           /* Snapshot of count at copy time */
        int64_t total_ticks; /* Snapshot of total_ticks */

        bool valid; /* True if snapshot was taken successfully */
    } RBPFTrajectorySnapshot;

    /**
     * Extraction result
     */
    typedef struct
    {
        int T;              /* Extracted length */
        int64_t start_tick; /* Tick number of first entry */
        int64_t end_tick;   /* Tick number of last entry */
        float fill_ratio;   /* How full the buffer is (count / T_max) */
    } RBPFTrajectoryExtractResult;

    /*═══════════════════════════════════════════════════════════════════════════
     * API - LIFECYCLE
     *═══════════════════════════════════════════════════════════════════════════*/

    /**
     * Get default configuration
     */
    RBPFTrajectoryConfig rbpf_trajectory_config_defaults(int T_max, int n_regimes);

    /**
     * Initialize trajectory buffer
     *
     * @param traj      Trajectory state
     * @param config    Configuration (NULL for defaults with T_max=512, K=4)
     * @return          0 on success, -1 on failure
     */
    int rbpf_trajectory_init(RBPFTrajectory *traj, const RBPFTrajectoryConfig *config);

    /**
     * Initialize with simple parameters
     */
    int rbpf_trajectory_init_simple(RBPFTrajectory *traj, int T_max, int n_regimes);

    /**
     * Reset buffer (keep config)
     */
    void rbpf_trajectory_reset(RBPFTrajectory *traj);

    /**
     * Free resources
     */
    void rbpf_trajectory_free(RBPFTrajectory *traj);

    /*═══════════════════════════════════════════════════════════════════════════
     * API - RECORDING
     *═══════════════════════════════════════════════════════════════════════════*/

    /**
     * Record one tick of trajectory
     *
     * Call this after each rbpf_ksc_step() with the output values.
     *
     * @param traj      Trajectory state
     * @param regime    Dominant regime from RBPF output
     * @param h         Log-vol mean from RBPF output
     */
    void rbpf_trajectory_record(RBPFTrajectory *traj, int regime, float h);

/**
 * Record from RBPF output struct (convenience wrapper)
 *
 * Assumes output has fields: dominant_regime, log_vol_mean
 * Use rbpf_trajectory_record() directly if your struct differs.
 */
#define rbpf_trajectory_record_output(traj, output) \
    rbpf_trajectory_record((traj), (output)->dominant_regime, (float)(output)->log_vol_mean)

    /*═══════════════════════════════════════════════════════════════════════════
     * API - EXTRACTION
     *═══════════════════════════════════════════════════════════════════════════*/

    /**
     * Extract trajectory for PGAS reference path
     *
     * Extracts up to T_requested ticks in chronological order.
     * If buffer has fewer entries, returns what's available.
     *
     * @param traj          Trajectory state
     * @param regimes_out   Output buffer for regimes [T_requested]
     * @param h_out         Output buffer for log-vol [T_requested]
     * @param T_requested   Requested length
     * @return              Extraction result with actual length
     */
    RBPFTrajectoryExtractResult rbpf_trajectory_extract(
        const RBPFTrajectory *traj,
        int *regimes_out,
        float *h_out,
        int T_requested);

    /**
     * Extract to double precision (for PGAS interface)
     */
    RBPFTrajectoryExtractResult rbpf_trajectory_extract_double(
        const RBPFTrajectory *traj,
        int *regimes_out,
        double *h_out,
        int T_requested);

    /**
     * Get current buffer length
     */
    int rbpf_trajectory_length(const RBPFTrajectory *traj);

    /**
     * Check if buffer has enough data for Oracle
     */
    bool rbpf_trajectory_ready(const RBPFTrajectory *traj, int min_length);

    /*═══════════════════════════════════════════════════════════════════════════
     * API - THREAD-SAFE SNAPSHOT (Recommended for HFT)
     *
     * The snapshot approach copies the entire buffer state atomically,
     * allowing the Oracle thread to extract without contention.
     *═══════════════════════════════════════════════════════════════════════════*/

    /**
     * Allocate snapshot buffers
     *
     * Must be called before rbpf_trajectory_snapshot().
     * Typically called once during Oracle initialization.
     *
     * @param snap      Snapshot state to initialize
     * @param T_max     Buffer size (must match trajectory)
     * @param n_regimes Number of regimes
     * @return          0 on success, -1 on failure
     */
    int rbpf_trajectory_snapshot_alloc(RBPFTrajectorySnapshot *snap, int T_max, int n_regimes);

    /**
     * Free snapshot buffers
     */
    void rbpf_trajectory_snapshot_free(RBPFTrajectorySnapshot *snap);

    /**
     * Take atomic snapshot of trajectory buffer
     *
     * Uses seqlock to ensure consistent copy:
     * - Waits for any in-progress write to complete
     * - Copies buffer state
     * - Verifies no write occurred during copy
     * - Retries if necessary
     *
     * @param traj      Source trajectory (RBPF thread may be writing)
     * @param snap      Destination snapshot (pre-allocated)
     * @param max_retries Maximum retry attempts (default: 3)
     * @return          0 on success, -1 if retries exhausted
     */
    int rbpf_trajectory_snapshot(const RBPFTrajectory *traj,
                                 RBPFTrajectorySnapshot *snap,
                                 int max_retries);

    /**
     * Extract from snapshot (no thread-safety concerns)
     */
    RBPFTrajectoryExtractResult rbpf_trajectory_extract_from_snapshot(
        const RBPFTrajectorySnapshot *snap,
        int *regimes_out,
        float *h_out,
        int T_requested);

    /**
     * Extract to double precision from snapshot
     */
    RBPFTrajectoryExtractResult rbpf_trajectory_extract_from_snapshot_double(
        const RBPFTrajectorySnapshot *snap,
        int *regimes_out,
        double *h_out,
        int T_requested);

    /*═══════════════════════════════════════════════════════════════════════════
     * API - TEMPERING
     *
     * Tempered Injection (Chopin & Papaspiliopoulos 2020):
     * Randomly flip a fraction of regime assignments to break confirmation bias.
     * Without tempering, PGAS conditions on RBPF's MAP path and may reinforce
     * errors. With 5% tempering, we inject exploration.
     *═══════════════════════════════════════════════════════════════════════════*/

    /**
     * Temper a regime trajectory (in-place)
     *
     * Randomly flips each regime with probability `flip_prob` to a
     * uniformly chosen different regime.
     *
     * @param traj          Trajectory state (for RNG and n_regimes)
     * @param regimes       Regime trajectory to temper [T] (modified in-place)
     * @param T             Length of trajectory
     * @param flip_prob     Probability of flipping each entry (default: 0.05)
     * @return              Number of flips applied
     */
    int rbpf_trajectory_temper(
        RBPFTrajectory *traj,
        int *regimes,
        int T,
        float flip_prob);

    /**
     * Temper with default probability (from config)
     */
    int rbpf_trajectory_temper_default(RBPFTrajectory *traj, int *regimes, int T);

    /*═══════════════════════════════════════════════════════════════════════════
     * API - QUERIES
     *═══════════════════════════════════════════════════════════════════════════*/

    /**
     * Get most recent regime
     */
    int rbpf_trajectory_last_regime(const RBPFTrajectory *traj);

    /**
     * Get most recent log-vol
     */
    float rbpf_trajectory_last_h(const RBPFTrajectory *traj);

    /**
     * Get total ticks recorded
     */
    int64_t rbpf_trajectory_total_ticks(const RBPFTrajectory *traj);

    /**
     * Compute regime distribution over buffer
     *
     * @param traj          Trajectory state
     * @param probs_out     Output probabilities [n_regimes]
     */
    void rbpf_trajectory_regime_distribution(
        const RBPFTrajectory *traj,
        float *probs_out);

    /*═══════════════════════════════════════════════════════════════════════════
     * API - DIAGNOSTICS
     *═══════════════════════════════════════════════════════════════════════════*/

    /**
     * Print buffer state
     */
    void rbpf_trajectory_print_state(const RBPFTrajectory *traj);

    /**
     * Print last N entries
     */
    void rbpf_trajectory_print_tail(const RBPFTrajectory *traj, int n);

#ifdef __cplusplus
}
#endif

#endif /* RBPF_TRAJECTORY_H */