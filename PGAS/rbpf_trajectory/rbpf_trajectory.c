/**
 * @file rbpf_trajectory.c
 * @brief Trajectory Buffer Implementation
 *
 * THREAD SAFETY:
 * - Uses cross-platform atomics (C11 on GCC/Clang, Interlocked on MSVC)
 * - Seqlock pattern for consistent snapshots
 * - Memory barriers ensure visibility across threads
 */

#include "rbpf_trajectory.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/*═══════════════════════════════════════════════════════════════════════════
 * CROSS-PLATFORM MEMORY BARRIERS
 *═══════════════════════════════════════════════════════════════════════════*/

#if defined(_MSC_VER)
/* MSVC: Use compiler intrinsics */
#include <intrin.h>
#define COMPILER_BARRIER() _ReadWriteBarrier()
#define MEMORY_FENCE() MemoryBarrier()
#elif defined(__GNUC__) || defined(__clang__)
/* GCC/Clang: Use builtins */
#define COMPILER_BARRIER() __asm__ __volatile__("" ::: "memory")
#define MEMORY_FENCE() __sync_synchronize()
#else
#define COMPILER_BARRIER()
#define MEMORY_FENCE()
#endif

/*═══════════════════════════════════════════════════════════════════════════
 * CROSS-PLATFORM ALIGNED ALLOCATION (64-byte for AVX-512)
 *
 * pgas_mkl_set_reference may use AVX-512 instructions (vmovaps) which
 * require 64-byte alignment for optimal performance.
 *═══════════════════════════════════════════════════════════════════════════*/

#define RBPF_TRAJ_ALIGNMENT 64 /* AVX-512 / cache line */

#if defined(_MSC_VER) || defined(__INTEL_COMPILER)
/* MSVC/Intel: Use _mm_malloc/_mm_free */
#include <malloc.h>
#define RBPF_ALIGNED_ALLOC(size, align) _mm_malloc((size), (align))
#define RBPF_ALIGNED_FREE(ptr) _mm_free(ptr)
#elif defined(__GNUC__) && !defined(__APPLE__)
/* GCC on Linux: Use aligned_alloc (C11) */
#define RBPF_ALIGNED_ALLOC(size, align) aligned_alloc((align), (size))
#define RBPF_ALIGNED_FREE(ptr) free(ptr)
#elif defined(__APPLE__)
/* macOS: aligned_alloc available since macOS 10.15 */
#include <stdlib.h>
static inline void *rbpf_aligned_alloc_posix(size_t align, size_t size)
{
    void *ptr = NULL;
    posix_memalign(&ptr, align, size);
    return ptr;
}
#define RBPF_ALIGNED_ALLOC(size, align) rbpf_aligned_alloc_posix((align), (size))
#define RBPF_ALIGNED_FREE(ptr) free(ptr)
#else
/* Fallback: Regular malloc (may not be aligned) */
#define RBPF_ALIGNED_ALLOC(size, align) malloc(size)
#define RBPF_ALIGNED_FREE(ptr) free(ptr)
#endif

/*═══════════════════════════════════════════════════════════════════════════
 * RNG (xoroshiro128+)
 *═══════════════════════════════════════════════════════════════════════════*/

static inline uint64_t rotl(const uint64_t x, int k)
{
    return (x << k) | (x >> (64 - k));
}

static uint64_t xoroshiro128plus(uint64_t *s)
{
    const uint64_t s0 = s[0];
    uint64_t s1 = s[1];
    const uint64_t result = s0 + s1;

    s1 ^= s0;
    s[0] = rotl(s0, 24) ^ s1 ^ (s1 << 16);
    s[1] = rotl(s1, 37);

    return result;
}

static inline float rand_uniform(uint64_t *s)
{
    return (xoroshiro128plus(s) >> 11) * (1.0f / 9007199254740992.0f);
}

static inline int rand_int(uint64_t *s, int max)
{
    return (int)(rand_uniform(s) * max);
}

static void seed_rng(uint64_t *s, uint64_t seed)
{
    uint64_t z = seed;

    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    s[0] = z ^ (z >> 31);

    z = seed + 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    s[1] = z ^ (z >> 31);

    if (s[0] == 0 && s[1] == 0)
    {
        s[0] = 1;
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * CONFIGURATION
 *═══════════════════════════════════════════════════════════════════════════*/

RBPFTrajectoryConfig rbpf_trajectory_config_defaults(int T_max, int n_regimes)
{
    RBPFTrajectoryConfig cfg;
    memset(&cfg, 0, sizeof(cfg));

    cfg.T_max = (T_max > 0 && T_max <= RBPF_TRAJ_MAX_T) ? T_max : 512;
    cfg.n_regimes = (n_regimes > 0 && n_regimes <= RBPF_TRAJ_MAX_REGIMES) ? n_regimes : 4;
    cfg.temper_prob = 0.05f; /* 5% default per Chopin & Papaspiliopoulos */
    cfg.seed = 0xDEADBEEF;

    return cfg;
}

/*═══════════════════════════════════════════════════════════════════════════
 * LIFECYCLE
 *═══════════════════════════════════════════════════════════════════════════*/

int rbpf_trajectory_init(RBPFTrajectory *traj, const RBPFTrajectoryConfig *config)
{
    if (!traj)
        return -1;

    memset(traj, 0, sizeof(*traj));

    traj->config = config ? *config : rbpf_trajectory_config_defaults(512, 4);

    int T = traj->config.T_max;

    /* Allocate 64-byte aligned buffers for AVX-512 compatibility */
    traj->regimes = (int *)RBPF_ALIGNED_ALLOC(T * sizeof(int), RBPF_TRAJ_ALIGNMENT);
    traj->h = (float *)RBPF_ALIGNED_ALLOC(T * sizeof(float), RBPF_TRAJ_ALIGNMENT);

    if (!traj->regimes || !traj->h)
    {
        rbpf_trajectory_free(traj);
        return -1;
    }

    /* Zero the buffers */
    memset(traj->regimes, 0, T * sizeof(int));
    memset(traj->h, 0, T * sizeof(float));

    /* Initialize RNG */
    seed_rng(traj->rng_state, traj->config.seed);

    /* Initialize atomic fields */
    RBPF_ATOMIC_STORE_INT(&traj->head, 0);
    RBPF_ATOMIC_STORE_INT(&traj->count, 0);
    RBPF_ATOMIC_STORE_INT64(&traj->total_ticks, 0);
    RBPF_ATOMIC_STORE_UINT32(&traj->seq, 0); /* Even = no write in progress */

    traj->oldest_tick = 0;
    traj->initialized = true;

    return 0;
}

int rbpf_trajectory_init_simple(RBPFTrajectory *traj, int T_max, int n_regimes)
{
    RBPFTrajectoryConfig cfg = rbpf_trajectory_config_defaults(T_max, n_regimes);
    return rbpf_trajectory_init(traj, &cfg);
}

void rbpf_trajectory_reset(RBPFTrajectory *traj)
{
    if (!traj || !traj->initialized)
        return;

    /* Acquire seqlock for write */
    uint32_t seq = RBPF_ATOMIC_LOAD_UINT32(&traj->seq);
    RBPF_ATOMIC_STORE_UINT32(&traj->seq, seq + 1); /* Odd = write in progress */
    MEMORY_FENCE();

    int T = traj->config.T_max;
    memset(traj->regimes, 0, T * sizeof(int));
    memset(traj->h, 0, T * sizeof(float));

    RBPF_ATOMIC_STORE_INT(&traj->head, 0);
    RBPF_ATOMIC_STORE_INT(&traj->count, 0);
    RBPF_ATOMIC_STORE_INT64(&traj->total_ticks, 0);
    traj->oldest_tick = 0;

    seed_rng(traj->rng_state, traj->config.seed);

    /* Release seqlock */
    MEMORY_FENCE();
    RBPF_ATOMIC_STORE_UINT32(&traj->seq, seq + 2); /* Even = write complete */
}

void rbpf_trajectory_free(RBPFTrajectory *traj)
{
    if (!traj)
        return;

    RBPF_ALIGNED_FREE(traj->regimes);
    RBPF_ALIGNED_FREE(traj->h);

    memset(traj, 0, sizeof(*traj));
}

/*═══════════════════════════════════════════════════════════════════════════
 * RECORDING
 *═══════════════════════════════════════════════════════════════════════════*/

/*═══════════════════════════════════════════════════════════════════════════
 * RECORDING (RBPF thread - writer)
 *
 * Uses seqlock pattern:
 * 1. Increment seq to odd (write in progress)
 * 2. Write data
 * 3. Increment seq to even (write complete)
 *
 * Readers check seq before/after and retry if changed or odd.
 *═══════════════════════════════════════════════════════════════════════════*/

void rbpf_trajectory_record(RBPFTrajectory *traj, int regime, float h)
{
    if (!traj || !traj->initialized)
        return;

    int T_max = traj->config.T_max;

    /* Begin write - set seq to odd */
    uint32_t seq = RBPF_ATOMIC_LOAD_UINT32(&traj->seq);
    RBPF_ATOMIC_STORE_UINT32(&traj->seq, seq + 1);

    /* Get current position */
    int head = RBPF_ATOMIC_LOAD_INT(&traj->head);
    int count = RBPF_ATOMIC_LOAD_INT(&traj->count);

    /* Write data */
    traj->regimes[head] = regime;
    traj->h[head] = h;

    /* Update position (circular) */
    int new_head = (head + 1) % T_max;
    int new_count = (count < T_max) ? count + 1 : T_max;

    /* Update atomic state */
    RBPF_ATOMIC_STORE_INT(&traj->head, new_head);
    RBPF_ATOMIC_STORE_INT(&traj->count, new_count);
    RBPF_ATOMIC_FETCH_ADD_INT64(&traj->total_ticks, 1);

    if (count >= T_max)
    {
        traj->oldest_tick++;
    }

    /* End write - set seq to even */
    RBPF_ATOMIC_STORE_UINT32(&traj->seq, seq + 2);
}

/*═══════════════════════════════════════════════════════════════════════════
 * EXTRACTION (Oracle thread - reader)
 *
 * For direct extraction (without snapshot), we use seqlock checking.
 * For maximum safety in HFT, prefer the snapshot approach.
 *═══════════════════════════════════════════════════════════════════════════*/

RBPFTrajectoryExtractResult rbpf_trajectory_extract(
    const RBPFTrajectory *traj,
    int *regimes_out,
    float *h_out,
    int T_requested)
{

    RBPFTrajectoryExtractResult result;
    memset(&result, 0, sizeof(result));

    if (!traj || !traj->initialized)
    {
        return result;
    }

    int T_max = traj->config.T_max;

    /* Read with seqlock - retry if write in progress */
    uint32_t seq1, seq2;
    int head, count;
    int64_t total_ticks;

    int retries = 0;
    const int max_retries = 10;

    do
    {
        seq1 = RBPF_ATOMIC_LOAD_UINT32(&traj->seq);

        /* Wait if write in progress (seq is odd) */
        while (seq1 & 1)
        {
            if (++retries > max_retries * 100)
            {
                return result; /* Give up */
            }
            COMPILER_BARRIER();
            seq1 = RBPF_ATOMIC_LOAD_UINT32(&traj->seq);
        }

        /* Read state */
        head = RBPF_ATOMIC_LOAD_INT(&traj->head);
        count = RBPF_ATOMIC_LOAD_INT(&traj->count);
        total_ticks = RBPF_ATOMIC_LOAD_INT64(&traj->total_ticks);

        /* Verify no concurrent write */
        seq2 = RBPF_ATOMIC_LOAD_UINT32(&traj->seq);

    } while (seq1 != seq2 && ++retries < max_retries);

    if (seq1 != seq2)
    {
        return result; /* Failed to get consistent read */
    }

    int available = count;
    int T = (T_requested < available) ? T_requested : available;

    result.T = T;
    result.fill_ratio = (float)available / (float)T_max;
    result.end_tick = total_ticks - 1;
    result.start_tick = total_ticks - T;

    if (T == 0)
    {
        return result;
    }

    /* Extract from consistent snapshot of state */
    int start_offset = count - T;
    int start_idx = (head - count + start_offset + T_max) % T_max;

    for (int i = 0; i < T; i++)
    {
        int idx = (start_idx + i) % T_max;
        if (regimes_out)
            regimes_out[i] = traj->regimes[idx];
        if (h_out)
            h_out[i] = traj->h[idx];
    }

    return result;
}

RBPFTrajectoryExtractResult rbpf_trajectory_extract_double(
    const RBPFTrajectory *traj,
    int *regimes_out,
    double *h_out,
    int T_requested)
{

    RBPFTrajectoryExtractResult result;
    memset(&result, 0, sizeof(result));

    if (!traj || !traj->initialized)
    {
        return result;
    }

    int T_max = traj->config.T_max;

    /* Read with seqlock */
    uint32_t seq1, seq2;
    int head, count;
    int64_t total_ticks;

    int retries = 0;
    const int max_retries = 10;

    do
    {
        seq1 = RBPF_ATOMIC_LOAD_UINT32(&traj->seq);
        while (seq1 & 1)
        {
            if (++retries > max_retries * 100)
                return result;
            COMPILER_BARRIER();
            seq1 = RBPF_ATOMIC_LOAD_UINT32(&traj->seq);
        }

        head = RBPF_ATOMIC_LOAD_INT(&traj->head);
        count = RBPF_ATOMIC_LOAD_INT(&traj->count);
        total_ticks = RBPF_ATOMIC_LOAD_INT64(&traj->total_ticks);

        seq2 = RBPF_ATOMIC_LOAD_UINT32(&traj->seq);
    } while (seq1 != seq2 && ++retries < max_retries);

    if (seq1 != seq2)
        return result;

    int available = count;
    int T = (T_requested < available) ? T_requested : available;

    result.T = T;
    result.fill_ratio = (float)available / (float)T_max;
    result.end_tick = total_ticks - 1;
    result.start_tick = total_ticks - T;

    if (T == 0)
        return result;

    int start_offset = count - T;
    int start_idx = (head - count + start_offset + T_max) % T_max;

    for (int i = 0; i < T; i++)
    {
        int idx = (start_idx + i) % T_max;
        if (regimes_out)
            regimes_out[i] = traj->regimes[idx];
        if (h_out)
            h_out[i] = (double)traj->h[idx];
    }

    return result;
}

int rbpf_trajectory_length(const RBPFTrajectory *traj)
{
    if (!traj || !traj->initialized)
        return 0;
    return RBPF_ATOMIC_LOAD_INT(&traj->count);
}

bool rbpf_trajectory_ready(const RBPFTrajectory *traj, int min_length)
{
    if (!traj || !traj->initialized)
        return false;
    return RBPF_ATOMIC_LOAD_INT(&traj->count) >= min_length;
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEMPERING
 *
 * Per Chopin & Papaspiliopoulos (2020), we inject small perturbations
 * into the reference path to break confirmation bias.
 *═══════════════════════════════════════════════════════════════════════════*/

int rbpf_trajectory_temper(
    RBPFTrajectory *traj,
    int *regimes,
    int T,
    float flip_prob)
{

    if (!traj || !regimes || T <= 0 || flip_prob <= 0.0f)
    {
        return 0;
    }

    int n_regimes = traj->config.n_regimes;
    int flips = 0;

    for (int t = 0; t < T; t++)
    {
        if (rand_uniform(traj->rng_state) < flip_prob)
        {
            int old_regime = regimes[t];

            /* Pick a different regime uniformly */
            int offset = 1 + rand_int(traj->rng_state, n_regimes - 1);
            int new_regime = (old_regime + offset) % n_regimes;

            regimes[t] = new_regime;
            flips++;
        }
    }

    return flips;
}

int rbpf_trajectory_temper_default(RBPFTrajectory *traj, int *regimes, int T)
{
    if (!traj)
        return 0;
    return rbpf_trajectory_temper(traj, regimes, T, traj->config.temper_prob);
}

/*═══════════════════════════════════════════════════════════════════════════
 * QUERIES
 *═══════════════════════════════════════════════════════════════════════════*/

int rbpf_trajectory_last_regime(const RBPFTrajectory *traj)
{
    if (!traj || !traj->initialized)
        return -1;

    int count = RBPF_ATOMIC_LOAD_INT(&traj->count);
    if (count == 0)
        return -1;

    int head = RBPF_ATOMIC_LOAD_INT(&traj->head);
    int T_max = traj->config.T_max;
    int last_idx = (head - 1 + T_max) % T_max;
    return traj->regimes[last_idx];
}

float rbpf_trajectory_last_h(const RBPFTrajectory *traj)
{
    if (!traj || !traj->initialized)
        return 0.0f;

    int count = RBPF_ATOMIC_LOAD_INT(&traj->count);
    if (count == 0)
        return 0.0f;

    int head = RBPF_ATOMIC_LOAD_INT(&traj->head);
    int T_max = traj->config.T_max;
    int last_idx = (head - 1 + T_max) % T_max;
    return traj->h[last_idx];
}

int64_t rbpf_trajectory_total_ticks(const RBPFTrajectory *traj)
{
    if (!traj || !traj->initialized)
        return 0;
    return RBPF_ATOMIC_LOAD_INT64(&traj->total_ticks);
}

void rbpf_trajectory_regime_distribution(
    const RBPFTrajectory *traj,
    float *probs_out)
{

    if (!traj || !traj->initialized || !probs_out)
        return;

    int n_regimes = traj->config.n_regimes;
    int T_max = traj->config.T_max;

    /* Zero output */
    for (int r = 0; r < n_regimes; r++)
    {
        probs_out[r] = 0.0f;
    }

    int count = RBPF_ATOMIC_LOAD_INT(&traj->count);
    if (count == 0)
        return;

    int head = RBPF_ATOMIC_LOAD_INT(&traj->head);

    /* Count occurrences */
    int counts[RBPF_TRAJ_MAX_REGIMES] = {0};

    int start_idx = (head - count + T_max) % T_max;
    for (int i = 0; i < count; i++)
    {
        int idx = (start_idx + i) % T_max;
        int r = traj->regimes[idx];
        if (r >= 0 && r < n_regimes)
        {
            counts[r]++;
        }
    }

    /* Normalize */
    float inv_count = 1.0f / (float)count;
    for (int r = 0; r < n_regimes; r++)
    {
        probs_out[r] = (float)counts[r] * inv_count;
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * THREAD-SAFE SNAPSHOT (Recommended for HFT)
 *
 * The snapshot approach provides guaranteed zero-contention extraction:
 * 1. Oracle allocates snapshot once at startup
 * 2. When triggered, takes atomic snapshot (waits for any write to complete)
 * 3. Extracts from snapshot with no thread-safety concerns
 *═══════════════════════════════════════════════════════════════════════════*/

int rbpf_trajectory_snapshot_alloc(RBPFTrajectorySnapshot *snap, int T_max, int n_regimes)
{
    if (!snap || T_max <= 0)
        return -1;

    memset(snap, 0, sizeof(*snap));

    snap->T_max = T_max;
    snap->n_regimes = n_regimes;

    snap->regimes = (int *)calloc(T_max, sizeof(int));
    snap->h = (float *)calloc(T_max, sizeof(float));

    if (!snap->regimes || !snap->h)
    {
        rbpf_trajectory_snapshot_free(snap);
        return -1;
    }

    snap->valid = false;
    return 0;
}

void rbpf_trajectory_snapshot_free(RBPFTrajectorySnapshot *snap)
{
    if (!snap)
        return;

    free(snap->regimes);
    free(snap->h);
    memset(snap, 0, sizeof(*snap));
}

int rbpf_trajectory_snapshot(const RBPFTrajectory *traj,
                             RBPFTrajectorySnapshot *snap,
                             int max_retries)
{
    if (!traj || !traj->initialized || !snap || !snap->regimes || !snap->h)
    {
        return -1;
    }

    if (max_retries <= 0)
        max_retries = 3;

    int T_max = traj->config.T_max;
    if (snap->T_max != T_max)
    {
        return -1; /* Snapshot buffer size mismatch */
    }

    uint32_t seq1, seq2;
    int retries = 0;

    do
    {
        /* Wait for any write to complete */
        seq1 = RBPF_ATOMIC_LOAD_UINT32(&traj->seq);
        while (seq1 & 1)
        {
            if (++retries > max_retries * 100)
            {
                snap->valid = false;
                return -1;
            }
            COMPILER_BARRIER();
            seq1 = RBPF_ATOMIC_LOAD_UINT32(&traj->seq);
        }

        /* Copy state */
        snap->head = RBPF_ATOMIC_LOAD_INT(&traj->head);
        snap->count = RBPF_ATOMIC_LOAD_INT(&traj->count);
        snap->total_ticks = RBPF_ATOMIC_LOAD_INT64(&traj->total_ticks);

        /* Copy data buffers */
        memcpy(snap->regimes, traj->regimes, T_max * sizeof(int));
        memcpy(snap->h, traj->h, T_max * sizeof(float));

        /* Verify no write occurred during copy */
        MEMORY_FENCE();
        seq2 = RBPF_ATOMIC_LOAD_UINT32(&traj->seq);

    } while (seq1 != seq2 && ++retries < max_retries);

    if (seq1 != seq2)
    {
        snap->valid = false;
        return -1;
    }

    snap->valid = true;
    return 0;
}

RBPFTrajectoryExtractResult rbpf_trajectory_extract_from_snapshot(
    const RBPFTrajectorySnapshot *snap,
    int *regimes_out,
    float *h_out,
    int T_requested)
{

    RBPFTrajectoryExtractResult result;
    memset(&result, 0, sizeof(result));

    if (!snap || !snap->valid)
    {
        return result;
    }

    int T_max = snap->T_max;
    int available = snap->count;
    int T = (T_requested < available) ? T_requested : available;

    result.T = T;
    result.fill_ratio = (float)available / (float)T_max;
    result.end_tick = snap->total_ticks - 1;
    result.start_tick = snap->total_ticks - T;

    if (T == 0)
    {
        return result;
    }

    int start_offset = snap->count - T;
    int start_idx = (snap->head - snap->count + start_offset + T_max) % T_max;

    for (int i = 0; i < T; i++)
    {
        int idx = (start_idx + i) % T_max;
        if (regimes_out)
            regimes_out[i] = snap->regimes[idx];
        if (h_out)
            h_out[i] = snap->h[idx];
    }

    return result;
}

RBPFTrajectoryExtractResult rbpf_trajectory_extract_from_snapshot_double(
    const RBPFTrajectorySnapshot *snap,
    int *regimes_out,
    double *h_out,
    int T_requested)
{

    RBPFTrajectoryExtractResult result;
    memset(&result, 0, sizeof(result));

    if (!snap || !snap->valid)
    {
        return result;
    }

    int T_max = snap->T_max;
    int available = snap->count;
    int T = (T_requested < available) ? T_requested : available;

    result.T = T;
    result.fill_ratio = (float)available / (float)T_max;
    result.end_tick = snap->total_ticks - 1;
    result.start_tick = snap->total_ticks - T;

    if (T == 0)
    {
        return result;
    }

    int start_offset = snap->count - T;
    int start_idx = (snap->head - snap->count + start_offset + T_max) % T_max;

    for (int i = 0; i < T; i++)
    {
        int idx = (start_idx + i) % T_max;
        if (regimes_out)
            regimes_out[i] = snap->regimes[idx];
        if (h_out)
            h_out[i] = (double)snap->h[idx];
    }

    return result;
}

/*═══════════════════════════════════════════════════════════════════════════
 * DIAGNOSTICS
 *═══════════════════════════════════════════════════════════════════════════*/

void rbpf_trajectory_print_state(const RBPFTrajectory *traj)
{
    if (!traj)
    {
        printf("RBPFTrajectory: NULL\n");
        return;
    }

    int count = RBPF_ATOMIC_LOAD_INT(&traj->count);
    int head = RBPF_ATOMIC_LOAD_INT(&traj->head);
    int64_t total = RBPF_ATOMIC_LOAD_INT64(&traj->total_ticks);

    printf("═══════════════════════════════════════════════════════════\n");
    printf("RBPF TRAJECTORY BUFFER STATE\n");
    printf("═══════════════════════════════════════════════════════════\n");
    printf("Initialized:    %s\n", traj->initialized ? "true" : "false");
    printf("T_max:          %d\n", traj->config.T_max);
    printf("n_regimes:      %d\n", traj->config.n_regimes);
    printf("temper_prob:    %.2f%%\n", traj->config.temper_prob * 100.0f);
    printf("Buffer count:   %d / %d (%.1f%%)\n",
           count, traj->config.T_max,
           100.0f * (float)count / (float)traj->config.T_max);
    printf("Total ticks:    %lld\n", (long long)total);
    printf("Head position:  %d\n", head);
    printf("Seqlock:        %u\n", RBPF_ATOMIC_LOAD_UINT32(&traj->seq));

    if (count > 0)
    {
        printf("Last regime:    %d\n", rbpf_trajectory_last_regime(traj));
        printf("Last h:         %.4f\n", rbpf_trajectory_last_h(traj));

        float probs[RBPF_TRAJ_MAX_REGIMES];
        rbpf_trajectory_regime_distribution(traj, probs);
        printf("Regime dist:    [");
        for (int r = 0; r < traj->config.n_regimes; r++)
        {
            printf("%.1f%%", probs[r] * 100.0f);
            if (r < traj->config.n_regimes - 1)
                printf(", ");
        }
        printf("]\n");
    }
    printf("═══════════════════════════════════════════════════════════\n");
}

void rbpf_trajectory_print_tail(const RBPFTrajectory *traj, int n)
{
    if (!traj || !traj->initialized)
    {
        printf("RBPFTrajectory: empty or not initialized\n");
        return;
    }

    int count = RBPF_ATOMIC_LOAD_INT(&traj->count);
    int head = RBPF_ATOMIC_LOAD_INT(&traj->head);
    int64_t total = RBPF_ATOMIC_LOAD_INT64(&traj->total_ticks);

    if (count == 0)
    {
        printf("RBPFTrajectory: empty\n");
        return;
    }

    int T_max = traj->config.T_max;
    int to_print = (n < count) ? n : count;

    printf("Last %d entries (newest last):\n", to_print);
    printf("  %-6s  %-8s  %-10s\n", "t", "regime", "h");
    printf("  %-6s  %-8s  %-10s\n", "---", "------", "-------");

    /* Start from (most recent - to_print + 1) */
    int start_idx = (head - to_print + T_max) % T_max;

    for (int i = 0; i < to_print; i++)
    {
        int idx = (start_idx + i) % T_max;
        int64_t tick = total - to_print + i;
        printf("  %-6lld  %-8d  %-10.4f\n",
               (long long)tick, traj->regimes[idx], traj->h[idx]);
    }
}