/**
 * @file pi_staging.h
 * @brief Lock-Free Π Staging Buffer
 *
 * Double-buffered handoff of Π from Oracle thread to RBPF thread.
 * Oracle produces Π estimates continuously.
 * RBPF consumes when ready and conditions are met.
 */

#ifndef PI_STAGING_H
#define PI_STAGING_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*═══════════════════════════════════════════════════════════════════════════
 * CONFIGURATION
 *═══════════════════════════════════════════════════════════════════════════*/

#ifndef PI_STAGING_MAX_K
#define PI_STAGING_MAX_K 8
#endif

/*═══════════════════════════════════════════════════════════════════════════
 * PLATFORM-SPECIFIC
 *═══════════════════════════════════════════════════════════════════════════*/

#if defined(_MSC_VER)
    #define PI_ALIGNED64 __declspec(align(64))
    #include <intrin.h>
    #define PI_MEMORY_FENCE() _mm_mfence()
#elif defined(__GNUC__) || defined(__clang__)
    #define PI_ALIGNED64 __attribute__((aligned(64)))
    #define PI_MEMORY_FENCE() __sync_synchronize()
#else
    #define PI_ALIGNED64
    #define PI_MEMORY_FENCE()
#endif

/*═══════════════════════════════════════════════════════════════════════════
 * ORACLE CONFIDENCE (Simplified 3 Gates)
 *═══════════════════════════════════════════════════════════════════════════*/

typedef struct {
    /* Gate A: Did we explore? (Mixing) */
    float mixing_score;            /* 0-1, from acceptance rate */
    float acceptance_rate;         /* Raw acceptance rate */
    
    /* Gate B: Did we learn? (Information) */
    float information_score;       /* 0-1, from count magnitude */
    float min_row_count;           /* Minimum counts in any row */
    
    /* Gate C: Is it new? (Innovation) */
    float innovation_score;        /* 0-1, from ||ΔΠ|| */
    float frobenius_diff;          /* Raw difference */
    
    /* Combined */
    float overall;                 /* Geometric mean of gates */
    bool  is_valid;                /* Hard fail if mixing < 1% */
    
    /* Diagnostics */
    int   sweeps_used;
    bool  converged_early;
    
} OracleConfidence;

/*═══════════════════════════════════════════════════════════════════════════
 * STAGED Π ENTRY
 *═══════════════════════════════════════════════════════════════════════════*/

typedef struct {
    /* Matrix data */
    float Pi[PI_STAGING_MAX_K * PI_STAGING_MAX_K];
    float Q[PI_STAGING_MAX_K * PI_STAGING_MAX_K];  /* Sufficient stats for Thompson */
    int   K;
    
    /* Confidence metrics */
    OracleConfidence confidence;
    
    /* Timestamp */
    int64_t timestamp;             /* Tick when generated */
    int64_t window_start;          /* Start of observation window */
    int64_t window_end;            /* End of observation window */
    
    /* Valid flag */
    bool valid;
    
} StagedPi;

/*═══════════════════════════════════════════════════════════════════════════
 * STAGING BUFFER (Double-Buffered)
 *═══════════════════════════════════════════════════════════════════════════*/

typedef struct {
    /* Double buffer */
    StagedPi buffers[2];
    
    /* Indices (cache-line isolated for no false sharing) */
    PI_ALIGNED64 volatile int write_idx;      /* 0 or 1 */
    PI_ALIGNED64 volatile int read_ready;     /* New Π available? */
    
    /* Generation counter (for sequencing) */
    PI_ALIGNED64 volatile int64_t generation;
    
} PiStagingBuffer;

/*═══════════════════════════════════════════════════════════════════════════
 * API - INITIALIZATION
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Initialize staging buffer
 */
void pi_staging_init(PiStagingBuffer *buf);

/*═══════════════════════════════════════════════════════════════════════════
 * API - ORACLE THREAD (Producer)
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Submit new Π from Oracle
 * 
 * @param buf         Staging buffer
 * @param Pi          New transition matrix (K×K, row-stochastic)
 * @param Q           Sufficient statistics (for Thompson sampling)
 * @param K           Number of regimes
 * @param confidence  Confidence metrics
 * @param tick        Current tick
 * @param window_start Start of observation window
 * @param window_end   End of observation window
 */
void pi_staging_submit(
    PiStagingBuffer *buf,
    const float *Pi,
    const float *Q,
    int K,
    const OracleConfidence *confidence,
    int64_t tick,
    int64_t window_start,
    int64_t window_end);

/*═══════════════════════════════════════════════════════════════════════════
 * API - RBPF THREAD (Consumer)
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Check if new Π is available
 */
bool pi_staging_available(const PiStagingBuffer *buf);

/**
 * Get current generation (for change detection)
 */
int64_t pi_staging_get_generation(const PiStagingBuffer *buf);

/**
 * Peek at staged Π without consuming
 * Returns pointer to internal buffer (valid until next submit)
 */
const StagedPi *pi_staging_peek(const PiStagingBuffer *buf);

/**
 * Consume staged Π
 * Copies data and clears ready flag
 * 
 * @param buf         Staging buffer
 * @param Pi_out      Output: Π matrix
 * @param Q_out       Output: Q matrix (can be NULL)
 * @param K_out       Output: K value
 * @param conf_out    Output: Confidence (can be NULL)
 * @return            true if consumed, false if nothing available
 */
bool pi_staging_consume(
    PiStagingBuffer *buf,
    float *Pi_out,
    float *Q_out,
    int *K_out,
    OracleConfidence *conf_out);

/**
 * Consume with full metadata
 */
bool pi_staging_consume_full(
    PiStagingBuffer *buf,
    StagedPi *out);

/*═══════════════════════════════════════════════════════════════════════════
 * API - CONFIDENCE UTILITIES
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Compute confidence from PGAS results
 * 
 * @param acceptance_rate  PGAS acceptance rate
 * @param min_row_count    Minimum row sum in Q
 * @param frobenius_diff   ||Π_new - Π_old||
 * @param sweeps_used      Number of sweeps run
 * @return                 OracleConfidence struct
 */
OracleConfidence pi_staging_compute_confidence(
    float acceptance_rate,
    float min_row_count,
    float frobenius_diff,
    int sweeps_used);

#ifdef __cplusplus
}
#endif

#endif /* PI_STAGING_H */
