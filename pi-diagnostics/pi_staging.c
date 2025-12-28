/**
 * @file pi_staging.c
 * @brief Lock-Free Π Staging Buffer Implementation
 */

#include "pi_staging.h"
#include <string.h>
#include <math.h>

/*═══════════════════════════════════════════════════════════════════════════
 * CONSTANTS
 *═══════════════════════════════════════════════════════════════════════════*/

#define MIN_ACCEPTANCE_VALID  0.01f   /* Below this = chain stuck */
#define GOOD_ACCEPTANCE       0.20f   /* Above this = excellent mixing */
#define MIN_COUNTS_CONFIDENT  50.0f   /* Minimum counts for confidence */
#define MIN_INNOVATION        0.01f   /* Below this = no change */
#define MAX_INNOVATION        0.50f   /* Above this = suspicious */

/*═══════════════════════════════════════════════════════════════════════════
 * INITIALIZATION
 *═══════════════════════════════════════════════════════════════════════════*/

void pi_staging_init(PiStagingBuffer *buf) {
    if (!buf) return;
    
    memset(buf, 0, sizeof(*buf));
    buf->write_idx = 0;
    buf->read_ready = 0;
    buf->generation = 0;
}

/*═══════════════════════════════════════════════════════════════════════════
 * ORACLE THREAD (Producer)
 *═══════════════════════════════════════════════════════════════════════════*/

void pi_staging_submit(
    PiStagingBuffer *buf,
    const float *Pi,
    const float *Q,
    int K,
    const OracleConfidence *confidence,
    int64_t tick,
    int64_t window_start,
    int64_t window_end)
{
    if (!buf || !Pi || K <= 0 || K > PI_STAGING_MAX_K) return;
    
    /* Write to inactive buffer */
    int write = 1 - buf->write_idx;
    StagedPi *staged = &buf->buffers[write];
    
    /* Copy matrix */
    int size = K * K;
    memcpy(staged->Pi, Pi, size * sizeof(float));
    
    /* Copy Q if provided */
    if (Q) {
        memcpy(staged->Q, Q, size * sizeof(float));
    } else {
        memset(staged->Q, 0, size * sizeof(float));
    }
    
    staged->K = K;
    
    /* Copy confidence */
    if (confidence) {
        staged->confidence = *confidence;
    } else {
        memset(&staged->confidence, 0, sizeof(staged->confidence));
    }
    
    /* Timestamps */
    staged->timestamp = tick;
    staged->window_start = window_start;
    staged->window_end = window_end;
    staged->valid = true;
    
    /* Memory fence before making visible */
    PI_MEMORY_FENCE();
    
    /* Swap buffers */
    buf->write_idx = write;
    buf->generation++;
    buf->read_ready = 1;
}

/*═══════════════════════════════════════════════════════════════════════════
 * RBPF THREAD (Consumer)
 *═══════════════════════════════════════════════════════════════════════════*/

bool pi_staging_available(const PiStagingBuffer *buf) {
    if (!buf) return false;
    return buf->read_ready != 0;
}

int64_t pi_staging_get_generation(const PiStagingBuffer *buf) {
    if (!buf) return 0;
    return buf->generation;
}

const StagedPi *pi_staging_peek(const PiStagingBuffer *buf) {
    if (!buf || !buf->read_ready) return NULL;
    return &buf->buffers[buf->write_idx];
}

bool pi_staging_consume(
    PiStagingBuffer *buf,
    float *Pi_out,
    float *Q_out,
    int *K_out,
    OracleConfidence *conf_out)
{
    if (!buf || !buf->read_ready) return false;
    
    /* Read from active buffer */
    int read = buf->write_idx;
    const StagedPi *staged = &buf->buffers[read];
    
    if (!staged->valid) return false;
    
    int K = staged->K;
    int size = K * K;
    
    /* Copy data */
    if (Pi_out) {
        memcpy(Pi_out, staged->Pi, size * sizeof(float));
    }
    if (Q_out) {
        memcpy(Q_out, staged->Q, size * sizeof(float));
    }
    if (K_out) {
        *K_out = K;
    }
    if (conf_out) {
        *conf_out = staged->confidence;
    }
    
    /* Clear ready flag */
    buf->read_ready = 0;
    
    return true;
}

bool pi_staging_consume_full(PiStagingBuffer *buf, StagedPi *out) {
    if (!buf || !out || !buf->read_ready) return false;
    
    int read = buf->write_idx;
    const StagedPi *staged = &buf->buffers[read];
    
    if (!staged->valid) return false;
    
    /* Copy entire struct */
    *out = *staged;
    
    /* Clear ready flag */
    buf->read_ready = 0;
    
    return true;
}

/*═══════════════════════════════════════════════════════════════════════════
 * CONFIDENCE COMPUTATION
 *═══════════════════════════════════════════════════════════════════════════*/

OracleConfidence pi_staging_compute_confidence(
    float acceptance_rate,
    float min_row_count,
    float frobenius_diff,
    int sweeps_used)
{
    OracleConfidence conf;
    memset(&conf, 0, sizeof(conf));
    
    conf.acceptance_rate = acceptance_rate;
    conf.min_row_count = min_row_count;
    conf.frobenius_diff = frobenius_diff;
    conf.sweeps_used = sweeps_used;
    
    /*═══════════════════════════════════════════════════════════════════
     * GATE A: MIXING (Did we explore?)
     *
     * Acceptance < 1%  → Chain stuck, INVALID
     * Acceptance < 5%  → Poor mixing, low confidence
     * Acceptance < 20% → Moderate mixing
     * Acceptance > 20% → Good mixing
     *═══════════════════════════════════════════════════════════════════*/
    
    if (acceptance_rate < MIN_ACCEPTANCE_VALID) {
        /* Hard fail - chain stuck */
        conf.mixing_score = 0.0f;
        conf.is_valid = false;
        conf.overall = 0.0f;
        return conf;
    }
    
    if (acceptance_rate >= GOOD_ACCEPTANCE) {
        conf.mixing_score = 1.0f;
    } else {
        /* Linear interpolation from 1% to 20% */
        conf.mixing_score = (acceptance_rate - MIN_ACCEPTANCE_VALID) / 
                            (GOOD_ACCEPTANCE - MIN_ACCEPTANCE_VALID);
    }
    
    /*═══════════════════════════════════════════════════════════════════
     * GATE B: INFORMATION (Did we learn?)
     *
     * min_row_count < 10  → Very little data
     * min_row_count < 50  → Some data, low confidence
     * min_row_count > 50  → Enough data for confidence
     *═══════════════════════════════════════════════════════════════════*/
    
    if (min_row_count >= MIN_COUNTS_CONFIDENT) {
        conf.information_score = 1.0f;
    } else if (min_row_count > 0) {
        conf.information_score = min_row_count / MIN_COUNTS_CONFIDENT;
    } else {
        conf.information_score = 0.0f;
    }
    
    /*═══════════════════════════════════════════════════════════════════
     * GATE C: INNOVATION (Is it new?)
     *
     * ||ΔΠ|| < 0.01 → No change, don't bother
     * ||ΔΠ|| > 0.50 → Suspicious, penalize unless mixing excellent
     * Otherwise     → Normal innovation
     *═══════════════════════════════════════════════════════════════════*/
    
    if (frobenius_diff < MIN_INNOVATION) {
        /* No meaningful change */
        conf.innovation_score = 0.0f;
    } else if (frobenius_diff > MAX_INNOVATION) {
        /* Suspicious - large change. Trust if mixing is good. */
        conf.innovation_score = 0.5f * conf.mixing_score;
    } else {
        conf.innovation_score = 1.0f;
    }
    
    /*═══════════════════════════════════════════════════════════════════
     * OVERALL CONFIDENCE
     * 
     * Geometric mean emphasizes that ALL gates matter.
     * If any gate is zero, overall is zero.
     *═══════════════════════════════════════════════════════════════════*/
    
    float product = conf.mixing_score * 
                    conf.information_score * 
                    conf.innovation_score;
    
    if (product > 0) {
        conf.overall = powf(product, 1.0f / 3.0f);
    } else {
        conf.overall = 0.0f;
    }
    
    conf.is_valid = (conf.overall > 0.1f);
    
    return conf;
}
