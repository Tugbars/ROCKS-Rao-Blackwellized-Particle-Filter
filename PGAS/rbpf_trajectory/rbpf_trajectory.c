/**
 * @file rbpf_trajectory.c
 * @brief Trajectory Buffer Implementation
 */

#include "rbpf_trajectory.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/*═══════════════════════════════════════════════════════════════════════════
 * RNG (xoroshiro128+)
 *═══════════════════════════════════════════════════════════════════════════*/

static inline uint64_t rotl(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

static uint64_t xoroshiro128plus(uint64_t *s) {
    const uint64_t s0 = s[0];
    uint64_t s1 = s[1];
    const uint64_t result = s0 + s1;
    
    s1 ^= s0;
    s[0] = rotl(s0, 24) ^ s1 ^ (s1 << 16);
    s[1] = rotl(s1, 37);
    
    return result;
}

static inline float rand_uniform(uint64_t *s) {
    return (xoroshiro128plus(s) >> 11) * (1.0f / 9007199254740992.0f);
}

static inline int rand_int(uint64_t *s, int max) {
    return (int)(rand_uniform(s) * max);
}

static void seed_rng(uint64_t *s, uint64_t seed) {
    uint64_t z = seed;
    
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    s[0] = z ^ (z >> 31);
    
    z = seed + 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    s[1] = z ^ (z >> 31);
    
    if (s[0] == 0 && s[1] == 0) {
        s[0] = 1;
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * CONFIGURATION
 *═══════════════════════════════════════════════════════════════════════════*/

RBPFTrajectoryConfig rbpf_trajectory_config_defaults(int T_max, int n_regimes) {
    RBPFTrajectoryConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    
    cfg.T_max = (T_max > 0 && T_max <= RBPF_TRAJ_MAX_T) ? T_max : 512;
    cfg.n_regimes = (n_regimes > 0 && n_regimes <= RBPF_TRAJ_MAX_REGIMES) ? n_regimes : 4;
    cfg.temper_prob = 0.05f;  /* 5% default per Chopin & Papaspiliopoulos */
    cfg.seed = 0xDEADBEEF;
    
    return cfg;
}

/*═══════════════════════════════════════════════════════════════════════════
 * LIFECYCLE
 *═══════════════════════════════════════════════════════════════════════════*/

int rbpf_trajectory_init(RBPFTrajectory *traj, const RBPFTrajectoryConfig *config) {
    if (!traj) return -1;
    
    memset(traj, 0, sizeof(*traj));
    
    traj->config = config ? *config : rbpf_trajectory_config_defaults(512, 4);
    
    int T = traj->config.T_max;
    
    /* Allocate buffers */
    traj->regimes = (int *)calloc(T, sizeof(int));
    traj->h = (float *)calloc(T, sizeof(float));
    
    if (!traj->regimes || !traj->h) {
        rbpf_trajectory_free(traj);
        return -1;
    }
    
    /* Initialize RNG */
    seed_rng(traj->rng_state, traj->config.seed);
    
    traj->head = 0;
    traj->count = 0;
    traj->total_ticks = 0;
    traj->oldest_tick = 0;
    traj->initialized = true;
    
    return 0;
}

int rbpf_trajectory_init_simple(RBPFTrajectory *traj, int T_max, int n_regimes) {
    RBPFTrajectoryConfig cfg = rbpf_trajectory_config_defaults(T_max, n_regimes);
    return rbpf_trajectory_init(traj, &cfg);
}

void rbpf_trajectory_reset(RBPFTrajectory *traj) {
    if (!traj || !traj->initialized) return;
    
    int T = traj->config.T_max;
    memset(traj->regimes, 0, T * sizeof(int));
    memset(traj->h, 0, T * sizeof(float));
    
    traj->head = 0;
    traj->count = 0;
    traj->total_ticks = 0;
    traj->oldest_tick = 0;
    
    seed_rng(traj->rng_state, traj->config.seed);
}

void rbpf_trajectory_free(RBPFTrajectory *traj) {
    if (!traj) return;
    
    free(traj->regimes);
    free(traj->h);
    
    memset(traj, 0, sizeof(*traj));
}

/*═══════════════════════════════════════════════════════════════════════════
 * RECORDING
 *═══════════════════════════════════════════════════════════════════════════*/

void rbpf_trajectory_record(RBPFTrajectory *traj, int regime, float h) {
    if (!traj || !traj->initialized) return;
    
    int T_max = traj->config.T_max;
    
    /* Write to current head position */
    traj->regimes[traj->head] = regime;
    traj->h[traj->head] = h;
    
    /* Advance head (circular) */
    traj->head = (traj->head + 1) % T_max;
    
    /* Update count */
    if (traj->count < T_max) {
        traj->count++;
    } else {
        /* Buffer is full, oldest entry is being overwritten */
        traj->oldest_tick++;
    }
    
    traj->total_ticks++;
}

/*═══════════════════════════════════════════════════════════════════════════
 * EXTRACTION
 *═══════════════════════════════════════════════════════════════════════════*/

RBPFTrajectoryExtractResult rbpf_trajectory_extract(
    const RBPFTrajectory *traj,
    int *regimes_out,
    float *h_out,
    int T_requested) {
    
    RBPFTrajectoryExtractResult result;
    memset(&result, 0, sizeof(result));
    
    if (!traj || !traj->initialized) {
        return result;
    }
    
    int T_max = traj->config.T_max;
    int available = traj->count;
    int T = (T_requested < available) ? T_requested : available;
    
    result.T = T;
    result.fill_ratio = (float)available / (float)T_max;
    result.end_tick = traj->total_ticks - 1;
    result.start_tick = traj->total_ticks - T;
    
    if (T == 0) {
        return result;
    }
    
    /* Find start position in circular buffer */
    /* head points to next write, so most recent is at (head - 1) */
    /* oldest in buffer is at (head - count) */
    /* We want the most recent T entries */
    
    int start_offset = traj->count - T;  /* How many entries to skip from oldest */
    int start_idx = (traj->head - traj->count + start_offset + T_max) % T_max;
    
    /* Extract in chronological order */
    for (int i = 0; i < T; i++) {
        int idx = (start_idx + i) % T_max;
        if (regimes_out) regimes_out[i] = traj->regimes[idx];
        if (h_out) h_out[i] = traj->h[idx];
    }
    
    return result;
}

RBPFTrajectoryExtractResult rbpf_trajectory_extract_double(
    const RBPFTrajectory *traj,
    int *regimes_out,
    double *h_out,
    int T_requested) {
    
    RBPFTrajectoryExtractResult result;
    memset(&result, 0, sizeof(result));
    
    if (!traj || !traj->initialized) {
        return result;
    }
    
    int T_max = traj->config.T_max;
    int available = traj->count;
    int T = (T_requested < available) ? T_requested : available;
    
    result.T = T;
    result.fill_ratio = (float)available / (float)T_max;
    result.end_tick = traj->total_ticks - 1;
    result.start_tick = traj->total_ticks - T;
    
    if (T == 0) {
        return result;
    }
    
    int start_offset = traj->count - T;
    int start_idx = (traj->head - traj->count + start_offset + T_max) % T_max;
    
    for (int i = 0; i < T; i++) {
        int idx = (start_idx + i) % T_max;
        if (regimes_out) regimes_out[i] = traj->regimes[idx];
        if (h_out) h_out[i] = (double)traj->h[idx];
    }
    
    return result;
}

int rbpf_trajectory_length(const RBPFTrajectory *traj) {
    return (traj && traj->initialized) ? traj->count : 0;
}

bool rbpf_trajectory_ready(const RBPFTrajectory *traj, int min_length) {
    return (traj && traj->initialized && traj->count >= min_length);
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
    float flip_prob) {
    
    if (!traj || !regimes || T <= 0 || flip_prob <= 0.0f) {
        return 0;
    }
    
    int n_regimes = traj->config.n_regimes;
    int flips = 0;
    
    for (int t = 0; t < T; t++) {
        if (rand_uniform(traj->rng_state) < flip_prob) {
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

int rbpf_trajectory_temper_default(RBPFTrajectory *traj, int *regimes, int T) {
    if (!traj) return 0;
    return rbpf_trajectory_temper(traj, regimes, T, traj->config.temper_prob);
}

/*═══════════════════════════════════════════════════════════════════════════
 * QUERIES
 *═══════════════════════════════════════════════════════════════════════════*/

int rbpf_trajectory_last_regime(const RBPFTrajectory *traj) {
    if (!traj || !traj->initialized || traj->count == 0) {
        return -1;
    }
    
    int T_max = traj->config.T_max;
    int last_idx = (traj->head - 1 + T_max) % T_max;
    return traj->regimes[last_idx];
}

float rbpf_trajectory_last_h(const RBPFTrajectory *traj) {
    if (!traj || !traj->initialized || traj->count == 0) {
        return 0.0f;
    }
    
    int T_max = traj->config.T_max;
    int last_idx = (traj->head - 1 + T_max) % T_max;
    return traj->h[last_idx];
}

int64_t rbpf_trajectory_total_ticks(const RBPFTrajectory *traj) {
    return (traj && traj->initialized) ? traj->total_ticks : 0;
}

void rbpf_trajectory_regime_distribution(
    const RBPFTrajectory *traj,
    float *probs_out) {
    
    if (!traj || !traj->initialized || !probs_out) return;
    
    int n_regimes = traj->config.n_regimes;
    int T_max = traj->config.T_max;
    
    /* Zero output */
    for (int r = 0; r < n_regimes; r++) {
        probs_out[r] = 0.0f;
    }
    
    if (traj->count == 0) return;
    
    /* Count occurrences */
    int counts[RBPF_TRAJ_MAX_REGIMES] = {0};
    
    int start_idx = (traj->head - traj->count + T_max) % T_max;
    for (int i = 0; i < traj->count; i++) {
        int idx = (start_idx + i) % T_max;
        int r = traj->regimes[idx];
        if (r >= 0 && r < n_regimes) {
            counts[r]++;
        }
    }
    
    /* Normalize */
    float inv_count = 1.0f / (float)traj->count;
    for (int r = 0; r < n_regimes; r++) {
        probs_out[r] = (float)counts[r] * inv_count;
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * DIAGNOSTICS
 *═══════════════════════════════════════════════════════════════════════════*/

void rbpf_trajectory_print_state(const RBPFTrajectory *traj) {
    if (!traj) {
        printf("RBPFTrajectory: NULL\n");
        return;
    }
    
    printf("═══════════════════════════════════════════════════════════\n");
    printf("RBPF TRAJECTORY BUFFER STATE\n");
    printf("═══════════════════════════════════════════════════════════\n");
    printf("Initialized:    %s\n", traj->initialized ? "true" : "false");
    printf("T_max:          %d\n", traj->config.T_max);
    printf("n_regimes:      %d\n", traj->config.n_regimes);
    printf("temper_prob:    %.2f%%\n", traj->config.temper_prob * 100.0f);
    printf("Buffer count:   %d / %d (%.1f%%)\n", 
           traj->count, traj->config.T_max,
           100.0f * (float)traj->count / (float)traj->config.T_max);
    printf("Total ticks:    %lld\n", (long long)traj->total_ticks);
    printf("Head position:  %d\n", traj->head);
    
    if (traj->count > 0) {
        printf("Last regime:    %d\n", rbpf_trajectory_last_regime(traj));
        printf("Last h:         %.4f\n", rbpf_trajectory_last_h(traj));
        
        float probs[RBPF_TRAJ_MAX_REGIMES];
        rbpf_trajectory_regime_distribution(traj, probs);
        printf("Regime dist:    [");
        for (int r = 0; r < traj->config.n_regimes; r++) {
            printf("%.1f%%", probs[r] * 100.0f);
            if (r < traj->config.n_regimes - 1) printf(", ");
        }
        printf("]\n");
    }
    printf("═══════════════════════════════════════════════════════════\n");
}

void rbpf_trajectory_print_tail(const RBPFTrajectory *traj, int n) {
    if (!traj || !traj->initialized || traj->count == 0) {
        printf("RBPFTrajectory: empty or not initialized\n");
        return;
    }
    
    int T_max = traj->config.T_max;
    int to_print = (n < traj->count) ? n : traj->count;
    
    printf("Last %d entries (newest last):\n", to_print);
    printf("  %-6s  %-8s  %-10s\n", "t", "regime", "h");
    printf("  %-6s  %-8s  %-10s\n", "---", "------", "-------");
    
    /* Start from (most recent - to_print + 1) */
    int start_idx = (traj->head - to_print + T_max) % T_max;
    
    for (int i = 0; i < to_print; i++) {
        int idx = (start_idx + i) % T_max;
        int64_t tick = traj->total_ticks - to_print + i;
        printf("  %-6lld  %-8d  %-10.4f\n", 
               (long long)tick, traj->regimes[idx], traj->h[idx]);
    }
}
