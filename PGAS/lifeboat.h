/**
 * @file lifeboat.h
 * @brief Lifeboat Injection: Replace RBPF particles with PARIS-smoothed cloud
 *
 * Architecture:
 * ┌─────────────────────────────────────────────────────────────────────────┐
 * │                         MAIN THREAD (RBPF)                              │
 * │  tick → RBPF.update() → VI approx → [trigger?] → inject(lifeboat)      │
 * └───────────────────────────────────────┬─────────────────────────────────┘
 *                                         │ spawn when triggered
 *                                         ▼
 * ┌─────────────────────────────────────────────────────────────────────────┐
 * │                      BACKGROUND THREAD (PGAS+PARIS)                     │
 * │  snapshot(buffer) → PGAS.run_adaptive() → PARIS.smooth() → signal_done │
 * └─────────────────────────────────────────────────────────────────────────┘
 *
 * Trigger conditions:
 *   1. Periodic: Every N ticks (e.g., 100 ticks = 5 seconds at 20Hz)
 *   2. ESS drop: When ESS < threshold (particle degeneracy)
 *   3. KL divergence: When KL(PGAS || VI) > threshold
 *   4. Manual: External trigger (e.g., detected regime change)
 *
 * Injection modes:
 *   1. REPLACE: Overwrite all RBPF particles with smoothed cloud
 *   2. MIX: Weighted combination (1-α)*RBPF + α*lifeboat
 *   3. RESAMPLE: Resample RBPF from combined pool
 */

#ifndef LIFEBOAT_H
#define LIFEBOAT_H

#include <stdint.h>
#include <stdbool.h>
#include <pthread.h>

#ifdef __cplusplus
extern "C" {
#endif

/*═══════════════════════════════════════════════════════════════════════════════
 * CONFIGURATION
 *═══════════════════════════════════════════════════════════════════════════════*/

#define LIFEBOAT_MAX_PARTICLES 256
#define LIFEBOAT_MAX_REGIMES 8
#define LIFEBOAT_ALIGN 64

/*═══════════════════════════════════════════════════════════════════════════════
 * ENUMS
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * Injection mode
 */
typedef enum {
    LIFEBOAT_MODE_REPLACE,      /**< Full replacement */
    LIFEBOAT_MODE_MIX,          /**< Weighted mixing */
    LIFEBOAT_MODE_RESAMPLE,     /**< Resample from combined pool */
} LifeboatMode;

/**
 * Trigger type
 */
typedef enum {
    LIFEBOAT_TRIGGER_NONE = 0,
    LIFEBOAT_TRIGGER_PERIODIC,  /**< Every N ticks */
    LIFEBOAT_TRIGGER_ESS,       /**< ESS below threshold */
    LIFEBOAT_TRIGGER_KL,        /**< KL divergence high */
    LIFEBOAT_TRIGGER_MANUAL,    /**< External trigger */
} LifeboatTrigger;

/**
 * Background thread state
 */
typedef enum {
    LIFEBOAT_THREAD_IDLE,       /**< Waiting for work */
    LIFEBOAT_THREAD_RUNNING,    /**< PGAS+PARIS in progress */
    LIFEBOAT_THREAD_READY,      /**< Results available */
    LIFEBOAT_THREAD_SHUTDOWN,   /**< Terminating */
} LifeboatThreadState;

/*═══════════════════════════════════════════════════════════════════════════════
 * DATA STRUCTURES
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * Smoothed particle cloud (output from PARIS)
 */
typedef struct {
    int N;                      /**< Particle count */
    int K;                      /**< Regime count */
    int T;                      /**< Buffer length used */
    
    /* Final state at t=T-1 (what we inject) */
    int regimes[LIFEBOAT_MAX_PARTICLES];
    float h[LIFEBOAT_MAX_PARTICLES];
    float weights[LIFEBOAT_MAX_PARTICLES];  /**< Uniform after smoothing */
    
    /* Model parameters (for RBPF to adopt) */
    float trans[LIFEBOAT_MAX_REGIMES * LIFEBOAT_MAX_REGIMES];
    float mu_vol[LIFEBOAT_MAX_REGIMES];
    float sigma_vol[LIFEBOAT_MAX_REGIMES];
    float phi;
    float sigma_h;
    
    /* Diagnostics */
    float ancestor_acceptance;
    int pgas_sweeps;
    double compute_time_ms;
    uint64_t source_tick_id;    /**< Last tick in buffer when generated */
    
    /* Validity */
    bool valid;
    
} LifeboatCloud;

/**
 * Trigger configuration
 */
typedef struct {
    /* Periodic trigger */
    bool enable_periodic;
    int periodic_interval;      /**< Ticks between runs (e.g., 100) */
    
    /* ESS trigger */
    bool enable_ess;
    float ess_threshold;        /**< Trigger if ESS < N * threshold */
    
    /* KL trigger */
    bool enable_kl;
    float kl_threshold;         /**< Trigger if KL > threshold */
    
    /* Injection settings */
    LifeboatMode mode;
    float mix_alpha;            /**< For MIX mode: weight of lifeboat */
    
    /* Cooldown */
    int cooldown_ticks;         /**< Minimum ticks between injections */
    
} LifeboatConfig;

/**
 * Statistics
 */
typedef struct {
    uint64_t total_triggers;
    uint64_t total_injections;
    uint64_t failed_runs;
    uint64_t periodic_triggers;
    uint64_t ess_triggers;
    uint64_t kl_triggers;
    uint64_t manual_triggers;
    double total_compute_time_ms;
    double avg_compute_time_ms;
    float avg_acceptance_rate;
    uint64_t last_injection_tick;
} LifeboatStats;

/**
 * Main Lifeboat manager
 */
typedef struct {
    /* Configuration */
    int N;                      /**< Particle count */
    int K;                      /**< Regime count */
    int buffer_size;            /**< Circular buffer size (T) */
    LifeboatConfig config;
    
    /* Thread management */
    pthread_t worker_thread;
    pthread_mutex_t mutex;
    pthread_cond_t cond_start;
    pthread_cond_t cond_done;
    
    /* Atomic state for lock-free hot-path check */
    volatile int thread_state_atomic;  /**< LifeboatThreadState - atomic read */
    volatile bool shutdown_requested;
    
    /* Double-buffered clouds (zero-copy swap) */
    LifeboatCloud clouds[2];
    int active_cloud;           /**< Index of cloud being written */
    int ready_cloud;            /**< Index of cloud ready for injection (-1 if none) */
    
    /* Trigger state */
    uint64_t current_tick;
    uint64_t last_trigger_tick;
    uint64_t last_injection_tick;
    LifeboatTrigger last_trigger_type;
    
    /* Buffer snapshot for PGAS (copied at trigger time) */
    float *snapshot_obs;        /**< [buffer_size] observations */
    uint64_t *snapshot_ticks;   /**< [buffer_size] tick IDs */
    int snapshot_count;
    uint64_t snapshot_start_tick;  /**< First tick in snapshot */
    uint64_t snapshot_end_tick;    /**< Last tick in snapshot */
    
    /* Reference trajectory for PGAS (from previous run or VI) */
    int *ref_regimes;           /**< [buffer_size] */
    float *ref_h;               /**< [buffer_size] */
    
    /* Pre-allocated worker thread buffers (avoid malloc in hot path) */
    float *worker_obs;          /**< [buffer_size] */
    double *worker_obs_d;       /**< [buffer_size] double conversion */
    int *worker_ref_regimes;    /**< [buffer_size] */
    float *worker_ref_h;        /**< [buffer_size] */
    double *worker_ref_h_d;     /**< [buffer_size] double conversion */
    
    /* Statistics */
    LifeboatStats stats;
    
    /* RNG seed */
    uint32_t seed;
    
} LifeboatManager;

/*═══════════════════════════════════════════════════════════════════════════════
 * LIFECYCLE
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Create Lifeboat manager
 * @param N           Particle count
 * @param K           Regime count
 * @param buffer_size Circular buffer size
 * @param seed        RNG seed
 * @return            Manager (call lifeboat_destroy to free)
 */
LifeboatManager *lifeboat_create(int N, int K, int buffer_size, uint32_t seed);

/**
 * @brief Destroy Lifeboat manager
 */
void lifeboat_destroy(LifeboatManager *mgr);

/**
 * @brief Configure trigger and injection settings
 */
void lifeboat_configure(LifeboatManager *mgr, const LifeboatConfig *config);

/**
 * @brief Set model parameters (called when RBPF model updates)
 */
void lifeboat_set_model(LifeboatManager *mgr,
                        const float *trans,
                        const float *mu_vol,
                        const float *sigma_vol,
                        float phi,
                        float sigma_h);

/*═══════════════════════════════════════════════════════════════════════════════
 * TRIGGER & CHECK
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Check if lifeboat should be triggered
 * @param mgr         Manager
 * @param tick_id     Current tick
 * @param ess         Current ESS (or -1 if unknown)
 * @param kl          Current KL divergence (or -1 if unknown)
 * @return            Trigger type (NONE if no trigger)
 */
LifeboatTrigger lifeboat_check_trigger(LifeboatManager *mgr,
                                        uint64_t tick_id,
                                        float ess,
                                        float kl);

/**
 * @brief Manually trigger lifeboat
 * @return true if trigger accepted, false if already running
 */
bool lifeboat_trigger_manual(LifeboatManager *mgr);

/**
 * @brief Start background PGAS+PARIS run
 * @param mgr         Manager
 * @param observations Circular buffer observations
 * @param tick_ids    Circular buffer tick IDs
 * @param count       Number of observations
 * @param ref_regimes Reference trajectory regimes (or NULL for random init)
 * @param ref_h       Reference trajectory h (or NULL)
 * @return            true if started, false if already running
 */
bool lifeboat_start_run(LifeboatManager *mgr,
                        const float *observations,
                        const uint64_t *tick_ids,
                        int count,
                        const int *ref_regimes,
                        const float *ref_h);

/**
 * @brief Check if results are ready (LOCK-FREE for hot path!)
 * 
 * This uses atomic read - safe to call every tick without mutex.
 * Only acquire mutex when this returns true.
 */
bool lifeboat_is_ready(const LifeboatManager *mgr);

/**
 * @brief Check if background thread is running (LOCK-FREE)
 */
bool lifeboat_is_running(const LifeboatManager *mgr);

/**
 * @brief Get source tick ID of ready cloud (for catch-up calculation)
 * @return Last tick ID in the snapshot used to generate the cloud, or 0 if not ready
 */
uint64_t lifeboat_get_source_tick(const LifeboatManager *mgr);

/*═══════════════════════════════════════════════════════════════════════════════
 * INJECTION
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Get ready cloud for injection (does not consume it)
 * @return Pointer to cloud, or NULL if not ready
 */
const LifeboatCloud *lifeboat_get_cloud(const LifeboatManager *mgr);

/**
 * @brief Inject lifeboat cloud into RBPF particles
 *
 * Depending on mode:
 *   REPLACE:  rbpf_particles = lifeboat_cloud
 *   MIX:      rbpf_particles = (1-α)*rbpf + α*lifeboat
 *   RESAMPLE: resample from combined pool
 *
 * @param mgr         Manager
 * @param rbpf_regimes RBPF particle regimes [N] (modified in place)
 * @param rbpf_h      RBPF particle h values [N] (modified in place)
 * @param rbpf_weights RBPF particle weights [N] (modified in place)
 * @param source_tick_out Output: tick ID the cloud was generated from (for catch-up)
 * @param rng_state   Per-thread RNG state [2] for RESAMPLE mode (or NULL)
 * @return            true if injection performed
 */
bool lifeboat_inject(LifeboatManager *mgr,
                     int *rbpf_regimes,
                     float *rbpf_h,
                     float *rbpf_weights,
                     uint64_t *source_tick_out,
                     uint64_t *rng_state);

/**
 * @brief Consume the ready cloud (marks it as used)
 */
void lifeboat_consume_cloud(LifeboatManager *mgr);

/*═══════════════════════════════════════════════════════════════════════════════
 * FAST-FORWARD (CATCH-UP)
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Fast-forward particles through missed ticks
 *
 * After injection, particles are "correct" for source_tick but we're now
 * at current_tick. This function propagates particles through the gap.
 *
 * IMPORTANT: Pass the cloud returned by lifeboat_get_cloud() to ensure
 * particles are propagated using the correct model parameters.
 *
 * @param cloud           Cloud that was injected (from lifeboat_get_cloud)
 * @param regimes         Particle regimes [N] (modified in place)
 * @param h               Particle h values [N] (modified in place)
 * @param weights         Particle weights [N] (modified in place)
 * @param observations    Observations for catch-up ticks [n_ticks]
 * @param n_ticks         Number of ticks to fast-forward
 * @param reweight        If true, reweight by likelihood (slower but more accurate)
 * @param rng_state       Per-thread RNG state [2] (xoroshiro128+), modified in place
 */
void lifeboat_fast_forward(const LifeboatCloud *cloud,
                           int *regimes,
                           float *h,
                           float *weights,
                           const float *observations,
                           int n_ticks,
                           bool reweight,
                           uint64_t *rng_state);

/*═══════════════════════════════════════════════════════════════════════════════
 * FAST RNG (Lock-free xoroshiro128+)
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Initialize xoroshiro128+ RNG state
 * @param state  Output state [2]
 * @param seed   Seed value
 */
static inline void lifeboat_rng_init(uint64_t state[2], uint64_t seed)
{
    /* SplitMix64 to initialize state */
    uint64_t z = seed;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    state[0] = z ^ (z >> 31);
    z = seed + 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    state[1] = z ^ (z >> 31);
}

/**
 * @brief Generate next random uint64 (xoroshiro128+)
 * @param state  RNG state [2], modified in place
 * @return       Random uint64
 */
static inline uint64_t lifeboat_rng_next(uint64_t state[2])
{
    const uint64_t s0 = state[0];
    uint64_t s1 = state[1];
    const uint64_t result = s0 + s1;
    
    s1 ^= s0;
    state[0] = ((s0 << 24) | (s0 >> 40)) ^ s1 ^ (s1 << 16);
    state[1] = (s1 << 37) | (s1 >> 27);
    
    return result;
}

/**
 * @brief Generate uniform float in [0, 1)
 */
static inline float lifeboat_rng_uniform(uint64_t state[2])
{
    return (float)(lifeboat_rng_next(state) >> 40) * 0x1.0p-24f;
}

/**
 * @brief Generate approximate Gaussian using fast Box-Muller
 * @param state  RNG state [2]
 * @return       Approximate N(0,1) sample
 */
static inline float lifeboat_rng_normal(uint64_t state[2])
{
    /* Fast approximation: sum of 4 uniforms - 2 */
    float u1 = lifeboat_rng_uniform(state);
    float u2 = lifeboat_rng_uniform(state);
    float u3 = lifeboat_rng_uniform(state);
    float u4 = lifeboat_rng_uniform(state);
    return (u1 + u2 + u3 + u4 - 2.0f) * 1.1547f;  /* Scale to match σ=1 */
}

/*═══════════════════════════════════════════════════════════════════════════════
 * DIAGNOSTICS
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Get statistics
 */
const LifeboatStats *lifeboat_get_stats(const LifeboatManager *mgr);

/**
 * @brief Reset statistics
 */
void lifeboat_reset_stats(LifeboatManager *mgr);

/**
 * @brief Print diagnostics
 */
void lifeboat_print_diagnostics(const LifeboatManager *mgr);

#ifdef __cplusplus
}
#endif

#endif /* LIFEBOAT_H */
