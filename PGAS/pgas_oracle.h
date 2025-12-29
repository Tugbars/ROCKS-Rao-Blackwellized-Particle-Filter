/**
 * @file pgas_oracle.h
 * @brief PGAS Oracle - Background Parameter Server for RBPF
 *
 * Architecture from Integration Schema:
 *
 *   PGAS = Parameter Server. Provides the physics (Π).
 *   RBPF = Game Client. Plays the game with given physics.
 *   Hawkes = Circuit Breaker. Disconnects during crisis.
 *
 * This module implements:
 *   1. Background PGAS loop (Cores 2-31, OpenMP)
 *   2. Double-buffer channel for lock-free Π publication
 *   3. Integration points for tick loop hot swap
 *
 * Threading:
 *   ┌─────────────────────────────────────────────────────────────────┐
 *   │   RBPF (2 pthreads)              PGAS Oracle (30 OpenMP)       │
 *   │   Cores 0-1                      Cores 2-31                    │
 *   │   40μs per tick                  ~100ms-1s per batch           │
 *   └─────────────────────────────────────────────────────────────────┘
 *
 * Data Flow:
 *   Tick Loop:  push observation → [ring buffer] 
 *   PGAS Loop:  [ring buffer] → extract → sweeps → Π → [channel]
 *   Tick Loop:  [channel] → hot swap (if ready AND not crisis)
 */

#ifndef PGAS_ORACLE_H
#define PGAS_ORACLE_H

#include <stdint.h>
#include <stdbool.h>
#include <stdatomic.h>
#include <pthread.h>

#ifdef __cplusplus
extern "C" {
#endif

/*═══════════════════════════════════════════════════════════════════════════════
 * FORWARD DECLARATIONS
 *═══════════════════════════════════════════════════════════════════════════════*/

typedef struct PGASSlidingState PGASSlidingState;

/*═══════════════════════════════════════════════════════════════════════════════
 * CONFIGURATION
 *═══════════════════════════════════════════════════════════════════════════════*/

#define PGAS_ORACLE_MAX_K       8
#define PGAS_ORACLE_ALIGN       64

/*═══════════════════════════════════════════════════════════════════════════════
 * DOUBLE BUFFER CHANNEL (PGAS → RBPF)
 *
 * Lock-free publication of transition matrix.
 * Writer (PGAS) and reader (tick loop) never block.
 *═══════════════════════════════════════════════════════════════════════════════*/

typedef struct {
    float Pi[PGAS_ORACLE_MAX_K * PGAS_ORACLE_MAX_K];
    int64_t tick;                   /* Window end tick */
    float acceptance_rate;          /* Mixing diagnostic */
    int sweeps_used;
    int windows_completed;          /* How many windows PGAS has processed */
    float avg_acceptance;           /* Running average acceptance rate */
} PGASChannelBuffer;

typedef struct {
    _Atomic int write_idx;          /* Which buffer PGAS is writing to */
    _Atomic int ready;              /* New data available for consumption */
    
    PGASChannelBuffer buf[2] __attribute__((aligned(64)));
    
    /* Metadata */
    int K;                          /* Regime count */
    int64_t last_consumed_tick;     /* Last tick consumed by RBPF */
    
} PGASChannel;

/*═══════════════════════════════════════════════════════════════════════════════
 * PGAS ORACLE STATE
 *═══════════════════════════════════════════════════════════════════════════════*/

typedef struct PGASOracleState {
    /* ═══════════════════════════════════════════════════════════════════════
     * OWNED COMPONENTS
     * ═══════════════════════════════════════════════════════════════════════*/
    PGASSlidingState* sliding;      /* Sliding window PGAS */
    PGASChannel channel;            /* Double-buffer output */
    
    /* ═══════════════════════════════════════════════════════════════════════
     * THREAD CONTROL
     * ═══════════════════════════════════════════════════════════════════════*/
    pthread_t thread;
    _Atomic int running;            /* 0 = stop, 1 = run */
    _Atomic int started;            /* Thread has started */
    
    /* ═══════════════════════════════════════════════════════════════════════
     * CONFIGURATION
     * ═══════════════════════════════════════════════════════════════════════*/
    int n_sweeps;                   /* Gibbs sweeps per iteration */
    int core_start;                 /* First core for affinity (e.g., 2) */
    int n_cores;                    /* Number of cores to use (e.g., 30) */
    
    /* ═══════════════════════════════════════════════════════════════════════
     * DIAGNOSTICS
     * ═══════════════════════════════════════════════════════════════════════*/
    _Atomic int64_t iterations;     /* Windows processed */
    _Atomic int64_t observations_pushed;
    float avg_acceptance_rate;
    float avg_iteration_time_ms;
    
} PGASOracleState;

/*═══════════════════════════════════════════════════════════════════════════════
 * LIFECYCLE
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * Allocate PGAS Oracle
 *
 * @param window_size   PGAS window length (e.g., 2000)
 * @param slide_step    Observations per slide (e.g., 500)
 * @param N             Particles (e.g., 256)
 * @param K             Regimes (e.g., 4)
 * @param n_sweeps      Gibbs sweeps per iteration (e.g., 5)
 * @param seed          RNG seed
 * @return              Allocated oracle, or NULL on failure
 */
PGASOracleState* pgas_oracle_alloc(int window_size, int slide_step,
                                    int N, int K, int n_sweeps,
                                    uint32_t seed);

/**
 * Free PGAS Oracle
 */
void pgas_oracle_free(PGASOracleState* oracle);

/*═══════════════════════════════════════════════════════════════════════════════
 * MODEL CONFIGURATION
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * Set model parameters (before starting thread)
 */
void pgas_oracle_set_model(PGASOracleState* oracle,
                           const double* trans,
                           const double* mu_vol,
                           const double* sigma_vol,
                           double phi);

/**
 * Set sticky prior (before starting thread)
 */
void pgas_oracle_set_prior(PGASOracleState* oracle,
                           float alpha, float kappa);

/**
 * Set recency weighting (before starting thread)
 */
void pgas_oracle_set_recency(PGASOracleState* oracle, float lambda);

/**
 * Set core affinity range (before starting thread)
 *
 * @param oracle      Oracle state
 * @param core_start  First core ID (e.g., 2)
 * @param n_cores     Number of cores (e.g., 30)
 */
void pgas_oracle_set_affinity(PGASOracleState* oracle,
                              int core_start, int n_cores);

/*═══════════════════════════════════════════════════════════════════════════════
 * THREAD CONTROL
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * Start background PGAS thread
 *
 * Thread will:
 *   1. Wait for window to fill
 *   2. Run Gibbs sweeps
 *   3. Publish Π to channel
 *   4. Slide window
 *   5. Loop
 */
int pgas_oracle_start(PGASOracleState* oracle);

/**
 * Stop background thread (blocking)
 */
void pgas_oracle_stop(PGASOracleState* oracle);

/**
 * Check if oracle thread is running
 */
bool pgas_oracle_is_running(const PGASOracleState* oracle);

/*═══════════════════════════════════════════════════════════════════════════════
 * OBSERVATION INPUT (Tick Loop)
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * Push observation to oracle's ring buffer
 *
 * Thread-safe: Called from tick loop while PGAS thread runs.
 *
 * @param oracle  Oracle state
 * @param obs     Observation (y_t = log(r_t²))
 * @param tick    Global tick number
 * @return        true if accepted, false if overflow
 */
bool pgas_oracle_push(PGASOracleState* oracle, float obs, int64_t tick);

/*═══════════════════════════════════════════════════════════════════════════════
 * CHANNEL ACCESS (Tick Loop)
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * Check if new Π is available
 *
 * Thread-safe: Called from tick loop.
 */
bool pgas_oracle_ready(const PGASOracleState* oracle);

/**
 * Consume Π from channel (marks as read)
 *
 * Thread-safe: Called from tick loop.
 *
 * @param oracle   Oracle state
 * @param pi_out   Output buffer [K*K]
 * @param tick_out Output: window end tick, or NULL
 * @return         true if consumed, false if not ready
 */
bool pgas_oracle_consume(PGASOracleState* oracle, float* pi_out, int64_t* tick_out);

/**
 * Get channel metadata (non-consuming peek)
 */
void pgas_oracle_peek(const PGASOracleState* oracle,
                      float* acceptance_rate_out,
                      int* sweeps_out);

/*═══════════════════════════════════════════════════════════════════════════════
 * HOT SWAP INTEGRATION (Tick Loop)
 *
 * These functions implement the hot swap logic from the integration schema.
 * Called from tick_update() after Hawkes check.
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * Attempt hot swap: consume Π if ready AND not in crisis
 *
 * Implements the integration schema logic:
 *   if (oracle_ready AND !crisis) {
 *       swap Pi from channel
 *       reset_sufficient_statistics()
 *   }
 *
 * @param oracle    Oracle state
 * @param crisis    Hawkes crisis flag (if true, veto swap)
 * @param pi_out    Output buffer [K*K], receives Π if swapped
 * @param tick_out  Output: window end tick
 * @return          true if swapped, false if vetoed or not ready
 */
bool pgas_oracle_try_hot_swap(PGASOracleState* oracle,
                              bool crisis,
                              float* pi_out,
                              int64_t* tick_out);

/*═══════════════════════════════════════════════════════════════════════════════
 * VALIDATION
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * Validate Π matrix (row stochastic, no NaN)
 *
 * @param pi  Transition matrix [K*K]
 * @param K   Number of regimes
 * @return    true if valid
 */
bool pgas_oracle_validate_pi(const float* pi, int K);

/*═══════════════════════════════════════════════════════════════════════════════
 * DIAGNOSTICS
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * Get iteration count
 */
int64_t pgas_oracle_get_iterations(const PGASOracleState* oracle);

/**
 * Get observations pushed count
 */
int64_t pgas_oracle_get_observations(const PGASOracleState* oracle);

/**
 * Print diagnostics
 */
void pgas_oracle_print_diagnostics(const PGASOracleState* oracle);

#ifdef __cplusplus
}
#endif

#endif /* PGAS_ORACLE_H */
