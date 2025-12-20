/**
 * @file pgas.h
 * @brief Particle Gibbs with Ancestor Sampling (PGAS) + PARIS Smoother
 *
 * Implementation based on:
 *   - Lindsten, Jordan & Schön (2014): Particle Gibbs with Ancestor Sampling
 *   - Olsson & Westerborn (2017): PARIS (Particle-based, Rapid Incremental Smoother)
 *
 * Architecture:
 *   1. PGAS runs on background thread to learn θ (transition matrix, emissions)
 *   2. PARIS backward smoothing produces bias-reduced particle cloud
 *   3. Smoothed cloud is injected into main-thread RBPF via Lifeboat Protocol
 *
 * Key features:
 *   - Ancestor Sampling acceptance rate for mixing diagnostics
 *   - O(N) backward kernel (PARIS)
 *   - Adaptive sweep count (min 3, max 10, target acceptance > 0.15)
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 */

#ifndef PGAS_H
#define PGAS_H

#include <stdbool.h>
#include <stdint.h>
#include "circular_buffer.h"

#ifdef __cplusplus
extern "C" {
#endif

/*═══════════════════════════════════════════════════════════════════════════════
 * CONFIGURATION
 *═══════════════════════════════════════════════════════════════════════════════*/

#ifndef PGAS_MAX_PARTICLES
#define PGAS_MAX_PARTICLES 128
#endif

#ifndef PGAS_MAX_TIME
#define PGAS_MAX_TIME 512       /**< Max buffer length (matches circular buffer) */
#endif

#ifndef PGAS_MAX_REGIMES
#define PGAS_MAX_REGIMES 8
#endif

/** Adaptive sweep configuration */
#define PGAS_MIN_SWEEPS 3
#define PGAS_MAX_SWEEPS 10
#define PGAS_TARGET_ACCEPTANCE 0.15
#define PGAS_ABORT_ACCEPTANCE 0.10

/*═══════════════════════════════════════════════════════════════════════════════
 * DATA STRUCTURES
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * Single particle state
 */
typedef struct {
    int regime;         /**< Discrete regime index */
    double h;           /**< Continuous state (log-volatility) */
    int ancestor;       /**< Ancestor index at previous timestep */
} PGASParticle;

/**
 * Model parameters being learned
 */
typedef struct {
    int K;                                          /**< Number of regimes */
    double trans[PGAS_MAX_REGIMES * PGAS_MAX_REGIMES]; /**< Transition matrix [K×K] */
    double mu_vol[PGAS_MAX_REGIMES];                /**< Emission means */
    double sigma_vol[PGAS_MAX_REGIMES];             /**< Emission std devs */
    double phi;                                     /**< AR(1) persistence for h */
    double sigma_h;                                 /**< Innovation std for h */
} PGASModel;

/**
 * Sufficient statistics for parameter learning
 */
typedef struct {
    double trans_counts[PGAS_MAX_REGIMES * PGAS_MAX_REGIMES];  /**< Transition counts */
    double regime_counts[PGAS_MAX_REGIMES];                    /**< Regime occupancy */
    double h_sum[PGAS_MAX_REGIMES];                            /**< Sum of h per regime */
    double h_sq_sum[PGAS_MAX_REGIMES];                         /**< Sum of h² per regime */
} PGASSuffStats;

/**
 * Main PGAS state structure
 */
typedef struct {
    /*───────────────────────────────────────────────────────────────────────────
     * Configuration
     *───────────────────────────────────────────────────────────────────────────*/
    int N;              /**< Number of particles */
    int T;              /**< Current buffer length */
    int K;              /**< Number of regimes */
    
    /*───────────────────────────────────────────────────────────────────────────
     * Particle Storage [T][N]
     * Memory layout: time-major for cache-friendly sequential access
     *───────────────────────────────────────────────────────────────────────────*/
    int regimes[PGAS_MAX_TIME][PGAS_MAX_PARTICLES];
    double h[PGAS_MAX_TIME][PGAS_MAX_PARTICLES];
    double weights[PGAS_MAX_TIME][PGAS_MAX_PARTICLES];    /**< Normalized */
    double log_weights[PGAS_MAX_TIME][PGAS_MAX_PARTICLES];/**< Unnormalized log */
    int ancestors[PGAS_MAX_TIME][PGAS_MAX_PARTICLES];
    
    /*───────────────────────────────────────────────────────────────────────────
     * Reference Trajectory (Conditioned in CSMC)
     *───────────────────────────────────────────────────────────────────────────*/
    int ref_regimes[PGAS_MAX_TIME];
    double ref_h[PGAS_MAX_TIME];
    int ref_ancestors[PGAS_MAX_TIME];
    int ref_idx;        /**< Which particle index is the reference (usually N-1) */
    
    /*───────────────────────────────────────────────────────────────────────────
     * Model Parameters
     *───────────────────────────────────────────────────────────────────────────*/
    PGASModel model;
    PGASModel model_prior;      /**< Prior for Bayesian parameter updates */
    PGASSuffStats suff_stats;
    
    /*───────────────────────────────────────────────────────────────────────────
     * Observations (copied from circular buffer)
     *───────────────────────────────────────────────────────────────────────────*/
    double observations[PGAS_MAX_TIME];
    
    /*───────────────────────────────────────────────────────────────────────────
     * PARIS Backward Smoothing Output
     *───────────────────────────────────────────────────────────────────────────*/
    int smoothed_ancestors[PGAS_MAX_TIME][PGAS_MAX_PARTICLES];
    bool smoothing_done;
    
    /*───────────────────────────────────────────────────────────────────────────
     * Mixing Diagnostics
     *───────────────────────────────────────────────────────────────────────────*/
    int ancestor_proposals;     /**< Total AS attempts in current sweep */
    int ancestor_accepts;       /**< AS moves accepted (ancestor changed) */
    double acceptance_rate;     /**< accepts / proposals */
    
    /*───────────────────────────────────────────────────────────────────────────
     * Sweep Statistics
     *───────────────────────────────────────────────────────────────────────────*/
    int total_sweeps;
    int current_sweep;
    double sweep_times_us[PGAS_MAX_SWEEPS];
    
    /*───────────────────────────────────────────────────────────────────────────
     * RNG State (for reproducibility)
     *───────────────────────────────────────────────────────────────────────────*/
    uint64_t rng_state[2];      /**< xoroshiro128+ state */
    
} PGASState;

/**
 * Result of PGAS inference
 */
typedef enum {
    PGAS_SUCCESS,               /**< Converged, ready for Lifeboat */
    PGAS_STILL_MIXING,          /**< More sweeps needed */
    PGAS_FAILED_TO_MIX,         /**< Aborted after max sweeps */
    PGAS_INVALID_INPUT,         /**< Bad input data */
} PGASResult;

/**
 * Lifeboat packet: output of PGAS+PARIS for main thread injection
 */
typedef struct {
    /*───────────────────────────────────────────────────────────────────────────
     * Model Parameters (new θ)
     *───────────────────────────────────────────────────────────────────────────*/
    int K;
    double trans[PGAS_MAX_REGIMES * PGAS_MAX_REGIMES];
    double mu_vol[PGAS_MAX_REGIMES];
    double sigma_vol[PGAS_MAX_REGIMES];
    double phi;
    double sigma_h;
    
    /*───────────────────────────────────────────────────────────────────────────
     * Smoothed Particle Cloud at Final Time
     *───────────────────────────────────────────────────────────────────────────*/
    int n_particles;
    int final_regimes[PGAS_MAX_PARTICLES];
    double final_h[PGAS_MAX_PARTICLES];
    double final_weights[PGAS_MAX_PARTICLES];
    
    /*───────────────────────────────────────────────────────────────────────────
     * Diagnostics
     *───────────────────────────────────────────────────────────────────────────*/
    double ancestor_acceptance;
    int sweeps_used;
    double total_time_us;
    
    /*───────────────────────────────────────────────────────────────────────────
     * Timing
     *───────────────────────────────────────────────────────────────────────────*/
    uint64_t first_tick;        /**< Oldest tick in buffer */
    uint64_t last_tick;         /**< Newest tick in buffer */
    int buffer_len;             /**< Number of observations processed */
    
} LifeboatPacket;

/*═══════════════════════════════════════════════════════════════════════════════
 * INITIALIZATION
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Initialize PGAS state
 *
 * @param pgas      State to initialize
 * @param N         Number of particles (recommend 100)
 * @param K         Number of regimes
 * @param seed      RNG seed
 */
void pgas_init(PGASState *pgas, int N, int K, uint64_t seed);

/**
 * @brief Set model parameters (from current RBPF model)
 *
 * @param pgas      PGAS state
 * @param trans     Transition matrix [K×K]
 * @param mu_vol    Emission means [K]
 * @param sigma_vol Emission std devs [K]
 * @param phi       AR(1) persistence
 * @param sigma_h   Innovation std
 */
void pgas_set_model(PGASState *pgas,
                    const double *trans,
                    const double *mu_vol,
                    const double *sigma_vol,
                    double phi,
                    double sigma_h);

/**
 * @brief Set initial reference trajectory (from current RBPF MAP path)
 *
 * @param pgas    PGAS state
 * @param regimes Regime sequence [T]
 * @param h       Log-vol sequence [T]
 * @param T       Sequence length
 */
void pgas_set_reference(PGASState *pgas,
                        const int *regimes,
                        const double *h,
                        int T);

/**
 * @brief Load observations from buffer snapshot
 *
 * @param pgas     PGAS state
 * @param snapshot Buffer snapshot from main thread
 */
void pgas_load_observations(PGASState *pgas, const BufferSnapshot *snapshot);

/*═══════════════════════════════════════════════════════════════════════════════
 * CORE PGAS OPERATIONS
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Run one CSMC sweep with ancestor sampling
 *
 * This is the core PGAS operation:
 *   1. Forward filter with reference trajectory fixed
 *   2. Ancestor sampling at each timestep
 *   3. Update model parameters from sufficient statistics
 *
 * @param pgas PGAS state
 * @return     Acceptance rate for this sweep
 */
double pgas_csmc_sweep(PGASState *pgas);

/**
 * @brief Run adaptive PGAS until convergence or max sweeps
 *
 * Implements the adaptive sweep strategy:
 *   - Minimum 3 sweeps
 *   - Continue until acceptance_rate > 0.15
 *   - Abort if acceptance_rate < 0.10 after 10 sweeps
 *
 * @param pgas PGAS state
 * @return     PGAS_SUCCESS, PGAS_STILL_MIXING, or PGAS_FAILED_TO_MIX
 */
PGASResult pgas_run_adaptive(PGASState *pgas);

/**
 * @brief Check if PGAS has converged (acceptance > target)
 *
 * @param pgas PGAS state
 * @return     true if ready for PARIS smoothing
 */
bool pgas_has_converged(const PGASState *pgas);

/*═══════════════════════════════════════════════════════════════════════════════
 * PARIS BACKWARD SMOOTHING
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Run PARIS backward smoothing
 *
 * For each particle n at time T, traces back through ancestors
 * using backward kernel to correct forward-filter bias.
 *
 * O(N) per timestep, O(N×T) total.
 *
 * @param pgas PGAS state (must have completed at least one sweep)
 */
void paris_backward_smooth(PGASState *pgas);

/**
 * @brief Extract smoothed particles at time t
 *
 * @param pgas    PGAS state (after PARIS smoothing)
 * @param t       Time index
 * @param regimes Output: regime indices [N]
 * @param h       Output: log-vol values [N]
 * @param weights Output: weights [N] (uniform after smoothing)
 */
void paris_get_smoothed_particles(const PGASState *pgas,
                                   int t,
                                   int *regimes,
                                   double *h,
                                   double *weights);

/*═══════════════════════════════════════════════════════════════════════════════
 * LIFEBOAT PACKET GENERATION
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Generate Lifeboat packet from PGAS+PARIS output
 *
 * Call after pgas_run_adaptive() returns PGAS_SUCCESS and paris_backward_smooth().
 *
 * @param pgas   PGAS state
 * @param packet Output: Lifeboat packet for main thread
 */
void pgas_generate_lifeboat(const PGASState *pgas, LifeboatPacket *packet);

/**
 * @brief Validate Lifeboat packet before injection
 *
 * Checks:
 *   - Acceptance rate > threshold
 *   - Model parameters are valid (rows sum to 1, etc.)
 *   - Particle weights are normalized
 *
 * @param packet Lifeboat packet
 * @return       true if safe to inject
 */
bool lifeboat_validate(const LifeboatPacket *packet);

/*═══════════════════════════════════════════════════════════════════════════════
 * DIAGNOSTICS
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Get current acceptance rate
 */
double pgas_get_acceptance_rate(const PGASState *pgas);

/**
 * @brief Get number of sweeps performed
 */
int pgas_get_sweep_count(const PGASState *pgas);

/**
 * @brief Get effective sample size of particle cloud at time t
 */
double pgas_get_ess(const PGASState *pgas, int t);

/**
 * @brief Print PGAS diagnostics
 */
void pgas_print_diagnostics(const PGASState *pgas);

#ifdef __cplusplus
}
#endif

#endif /* PGAS_H */
