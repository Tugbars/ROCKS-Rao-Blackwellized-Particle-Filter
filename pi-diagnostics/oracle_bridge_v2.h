/**
 * @file oracle_bridge_v2.h
 * @brief Oracle Bridge V2 - Clean Dual-Loop Architecture
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * ARCHITECTURE
 * ═══════════════════════════════════════════════════════════════════════════
 *
 *                      Observation Buffer
 *                    (Lock-Free Circular)
 *                            │
 *              ┌─────────────┴─────────────┐
 *              │                           │
 *              ▼                           ▼
 *   ┌────────────────────┐     ┌────────────────────┐
 *   │    RBPF Thread     │     │   Oracle Thread    │
 *   │    (Consumer)      │     │   (Producer)       │
 *   │                    │     │                    │
 *   │  - Uses Π          │     │  - Owns ref path   │
 *   │  - Filters z,h     │◄────│  - Full Gibbs      │
 *   │  - Monitors quality│  Π  │  - Learns Π        │
 *   │                    │     │                    │
 *   └────────────────────┘     └────────────────────┘
 *
 * KEY PRINCIPLES:
 * 1. Oracle is INDEPENDENT - learns Π from raw observations, not RBPF
 * 2. RBPF monitors its own quality (RMSE, lag, D₃, ESS)
 * 3. Injection decision based on quality + divergences + confidence
 * 4. Thompson sampling for variance shots when uncertain
 *
 * ═══════════════════════════════════════════════════════════════════════════
 */

#ifndef ORACLE_BRIDGE_V2_H
#define ORACLE_BRIDGE_V2_H

#include <stdbool.h>
#include <stdint.h>

#include "observation_buffer.h"
#include "pi_quality.h"
#include "pi_divergence.h"
#include "injection_decision.h"
#include "pi_staging.h"
#include "thompson_sampler.h"

#ifdef __cplusplus
extern "C" {
#endif

/*═══════════════════════════════════════════════════════════════════════════
 * CONFIGURATION
 *═══════════════════════════════════════════════════════════════════════════*/

#ifndef ORACLE_MAX_K
#define ORACLE_MAX_K 8
#endif

#ifndef ORACLE_MAX_WINDOW
#define ORACLE_MAX_WINDOW 1000
#endif

typedef struct {
    /* Number of regimes */
    int K;
    
    /* Observation window for PGAS */
    int observation_window;       /* T for PGAS (default: 500) */
    
    /* Injection decision config */
    InjectionConfig injection;
    
    /* Divergence thresholds */
    DivergenceThresholds divergence;
    
    /* Thompson prior */
    float thompson_prior_alpha;   /* Dirichlet prior (default: 1.0) */
    
    /* Storvik effective count after injection */
    float storvik_reset_count;    /* Pseudo-count per row (default: 100) */
    
    /* Diagnostics */
    bool verbose;
    
} OracleBridgeConfig;

/*═══════════════════════════════════════════════════════════════════════════
 * RBPF-SIDE STATE
 *
 * This is what RBPF maintains and updates every tick.
 * The Oracle runs separately and doesn't depend on this.
 *═══════════════════════════════════════════════════════════════════════════*/

typedef struct {
    /* Current Π matrices */
    float Pi_operating[ORACLE_MAX_K * ORACLE_MAX_K];  /* Used for predictions */
    float Q_storvik[ORACLE_MAX_K * ORACLE_MAX_K];     /* Storvik sufficient stats */
    float Pi_storvik[ORACLE_MAX_K * ORACLE_MAX_K];    /* Derived from Q (normalized) */
    int   K;
    
    /* Quality monitoring */
    PiQualityState quality;
    
    /* Hawkes intensity (market activity) */
    float hawkes_intensity;
    float hawkes_surprise_sigma;
    
    /* RNG for Thompson sampling */
    ThompsonRNG rng;
    
    /* Statistics */
    int   total_injections;
    int   thompson_samples;
    int   storvik_self_corrections;
    float cumulative_gamma;
    
    /* Last injection info */
    int64_t last_injection_tick;
    float   last_gamma;
    InjectionSource last_source;
    
} RBPFSideState;

/*═══════════════════════════════════════════════════════════════════════════
 * COMPLETE BRIDGE STATE
 *═══════════════════════════════════════════════════════════════════════════*/

typedef struct {
    /* Configuration */
    OracleBridgeConfig config;
    
    /* Shared observation buffer (RBPF writes, Oracle reads) */
    ObservationBuffer *obs_buffer;  /* External, not owned */
    
    /* Π staging (Oracle writes, RBPF reads) */
    PiStagingBuffer pi_staging;
    
    /* RBPF-side state */
    RBPFSideState rbpf;
    
    /* Last Oracle result (for diagnostics) */
    OracleConfidence last_oracle_confidence;
    DivergenceTriad last_divergence;
    
    /* Validation */
    bool initialized;
    
} OracleBridgeV2;

/*═══════════════════════════════════════════════════════════════════════════
 * API - LIFECYCLE
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Get default configuration
 */
OracleBridgeConfig oracle_bridge_config_defaults(int K);

/**
 * Initialize bridge
 * 
 * @param bridge      Bridge state
 * @param config      Configuration
 * @param obs_buffer  Shared observation buffer (external, caller manages)
 * @param Pi_initial  Initial transition matrix (can be NULL for uniform)
 * @return            0 on success
 */
int oracle_bridge_init(
    OracleBridgeV2 *bridge,
    const OracleBridgeConfig *config,
    ObservationBuffer *obs_buffer,
    const float *Pi_initial);

/**
 * Free bridge resources
 */
void oracle_bridge_free(OracleBridgeV2 *bridge);

/**
 * Reset state (keep config)
 */
void oracle_bridge_reset(OracleBridgeV2 *bridge);

/*═══════════════════════════════════════════════════════════════════════════
 * API - RBPF THREAD (Called Every Tick)
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * RBPF tick update - call this every tick
 *
 * This updates quality monitoring and checks for Oracle Π.
 * 
 * @param bridge          Bridge state
 * @param observation     Current observation (y_t)
 * @param prediction      RBPF's prediction (E[y_t|y_{1:t-1}])
 * @param log_likelihood  Marginal log-likelihood
 * @param map_regime      Current MAP regime estimate
 * @param weights         Particle weights (normalized)
 * @param N               Number of particles
 * @param tick            Current tick index
 * @return                InjectionDecision (check should_inject)
 */
InjectionDecision oracle_bridge_rbpf_tick(
    OracleBridgeV2 *bridge,
    double observation,
    double prediction,
    double log_likelihood,
    int map_regime,
    const double *weights,
    int N,
    int64_t tick);

/**
 * Update Hawkes intensity (call separately if using external Hawkes)
 */
void oracle_bridge_update_hawkes(
    OracleBridgeV2 *bridge,
    float intensity,
    float surprise_sigma);

/**
 * Update Storvik counts (after RBPF particle update)
 * 
 * @param bridge    Bridge state
 * @param Q_new     New sufficient statistics from RBPF
 */
void oracle_bridge_update_storvik(
    OracleBridgeV2 *bridge,
    const float *Q_new);

/**
 * Apply injection decision
 * 
 * Call this after oracle_bridge_rbpf_tick if decision.should_inject
 * 
 * @param bridge    Bridge state
 * @param decision  Decision from rbpf_tick
 * @return          0 on success
 */
int oracle_bridge_apply_injection(
    OracleBridgeV2 *bridge,
    const InjectionDecision *decision);

/**
 * Get current operating Π
 */
const float *oracle_bridge_get_pi(const OracleBridgeV2 *bridge);

/**
 * Get Storvik counts
 */
const float *oracle_bridge_get_storvik(const OracleBridgeV2 *bridge);

/*═══════════════════════════════════════════════════════════════════════════
 * API - ORACLE THREAD (Called by external PGAS)
 *
 * Note: The actual PGAS computation is external. These functions
 * handle the handoff of results.
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Submit Oracle result (call from Oracle thread after PGAS)
 * 
 * @param bridge          Bridge state
 * @param Pi_oracle       Learned transition matrix
 * @param Q_oracle        Sufficient statistics
 * @param acceptance_rate PGAS acceptance rate
 * @param min_row_count   Minimum row sum in Q
 * @param sweeps_used     Number of PGAS sweeps
 * @param window_start    Start tick of observation window
 * @param window_end      End tick of observation window
 */
void oracle_bridge_submit_oracle(
    OracleBridgeV2 *bridge,
    const float *Pi_oracle,
    const float *Q_oracle,
    float acceptance_rate,
    float min_row_count,
    int sweeps_used,
    int64_t window_start,
    int64_t window_end);

/*═══════════════════════════════════════════════════════════════════════════
 * API - DIAGNOSTICS
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Get current quality snapshot
 */
PiQualitySnapshot oracle_bridge_get_quality(const OracleBridgeV2 *bridge);

/**
 * Get current divergences
 */
DivergenceTriad oracle_bridge_get_divergence(const OracleBridgeV2 *bridge);

/**
 * Get last Oracle confidence
 */
OracleConfidence oracle_bridge_get_oracle_confidence(const OracleBridgeV2 *bridge);

/**
 * Print state summary
 */
void oracle_bridge_print_state(const OracleBridgeV2 *bridge);

/**
 * Get statistics
 */
typedef struct {
    int   total_injections;
    int   thompson_samples;
    int   storvik_self_corrections;
    float avg_gamma;
    float current_quality_score;
    float current_ess;
    float current_rmse_ratio;
    float current_d3;
} OracleBridgeStats;

OracleBridgeStats oracle_bridge_get_stats(const OracleBridgeV2 *bridge);

#ifdef __cplusplus
}
#endif

#endif /* ORACLE_BRIDGE_V2_H */
