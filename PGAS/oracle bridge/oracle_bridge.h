/**
 * @file oracle_bridge.h
 * @brief Bridge between PGAS Oracle and SAEM Blender
 *
 * Connects:
 *   - Hawkes Trigger (Layer 1) → decides WHEN to call Oracle
 *   - PGAS Oracle (Layer 2) → runs particle Gibbs sampling
 *   - SAEM Blender (Layer 3) → safely blends results into RBPF
 *
 * Usage:
 *   1. oracle_bridge_init() - setup with PGAS and SAEM handles
 *   2. oracle_bridge_check_trigger() - check if Oracle should fire
 *   3. oracle_bridge_run() - execute Oracle and blend results
 *   4. oracle_bridge_get_Pi() - get updated transition matrix for RBPF
 */

#ifndef ORACLE_BRIDGE_H
#define ORACLE_BRIDGE_H

#include "hawkes_integrator.h"
#include "saem_blender.h"
#include "pgas_mkl.h"

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/*═══════════════════════════════════════════════════════════════════════════
 * CONFIGURATION
 *═══════════════════════════════════════════════════════════════════════════*/

typedef struct {
    /* PGAS settings */
    int   pgas_particles;          /* N particles (default: 256) */
    int   pgas_sweeps_min;         /* Minimum Gibbs sweeps (default: 3) */
    int   pgas_sweeps_max;         /* Maximum Gibbs sweeps (default: 10) */
    float pgas_target_accept;      /* Target acceptance rate (default: 0.15) */
    
    /* Exponential Weighting (Window Paradox Solution)
     * Half-life = ln(2) / lambda ≈ 693 / lambda
     * lambda=0.001 → half-life ~693 ticks (prioritizes "Now") */
    float recency_lambda;          /* Decay rate (default: 0.001, 0=disabled) */
    
    /* Dual-gate trigger (Hawkes + KL) */
    bool  use_dual_gate;           /* Require both signals (default: true) */
    float kl_threshold_sigma;      /* KL surprise threshold in σ (default: 2.0) */
    
    /* Reference path tempering */
    bool  use_tempered_path;       /* Enable anti-confirmation bias (default: true) */
    float temper_flip_prob;        /* Flip probability (default: 0.05) */
    
    /* Diagnostics */
    bool  verbose;                 /* Print diagnostics (default: false) */
    
} OracleBridgeConfig;

/*═══════════════════════════════════════════════════════════════════════════
 * STATE
 *═══════════════════════════════════════════════════════════════════════════*/

typedef struct {
    /* Configuration */
    OracleBridgeConfig config;
    int n_regimes;
    
    /* Component handles (not owned - caller manages lifecycle) */
    HawkesIntegrator *hawkes;      /* Trigger detector */
    SAEMBlender *blender;          /* Parameter blender */
    PGASMKLState *pgas;            /* Oracle sampler */
    
    /* Last trigger info */
    float last_hawkes_surprise;
    float last_kl_surprise;
    int   last_trigger_tick;
    
    /* Statistics */
    int   total_oracle_calls;
    int   successful_blends;
    float cumulative_kl_change;
    
    /* Validation */
    bool initialized;
    
} OracleBridge;

/*═══════════════════════════════════════════════════════════════════════════
 * API - LIFECYCLE
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Get default configuration
 */
OracleBridgeConfig oracle_bridge_config_defaults(void);

/**
 * Initialize bridge
 *
 * @param bridge    Bridge state to initialize
 * @param cfg       Configuration (NULL for defaults)
 * @param hawkes    Hawkes trigger (caller owns)
 * @param blender   SAEM blender (caller owns)
 * @param pgas      PGAS sampler (caller owns)
 * @return 0 on success
 */
int oracle_bridge_init(OracleBridge *bridge,
                       const OracleBridgeConfig *cfg,
                       HawkesIntegrator *hawkes,
                       SAEMBlender *blender,
                       PGASMKLState *pgas);

/**
 * Reset bridge state
 */
void oracle_bridge_reset(OracleBridge *bridge);

/*═══════════════════════════════════════════════════════════════════════════
 * API - TRIGGER CHECK
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Trigger check result
 */
typedef struct {
    bool  should_trigger;          /* Fire Oracle? */
    float hawkes_surprise;         /* Hawkes surprise (σ) */
    float kl_surprise;             /* KL surprise (σ) - if dual-gate */
    bool  triggered_by_panic;      /* Absolute panic override? */
    int   ticks_since_last;        /* Ticks since last Oracle call */
} OracleTriggerResult;

/**
 * Check if Oracle should fire
 *
 * Call this every tick with Hawkes result and optional KL surprise.
 *
 * @param bridge          Bridge state
 * @param hawkes_result   Result from hawkes_integrator_update()
 * @param kl_surprise     KL surprise in σ (0 to disable dual-gate)
 * @param current_tick    Current tick number
 * @return Trigger decision
 */
OracleTriggerResult oracle_bridge_check_trigger(
    OracleBridge *bridge,
    const HawkesIntegratorResult *hawkes_result,
    float kl_surprise,
    int current_tick);

/*═══════════════════════════════════════════════════════════════════════════
 * API - ORACLE EXECUTION
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Oracle run result
 */
typedef struct {
    bool  success;                 /* Oracle completed successfully */
    float acceptance_rate;         /* PGAS acceptance rate */
    float ess_fraction;            /* Final ESS / N */
    int   sweeps_used;             /* Gibbs sweeps performed */
    float kl_divergence;           /* KL(Π_new || Π_old) */
    float diag_before;             /* Avg diagonal before blend */
    float diag_after;              /* Avg diagonal after blend */
    bool  stickiness_adjusted;     /* κ control triggered? */
    int   temper_flips;            /* Path flips injected */
} OracleRunResult;

/**
 * Run Oracle and blend results
 *
 * This is the main entry point when trigger fires.
 *
 * @param bridge          Bridge state
 * @param rbpf_path       RBPF MAP path [T] (regime indices)
 * @param rbpf_h          RBPF MAP log-vol [T]
 * @param observations    Log-squared returns [T]
 * @param T               Sequence length
 * @param trigger_surprise Hawkes surprise that triggered (for adaptive γ)
 * @return Run result with diagnostics
 */
OracleRunResult oracle_bridge_run(
    OracleBridge *bridge,
    const int *rbpf_path,
    const double *rbpf_h,
    const double *observations,
    int T,
    float trigger_surprise);

/**
 * Get current transition matrix
 *
 * @param bridge   Bridge state
 * @param Pi_out   Output buffer [K×K] row-major
 */
void oracle_bridge_get_Pi(const OracleBridge *bridge, float *Pi_out);

/*═══════════════════════════════════════════════════════════════════════════
 * API - DIAGNOSTICS
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Bridge statistics
 */
typedef struct {
    int   total_oracle_calls;
    int   successful_blends;
    float avg_acceptance_rate;
    float avg_kl_change;
    float current_gamma;
    float current_avg_diagonal;
} OracleBridgeStats;

/**
 * Get statistics
 */
void oracle_bridge_get_stats(const OracleBridge *bridge, OracleBridgeStats *stats);

/**
 * Print diagnostics
 */
void oracle_bridge_print_state(const OracleBridge *bridge);

#ifdef __cplusplus
}
#endif

#endif /* ORACLE_BRIDGE_H */
