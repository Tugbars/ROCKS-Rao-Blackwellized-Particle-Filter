/**
 * @file regime_system.h
 * @brief Complete Regime Trading System Integration
 *
 * Architecture from Integration Schema:
 *
 *   PGAS = Parameter Server. Provides the physics (Π).
 *   RBPF = Game Client. Plays the game with given physics.
 *   Hawkes = Circuit Breaker. Disconnects during crisis.
 *   λ = Plasticity Dial. How fast client adapts when disconnected.
 *
 * Threading Model:
 *   ┌─────────────────────────────────────────────────────────────────┐
 *   │   Tick Loop (Cores 0-1)         PGAS Oracle (Cores 2-31)       │
 *   │   40μs per tick                 ~100ms-1s per batch            │
 *   │   regime_system_tick()          (background pthread)           │
 *   └─────────────────────────────────────────────────────────────────┘
 *
 * Usage:
 *   1. RegimeSystemConfig cfg = regime_system_config_defaults();
 *   2. RegimeSystem* sys = regime_system_alloc(&cfg);
 *   3. regime_system_start(sys);  // Start PGAS background thread
 *   4. for each tick: regime_system_tick(sys, obs, tick);
 *   5. regime_system_stop(sys);
 *   6. regime_system_free(sys);
 */

#ifndef REGIME_SYSTEM_H
#define REGIME_SYSTEM_H

#include <stdint.h>
#include <stdbool.h>

#include "pgas_oracle.h"
#include "hawkes_integrator.h"

#ifdef __cplusplus
extern "C" {
#endif

/*═══════════════════════════════════════════════════════════════════════════════
 * CONFIGURATION
 *═══════════════════════════════════════════════════════════════════════════════*/

#define REGIME_SYSTEM_MAX_K  8

/**
 * System configuration
 */
typedef struct {
    /* ═══════════════════════════════════════════════════════════════════════
     * MODEL PARAMETERS
     * ═══════════════════════════════════════════════════════════════════════*/
    int K;                                  /* Number of regimes */
    double trans[REGIME_SYSTEM_MAX_K * REGIME_SYSTEM_MAX_K];  /* Initial Π */
    double mu_vol[REGIME_SYSTEM_MAX_K];     /* Regime log-vol means */
    double sigma_vol[REGIME_SYSTEM_MAX_K];  /* Per-regime AR process noise */
    double phi;                             /* AR(1) persistence */
    
    /* ═══════════════════════════════════════════════════════════════════════
     * PGAS ORACLE
     * ═══════════════════════════════════════════════════════════════════════*/
    int pgas_window_size;       /* PGAS window (e.g., 2000) */
    int pgas_slide_step;        /* Slide amount (e.g., 100) */
    int pgas_particles;         /* PGAS particles (e.g., 256) */
    int pgas_sweeps;            /* Gibbs sweeps per iteration (e.g., 3) */
    float pgas_prior_alpha;     /* Dirichlet prior (e.g., 1.0) */
    float pgas_sticky_kappa;    /* Sticky prior (e.g., 50.0) */
    float pgas_recency_lambda;  /* Recency weighting (e.g., 0.001) */
    
    /* ═══════════════════════════════════════════════════════════════════════
     * RBPF (stub - to be replaced with actual RBPF config)
     * ═══════════════════════════════════════════════════════════════════════*/
    int rbpf_particles;         /* RBPF particles (e.g., 1024) */
    
    /* ═══════════════════════════════════════════════════════════════════════
     * HAWKES CIRCUIT BREAKER
     * ═══════════════════════════════════════════════════════════════════════*/
    HawkesIntegratorConfig hawkes_config;
    
    /* ═══════════════════════════════════════════════════════════════════════
     * ADAPTIVE FORGETTING
     * ═══════════════════════════════════════════════════════════════════════*/
    float lambda_steady;        /* λ when Hawkes is calm (e.g., 0.995) */
    float lambda_crisis;        /* λ when Hawkes is armed/fired (e.g., 0.5) */
    float n_eff_reset;          /* N_eff for count reset after swap (e.g., 1000) */
    
    /* ═══════════════════════════════════════════════════════════════════════
     * CORE AFFINITY
     * ═══════════════════════════════════════════════════════════════════════*/
    int pgas_core_start;        /* First core for PGAS (e.g., 2) */
    int pgas_n_cores;           /* Number of PGAS cores (e.g., 30) */
    
    /* ═══════════════════════════════════════════════════════════════════════
     * RNG
     * ═══════════════════════════════════════════════════════════════════════*/
    uint32_t seed;
    
} RegimeSystemConfig;

/*═══════════════════════════════════════════════════════════════════════════════
 * TICK RESULT
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * Result from each tick update
 */
typedef struct {
    /* Crisis state */
    bool crisis;                    /* Hawkes detected regime break */
    HawkesTriggerState hawkes_state;
    float hawkes_surprise;          /* Surprise in σ units */
    
    /* Hot swap info */
    bool swapped;                   /* PGAS Π was swapped this tick */
    int64_t swap_tick;              /* Window end tick of swapped Π */
    
    /* Current λ */
    float lambda;                   /* Current forgetting factor */
    
    /* RBPF output (stub - to be filled by actual RBPF) */
    int dominant_regime;            /* Most likely regime */
    float regime_probs[REGIME_SYSTEM_MAX_K];
    
} RegimeSystemTickResult;

/*═══════════════════════════════════════════════════════════════════════════════
 * SYSTEM STATE
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * Complete system state
 */
typedef struct RegimeSystem {
    /* ═══════════════════════════════════════════════════════════════════════
     * COMPONENTS
     * ═══════════════════════════════════════════════════════════════════════*/
    PGASOracleState* oracle;        /* PGAS background oracle */
    HawkesIntegrator hawkes;        /* Circuit breaker */
    
    /* RBPF placeholder - replace with actual RBPF state */
    float Pi[REGIME_SYSTEM_MAX_K * REGIME_SYSTEM_MAX_K];
    float lambda;                   /* Current forgetting factor */
    
    /* ═══════════════════════════════════════════════════════════════════════
     * CONFIGURATION
     * ═══════════════════════════════════════════════════════════════════════*/
    int K;
    float lambda_steady;
    float lambda_crisis;
    float n_eff_reset;
    
    /* ═══════════════════════════════════════════════════════════════════════
     * DIAGNOSTICS
     * ═══════════════════════════════════════════════════════════════════════*/
    int64_t total_ticks;
    int64_t crisis_ticks;
    int64_t swaps_performed;
    int64_t swaps_vetoed;
    
} RegimeSystem;

/*═══════════════════════════════════════════════════════════════════════════════
 * LIFECYCLE
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * Get default configuration
 */
RegimeSystemConfig regime_system_config_defaults(void);

/**
 * Allocate system
 */
RegimeSystem* regime_system_alloc(const RegimeSystemConfig* cfg);

/**
 * Free system
 */
void regime_system_free(RegimeSystem* sys);

/**
 * Start PGAS background thread
 */
int regime_system_start(RegimeSystem* sys);

/**
 * Stop PGAS background thread
 */
void regime_system_stop(RegimeSystem* sys);

/*═══════════════════════════════════════════════════════════════════════════════
 * CORE UPDATE
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * Process one tick
 *
 * This is the main entry point called at each market tick.
 * Implements the integration schema:
 *   1. Push observation to PGAS
 *   2. Update Hawkes (crisis detection)
 *   3. Set adaptive λ
 *   4. Hot swap with Hawkes veto
 *   5. RBPF update (stub)
 *
 * @param sys          System state
 * @param observation  Market observation (log-return, etc.)
 * @param tick         Global tick number
 * @return             Tick result with diagnostics
 */
RegimeSystemTickResult regime_system_tick(RegimeSystem* sys, 
                                           float observation, 
                                           int64_t tick);

/*═══════════════════════════════════════════════════════════════════════════════
 * QUERIES
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * Get current transition matrix
 */
void regime_system_get_pi(const RegimeSystem* sys, float* pi_out);

/**
 * Get current λ
 */
float regime_system_get_lambda(const RegimeSystem* sys);

/**
 * Check if in crisis mode
 */
bool regime_system_is_crisis(const RegimeSystem* sys);

/**
 * Get PGAS oracle state (for detailed diagnostics)
 */
const PGASOracleState* regime_system_get_oracle(const RegimeSystem* sys);

/**
 * Get Hawkes state (for detailed diagnostics)
 */
const HawkesIntegrator* regime_system_get_hawkes(const RegimeSystem* sys);

/*═══════════════════════════════════════════════════════════════════════════════
 * DIAGNOSTICS
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * Print system diagnostics
 */
void regime_system_print_diagnostics(const RegimeSystem* sys);

#ifdef __cplusplus
}
#endif

#endif /* REGIME_SYSTEM_H */
