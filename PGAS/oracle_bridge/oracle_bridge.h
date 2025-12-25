/**
 * @file oracle_bridge.h
 * @brief Bridge between PGAS Oracle and SAEM Blender
 *
 * Full Oracle integration pipeline:
 *
 *   ┌─────────────────────────────────────────────────────────────────┐
 *   │  RBPF Thread                     │  Oracle Thread               │
 *   ├─────────────────────────────────────────────────────────────────┤
 *   │                                  │                              │
 *   │  [Hawkes + KL] ──trigger?──────► │  [Scout] pre-validate        │
 *   │                                  │     │                        │
 *   │                                  │     ▼                        │
 *   │  [RBPFTrajectory] ──snapshot───► │  [PGAS] run + confidence     │
 *   │                                  │     │                        │
 *   │                                  │     ▼                        │
 *   │                                  │  [SAEM] blend with γ(conf)   │
 *   │                                  │     │                        │
 *   │                                  │     ▼                        │
 *   │  [RBPF Π update] ◄──Thompson─── │  [Thompson] sample/exploit   │
 *   │                                  │                              │
 *   └─────────────────────────────────────────────────────────────────┘
 *
 * Key features:
 *   - Dual-gate trigger (Hawkes + KL)
 *   - Scout sweep pre-validation (degeneracy check)
 *   - PGAS Confidence → adaptive γ
 *   - Thompson sampling for explore/exploit
 *   - Tempered path injection (anti-confirmation bias)
 *
 * Reference: ORACLE_INTEGRATION_PLAN.md
 */

#ifndef ORACLE_BRIDGE_H
#define ORACLE_BRIDGE_H

#include "hawkes_integrator.h"
#include "saem_blender.h"
#include "pgas_mkl.h"
#include "kl_trigger.h"
#include "pgas_confidence.h"
#include "thompson_sampler.h"
#include "paris_mkl.h"

#include <stdbool.h>

#ifdef __cplusplus
extern "C"
{
#endif

    /*═══════════════════════════════════════════════════════════════════════════
     * CONFIGURATION
     *═══════════════════════════════════════════════════════════════════════════*/

    typedef struct
    {
        /* PGAS settings */
        int pgas_particles;       /* N particles (default: 256) */
        int pgas_sweeps_min;      /* Minimum Gibbs sweeps (default: 3) */
        int pgas_sweeps_max;      /* Maximum Gibbs sweeps (default: 10) */
        float pgas_target_accept; /* Target acceptance rate (default: 0.15) */

        /* Exponential Weighting (Window Paradox Solution)
         * Half-life = ln(2) / lambda ≈ 693 / lambda
         * lambda=0.001 → half-life ~693 ticks (prioritizes "Now") */
        float recency_lambda; /* Decay rate (default: 0.001, 0=disabled) */

        /* Dual-gate trigger (Hawkes + KL) */
        bool use_dual_gate;       /* Require both signals (default: true) */
        float kl_threshold_sigma; /* KL surprise threshold in σ (default: 2.0) */

        /* Scout sweep pre-validation */
        bool use_scout_sweep;        /* Enable scout validation (default: true) */
        int scout_sweeps;            /* Number of scout sweeps (default: 5) */
        float scout_min_acceptance;  /* Minimum acceptance (default: 0.10) */
        float scout_min_unique_frac; /* Minimum unique fraction (default: 0.25) */
        float scout_entropy_skip;    /* Skip PGAS if entropy below (default: 0.1) */

        /* Reference path tempering */
        bool use_tempered_path; /* Enable anti-confirmation bias (default: true) */
        float temper_flip_prob; /* Flip probability (default: 0.05) */

        /* Thompson sampling */
        float thompson_exploit_thresh; /* Row sum threshold (default: 500) */

        /* Confidence-based γ control */
        float gamma_on_regime_change; /* γ after tier-2 reset (default: 0.50) */
        float gamma_on_degeneracy;    /* γ when PGAS fails (default: 0.02) */

        /* Diagnostics */
        bool verbose; /* Print diagnostics (default: false) */

    } OracleBridgeConfig;

    /*═══════════════════════════════════════════════════════════════════════════
     * STATE
     *═══════════════════════════════════════════════════════════════════════════*/

    typedef struct
    {
        /* Configuration */
        OracleBridgeConfig config;
        int n_regimes;

        /* Component handles (not owned - caller manages lifecycle) */
        HawkesIntegrator *hawkes;  /* Trigger detector */
        KLTrigger *kl_trigger;     /* KL surprise detector (dual-gate) */
        SAEMBlender *blender;      /* Parameter blender */
        PGASMKLState *pgas;        /* Oracle sampler */
        PARISMKLState *paris;      /* Scout sweep sampler (can be NULL) */
        ThompsonSampler *thompson; /* Explore/exploit sampler (can be NULL) */

        /* Last trigger info */
        float last_hawkes_surprise;
        float last_kl_surprise;
        int last_trigger_tick;

        /* Last run diagnostics */
        PGASConfidence last_confidence;
        bool last_scout_valid;
        bool last_scout_skipped_pgas;

        /* Statistics */
        int total_oracle_calls;
        int successful_blends;
        int scout_skip_count;    /* Times scout allowed PGAS skip */
        int regime_change_count; /* Tier-2 resets triggered */
        int degeneracy_count;    /* PGAS failures detected */
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
     * Initialize bridge (extended)
     *
     * @param bridge     Bridge state to initialize
     * @param cfg        Configuration (NULL for defaults)
     * @param hawkes     Hawkes trigger (caller owns, can be NULL)
     * @param kl_trigger KL surprise trigger (caller owns, NULL to disable dual-gate)
     * @param blender    SAEM blender (caller owns, required)
     * @param pgas       PGAS sampler (caller owns, required)
     * @param paris      PARIS smoother for scout (caller owns, NULL to disable scout)
     * @param thompson   Thompson sampler (caller owns, NULL for mean only)
     * @return 0 on success
     */
    int oracle_bridge_init_full(OracleBridge *bridge,
                                const OracleBridgeConfig *cfg,
                                HawkesIntegrator *hawkes,
                                KLTrigger *kl_trigger,
                                SAEMBlender *blender,
                                PGASMKLState *pgas,
                                PARISMKLState *paris,
                                ThompsonSampler *thompson);

    /**
     * Initialize bridge (backward compatible)
     */
    int oracle_bridge_init(OracleBridge *bridge,
                           const OracleBridgeConfig *cfg,
                           HawkesIntegrator *hawkes,
                           KLTrigger *kl_trigger,
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
    typedef struct
    {
        bool should_trigger;     /* Fire Oracle? */
        float hawkes_surprise;   /* Hawkes surprise (σ) */
        float kl_surprise;       /* KL surprise (σ) - if dual-gate */
        bool triggered_by_panic; /* Absolute panic override? */
        int ticks_since_last;    /* Ticks since last Oracle call */
    } OracleTriggerResult;

    /**
     * Check if Oracle should fire
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
     * Oracle run result (extended)
     */
    typedef struct
    {
        bool success; /* Oracle completed successfully */

        /* Scout phase */
        bool scout_ran;          /* Did scout run? */
        bool scout_valid;        /* Was scout mixing adequate? */
        bool scout_skipped_pgas; /* Did scout allow PGAS skip? */
        float scout_entropy;     /* Scout path entropy */
        int scout_unique_paths;  /* Scout unique path count */

        /* PGAS phase */
        bool pgas_ran;         /* Did PGAS run? */
        float acceptance_rate; /* PGAS acceptance rate */
        float ess_fraction;    /* Final ESS / N */
        int sweeps_used;       /* Gibbs sweeps performed */
        int temper_flips;      /* Path flips injected */

        /* Confidence metrics */
        float confidence_score;      /* Overall PGAS confidence */
        float path_divergence;       /* Fraction of path changed */
        bool regime_change_detected; /* Tier-2 reset triggered? */
        bool degeneracy_detected;    /* PGAS failed to mix? */

        /* SAEM blend */
        float gamma_used;         /* γ value from confidence */
        float kl_divergence;      /* KL(Π_new || Π_old) */
        float diag_before;        /* Avg diagonal before blend */
        float diag_after;         /* Avg diagonal after blend */
        bool stickiness_adjusted; /* κ control triggered? */

        /* Thompson */
        bool thompson_explored; /* Did Thompson sample (vs exploit)? */

    } OracleRunResult;

    /**
     * Run Oracle with full pipeline
     *
     * This is the main entry point when trigger fires.
     * Flow: Scout → PGAS → Confidence → SAEM → Thompson
     *
     * @param bridge           Bridge state
     * @param rbpf_path        RBPF MAP path [T] (regime indices)
     * @param rbpf_h           RBPF MAP log-vol [T]
     * @param observations     Log-squared returns [T]
     * @param T                Sequence length
     * @param trigger_surprise Hawkes surprise that triggered
     * @return Run result with full diagnostics
     */
    OracleRunResult oracle_bridge_run(
        OracleBridge *bridge,
        const int *rbpf_path,
        const double *rbpf_h,
        const double *observations,
        int T,
        float trigger_surprise);

    /**
     * Get current transition matrix (from SAEM)
     */
    void oracle_bridge_get_Pi(const OracleBridge *bridge, float *Pi_out);

    /**
     * Get Thompson-sampled Π (explore/exploit)
     *
     * Call this after oracle_bridge_run() to get the actual Π to use.
     * If Thompson is NULL, returns SAEM mean.
     */
    void oracle_bridge_get_Pi_thompson(OracleBridge *bridge, float *Pi_out);

    /**
     * Get sufficient statistics Q (for external Thompson)
     */
    void oracle_bridge_get_Q(const OracleBridge *bridge, float *Q_out);

    /*═══════════════════════════════════════════════════════════════════════════
     * API - DIAGNOSTICS
     *═══════════════════════════════════════════════════════════════════════════*/

    /**
     * Bridge statistics (extended)
     */
    typedef struct
    {
        int total_oracle_calls;
        int successful_blends;
        int scout_skip_count;
        int regime_change_count;
        int degeneracy_count;
        float avg_acceptance_rate;
        float avg_kl_change;
        float current_gamma;
        float current_avg_diagonal;
        float thompson_explore_ratio;
    } OracleBridgeStats;

    /**
     * Get statistics
     */
    void oracle_bridge_get_stats(const OracleBridge *bridge, OracleBridgeStats *stats);

    /**
     * Get last confidence metrics
     */
    void oracle_bridge_get_last_confidence(const OracleBridge *bridge, PGASConfidence *conf);

    /**
     * Print diagnostics
     */
    void oracle_bridge_print_state(const OracleBridge *bridge);

#ifdef __cplusplus
}
#endif

#endif /* ORACLE_BRIDGE_H */