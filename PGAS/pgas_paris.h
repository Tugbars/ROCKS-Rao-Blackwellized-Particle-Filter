/*═══════════════════════════════════════════════════════════════════════════════
 * PGAS-PARIS Integration
 *
 * Extends PGAS with PARIS backward smoothing for path degeneracy fix.
 *
 * Literature:
 *   - Lindsten, Jordan, Schön (2014) - "Particle Gibbs with Ancestor Sampling"
 *   - Olsson & Westerborn (2017) - "Efficient particle-based online smoothing:
 *     the PaRIS algorithm"
 *
 * Problem:
 *   Standard PGAS suffers from path degeneracy - all particles collapse to
 *   single ancestor when tracing backward. Transition counts n_trans[i][j]
 *   come from essentially ONE path, not the full posterior.
 *
 * Solution:
 *   PARIS backward pass re-samples ancestry using full observation sequence
 *   y_{1:T}, giving proper smoothed marginals P(z_t | y_{1:T}).
 *
 * Usage:
 *   PGASParisState *pp = pgas_paris_alloc(pgas, 8);  // 8 trajectories
 *   
 *   // Instead of pgas_mkl_gibbs_sweep():
 *   pgas_paris_gibbs_sweep(pp);
 *   
 *   // Get learned transitions
 *   pgas_paris_get_transitions(pp, trans, K);
 *   
 *   pgas_paris_free(pp);
 *
 *═══════════════════════════════════════════════════════════════════════════════*/

#ifndef PGAS_PARIS_H
#define PGAS_PARIS_H

#include "pgas_mkl.h"
#include "paris_mkl.h"

#ifdef __cplusplus
extern "C" {
#endif

/*═══════════════════════════════════════════════════════════════════════════════
 * CONFIGURATION
 *═══════════════════════════════════════════════════════════════════════════════*/

/* Maximum smoothed trajectories to sample for transition counting */
#define PGAS_PARIS_MAX_TRAJECTORIES 32

/* Default number of trajectories for ensemble transition counting */
#define PGAS_PARIS_DEFAULT_TRAJECTORIES 8

/*═══════════════════════════════════════════════════════════════════════════════
 * SMOOTHED TRAJECTORY STORAGE
 *
 * After PARIS backward pass, we sample M smoothed trajectories and count
 * transitions from ALL of them, not just the reference.
 *═══════════════════════════════════════════════════════════════════════════════*/

typedef struct {
    int T;                  /* Time series length */
    int K;                  /* Number of regimes */
    int M;                  /* Number of trajectories */
    
    /* Smoothed trajectories: [M × T] regime indices */
    int *trajectories;      /* trajectories[m * T + t] = regime at time t, trajectory m */
    
    /* Ensemble transition counts: sum across all M trajectories */
    int n_trans[PGAS_MKL_MAX_REGIMES * PGAS_MKL_MAX_REGIMES];
    
    /* Per-trajectory counts (for diagnostics) */
    int per_traj_counts[PGAS_PARIS_MAX_TRAJECTORIES * PGAS_MKL_MAX_REGIMES * PGAS_MKL_MAX_REGIMES];
    
    /* Statistics */
    float trajectory_diversity;  /* Fraction of unique trajectories */
    float regime_entropy;        /* Entropy of regime distribution */
    
} PGASParisTrajectories;

/*═══════════════════════════════════════════════════════════════════════════════
 * PGAS-PARIS STATE
 *
 * Wrapper that holds both PGAS state and PARIS workspace.
 *═══════════════════════════════════════════════════════════════════════════════*/

typedef struct {
    PGASMKLState *pgas;         /* Underlying PGAS state (NOT owned) */
    PARISMKLState *paris;       /* PARIS backward smoother (owned) */
    
    /* Smoothed trajectory storage */
    PGASParisTrajectories traj;
    
    /* Conversion buffers (PGAS uses float, PARIS load_particles wants double) */
    double *h_double;           /* [T × N] h converted to double */
    double *weights_double;     /* [T × N] weights converted to double */
    
    /* Configuration */
    int n_trajectories;         /* Number of trajectories to sample */
    int use_ensemble_counts;    /* 1 = count from M trajectories, 0 = single ref */
    
    /* Diagnostics */
    float avg_backward_ess;     /* Average ESS during backward pass */
    float path_degeneracy;      /* Measure of path collapse (0=diverse, 1=degenerate) */
    int total_sweeps;           /* Total Gibbs sweeps performed */
    
} PGASParisState;

/*═══════════════════════════════════════════════════════════════════════════════
 * LIFECYCLE
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * Allocate PGAS-PARIS integration state.
 *
 * @param pgas           Existing PGAS state (NOT owned, must outlive this)
 * @param n_trajectories Number of smoothed trajectories for transition counting
 * @return               New state, or NULL on failure
 */
PGASParisState* pgas_paris_alloc(PGASMKLState *pgas, int n_trajectories);

/**
 * Free PGAS-PARIS state (does NOT free underlying PGAS state).
 */
void pgas_paris_free(PGASParisState *state);

/*═══════════════════════════════════════════════════════════════════════════════
 * CORE OPERATIONS
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * Full Gibbs sweep with PARIS backward smoothing.
 *
 * Sequence:
 *   1. CSMC forward pass (ancestor sampling)
 *   2. Copy particles to PARIS state
 *   3. PARIS backward pass (fixes path degeneracy)
 *   4. Sample M smoothed trajectories
 *   5. Count transitions from ALL trajectories
 *   6. Sample new Π from Dirichlet posterior (using ensemble counts)
 *   7. Update adaptive κ (if enabled)
 *
 * @param state  PGAS-PARIS state
 * @return       CSMC acceptance rate
 */
float pgas_paris_gibbs_sweep(PGASParisState *state);

/**
 * Copy PGAS particles to PARIS state.
 *
 * Converts float → double for PARIS API compatibility.
 * Call after pgas_mkl_csmc_sweep().
 *
 * @param state  PGAS-PARIS state
 */
void pgas_paris_copy_particles(PGASParisState *state);

/**
 * Run PARIS backward smoothing.
 *
 * Call after pgas_paris_copy_particles().
 * Populates paris->smoothed with backward-sampled indices.
 *
 * @param state  PGAS-PARIS state
 */
void pgas_paris_backward_smooth(PGASParisState *state);

/**
 * Sample M smoothed trajectories from PARIS output.
 *
 * Extracts regime sequences using paris_mkl_get_trajectory().
 * Populates state->traj.trajectories.
 *
 * @param state  PGAS-PARIS state
 */
void pgas_paris_sample_trajectories(PGASParisState *state);

/**
 * Count transitions from smoothed trajectories.
 *
 * Aggregates counts across all M trajectories into state->traj.n_trans.
 * These counts are used for Dirichlet posterior sampling.
 *
 * @param state  PGAS-PARIS state
 */
void pgas_paris_count_transitions(PGASParisState *state);

/**
 * Sample transition matrix from Dirichlet posterior.
 *
 * Uses PARIS ensemble counts instead of single-trajectory counts.
 * Updates pgas->model.trans and pgas->model.log_trans.
 *
 * @param state  PGAS-PARIS state
 */
void pgas_paris_sample_trans_matrix(PGASParisState *state);

/*═══════════════════════════════════════════════════════════════════════════════
 * CONVENIENCE: RUN MULTIPLE SWEEPS
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * Run multiple Gibbs sweeps with PARIS.
 *
 * @param state     PGAS-PARIS state
 * @param n_sweeps  Number of sweeps to run
 * @param burnin    Number of initial sweeps to discard (no accumulation)
 * @param callback  Optional callback after each sweep (can be NULL)
 * @param user_data Passed to callback
 */
typedef void (*PGASParisSweepCallback)(PGASParisState *state, int sweep, void *user_data);

void pgas_paris_run_sweeps(
    PGASParisState *state,
    int n_sweeps,
    int burnin,
    PGASParisSweepCallback callback,
    void *user_data
);

/*═══════════════════════════════════════════════════════════════════════════════
 * DIAGNOSTICS
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * Get path degeneracy measure.
 *
 * Computes fraction of trajectories that are identical.
 * @return 0.0 = all trajectories unique, 1.0 = all collapsed to single path
 */
float pgas_paris_get_path_degeneracy(const PGASParisState *state);

/**
 * Get trajectory diversity.
 *
 * @return Fraction of unique trajectories among M sampled
 */
float pgas_paris_get_trajectory_diversity(const PGASParisState *state);

/**
 * Print diagnostic summary.
 */
void pgas_paris_print_diagnostics(const PGASParisState *state);

/*═══════════════════════════════════════════════════════════════════════════════
 * TRANSITION MATRIX ACCESS
 *
 * These forward to underlying PGAS state for convenience.
 *═══════════════════════════════════════════════════════════════════════════════*/

static inline void pgas_paris_get_transitions(const PGASParisState *state, float *out, int K) {
    pgas_mkl_get_transitions(state->pgas, out, K);
}

static inline void pgas_paris_get_transition_counts(const PGASParisState *state, int *out, int K) {
    /* Return PARIS ensemble counts, not PGAS single-trajectory counts */
    for (int i = 0; i < K * K; i++) {
        out[i] = state->traj.n_trans[i];
    }
}

static inline float pgas_paris_get_sticky_kappa(const PGASParisState *state) {
    return pgas_mkl_get_sticky_kappa(state->pgas);
}

static inline float pgas_paris_get_chatter_ratio(const PGASParisState *state) {
    return pgas_mkl_get_chatter_ratio(state->pgas);
}

static inline float pgas_paris_get_acceptance_rate(const PGASParisState *state) {
    return pgas_mkl_get_acceptance_rate(state->pgas);
}

/*═══════════════════════════════════════════════════════════════════════════════
 * ENSEMBLE ACCUMULATOR (for posterior mean extraction)
 *
 * Accumulates post-burnin samples to extract posterior mean Π.
 * Fixes the "Golden Sample Fallacy" - never inject single MCMC sample.
 *═══════════════════════════════════════════════════════════════════════════════*/

typedef struct {
    int K;
    int n_samples;
    double trans_sum[PGAS_MKL_MAX_REGIMES * PGAS_MKL_MAX_REGIMES];
    double trans_sum_sq[PGAS_MKL_MAX_REGIMES * PGAS_MKL_MAX_REGIMES];
    double kappa_sum;
    double kappa_sum_sq;
} PGASParisEnsemble;

/**
 * Initialize ensemble accumulator.
 */
void pgas_paris_ensemble_init(PGASParisEnsemble *ens, int K);

/**
 * Accumulate current state into ensemble.
 */
void pgas_paris_ensemble_accumulate(PGASParisEnsemble *ens, const PGASParisState *state);

/**
 * Get posterior mean transition matrix.
 */
void pgas_paris_ensemble_get_mean(const PGASParisEnsemble *ens, float *out);

/**
 * Get posterior standard deviation for element (i,j).
 */
float pgas_paris_ensemble_get_std(const PGASParisEnsemble *ens, int i, int j);

/**
 * Get maximum standard deviation across all elements (convergence check).
 */
float pgas_paris_ensemble_get_max_std(const PGASParisEnsemble *ens);

/**
 * Check if ensemble has converged (max_std < threshold).
 */
int pgas_paris_ensemble_is_converged(const PGASParisEnsemble *ens, float threshold);

#ifdef __cplusplus
}
#endif

#endif /* PGAS_PARIS_H */
