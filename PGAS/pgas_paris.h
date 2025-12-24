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
extern "C"
{
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

    typedef struct
    {
        int T; /* Time series length */
        int K; /* Number of regimes */
        int M; /* Number of trajectories */

        /* Smoothed trajectories: [M × T] regime indices */
        int *trajectories; /* trajectories[m * T + t] = regime at time t, trajectory m */

        /* Ensemble transition counts: sum across all M trajectories */
        int n_trans[PGAS_MKL_MAX_REGIMES * PGAS_MKL_MAX_REGIMES];

        /* Per-trajectory counts (for diagnostics) */
        int per_traj_counts[PGAS_PARIS_MAX_TRAJECTORIES * PGAS_MKL_MAX_REGIMES * PGAS_MKL_MAX_REGIMES];

        /* Statistics */
        float trajectory_diversity; /* Fraction of unique trajectories */
        float regime_entropy;       /* Entropy of regime distribution */

    } PGASParisTrajectories;

    /*═══════════════════════════════════════════════════════════════════════════════
     * PGAS-PARIS STATE
     *
     * Wrapper that holds both PGAS state and PARIS workspace.
     *═══════════════════════════════════════════════════════════════════════════════*/

    typedef struct
    {
        PGASMKLState *pgas;   /* Underlying PGAS state (NOT owned) */
        PARISMKLState *paris; /* PARIS backward smoother (owned) */

        /* Smoothed trajectory storage */
        PGASParisTrajectories traj;

        /* Conversion buffers (PGAS uses float, PARIS load_particles wants double) */
        double *h_double;       /* [T × N] h converted to double */
        double *weights_double; /* [T × N] weights converted to double */

        /* OPTIMIZATION: Pre-allocated copy buffers (avoid malloc per sweep) */
        int *regimes_nopad;   /* [T × N] regimes without padding */
        int *ancestors_nopad; /* [T × N] ancestors without padding */
        float *h_traj_cache;  /* [M × T] h values for trajectories */

        /* Configuration */
        int n_trajectories;      /* Number of trajectories to sample */
        int use_ensemble_counts; /* 1 = count from M trajectories, 0 = single ref */

        /* Diagnostics */
        float avg_backward_ess; /* Average ESS during backward pass */
        float path_degeneracy;  /* Measure of path collapse (0=diverse, 1=degenerate) */
        int total_sweeps;       /* Total Gibbs sweeps performed */

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
    PGASParisState *pgas_paris_alloc(PGASMKLState *pgas, int n_trajectories);

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
    void pgas_paris_run_backward(PGASParisState *state);

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
        void *user_data);

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

    static inline void pgas_paris_get_transitions(const PGASParisState *state, float *out, int K)
    {
        pgas_mkl_get_transitions(state->pgas, out, K);
    }

    static inline void pgas_paris_get_transition_counts(const PGASParisState *state, int *out, int K)
    {
        /* Return PARIS ensemble counts, not PGAS single-trajectory counts */
        for (int i = 0; i < K * K; i++)
        {
            out[i] = state->traj.n_trans[i];
        }
    }

    static inline float pgas_paris_get_sticky_kappa(const PGASParisState *state)
    {
        return pgas_mkl_get_sticky_kappa(state->pgas);
    }

    static inline float pgas_paris_get_chatter_ratio(const PGASParisState *state)
    {
        return pgas_mkl_get_chatter_ratio(state->pgas);
    }

    static inline float pgas_paris_get_acceptance_rate(const PGASParisState *state)
    {
        return pgas_mkl_get_acceptance_rate(state->pgas);
    }

    /*═══════════════════════════════════════════════════════════════════════════════
     * ENSEMBLE ACCUMULATOR (for posterior mean extraction)
     *
     * Accumulates post-burnin samples to extract posterior mean Π.
     * Fixes the "Golden Sample Fallacy" - never inject single MCMC sample.
     *═══════════════════════════════════════════════════════════════════════════════*/

    typedef struct
    {
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

    /*═══════════════════════════════════════════════════════════════════════════════
     * REGIME PARAMETER LEARNING
     *
     * Extends PGAS-PARIS to learn regime-specific parameters:
     *   - μ_vol[k]: Long-run mean log-volatility per regime
     *   - σ_vol[k]: Emission spread per regime (optional)
     *
     * Uses conjugate priors for closed-form Gibbs updates:
     *   - μ_k ~ Normal(m₀, s₀²)  →  Normal posterior
     *   - σ_k ~ InvGamma(a, b)   →  InvGamma posterior
     *
     * Literature:
     *   - Lindsten et al. (2014) for Rao-Blackwellized parameter learning
     *   - Fox et al. (2011) for sticky HDP-HMM parameter sampling
     *═══════════════════════════════════════════════════════════════════════════════*/

    /**
     * Prior configuration for regime parameter learning
     */
    typedef struct
    {
        /* μ_vol prior: Normal(mu_prior_mean, mu_prior_var) per regime */
        float mu_prior_mean[PGAS_MKL_MAX_REGIMES]; /**< Prior mean for μ_k */
        float mu_prior_var[PGAS_MKL_MAX_REGIMES];  /**< Prior variance for μ_k */

        /* σ_vol prior: InvGamma(sigma_prior_shape, sigma_prior_scale) */
        float sigma_prior_shape; /**< Shape parameter (a) for all regimes */
        float sigma_prior_scale; /**< Scale parameter (b) for all regimes */

        /* σ_h prior (shared AR innovation): InvGamma(sigma_h_shape, sigma_h_scale) */
        float sigma_h_prior_shape; /**< Shape parameter for σ_h prior */
        float sigma_h_prior_scale; /**< Scale parameter for σ_h prior */

        /* φ prior (AR persistence): Beta(phi_a, phi_b) mapped to [0, 1)
         * For highly persistent SV, use phi_a=20, phi_b=1.5 → mode ~0.95 */
        float phi_prior_a;      /**< Beta shape a (higher → more mass near 1) */
        float phi_prior_b;      /**< Beta shape b (higher → more mass near 0) */
        float phi_proposal_std; /**< MH random walk std on logit(φ) scale */
        float phi_min;          /**< Lower bound for φ (e.g., 0.8) */
        float phi_max;          /**< Upper bound for φ (e.g., 0.995) */

        /* Learning control */
        int learn_mu;      /**< Enable μ_vol learning */
        int learn_sigma;   /**< Enable σ_vol learning */
        int learn_sigma_h; /**< Enable σ_h learning (shared) */
        int learn_phi;     /**< Enable φ learning via MH (shared) */

        /* Ordering constraint to prevent label switching */
        int enforce_ordering; /**< If 1, enforce μ_0 < μ_1 < ... < μ_{K-1} */

        /* MH diagnostics */
        int phi_mh_accepts; /**< Cumulative φ MH accepts */
        int phi_mh_total;   /**< Cumulative φ MH proposals */
    } PGASParisRegimePrior;

    /**
     * Sufficient statistics for regime parameter learning
     * Collected from PARIS ensemble trajectories (Rao-Blackwellized)
     */
    typedef struct
    {
        int K;

        /* Per-regime statistics (averaged across M trajectories) */
        double n_k[PGAS_MKL_MAX_REGIMES];        /**< Expected count in regime k */
        double sum_h_k[PGAS_MKL_MAX_REGIMES];    /**< Sum of h values in regime k */
        double sum_h_sq_k[PGAS_MKL_MAX_REGIMES]; /**< Sum of h² values in regime k */

        /* For AR(1) dynamics: compute residuals h_t - φ*h_{t-1} */
        double sum_resid_k[PGAS_MKL_MAX_REGIMES];    /**< Sum of (h_t - φ*h_{t-1}) in regime k */
        double sum_resid_sq_k[PGAS_MKL_MAX_REGIMES]; /**< Sum of (h_t - φ*h_{t-1})² in regime k */

        /* Global statistics for shared parameter learning (σ_h, φ) */
        double total_T;           /**< Total timesteps (T-1 for transitions) */
        double sum_h_all;         /**< Sum of all h values */
        double sum_h_sq_all;      /**< Sum of all h² values */
        double sum_h_lag_all;     /**< Sum of h_t * h_{t-1} */
        double sum_h_prev_sq_all; /**< Sum of h_{t-1}² */
        double sum_resid_all;     /**< Sum of all residuals (for current φ) */
        double sum_resid_sq_all;  /**< Sum of all residuals² (for current φ) */
    } PGASParisRegimeStats;

    /**
     * Initialize regime learning prior with defaults
     *
     * Defaults:
     *   - μ_k prior: N(-3.0, 2.0²) for all k (weak prior centered at typical vol)
     *   - σ_vol prior: InvGamma(3, 0.1) (weak prior on per-regime spread)
     *   - σ_h prior: InvGamma(3, 0.03) → σ_h ~ 0.1-0.2
     *   - φ prior: Beta(20, 1.5) on [0.8, 0.995] → mode ~ 0.97
     *   - φ MH proposal_std: 0.1 (on logit scale)
     *   - All learning disabled by default
     *   - Ordering enforced by default (prevents label switching)
     *
     * @param prior  Prior struct to initialize
     * @param K      Number of regimes
     */
    void pgas_paris_regime_prior_init(PGASParisRegimePrior *prior, int K);

    /**
     * Set μ_vol prior for a specific regime
     *
     * @param prior      Prior struct
     * @param k          Regime index
     * @param mean       Prior mean
     * @param variance   Prior variance
     */
    void pgas_paris_set_mu_prior(PGASParisRegimePrior *prior, int k,
                                 float mean, float variance);

    /**
     * Set σ_vol prior (per-regime emission spread)
     *
     * Prior: σ²_vol[k] ~ InvGamma(shape, scale)
     * Mode = scale / (shape + 1)
     *
     * Example: shape=3, scale=0.1 → mode ≈ 0.025, giving σ_vol ~ 0.15
     *
     * @param prior  Prior struct
     * @param shape  Inv-Gamma shape parameter (a > 0)
     * @param scale  Inv-Gamma scale parameter (b > 0)
     */
    void pgas_paris_set_sigma_vol_prior(PGASParisRegimePrior *prior,
                                        float shape, float scale);

    /**
     * Set σ_h prior (shared AR innovation std)
     *
     * Prior: σ²_h ~ InvGamma(shape, scale)
     * Mode = scale / (shape + 1)
     *
     * Example: shape=3, scale=0.03 → mode ≈ 0.0075, giving σ_h ~ 0.1
     *
     * @param prior  Prior struct
     * @param shape  Inv-Gamma shape parameter (a > 0)
     * @param scale  Inv-Gamma scale parameter (b > 0)
     */
    void pgas_paris_set_sigma_h_prior(PGASParisRegimePrior *prior,
                                      float shape, float scale);

    /**
     * Set φ prior (AR persistence coefficient)
     *
     * Prior: φ ~ Beta(a, b) on [phi_min, phi_max]
     * Mode = (a - 1) / (a + b - 2) scaled to [phi_min, phi_max]
     *
     * For highly persistent SV models:
     *   - a=20, b=1.5 → mode ≈ 0.974
     *   - phi_min=0.85, phi_max=0.995 keeps φ in realistic range
     *
     * The proposal_std controls MH random walk step size on logit scale:
     *   - Too small (< 0.05): slow mixing, high autocorrelation
     *   - Too large (> 0.3): low acceptance rate
     *   - Target: 20-50% acceptance rate
     *
     * @param prior         Prior struct
     * @param a             Beta shape a (higher → more mass near phi_max)
     * @param b             Beta shape b (higher → more mass near phi_min)
     * @param phi_min       Lower bound for φ
     * @param phi_max       Upper bound for φ
     * @param proposal_std  MH random walk std on logit(φ) scale
     */
    void pgas_paris_set_phi_prior(PGASParisRegimePrior *prior,
                                  float a, float b,
                                  float phi_min, float phi_max,
                                  float proposal_std);

    /**
     * Get φ MH acceptance rate
     *
     * Returns the cumulative acceptance rate for φ proposals.
     * Target: 20-50% for good mixing. Adjust proposal_std if outside range.
     *
     * @param prior  Prior struct (must have been used in sampling)
     * @return       Acceptance rate in [0, 1], or 0 if no proposals yet
     */
    float pgas_paris_get_phi_acceptance_rate(const PGASParisRegimePrior *prior);

    /**
     * Collect sufficient statistics from PARIS ensemble
     *
     * Computes Rao-Blackwellized statistics across M trajectories:
     *   - E[n_k]: Expected count in regime k
     *   - E[Σh | z=k]: Expected sum of h values in regime k
     *   - E[Σh² | z=k]: Expected sum of h² in regime k
     *
     * @param state  PGAS-PARIS state (after backward smoothing + trajectory sampling)
     * @param stats  Output statistics struct
     */
    void pgas_paris_collect_regime_stats(PGASParisState *state,
                                         PGASParisRegimeStats *stats);

    /**
     * Sample μ_vol from Normal posterior
     *
     * For AR(1) SV model: h_t = μ_k(1-φ) + φ*h_{t-1} + σ_h*ε
     * We estimate μ_k from the "target" that h regresses toward.
     *
     * Posterior: μ_k | data ~ N(μ_post, σ²_post)
     * where:
     *   σ²_post = 1 / (1/s₀² + n_k*(1-φ)²/σ_h²)
     *   μ_post = σ²_post * (m₀/s₀² + (1-φ)*Σresid_k/σ_h²)
     *
     * @param state  PGAS-PARIS state
     * @param stats  Sufficient statistics from collect_regime_stats
     * @param prior  Prior configuration
     */
    void pgas_paris_sample_mu_vol(PGASParisState *state,
                                  const PGASParisRegimeStats *stats,
                                  const PGASParisRegimePrior *prior);

    /**
     * Sample σ_vol from Inverse-Gamma posterior (optional)
     *
     * Posterior: σ²_k | data ~ InvGamma(a_post, b_post)
     * where:
     *   a_post = a₀ + n_k/2
     *   b_post = b₀ + 0.5 * Σ(h_t - μ_k)²
     *
     * @param state  PGAS-PARIS state
     * @param stats  Sufficient statistics
     * @param prior  Prior configuration
     */
    void pgas_paris_sample_sigma_vol(PGASParisState *state,
                                     const PGASParisRegimeStats *stats,
                                     const PGASParisRegimePrior *prior);

    /**
     * Enforce μ-ordering to prevent label switching
     *
     * After sampling, sorts regimes so μ_0 < μ_1 < ... < μ_{K-1}
     * Also permutes Π rows/columns accordingly.
     *
     * This is the ALIGN step in the Lifeboat protocol.
     *
     * @param state  PGAS-PARIS state
     */
    void pgas_paris_enforce_mu_ordering(PGASParisState *state);

    /**
     * Get learned μ_vol values
     *
     * @param state    PGAS-PARIS state
     * @param mu_out   Output array [K]
     * @param K        Number of regimes
     */
    void pgas_paris_get_mu_vol(const PGASParisState *state, float *mu_out, int K);

    /**
     * Get learned σ_vol values
     *
     * @param state      PGAS-PARIS state
     * @param sigma_out  Output array [K]
     * @param K          Number of regimes
     */
    void pgas_paris_get_sigma_vol(const PGASParisState *state, float *sigma_out, int K);

    /**
     * Sample σ_h from Inverse-Gamma posterior (shared AR innovation std)
     *
     * Model: h_t = μ_{z_t}(1-φ) + φ*h_{t-1} + σ_h*ε_t
     *
     * Posterior: σ²_h | data ~ InvGamma(a_post, b_post)
     * where:
     *   a_post = a₀ + (T-1)/2
     *   b_post = b₀ + 0.5 * Σ(h_t - μ_{z_t}(1-φ) - φ*h_{t-1})²
     *
     * @param state  PGAS-PARIS state
     * @param stats  Sufficient statistics
     * @param prior  Prior configuration
     */
    void pgas_paris_sample_sigma_h(PGASParisState *state,
                                   const PGASParisRegimeStats *stats,
                                   PGASParisRegimePrior *prior);

    /**
     * Sample φ via Metropolis-Hastings (AR persistence coefficient)
     *
     * Non-conjugate, so we use random walk MH on logit(φ) scale.
     * Prior: φ ~ Beta(a, b) on [phi_min, phi_max]
     *
     * Proposal: logit(φ') = logit(φ) + N(0, proposal_std²)
     * Accept with Metropolis ratio including Jacobian.
     *
     * @param state  PGAS-PARIS state
     * @param stats  Sufficient statistics
     * @param prior  Prior configuration (updated with MH diagnostics)
     */
    void pgas_paris_sample_phi(PGASParisState *state,
                               const PGASParisRegimeStats *stats,
                               PGASParisRegimePrior *prior);

    /**
     * Get learned σ_h value
     *
     * @param state  PGAS-PARIS state
     * @return       Current σ_h value
     */
    float pgas_paris_get_sigma_h(const PGASParisState *state);

    /**
     * Get learned φ value
     *
     * @param state  PGAS-PARIS state
     * @return       Current φ value
     */
    float pgas_paris_get_phi(const PGASParisState *state);

    /**
     * Full Gibbs sweep with regime parameter learning
     *
     * Extended sweep:
     *   1. CSMC forward pass
     *   2. PARIS backward smoothing
     *   3. Sample trajectories
     *   4. Count transitions → Sample Π
     *   5. Collect regime stats → Sample μ_vol (if enabled)
     *   6. Sample σ_vol (if enabled)
     *   7. Sample σ_h (if enabled)
     *   8. Sample φ via MH (if enabled)
     *   9. Enforce μ-ordering (if enabled)
     *
     * @param state  PGAS-PARIS state
     * @param prior  Regime learning prior (NULL to skip regime learning)
     * @return       CSMC acceptance rate
     */
    float pgas_paris_gibbs_sweep_full(PGASParisState *state,
                                      PGASParisRegimePrior *prior);

#ifdef __cplusplus
}
#endif

#endif /* PGAS_PARIS_H */