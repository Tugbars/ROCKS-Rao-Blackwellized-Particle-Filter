/**
 * @file pgas_mkl.h
 * @brief MKL-optimized PGAS + PARIS implementation
 *
 * Uses Intel MKL for:
 *   - VML: vsExp, vsLn, vsSub, vsMul (vectorized math)
 *   - VSL: RNG (Mersenne Twister / SFMT)
 *   - Potential BLAS for larger operations
 *
 * Combined with:
 *   - AVX2/AVX-512 SIMD for custom kernels
 *   - OpenMP for particle-level parallelism
 *   - SoA memory layout with 64-byte alignment
 *
 * UPDATED: OCSN 10-component emission + Transition learning for oracle validation
 */

#ifndef PGAS_MKL_H
#define PGAS_MKL_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C"
{
#endif

    /*═══════════════════════════════════════════════════════════════════════════════
     * CONFIGURATION
     *═══════════════════════════════════════════════════════════════════════════════*/

#ifndef PGAS_MKL_MAX_PARTICLES
#define PGAS_MKL_MAX_PARTICLES 256
#endif

#ifndef PGAS_MKL_MAX_TIME
#define PGAS_MKL_MAX_TIME 512
#endif

#ifndef PGAS_MKL_MAX_REGIMES
#define PGAS_MKL_MAX_REGIMES 8
#endif

#ifndef PGAS_MKL_MAX_K
#define PGAS_MKL_MAX_K PGAS_MKL_MAX_REGIMES
#endif

#define PGAS_MKL_ALIGN 64 /**< AVX-512 / cache line alignment */

/**
 * Pad N to multiple of 16 for optimal SIMD (eliminates tail masking)
 * Example: N=100 → N_padded=112, N=128 → N_padded=128
 */
#define PGAS_MKL_SIMD_WIDTH 16
#define PGAS_MKL_PAD_N(n) (((n) + PGAS_MKL_SIMD_WIDTH - 1) & ~(PGAS_MKL_SIMD_WIDTH - 1))

    /*═══════════════════════════════════════════════════════════════════════════════
     * DATA STRUCTURES
     *═══════════════════════════════════════════════════════════════════════════════*/

    /**
     * MKL RNG stream wrapper
     */
    typedef struct
    {
        void *stream; /**< VSLStreamStatePtr */
        int brng;     /**< Basic RNG type (MT19937, SFMT, etc.) */
    } MKLRngStream;

    /**
     * Model parameters
     */
    typedef struct
    {
        int K;
        float trans[PGAS_MKL_MAX_REGIMES * PGAS_MKL_MAX_REGIMES];
        float log_trans[PGAS_MKL_MAX_REGIMES * PGAS_MKL_MAX_REGIMES];
        float log_trans_T[PGAS_MKL_MAX_REGIMES * PGAS_MKL_MAX_REGIMES]; /**< Transposed for column access */
        float mu_vol[PGAS_MKL_MAX_REGIMES];
        float sigma_vol[PGAS_MKL_MAX_REGIMES];
        float mu_shift[PGAS_MKL_MAX_REGIMES]; /**< mu_vol[k] * (1 - phi) for Rank-1 optimization */
        float phi;
        float sigma_h;
        float inv_sigma_h_sq;
        float neg_half_inv_sigma_h_sq; /**< -0.5 / sigma_h² precomputed */
    } PGASMKLModel;

    /**
     * Per-thread workspace for PARIS backward pass
     * Pre-allocated to avoid mkl_malloc in hot path
     */
    typedef struct
    {
        float *log_bw;    /**< [N_padded] backward log weights */
        float *bw;        /**< [N_padded] backward weights */
        float *workspace; /**< [N_padded] scratch space */
        float *cumsum;    /**< [N_padded] cumulative sum */
    } PGASThreadWorkspace;

    /**
     * SoA particle storage with MKL-friendly alignment
     */
    typedef struct
    {
        int N;        /**< Particle count (user-specified) */
        int N_padded; /**< Padded to multiple of 16 for SIMD */
        int T;        /**< Time steps */
        int K;        /**< Regime count */

        /* SoA arrays: contiguous [T * N_padded] with 64-byte alignment */
        int *regimes;       /**< [T × N_padded] */
        float *h;           /**< [T × N_padded] */
        float *weights;     /**< [T × N_padded] normalized */
        float *log_weights; /**< [T × N_padded] */
        int *ancestors;     /**< [T × N_padded] */
        int *smoothed;      /**< [T × N_padded] PARIS output */

        /* Observations */
        float *observations; /**< [T] */

        /* Reference trajectory (for CSMC) */
        int *ref_regimes;   /**< [T] */
        float *ref_h;       /**< [T] */
        int *ref_ancestors; /**< [T] */
        int ref_idx;        /**< Which particle is reference */

        /* Model */
        PGASMKLModel model;

        /* ═══════════════════════════════════════════════════════════════════
         * TRANSITION LEARNING (NEW)
         * Used for Gibbs sampling of transition matrix
         * ═══════════════════════════════════════════════════════════════════*/
        int n_trans[PGAS_MKL_MAX_K * PGAS_MKL_MAX_K]; /**< Transition counts from ref trajectory */
        float prior_alpha;                            /**< Symmetric Dirichlet prior (default: 1.0) */
        float sticky_kappa;                           /**< Self-transition bias for sticky prior (default: 10.0) */

        /* ═══════════════════════════════════════════════════════════════════
         * ADAPTIVE KAPPA (Optional)
         * Uses chatter ratio as feedback to auto-tune stickiness prior
         * ═══════════════════════════════════════════════════════════════════*/
        int adaptive_kappa_enabled; /**< 0=disabled (default), 1=enabled */
        float kappa_min;            /**< Lower bound (default: 20.0) */
        float kappa_max;            /**< Upper bound (default: 500.0) */
        float kappa_up_rate;        /**< (Legacy) Rate for increasing κ */
        float kappa_down_rate;      /**< (Legacy) Rate for decreasing κ */
        float last_chatter_ratio;   /**< EMA-smoothed chatter ratio */
        float rls_chatter_estimate; /**< RLS estimate of true chatter */
        float rls_variance;         /**< RLS estimation variance P */
        float rls_forgetting;       /**< RLS forgetting factor λ (default: 0.97) */
        int last_off_diag_count;    /**< Off-diagonal transitions in last sweep */
        int last_total_count;       /**< Total transitions in last sweep */

        /* MKL RNG - main stream */
        MKLRngStream rng;

/* Per-thread RNG streams for PARIS (avoid vslNewStream in hot path) */
#define PGAS_MKL_MAX_THREADS 64
        void *thread_rng_streams[PGAS_MKL_MAX_THREADS];
        int n_thread_streams;

        /* Per-thread workspaces for PARIS (avoid mkl_malloc in hot path) */
        PGASThreadWorkspace thread_ws[PGAS_MKL_MAX_THREADS];

        /* Workspace buffers (pre-allocated, N_padded size for SIMD) */
        float *ws_log_bw;  /**< [N_padded] backward log weights */
        float *ws_bw;      /**< [N_padded] backward weights */
        float *ws_uniform; /**< [N_padded] uniform random numbers */
        float *ws_normal;  /**< [N_padded] normal random numbers */
        int *ws_indices;   /**< [N_padded] sampled indices */
        float *ws_cumsum;  /**< [N_padded] cumulative sum for sampling */
        float *ws_ocsn;    /**< [11 * N_padded] OCSN batch workspace */

        /* Walker's Alias Table workspace for O(1) sampling */
        float *ws_alias_prob; /**< [N_padded] alias probability table */
        int *ws_alias_idx;    /**< [N_padded] alias index table */
        int *ws_alias_small;  /**< [N_padded] small stack for building */
        int *ws_alias_large;  /**< [N_padded] large stack for building */

        /* Diagnostics */
        int ancestor_proposals;
        int ancestor_accepts;
        float acceptance_rate;
        int total_sweeps;
        int current_sweep;

    } PGASMKLState;

    /**
     * Lifeboat packet output
     */
    typedef struct
    {
        int K;
        int N;
        int T;
        float trans[PGAS_MKL_MAX_REGIMES * PGAS_MKL_MAX_REGIMES];
        float mu_vol[PGAS_MKL_MAX_REGIMES];
        float sigma_vol[PGAS_MKL_MAX_REGIMES];
        float phi;
        float sigma_h;

        int *final_regimes;   /**< [N] at time T-1 */
        float *final_h;       /**< [N] at time T-1 */
        float *final_weights; /**< [N] uniform after smoothing */

        float ancestor_acceptance;
        int sweeps_used;
        double total_time_us;
    } LifeboatPacketMKL;

    /*═══════════════════════════════════════════════════════════════════════════════
     * INITIALIZATION
     *═══════════════════════════════════════════════════════════════════════════════*/

    /**
     * @brief Allocate and initialize MKL PGAS state
     * @param N     Particle count
     * @param T     Max time steps
     * @param K     Regime count
     * @param seed  RNG seed
     * @return      Allocated state (call pgas_mkl_free to release)
     */
    PGASMKLState *pgas_mkl_alloc(int N, int T, int K, uint32_t seed);

    /**
     * @brief Free MKL PGAS state
     */
    void pgas_mkl_free(PGASMKLState *state);

    /**
     * @brief Set model parameters
     */
    void pgas_mkl_set_model(PGASMKLState *state,
                            const double *trans,
                            const double *mu_vol,
                            const double *sigma_vol,
                            double phi,
                            double sigma_h);

    /**
     * @brief Set reference trajectory
     */
    void pgas_mkl_set_reference(PGASMKLState *state,
                                const int *regimes,
                                const double *h,
                                int T);

    /**
     * @brief Load observations
     */
    void pgas_mkl_load_observations(PGASMKLState *state,
                                    const double *observations,
                                    int T);

    /*═══════════════════════════════════════════════════════════════════════════════
     * PGAS OPERATIONS
     *═══════════════════════════════════════════════════════════════════════════════*/

    /**
     * @brief Run one CSMC sweep with MKL acceleration
     * @return Ancestor acceptance rate
     */
    float pgas_mkl_csmc_sweep(PGASMKLState *state);

    /**
     * @brief Run adaptive PGAS (min 3, max 10 sweeps)
     * @return 0=success, 1=still_mixing, 2=failed
     */
    int pgas_mkl_run_adaptive(PGASMKLState *state);

    /*═══════════════════════════════════════════════════════════════════════════════
     * TRANSITION LEARNING (NEW)
     *═══════════════════════════════════════════════════════════════════════════════*/

    /**
     * @brief Sample transition matrix from Dirichlet posterior
     *
     * Uses counts from reference trajectory + sticky prior.
     * π_i ~ Dirichlet(α + n_{i1}, ..., α + n_{iK} + κ·I(j=i))
     *
     * @param state  PGAS state with ref_regimes containing sampled trajectory
     */
    void pgas_mkl_sample_transitions(PGASMKLState *state);

    /**
     * @brief Full Gibbs sweep: CSMC for states + Dirichlet for transitions
     *
     * This is the main entry point for using PGAS as an oracle
     * to learn the optimal transition matrix from data.
     *
     * @param state  PGAS state
     * @return       Ancestor acceptance rate from CSMC sweep
     */
    float pgas_mkl_gibbs_sweep(PGASMKLState *state);

    /**
     * @brief Set transition learning priors
     *
     * @param state   PGAS state
     * @param alpha   Symmetric Dirichlet prior (default: 1.0)
     * @param kappa   Sticky self-transition bias (default: 10.0)
     */
    static inline void pgas_mkl_set_transition_prior(PGASMKLState *state,
                                                     float alpha, float kappa)
    {
        if (state)
        {
            state->prior_alpha = alpha;
            state->sticky_kappa = kappa;
        }
    }

    /*═══════════════════════════════════════════════════════════════════════════════
     * ADAPTIVE KAPPA
     *
     * Uses CHATTER-CORRECTED MOMENT MATCHING with RLS SMOOTHING.
     * Disabled by default — enable explicitly with pgas_mkl_enable_adaptive_kappa().
     *
     * The Problem: "Adaptive MCMC Death Spiral"
     *   - Naive moment-matching uses obs_diag from PGAS counts
     *   - But PGAS counts are BIASED by current κ
     *   - High κ → sticky trajectory → high obs_diag → higher κ (feedback!)
     *
     * The Solution: Use CHATTER RATIO as negative feedback
     *   - chatter = observed_switches / expected_switches
     *   - If chatter > 1: data fights prior → decrease κ
     *   - If chatter < 1: data calmer than prior → increase κ
     *
     * Smoothing: Recursive Least Squares (RLS) with forgetting factor
     *   - More principled than EMA
     *   - Adaptive gain based on estimation uncertainty
     *   - Tracks non-stationary signals optimally
     *
     * Reference: Ljung & Söderström, "Theory and Practice of RLS"
     *═══════════════════════════════════════════════════════════════════════════════*/

    /**
     * @brief Enable adaptive kappa adjustment
     *
     * When enabled, PGAS will adjust sticky_kappa after each Gibbs sweep
     * based on the observed chatter ratio.
     *
     * @param state   PGAS state
     * @param enable  1=enabled, 0=disabled
     */
    static inline void pgas_mkl_enable_adaptive_kappa(PGASMKLState *state, int enable)
    {
        if (state)
        {
            state->adaptive_kappa_enabled = enable;
        }
    }

    /**
     * @brief Configure adaptive kappa bounds
     *
     * Uses RLS-smoothed chatter-corrected moment matching.
     *
     * @param state         PGAS state
     * @param kappa_min     Lower bound for κ (default: 20.0)
     * @param kappa_max     Upper bound for κ (default: 500.0)
     * @param up_rate       (Legacy) Not used in current implementation
     * @param down_rate     (Legacy) Not used in current implementation
     */
    static inline void pgas_mkl_configure_adaptive_kappa(PGASMKLState *state,
                                                         float kappa_min,
                                                         float kappa_max,
                                                         float up_rate,
                                                         float down_rate)
    {
        if (state)
        {
            state->kappa_min = kappa_min;
            state->kappa_max = kappa_max;
            state->kappa_up_rate = up_rate;
            state->kappa_down_rate = down_rate;
        }
    }

    /**
     * @brief Set RLS forgetting factor for adaptive kappa
     *
     * λ = 0.95 → ~20 sweep effective window (faster adaptation)
     * λ = 0.97 → ~33 sweep effective window (default)
     * λ = 0.99 → ~100 sweep effective window (slower, more stable)
     *
     * @param state         PGAS state
     * @param forgetting    Forgetting factor λ ∈ (0.9, 1.0)
     */
    static inline void pgas_mkl_set_rls_forgetting(PGASMKLState *state,
                                                   float forgetting)
    {
        if (state && forgetting > 0.9f && forgetting < 1.0f)
        {
            state->rls_forgetting = forgetting;
        }
    }

    /**
     * @brief Get RLS estimation variance (for diagnostics)
     *
     * Low P → confident in chatter estimate
     * High P → uncertain, adapting faster
     */
    static inline float pgas_mkl_get_rls_variance(const PGASMKLState *state)
    {
        return state ? state->rls_variance : 0.0f;
    }

    /**
     * @brief Get current chatter ratio
     *
     * @param state  PGAS state
     * @return       Ratio of observed to expected transitions (1.0 = perfect)
     */
    static inline float pgas_mkl_get_chatter_ratio(const PGASMKLState *state)
    {
        return state ? state->last_chatter_ratio : 0.0f;
    }

    /**
     * @brief Get current sticky kappa
     *
     * @param state  PGAS state
     * @return       Current κ value
     */
    static inline float pgas_mkl_get_sticky_kappa(const PGASMKLState *state)
    {
        return state ? state->sticky_kappa : 0.0f;
    }

    /**
     * @brief Get learned transition matrix
     *
     * @param state      PGAS state
     * @param trans_out  Output array [K × K]
     * @param K          Number of regimes
     */
    static inline void pgas_mkl_get_transitions(const PGASMKLState *state,
                                                float *trans_out, int K)
    {
        if (!state || !trans_out)
            return;
        int K_copy = (K < state->K) ? K : state->K;
        for (int i = 0; i < K_copy; i++)
        {
            for (int j = 0; j < K_copy; j++)
            {
                trans_out[i * K + j] = state->model.trans[i * state->K + j];
            }
        }
    }

    /**
     * @brief Get transition counts from last trajectory
     *
     * @param state      PGAS state
     * @param counts_out Output array [K × K]
     * @param K          Number of regimes
     */
    static inline void pgas_mkl_get_transition_counts(const PGASMKLState *state,
                                                      int *counts_out, int K)
    {
        if (!state || !counts_out)
            return;
        int K_copy = (K < state->K) ? K : state->K;
        for (int i = 0; i < K_copy; i++)
        {
            for (int j = 0; j < K_copy; j++)
            {
                counts_out[i * K + j] = state->n_trans[i * state->K + j];
            }
        }
    }

    /*═══════════════════════════════════════════════════════════════════════════════
     * PARIS OPERATIONS (integrated with PGAS state)
     *═══════════════════════════════════════════════════════════════════════════════*/

    /**
     * @brief Run MKL-accelerated PARIS backward smoothing on PGAS state
     *
     * Uses:
     *   - vsExp for batch exponential
     *   - VSL RNG for sampling
     *   - OpenMP for particle parallelism
     */
    void pgas_paris_backward_smooth(PGASMKLState *state);

    /**
     * @brief Get smoothed particles at time t from PGAS state
     */
    void pgas_paris_get_smoothed(const PGASMKLState *state,
                                 int t,
                                 int *regimes,
                                 float *h);

    /*═══════════════════════════════════════════════════════════════════════════════
     * LIFEBOAT
     *═══════════════════════════════════════════════════════════════════════════════*/

    /**
     * @brief Generate Lifeboat packet from PGAS+PARIS output
     */
    void pgas_mkl_generate_lifeboat(const PGASMKLState *state,
                                    LifeboatPacketMKL *packet);

    /**
     * @brief Validate Lifeboat packet
     */
    bool lifeboat_mkl_validate(const LifeboatPacketMKL *packet);

    /*═══════════════════════════════════════════════════════════════════════════════
     * DIAGNOSTICS
     *═══════════════════════════════════════════════════════════════════════════════*/

    float pgas_mkl_get_acceptance_rate(const PGASMKLState *state);
    int pgas_mkl_get_sweep_count(const PGASMKLState *state);
    float pgas_mkl_get_ess(const PGASMKLState *state, int t);
    void pgas_mkl_print_diagnostics(const PGASMKLState *state);

#ifdef __cplusplus
}
#endif

#endif /* PGAS_MKL_H */