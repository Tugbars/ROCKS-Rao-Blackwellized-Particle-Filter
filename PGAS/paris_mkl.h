/**
 * @file paris_mkl.h
 * @brief Standalone MKL-optimized PARIS backward smoother
 *
 * ALIGNED WITH RBPF: Uses per-regime sigma_vol[k] for AR process noise.
 * This matches the RBPF model where crisis regimes have higher process noise.
 *
 * Key optimizations:
 *   1. N_padded stride eliminates SIMD tail masking
 *   2. Pre-allocated per-thread RNG streams
 *   3. Rank-1 log_trans column access pattern
 *   4. AVX2/AVX-512 vectorized backward kernels
 */

#ifndef PARIS_MKL_H
#define PARIS_MKL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

    /*═══════════════════════════════════════════════════════════════════════════════
     * CONFIGURATION
     *═══════════════════════════════════════════════════════════════════════════════*/

#ifndef PARIS_MKL_MAX_PARTICLES
#define PARIS_MKL_MAX_PARTICLES 256
#endif

#ifndef PARIS_MKL_MAX_TIME
#define PARIS_MKL_MAX_TIME 512
#endif

#ifndef PARIS_MKL_MAX_REGIMES
#define PARIS_MKL_MAX_REGIMES 8
#endif

#define PARIS_MKL_ALIGN 64 /**< AVX-512 / cache line alignment */

#define PARIS_MKL_SIMD_WIDTH 16
#define PARIS_MKL_PAD_N(n) (((n) + PARIS_MKL_SIMD_WIDTH - 1) & ~(PARIS_MKL_SIMD_WIDTH - 1))

#define PARIS_MKL_MAX_THREADS 64

    /*═══════════════════════════════════════════════════════════════════════════════
     * DATA STRUCTURES
     *═══════════════════════════════════════════════════════════════════════════════*/

    /**
     * Model parameters
     *
     * ALIGNED WITH RBPF: Uses per-regime sigma_vol[k] for AR process noise.
     * Crisis regimes have higher process noise, which is physically correct.
     *
     * AR dynamics: h_t = μ_k(1-φ) + φ*h_{t-1} + σ_vol[k]*ε_t
     */
    typedef struct
    {
        int K;
        float trans[PARIS_MKL_MAX_REGIMES * PARIS_MKL_MAX_REGIMES];
        float log_trans[PARIS_MKL_MAX_REGIMES * PARIS_MKL_MAX_REGIMES];
        float mu_vol[PARIS_MKL_MAX_REGIMES];
        float mu_shifts[PARIS_MKL_MAX_REGIMES]; /**< mu_vol[k] * (1 - phi) precomputed */
        float phi;

        /* Per-regime AR process noise (aligned with RBPF) */
        float sigma_vol[PARIS_MKL_MAX_REGIMES];
        float inv_sigma_vol_sq[PARIS_MKL_MAX_REGIMES];          /**< 1 / σ_vol[k]² */
        float neg_half_inv_sigma_vol_sq[PARIS_MKL_MAX_REGIMES]; /**< -0.5 / σ_vol[k]² */
    } PARISMKLModel;

    /**
     * PARIS state with MKL-friendly memory layout
     */
    typedef struct
    {
        int N;        /**< Particle count (user-specified) */
        int N_padded; /**< Padded to multiple of 16 for SIMD */
        int T;        /**< Time steps */
        int K;        /**< Regime count */

        /* SoA arrays: [T × N_padded] with 64-byte alignment */
        int *regimes;       /**< [T × N_padded] */
        float *h;           /**< [T × N_padded] */
        float *log_weights; /**< [T × N_padded] */
        int *ancestors;     /**< [T × N_padded] */
        int *smoothed;      /**< [T × N_padded] PARIS output */

        /* Model */
        PARISMKLModel model;

        /* Main RNG stream */
        void *rng_stream;

        /* Per-thread RNG streams */
        void *thread_rng_streams[PARIS_MKL_MAX_THREADS];
        int n_thread_streams;

        /* Per-thread workspace */
        float *thread_ws;
        int thread_ws_stride;

        /* Workspace buffers */
        float *ws_log_bw;
        float *ws_bw;
        float *ws_workspace;
        float *ws_cumsum;
        float *ws_scaled_h;

    } PARISMKLState;

    /*═══════════════════════════════════════════════════════════════════════════════
     * LIFECYCLE
     *═══════════════════════════════════════════════════════════════════════════════*/

    /**
     * @brief Allocate PARIS state
     */
    PARISMKLState *paris_mkl_alloc(int N, int T, int K, uint32_t seed);

    /**
     * @brief Free PARIS state
     */
    void paris_mkl_free(PARISMKLState *state);

    /*═══════════════════════════════════════════════════════════════════════════════
     * CONFIGURATION
     *═══════════════════════════════════════════════════════════════════════════════*/

    /**
     * @brief Set model parameters
     *
     * ALIGNED WITH RBPF: Uses per-regime sigma_vol[k] for AR process noise.
     *
     * @param state      PARIS state
     * @param trans      Transition matrix [K×K]
     * @param mu_vol     Per-regime log-vol means [K]
     * @param sigma_vol  Per-regime AR process noise [K]
     * @param phi        AR persistence (shared across regimes)
     */
    void paris_mkl_set_model(PARISMKLState *state,
                             const double *trans,
                             const double *mu_vol,
                             const double *sigma_vol,
                             double phi);

    /**
     * @brief Load particles from forward pass
     */
    void paris_mkl_load_particles(PARISMKLState *state,
                                  const int *regimes,
                                  const double *h,
                                  const double *weights,
                                  const int *ancestors,
                                  int T);

    /*═══════════════════════════════════════════════════════════════════════════════
     * BACKWARD SMOOTHING
     *═══════════════════════════════════════════════════════════════════════════════*/

    /**
     * @brief Run backward smoothing pass
     *
     * Uses per-regime sigma_vol[k] for AR transition likelihood.
     */
    void paris_mkl_backward_smooth(PARISMKLState *state);

    /*═══════════════════════════════════════════════════════════════════════════════
     * OUTPUT EXTRACTION
     *═══════════════════════════════════════════════════════════════════════════════*/

    /**
     * @brief Get smoothed particles at time t
     */
    void paris_mkl_get_smoothed(const PARISMKLState *state, int t,
                                int *regimes, float *h);

    /**
     * @brief Get full smoothed trajectory for particle n
     */
    void paris_mkl_get_trajectory(const PARISMKLState *state, int n,
                                  int *regimes, float *h);

    /*═══════════════════════════════════════════════════════════════════════════════
     * SCOUT SWEEP (Pre-validation before triggering PGAS)
     *
     * Quick PARIS sweep to check if the filter is degenerate before committing
     * to a full PGAS run. This prevents wasting compute on a broken filter.
     *
     * Flow:
     *   1. Hawkes/KL triggers suspicious
     *   2. Run scout sweep (few backward passes, ~5ms)
     *   3. If scout INVALID → force PGAS (can't trust filter)
     *   4. If scout VALID + low entropy → filter confident, skip PGAS
     *   5. If scout VALID + high entropy → filter uncertain, run PGAS
     *═══════════════════════════════════════════════════════════════════════════════*/

    /**
     * Scout sweep result
     */
    typedef struct
    {
        float entropy;         /**< Path entropy (lower = more confident) */
        float acceptance_rate; /**< Fraction of proposals accepted */
        int unique_paths;      /**< Number of distinct paths after sweeps */
        int total_proposals;   /**< Total ancestor proposals made */
        int total_accepts;     /**< Total proposals accepted */
        int sweeps_run;        /**< Number of sweeps completed */
        bool is_valid;         /**< True if scout mixing is adequate */
    } PARISScoutResult;

    /**
     * Scout sweep configuration
     */
    typedef struct
    {
        int n_sweeps;              /**< Number of backward sweeps (default: 5) */
        float min_acceptance;      /**< Minimum acceptance rate (default: 0.10) */
        float min_unique_fraction; /**< Minimum unique paths / N (default: 0.25) */
    } PARISScoutConfig;

    /**
     * @brief Get default scout configuration
     */
    PARISScoutConfig paris_mkl_scout_config_defaults(void);

    /**
     * @brief Run scout sweep with validation
     *
     * Runs a few backward sweeps and checks if the sampler is mixing properly.
     * Use this to decide whether to trigger full PGAS.
     *
     * @param state   PARIS state (must have particles loaded from forward pass)
     * @param config  Scout configuration (NULL for defaults)
     * @return        Scout result with validity flag
     */
    PARISScoutResult paris_mkl_scout_sweep(PARISMKLState *state,
                                           const PARISScoutConfig *config);

    /**
     * @brief Compute path entropy from smoothed trajectories
     *
     * Lower entropy = more agreement among particles = higher confidence
     *
     * @param state   PARIS state after backward smoothing
     * @return        Path entropy in nats
     */
    float paris_mkl_compute_path_entropy(const PARISMKLState *state);

    /**
     * @brief Count unique paths in smoothed trajectories
     *
     * @param state   PARIS state after backward smoothing
     * @return        Number of distinct regime paths
     */
    int paris_mkl_count_unique_paths(const PARISMKLState *state);

    /*═══════════════════════════════════════════════════════════════════════════════
     * DIAGNOSTICS
     *═══════════════════════════════════════════════════════════════════════════════*/

    void paris_mkl_print_info(const PARISMKLState *state);

#ifdef __cplusplus
}
#endif

#endif /* PARIS_MKL_H */