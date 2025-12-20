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
        float mu_vol[PGAS_MKL_MAX_REGIMES];
        float sigma_vol[PGAS_MKL_MAX_REGIMES];
        float phi;
        float sigma_h;
        float inv_sigma_h_sq;
    } PGASMKLModel;

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

        /* MKL RNG - main stream */
        MKLRngStream rng;

/* Per-thread RNG streams for PARIS (avoid vslNewStream in hot path) */
#define PGAS_MKL_MAX_THREADS 32
        void *thread_rng_streams[PGAS_MKL_MAX_THREADS];
        int n_thread_streams;

        /* Workspace buffers (pre-allocated, N_padded size for SIMD) */
        float *ws_log_bw;  /**< [N_padded] backward log weights */
        float *ws_bw;      /**< [N_padded] backward weights */
        float *ws_uniform; /**< [N_padded] uniform random numbers */
        float *ws_normal;  /**< [N_padded] normal random numbers */
        int *ws_indices;   /**< [N_padded] sampled indices */
        float *ws_cumsum;  /**< [N_padded] cumulative sum for sampling */

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