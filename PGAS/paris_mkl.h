/**
 * @file paris_mkl.h
 * @brief Standalone MKL-optimized PARIS backward smoother
 *
 * MKL optimizations:
 *   - VML vsExp for batch exponential
 *   - VSL SFMT RNG for fast sampling
 *   - CBLAS isamax/sasum/sscal for vector operations
 *   - N_padded stride for full SIMD utilization
 *   - Pre-allocated per-thread RNG streams
 *   - 64-byte aligned memory
 *
 * Usage:
 *   PARISMKLState *paris = paris_mkl_alloc(N, T, K, seed);
 *   paris_mkl_set_model(paris, trans, mu_vol, phi, sigma_h);
 *   paris_mkl_load_particles(paris, regimes, h, weights, ancestors, T);
 *   paris_mkl_backward_smooth(paris);
 *   paris_mkl_free(paris);
 */

#ifndef PARIS_MKL_H
#define PARIS_MKL_H

#include <stdint.h>
#include <stdbool.h>

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

#define PARIS_MKL_MAX_THREADS 32
#define PARIS_MKL_ALIGN 64      /**< AVX-512 / cache line alignment */
#define PARIS_MKL_SIMD_WIDTH 16 /**< Pad to multiple of 16 floats */

/**
 * Recommended parameters for HFT SV-HMM:
 *   T (time):     128-200 ticks (recent history window)
 *   N (particles): 64-128 (64 min for ESS, 128 for stability)
 *   K (regimes):   4 (low/normal/high/extreme volatility)
 *
 * Powers of 2 recommended for optimal SIMD alignment.
 */
#define PARIS_MKL_RECOMMENDED_T 128
#define PARIS_MKL_RECOMMENDED_N 64
#define PARIS_MKL_RECOMMENDED_K 4

/**
 * Pad N to multiple of 16 for optimal SIMD (eliminates tail masking)
 */
#define PARIS_MKL_PAD_N(n) (((n) + PARIS_MKL_SIMD_WIDTH - 1) & ~(PARIS_MKL_SIMD_WIDTH - 1))

    /*═══════════════════════════════════════════════════════════════════════════════
     * DATA STRUCTURES
     *═══════════════════════════════════════════════════════════════════════════════*/

    /**
     * Model parameters for PARIS
     */
    typedef struct
    {
        int K; /**< Number of regimes */
        float trans[PARIS_MKL_MAX_REGIMES * PARIS_MKL_MAX_REGIMES];
        float log_trans[PARIS_MKL_MAX_REGIMES * PARIS_MKL_MAX_REGIMES];
        float mu_vol[PARIS_MKL_MAX_REGIMES];
        float mu_shifts[PARIS_MKL_MAX_REGIMES]; /**< Precomputed mu_k * (1 - phi) for Rank-1 opt */
        float phi;                              /**< AR(1) persistence */
        float sigma_h;                          /**< Volatility of h */
        float inv_sigma_h_sq;                   /**< Precomputed 1/σ_h² */
    } PARISMKLModel;

    /**
     * MKL-optimized PARIS state with SoA layout
     *
     * All T×N arrays use N_padded as stride for SIMD alignment
     */
    typedef struct
    {
        int N;        /**< Particle count (user-specified) */
        int N_padded; /**< Padded to multiple of 16 */
        int T;        /**< Time steps loaded */
        int K;        /**< Regime count */

        /* SoA particle storage [T × N_padded] */
        int *regimes;       /**< Regime indices */
        float *h;           /**< Log-volatility */
        float *log_weights; /**< Log weights from filtering */
        int *ancestors;     /**< Ancestor indices */
        int *smoothed;      /**< PARIS output: smoothed indices */

        /* Model */
        PARISMKLModel model;

        /* MKL RNG - main stream */
        void *rng_stream;

        /* Per-thread RNG streams (pre-allocated, reused) */
        void *thread_rng_streams[PARIS_MKL_MAX_THREADS];
        int n_thread_streams;

        /* Workspace buffers [N_padded] */
        float *ws_log_bw;    /**< Backward log weights */
        float *ws_bw;        /**< Backward weights (normalized) */
        float *ws_workspace; /**< Temp for logsumexp */
        float *ws_cumsum;    /**< Cumulative sum for sampling */
        float *ws_scaled_h;  /**< Pre-scaled h: phi * h_t (Rank-1 opt) */

        /* Pre-allocated per-thread workspaces (avoid malloc in hot path + false sharing) */
        float *thread_ws;     /**< [MAX_THREADS × 4 × (N_padded + 32)] with 128B padding */
        int thread_ws_stride; /**< Stride per thread (includes padding) */

    } PARISMKLState;

    /*═══════════════════════════════════════════════════════════════════════════════
     * LIFECYCLE
     *═══════════════════════════════════════════════════════════════════════════════*/

    /**
     * @brief Allocate MKL PARIS state
     * @param N     Number of particles
     * @param T     Maximum time steps
     * @param K     Number of regimes
     * @param seed  RNG seed
     * @return      Allocated state (call paris_mkl_free to release)
     */
    PARISMKLState *paris_mkl_alloc(int N, int T, int K, uint32_t seed);

    /**
     * @brief Free MKL PARIS state
     */
    void paris_mkl_free(PARISMKLState *state);

    /*═══════════════════════════════════════════════════════════════════════════════
     * CONFIGURATION
     *═══════════════════════════════════════════════════════════════════════════════*/

    /**
     * @brief Set model parameters
     * @param state     PARIS state
     * @param trans     Transition matrix [K×K] row-major (double for API compat)
     * @param mu_vol    Regime means [K]
     * @param phi       AR(1) persistence
     * @param sigma_h   Volatility of h process
     */
    void paris_mkl_set_model(PARISMKLState *state,
                             const double *trans,
                             const double *mu_vol,
                             double phi,
                             double sigma_h);

    /**
     * @brief Load particle data from filtering pass
     * @param state     PARIS state
     * @param regimes   Regime indices [T×N] (will be copied with padding)
     * @param h         Log-volatility [T×N] (double for API compat)
     * @param weights   Normalized weights [T×N] (converted to log)
     * @param ancestors Ancestor indices [T×N]
     * @param T         Number of time steps
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
     * @brief Run MKL-optimized PARIS backward smoothing
     *
     * Algorithm:
     *   For t = T-2 down to 0:
     *     For each particle n:
     *       - Get smoothed state at t+1
     *       - Compute backward weights using MKL vsExp
     *       - Sample ancestor using VSL RNG
     *
     * Complexity: O(T × N²) but with full SIMD utilization
     */
    void paris_mkl_backward_smooth(PARISMKLState *state);

    /**
     * @brief Get smoothed particle states at time t
     * @param state     PARIS state (after backward_smooth)
     * @param t         Time index
     * @param regimes   Output regime indices [N] (or NULL)
     * @param h         Output log-volatility [N] (or NULL)
     */
    void paris_mkl_get_smoothed(const PARISMKLState *state,
                                int t,
                                int *regimes,
                                float *h);

    /**
     * @brief Extract full smoothed trajectory for particle n
     * @param state     PARIS state
     * @param n         Particle index
     * @param regimes   Output regime trajectory [T] (or NULL)
     * @param h         Output h trajectory [T] (or NULL)
     */
    void paris_mkl_get_trajectory(const PARISMKLState *state,
                                  int n,
                                  int *regimes,
                                  float *h);

    /*═══════════════════════════════════════════════════════════════════════════════
     * DIAGNOSTICS
     *═══════════════════════════════════════════════════════════════════════════════*/

    /**
     * @brief Print PARIS diagnostics
     */
    void paris_mkl_print_info(const PARISMKLState *state);

#ifdef __cplusplus
}
#endif

#endif /* PARIS_MKL_H */