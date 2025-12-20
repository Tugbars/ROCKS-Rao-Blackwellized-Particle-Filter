/**
 * @file paris_fast.h
 * @brief High-performance PARIS backward smoother
 *
 * Optimizations:
 *   1. AVX2/AVX-512 vectorized backward kernel
 *   2. Structure-of-Arrays (SoA) memory layout
 *   3. Walker's Alias Method for O(1) sampling
 *   4. Single precision option (2x throughput)
 *   5. OpenMP task parallelism
 *
 * Performance target: <10ms for T=200, N=100 (vs 50ms baseline)
 */

#ifndef PARIS_FAST_H
#define PARIS_FAST_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/*═══════════════════════════════════════════════════════════════════════════════
 * CONFIGURATION
 *═══════════════════════════════════════════════════════════════════════════════*/

#ifndef PARIS_MAX_PARTICLES
#define PARIS_MAX_PARTICLES 128
#endif

#ifndef PARIS_MAX_TIME
#define PARIS_MAX_TIME 512
#endif

#ifndef PARIS_MAX_REGIMES
#define PARIS_MAX_REGIMES 8
#endif

/** Use single precision for 2x SIMD throughput */
#ifndef PARIS_USE_FLOAT
#define PARIS_USE_FLOAT 1
#endif

#if PARIS_USE_FLOAT
typedef float paris_real;
#define PARIS_ALIGN 32   /* AVX: 8 floats */
#else
typedef double paris_real;
#define PARIS_ALIGN 64   /* AVX-512: 8 doubles */
#endif

/*═══════════════════════════════════════════════════════════════════════════════
 * WALKER'S ALIAS TABLE
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * Alias table for O(1) categorical sampling
 * 
 * After O(N) setup, each sample is O(1):
 *   1. Pick random index i ∈ [0, N)
 *   2. Pick random u ∈ [0, 1)
 *   3. If u < prob[i], return i; else return alias[i]
 */
typedef struct {
    int N;
    paris_real prob[PARIS_MAX_PARTICLES] __attribute__((aligned(PARIS_ALIGN)));
    int alias[PARIS_MAX_PARTICLES];
} AliasTable;

/**
 * @brief Build alias table from weights (O(N))
 * @param table  Output alias table
 * @param weights  Unnormalized weights [N]
 * @param N  Number of elements
 */
void alias_build(AliasTable *table, const paris_real *weights, int N);

/**
 * @brief Sample from alias table (O(1))
 * @param table  Alias table
 * @param rng  RNG state (xoroshiro128+)
 * @return  Sampled index
 */
int alias_sample(const AliasTable *table, uint64_t *rng);

/*═══════════════════════════════════════════════════════════════════════════════
 * SOA PARTICLE STORAGE
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * Structure-of-Arrays particle storage for cache efficiency
 * 
 * All regimes at time t are contiguous, all h values contiguous.
 * Enables vectorized access patterns.
 */
typedef struct {
    int N;      /**< Particle count */
    int T;      /**< Time steps */
    int K;      /**< Regime count */
    
    /* SoA layout: [T][N] but stored as [T * N] for alignment */
    int *regimes;           /**< [T × N] regime indices */
    paris_real *h;          /**< [T × N] log-volatility */
    paris_real *weights;    /**< [T × N] normalized weights */
    paris_real *log_weights;/**< [T × N] log weights */
    int *ancestors;         /**< [T × N] ancestor indices */
    int *smoothed;          /**< [T × N] smoothed ancestor indices */
    
    /* Model parameters (row-major for cache efficiency) */
    paris_real *log_trans;  /**< [K × K] log transition matrix */
    paris_real *mu_vol;     /**< [K] emission means */
    paris_real phi;
    paris_real sigma_h;
    paris_real inv_sigma_h_sq;  /**< Precomputed 1/(σ_h²) */
    
    /* Observations */
    paris_real *observations;   /**< [T] */
    
    /* RNG */
    uint64_t rng_state[2];
    
    /* Alias tables (one per particle for parallel sampling) */
    AliasTable alias_tables[PARIS_MAX_PARTICLES];
    
    /* Workspace for vectorized operations */
    paris_real *bw_weights_workspace;  /**< [N] aligned buffer */
    
} PARISState;

/*═══════════════════════════════════════════════════════════════════════════════
 * INITIALIZATION
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Allocate and initialize PARIS state
 * @param N  Particle count
 * @param T  Max time steps
 * @param K  Regime count
 * @param seed  RNG seed
 * @return  Allocated state (caller must free with paris_free)
 */
PARISState *paris_alloc(int N, int T, int K, uint64_t seed);

/**
 * @brief Free PARIS state
 */
void paris_free(PARISState *state);

/**
 * @brief Set model parameters
 */
void paris_set_model(PARISState *state,
                     const double *trans,      /* [K×K] transition matrix */
                     const double *mu_vol,     /* [K] emission means */
                     double phi,
                     double sigma_h);

/**
 * @brief Load particle cloud from PGAS output
 */
void paris_load_particles(PARISState *state,
                          const int *regimes,      /* [T×N] */
                          const double *h,         /* [T×N] */
                          const double *weights,   /* [T×N] */
                          const int *ancestors,    /* [T×N] */
                          int T);

/*═══════════════════════════════════════════════════════════════════════════════
 * CORE OPERATIONS
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Run PARIS backward smoothing (vectorized + parallel)
 *
 * For each particle n at time T, traces back through ancestors
 * using the backward kernel with AVX2/AVX-512 acceleration.
 *
 * Complexity: O(N² × T) but with ~8x SIMD speedup + OpenMP parallelism
 *
 * @param state  PARIS state (must have loaded particles)
 */
void paris_backward_smooth_fast(PARISState *state);

/**
 * @brief Get smoothed particles at time t
 * @param state   PARIS state (after smoothing)
 * @param t       Time index
 * @param regimes Output: regime indices [N]
 * @param h       Output: log-vol values [N]
 */
void paris_get_smoothed(const PARISState *state,
                        int t,
                        int *regimes,
                        double *h);

/*═══════════════════════════════════════════════════════════════════════════════
 * VECTORIZED KERNELS
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Compute backward weights for all particles (AVX2)
 *
 * bw_weight[m] = w_t[m] × exp(log_trans[regime_m][regime_next] + log_h_trans)
 *
 * @param state      PARIS state
 * @param t          Time index (computing weights at time t)
 * @param regime_next Regime of particle at t+1
 * @param h_next     Log-vol of particle at t+1
 * @param bw_weights Output: backward weights [N]
 */
void paris_compute_bw_weights_avx2(const PARISState *state,
                                    int t,
                                    int regime_next,
                                    paris_real h_next,
                                    paris_real *bw_weights);

#ifdef __AVX512F__
/**
 * @brief Compute backward weights (AVX-512 version)
 */
void paris_compute_bw_weights_avx512(const PARISState *state,
                                      int t,
                                      int regime_next,
                                      paris_real h_next,
                                      paris_real *bw_weights);
#endif

/**
 * @brief Vectorized log-sum-exp with max subtraction
 * @param log_weights  Log weights [N]
 * @param N  Count
 * @param weights  Output: normalized weights [N]
 */
void paris_logsumexp_normalize_avx2(const paris_real *log_weights,
                                     int N,
                                     paris_real *weights);

#ifdef __cplusplus
}
#endif

#endif /* PARIS_FAST_H */
