/**
 * @file rbpf_kl_tempering.h
 * @brief Information-Geometric Weight Normalization
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * THE INFORMATION BOTTLENECK
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Standard particle filter: w_new ∝ w_old × p(y|x)
 * 
 * Problem: A single extreme observation can "execute" 99% of particles,
 * collapsing ESS to 1 and destroying diversity. This is "Numerical Genocide."
 *
 * Solution: Tempered update with β ∈ (0, 1]:
 *   w_new ∝ w_old × p(y|x)^β
 *
 * β is computed from the KL divergence between proposed and current weights:
 *   - If KL < p95 (normal): β = 1.0 (full update)
 *   - If KL > p95 (shock): β = p95 / KL (continuous dampening)
 *   - If KL > log(N) (ceiling): β = log(N) / KL (hard clamp)
 *
 * The log(N) ceiling is the "Speed of Light" - a single observation cannot
 * provide more information than distinguishing among N particles.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * ZOMBIE PREVENTION
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * If β < 0.5 for too many consecutive ticks, the particles are "zombies" -
 * alive but representing an invalid state. We force a circuit breaker reset.
 *
 * Additionally, when β is low, we couple with Storvik by increasing λ
 * (forgetting factor) to prevent parameter stagnation.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * REFERENCES
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * - Grünwald & van Ommen (2017), "Inconsistency of Bayesian Inference for
 *   Misspecified Linear Models, and a Proposal for Repairing It"
 * - Bissiri et al. (2016), "A General Framework for Updating Belief
 *   Distributions" (Power Likelihood / SafeBayes)
 * - Amari & Nagaoka (2000), "Methods of Information Geometry"
 */

#ifndef RBPF_KL_TEMPERING_H
#define RBPF_KL_TEMPERING_H

#include "p2_quantile.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*═══════════════════════════════════════════════════════════════════════════
 * CONFIGURATION
 *═══════════════════════════════════════════════════════════════════════════*/

/** Minimum β to prevent complete likelihood rejection */
#define RBPF_KL_BETA_FLOOR          0.1f

/** Default P² quantile for soft dampening threshold */
#define RBPF_KL_DEFAULT_QUANTILE    0.95

/** Warmup ticks before P² threshold is trusted */
#define RBPF_KL_WARMUP_TICKS        500

/** Max consecutive damped ticks before zombie reset */
#define RBPF_KL_MAX_DAMPED_TICKS    10

/** β threshold for "heavily damped" state */
#define RBPF_KL_DAMPED_THRESHOLD    0.5f

/** Emergency λ when heavily damped (couples with Storvik) */
#define RBPF_KL_EMERGENCY_LAMBDA    0.99f

/*═══════════════════════════════════════════════════════════════════════════
 * KL STATE STRUCTURE
 *═══════════════════════════════════════════════════════════════════════════*/

typedef struct {
    /*───────────────────────────────────────────────────────────────────────
     * NORMALIZING CONSTANT TRACKING
     *───────────────────────────────────────────────────────────────────────*/
    float log_Z_old;                /**< Log normalizing constant from previous tick */
    
    /*───────────────────────────────────────────────────────────────────────
     * P² QUANTILE LEARNING
     *───────────────────────────────────────────────────────────────────────*/
    P2Quantile kl_quantile;         /**< Learns p95 of KL distribution */
    int warmup_complete;            /**< 1 after RBPF_KL_WARMUP_TICKS */
    
    /*───────────────────────────────────────────────────────────────────────
     * ZOMBIE PREVENTION STATE MACHINE
     *───────────────────────────────────────────────────────────────────────*/
    int consecutive_damped_ticks;   /**< Count of β < 0.5 ticks */
    int max_damped_before_reset;    /**< Threshold for zombie reset */
    
    /*───────────────────────────────────────────────────────────────────────
     * CONFIGURATION
     *───────────────────────────────────────────────────────────────────────*/
    int enabled;                    /**< 0 = disabled, use standard weights */
    float beta_floor;               /**< Minimum allowed β */
    float damped_threshold;         /**< β below this is "heavily damped" */
    float emergency_lambda;         /**< Storvik λ when heavily damped */
    
    /*───────────────────────────────────────────────────────────────────────
     * OUTPUT / DIAGNOSTICS
     *───────────────────────────────────────────────────────────────────────*/
    float last_kl;                  /**< KL from most recent tick */
    float last_beta;                /**< β applied on most recent tick */
    float kl_ceiling;               /**< log(N) - hard limit */
    float kl_p95;                   /**< Current P² estimate of p95 */
    
    /*───────────────────────────────────────────────────────────────────────
     * STATISTICS
     *───────────────────────────────────────────────────────────────────────*/
    uint64_t ticks_processed;       /**< Total ticks */
    uint64_t soft_damp_count;       /**< Ticks with β < 1 due to p95 */
    uint64_t hard_clamp_count;      /**< Ticks with β clamped to ceiling */
    uint64_t zombie_resets;         /**< Forced resets due to sustained β */
    float min_beta_seen;            /**< Lowest β ever applied */
    float max_kl_seen;              /**< Highest KL ever observed */
    
} RBPF_KL_State;

/*═══════════════════════════════════════════════════════════════════════════
 * LIFECYCLE
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Initialize KL tempering state
 *
 * @param state     State structure to initialize
 * @param n         Number of particles (for log(N) ceiling)
 */
void rbpf_kl_init(RBPF_KL_State *state, int n);

/**
 * @brief Reset KL state (e.g., after circuit breaker)
 *
 * Resets zombie counter and log_Z_old, but preserves learned P² quantile.
 */
void rbpf_kl_reset(RBPF_KL_State *state);

/**
 * @brief Full reset including P² quantile (e.g., new trading session)
 */
void rbpf_kl_reset_full(RBPF_KL_State *state, int n);

/*═══════════════════════════════════════════════════════════════════════════
 * CORE COMPUTATION
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Compute KL divergence between proposed and current weights
 *
 * KL(proposed || old) where proposed ∝ old × exp(log_lik)
 *
 * @param log_weight        Current log weights [n]
 * @param log_lik_increment Log-likelihood increments [n]
 * @param n                 Number of particles
 * @param log_Z_old         Log normalizing constant of current distribution
 * @return                  KL divergence in nats
 */
float rbpf_kl_compute(
    const float *log_weight,
    const float *log_lik_increment,
    int n,
    float log_Z_old);

/**
 * @brief Compute tempering factor β from proposed KL
 *
 * @param state         KL state (provides p95, ceiling)
 * @param proposed_kl   KL divergence from rbpf_kl_compute
 * @return              β ∈ [beta_floor, 1.0]
 */
float rbpf_kl_compute_beta(RBPF_KL_State *state, float proposed_kl);

/**
 * @brief Apply tempered weight update
 *
 * log_weight[i] += β × log_lik_increment[i]
 *
 * @param log_weight        Log weights to update [n]
 * @param log_lik_increment Log-likelihood increments [n]
 * @param beta              Tempering factor
 * @param n                 Number of particles
 */
void rbpf_kl_apply_tempered(
    float *log_weight,
    const float *log_lik_increment,
    float beta,
    int n);

/**
 * @brief Compute log normalizing constant
 *
 * log(Σ exp(log_weight[i]))
 *
 * @param log_weight    Log weights [n]
 * @param n             Number of particles
 * @return              Log normalizing constant
 */
float rbpf_kl_compute_log_Z(const float *log_weight, int n);

/*═══════════════════════════════════════════════════════════════════════════
 * FULL UPDATE STEP
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Result of KL-tempered weight update
 */
typedef struct {
    float beta;                 /**< Tempering factor applied */
    float kl;                   /**< Proposed KL divergence */
    int zombie_detected;        /**< 1 if zombie reset triggered */
    int heavily_damped;         /**< 1 if β < damped_threshold */
} RBPF_KL_Result;

/**
 * @brief Full KL-tempered weight update step
 *
 * This is the main entry point. It:
 * 1. Computes KL divergence
 * 2. Updates P² quantile
 * 3. Computes β with continuous dampening
 * 4. Checks zombie state
 * 5. Applies tempered weights
 * 6. Updates log_Z_old
 *
 * @param state                 KL state
 * @param log_weight            Log weights [n] (modified in place)
 * @param log_lik_increment     Log-likelihood increments [n]
 * @param n                     Number of particles
 * @param resampled             1 if resampling occurred this tick
 * @return                      Result structure with diagnostics
 */
RBPF_KL_Result rbpf_kl_step(
    RBPF_KL_State *state,
    float *log_weight,
    const float *log_lik_increment,
    int n,
    int resampled);

/*═══════════════════════════════════════════════════════════════════════════
 * AVX-512 OPTIMIZED KERNELS
 *═══════════════════════════════════════════════════════════════════════════*/

#if defined(__AVX512F__) && !defined(RBPF_KL_NO_AVX512)
#define RBPF_KL_USE_AVX512 1

/**
 * @brief AVX-512 KL divergence computation
 *
 * Requires n to be multiple of 16 and arrays to be 64-byte aligned.
 */
float rbpf_kl_compute_avx512(
    const float *log_weight,
    const float *log_lik_increment,
    int n,
    float log_Z_old);

/**
 * @brief AVX-512 tempered weight application
 */
void rbpf_kl_apply_tempered_avx512(
    float *log_weight,
    const float *log_lik_increment,
    float beta,
    int n);

/**
 * @brief AVX-512 log normalizing constant
 */
float rbpf_kl_compute_log_Z_avx512(const float *log_weight, int n);

#endif /* AVX512 */

/*═══════════════════════════════════════════════════════════════════════════
 * DIAGNOSTICS
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Print KL tempering diagnostics
 */
void rbpf_kl_print_diagnostics(const RBPF_KL_State *state);

#ifdef __cplusplus
}
#endif

#endif /* RBPF_KL_TEMPERING_H */
