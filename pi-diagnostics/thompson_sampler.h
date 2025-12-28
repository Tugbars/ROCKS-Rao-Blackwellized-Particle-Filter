/**
 * @file thompson_sampler.h
 * @brief Thompson Sampling for Π (Variance Shot)
 *
 * When RBPF is critical and Oracle confidence is low, we don't just
 * inject the Oracle's point estimate. We sample from the posterior
 * to inject variance (exploration) rather than committing to potentially
 * wrong point estimate.
 *
 * Π rows are Dirichlet-distributed: Π_i ~ Dir(Q_i + α)
 * where Q_i are sufficient statistics and α is prior.
 */

#ifndef THOMPSON_SAMPLER_H
#define THOMPSON_SAMPLER_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/*═══════════════════════════════════════════════════════════════════════════
 * RNG STATE (xoroshiro128+)
 *═══════════════════════════════════════════════════════════════════════════*/

typedef struct {
    uint64_t s[2];
} ThompsonRNG;

/*═══════════════════════════════════════════════════════════════════════════
 * API - RNG
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Initialize RNG with seed
 */
void thompson_rng_init(ThompsonRNG *rng, uint64_t seed);

/**
 * Generate uniform [0,1)
 */
double thompson_rng_uniform(ThompsonRNG *rng);

/**
 * Generate standard normal (Box-Muller)
 */
double thompson_rng_normal(ThompsonRNG *rng);

/**
 * Generate Gamma(shape, 1.0) using Marsaglia-Tsang
 */
double thompson_rng_gamma(ThompsonRNG *rng, double shape);

/*═══════════════════════════════════════════════════════════════════════════
 * API - THOMPSON SAMPLING
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Sample Π from Dirichlet posterior
 * 
 * Each row Π_i is sampled from Dir(Q_i + prior_alpha)
 * 
 * @param Q            Sufficient statistics (K×K)
 * @param prior_alpha  Prior pseudo-count per cell (default: 1.0)
 * @param K            Number of regimes
 * @param Pi_out       Output: Sampled Π (K×K)
 * @param rng          RNG state
 */
void thompson_sample_pi(
    const float *Q,
    float prior_alpha,
    int K,
    float *Pi_out,
    ThompsonRNG *rng);

/**
 * Sample Π row from Dirichlet
 * 
 * @param alpha   Dirichlet parameters (length K)
 * @param K       Number of elements
 * @param out     Output: Sampled probabilities (length K)
 * @param rng     RNG state
 */
void thompson_sample_dirichlet(
    const float *alpha,
    int K,
    float *out,
    ThompsonRNG *rng);

/**
 * Compute mean Π from Q (equivalent to posterior mean)
 * 
 * @param Q            Sufficient statistics (K×K)
 * @param prior_alpha  Prior pseudo-count
 * @param K            Number of regimes
 * @param Pi_out       Output: Mean Π
 */
void thompson_mean_pi(
    const float *Q,
    float prior_alpha,
    int K,
    float *Pi_out);

/**
 * Blend between mean and sample based on confidence
 * 
 * High confidence → use mean
 * Low confidence  → use sample (variance shot)
 * 
 * @param Q            Sufficient statistics
 * @param prior_alpha  Prior
 * @param K            Number of regimes
 * @param confidence   0-1 (0 = pure sample, 1 = pure mean)
 * @param Pi_out       Output: Blended Π
 * @param rng          RNG state
 */
void thompson_sample_with_confidence(
    const float *Q,
    float prior_alpha,
    int K,
    float confidence,
    float *Pi_out,
    ThompsonRNG *rng);

#ifdef __cplusplus
}
#endif

#endif /* THOMPSON_SAMPLER_H */
