/**
 * @file thompson_sampler.h
 * @brief Thompson Sampling for Transition Matrix Handoff
 *
 * When handing off Π from SAEM to RBPF, we have a posterior distribution
 * over transition matrices represented by sufficient statistics Q.
 *
 * The posterior for each row i is: Π[i,:] ~ Dirichlet(Q[i,:])
 *
 * Thompson Sampling Strategy:
 *   - EXPLORE: When row_sum < threshold, sample Π ~ Dirichlet(Q)
 *   - EXPLOIT: When row_sum >= threshold, use mean Π = Q / sum(Q)
 *
 * This balances exploration (uncertainty in low-data regimes) with
 * exploitation (confidence in high-data regimes).
 *
 * Reference: ORACLE_INTEGRATION_PLAN.md
 */

#ifndef THOMPSON_SAMPLER_H
#define THOMPSON_SAMPLER_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*═══════════════════════════════════════════════════════════════════════════
 * CONFIGURATION
 *═══════════════════════════════════════════════════════════════════════════*/

#define THOMPSON_MAX_REGIMES 8

/**
 * Thompson Sampler configuration
 */
typedef struct {
    int n_regimes;                 /* Number of regimes (K) */
    
    /* Explore/Exploit threshold */
    float exploit_threshold;       /* Row sum above which we exploit (default: 500) */
    
    /* Sampling parameters */
    float min_concentration;       /* Minimum Dirichlet concentration (default: 0.1) */
    float floor_probability;       /* Minimum probability in sampled Π (default: 1e-5) */
    
    /* RNG seed */
    uint64_t seed;
    
} ThompsonSamplerConfig;

/*═══════════════════════════════════════════════════════════════════════════
 * STATE
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Thompson Sampler state
 */
typedef struct {
    ThompsonSamplerConfig config;
    
    /* RNG state (xoroshiro128+) */
    uint64_t rng_state[2];
    
    /* Statistics */
    int total_samples;
    int explore_count;             /* Times we sampled (explored) */
    int exploit_count;             /* Times we used mean (exploited) */
    
    bool initialized;
    
} ThompsonSampler;

/**
 * Result of a sample operation
 */
typedef struct {
    bool explored;                 /* True if we sampled, false if we used mean */
    float row_sums[THOMPSON_MAX_REGIMES];  /* Row sums for each regime */
    float min_row_sum;             /* Minimum row sum (determines exploration) */
    float max_row_sum;             /* Maximum row sum */
} ThompsonSampleResult;

/*═══════════════════════════════════════════════════════════════════════════
 * API - LIFECYCLE
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Get default configuration
 */
ThompsonSamplerConfig thompson_sampler_config_defaults(int n_regimes);

/**
 * Initialize sampler
 */
int thompson_sampler_init(ThompsonSampler *sampler, const ThompsonSamplerConfig *config);

/**
 * Reset state (keep config)
 */
void thompson_sampler_reset(ThompsonSampler *sampler);

/**
 * Free resources
 */
void thompson_sampler_free(ThompsonSampler *sampler);

/*═══════════════════════════════════════════════════════════════════════════
 * API - CORE OPERATIONS
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Sample or compute mean transition matrix from sufficient statistics
 *
 * This is the main entry point. Given Q (sufficient statistics from SAEM),
 * either samples Π ~ Dirichlet(Q) or computes mean Π = Q/sum(Q) based on
 * the explore/exploit threshold.
 *
 * @param sampler   Sampler state
 * @param Q         Sufficient statistics [K×K] row-major
 * @param Pi_out    Output transition matrix [K×K] row-major
 * @return          Sample result with diagnostics
 */
ThompsonSampleResult thompson_sampler_sample(
    ThompsonSampler *sampler,
    const float Q[THOMPSON_MAX_REGIMES][THOMPSON_MAX_REGIMES],
    float Pi_out[THOMPSON_MAX_REGIMES][THOMPSON_MAX_REGIMES]);

/**
 * Force exploration (always sample from Dirichlet)
 */
ThompsonSampleResult thompson_sampler_explore(
    ThompsonSampler *sampler,
    const float Q[THOMPSON_MAX_REGIMES][THOMPSON_MAX_REGIMES],
    float Pi_out[THOMPSON_MAX_REGIMES][THOMPSON_MAX_REGIMES]);

/**
 * Force exploitation (always use mean)
 */
ThompsonSampleResult thompson_sampler_exploit(
    ThompsonSampler *sampler,
    const float Q[THOMPSON_MAX_REGIMES][THOMPSON_MAX_REGIMES],
    float Pi_out[THOMPSON_MAX_REGIMES][THOMPSON_MAX_REGIMES]);

/*═══════════════════════════════════════════════════════════════════════════
 * API - DIRICHLET SAMPLING
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Sample a single row from Dirichlet distribution
 *
 * @param sampler   Sampler state (for RNG)
 * @param alpha     Dirichlet concentration parameters [K]
 * @param K         Number of categories
 * @param out       Output sample [K] (sums to 1)
 */
void thompson_sampler_dirichlet(
    ThompsonSampler *sampler,
    const float *alpha,
    int K,
    float *out);

/**
 * Sample from Gamma distribution (helper for Dirichlet)
 *
 * @param sampler   Sampler state (for RNG)
 * @param shape     Shape parameter (alpha > 0)
 * @param scale     Scale parameter (beta > 0, typically 1.0)
 * @return          Gamma sample
 */
float thompson_sampler_gamma(ThompsonSampler *sampler, float shape, float scale);

/*═══════════════════════════════════════════════════════════════════════════
 * API - QUERIES
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Get explore/exploit ratio
 */
float thompson_sampler_get_explore_ratio(const ThompsonSampler *sampler);

/**
 * Get total samples
 */
int thompson_sampler_get_total_samples(const ThompsonSampler *sampler);

/**
 * Check if row would explore (without actually sampling)
 */
bool thompson_sampler_would_explore(const ThompsonSampler *sampler, float row_sum);

/*═══════════════════════════════════════════════════════════════════════════
 * API - DIAGNOSTICS
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Print sampler state
 */
void thompson_sampler_print_state(const ThompsonSampler *sampler);

/**
 * Print configuration
 */
void thompson_sampler_print_config(const ThompsonSamplerConfig *config);

#ifdef __cplusplus
}
#endif

#endif /* THOMPSON_SAMPLER_H */
