/**
 * @file rbpf_silverman.h
 * @brief Silverman's Rule of Thumb for Kernel Density Estimation Bandwidth
 *
 * Replaces fixed jitter scale with density-based bandwidth estimation.
 * This prevents particle collapse after resampling while avoiding over-smoothing.
 *
 * Silverman's Rule (1986):
 *   h = 0.9 × min(σ, IQR/1.34) × N^(-1/5)
 *
 * where:
 *   σ   = standard deviation of particle states
 *   IQR = interquartile range (Q3 - Q1), robust to outliers
 *   N   = number of particles
 *
 * The min(σ, IQR/1.34) provides robustness:
 *   - For Gaussian data: σ ≈ IQR/1.34, so they're equal
 *   - For heavy-tailed data: IQR/1.34 < σ, so IQR wins (more conservative)
 *   - For multimodal data: IQR/1.34 < σ, so IQR wins
 *
 * Usage in particle filter:
 *   After resampling, apply jitter:
 *     mu[i] += silverman_bandwidth(mu, n) * randn()
 */

#ifndef RBPF_SILVERMAN_H
#define RBPF_SILVERMAN_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Compute Silverman's bandwidth for kernel density estimation
 *
 * @param x         Particle states (log-volatility)
 * @param n         Number of particles
 * @param scratch   Scratch buffer of size n (for IQR computation)
 * @return          Optimal bandwidth h
 *
 * @note scratch buffer will be modified (partial sort for quartiles)
 */
double rbpf_silverman_bandwidth(const double *x, int n, double *scratch);

/**
 * @brief Float version for single-precision builds
 */
float rbpf_silverman_bandwidth_f(const float *x, int n, float *scratch);

/**
 * @brief Simplified Silverman (σ only, no IQR) - faster but less robust
 *
 * h = 1.06 × σ × N^(-1/5)
 *
 * Use when particles are approximately Gaussian (normal operation).
 * The full IQR version is better after shocks (heavy tails).
 */
double rbpf_silverman_bandwidth_simple(const double *x, int n);

/**
 * @brief Configuration for adaptive bandwidth selection
 */
typedef struct {
    int use_silverman;          /**< 1 = Silverman, 0 = fixed bandwidth */
    double fixed_bandwidth;     /**< Fallback fixed bandwidth */
    double min_bandwidth;       /**< Floor to prevent under-smoothing */
    double max_bandwidth;       /**< Ceiling to prevent over-smoothing */
    double ess_scale_min;       /**< Minimum ESS scaling factor */
    double ess_scale_max;       /**< Maximum ESS scaling factor */
} RBPF_BandwidthConfig;

/**
 * @brief Initialize bandwidth config with sensible defaults
 */
void rbpf_bandwidth_config_defaults(RBPF_BandwidthConfig *cfg);

/**
 * @brief Compute adaptive bandwidth considering ESS
 *
 * @param x         Particle states
 * @param n         Number of particles
 * @param ess       Current effective sample size
 * @param cfg       Configuration
 * @param scratch   Scratch buffer of size n
 * @return          Bandwidth to use for jitter
 *
 * Logic:
 *   1. Compute Silverman bandwidth from particle distribution
 *   2. Scale by ESS ratio (low ESS → more jitter)
 *   3. Clamp to [min_bandwidth, max_bandwidth]
 */
double rbpf_adaptive_bandwidth(const double *x, int n, double ess,
                               const RBPF_BandwidthConfig *cfg,
                               double *scratch);

#ifdef __cplusplus
}
#endif

#endif /* RBPF_SILVERMAN_H */
