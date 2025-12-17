/**
 * @file rbpf_fisher_rao.h
 * @brief Fisher-Rao Geodesic for Gaussian State Interpolation
 *
 * The space of univariate Gaussians N(μ, σ²) equipped with the Fisher
 * information metric is isometric to the hyperbolic half-plane H².
 *
 * Geodesics in this space are:
 *   - Vertical lines (when μ₁ = μ₂)
 *   - Semicircles centered on the μ-axis (general case)
 *
 * This provides the PRINCIPLED way to interpolate between Gaussian states,
 * replacing arbitrary blending heuristics like "70/30".
 *
 * Reference:
 *   Amari & Nagaoka (2000), "Methods of Information Geometry"
 *   Chapter 2: Geometry of Statistical Models
 */

#ifndef RBPF_FISHER_RAO_H
#define RBPF_FISHER_RAO_H

#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/*═══════════════════════════════════════════════════════════════════════════
 * TYPES
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Gaussian state in (μ, σ) parametrization
 * 
 * We use σ (std dev) not σ² (variance) because the Fisher metric
 * is more natural in (μ, σ) coordinates.
 */
typedef struct {
    double mu;     /**< Mean */
    double sigma;  /**< Standard deviation (NOT variance) */
} FisherRaoGaussian;

/*═══════════════════════════════════════════════════════════════════════════
 * GEODESIC COMPUTATION
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Compute point on Fisher-Rao geodesic between two Gaussians
 *
 * @param p1  Starting Gaussian (t=0)
 * @param p2  Ending Gaussian (t=1)
 * @param t   Interpolation parameter ∈ [0, 1]
 * @return    Gaussian at parameter t along geodesic
 *
 * Properties:
 *   - t=0 returns p1 exactly
 *   - t=1 returns p2 exactly
 *   - Path minimizes Fisher-Rao distance
 *   - Geodesic is a semicircle in (μ, σ) space (or vertical line if μ₁=μ₂)
 */
static inline FisherRaoGaussian fisher_rao_geodesic(
    FisherRaoGaussian p1, 
    FisherRaoGaussian p2, 
    double t)
{
    FisherRaoGaussian result;
    
    double mu1 = p1.mu, sigma1 = p1.sigma;
    double mu2 = p2.mu, sigma2 = p2.sigma;
    
    /* Clamp t to [0, 1] */
    if (t < 0.0) t = 0.0;
    if (t > 1.0) t = 1.0;
    
    /* Case 1: Vertical geodesic (same mean) */
    if (fabs(mu2 - mu1) < 1e-10) {
        result.mu = mu1;
        /* Geometric interpolation of sigma (exact geodesic) */
        result.sigma = pow(sigma1, 1.0 - t) * pow(sigma2, t);
        return result;
    }
    
    /* Case 2: Semicircle geodesic (different means) */
    
    /* Find center of semicircle on μ-axis */
    double c = (mu2*mu2 - mu1*mu1 + sigma2*sigma2 - sigma1*sigma1) / 
               (2.0 * (mu2 - mu1));
    
    /* Radius of semicircle */
    double r = sqrt((mu1 - c)*(mu1 - c) + sigma1*sigma1);
    
    /* Angles from center to each point */
    double theta1 = atan2(sigma1, mu1 - c);
    double theta2 = atan2(sigma2, mu2 - c);
    
    /* Linear interpolation of angle */
    double theta = (1.0 - t) * theta1 + t * theta2;
    
    /* Point on semicircle */
    result.mu = c + r * cos(theta);
    result.sigma = r * sin(theta);
    
    /* Safety: sigma must be positive */
    if (result.sigma < 1e-6) result.sigma = 1e-6;
    
    return result;
}

/*═══════════════════════════════════════════════════════════════════════════
 * FISHER-RAO DISTANCE
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Compute Fisher-Rao distance between two Gaussians
 *
 * The Fisher-Rao distance is the geodesic distance in information geometry.
 * For univariate Gaussians, it has a closed form via the hyperbolic metric.
 *
 * @param p1  First Gaussian
 * @param p2  Second Gaussian
 * @return    Fisher-Rao (geodesic) distance
 */
static inline double fisher_rao_distance(FisherRaoGaussian p1, FisherRaoGaussian p2)
{
    double mu1 = p1.mu, sigma1 = p1.sigma;
    double mu2 = p2.mu, sigma2 = p2.sigma;
    
    /* Hyperbolic distance formula for Poincaré half-plane:
     * d = arccosh(1 + (Δμ² + Δσ²) / (2σ₁σ₂))
     * 
     * Adjusted for Fisher metric scaling (factor of √2 on σ axis):
     * d = √2 × arccosh(1 + (Δμ²/2 + Δσ²) / (2σ₁σ₂))
     */
    double delta_mu = mu2 - mu1;
    double delta_sigma = sigma2 - sigma1;
    
    double arg = 1.0 + (delta_mu*delta_mu + 2.0*delta_sigma*delta_sigma) / 
                       (2.0 * sigma1 * sigma2);
    
    return sqrt(2.0) * acosh(arg);
}

/*═══════════════════════════════════════════════════════════════════════════
 * PRECISION-WEIGHTED BLEND PARAMETER
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Compute principled blend parameter t using precision weighting
 *
 * The blend parameter determines how far along the geodesic to move.
 * Using precision (inverse variance) weighting:
 *   - High particle precision → small t → stay near old state
 *   - Low particle precision → large t → move toward new regime
 *
 * @param var_particle    Current particle variance
 * @param var_stationary  Stationary variance of target regime: σ²/(2θ)
 * @return                Blend parameter t ∈ [0, 1]
 */
static inline double fisher_rao_blend_parameter(
    double var_particle, 
    double var_stationary)
{
    /* Precisions */
    double prec_particle = 1.0 / var_particle;
    double prec_regime = 1.0 / var_stationary;
    
    /* Precision-weighted blend: move toward higher precision */
    double t = prec_regime / (prec_particle + prec_regime);
    
    return t;
}

/*═══════════════════════════════════════════════════════════════════════════
 * CONVENIENCE: FULL MUTATION FUNCTION
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Mutate particle state toward new regime along Fisher-Rao geodesic
 *
 * This is the principled replacement for arbitrary "70/30" blending.
 *
 * @param mu_particle     Current particle mean (log-vol)
 * @param var_particle    Current particle variance
 * @param mu_regime       Target regime mean (log-vol)
 * @param theta_regime    Target regime mean-reversion speed
 * @param sigma_regime    Target regime vol-of-vol
 * @param mu_out          Output: new particle mean
 * @param var_out         Output: new particle variance
 */
static inline void fisher_rao_mutate(
    double mu_particle, double var_particle,
    double mu_regime, double theta_regime, double sigma_regime,
    double *mu_out, double *var_out)
{
    /* Stationary variance of target regime from OU process:
     * Var(ℓ_∞) = σ² / (2θ)
     */
    double var_stationary = (sigma_regime * sigma_regime) / (2.0 * theta_regime);
    
    /* Clamp to reasonable range */
    if (var_stationary < 0.01) var_stationary = 0.01;
    if (var_stationary > 10.0) var_stationary = 10.0;
    
    /* Compute blend parameter from precision weighting */
    double t = fisher_rao_blend_parameter(var_particle, var_stationary);
    
    /* Set up Gaussians in (μ, σ) form */
    FisherRaoGaussian p_old = {mu_particle, sqrt(var_particle)};
    FisherRaoGaussian p_new = {mu_regime, sqrt(var_stationary)};
    
    /* Compute point on geodesic */
    FisherRaoGaussian result = fisher_rao_geodesic(p_old, p_new, t);
    
    /* Output */
    *mu_out = result.mu;
    *var_out = result.sigma * result.sigma;
}

/*═══════════════════════════════════════════════════════════════════════════
 * DIAGNOSTICS
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Print geodesic interpolation for debugging
 */
static inline void fisher_rao_print_geodesic(
    FisherRaoGaussian p1, 
    FisherRaoGaussian p2,
    int n_steps)
{
    printf("Fisher-Rao Geodesic from N(%.3f, %.3f²) to N(%.3f, %.3f²):\n",
           p1.mu, p1.sigma, p2.mu, p2.sigma);
    printf("  Distance: %.4f\n", fisher_rao_distance(p1, p2));
    printf("  Path:\n");
    
    for (int i = 0; i <= n_steps; i++) {
        double t = (double)i / n_steps;
        FisherRaoGaussian p = fisher_rao_geodesic(p1, p2, t);
        printf("    t=%.2f: μ=%.4f, σ=%.4f\n", t, p.mu, p.sigma);
    }
}

#ifdef __cplusplus
}
#endif

#endif /* RBPF_FISHER_RAO_H */
