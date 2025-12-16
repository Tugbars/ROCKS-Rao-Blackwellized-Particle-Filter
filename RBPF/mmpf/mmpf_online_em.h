/**
 * @file mmpf_online_em.h
 * @brief Online Expectation-Maximization for Regime Discovery
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * THE PROBLEM: STATIC SWIM LANES
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Current regime definitions are hard-coded:
 *   Calm:   μ ∈ [-5.5, -3.5]
 *   Crisis: μ ∈ [-2.5, 0.0]
 *
 * These fail when:
 *   - Asset changes (BTC ≠ Eurodollar)
 *   - Market enters new secular era (2022 inflation)
 *   - Volatility profile drifts over time
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * THE SOLUTION: GAUSSIAN MIXTURE MODEL WITH ONLINE EM
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Model log-volatility as mixture of K Gaussians:
 *
 *   P(y) = Σ_k π_k × N(y | μ_k, σ_k²)
 *
 * Online EM learns {μ_k, σ_k², π_k} from streaming data:
 *
 * E-Step: Compute responsibility (soft assignment)
 *   γ_k ∝ π_k × N(y | μ_k, σ_k²)
 *
 * M-Step: Update parameters via stochastic approximation
 *   π_k ← (1-η)π_k + η×γ_k
 *   μ_k ← μ_k + η_k×(y - μ_k)
 *   σ_k² ← (1-η_k)σ_k² + η_k×(y - μ_k)²
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * THE ORDERING CONSTRAINT
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * EM doesn't care about labels. Cluster 0 might become "Crisis" if it drifts.
 * We enforce:
 *
 *   μ₀ < μ₁ < μ₂
 *
 * After each update, sort clusters by μ. This ensures:
 *   - Cluster 0 → Lowest μ → CALM
 *   - Cluster K-1 → Highest μ → CRISIS
 */

#ifndef MMPF_ONLINE_EM_H
#define MMPF_ONLINE_EM_H

#ifdef __cplusplus
extern "C" {
#endif

/*═══════════════════════════════════════════════════════════════════════════
 * CONFIGURATION
 *═══════════════════════════════════════════════════════════════════════════*/

#define MMPF_EM_MAX_CLUSTERS 8

/**
 * @brief Online EM state for regime discovery
 */
typedef struct {
    int n_clusters;           /**< Number of clusters (typically 3: Calm, Trend, Crisis) */
    
    /* Cluster parameters */
    double mu[MMPF_EM_MAX_CLUSTERS];    /**< Cluster means (log-volatility centers) */
    double var[MMPF_EM_MAX_CLUSTERS];   /**< Cluster variances */
    double pi[MMPF_EM_MAX_CLUSTERS];    /**< Cluster weights (mixing proportions) */
    
    /* Learning rate */
    double eta;               /**< Base learning rate (default: 0.001 ≈ 1000 tick memory) */
    
    /* Bounds */
    double min_var;           /**< Variance floor (prevents singularity) */
    double min_pi;            /**< Weight floor (prevents cluster death) */
    double min_mu;            /**< Minimum allowed μ */
    double max_mu;            /**< Maximum allowed μ */
    
    /* Statistics */
    int tick_count;           /**< Total observations processed */
    double last_responsibility[MMPF_EM_MAX_CLUSTERS];  /**< Last γ_k values */
} MMPF_OnlineEM;

/**
 * @brief Initialize Online EM with default parameters
 *
 * @param em          EM state to initialize
 * @param n_clusters  Number of clusters (2-8)
 *
 * Default initialization uses tuned values from Stage 1:
 *   Cluster 0 (Calm):   μ = -4.5
 *   Cluster 1 (Trend):  μ = -3.0
 *   Cluster 2 (Crisis): μ = -1.25
 */
void mmpf_online_em_init(MMPF_OnlineEM *em, int n_clusters);

/**
 * @brief Initialize with custom starting values
 *
 * @param em          EM state to initialize
 * @param n_clusters  Number of clusters
 * @param mu          Initial cluster means [n_clusters]
 * @param var         Initial cluster variances [n_clusters] (NULL for defaults)
 * @param pi          Initial cluster weights [n_clusters] (NULL for uniform)
 */
void mmpf_online_em_init_custom(MMPF_OnlineEM *em, int n_clusters,
                                 const double *mu, const double *var,
                                 const double *pi);

/**
 * @brief Update EM with new observation
 *
 * @param em          EM state
 * @param y_log_vol   Observed log-volatility (or proxy like weighted particle mean)
 *
 * This performs one step of online EM:
 * 1. E-step: Compute responsibilities γ_k
 * 2. M-step: Update parameters via stochastic approximation
 * 3. Enforce ordering constraint μ₀ < μ₁ < ... < μ_{K-1}
 */
void mmpf_online_em_update(MMPF_OnlineEM *em, double y_log_vol);

/**
 * @brief Get cluster assignment for observation
 *
 * @param em          EM state
 * @param y_log_vol   Observed log-volatility
 * @return            Index of most likely cluster (0 = Calm, K-1 = Crisis)
 */
int mmpf_online_em_classify(const MMPF_OnlineEM *em, double y_log_vol);

/**
 * @brief Get responsibilities for observation
 *
 * @param em          EM state
 * @param y_log_vol   Observed log-volatility
 * @param gamma       Output responsibilities [n_clusters]
 */
void mmpf_online_em_responsibilities(const MMPF_OnlineEM *em, double y_log_vol,
                                      double *gamma);

/**
 * @brief Set learning rate
 *
 * @param em   EM state
 * @param eta  Learning rate (0.0001 to 0.1)
 *
 * Lower η → slower adaptation, more stable
 * Higher η → faster adaptation, more responsive
 *
 * Effective memory ≈ 1/η ticks
 * Default η=0.001 gives ~1000 tick memory
 */
void mmpf_online_em_set_learning_rate(MMPF_OnlineEM *em, double eta);

/**
 * @brief Get cluster centers for MMPF hypothesis configuration
 *
 * @param em      EM state
 * @param mu_out  Output array of cluster means [n_clusters]
 */
void mmpf_online_em_get_centers(const MMPF_OnlineEM *em, double *mu_out);

/**
 * @brief Check if EM has converged (variances stable)
 *
 * @param em            EM state
 * @param var_threshold Maximum variance change considered stable
 * @return              1 if converged, 0 otherwise
 */
int mmpf_online_em_converged(const MMPF_OnlineEM *em, double var_threshold);

/**
 * @brief Reset EM to initial state
 */
void mmpf_online_em_reset(MMPF_OnlineEM *em);

/*═══════════════════════════════════════════════════════════════════════════
 * INTEGRATION HELPERS
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Apply learned centers to MMPF hypotheses
 *
 * Call after em_update to sync EM clusters with MMPF configuration.
 * This maps:
 *   em->mu[0] → hypotheses[MMPF_CALM].mu_vol
 *   em->mu[1] → hypotheses[MMPF_TREND].mu_vol
 *   em->mu[2] → hypotheses[MMPF_CRISIS].mu_vol
 */
/* void mmpf_online_em_apply_to_config(const MMPF_OnlineEM *em, MMPF_Config *cfg); */

#ifdef __cplusplus
}
#endif

#endif /* MMPF_ONLINE_EM_H */
