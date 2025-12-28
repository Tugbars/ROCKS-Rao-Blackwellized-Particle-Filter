/**
 * @file pi_divergence.h
 * @brief Three-Way Π Divergence Monitoring
 *
 * Monitors divergences between three Π matrices:
 *
 *   Π_oracle    - What PGAS suggests (ground truth)
 *   Π_storvik   - What RBPF is learning online (from counts)
 *   Π_operating - What RBPF uses for predictions (current assumption)
 *
 *              Π_oracle
 *                 /\
 *                /  \
 *          D₁   /    \   D₂
 *              /      \
 *             /        \
 *      Π_storvik ──── Π_operating
 *                 D₃
 *
 * D₁ = KL(Π_oracle || Π_storvik)    - Oracle vs RBPF's learning
 * D₂ = KL(Π_oracle || Π_operating)  - Oracle vs RBPF's assumption (innovation)
 * D₃ = KL(Π_storvik || Π_operating) - RBPF contradicting itself!
 *
 * D₃ is the hidden gold: pure RBPF internal signal, no Oracle needed!
 */

#ifndef PI_DIVERGENCE_H
#define PI_DIVERGENCE_H

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/*═══════════════════════════════════════════════════════════════════════════
 * DIVERGENCE TRIAD
 *═══════════════════════════════════════════════════════════════════════════*/

typedef struct {
    /* The three divergences */
    double d1_oracle_vs_storvik;     /* Do Oracle and Storvik agree? */
    double d2_oracle_vs_operating;   /* Does Oracle differ from current? */
    double d3_storvik_vs_operating;  /* Is RBPF self-contradicting? */
    
    /* Validity flags */
    bool oracle_available;           /* Is Oracle Π available? */
    bool d3_valid;                   /* D₃ always computable if RBPF running */
    
} DivergenceTriad;

/*═══════════════════════════════════════════════════════════════════════════
 * SCENARIO CLASSIFICATION
 *═══════════════════════════════════════════════════════════════════════════*/

typedef enum {
    /* All agree - no action needed */
    DIV_SCENARIO_HEALTHY = 0,
    
    /* D₃ high: RBPF internal conflict */
    DIV_SCENARIO_SELF_CONTRADICTION,
    
    /* D₃ high + D₁ low: Oracle and Storvik agree against Operating */
    DIV_SCENARIO_STRONG_AGREEMENT,
    
    /* D₂ high, D₃ low: Oracle sees something new */
    DIV_SCENARIO_ORACLE_INNOVATION,
    
    /* D₁ high, D₂ high, D₃ low: Three-way disagreement */
    DIV_SCENARIO_CONFUSION,
    
} DivergenceScenario;

/*═══════════════════════════════════════════════════════════════════════════
 * THRESHOLDS
 *═══════════════════════════════════════════════════════════════════════════*/

typedef struct {
    double d_agree;          /* Below this = "agree" (default: 0.05) */
    double d_disagree;       /* Above this = "disagree" (default: 0.15) */
    double d3_danger;        /* D₃ above this = self-contradiction (default: 0.10) */
    
} DivergenceThresholds;

/*═══════════════════════════════════════════════════════════════════════════
 * API
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Get default thresholds
 */
DivergenceThresholds divergence_thresholds_defaults(void);

/**
 * Compute all three divergences
 * 
 * @param Pi_oracle     From PGAS (can be NULL if not available)
 * @param Pi_storvik    From Storvik counts (row-normalized)
 * @param Pi_operating  Currently used by RBPF
 * @param K             Number of regimes
 * @return              DivergenceTriad with all divergences
 */
DivergenceTriad divergence_compute(
    const float *Pi_oracle,
    const float *Pi_storvik,
    const float *Pi_operating,
    int K);

/**
 * Classify the divergence scenario
 */
DivergenceScenario divergence_classify(
    const DivergenceTriad *div,
    const DivergenceThresholds *thresh);

/**
 * Get scenario description string
 */
const char *divergence_scenario_str(DivergenceScenario scenario);

/**
 * Compute symmetric KL divergence (average of both directions)
 */
double divergence_symmetric_kl(
    const float *P,
    const float *Q,
    int K);

/**
 * Compute Frobenius norm of difference (for sanity check)
 */
double divergence_frobenius(
    const float *P,
    const float *Q,
    int K);

#ifdef __cplusplus
}
#endif

#endif /* PI_DIVERGENCE_H */
