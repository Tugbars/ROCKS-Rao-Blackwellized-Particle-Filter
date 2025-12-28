/**
 * @file pi_divergence.c
 * @brief Three-Way Π Divergence Implementation
 */

#include "pi_divergence.h"
#include <math.h>
#include <string.h>

/*═══════════════════════════════════════════════════════════════════════════
 * CONSTANTS
 *═══════════════════════════════════════════════════════════════════════════*/

#define MIN_PROB 1e-10
#define KL_CAP   10.0

/*═══════════════════════════════════════════════════════════════════════════
 * DEFAULTS
 *═══════════════════════════════════════════════════════════════════════════*/

DivergenceThresholds divergence_thresholds_defaults(void) {
    DivergenceThresholds t;
    t.d_agree = 0.05;
    t.d_disagree = 0.15;
    t.d3_danger = 0.10;
    return t;
}

/*═══════════════════════════════════════════════════════════════════════════
 * KL DIVERGENCE
 *═══════════════════════════════════════════════════════════════════════════*/

static double kl_divergence_matrix(const float *P, const float *Q, int K) {
    if (!P || !Q || K <= 0) return 0.0;
    
    double total_kl = 0.0;
    
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < K; j++) {
            double p = P[i * K + j];
            double q = Q[i * K + j];
            
            if (p > MIN_PROB && q > MIN_PROB) {
                total_kl += p * log(p / q);
            } else if (p > MIN_PROB) {
                total_kl += KL_CAP;  /* Cap infinite divergence */
            }
        }
    }
    
    return total_kl / K;  /* Average per row */
}

/*═══════════════════════════════════════════════════════════════════════════
 * COMPUTE TRIAD
 *═══════════════════════════════════════════════════════════════════════════*/

DivergenceTriad divergence_compute(
    const float *Pi_oracle,
    const float *Pi_storvik,
    const float *Pi_operating,
    int K)
{
    DivergenceTriad div;
    memset(&div, 0, sizeof(div));
    
    if (!Pi_storvik || !Pi_operating || K <= 0) {
        return div;
    }
    
    /* D₃: RBPF internal consistency (always computable) */
    div.d3_storvik_vs_operating = kl_divergence_matrix(Pi_storvik, Pi_operating, K);
    div.d3_valid = true;
    
    /* D₁, D₂: Only if Oracle available */
    if (Pi_oracle) {
        div.d1_oracle_vs_storvik = kl_divergence_matrix(Pi_oracle, Pi_storvik, K);
        div.d2_oracle_vs_operating = kl_divergence_matrix(Pi_oracle, Pi_operating, K);
        div.oracle_available = true;
    }
    
    return div;
}

/*═══════════════════════════════════════════════════════════════════════════
 * CLASSIFY SCENARIO
 *═══════════════════════════════════════════════════════════════════════════*/

DivergenceScenario divergence_classify(
    const DivergenceTriad *div,
    const DivergenceThresholds *thresh)
{
    if (!div || !thresh) return DIV_SCENARIO_HEALTHY;
    
    /* Check D₃ first (RBPF internal signal) */
    bool d3_high = (div->d3_valid && div->d3_storvik_vs_operating > thresh->d3_danger);
    
    if (!div->oracle_available) {
        /* No Oracle - can only check D₃ */
        if (d3_high) {
            return DIV_SCENARIO_SELF_CONTRADICTION;
        }
        return DIV_SCENARIO_HEALTHY;
    }
    
    /* Oracle available - check all three */
    bool d1_low = (div->d1_oracle_vs_storvik < thresh->d_agree);
    bool d1_high = (div->d1_oracle_vs_storvik > thresh->d_disagree);
    bool d2_high = (div->d2_oracle_vs_operating > thresh->d_disagree);
    
    /*
     * SCENARIO CLASSIFICATION:
     *
     * D₃ high + D₁ low → Oracle and Storvik agree, Operating is wrong
     *                    STRONGEST signal: two sources agree
     *
     * D₃ high + D₁ high → Three-way disagreement (confusion)
     *
     * D₃ low + D₂ high → Oracle sees something new, RBPF is consistent
     *                    Standard innovation case
     *
     * D₃ low + D₂ low → All agree, healthy
     */
    
    if (d3_high && d1_low) {
        return DIV_SCENARIO_STRONG_AGREEMENT;  /* Oracle + Storvik agree */
    }
    
    if (d3_high && d1_high) {
        return DIV_SCENARIO_CONFUSION;  /* Everyone disagrees */
    }
    
    if (d3_high) {
        return DIV_SCENARIO_SELF_CONTRADICTION;
    }
    
    if (d2_high) {
        return DIV_SCENARIO_ORACLE_INNOVATION;
    }
    
    return DIV_SCENARIO_HEALTHY;
}

/*═══════════════════════════════════════════════════════════════════════════
 * SCENARIO STRINGS
 *═══════════════════════════════════════════════════════════════════════════*/

const char *divergence_scenario_str(DivergenceScenario scenario) {
    switch (scenario) {
        case DIV_SCENARIO_HEALTHY:
            return "HEALTHY (all agree)";
        case DIV_SCENARIO_SELF_CONTRADICTION:
            return "SELF-CONTRADICTION (D3 high, RBPF internal conflict)";
        case DIV_SCENARIO_STRONG_AGREEMENT:
            return "STRONG AGREEMENT (Oracle + Storvik agree, Operating stale)";
        case DIV_SCENARIO_ORACLE_INNOVATION:
            return "ORACLE INNOVATION (Oracle sees something new)";
        case DIV_SCENARIO_CONFUSION:
            return "CONFUSION (three-way disagreement)";
        default:
            return "UNKNOWN";
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * UTILITY FUNCTIONS
 *═══════════════════════════════════════════════════════════════════════════*/

double divergence_symmetric_kl(const float *P, const float *Q, int K) {
    double kl_pq = kl_divergence_matrix(P, Q, K);
    double kl_qp = kl_divergence_matrix(Q, P, K);
    return 0.5 * (kl_pq + kl_qp);
}

double divergence_frobenius(const float *P, const float *Q, int K) {
    if (!P || !Q || K <= 0) return 0.0;
    
    double sum_sq = 0.0;
    for (int i = 0; i < K * K; i++) {
        double diff = P[i] - Q[i];
        sum_sq += diff * diff;
    }
    
    return sqrt(sum_sq);
}
