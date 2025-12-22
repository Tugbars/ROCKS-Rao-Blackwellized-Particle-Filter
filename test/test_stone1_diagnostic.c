/**
 * @file test_stone1_diagnostic.c
 * @brief Stone #1 Diagnostic: OCSN vs Exact Log-χ² Likelihood Discrimination
 *
 * Measures the Bayes Factor (likelihood ratio) for regime transitions
 * using both the 10-component OCSN mixture and the exact log-χ²(1) PDF.
 *
 * If K_exact >> K_ocsn, then OCSN was "stealing alpha" from regime detection.
 */

#include <stdio.h>
#include <math.h>
#include "rbpf_sprt.h"  /* For sprt_logchisq_loglik (OCSN) */

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/*═══════════════════════════════════════════════════════════════════════════
 * EXACT LOG-χ²(1) LIKELIHOOD
 *
 * The log-χ²(1) distribution arises from:
 *   y = log(r²) where r ~ N(0, σ²)
 *   y = h + ε   where h = log(σ²), ε ~ log-χ²(1)
 *
 * PDF of ε ~ log-χ²(1):
 *   f(ε) = (1/√2π) × exp((ε - exp(ε))/2)
 *
 * Log-PDF (numerically stable):
 *   log f(ε) = -0.5×log(2π) + 0.5×ε - 0.5×exp(ε)
 *
 * Given observation y and state h:
 *   ε = y - h
 *   log P(y|h) = log f(y - h)
 *═══════════════════════════════════════════════════════════════════════════*/

static const double LOG_2PI = 1.8378770664093454835606594728112;  /* log(2π) */

/**
 * @brief Exact log-χ²(1) log-likelihood
 *
 * @param y   Observation: y = log(r²)
 * @param h   Log-volatility state
 * @return    log P(y | h)
 */
double logchi2_exact_logpdf(double y, double h)
{
    double z = y - h;
    
    /* Numerical stability: clamp exp(z) to avoid overflow
     * When z > 700, exp(z) overflows to Inf
     * When z < -700, exp(z) underflows to 0 (which is fine) */
    if (z > 700.0) {
        /* exp(z) dominates, log-likelihood → -∞ */
        return -1e100;
    }
    
    return -0.5 * LOG_2PI + 0.5 * z - 0.5 * exp(z);
}

/**
 * @brief Run discrimination diagnostic for a single boundary
 */
void test_boundary(double y_obs, double h_old, double h_new, const char *label)
{
    /* OCSN (10-component mixture) */
    double loglik_ocsn_old = sprt_logchisq_loglik(y_obs, h_old);
    double loglik_ocsn_new = sprt_logchisq_loglik(y_obs, h_new);
    double log_ratio_ocsn = loglik_ocsn_new - loglik_ocsn_old;
    double ratio_ocsn = exp(log_ratio_ocsn);
    
    /* Exact log-χ²(1) */
    double loglik_exact_old = logchi2_exact_logpdf(y_obs, h_old);
    double loglik_exact_new = logchi2_exact_logpdf(y_obs, h_new);
    double log_ratio_exact = loglik_exact_new - loglik_exact_old;
    double ratio_exact = exp(log_ratio_exact);
    
    /* Discrimination gain */
    double gain_pct = (ratio_exact / ratio_ocsn - 1.0) * 100.0;
    double log_gain = log_ratio_exact - log_ratio_ocsn;
    
    printf("%-20s  y=%.2f  h: %.1f → %.1f\n", label, y_obs, h_old, h_new);
    printf("  OCSN:   log-ratio = %+.4f  ratio = %.4f\n", log_ratio_ocsn, ratio_ocsn);
    printf("  Exact:  log-ratio = %+.4f  ratio = %.4f\n", log_ratio_exact, ratio_exact);
    printf("  Gain:   %+.1f%% (log-gain: %+.4f nats)\n", gain_pct, log_gain);
    printf("\n");
}

/**
 * @brief Test likelihood shape across a range
 */
void test_likelihood_shape(double y_obs)
{
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  Likelihood Shape at y = %.2f\n", y_obs);
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("    h       OCSN log-lik    Exact log-lik    Diff\n");
    printf("───────────────────────────────────────────────────────────────\n");
    
    for (double h = -6.0; h <= -1.0; h += 0.5) {
        double ll_ocsn = sprt_logchisq_loglik(y_obs, h);
        double ll_exact = logchi2_exact_logpdf(y_obs, h);
        double diff = ll_exact - ll_ocsn;
        printf("  %5.1f    %+10.4f      %+10.4f     %+.4f\n", h, ll_ocsn, ll_exact, diff);
    }
    printf("\n");
}

/**
 * @brief Simulate regime transition and measure cumulative evidence
 */
void test_transition_speed(double h_calm, double h_crisis, int n_ticks)
{
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  Transition Speed: h=%.1f → h=%.1f over %d ticks\n", h_calm, h_crisis, n_ticks);
    printf("═══════════════════════════════════════════════════════════════\n");
    
    /* Simulate observations from crisis regime */
    double cum_log_ratio_ocsn = 0.0;
    double cum_log_ratio_exact = 0.0;
    
    /* Threshold: log(34) ≈ 3.53 (needed to overcome 0.92 stickiness) */
    double threshold = log(34.0);
    int ticks_ocsn = -1;
    int ticks_exact = -1;
    
    printf("  Tick   y_obs    OCSN cumul   Exact cumul\n");
    printf("───────────────────────────────────────────────────────────────\n");
    
    /* Use deterministic y values at regime center for clean comparison */
    for (int t = 0; t < n_ticks; t++) {
        /* y drawn from crisis regime: y ≈ h_crisis + E[log-χ²(1)]
         * E[log-χ²(1)] ≈ -1.27 (Euler-Mascheroni related) */
        double y_obs = h_crisis - 1.27 + 0.5 * sin(t * 0.7);  /* Add variation */
        
        double ll_ocsn_old = sprt_logchisq_loglik(y_obs, h_calm);
        double ll_ocsn_new = sprt_logchisq_loglik(y_obs, h_crisis);
        double ll_exact_old = logchi2_exact_logpdf(y_obs, h_calm);
        double ll_exact_new = logchi2_exact_logpdf(y_obs, h_crisis);
        
        cum_log_ratio_ocsn += (ll_ocsn_new - ll_ocsn_old);
        cum_log_ratio_exact += (ll_exact_new - ll_exact_old);
        
        if (ticks_ocsn < 0 && cum_log_ratio_ocsn >= threshold) {
            ticks_ocsn = t + 1;
        }
        if (ticks_exact < 0 && cum_log_ratio_exact >= threshold) {
            ticks_exact = t + 1;
        }
        
        if (t < 20 || t == n_ticks - 1) {
            printf("  %4d   %+.2f    %+8.2f     %+8.2f%s\n", 
                   t, y_obs, cum_log_ratio_ocsn, cum_log_ratio_exact,
                   (t+1 == ticks_exact) ? " ← Exact crosses" : 
                   (t+1 == ticks_ocsn) ? " ← OCSN crosses" : "");
        }
    }
    
    printf("───────────────────────────────────────────────────────────────\n");
    printf("  Threshold (log 34): %.2f\n", threshold);
    printf("  OCSN crosses at:    tick %d\n", ticks_ocsn);
    printf("  Exact crosses at:   tick %d\n", ticks_exact);
    if (ticks_ocsn > 0 && ticks_exact > 0) {
        printf("  Speedup:            %.1fx faster\n", (double)ticks_ocsn / ticks_exact);
    }
    printf("\n");
}

int main(void)
{
    printf("\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  STONE #1 DIAGNOSTIC: OCSN vs Exact Log-χ² Likelihood\n");
    printf("═══════════════════════════════════════════════════════════════\n\n");
    
    /* Test 1: Boundary discrimination at various points */
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  Boundary Discrimination Tests\n");
    printf("═══════════════════════════════════════════════════════════════\n\n");
    
    /* Adjacent regimes: R1 (-4) vs R2 (-3) */
    test_boundary(-3.5, -4.0, -3.0, "R1→R2 boundary");
    test_boundary(-3.0, -4.0, -3.0, "R1→R2 favor R2");
    test_boundary(-4.0, -4.0, -3.0, "R1→R2 favor R1");
    
    /* Calm to crisis: R0 (-5) vs R3 (-2) */
    test_boundary(-2.5, -5.0, -2.0, "Calm→Crisis");
    test_boundary(-1.5, -5.0, -2.0, "Extreme crisis");
    
    /* Test 2: Likelihood shape */
    test_likelihood_shape(-3.5);
    test_likelihood_shape(-2.0);
    
    /* Test 3: Transition speed simulation */
    test_transition_speed(-4.0, -2.0, 50);  /* 2 regime jump */
    test_transition_speed(-5.0, -2.0, 50);  /* 3 regime jump (calm → crisis) */
    
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  Conclusion\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  If Exact crosses threshold significantly faster than OCSN,\n");
    printf("  then OCSN mixture is 'blurring' the likelihood and causing\n");
    printf("  regime detection lag. Replace with exact log-χ².\n");
    printf("═══════════════════════════════════════════════════════════════\n\n");
    
    return 0;
}
