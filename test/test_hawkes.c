/**
 * @file test_hawkes.c
 * @brief Test Hawkes process and RBPF integration
 *
 * Demonstrates:
 *   1. Basic Hawkes updates and intensity tracking
 *   2. Calibration from data
 *   3. Regime transition modification
 *   4. Comparison with Omori decay
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "hawkes_intensity.h"

/*═══════════════════════════════════════════════════════════════════════════
 * RNG
 *═══════════════════════════════════════════════════════════════════════════*/

static unsigned int rng_state = 12345;

static float rng_uniform(void) {
    rng_state = rng_state * 1103515245 + 12345;
    return (float)((rng_state >> 16) & 0x7FFF) / 32768.0f;
}

static float rng_normal(void) {
    float u1 = rng_uniform();
    float u2 = rng_uniform();
    while (u1 < 1e-10f) u1 = rng_uniform();
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
}

/*═══════════════════════════════════════════════════════════════════════════
 * SYNTHETIC DATA WITH CLUSTERING
 *═══════════════════════════════════════════════════════════════════════════*/

typedef struct {
    float *returns;
    int *regimes;
    int n;
} TestData;

/**
 * Generate returns with volatility clustering (self-exciting)
 */
static TestData generate_clustered_data(int n, unsigned int seed)
{
    rng_state = seed;
    
    TestData data;
    data.returns = (float*)malloc(n * sizeof(float));
    data.regimes = (int*)malloc(n * sizeof(int));
    data.n = n;
    
    /* True Hawkes parameters for data generation */
    float mu = 0.03f;       /* Baseline event rate */
    float alpha = 0.6f;     /* Excitation */
    float beta = 0.1f;      /* Decay */
    
    float intensity = mu;
    int regime = 0;
    float base_vol = 0.01f;
    
    /* Event history for generation */
    float event_times[256];
    int n_events = 0;
    
    for (int t = 0; t < n; t++) {
        /* Decay intensity */
        float new_intensity = mu;
        for (int i = 0; i < n_events; i++) {
            float dt = (float)t - event_times[i];
            if (dt > 0) {
                new_intensity += alpha * expf(-beta * dt);
            }
        }
        intensity = new_intensity;
        
        /* Regime based on intensity */
        if (intensity < 0.1f) regime = 0;
        else if (intensity < 0.3f) regime = 1;
        else if (intensity < 0.6f) regime = 2;
        else regime = 3;
        
        /* Volatility scales with regime */
        float vol = base_vol * (1.0f + regime * 1.5f);
        
        /* Generate return */
        float ret = vol * rng_normal();
        
        /* Jumps during high intensity */
        if (intensity > 0.5f && rng_uniform() < 0.1f) {
            ret += (rng_uniform() < 0.5f ? -1.0f : 1.0f) * vol * 4.0f;
        }
        
        /* Self-excitation: large moves trigger events */
        float abs_ret = fabsf(ret);
        float threshold = 0.02f + 0.01f * regime;
        if (abs_ret > threshold) {
            if (n_events < 256) {
                event_times[n_events++] = (float)t;
            }
        }
        
        data.returns[t] = ret;
        data.regimes[t] = regime;
    }
    
    return data;
}

static void free_test_data(TestData *data) {
    free(data->returns);
    free(data->regimes);
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST 1: BASIC HAWKES UPDATE
 *═══════════════════════════════════════════════════════════════════════════*/

static void test_basic_update(void)
{
    printf("\n");
    printf("================================================================\n");
    printf("  TEST 1: Basic Hawkes Update\n");
    printf("================================================================\n");
    
    HawkesState state;
    hawkes_init(&state, NULL);  /* Default config */
    
    hawkes_print_config(&state.config);
    
    /* Simulate a sequence with a shock */
    printf("\nSimulating sequence with shock at t=50:\n");
    printf("  t     return   intensity   excited   events\n");
    printf("  ---   ------   ---------   -------   ------\n");
    
    for (int t = 0; t < 100; t++) {
        float ret;
        
        if (t == 50) {
            ret = 0.08f;  /* Large shock */
        } else if (t > 50 && t < 55 && rng_uniform() < 0.3f) {
            ret = 0.04f * (rng_uniform() < 0.5f ? 1.0f : -1.0f);  /* Aftershocks */
        } else {
            ret = 0.01f * rng_normal();  /* Normal */
        }
        
        int regime = (t >= 50 && t < 60) ? 2 : 0;
        hawkes_update(&state, (float)t, ret, regime);
        
        if (t < 10 || (t >= 45 && t <= 70) || t >= 95) {
            printf("  %3d   %+.4f   %.4f     %.4f    %d\n",
                   t, ret, state.intensity, state.intensity_excited, state.total_events);
        } else if (t == 10) {
            printf("  ...   (normal period)\n");
        }
    }
    
    hawkes_print_state(&state);
    hawkes_free(&state);
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST 2: CALIBRATION
 *═══════════════════════════════════════════════════════════════════════════*/

static void test_calibration(void)
{
    printf("\n");
    printf("================================================================\n");
    printf("  TEST 2: Hawkes Calibration\n");
    printf("================================================================\n");
    
    /* Generate data with known parameters */
    printf("\nGenerating clustered data (n=5000)...\n");
    TestData data = generate_clustered_data(5000, 42);
    
    /* Count events */
    int n_events = 0;
    float threshold = 0.025f;
    for (int i = 0; i < data.n; i++) {
        if (fabsf(data.returns[i]) > threshold) n_events++;
    }
    printf("  Events above threshold (%.1f%%): %d (%.1f%% of ticks)\n",
           threshold * 100, n_events, 100.0f * n_events / data.n);
    
    /* Calibrate with method of moments */
    float mu_mom, alpha_mom, beta_mom;
    hawkes_calibrate_moments(data.returns, data.n, threshold,
                             &mu_mom, &alpha_mom, &beta_mom);
    
    printf("\nMethod of Moments Estimates:\n");
    printf("  mu = %.4f\n", mu_mom);
    printf("  alpha = %.4f\n", alpha_mom);
    printf("  beta = %.4f\n", beta_mom);
    printf("  branching ratio n = %.3f\n", alpha_mom / beta_mom);
    printf("  half-life = %.1f ticks\n", 0.693f / beta_mom);
    
    /* Calibrate with MLE */
    float mu_mle, alpha_mle, beta_mle;
    hawkes_calibrate_mle(data.returns, data.n, threshold,
                         &mu_mle, &alpha_mle, &beta_mle);
    
    printf("\nMLE Estimates:\n");
    printf("  mu = %.4f\n", mu_mle);
    printf("  alpha = %.4f\n", alpha_mle);
    printf("  beta = %.4f\n", beta_mle);
    printf("  branching ratio n = %.3f\n", alpha_mle / beta_mle);
    printf("  half-life = %.1f ticks\n", 0.693f / beta_mle);
    
    printf("\nTrue parameters (approx):\n");
    printf("  mu ≈ 0.03, alpha ≈ 0.6, beta ≈ 0.1\n");
    
    free_test_data(&data);
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST 3: REGIME TRANSITION MODIFICATION
 *═══════════════════════════════════════════════════════════════════════════*/

static void test_transition_modification(void)
{
    printf("\n");
    printf("================================================================\n");
    printf("  TEST 3: Regime Transition Modification\n");
    printf("================================================================\n");
    
    HawkesState state;
    HawkesConfig cfg = hawkes_config_defaults();
    cfg.intensity_threshold = 0.2f;
    cfg.max_transition_boost = 0.15f;
    hawkes_init(&state, &cfg);
    
    int n_regimes = 4;
    
    /* Test at different intensity levels */
    float intensities[] = {0.05f, 0.15f, 0.25f, 0.40f, 0.60f, 0.80f};
    int n_tests = sizeof(intensities) / sizeof(intensities[0]);
    
    printf("\nTransition probabilities from R1 at different intensities:\n");
    printf("(Base: P(stay)=0.98, P(up)=0.01, P(down)=0.01)\n\n");
    printf("  Intensity   P(R0)   P(R1)   P(R2)   P(R3)   Action\n");
    printf("  ---------   -----   -----   -----   -----   ------\n");
    
    for (int i = 0; i < n_tests; i++) {
        /* Set intensity manually for test */
        state.intensity = intensities[i];
        
        /* Base transition probabilities from R1 */
        float trans[4] = {0.01f, 0.98f, 0.01f, 0.00f};
        
        hawkes_modify_transition_probs(&state, trans, 1, n_regimes);
        
        const char *action = "";
        if (intensities[i] < cfg.intensity_threshold - 0.05f) {
            action = "Boost ↓";
        } else if (intensities[i] > cfg.intensity_threshold + 0.05f) {
            action = "Boost ↑";
        } else {
            action = "Neutral";
        }
        
        printf("  %.2f        %.3f   %.3f   %.3f   %.3f   %s\n",
               intensities[i], trans[0], trans[1], trans[2], trans[3], action);
    }
    
    hawkes_free(&state);
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST 4: HAWKES VS OMORI COMPARISON
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Simple Omori decay (for comparison)
 */
static float omori_intensity(float t, float t_shock, float K, float c, float p)
{
    if (t <= t_shock) return 0.0f;
    float dt = t - t_shock;
    return K / powf(dt + c, p);
}

static void test_hawkes_vs_omori(void)
{
    printf("\n");
    printf("================================================================\n");
    printf("  TEST 4: Hawkes vs Omori Decay Comparison\n");
    printf("================================================================\n");
    
    HawkesState hawkes;
    hawkes_init(&hawkes, NULL);
    
    /* Omori parameters (typical) */
    float K = 1.0f, c = 1.0f, p = 1.2f;
    
    /* Simulate shock at t=10, aftershocks, then compare decay */
    printf("\nScenario: Main shock at t=10, aftershocks at t=12,15,18\n\n");
    printf("  t     Hawkes   Omori    Ratio   Note\n");
    printf("  ---   ------   ------   -----   ----\n");
    
    for (int t = 0; t <= 60; t++) {
        float ret = 0.01f * rng_normal();  /* Background noise */
        
        /* Shocks */
        if (t == 10) ret = 0.10f;       /* Main shock */
        else if (t == 12) ret = 0.05f;  /* Aftershock 1 */
        else if (t == 15) ret = 0.04f;  /* Aftershock 2 */
        else if (t == 18) ret = 0.03f;  /* Aftershock 3 */
        
        hawkes_update(&hawkes, (float)t, ret, fabsf(ret) > 0.03f ? 2 : 0);
        
        /* Omori only knows about main shock */
        float omori = 0.02f + omori_intensity((float)t, 10.0f, K, c, p);
        
        float ratio = hawkes.intensity / (omori + 1e-6f);
        
        const char *note = "";
        if (t == 10) note = "MAIN SHOCK";
        else if (t == 12 || t == 15 || t == 18) note = "aftershock";
        
        if (t <= 5 || (t >= 8 && t <= 25) || t % 10 == 0) {
            printf("  %2d    %.4f   %.4f   %.2f    %s\n",
                   t, hawkes.intensity, omori, ratio, note);
        }
    }
    
    printf("\nKey differences:\n");
    printf("  - Hawkes intensity JUMPS at each aftershock (self-excitation)\n");
    printf("  - Omori only decays from main shock (no aftershock response)\n");
    printf("  - Hawkes ratio > 1 during aftershock clustering\n");
    printf("  - Both decay similarly in quiet periods\n");
    
    hawkes_free(&hawkes);
}

/*═══════════════════════════════════════════════════════════════════════════
 * TEST 5: RBPF-STYLE INTEGRATION DEMO
 *═══════════════════════════════════════════════════════════════════════════*/

static void test_rbpf_integration(void)
{
    printf("\n");
    printf("================================================================\n");
    printf("  TEST 5: RBPF Integration Pattern\n");
    printf("================================================================\n");
    
    /* This shows how Hawkes integrates with RBPF loop */
    
    printf("\n/* Example RBPF + Hawkes integration code: */\n\n");
    
    printf("HawkesState hawkes;\n");
    printf("hawkes_init(&hawkes, NULL);\n\n");
    
    printf("for (int t = 0; t < n_ticks; t++) {\n");
    printf("    /* 1. RBPF step */\n");
    printf("    rbpf_ksc_step(rbpf, returns[t], &out);\n");
    printf("    \n");
    printf("    /* 2. Update Hawkes intensity */\n");
    printf("    float intensity = hawkes_update(&hawkes, t, returns[t], out.smoothed_regime);\n");
    printf("    \n");
    printf("    /* 3. Modify next step's transition probabilities */\n");
    printf("    float trans[16];\n");
    printf("    rbpf_ksc_get_transition_probs(rbpf, out.smoothed_regime, trans);\n");
    printf("    hawkes_modify_transition_probs(&hawkes, trans, out.smoothed_regime, n_regimes);\n");
    printf("    rbpf_ksc_set_transition_override(rbpf, out.smoothed_regime, trans);\n");
    printf("    \n");
    printf("    /* 4. (Optional) Use intensity for vol adjustment */\n");
    printf("    float hawkes_vol = hawkes_expected_vol(&hawkes, out.smoothed_regime);\n");
    printf("    /* Blend with RBPF vol: vol = 0.8*rbpf_vol + 0.2*hawkes_vol */\n");
    printf("}\n");
    
    printf("\n/* Key integration points: */\n");
    printf("1. hawkes_update() after each RBPF step\n");
    printf("2. hawkes_modify_transition_probs() before next RBPF step\n");
    printf("3. Optional: blend Hawkes vol with RBPF vol estimate\n");
    printf("4. Optional: use hawkes_regime_loglik() in particle weighting\n");
}

/*═══════════════════════════════════════════════════════════════════════════
 * MAIN
 *═══════════════════════════════════════════════════════════════════════════*/

int main(int argc, char *argv[])
{
    printf("\n");
    printf("================================================================\n");
    printf("  HAWKES PROCESS TEST SUITE\n");
    printf("================================================================\n");
    
    rng_state = (unsigned int)time(NULL);
    
    test_basic_update();
    test_calibration();
    test_transition_modification();
    test_hawkes_vs_omori();
    test_rbpf_integration();
    
    printf("\n");
    printf("================================================================\n");
    printf("  ALL TESTS COMPLETE\n");
    printf("================================================================\n");
    
    return 0;
}
