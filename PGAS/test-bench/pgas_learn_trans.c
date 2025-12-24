/*═══════════════════════════════════════════════════════════════════════════════
 * PGAS Transition Matrix Learner
 *
 * Purpose: Learn transition matrix from synthetic data, output copy-pasteable C.
 *
 * Data generation: EXACT copy from test_mmpf_comparison.c (8000 ticks, 7 scenarios)
 * Volatility params: Your tuned values (fixed, not learned)
 * Output: rbpf_real_t trans[16] = {...};
 *
 * Usage:
 *   pgas_learn_trans [seed]
 *   Default seed = 42 (matches your tuner)
 *
 * Updated for per-regime sigma_vol API (ALIGNED WITH RBPF)
 *═══════════════════════════════════════════════════════════════════════════════*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "pgas_mkl.h"

/*═══════════════════════════════════════════════════════════════════════════════
 * CONFIGURATION - YOUR TUNED VALUES
 *═══════════════════════════════════════════════════════════════════════════════*/

#define N_REGIMES 4
#define N_PARTICLES 128
#define N_BURNIN 300
#define N_SAMPLES 500
#define N_TICKS 8000

/* Your tuned volatility parameters (FIXED - PGAS only learns transitions) */
static const double TUNED_MU_VOL[N_REGIMES] = {-4.50, -3.67, -2.83, -2.00};
static const double TUNED_SIGMA_VOL[N_REGIMES] = {0.080, 0.267, 0.453, 0.640};
static const double TUNED_PHI = 0.97;

/*═══════════════════════════════════════════════════════════════════════════════
 * PCG32 RNG - EXACT COPY FROM test_mmpf_comparison.c
 *═══════════════════════════════════════════════════════════════════════════════*/

typedef struct
{
    uint64_t state;
    uint64_t inc;
} pcg32_t;

static uint32_t pcg32_random(pcg32_t *rng)
{
    uint64_t oldstate = rng->state;
    rng->state = oldstate * 6364136223846793005ULL + rng->inc;
    uint32_t xorshifted = (uint32_t)(((oldstate >> 18u) ^ oldstate) >> 27u);
    uint32_t rot = (uint32_t)(oldstate >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((0u - rot) & 31));
}

static double pcg32_double(pcg32_t *rng)
{
    return (double)pcg32_random(rng) / 4294967296.0;
}

static double pcg32_gaussian(pcg32_t *rng)
{
    double u1 = pcg32_double(rng);
    double u2 = pcg32_double(rng);
    if (u1 < 1e-10)
        u1 = 1e-10;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979 * u2);
}

/*═══════════════════════════════════════════════════════════════════════════════
 * HYPOTHESIS PARAMETERS - EXACT COPY FROM test_mmpf_comparison.c
 *═══════════════════════════════════════════════════════════════════════════════*/

typedef enum
{
    HYPO_CALM = 0,
    HYPO_TREND = 1,
    HYPO_CRISIS = 2,
    N_HYPOTHESES = 3
} Hypothesis;

typedef struct
{
    double mu_vol;
    double phi;
    double sigma_eta;
} HypothesisParams;

static const HypothesisParams TRUE_PARAMS[N_HYPOTHESES] = {
    /* CALM: Low vol, high persistence, smooth */
    {.mu_vol = -5.0, .phi = 0.995, .sigma_eta = 0.08},
    /* TREND: Medium vol, medium persistence */
    {.mu_vol = -3.5, .phi = 0.95, .sigma_eta = 0.20},
    /* CRISIS: High vol, fast mean reversion, explosive */
    {.mu_vol = -1.5, .phi = 0.85, .sigma_eta = 0.50}};

/*═══════════════════════════════════════════════════════════════════════════════
 * DATA GENERATION - EXACT COPY FROM test_mmpf_comparison.c
 *═══════════════════════════════════════════════════════════════════════════════*/

typedef struct
{
    double *returns;
    double *log_sq_returns; /* y_t = log(r_t^2) for PGAS */
    int n_ticks;
} SyntheticData;

static SyntheticData *generate_test_data(int seed)
{
    SyntheticData *data = (SyntheticData *)calloc(1, sizeof(SyntheticData));

    int n = N_TICKS;
    data->n_ticks = n;
    data->returns = (double *)malloc(n * sizeof(double));
    data->log_sq_returns = (double *)malloc(n * sizeof(double));

    pcg32_t rng = {seed * 12345ULL + 1, seed * 67890ULL | 1};

    /* Start in CALM */
    double log_vol = TRUE_PARAMS[HYPO_CALM].mu_vol;
    int t = 0;

#define EVOLVE_STATE(H)                                                                       \
    do                                                                                        \
    {                                                                                         \
        const HypothesisParams *p = &TRUE_PARAMS[H];                                          \
        double theta = 1.0 - p->phi;                                                          \
        log_vol = p->phi * log_vol + theta * p->mu_vol + p->sigma_eta * pcg32_gaussian(&rng); \
        double vol = exp(log_vol);                                                            \
        double ret = vol * pcg32_gaussian(&rng);                                              \
        data->returns[t] = ret;                                                               \
        /* Convert to OCSN observation: y = log(r^2) */                                       \
        double r_sq = ret * ret;                                                              \
        if (r_sq < 1e-20)                                                                     \
            r_sq = 1e-20;                                                                     \
        data->log_sq_returns[t] = log(r_sq);                                                  \
    } while (0)

    /*═══════════════════════════════════════════════════════════════════════
     * Scenario 1: Extended Calm (0-1499)
     *═══════════════════════════════════════════════════════════════════════*/
    for (; t < 1500; t++)
    {
        EVOLVE_STATE(HYPO_CALM);
    }

    /*═══════════════════════════════════════════════════════════════════════
     * Scenario 2: Slow Trend (1500-2499)
     *═══════════════════════════════════════════════════════════════════════*/
    for (; t < 2500; t++)
    {
        Hypothesis h = (t < 1800) ? HYPO_CALM : HYPO_TREND;
        EVOLVE_STATE(h);
    }

    /*═══════════════════════════════════════════════════════════════════════
     * Scenario 3: Sudden Crisis (2500-2999)
     *═══════════════════════════════════════════════════════════════════════*/
    for (; t < 3000; t++)
    {
        EVOLVE_STATE(HYPO_CRISIS);
    }

    /*═══════════════════════════════════════════════════════════════════════
     * Scenario 4: Crisis Persistence (3000-3999)
     *═══════════════════════════════════════════════════════════════════════*/
    for (; t < 4000; t++)
    {
        EVOLVE_STATE(HYPO_CRISIS);
    }

    /*═══════════════════════════════════════════════════════════════════════
     * Scenario 5: Recovery (4000-5199)
     *═══════════════════════════════════════════════════════════════════════*/
    for (; t < 5200; t++)
    {
        Hypothesis h;
        if (t < 4400)
            h = HYPO_CRISIS;
        else if (t < 4800)
            h = HYPO_TREND;
        else
            h = HYPO_CALM;
        EVOLVE_STATE(h);
    }

    /*═══════════════════════════════════════════════════════════════════════
     * Scenario 6: Flash Crash (5200-5699)
     *═══════════════════════════════════════════════════════════════════════*/
    for (; t < 5700; t++)
    {
        Hypothesis h;
        if (t >= 5350 && t < 5410)
            h = HYPO_CRISIS;
        else
            h = HYPO_CALM;
        EVOLVE_STATE(h);
    }

    /*═══════════════════════════════════════════════════════════════════════
     * Scenario 7: Choppy (5700-7999)
     *═══════════════════════════════════════════════════════════════════════*/
    Hypothesis current_h = HYPO_TREND;
    int next_switch = 5700 + 80 + (int)(pcg32_double(&rng) * 120);

    for (; t < N_TICKS; t++)
    {
        if (t >= next_switch)
        {
            int delta = (pcg32_double(&rng) < 0.5) ? -1 : 1;
            current_h = (Hypothesis)((current_h + delta + N_HYPOTHESES) % N_HYPOTHESES);
            next_switch = t + 80 + (int)(pcg32_double(&rng) * 150);
        }
        EVOLVE_STATE(current_h);
    }

#undef EVOLVE_STATE

    return data;
}

static void free_data(SyntheticData *data)
{
    if (!data)
        return;
    free(data->returns);
    free(data->log_sq_returns);
    free(data);
}

/*═══════════════════════════════════════════════════════════════════════════════
 * REFERENCE TRAJECTORY INITIALIZATION (from observations only)
 *
 * Simple heuristic: assign regime based on observation magnitude
 * This washes out after burn-in anyway
 *═══════════════════════════════════════════════════════════════════════════════*/

static void init_reference_from_observations(
    const double *y, /* log-squared returns */
    int T,
    int K,
    const double *mu_vol, /* regime means */
    double phi,
    int *out_regimes,
    double *out_h)
{
    /*
     * Simple assignment: Pick regime whose mu_vol is closest to observation
     * Then set h = observation (crude but burn-in will fix it)
     */
    for (int t = 0; t < T; t++)
    {
        double obs = y[t];

        /* Find closest regime by mu_vol */
        int best_k = 0;
        double best_dist = fabs(obs - mu_vol[0]);
        for (int k = 1; k < K; k++)
        {
            double dist = fabs(obs - mu_vol[k]);
            if (dist < best_dist)
            {
                best_dist = dist;
                best_k = k;
            }
        }

        out_regimes[t] = best_k;
        out_h[t] = mu_vol[best_k]; /* Start at regime mean */
    }

    /* Smooth h with simple AR(1) pass */
    for (int t = 1; t < T; t++)
    {
        out_h[t] = phi * out_h[t - 1] + (1.0 - phi) * mu_vol[out_regimes[t]];
    }
}

/*═══════════════════════════════════════════════════════════════════════════════
 * ENSEMBLE ACCUMULATOR
 *═══════════════════════════════════════════════════════════════════════════════*/

typedef struct
{
    int K;
    int n_samples;
    double trans_sum[16];
    double trans_sum_sq[16];
} Ensemble;

static void ensemble_init(Ensemble *ens, int K)
{
    memset(ens, 0, sizeof(Ensemble));
    ens->K = K;
}

static void ensemble_accumulate(Ensemble *ens, const float *trans)
{
    int K = ens->K;
    ens->n_samples++;
    for (int i = 0; i < K * K; i++)
    {
        ens->trans_sum[i] += trans[i];
        ens->trans_sum_sq[i] += trans[i] * trans[i];
    }
}

static void ensemble_get_mean(const Ensemble *ens, float *out)
{
    int n = ens->n_samples;
    if (n == 0)
        return;
    for (int i = 0; i < ens->K * ens->K; i++)
    {
        out[i] = (float)(ens->trans_sum[i] / n);
    }
}

static double ensemble_get_std(const Ensemble *ens, int i, int j)
{
    int n = ens->n_samples;
    if (n < 2)
        return 0.0;
    int idx = i * ens->K + j;
    double mean = ens->trans_sum[idx] / n;
    double var = (ens->trans_sum_sq[idx] / n) - (mean * mean);
    if (var < 0)
        var = 0;
    return sqrt(var);
}

/*═══════════════════════════════════════════════════════════════════════════════
 * MAIN
 *═══════════════════════════════════════════════════════════════════════════════*/

int main(int argc, char **argv)
{
    int seed = 42;
    if (argc > 1)
        seed = atoi(argv[1]);

    printf("╔═══════════════════════════════════════════════════════════════════════╗\n");
    printf("║           PGAS Transition Matrix Learner                             ║\n");
    printf("╠═══════════════════════════════════════════════════════════════════════╣\n");
    printf("║  Seed: %-5d                                                         ║\n", seed);
    printf("║  Data: 8000 ticks (same as test_mmpf_comparison)                     ║\n");
    printf("║  K=4, Particles=%d, Burn-in=%d, Samples=%d                        ║\n",
           N_PARTICLES, N_BURNIN, N_SAMPLES);
    printf("╚═══════════════════════════════════════════════════════════════════════╝\n\n");

    /* Generate data */
    printf("Generating synthetic data (seed=%d)...\n", seed);
    SyntheticData *data = generate_test_data(seed);
    printf("  %d ticks generated\n\n", data->n_ticks);

    /* Allocate PGAS */
    printf("Initializing PGAS...\n");
    PGASMKLState *pgas = pgas_mkl_alloc(N_PARTICLES, N_TICKS, N_REGIMES, seed + 1000);
    if (!pgas)
    {
        fprintf(stderr, "ERROR: Failed to allocate PGAS\n");
        free_data(data);
        return 1;
    }

    /* Set model with YOUR tuned volatility params */
    /* Start with uniform transitions - PGAS will learn */
    double init_trans[16];
    for (int i = 0; i < N_REGIMES; i++)
    {
        for (int j = 0; j < N_REGIMES; j++)
        {
            init_trans[i * N_REGIMES + j] = (i == j) ? 0.85 : 0.05;
        }
    }

    /* Updated API: no sigma_h argument (per-regime sigma_vol is used) */
    pgas_mkl_set_model(pgas, init_trans, TUNED_MU_VOL, TUNED_SIGMA_VOL, TUNED_PHI);

    /* Set transition prior: κ=50 + adaptive (capped at 150) */
    pgas_mkl_set_transition_prior(pgas, 1.0f, 50.0f);
    pgas_mkl_enable_adaptive_kappa(pgas, 1);
    pgas_mkl_configure_adaptive_kappa(pgas, 30.0f, 150.0f, 0.0f, 0.0f);

    printf("  mu_vol  = {%.2f, %.2f, %.2f, %.2f}\n",
           TUNED_MU_VOL[0], TUNED_MU_VOL[1], TUNED_MU_VOL[2], TUNED_MU_VOL[3]);
    printf("  sigma_vol = {%.3f, %.3f, %.3f, %.3f}\n",
           TUNED_SIGMA_VOL[0], TUNED_SIGMA_VOL[1], TUNED_SIGMA_VOL[2], TUNED_SIGMA_VOL[3]);
    printf("  phi=%.2f\n", TUNED_PHI);
    printf("  κ=50 (adaptive, max=150)\n\n");

    /* Load observations */
    pgas_mkl_load_observations(pgas, data->log_sq_returns, N_TICKS);

    /* Initialize reference trajectory from observations (no ground truth) */
    printf("Initializing reference trajectory from observations...\n");
    int *init_regimes = (int *)malloc(N_TICKS * sizeof(int));
    double *init_h = (double *)malloc(N_TICKS * sizeof(double));

    init_reference_from_observations(data->log_sq_returns, N_TICKS, N_REGIMES,
                                     TUNED_MU_VOL, TUNED_PHI,
                                     init_regimes, init_h);

    pgas_mkl_set_reference(pgas, init_regimes, init_h, N_TICKS);
    free(init_regimes);
    free(init_h);

    /* Burn-in */
    printf("\nBurn-in (%d sweeps)...\n", N_BURNIN);
    for (int i = 0; i < N_BURNIN; i++)
    {
        pgas_mkl_gibbs_sweep(pgas);
        if ((i + 1) % 100 == 0)
        {
            printf("  Sweep %d: κ=%.1f, chatter=%.2f, accept=%.3f\n",
                   i + 1,
                   pgas_mkl_get_sticky_kappa(pgas),
                   pgas_mkl_get_chatter_ratio(pgas),
                   pgas_mkl_get_acceptance_rate(pgas));
        }
    }

    /* Sampling */
    printf("\nSampling (%d sweeps)...\n", N_SAMPLES);
    Ensemble ens;
    ensemble_init(&ens, N_REGIMES);

    float trans_current[16];

    for (int i = 0; i < N_SAMPLES; i++)
    {
        pgas_mkl_gibbs_sweep(pgas);
        pgas_mkl_get_transitions(pgas, trans_current, N_REGIMES);
        ensemble_accumulate(&ens, trans_current);

        if ((i + 1) % 100 == 0)
        {
            printf("  Sample %d: κ=%.1f, chatter=%.2f\n",
                   i + 1,
                   pgas_mkl_get_sticky_kappa(pgas),
                   pgas_mkl_get_chatter_ratio(pgas));
        }
    }

    /* Get posterior mean */
    float trans_learned[16];
    ensemble_get_mean(&ens, trans_learned);

    /* Results */
    printf("\n");
    printf("═══════════════════════════════════════════════════════════════════════\n");
    printf("RESULTS\n");
    printf("═══════════════════════════════════════════════════════════════════════\n\n");

    printf("Final κ: %.1f\n", pgas_mkl_get_sticky_kappa(pgas));
    printf("Final chatter: %.2f\n", pgas_mkl_get_chatter_ratio(pgas));
    printf("Final acceptance: %.3f\n\n", pgas_mkl_get_acceptance_rate(pgas));

    printf("Learned transition matrix (mean ± std):\n");
    for (int i = 0; i < N_REGIMES; i++)
    {
        printf("  [");
        for (int j = 0; j < N_REGIMES; j++)
        {
            float mean = trans_learned[i * N_REGIMES + j];
            double std = ensemble_get_std(&ens, i, j);
            printf(" %.4f±%.3f", mean, std);
        }
        printf(" ]\n");
    }

    printf("\nDiagonal (stickiness): %.4f, %.4f, %.4f, %.4f\n",
           trans_learned[0], trans_learned[5], trans_learned[10], trans_learned[15]);

    float avg_diag = (trans_learned[0] + trans_learned[5] +
                      trans_learned[10] + trans_learned[15]) /
                     4.0f;
    printf("Average diagonal: %.4f\n", avg_diag);

    /* ═══════════════════════════════════════════════════════════════════════
     * COPY-PASTEABLE OUTPUT
     * ═══════════════════════════════════════════════════════════════════════*/
    printf("\n");
    printf("═══════════════════════════════════════════════════════════════════════\n");
    printf("COPY-PASTE THIS INTO test_mmpf_comparison.c:\n");
    printf("═══════════════════════════════════════════════════════════════════════\n\n");

    printf("/* PGAS-learned transition matrix (seed=%d, %d ticks, κ_final=%.1f) */\n",
           seed, N_TICKS, pgas_mkl_get_sticky_kappa(pgas));
    printf("rbpf_real_t trans[16] = {\n");
    printf("    %.4ff, %.4ff, %.4ff, %.4ff,\n",
           trans_learned[0], trans_learned[1], trans_learned[2], trans_learned[3]);
    printf("    %.4ff, %.4ff, %.4ff, %.4ff,\n",
           trans_learned[4], trans_learned[5], trans_learned[6], trans_learned[7]);
    printf("    %.4ff, %.4ff, %.4ff, %.4ff,\n",
           trans_learned[8], trans_learned[9], trans_learned[10], trans_learned[11]);
    printf("    %.4ff, %.4ff, %.4ff, %.4ff\n",
           trans_learned[12], trans_learned[13], trans_learned[14], trans_learned[15]);
    printf("};\n");

    printf("\n═══════════════════════════════════════════════════════════════════════\n");

    /* Compare to your tuned matrix for reference */
    printf("\nFor reference, your TUNED matrix was:\n");
    printf("rbpf_real_t trans[16] = {\n");
    printf("    0.920f, 0.056f, 0.020f, 0.004f,\n");
    printf("    0.032f, 0.920f, 0.036f, 0.012f,\n");
    printf("    0.012f, 0.036f, 0.920f, 0.032f,\n");
    printf("    0.004f, 0.020f, 0.056f, 0.920f\n");
    printf("};\n");

    /* Frobenius distance */
    float tuned[16] = {
        0.920f, 0.056f, 0.020f, 0.004f,
        0.032f, 0.920f, 0.036f, 0.012f,
        0.012f, 0.036f, 0.920f, 0.032f,
        0.004f, 0.020f, 0.056f, 0.920f};

    double frob = 0;
    for (int i = 0; i < 16; i++)
    {
        double diff = trans_learned[i] - tuned[i];
        frob += diff * diff;
    }
    frob = sqrt(frob);

    printf("\nFrobenius distance (PGAS vs Tuned): %.4f\n", frob);

    /* Cleanup */
    pgas_mkl_free(pgas);
    free_data(data);

    printf("\n╔═══════════════════════════════════════════════════════════════════════╗\n");
    printf("║  Done! Paste the learned matrix into your test and compare.          ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════════╝\n");

    return 0;
}