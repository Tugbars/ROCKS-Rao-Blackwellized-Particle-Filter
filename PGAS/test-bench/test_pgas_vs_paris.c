/*═══════════════════════════════════════════════════════════════════════════════
 * PGAS vs PGAS-PARIS Comparison Test
 *
 * Purpose: Validate that PGAS-PARIS integration is working correctly
 *
 * Checks:
 *   1. Both produce valid Π (rows sum to 1, no NaN/Inf)
 *   2. Transition counts are reasonable
 *   3. PARIS trajectory diversity > 0%
 *   4. Learned Π should be similar (not identical, but not wildly different)
 *
 * Red flags:
 *   - Trajectory diversity = 0% → PARIS backward pass broken
 *   - trans_paris all zeros or NaN → counts not propagating
 *   - Frobenius > 0.5 → fundamentally different (likely bug)
 *   - Identical matrices → PARIS not actually being used
 *
 * Updated for per-regime sigma_vol API (ALIGNED WITH RBPF)
 *═══════════════════════════════════════════════════════════════════════════════*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "pgas_mkl.h"
#include "pgas_paris.h"

/*═══════════════════════════════════════════════════════════════════════════════
 * PCG32 RNG (same as your test harness)
 *═══════════════════════════════════════════════════════════════════════════════*/

typedef struct
{
    uint64_t state;
    uint64_t inc;
} pcg32_random_t;

static inline uint32_t pcg32_random_r(pcg32_random_t *rng)
{
    uint64_t oldstate = rng->state;
    rng->state = oldstate * 6364136223846793005ULL + rng->inc;
    uint32_t xorshifted = (uint32_t)(((oldstate >> 18u) ^ oldstate) >> 27u);
    uint32_t rot = (uint32_t)(oldstate >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((0u - rot) & 31));
}

static inline void pcg32_srandom_r(pcg32_random_t *rng, uint64_t seed, uint64_t seq)
{
    rng->state = 0U;
    rng->inc = (seq << 1u) | 1u;
    pcg32_random_r(rng);
    rng->state += seed;
    pcg32_random_r(rng);
}

static inline double pcg32_double(pcg32_random_t *rng)
{
    return (double)pcg32_random_r(rng) / 4294967296.0;
}

static inline double pcg32_normal(pcg32_random_t *rng)
{
    double u1 = pcg32_double(rng);
    double u2 = pcg32_double(rng);
    if (u1 < 1e-10)
        u1 = 1e-10;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2);
}

/*═══════════════════════════════════════════════════════════════════════════════
 * SYNTHETIC DATA GENERATION
 *═══════════════════════════════════════════════════════════════════════════════*/

typedef struct
{
    int T;
    float *y;      /* Observations: y_t = log(r_t^2) */
    float *h_true; /* True log-volatility */
    int *z_true;   /* True regimes */
} SyntheticData;

static void generate_synthetic_data(SyntheticData *data, int T, uint64_t seed)
{
    pcg32_random_t rng;
    pcg32_srandom_r(&rng, seed, 1);

    data->T = T;
    data->y = (float *)malloc(T * sizeof(float));
    data->h_true = (float *)malloc(T * sizeof(float));
    data->z_true = (int *)malloc(T * sizeof(int));

    /* Model parameters */
    const int K = 4;
    const float mu_vol[4] = {-4.50f, -3.67f, -2.83f, -2.00f};
    const float sigma_vol[4] = {0.08f, 0.267f, 0.453f, 0.64f}; /* Per-regime */
    const float phi = 0.97f;

    /* True transition matrix (what we want PGAS to learn) */
    const float true_trans[16] = {
        0.95f, 0.03f, 0.01f, 0.01f,
        0.02f, 0.94f, 0.03f, 0.01f,
        0.01f, 0.03f, 0.94f, 0.02f,
        0.01f, 0.01f, 0.03f, 0.95f};

    /* Initialize */
    int z = 0;
    float h = mu_vol[z];

    for (int t = 0; t < T; t++)
    {
        /* Regime transition */
        float u = (float)pcg32_double(&rng);
        float cumsum = 0.0f;
        int new_z = z;
        for (int j = 0; j < K; j++)
        {
            cumsum += true_trans[z * K + j];
            if (u < cumsum)
            {
                new_z = j;
                break;
            }
        }
        z = new_z;

        /* h dynamics: h_t = mu_z * (1-phi) + phi * h_{t-1} + sigma_vol[z] * eps */
        float eps_h = (float)pcg32_normal(&rng);
        h = mu_vol[z] * (1.0f - phi) + phi * h + sigma_vol[z] * eps_h;

        /* Observation: y_t = log(r_t^2) where r_t = exp(h_t/2) * eps */
        float eps_r = (float)pcg32_normal(&rng);
        float r = expf(h * 0.5f) * eps_r;
        float y = logf(r * r + 1e-10f);

        data->y[t] = y;
        data->h_true[t] = h;
        data->z_true[t] = z;
    }
}

static void free_synthetic_data(SyntheticData *data)
{
    free(data->y);
    free(data->h_true);
    free(data->z_true);
}

/*═══════════════════════════════════════════════════════════════════════════════
 * UTILITY FUNCTIONS
 *═══════════════════════════════════════════════════════════════════════════════*/

static void print_matrix(const char *name, const float *trans, int K)
{
    printf("\n%s:\n", name);
    for (int i = 0; i < K; i++)
    {
        printf("  [");
        for (int j = 0; j < K; j++)
        {
            printf(" %.4f", trans[i * K + j]);
        }
        printf(" ]\n");
    }
}

static float frobenius_distance(const float *A, const float *B, int K)
{
    float sum = 0.0f;
    for (int i = 0; i < K * K; i++)
    {
        float diff = A[i] - B[i];
        sum += diff * diff;
    }
    return sqrtf(sum);
}

static int validate_stochastic_matrix(const float *trans, int K, const char *name)
{
    int valid = 1;

    for (int i = 0; i < K; i++)
    {
        float row_sum = 0.0f;
        for (int j = 0; j < K; j++)
        {
            float p = trans[i * K + j];
            if (isnan(p) || isinf(p))
            {
                printf("  ERROR: %s[%d][%d] = %f (NaN/Inf)\n", name, i, j, p);
                valid = 0;
            }
            if (p < 0.0f || p > 1.0f)
            {
                printf("  WARNING: %s[%d][%d] = %f (out of [0,1])\n", name, i, j, p);
            }
            row_sum += p;
        }
        if (fabsf(row_sum - 1.0f) > 0.01f)
        {
            printf("  ERROR: %s row %d sums to %f (not 1.0)\n", name, i, row_sum);
            valid = 0;
        }
    }

    return valid;
}

static float compute_avg_diagonal(const float *trans, int K)
{
    float sum = 0.0f;
    for (int i = 0; i < K; i++)
    {
        sum += trans[i * K + i];
    }
    return sum / K;
}

/*═══════════════════════════════════════════════════════════════════════════════
 * MAIN TEST
 *═══════════════════════════════════════════════════════════════════════════════*/

int main(void)
{
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("  PGAS vs PGAS-PARIS Comparison Test\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");

    /* Configuration */
    const int K = 4;
    const int N = 64;
    const int T = 2000;
    const int N_SWEEPS = 500;
    const int BURNIN = 200;
    const int M_TRAJECTORIES = 8;
    const uint64_t SEED = 42;

    printf("Configuration:\n");
    printf("  K (regimes):       %d\n", K);
    printf("  N (particles):     %d\n", N);
    printf("  T (timesteps):     %d\n", T);
    printf("  Sweeps:            %d (burnin: %d)\n", N_SWEEPS, BURNIN);
    printf("  PARIS trajectories: %d\n", M_TRAJECTORIES);
    printf("  Seed:              %lu\n", (unsigned long)SEED);

    /* Generate data */
    printf("\nGenerating synthetic data...\n");
    SyntheticData data;
    generate_synthetic_data(&data, T, SEED);

    /* Count true regime transitions */
    int true_trans_counts[16] = {0};
    for (int t = 1; t < T; t++)
    {
        int from = data.z_true[t - 1];
        int to = data.z_true[t];
        true_trans_counts[from * K + to]++;
    }

    printf("True transition counts:\n");
    for (int i = 0; i < K; i++)
    {
        printf("  [");
        for (int j = 0; j < K; j++)
        {
            printf(" %4d", true_trans_counts[i * K + j]);
        }
        printf(" ]\n");
    }

    /* Model parameters - per-regime sigma_vol */
    const double mu_vol_d[4] = {-4.50, -3.67, -2.83, -2.00};
    const double sigma_vol_d[4] = {0.08, 0.267, 0.453, 0.64};
    const double phi_d = 0.97;

    /* Initial uniform transition matrix */
    double init_trans[16];
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            init_trans[i * 4 + j] = (i == j) ? 0.85 : 0.05; /* Slightly sticky init */
        }
    }

    /* Convert observations to double for API */
    double *obs_d = (double *)malloc(T * sizeof(double));
    for (int t = 0; t < T; t++)
    {
        obs_d[t] = (double)data.y[t];
    }

    /* Initialize reference trajectory from data (simple heuristic) */
    int *init_regimes = (int *)malloc(T * sizeof(int));
    double *init_h = (double *)malloc(T * sizeof(double));
    for (int t = 0; t < T; t++)
    {
        /* Simple regime assignment based on observation magnitude */
        float y = data.y[t];
        if (y < -6.0f)
            init_regimes[t] = 0;
        else if (y < -4.0f)
            init_regimes[t] = 1;
        else if (y < -2.0f)
            init_regimes[t] = 2;
        else
            init_regimes[t] = 3;
        init_h[t] = -3.0 + y * 0.1; /* Rough h estimate */
    }

    /*═══════════════════════════════════════════════════════════════════════════
     * VARIANT A: Vanilla PGAS
     *═══════════════════════════════════════════════════════════════════════════*/
    printf("\n═══════════════════════════════════════════════════════════════════\n");
    printf("  VARIANT A: Vanilla PGAS\n");
    printf("═══════════════════════════════════════════════════════════════════\n");

    PGASMKLState *pgas_A = pgas_mkl_alloc(N, T, K, (uint32_t)SEED);
    if (!pgas_A)
    {
        fprintf(stderr, "Failed to allocate PGAS state A\n");
        return 1;
    }

    /* Configure - Updated API: no sigma_h argument */
    pgas_mkl_set_model(pgas_A, init_trans, mu_vol_d, sigma_vol_d, phi_d);
    pgas_mkl_set_transition_prior(pgas_A, 1.0f, 50.0f);
    pgas_mkl_enable_adaptive_kappa(pgas_A, 1);
    pgas_mkl_configure_adaptive_kappa(pgas_A, 30.0f, 150.0f, 0.0f, 0.0f);
    pgas_mkl_load_observations(pgas_A, obs_d, T);
    pgas_mkl_set_reference(pgas_A, init_regimes, init_h, T);

    /* Run sweeps */
    PGASParisEnsemble ens_A;
    pgas_paris_ensemble_init(&ens_A, K);

    float total_accept_A = 0.0f;
    printf("Running %d sweeps...\n", N_SWEEPS);

    for (int s = 0; s < N_SWEEPS; s++)
    {
        float accept = pgas_mkl_gibbs_sweep(pgas_A);
        total_accept_A += accept;

        if (s >= BURNIN)
        {
            /* Manual accumulation since we don't have PGASParisState */
            float trans[16];
            pgas_mkl_get_transitions(pgas_A, trans, K);
            ens_A.n_samples++;
            for (int i = 0; i < K * K; i++)
            {
                ens_A.trans_sum[i] += trans[i];
                ens_A.trans_sum_sq[i] += (double)trans[i] * trans[i];
            }
        }

        if ((s + 1) % 100 == 0)
        {
            printf("  Sweep %d: accept=%.3f, κ=%.1f, chatter=%.2f\n",
                   s + 1, accept,
                   pgas_mkl_get_sticky_kappa(pgas_A),
                   pgas_mkl_get_chatter_ratio(pgas_A));
        }
    }

    float trans_A[16];
    pgas_paris_ensemble_get_mean(&ens_A, trans_A);

    printf("\nVanilla PGAS Results:\n");
    printf("  Avg acceptance: %.3f\n", total_accept_A / N_SWEEPS);
    printf("  Final κ:        %.1f\n", pgas_mkl_get_sticky_kappa(pgas_A));
    printf("  Ensemble size:  %d samples\n", ens_A.n_samples);
    print_matrix("Learned Π (vanilla)", trans_A, K);

    int valid_A = validate_stochastic_matrix(trans_A, K, "trans_A");
    printf("  Valid stochastic: %s\n", valid_A ? "YES" : "NO");
    printf("  Avg diagonal:     %.4f\n", compute_avg_diagonal(trans_A, K));

    pgas_mkl_free(pgas_A);

    /*═══════════════════════════════════════════════════════════════════════════
     * VARIANT B: PGAS-PARIS
     *═══════════════════════════════════════════════════════════════════════════*/
    printf("\n═══════════════════════════════════════════════════════════════════\n");
    printf("  VARIANT B: PGAS-PARIS\n");
    printf("═══════════════════════════════════════════════════════════════════\n");

    /* Fresh PGAS state with same seed */
    PGASMKLState *pgas_B = pgas_mkl_alloc(N, T, K, (uint32_t)SEED);
    if (!pgas_B)
    {
        fprintf(stderr, "Failed to allocate PGAS state B\n");
        return 1;
    }

    /* Same configuration - Updated API: no sigma_h argument */
    pgas_mkl_set_model(pgas_B, init_trans, mu_vol_d, sigma_vol_d, phi_d);
    pgas_mkl_set_transition_prior(pgas_B, 1.0f, 50.0f);
    pgas_mkl_enable_adaptive_kappa(pgas_B, 1);
    pgas_mkl_configure_adaptive_kappa(pgas_B, 30.0f, 150.0f, 0.0f, 0.0f);
    pgas_mkl_load_observations(pgas_B, obs_d, T);
    pgas_mkl_set_reference(pgas_B, init_regimes, init_h, T);

    /* Wrap with PARIS */
    PGASParisState *pp = pgas_paris_alloc(pgas_B, M_TRAJECTORIES);
    if (!pp)
    {
        fprintf(stderr, "Failed to allocate PGAS-PARIS state\n");
        pgas_mkl_free(pgas_B);
        return 1;
    }

    /* Run sweeps */
    PGASParisEnsemble ens_B;
    pgas_paris_ensemble_init(&ens_B, K);

    float total_accept_B = 0.0f;
    float total_diversity = 0.0f;
    printf("Running %d sweeps...\n", N_SWEEPS);

    for (int s = 0; s < N_SWEEPS; s++)
    {
        float accept = pgas_paris_gibbs_sweep(pp);
        total_accept_B += accept;

        if (s >= BURNIN)
        {
            pgas_paris_ensemble_accumulate(&ens_B, pp);
            total_diversity += pgas_paris_get_trajectory_diversity(pp);
        }

        if ((s + 1) % 100 == 0)
        {
            printf("  Sweep %d: accept=%.3f, κ=%.1f, chatter=%.2f, diversity=%.1f%%\n",
                   s + 1, accept,
                   pgas_paris_get_sticky_kappa(pp),
                   pgas_paris_get_chatter_ratio(pp),
                   pgas_paris_get_trajectory_diversity(pp) * 100.0f);
        }
    }

    float trans_B[16];
    pgas_paris_ensemble_get_mean(&ens_B, trans_B);

    printf("\nPGAS-PARIS Results:\n");
    printf("  Avg acceptance:   %.3f\n", total_accept_B / N_SWEEPS);
    printf("  Final κ:          %.1f\n", pgas_paris_get_sticky_kappa(pp));
    printf("  Ensemble size:    %d samples\n", ens_B.n_samples);
    printf("  Avg diversity:    %.1f%%\n", (total_diversity / (N_SWEEPS - BURNIN)) * 100.0f);
    printf("  Path degeneracy:  %.1f%%\n", pgas_paris_get_path_degeneracy(pp) * 100.0f);
    print_matrix("Learned Π (PARIS)", trans_B, K);

    int valid_B = validate_stochastic_matrix(trans_B, K, "trans_B");
    printf("  Valid stochastic: %s\n", valid_B ? "YES" : "NO");
    printf("  Avg diagonal:     %.4f\n", compute_avg_diagonal(trans_B, K));

    /* Print PARIS diagnostics */
    printf("\n");
    pgas_paris_print_diagnostics(pp);

    pgas_paris_free(pp);
    pgas_mkl_free(pgas_B);

    /*═══════════════════════════════════════════════════════════════════════════
     * COMPARISON
     *═══════════════════════════════════════════════════════════════════════════*/
    printf("\n═══════════════════════════════════════════════════════════════════\n");
    printf("  COMPARISON\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");

    float frob = frobenius_distance(trans_A, trans_B, K);

    printf("Frobenius distance (A vs B): %.4f\n", frob);
    printf("\nElement-wise difference (B - A):\n");
    for (int i = 0; i < K; i++)
    {
        printf("  [");
        for (int j = 0; j < K; j++)
        {
            float diff = trans_B[i * K + j] - trans_A[i * K + j];
            printf(" %+.4f", diff);
        }
        printf(" ]\n");
    }

    /*═══════════════════════════════════════════════════════════════════════════
     * VALIDATION CHECKS
     *═══════════════════════════════════════════════════════════════════════════*/
    printf("\n═══════════════════════════════════════════════════════════════════\n");
    printf("  VALIDATION SUMMARY\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");

    int all_pass = 1;

    /* Check 1: Valid stochastic matrices */
    printf("✓ Valid stochastic matrices: ");
    if (valid_A && valid_B)
    {
        printf("PASS\n");
    }
    else
    {
        printf("FAIL\n");
        all_pass = 0;
    }

    /* Check 2: Trajectory diversity > 0 */
    float avg_diversity = total_diversity / (N_SWEEPS - BURNIN);
    printf("✓ PARIS trajectory diversity > 0: ");
    if (avg_diversity > 0.01f)
    {
        printf("PASS (%.1f%%)\n", avg_diversity * 100.0f);
    }
    else
    {
        printf("FAIL (%.1f%%) - PARIS backward pass may be broken\n", avg_diversity * 100.0f);
        all_pass = 0;
    }

    /* Check 3: Matrices not identical */
    printf("✓ Matrices not identical: ");
    if (frob > 0.001f)
    {
        printf("PASS (Frob=%.4f)\n", frob);
    }
    else
    {
        printf("FAIL (Frob=%.4f) - PARIS may not be used\n", frob);
        all_pass = 0;
    }

    /* Check 4: Matrices reasonably similar */
    printf("✓ Matrices reasonably similar (Frob < 0.3): ");
    if (frob < 0.3f)
    {
        printf("PASS (Frob=%.4f)\n", frob);
    }
    else
    {
        printf("WARNING (Frob=%.4f) - may indicate bug\n", frob);
        /* Don't fail, just warn */
    }

    /* Check 5: Similar avg diagonal */
    float diag_A = compute_avg_diagonal(trans_A, K);
    float diag_B = compute_avg_diagonal(trans_B, K);
    printf("✓ Similar stickiness: ");
    if (fabsf(diag_A - diag_B) < 0.05f)
    {
        printf("PASS (A=%.3f, B=%.3f)\n", diag_A, diag_B);
    }
    else
    {
        printf("WARNING (A=%.3f, B=%.3f)\n", diag_A, diag_B);
    }

    printf("\n═══════════════════════════════════════════════════════════════════\n");
    if (all_pass)
    {
        printf("  ALL CHECKS PASSED - PGAS-PARIS integration looks correct\n");
    }
    else
    {
        printf("  SOME CHECKS FAILED - investigate issues above\n");
    }
    printf("═══════════════════════════════════════════════════════════════════\n");

    /* Cleanup */
    free(obs_d);
    free(init_regimes);
    free(init_h);
    free_synthetic_data(&data);

    return all_pass ? 0 : 1;
}