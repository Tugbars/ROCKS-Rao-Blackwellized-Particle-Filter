/*═══════════════════════════════════════════════════════════════════════════════
 * Regime Parameter Learning: PGAS vs PGAS-PARIS Comparison
 *
 * Research test comparing two approaches to learning μ_vol:
 *
 *   VARIANT A: Vanilla PGAS
 *     - Single reference trajectory
 *     - Subject to path degeneracy
 *     - Lower computational cost
 *
 *   VARIANT B: PGAS-PARIS
 *     - M smoothed trajectories (Rao-Blackwellized)
 *     - Fixes path degeneracy
 *     - Lower variance estimates
 *
 * Both use same:
 *   - Synthetic data (known ground truth)
 *   - Prior configuration
 *   - Random seed
 *   - Number of sweeps
 *
 * Metrics compared:
 *   - μ_vol RMSE vs ground truth
 *   - μ_vol estimation variance
 *   - Convergence speed
 *   - Π learning quality
 *
 * Usage:
 *   ./test_regime_learning [seed]
 *
 *═══════════════════════════════════════════════════════════════════════════════*/

#include "pgas_mkl.h"
#include "pgas_paris.h"
#include "mkl_tuning.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include <mkl.h>
#include <mkl_vsl.h>

/*═══════════════════════════════════════════════════════════════════════════════
 * CONFIGURATION
 *═══════════════════════════════════════════════════════════════════════════════*/

/* Data generation */
#define DEFAULT_SEED 42
#define T_DATA 2000
#define K_REGIMES 4
#define N_PARTICLES 64

/* MCMC settings */
#define N_SWEEPS 500
#define BURNIN 200
#define M_TRAJECTORIES 8

/* Ground truth parameters
 *
 * NOTE: OCSN Mixture (Omori et al. 2007)
 * The model is: y_t = log(r_t²) = h_t + log(η_t²)
 * where η_t ~ N(0,1), so E[log(η²)] ≈ -1.27
 *
 * OCSN mixture means are pre-adjusted for this offset.
 * Our synthetic generator produces h_t directly, and the OCSN
 * likelihood correctly handles the log(η²) distribution.
 */
static const float TRUE_MU_VOL[4] = {-4.50f, -3.67f, -2.83f, -2.00f};
static const float TRUE_SIGMA_VOL[4] = {0.08f, 0.267f, 0.453f, 0.64f};
static const float TRUE_PHI = 0.97f;
static const float TRUE_SIGMA_H = 0.15f;

/* True transition matrix (sticky) */
static const double TRUE_TRANS[16] = {
    0.96, 0.02, 0.01, 0.01,
    0.02, 0.95, 0.02, 0.01,
    0.01, 0.03, 0.94, 0.02,
    0.01, 0.02, 0.03, 0.94};

/*═══════════════════════════════════════════════════════════════════════════════
 * SYNTHETIC DATA GENERATION
 *═══════════════════════════════════════════════════════════════════════════════*/

typedef struct
{
    int T;
    int *true_regimes; /* Ground truth regimes */
    float *true_h;     /* Ground truth log-volatility */
    float *y;          /* Observations */
} SyntheticData;

static void generate_synthetic_data(SyntheticData *data, int T, uint32_t seed)
{
    data->T = T;
    data->true_regimes = (int *)malloc(T * sizeof(int));
    data->true_h = (float *)malloc(T * sizeof(float));
    data->y = (float *)malloc(T * sizeof(float));

    VSLStreamStatePtr stream;
    vslNewStream(&stream, VSL_BRNG_MT19937, seed);

    /* Initial state */
    data->true_regimes[0] = 1; /* Start in regime 1 */
    data->true_h[0] = TRUE_MU_VOL[1];

    float *uniform = (float *)malloc(T * sizeof(float));
    float *normal = (float *)malloc(T * sizeof(float));
    float *obs_noise = (float *)malloc(T * sizeof(float));

    vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, T, uniform, 0.0f, 1.0f);
    vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, stream, T, normal, 0.0f, TRUE_SIGMA_H);
    vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, stream, T, obs_noise, 0.0f, 1.0f);

    /* Generate sequence */
    for (int t = 1; t < T; t++)
    {
        /* Sample regime transition */
        int prev_regime = data->true_regimes[t - 1];
        float u = uniform[t];
        float cumsum = 0.0f;
        int new_regime = K_REGIMES - 1;

        for (int k = 0; k < K_REGIMES; k++)
        {
            cumsum += (float)TRUE_TRANS[prev_regime * K_REGIMES + k];
            if (u < cumsum)
            {
                new_regime = k;
                break;
            }
        }
        data->true_regimes[t] = new_regime;

        /* AR(1) dynamics for h */
        float mu_k = TRUE_MU_VOL[new_regime];
        data->true_h[t] = mu_k * (1.0f - TRUE_PHI) + TRUE_PHI * data->true_h[t - 1] + normal[t];
    }

    /* Generate observations */
    for (int t = 0; t < T; t++)
    {
        float vol = expf(0.5f * data->true_h[t]);
        data->y[t] = vol * obs_noise[t];
    }

    free(uniform);
    free(normal);
    free(obs_noise);
    vslDeleteStream(&stream);
}

static void free_synthetic_data(SyntheticData *data)
{
    free(data->true_regimes);
    free(data->true_h);
    free(data->y);
}

/*═══════════════════════════════════════════════════════════════════════════════
 * ACCUMULATOR FOR μ_vol POSTERIOR
 *═══════════════════════════════════════════════════════════════════════════════*/

typedef struct
{
    int K;
    int n_samples;
    double mu_sum[PGAS_MKL_MAX_REGIMES];
    double mu_sum_sq[PGAS_MKL_MAX_REGIMES];
    double trans_sum[PGAS_MKL_MAX_REGIMES * PGAS_MKL_MAX_REGIMES];
} RegimeLearningAccumulator;

static void accumulator_init(RegimeLearningAccumulator *acc, int K)
{
    memset(acc, 0, sizeof(RegimeLearningAccumulator));
    acc->K = K;
}

static void accumulator_add_pgas(RegimeLearningAccumulator *acc, const PGASMKLState *pgas)
{
    int K = acc->K;
    acc->n_samples++;

    for (int k = 0; k < K; k++)
    {
        float mu = pgas->model.mu_vol[k];
        acc->mu_sum[k] += mu;
        acc->mu_sum_sq[k] += mu * mu;
    }

    for (int i = 0; i < K * K; i++)
    {
        acc->trans_sum[i] += pgas->model.trans[i];
    }
}

static void accumulator_add_paris(RegimeLearningAccumulator *acc, const PGASParisState *pp)
{
    accumulator_add_pgas(acc, pp->pgas);
}

static float accumulator_get_mu_mean(const RegimeLearningAccumulator *acc, int k)
{
    if (acc->n_samples == 0)
        return 0.0f;
    return (float)(acc->mu_sum[k] / acc->n_samples);
}

static float accumulator_get_mu_std(const RegimeLearningAccumulator *acc, int k)
{
    if (acc->n_samples < 2)
        return 0.0f;
    double mean = acc->mu_sum[k] / acc->n_samples;
    double var = (acc->mu_sum_sq[k] / acc->n_samples) - mean * mean;
    if (var < 0)
        var = 0;
    return (float)sqrt(var);
}

static float accumulator_get_trans_mean(const RegimeLearningAccumulator *acc, int i, int j)
{
    if (acc->n_samples == 0)
        return 0.0f;
    return (float)(acc->trans_sum[i * acc->K + j] / acc->n_samples);
}

/*═══════════════════════════════════════════════════════════════════════════════
 * VANILLA PGAS μ_vol LEARNING
 *
 * Uses single reference trajectory for sufficient statistics.
 * Subject to path degeneracy.
 *═══════════════════════════════════════════════════════════════════════════════*/

typedef struct
{
    float mu_prior_mean[PGAS_MKL_MAX_REGIMES];
    float mu_prior_var[PGAS_MKL_MAX_REGIMES];
    int enforce_ordering;
} PGASRegimePrior;

static void pgas_regime_prior_init(PGASRegimePrior *prior, int K)
{
    for (int k = 0; k < K; k++)
    {
        prior->mu_prior_mean[k] = -3.0f;
        prior->mu_prior_var[k] = 4.0f;
    }
    prior->enforce_ordering = 1;
}

/* Collect sufficient stats from single reference trajectory */
static void pgas_collect_regime_stats_single(
    const PGASMKLState *pgas,
    double *n_k,
    double *sum_resid_k)
{
    const int K = pgas->K;
    const int T = pgas->T;
    const float phi = pgas->model.phi;

    for (int k = 0; k < K; k++)
    {
        n_k[k] = 0.0;
        sum_resid_k[k] = 0.0;
    }

    /* Use reference trajectory */
    for (int t = 1; t < T; t++)
    {
        int z_t = pgas->ref_regimes[t];
        float h_t = (float)pgas->ref_h[t];
        float h_prev = (float)pgas->ref_h[t - 1];
        float resid = h_t - phi * h_prev;

        n_k[z_t] += 1.0;
        sum_resid_k[z_t] += resid;
    }
}

/* Sample μ_vol using single-trajectory stats */
static void pgas_sample_mu_vol_single(
    PGASMKLState *pgas,
    const PGASRegimePrior *prior)
{
    const int K = pgas->K;
    const float phi = pgas->model.phi;
    const float sigma_h = pgas->model.sigma_h;
    const float sigma_h_sq = sigma_h * sigma_h;
    const float one_minus_phi = 1.0f - phi;
    const float one_minus_phi_sq = one_minus_phi * one_minus_phi;

    VSLStreamStatePtr stream = (VSLStreamStatePtr)pgas->rng.stream;

    double n_k[PGAS_MKL_MAX_REGIMES];
    double sum_resid_k[PGAS_MKL_MAX_REGIMES];

    pgas_collect_regime_stats_single(pgas, n_k, sum_resid_k);

    for (int k = 0; k < K; k++)
    {
        float m0 = prior->mu_prior_mean[k];
        float s0_sq = prior->mu_prior_var[k];

        if (n_k[k] < 1.0)
        {
            float sample;
            vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, stream, 1,
                          &sample, m0, sqrtf(s0_sq));
            pgas->model.mu_vol[k] = sample;
            pgas->model.mu_shift[k] = sample * one_minus_phi;
            continue;
        }

        double precision_prior = 1.0 / s0_sq;
        double precision_data = n_k[k] * one_minus_phi_sq / sigma_h_sq;
        double precision_post = precision_prior + precision_data;
        double var_post = 1.0 / precision_post;

        double data_contribution = one_minus_phi * sum_resid_k[k] / sigma_h_sq;
        double prior_contribution = m0 / s0_sq;
        double mean_post = var_post * (prior_contribution + data_contribution);

        float sample;
        vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, stream, 1,
                      &sample, (float)mean_post, sqrtf((float)var_post));

        pgas->model.mu_vol[k] = sample;
        pgas->model.mu_shift[k] = sample * one_minus_phi;
    }
}

/* Enforce μ ordering to prevent label switching */
static void pgas_enforce_mu_ordering_single(PGASMKLState *pgas)
{
    const int K = pgas->K;

    int perm[PGAS_MKL_MAX_REGIMES];
    for (int k = 0; k < K; k++)
        perm[k] = k;

    /* Insertion sort */
    for (int i = 1; i < K; i++)
    {
        int key = perm[i];
        float key_mu = pgas->model.mu_vol[key];
        int j = i - 1;
        while (j >= 0 && pgas->model.mu_vol[perm[j]] > key_mu)
        {
            perm[j + 1] = perm[j];
            j--;
        }
        perm[j + 1] = key;
    }

    int already_sorted = 1;
    for (int k = 0; k < K; k++)
    {
        if (perm[k] != k)
        {
            already_sorted = 0;
            break;
        }
    }
    if (already_sorted)
        return;

    float new_mu_vol[PGAS_MKL_MAX_REGIMES];
    float new_sigma_vol[PGAS_MKL_MAX_REGIMES];
    float new_mu_shift[PGAS_MKL_MAX_REGIMES];
    float new_trans[PGAS_MKL_MAX_REGIMES * PGAS_MKL_MAX_REGIMES];
    float new_log_trans[PGAS_MKL_MAX_REGIMES * PGAS_MKL_MAX_REGIMES];

    for (int k = 0; k < K; k++)
    {
        new_mu_vol[k] = pgas->model.mu_vol[perm[k]];
        new_sigma_vol[k] = pgas->model.sigma_vol[perm[k]];
        new_mu_shift[k] = pgas->model.mu_shift[perm[k]];
    }

    for (int i = 0; i < K; i++)
    {
        for (int j = 0; j < K; j++)
        {
            new_trans[i * K + j] = pgas->model.trans[perm[i] * K + perm[j]];
            new_log_trans[i * K + j] = pgas->model.log_trans[perm[i] * K + perm[j]];
        }
    }

    for (int k = 0; k < K; k++)
    {
        pgas->model.mu_vol[k] = new_mu_vol[k];
        pgas->model.sigma_vol[k] = new_sigma_vol[k];
        pgas->model.mu_shift[k] = new_mu_shift[k];
    }
    for (int i = 0; i < K * K; i++)
    {
        pgas->model.trans[i] = new_trans[i];
        pgas->model.log_trans[i] = new_log_trans[i];
    }
}

/* Full vanilla PGAS Gibbs sweep with μ_vol learning */
static float pgas_gibbs_sweep_with_mu(PGASMKLState *pgas, const PGASRegimePrior *prior)
{
    /* 1. CSMC forward */
    float accept = pgas_mkl_csmc_sweep(pgas);

    /* 2. Sample Π */
    pgas_mkl_sample_transitions(pgas);

    /* 3. Sample μ_vol */
    pgas_sample_mu_vol_single(pgas, prior);

    /* 4. Enforce ordering */
    if (prior->enforce_ordering)
    {
        pgas_enforce_mu_ordering_single(pgas);
    }

    return accept;
}

/*═══════════════════════════════════════════════════════════════════════════════
 * UTILITY FUNCTIONS
 *═══════════════════════════════════════════════════════════════════════════════*/

static float compute_mu_rmse(const float *learned, const float *true_vals, int K)
{
    float sum_sq = 0.0f;
    for (int k = 0; k < K; k++)
    {
        float diff = learned[k] - true_vals[k];
        sum_sq += diff * diff;
    }
    return sqrtf(sum_sq / K);
}

static void print_mu_comparison(const char *name, const float *learned,
                                const float *true_vals, int K)
{
    printf("\n%s:\n", name);
    printf("  Regime    Learned     True       Diff\n");
    printf("  ─────────────────────────────────────\n");

    for (int k = 0; k < K; k++)
    {
        float diff = learned[k] - true_vals[k];
        printf("     %d      %6.3f    %6.3f    %+6.3f\n",
               k, learned[k], true_vals[k], diff);
    }

    printf("  ─────────────────────────────────────\n");
    printf("  RMSE: %.4f\n", compute_mu_rmse(learned, true_vals, K));
}

static void print_trans_comparison(const char *name, const RegimeLearningAccumulator *acc, int K)
{
    printf("\n%s:\n", name);
    for (int i = 0; i < K; i++)
    {
        printf("  [");
        for (int j = 0; j < K; j++)
        {
            printf(" %5.3f", accumulator_get_trans_mean(acc, i, j));
        }
        printf(" ]\n");
    }
}

/*═══════════════════════════════════════════════════════════════════════════════
 * MAIN TEST
 *═══════════════════════════════════════════════════════════════════════════════*/

int main(int argc, char **argv)
{

    mkl_tuning_init(8, 0);  /* 8 P-cores, quiet mode */
    uint32_t seed = DEFAULT_SEED;
    if (argc > 1)
    {
        seed = (uint32_t)atoi(argv[1]);
    }

    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("  Regime Parameter Learning: PGAS vs PGAS-PARIS\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");

    printf("Configuration:\n");
    printf("  K (regimes):       %d\n", K_REGIMES);
    printf("  N (particles):     %d\n", N_PARTICLES);
    printf("  T (timesteps):     %d\n", T_DATA);
    printf("  Sweeps:            %d (burnin: %d)\n", N_SWEEPS, BURNIN);
    printf("  PARIS trajectories: %d\n", M_TRAJECTORIES);
    printf("  Seed:              %u\n", seed);

    printf("\nGround Truth μ_vol:\n");
    printf("  [");
    for (int k = 0; k < K_REGIMES; k++)
    {
        printf(" %5.2f", TRUE_MU_VOL[k]);
    }
    printf(" ]\n");

    /* Generate data */
    printf("\nGenerating synthetic data...\n");
    SyntheticData data;
    generate_synthetic_data(&data, T_DATA, seed);

    /* Convert for API */
    double *obs_d = (double *)malloc(T_DATA * sizeof(double));
    double init_trans[16];
    double mu_vol_d[4], sigma_vol_d[4];

    /* FIX 1: Log-transform observations for OCSN
     * PGAS expects y = log(r²), not raw returns
     * Without this, model sees ~0.01 instead of ~-9.2 */
    for (int t = 0; t < T_DATA; t++)
    {
        float r = data.y[t];
        obs_d[t] = log(r * r + 1e-10); /* log(r²) with epsilon for safety */
    }

    /* Initial transition matrix (rows sum to 1.0) */
    for (int i = 0; i < K_REGIMES; i++)
    {
        for (int j = 0; j < K_REGIMES; j++)
        {
            init_trans[i * K_REGIMES + j] = (i == j) ? 0.85 : 0.05;
        }
    }
    for (int k = 0; k < 4; k++)
    {
        mu_vol_d[k] = -3.0; /* Start with uninformative init */
        sigma_vol_d[k] = TRUE_SIGMA_VOL[k];
    }

    /* Initialize reference trajectory */
    int *init_regimes = (int *)malloc(T_DATA * sizeof(int));
    double *init_h = (double *)malloc(T_DATA * sizeof(double));
    for (int t = 0; t < T_DATA; t++)
    {
        init_regimes[t] = data.true_regimes[t]; /* Warm start with truth */
        init_h[t] = data.true_h[t];
    }

    /*═══════════════════════════════════════════════════════════════════════════
     * VARIANT A: Vanilla PGAS with μ_vol Learning
     *═══════════════════════════════════════════════════════════════════════════*/
    printf("\n═══════════════════════════════════════════════════════════════════\n");
    printf("  VARIANT A: Vanilla PGAS + μ_vol Learning\n");
    printf("═══════════════════════════════════════════════════════════════════\n");

    PGASMKLState *pgas_A = pgas_mkl_alloc(N_PARTICLES, T_DATA, K_REGIMES, seed);
    pgas_mkl_set_model(pgas_A, init_trans, mu_vol_d, sigma_vol_d, TRUE_PHI, TRUE_SIGMA_H);
    pgas_mkl_set_transition_prior(pgas_A, 1.0f, 50.0f);
    pgas_mkl_enable_adaptive_kappa(pgas_A, 1);
    pgas_mkl_configure_adaptive_kappa(pgas_A, 20.0f, 150.0f, 0.0f, 0.0f);
    pgas_mkl_load_observations(pgas_A, obs_d, T_DATA);
    pgas_mkl_set_reference(pgas_A, init_regimes, init_h, T_DATA);

    PGASRegimePrior prior_A;
    pgas_regime_prior_init(&prior_A, K_REGIMES);

    RegimeLearningAccumulator acc_A;
    accumulator_init(&acc_A, K_REGIMES);

    printf("Running %d sweeps...\n", N_SWEEPS);
    clock_t start_A = clock();

    for (int s = 0; s < N_SWEEPS; s++)
    {
        float accept = pgas_gibbs_sweep_with_mu(pgas_A, &prior_A);

        if (s >= BURNIN)
        {
            accumulator_add_pgas(&acc_A, pgas_A);
        }

        if ((s + 1) % 100 == 0)
        {
            printf("  Sweep %d: accept=%.3f, κ=%.1f, μ=[%.2f, %.2f, %.2f, %.2f]\n",
                   s + 1, accept,
                   pgas_mkl_get_sticky_kappa(pgas_A),
                   pgas_A->model.mu_vol[0], pgas_A->model.mu_vol[1],
                   pgas_A->model.mu_vol[2], pgas_A->model.mu_vol[3]);
        }
    }

    clock_t end_A = clock();
    double time_A = (double)(end_A - start_A) / CLOCKS_PER_SEC;

    float mu_learned_A[4];
    for (int k = 0; k < 4; k++)
    {
        mu_learned_A[k] = accumulator_get_mu_mean(&acc_A, k);
    }

    printf("\nVanilla PGAS Results:\n");
    printf("  Time:            %.2f sec\n", time_A);
    printf("  Post-burnin:     %d samples\n", acc_A.n_samples);
    print_mu_comparison("Learned μ_vol (Vanilla PGAS)", mu_learned_A, TRUE_MU_VOL, K_REGIMES);

    printf("\nμ_vol Posterior Std:\n  [");
    for (int k = 0; k < K_REGIMES; k++)
    {
        printf(" %.4f", accumulator_get_mu_std(&acc_A, k));
    }
    printf(" ]\n");

    print_trans_comparison("Learned Π (Vanilla PGAS)", &acc_A, K_REGIMES);

    pgas_mkl_free(pgas_A);

    /*═══════════════════════════════════════════════════════════════════════════
     * VARIANT B: PGAS-PARIS with μ_vol Learning
     *═══════════════════════════════════════════════════════════════════════════*/
    printf("\n═══════════════════════════════════════════════════════════════════\n");
    printf("  VARIANT B: PGAS-PARIS + μ_vol Learning (Rao-Blackwellized)\n");
    printf("═══════════════════════════════════════════════════════════════════\n");

    /* Fresh PGAS state with same seed */
    PGASMKLState *pgas_B = pgas_mkl_alloc(N_PARTICLES, T_DATA, K_REGIMES, seed);
    pgas_mkl_set_model(pgas_B, init_trans, mu_vol_d, sigma_vol_d, TRUE_PHI, TRUE_SIGMA_H);
    pgas_mkl_set_transition_prior(pgas_B, 1.0f, 50.0f);
    pgas_mkl_enable_adaptive_kappa(pgas_B, 1);
    pgas_mkl_configure_adaptive_kappa(pgas_B, 20.0f, 150.0f, 0.0f, 0.0f);
    pgas_mkl_load_observations(pgas_B, obs_d, T_DATA);
    pgas_mkl_set_reference(pgas_B, init_regimes, init_h, T_DATA);

    PGASParisState *pp = pgas_paris_alloc(pgas_B, M_TRAJECTORIES);

    PGASParisRegimePrior prior_B;
    pgas_paris_regime_prior_init(&prior_B, K_REGIMES);
    prior_B.learn_mu = 1;
    prior_B.learn_sigma = 0;
    prior_B.enforce_ordering = 1;

    RegimeLearningAccumulator acc_B;
    accumulator_init(&acc_B, K_REGIMES);

    printf("Running %d sweeps...\n", N_SWEEPS);
    clock_t start_B = clock();

    for (int s = 0; s < N_SWEEPS; s++)
    {
        float accept = pgas_paris_gibbs_sweep_full(pp, &prior_B);

        if (s >= BURNIN)
        {
            accumulator_add_paris(&acc_B, pp);
        }

        if ((s + 1) % 100 == 0)
        {
            printf("  Sweep %d: accept=%.3f, κ=%.1f, div=%.0f%%, μ=[%.2f, %.2f, %.2f, %.2f]\n",
                   s + 1, accept,
                   pgas_paris_get_sticky_kappa(pp),
                   pgas_paris_get_trajectory_diversity(pp) * 100.0f,
                   pgas_B->model.mu_vol[0], pgas_B->model.mu_vol[1],
                   pgas_B->model.mu_vol[2], pgas_B->model.mu_vol[3]);
        }
    }

    clock_t end_B = clock();
    double time_B = (double)(end_B - start_B) / CLOCKS_PER_SEC;

    float mu_learned_B[4];
    for (int k = 0; k < 4; k++)
    {
        mu_learned_B[k] = accumulator_get_mu_mean(&acc_B, k);
    }

    printf("\nPGAS-PARIS Results:\n");
    printf("  Time:            %.2f sec\n", time_B);
    printf("  Post-burnin:     %d samples\n", acc_B.n_samples);
    print_mu_comparison("Learned μ_vol (PGAS-PARIS)", mu_learned_B, TRUE_MU_VOL, K_REGIMES);

    printf("\nμ_vol Posterior Std:\n  [");
    for (int k = 0; k < K_REGIMES; k++)
    {
        printf(" %.4f", accumulator_get_mu_std(&acc_B, k));
    }
    printf(" ]\n");

    print_trans_comparison("Learned Π (PGAS-PARIS)", &acc_B, K_REGIMES);

    pgas_paris_free(pp);
    pgas_mkl_free(pgas_B);

    /*═══════════════════════════════════════════════════════════════════════════
     * COMPARISON SUMMARY
     *═══════════════════════════════════════════════════════════════════════════*/
    printf("\n═══════════════════════════════════════════════════════════════════\n");
    printf("  COMPARISON SUMMARY\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");

    float rmse_A = compute_mu_rmse(mu_learned_A, TRUE_MU_VOL, K_REGIMES);
    float rmse_B = compute_mu_rmse(mu_learned_B, TRUE_MU_VOL, K_REGIMES);

    printf("                        Vanilla PGAS    PGAS-PARIS    Winner\n");
    printf("───────────────────────────────────────────────────────────────\n");
    printf("μ_vol RMSE              %8.4f        %8.4f      %s\n",
           rmse_A, rmse_B, rmse_A < rmse_B ? "PGAS" : "PARIS");

    float avg_std_A = 0, avg_std_B = 0;
    for (int k = 0; k < K_REGIMES; k++)
    {
        avg_std_A += accumulator_get_mu_std(&acc_A, k);
        avg_std_B += accumulator_get_mu_std(&acc_B, k);
    }
    avg_std_A /= K_REGIMES;
    avg_std_B /= K_REGIMES;

    printf("μ_vol Avg Std           %8.4f        %8.4f      %s\n",
           avg_std_A, avg_std_B, avg_std_A < avg_std_B ? "PGAS" : "PARIS");
    printf("Time (sec)              %8.2f        %8.2f      %s\n",
           time_A, time_B, time_A < time_B ? "PGAS" : "PARIS");
    printf("───────────────────────────────────────────────────────────────\n");

    printf("\nElement-wise μ_vol comparison:\n");
    printf("  Regime   True    PGAS    PARIS   PGAS err  PARIS err  Winner\n");
    printf("  ─────────────────────────────────────────────────────────────\n");

    for (int k = 0; k < K_REGIMES; k++)
    {
        float err_A = fabsf(mu_learned_A[k] - TRUE_MU_VOL[k]);
        float err_B = fabsf(mu_learned_B[k] - TRUE_MU_VOL[k]);
        printf("     %d    %5.2f   %5.2f   %5.2f    %5.3f     %5.3f     %s\n",
               k, TRUE_MU_VOL[k], mu_learned_A[k], mu_learned_B[k],
               err_A, err_B, err_A < err_B ? "PGAS" : "PARIS");
    }

    printf("\n═══════════════════════════════════════════════════════════════════\n");
    if (rmse_B < rmse_A && avg_std_B < avg_std_A)
    {
        printf("  ★ PGAS-PARIS wins on both accuracy AND variance\n");
    }
    else if (rmse_B < rmse_A)
    {
        printf("  ★ PGAS-PARIS wins on accuracy (lower RMSE)\n");
    }
    else if (avg_std_B < avg_std_A)
    {
        printf("  ★ PGAS-PARIS wins on variance (lower posterior std)\n");
    }
    else
    {
        printf("  ★ Vanilla PGAS wins on this run\n");
    }
    printf("═══════════════════════════════════════════════════════════════════\n");

    /* Cleanup */
    free(obs_d);
    free(init_regimes);
    free(init_h);
    free_synthetic_data(&data);

    return 0;
}