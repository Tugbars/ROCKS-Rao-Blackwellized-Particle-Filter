/*═══════════════════════════════════════════════════════════════════════════════
 * Full Parameter Learning Test: μ_vol + σ_h + φ
 *
 * Research test that attempts to learn ALL model parameters:
 *   - μ_vol[k]: Per-regime log-volatility mean (conjugate Normal) ✅ Works
 *   - σ_h: Shared AR innovation std (Inverse-Gamma) ⚠️ Confounded
 *   - φ: AR persistence coefficient (Metropolis-Hastings) ⚠️ Needs tuning
 *
 * Purpose: Document which parameters are identifiable vs confounded.
 *
 * Expected Results:
 *   - μ_vol: RMSE < 0.20 (works reliably)
 *   - φ: May recover with good MH tuning, but adds heuristics
 *   - σ_h: Biased high - confounded with OCSN observation noise
 *
 * Model:
 *   z_t ~ Categorical(Π[z_{t-1},:])                  [regime transitions]
 *   h_t = μ_{z_t}(1-φ) + φ*h_{t-1} + σ_h*ε_t        [AR(1) dynamics]
 *   y_t = h_t + OCSN_noise                           [observation]
 *
 * Conclusion: For production, use test_regime_learning (μ_vol only).
 *             Fix φ and σ_h from domain knowledge.
 *
 * Usage:
 *   ./test_full_learning [seed]
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

#ifdef _WIN32
#include <windows.h>
static double get_time_ms(void)
{
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart * 1000.0 / (double)freq.QuadPart;
}
#else
#include <sys/time.h>
static double get_time_ms(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}
#endif

/*═══════════════════════════════════════════════════════════════════════════════
 * CONFIGURATION
 *
 * Practical approach: Learn only μ_vol (validated RMSE < 0.1).
 * Fix φ and σ_h from calibration - they're structural, not regime-specific.
 *═══════════════════════════════════════════════════════════════════════════════*/

#define DEFAULT_SEED 42
#define T_DATA 2000
#define K_REGIMES 4
#define N_PARTICLES 32 /* Same as test_regime_learning */
#define N_SWEEPS 500
#define BURNIN 200
#define M_TRAJECTORIES 8

/* Ground truth parameters - original realistic spread */
static const float TRUE_MU_VOL[4] = {-4.50f, -3.67f, -2.83f, -2.00f};
static const float TRUE_SIGMA_VOL[4] = {0.08f, 0.267f, 0.453f, 0.64f};
static const float TRUE_PHI = 0.97f;
static const float TRUE_SIGMA_H = 0.15f;

/* True transition matrix (sticky) - matches test_regime_learning */
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
    int *true_regimes;
    float *true_h;
    float *y;
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
    data->true_regimes[0] = 1;
    data->true_h[0] = TRUE_MU_VOL[1];

    float *uniform = (float *)malloc(T * sizeof(float));
    float *normal = (float *)malloc(T * sizeof(float));

    vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, T, uniform, 0.0f, 1.0f);
    vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, stream, T, normal, 0.0f, TRUE_SIGMA_H);

    /* Generate latent process */
    for (int t = 1; t < T; t++)
    {
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

        float mu_k = TRUE_MU_VOL[new_regime];
        data->true_h[t] = mu_k * (1.0f - TRUE_PHI) + TRUE_PHI * data->true_h[t - 1] + normal[t];
    }

    /* Generate observations: raw returns (matches test_regime_learning) */
    float *obs_noise = (float *)malloc(T * sizeof(float));
    vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, stream, T, obs_noise, 0.0f, 1.0f);
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
 * MAIN
 *═══════════════════════════════════════════════════════════════════════════════*/

int main(int argc, char **argv)
{
    uint32_t seed = DEFAULT_SEED;
    if (argc > 1)
        seed = (uint32_t)atoi(argv[1]);

    /* Initialize MKL tuning */
    mkl_tuning_init(8, 1);

    printf("\n");
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("  Full Parameter Learning: μ_vol + σ_h + φ\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");

    printf("Configuration:\n");
    printf("  K (regimes):       %d\n", K_REGIMES);
    printf("  N (particles):     %d\n", N_PARTICLES);
    printf("  T (timesteps):     %d\n", T_DATA);
    printf("  Sweeps:            %d (burnin: %d)\n", N_SWEEPS, BURNIN);
    printf("  PARIS trajectories: %d\n", M_TRAJECTORIES);
    printf("  Seed:              %u\n\n", seed);

    printf("Ground Truth μ_vol:\n");
    printf("  [ ");
    for (int k = 0; k < K_REGIMES; k++)
        printf("%.2f ", TRUE_MU_VOL[k]);
    printf("]\n\n");

    printf("Ground Truth (structural):\n");
    printf("  φ   = %.3f\n", TRUE_PHI);
    printf("  σ_h = %.3f\n\n", TRUE_SIGMA_H);

    printf("Initial Values (intentionally wrong - to test learning):\n");
    printf("  μ_vol = [-3.0, -3.0, -3.0, -3.0]\n");
    printf("  φ     = 0.90  (true: 0.97)\n");
    printf("  σ_h   = 0.25  (true: 0.15)\n\n");

    /* Generate synthetic data */
    printf("Generating synthetic data...\n\n");
    SyntheticData data;
    generate_synthetic_data(&data, T_DATA, seed);

    /* Convert to double for PGAS API
     * CRITICAL: Data is raw returns, PGAS expects y = log(r²) */
    double *obs_d = (double *)malloc(T_DATA * sizeof(double));
    for (int t = 0; t < T_DATA; t++)
    {
        float r = data.y[t];
        obs_d[t] = log(r * r + 1e-10); /* log(r²) with epsilon */
    }

    /* Initialize with intentionally wrong μ_vol (to test learning)
     * but CORRECT φ and σ_h (since we're not learning them) */
    double init_trans[16];
    for (int i = 0; i < K_REGIMES; i++)
    {
        for (int j = 0; j < K_REGIMES; j++)
        {
            init_trans[i * K_REGIMES + j] = (i == j) ? 0.85 : 0.05;
        }
    }

    double init_mu_vol[4] = {-3.0, -3.0, -3.0, -3.0}; /* Wrong - will be learned */
    double init_sigma_vol[4];
    for (int k = 0; k < K_REGIMES; k++)
    {
        init_sigma_vol[k] = TRUE_SIGMA_VOL[k]; /* Use true values */
    }

    /* Allocate PGAS state - start with WRONG φ and σ_h to test learning */
    float init_phi = 0.90f;     /* Wrong - true is 0.97 */
    float init_sigma_h = 0.25f; /* Wrong - true is 0.15 */

    PGASMKLState *pgas = pgas_mkl_alloc(N_PARTICLES, T_DATA, K_REGIMES, seed);
    pgas_mkl_set_model(pgas, init_trans, init_mu_vol, init_sigma_vol, init_phi, init_sigma_h);
    pgas_mkl_set_transition_prior(pgas, 1.0f, 50.0f);
    pgas_mkl_enable_adaptive_kappa(pgas, 1);
    pgas_mkl_configure_adaptive_kappa(pgas, 20.0f, 150.0f, 0.0f, 0.0f);
    pgas_mkl_load_observations(pgas, obs_d, T_DATA);

    /* Initialize reference with true trajectory */
    double *init_h = (double *)malloc(T_DATA * sizeof(double));
    for (int t = 0; t < T_DATA; t++)
        init_h[t] = data.true_h[t];
    pgas_mkl_set_reference(pgas, data.true_regimes, init_h, T_DATA);

    /* Allocate PGAS-PARIS state */
    PGASParisState *pp = pgas_paris_alloc(pgas, M_TRAJECTORIES);

    /* Configure prior with ALL learning enabled */
    PGASParisRegimePrior prior;
    pgas_paris_regime_prior_init(&prior, K_REGIMES);

    /* FULL LEARNING: Attempt to learn ALL parameters
     *
     * This test explores learning everything to document:
     *   ✅ μ_vol: Learns well
     *   ⚠️ φ: Learns but requires MH tuning
     *   ❌ σ_h: Confounded with OCSN noise
     *
     * Results show why production uses only μ_vol learning.
     */
    prior.learn_mu = 1;
    prior.learn_sigma = 0;   /* Conflicts with σ_h */
    prior.learn_sigma_h = 1; /* Try to learn */
    prior.learn_phi = 1;     /* Try to learn */
    prior.enforce_ordering = 1;

    /* φ prior: Beta(20, 1.5) on [0.85, 0.995] with MH step */
    pgas_paris_set_phi_prior(&prior, 20.0f, 1.5f, 0.85f, 0.995f, 0.35f);

    /* σ_h prior: InvGamma(5, 0.05) - moderately informative */
    pgas_paris_set_sigma_h_prior(&prior, 5.0f, 0.05f);

    /* Storage for post-burnin samples */
    int n_samples = N_SWEEPS - BURNIN;
    float *mu_samples = (float *)malloc(n_samples * K_REGIMES * sizeof(float));
    float *phi_samples = (float *)malloc(n_samples * sizeof(float));
    float *sigma_h_samples = (float *)malloc(n_samples * sizeof(float));

    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("  Running PGAS-PARIS (FULL parameter learning: μ_vol, σ_h, φ)\n");
    printf("═══════════════════════════════════════════════════════════════════\n");

    double start_time = get_time_ms();

    for (int s = 0; s < N_SWEEPS; s++)
    {
        float accept = pgas_paris_gibbs_sweep_full(pp, &prior);

        /* Store post-burnin samples */
        if (s >= BURNIN)
        {
            int idx = s - BURNIN;
            for (int k = 0; k < K_REGIMES; k++)
            {
                mu_samples[idx * K_REGIMES + k] = pgas->model.mu_vol[k];
            }
            phi_samples[idx] = pgas->model.phi;
            sigma_h_samples[idx] = pgas->model.sigma_h;
        }

        /* Progress update every 100 sweeps */
        if ((s + 1) % 100 == 0)
        {
            printf("  Sweep %d: accept=%.3f, μ=[",
                   s + 1, accept);
            for (int k = 0; k < K_REGIMES; k++)
            {
                printf("%.2f%s", pgas->model.mu_vol[k], k < K_REGIMES - 1 ? ", " : "");
            }
            printf("]\n");
        }
    }

    double end_time = get_time_ms();

    printf("\nResults:\n");
    printf("  Time: %.2f sec\n", (end_time - start_time) / 1000.0);
    printf("  Post-burnin samples: %d\n\n", n_samples);

    /* Compute posterior means and stds */
    float mu_mean[K_REGIMES] = {0}, mu_std[K_REGIMES] = {0};
    float phi_mean = 0, phi_std = 0;
    float sigma_h_mean = 0, sigma_h_std = 0;

    for (int i = 0; i < n_samples; i++)
    {
        for (int k = 0; k < K_REGIMES; k++)
        {
            mu_mean[k] += mu_samples[i * K_REGIMES + k];
        }
        phi_mean += phi_samples[i];
        sigma_h_mean += sigma_h_samples[i];
    }

    for (int k = 0; k < K_REGIMES; k++)
        mu_mean[k] /= n_samples;
    phi_mean /= n_samples;
    sigma_h_mean /= n_samples;

    for (int i = 0; i < n_samples; i++)
    {
        for (int k = 0; k < K_REGIMES; k++)
        {
            float d = mu_samples[i * K_REGIMES + k] - mu_mean[k];
            mu_std[k] += d * d;
        }
        float d_phi = phi_samples[i] - phi_mean;
        float d_sh = sigma_h_samples[i] - sigma_h_mean;
        phi_std += d_phi * d_phi;
        sigma_h_std += d_sh * d_sh;
    }

    for (int k = 0; k < K_REGIMES; k++)
        mu_std[k] = sqrtf(mu_std[k] / n_samples);
    phi_std = sqrtf(phi_std / n_samples);
    sigma_h_std = sqrtf(sigma_h_std / n_samples);

    /* Print results */
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("  LEARNED PARAMETERS\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");

    printf("μ_vol:\n");
    printf("  Regime    Learned     True        Error       Std\n");
    printf("  ─────────────────────────────────────────────────────\n");
    float mu_rmse = 0;
    for (int k = 0; k < K_REGIMES; k++)
    {
        float err = mu_mean[k] - TRUE_MU_VOL[k];
        mu_rmse += err * err;
        printf("     %d      %7.3f    %7.3f     %+6.3f     %.3f\n",
               k, mu_mean[k], TRUE_MU_VOL[k], err, mu_std[k]);
    }
    mu_rmse = sqrtf(mu_rmse / K_REGIMES);
    printf("  ─────────────────────────────────────────────────────\n");
    printf("  RMSE: %.4f\n\n", mu_rmse);

    float phi_err = phi_mean - TRUE_PHI;
    float sigma_h_err = sigma_h_mean - TRUE_SIGMA_H;

    printf("Structural Parameters:\n");
    printf("  Param    Learned     True        Error       Std\n");
    printf("  ─────────────────────────────────────────────────────\n");
    printf("  φ        %7.4f    %7.4f     %+6.4f     %.4f\n",
           phi_mean, TRUE_PHI, phi_err, phi_std);
    printf("  σ_h      %7.4f    %7.4f     %+6.4f     %.4f\n",
           sigma_h_mean, TRUE_SIGMA_H, sigma_h_err, sigma_h_std);
    printf("  ─────────────────────────────────────────────────────\n");

    float phi_accept = pgas_paris_get_phi_acceptance_rate(&prior);
    printf("  φ MH acceptance rate: %.1f%%\n\n", phi_accept * 100.0f);

    /* Summary */
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("  SUMMARY\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");

    int mu_pass = (mu_rmse < 0.20f);
    int phi_pass = (fabsf(phi_err) < 0.02f);         /* Within 0.02 of true */
    int sigma_h_pass = (fabsf(sigma_h_err) < 0.05f); /* Within 0.05 of true */

    printf("  ✓/✗  Parameter     Status\n");
    printf("  ─────────────────────────────────────────────────────\n");
    printf("   %s   μ_vol        RMSE=%.3f (threshold: <0.20)\n",
           mu_pass ? "✓" : "✗", mu_rmse);
    printf("   %s   φ            error=%+.4f (threshold: <0.02)\n",
           phi_pass ? "✓" : "✗", phi_err);
    printf("   %s   σ_h          error=%+.4f (threshold: <0.05)\n",
           sigma_h_pass ? "✓" : "✗", sigma_h_err);
    printf("  ─────────────────────────────────────────────────────\n\n");

    int all_pass = mu_pass && phi_pass && sigma_h_pass;

    if (all_pass)
    {
        printf("  ★ ALL PARAMETERS LEARNED SUCCESSFULLY ★\n\n");
    }
    else
    {
        printf("  Results (expected: μ_vol ✓, φ ⚠, σ_h ✗):\n");
        printf("    μ_vol: %s - Works reliably\n", mu_pass ? "PASS" : "FAIL");
        printf("    φ:     %s - Requires MH tuning\n", phi_pass ? "PASS" : "FAIL");
        printf("    σ_h:   %s - Confounded with OCSN noise\n\n", sigma_h_pass ? "PASS" : "FAIL");
    }

    /* C code output for production */
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("  LEARNED VALUES (for reference)\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");

    printf("/* Learned structural parameters */\n");
    printf("rbpf_ext_set_phi(ext, %.4ff);     /* learned (true: %.4f) */\n", phi_mean, TRUE_PHI);
    printf("rbpf_ext_set_sigma_h(ext, %.4ff); /* learned (true: %.4f) */\n\n", sigma_h_mean, TRUE_SIGMA_H);
    printf("/* Learned μ_vol from PGAS-PARIS */\n");
    for (int k = 0; k < K_REGIMES; k++)
    {
        printf("rbpf_ext_set_regime_params(ext, %d, %.4ff, %.3ff, %.3ff);\n",
               k, TRUE_SIGMA_VOL[k], mu_mean[k], TRUE_SIGMA_VOL[k]);
    }
    printf("\n");

    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("  RECOMMENDATION\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");
    printf("  For production, use test_regime_learning which learns only μ_vol.\n");
    printf("  Fix φ=%.3f and σ_h=%.3f from domain knowledge/calibration.\n\n", TRUE_PHI, TRUE_SIGMA_H);

    /* Cleanup */
    free(mu_samples);
    free(phi_samples);
    free(sigma_h_samples);
    free(init_h);
    free(obs_d);
    pgas_paris_free(pp);
    pgas_mkl_free(pgas);
    free_synthetic_data(&data);
    mkl_tuning_cleanup();

    return all_pass ? 0 : 1;
}