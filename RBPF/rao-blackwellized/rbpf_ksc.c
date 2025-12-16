/**
 * @file rbpf_ksc.c
 * @brief RBPF with Kim-Shephard-Chib (1998) - Optimized Implementation
 *
 * Key optimizations:
 *   - Zero malloc in hot path (all buffers preallocated)
 *   - Pointer swap instead of memcpy for resampling
 *   - PCG32 RNG (fast, good quality)
 *   - Transition LUT (no cumsum search)
 *   - Regularization after resample (prevents Kalman state degeneracy)
 *   - Self-aware detection signals (no external model)
 *
 * Student-t Extension (optional):
 *   - Compile with RBPF_ENABLE_STUDENT_T=1 (default) for fat-tail support
 *   - Compile with RBPF_ENABLE_STUDENT_T=0 for Gaussian-only (minimal overhead)
 *   - Runtime switch via rbpf_ksc_enable_student_t() / rbpf_ksc_disable_student_t()
 *
 * Latency target: <15μs for 1000 particles (Gaussian)
 *                 <20μs for 1000 particles (Student-t)
 */

#include "rbpf_ksc.h"
#include "rbpf_silverman.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <mkl_vml.h>

/*═══════════════════════════════════════════════════════════════════════════
 * STUDENT-T COMPILE-TIME SWITCH
 *
 * Set RBPF_ENABLE_STUDENT_T=0 before including header for Gaussian-only build.
 * Default is enabled (1).
 *═══════════════════════════════════════════════════════════════════════════*/

#ifndef RBPF_ENABLE_STUDENT_T
#define RBPF_ENABLE_STUDENT_T 1
#endif

#if RBPF_ENABLE_STUDENT_T
/* Forward declarations for Student-t extension (defined in rbpf_ksc_student_t.c) */
extern int rbpf_ksc_alloc_student_t(RBPF_KSC *rbpf);
extern void rbpf_ksc_free_student_t(RBPF_KSC *rbpf);
extern void rbpf_ksc_resample_student_t(RBPF_KSC *rbpf, const int *indices);
extern rbpf_real_t rbpf_ksc_update_student_t(RBPF_KSC *rbpf, rbpf_real_t y);
extern rbpf_real_t rbpf_ksc_update_student_t_robust(RBPF_KSC *rbpf, rbpf_real_t y,
                                                    rbpf_real_t nu,
                                                    const RBPF_RobustOCSN *ocsn);
#endif

/*─────────────────────────────────────────────────────────────────────────────
 * OMORI, CHIB, SHEPHARD & NAKAJIMA (2007) MIXTURE PARAMETERS
 *
 * 10-component Gaussian mixture approximation of log(χ²(1)):
 * p(log(ε²)) ≈ Σ_k π_k × N(m_k, v_k²)
 *
 * Upgrade from KSC (1998): better tail accuracy in both directions
 *───────────────────────────────────────────────────────────────────────────*/

static const rbpf_real_t KSC_PROB[KSC_N_COMPONENTS] = {
    RBPF_REAL(0.00609), RBPF_REAL(0.04775), RBPF_REAL(0.13057), RBPF_REAL(0.20674),
    RBPF_REAL(0.22715), RBPF_REAL(0.18842), RBPF_REAL(0.12047), RBPF_REAL(0.05591),
    RBPF_REAL(0.01575), RBPF_REAL(0.00115)};

static const rbpf_real_t KSC_MEAN[KSC_N_COMPONENTS] = {
    RBPF_REAL(1.92677), RBPF_REAL(1.34744), RBPF_REAL(0.73504), RBPF_REAL(0.02266),
    RBPF_REAL(-0.85173), RBPF_REAL(-1.97278), RBPF_REAL(-3.46788), RBPF_REAL(-5.55246),
    RBPF_REAL(-8.68384), RBPF_REAL(-14.65000)};

static const rbpf_real_t KSC_VAR[KSC_N_COMPONENTS] = {
    RBPF_REAL(0.11265), RBPF_REAL(0.17788), RBPF_REAL(0.26768), RBPF_REAL(0.40611),
    RBPF_REAL(0.62699), RBPF_REAL(0.98583), RBPF_REAL(1.57469), RBPF_REAL(2.54498),
    RBPF_REAL(4.16591), RBPF_REAL(7.33342)};

/* Precomputed: -0.5 * log(2π) = -0.9189385332 */
static const rbpf_real_t LOG_2PI_HALF = RBPF_REAL(-0.9189385332);

/*─────────────────────────────────────────────────────────────────────────────
 * HELPERS
 *───────────────────────────────────────────────────────────────────────────*/

static inline rbpf_real_t *aligned_alloc_real(int n)
{
    return (rbpf_real_t *)mkl_malloc(n * sizeof(rbpf_real_t), RBPF_ALIGN);
}

static inline int *aligned_alloc_int(int n)
{
    return (int *)mkl_malloc(n * sizeof(int), RBPF_ALIGN);
}

/*─────────────────────────────────────────────────────────────────────────────
 * CREATE / DESTROY
 *───────────────────────────────────────────────────────────────────────────*/

RBPF_KSC *rbpf_ksc_create(int n_particles, int n_regimes)
{
    RBPF_KSC *rbpf = (RBPF_KSC *)mkl_calloc(1, sizeof(RBPF_KSC), RBPF_ALIGN);
    if (!rbpf)
        return NULL;

    rbpf->n_particles = n_particles;
    rbpf->n_regimes = n_regimes < RBPF_MAX_REGIMES ? n_regimes : RBPF_MAX_REGIMES;
    rbpf->uniform_weight = RBPF_REAL(1.0) / n_particles;
    rbpf->inv_n = RBPF_REAL(1.0) / n_particles;

    rbpf->n_threads = omp_get_max_threads();
    if (rbpf->n_threads > RBPF_MAX_THREADS)
        rbpf->n_threads = RBPF_MAX_THREADS;

    int n = n_particles;

    /* Particle state */
    rbpf->mu = aligned_alloc_real(n);
    rbpf->var = aligned_alloc_real(n);
    rbpf->regime = aligned_alloc_int(n);
    rbpf->log_weight = aligned_alloc_real(n);

    /* Double buffers */
    rbpf->mu_tmp = aligned_alloc_real(n);
    rbpf->var_tmp = aligned_alloc_real(n);
    rbpf->regime_tmp = aligned_alloc_int(n);

    /* Workspace - ALL preallocated */
    rbpf->mu_pred = aligned_alloc_real(n);
    rbpf->var_pred = aligned_alloc_real(n);
    rbpf->theta_arr = aligned_alloc_real(n);
    rbpf->mu_vol_arr = aligned_alloc_real(n);
    rbpf->q_arr = aligned_alloc_real(n);
    rbpf->lik_total = aligned_alloc_real(n);
    rbpf->lik_comp = aligned_alloc_real(n);
    rbpf->innov = aligned_alloc_real(n);
    rbpf->S = aligned_alloc_real(n);
    rbpf->K = aligned_alloc_real(n);
    rbpf->w_norm = aligned_alloc_real(n);
    rbpf->cumsum = aligned_alloc_real(n);
    rbpf->mu_accum = aligned_alloc_real(n);
    rbpf->var_accum = aligned_alloc_real(n);
    rbpf->scratch1 = aligned_alloc_real(n);
    rbpf->scratch2 = aligned_alloc_real(n);
    rbpf->indices = aligned_alloc_int(n);

    /* Log-sum-exp buffers for numerical stability in K-mixture */
    rbpf->log_lik_buffer = aligned_alloc_real(KSC_N_COMPONENTS * n);
    rbpf->max_log_lik = aligned_alloc_real(n);

    /* Pre-generated Gaussian buffer for jitter (MKL ICDF) */
    rbpf->rng_gaussian = aligned_alloc_real(2 * n); /* 2n for mu and var jitter */
    rbpf->rng_buffer_size = 2 * n;

    /* Silverman bandwidth scratch buffer */
    rbpf->silverman_scratch = aligned_alloc_real(n);

    /* Check allocations */
    if (!rbpf->mu || !rbpf->var || !rbpf->regime || !rbpf->log_weight ||
        !rbpf->mu_tmp || !rbpf->var_tmp || !rbpf->regime_tmp ||
        !rbpf->mu_pred || !rbpf->var_pred || !rbpf->theta_arr ||
        !rbpf->mu_vol_arr || !rbpf->q_arr || !rbpf->lik_total ||
        !rbpf->lik_comp || !rbpf->innov || !rbpf->S || !rbpf->K ||
        !rbpf->w_norm || !rbpf->cumsum || !rbpf->mu_accum || !rbpf->var_accum ||
        !rbpf->scratch1 || !rbpf->scratch2 || !rbpf->indices ||
        !rbpf->log_lik_buffer || !rbpf->max_log_lik || !rbpf->rng_gaussian ||
        !rbpf->silverman_scratch)
    {
        rbpf_ksc_destroy(rbpf);
        return NULL;
    }

    /* Initialize RNG */
    for (int t = 0; t < rbpf->n_threads; t++)
    {
        rbpf_pcg32_seed(&rbpf->pcg[t], 42 + t * 12345, t * 67890);
        vslNewStream(&rbpf->mkl_rng[t], VSL_BRNG_SFMT19937, 42 + t * 8192);
    }

    /* Default regime parameters */
    for (int r = 0; r < n_regimes; r++)
    {
        rbpf->params[r].theta = RBPF_REAL(0.05);
        rbpf->params[r].mu_vol = rbpf_log(RBPF_REAL(0.01)); /* 1% daily vol */
        rbpf->params[r].sigma_vol = RBPF_REAL(0.1);
        rbpf->params[r].q = RBPF_REAL(0.01);
    }

    /* Regularization defaults
     *
     * With correct law-of-total-variance calculation, variance estimates
     * are more accurate. We can use lighter regularization:
     * - h_mu: jitter on state to prevent particle collapse
     * - h_var: lighter jitter on covariance (optional, mainly for robustness)
     */
    rbpf->reg_bandwidth_mu = RBPF_REAL(0.02);    /* ~2% jitter on log-vol */
    rbpf->reg_bandwidth_var = RBPF_REAL(0.0005); /* Reduced: correct variance calc */
    rbpf->reg_scale_min = RBPF_REAL(0.1);
    rbpf->reg_scale_max = RBPF_REAL(0.5);
    rbpf->last_ess = (rbpf_real_t)n;

    /* Silverman adaptive bandwidth - enabled by default */
    rbpf->use_silverman_bandwidth = 1;
    rbpf->last_silverman_bandwidth = RBPF_REAL(0.0);

    /* Regime diversity: prevent particle collapse to single regime
     * Without this, resampling can kill minority regimes, leaving
     * no particles to respond to sudden regime changes. */
    rbpf->min_particles_per_regime = n / (4 * n_regimes); /* ~6% per regime */
    if (rbpf->min_particles_per_regime < 2)
        rbpf->min_particles_per_regime = 2;
    rbpf->regime_mutation_prob = RBPF_REAL(0.02); /* 2% mutation rate */

    /* Detection state */
    rbpf->detection.vol_ema_short = RBPF_REAL(0.01);
    rbpf->detection.vol_ema_long = RBPF_REAL(0.01);
    rbpf->detection.prev_regime = 0;
    rbpf->detection.cooldown = 0;

    /* SPRT regime detection (MANDATORY - statistically principled)
     * α = 0.01 (1% false positive), β = 0.01 (1% false negative)
     * Thresholds: A = log(99) ≈ 4.6, B = log(0.0101) ≈ -4.6
     */
    for (int r = 0; r < RBPF_MAX_REGIMES; r++)
    {
        rbpf->detection.sprt_log_ratios[r] = 0.0;
    }
    rbpf->detection.sprt_threshold_high = 4.595; /* log((1-0.01)/0.01) */
    rbpf->detection.sprt_threshold_low = -4.595; /* log(0.01/(1-0.01)) */
    rbpf->detection.sprt_current_regime = 0;
    rbpf->detection.sprt_min_dwell = 3; /* Min 3 ticks before switch */
    rbpf->detection.sprt_ticks_in_current = 0;

    /* Legacy fields (kept for API compatibility, not used) */
    rbpf->detection.stable_regime = 0;

    /* Fixed-lag smoothing (dual output) - disabled by default */
    rbpf->smooth_lag = 0;
    rbpf->smooth_head = 0;
    rbpf->smooth_count = 0;
    for (int i = 0; i < RBPF_MAX_SMOOTH_LAG; i++)
    {
        rbpf->smooth_history[i].valid = 0;
    }

    /* Liu-West parameter learning (Phase 3) */
    int n_params = n * n_regimes;
    rbpf->particle_mu_vol = aligned_alloc_real(n_params);
    rbpf->particle_sigma_vol = aligned_alloc_real(n_params);
    rbpf->particle_mu_vol_tmp = aligned_alloc_real(n_params);
    rbpf->particle_sigma_vol_tmp = aligned_alloc_real(n_params);

    if (!rbpf->particle_mu_vol || !rbpf->particle_sigma_vol ||
        !rbpf->particle_mu_vol_tmp || !rbpf->particle_sigma_vol_tmp)
    {
        rbpf_ksc_destroy(rbpf);
        return NULL;
    }

    /* Parameter bounds (used for clamping learned params from MMPF Online EM) */
    rbpf->param_bounds.min_mu_vol = rbpf_log(0.001f); /* 0.1% vol floor */
    rbpf->param_bounds.max_mu_vol = rbpf_log(0.5f);   /* 50% vol ceiling */
    rbpf->param_bounds.min_sigma_vol = RBPF_REAL(0.01);
    rbpf->param_bounds.max_sigma_vol = RBPF_REAL(1.0);

    rbpf->use_learned_params = 0;

    /*═══════════════════════════════════════════════════════════════════════
     * STUDENT-T ALLOCATION
     *
     * Allocate auxiliary variable arrays for Student-t observation model.
     * These are only used when student_t_enabled is set at runtime.
     *═══════════════════════════════════════════════════════════════════════*/
#if RBPF_ENABLE_STUDENT_T
    if (rbpf_ksc_alloc_student_t(rbpf) < 0)
    {
        rbpf_ksc_destroy(rbpf);
        return NULL;
    }
#else
    /* Gaussian-only build: set pointers to NULL */
    rbpf->lambda = NULL;
    rbpf->lambda_tmp = NULL;
    rbpf->log_lambda = NULL;
    rbpf->student_t_enabled = 0;
#endif

    /* MKL fast math mode */
    vmlSetMode(VML_EP | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE);

    return rbpf;
}

void rbpf_ksc_destroy(RBPF_KSC *rbpf)
{
    if (!rbpf)
        return;

    for (int t = 0; t < rbpf->n_threads; t++)
    {
        if (rbpf->mkl_rng[t])
            vslDeleteStream(&rbpf->mkl_rng[t]);
    }

    mkl_free(rbpf->mu);
    mkl_free(rbpf->var);
    mkl_free(rbpf->regime);
    mkl_free(rbpf->log_weight);
    mkl_free(rbpf->mu_tmp);
    mkl_free(rbpf->var_tmp);
    mkl_free(rbpf->regime_tmp);
    mkl_free(rbpf->mu_pred);
    mkl_free(rbpf->var_pred);
    mkl_free(rbpf->theta_arr);
    mkl_free(rbpf->mu_vol_arr);
    mkl_free(rbpf->q_arr);
    mkl_free(rbpf->lik_total);
    mkl_free(rbpf->lik_comp);
    mkl_free(rbpf->innov);
    mkl_free(rbpf->S);
    mkl_free(rbpf->K);
    mkl_free(rbpf->w_norm);
    mkl_free(rbpf->cumsum);
    mkl_free(rbpf->mu_accum);
    mkl_free(rbpf->var_accum);
    mkl_free(rbpf->scratch1);
    mkl_free(rbpf->scratch2);
    mkl_free(rbpf->indices);

    /* Log-sum-exp and RNG buffers */
    mkl_free(rbpf->log_lik_buffer);
    mkl_free(rbpf->max_log_lik);
    mkl_free(rbpf->rng_gaussian);
    mkl_free(rbpf->silverman_scratch);

    /* Liu-West arrays */
    mkl_free(rbpf->particle_mu_vol);
    mkl_free(rbpf->particle_sigma_vol);
    mkl_free(rbpf->particle_mu_vol_tmp);
    mkl_free(rbpf->particle_sigma_vol_tmp);

    /*═══════════════════════════════════════════════════════════════════════
     * STUDENT-T CLEANUP
     *═══════════════════════════════════════════════════════════════════════*/
#if RBPF_ENABLE_STUDENT_T
    rbpf_ksc_free_student_t(rbpf);
#endif

    mkl_free(rbpf);
}

/*─────────────────────────────────────────────────────────────────────────────
 * CONFIGURATION
 *───────────────────────────────────────────────────────────────────────────*/

void rbpf_ksc_set_regime_params(RBPF_KSC *rbpf, int r,
                                rbpf_real_t theta, rbpf_real_t mu_vol, rbpf_real_t sigma_vol)
{
    if (r < 0 || r >= RBPF_MAX_REGIMES)
        return;
    rbpf->params[r].theta = theta;
    rbpf->params[r].mu_vol = mu_vol;
    rbpf->params[r].sigma_vol = sigma_vol;
    rbpf->params[r].q = sigma_vol * sigma_vol;

    /* Also update Liu-West cached means (these are the fallback values) */
    rbpf->lw_mu_vol_mean[r] = mu_vol;
    rbpf->lw_sigma_vol_mean[r] = sigma_vol;
}

void rbpf_ksc_build_transition_lut(RBPF_KSC *rbpf, const rbpf_real_t *trans_matrix)
{
    /* Build LUT for each regime: uniform → next regime */
    for (int r = 0; r < rbpf->n_regimes; r++)
    {
        rbpf_real_t cumsum[RBPF_MAX_REGIMES];
        cumsum[0] = trans_matrix[r * rbpf->n_regimes + 0];
        for (int j = 1; j < rbpf->n_regimes; j++)
        {
            cumsum[j] = cumsum[j - 1] + trans_matrix[r * rbpf->n_regimes + j];
        }

        for (int i = 0; i < 1024; i++)
        {
            rbpf_real_t u = (rbpf_real_t)i / RBPF_REAL(1024.0);
            int next = rbpf->n_regimes - 1;
            for (int j = 0; j < rbpf->n_regimes - 1; j++)
            {
                if (u < cumsum[j])
                {
                    next = j;
                    break;
                }
            }
            rbpf->trans_lut[r][i] = (uint8_t)next;
        }
    }
}

void rbpf_ksc_set_regularization(RBPF_KSC *rbpf, rbpf_real_t h_mu, rbpf_real_t h_var)
{
    rbpf->reg_bandwidth_mu = h_mu;
    rbpf->reg_bandwidth_var = h_var;
}

void rbpf_ksc_set_regime_diversity(RBPF_KSC *rbpf, int min_per_regime, rbpf_real_t mutation_prob)
{
    rbpf->min_particles_per_regime = min_per_regime;
    /* Clamp mutation probability to [0, 0.2] for stability */
    if (mutation_prob < RBPF_REAL(0.0))
        mutation_prob = RBPF_REAL(0.0);
    if (mutation_prob > RBPF_REAL(0.2))
        mutation_prob = RBPF_REAL(0.2);
    rbpf->regime_mutation_prob = mutation_prob;
}

void rbpf_ksc_set_sprt_params(RBPF_KSC *rbpf, double alpha, double beta, int min_dwell)
{
    if (!rbpf)
        return;

    /* Compute Wald thresholds from error rates */
    if (alpha < 0.001)
        alpha = 0.001;
    if (alpha > 0.5)
        alpha = 0.5;
    if (beta < 0.001)
        beta = 0.001;
    if (beta > 0.5)
        beta = 0.5;

    rbpf->detection.sprt_threshold_high = log((1.0 - beta) / alpha);
    rbpf->detection.sprt_threshold_low = log(beta / (1.0 - alpha));

    if (min_dwell < 1)
        min_dwell = 1;
    if (min_dwell > 100)
        min_dwell = 100;
    rbpf->detection.sprt_min_dwell = min_dwell;
}

void rbpf_ksc_reset_sprt(RBPF_KSC *rbpf)
{
    if (!rbpf)
        return;

    for (int r = 0; r < RBPF_MAX_REGIMES; r++)
    {
        rbpf->detection.sprt_log_ratios[r] = 0.0;
    }
    rbpf->detection.sprt_ticks_in_current = 0;
}

void rbpf_ksc_set_fixed_lag_smoothing(RBPF_KSC *rbpf, int lag)
{
    /* Clamp lag to valid range */
    if (lag < 0)
        lag = 0;
    if (lag > RBPF_MAX_SMOOTH_LAG)
        lag = RBPF_MAX_SMOOTH_LAG;

    rbpf->smooth_lag = lag;
    rbpf->smooth_head = 0;
    rbpf->smooth_count = 0;

    /* Clear history buffer */
    for (int i = 0; i < RBPF_MAX_SMOOTH_LAG; i++)
    {
        rbpf->smooth_history[i].valid = 0;
    }
}

void rbpf_ksc_set_learned_params_mode(RBPF_KSC *rbpf, int enable)
{
    if (rbpf)
    {
        rbpf->use_learned_params = enable;
    }
}

void rbpf_ksc_set_silverman_bandwidth(RBPF_KSC *rbpf, int enable)
{
    if (rbpf)
    {
        rbpf->use_silverman_bandwidth = enable;
    }
}

rbpf_real_t rbpf_ksc_get_last_silverman_bandwidth(const RBPF_KSC *rbpf)
{
    return rbpf ? rbpf->last_silverman_bandwidth : RBPF_REAL(0.0);
}

/*─────────────────────────────────────────────────────────────────────────────
 * PARAMETER LEARNING - DELETED (Liu-West removed)
 *
 * Liu-West parameter learning has been replaced by Online EM at the MMPF level.
 * See mmpf_online_em.c for the new implementation.
 *
 * The MMPF controller now:
 * 1. Learns regime centers via Online EM (streaming GMM)
 * 2. Pushes updated parameters to RBPF via rbpf_ksc_set_regime_params()
 *
 * This separation of concerns is cleaner:
 * - RBPF: Pure filtering (given parameters)
 * - MMPF: Model selection + parameter learning
 *───────────────────────────────────────────────────────────────────────────*/

/*─────────────────────────────────────────────────────────────────────────────
 * INITIALIZATION
 *───────────────────────────────────────────────────────────────────────────*/

void rbpf_ksc_init(RBPF_KSC *rbpf, rbpf_real_t mu0, rbpf_real_t var0)
{
    int n = rbpf->n_particles;
    int n_regimes = rbpf->n_regimes;

    /* Spread particles for state diversity */
    rbpf_real_t state_spread = RBPF_REAL(0.1);

    for (int i = 0; i < n; i++)
    {
        rbpf_real_t noise = rbpf_pcg32_gaussian(&rbpf->pcg[0]) * state_spread;
        rbpf->mu[i] = mu0 + noise;
        rbpf->var[i] = var0;
        rbpf->regime[i] = i % n_regimes;
        rbpf->log_weight[i] = RBPF_REAL(0.0); /* log(1) = 0 */
    }

    /* Initialize per-particle Liu-West parameters from global params
     *
     * With ORDER CONSTRAINT in place, spread is still useful for exploration.
     * The constraint ensures μ_vol[0] < μ_vol[1] < ... < μ_vol[n_regimes-1]
     * after sorting, so particles can explore while maintaining regime identity.
     *
     * Spread of 0.5 covers ±1.5 range (3σ) in log-vol space.
     */
    rbpf_real_t param_spread_mu = RBPF_REAL(0.5);     /* Wide spread on μ_vol */
    rbpf_real_t param_spread_sigma = RBPF_REAL(0.08); /* Moderate spread on σ_vol */

    for (int i = 0; i < n; i++)
    {
        for (int r = 0; r < n_regimes; r++)
        {
            int idx = i * n_regimes + r;

            /* Wide jitter to explore parameter space */
            rbpf_real_t jitter_mu = rbpf_pcg32_gaussian(&rbpf->pcg[0]) * param_spread_mu;
            rbpf_real_t jitter_sigma = rbpf_pcg32_gaussian(&rbpf->pcg[0]) * param_spread_sigma;

            rbpf_real_t mu_vol = rbpf->params[r].mu_vol + jitter_mu;
            rbpf_real_t sigma_vol = rbpf->params[r].sigma_vol + rbpf_fabs(jitter_sigma);

            /* Clamp to bounds */
            if (mu_vol < rbpf->param_bounds.min_mu_vol)
                mu_vol = rbpf->param_bounds.min_mu_vol;
            if (mu_vol > rbpf->param_bounds.max_mu_vol)
                mu_vol = rbpf->param_bounds.max_mu_vol;
            if (sigma_vol < rbpf->param_bounds.min_sigma_vol)
                sigma_vol = rbpf->param_bounds.min_sigma_vol;
            if (sigma_vol > rbpf->param_bounds.max_sigma_vol)
                sigma_vol = rbpf->param_bounds.max_sigma_vol;

            rbpf->particle_mu_vol[idx] = mu_vol;
            rbpf->particle_sigma_vol[idx] = sigma_vol;
        }

        /* ORDER CONSTRAINT: Enforce μ_vol[0] < μ_vol[1] < ... < μ_vol[n_regimes-1]
         * Must sort after initialization to prevent label switching from the start */
        for (int r = 0; r < n_regimes - 1; r++)
        {
            for (int s = r + 1; s < n_regimes; s++)
            {
                int idx_r = i * n_regimes + r;
                int idx_s = i * n_regimes + s;

                if (rbpf->particle_mu_vol[idx_r] > rbpf->particle_mu_vol[idx_s])
                {
                    rbpf_real_t temp = rbpf->particle_mu_vol[idx_r];
                    rbpf->particle_mu_vol[idx_r] = rbpf->particle_mu_vol[idx_s];
                    rbpf->particle_mu_vol[idx_s] = temp;

                    temp = rbpf->particle_sigma_vol[idx_r];
                    rbpf->particle_sigma_vol[idx_r] = rbpf->particle_sigma_vol[idx_s];
                    rbpf->particle_sigma_vol[idx_s] = temp;
                }
            }
        }

        /* MINIMUM SEPARATION: Ensure regimes start spread across vol spectrum */
        const rbpf_real_t min_sep_init = RBPF_REAL(0.5);
        for (int r = 1; r < n_regimes; r++)
        {
            int idx_prev = i * n_regimes + (r - 1);
            int idx_curr = i * n_regimes + r;

            rbpf_real_t gap = rbpf->particle_mu_vol[idx_curr] - rbpf->particle_mu_vol[idx_prev];
            if (gap < min_sep_init)
            {
                rbpf->particle_mu_vol[idx_curr] = rbpf->particle_mu_vol[idx_prev] + min_sep_init;

                /* Clamp to upper bound */
                if (rbpf->particle_mu_vol[idx_curr] > rbpf->param_bounds.max_mu_vol)
                {
                    rbpf->particle_mu_vol[idx_curr] = rbpf->param_bounds.max_mu_vol;
                }
            }
        }
    }

    /* Reset detection */
    rbpf->detection.vol_ema_short = rbpf_exp(mu0);
    rbpf->detection.vol_ema_long = rbpf_exp(mu0);
    rbpf->detection.prev_regime = 0;
    rbpf->detection.cooldown = 0;

    /* Reset SPRT */
    for (int r = 0; r < rbpf->n_regimes; r++)
    {
        rbpf->detection.sprt_log_ratios[r] = 0.0;
    }
    rbpf->detection.sprt_current_regime = 0;
    rbpf->detection.sprt_ticks_in_current = 0;

    /* Reset legacy counter-based smoothing */
    rbpf->detection.stable_regime = 0;
    rbpf->detection.candidate_regime = 0;
    rbpf->detection.hold_count = 0;

    /* Reset fixed-lag smoothing buffer */
    rbpf->smooth_head = 0;
    rbpf->smooth_count = 0;
    for (int i = 0; i < RBPF_MAX_SMOOTH_LAG; i++)
    {
        rbpf->smooth_history[i].valid = 0;
    }

    /*═══════════════════════════════════════════════════════════════════════
     * STUDENT-T INITIALIZATION
     *
     * Initialize λ arrays to 1.0 (Gaussian equivalent)
     *═══════════════════════════════════════════════════════════════════════*/
#if RBPF_ENABLE_STUDENT_T
    if (rbpf->lambda != NULL)
    {
        for (int i = 0; i < n; i++)
        {
            rbpf->lambda[i] = RBPF_REAL(1.0);
            if (rbpf->log_lambda)
                rbpf->log_lambda[i] = RBPF_REAL(0.0);
        }
    }
#endif
}

/*─────────────────────────────────────────────────────────────────────────────
 * PREDICT STEP (Option B: Decoupled per-particle flag)
 *
 * ℓ_t = (1-θ)ℓ_{t-1} + θμ + η_t,  η_t ~ N(0, q)
 *
 * Kalman predict:
 *   μ_pred = (1-θ)μ + θμ_vol
 *   P_pred = (1-θ)²P + q
 *
 * OPTION B LOGIC:
 *   Use per-particle parameters if:
 *   1. Explicitly requested via use_learned_params flag (Storvik Mode), OR
 *   2. Liu-West is enabled AND warmup is complete (Liu-West Mode)
 *
 * This decoupling allows Storvik to populate particle_mu_vol/particle_sigma_vol
 * without triggering Liu-West's shrinkage/jitter logic in resample.
 *───────────────────────────────────────────────────────────────────────────*/

void rbpf_ksc_predict(RBPF_KSC *rbpf)
{
    const int n = rbpf->n_particles;
    const RBPF_RegimeParams *params = rbpf->params;
    const int n_regimes = rbpf->n_regimes;

    /* Per-particle parameters: Set by external learner (Storvik via MMPF) */
    const int use_particles = rbpf->use_learned_params;

    rbpf_real_t *restrict mu = rbpf->mu;
    rbpf_real_t *restrict var = rbpf->var;
    const int *restrict regime = rbpf->regime;
    rbpf_real_t *restrict mu_pred = rbpf->mu_pred;
    rbpf_real_t *restrict var_pred = rbpf->var_pred;

    if (use_particles)
    {
        /* Use per-particle learned parameters (Storvik or Liu-West) */
        const rbpf_real_t *particle_mu_vol = rbpf->particle_mu_vol;
        const rbpf_real_t *particle_sigma_vol = rbpf->particle_sigma_vol;

        for (int i = 0; i < n; i++)
        {
            int r = regime[i];
            rbpf_real_t theta = params[r].theta; /* θ still from global (not learned) */
            rbpf_real_t omt = RBPF_REAL(1.0) - theta;
            rbpf_real_t omt2 = omt * omt;

            /* Per-particle μ_vol and σ_vol
             * In Storvik mode: populated by memcpy from Storvik's mu_cached/sigma_cached
             * In Liu-West mode: populated by Liu-West shrinkage+jitter */
            int idx = i * n_regimes + r;
            rbpf_real_t mv = particle_mu_vol[idx];
            rbpf_real_t sigma_vol = particle_sigma_vol[idx];
            rbpf_real_t q = sigma_vol * sigma_vol;

            mu_pred[i] = omt * mu[i] + theta * mv;
            var_pred[i] = omt2 * var[i] + q;
        }
    }
    else
    {
        /* Use global regime parameters (original behavior - no learning) */
        rbpf_real_t theta_r[RBPF_MAX_REGIMES];
        rbpf_real_t mu_vol_r[RBPF_MAX_REGIMES];
        rbpf_real_t q_r[RBPF_MAX_REGIMES];
        rbpf_real_t one_minus_theta_r[RBPF_MAX_REGIMES];
        rbpf_real_t one_minus_theta_sq_r[RBPF_MAX_REGIMES];

        for (int r = 0; r < n_regimes; r++)
        {
            theta_r[r] = params[r].theta;
            mu_vol_r[r] = params[r].mu_vol;
            q_r[r] = params[r].q;
            one_minus_theta_r[r] = RBPF_REAL(1.0) - theta_r[r];
            one_minus_theta_sq_r[r] = one_minus_theta_r[r] * one_minus_theta_r[r];
        }

        for (int i = 0; i < n; i++)
        {
            int r = regime[i];
            rbpf_real_t omt = one_minus_theta_r[r];
            rbpf_real_t omt2 = one_minus_theta_sq_r[r];
            rbpf_real_t th = theta_r[r];
            rbpf_real_t mv = mu_vol_r[r];
            rbpf_real_t q = q_r[r];

            mu_pred[i] = omt * mu[i] + th * mv;
            var_pred[i] = omt2 * var[i] + q;
        }
    }
}

/*─────────────────────────────────────────────────────────────────────────────
 * UPDATE STEP (optimized 10-component Omori mixture Kalman)
 *
 * Observation: y = log(r²) = 2ℓ + log(ε²)
 * Linear: y - m_k = H*ℓ + (log(ε²) - m_k), H = 2
 *
 * Optimizations:
 *   - Fused scalar loops for small n (avoids VML dispatch overhead)
 *   - Precomputed constants (H², log(π_k), etc.)
 *   - Single pass accumulation
 *───────────────────────────────────────────────────────────────────────────*/

/* Precomputed: log(π_k) for each Omori (2007) component */
static const rbpf_real_t KSC_LOG_PROB[KSC_N_COMPONENTS] = {
    RBPF_REAL(-5.101), /* log(0.00609) */
    RBPF_REAL(-3.042), /* log(0.04775) */
    RBPF_REAL(-2.036), /* log(0.13057) */
    RBPF_REAL(-1.577), /* log(0.20674) */
    RBPF_REAL(-1.482), /* log(0.22715) */
    RBPF_REAL(-1.669), /* log(0.18842) */
    RBPF_REAL(-2.116), /* log(0.12047) */
    RBPF_REAL(-2.884), /* log(0.05591) */
    RBPF_REAL(-4.151), /* log(0.01575) */
    RBPF_REAL(-6.768)  /* log(0.00115) */
};

rbpf_real_t rbpf_ksc_update(RBPF_KSC *rbpf, rbpf_real_t y)
{
    const int n = rbpf->n_particles;
    const rbpf_real_t H = RBPF_REAL(2.0);
    const rbpf_real_t H2 = RBPF_REAL(4.0);
    const rbpf_real_t NEG_HALF = RBPF_REAL(-0.5);

    rbpf_real_t *restrict mu_pred = rbpf->mu_pred;
    rbpf_real_t *restrict var_pred = rbpf->var_pred;
    rbpf_real_t *restrict mu = rbpf->mu;
    rbpf_real_t *restrict var = rbpf->var;
    rbpf_real_t *restrict log_weight = rbpf->log_weight;
    rbpf_real_t *restrict lik_total = rbpf->lik_total;
    rbpf_real_t *restrict mu_accum = rbpf->mu_accum;
    rbpf_real_t *restrict var_accum = rbpf->var_accum;
    rbpf_real_t *restrict log_lik_buf = rbpf->log_lik_buffer;
    rbpf_real_t *restrict max_ll = rbpf->max_log_lik;

    /*
     * Log-Sum-Exp approach for numerical stability:
     * 1. Compute log_lik for all K components, store in buffer
     * 2. Find max log_lik per particle
     * 3. Sum exp(log_lik - max) to avoid underflow
     * 4. log(sum) + max = log(total_lik)
     *
     * Variance calculation uses law of total variance:
     *   Var[X] = E[Var[X|K]] + Var[E[X|K]]
     *          = E[X²] - E[X]²
     *
     * We accumulate: E[X²] = Σ wₖ (σₖ² + μₖ²)
     * Then compute:  Var = E[X²] - (E[X])²
     */

    /* Initialize max to very negative */
    for (int i = 0; i < n; i++)
    {
        max_ll[i] = RBPF_REAL(-1e30);
    }

    /* Pass 1: Compute log-likelihoods for all components */
    for (int k = 0; k < KSC_N_COMPONENTS; k++)
    {
        const rbpf_real_t m_k = KSC_MEAN[k];
        const rbpf_real_t v2_k = KSC_VAR[k];
        const rbpf_real_t log_pi_k = KSC_LOG_PROB[k];
        const rbpf_real_t y_adj = y - m_k;

        rbpf_real_t *log_lik_k = log_lik_buf + k * n; /* Pointer to component k's buffer */

        RBPF_PRAGMA_SIMD
        for (int i = 0; i < n; i++)
        {
            /* Innovation */
            rbpf_real_t innov = y_adj - H * mu_pred[i];

            /* Innovation variance */
            rbpf_real_t S = H2 * var_pred[i] + v2_k;

            /* Log-likelihood: -0.5*(log(S) + innov²/S) + log(π_k) */
            rbpf_real_t innov2_S = innov * innov / S;
            rbpf_real_t log_lik = NEG_HALF * (rbpf_log(S) + innov2_S) + log_pi_k;

            log_lik_k[i] = log_lik;

            /* Track max for log-sum-exp */
            if (log_lik > max_ll[i])
                max_ll[i] = log_lik;
        }
    }

    /* Zero accumulators */
    memset(lik_total, 0, n * sizeof(rbpf_real_t));
    memset(mu_accum, 0, n * sizeof(rbpf_real_t));
    memset(var_accum, 0, n * sizeof(rbpf_real_t)); /* Now holds E[X²] = Σ wₖ(σₖ² + μₖ²) */

    /* Pass 2: Compute normalized likelihoods and accumulate */
    for (int k = 0; k < KSC_N_COMPONENTS; k++)
    {
        const rbpf_real_t m_k = KSC_MEAN[k];
        const rbpf_real_t v2_k = KSC_VAR[k];
        const rbpf_real_t y_adj = y - m_k;

        rbpf_real_t *log_lik_k = log_lik_buf + k * n;

        RBPF_PRAGMA_SIMD
        for (int i = 0; i < n; i++)
        {
            /* Stable exponential: exp(log_lik - max) */
            rbpf_real_t lik = rbpf_exp(log_lik_k[i] - max_ll[i]);

            /* Accumulate total likelihood */
            lik_total[i] += lik;

            /* Recompute Kalman update for this component */
            rbpf_real_t innov = y_adj - H * mu_pred[i];
            rbpf_real_t S = H2 * var_pred[i] + v2_k;
            rbpf_real_t K = H * var_pred[i] / S;
            rbpf_real_t mu_k = mu_pred[i] + K * innov;
            rbpf_real_t var_k = (RBPF_REAL(1.0) - K * H) * var_pred[i];

            /* Accumulate E[X] = Σ wₖ μₖ */
            mu_accum[i] += lik * mu_k;

            /* Accumulate E[X²] = Σ wₖ (σₖ² + μₖ²)
             * This captures BOTH the average variance AND the spread of means
             * (Law of total variance: Var = E[Var|K] + Var[E|K] = E[X²] - E[X]²) */
            var_accum[i] += lik * (var_k + mu_k * mu_k);
        }
    }

    /* Normalize and update weights */
    rbpf_real_t total_marginal = RBPF_REAL(0.0);
    for (int i = 0; i < n; i++)
    {
        rbpf_real_t inv_lik = RBPF_REAL(1.0) / (lik_total[i] + RBPF_REAL(1e-30));

        /* E[X] = Σ wₖ μₖ / Σ wₖ */
        rbpf_real_t mean_final = mu_accum[i] * inv_lik;

        /* E[X²] = Σ wₖ (σₖ² + μₖ²) / Σ wₖ */
        rbpf_real_t E_X2 = var_accum[i] * inv_lik;

        /* Var[X] = E[X²] - E[X]² (law of total variance) */
        rbpf_real_t var_final = E_X2 - mean_final * mean_final;

        mu[i] = mean_final;
        var[i] = var_final;

        /* Floor variance (should rarely trigger now with correct calculation) */
        if (var[i] < RBPF_REAL(1e-6))
            var[i] = RBPF_REAL(1e-6);

        /* Update log-weight: log(sum * exp(max)) = log(sum) + max */
        log_weight[i] += rbpf_log(lik_total[i] + RBPF_REAL(1e-30)) + max_ll[i];

        /* Marginal uses un-normalized likelihood */
        total_marginal += lik_total[i] * rbpf_exp(max_ll[i]);
    }

    return total_marginal / n;
}

/*─────────────────────────────────────────────────────────────────────────────
 * REGIME TRANSITION (LUT-based, no cumsum search)
 *───────────────────────────────────────────────────────────────────────────*/

void rbpf_ksc_transition(RBPF_KSC *rbpf)
{
    const int n = rbpf->n_particles;
    int *regime = rbpf->regime;
    rbpf_pcg32_t *rng = &rbpf->pcg[0];

    for (int i = 0; i < n; i++)
    {
        int r_old = regime[i];
        rbpf_real_t u = rbpf_pcg32_uniform(rng);
        int lut_idx = (int)(u * RBPF_REAL(1023.0));
        regime[i] = rbpf->trans_lut[r_old][lut_idx];
    }
}

/*─────────────────────────────────────────────────────────────────────────────
 * RESAMPLE (systematic + regularization)
 *───────────────────────────────────────────────────────────────────────────*/

int rbpf_ksc_resample(RBPF_KSC *rbpf)
{
    const int n = rbpf->n_particles;

    rbpf_real_t *log_weight = rbpf->log_weight;
    rbpf_real_t *w_norm = rbpf->w_norm;
    rbpf_real_t *cumsum = rbpf->cumsum;
    int *indices = rbpf->indices;

    /* Find max log-weight for numerical stability */
    rbpf_real_t max_lw = log_weight[0];
    for (int i = 1; i < n; i++)
    {
        if (log_weight[i] > max_lw)
            max_lw = log_weight[i];
    }

    /* Normalize: w = exp(lw - max) / sum */
    for (int i = 0; i < n; i++)
    {
        w_norm[i] = log_weight[i] - max_lw;
    }
    rbpf_vsExp(n, w_norm, w_norm);

    rbpf_real_t sum_w = rbpf_cblas_asum(n, w_norm, 1);
    if (sum_w < RBPF_REAL(1e-30))
    {
        /* All weights collapsed - reset to uniform */
        rbpf_real_t uw = rbpf->uniform_weight;
        for (int i = 0; i < n; i++)
        {
            w_norm[i] = uw;
        }
        sum_w = RBPF_REAL(1.0);
    }
    rbpf_cblas_scal(n, RBPF_REAL(1.0) / sum_w, w_norm, 1);

    /* Compute ESS */
    rbpf_real_t sum_w2 = rbpf_cblas_dot(n, w_norm, 1, w_norm, 1);
    rbpf_real_t ess = RBPF_REAL(1.0) / sum_w2;
    rbpf->last_ess = ess;

    /* Adaptive resampling threshold
     *
     * Normal mode: resample when ESS < 50% (standard)
     * Standard ESS threshold: resample when ESS < 50%
     */
    rbpf_real_t threshold = RBPF_REAL(0.5);

    /* Skip resample if ESS is high enough */
    if (ess > n * threshold)
    {
        return 0;
    }

    /* Cumulative sum */
    cumsum[0] = w_norm[0];
    for (int i = 1; i < n; i++)
    {
        cumsum[i] = cumsum[i - 1] + w_norm[i];
    }

    /* Fused systematic resampling + data copy
     * - Single pass: generate index and copy immediately
     * - Keeps source data in cache if selected multiple times */
    rbpf_real_t *mu = rbpf->mu;
    rbpf_real_t *var = rbpf->var;
    int *regime = rbpf->regime;
    rbpf_real_t *mu_tmp = rbpf->mu_tmp;
    rbpf_real_t *var_tmp = rbpf->var_tmp;
    int *regime_tmp = rbpf->regime_tmp;

    rbpf_real_t u0 = rbpf_pcg32_uniform(&rbpf->pcg[0]) * rbpf->inv_n;
    int j = 0;
    for (int i = 0; i < n; i++)
    {
        rbpf_real_t u = u0 + (rbpf_real_t)i * rbpf->inv_n;
        while (j < n - 1 && cumsum[j] < u)
            j++;

        /* Store index (still needed for Liu-West) */
        indices[i] = j;

        /* Copy immediately - keeps mu[j] hot in cache */
        mu_tmp[i] = mu[j];
        var_tmp[i] = var[j];
        regime_tmp[i] = regime[j];
    }

    /* Pointer swap (no memcpy!) */
    rbpf->mu = mu_tmp;
    rbpf->mu_tmp = mu;
    rbpf->var = var_tmp;
    rbpf->var_tmp = var;
    rbpf->regime = regime_tmp;
    rbpf->regime_tmp = regime;

    /* Reset log-weights to 0 */
    memset(rbpf->log_weight, 0, n * sizeof(rbpf_real_t));

    /* Apply regularization (kernel jitter)
     *
     * Two modes:
     * 1. Silverman (default): h = 0.9 × min(σ, IQR/1.34) × N^(-1/5)
     *    Automatically adapts to particle distribution. No ESS scaling needed.
     *
     * 2. Fixed (legacy): h = reg_bandwidth_mu × ESS_scale
     *    Uses ESS ratio to scale fixed bandwidth.
     */
    rbpf_real_t h_mu;

    if (rbpf->use_silverman_bandwidth && rbpf->silverman_scratch != NULL)
    {
        /* Silverman adaptive bandwidth - no ESS scaling needed */
#ifdef RBPF_USE_FLOAT
        h_mu = rbpf_silverman_bandwidth_f(mu, n, rbpf->silverman_scratch);
#else
        h_mu = (rbpf_real_t)rbpf_silverman_bandwidth(
            (const double *)mu, n, (double *)rbpf->silverman_scratch);
#endif

        /* Clamp to reasonable bounds */
        if (h_mu < RBPF_REAL(0.001))
            h_mu = RBPF_REAL(0.001);
        if (h_mu > RBPF_REAL(0.5))
            h_mu = RBPF_REAL(0.5);

        /* Store for diagnostics */
        rbpf->last_silverman_bandwidth = h_mu;
    }
    else
    {
        /* Fixed bandwidth with ESS scaling (legacy behavior) */
        rbpf_real_t ess_ratio = ess / (rbpf_real_t)n;
        rbpf_real_t scale = rbpf->reg_scale_max -
                            (rbpf->reg_scale_max - rbpf->reg_scale_min) * ess_ratio;
        if (scale < rbpf->reg_scale_min)
            scale = rbpf->reg_scale_min;
        if (scale > rbpf->reg_scale_max)
            scale = rbpf->reg_scale_max;

        h_mu = rbpf->reg_bandwidth_mu * scale;
    }

    /* Variance jitter: always use fixed small value
     * (Silverman on variance not needed - it's already tiny) */
    rbpf_real_t h_var = rbpf->reg_bandwidth_var;

    /* Generate Gaussian randoms in batch using MKL (ICDF method)
     * Much faster than scalar PCG32 calls in loop */
    rbpf_real_t *gauss = rbpf->rng_gaussian;
    RBPF_VSL_RNG_GAUSSIAN(VSL_RNG_METHOD_GAUSSIAN_ICDF, rbpf->mkl_rng[0],
                          2 * n, gauss, RBPF_REAL(0.0), RBPF_REAL(1.0));

    /* Apply jitter: first n randoms for mu, next n for var */
    mu = rbpf->mu;
    var = rbpf->var;
    regime = rbpf->regime;

    RBPF_PRAGMA_SIMD
    for (int i = 0; i < n; i++)
    {
        mu[i] += h_mu * gauss[i];
        var[i] += h_var * rbpf_fabs(gauss[n + i]);
        if (var[i] < RBPF_REAL(1e-6))
            var[i] = RBPF_REAL(1e-6);
    }

    /* Regime diversity preservation
     *
     * Problem: Standard resampling can kill minority regimes. When volatility
     * is calm, regime 3 (crisis) particles get low weight and die out. Later,
     * when a crisis hits, there are no regime 3 particles to respond!
     *
     * Solution: Ensure minimum particles per regime through two mechanisms:
     * 1. Random mutation: Some particles randomly switch regime
     * 2. Stratification: Force minimum count per regime (if needed)
     */
    if (rbpf->regime_mutation_prob > RBPF_REAL(0.0))
    {
        int n_regimes = rbpf->n_regimes;
        rbpf_pcg32_t *rng_mut = &rbpf->pcg[0];

        /* Count current regime distribution */
        int regime_count[RBPF_MAX_REGIMES] = {0};
        for (int i = 0; i < n; i++)
        {
            regime_count[regime[i]]++;
        }

        /* Find regimes that need more particles */
        int min_count = rbpf->min_particles_per_regime;

        for (int i = 0; i < n; i++)
        {
            int r = regime[i];

            /* Only mutate particles from over-represented regimes */
            if (regime_count[r] > min_count * 2)
            {
                if (rbpf_pcg32_uniform(rng_mut) < rbpf->regime_mutation_prob)
                {
                    /* Find under-represented regime */
                    for (int r_new = 0; r_new < n_regimes; r_new++)
                    {
                        if (regime_count[r_new] < min_count)
                        {
                            /* Mutate to new regime */
                            regime_count[r]--;
                            regime_count[r_new]++;
                            regime[i] = r_new;

                            /* Adapt state toward new regime's mu_vol */
                            rbpf_real_t mu_new = rbpf->params[r_new].mu_vol;
                            mu[i] = RBPF_REAL(0.7) * mu[i] + RBPF_REAL(0.3) * mu_new;
                            break;
                        }
                    }
                }
            }
        }
    }

    /*═══════════════════════════════════════════════════════════════════════
     * STUDENT-T RESAMPLE
     *
     * Resample the auxiliary variable λ arrays to match particle indices.
     *═══════════════════════════════════════════════════════════════════════*/
#if RBPF_ENABLE_STUDENT_T
    if (rbpf->student_t_enabled)
    {
        rbpf_ksc_resample_student_t(rbpf, indices);
    }
#endif

    return 1;
}

/*─────────────────────────────────────────────────────────────────────────────
 * COMPUTE OUTPUTS
 *───────────────────────────────────────────────────────────────────────────*/

void rbpf_ksc_compute_outputs(RBPF_KSC *rbpf, rbpf_real_t marginal_lik,
                              RBPF_KSC_Output *out)
{
    const int n = rbpf->n_particles;
    const int n_regimes = rbpf->n_regimes;

    rbpf_real_t *log_weight = rbpf->log_weight;
    rbpf_real_t *mu = rbpf->mu;
    rbpf_real_t *var = rbpf->var;
    int *regime = rbpf->regime;
    rbpf_real_t *w_norm = rbpf->w_norm;

    /* Normalize weights */
    rbpf_real_t max_lw = log_weight[0];
    for (int i = 1; i < n; i++)
    {
        if (log_weight[i] > max_lw)
            max_lw = log_weight[i];
    }

    for (int i = 0; i < n; i++)
    {
        w_norm[i] = log_weight[i] - max_lw;
    }
    rbpf_vsExp(n, w_norm, w_norm);

    rbpf_real_t sum_w = rbpf_cblas_asum(n, w_norm, 1);
    if (sum_w < RBPF_REAL(1e-30))
        sum_w = RBPF_REAL(1.0);
    rbpf_cblas_scal(n, RBPF_REAL(1.0) / sum_w, w_norm, 1);

    /* Log-vol mean and variance (using law of total variance) */
    rbpf_real_t log_vol_mean = RBPF_REAL(0.0);
    for (int i = 0; i < n; i++)
    {
        log_vol_mean += w_norm[i] * mu[i];
    }

    rbpf_real_t log_vol_var = RBPF_REAL(0.0);
    for (int i = 0; i < n; i++)
    {
        rbpf_real_t diff = mu[i] - log_vol_mean;
        /* Var[X] = E[Var[X|particle]] + Var[E[X|particle]] */
        log_vol_var += w_norm[i] * (var[i] + diff * diff);
    }

    out->log_vol_mean = log_vol_mean;
    out->log_vol_var = log_vol_var;

    /* Vol mean: TRUE Monte Carlo estimate over particle mixture
     *
     * Each particle i represents a Gaussian: ℓ ~ N(μ_i, σ²_i)
     * For a log-normal: E[exp(ℓ)|particle i] = exp(μ_i + ½σ²_i)
     *
     * True mixture expectation:
     *   E[exp(ℓ)] = Σ_i w_i × E[exp(ℓ)|i] = Σ_i w_i × exp(μ_i + ½var_i)
     *
     * This is more accurate than the single-Gaussian approximation:
     *   exp(E[ℓ] + ½Var[ℓ])  ← WRONG for mixtures
     */
    rbpf_real_t vol_mean = RBPF_REAL(0.0);
    for (int i = 0; i < n; i++)
    {
        vol_mean += w_norm[i] * rbpf_exp(mu[i] + RBPF_REAL(0.5) * var[i]);
    }
    out->vol_mean = vol_mean;

    /* ESS */
    rbpf_real_t sum_w2 = rbpf_cblas_dot(n, w_norm, 1, w_norm, 1);
    out->ess = RBPF_REAL(1.0) / sum_w2;

    /* Regime probabilities */
    memset(out->regime_probs, 0, sizeof(out->regime_probs));
    for (int i = 0; i < n; i++)
    {
        out->regime_probs[regime[i]] += w_norm[i];
    }

    /* Dominant regime */
    int dom = 0;
    rbpf_real_t max_prob = out->regime_probs[0];
    for (int r = 1; r < n_regimes; r++)
    {
        if (out->regime_probs[r] > max_prob)
        {
            max_prob = out->regime_probs[r];
            dom = r;
        }
    }
    out->dominant_regime = dom;

    RBPF_Detection *det = &rbpf->detection;

    /* SPRT regime detection
     *
     * Sequential Probability Ratio Test for statistically principled regime switching.
     * Uses log-likelihood ratios accumulated over time.
     *
     * For each regime k != current, accumulate:
     *   Λ_k = Σ log(P(y|k) / P(y|current))
     *
     * Switch to k if Λ_k > threshold_high (default: log(99) ≈ 4.6)
     * Reset Λ_k if it drops below threshold_low (default: log(0.01) ≈ -4.6)
     */
    int current = det->sprt_current_regime;
    det->sprt_ticks_in_current++;

    /* Current regime probability (with floor for numerical stability) */
    double p_current = (double)out->regime_probs[current];
    if (p_current < 1e-10)
        p_current = 1e-10;
    double log_p_current = log(p_current);

    int new_regime = current;
    double best_ratio = det->sprt_threshold_high; /* Need to exceed this */

    for (int k = 0; k < n_regimes; k++)
    {
        if (k == current)
            continue;

        /* Update log-likelihood ratio */
        double p_k = (double)out->regime_probs[k];
        if (p_k < 1e-10)
            p_k = 1e-10;
        double log_p_k = log(p_k);

        double delta = log_p_k - log_p_current;

        /* Clamp to prevent numerical explosion */
        if (delta > 10.0)
            delta = 10.0;
        if (delta < -10.0)
            delta = -10.0;

        det->sprt_log_ratios[k] += delta;

        /* Reset if evidence strongly against this regime */
        if (det->sprt_log_ratios[k] < det->sprt_threshold_low)
        {
            det->sprt_log_ratios[k] = 0.0;
        }

        /* Check if regime k should become new regime */
        if (det->sprt_log_ratios[k] > best_ratio &&
            det->sprt_ticks_in_current >= det->sprt_min_dwell)
        {
            best_ratio = det->sprt_log_ratios[k];
            new_regime = k;
        }
    }

    /* Switch regime if SPRT triggered */
    if (new_regime != current)
    {
        det->sprt_current_regime = new_regime;
        det->sprt_ticks_in_current = 0;

        /* Reset all log-ratios */
        for (int k = 0; k < n_regimes; k++)
        {
            det->sprt_log_ratios[k] = 0.0;
        }
    }

    out->smoothed_regime = det->sprt_current_regime;
    det->stable_regime = det->sprt_current_regime; /* Keep stable_regime in sync */

    /* Self-aware signals */
    out->marginal_lik = marginal_lik;
    out->surprise = -rbpf_log(marginal_lik + RBPF_REAL(1e-30));

    /* Regime entropy: -Σ p*log(p) */
    rbpf_real_t entropy = RBPF_REAL(0.0);
    for (int r = 0; r < n_regimes; r++)
    {
        rbpf_real_t p = out->regime_probs[r];
        if (p > RBPF_REAL(1e-10))
        {
            entropy -= p * rbpf_log(p);
        }
    }
    out->regime_entropy = entropy;

    /* Vol ratio (vs EMA) */
    det->vol_ema_short = RBPF_REAL(0.1) * out->vol_mean + RBPF_REAL(0.9) * det->vol_ema_short;
    det->vol_ema_long = RBPF_REAL(0.01) * out->vol_mean + RBPF_REAL(0.99) * det->vol_ema_long;
    out->vol_ratio = det->vol_ema_short / (det->vol_ema_long + RBPF_REAL(1e-10));

    /* Regime change detection */
    out->regime_changed = 0;
    out->change_type = 0;

    if (det->cooldown > 0)
    {
        det->cooldown--;
    }
    else
    {
        /* Structural: regime flipped with high confidence */
        int structural = (dom != det->prev_regime) && (max_prob > RBPF_REAL(0.7));

        /* Vol shock: >80% increase or >50% decrease */
        int vol_shock = (out->vol_ratio > 1.8f) || (out->vol_ratio < RBPF_REAL(0.5));

        /* Surprise: observation unlikely under model */
        int surprised = (out->surprise > RBPF_REAL(5.0));

        if (structural || vol_shock || surprised)
        {
            out->regime_changed = 1;
            out->change_type = structural ? 1 : (vol_shock ? 2 : 3);
            det->cooldown = 20; /* Suppress for 20 ticks */
        }
    }

    det->prev_regime = dom;

    /*========================================================================
     * FIXED-LAG SMOOTHING (Dual Output)
     *
     * Store current fast estimates in circular buffer.
     * Output K-lagged estimates for regime confirmation.
     *
     * This provides:
     *   - Fast signal (t):   Immediate reaction to volatility spikes
     *   - Smooth signal (t-K): Stable regime for position sizing
     *======================================================================*/

    const int lag = rbpf->smooth_lag;

    if (lag > 0)
    {
        /* Store current fast estimates at head position */
        RBPF_SmoothEntry *entry = &rbpf->smooth_history[rbpf->smooth_head];
        entry->vol_mean = out->vol_mean;
        entry->log_vol_mean = out->log_vol_mean;
        entry->log_vol_var = out->log_vol_var;
        entry->dominant_regime = out->dominant_regime;
        entry->ess = out->ess;
        entry->valid = 1;

        for (int r = 0; r < n_regimes; r++)
        {
            entry->regime_probs[r] = out->regime_probs[r];
        }

        /* Advance head (circular buffer) */
        rbpf->smooth_head = (rbpf->smooth_head + 1) % RBPF_MAX_SMOOTH_LAG;
        if (rbpf->smooth_count < lag)
        {
            rbpf->smooth_count++;
        }

        /* Output smooth signal if we have enough history */
        if (rbpf->smooth_count >= lag)
        {
            /* Read from K ticks ago (oldest valid entry) */
            int smooth_idx = (rbpf->smooth_head - lag + RBPF_MAX_SMOOTH_LAG) % RBPF_MAX_SMOOTH_LAG;
            const RBPF_SmoothEntry *smooth_entry = &rbpf->smooth_history[smooth_idx];

            out->smooth_valid = 1;
            out->smooth_lag = lag;
            out->vol_mean_smooth = smooth_entry->vol_mean;
            out->log_vol_mean_smooth = smooth_entry->log_vol_mean;
            out->log_vol_var_smooth = smooth_entry->log_vol_var;
            out->dominant_regime_smooth = smooth_entry->dominant_regime;

            for (int r = 0; r < n_regimes; r++)
            {
                out->regime_probs_smooth[r] = smooth_entry->regime_probs[r];
            }

            /* Regime confidence: max probability in smooth distribution */
            rbpf_real_t max_smooth_prob = out->regime_probs_smooth[0];
            for (int r = 1; r < n_regimes; r++)
            {
                if (out->regime_probs_smooth[r] > max_smooth_prob)
                {
                    max_smooth_prob = out->regime_probs_smooth[r];
                }
            }
            out->regime_confidence = max_smooth_prob;
        }
        else
        {
            /* Not enough history yet - output fast signal as fallback */
            out->smooth_valid = 0;
            out->smooth_lag = lag;
            out->vol_mean_smooth = out->vol_mean;
            out->log_vol_mean_smooth = out->log_vol_mean;
            out->log_vol_var_smooth = out->log_vol_var;
            out->dominant_regime_smooth = out->dominant_regime;
            out->regime_confidence = max_prob;

            for (int r = 0; r < n_regimes; r++)
            {
                out->regime_probs_smooth[r] = out->regime_probs[r];
            }
        }
    }
    else
    {
        /* Fixed-lag smoothing disabled - smooth = fast */
        out->smooth_valid = 1; /* Always valid when disabled (no lag) */
        out->smooth_lag = 0;
        out->vol_mean_smooth = out->vol_mean;
        out->log_vol_mean_smooth = out->log_vol_mean;
        out->log_vol_var_smooth = out->log_vol_var;
        out->dominant_regime_smooth = out->dominant_regime;
        out->regime_confidence = max_prob;

        for (int r = 0; r < n_regimes; r++)
        {
            out->regime_probs_smooth[r] = out->regime_probs[r];
        }
    }

    /*========================================================================
     * STUDENT-T OUTPUT DIAGNOSTICS
     *======================================================================*/
    out->student_t_active = 0;
    out->lambda_mean = RBPF_REAL(1.0);
    out->lambda_var = RBPF_REAL(0.0);
    out->nu_effective = RBPF_NU_CEIL;

#if RBPF_ENABLE_STUDENT_T
    if (rbpf->student_t_enabled && rbpf->lambda != NULL)
    {
        out->student_t_active = 1;

        /* Compute λ statistics */
        rbpf_real_t sum_lam = RBPF_REAL(0.0);
        rbpf_real_t sum_lam_sq = RBPF_REAL(0.0);

        for (int i = 0; i < n; i++)
        {
            sum_lam += w_norm[i] * rbpf->lambda[i];
            sum_lam_sq += w_norm[i] * rbpf->lambda[i] * rbpf->lambda[i];
        }

        out->lambda_mean = sum_lam;
        out->lambda_var = sum_lam_sq - sum_lam * sum_lam;
        if (out->lambda_var < RBPF_REAL(0.0))
            out->lambda_var = RBPF_REAL(0.0);

        /* Implied ν from observed λ variance: Var[λ] = 2/ν → ν = 2/Var[λ] */
        if (out->lambda_var > RBPF_REAL(0.01))
        {
            out->nu_effective = RBPF_REAL(2.0) / out->lambda_var;
        }
        else
        {
            out->nu_effective = RBPF_NU_CEIL; /* Near-Gaussian */
        }

        /* Copy learned ν estimates */
        for (int r = 0; r < n_regimes; r++)
        {
            if (rbpf->student_t[r].learn_nu)
            {
                out->learned_nu[r] = rbpf->student_t_stats[r].nu_estimate;
            }
            else
            {
                out->learned_nu[r] = rbpf->student_t[r].nu;
            }
        }
    }
#endif
}

/*─────────────────────────────────────────────────────────────────────────────
 * MAIN UPDATE - THE HOT PATH
 *───────────────────────────────────────────────────────────────────────────*/

void rbpf_ksc_step(RBPF_KSC *rbpf, rbpf_real_t obs, RBPF_KSC_Output *output)
{
    /* Transform observation: y = log(r²) */
    rbpf_real_t y;
    if (rbpf_fabs(obs) < RBPF_REAL(1e-10))
    {
        y = RBPF_REAL(-23.0); /* Floor at ~log(1e-10²) */
    }
    else
    {
        y = rbpf_log(obs * obs);
    }

    /* 1. Regime transition */
    rbpf_ksc_transition(rbpf);

    /* 2. Kalman predict */
    rbpf_ksc_predict(rbpf);

    /* 3. Mixture Kalman update
     *
     * Runtime branch: Gaussian vs Student-t observation model
     * When Student-t is enabled, route to specialized update function
     * that samples auxiliary variables λ and shifts observations.
     */
    rbpf_real_t marginal_lik;

#if RBPF_ENABLE_STUDENT_T
    if (rbpf->student_t_enabled)
    {
        marginal_lik = rbpf_ksc_update_student_t(rbpf, y);
    }
    else
    {
        marginal_lik = rbpf_ksc_update(rbpf, y);
    }
#else
    marginal_lik = rbpf_ksc_update(rbpf, y);
#endif

    /* 4. Compute outputs (before resample) */
    rbpf_ksc_compute_outputs(rbpf, marginal_lik, output);

    /* 5. Resample if needed (includes Liu-West update + Student-t λ resample) */
    output->resampled = rbpf_ksc_resample(rbpf);

    /* Output current regime parameters
     * (Either global params or learned from Storvik/Online EM via MMPF) */
    for (int r = 0; r < rbpf->n_regimes; r++)
    {
        output->learned_mu_vol[r] = rbpf->params[r].mu_vol;
        output->learned_sigma_vol[r] = rbpf->params[r].sigma_vol;
    }
}

/*─────────────────────────────────────────────────────────────────────────────
 * WARMUP
 *───────────────────────────────────────────────────────────────────────────*/

void rbpf_ksc_warmup(RBPF_KSC *rbpf)
{
    int n = rbpf->n_particles;

/* Force OpenMP thread creation */
#pragma omp parallel
    {
        volatile int tid = omp_get_thread_num();
        (void)tid;
    }

    /* Warmup MKL VML */
    rbpf_vsExp(n, rbpf->mu, rbpf->scratch1);
    rbpf_vsLn(n, rbpf->var, rbpf->scratch2);

    /* Warmup BLAS */
    volatile rbpf_real_t sum = rbpf_cblas_asum(n, rbpf->w_norm, 1);
    (void)sum;
}

/*─────────────────────────────────────────────────────────────────────────────
 * DEBUG
 *───────────────────────────────────────────────────────────────────────────*/

void rbpf_ksc_print_config(const RBPF_KSC *rbpf)
{
    printf("RBPF-KSC Configuration:\n");
    printf("  Particles:     %d\n", rbpf->n_particles);
    printf("  Regimes:       %d\n", rbpf->n_regimes);
    printf("  Threads:       %d\n", rbpf->n_threads);
    printf("  Reg bandwidth: mu=%.4f, var=%.4f\n",
           rbpf->reg_bandwidth_mu, rbpf->reg_bandwidth_var);
    printf("  Silverman:     %s (last h=%.4f)\n",
           rbpf->use_silverman_bandwidth ? "YES" : "NO",
           rbpf->last_silverman_bandwidth);

    printf("\n  Parameter Bounds:\n");
    printf("    μ_vol: [%.4f, %.4f]\n",
           rbpf->param_bounds.min_mu_vol, rbpf->param_bounds.max_mu_vol);
    printf("    σ_vol: [%.4f, %.4f]\n",
           rbpf->param_bounds.min_sigma_vol, rbpf->param_bounds.max_sigma_vol);
    printf("    Learned params: %s (set by MMPF Online EM)\n",
           rbpf->use_learned_params ? "ACTIVE" : "INACTIVE");

#if RBPF_ENABLE_STUDENT_T
    printf("\n  Student-t Observation Model:\n");
    printf("    Compiled:    YES\n");
    printf("    Enabled:     %s\n", rbpf->student_t_enabled ? "YES" : "NO");
    if (rbpf->student_t_enabled)
    {
        printf("    Per-regime ν:\n");
        for (int r = 0; r < rbpf->n_regimes; r++)
        {
            printf("      R%d: ν=%.2f%s\n", r, rbpf->student_t[r].nu,
                   rbpf->student_t[r].learn_nu ? " (learning)" : "");
        }
    }
#else
    printf("\n  Student-t Observation Model:\n");
    printf("    Compiled:    NO (Gaussian-only build)\n");
#endif

    printf("\n  Per-regime parameters (initial):\n");
    printf("  %-8s %8s %8s %8s\n", "Regime", "theta", "mu_vol", "sigma_vol");
    for (int r = 0; r < rbpf->n_regimes; r++)
    {
        const RBPF_RegimeParams *p = &rbpf->params[r];
        printf("  %-8d %8.4f %8.4f %8.4f\n",
               r, p->theta, p->mu_vol, p->sigma_vol);
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * STUDENT-T STUBS (when compiled without RBPF_ENABLE_STUDENT_T)
 *
 * These allow code to call Student-t API without #ifdefs everywhere.
 * The functions do nothing but the code compiles and links cleanly.
 *═══════════════════════════════════════════════════════════════════════════*/

#if !RBPF_ENABLE_STUDENT_T

void rbpf_ksc_enable_student_t(RBPF_KSC *rbpf, rbpf_real_t nu)
{
    (void)rbpf;
    (void)nu;
    /* No-op in Gaussian-only build */
}

void rbpf_ksc_disable_student_t(RBPF_KSC *rbpf)
{
    (void)rbpf;
}

void rbpf_ksc_set_student_t_nu(RBPF_KSC *rbpf, int regime, rbpf_real_t nu)
{
    (void)rbpf;
    (void)regime;
    (void)nu;
}

rbpf_real_t rbpf_ksc_get_nu(const RBPF_KSC *rbpf, int regime)
{
    (void)rbpf;
    (void)regime;
    return RBPF_REAL(1e30); /* "Infinite" ν = Gaussian */
}

void rbpf_ksc_enable_nu_learning(RBPF_KSC *rbpf, int regime, rbpf_real_t learning_rate)
{
    (void)rbpf;
    (void)regime;
    (void)learning_rate;
}

void rbpf_ksc_disable_nu_learning(RBPF_KSC *rbpf, int regime)
{
    (void)rbpf;
    (void)regime;
}

void rbpf_ksc_get_lambda_stats(const RBPF_KSC *rbpf, int regime,
                               rbpf_real_t *mean_out, rbpf_real_t *var_out,
                               rbpf_real_t *n_eff_out)
{
    (void)rbpf;
    (void)regime;
    if (mean_out)
        *mean_out = RBPF_REAL(1.0);
    if (var_out)
        *var_out = RBPF_REAL(0.0);
    if (n_eff_out)
        *n_eff_out = RBPF_REAL(0.0);
}

void rbpf_ksc_reset_nu_learning(RBPF_KSC *rbpf, int regime)
{
    (void)rbpf;
    (void)regime;
}

void rbpf_ksc_step_student_t(RBPF_KSC *rbpf, rbpf_real_t obs, RBPF_KSC_Output *output)
{
    /* Fall back to Gaussian step */
    rbpf_ksc_step(rbpf, obs, output);
}

void rbpf_ksc_step_student_t_nu(RBPF_KSC *rbpf, rbpf_real_t obs, rbpf_real_t nu,
                                RBPF_KSC_Output *output)
{
    (void)nu;
    rbpf_ksc_step(rbpf, obs, output);
}

int rbpf_ksc_alloc_student_t(RBPF_KSC *rbpf)
{
    (void)rbpf;
    return 0; /* Success (no-op) */
}

void rbpf_ksc_free_student_t(RBPF_KSC *rbpf)
{
    (void)rbpf;
}

void rbpf_ksc_resample_student_t(RBPF_KSC *rbpf, const int *indices)
{
    (void)rbpf;
    (void)indices;
}

rbpf_real_t rbpf_ksc_update_student_t(RBPF_KSC *rbpf, rbpf_real_t y)
{
    /* Fall back to Gaussian update when Student-t is disabled */
    return rbpf_ksc_update(rbpf, y);
}

rbpf_real_t rbpf_ksc_update_student_t_robust(RBPF_KSC *rbpf, rbpf_real_t y,
                                             rbpf_real_t nu,
                                             const RBPF_RobustOCSN *ocsn)
{
    (void)nu;
    /* Fall back to OCSN-only when Student-t is disabled */
    if (ocsn && ocsn->enabled)
    {
        return rbpf_ksc_update_robust(rbpf, y, ocsn);
    }
    return rbpf_ksc_update(rbpf, y);
}

#endif /* !RBPF_ENABLE_STUDENT_T */