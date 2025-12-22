/**
 * @file rbpf_ksc_init.c
 * @brief RBPF-KSC Lifecycle and Configuration
 *
 * This file contains:
 *   - create() / destroy()
 *   - All set_*() configuration functions
 *   - init()
 *   - warmup()
 *   - print_config()
 *   - Student-t stubs (when compiled without RBPF_ENABLE_STUDENT_T)
 *
 * Hot path is in rbpf_ksc.c
 * Output computation is in rbpf_ksc_output.c
 */

#include "rbpf_ksc.h"
#include "rbpf_sprt.h"
#include "rbpf_dirichlet_transition.h"
#include "bocpd.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <mkl_vml.h>

/*═══════════════════════════════════════════════════════════════════════════
 * STUDENT-T COMPILE-TIME SWITCH
 *═══════════════════════════════════════════════════════════════════════════*/

#ifndef RBPF_ENABLE_STUDENT_T
#define RBPF_ENABLE_STUDENT_T 1
#endif

#if RBPF_ENABLE_STUDENT_T
extern int rbpf_ksc_alloc_student_t(RBPF_KSC *rbpf);
extern void rbpf_ksc_free_student_t(RBPF_KSC *rbpf);
#endif

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
 * CREATE
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

    /* Log-sum-exp buffers */
    rbpf->log_lik_buffer = aligned_alloc_real(KSC_N_COMPONENTS * n);
    rbpf->max_log_lik = aligned_alloc_real(n);

    /* Pre-generated Gaussian buffer for jitter */
    rbpf->rng_gaussian = aligned_alloc_real(2 * n);
    rbpf->rng_buffer_size = 2 * n;

    /* Silverman bandwidth scratch buffer */
    rbpf->silverman_scratch = aligned_alloc_real(n);

    /*═══════════════════════════════════════════════════════════════════════
     * KL TEMPERING BUFFER
     *
     * Stores log-likelihood increments for deferred weight application.
     * When deferred_weight_mode=1, rbpf_ksc_update() stores increments here
     * but does NOT apply them. The Extended layer then computes KL divergence
     * and applies tempered weights: log_weight += β × log_lik_increment
     *═══════════════════════════════════════════════════════════════════════*/
    rbpf->log_lik_increment = aligned_alloc_real(n);
    rbpf->deferred_weight_mode = 0; /* Default: immediate weight application */

    /* Check allocations */
    if (!rbpf->mu || !rbpf->var || !rbpf->regime || !rbpf->log_weight ||
        !rbpf->mu_tmp || !rbpf->var_tmp || !rbpf->regime_tmp ||
        !rbpf->mu_pred || !rbpf->var_pred || !rbpf->theta_arr ||
        !rbpf->mu_vol_arr || !rbpf->q_arr || !rbpf->lik_total ||
        !rbpf->lik_comp || !rbpf->innov || !rbpf->S || !rbpf->K ||
        !rbpf->w_norm || !rbpf->cumsum || !rbpf->mu_accum || !rbpf->var_accum ||
        !rbpf->scratch1 || !rbpf->scratch2 || !rbpf->indices ||
        !rbpf->log_lik_buffer || !rbpf->max_log_lik || !rbpf->rng_gaussian ||
        !rbpf->silverman_scratch || !rbpf->log_lik_increment)
    {
        rbpf_ksc_destroy(rbpf);
        return NULL;
    }

    /* Zero the KL tempering buffer */
    memset(rbpf->log_lik_increment, 0, n * sizeof(rbpf_real_t));

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
        rbpf->params[r].mu_vol = rbpf_log(RBPF_REAL(0.01));
        rbpf->params[r].sigma_vol = RBPF_REAL(0.1);
        rbpf->params[r].q = RBPF_REAL(0.01);
    }

    /* Regularization defaults */
    rbpf->reg_bandwidth_mu = RBPF_REAL(0.02);
    rbpf->reg_bandwidth_var = RBPF_REAL(0.0005);
    rbpf->reg_scale_min = RBPF_REAL(0.1);
    rbpf->reg_scale_max = RBPF_REAL(0.5);
    rbpf->last_ess = (rbpf_real_t)n;

    /* Silverman adaptive bandwidth */
    rbpf->use_silverman_bandwidth = 1;
    rbpf->last_silverman_bandwidth = RBPF_REAL(0.0);

    /* Regime diversity - "Pilot Light" safety net
     * Prevents P(regime=k) = 0 (absorbing state).
     * Primary regime switching handled by BOCPD + SPRT,
     * but we keep 1-2 particles per regime as mathematical insurance.
     * Cost: negligible (2 particles out of 1000 = 0.2% noise). */
    rbpf->min_particles_per_regime = 2;
    rbpf->regime_mutation_prob = RBPF_REAL(0.001); /* 0.1% */

    /* Detection state */
    rbpf->detection.vol_ema_short = RBPF_REAL(0.01);
    rbpf->detection.vol_ema_long = RBPF_REAL(0.01);
    rbpf->detection.prev_regime = 0;
    rbpf->detection.cooldown = 0;

    /* SPRT regime detection - use dedicated module */
    sprt_multi_init(&rbpf->sprt, n_regimes, 0.01, 0.01, 3);

    /* Dirichlet transition learning - disabled by default
     *
     * Initialize with geometry-aware prior based on regime distances.
     * The prior encodes: "nearby regimes are more likely transitions"
     * Learning happens when SPRT confirms regime changes.
     */
    {
        float mu_vol_init[RBPF_MAX_REGIMES];
        for (int r = 0; r < n_regimes; r++)
        {
            mu_vol_init[r] = (float)rbpf->params[r].mu_vol;
        }

        dirichlet_transition_init_geometric(
            &rbpf->trans_prior,
            n_regimes,
            mu_vol_init,
            30.0f, /* stickiness: moderate */
            1.0f,  /* distance_scale: 1 log-vol unit */
            0.999f /* gamma: ~1000 tick memory */
        );
    }
    rbpf->trans_prior_enabled = 0; /* Off by default - use fixed matrix */

    rbpf->detection.stable_regime = 0;

    /* BOCPD changepoint detection - disabled by default
     * Call rbpf_ksc_attach_bocpd() to enable */
    rbpf->bocpd = NULL;
    rbpf->bocpd_delta = NULL;
    rbpf->bocpd_hazard = NULL;
    rbpf->bocpd_threshold = 3.0;     /* z-score threshold for detection */
    rbpf->bocpd_decay = 0.995;       /* Decay for delta detector */
    rbpf->bocpd_learn_window = 1000; /* Window for hazard learning */

    /* Fixed-lag smoothing */
    rbpf->smooth_lag = 0;
    rbpf->smooth_head = 0;
    rbpf->smooth_count = 0;
    for (int i = 0; i < RBPF_MAX_SMOOTH_LAG; i++)
    {
        rbpf->smooth_history[i].valid = 0;
    }

    /* Liu-West parameter learning arrays */
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

    /* Parameter bounds */
    rbpf->param_bounds.min_mu_vol = rbpf_log(0.001f);
    rbpf->param_bounds.max_mu_vol = rbpf_log(0.5f);
    rbpf->param_bounds.min_sigma_vol = RBPF_REAL(0.01);
    rbpf->param_bounds.max_sigma_vol = RBPF_REAL(1.0);

    rbpf->use_learned_params = 0;

    /* Student-t allocation */
#if RBPF_ENABLE_STUDENT_T
    if (rbpf_ksc_alloc_student_t(rbpf) < 0)
    {
        rbpf_ksc_destroy(rbpf);
        return NULL;
    }
#else
    rbpf->lambda = NULL;
    rbpf->lambda_tmp = NULL;
    rbpf->log_lambda = NULL;
    rbpf->student_t_enabled = 0;
#endif

    /* MKL fast math mode */
    vmlSetMode(VML_EP | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE);

    return rbpf;
}

/*─────────────────────────────────────────────────────────────────────────────
 * DESTROY
 *───────────────────────────────────────────────────────────────────────────*/

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

    mkl_free(rbpf->log_lik_buffer);
    mkl_free(rbpf->max_log_lik);
    mkl_free(rbpf->rng_gaussian);
    mkl_free(rbpf->silverman_scratch);

    /* KL tempering buffer */
    mkl_free(rbpf->log_lik_increment);

    mkl_free(rbpf->particle_mu_vol);
    mkl_free(rbpf->particle_sigma_vol);
    mkl_free(rbpf->particle_mu_vol_tmp);
    mkl_free(rbpf->particle_sigma_vol_tmp);

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
}

void rbpf_ksc_build_transition_lut(RBPF_KSC *rbpf, const rbpf_real_t *trans_matrix)
{
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
    if (!rbpf)
        return;

    /* Clamp to reasonable bounds */
    if (min_per_regime < 0)
        min_per_regime = 0;
    if (min_per_regime > rbpf->n_particles / 4)
        min_per_regime = rbpf->n_particles / 4;

    if (mutation_prob < RBPF_REAL(0.0))
        mutation_prob = RBPF_REAL(0.0);
    if (mutation_prob > RBPF_REAL(0.05)) /* Cap at 5% - was 20%, now stricter */
        mutation_prob = RBPF_REAL(0.05);

    rbpf->min_particles_per_regime = min_per_regime;
    rbpf->regime_mutation_prob = mutation_prob;
}

void rbpf_ksc_set_sprt_params(RBPF_KSC *rbpf, double alpha, double beta, int min_dwell)
{
    if (!rbpf)
        return;

    /* Clamp error rates */
    if (alpha < 0.001)
        alpha = 0.001;
    if (alpha > 0.5)
        alpha = 0.5;
    if (beta < 0.001)
        beta = 0.001;
    if (beta > 0.5)
        beta = 0.5;
    if (min_dwell < 1)
        min_dwell = 1;
    if (min_dwell > 100)
        min_dwell = 100;

    /* Re-initialize SPRT module with new params */
    sprt_multi_init(&rbpf->sprt, rbpf->n_regimes, alpha, beta, min_dwell);
}

void rbpf_ksc_reset_sprt(RBPF_KSC *rbpf)
{
    if (!rbpf)
        return;

    /* Reset all pairwise tests */
    for (int i = 0; i < rbpf->sprt.n_regimes; i++)
    {
        for (int j = i + 1; j < rbpf->sprt.n_regimes; j++)
        {
            sprt_binary_reset(&rbpf->sprt.tests[i][j]);
        }
        rbpf->sprt.regime_evidence[i] = 0.0;
    }
    rbpf->sprt.ticks_in_current = 0;
}

void rbpf_ksc_force_sprt_regime(RBPF_KSC *rbpf, int regime)
{
    if (!rbpf)
        return;
    if (regime < 0 || regime >= rbpf->n_regimes)
        return;

    /* Use SPRT module's force function */
    sprt_multi_force_regime(&rbpf->sprt, regime);
    rbpf->detection.stable_regime = regime;
}

/*─────────────────────────────────────────────────────────────────────────────
 * KL TEMPERING CONFIGURATION
 *───────────────────────────────────────────────────────────────────────────*/

void rbpf_ksc_set_deferred_weight_mode(RBPF_KSC *rbpf, int enable)
{
    if (!rbpf)
        return;
    rbpf->deferred_weight_mode = enable;
}

int rbpf_ksc_get_deferred_weight_mode(const RBPF_KSC *rbpf)
{
    return rbpf ? rbpf->deferred_weight_mode : 0;
}

rbpf_real_t *rbpf_ksc_get_log_lik_increment(RBPF_KSC *rbpf)
{
    return rbpf ? rbpf->log_lik_increment : NULL;
}

void rbpf_ksc_apply_weight_increments(RBPF_KSC *rbpf, rbpf_real_t beta)
{
    if (!rbpf)
        return;

    const int n = rbpf->n_particles;
    rbpf_real_t *log_weight = rbpf->log_weight;
    const rbpf_real_t *log_lik_inc = rbpf->log_lik_increment;

    for (int i = 0; i < n; i++)
    {
        log_weight[i] += beta * log_lik_inc[i];
    }
}

/*─────────────────────────────────────────────────────────────────────────────
 * BOCPD CHANGEPOINT DETECTION INTEGRATION
 *
 * BOCPD ("Afterburner") provides event-driven regime switching.
 * Complements the pilot light mutation for robust regime tracking.
 *───────────────────────────────────────────────────────────────────────────*/

void rbpf_ksc_attach_bocpd(RBPF_KSC *rbpf,
                           bocpd_t *bocpd,
                           bocpd_delta_detector_t *delta,
                           bocpd_hazard_t *hazard)
{
    if (!rbpf)
        return;

    rbpf->bocpd = bocpd;
    rbpf->bocpd_delta = delta;
    rbpf->bocpd_hazard = hazard;
}

void rbpf_ksc_detach_bocpd(RBPF_KSC *rbpf)
{
    if (!rbpf)
        return;

    rbpf->bocpd = NULL;
    rbpf->bocpd_delta = NULL;
    rbpf->bocpd_hazard = NULL;
}

void rbpf_ksc_set_bocpd_params(RBPF_KSC *rbpf,
                               double z_threshold,
                               double decay,
                               size_t learn_window)
{
    if (!rbpf)
        return;

    if (z_threshold < 1.0)
        z_threshold = 1.0;
    if (z_threshold > 10.0)
        z_threshold = 10.0;

    if (decay < 0.9)
        decay = 0.9;
    if (decay > 0.9999)
        decay = 0.9999;

    rbpf->bocpd_threshold = z_threshold;
    rbpf->bocpd_decay = decay;
    rbpf->bocpd_learn_window = learn_window;
}

int rbpf_ksc_bocpd_attached(const RBPF_KSC *rbpf)
{
    return (rbpf && rbpf->bocpd != NULL && rbpf->bocpd_delta != NULL);
}

void rbpf_ksc_set_fixed_lag_smoothing(RBPF_KSC *rbpf, int lag)
{
    if (lag < 0)
        lag = 0;
    if (lag > RBPF_MAX_SMOOTH_LAG)
        lag = RBPF_MAX_SMOOTH_LAG;

    rbpf->smooth_lag = lag;
    rbpf->smooth_head = 0;
    rbpf->smooth_count = 0;

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
 * INITIALIZATION
 *───────────────────────────────────────────────────────────────────────────*/

void rbpf_ksc_init(RBPF_KSC *rbpf, rbpf_real_t mu0, rbpf_real_t var0)
{
    int n = rbpf->n_particles;
    int n_regimes = rbpf->n_regimes;

    rbpf_real_t state_spread = RBPF_REAL(0.1);

    for (int i = 0; i < n; i++)
    {
        rbpf_real_t noise = rbpf_pcg32_gaussian(&rbpf->pcg[0]) * state_spread;
        rbpf->mu[i] = mu0 + noise;
        rbpf->var[i] = var0;
        rbpf->regime[i] = i % n_regimes;
        rbpf->log_weight[i] = RBPF_REAL(0.0);
    }

    /* Initialize per-particle parameters from global params */
    rbpf_real_t param_spread_mu = RBPF_REAL(0.5);
    rbpf_real_t param_spread_sigma = RBPF_REAL(0.08);

    for (int i = 0; i < n; i++)
    {
        for (int r = 0; r < n_regimes; r++)
        {
            int idx = i * n_regimes + r;

            rbpf_real_t jitter_mu = rbpf_pcg32_gaussian(&rbpf->pcg[0]) * param_spread_mu;
            rbpf_real_t jitter_sigma = rbpf_pcg32_gaussian(&rbpf->pcg[0]) * param_spread_sigma;

            rbpf_real_t mu_vol = rbpf->params[r].mu_vol + jitter_mu;
            rbpf_real_t sigma_vol = rbpf->params[r].sigma_vol + rbpf_fabs(jitter_sigma);

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

        /* ORDER CONSTRAINT */
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

        /* MINIMUM SEPARATION */
        const rbpf_real_t min_sep_init = RBPF_REAL(0.5);
        for (int r = 1; r < n_regimes; r++)
        {
            int idx_prev = i * n_regimes + (r - 1);
            int idx_curr = i * n_regimes + r;

            rbpf_real_t gap = rbpf->particle_mu_vol[idx_curr] - rbpf->particle_mu_vol[idx_prev];
            if (gap < min_sep_init)
            {
                rbpf->particle_mu_vol[idx_curr] = rbpf->particle_mu_vol[idx_prev] + min_sep_init;

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

    /* Initialize last_y for SPRT (neutral observation at initial vol) */
    rbpf->last_y = RBPF_REAL(2.0) * mu0;

    /* Reset SPRT module */
    rbpf_ksc_reset_sprt(rbpf);
    rbpf->detection.stable_regime = 0;

    /* Reset fixed-lag smoothing */
    rbpf->smooth_head = 0;
    rbpf->smooth_count = 0;
    for (int i = 0; i < RBPF_MAX_SMOOTH_LAG; i++)
    {
        rbpf->smooth_history[i].valid = 0;
    }

    /* Reset KL tempering buffer */
    memset(rbpf->log_lik_increment, 0, n * sizeof(rbpf_real_t));

    /* Student-t initialization */
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

/*═══════════════════════════════════════════════════════════════════════════════
 * DIRICHLET TRANSITION LEARNING CONFIGURATION
 *═══════════════════════════════════════════════════════════════════════════════*/

void rbpf_ksc_enable_transition_learning(RBPF_KSC *rbpf, int enable)
{
    if (!rbpf)
        return;

    rbpf->trans_prior_enabled = enable;

    if (enable)
    {
        /* Sync Dirichlet prior to current transition LUT */
        /* This ensures continuity if switching mid-run */
        printf("  Transition learning ENABLED (Discounted Dirichlet, γ=%.4f)\n",
               rbpf->trans_prior.gamma);
    }
}

void rbpf_ksc_set_transition_learning_params(RBPF_KSC *rbpf,
                                             float stickiness,
                                             float distance_scale,
                                             float gamma)
{
    if (!rbpf)
        return;

    /* Clamp parameters to reasonable ranges */
    if (stickiness < 1.0f)
        stickiness = 1.0f;
    if (stickiness > 1000.0f)
        stickiness = 1000.0f;

    if (distance_scale < 0.1f)
        distance_scale = 0.1f;
    if (distance_scale > 10.0f)
        distance_scale = 10.0f;

    if (gamma < 0.9f)
        gamma = 0.9f;
    if (gamma > 0.9999f)
        gamma = 0.9999f;

    /* Re-initialize with new parameters */
    float mu_vol_init[RBPF_MAX_REGIMES];
    for (int r = 0; r < rbpf->n_regimes; r++)
    {
        mu_vol_init[r] = (float)rbpf->params[r].mu_vol;
    }

    dirichlet_transition_init_geometric(
        &rbpf->trans_prior,
        rbpf->n_regimes,
        mu_vol_init,
        stickiness,
        distance_scale,
        gamma);
}

float rbpf_ksc_get_transition_prob(const RBPF_KSC *rbpf, int from, int to)
{
    if (!rbpf)
        return 0.0f;
    if (from < 0 || from >= rbpf->n_regimes)
        return 0.0f;
    if (to < 0 || to >= rbpf->n_regimes)
        return 0.0f;

    if (rbpf->trans_prior_enabled)
    {
        return dirichlet_transition_prob(&rbpf->trans_prior, from, to);
    }
    else
    {
        /* Return from fixed LUT (approximate - LUT is discretized) */
        /* Count how many LUT entries map to 'to' */
        int count = 0;
        for (int i = 0; i < 1024; i++)
        {
            if (rbpf->trans_lut[from][i] == (uint8_t)to)
                count++;
        }
        return (float)count / 1024.0f;
    }
}

void rbpf_ksc_print_transition_prior(const RBPF_KSC *rbpf)
{
    if (!rbpf)
        return;

    printf("\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  Transition Learning Status\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  Enabled: %s\n", rbpf->trans_prior_enabled ? "YES" : "NO");

    if (rbpf->trans_prior_enabled)
    {
        dirichlet_transition_print(&rbpf->trans_prior);
    }
    else
    {
        printf("  Using fixed transition matrix (call enable_transition_learning to learn)\n");
        printf("\n  Current fixed transition probabilities:\n");
        printf("       ");
        for (int j = 0; j < rbpf->n_regimes; j++)
            printf("    R%d   ", j);
        printf("\n");

        for (int i = 0; i < rbpf->n_regimes; i++)
        {
            printf("  R%d: ", i);
            for (int j = 0; j < rbpf->n_regimes; j++)
            {
                float p = rbpf_ksc_get_transition_prob(rbpf, i, j);
                if (i == j)
                    printf(" [%5.1f%%]", p * 100.0f);
                else
                    printf("  %5.1f%% ", p * 100.0f);
            }
            printf("\n");
        }
    }
    printf("═══════════════════════════════════════════════════════════════\n");
}

/*─────────────────────────────────────────────────────────────────────────────
 * WARMUP
 *───────────────────────────────────────────────────────────────────────────*/

void rbpf_ksc_warmup(RBPF_KSC *rbpf)
{
    int n = rbpf->n_particles;

#pragma omp parallel
    {
        volatile int tid = omp_get_thread_num();
        (void)tid;
    }

    rbpf_vsExp(n, rbpf->mu, rbpf->scratch1);
    rbpf_vsLn(n, rbpf->var, rbpf->scratch2);

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

    printf("\n  KL Tempering:\n");
    printf("    Deferred weight mode: %s\n",
           rbpf->deferred_weight_mode ? "ENABLED" : "DISABLED");

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

    printf("\n  Transition Learning:\n");
    printf("    Enabled:     %s\n", rbpf->trans_prior_enabled ? "YES" : "NO");
    if (rbpf->trans_prior_enabled)
    {
        DirichletTransitionStats stats = dirichlet_transition_stats(&rbpf->trans_prior);
        printf("    γ (decay):   %.4f (~%.0f tick memory)\n",
               rbpf->trans_prior.gamma,
               1.0f / (1.0f - rbpf->trans_prior.gamma));
        printf("    Avg sticky:  %.1f%%\n", stats.avg_stickiness * 100.0f);
        printf("    Transitions: %d observed\n", stats.total_transitions);
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * STUDENT-T STUBS (when compiled without RBPF_ENABLE_STUDENT_T)
 *═══════════════════════════════════════════════════════════════════════════*/

#if !RBPF_ENABLE_STUDENT_T

void rbpf_ksc_enable_student_t(RBPF_KSC *rbpf, rbpf_real_t nu)
{
    (void)rbpf;
    (void)nu;
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
    return RBPF_REAL(1e30);
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
    return 0;
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