/**
 * @file mmpf_mcmc.c
 * @brief MCMC Move Step for Shock Response - Particle Teleportation
 *
 * AVX2 + MKL optimized implementation of Metropolis-Hastings MCMC
 * for instant tracking of volatility shocks.
 */

#include "mmpf_mcmc.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

#ifdef __AVX2__
#include <immintrin.h>
#endif

/* MKL for batch RNG and aligned allocation */
#include <mkl.h>
#include <mkl_vsl.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/*═══════════════════════════════════════════════════════════════════════════
 * LOG-CHI-SQUARED LIKELIHOOD CONSTANTS
 *
 * For y = log(r²) = 2h + log(ε²), where ε ~ N(0,1):
 * log(ε²) ~ log-χ²(1), which has mean ≈ -1.27 and variance ≈ 4.93
 *
 * Fast approximation: treat as single Gaussian N(-1.27, 4.93)
 *═══════════════════════════════════════════════════════════════════════════*/

#define LCHI_MEAN -1.2704     /* E[log(χ²(1))] = ψ(0.5) + log(2) */
#define LCHI_VAR 4.9348       /* Var[log(χ²(1))] */
#define LCHI_STD 2.2215       /* sqrt(variance) */
#define LCHI_INV_VAR 0.2026   /* 1 / variance */
#define LCHI_LOG_NORM -1.8379 /* -0.5 * log(2π × variance) */

/*═══════════════════════════════════════════════════════════════════════════
 * CONFIGURATION
 *═══════════════════════════════════════════════════════════════════════════*/

void mmpf_mcmc_config_defaults(MMPF_MCMC_Config *cfg)
{
    cfg->n_steps = 5;
    cfg->proposal_sigma = 1.5;
    cfg->var_reset = 1.0;
    cfg->min_log_vol = -10.0;
    cfg->max_log_vol = 2.0;
    cfg->flatten_transitions = 1;
}

/*═══════════════════════════════════════════════════════════════════════════
 * MMPF INTEGRATION: INIT / DESTROY
 *═══════════════════════════════════════════════════════════════════════════*/

void mmpf_mcmc_init(MMPF_ROCKS *mmpf, int n_part, int n_models, int n_steps)
{
    /* Initialize scratch buffers directly */
    int total = n_part * n_steps;
    total = (total + 7) & ~7; /* Round up to cache line */

    mmpf->mcmc_scratch.rng_gauss = (double *)mkl_malloc(total * sizeof(double), 64);
    mmpf->mcmc_scratch.rng_log_u = (double *)mkl_malloc(total * sizeof(double), 64);
    mmpf->mcmc_scratch.capacity = total;

    (void)n_models; /* Reserved for future use */

    /* Create dedicated VSL stream for MCMC */
    vslNewStream((VSLStreamStatePtr *)&mmpf->mcmc_vsl_stream, VSL_BRNG_SFMT19937,
                 (unsigned int)(12345 + (size_t)mmpf));

    /* Initialize statistics */
    memset(&mmpf->mcmc_stats, 0, sizeof(mmpf->mcmc_stats));
}

void mmpf_mcmc_destroy(MMPF_ROCKS *mmpf)
{
    /* Free scratch buffers */
    if (mmpf->mcmc_scratch.rng_gauss)
    {
        mkl_free(mmpf->mcmc_scratch.rng_gauss);
        mkl_free(mmpf->mcmc_scratch.rng_log_u);
        mmpf->mcmc_scratch.rng_gauss = NULL;
        mmpf->mcmc_scratch.rng_log_u = NULL;
        mmpf->mcmc_scratch.capacity = 0;
    }

    if (mmpf->mcmc_vsl_stream)
    {
        vslDeleteStream((VSLStreamStatePtr *)&mmpf->mcmc_vsl_stream);
        mmpf->mcmc_vsl_stream = NULL;
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * SCALAR LIKELIHOOD
 *═══════════════════════════════════════════════════════════════════════════*/

double mmpf_mcmc_loglik(double y_log_sq, double h)
{
    /* y = log(r²) = 2h + log(ε²)
     * y - 2h ~ log-χ²(1) ≈ N(-1.27, 4.93)
     *
     * log P(y|h) ∝ -0.5 × (y - 2h - (-1.27))² / 4.93
     *            = -0.5 × (y - 2h + 1.27)² / 4.93
     */
    double residual = y_log_sq - 2.0 * h + LCHI_MEAN;
    return -0.5 * residual * residual * LCHI_INV_VAR;
}

/*═══════════════════════════════════════════════════════════════════════════
 * AVX2 VECTORIZED LIKELIHOOD
 *═══════════════════════════════════════════════════════════════════════════*/

#ifdef __AVX2__

__m256d mmpf_mcmc_loglik_avx(__m256d h_vec, double y_log_sq)
{
    /* Constants */
    const __m256d v_two = _mm256_set1_pd(2.0);
    const __m256d v_y = _mm256_set1_pd(y_log_sq);
    const __m256d v_lchi_mean = _mm256_set1_pd(LCHI_MEAN);
    const __m256d v_neg_half_inv_var = _mm256_set1_pd(-0.5 * LCHI_INV_VAR);

    /* residual = y - 2h + LCHI_MEAN */
    __m256d two_h = _mm256_mul_pd(v_two, h_vec);
    __m256d residual = _mm256_sub_pd(v_y, two_h);
    residual = _mm256_add_pd(residual, v_lchi_mean);

    /* log_lik = -0.5 × residual² / variance */
    __m256d residual_sq = _mm256_mul_pd(residual, residual);
    __m256d log_lik = _mm256_mul_pd(residual_sq, v_neg_half_inv_var);

    return log_lik;
}

#endif /* __AVX2__ */

/*═══════════════════════════════════════════════════════════════════════════
 * SCALAR MCMC IMPLEMENTATION
 *═══════════════════════════════════════════════════════════════════════════*/

static void mcmc_scalar(MMPF_ROCKS *mmpf, double y_log_sq,
                        const MMPF_MCMC_Config *cfg)
{
    int n_steps = cfg->n_steps;
    double sigma = cfg->proposal_sigma;
    double var_reset = cfg->var_reset;

    /* For each model */
    for (int k = 0; k < MMPF_N_MODELS; k++)
    {
        RBPF_KSC *rbpf = mmpf->ext[k]->rbpf;
        int n = rbpf->n_particles;
        double *mu = (double *)rbpf->mu;
        double *var = (double *)rbpf->var;

        /* MCMC for each particle */
        for (int i = 0; i < n; i++)
        {
            double h_curr = mu[i];
            double ll_curr = mmpf_mcmc_loglik(y_log_sq, h_curr);

            /* Run MH steps */
            for (int step = 0; step < n_steps; step++)
            {
                /* Generate randoms inline using MKL */
                double noise, u;
                vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER,
                              (VSLStreamStatePtr)mmpf->mcmc_vsl_stream, 1, &noise, 0.0, sigma);
                vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD,
                             (VSLStreamStatePtr)mmpf->mcmc_vsl_stream, 1, &u, 0.0, 1.0);

                /* Propose */
                double h_prop = h_curr + noise;

                /* Clamp to bounds */
                if (h_prop < cfg->min_log_vol)
                    h_prop = cfg->min_log_vol;
                if (h_prop > cfg->max_log_vol)
                    h_prop = cfg->max_log_vol;

                /* Evaluate */
                double ll_prop = mmpf_mcmc_loglik(y_log_sq, h_prop);

                /* Accept/reject */
                double log_alpha = ll_prop - ll_curr;
                if (log_alpha > 0.0 || log(u + 1e-10) < log_alpha)
                {
                    h_curr = h_prop;
                    ll_curr = ll_prop;
                }
            }

            /* Store result */
            mu[i] = h_curr;
            var[i] = var_reset;
        }
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * AVX2 + MKL OPTIMIZED MCMC
 *═══════════════════════════════════════════════════════════════════════════*/

#ifdef __AVX2__

static void mcmc_avx_mkl(MMPF_ROCKS *mmpf, double y_log_sq,
                         const MMPF_MCMC_Config *cfg)
{
    int n_steps = cfg->n_steps;
    double sigma = cfg->proposal_sigma;
    double var_reset = cfg->var_reset;

    /* Compute total random numbers needed */
    int max_particles = 0;
    for (int k = 0; k < MMPF_N_MODELS; k++)
    {
        if (mmpf->ext[k]->rbpf->n_particles > max_particles)
        {
            max_particles = mmpf->ext[k]->rbpf->n_particles;
        }
    }
    int total_rng = max_particles * n_steps;

    /* Verify scratch capacity (should be pre-allocated at init) */
    if (total_rng > mmpf->mcmc_scratch.capacity)
    {
        /* Emergency resize - should not happen in production */
        if (mmpf->mcmc_scratch.rng_gauss)
        {
            mkl_free(mmpf->mcmc_scratch.rng_gauss);
            mkl_free(mmpf->mcmc_scratch.rng_log_u);
        }
        int new_cap = (total_rng + 7) & ~7;
        mmpf->mcmc_scratch.rng_gauss = (double *)mkl_malloc(new_cap * sizeof(double), 64);
        mmpf->mcmc_scratch.rng_log_u = (double *)mkl_malloc(new_cap * sizeof(double), 64);
        mmpf->mcmc_scratch.capacity = new_cap;
    }

    double *rng_gauss = mmpf->mcmc_scratch.rng_gauss;
    double *rng_log_u = mmpf->mcmc_scratch.rng_log_u;

    /* Use VSL stream from MMPF (or create temp if not available) */
    VSLStreamStatePtr vsl_stream = (VSLStreamStatePtr)mmpf->mcmc_vsl_stream;
    int own_stream = 0;

    if (!vsl_stream)
    {
        /* Fallback: create temporary stream (suboptimal) */
        vslNewStream(&vsl_stream, VSL_BRNG_SFMT19937, 12345);
        own_stream = 1;
    }

    /* Generate ALL random numbers at once */
    vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, vsl_stream,
                  total_rng, rng_gauss, 0.0, sigma);
    vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, vsl_stream,
                 total_rng, rng_log_u, 0.0, 1.0);

    /* Pre-compute log(uniform) for acceptance check */
    vdLn(total_rng, rng_log_u, rng_log_u);

    /* Clean up temp stream if we created one */
    if (own_stream)
    {
        vslDeleteStream(&vsl_stream);
    }

    /* AVX constants */
    const __m256d v_min = _mm256_set1_pd(cfg->min_log_vol);
    const __m256d v_max = _mm256_set1_pd(cfg->max_log_vol);
    const __m256d v_var_reset = _mm256_set1_pd(var_reset);

    /* Process each model */
    for (int k = 0; k < MMPF_N_MODELS; k++)
    {
        RBPF_KSC *rbpf = mmpf->ext[k]->rbpf;
        int n = rbpf->n_particles;
        double *mu = (double *)rbpf->mu;
        double *var = (double *)rbpf->var;

        /* Process 4 particles at a time */
        int i = 0;
        for (; i + 4 <= n; i += 4)
        {
            __m256d h_curr = _mm256_loadu_pd(&mu[i]);
            __m256d ll_curr = mmpf_mcmc_loglik_avx(h_curr, y_log_sq);

            /* Run MH steps */
            for (int step = 0; step < n_steps; step++)
            {
                int rng_idx = step * max_particles + i;

                /* Load pre-generated randoms */
                __m256d noise = _mm256_loadu_pd(&rng_gauss[rng_idx]);
                __m256d log_u = _mm256_loadu_pd(&rng_log_u[rng_idx]);

                /* Propose: h_prop = h_curr + noise (noise already scaled by sigma) */
                __m256d h_prop = _mm256_add_pd(h_curr, noise);

                /* Clamp to bounds */
                h_prop = _mm256_max_pd(h_prop, v_min);
                h_prop = _mm256_min_pd(h_prop, v_max);

                /* Evaluate proposal likelihood */
                __m256d ll_prop = mmpf_mcmc_loglik_avx(h_prop, y_log_sq);

                /* log_alpha = ll_prop - ll_curr */
                __m256d log_alpha = _mm256_sub_pd(ll_prop, ll_curr);

                /* Accept if log_alpha >= log_u */
                __m256d accept_mask = _mm256_cmp_pd(log_alpha, log_u, _CMP_GE_OQ);

                /* Branchless update */
                h_curr = _mm256_blendv_pd(h_curr, h_prop, accept_mask);
                ll_curr = _mm256_blendv_pd(ll_curr, ll_prop, accept_mask);
            }

            /* Store results */
            _mm256_storeu_pd(&mu[i], h_curr);
            _mm256_storeu_pd(&var[i], v_var_reset);
        }

        /* Handle remainder (scalar) */
        for (; i < n; i++)
        {
            double h_curr = mu[i];
            double ll_curr = mmpf_mcmc_loglik(y_log_sq, h_curr);

            for (int step = 0; step < n_steps; step++)
            {
                int rng_idx = step * max_particles + i;
                double noise = rng_gauss[rng_idx];
                double log_u = rng_log_u[rng_idx];

                double h_prop = h_curr + noise;
                if (h_prop < cfg->min_log_vol)
                    h_prop = cfg->min_log_vol;
                if (h_prop > cfg->max_log_vol)
                    h_prop = cfg->max_log_vol;

                double ll_prop = mmpf_mcmc_loglik(y_log_sq, h_prop);
                double log_alpha = ll_prop - ll_curr;

                if (log_alpha >= log_u)
                {
                    h_curr = h_prop;
                    ll_curr = ll_prop;
                }
            }

            mu[i] = h_curr;
            var[i] = var_reset;
        }
    }
}

#endif /* __AVX2__ */

/*═══════════════════════════════════════════════════════════════════════════
 * PUBLIC API
 *═══════════════════════════════════════════════════════════════════════════*/

void mmpf_inject_shock_mcmc(MMPF_ROCKS *mmpf, double y_log_sq,
                            const MMPF_MCMC_Config *cfg)
{
    MMPF_MCMC_Config default_cfg;
    if (!cfg)
    {
        mmpf_mcmc_config_defaults(&default_cfg);
        cfg = &default_cfg;
    }

    /* Run MCMC (scalar version) */
    mcmc_scalar(mmpf, y_log_sq, cfg);

    /* Flatten transition matrix */
    if (cfg->flatten_transitions)
    {
        double uniform = 1.0 / MMPF_N_MODELS;
        for (int i = 0; i < MMPF_N_MODELS; i++)
        {
            for (int j = 0; j < MMPF_N_MODELS; j++)
            {
                /* Save original */
                mmpf->saved_transition[i][j] = mmpf->transition[i][j];
                /* Set uniform */
                mmpf->transition[i][j] = uniform;
            }
        }
    }

    mmpf->shock_active = 1;
}

void mmpf_inject_shock_mcmc_avx(MMPF_ROCKS *mmpf, double y_log_sq,
                                const MMPF_MCMC_Config *cfg)
{
    MMPF_MCMC_Config default_cfg;
    if (!cfg)
    {
        mmpf_mcmc_config_defaults(&default_cfg);
        cfg = &default_cfg;
    }

#ifdef __AVX2__
    mcmc_avx_mkl(mmpf, y_log_sq, cfg);
#else
    /* Fall back to scalar */
    mcmc_scalar(mmpf, y_log_sq, cfg);
#endif

    /* Flatten transition matrix */
    if (cfg->flatten_transitions)
    {
        double uniform = 1.0 / MMPF_N_MODELS;
        for (int i = 0; i < MMPF_N_MODELS; i++)
        {
            for (int j = 0; j < MMPF_N_MODELS; j++)
            {
                mmpf->saved_transition[i][j] = mmpf->transition[i][j];
                mmpf->transition[i][j] = uniform;
            }
        }
    }

    mmpf->shock_active = 1;
}

/*═══════════════════════════════════════════════════════════════════════════
 * DIAGNOSTICS
 *═══════════════════════════════════════════════════════════════════════════*/

void mmpf_mcmc_get_stats(const MMPF_ROCKS *mmpf, MMPF_MCMC_Stats *stats)
{
    if (!mmpf || !stats)
        return;

    /* Copy field by field to avoid type mismatch */
    stats->total_shocks = mmpf->mcmc_stats.total_shocks;
    stats->total_proposals = mmpf->mcmc_stats.total_proposals;
    stats->total_accepts = mmpf->mcmc_stats.total_accepts;
    stats->avg_acceptance = mmpf->mcmc_stats.avg_acceptance;
    stats->last_pre_mean = mmpf->mcmc_stats.last_pre_mean;
    stats->last_post_mean = mmpf->mcmc_stats.last_post_mean;
    stats->last_teleport = mmpf->mcmc_stats.last_teleport;
}

void mmpf_mcmc_reset_stats(MMPF_ROCKS *mmpf)
{
    if (!mmpf)
        return;

    mmpf->mcmc_stats.total_shocks = 0;
    mmpf->mcmc_stats.total_proposals = 0;
    mmpf->mcmc_stats.total_accepts = 0;
    mmpf->mcmc_stats.avg_acceptance = 0.0;
    mmpf->mcmc_stats.last_pre_mean = 0.0;
    mmpf->mcmc_stats.last_post_mean = 0.0;
    mmpf->mcmc_stats.last_teleport = 0.0;
}