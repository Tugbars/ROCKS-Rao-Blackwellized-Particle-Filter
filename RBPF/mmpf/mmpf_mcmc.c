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

/* MKL for batch RNG */
#ifdef USE_MKL
#include <mkl.h>
#include <mkl_vsl.h>
#define HAS_MKL 1
#else
#define HAS_MKL 0
#endif

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

#define LCHI_MEAN       -1.2704    /* E[log(χ²(1))] = ψ(0.5) + log(2) */
#define LCHI_VAR         4.9348    /* Var[log(χ²(1))] */
#define LCHI_STD         2.2215    /* sqrt(variance) */
#define LCHI_INV_VAR     0.2026    /* 1 / variance */
#define LCHI_LOG_NORM   -1.8379    /* -0.5 * log(2π × variance) */

/*═══════════════════════════════════════════════════════════════════════════
 * CONFIGURATION
 *═══════════════════════════════════════════════════════════════════════════*/

void mmpf_mcmc_config_defaults(MMPF_MCMC_Config *cfg) {
    cfg->n_steps = 5;
    cfg->proposal_sigma = 1.5;
    cfg->var_reset = 1.0;
    cfg->min_log_vol = -10.0;
    cfg->max_log_vol = 2.0;
    cfg->flatten_transitions = 1;
}

/*═══════════════════════════════════════════════════════════════════════════
 * SCRATCH BUFFER MANAGEMENT
 *═══════════════════════════════════════════════════════════════════════════*/

void mmpf_mcmc_scratch_alloc(MMPF_MCMC_Scratch *scratch,
                              int n_part, int n_models, int n_steps) {
    int total = n_part * n_models * n_steps;
    
    /* Round up to cache line (64 bytes = 8 doubles) */
    total = (total + 7) & ~7;
    
#if HAS_MKL
    scratch->rng_gauss = (double *)mkl_malloc(total * sizeof(double), 64);
    scratch->rng_log_u = (double *)mkl_malloc(total * sizeof(double), 64);
#else
    scratch->rng_gauss = (double *)aligned_alloc(64, total * sizeof(double));
    scratch->rng_log_u = (double *)aligned_alloc(64, total * sizeof(double));
#endif
    
    scratch->capacity = total;
}

void mmpf_mcmc_scratch_free(MMPF_MCMC_Scratch *scratch) {
    if (scratch->rng_gauss) {
#if HAS_MKL
        mkl_free(scratch->rng_gauss);
        mkl_free(scratch->rng_log_u);
#else
        free(scratch->rng_gauss);
        free(scratch->rng_log_u);
#endif
    }
    scratch->rng_gauss = NULL;
    scratch->rng_log_u = NULL;
    scratch->capacity = 0;
}

/*═══════════════════════════════════════════════════════════════════════════
 * SCALAR LIKELIHOOD
 *═══════════════════════════════════════════════════════════════════════════*/

double mmpf_mcmc_loglik(double y_log_sq, double h) {
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

__m256d mmpf_mcmc_loglik_avx(__m256d h_vec, double y_log_sq) {
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
 * SIMPLE RNG FOR NON-MKL BUILDS
 *═══════════════════════════════════════════════════════════════════════════*/

#if !HAS_MKL

/* xoroshiro128+ */
static uint64_t s_rng_state[2] = {0x853c49e6748fea9bULL, 0xda3e39cb94b95bdbULL};

static inline uint64_t rotl(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

static inline uint64_t xoro_next(void) {
    uint64_t s0 = s_rng_state[0];
    uint64_t s1 = s_rng_state[1];
    uint64_t result = s0 + s1;
    
    s1 ^= s0;
    s_rng_state[0] = rotl(s0, 24) ^ s1 ^ (s1 << 16);
    s_rng_state[1] = rotl(s1, 37);
    
    return result;
}

static inline double rand_uniform(void) {
    return (xoro_next() >> 11) * (1.0 / 9007199254740992.0);
}

/* Box-Muller for Gaussian */
static inline double rand_gaussian(void) {
    double u1 = rand_uniform();
    double u2 = rand_uniform();
    if (u1 < 1e-10) u1 = 1e-10;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

#endif /* !HAS_MKL */

/*═══════════════════════════════════════════════════════════════════════════
 * SCALAR MCMC IMPLEMENTATION
 *═══════════════════════════════════════════════════════════════════════════*/

static void mcmc_scalar(MMPF_ROCKS *mmpf, double y_log_sq,
                        const MMPF_MCMC_Config *cfg) {
    int n_steps = cfg->n_steps;
    double sigma = cfg->proposal_sigma;
    double var_reset = cfg->var_reset;
    
    /* For each model */
    for (int k = 0; k < MMPF_N_MODELS; k++) {
        RBPF_KSC *rbpf = mmpf->ext[k]->rbpf;
        int n = rbpf->n_particles;
        double *mu = (double *)rbpf->mu;
        double *var = (double *)rbpf->var;
        
        /* MCMC for each particle */
        for (int i = 0; i < n; i++) {
            double h_curr = mu[i];
            double ll_curr = mmpf_mcmc_loglik(y_log_sq, h_curr);
            
            /* Run MH steps */
            for (int step = 0; step < n_steps; step++) {
#if HAS_MKL
                /* MKL path uses pre-generated randoms (handled in AVX version) */
                double noise = 0.0;
                double u = 0.5;
#else
                double noise = rand_gaussian() * sigma;
                double u = rand_uniform();
#endif
                
                /* Propose */
                double h_prop = h_curr + noise;
                
                /* Clamp to bounds */
                if (h_prop < cfg->min_log_vol) h_prop = cfg->min_log_vol;
                if (h_prop > cfg->max_log_vol) h_prop = cfg->max_log_vol;
                
                /* Evaluate */
                double ll_prop = mmpf_mcmc_loglik(y_log_sq, h_prop);
                
                /* Accept/reject */
                double log_alpha = ll_prop - ll_curr;
                if (log_alpha > 0.0 || log(u + 1e-10) < log_alpha) {
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

#if defined(__AVX2__) && HAS_MKL

static void mcmc_avx_mkl(MMPF_ROCKS *mmpf, double y_log_sq,
                          const MMPF_MCMC_Config *cfg) {
    int n_steps = cfg->n_steps;
    double sigma = cfg->proposal_sigma;
    double var_reset = cfg->var_reset;
    
    /* Thread-local scratch buffers */
    static __thread double *rng_gauss = NULL;
    static __thread double *rng_uniform = NULL;
    static __thread int rng_capacity = 0;
    static __thread VSLStreamStatePtr vsl_stream = NULL;
    
    /* Compute total random numbers needed */
    int max_particles = 0;
    for (int k = 0; k < MMPF_N_MODELS; k++) {
        if (mmpf->ext[k]->rbpf->n_particles > max_particles) {
            max_particles = mmpf->ext[k]->rbpf->n_particles;
        }
    }
    int total_rng = max_particles * n_steps;
    
    /* Allocate/resize buffers if needed */
    if (total_rng > rng_capacity) {
        if (rng_gauss) mkl_free(rng_gauss);
        if (rng_uniform) mkl_free(rng_uniform);
        
        rng_capacity = (total_rng + 255) & ~255;  /* Round up to 256 */
        rng_gauss = (double *)mkl_malloc(rng_capacity * sizeof(double), 64);
        rng_uniform = (double *)mkl_malloc(rng_capacity * sizeof(double), 64);
    }
    
    /* Create VSL stream if needed */
    if (!vsl_stream) {
        vslNewStream(&vsl_stream, VSL_BRNG_SFMT19937, 12345);
    }
    
    /* Generate ALL random numbers at once */
    vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, vsl_stream, 
                  total_rng, rng_gauss, 0.0, sigma);
    vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, vsl_stream,
                 total_rng, rng_uniform, 0.0, 1.0);
    
    /* Pre-compute log(uniform) for acceptance check */
    vdLn(total_rng, rng_uniform, rng_uniform);
    
    /* AVX constants */
    const __m256d v_min = _mm256_set1_pd(cfg->min_log_vol);
    const __m256d v_max = _mm256_set1_pd(cfg->max_log_vol);
    const __m256d v_var_reset = _mm256_set1_pd(var_reset);
    
    /* Process each model */
    for (int k = 0; k < MMPF_N_MODELS; k++) {
        RBPF_KSC *rbpf = mmpf->ext[k]->rbpf;
        int n = rbpf->n_particles;
        double *mu = (double *)rbpf->mu;
        double *var = (double *)rbpf->var;
        
        /* Process 4 particles at a time */
        int i = 0;
        for (; i + 4 <= n; i += 4) {
            __m256d h_curr = _mm256_loadu_pd(&mu[i]);
            __m256d ll_curr = mmpf_mcmc_loglik_avx(h_curr, y_log_sq);
            
            /* Run MH steps */
            for (int step = 0; step < n_steps; step++) {
                int rng_idx = step * max_particles + i;
                
                /* Load pre-generated randoms */
                __m256d noise = _mm256_loadu_pd(&rng_gauss[rng_idx]);
                __m256d log_u = _mm256_loadu_pd(&rng_uniform[rng_idx]);
                
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
        for (; i < n; i++) {
            double h_curr = mu[i];
            double ll_curr = mmpf_mcmc_loglik(y_log_sq, h_curr);
            
            for (int step = 0; step < n_steps; step++) {
                int rng_idx = step * max_particles + i;
                double noise = rng_gauss[rng_idx];
                double log_u = rng_uniform[rng_idx];
                
                double h_prop = h_curr + noise;
                if (h_prop < cfg->min_log_vol) h_prop = cfg->min_log_vol;
                if (h_prop > cfg->max_log_vol) h_prop = cfg->max_log_vol;
                
                double ll_prop = mmpf_mcmc_loglik(y_log_sq, h_prop);
                double log_alpha = ll_prop - ll_curr;
                
                if (log_alpha >= log_u) {
                    h_curr = h_prop;
                    ll_curr = ll_prop;
                }
            }
            
            mu[i] = h_curr;
            var[i] = var_reset;
        }
    }
}

#endif /* __AVX2__ && HAS_MKL */

/*═══════════════════════════════════════════════════════════════════════════
 * PUBLIC API
 *═══════════════════════════════════════════════════════════════════════════*/

void mmpf_inject_shock_mcmc(MMPF_ROCKS *mmpf, double y_log_sq,
                            const MMPF_MCMC_Config *cfg) {
    MMPF_MCMC_Config default_cfg;
    if (!cfg) {
        mmpf_mcmc_config_defaults(&default_cfg);
        cfg = &default_cfg;
    }
    
    /* Run MCMC (scalar version) */
    mcmc_scalar(mmpf, y_log_sq, cfg);
    
    /* Flatten transition matrix */
    if (cfg->flatten_transitions) {
        double uniform = 1.0 / MMPF_N_MODELS;
        for (int i = 0; i < MMPF_N_MODELS; i++) {
            for (int j = 0; j < MMPF_N_MODELS; j++) {
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
                                 const MMPF_MCMC_Config *cfg) {
    MMPF_MCMC_Config default_cfg;
    if (!cfg) {
        mmpf_mcmc_config_defaults(&default_cfg);
        cfg = &default_cfg;
    }
    
#if defined(__AVX2__) && HAS_MKL
    mcmc_avx_mkl(mmpf, y_log_sq, cfg);
#else
    /* Fall back to scalar */
    mcmc_scalar(mmpf, y_log_sq, cfg);
#endif
    
    /* Flatten transition matrix */
    if (cfg->flatten_transitions) {
        double uniform = 1.0 / MMPF_N_MODELS;
        for (int i = 0; i < MMPF_N_MODELS; i++) {
            for (int j = 0; j < MMPF_N_MODELS; j++) {
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

void mmpf_mcmc_get_stats(const MMPF_ROCKS *mmpf, MMPF_MCMC_Stats *stats) {
    /* TODO: Implement statistics tracking */
    memset(stats, 0, sizeof(MMPF_MCMC_Stats));
    (void)mmpf;
}

void mmpf_mcmc_reset_stats(MMPF_ROCKS *mmpf) {
    /* TODO: Implement statistics reset */
    (void)mmpf;
}
