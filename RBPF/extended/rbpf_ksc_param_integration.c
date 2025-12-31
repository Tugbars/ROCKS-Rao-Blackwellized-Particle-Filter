/**
 * @file rbpf_ksc_param_integration.c
 * @brief Core: Lifecycle + Step Function + Internal Helpers
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * This file contains:
 *   - rbpf_ext_create(), rbpf_ext_destroy(), rbpf_ext_init()
 *   - rbpf_ext_step() - the hot path
 *   - Internal helpers (SIMD, sync, lag buffers, transition counts)
 *
 * MODIFIED: Soft-voting transition learning + Crisis mode integration
 *
 * Related files:
 *   - rbpf_ext_config.c             Configuration functions
 *   - rbpf_ext_diagnostics.c        Getters and print functions
 *   - rbpf_ext_hawkes.c             Hawkes + Robust OCSN + Presets
 *   - rbpf_ext_smoothed_storvik.c   PARIS fixed-lag smoother
 *   - rbpf_ext_crisis.c             Crisis mode API (NEW)
 * ═══════════════════════════════════════════════════════════════════════════
 */

#include "rbpf_ksc_param_integration.h"
#include "rbpf_ext_crisis.h" /* NEW: Crisis mode API */
#include "rbpf_fixed_lag_smoother.h"
#include "rbpf_sprt.h"
#include "rbpf_dirichlet_transition.h"
#include "rbpf_kl_tempering.h"
#include "rbpf_apf_kick.h"
#include "hawkes_integrator.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <inttypes.h>

/* Forward declarations */
extern void rbpf_rebuild_trans_lut_from_dirichlet(RBPF_KSC *rbpf);

/*═══════════════════════════════════════════════════════════════════════════
 * SIMD AND CACHE CONFIGURATION
 *═══════════════════════════════════════════════════════════════════════════*/

#define CACHE_LINE 64

#if defined(__AVX512F__) && !defined(_MSC_VER)
#define USE_AVX512 1
#include <immintrin.h>
#elif defined(__AVX2__)
#define USE_AVX2 1
#include <immintrin.h>
#endif

#ifdef PARAM_LEARN_USE_MKL
#include <mkl.h>
#include <mkl_vml.h>
#endif

/* Compiler hints */
#if defined(__GNUC__) || defined(__clang__)
#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
#define RESTRICT __restrict__
#define FORCE_INLINE __attribute__((always_inline)) inline
#define PREFETCH_R(p) __builtin_prefetch((p), 0, 3)
#define PREFETCH_W(p) __builtin_prefetch((p), 1, 3)
#elif defined(_MSC_VER)
#define LIKELY(x) (x)
#define UNLIKELY(x) (x)
#define RESTRICT __restrict
#define FORCE_INLINE __forceinline
#define PREFETCH_R(p) _mm_prefetch((const char *)(p), _MM_HINT_T0)
#define PREFETCH_W(p) _mm_prefetch((const char *)(p), _MM_HINT_T0)
#else
#define LIKELY(x) (x)
#define UNLIKELY(x) (x)
#define RESTRICT
#define FORCE_INLINE inline
#define PREFETCH_R(p)
#define PREFETCH_W(p)
#endif

/*═══════════════════════════════════════════════════════════════════════════
 * SIMD HELPERS
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * SIMD double→float conversion (for Storvik→RBPF sync)
 */
static FORCE_INLINE void convert_double_to_float_aligned(
    const double *RESTRICT src,
    float *RESTRICT dst,
    int n)
{
#if defined(USE_AVX512)
    int i = 0;
    for (; i + 8 <= n; i += 8)
    {
        __m512d vd = _mm512_load_pd(src + i);
        __m256 vf = _mm512_cvtpd_ps(vd);
        _mm256_store_ps(dst + i, vf);
    }
    for (; i < n; i++)
    {
        dst[i] = (float)src[i];
    }
#elif defined(USE_AVX2)
    int i = 0;
    for (; i + 4 <= n; i += 4)
    {
        __m256d vd = _mm256_load_pd(src + i);
        __m128 vf = _mm256_cvtpd_ps(vd);
        _mm_store_ps(dst + i, vf);
    }
    for (; i < n; i++)
    {
        dst[i] = (float)src[i];
    }
#else
    for (int i = 0; i < n; i++)
    {
        dst[i] = (float)src[i];
    }
#endif
}

static FORCE_INLINE void memory_store_fence(void)
{
#if defined(USE_AVX512) || defined(USE_AVX2) || defined(__SSE2__)
    _mm_sfence();
#elif defined(_MSC_VER)
    _WriteBarrier();
    MemoryBarrier();
#else
    __sync_synchronize();
#endif
}

/*═══════════════════════════════════════════════════════════════════════════
 * INTERNAL: PARTICLE INFO EXTRACTION
 *═══════════════════════════════════════════════════════════════════════════*/

static void extract_particle_info_optimized(RBPF_Extended *ext, int resampled)
{
    RBPF_KSC *rbpf = ext->rbpf;
    const int n = rbpf->n_particles;

    const rbpf_real_t *RESTRICT w_norm = rbpf->w_norm;
    const rbpf_real_t *RESTRICT mu = rbpf->mu;
    const int *RESTRICT regime = rbpf->regime;
    const int *RESTRICT indices = rbpf->indices;

    ParticleInfo *RESTRICT info = ext->particle_info;
    rbpf_real_t *RESTRICT ell_lag = ext->ell_lag_buffer;
    int *RESTRICT prev_regime = ext->prev_regime;

    PREFETCH_W(info);
    PREFETCH_W(info + 8);

    for (int i = 0; i < n; i++)
    {
        ParticleInfo *p = &info[i];
        p->regime = regime[i];
        p->ell = mu[i];
        p->weight = w_norm[i];

        int parent_idx = resampled ? indices[i] : i;
        p->ell_lag = ell_lag[parent_idx];
        p->prev_regime = prev_regime[parent_idx];
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * INTERNAL: STORVIK→RBPF SYNC
 *═══════════════════════════════════════════════════════════════════════════*/

static void sync_storvik_to_rbpf_optimized(RBPF_Extended *ext)
{
    if (!ext->storvik_initialized)
        return;
    if (ext->param_mode != RBPF_PARAM_STORVIK)
        return;

    RBPF_KSC *rbpf = ext->rbpf;
    StorvikSoA *soa = param_learn_get_active_soa(&ext->storvik);
    const int total = rbpf->n_particles * rbpf->n_regimes;

    convert_double_to_float_aligned(soa->mu_cached, rbpf->particle_mu_vol, total);
    convert_double_to_float_aligned(soa->sigma_cached, rbpf->particle_sigma_vol, total);

    memory_store_fence();
}

/*═══════════════════════════════════════════════════════════════════════════
 * INTERNAL: LAG BUFFER UPDATE
 *═══════════════════════════════════════════════════════════════════════════*/

static FORCE_INLINE void update_lag_buffers(RBPF_Extended *ext)
{
    RBPF_KSC *rbpf = ext->rbpf;
    const int n = rbpf->n_particles;

    rbpf_real_t *RESTRICT ell_lag = ext->ell_lag_buffer;
    int *RESTRICT prev_regime = ext->prev_regime;
    const rbpf_real_t *RESTRICT mu = rbpf->mu;
    const int *RESTRICT regime = rbpf->regime;

    for (int i = 0; i < n; i += 8)
    {
        PREFETCH_R(mu + i + 16);
        PREFETCH_R(regime + i + 16);

        int end = (i + 8 < n) ? i + 8 : n;
        for (int j = i; j < end; j++)
        {
            ell_lag[j] = mu[j];
            prev_regime[j] = regime[j];
        }
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * LIFECYCLE
 *═══════════════════════════════════════════════════════════════════════════*/

RBPF_Extended *rbpf_ext_create(int n_particles, int n_regimes, RBPF_ParamMode mode)
{
    RBPF_Extended *ext = (RBPF_Extended *)calloc(1, sizeof(RBPF_Extended));
    if (!ext)
        return NULL;

    ext->param_mode = mode;

    /* Create core RBPF-KSC */
    ext->rbpf = rbpf_ksc_create(n_particles, n_regimes);
    if (!ext->rbpf)
    {
        free(ext);
        return NULL;
    }

    /* Allocate workspace (cache-line aligned) */
#if defined(_MSC_VER)
    ext->particle_info = (ParticleInfo *)_aligned_malloc(
        n_particles * sizeof(ParticleInfo), CACHE_LINE);
    ext->prev_regime = (int *)_aligned_malloc(
        n_particles * sizeof(int), CACHE_LINE);
    ext->ell_lag_buffer = (rbpf_real_t *)_aligned_malloc(
        n_particles * sizeof(rbpf_real_t), CACHE_LINE);
#else
    posix_memalign((void **)&ext->particle_info, CACHE_LINE,
                   n_particles * sizeof(ParticleInfo));
    posix_memalign((void **)&ext->prev_regime, CACHE_LINE,
                   n_particles * sizeof(int));
    posix_memalign((void **)&ext->ell_lag_buffer, CACHE_LINE,
                   n_particles * sizeof(rbpf_real_t));
#endif

    if (!ext->particle_info || !ext->prev_regime || !ext->ell_lag_buffer)
    {
        rbpf_ext_destroy(ext);
        return NULL;
    }

    /* Initialize Storvik if needed */
    if (mode == RBPF_PARAM_STORVIK || mode == RBPF_PARAM_HYBRID)
    {
        ParamLearnConfig cfg = param_learn_config_defaults();
        cfg.sample_on_regime_change = true;
        cfg.sample_on_structural_break = true;
        cfg.sample_after_resampling = true;

        if (param_learn_init(&ext->storvik, &cfg, n_particles, n_regimes) != 0)
        {
            rbpf_ext_destroy(ext);
            return NULL;
        }
        ext->storvik_initialized = 1;
    }

    /*═══════════════════════════════════════════════════════════════════════
     * TRANSITION LEARNING (DEPRECATED) + CRISIS MODE
     *═══════════════════════════════════════════════════════════════════════*/

    /* Transition learning disabled - PGAS owns Π */
    ext->trans_learn_enabled = 0;
    ext->trans_update_interval = 100;
    ext->trans_ticks_since_update = 0;
    ext->trans_forgetting = 0.995;
    ext->trans_prior_diag = 50.0;
    ext->trans_prior_off = 1.0;

    /* Crisis mode defaults */
    ext->trans_mode = TRANS_MODE_NORMAL;
    ext->ticks_in_crisis_mode = 0;
    ext->emission_lambda_override_active = 0;
    ext->emission_lambda_override = 0.95f;
    ext->crisis_entries = 0;
    ext->crisis_exits = 0;

    /* Per-particle parameter mode */
    ext->rbpf->use_learned_params = 1;

    /* Hawkes Integrator for APF Kick */
    HawkesIntegratorConfig hawkes_cfg = hawkes_integrator_config_defaults();
    hawkes_integrator_init(&ext->hawkes_integrator, &hawkes_cfg);
    ext->apf_kick_enabled = 0;
    ext->apf_surprise_threshold = 0.5f;

    memset(ext->base_trans_matrix, 0, sizeof(ext->base_trans_matrix));

    /* Robust OCSN defaults (disabled) */
    ext->robust_ocsn.enabled = 0;
    for (int r = 0; r < RBPF_MAX_REGIMES; r++)
    {
        ext->robust_ocsn.regime[r].prob = RBPF_REAL(0.01) + r * RBPF_REAL(0.005);
        ext->robust_ocsn.regime[r].variance = RBPF_REAL(18.0) + r * RBPF_REAL(4.0);
    }

    /* Smoothed Storvik defaults (disabled) */
    ext->smoothed_storvik_enabled = 0;
    ext->smoothed_storvik_lag = 50;
    ext->smoother = NULL;
    ext->cooldown_remaining = 0;
    ext->min_buffer_for_flush = 10;
    ext->ess_collapse_threshold = (float)n_particles / 20.0f;
    ext->flush_count = 0;
    ext->reset_count = 0;

    /* KL Tempering (disabled by default, but allocated) */
    ext->kl_state = (RBPF_KL_State *)calloc(1, sizeof(RBPF_KL_State));
    if (ext->kl_state)
    {
        rbpf_kl_init(ext->kl_state, n_particles);
    }
    ext->kl_tempering_enabled = 0;
    ext->last_resampled = 0;

    /* Misc */
    ext->current_preset = RBPF_PRESET_CUSTOM;
    ext->tick_count = 0;
    ext->last_hawkes_intensity = RBPF_REAL(0.0);
    ext->last_outlier_fraction = RBPF_REAL(0.0);
    ext->structural_break_signaled = 0;

    /* Policy engine state */
    ext->prev_sprt_regime = 0;

    rbpf_adaptive_forgetting_init(&ext->adaptive_forgetting);

    return ext;
}

void rbpf_ext_destroy(RBPF_Extended *ext)
{
    if (!ext)
        return;

    if (ext->rbpf)
        rbpf_ksc_destroy(ext->rbpf);
    if (ext->storvik_initialized)
        param_learn_free(&ext->storvik);
    if (ext->smoother)
        fls_destroy(ext->smoother);

    /* KL Tempering state */
    free(ext->kl_state);

    /* Hawkes Integrator */
    hawkes_integrator_free(&ext->hawkes_integrator);

#if defined(_MSC_VER)
    _aligned_free(ext->particle_info);
    _aligned_free(ext->prev_regime);
    _aligned_free(ext->ell_lag_buffer);
#else
    free(ext->particle_info);
    free(ext->prev_regime);
    free(ext->ell_lag_buffer);
#endif

    free(ext);
}

void rbpf_ext_init(RBPF_Extended *ext, rbpf_real_t mu0, rbpf_real_t var0)
{
    if (!ext)
        return;

    rbpf_ksc_init(ext->rbpf, mu0, var0);

    const int n = ext->rbpf->n_particles;
    for (int i = 0; i < n; i++)
    {
        ext->ell_lag_buffer[i] = mu0;
        ext->prev_regime[i] = ext->rbpf->regime[i];
    }

    if (ext->storvik_initialized)
    {
        const int nr = ext->rbpf->n_regimes;
        for (int r = 0; r < nr; r++)
        {
            const RBPF_RegimeParams *p = &ext->rbpf->params[r];
            rbpf_real_t phi = RBPF_REAL(1.0) - p->theta;
            param_learn_set_prior(&ext->storvik, r, p->mu_vol, phi, p->sigma_vol);
        }
        param_learn_broadcast_priors(&ext->storvik);
        sync_storvik_to_rbpf_optimized(ext);
    }

    /* Reset smoother if enabled */
    if (ext->smoother)
    {
        fls_reset(ext->smoother);

        if (ext->smoothed_storvik_enabled)
        {
            const int nr = ext->rbpf->n_regimes;
            double trans[RBPF_MAX_REGIMES * RBPF_MAX_REGIMES];
            double mu_vol[RBPF_MAX_REGIMES];
            double sigma_vol[RBPF_MAX_REGIMES];
            double phi = 1.0 - ext->rbpf->params[0].theta;

            for (int i = 0; i < nr; i++)
            {
                mu_vol[i] = ext->rbpf->params[i].mu_vol;
                sigma_vol[i] = ext->rbpf->params[i].sigma_vol;
                for (int j = 0; j < nr; j++)
                {
                    trans[i * nr + j] = ext->base_trans_matrix[i * nr + j];
                }
            }
            fls_set_model(ext->smoother, trans, mu_vol, sigma_vol, phi);
        }
    }
    ext->cooldown_remaining = 0;

    /* Reset KL tempering state */
    if (ext->kl_state)
    {
        rbpf_kl_reset(ext->kl_state);
    }
    ext->last_resampled = 0;

    /* Reset Hawkes integrator */
    hawkes_integrator_reset(&ext->hawkes_integrator);

    /* Initialize policy engine state */
    ext->prev_sprt_regime = 0;
    ext->structural_break_signaled = 0;

    /*═══════════════════════════════════════════════════════════════════════
     * RESET CRISIS MODE STATE
     *═══════════════════════════════════════════════════════════════════════*/
    ext->trans_mode = TRANS_MODE_NORMAL;
    ext->ticks_in_crisis_mode = 0;
    ext->emission_lambda_override_active = 0;
}

/*═══════════════════════════════════════════════════════════════════════════
 * MAIN STEP FUNCTION
 *═══════════════════════════════════════════════════════════════════════════*/

void rbpf_ext_step(RBPF_Extended *ext, rbpf_real_t obs, RBPF_KSC_Output *output)
{
    if (!ext || !ext->rbpf)
        return;

    RBPF_KSC *rbpf = ext->rbpf;
    const int n = rbpf->n_particles;

    ext->tick_count++;

    /*═══════════════════════════════════════════════════════════════════════
     * PHASE 0: STRUCTURAL BREAK SIGNAL
     *═══════════════════════════════════════════════════════════════════════*/
    if (ext->structural_break_signaled && ext->storvik_initialized)
    {
        param_learn_signal_structural_break(&ext->storvik);
    }

    /*═══════════════════════════════════════════════════════════════════════
     * PHASE 1: TRANSFORM OBSERVATION
     *═══════════════════════════════════════════════════════════════════════*/
    rbpf_real_t y;
    if (rbpf_fabs(obs) < RBPF_REAL(1e-10))
    {
        y = RBPF_REAL(-23.0);
    }
    else
    {
        y = rbpf_log(obs * obs);
    }

    /*═══════════════════════════════════════════════════════════════════════
     * PHASE 2: HAWKES UPDATE + REGIME TRANSITION (APF Kick when crisis)
     *═══════════════════════════════════════════════════════════════════════*/
    int apf_active = 0;

    if (ext->apf_kick_enabled)
    {
        HawkesIntegratorResult hawkes_result = hawkes_integrator_update(
            &ext->hawkes_integrator,
            (float)ext->tick_count,
            (float)obs);

        apf_active = (hawkes_result.surprise_sigma > ext->apf_surprise_threshold);
        ext->last_hawkes_intensity = (rbpf_real_t)hawkes_result.integrated_intensity;
    }

    if (apf_active)
    {
        rbpf_ksc_transition_apf(rbpf, y);
    }
    else
    {
        rbpf_ksc_transition(rbpf);
    }

    /*═══════════════════════════════════════════════════════════════════════
     * PHASE 3: KALMAN PREDICT + UPDATE
     *═══════════════════════════════════════════════════════════════════════*/
    rbpf_ksc_predict(rbpf);

    rbpf_real_t marginal_lik;
    if (ext->robust_ocsn.enabled)
    {
        marginal_lik = rbpf_ksc_update_robust(rbpf, y, &ext->robust_ocsn);
    }
    else
    {
        marginal_lik = rbpf_ksc_update(rbpf, y);
    }

    /*═══════════════════════════════════════════════════════════════════════
     * PHASE 4: KL TEMPERING
     *═══════════════════════════════════════════════════════════════════════*/
    if (ext->kl_tempering_enabled && ext->kl_state)
    {
#ifndef RBPF_USE_DOUBLE
        RBPF_KL_Result kl_result = rbpf_kl_step(
            ext->kl_state,
            rbpf->log_weight,
            rbpf->log_lik_increment,
            n,
            ext->last_resampled);

        if (kl_result.zombie_detected)
        {
            ext->structural_break_signaled = 1;
        }
#endif
    }

    /*═══════════════════════════════════════════════════════════════════════
     * PHASE 5: COMPUTE OUTPUTS + RESAMPLE
     *═══════════════════════════════════════════════════════════════════════*/
    rbpf_ksc_compute_outputs(rbpf, marginal_lik, output);
    output->resampled = rbpf_ksc_resample(rbpf);
    ext->last_resampled = output->resampled;

    /*═══════════════════════════════════════════════════════════════════════
     * PHASE 6: OUTLIER FRACTION
     *═══════════════════════════════════════════════════════════════════════*/
    if (ext->robust_ocsn.enabled)
    {
        ext->last_outlier_fraction = rbpf_ksc_compute_outlier_fraction(
            rbpf, y, &ext->robust_ocsn);
    }
    else
    {
        ext->last_outlier_fraction = RBPF_REAL(0.0);
    }
    output->outlier_fraction = ext->last_outlier_fraction;

    /*═══════════════════════════════════════════════════════════════════════
     * PHASE 7: ADAPTIVE FORGETTING (with crisis override)
     *═══════════════════════════════════════════════════════════════════════*/
    if (ext->adaptive_forgetting.enabled)
    {
        int regime_counts[RBPF_MAX_REGIMES] = {0};
        for (int i = 0; i < n; i++)
        {
            int r = rbpf->regime[i];
            if (r >= 0 && r < rbpf->n_regimes)
            {
                regime_counts[r]++;
            }
        }
        int dominant_regime = 0;
        int max_count = 0;
        for (int r = 0; r < rbpf->n_regimes; r++)
        {
            if (regime_counts[r] > max_count)
            {
                max_count = regime_counts[r];
                dominant_regime = r;
            }
        }
        rbpf_adaptive_forgetting_update(ext, marginal_lik, dominant_regime);

        /* Crisis override takes precedence */
        if (ext->emission_lambda_override_active)
        {
            ext->adaptive_forgetting.lambda_current = ext->emission_lambda_override;
        }

        /* Push to Storvik */
        if (ext->storvik_initialized)
        {
            param_learn_set_forgetting(&ext->storvik, 1,
                                       ext->adaptive_forgetting.lambda_current);
        }
    }

    /*═══════════════════════════════════════════════════════════════════════
     * PHASE 8: POLICY ENGINE (Regime Change Detection)
     *═══════════════════════════════════════════════════════════════════════*/
    {
        int p2_tail_event = rbpf_ext_structural_break_detected(ext);
        int sprt_flip = (output->smoothed_regime != ext->prev_sprt_regime);

        if (p2_tail_event)
        {
            output->regime_changed = 1;
            output->change_type = 1;

            sprt_multi_force_regime(&rbpf->sprt, output->dominant_regime);
            ext->structural_break_signaled = 1;
            ext->prev_sprt_regime = output->dominant_regime;
        }
        else if (sprt_flip)
        {
            output->regime_changed = 1;
            output->change_type = 2;

            if (rbpf->trans_prior_enabled)
            {
                dirichlet_transition_update(&rbpf->trans_prior,
                                            ext->prev_sprt_regime,
                                            output->smoothed_regime);
                rbpf_rebuild_trans_lut_from_dirichlet(rbpf);
            }

            ext->prev_sprt_regime = output->smoothed_regime;
        }
        else
        {
            output->regime_changed = 0;
            output->change_type = 0;
        }
    }

    /*═══════════════════════════════════════════════════════════════════════
     * PHASE 9: STORVIK PARAMETER LEARNING
     *═══════════════════════════════════════════════════════════════════════*/
    if (ext->storvik_initialized)
    {
        if (output->resampled)
        {
            param_learn_apply_resampling(&ext->storvik, rbpf->indices, n);
        }

        extract_particle_info_optimized(ext, output->resampled);
        param_learn_update(&ext->storvik, ext->particle_info, n);

        if (ext->structural_break_signaled)
        {
            param_learn_reset_to_priors(&ext->storvik);
        }
    }

    /* Clear structural break flag */
    ext->structural_break_signaled = 0;

    /*═══════════════════════════════════════════════════════════════════════
     * PHASE 10: CRISIS MODE TICK COUNTER
     *
     * Just increment ticks_in_crisis_mode for PGAS veto timing.
     * PGAS handles Π learning.
     *═══════════════════════════════════════════════════════════════════════*/
    if (ext->trans_mode == TRANS_MODE_CRISIS)
    {
        ext->ticks_in_crisis_mode++;
    }

    /*═══════════════════════════════════════════════════════════════════════
     * PHASE 11: LAG BUFFERS & SYNC
     *═══════════════════════════════════════════════════════════════════════*/
    update_lag_buffers(ext);
    sync_storvik_to_rbpf_optimized(ext);

    /*═══════════════════════════════════════════════════════════════════════
     * PHASE 12: POPULATE OUTPUT
     *═══════════════════════════════════════════════════════════════════════*/
    for (int r = 0; r < rbpf->n_regimes; r++)
    {
        rbpf_ext_get_learned_params(ext, r,
                                    &output->learned_mu_vol[r],
                                    &output->learned_sigma_vol[r]);
    }
}