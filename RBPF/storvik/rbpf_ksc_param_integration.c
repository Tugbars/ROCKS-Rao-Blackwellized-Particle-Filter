/**
 * @file rbpf_ksc_param_integration.c
 * @brief Core: Lifecycle, Step Function, Basic Configuration
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * This file contains:
 *   - rbpf_ext_create(), rbpf_ext_destroy(), rbpf_ext_init()
 *   - rbpf_ext_step() - the hot path
 *   - Basic configuration functions
 *   - Transition learning
 *   - Parameter access
 *   - Diagnostics
 *
 * Related files:
 *   - rbpf_ext_hawkes.c         Hawkes + Robust OCSN + Presets
 *   - rbpf_ext_smoothed_storvik.c   PARIS fixed-lag smoother
 * ═══════════════════════════════════════════════════════════════════════════
 */

#include "rbpf_ksc_param_integration.h"
#include "rbpf_fixed_lag_smoother.h"
#include "rbpf_sprt.h"
#include "rbpf_dirichlet_transition.h"
#include "rbpf_kl_tempering.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <inttypes.h>

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
 * INTERNAL: TRANSITION COUNT UPDATE
 *═══════════════════════════════════════════════════════════════════════════*/

static void update_transition_counts_optimized(RBPF_Extended *ext)
{
    if (!ext->trans_learn_enabled)
        return;

    RBPF_KSC *rbpf = ext->rbpf;
    const int n = rbpf->n_particles;
    const int nr = rbpf->n_regimes;
    const double forget = ext->trans_forgetting;

    /* Decay old counts */
#if defined(USE_AVX512)
    __m512d vforget = _mm512_set1_pd(forget);
    for (int i = 0; i < nr; i++)
    {
        int j = 0;
        for (; j + 8 <= nr; j += 8)
        {
            __m512d counts = _mm512_loadu_pd(&ext->trans_counts[i][j]);
            counts = _mm512_mul_pd(counts, vforget);
            _mm512_storeu_pd(&ext->trans_counts[i][j], counts);
        }
        for (; j < nr; j++)
        {
            ext->trans_counts[i][j] *= forget;
        }
    }
#else
    for (int i = 0; i < nr; i++)
    {
        for (int j = 0; j < nr; j++)
        {
            ext->trans_counts[i][j] *= forget;
        }
    }
#endif

    /* Accumulate new counts */
    int local_counts[RBPF_MAX_REGIMES][RBPF_MAX_REGIMES] = {{0}};
    const int *RESTRICT regime = rbpf->regime;
    const int *RESTRICT prev = ext->prev_regime;

    for (int k = 0; k < n; k++)
    {
        int r_prev = prev[k];
        int r_curr = regime[k];
        if (r_prev >= 0 && r_prev < nr && r_curr >= 0 && r_curr < nr)
        {
            local_counts[r_prev][r_curr]++;
        }
    }

    const double inv_n = 1.0 / n;
    for (int i = 0; i < nr; i++)
    {
        for (int j = 0; j < nr; j++)
        {
            ext->trans_counts[i][j] += local_counts[i][j] * inv_n;
        }
    }
}

static void rebuild_transition_lut(RBPF_Extended *ext)
{
    if (!ext->trans_learn_enabled)
        return;

    RBPF_KSC *rbpf = ext->rbpf;
    const int nr = rbpf->n_regimes;
    rbpf_real_t flat_matrix[RBPF_MAX_REGIMES * RBPF_MAX_REGIMES];

    for (int i = 0; i < nr; i++)
    {
        double row_sum = 0.0;
        for (int j = 0; j < nr; j++)
        {
            double prior = (i == j) ? ext->trans_prior_diag : ext->trans_prior_off;
            row_sum += ext->trans_counts[i][j] + prior;
        }
        for (int j = 0; j < nr; j++)
        {
            double prior = (i == j) ? ext->trans_prior_diag : ext->trans_prior_off;
            double count = ext->trans_counts[i][j] + prior;
            flat_matrix[i * nr + j] = (rbpf_real_t)(count / row_sum);
        }
    }

    rbpf_ksc_build_transition_lut(rbpf, flat_matrix);
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

    /* Transition learning defaults */
    ext->trans_learn_enabled = 0;
    ext->trans_forgetting = 0.995;
    ext->trans_prior_diag = 50.0;
    ext->trans_prior_off = 1.0;
    ext->trans_update_interval = 100;
    ext->trans_ticks_since_update = 0;
    memset(ext->trans_counts, 0, sizeof(ext->trans_counts));

    /* Per-particle parameter mode */
    ext->rbpf->use_learned_params = 1;

    /* Hawkes defaults (disabled) */
    ext->hawkes.enabled = 0;
    ext->hawkes.mu = RBPF_REAL(0.05);
    ext->hawkes.alpha = RBPF_REAL(0.3);
    ext->hawkes.beta = RBPF_REAL(0.1);
    ext->hawkes.threshold = RBPF_REAL(0.03);
    ext->hawkes.intensity = ext->hawkes.mu;
    ext->hawkes.intensity_prev = ext->hawkes.mu;
    ext->hawkes.boost_scale = RBPF_REAL(0.1);
    ext->hawkes.boost_cap = RBPF_REAL(0.25);
    ext->hawkes.lut_dirty = 0;
    ext->hawkes.adaptive_beta_enabled = 1;
    for (int r = 0; r < RBPF_MAX_REGIMES; r++)
    {
        ext->hawkes.beta_regime_scale[r] = RBPF_REAL(1.0);
    }
    ext->hawkes.beta_regime_scale[0] = RBPF_REAL(2.0);
    ext->hawkes.beta_regime_scale[1] = RBPF_REAL(1.5);
    ext->hawkes.beta_regime_scale[2] = RBPF_REAL(1.0);
    ext->hawkes.beta_regime_scale[3] = RBPF_REAL(0.5);

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
    }
    ext->cooldown_remaining = 0;

    /* Initialize policy engine state */
    ext->prev_sprt_regime = 0; /* Start in regime 0 (typically calm) */
    ext->structural_break_signaled = 0;
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
        /* Don't clear yet - smoother needs to see it */
    }

    /*═══════════════════════════════════════════════════════════════════════
     * PHASE 1: HAWKES PRE-STEP (modify transitions)
     *═══════════════════════════════════════════════════════════════════════*/
    if (ext->hawkes.enabled)
    {
        rbpf_ext_hawkes_apply_to_transitions(ext);
    }

    /*═══════════════════════════════════════════════════════════════════════
     * PHASE 2: RBPF FORWARD PASS
     *═══════════════════════════════════════════════════════════════════════*/

    /* Transform: y = log(r²) */
    rbpf_real_t y;
    if (rbpf_fabs(obs) < RBPF_REAL(1e-10))
    {
        y = RBPF_REAL(-23.0);
    }
    else
    {
        y = rbpf_log(obs * obs);
    }

    rbpf_ksc_transition(rbpf);
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

    rbpf_ksc_compute_outputs(rbpf, marginal_lik, output);
    output->resampled = rbpf_ksc_resample(rbpf);

    /*═══════════════════════════════════════════════════════════════════════
     * PHASE 3: HAWKES POST-STEP (update intensity)
     *═══════════════════════════════════════════════════════════════════════*/
    if (ext->hawkes.enabled)
    {
        rbpf_ext_hawkes_update_intensity(ext, obs);
        rbpf_ext_hawkes_restore_base_transitions(ext);
    }

    /*═══════════════════════════════════════════════════════════════════════
     * PHASE 4: OUTLIER FRACTION
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
    if (output)
    {
        output->outlier_fraction = ext->last_outlier_fraction;
    }

    /*═══════════════════════════════════════════════════════════════════════
     * PHASE 5: ADAPTIVE FORGETTING
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
    }

    /*═══════════════════════════════════════════════════════════════════════
     * PHASE 5.5: POLICY ENGINE (Regime Change Detection)
     *
     * The Extended layer owns the decision logic. Core only computes signals.
     *
     * Two independent detectors:
     *   1. P² Circuit Breaker: "Tail event - old world is dead"
     *   2. SPRT Transition: "Statistically confirmed regime flip"
     *
     * P² fires on 99.9th percentile surprise (data-driven, no heuristics).
     * SPRT fires on accumulated log-likelihood evidence.
     *
     * When P² fires, we synchronize SPRT with the particle filter's
     * dominant regime to prevent "ghost evidence" fighting the new reality.
     *
     * NOTE: Must run AFTER adaptive forgetting (which sets P² flag for
     * THIS tick) but BEFORE Storvik (which needs structural_break_signaled).
     *═══════════════════════════════════════════════════════════════════════*/
    {
        int p2_tail_event = rbpf_ext_structural_break_detected(ext);
        int sprt_flip = (output->smoothed_regime != ext->prev_sprt_regime);

        if (p2_tail_event)
        {
            /*───────────────────────────────────────────────────────────────
             * PATH 1: UNPRECEDENTED SURPRISE (Circuit Breaker)
             *
             * The P² algorithm says this tick is in the extreme tail.
             * The old world model is dead. Full reset.
             *───────────────────────────────────────────────────────────────*/
            output->regime_changed = 1;
            output->change_type = 1; /* Tail event */

            /* Synchronize SPRT with reality:
             * - Reset accumulated evidence (no "ghost" from old regime)
             * - Force to particle filter's dominant regime
             * - SPRT now validates the NEW regime instead of fighting it */
            sprt_multi_force_regime(&rbpf->sprt, output->dominant_regime);

            /* Signal smoother for emergency flush (if enabled) */
            ext->structural_break_signaled = 1;

            /* Update our tracking */
            ext->prev_sprt_regime = output->dominant_regime;
        }
        else if (sprt_flip)
        {
            /*───────────────────────────────────────────────────────────────
             * PATH 2: STATISTICAL TRANSITION (Drift/Regime Switch)
             *
             * SPRT has accumulated enough evidence to confirm a regime
             * change. This is a "normal" transition - learn from it.
             *───────────────────────────────────────────────────────────────*/
            output->regime_changed = 1;
            output->change_type = 2; /* SPRT transition */

            /* Learn from this transition (Soft Dirichlet update) */
            if (rbpf->trans_prior_enabled)
            {
                dirichlet_transition_update(&rbpf->trans_prior,
                                            ext->prev_sprt_regime,
                                            output->smoothed_regime);
                rbpf_rebuild_trans_lut_from_dirichlet(rbpf);
            }

            /* Update tracking */
            ext->prev_sprt_regime = output->smoothed_regime;
        }
        else
        {
            /*───────────────────────────────────────────────────────────────
             * PATH 3: STEADY STATE
             *
             * Normal operation. No regime change detected.
             *───────────────────────────────────────────────────────────────*/
            output->regime_changed = 0;
            output->change_type = 0;
        }
    }

    /*═══════════════════════════════════════════════════════════════════════
     * PHASE 6: STORVIK PARAMETER LEARNING
     *═══════════════════════════════════════════════════════════════════════*/
    if (ext->storvik_initialized)
    {
        if (output->resampled)
        {
            param_learn_apply_resampling(&ext->storvik, rbpf->indices, n);
        }

        /*───────────────────────────────────────────────────────────────────
         * HYBRID APPROACH:
         *   1. ALWAYS run filtered Storvik (real-time, every tick)
         *   2. If smoother enabled, ALSO push to buffer
         *   3. On structural break, smoother does PARIS flush
         *
         * This gives us:
         *   - Real-time parameter tracking (no blind period)
         *   - Smoothed corrections on regime changes
         *───────────────────────────────────────────────────────────────────*/

        /* Always: Filtered Storvik update */
        extract_particle_info_optimized(ext, output->resampled);
        param_learn_update(&ext->storvik, ext->particle_info, n);

        /* Additionally: Smoother buffer management (if enabled) */
        if (ext->smoothed_storvik_enabled && ext->smoother)
        {
            rbpf_ext_smoother_step(ext, output);
        }
    }

    /* Clear structural break flag after smoother has seen it */
    ext->structural_break_signaled = 0;

    /*═══════════════════════════════════════════════════════════════════════
     * PHASE 7: TRANSITION LEARNING
     *═══════════════════════════════════════════════════════════════════════*/
    if (ext->trans_learn_enabled)
    {
        update_transition_counts_optimized(ext);

        ext->trans_ticks_since_update++;
        if (ext->trans_ticks_since_update >= ext->trans_update_interval)
        {
            rebuild_transition_lut(ext);
            ext->trans_ticks_since_update = 0;

            /* Update base matrix for Hawkes */
            const int nr = rbpf->n_regimes;
            for (int i = 0; i < nr; i++)
            {
                for (int j = 0; j < nr; j++)
                {
                    int count = 0;
                    for (int k = 0; k < 1024; k++)
                    {
                        if (rbpf->trans_lut[i][k] == j)
                            count++;
                    }
                    ext->base_trans_matrix[i * nr + j] =
                        (rbpf_real_t)count / RBPF_REAL(1024.0);
                }
            }
        }
    }

    /*═══════════════════════════════════════════════════════════════════════
     * PHASE 8: LAG BUFFERS & SYNC
     *═══════════════════════════════════════════════════════════════════════*/
    update_lag_buffers(ext);
    sync_storvik_to_rbpf_optimized(ext);

    /*═══════════════════════════════════════════════════════════════════════
     * PHASE 9: POPULATE OUTPUT
     *═══════════════════════════════════════════════════════════════════════*/
    for (int r = 0; r < rbpf->n_regimes; r++)
    {
        rbpf_ext_get_learned_params(ext, r,
                                    &output->learned_mu_vol[r],
                                    &output->learned_sigma_vol[r]);
    }
}

void rbpf_ext_step_apf(RBPF_Extended *ext, rbpf_real_t obs_current,
                       rbpf_real_t obs_next, RBPF_KSC_Output *output)
{
    /* APF disabled - fallback to standard step */
    rbpf_ext_step(ext, obs_current, output);
    (void)obs_next;
}

/*═══════════════════════════════════════════════════════════════════════════
 * BASIC CONFIGURATION
 *═══════════════════════════════════════════════════════════════════════════*/

void rbpf_ext_set_regime_params(RBPF_Extended *ext, int regime,
                                rbpf_real_t theta, rbpf_real_t mu_vol,
                                rbpf_real_t sigma_vol)
{
    if (!ext || regime < 0 || regime >= RBPF_MAX_REGIMES)
        return;

    rbpf_ksc_set_regime_params(ext->rbpf, regime, theta, mu_vol, sigma_vol);

    if (ext->storvik_initialized)
    {
        rbpf_real_t phi = RBPF_REAL(1.0) - theta;
        param_learn_set_prior(&ext->storvik, regime, mu_vol, phi, sigma_vol);
    }
}

void rbpf_ext_build_transition_lut(RBPF_Extended *ext, const rbpf_real_t *trans_matrix)
{
    if (!ext)
        return;

    const int n = ext->rbpf->n_regimes;
    memcpy(ext->base_trans_matrix, trans_matrix, n * n * sizeof(rbpf_real_t));
    rbpf_ksc_build_transition_lut(ext->rbpf, trans_matrix);
    ext->hawkes.lut_dirty = 0;
}

void rbpf_ext_set_storvik_interval(RBPF_Extended *ext, int regime, int interval)
{
    if (!ext || !ext->storvik_initialized)
        return;
    if (regime < 0 || regime >= PARAM_LEARN_MAX_REGIMES)
        return;
    ext->storvik.config.sample_interval[regime] = interval;
}

void rbpf_ext_set_hft_mode(RBPF_Extended *ext, int enable)
{
    if (!ext || !ext->storvik_initialized)
        return;

    if (enable)
    {
        ext->storvik.config.sample_interval[0] = 100;
        ext->storvik.config.sample_interval[1] = 50;
        ext->storvik.config.sample_interval[2] = 20;
        ext->storvik.config.sample_interval[3] = 5;
    }
    else
    {
        for (int r = 0; r < PARAM_LEARN_MAX_REGIMES; r++)
        {
            ext->storvik.config.sample_interval[r] = 1;
        }
    }
}

void rbpf_ext_set_full_update_mode(RBPF_Extended *ext)
{
    if (!ext || !ext->storvik_initialized)
        return;

    for (int r = 0; r < PARAM_LEARN_MAX_REGIMES; r++)
    {
        ext->storvik.config.sample_interval[r] = 1;
    }
    ext->storvik.config.enable_global_tick_skip = false;
    ext->storvik.config.enable_forgetting = true;
    ext->storvik.config.forgetting_lambda = 0.997;
}

void rbpf_ext_signal_structural_break(RBPF_Extended *ext)
{
    if (!ext)
        return;
    ext->structural_break_signaled = 1;
    if (ext->storvik_initialized)
    {
        param_learn_signal_structural_break(&ext->storvik);
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * TRANSITION LEARNING
 *═══════════════════════════════════════════════════════════════════════════*/

void rbpf_ext_enable_transition_learning(RBPF_Extended *ext, int enable)
{
    if (!ext)
        return;
    ext->trans_learn_enabled = enable;
    if (enable)
        rbpf_ext_reset_transition_counts(ext);
}

void rbpf_ext_configure_transition_learning(RBPF_Extended *ext,
                                            double forgetting,
                                            double prior_diag,
                                            double prior_off,
                                            int update_interval)
{
    if (!ext)
        return;
    ext->trans_forgetting = forgetting;
    ext->trans_prior_diag = prior_diag;
    ext->trans_prior_off = prior_off;
    ext->trans_update_interval = update_interval;
}

void rbpf_ext_reset_transition_counts(RBPF_Extended *ext)
{
    if (!ext)
        return;
    memset(ext->trans_counts, 0, sizeof(ext->trans_counts));
    ext->trans_ticks_since_update = 0;
}

double rbpf_ext_get_transition_prob(const RBPF_Extended *ext, int from, int to)
{
    if (!ext || !ext->rbpf)
        return 0.0;
    if (from < 0 || from >= ext->rbpf->n_regimes)
        return 0.0;
    if (to < 0 || to >= ext->rbpf->n_regimes)
        return 0.0;

    const int nr = ext->rbpf->n_regimes;
    const double prior = (from == to) ? ext->trans_prior_diag : ext->trans_prior_off;

    double row_sum = 0.0;
    for (int j = 0; j < nr; j++)
    {
        double p = (from == j) ? ext->trans_prior_diag : ext->trans_prior_off;
        row_sum += ext->trans_counts[from][j] + p;
    }

    return (ext->trans_counts[from][to] + prior) / row_sum;
}

/*═══════════════════════════════════════════════════════════════════════════
 * PARAMETER ACCESS
 *═══════════════════════════════════════════════════════════════════════════*/

void rbpf_ext_get_learned_params(const RBPF_Extended *ext, int regime,
                                 rbpf_real_t *mu_vol, rbpf_real_t *sigma_vol)
{
    if (!ext || regime < 0 || regime >= RBPF_MAX_REGIMES)
    {
        if (mu_vol)
            *mu_vol = RBPF_REAL(-4.6);
        if (sigma_vol)
            *sigma_vol = RBPF_REAL(0.1);
        return;
    }

    switch (ext->param_mode)
    {
    case RBPF_PARAM_STORVIK:
    case RBPF_PARAM_HYBRID:
        if (ext->storvik_initialized)
        {
            RegimeParams params;
            param_learn_get_params(&ext->storvik, 0, regime, &params);
            if (mu_vol)
                *mu_vol = (rbpf_real_t)params.mu;
            if (sigma_vol)
                *sigma_vol = (rbpf_real_t)params.sigma;
        }
        else
        {
            if (mu_vol)
                *mu_vol = ext->rbpf->params[regime].mu_vol;
            if (sigma_vol)
                *sigma_vol = ext->rbpf->params[regime].sigma_vol;
        }
        break;

    default:
        if (mu_vol)
            *mu_vol = ext->rbpf->params[regime].mu_vol;
        if (sigma_vol)
            *sigma_vol = ext->rbpf->params[regime].sigma_vol;
        break;
    }
}

void rbpf_ext_get_storvik_summary(const RBPF_Extended *ext, int regime,
                                  RegimeParams *summary)
{
    if (!ext || !summary || !ext->storvik_initialized)
    {
        if (summary)
            memset(summary, 0, sizeof(RegimeParams));
        return;
    }
    param_learn_get_params(&ext->storvik, 0, regime, summary);
}

void rbpf_ext_get_learning_stats(const RBPF_Extended *ext,
                                 uint64_t *stat_updates,
                                 uint64_t *samples_drawn,
                                 uint64_t *samples_skipped)
{
    if (!ext || !ext->storvik_initialized)
    {
        if (stat_updates)
            *stat_updates = 0;
        if (samples_drawn)
            *samples_drawn = 0;
        if (samples_skipped)
            *samples_skipped = 0;
        return;
    }
    if (stat_updates)
        *stat_updates = ext->storvik.total_stat_updates;
    if (samples_drawn)
        *samples_drawn = ext->storvik.total_samples_drawn;
    if (samples_skipped)
        *samples_skipped = ext->storvik.samples_skipped_load;
}

/*═══════════════════════════════════════════════════════════════════════════
 * ADAPTIVE FORGETTING
 *
 * NOTE: Main functions (rbpf_ext_enable_adaptive_forgetting,
 *       rbpf_ext_enable_adaptive_forgetting_mode, rbpf_ext_enable_circuit_breaker)
 *       are implemented in rbpf_adaptive_forgetting.c
 *═══════════════════════════════════════════════════════════════════════════*/

/*═══════════════════════════════════════════════════════════════════════════
 * DIAGNOSTICS
 *═══════════════════════════════════════════════════════════════════════════*/

void rbpf_ext_print_config(const RBPF_Extended *ext)
{
    if (!ext)
        return;

    printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    printf("║   RBPF-KSC Extended Configuration                            ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    const char *mode_str;
    switch (ext->param_mode)
    {
    case RBPF_PARAM_DISABLED:
        mode_str = "DISABLED";
        break;
    case RBPF_PARAM_LIU_WEST:
        mode_str = "LIU-WEST";
        break;
    case RBPF_PARAM_STORVIK:
        mode_str = "STORVIK";
        break;
    case RBPF_PARAM_HYBRID:
        mode_str = "HYBRID";
        break;
    default:
        mode_str = "UNKNOWN";
        break;
    }

    printf("Parameter Learning: %s\n", mode_str);
    printf("Particles:          %d\n", ext->rbpf->n_particles);
    printf("Regimes:            %d\n", ext->rbpf->n_regimes);

#if defined(USE_AVX512)
    printf("SIMD:               AVX-512\n");
#elif defined(USE_AVX2)
    printf("SIMD:               AVX2\n");
#else
    printf("SIMD:               Scalar\n");
#endif

    if (ext->storvik_initialized)
    {
        printf("\nStorvik Sampling Intervals:\n");
        for (int r = 0; r < ext->rbpf->n_regimes; r++)
        {
            printf("  R%d: every %d ticks\n", r,
                   ext->storvik.config.sample_interval[r]);
        }
    }

    printf("\n  Hawkes Self-Excitation:\n");
    if (ext->hawkes.enabled)
    {
        printf("    Enabled:     YES\n");
        printf("    μ (base):    %.4f\n", (float)ext->hawkes.mu);
        printf("    α (jump):    %.4f\n", (float)ext->hawkes.alpha);
        printf("    β (decay):   %.4f (half-life: %.1f ticks)\n",
               (float)ext->hawkes.beta, 0.693f / (float)ext->hawkes.beta);
        printf("    Threshold:   %.2f%%\n", (float)ext->hawkes.threshold * 100);
    }
    else
    {
        printf("    Enabled:     NO\n");
    }

    printf("\n  Robust OCSN (11th Component):\n");
    if (ext->robust_ocsn.enabled)
    {
        printf("    Enabled:     YES\n");
        for (int r = 0; r < ext->rbpf->n_regimes; r++)
        {
            printf("      R%d: prob=%.1f%%, var=%.1f\n", r,
                   (float)ext->robust_ocsn.regime[r].prob * 100,
                   (float)ext->robust_ocsn.regime[r].variance);
        }
    }
    else
    {
        printf("    Enabled:     NO\n");
    }

    /* Print smoother config */
    rbpf_ext_print_smoother_config(ext);

    printf("\n");
}

void rbpf_ext_print_storvik_stats(const RBPF_Extended *ext, int regime)
{
    if (!ext || !ext->storvik_initialized)
        return;
    printf("\nStorvik Statistics (Regime %d):\n", regime);
    param_learn_print_regime_stats(&ext->storvik, regime);
}

/*═══════════════════════════════════════════════════════════════════════════
 * KL TEMPERING API
 *
 * Information-geometric weight normalization. When enabled:
 * - rbpf_ksc_update() stores log_lik_increment but doesn't apply
 * - rbpf_kl_step() computes KL divergence and applies tempered weights
 * - Zombie detection triggers structural break signals
 *═══════════════════════════════════════════════════════════════════════════*/

void rbpf_ext_enable_kl_tempering(RBPF_Extended *ext)
{
    if (!ext)
        return;

    ext->kl_tempering_enabled = 1;

    /* Enable deferred weight mode on underlying RBPF */
    rbpf_ksc_set_deferred_weight_mode(ext->rbpf, 1);

    /* Initialize KL state if not already done */
    if (ext->kl_state)
    {
        rbpf_kl_state_reset(ext->kl_state);
    }
}

void rbpf_ext_disable_kl_tempering(RBPF_Extended *ext)
{
    if (!ext)
        return;

    ext->kl_tempering_enabled = 0;

    /* Disable deferred weight mode - weights applied immediately */
    rbpf_ksc_set_deferred_weight_mode(ext->rbpf, 0);
}

int rbpf_ext_kl_tempering_enabled(const RBPF_Extended *ext)
{
    return ext ? ext->kl_tempering_enabled : 0;
}

float rbpf_ext_get_last_beta(const RBPF_Extended *ext)
{
    if (!ext || !ext->kl_state)
        return 1.0f;
    return ext->kl_state->last_beta;
}

float rbpf_ext_get_last_kl(const RBPF_Extended *ext)
{
    if (!ext || !ext->kl_state)
        return 0.0f;
    return ext->kl_state->last_kl;
}

uint64_t rbpf_ext_get_zombie_resets(const RBPF_Extended *ext)
{
    if (!ext || !ext->kl_state)
        return 0;
    return ext->kl_state->zombie_reset_count;
}

void rbpf_ext_print_kl_diagnostics(const RBPF_Extended *ext)
{
    if (!ext)
    {
        printf("KL Tempering: ext is NULL\n");
        return;
    }

    printf("\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  KL Tempering Diagnostics\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  Enabled:           %s\n", ext->kl_tempering_enabled ? "YES" : "NO");

    if (ext->kl_tempering_enabled && ext->kl_state)
    {
        RBPF_KL_State *kl = ext->kl_state;

        printf("  Particles:         %d\n", kl->n_particles);
        printf("  KL ceiling:        %.4f nats (log N)\n", kl->max_kl);
        printf("  Beta floor:        %.2f\n", kl->beta_floor);
        printf("\n");
        printf("  Last tick:\n");
        printf("    KL divergence:   %.4f nats\n", kl->last_kl);
        printf("    Beta applied:    %.4f\n", kl->last_beta);
        printf("    log(Z_old):      %.4f\n", kl->log_Z_old);
        printf("\n");
        printf("  Counters:\n");
        printf("    Ticks processed: %" PRIu64 "\n", kl->ticks_processed);
        printf("    Hard clamps:     %" PRIu64 " (%.3f%%)\n",
               kl->hard_clamp_count,
               kl->ticks_processed > 0 ? 100.0 * kl->hard_clamp_count / kl->ticks_processed : 0.0);
        printf("    Soft dampens:    %" PRIu64 " (%.3f%%)\n",
               kl->soft_dampen_count,
               kl->ticks_processed > 0 ? 100.0 * kl->soft_dampen_count / kl->ticks_processed : 0.0);
        printf("    Zombie resets:   %" PRIu64 "\n", kl->zombie_reset_count);
        printf("\n");
        printf("  Zombie state:\n");
        printf("    Consecutive low: %d / %d\n",
               kl->consecutive_low_beta, kl->zombie_threshold);
        printf("    Currently zombie: %s\n",
               kl->consecutive_low_beta >= kl->zombie_threshold ? "YES" : "NO");
        printf("\n");
        printf("  P² Quantile (p95):\n");
        printf("    Warmup:          %s (%" PRIu64 " / %d)\n",
               kl->warmup_complete ? "COMPLETE" : "IN PROGRESS",
               kl->ticks_processed, kl->warmup_ticks);
        printf("    Current p95:     %.4f nats\n", kl->kl_p95);
    }
    else if (!ext->kl_tempering_enabled)
    {
        printf("  (Enable with rbpf_ext_enable_kl_tempering())\n");
    }
    else
    {
        printf("  KL state not initialized\n");
    }

    printf("═══════════════════════════════════════════════════════════════\n");
}