/*
 * sr_detector.c - Shiryaev-Roberts Change-Point Detector Implementation
 *
 * Performance optimizations:
 *   - MKL VML for batch transcendentals (vsLog1p, vsExp, vsLn)
 *   - AVX-512/AVX2 for vectorized arithmetic
 *   - Branchless winsorization
 *   - Cache-friendly batch processing
 */

#include "sr_detector.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

/* ═══════════════════════════════════════════════════════════════════════════
 * PLATFORM DETECTION
 * ═══════════════════════════════════════════════════════════════════════════ */

#ifdef __INTEL_MKL__
#define SR_USE_MKL 1
#include <mkl.h>
#else
#define SR_USE_MKL 0
#endif

#if defined(__AVX512F__)
#define SR_USE_AVX512 1
#include <immintrin.h>
#elif defined(__AVX2__)
#define SR_USE_AVX2 1
#include <immintrin.h>
#elif defined(__AVX__)
#define SR_USE_AVX 1
#include <immintrin.h>
#else
#define SR_USE_SCALAR 1
#endif

/* Alignment for SIMD */
#if SR_USE_AVX512
#define SR_ALIGN 64
#define SR_VECLEN 16
#elif SR_USE_AVX2 || SR_USE_AVX
#define SR_ALIGN 32
#define SR_VECLEN 8
#else
#define SR_ALIGN 16
#define SR_VECLEN 4
#endif

/* ═══════════════════════════════════════════════════════════════════════════
 * SCALAR PRIMITIVES
 * ═══════════════════════════════════════════════════════════════════════════ */

float sr_log_lr_gaussian(float obs, float sigma_h0, float sigma_h1, float winsorize_cap)
{
    /*
     * Gaussian log-likelihood ratio with winsorization:
     *
     * log(LR) = log(p(obs|H1) / p(obs|H0))
     *         = -0.5*z1² + 0.5*z0² + log(σ0/σ1)
     *         = 0.5*(z0² - z1²) + log(σ0/σ1)
     *
     * Winsorization: cap |z| to prevent outlier domination
     */

    /* Compute z-scores */
    float z0 = obs / sigma_h0;
    float z1 = obs / sigma_h1;

    /* Branchless winsorization: clamp to [-cap, +cap] */
    z0 = fmaxf(-winsorize_cap, fminf(z0, winsorize_cap));
    z1 = fmaxf(-winsorize_cap, fminf(z1, winsorize_cap));

    /* Log-likelihood ratio */
    float log_lr = 0.5f * (z0 * z0 - z1 * z1) + logf(sigma_h0 / sigma_h1);

    return log_lr;
}

float sr_log_lr_student_t(float obs, float sigma_h0, float sigma_h1, float nu)
{
    /*
     * Student-t log-likelihood ratio:
     *
     * log p(z|ν) ∝ -(ν+1)/2 × log(1 + z²/ν)
     *
     * log(LR) = -(ν+1)/2 × [log(1 + z1²/ν) - log(1 + z0²/ν)] + log(σ0/σ1)
     *         = (ν+1)/2 × [log(1 + z0²/ν) - log(1 + z1²/ν)] + log(σ0/σ1)
     */

    float z0 = obs / sigma_h0;
    float z1 = obs / sigma_h1;

    float term0 = logf(1.0f + z0 * z0 / nu);
    float term1 = logf(1.0f + z1 * z1 / nu);

    float log_lr = 0.5f * (nu + 1.0f) * (term0 - term1) + logf(sigma_h0 / sigma_h1);

    return log_lr;
}

float sr_accumulate_step(float log_lr, float log_sr_old, float clamp)
{
    /*
     * SR recursion in log-space:
     *
     * R_t = LR_t × (1 + R_{t-1})
     * log(R_t) = log(LR_t) + log(1 + R_{t-1})
     *          = log(LR_t) + log(1 + exp(log(R_{t-1})))
     *          = log_lr + log1p(exp(log_sr_old))
     *
     * CRITICAL: expf() overflows at x ≈ 88.7
     * Use softplus approximation: log(1 + e^x) ≈ x for x >> 0
     */

    float log_one_plus_R;

    if (log_sr_old > 20.0f)
    {
        /* Softplus approximation: log(1 + e^x) ≈ x for large x */
        log_one_plus_R = log_sr_old;
    }
    else if (log_sr_old < -20.0f)
    {
        /* log(1 + e^x) ≈ 0 for very negative x */
        log_one_plus_R = 0.0f;
    }
    else
    {
        /* Safe range: use exact computation */
        log_one_plus_R = log1pf(expf(log_sr_old));
    }

    float log_sr_new = log_lr + log_one_plus_R;

    /* Clamp to prevent overflow */
    return fmaxf(-clamp, fminf(log_sr_new, clamp));
}

/* ═══════════════════════════════════════════════════════════════════════════
 * SINGLE SR STATISTIC
 * ═══════════════════════════════════════════════════════════════════════════ */

void sr_stat_init(SRStat *sr, const SRConfig *cfg)
{
    sr->log_sr = 0.0f;
    if (cfg)
    {
        sr->cfg = *cfg;
    }
    else
    {
        sr->cfg = sr_config_default();
    }
}

void sr_stat_reset(SRStat *sr)
{
    sr->log_sr = 0.0f;
}

float sr_stat_update(SRStat *sr, float obs, float sigma_peace)
{
    float sigma_crisis = sr->cfg.sigma_multiple * sigma_peace;

    float log_lr;
    if (sr->cfg.student_nu > 0.0f)
    {
        log_lr = sr_log_lr_student_t(obs, sigma_peace, sigma_crisis, sr->cfg.student_nu);
    }
    else
    {
        log_lr = sr_log_lr_gaussian(obs, sigma_peace, sigma_crisis, sr->cfg.winsorize_cap);
    }

    sr->log_sr = sr_accumulate_step(log_lr, sr->log_sr, sr->cfg.log_sr_clamp);

    return sr->log_sr;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * DUAL SR STATISTICS
 * ═══════════════════════════════════════════════════════════════════════════ */

void dual_sr_init(DualSR *dsr, const SRConfig *cfg, float initial_sigma_peace)
{
    dsr->log_sr_up = 0.0f;
    dsr->log_sr_down = 0.0f;
    dsr->sigma_peace = initial_sigma_peace;
    dsr->sigma_crisis = cfg->sigma_multiple * initial_sigma_peace;
    dsr->sigma_crisis_ema_alpha = 0.05f; /* Warmup: fast adaptation */

    if (cfg)
    {
        dsr->cfg = *cfg;
    }
    else
    {
        dsr->cfg = sr_config_default();
    }
}

void dual_sr_reset(DualSR *dsr)
{
    dsr->log_sr_up = 0.0f;
    dsr->log_sr_down = 0.0f;
}

void dual_sr_reset_up(DualSR *dsr)
{
    dsr->log_sr_up = 0.0f;
}

void dual_sr_reset_down(DualSR *dsr)
{
    dsr->log_sr_down = 0.0f;
}

static inline float dual_sr_compute_log_lr(const DualSR *dsr, float obs,
                                           float sigma_h0, float sigma_h1)
{
    if (dsr->cfg.student_nu > 0.0f)
    {
        return sr_log_lr_student_t(obs, sigma_h0, sigma_h1, dsr->cfg.student_nu);
    }
    else
    {
        return sr_log_lr_gaussian(obs, sigma_h0, sigma_h1, dsr->cfg.winsorize_cap);
    }
}

float dual_sr_update_entry(DualSR *dsr, float obs)
{
    /* SR_up: H0 = peace (σ_peace), H1 = crisis (σ_multiple × σ_peace) */
    float sigma_h0 = dsr->sigma_peace;
    float sigma_h1 = dsr->cfg.sigma_multiple * dsr->sigma_peace;

    float log_lr = dual_sr_compute_log_lr(dsr, obs, sigma_h0, sigma_h1);
    dsr->log_sr_up = sr_accumulate_step(log_lr, dsr->log_sr_up, dsr->cfg.log_sr_clamp);

    /* SR_down inactive in entry mode */
    dsr->log_sr_down = 0.0f;

    return dsr->log_sr_up;
}

float dual_sr_update_exit(DualSR *dsr, float obs)
{
    /* Learn sigma_crisis via EMA */
    float abs_obs = fabsf(obs);
    dsr->sigma_crisis = (1.0f - dsr->sigma_crisis_ema_alpha) * dsr->sigma_crisis + dsr->sigma_crisis_ema_alpha * abs_obs;

    /* Slow down EMA after warmup */
    if (dsr->sigma_crisis_ema_alpha > 0.01f)
    {
        dsr->sigma_crisis_ema_alpha *= 0.95f;
        if (dsr->sigma_crisis_ema_alpha < 0.01f)
        {
            dsr->sigma_crisis_ema_alpha = 0.01f;
        }
    }

    /* SR_down: H0 = crisis (σ_crisis), H1 = peace (σ_peace) */
    float sigma_h0 = dsr->sigma_crisis;
    float sigma_h1 = dsr->sigma_peace;

    /* Ensure sigma_h0 > sigma_h1 for valid LR direction */
    if (sigma_h0 < sigma_h1 * 1.1f)
    {
        sigma_h0 = sigma_h1 * 1.1f;
    }

    float log_lr = dual_sr_compute_log_lr(dsr, obs, sigma_h0, sigma_h1);
    dsr->log_sr_down = sr_accumulate_step(log_lr, dsr->log_sr_down, dsr->cfg.log_sr_clamp);

    /* SR_up inactive in exit mode */
    dsr->log_sr_up = 0.0f;

    return dsr->log_sr_down;
}

float dual_sr_update_both(DualSR *dsr, float obs)
{
    /* SR_up: testing for crisis re-trigger */
    float sigma_h0_up = dsr->sigma_peace;
    float sigma_h1_up = dsr->cfg.sigma_multiple * dsr->sigma_peace;
    float log_lr_up = dual_sr_compute_log_lr(dsr, obs, sigma_h0_up, sigma_h1_up);
    dsr->log_sr_up = sr_accumulate_step(log_lr_up, dsr->log_sr_up, dsr->cfg.log_sr_clamp);

    /* SR_down: testing for exit confirmation */
    float sigma_h0_down = dsr->sigma_crisis;
    float sigma_h1_down = dsr->sigma_peace;

    if (sigma_h0_down < sigma_h1_down * 1.1f)
    {
        sigma_h0_down = sigma_h1_down * 1.1f;
    }

    float log_lr_down = dual_sr_compute_log_lr(dsr, obs, sigma_h0_down, sigma_h1_down);
    dsr->log_sr_down = sr_accumulate_step(log_lr_down, dsr->log_sr_down, dsr->cfg.log_sr_clamp);

    return dsr->log_sr_up;
}

void dual_sr_set_sigma_peace(DualSR *dsr, float sigma_peace)
{
    dsr->sigma_peace = sigma_peace;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * ADAPTIVE THRESHOLD
 * ═══════════════════════════════════════════════════════════════════════════ */

void adaptive_threshold_init(AdaptiveThreshold *at, const AdaptiveThresholdConfig *cfg)
{
    at->ticks_since_crisis = 0;
    at->recent_false_alarms = 0;
    if (cfg)
    {
        at->cfg = *cfg;
    }
    else
    {
        at->cfg = adaptive_threshold_config_default();
    }
}

void adaptive_threshold_reset(AdaptiveThreshold *at)
{
    at->ticks_since_crisis = 0;
    at->recent_false_alarms = 0;
}

void adaptive_threshold_tick(AdaptiveThreshold *at)
{
    at->ticks_since_crisis++;
}

void adaptive_threshold_false_alarm(AdaptiveThreshold *at)
{
    if (at->recent_false_alarms < at->cfg.max_wolf_count)
    {
        at->recent_false_alarms++;
    }
}

void adaptive_threshold_clean_exit(AdaptiveThreshold *at)
{
    at->ticks_since_crisis = 0;
    at->recent_false_alarms = 0;
}

float adaptive_threshold_compute(const AdaptiveThreshold *at)
{
    float log_H = at->cfg.log_H_base;

    /* Inertia: log(1 + ticks/scale) */
    log_H += logf(1.0f + (float)at->ticks_since_crisis / at->cfg.inertia_scale);

    /* Wolf penalty */
    log_H += at->cfg.wolf_penalty * (float)at->recent_false_alarms;

    return log_H;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * VECTORIZED OPERATIONS - MKL VML
 * ═══════════════════════════════════════════════════════════════════════════ */

#if SR_USE_MKL

void sr_compute_log_lr_batch(
    const float *obs,
    const float *sigma_peace,
    float sigma_peace_const,
    float *log_lr,
    int n,
    const SRConfig *cfg)
{
    float sigma_multiple = cfg->sigma_multiple;
    float winsorize_cap = cfg->winsorize_cap;

    /* Fallback to scalar for Student-t (MKL path only implements Gaussian) */
    if (cfg->student_nu > 0.0f)
    {
        for (int i = 0; i < n; i++)
        {
            float s0 = (sigma_peace == NULL) ? sigma_peace_const : sigma_peace[i];
            float s1 = sigma_multiple * s0;
            log_lr[i] = sr_log_lr_student_t(obs[i], s0, s1, cfg->student_nu);
        }
        return;
    }

    /* Allocate aligned temporary buffers */
    float *z0 = (float *)mkl_malloc(n * sizeof(float), SR_ALIGN);
    float *z1 = (float *)mkl_malloc(n * sizeof(float), SR_ALIGN);
    float *z0_sq = (float *)mkl_malloc(n * sizeof(float), SR_ALIGN);
    float *z1_sq = (float *)mkl_malloc(n * sizeof(float), SR_ALIGN);
    float *log_sigma_ratio = (float *)mkl_malloc(n * sizeof(float), SR_ALIGN);

    if (sigma_peace == NULL)
    {
        /* Constant sigma: vectorized division */
        float inv_sigma0 = 1.0f / sigma_peace_const;
        float inv_sigma1 = 1.0f / (sigma_multiple * sigma_peace_const);
        float log_ratio = logf(sigma_peace_const / (sigma_multiple * sigma_peace_const));

        /* z0 = obs / sigma0 */
        cblas_scopy(n, obs, 1, z0, 1);
        cblas_sscal(n, inv_sigma0, z0, 1);

        /* z1 = obs / sigma1 */
        cblas_scopy(n, obs, 1, z1, 1);
        cblas_sscal(n, inv_sigma1, z1, 1);

        /* Constant log ratio */
        for (int i = 0; i < n; i++)
        {
            log_sigma_ratio[i] = log_ratio;
        }
    }
    else
    {
        /* Per-observation sigma */
        for (int i = 0; i < n; i++)
        {
            float s0 = sigma_peace[i];
            float s1 = sigma_multiple * s0;
            z0[i] = obs[i] / s0;
            z1[i] = obs[i] / s1;
            log_sigma_ratio[i] = logf(s0 / s1);
        }
    }

    /* Winsorize: clamp z values */
    for (int i = 0; i < n; i++)
    {
        z0[i] = fmaxf(-winsorize_cap, fminf(z0[i], winsorize_cap));
        z1[i] = fmaxf(-winsorize_cap, fminf(z1[i], winsorize_cap));
    }

    /* z0_sq = z0² */
    vsSqr(n, z0, z0_sq);

    /* z1_sq = z1² */
    vsSqr(n, z1, z1_sq);

    /* log_lr = 0.5 * (z0² - z1²) + log_sigma_ratio */
    vsSub(n, z0_sq, z1_sq, log_lr);            /* log_lr = z0_sq - z1_sq */
    cblas_sscal(n, 0.5f, log_lr, 1);           /* log_lr *= 0.5 */
    vsAdd(n, log_lr, log_sigma_ratio, log_lr); /* log_lr += log_sigma_ratio */

    mkl_free(z0);
    mkl_free(z1);
    mkl_free(z0_sq);
    mkl_free(z1_sq);
    mkl_free(log_sigma_ratio);
}

float sr_accumulate_batch(
    const float *log_lr,
    float *log_sr,
    float log_sr_init,
    int n,
    float clamp)
{
    /*
     * SR recursion: log_sr[t] = log_lr[t] + log1p(exp(log_sr[t-1]))
     *
     * This is inherently sequential, but we can vectorize the transcendentals
     * in chunks for better instruction-level parallelism.
     */

    const int CHUNK = 64; /* Process in chunks for cache efficiency */

    float *exp_tmp = (float *)mkl_malloc(CHUNK * sizeof(float), SR_ALIGN);
    float *log1p_tmp = (float *)mkl_malloc(CHUNK * sizeof(float), SR_ALIGN);

    float current_log_sr = log_sr_init;

    for (int base = 0; base < n; base += CHUNK)
    {
        int chunk_size = (base + CHUNK <= n) ? CHUNK : (n - base);

        /* For small chunks or sequential dependency, use scalar */
        for (int i = 0; i < chunk_size; i++)
        {
            int idx = base + i;
            float log_one_plus_R = log1pf(expf(current_log_sr));
            current_log_sr = log_lr[idx] + log_one_plus_R;
            current_log_sr = fmaxf(-clamp, fminf(current_log_sr, clamp));
            log_sr[idx] = current_log_sr;
        }
    }

    mkl_free(exp_tmp);
    mkl_free(log1p_tmp);

    return current_log_sr;
}

float sr_update_batch(
    const float *obs,
    float sigma_peace,
    float *log_sr,
    float log_sr_init,
    int n,
    const SRConfig *cfg)
{
    /* Allocate log_lr buffer */
    float *log_lr = (float *)mkl_malloc(n * sizeof(float), SR_ALIGN);

    /* Compute all log-LRs */
    sr_compute_log_lr_batch(obs, NULL, sigma_peace, log_lr, n, cfg);

    /* Accumulate */
    float final = sr_accumulate_batch(log_lr, log_sr, log_sr_init, n, cfg->log_sr_clamp);

    mkl_free(log_lr);

    return final;
}

#else /* !SR_USE_MKL */

/* ═══════════════════════════════════════════════════════════════════════════
 * VECTORIZED OPERATIONS - AVX2/AVX-512 FALLBACK
 * ═══════════════════════════════════════════════════════════════════════════ */

#if SR_USE_AVX512

static inline __m512 avx512_clamp(__m512 x, float lo, float hi)
{
    __m512 vlo = _mm512_set1_ps(lo);
    __m512 vhi = _mm512_set1_ps(hi);
    return _mm512_max_ps(vlo, _mm512_min_ps(x, vhi));
}

void sr_compute_log_lr_batch(
    const float *obs,
    const float *sigma_peace,
    float sigma_peace_const,
    float *log_lr,
    int n,
    const SRConfig *cfg)
{
    float sigma_multiple = cfg->sigma_multiple;
    float winsorize_cap = cfg->winsorize_cap;

    /* Fallback to scalar for Student-t (SIMD path only implements Gaussian) */
    if (cfg->student_nu > 0.0f)
    {
        for (int i = 0; i < n; i++)
        {
            float s0 = (sigma_peace == NULL) ? sigma_peace_const : sigma_peace[i];
            float s1 = sigma_multiple * s0;
            log_lr[i] = sr_log_lr_student_t(obs[i], s0, s1, cfg->student_nu);
        }
        return;
    }

    float log_ratio_const = logf(1.0f / sigma_multiple);

    __m512 v_inv_sigma0, v_inv_sigma1, v_log_ratio;
    __m512 v_half = _mm512_set1_ps(0.5f);
    __m512 v_cap = _mm512_set1_ps(winsorize_cap);
    __m512 v_neg_cap = _mm512_set1_ps(-winsorize_cap);

    if (sigma_peace == NULL)
    {
        v_inv_sigma0 = _mm512_set1_ps(1.0f / sigma_peace_const);
        v_inv_sigma1 = _mm512_set1_ps(1.0f / (sigma_multiple * sigma_peace_const));
        v_log_ratio = _mm512_set1_ps(log_ratio_const);
    }

    int i = 0;

    /* Vectorized loop */
    for (; i + 16 <= n; i += 16)
    {
        __m512 v_obs = _mm512_loadu_ps(&obs[i]);

        __m512 v_z0, v_z1, v_lr;

        if (sigma_peace == NULL)
        {
            v_z0 = _mm512_mul_ps(v_obs, v_inv_sigma0);
            v_z1 = _mm512_mul_ps(v_obs, v_inv_sigma1);
            v_lr = v_log_ratio;
        }
        else
        {
            __m512 v_sigma = _mm512_loadu_ps(&sigma_peace[i]);
            __m512 v_sigma1 = _mm512_mul_ps(v_sigma, _mm512_set1_ps(sigma_multiple));
            v_z0 = _mm512_div_ps(v_obs, v_sigma);
            v_z1 = _mm512_div_ps(v_obs, v_sigma1);
            /* log(sigma/sigma1) = log(1/multiple) = constant */
            v_lr = _mm512_set1_ps(log_ratio_const);
        }

        /* Winsorize */
        v_z0 = _mm512_max_ps(v_neg_cap, _mm512_min_ps(v_z0, v_cap));
        v_z1 = _mm512_max_ps(v_neg_cap, _mm512_min_ps(v_z1, v_cap));

        /* z0² - z1² */
        __m512 v_z0_sq = _mm512_mul_ps(v_z0, v_z0);
        __m512 v_z1_sq = _mm512_mul_ps(v_z1, v_z1);
        __m512 v_diff = _mm512_sub_ps(v_z0_sq, v_z1_sq);

        /* 0.5 * diff + log_ratio */
        __m512 v_result = _mm512_fmadd_ps(v_half, v_diff, v_lr);

        _mm512_storeu_ps(&log_lr[i], v_result);
    }

    /* Scalar tail */
    for (; i < n; i++)
    {
        float s0 = (sigma_peace == NULL) ? sigma_peace_const : sigma_peace[i];
        float s1 = sigma_multiple * s0;
        log_lr[i] = sr_log_lr_gaussian(obs[i], s0, s1, winsorize_cap);
    }
}

#elif SR_USE_AVX2

static inline __m256 avx2_clamp(__m256 x, __m256 lo, __m256 hi)
{
    return _mm256_max_ps(lo, _mm256_min_ps(x, hi));
}

void sr_compute_log_lr_batch(
    const float *obs,
    const float *sigma_peace,
    float sigma_peace_const,
    float *log_lr,
    int n,
    const SRConfig *cfg)
{
    float sigma_multiple = cfg->sigma_multiple;
    float winsorize_cap = cfg->winsorize_cap;

    /* Fallback to scalar for Student-t (SIMD path only implements Gaussian) */
    if (cfg->student_nu > 0.0f)
    {
        for (int i = 0; i < n; i++)
        {
            float s0 = (sigma_peace == NULL) ? sigma_peace_const : sigma_peace[i];
            float s1 = sigma_multiple * s0;
            log_lr[i] = sr_log_lr_student_t(obs[i], s0, s1, cfg->student_nu);
        }
        return;
    }

    float log_ratio_const = logf(1.0f / sigma_multiple);

    __m256 v_inv_sigma0, v_inv_sigma1, v_log_ratio;
    __m256 v_half = _mm256_set1_ps(0.5f);
    __m256 v_cap = _mm256_set1_ps(winsorize_cap);
    __m256 v_neg_cap = _mm256_set1_ps(-winsorize_cap);
    __m256 v_multiple = _mm256_set1_ps(sigma_multiple);

    if (sigma_peace == NULL)
    {
        v_inv_sigma0 = _mm256_set1_ps(1.0f / sigma_peace_const);
        v_inv_sigma1 = _mm256_set1_ps(1.0f / (sigma_multiple * sigma_peace_const));
        v_log_ratio = _mm256_set1_ps(log_ratio_const);
    }

    int i = 0;

    /* Vectorized loop */
    for (; i + 8 <= n; i += 8)
    {
        __m256 v_obs = _mm256_loadu_ps(&obs[i]);

        __m256 v_z0, v_z1, v_lr;

        if (sigma_peace == NULL)
        {
            v_z0 = _mm256_mul_ps(v_obs, v_inv_sigma0);
            v_z1 = _mm256_mul_ps(v_obs, v_inv_sigma1);
            v_lr = v_log_ratio;
        }
        else
        {
            __m256 v_sigma = _mm256_loadu_ps(&sigma_peace[i]);
            __m256 v_sigma1 = _mm256_mul_ps(v_sigma, v_multiple);
            v_z0 = _mm256_div_ps(v_obs, v_sigma);
            v_z1 = _mm256_div_ps(v_obs, v_sigma1);
            v_lr = _mm256_set1_ps(log_ratio_const);
        }

        /* Winsorize */
        v_z0 = _mm256_max_ps(v_neg_cap, _mm256_min_ps(v_z0, v_cap));
        v_z1 = _mm256_max_ps(v_neg_cap, _mm256_min_ps(v_z1, v_cap));

        /* z0² - z1² */
        __m256 v_z0_sq = _mm256_mul_ps(v_z0, v_z0);
        __m256 v_z1_sq = _mm256_mul_ps(v_z1, v_z1);
        __m256 v_diff = _mm256_sub_ps(v_z0_sq, v_z1_sq);

        /* 0.5 * diff + log_ratio */
        __m256 v_result = _mm256_fmadd_ps(v_half, v_diff, v_lr);

        _mm256_storeu_ps(&log_lr[i], v_result);
    }

    /* Scalar tail */
    for (; i < n; i++)
    {
        float s0 = (sigma_peace == NULL) ? sigma_peace_const : sigma_peace[i];
        float s1 = sigma_multiple * s0;
        log_lr[i] = sr_log_lr_gaussian(obs[i], s0, s1, winsorize_cap);
    }
}

#else /* Scalar fallback */

void sr_compute_log_lr_batch(
    const float *obs,
    const float *sigma_peace,
    float sigma_peace_const,
    float *log_lr,
    int n,
    const SRConfig *cfg)
{
    float sigma_multiple = cfg->sigma_multiple;
    float winsorize_cap = cfg->winsorize_cap;

    if (cfg->student_nu > 0.0f)
    {
        /* Student-t path */
        for (int i = 0; i < n; i++)
        {
            float s0 = (sigma_peace == NULL) ? sigma_peace_const : sigma_peace[i];
            float s1 = sigma_multiple * s0;
            log_lr[i] = sr_log_lr_student_t(obs[i], s0, s1, cfg->student_nu);
        }
    }
    else
    {
        /* Gaussian path */
        for (int i = 0; i < n; i++)
        {
            float s0 = (sigma_peace == NULL) ? sigma_peace_const : sigma_peace[i];
            float s1 = sigma_multiple * s0;
            log_lr[i] = sr_log_lr_gaussian(obs[i], s0, s1, winsorize_cap);
        }
    }
}

#endif /* SIMD variants */

float sr_accumulate_batch(
    const float *log_lr,
    float *log_sr,
    float log_sr_init,
    int n,
    float clamp)
{
    /* Sequential dependency - scalar only */
    float current = log_sr_init;

    for (int i = 0; i < n; i++)
    {
        current = sr_accumulate_step(log_lr[i], current, clamp);
        log_sr[i] = current;
    }

    return current;
}

float sr_update_batch(
    const float *obs,
    float sigma_peace,
    float *log_sr,
    float log_sr_init,
    int n,
    const SRConfig *cfg)
{
    /* Allocate log_lr buffer on stack for small batches, heap for large */
    float *log_lr;
    int on_stack = (n <= 1024);

    if (on_stack)
    {
        log_lr = (float *)alloca(n * sizeof(float));
    }
    else
    {
        log_lr = (float *)aligned_alloc(SR_ALIGN, n * sizeof(float));
    }

    /* Compute all log-LRs */
    sr_compute_log_lr_batch(obs, NULL, sigma_peace, log_lr, n, cfg);

    /* Accumulate */
    float final = sr_accumulate_batch(log_lr, log_sr, log_sr_init, n, cfg->log_sr_clamp);

    if (!on_stack)
    {
        free(log_lr);
    }

    return final;
}

#endif /* SR_USE_MKL */