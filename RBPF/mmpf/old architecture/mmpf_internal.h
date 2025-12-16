/**
 * @file mmpf_internal.h
 * @brief MMPF-ROCKS Internal Declarations
 *
 * Shared helpers, macros, and internal function declarations.
 * Not part of public API - include only from mmpf_*.c files.
 */

#ifndef MMPF_INTERNAL_H
#define MMPF_INTERNAL_H

#include "mmpf_rocks.h"
#include "rbpf_ksc_param_integration.h" /* For adaptive forgetting API */
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/* SIMD intrinsics for double→float conversion */
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <immintrin.h>
#endif

/*═══════════════════════════════════════════════════════════════════════════
 * COMPILER HINTS
 *═══════════════════════════════════════════════════════════════════════════*/

#if defined(__GNUC__) || defined(__clang__)
#define MMPF_RESTRICT __restrict__
#elif defined(_MSC_VER)
#define MMPF_RESTRICT __restrict
#else
#define MMPF_RESTRICT
#endif

/*═══════════════════════════════════════════════════════════════════════════
 * ALIGNED ALLOCATION HELPERS
 *═══════════════════════════════════════════════════════════════════════════*/

static inline void *mmpf_aligned_alloc(size_t size, size_t alignment)
{
#ifdef _MSC_VER
    return _aligned_malloc(size, alignment);
#else
    void *ptr = NULL;
    if (posix_memalign(&ptr, alignment, size) != 0)
    {
        return NULL;
    }
    return ptr;
#endif
}

static inline void mmpf_aligned_free(void *ptr)
{
#ifdef _MSC_VER
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

/*═══════════════════════════════════════════════════════════════════════════
 * NUMERICAL HELPERS
 *═══════════════════════════════════════════════════════════════════════════*/

/* Normalize weights (in-place) */
static inline void mmpf_normalize_weights(rbpf_real_t *weights, int n)
{
    rbpf_real_t sum = RBPF_REAL(0.0);
    int i;

    for (i = 0; i < n; i++)
    {
        sum += weights[i];
    }

    if (sum > RBPF_EPS)
    {
        rbpf_real_t inv_sum = RBPF_REAL(1.0) / sum;
        for (i = 0; i < n; i++)
        {
            weights[i] *= inv_sum;
        }
    }
    else
    {
        /* Uniform fallback */
        rbpf_real_t uniform = RBPF_REAL(1.0) / n;
        for (i = 0; i < n; i++)
        {
            weights[i] = uniform;
        }
    }
}

/* Log-sum-exp for numerical stability */
static inline rbpf_real_t mmpf_log_sum_exp(const rbpf_real_t *log_w, int n)
{
    rbpf_real_t max_log = log_w[0];
    rbpf_real_t sum = RBPF_REAL(0.0);
    int i;

    for (i = 1; i < n; i++)
    {
        if (log_w[i] > max_log)
            max_log = log_w[i];
    }

    for (i = 0; i < n; i++)
    {
        sum += rbpf_exp(log_w[i] - max_log);
    }

    return max_log + rbpf_log(sum);
}

/* Normalize log-weights to linear weights */
static inline void mmpf_log_to_linear(const rbpf_real_t *log_w, rbpf_real_t *w, int n)
{
    rbpf_real_t lse = mmpf_log_sum_exp(log_w, n);
    int i;

    for (i = 0; i < n; i++)
    {
        w[i] = rbpf_exp(log_w[i] - lse);
    }
}

/* Argmax */
static inline int mmpf_argmax(const rbpf_real_t *arr, int n)
{
    int idx = 0;
    rbpf_real_t max_val = arr[0];
    int i;

    for (i = 1; i < n; i++)
    {
        if (arr[i] > max_val)
        {
            max_val = arr[i];
            idx = i;
        }
    }
    return idx;
}

/*═══════════════════════════════════════════════════════════════════════════
 * SIMD DOUBLE→FLOAT CONVERSION
 *═══════════════════════════════════════════════════════════════════════════*/

static inline void mmpf_convert_double_to_float(
    const param_real *MMPF_RESTRICT src,
    rbpf_real_t *MMPF_RESTRICT dst,
    int n)
{
    int i;

#if defined(__AVX512F__) && !defined(_MSC_VER)
    /* AVX-512: 8 doubles → 8 floats per iteration */
    for (i = 0; i + 8 <= n; i += 8)
    {
        __m512d vd = _mm512_loadu_pd(src + i);
        __m256 vf = _mm512_cvtpd_ps(vd);
        _mm256_storeu_ps(dst + i, vf);
    }
    for (; i < n; i++)
    {
        dst[i] = (rbpf_real_t)src[i];
    }

#elif defined(__AVX2__) || defined(__AVX__)
    /* AVX2: 4 doubles → 4 floats per iteration */
    for (i = 0; i + 4 <= n; i += 4)
    {
        __m256d vd = _mm256_loadu_pd(src + i);
        __m128 vf = _mm256_cvtpd_ps(vd);
        _mm_storeu_ps(dst + i, vf);
    }
    for (; i < n; i++)
    {
        dst[i] = (rbpf_real_t)src[i];
    }

#else
    /* Scalar fallback */
    for (i = 0; i < n; i++)
    {
        dst[i] = (rbpf_real_t)src[i];
    }
#endif
}

/*═══════════════════════════════════════════════════════════════════════════
 * INTERNAL FUNCTION DECLARATIONS
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Sync Storvik learned params → RBPF particle arrays
 *
 * CRITICAL: Bridges ParamLearner (double) → RBPF (float).
 * Must be called after buffer_import and before rbpf_ksc_step.
 */
void mmpf_sync_parameters(RBPF_Extended *ext);

/**
 * @brief Update Storvik stats for a single hypothesis
 *
 * Called from mmpf_step when gated learning is enabled.
 */
void mmpf_update_storvik_for_hypothesis(MMPF_ROCKS *mmpf, int k, int resampled);

#endif /* MMPF_INTERNAL_H */
