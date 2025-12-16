/**
 * @file rbpf_silverman.c
 * @brief Silverman's Rule of Thumb for Kernel Density Estimation Bandwidth
 *
 * Implementation of Silverman (1986) bandwidth selection for particle filter
 * regularization. Provides density-adaptive jitter that prevents particle
 * collapse without over-smoothing.
 */

#include "rbpf_silverman.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

#ifdef __AVX2__
#include <immintrin.h>
#endif

/*═══════════════════════════════════════════════════════════════════════════
 * QUICKSELECT FOR QUARTILES
 *
 * O(n) expected time algorithm to find k-th smallest element.
 * Used to compute Q1 (25th percentile) and Q3 (75th percentile) for IQR.
 *═══════════════════════════════════════════════════════════════════════════*/

static inline void swap_double(double *a, double *b) {
    double t = *a;
    *a = *b;
    *b = t;
}

/**
 * @brief Partition array around pivot (Hoare partition scheme)
 * @return Final position of pivot
 */
static int partition(double *arr, int lo, int hi) {
    /* Use median-of-three for pivot selection (avoids worst case) */
    int mid = lo + (hi - lo) / 2;
    
    /* Sort lo, mid, hi */
    if (arr[mid] < arr[lo]) swap_double(&arr[lo], &arr[mid]);
    if (arr[hi] < arr[lo]) swap_double(&arr[lo], &arr[hi]);
    if (arr[hi] < arr[mid]) swap_double(&arr[mid], &arr[hi]);
    
    /* Use mid as pivot, move to hi-1 */
    swap_double(&arr[mid], &arr[hi - 1]);
    double pivot = arr[hi - 1];
    
    int i = lo;
    int j = hi - 1;
    
    for (;;) {
        while (arr[++i] < pivot);
        while (arr[--j] > pivot);
        if (i >= j) break;
        swap_double(&arr[i], &arr[j]);
    }
    
    swap_double(&arr[i], &arr[hi - 1]);
    return i;
}

/**
 * @brief Find k-th smallest element (0-indexed)
 * @param arr   Array to search (WILL BE MODIFIED)
 * @param n     Array length
 * @param k     Index of element to find (0 = smallest)
 * @return      Value of k-th smallest element
 */
static double quickselect(double *arr, int n, int k) {
    int lo = 0, hi = n - 1;
    
    while (lo < hi) {
        int pivot_idx = partition(arr, lo, hi);
        
        if (pivot_idx == k) {
            return arr[k];
        } else if (pivot_idx > k) {
            hi = pivot_idx - 1;
        } else {
            lo = pivot_idx + 1;
        }
    }
    
    return arr[lo];
}

/*═══════════════════════════════════════════════════════════════════════════
 * STATISTICS HELPERS
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Compute mean and variance in single pass (Welford's algorithm)
 */
static void compute_mean_var(const double *x, int n, double *mean_out, double *var_out) {
    if (n <= 0) {
        *mean_out = 0.0;
        *var_out = 0.0;
        return;
    }
    
    double mean = 0.0;
    double M2 = 0.0;
    
#ifdef __AVX2__
    /* AVX2 vectorized mean computation */
    if (n >= 8) {
        __m256d sum_vec = _mm256_setzero_pd();
        int i = 0;
        
        for (; i + 4 <= n; i += 4) {
            __m256d v = _mm256_loadu_pd(&x[i]);
            sum_vec = _mm256_add_pd(sum_vec, v);
        }
        
        /* Horizontal sum */
        __m128d lo = _mm256_castpd256_pd128(sum_vec);
        __m128d hi = _mm256_extractf128_pd(sum_vec, 1);
        lo = _mm_add_pd(lo, hi);
        lo = _mm_add_pd(lo, _mm_shuffle_pd(lo, lo, 1));
        mean = _mm_cvtsd_f64(lo);
        
        /* Add remaining elements */
        for (; i < n; i++) {
            mean += x[i];
        }
        mean /= n;
        
        /* Now compute variance (second pass for numerical stability) */
        __m256d mean_vec = _mm256_set1_pd(mean);
        __m256d m2_vec = _mm256_setzero_pd();
        
        for (i = 0; i + 4 <= n; i += 4) {
            __m256d v = _mm256_loadu_pd(&x[i]);
            __m256d diff = _mm256_sub_pd(v, mean_vec);
            m2_vec = _mm256_fmadd_pd(diff, diff, m2_vec);
        }
        
        lo = _mm256_castpd256_pd128(m2_vec);
        hi = _mm256_extractf128_pd(m2_vec, 1);
        lo = _mm_add_pd(lo, hi);
        lo = _mm_add_pd(lo, _mm_shuffle_pd(lo, lo, 1));
        M2 = _mm_cvtsd_f64(lo);
        
        for (; i < n; i++) {
            double diff = x[i] - mean;
            M2 += diff * diff;
        }
    } else
#endif
    {
        /* Scalar Welford's algorithm */
        for (int i = 0; i < n; i++) {
            double delta = x[i] - mean;
            mean += delta / (i + 1);
            double delta2 = x[i] - mean;
            M2 += delta * delta2;
        }
    }
    
    *mean_out = mean;
    *var_out = (n > 1) ? M2 / (n - 1) : 0.0;  /* Bessel's correction */
}

/*═══════════════════════════════════════════════════════════════════════════
 * SILVERMAN BANDWIDTH COMPUTATION
 *═══════════════════════════════════════════════════════════════════════════*/

double rbpf_silverman_bandwidth(const double *x, int n, double *scratch) {
    if (n <= 1) return 0.0;
    
    /* 1. Compute standard deviation */
    double mean, var;
    compute_mean_var(x, n, &mean, &var);
    double sigma = sqrt(var);
    
    /* 2. Compute IQR using quickselect */
    /* Copy to scratch buffer (quickselect modifies array) */
    memcpy(scratch, x, n * sizeof(double));
    
    int q1_idx = n / 4;          /* 25th percentile index */
    int q3_idx = (3 * n) / 4;    /* 75th percentile index */
    
    double q1 = quickselect(scratch, n, q1_idx);
    
    /* Re-copy since quickselect partially sorts */
    memcpy(scratch, x, n * sizeof(double));
    double q3 = quickselect(scratch, n, q3_idx);
    
    double iqr = q3 - q1;
    double iqr_scaled = iqr / 1.34;  /* For Gaussian: IQR/1.34 ≈ σ */
    
    /* 3. Silverman's Rule: h = 0.9 × min(σ, IQR/1.34) × N^(-1/5) */
    double scale = (sigma < iqr_scaled) ? sigma : iqr_scaled;
    
    /* Handle edge case: if all particles identical, use small fixed bandwidth */
    if (scale < 1e-10) {
        scale = 0.01;  /* Fallback: 1% in log-vol space */
    }
    
    double n_factor = pow((double)n, -0.2);  /* N^(-1/5) */
    double h = 0.9 * scale * n_factor;
    
    return h;
}

float rbpf_silverman_bandwidth_f(const float *x, int n, float *scratch) {
    /* Convert to double, compute, convert back */
    /* For performance-critical code, implement native float version */
    if (n <= 1) return 0.0f;
    
    double *x_d = (double *)malloc(n * sizeof(double));
    double *s_d = (double *)malloc(n * sizeof(double));
    
    for (int i = 0; i < n; i++) x_d[i] = (double)x[i];
    
    double h = rbpf_silverman_bandwidth(x_d, n, s_d);
    
    free(x_d);
    free(s_d);
    
    return (float)h;
}

double rbpf_silverman_bandwidth_simple(const double *x, int n) {
    if (n <= 1) return 0.0;
    
    /* Compute standard deviation only (no IQR) */
    double mean, var;
    compute_mean_var(x, n, &mean, &var);
    double sigma = sqrt(var);
    
    /* Handle edge case */
    if (sigma < 1e-10) {
        sigma = 0.01;
    }
    
    /* Simplified rule: h = 1.06 × σ × N^(-1/5) */
    double n_factor = pow((double)n, -0.2);
    double h = 1.06 * sigma * n_factor;
    
    return h;
}

/*═══════════════════════════════════════════════════════════════════════════
 * ADAPTIVE BANDWIDTH WITH ESS SCALING
 *═══════════════════════════════════════════════════════════════════════════*/

void rbpf_bandwidth_config_defaults(RBPF_BandwidthConfig *cfg) {
    cfg->use_silverman = 1;
    cfg->fixed_bandwidth = 0.02;   /* 2% jitter in log-vol space */
    cfg->min_bandwidth = 0.001;    /* Floor: 0.1% */
    cfg->max_bandwidth = 0.5;      /* Ceiling: 50% */
    cfg->ess_scale_min = 0.5;      /* Low ESS → up to 2x bandwidth */
    cfg->ess_scale_max = 1.5;      /* High ESS → down to 0.67x bandwidth */
}

double rbpf_adaptive_bandwidth(const double *x, int n, double ess,
                               const RBPF_BandwidthConfig *cfg,
                               double *scratch) {
    double h;
    
    /* 1. Compute base bandwidth */
    if (cfg->use_silverman && scratch != NULL) {
        h = rbpf_silverman_bandwidth(x, n, scratch);
    } else if (cfg->use_silverman) {
        /* No scratch buffer: use simplified version */
        h = rbpf_silverman_bandwidth_simple(x, n);
    } else {
        h = cfg->fixed_bandwidth;
    }
    
    /* 2. Scale by ESS ratio */
    /* Low ESS (particles collapsed) → need more jitter */
    /* High ESS (particles diverse) → need less jitter */
    double ess_ratio = ess / (double)n;
    
    /* Linear interpolation: ess_ratio=0 → scale_max, ess_ratio=1 → scale_min */
    double scale = cfg->ess_scale_max - 
                   (cfg->ess_scale_max - cfg->ess_scale_min) * ess_ratio;
    
    h *= scale;
    
    /* 3. Clamp to bounds */
    if (h < cfg->min_bandwidth) h = cfg->min_bandwidth;
    if (h > cfg->max_bandwidth) h = cfg->max_bandwidth;
    
    return h;
}

/*═══════════════════════════════════════════════════════════════════════════
 * SINGLE-PRECISION SUPPORT FOR rbpf_real_t BUILDS
 *
 * If RBPF uses float internally, provide optimized float path.
 *═══════════════════════════════════════════════════════════════════════════*/

#ifdef RBPF_USE_FLOAT

/* Swap helper for float */
static inline void swap_float(float *a, float *b) {
    float t = *a;
    *a = *b;
    *b = t;
}

static int partition_f(float *arr, int lo, int hi) {
    int mid = lo + (hi - lo) / 2;
    
    if (arr[mid] < arr[lo]) swap_float(&arr[lo], &arr[mid]);
    if (arr[hi] < arr[lo]) swap_float(&arr[lo], &arr[hi]);
    if (arr[hi] < arr[mid]) swap_float(&arr[mid], &arr[hi]);
    
    swap_float(&arr[mid], &arr[hi - 1]);
    float pivot = arr[hi - 1];
    
    int i = lo;
    int j = hi - 1;
    
    for (;;) {
        while (arr[++i] < pivot);
        while (arr[--j] > pivot);
        if (i >= j) break;
        swap_float(&arr[i], &arr[j]);
    }
    
    swap_float(&arr[i], &arr[hi - 1]);
    return i;
}

static float quickselect_f(float *arr, int n, int k) {
    int lo = 0, hi = n - 1;
    
    while (lo < hi) {
        int pivot_idx = partition_f(arr, lo, hi);
        
        if (pivot_idx == k) {
            return arr[k];
        } else if (pivot_idx > k) {
            hi = pivot_idx - 1;
        } else {
            lo = pivot_idx + 1;
        }
    }
    
    return arr[lo];
}

float rbpf_silverman_bandwidth_native_f(const float *x, int n, float *scratch) {
    if (n <= 1) return 0.0f;
    
    /* Compute mean and variance */
    float mean = 0.0f, M2 = 0.0f;
    for (int i = 0; i < n; i++) {
        float delta = x[i] - mean;
        mean += delta / (i + 1);
        float delta2 = x[i] - mean;
        M2 += delta * delta2;
    }
    float var = (n > 1) ? M2 / (n - 1) : 0.0f;
    float sigma = sqrtf(var);
    
    /* Compute IQR */
    memcpy(scratch, x, n * sizeof(float));
    int q1_idx = n / 4;
    int q3_idx = (3 * n) / 4;
    
    float q1 = quickselect_f(scratch, n, q1_idx);
    memcpy(scratch, x, n * sizeof(float));
    float q3 = quickselect_f(scratch, n, q3_idx);
    
    float iqr = q3 - q1;
    float iqr_scaled = iqr / 1.34f;
    
    float scale = (sigma < iqr_scaled) ? sigma : iqr_scaled;
    if (scale < 1e-6f) scale = 0.01f;
    
    float n_factor = powf((float)n, -0.2f);
    float h = 0.9f * scale * n_factor;
    
    return h;
}

#endif /* RBPF_USE_FLOAT */
