/**
 * @file paris_fast.c
 * @brief High-performance PARIS backward smoother implementation
 *
 * Optimizations implemented:
 *   1. AVX2 vectorized backward kernel (8 floats per instruction)
 *   2. SoA memory layout for cache efficiency
 *   3. Walker's Alias Method for O(1) sampling
 *   4. Single precision (2x throughput vs double)
 *   5. OpenMP parallel backward pass
 */

#include "paris_fast.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __AVX2__
#include <immintrin.h>
#endif

/*═══════════════════════════════════════════════════════════════════════════════
 * CONSTANTS
 *═══════════════════════════════════════════════════════════════════════════════*/

#define PARIS_EPS 1e-10f
#define PARIS_LOG_EPS -23.0259f

/*═══════════════════════════════════════════════════════════════════════════════
 * RNG (xoroshiro128+)
 *═══════════════════════════════════════════════════════════════════════════════*/

static inline uint64_t rotl(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

static inline uint64_t xoro_next(uint64_t *s) {
    const uint64_t s0 = s[0];
    uint64_t s1 = s[1];
    const uint64_t result = s0 + s1;
    s1 ^= s0;
    s[0] = rotl(s0, 24) ^ s1 ^ (s1 << 16);
    s[1] = rotl(s1, 37);
    return result;
}

static inline float rand_uniform_f(uint64_t *s) {
    return (xoro_next(s) >> 40) * 0x1.0p-24f;
}

static inline int rand_int(uint64_t *s, int n) {
    return (int)(rand_uniform_f(s) * n);
}

/*═══════════════════════════════════════════════════════════════════════════════
 * WALKER'S ALIAS METHOD
 *═══════════════════════════════════════════════════════════════════════════════*/

void alias_build(AliasTable *table, const paris_real *weights, int N)
{
    table->N = N;
    
    /* Normalize weights */
    paris_real sum = 0;
    for (int i = 0; i < N; i++) {
        sum += weights[i];
    }
    paris_real scale = (paris_real)N / sum;
    
    /* Initialize prob and alias */
    paris_real scaled[PARIS_MAX_PARTICLES];
    for (int i = 0; i < N; i++) {
        scaled[i] = weights[i] * scale;
        table->prob[i] = scaled[i];
        table->alias[i] = i;
    }
    
    /* Build small and large stacks */
    int small[PARIS_MAX_PARTICLES];
    int large[PARIS_MAX_PARTICLES];
    int n_small = 0, n_large = 0;
    
    for (int i = 0; i < N; i++) {
        if (scaled[i] < 1.0f) {
            small[n_small++] = i;
        } else {
            large[n_large++] = i;
        }
    }
    
    /* Build alias table */
    while (n_small > 0 && n_large > 0) {
        int s = small[--n_small];
        int l = large[--n_large];
        
        table->prob[s] = scaled[s];
        table->alias[s] = l;
        
        scaled[l] = (scaled[l] + scaled[s]) - 1.0f;
        
        if (scaled[l] < 1.0f) {
            small[n_small++] = l;
        } else {
            large[n_large++] = l;
        }
    }
    
    /* Handle remaining (numerical precision) */
    while (n_large > 0) {
        table->prob[large[--n_large]] = 1.0f;
    }
    while (n_small > 0) {
        table->prob[small[--n_small]] = 1.0f;
    }
}

int alias_sample(const AliasTable *table, uint64_t *rng)
{
    int i = rand_int(rng, table->N);
    paris_real u = rand_uniform_f(rng);
    return (u < table->prob[i]) ? i : table->alias[i];
}

/*═══════════════════════════════════════════════════════════════════════════════
 * MEMORY ALLOCATION
 *═══════════════════════════════════════════════════════════════════════════════*/

static void *aligned_malloc(size_t size, size_t align) {
    void *ptr = NULL;
    if (posix_memalign(&ptr, align, size) != 0) {
        return NULL;
    }
    return ptr;
}

PARISState *paris_alloc(int N, int T, int K, uint64_t seed)
{
    PARISState *state = calloc(1, sizeof(PARISState));
    if (!state) return NULL;
    
    state->N = N;
    state->T = T;
    state->K = K;
    
    size_t NT = (size_t)N * T;
    size_t align = PARIS_ALIGN;
    
    /* Allocate SoA arrays */
    state->regimes = aligned_malloc(NT * sizeof(int), align);
    state->h = aligned_malloc(NT * sizeof(paris_real), align);
    state->weights = aligned_malloc(NT * sizeof(paris_real), align);
    state->log_weights = aligned_malloc(NT * sizeof(paris_real), align);
    state->ancestors = aligned_malloc(NT * sizeof(int), align);
    state->smoothed = aligned_malloc(NT * sizeof(int), align);
    
    state->log_trans = aligned_malloc(K * K * sizeof(paris_real), align);
    state->mu_vol = aligned_malloc(K * sizeof(paris_real), align);
    state->observations = aligned_malloc(T * sizeof(paris_real), align);
    
    state->bw_weights_workspace = aligned_malloc(N * sizeof(paris_real), align);
    
    /* Initialize RNG */
    state->rng_state[0] = seed;
    state->rng_state[1] = seed ^ 0x9E3779B97F4A7C15ULL;
    
    return state;
}

void paris_free(PARISState *state)
{
    if (!state) return;
    
    free(state->regimes);
    free(state->h);
    free(state->weights);
    free(state->log_weights);
    free(state->ancestors);
    free(state->smoothed);
    free(state->log_trans);
    free(state->mu_vol);
    free(state->observations);
    free(state->bw_weights_workspace);
    free(state);
}

/*═══════════════════════════════════════════════════════════════════════════════
 * MODEL SETUP
 *═══════════════════════════════════════════════════════════════════════════════*/

void paris_set_model(PARISState *state,
                     const double *trans,
                     const double *mu_vol,
                     double phi,
                     double sigma_h)
{
    if (!state) return;
    
    int K = state->K;
    
    /* Convert transition matrix to log and float */
    for (int i = 0; i < K * K; i++) {
        state->log_trans[i] = (paris_real)log(trans[i] + PARIS_EPS);
    }
    
    for (int i = 0; i < K; i++) {
        state->mu_vol[i] = (paris_real)mu_vol[i];
    }
    
    state->phi = (paris_real)phi;
    state->sigma_h = (paris_real)sigma_h;
    state->inv_sigma_h_sq = 1.0f / (state->sigma_h * state->sigma_h);
}

void paris_load_particles(PARISState *state,
                          const int *regimes,
                          const double *h,
                          const double *weights,
                          const int *ancestors,
                          int T)
{
    if (!state) return;
    
    state->T = T;
    int N = state->N;
    
    /* Copy and convert to SoA float layout */
    for (int t = 0; t < T; t++) {
        for (int n = 0; n < N; n++) {
            int idx = t * N + n;
            state->regimes[idx] = regimes[idx];
            state->h[idx] = (paris_real)h[idx];
            state->weights[idx] = (paris_real)weights[idx];
            state->log_weights[idx] = logf(state->weights[idx] + PARIS_EPS);
            state->ancestors[idx] = ancestors[idx];
        }
    }
}

/*═══════════════════════════════════════════════════════════════════════════════
 * VECTORIZED BACKWARD KERNEL (AVX2)
 *═══════════════════════════════════════════════════════════════════════════════*/

#ifdef __AVX2__

/**
 * Fast exp approximation for float (Schraudolph's method, improved)
 * Good for ~12 bits of precision, sufficient for probability ratios
 */
static inline __m256 fast_exp_avx(__m256 x) {
    /* exp(x) ≈ 2^(x/ln2) using IEEE 754 float trick */
    const __m256 log2e = _mm256_set1_ps(1.442695040f);
    const __m256 c1 = _mm256_set1_ps(12102203.0f);  /* 2^23 / ln2 */
    const __m256 c2 = _mm256_set1_ps(1065353216.0f); /* 127 * 2^23 */
    
    __m256 t = _mm256_fmadd_ps(x, c1, c2);
    return _mm256_castsi256_ps(_mm256_cvtps_epi32(t));
}

void paris_compute_bw_weights_avx2(const PARISState *state,
                                    int t,
                                    int regime_next,
                                    paris_real h_next,
                                    paris_real *bw_weights)
{
    const int N = state->N;
    const int K = state->K;
    const paris_real *log_trans_row = &state->log_trans[0];  /* Will index by regime */
    const paris_real inv_sigma_h_sq = state->inv_sigma_h_sq;
    const paris_real *mu_vol = state->mu_vol;
    const paris_real phi = state->phi;
    
    /* Pointers to time t data */
    const int *regimes_t = &state->regimes[t * N];
    const paris_real *h_t = &state->h[t * N];
    const paris_real *log_w_t = &state->log_weights[t * N];
    
    /* Broadcast constants */
    __m256 h_next_vec = _mm256_set1_ps(h_next);
    __m256 phi_vec = _mm256_set1_ps(phi);
    __m256 inv_var_vec = _mm256_set1_ps(-0.5f * inv_sigma_h_sq);
    
    /* Process 8 particles at a time */
    int n = 0;
    for (; n + 8 <= N; n += 8) {
        /* Load log weights */
        __m256 log_w = _mm256_loadu_ps(&log_w_t[n]);
        
        /* Load h values */
        __m256 h_m = _mm256_loadu_ps(&h_t[n]);
        
        /* Compute h transition log-prob for each particle */
        /* log P(h_next | h_m, regime_next) = -0.5 * (h_next - mean)² / σ² */
        /* mean = μ_k + φ(h_m - μ_k) where k = regime_next */
        paris_real mu_k = mu_vol[regime_next];
        __m256 mu_k_vec = _mm256_set1_ps(mu_k);
        
        /* mean = mu_k + phi * (h_m - mu_k) */
        __m256 h_diff = _mm256_sub_ps(h_m, mu_k_vec);
        __m256 mean = _mm256_fmadd_ps(phi_vec, h_diff, mu_k_vec);
        
        /* diff = h_next - mean */
        __m256 diff = _mm256_sub_ps(h_next_vec, mean);
        
        /* log_h_trans = -0.5 * diff² / σ² */
        __m256 diff_sq = _mm256_mul_ps(diff, diff);
        __m256 log_h_trans = _mm256_mul_ps(diff_sq, inv_var_vec);
        
        /* Load regime transition log-probs (scalar, must gather) */
        __m256 log_regime_trans = _mm256_setzero_ps();
        float log_trans_buf[8];
        for (int i = 0; i < 8; i++) {
            int regime_m = regimes_t[n + i];
            log_trans_buf[i] = log_trans_row[regime_m * K + regime_next];
        }
        log_regime_trans = _mm256_loadu_ps(log_trans_buf);
        
        /* Total log weight = log_w + log_trans + log_h_trans */
        __m256 log_bw = _mm256_add_ps(log_w, log_regime_trans);
        log_bw = _mm256_add_ps(log_bw, log_h_trans);
        
        _mm256_storeu_ps(&bw_weights[n], log_bw);
    }
    
    /* Scalar tail */
    for (; n < N; n++) {
        int regime_m = regimes_t[n];
        paris_real h_m = h_t[n];
        paris_real mu_k = mu_vol[regime_next];
        
        paris_real mean = mu_k + phi * (h_m - mu_k);
        paris_real diff = h_next - mean;
        paris_real log_h_trans = -0.5f * diff * diff * inv_sigma_h_sq;
        
        bw_weights[n] = log_w_t[n] + log_trans_row[regime_m * K + regime_next] + log_h_trans;
    }
}

void paris_logsumexp_normalize_avx2(const paris_real *log_weights,
                                     int N,
                                     paris_real *weights)
{
    /* Find max */
    __m256 max_vec = _mm256_set1_ps(-1e30f);
    int n = 0;
    
    for (; n + 8 <= N; n += 8) {
        __m256 lw = _mm256_loadu_ps(&log_weights[n]);
        max_vec = _mm256_max_ps(max_vec, lw);
    }
    
    /* Horizontal max */
    __m128 hi = _mm256_extractf128_ps(max_vec, 1);
    __m128 lo = _mm256_castps256_ps128(max_vec);
    __m128 max4 = _mm_max_ps(hi, lo);
    max4 = _mm_max_ps(max4, _mm_shuffle_ps(max4, max4, _MM_SHUFFLE(2, 3, 0, 1)));
    max4 = _mm_max_ps(max4, _mm_shuffle_ps(max4, max4, _MM_SHUFFLE(1, 0, 3, 2)));
    float max_val = _mm_cvtss_f32(max4);
    
    /* Scalar tail for max */
    for (; n < N; n++) {
        if (log_weights[n] > max_val) max_val = log_weights[n];
    }
    
    /* Subtract max and exp, accumulate sum */
    __m256 max_broadcast = _mm256_set1_ps(max_val);
    __m256 sum_vec = _mm256_setzero_ps();
    
    n = 0;
    for (; n + 8 <= N; n += 8) {
        __m256 lw = _mm256_loadu_ps(&log_weights[n]);
        __m256 shifted = _mm256_sub_ps(lw, max_broadcast);
        __m256 w = fast_exp_avx(shifted);
        _mm256_storeu_ps(&weights[n], w);
        sum_vec = _mm256_add_ps(sum_vec, w);
    }
    
    /* Horizontal sum */
    hi = _mm256_extractf128_ps(sum_vec, 1);
    lo = _mm256_castps256_ps128(sum_vec);
    __m128 sum4 = _mm_add_ps(hi, lo);
    sum4 = _mm_add_ps(sum4, _mm_shuffle_ps(sum4, sum4, _MM_SHUFFLE(2, 3, 0, 1)));
    sum4 = _mm_add_ps(sum4, _mm_shuffle_ps(sum4, sum4, _MM_SHUFFLE(1, 0, 3, 2)));
    float sum = _mm_cvtss_f32(sum4);
    
    /* Scalar tail */
    for (; n < N; n++) {
        weights[n] = expf(log_weights[n] - max_val);
        sum += weights[n];
    }
    
    /* Normalize */
    __m256 inv_sum = _mm256_set1_ps(1.0f / sum);
    n = 0;
    for (; n + 8 <= N; n += 8) {
        __m256 w = _mm256_loadu_ps(&weights[n]);
        w = _mm256_mul_ps(w, inv_sum);
        _mm256_storeu_ps(&weights[n], w);
    }
    for (; n < N; n++) {
        weights[n] /= sum;
    }
}

#else /* Scalar fallback */

void paris_compute_bw_weights_avx2(const PARISState *state,
                                    int t,
                                    int regime_next,
                                    paris_real h_next,
                                    paris_real *bw_weights)
{
    const int N = state->N;
    const int K = state->K;
    const paris_real inv_sigma_h_sq = state->inv_sigma_h_sq;
    const paris_real *mu_vol = state->mu_vol;
    const paris_real phi = state->phi;
    
    const int *regimes_t = &state->regimes[t * N];
    const paris_real *h_t = &state->h[t * N];
    const paris_real *log_w_t = &state->log_weights[t * N];
    
    paris_real mu_k = mu_vol[regime_next];
    
    for (int n = 0; n < N; n++) {
        int regime_m = regimes_t[n];
        paris_real h_m = h_t[n];
        
        paris_real mean = mu_k + phi * (h_m - mu_k);
        paris_real diff = h_next - mean;
        paris_real log_h_trans = -0.5f * diff * diff * inv_sigma_h_sq;
        
        bw_weights[n] = log_w_t[n] + 
                        state->log_trans[regime_m * K + regime_next] + 
                        log_h_trans;
    }
}

void paris_logsumexp_normalize_avx2(const paris_real *log_weights,
                                     int N,
                                     paris_real *weights)
{
    /* Find max */
    paris_real max_val = log_weights[0];
    for (int n = 1; n < N; n++) {
        if (log_weights[n] > max_val) max_val = log_weights[n];
    }
    
    /* Exp and sum */
    paris_real sum = 0;
    for (int n = 0; n < N; n++) {
        weights[n] = expf(log_weights[n] - max_val);
        sum += weights[n];
    }
    
    /* Normalize */
    paris_real inv_sum = 1.0f / sum;
    for (int n = 0; n < N; n++) {
        weights[n] *= inv_sum;
    }
}

#endif /* __AVX2__ */

/*═══════════════════════════════════════════════════════════════════════════════
 * PARIS BACKWARD SMOOTHING (OPTIMIZED)
 *═══════════════════════════════════════════════════════════════════════════════*/

void paris_backward_smooth_fast(PARISState *state)
{
    if (!state || state->T < 2) return;
    
    const int N = state->N;
    const int T = state->T;
    
    /* Initialize at final time: smoothed = self */
    for (int n = 0; n < N; n++) {
        state->smoothed[(T-1) * N + n] = n;
    }
    
    /* Backward pass with OpenMP parallelism */
    for (int t = T - 2; t >= 0; t--) {
        
        #ifdef _OPENMP
        #pragma omp parallel
        {
            /* Each thread has its own RNG state and workspace */
            uint64_t local_rng[2];
            local_rng[0] = state->rng_state[0] ^ (uint64_t)(t * 1000 + omp_get_thread_num());
            local_rng[1] = state->rng_state[1] ^ (uint64_t)(t * 1000 + omp_get_thread_num());
            
            paris_real *bw_weights = aligned_alloc(PARIS_ALIGN, N * sizeof(paris_real));
            paris_real *norm_weights = aligned_alloc(PARIS_ALIGN, N * sizeof(paris_real));
            
            #pragma omp for schedule(dynamic, 8)
            for (int n = 0; n < N; n++) {
                /* Get smoothed state at t+1 */
                int idx_next = state->smoothed[(t+1) * N + n];
                int regime_next = state->regimes[(t+1) * N + idx_next];
                paris_real h_next = state->h[(t+1) * N + idx_next];
                
                /* Compute backward weights (vectorized) */
                paris_compute_bw_weights_avx2(state, t, regime_next, h_next, bw_weights);
                
                /* Normalize (vectorized) */
                paris_logsumexp_normalize_avx2(bw_weights, N, norm_weights);
                
                /* Build alias table and sample (O(N) setup, O(1) sample) */
                AliasTable alias;
                alias_build(&alias, norm_weights, N);
                state->smoothed[t * N + n] = alias_sample(&alias, local_rng);
            }
            
            free(bw_weights);
            free(norm_weights);
        }
        #else
        /* Sequential version */
        paris_real *bw_weights = state->bw_weights_workspace;
        paris_real norm_weights[PARIS_MAX_PARTICLES];
        
        for (int n = 0; n < N; n++) {
            int idx_next = state->smoothed[(t+1) * N + n];
            int regime_next = state->regimes[(t+1) * N + idx_next];
            paris_real h_next = state->h[(t+1) * N + idx_next];
            
            paris_compute_bw_weights_avx2(state, t, regime_next, h_next, bw_weights);
            paris_logsumexp_normalize_avx2(bw_weights, N, norm_weights);
            
            AliasTable alias;
            alias_build(&alias, norm_weights, N);
            state->smoothed[t * N + n] = alias_sample(&alias, state->rng_state);
        }
        #endif
    }
}

void paris_get_smoothed(const PARISState *state,
                        int t,
                        int *regimes,
                        double *h)
{
    if (!state || t < 0 || t >= state->T) return;
    
    const int N = state->N;
    
    for (int n = 0; n < N; n++) {
        int idx = state->smoothed[t * N + n];
        if (regimes) regimes[n] = state->regimes[t * N + idx];
        if (h) h[n] = (double)state->h[t * N + idx];
    }
}
