/**
 * @file bocpd.c
 * @brief Bayesian Online Changepoint Detection - Streamlined for MMPF
 */

#define _USE_MATH_DEFINES
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
#include "bocpd.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

/*═══════════════════════════════════════════════════════════════════════════
 * INTERLEAVED BLOCK ACCESSORS
 *═══════════════════════════════════════════════════════════════════════════*/

static inline double iblk_get(const double *buf, size_t idx, size_t field_offset) {
    size_t block = idx / 4;
    size_t lane = idx & 3;
    return buf[block * BOCPD_IBLK_DOUBLES + field_offset / 8 + lane];
}

static inline void iblk_set(double *buf, size_t idx, size_t field_offset, double val) {
    size_t block = idx / 4;
    size_t lane = idx & 3;
    buf[block * BOCPD_IBLK_DOUBLES + field_offset / 8 + lane] = val;
}

#define IBLK_GET_MU(buf, i)      iblk_get(buf, i, BOCPD_IBLK_MU)
#define IBLK_GET_C1(buf, i)      iblk_get(buf, i, BOCPD_IBLK_C1)
#define IBLK_GET_C2(buf, i)      iblk_get(buf, i, BOCPD_IBLK_C2)
#define IBLK_GET_INV_SSN(buf, i) iblk_get(buf, i, BOCPD_IBLK_INV_SSN)
#define IBLK_GET_KAPPA(buf, i)   iblk_get(buf, i, BOCPD_IBLK_KAPPA)
#define IBLK_GET_ALPHA(buf, i)   iblk_get(buf, i, BOCPD_IBLK_ALPHA)
#define IBLK_GET_BETA(buf, i)    iblk_get(buf, i, BOCPD_IBLK_BETA)
#define IBLK_GET_SS_N(buf, i)    iblk_get(buf, i, BOCPD_IBLK_SS_N)

#define IBLK_SET_MU(buf, i, v)      iblk_set(buf, i, BOCPD_IBLK_MU, v)
#define IBLK_SET_C1(buf, i, v)      iblk_set(buf, i, BOCPD_IBLK_C1, v)
#define IBLK_SET_C2(buf, i, v)      iblk_set(buf, i, BOCPD_IBLK_C2, v)
#define IBLK_SET_INV_SSN(buf, i, v) iblk_set(buf, i, BOCPD_IBLK_INV_SSN, v)
#define IBLK_SET_KAPPA(buf, i, v)   iblk_set(buf, i, BOCPD_IBLK_KAPPA, v)
#define IBLK_SET_ALPHA(buf, i, v)   iblk_set(buf, i, BOCPD_IBLK_ALPHA, v)
#define IBLK_SET_BETA(buf, i, v)    iblk_set(buf, i, BOCPD_IBLK_BETA, v)
#define IBLK_SET_SS_N(buf, i, v)    iblk_set(buf, i, BOCPD_IBLK_SS_N, v)

/*═══════════════════════════════════════════════════════════════════════════
 * FAST MATH
 *═══════════════════════════════════════════════════════════════════════════*/

static inline double fast_log(double x) {
    union { double d; uint64_t u; } u = {.d = x};
    int64_t e = (int64_t)((u.u >> 52) & 0x7FF) - 1023;
    u.u = (u.u & 0x000FFFFFFFFFFFFFULL) | 0x3FF0000000000000ULL;
    double m = u.d;
    double t = (m - 1.0) / (m + 1.0);
    double t2 = t * t;
    double poly = 1.0 + t2 * (0.3333333333333333 + t2 * (0.2 + 
                       t2 * (0.1428571428571429 + t2 * 0.1111111111111111)));
    return (double)e * 0.6931471805599453 + 2.0 * t * poly;
}

static inline __m256d fast_log_avx2(__m256d x) {
    const __m256d one = _mm256_set1_pd(1.0);
    const __m256d two = _mm256_set1_pd(2.0);
    const __m256d ln2 = _mm256_set1_pd(0.6931471805599453);
    const __m256d c3 = _mm256_set1_pd(0.3333333333333333);
    const __m256d c5 = _mm256_set1_pd(0.2);
    const __m256d c7 = _mm256_set1_pd(0.1428571428571429);
    const __m256d c9 = _mm256_set1_pd(0.1111111111111111);
    const __m256i exp_mask = _mm256_set1_epi64x(0x7FF0000000000000ULL);
    const __m256i mantissa_mask = _mm256_set1_epi64x(0x000FFFFFFFFFFFFFULL);
    const __m256i exp_bias_bits = _mm256_set1_epi64x(0x3FF0000000000000ULL);
    const __m256i magic_i = _mm256_set1_epi64x(0x4330000000000000ULL);
    const __m256d magic_d = _mm256_set1_pd(4503599627370496.0);
    const __m256d bias_1023 = _mm256_set1_pd(1023.0);

    __m256i xi = _mm256_castpd_si256(x);
    __m256i exp_bits = _mm256_srli_epi64(_mm256_and_si256(xi, exp_mask), 52);
    __m256i exp_biased = _mm256_or_si256(exp_bits, magic_i);
    __m256d exp_double = _mm256_sub_pd(_mm256_castsi256_pd(exp_biased), magic_d);
    __m256d e = _mm256_sub_pd(exp_double, bias_1023);
    __m256i mi = _mm256_or_si256(_mm256_and_si256(xi, mantissa_mask), exp_bias_bits);
    __m256d m = _mm256_castsi256_pd(mi);
    __m256d num = _mm256_sub_pd(m, one);
    __m256d den = _mm256_add_pd(m, one);
    __m256d t = _mm256_div_pd(num, den);
    __m256d t2 = _mm256_mul_pd(t, t);
    __m256d poly = _mm256_fmadd_pd(t2, c9, c7);
    poly = _mm256_fmadd_pd(t2, poly, c5);
    poly = _mm256_fmadd_pd(t2, poly, c3);
    poly = _mm256_fmadd_pd(t2, poly, one);
    return _mm256_fmadd_pd(e, ln2, _mm256_mul_pd(two, _mm256_mul_pd(t, poly)));
}

static inline double fast_lgamma_stirling(double x) {
    const double half_ln2pi = 0.9189385332046727;
    const double s1 = 0.0833333333333333333;
    const double s2 = -0.00277777777777777778;
    const double s3 = 0.000793650793650793651;
    const double s4 = -0.000595238095238095238;
    const double s5 = 0.000841750841750841751;
    const double s6 = -0.00191752691752691753;
    double ln_x = fast_log(x);
    double base = (x - 0.5) * ln_x - x + half_ln2pi;
    double inv_x = 1.0 / x;
    double inv_x2 = inv_x * inv_x;
    double correction = s6;
    correction = correction * inv_x2 + s5;
    correction = correction * inv_x2 + s4;
    correction = correction * inv_x2 + s3;
    correction = correction * inv_x2 + s2;
    correction = correction * inv_x2 + s1;
    return base + correction * inv_x;
}

static inline double fast_lgamma_lanczos(double x) {
    const double half_ln2pi = 0.9189385332046727;
    const double g = 4.7421875;
    const double c0 = 1.000000000190015;
    const double c1 = 76.18009172947146;
    const double c2 = -86.50532032941677;
    const double c3 = 24.01409824083091;
    const double c4 = -1.231739572450155;
    const double c5 = 0.001208650973866179;
    double Ag = c0 + c1/x + c2/(x+1) + c3/(x+2) + c4/(x+3) + c5/(x+4);
    double t = x + g - 0.5;
    return half_ln2pi + (x - 0.5) * fast_log(t) - t + fast_log(Ag);
}

static inline double fast_lgamma(double x) {
    return (x > 40.0) ? fast_lgamma_stirling(x) : fast_lgamma_lanczos(x);
}

static inline __m256d lgamma_stirling_avx2(__m256d x) {
    const __m256d half = _mm256_set1_pd(0.5);
    const __m256d one = _mm256_set1_pd(1.0);
    const __m256d half_ln2pi = _mm256_set1_pd(0.9189385332046727);
    const __m256d s1 = _mm256_set1_pd(0.0833333333333333333);
    const __m256d s2 = _mm256_set1_pd(-0.00277777777777777778);
    const __m256d s3 = _mm256_set1_pd(0.000793650793650793651);
    const __m256d s4 = _mm256_set1_pd(-0.000595238095238095238);
    const __m256d s5 = _mm256_set1_pd(0.000841750841750841751);
    const __m256d s6 = _mm256_set1_pd(-0.00191752691752691753);

    __m256d ln_x = fast_log_avx2(x);
    __m256d base = _mm256_fmadd_pd(_mm256_sub_pd(x, half), ln_x,
                                   _mm256_sub_pd(half_ln2pi, x));
    __m256d inv_x = _mm256_div_pd(one, x);
    __m256d inv_x2 = _mm256_mul_pd(inv_x, inv_x);
    __m256d correction = s6;
    correction = _mm256_fmadd_pd(correction, inv_x2, s5);
    correction = _mm256_fmadd_pd(correction, inv_x2, s4);
    correction = _mm256_fmadd_pd(correction, inv_x2, s3);
    correction = _mm256_fmadd_pd(correction, inv_x2, s2);
    correction = _mm256_fmadd_pd(correction, inv_x2, s1);
    correction = _mm256_mul_pd(correction, inv_x);
    return _mm256_add_pd(base, correction);
}

static inline __m256d lgamma_lanczos_avx2(__m256d x) {
    const __m256d half = _mm256_set1_pd(0.5);
    const __m256d one = _mm256_set1_pd(1.0);
    const __m256d half_ln2pi = _mm256_set1_pd(0.9189385332046727);
    const __m256d g = _mm256_set1_pd(4.7421875);
    const __m256d c0 = _mm256_set1_pd(1.000000000190015);
    const __m256d c1 = _mm256_set1_pd(76.18009172947146);
    const __m256d c2 = _mm256_set1_pd(-86.50532032941677);
    const __m256d c3 = _mm256_set1_pd(24.01409824083091);
    const __m256d c4 = _mm256_set1_pd(-1.231739572450155);
    const __m256d c5 = _mm256_set1_pd(0.001208650973866179);

    __m256d xp0 = x;
    __m256d xp1 = _mm256_add_pd(x, one);
    __m256d xp2 = _mm256_add_pd(x, _mm256_set1_pd(2.0));
    __m256d xp3 = _mm256_add_pd(x, _mm256_set1_pd(3.0));
    __m256d xp4 = _mm256_add_pd(x, _mm256_set1_pd(4.0));

    __m256d Ag = c0;
    Ag = _mm256_add_pd(Ag, _mm256_div_pd(c1, xp0));
    Ag = _mm256_add_pd(Ag, _mm256_div_pd(c2, xp1));
    Ag = _mm256_add_pd(Ag, _mm256_div_pd(c3, xp2));
    Ag = _mm256_add_pd(Ag, _mm256_div_pd(c4, xp3));
    Ag = _mm256_add_pd(Ag, _mm256_div_pd(c5, xp4));

    __m256d t = _mm256_add_pd(x, _mm256_sub_pd(g, half));
    __m256d ln_t = fast_log_avx2(t);
    __m256d ln_Ag = fast_log_avx2(Ag);

    __m256d result = half_ln2pi;
    result = _mm256_fmadd_pd(_mm256_sub_pd(x, half), ln_t, result);
    result = _mm256_sub_pd(result, t);
    return _mm256_add_pd(result, ln_Ag);
}

static inline __m256d fast_lgamma_avx2(__m256d x) {
    const __m256d forty = _mm256_set1_pd(40.0);
    __m256d mask_large = _mm256_cmp_pd(x, forty, _CMP_GT_OQ);
    int mask_bits = _mm256_movemask_pd(mask_large);
    
    if (mask_bits == 0) return lgamma_lanczos_avx2(x);
    if (mask_bits == 0xF) return lgamma_stirling_avx2(x);
    
    __m256d result_small = lgamma_lanczos_avx2(x);
    __m256d result_large = lgamma_stirling_avx2(x);
    return _mm256_blendv_pd(result_small, result_large, mask_large);
}

/*═══════════════════════════════════════════════════════════════════════════
 * SHIFTED STORE FOR POSTERIOR UPDATE
 *═══════════════════════════════════════════════════════════════════════════*/

static inline void store_shifted_field(double *buf, size_t block_idx,
                                       size_t field_offset, __m256d vals) {
    __m256d rotated = _mm256_permute4x64_pd(vals, 0x93);
    double *block_k = buf + block_idx * BOCPD_IBLK_DOUBLES + field_offset / 8;
    double *block_k1 = buf + (block_idx + 1) * BOCPD_IBLK_DOUBLES + field_offset / 8;
    __m256d existing_k = _mm256_loadu_pd(block_k);
    __m256d existing_k1 = _mm256_loadu_pd(block_k1);
    __m256d merged_k = _mm256_blend_pd(existing_k, rotated, 0b1110);
    __m256d merged_k1 = _mm256_blend_pd(existing_k1, rotated, 0b0001);
    _mm256_storeu_pd(block_k, merged_k);
    _mm256_storeu_pd(block_k1, merged_k1);
}

/*═══════════════════════════════════════════════════════════════════════════
 * HAZARD FUNCTIONS
 *═══════════════════════════════════════════════════════════════════════════*/

int bocpd_hazard_init_constant(bocpd_hazard_t *h, double lambda, size_t max_run) {
    if (!h || lambda <= 0.0 || max_run < 16) return -1;
    
    memset(h, 0, sizeof(*h));
    h->type = HAZARD_CONSTANT;
    h->max_run_length = max_run;
    h->params.constant.lambda = lambda;
    
    h->h = (double *)malloc(max_run * sizeof(double));
    h->one_minus_h = (double *)malloc(max_run * sizeof(double));
    if (!h->h || !h->one_minus_h) {
        bocpd_hazard_free(h);
        return -1;
    }
    
    double hval = 1.0 / lambda;
    double omh = 1.0 - hval;
    for (size_t i = 0; i < max_run; i++) {
        h->h[i] = hval;
        h->one_minus_h[i] = omh;
    }
    return 0;
}

int bocpd_hazard_init_power_law(bocpd_hazard_t *h, double alpha, size_t max_run) {
    if (!h || alpha <= 0.0 || max_run < 16) return -1;
    
    memset(h, 0, sizeof(*h));
    h->type = HAZARD_POWER_LAW;
    h->max_run_length = max_run;
    h->params.power_law.alpha = alpha;
    
    h->h = (double *)malloc(max_run * sizeof(double));
    h->one_minus_h = (double *)malloc(max_run * sizeof(double));
    if (!h->h || !h->one_minus_h) {
        bocpd_hazard_free(h);
        return -1;
    }
    
    for (size_t r = 0; r < max_run; r++) {
        double hval = alpha / (double)(r + 1);
        if (hval > 1.0) hval = 1.0;
        h->h[r] = hval;
        h->one_minus_h[r] = 1.0 - hval;
    }
    return 0;
}

int bocpd_hazard_init_learned(bocpd_hazard_t *h, double a, double b, size_t max_run) {
    if (!h || a <= 0.0 || b <= 0.0 || max_run < 16) return -1;
    
    memset(h, 0, sizeof(*h));
    h->type = HAZARD_LEARNED;
    h->max_run_length = max_run;
    h->params.learned.a = a;
    h->params.learned.b = b;
    h->params.learned.n_obs = 0;
    h->params.learned.n_cp = 0;
    
    h->h = (double *)malloc(max_run * sizeof(double));
    h->one_minus_h = (double *)malloc(max_run * sizeof(double));
    if (!h->h || !h->one_minus_h) {
        bocpd_hazard_free(h);
        return -1;
    }
    
    double hval = a / (a + b);
    double omh = 1.0 - hval;
    for (size_t i = 0; i < max_run; i++) {
        h->h[i] = hval;
        h->one_minus_h[i] = omh;
    }
    return 0;
}

void bocpd_hazard_learn_update(bocpd_hazard_t *h, int is_changepoint, size_t decay_window) {
    if (!h || h->type != HAZARD_LEARNED) return;
    
    h->params.learned.n_obs++;
    if (is_changepoint) h->params.learned.n_cp++;
    
    /* Apply decay if needed */
    if (decay_window > 0 && h->params.learned.n_obs > decay_window) {
        double ratio = (double)decay_window / h->params.learned.n_obs;
        h->params.learned.n_obs = decay_window;
        h->params.learned.n_cp = (size_t)(h->params.learned.n_cp * ratio);
    }
    
    /* Update hazard tables */
    double a = h->params.learned.a + h->params.learned.n_cp;
    double b = h->params.learned.b + h->params.learned.n_obs - h->params.learned.n_cp;
    double hval = a / (a + b);
    double omh = 1.0 - hval;
    
    for (size_t i = 0; i < h->max_run_length; i++) {
        h->h[i] = hval;
        h->one_minus_h[i] = omh;
    }
}

void bocpd_hazard_free(bocpd_hazard_t *h) {
    if (!h) return;
    free(h->h);
    free(h->one_minus_h);
    memset(h, 0, sizeof(*h));
}

/*═══════════════════════════════════════════════════════════════════════════
 * DELTA DETECTOR
 *═══════════════════════════════════════════════════════════════════════════*/

void bocpd_delta_init(bocpd_delta_detector_t *d, size_t warmup) {
    if (!d) return;
    memset(d, 0, sizeof(*d));
    d->kappa = 1.0;
    d->mu = 0.0;
    d->alpha = 1.0;
    d->beta = 0.01;
    d->prev_short_mass = 0.0;
    d->n_observations = 0;
    d->warmup_period = warmup;
}

double bocpd_delta_update(bocpd_delta_detector_t *d, const double *r, 
                          size_t active_len, double decay) {
    if (!d || !r) return 0.0;
    
    /* Compute short-run mass P(r < BOCPD_SHORT_RUN_WINDOW) */
    double short_mass = 0.0;
    size_t limit = (active_len < BOCPD_SHORT_RUN_WINDOW) ? active_len : BOCPD_SHORT_RUN_WINDOW;
    for (size_t i = 0; i < limit; i++) {
        short_mass += r[i];
    }
    
    double delta = short_mass - d->prev_short_mass;
    d->prev_short_mass = short_mass;
    d->n_observations++;
    
    /* Storvik update for delta distribution */
    double kappa_new = d->kappa * decay + 1.0;
    double mu_new = (d->kappa * decay * d->mu + delta) / kappa_new;
    double alpha_new = d->alpha * decay + 0.5;
    double delta_diff = delta - d->mu;
    double mu_diff = delta - mu_new;
    double beta_new = d->beta * decay + 0.5 * delta_diff * mu_diff;
    
    d->kappa = kappa_new;
    d->mu = mu_new;
    d->alpha = alpha_new;
    d->beta = beta_new;
    
    return delta;
}

int bocpd_delta_check(const bocpd_delta_detector_t *d, double z_threshold) {
    if (!d || d->n_observations < d->warmup_period) return 0;
    
    double delta = d->prev_short_mass;  /* Most recent delta is stored here indirectly */
    double z = bocpd_delta_zscore(d, delta);
    return (z > z_threshold) ? 1 : 0;
}

double bocpd_delta_zscore(const bocpd_delta_detector_t *d, double delta) {
    if (!d || d->alpha < 1.0) return 0.0;
    
    double sigma_sq = d->beta / d->alpha;
    if (sigma_sq < 1e-10) sigma_sq = 1e-10;
    double sigma = sqrt(sigma_sq);
    
    return (delta - d->mu) / sigma;
}

/*═══════════════════════════════════════════════════════════════════════════
 * POSTERIOR UPDATE
 *═══════════════════════════════════════════════════════════════════════════*/

static inline void init_slot_zero(bocpd_t *b) {
    double *next = BOCPD_NEXT_BUF(b);
    const double kappa0 = b->prior.kappa0;
    const double mu0 = b->prior.mu0;
    const double alpha0 = b->prior.alpha0;
    const double beta0 = b->prior.beta0;

    IBLK_SET_MU(next, 0, mu0);
    IBLK_SET_KAPPA(next, 0, kappa0);
    IBLK_SET_ALPHA(next, 0, alpha0);
    IBLK_SET_BETA(next, 0, beta0);
    IBLK_SET_SS_N(next, 0, 0.0);

    double sigma_sq = beta0 * (kappa0 + 1.0) / (alpha0 * kappa0);
    double nu = 2.0 * alpha0;
    double C1 = b->prior_lgamma_alpha_p5 - b->prior_lgamma_alpha -
                0.5 * fast_log(nu * M_PI) - 0.5 * fast_log(sigma_sq);
    double C2 = alpha0 + 0.5;

    IBLK_SET_C1(next, 0, C1);
    IBLK_SET_C2(next, 0, C2);
    IBLK_SET_INV_SSN(next, 0, 1.0 / (sigma_sq * nu));
}

static void update_posteriors(bocpd_t *b, double x, size_t n_old) {
    init_slot_zero(b);
    if (n_old == 0) {
        b->cur_buf = 1 - b->cur_buf;
        return;
    }

    const double *cur = BOCPD_CUR_BUF(b);
    double *next = BOCPD_NEXT_BUF(b);

    const __m256d x_vec = _mm256_set1_pd(x);
    const __m256d one = _mm256_set1_pd(1.0);
    const __m256d two = _mm256_set1_pd(2.0);
    const __m256d half = _mm256_set1_pd(0.5);
    const __m256d pi = _mm256_set1_pd(M_PI);

    size_t i = 0;
    for (; i + 4 <= n_old; i += 4) {
        size_t block = i / 4;
        const double *src = cur + block * BOCPD_IBLK_DOUBLES;

        __m256d mu_old = _mm256_loadu_pd(src + BOCPD_IBLK_MU / 8);
        __m256d kappa_old = _mm256_loadu_pd(src + BOCPD_IBLK_KAPPA / 8);
        __m256d alpha_old = _mm256_loadu_pd(src + BOCPD_IBLK_ALPHA / 8);
        __m256d beta_old = _mm256_loadu_pd(src + BOCPD_IBLK_BETA / 8);
        __m256d ss_n_old = _mm256_loadu_pd(src + BOCPD_IBLK_SS_N / 8);

        __m256d ss_n_new = _mm256_add_pd(ss_n_old, one);
        __m256d kappa_new = _mm256_add_pd(kappa_old, one);
        __m256d mu_new = _mm256_div_pd(
            _mm256_fmadd_pd(kappa_old, mu_old, x_vec), kappa_new);
        __m256d alpha_new = _mm256_add_pd(alpha_old, half);

        __m256d delta1 = _mm256_sub_pd(x_vec, mu_old);
        __m256d delta2 = _mm256_sub_pd(x_vec, mu_new);
        __m256d beta_inc = _mm256_mul_pd(_mm256_mul_pd(delta1, delta2), half);
        __m256d beta_new = _mm256_add_pd(beta_old, beta_inc);

        __m256d kappa_p1 = _mm256_add_pd(kappa_new, one);
        __m256d sigma_sq = _mm256_div_pd(
            _mm256_mul_pd(beta_new, kappa_p1),
            _mm256_mul_pd(alpha_new, kappa_new));
        __m256d nu = _mm256_mul_pd(two, alpha_new);
        __m256d sigma_sq_nu = _mm256_mul_pd(sigma_sq, nu);
        __m256d inv_ssn = _mm256_div_pd(one, sigma_sq_nu);

        __m256d lg_a = fast_lgamma_avx2(alpha_new);
        __m256d alpha_p5 = _mm256_add_pd(alpha_new, half);
        __m256d lg_ap5 = fast_lgamma_avx2(alpha_p5);

        __m256d nu_pi_s2 = _mm256_mul_pd(_mm256_mul_pd(nu, pi), sigma_sq);
        __m256d ln_term = fast_log_avx2(nu_pi_s2);
        __m256d C1 = _mm256_sub_pd(lg_ap5, lg_a);
        C1 = _mm256_fnmadd_pd(half, ln_term, C1);
        __m256d C2 = alpha_p5;

        store_shifted_field(next, block, BOCPD_IBLK_MU, mu_new);
        store_shifted_field(next, block, BOCPD_IBLK_KAPPA, kappa_new);
        store_shifted_field(next, block, BOCPD_IBLK_ALPHA, alpha_new);
        store_shifted_field(next, block, BOCPD_IBLK_BETA, beta_new);
        store_shifted_field(next, block, BOCPD_IBLK_SS_N, ss_n_new);
        store_shifted_field(next, block, BOCPD_IBLK_C1, C1);
        store_shifted_field(next, block, BOCPD_IBLK_C2, C2);
        store_shifted_field(next, block, BOCPD_IBLK_INV_SSN, inv_ssn);
    }

    /* Scalar tail */
    for (; i < n_old; i++) {
        double ss_n_old = IBLK_GET_SS_N(cur, i);
        double kappa_old = IBLK_GET_KAPPA(cur, i);
        double mu_old = IBLK_GET_MU(cur, i);
        double alpha_old = IBLK_GET_ALPHA(cur, i);
        double beta_old = IBLK_GET_BETA(cur, i);

        double ss_n_new = ss_n_old + 1.0;
        double kappa_new = kappa_old + 1.0;
        double mu_new = (kappa_old * mu_old + x) / kappa_new;
        double alpha_new = alpha_old + 0.5;
        double beta_new = beta_old + 0.5 * (x - mu_old) * (x - mu_new);

        double sigma_sq = beta_new * (kappa_new + 1.0) / (alpha_new * kappa_new);
        double nu = 2.0 * alpha_new;
        double inv_ssn = 1.0 / (sigma_sq * nu);

        double lg_a = fast_lgamma(alpha_new);
        double lg_ap5 = fast_lgamma(alpha_new + 0.5);
        double C1 = lg_ap5 - lg_a - 0.5 * fast_log(nu * M_PI * sigma_sq);
        double C2 = alpha_new + 0.5;

        size_t out_idx = i + 1;
        IBLK_SET_MU(next, out_idx, mu_new);
        IBLK_SET_KAPPA(next, out_idx, kappa_new);
        IBLK_SET_ALPHA(next, out_idx, alpha_new);
        IBLK_SET_BETA(next, out_idx, beta_new);
        IBLK_SET_SS_N(next, out_idx, ss_n_new);
        IBLK_SET_C1(next, out_idx, C1);
        IBLK_SET_C2(next, out_idx, C2);
        IBLK_SET_INV_SSN(next, out_idx, inv_ssn);
    }

    b->cur_buf = 1 - b->cur_buf;
}

/*═══════════════════════════════════════════════════════════════════════════
 * PREDICTION STEP
 *═══════════════════════════════════════════════════════════════════════════*/

static void prediction_step(bocpd_t *b, double x) {
    const size_t n = b->active_len;
    if (n == 0) return;

    const double thresh = b->trunc_thresh;
    double *params = BOCPD_CUR_BUF(b);
    double *r = b->r;
    double *r_new = b->r_scratch;
    
    /* Get hazard values (constant or from table) */
    const int use_table = (b->hazard_table != NULL);
    const double h_const = b->hazard;
    const double omh_const = b->one_minus_h;

    const size_t n_padded = (n + 7) & ~7ULL;

    for (size_t i = n; i < n_padded + 8; i++) r[i] = 0.0;
    memset(r_new, 0, (n_padded + 16) * sizeof(double));

    const __m256d x_vec = _mm256_set1_pd(x);
    const __m256d thresh_vec = _mm256_set1_pd(thresh);
    const __m256d min_pp = _mm256_set1_pd(1e-300);
    const __m256d const_one = _mm256_set1_pd(1.0);

    /* log1p coefficients */
    const __m256d log1p_c2 = _mm256_set1_pd(-0.5);
    const __m256d log1p_c3 = _mm256_set1_pd(0.3333333333333333);
    const __m256d log1p_c4 = _mm256_set1_pd(-0.25);
    const __m256d log1p_c5 = _mm256_set1_pd(0.2);
    const __m256d log1p_c6 = _mm256_set1_pd(-0.1666666666666667);

    /* exp coefficients */
    const __m256d exp_inv_ln2 = _mm256_set1_pd(1.4426950408889634);
    const __m256d exp_min_x = _mm256_set1_pd(-700.0);
    const __m256d exp_max_x = _mm256_set1_pd(700.0);
    const __m256d exp_c1 = _mm256_set1_pd(0.6931471805599453);
    const __m256d exp_c2 = _mm256_set1_pd(0.24022650695910072);
    const __m256d exp_c3 = _mm256_set1_pd(0.05550410866482158);
    const __m256d exp_c4 = _mm256_set1_pd(0.009618129107628477);
    const __m256d exp_c5 = _mm256_set1_pd(0.0013333558146428443);
    const __m256d exp_c6 = _mm256_set1_pd(0.00015403530393381608);
    const __m256i exp_bias = _mm256_set1_epi64x(1023);

    __m256d r0_acc = _mm256_setzero_pd();
    __m256d max_growth = _mm256_setzero_pd();
    __m256i max_idx_vec = _mm256_setzero_si256();
    __m256i idx_vec = _mm256_set_epi64x(4, 3, 2, 1);
    const __m256i idx_inc = _mm256_set1_epi64x(4);
    size_t last_valid = 0;

    for (size_t i = 0; i < n_padded; i += 4) {
        size_t block = i / 4;
        const double *blk = params + block * BOCPD_IBLK_DOUBLES;

        __m256d mu = _mm256_loadu_pd(blk + BOCPD_IBLK_MU / 8);
        __m256d C1 = _mm256_loadu_pd(blk + BOCPD_IBLK_C1 / 8);
        __m256d C2 = _mm256_loadu_pd(blk + BOCPD_IBLK_C2 / 8);
        __m256d inv_ssn = _mm256_loadu_pd(blk + BOCPD_IBLK_INV_SSN / 8);
        __m256d r_old = _mm256_loadu_pd(&r[i]);

        /* Load hazard (constant or from table) */
        __m256d h_vec, omh_vec;
        if (use_table) {
            h_vec = _mm256_loadu_pd(&b->hazard_table->h[i]);
            omh_vec = _mm256_loadu_pd(&b->hazard_table->one_minus_h[i]);
        } else {
            h_vec = _mm256_set1_pd(h_const);
            omh_vec = _mm256_set1_pd(omh_const);
        }

        /* Student-t log-likelihood */
        __m256d z = _mm256_sub_pd(x_vec, mu);
        __m256d z2 = _mm256_mul_pd(z, z);
        __m256d t = _mm256_mul_pd(z2, inv_ssn);

        /* log1p(t) polynomial */
        __m256d poly = _mm256_fmadd_pd(t, log1p_c6, log1p_c5);
        poly = _mm256_fmadd_pd(t, poly, log1p_c4);
        poly = _mm256_fmadd_pd(t, poly, log1p_c3);
        poly = _mm256_fmadd_pd(t, poly, log1p_c2);
        poly = _mm256_fmadd_pd(t, poly, const_one);
        __m256d log1p_t = _mm256_mul_pd(t, poly);

        __m256d ln_pp = _mm256_fnmadd_pd(C2, log1p_t, C1);

        /* exp via 2^(x/ln2) */
        __m256d x_clamp = _mm256_max_pd(_mm256_min_pd(ln_pp, exp_max_x), exp_min_x);
        __m256d t_exp = _mm256_mul_pd(x_clamp, exp_inv_ln2);
        __m256d k = _mm256_round_pd(t_exp, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m256d f = _mm256_sub_pd(t_exp, k);

        __m256d f2 = _mm256_mul_pd(f, f);
        __m256d p01 = _mm256_fmadd_pd(f, exp_c1, const_one);
        __m256d p23 = _mm256_fmadd_pd(f, exp_c3, exp_c2);
        __m256d p45 = _mm256_fmadd_pd(f, exp_c5, exp_c4);
        __m256d q0123 = _mm256_fmadd_pd(f2, p23, p01);
        __m256d q456 = _mm256_fmadd_pd(f2, exp_c6, p45);
        __m256d f4 = _mm256_mul_pd(f2, f2);
        __m256d exp_p = _mm256_fmadd_pd(f4, q456, q0123);

        __m128i k32 = _mm256_cvtpd_epi32(k);
        __m256i k64 = _mm256_cvtepi32_epi64(k32);
        __m256i biased = _mm256_add_epi64(k64, exp_bias);
        __m256i bits = _mm256_slli_epi64(biased, 52);
        __m256d scale = _mm256_castsi256_pd(bits);

        __m256d pp = _mm256_mul_pd(exp_p, scale);
        pp = _mm256_max_pd(pp, min_pp);

        /* BOCPD update */
        __m256d r_pp = _mm256_mul_pd(r_old, pp);
        __m256d growth = _mm256_mul_pd(r_pp, omh_vec);
        __m256d change = _mm256_mul_pd(r_pp, h_vec);

        _mm256_storeu_pd(&r_new[i + 1], growth);
        r0_acc = _mm256_add_pd(r0_acc, change);

        /* MAP tracking */
        __m256d cmp = _mm256_cmp_pd(growth, max_growth, _CMP_GT_OQ);
        max_growth = _mm256_blendv_pd(max_growth, growth, cmp);
        max_idx_vec = _mm256_castpd_si256(_mm256_blendv_pd(
            _mm256_castsi256_pd(max_idx_vec),
            _mm256_castsi256_pd(idx_vec), cmp));

        /* Truncation tracking */
        __m256d thresh_cmp = _mm256_cmp_pd(growth, thresh_vec, _CMP_GT_OQ);
        int mask = _mm256_movemask_pd(thresh_cmp);
        if (mask) {
            if (mask & 8) last_valid = i + 4;
            else if (mask & 4) last_valid = i + 3;
            else if (mask & 2) last_valid = i + 2;
            else if (mask & 1) last_valid = i + 1;
        }

        idx_vec = _mm256_add_epi64(idx_vec, idx_inc);
    }

    /* Horizontal sum for r0 */
    __m128d lo = _mm256_castpd256_pd128(r0_acc);
    __m128d hi = _mm256_extractf128_pd(r0_acc, 1);
    lo = _mm_add_pd(lo, hi);
    lo = _mm_add_pd(lo, _mm_shuffle_pd(lo, lo, 1));
    double r0 = _mm_cvtsd_f64(lo);

    r_new[0] = r0;
    if (r0 > thresh && last_valid == 0) last_valid = 1;

    /* Find global max for MAP */
    double max_arr[4];
    int64_t idx_arr[4];
    _mm256_storeu_pd(max_arr, max_growth);
    _mm256_storeu_si256((__m256i *)idx_arr, max_idx_vec);

    double map_val = r0;
    size_t map_idx = 0;
    for (int j = 0; j < 4; j++) {
        if (max_arr[j] > map_val) {
            map_val = max_arr[j];
            map_idx = idx_arr[j];
        }
    }

    /* Normalize */
    size_t new_len = (last_valid > 0) ? last_valid + 1 : n + 1;
    if (new_len > b->capacity) new_len = b->capacity;

    size_t new_len_padded = (new_len + 3) & ~3ULL;

    __m256d sum_acc = _mm256_setzero_pd();
    for (size_t j = 0; j < new_len_padded; j += 4)
        sum_acc = _mm256_add_pd(sum_acc, _mm256_loadu_pd(&r_new[j]));

    lo = _mm256_castpd256_pd128(sum_acc);
    hi = _mm256_extractf128_pd(sum_acc, 1);
    lo = _mm_add_pd(lo, hi);
    lo = _mm_add_pd(lo, _mm_shuffle_pd(lo, lo, 1));
    double r_sum = _mm_cvtsd_f64(lo);

    if (r_sum > 1e-300) {
        __m256d inv_sum = _mm256_set1_pd(1.0 / r_sum);
        for (size_t j = 0; j < new_len_padded; j += 4) {
            __m256d rv = _mm256_loadu_pd(&r_new[j]);
            _mm256_storeu_pd(&r[j], _mm256_mul_pd(rv, inv_sum));
        }
    }

    b->active_len = new_len;
    b->map_runlength = map_idx;
}

/*═══════════════════════════════════════════════════════════════════════════
 * PUBLIC API
 *═══════════════════════════════════════════════════════════════════════════*/

int bocpd_init(bocpd_t *b, double hazard_lambda, bocpd_prior_t prior, 
               size_t max_run_length) {
    if (!b || hazard_lambda <= 0.0 || max_run_length < 16) return -1;

    memset(b, 0, sizeof(*b));

    size_t cap = 32;
    while (cap < max_run_length) cap <<= 1;

    b->capacity = cap;
    b->hazard = 1.0 / hazard_lambda;
    b->one_minus_h = 1.0 - b->hazard;
    b->trunc_thresh = 1e-6;
    b->prior = prior;
    b->cur_buf = 0;
    b->hazard_table = NULL;

    b->prior_lgamma_alpha = fast_lgamma(prior.alpha0);
    b->prior_lgamma_alpha_p5 = fast_lgamma(prior.alpha0 + 0.5);

    size_t n_blocks = cap / 4 + 2;
    size_t bytes_interleaved = n_blocks * BOCPD_IBLK_STRIDE;
    size_t bytes_r = (cap + 32) * sizeof(double);
    size_t total = 2 * bytes_interleaved + 2 * bytes_r + 64;

#ifdef _WIN32
    void *mega = _aligned_malloc(total, 64);
#else
    void *mega = NULL;
    if (posix_memalign(&mega, 64, total) != 0) mega = NULL;
#endif

    if (!mega) return -1;
    memset(mega, 0, total);

    uint8_t *ptr = (uint8_t *)mega;
    b->interleaved[0] = (double *)ptr; ptr += bytes_interleaved;
    b->interleaved[1] = (double *)ptr; ptr += bytes_interleaved;
    b->r = (double *)ptr; ptr += bytes_r;
    b->r_scratch = (double *)ptr;

    b->mega = mega;
    b->mega_bytes = total;
    b->t = 0;
    b->active_len = 0;

    return 0;
}

int bocpd_init_with_hazard(bocpd_t *b, bocpd_hazard_t *hazard, bocpd_prior_t prior) {
    if (!b || !hazard) return -1;
    
    int ret = bocpd_init(b, 100.0, prior, hazard->max_run_length);
    if (ret != 0) return ret;
    
    b->hazard_table = hazard;
    return 0;
}

void bocpd_free(bocpd_t *b) {
    if (!b) return;
#ifdef _WIN32
    if (b->mega) _aligned_free(b->mega);
#else
    free(b->mega);
#endif
    memset(b, 0, sizeof(*b));
}

void bocpd_reset(bocpd_t *b) {
    if (!b) return;
    memset(b->r, 0, (b->capacity + 32) * sizeof(double));
    memset(b->r_scratch, 0, (b->capacity + 32) * sizeof(double));
    size_t n_blocks = b->capacity / 4 + 2;
    size_t bytes_interleaved = n_blocks * BOCPD_IBLK_STRIDE;
    memset(b->interleaved[0], 0, bytes_interleaved);
    memset(b->interleaved[1], 0, bytes_interleaved);
    b->t = 0;
    b->active_len = 0;
    b->cur_buf = 0;
    b->map_runlength = 0;
    b->p_changepoint = 0.0;
}

void bocpd_step(bocpd_t *b, double x) {
    if (!b) return;

    if (b->t == 0) {
        b->r[0] = 1.0;
        double *cur = BOCPD_CUR_BUF(b);

        double k0 = b->prior.kappa0;
        double mu0 = b->prior.mu0;
        double a0 = b->prior.alpha0;
        double b0 = b->prior.beta0;

        double k1 = k0 + 1.0;
        double mu1 = (k0 * mu0 + x) / k1;
        double a1 = a0 + 0.5;
        double beta1 = b0 + 0.5 * (x - mu0) * (x - mu1);

        IBLK_SET_MU(cur, 0, mu1);
        IBLK_SET_KAPPA(cur, 0, k1);
        IBLK_SET_ALPHA(cur, 0, a1);
        IBLK_SET_BETA(cur, 0, beta1);
        IBLK_SET_SS_N(cur, 0, 1.0);

        double sigma_sq = beta1 * (k1 + 1.0) / (a1 * k1);
        double nu = 2.0 * a1;

        double lg_a = fast_lgamma(a1);
        double lg_ap5 = fast_lgamma(a1 + 0.5);
        double C1 = lg_ap5 - lg_a - 0.5 * fast_log(nu * M_PI * sigma_sq);
        double C2 = a1 + 0.5;

        IBLK_SET_C1(cur, 0, C1);
        IBLK_SET_C2(cur, 0, C2);
        IBLK_SET_INV_SSN(cur, 0, 1.0 / (sigma_sq * nu));

        b->active_len = 1;
        b->t = 1;
        b->map_runlength = 0;
        b->p_changepoint = 1.0;
        return;
    }

    size_t n_old = b->active_len;
    prediction_step(b, x);
    update_posteriors(b, x, n_old);
    b->t++;

    double p = 0.0;
    size_t lim = (b->active_len < 5) ? b->active_len : 5;
    for (size_t j = 0; j < lim; j++) p += b->r[j];
    b->p_changepoint = p;
}

double bocpd_short_mass(const bocpd_t *b, size_t window) {
    if (!b) return 0.0;
    double mass = 0.0;
    size_t lim = (b->active_len < window) ? b->active_len : window;
    for (size_t i = 0; i < lim; i++) mass += b->r[i];
    return mass;
}
