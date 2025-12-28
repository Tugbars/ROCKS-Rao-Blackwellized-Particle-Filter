/**
 * @file thompson_sampler.c
 * @brief Thompson Sampling Implementation
 */

#include "thompson_sampler.h"
#include <math.h>
#include <string.h>

/*═══════════════════════════════════════════════════════════════════════════
 * CONSTANTS
 *═══════════════════════════════════════════════════════════════════════════*/

#define MIN_ALPHA 1e-6
#define PI_VAL 3.14159265358979323846

/*═══════════════════════════════════════════════════════════════════════════
 * RNG (xoroshiro128+)
 *═══════════════════════════════════════════════════════════════════════════*/

static inline uint64_t rotl(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

static inline uint64_t xoroshiro128_next(ThompsonRNG *rng) {
    const uint64_t s0 = rng->s[0];
    uint64_t s1 = rng->s[1];
    const uint64_t result = s0 + s1;
    
    s1 ^= s0;
    rng->s[0] = rotl(s0, 24) ^ s1 ^ (s1 << 16);
    rng->s[1] = rotl(s1, 37);
    
    return result;
}

static inline uint64_t splitmix64(uint64_t *state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

void thompson_rng_init(ThompsonRNG *rng, uint64_t seed) {
    if (!rng) return;
    
    /* Initialize using splitmix64 */
    uint64_t state = seed;
    rng->s[0] = splitmix64(&state);
    rng->s[1] = splitmix64(&state);
    
    /* Ensure not all zeros */
    if (rng->s[0] == 0 && rng->s[1] == 0) {
        rng->s[0] = 1;
    }
}

double thompson_rng_uniform(ThompsonRNG *rng) {
    if (!rng) return 0.5;
    uint64_t u = xoroshiro128_next(rng);
    return (u >> 11) * 0x1.0p-53;  /* [0, 1) */
}

/*═══════════════════════════════════════════════════════════════════════════
 * NORMAL DISTRIBUTION (Box-Muller)
 *═══════════════════════════════════════════════════════════════════════════*/

double thompson_rng_normal(ThompsonRNG *rng) {
    if (!rng) return 0.0;
    
    double u1, u2;
    do {
        u1 = thompson_rng_uniform(rng);
    } while (u1 < 1e-10);
    
    u2 = thompson_rng_uniform(rng);
    
    return sqrt(-2.0 * log(u1)) * cos(2.0 * PI_VAL * u2);
}

/*═══════════════════════════════════════════════════════════════════════════
 * GAMMA DISTRIBUTION (Marsaglia-Tsang)
 *═══════════════════════════════════════════════════════════════════════════*/

double thompson_rng_gamma(ThompsonRNG *rng, double shape) {
    if (!rng || shape <= 0) return 0.0;
    
    /* For shape < 1, use Ahrens-Dieter method with boost */
    if (shape < 1.0) {
        double u = thompson_rng_uniform(rng);
        return thompson_rng_gamma(rng, shape + 1.0) * pow(u, 1.0 / shape);
    }
    
    /* Marsaglia-Tsang for shape >= 1 */
    double d = shape - 1.0 / 3.0;
    double c = 1.0 / sqrt(9.0 * d);
    
    while (1) {
        double x, v;
        do {
            x = thompson_rng_normal(rng);
            v = 1.0 + c * x;
        } while (v <= 0);
        
        v = v * v * v;
        double u = thompson_rng_uniform(rng);
        
        /* Quick accept */
        if (u < 1.0 - 0.0331 * (x * x) * (x * x)) {
            return d * v;
        }
        
        /* Full check */
        if (log(u) < 0.5 * x * x + d * (1.0 - v + log(v))) {
            return d * v;
        }
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * DIRICHLET SAMPLING
 *═══════════════════════════════════════════════════════════════════════════*/

void thompson_sample_dirichlet(
    const float *alpha,
    int K,
    float *out,
    ThompsonRNG *rng)
{
    if (!alpha || !out || !rng || K <= 0) return;
    
    double sum = 0.0;
    
    /* Sample K independent Gamma(alpha_i, 1) */
    for (int i = 0; i < K; i++) {
        double a = (double)alpha[i];
        if (a < MIN_ALPHA) a = MIN_ALPHA;
        
        double g = thompson_rng_gamma(rng, a);
        out[i] = (float)g;
        sum += g;
    }
    
    /* Normalize to get Dirichlet sample */
    if (sum > 0) {
        float inv_sum = (float)(1.0 / sum);
        for (int i = 0; i < K; i++) {
            out[i] *= inv_sum;
        }
    } else {
        /* Fallback to uniform */
        float uniform = 1.0f / K;
        for (int i = 0; i < K; i++) {
            out[i] = uniform;
        }
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * THOMPSON SAMPLING FOR Π
 *═══════════════════════════════════════════════════════════════════════════*/

void thompson_sample_pi(
    const float *Q,
    float prior_alpha,
    int K,
    float *Pi_out,
    ThompsonRNG *rng)
{
    if (!Q || !Pi_out || !rng || K <= 0) return;
    
    float alpha[64];  /* Assume K <= 64 */
    if (K > 64) K = 64;
    
    /* Sample each row from Dirichlet(Q_row + prior) */
    for (int i = 0; i < K; i++) {
        /* Build alpha = Q_row + prior */
        for (int j = 0; j < K; j++) {
            alpha[j] = Q[i * K + j] + prior_alpha;
        }
        
        /* Sample row */
        thompson_sample_dirichlet(alpha, K, Pi_out + i * K, rng);
    }
}

void thompson_mean_pi(
    const float *Q,
    float prior_alpha,
    int K,
    float *Pi_out)
{
    if (!Q || !Pi_out || K <= 0) return;
    
    /* Each row: mean = (Q_row + prior) / sum(Q_row + K*prior) */
    for (int i = 0; i < K; i++) {
        float row_sum = K * prior_alpha;
        for (int j = 0; j < K; j++) {
            row_sum += Q[i * K + j];
        }
        
        if (row_sum > 0) {
            for (int j = 0; j < K; j++) {
                Pi_out[i * K + j] = (Q[i * K + j] + prior_alpha) / row_sum;
            }
        } else {
            /* Uniform */
            float uniform = 1.0f / K;
            for (int j = 0; j < K; j++) {
                Pi_out[i * K + j] = uniform;
            }
        }
    }
}

void thompson_sample_with_confidence(
    const float *Q,
    float prior_alpha,
    int K,
    float confidence,
    float *Pi_out,
    ThompsonRNG *rng)
{
    if (!Q || !Pi_out || !rng || K <= 0) return;
    
    /* Clamp confidence */
    if (confidence < 0.0f) confidence = 0.0f;
    if (confidence > 1.0f) confidence = 1.0f;
    
    /* Get mean */
    float Pi_mean[64 * 64];
    thompson_mean_pi(Q, prior_alpha, K, Pi_mean);
    
    /* Get sample */
    float Pi_sample[64 * 64];
    thompson_sample_pi(Q, prior_alpha, K, Pi_sample, rng);
    
    /* Blend: high confidence → mean, low confidence → sample */
    int size = K * K;
    for (int i = 0; i < size; i++) {
        Pi_out[i] = confidence * Pi_mean[i] + (1.0f - confidence) * Pi_sample[i];
    }
    
    /* Renormalize rows (blending may break row-stochastic property) */
    for (int i = 0; i < K; i++) {
        float row_sum = 0.0f;
        for (int j = 0; j < K; j++) {
            row_sum += Pi_out[i * K + j];
        }
        if (row_sum > 0) {
            for (int j = 0; j < K; j++) {
                Pi_out[i * K + j] /= row_sum;
            }
        }
    }
}
