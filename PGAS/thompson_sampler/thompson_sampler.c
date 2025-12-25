/**
 * @file thompson_sampler.c
 * @brief Thompson Sampling for Transition Matrix Handoff
 *
 * Implementation of Dirichlet sampling and explore/exploit logic.
 */

#include "thompson_sampler.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

/*═══════════════════════════════════════════════════════════════════════════
 * RNG (xoroshiro128+)
 *═══════════════════════════════════════════════════════════════════════════*/

static inline uint64_t rotl(const uint64_t x, int k)
{
    return (x << k) | (x >> (64 - k));
}

static uint64_t xoroshiro128plus(uint64_t *s)
{
    const uint64_t s0 = s[0];
    uint64_t s1 = s[1];
    const uint64_t result = s0 + s1;

    s1 ^= s0;
    s[0] = rotl(s0, 24) ^ s1 ^ (s1 << 16);
    s[1] = rotl(s1, 37);

    return result;
}

/* Uniform [0, 1) */
static inline float rand_uniform(uint64_t *s)
{
    return (xoroshiro128plus(s) >> 11) * (1.0f / 9007199254740992.0f);
}

/* Standard normal via Box-Muller */
static float rand_normal(uint64_t *s)
{
    float u1 = rand_uniform(s);
    float u2 = rand_uniform(s);

    /* Avoid log(0) */
    if (u1 < 1e-10f)
        u1 = 1e-10f;

    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265358979f * u2);
}

/* Seed the RNG */
static void seed_rng(uint64_t *s, uint64_t seed)
{
    /* SplitMix64 to initialize */
    uint64_t z = seed;

    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    s[0] = z ^ (z >> 31);

    z = seed + 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    s[1] = z ^ (z >> 31);

    /* Ensure non-zero */
    if (s[0] == 0 && s[1] == 0)
    {
        s[0] = 1;
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * GAMMA SAMPLING
 *
 * Marsaglia & Tsang's method for shape >= 1
 * Ahrens-Dieter method for shape < 1
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Sample from Gamma(shape, 1) where shape >= 1
 * Uses Marsaglia & Tsang's method (2000)
 */
static float gamma_mt(uint64_t *rng, float shape)
{
    float d = shape - 1.0f / 3.0f;
    float c = 1.0f / sqrtf(9.0f * d);

    while (1)
    {
        float x, v;

        do
        {
            x = rand_normal(rng);
            v = 1.0f + c * x;
        } while (v <= 0.0f);

        v = v * v * v;
        float u = rand_uniform(rng);

        /* Squeeze test */
        if (u < 1.0f - 0.0331f * (x * x) * (x * x))
        {
            return d * v;
        }

        /* Full test */
        if (logf(u) < 0.5f * x * x + d * (1.0f - v + logf(v)))
        {
            return d * v;
        }
    }
}

/**
 * Sample from Gamma(shape, scale) for any shape > 0
 */
float thompson_sampler_gamma(ThompsonSampler *sampler, float shape, float scale)
{
    if (!sampler || shape <= 0.0f || scale <= 0.0f)
    {
        return 0.0f;
    }

    float result;

    if (shape >= 1.0f)
    {
        /* Marsaglia & Tsang directly */
        result = gamma_mt(sampler->rng_state, shape);
    }
    else
    {
        /* For shape < 1, use: Gamma(a) = Gamma(a+1) * U^(1/a)
         * where U ~ Uniform(0,1) */
        float g = gamma_mt(sampler->rng_state, shape + 1.0f);
        float u = rand_uniform(sampler->rng_state);
        result = g * powf(u, 1.0f / shape);
    }

    return result * scale;
}

/*═══════════════════════════════════════════════════════════════════════════
 * DIRICHLET SAMPLING
 *
 * Sample from Dirichlet(alpha) by:
 *   1. Sample X_i ~ Gamma(alpha_i, 1) for each i
 *   2. Normalize: theta_i = X_i / sum(X)
 *═══════════════════════════════════════════════════════════════════════════*/

void thompson_sampler_dirichlet(
    ThompsonSampler *sampler,
    const float *alpha,
    int K,
    float *out)
{

    if (!sampler || !alpha || !out || K <= 0)
        return;

    float min_alpha = sampler->config.min_concentration;
    float sum = 0.0f;

    /* Sample gamma variates */
    for (int i = 0; i < K; i++)
    {
        float a = alpha[i];
        if (a < min_alpha)
            a = min_alpha; /* Ensure valid concentration */

        out[i] = thompson_sampler_gamma(sampler, a, 1.0f);
        sum += out[i];
    }

    /* Normalize to simplex */
    if (sum > 1e-10f)
    {
        float inv_sum = 1.0f / sum;
        for (int i = 0; i < K; i++)
        {
            out[i] *= inv_sum;
        }
    }
    else
    {
        /* Fallback to uniform if degenerate */
        float uniform = 1.0f / K;
        for (int i = 0; i < K; i++)
        {
            out[i] = uniform;
        }
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * CONFIGURATION
 *═══════════════════════════════════════════════════════════════════════════*/

ThompsonSamplerConfig thompson_sampler_config_defaults(int n_regimes)
{
    ThompsonSamplerConfig cfg;
    memset(&cfg, 0, sizeof(cfg));

    cfg.n_regimes = (n_regimes > 0 && n_regimes <= THOMPSON_MAX_REGIMES)
                        ? n_regimes
                        : 4;

    cfg.exploit_threshold = 500.0f; /* Need 500 pseudo-counts to exploit */
    cfg.min_concentration = 0.1f;   /* Minimum Dirichlet parameter */
    cfg.floor_probability = 1e-5f;  /* Minimum probability */
    cfg.seed = 0xCAFEBABE;

    return cfg;
}

/*═══════════════════════════════════════════════════════════════════════════
 * LIFECYCLE
 *═══════════════════════════════════════════════════════════════════════════*/

int thompson_sampler_init(ThompsonSampler *sampler, const ThompsonSamplerConfig *config)
{
    if (!sampler)
        return -1;

    memset(sampler, 0, sizeof(*sampler));

    sampler->config = config ? *config : thompson_sampler_config_defaults(4);

    seed_rng(sampler->rng_state, sampler->config.seed);

    sampler->initialized = true;
    return 0;
}

void thompson_sampler_reset(ThompsonSampler *sampler)
{
    if (!sampler || !sampler->initialized)
        return;

    seed_rng(sampler->rng_state, sampler->config.seed);

    sampler->total_samples = 0;
    sampler->explore_count = 0;
    sampler->exploit_count = 0;
}

void thompson_sampler_free(ThompsonSampler *sampler)
{
    if (sampler)
    {
        memset(sampler, 0, sizeof(*sampler));
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * CORE SAMPLING
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Compute row sums and determine if any row should explore
 */
static void compute_row_stats(
    const float Q[THOMPSON_MAX_REGIMES][THOMPSON_MAX_REGIMES],
    int K,
    float *row_sums,
    float *min_sum,
    float *max_sum)
{

    *min_sum = FLT_MAX;
    *max_sum = 0.0f;

    for (int i = 0; i < K; i++)
    {
        float sum = 0.0f;
        for (int j = 0; j < K; j++)
        {
            sum += Q[i][j];
        }
        row_sums[i] = sum;

        if (sum < *min_sum)
            *min_sum = sum;
        if (sum > *max_sum)
            *max_sum = sum;
    }
}

/**
 * Apply floor and renormalize row
 *
 * When flooring, we need to ensure:
 * 1. All values >= floor
 * 2. Sum = 1.0
 *
 * Strategy: Set floored values aside, distribute remaining mass among non-floored.
 */
static void apply_floor_and_normalize(float *row, int K, float floor_prob)
{
    /* First pass: identify floored cells and available mass */
    int n_floored = 0;
    float non_floored_sum = 0.0f;

    for (int i = 0; i < K; i++)
    {
        if (row[i] < floor_prob)
        {
            n_floored++;
        }
        else
        {
            non_floored_sum += row[i];
        }
    }

    if (n_floored == 0)
    {
        /* Just normalize */
        float sum = 0.0f;
        for (int i = 0; i < K; i++)
            sum += row[i];
        if (sum > 1e-10f && fabsf(sum - 1.0f) > 1e-6f)
        {
            float inv_sum = 1.0f / sum;
            for (int i = 0; i < K; i++)
                row[i] *= inv_sum;
        }
        return;
    }

    /* Mass needed for floored cells */
    float floored_mass = n_floored * floor_prob;

    /* Mass available for non-floored cells */
    float available_mass = 1.0f - floored_mass;

    if (available_mass < floor_prob || non_floored_sum < 1e-10f)
    {
        /* Degenerate case: not enough mass, use uniform */
        float uniform = 1.0f / K;
        for (int i = 0; i < K; i++)
            row[i] = uniform;
        return;
    }

    /* Scale factor for non-floored cells */
    float scale = available_mass / non_floored_sum;

    /* Apply: floored cells get floor, non-floored get scaled */
    for (int i = 0; i < K; i++)
    {
        if (row[i] < floor_prob)
        {
            row[i] = floor_prob;
        }
        else
        {
            row[i] *= scale;
            /* Safety: ensure still above floor after scaling */
            if (row[i] < floor_prob)
                row[i] = floor_prob;
        }
    }
    /* Note: Do NOT renormalize after this - the math is exact */
}

ThompsonSampleResult thompson_sampler_sample(
    ThompsonSampler *sampler,
    const float Q[THOMPSON_MAX_REGIMES][THOMPSON_MAX_REGIMES],
    float Pi_out[THOMPSON_MAX_REGIMES][THOMPSON_MAX_REGIMES])
{

    ThompsonSampleResult result;
    memset(&result, 0, sizeof(result));

    if (!sampler || !sampler->initialized || !Q || !Pi_out)
    {
        return result;
    }

    int K = sampler->config.n_regimes;
    float threshold = sampler->config.exploit_threshold;
    float floor_prob = sampler->config.floor_probability;

    /* Compute row statistics */
    compute_row_stats(Q, K, result.row_sums, &result.min_row_sum, &result.max_row_sum);

    /* Decision: explore if ANY row has insufficient data */
    bool should_explore = (result.min_row_sum < threshold);
    result.explored = should_explore;

    sampler->total_samples++;

    if (should_explore)
    {
        /* EXPLORE: Sample Π ~ Dirichlet(Q) for each row */
        sampler->explore_count++;

        for (int i = 0; i < K; i++)
        {
            /* Use Q[i,:] as Dirichlet concentration parameters */
            float alpha[THOMPSON_MAX_REGIMES];
            for (int j = 0; j < K; j++)
            {
                alpha[j] = Q[i][j];
            }

            thompson_sampler_dirichlet(sampler, alpha, K, Pi_out[i]);
            apply_floor_and_normalize(Pi_out[i], K, floor_prob);
        }
    }
    else
    {
        /* EXPLOIT: Use mean Π = Q / sum(Q) */
        sampler->exploit_count++;

        for (int i = 0; i < K; i++)
        {
            float row_sum = result.row_sums[i];
            if (row_sum < 1e-10f)
                row_sum = 1e-10f;

            for (int j = 0; j < K; j++)
            {
                Pi_out[i][j] = Q[i][j] / row_sum;
            }

            apply_floor_and_normalize(Pi_out[i], K, floor_prob);
        }
    }

    return result;
}

ThompsonSampleResult thompson_sampler_explore(
    ThompsonSampler *sampler,
    const float Q[THOMPSON_MAX_REGIMES][THOMPSON_MAX_REGIMES],
    float Pi_out[THOMPSON_MAX_REGIMES][THOMPSON_MAX_REGIMES])
{

    ThompsonSampleResult result;
    memset(&result, 0, sizeof(result));

    if (!sampler || !sampler->initialized || !Q || !Pi_out)
    {
        return result;
    }

    int K = sampler->config.n_regimes;
    float floor_prob = sampler->config.floor_probability;

    compute_row_stats(Q, K, result.row_sums, &result.min_row_sum, &result.max_row_sum);
    result.explored = true;

    sampler->total_samples++;
    sampler->explore_count++;

    /* Force Dirichlet sampling */
    for (int i = 0; i < K; i++)
    {
        float alpha[THOMPSON_MAX_REGIMES];
        for (int j = 0; j < K; j++)
        {
            alpha[j] = Q[i][j];
        }

        thompson_sampler_dirichlet(sampler, alpha, K, Pi_out[i]);
        apply_floor_and_normalize(Pi_out[i], K, floor_prob);
    }

    return result;
}

ThompsonSampleResult thompson_sampler_exploit(
    ThompsonSampler *sampler,
    const float Q[THOMPSON_MAX_REGIMES][THOMPSON_MAX_REGIMES],
    float Pi_out[THOMPSON_MAX_REGIMES][THOMPSON_MAX_REGIMES])
{

    ThompsonSampleResult result;
    memset(&result, 0, sizeof(result));

    if (!sampler || !sampler->initialized || !Q || !Pi_out)
    {
        return result;
    }

    int K = sampler->config.n_regimes;
    float floor_prob = sampler->config.floor_probability;

    compute_row_stats(Q, K, result.row_sums, &result.min_row_sum, &result.max_row_sum);
    result.explored = false;

    sampler->total_samples++;
    sampler->exploit_count++;

    /* Force mean computation */
    for (int i = 0; i < K; i++)
    {
        float row_sum = result.row_sums[i];
        if (row_sum < 1e-10f)
            row_sum = 1e-10f;

        for (int j = 0; j < K; j++)
        {
            Pi_out[i][j] = Q[i][j] / row_sum;
        }

        apply_floor_and_normalize(Pi_out[i], K, floor_prob);
    }

    return result;
}

/*═══════════════════════════════════════════════════════════════════════════
 * FLAT ARRAY VERSIONS (for SAEM compatibility)
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Helper: copy flat to 2D
 */
static void flat_to_2d(const float *flat,
                       float out[THOMPSON_MAX_REGIMES][THOMPSON_MAX_REGIMES],
                       int K)
{
    for (int i = 0; i < K; i++)
    {
        for (int j = 0; j < K; j++)
        {
            out[i][j] = flat[i * K + j];
        }
    }
}

/**
 * Helper: copy 2D to flat
 */
static void copy_2d_to_flat(const float in[THOMPSON_MAX_REGIMES][THOMPSON_MAX_REGIMES],
                            float *flat,
                            int K)
{
    for (int i = 0; i < K; i++)
    {
        for (int j = 0; j < K; j++)
        {
            flat[i * K + j] = in[i][j];
        }
    }
}

ThompsonSampleResult thompson_sampler_sample_flat(
    ThompsonSampler *sampler,
    const float *Q_flat,
    int K,
    float *Pi_flat)
{

    ThompsonSampleResult result;
    memset(&result, 0, sizeof(result));

    if (!sampler || !Q_flat || !Pi_flat || K <= 0 || K > THOMPSON_MAX_REGIMES)
    {
        return result;
    }

    /* Convert flat to 2D */
    float Q_2d[THOMPSON_MAX_REGIMES][THOMPSON_MAX_REGIMES];
    float Pi_2d[THOMPSON_MAX_REGIMES][THOMPSON_MAX_REGIMES];

    flat_to_2d(Q_flat, Q_2d, K);

    /* Call the 2D version */
    result = thompson_sampler_sample(sampler, Q_2d, Pi_2d);

    /* Convert back to flat */
    copy_2d_to_flat(Pi_2d, Pi_flat, K);

    return result;
}

ThompsonSampleResult thompson_sampler_explore_flat(
    ThompsonSampler *sampler,
    const float *Q_flat,
    int K,
    float *Pi_flat)
{

    ThompsonSampleResult result;
    memset(&result, 0, sizeof(result));

    if (!sampler || !Q_flat || !Pi_flat || K <= 0 || K > THOMPSON_MAX_REGIMES)
    {
        return result;
    }

    float Q_2d[THOMPSON_MAX_REGIMES][THOMPSON_MAX_REGIMES];
    float Pi_2d[THOMPSON_MAX_REGIMES][THOMPSON_MAX_REGIMES];

    flat_to_2d(Q_flat, Q_2d, K);
    result = thompson_sampler_explore(sampler, Q_2d, Pi_2d);
    copy_2d_to_flat(Pi_2d, Pi_flat, K);

    return result;
}

ThompsonSampleResult thompson_sampler_exploit_flat(
    ThompsonSampler *sampler,
    const float *Q_flat,
    int K,
    float *Pi_flat)
{

    ThompsonSampleResult result;
    memset(&result, 0, sizeof(result));

    if (!sampler || !Q_flat || !Pi_flat || K <= 0 || K > THOMPSON_MAX_REGIMES)
    {
        return result;
    }

    float Q_2d[THOMPSON_MAX_REGIMES][THOMPSON_MAX_REGIMES];
    float Pi_2d[THOMPSON_MAX_REGIMES][THOMPSON_MAX_REGIMES];

    flat_to_2d(Q_flat, Q_2d, K);
    result = thompson_sampler_exploit(sampler, Q_2d, Pi_2d);
    copy_2d_to_flat(Pi_2d, Pi_flat, K);

    return result;
}

/*═══════════════════════════════════════════════════════════════════════════
 * QUERIES
 *═══════════════════════════════════════════════════════════════════════════*/

float thompson_sampler_get_explore_ratio(const ThompsonSampler *sampler)
{
    if (!sampler || sampler->total_samples == 0)
        return 0.0f;
    return (float)sampler->explore_count / (float)sampler->total_samples;
}

int thompson_sampler_get_total_samples(const ThompsonSampler *sampler)
{
    return sampler ? sampler->total_samples : 0;
}

bool thompson_sampler_would_explore(const ThompsonSampler *sampler, float row_sum)
{
    if (!sampler)
        return false;
    return row_sum < sampler->config.exploit_threshold;
}

/*═══════════════════════════════════════════════════════════════════════════
 * DIAGNOSTICS
 *═══════════════════════════════════════════════════════════════════════════*/

void thompson_sampler_print_state(const ThompsonSampler *sampler)
{
    if (!sampler)
    {
        printf("ThompsonSampler: NULL\n");
        return;
    }

    printf("═══════════════════════════════════════════════════════════\n");
    printf("THOMPSON SAMPLER STATE\n");
    printf("═══════════════════════════════════════════════════════════\n");
    printf("Initialized:      %s\n", sampler->initialized ? "true" : "false");
    printf("Total samples:    %d\n", sampler->total_samples);
    printf("Explore count:    %d\n", sampler->explore_count);
    printf("Exploit count:    %d\n", sampler->exploit_count);
    printf("Explore ratio:    %.2f%%\n",
           thompson_sampler_get_explore_ratio(sampler) * 100.0f);
    printf("═══════════════════════════════════════════════════════════\n");
}

void thompson_sampler_print_config(const ThompsonSamplerConfig *cfg)
{
    if (!cfg)
    {
        printf("ThompsonSamplerConfig: NULL\n");
        return;
    }

    printf("═══════════════════════════════════════════════════════════\n");
    printf("THOMPSON SAMPLER CONFIG\n");
    printf("═══════════════════════════════════════════════════════════\n");
    printf("n_regimes:         %d\n", cfg->n_regimes);
    printf("exploit_threshold: %.1f\n", cfg->exploit_threshold);
    printf("min_concentration: %.3f\n", cfg->min_concentration);
    printf("floor_probability: %.1e\n", cfg->floor_probability);
    printf("seed:              0x%llX\n", (unsigned long long)cfg->seed);
    printf("═══════════════════════════════════════════════════════════\n");
}