/*
   * ═══════════════════════════════════════════════════════════════════════════
   * RBPF Parameter Learning: Storvik with Adaptive Forgetting (LEAN)
   * ═══════════════════════════════════════════════════════════════════════════
   *
   * Full Bayesian update every tick. No sleeping, no throttling.
   *
   * Core operations:
   *   - Update sufficient statistics (m, κ, α, β) with forgetting
   *   - Sample parameters (μ, σ) from posterior
   *   - Handle resampling (double-buffer swap)
   *   - Reset to priors on structural break
   *
   * ═══════════════════════════════════════════════════════════════════════════
   */

#include "rbpf_param_learn.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/*═══════════════════════════════════════════════════════════════════════════
 * CONSTANTS
 *═══════════════════════════════════════════════════════════════════════════*/

#define PHI_MAX 0.995
#define ONE_MINUS_PHI_MIN 0.005

#if defined(_MSC_VER)
#define THREAD_LOCAL __declspec(thread)
#elif defined(__GNUC__) || defined(__clang__)
#define THREAD_LOCAL __thread
#else
#define THREAD_LOCAL
#endif

#if defined(__GNUC__) || defined(__clang__)
#define STORE_FENCE() __sync_synchronize()
#elif defined(_MSC_VER)
#include <intrin.h>
#define STORE_FENCE() _mm_sfence()
#else
#define STORE_FENCE()
#endif

    /*═══════════════════════════════════════════════════════════════════════════
     * ALIGNED MEMORY
     *═══════════════════════════════════════════════════════════════════════════*/

    static void *aligned_alloc_64(size_t size)
{
    size = (size + PL_CACHE_LINE - 1) & ~(size_t)(PL_CACHE_LINE - 1);
#if defined(_MSC_VER)
    return _aligned_malloc(size, PL_CACHE_LINE);
#else
    void *ptr = NULL;
    if (posix_memalign(&ptr, PL_CACHE_LINE, size) != 0)
        return NULL;
    return ptr;
#endif
}

static void aligned_free_64(void *ptr)
{
#if defined(_MSC_VER)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

/*═══════════════════════════════════════════════════════════════════════════
 * RNG: xoroshiro128+
 *═══════════════════════════════════════════════════════════════════════════*/

static inline uint64_t rotl(const uint64_t x, int k)
{
    return (x << k) | (x >> (64 - k));
}

static inline uint64_t xoro_next(uint64_t *s)
{
    const uint64_t s0 = s[0];
    uint64_t s1 = s[1];
    const uint64_t result = s0 + s1;
    s1 ^= s0;
    s[0] = rotl(s0, 24) ^ s1 ^ (s1 << 16);
    s[1] = rotl(s1, 37);
    return result;
}

static inline param_real rand_u01(uint64_t *s)
{
    return (xoro_next(s) >> 11) * (1.0 / 9007199254740992.0);
}

static void rng_seed(uint64_t *s, uint64_t seed)
{
    uint64_t z = seed;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    s[0] = z ^ (z >> 31);
    z = seed + 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    s[1] = z ^ (z >> 31);
}

/*═══════════════════════════════════════════════════════════════════════════
 * FAST NORMAL SAMPLER (Box-Muller with caching)
 *═══════════════════════════════════════════════════════════════════════════*/

typedef struct
{
    param_real cached;
    bool has_cached;
} NormalCache;

static THREAD_LOCAL NormalCache g_normal_cache = {0, false};

static inline param_real rand_normal(uint64_t *s)
{
    if (g_normal_cache.has_cached)
    {
        g_normal_cache.has_cached = false;
        return g_normal_cache.cached;
    }

    param_real u, v, s2;
    do
    {
        u = 2.0 * rand_u01(s) - 1.0;
        v = 2.0 * rand_u01(s) - 1.0;
        s2 = u * u + v * v;
    } while (s2 >= 1.0 || s2 == 0.0);

    param_real mult = sqrt(-2.0 * log(s2) / s2);
    g_normal_cache.cached = v * mult;
    g_normal_cache.has_cached = true;
    return u * mult;
}

/*═══════════════════════════════════════════════════════════════════════════
 * BATCH RNG (MKL or fallback)
 *═══════════════════════════════════════════════════════════════════════════*/

#ifdef PARAM_LEARN_USE_MKL
#include <mkl_vsl.h>

static void entropy_buffer_fill(EntropyBuffer *eb, int n_normal, int n_uniform)
{
    if (n_normal > eb->buffer_size)
        n_normal = eb->buffer_size;
    if (n_uniform > eb->buffer_size)
        n_uniform = eb->buffer_size;

    VSLStreamStatePtr stream = (VSLStreamStatePtr)eb->mkl_stream;
    if (!stream)
    {
        vslNewStream(&stream, VSL_BRNG_MT19937, (unsigned int)eb->rng_state[0]);
        eb->mkl_stream = stream;
    }

    vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, n_normal, eb->normal, 0.0, 1.0);
    vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, n_uniform, eb->uniform, 0.0, 1.0);
    eb->normal_cursor = 0;
    eb->uniform_cursor = 0;
}
#else
static void entropy_buffer_fill(EntropyBuffer *eb, int n_normal, int n_uniform)
{
    if (n_normal > eb->buffer_size)
        n_normal = eb->buffer_size;
    if (n_uniform > eb->buffer_size)
        n_uniform = eb->buffer_size;

    int i = 0;
    while (i < n_normal)
    {
        param_real u = 2.0 * rand_u01(eb->rng_state) - 1.0;
        param_real v = 2.0 * rand_u01(eb->rng_state) - 1.0;
        param_real s2 = u * u + v * v;
        if (s2 < 1.0 && s2 > 0.0)
        {
            param_real mult = sqrt(-2.0 * log(s2) / s2);
            eb->normal[i++] = u * mult;
            if (i < n_normal)
                eb->normal[i++] = v * mult;
        }
    }

    for (i = 0; i < n_uniform; i++)
    {
        eb->uniform[i] = rand_u01(eb->rng_state);
    }
    eb->normal_cursor = 0;
    eb->uniform_cursor = 0;
}
#endif

static inline param_real entropy_normal(EntropyBuffer *eb)
{
    if (eb->normal_cursor >= eb->buffer_size)
    {
        entropy_buffer_fill(eb, eb->buffer_size, 0);
    }
    return eb->normal[eb->normal_cursor++];
}

static inline param_real entropy_uniform(EntropyBuffer *eb)
{
    if (eb->uniform_cursor >= eb->buffer_size)
    {
        entropy_buffer_fill(eb, 0, eb->buffer_size);
    }
    return eb->uniform[eb->uniform_cursor++];
}

/*═══════════════════════════════════════════════════════════════════════════
 * CONFIGURATION
 *═══════════════════════════════════════════════════════════════════════════*/

ParamLearnConfig param_learn_config_defaults(void)
{
    ParamLearnConfig cfg;
    memset(&cfg, 0, sizeof(cfg));

    /* Adaptive forgetting */
    cfg.enable_forgetting = true;
    cfg.forgetting_lambda = 0.997;    /* Global fallback */
    cfg.forgetting_kappa_floor = 5.0; /* Prevent posterior collapse */
    cfg.forgetting_alpha_floor = 3.0; /* Keep inverse-gamma proper */

    /* Per-regime forgetting - default to global */
    cfg.enable_regime_adaptive_forgetting = false;
    for (int r = 0; r < PARAM_LEARN_MAX_REGIMES; r++)
    {
        cfg.forgetting_lambda_regime[r] = cfg.forgetting_lambda;
    }

    /* Bounds */
    cfg.sigma_floor_mult = 0.1;
    cfg.sigma_ceil_mult = 5.0;
    cfg.mu_drift_max = 1.0;

    /* Priors */
    cfg.prior_strength = 10.0;
    cfg.rng_seed = 42;

    /* Sleeping intervals - sample less frequently in calm regimes */
    cfg.sample_interval[0] = 50; /* Calm: every 50 ticks */
    cfg.sample_interval[1] = 20; /* Mild: every 20 ticks */
    cfg.sample_interval[2] = 5;  /* Trend: every 5 ticks */
    cfg.sample_interval[3] = 1;  /* Crisis: every tick */
    for (int r = 4; r < PARAM_LEARN_MAX_REGIMES; r++)
    {
        cfg.sample_interval[r] = 1;
    }
    cfg.sample_on_regime_change = true;
    cfg.sample_on_structural_break = true;
    cfg.sample_after_resampling = true;
    cfg.enable_global_tick_skip = false;
    cfg.global_skip_modulo = 1;

    return cfg;
}

/*═══════════════════════════════════════════════════════════════════════════
 * SOA ALLOCATION
 *═══════════════════════════════════════════════════════════════════════════*/

static int storvik_soa_alloc(StorvikSoA *soa, int total_size)
{
    size_t arr_size = total_size * sizeof(param_real);
    size_t int_size = total_size * sizeof(int);

    soa->m = (param_real *)aligned_alloc_64(arr_size);
    soa->kappa = (param_real *)aligned_alloc_64(arr_size);
    soa->alpha = (param_real *)aligned_alloc_64(arr_size);
    soa->beta = (param_real *)aligned_alloc_64(arr_size);
    soa->mu_cached = (param_real *)aligned_alloc_64(arr_size);
    soa->sigma2_cached = (param_real *)aligned_alloc_64(arr_size);
    soa->sigma_cached = (param_real *)aligned_alloc_64(arr_size);
    soa->n_obs = (int *)aligned_alloc_64(int_size);
    soa->ticks_since_sample = (int *)aligned_alloc_64(int_size);

    if (!soa->m || !soa->kappa || !soa->alpha || !soa->beta ||
        !soa->mu_cached || !soa->sigma2_cached || !soa->sigma_cached ||
        !soa->n_obs || !soa->ticks_since_sample)
    {
        return -1;
    }

    memset(soa->m, 0, arr_size);
    memset(soa->kappa, 0, arr_size);
    memset(soa->alpha, 0, arr_size);
    memset(soa->beta, 0, arr_size);
    memset(soa->mu_cached, 0, arr_size);
    memset(soa->sigma2_cached, 0, arr_size);
    memset(soa->sigma_cached, 0, arr_size);
    memset(soa->n_obs, 0, int_size);
    memset(soa->ticks_since_sample, 0, int_size);

    return 0;
}

static void storvik_soa_free(StorvikSoA *soa)
{
    aligned_free_64(soa->m);
    aligned_free_64(soa->kappa);
    aligned_free_64(soa->alpha);
    aligned_free_64(soa->beta);
    aligned_free_64(soa->mu_cached);
    aligned_free_64(soa->sigma2_cached);
    aligned_free_64(soa->sigma_cached);
    aligned_free_64(soa->n_obs);
    aligned_free_64(soa->ticks_since_sample);
    memset(soa, 0, sizeof(*soa));
}

/*═══════════════════════════════════════════════════════════════════════════
 * LIFECYCLE
 *═══════════════════════════════════════════════════════════════════════════*/

int param_learn_init(ParamLearner *learner,
                     const ParamLearnConfig *config,
                     int n_particles,
                     int n_regimes)
{
    if (!learner || n_regimes < 1 || n_regimes > PARAM_LEARN_MAX_REGIMES ||
        n_particles < 1 || n_particles > PARAM_LEARN_MAX_PARTICLES)
    {
        return -1;
    }

    memset(learner, 0, sizeof(*learner));

    learner->config = config ? *config : param_learn_config_defaults();
    learner->n_regimes = n_regimes;
    learner->n_particles = n_particles;
    learner->storvik_total_size = n_particles * n_regimes;
    learner->active_buffer = 0;

    rng_seed(learner->rng, learner->config.rng_seed);

    /* Allocate double buffers */
    if (storvik_soa_alloc(&learner->storvik[0], learner->storvik_total_size) < 0)
    {
        return -1;
    }
    if (storvik_soa_alloc(&learner->storvik[1], learner->storvik_total_size) < 0)
    {
        storvik_soa_free(&learner->storvik[0]);
        return -1;
    }

    /* Entropy buffer */
    learner->entropy.buffer_size = PL_RNG_BUFFER_SIZE;
    learner->entropy.normal = (param_real *)aligned_alloc_64(PL_RNG_BUFFER_SIZE * sizeof(param_real));
    learner->entropy.uniform = (param_real *)aligned_alloc_64(PL_RNG_BUFFER_SIZE * sizeof(param_real));
    if (!learner->entropy.normal || !learner->entropy.uniform)
    {
        storvik_soa_free(&learner->storvik[0]);
        storvik_soa_free(&learner->storvik[1]);
        return -1;
    }
    learner->entropy.rng_state[0] = learner->rng[0];
    learner->entropy.rng_state[1] = learner->rng[1];

#ifdef PARAM_LEARN_USE_MKL
    VSLStreamStatePtr stream;
    if (vslNewStream(&stream, VSL_BRNG_MT19937, (unsigned int)learner->rng[0]) != VSL_STATUS_OK)
    {
        storvik_soa_free(&learner->storvik[0]);
        storvik_soa_free(&learner->storvik[1]);
        aligned_free_64(learner->entropy.normal);
        aligned_free_64(learner->entropy.uniform);
        return -1;
    }
    learner->entropy.mkl_stream = stream;
#endif

    entropy_buffer_fill(&learner->entropy, PL_RNG_BUFFER_SIZE, PL_RNG_BUFFER_SIZE);

    /* Initialize default priors */
    for (int r = 0; r < n_regimes; r++)
    {
        RegimePrior *p = &learner->priors[r];
        p->m = -2.0 + 0.5 * r;
        p->kappa = learner->config.prior_strength;
        p->alpha = learner->config.prior_strength / 2.0 + 1.0;
        p->beta = (p->alpha - 1.0) * 0.01 * (1.0 + 0.5 * r);

        param_real phi_raw = 0.98 - 0.03 * r;
        p->phi = fmin(PHI_MAX, phi_raw);
        p->sigma_prior = sqrt(p->beta / (p->alpha - 1.0));

        p->one_minus_phi = fmax(ONE_MINUS_PHI_MIN, 1.0 - p->phi);
        p->one_minus_phi_sq = p->one_minus_phi * p->one_minus_phi;
        p->inv_one_minus_phi = 1.0 / p->one_minus_phi;
        p->inv_one_minus_phi_sq = 1.0 / p->one_minus_phi_sq;
    }

    return 0;
}

void param_learn_free(ParamLearner *learner)
{
    if (!learner)
        return;

#ifdef PARAM_LEARN_USE_MKL
    if (learner->entropy.mkl_stream)
    {
        vslDeleteStream((VSLStreamStatePtr *)&learner->entropy.mkl_stream);
    }
#endif

    storvik_soa_free(&learner->storvik[0]);
    storvik_soa_free(&learner->storvik[1]);
    aligned_free_64(learner->entropy.normal);
    aligned_free_64(learner->entropy.uniform);
    memset(learner, 0, sizeof(*learner));
}

void param_learn_reset(ParamLearner *learner)
{
    if (!learner)
        return;

    learner->tick = 0;
    learner->structural_break_flag = false;
    learner->active_buffer = 0;
    learner->total_stat_updates = 0;
    learner->total_samples_drawn = 0;
    learner->total_resets = 0;

    param_learn_broadcast_priors(learner);
}

/*═══════════════════════════════════════════════════════════════════════════
 * PRIOR SPECIFICATION
 *═══════════════════════════════════════════════════════════════════════════*/

void param_learn_set_prior(ParamLearner *learner, int regime,
                           param_real mu, param_real phi, param_real sigma)
{
    if (!learner || regime < 0 || regime >= learner->n_regimes)
        return;

    RegimePrior *p = &learner->priors[regime];
    param_real n = learner->config.prior_strength;

    p->m = mu;
    p->kappa = n;
    p->alpha = n / 2.0 + 1.0;
    p->beta = (p->alpha - 1.0) * sigma * sigma;
    p->phi = fmin(PHI_MAX, phi);
    p->sigma_prior = sigma;

    p->one_minus_phi = fmax(ONE_MINUS_PHI_MIN, 1.0 - p->phi);
    p->one_minus_phi_sq = p->one_minus_phi * p->one_minus_phi;
    p->inv_one_minus_phi = 1.0 / p->one_minus_phi;
    p->inv_one_minus_phi_sq = 1.0 / p->one_minus_phi_sq;
}

void param_learn_broadcast_priors(ParamLearner *learner)
{
    if (!learner)
        return;

    for (int buf = 0; buf < 2; buf++)
    {
        StorvikSoA *soa = &learner->storvik[buf];
        int nr = learner->n_regimes;

        for (int i = 0; i < learner->n_particles; i++)
        {
            for (int r = 0; r < nr; r++)
            {
                int idx = i * nr + r;
                const RegimePrior *p = &learner->priors[r];

                soa->m[idx] = p->m;
                soa->kappa[idx] = p->kappa;
                soa->alpha[idx] = p->alpha;
                soa->beta[idx] = p->beta;
                soa->sigma2_cached[idx] = p->beta / (p->alpha - 1.0 + 1e-10);
                soa->sigma_cached[idx] = sqrt(soa->sigma2_cached[idx]);
                soa->mu_cached[idx] = p->m;
                soa->n_obs[idx] = 0;
                soa->ticks_since_sample[idx] = 0;
            }
        }
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * FORGETTING API
 *═══════════════════════════════════════════════════════════════════════════*/

void param_learn_set_forgetting(ParamLearner *learner, bool enable, param_real lambda)
{
    if (!learner)
        return;
    learner->config.enable_forgetting = enable;
    if (lambda > 0.0 && lambda <= 1.0)
    {
        learner->config.forgetting_lambda = lambda;
    }
}

param_real param_learn_get_forgetting_lambda(const ParamLearner *learner, int regime)
{
    if (!learner || !learner->config.enable_forgetting)
        return 1.0;

    /* Per-regime if enabled and valid regime */
    if (learner->config.enable_regime_adaptive_forgetting &&
        regime >= 0 && regime < learner->n_regimes)
    {
        return learner->config.forgetting_lambda_regime[regime];
    }

    return learner->config.forgetting_lambda;
}

param_real param_learn_get_effective_sample_size(const ParamLearner *learner, int regime)
{
    if (!learner || !learner->config.enable_forgetting)
        return (param_real)learner->tick;
    param_real lambda = param_learn_get_forgetting_lambda(learner, regime);
    if (lambda >= 1.0)
        return (param_real)learner->tick;
    return 1.0 / (1.0 - lambda);
}

/* Per-regime forgetting */
void param_learn_set_regime_forgetting(ParamLearner *learner, int regime, param_real lambda)
{
    if (!learner || regime < 0 || regime >= PARAM_LEARN_MAX_REGIMES)
        return;
    if (lambda > 0.0 && lambda <= 1.0)
    {
        learner->config.forgetting_lambda_regime[regime] = lambda;
        learner->config.enable_regime_adaptive_forgetting = true;
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * CORE: SUFFICIENT STATISTICS UPDATE WITH FORGETTING
 *═══════════════════════════════════════════════════════════════════════════*/

static PL_FORCE_INLINE void storvik_update_single(
    StorvikSoA *soa,
    int idx,
    param_real ell,
    param_real ell_lag,
    const RegimePrior *prior,
    param_real lambda,
    param_real kappa_floor,
    param_real alpha_floor)
{
    /* Transform observation */
    param_real z = ell - prior->phi * ell_lag;
    param_real z_scaled = z * prior->inv_one_minus_phi;
    param_real var_scale = prior->inv_one_minus_phi_sq;

    /* Load current stats */
    param_real kappa_old = soa->kappa[idx];
    param_real m_old = soa->m[idx];
    param_real alpha_old = soa->alpha[idx];
    param_real beta_old = soa->beta[idx];

    /* Discount old stats */
    param_real kappa_disc = lambda * kappa_old;
    if (kappa_disc < kappa_floor)
        kappa_disc = kappa_floor;

    param_real alpha_excess = alpha_old - alpha_floor;
    if (alpha_excess < 0)
        alpha_excess = 0;
    param_real alpha_disc = alpha_floor + lambda * alpha_excess;

    param_real beta_disc = lambda * beta_old;
    param_real m_weighted = kappa_disc * m_old;

    /* Accumulate new observation */
    param_real kappa_new = kappa_disc + prior->one_minus_phi_sq;
    param_real m_new = (m_weighted + z * prior->one_minus_phi) / kappa_new;
    param_real alpha_new = alpha_disc + 0.5;

    param_real diff = z_scaled - m_old;
    param_real total_var = 1.0 / kappa_disc + var_scale;
    param_real beta_new = beta_disc + 0.5 * diff * diff / total_var;

    /* Store */
    soa->m[idx] = m_new;
    soa->kappa[idx] = kappa_new;
    soa->alpha[idx] = alpha_new;
    soa->beta[idx] = beta_new;
    soa->n_obs[idx]++;
}

/*═══════════════════════════════════════════════════════════════════════════
 * CORE: PARAMETER SAMPLING
 *═══════════════════════════════════════════════════════════════════════════*/

static void storvik_sample(ParamLearner *learner, StorvikSoA *soa, int idx, int regime)
{
    const RegimePrior *p = &learner->priors[regime];
    const ParamLearnConfig *cfg = &learner->config;
    EntropyBuffer *eb = &learner->entropy;

    param_real alpha = soa->alpha[idx];
    param_real beta = soa->beta[idx];

    /* Gamma sampling (Marsaglia-Tsang) */
    param_real d = alpha - 1.0 / 3.0;
    param_real c = 1.0 / sqrt(9.0 * d);
    param_real gamma_sample;

    for (;;)
    {
        param_real x = entropy_normal(eb);
        param_real v = 1.0 + c * x;
        if (v > 0)
        {
            v = v * v * v;
            param_real u = entropy_uniform(eb);
            if (u < 1.0 - 0.0331 * (x * x) * (x * x) ||
                log(u) < 0.5 * x * x + d * (1.0 - v + log(v)))
            {
                gamma_sample = d * v;
                break;
            }
        }
    }

    param_real sigma2 = beta / gamma_sample;

    /* Clamp σ² */
    param_real sigma2_prior = p->sigma_prior * p->sigma_prior;
    param_real sigma2_min = cfg->sigma_floor_mult * cfg->sigma_floor_mult * sigma2_prior;
    param_real sigma2_max = cfg->sigma_ceil_mult * cfg->sigma_ceil_mult * sigma2_prior;
    sigma2 = fmax(sigma2_min, fmin(sigma2_max, sigma2));

    /* Sample μ | σ² */
    param_real mu_std = sqrt(sigma2 / soa->kappa[idx]);
    param_real mu = soa->m[idx] + mu_std * entropy_normal(eb);

    /* Clamp μ drift */
    param_real mu_drift = mu - p->m;
    if (fabs(mu_drift) > cfg->mu_drift_max)
    {
        mu = p->m + (mu_drift > 0 ? cfg->mu_drift_max : -cfg->mu_drift_max);
    }

    /* Store */
    soa->mu_cached[idx] = mu;
    soa->sigma2_cached[idx] = sigma2;
    soa->sigma_cached[idx] = sqrt(sigma2);

    learner->total_samples_drawn++;
}

/*═══════════════════════════════════════════════════════════════════════════
 * MAIN UPDATE (FULL BAYESIAN - EVERY TICK)
 *═══════════════════════════════════════════════════════════════════════════*/

void param_learn_update(ParamLearner *learner,
                        const ParticleInfo *particles,
                        int n)
{
    if (!learner || !particles || n < 1)
        return;

    const ParamLearnConfig *cfg = &learner->config;
    learner->tick++;

    StorvikSoA *soa = param_learn_get_active_soa(learner);
    const int nr = learner->n_regimes;
    const int np = learner->n_particles;

    /* Forgetting parameters */
    param_real kappa_floor = cfg->forgetting_kappa_floor;
    param_real alpha_floor = cfg->forgetting_alpha_floor;

    /* Refill entropy if needed */
    int samples_needed = n * 4;
    if (learner->entropy.normal_cursor + samples_needed > learner->entropy.buffer_size)
    {
        entropy_buffer_fill(&learner->entropy, PL_RNG_BUFFER_SIZE, PL_RNG_BUFFER_SIZE);
    }

    /* Process each particle */
    for (int i = 0; i < n && i < np; i++)
    {
        const ParticleInfo *p = &particles[i];
        int r = p->regime;
        if (r < 0 || r >= nr)
            continue;

        int idx = i * nr + r;
        const RegimePrior *prior = &learner->priors[r];

        /* Get per-regime lambda */
        param_real lambda = param_learn_get_forgetting_lambda(learner, r);

        /* ALWAYS update sufficient statistics */
        storvik_update_single(soa, idx, p->ell, p->ell_lag, prior,
                              lambda, kappa_floor, alpha_floor);
        learner->total_stat_updates++;

        /* SLEEPING: Only sample when interval is reached */
        soa->ticks_since_sample[idx]++;
        int interval = cfg->sample_interval[r];
        if (interval <= 1 || soa->ticks_since_sample[idx] >= interval)
        {
            storvik_sample(learner, soa, idx, r);
            soa->ticks_since_sample[idx] = 0;
        }
    }

    /* Clear structural break flag */
    if (learner->structural_break_flag)
    {
        learner->structural_break_flag = false;
    }
}

void param_learn_signal_structural_break(ParamLearner *learner)
{
    if (learner)
    {
        learner->structural_break_flag = true;
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * RESAMPLING (DOUBLE-BUFFER SWAP)
 *═══════════════════════════════════════════════════════════════════════════*/

void param_learn_apply_resampling(ParamLearner *learner, const int *ancestors, int n)
{
    if (!learner || !ancestors)
        return;

    const int nr = learner->n_regimes;
    const int np = learner->n_particles;

    int active = learner->active_buffer;
    int inactive = 1 - active;

    StorvikSoA *src = &learner->storvik[active];
    StorvikSoA *dst = &learner->storvik[inactive];

    /* Gather from ancestors */
    for (int i = 0; i < n && i < np; i++)
    {
        int anc = ancestors[i];
        if (anc < 0 || anc >= np)
            anc = i;

        int dst_base = i * nr;
        int src_base = anc * nr;

        memcpy(&dst->m[dst_base], &src->m[src_base], nr * sizeof(param_real));
        memcpy(&dst->kappa[dst_base], &src->kappa[src_base], nr * sizeof(param_real));
        memcpy(&dst->alpha[dst_base], &src->alpha[src_base], nr * sizeof(param_real));
        memcpy(&dst->beta[dst_base], &src->beta[src_base], nr * sizeof(param_real));
        memcpy(&dst->mu_cached[dst_base], &src->mu_cached[src_base], nr * sizeof(param_real));
        memcpy(&dst->sigma2_cached[dst_base], &src->sigma2_cached[src_base], nr * sizeof(param_real));
        memcpy(&dst->sigma_cached[dst_base], &src->sigma_cached[src_base], nr * sizeof(param_real));
        memcpy(&dst->n_obs[dst_base], &src->n_obs[src_base], nr * sizeof(int));
        memcpy(&dst->ticks_since_sample[dst_base], &src->ticks_since_sample[src_base], nr * sizeof(int));
    }

    /* Swap */
    learner->active_buffer = inactive;
    STORE_FENCE();
}

/*═══════════════════════════════════════════════════════════════════════════
 * RESET TO PRIORS
 *═══════════════════════════════════════════════════════════════════════════*/

void param_learn_reset_to_priors(ParamLearner *learner)
{
    if (!learner)
        return;

    const int np = learner->n_particles;
    const int nr = learner->n_regimes;

    StorvikSoA *soa = param_learn_get_active_soa(learner);

    for (int r = 0; r < nr; r++)
    {
        const RegimePrior *prior = &learner->priors[r];

        param_real m_0 = prior->m;
        param_real kappa_0 = prior->kappa;
        param_real alpha_0 = prior->alpha;
        param_real beta_0 = prior->beta;
        param_real sigma_0 = prior->sigma_prior;
        param_real sigma2_0 = sigma_0 * sigma_0;

        for (int i = 0; i < np; i++)
        {
            int idx = i * nr + r;

            soa->m[idx] = m_0;
            soa->kappa[idx] = kappa_0;
            soa->alpha[idx] = alpha_0;
            soa->beta[idx] = beta_0;
            soa->mu_cached[idx] = m_0;
            soa->sigma_cached[idx] = sigma_0;
            soa->sigma2_cached[idx] = sigma2_0;
            soa->n_obs[idx] = 0;
            soa->ticks_since_sample[idx] = 0;
        }
    }

    learner->structural_break_flag = false;
    learner->total_resets++;
}

/*═══════════════════════════════════════════════════════════════════════════
 * GET PARAMETERS
 *═══════════════════════════════════════════════════════════════════════════*/

void param_learn_get_params(const ParamLearner *learner,
                            int particle_idx, int regime,
                            RegimeParams *params)
{
    if (!learner || !params)
    {
        if (params)
            memset(params, 0, sizeof(*params));
        return;
    }

    if (regime < 0 || regime >= learner->n_regimes ||
        particle_idx < 0 || particle_idx >= learner->n_particles)
    {
        const RegimePrior *p = &learner->priors[regime < learner->n_regimes ? regime : 0];
        params->mu = p->m;
        params->phi = p->phi;
        params->sigma = p->sigma_prior;
        params->sigma2 = p->sigma_prior * p->sigma_prior;
        params->n_obs = 0;
        return;
    }

    const StorvikSoA *soa = param_learn_get_active_soa_const(learner);
    int idx = particle_idx * learner->n_regimes + regime;

    params->mu = soa->mu_cached[idx];
    params->phi = learner->priors[regime].phi;
    params->sigma = soa->sigma_cached[idx];
    params->sigma2 = soa->sigma2_cached[idx];
    params->n_obs = soa->n_obs[idx];
}

/*═══════════════════════════════════════════════════════════════════════════
 * ACCESSORS
 *═══════════════════════════════════════════════════════════════════════════*/

StorvikSoA *param_learn_get_active_soa(ParamLearner *learner)
{
    return &learner->storvik[learner->active_buffer];
}

const StorvikSoA *param_learn_get_active_soa_const(const ParamLearner *learner)
{
    return &learner->storvik[learner->active_buffer];
}

/*═══════════════════════════════════════════════════════════════════════════
 * DIAGNOSTICS
 *═══════════════════════════════════════════════════════════════════════════*/

void param_learn_print_summary(const ParamLearner *learner)
{
    if (!learner)
        return;

    printf("\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  Storvik Parameter Learner (LEAN)\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  Particles: %d   Regimes: %d   Tick: %d\n",
           learner->n_particles, learner->n_regimes, learner->tick);
    printf("  Active buffer: %d\n", learner->active_buffer);
    printf("\n");
    printf("  Statistics:\n");
    printf("    Stat updates:  %llu\n", (unsigned long long)learner->total_stat_updates);
    printf("    Samples drawn: %llu\n", (unsigned long long)learner->total_samples_drawn);
    printf("    Resets:        %llu\n", (unsigned long long)learner->total_resets);
    printf("\n");
    printf("  Forgetting: %s\n", learner->config.enable_forgetting ? "ENABLED" : "DISABLED");
    if (learner->config.enable_forgetting)
    {
        param_real lambda = learner->config.forgetting_lambda;
        printf("    λ = %.4f  (N_eff ≈ %.0f)\n", lambda, 1.0 / (1.0 - lambda));
    }
    printf("═══════════════════════════════════════════════════════════════\n");
}

void param_learn_print_regime_stats(const ParamLearner *learner, int regime)
{
    if (!learner || regime < 0 || regime >= learner->n_regimes)
        return;

    const RegimePrior *p = &learner->priors[regime];
    const StorvikSoA *soa = param_learn_get_active_soa_const(learner);

    printf("\n─── Regime %d ───\n", regime);
    printf("Prior: μ=%.4f, φ=%.4f, σ=%.4f\n", p->m, p->phi, p->sigma_prior);

    double sum_mu = 0, sum_sigma = 0;
    int count = 0;
    for (int i = 0; i < learner->n_particles; i++)
    {
        int idx = i * learner->n_regimes + regime;
        if (soa->n_obs[idx] > 0)
        {
            sum_mu += soa->mu_cached[idx];
            sum_sigma += soa->sigma_cached[idx];
            count++;
        }
    }
    if (count > 0)
    {
        printf("Posterior (avg %d particles): μ=%.4f, σ=%.4f\n",
               count, sum_mu / count, sum_sigma / count);
    }
}