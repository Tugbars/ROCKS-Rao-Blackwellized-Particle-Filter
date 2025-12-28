/*
 * ═══════════════════════════════════════════════════════════════════════════
 * RBPF Parameter Learning: Storvik with Adaptive Forgetting (LEAN)
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Full Bayesian update every tick. No sleeping, no throttling.
 *
 * Normal-Inverse-Gamma conjugate prior for (μ, σ²):
 *   μ | σ² ~ N(m, σ²/κ)
 *   σ²     ~ IG(α, β)
 *
 * Sufficient statistics: (m, κ, α, β) updated with exponential forgetting.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 */

#ifndef RBPF_PARAM_LEARN_H
#define RBPF_PARAM_LEARN_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif

    /*═══════════════════════════════════════════════════════════════════════════
     * CONFIGURATION
     *═══════════════════════════════════════════════════════════════════════════*/

#define PARAM_LEARN_MAX_REGIMES 8
#define PARAM_LEARN_MAX_PARTICLES 4096
#define PL_CACHE_LINE 64
#define PL_RNG_BUFFER_SIZE 1024

    /* Precision */
    typedef double param_real;

/* Force inline */
#if defined(_MSC_VER)
#define PL_FORCE_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define PL_FORCE_INLINE __attribute__((always_inline)) inline
#else
#define PL_FORCE_INLINE inline
#endif

/* Restrict */
#if defined(_MSC_VER)
#define PL_RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
#define PL_RESTRICT __restrict__
#else
#define PL_RESTRICT
#endif

    /*═══════════════════════════════════════════════════════════════════════════
     * STRUCTURES
     *═══════════════════════════════════════════════════════════════════════════*/

    /**
     * Configuration
     */
    typedef struct
    {
        /* Forgetting */
        bool enable_forgetting;
        param_real forgetting_lambda;      /* Global discount factor (fallback) */
        param_real forgetting_kappa_floor; /* Prevent posterior collapse */
        param_real forgetting_alpha_floor; /* Keep inverse-gamma proper */

        /* Per-regime forgetting (if enable_regime_adaptive_forgetting) */
        bool enable_regime_adaptive_forgetting;
        param_real forgetting_lambda_regime[PARAM_LEARN_MAX_REGIMES];

        /* Bounds */
        param_real sigma_floor_mult;
        param_real sigma_ceil_mult;
        param_real mu_drift_max;

        /* Priors */
        param_real prior_strength;
        uint64_t rng_seed;

        /* Compatibility stubs (not used in lean version - always update every tick) */
        int sample_interval[PARAM_LEARN_MAX_REGIMES]; /* Ignored - always 1 */
        bool sample_on_regime_change;                 /* Ignored - always sample */
        bool sample_on_structural_break;              /* Ignored - always sample */
        bool sample_after_resampling;                 /* Ignored - always sample */
        bool enable_global_tick_skip;                 /* Ignored - never skip */
        int global_skip_modulo;                       /* Ignored */
    } ParamLearnConfig;

    /**
     * Per-regime prior specification
     */
    typedef struct
    {
        param_real m;           /* Prior mean for μ */
        param_real kappa;       /* Prior precision for μ */
        param_real alpha;       /* IG shape for σ² */
        param_real beta;        /* IG scale for σ² */
        param_real phi;         /* AR(1) coefficient (fixed, not learned) */
        param_real sigma_prior; /* Prior σ (for bounds) */

        /* Precomputed */
        param_real one_minus_phi;
        param_real one_minus_phi_sq;
        param_real inv_one_minus_phi;
        param_real inv_one_minus_phi_sq;
    } RegimePrior;

    /**
     * Sufficient statistics SoA layout
     * Indexed as: particle_idx * n_regimes + regime
     */
    typedef struct
    {
        param_real *m;             /* Posterior mean for μ */
        param_real *kappa;         /* Posterior precision for μ */
        param_real *alpha;         /* IG shape */
        param_real *beta;          /* IG scale */
        param_real *mu_cached;     /* Last sampled μ */
        param_real *sigma2_cached; /* Last sampled σ² */
        param_real *sigma_cached;  /* Last sampled σ */
        int *n_obs;                /* Observation count */
        int *ticks_since_sample;   /* Ticks since last sample (for sleeping) */
    } StorvikSoA;

    /**
     * Entropy buffer for batched RNG
     */
    typedef struct
    {
        param_real *normal;
        param_real *uniform;
        int normal_cursor;
        int uniform_cursor;
        int buffer_size;
        uint64_t rng_state[2];
        void *mkl_stream; /* VSLStreamStatePtr if MKL */
    } EntropyBuffer;

    /**
     * Particle info passed from RBPF
     */
    typedef struct
    {
        int regime;
        int prev_regime;
        param_real ell;     /* Current log-vol */
        param_real ell_lag; /* Previous log-vol */
        param_real weight;
    } ParticleInfo;

    /**
     * Output parameters
     */
    typedef struct
    {
        param_real mu;
        param_real phi;
        param_real sigma;
        param_real sigma2;
        int n_obs;
    } RegimeParams;

    /**
     * Main learner state
     */
    typedef struct
    {
        ParamLearnConfig config;
        int n_regimes;
        int n_particles;
        int storvik_total_size;

        /* Double-buffered SoA */
        StorvikSoA storvik[2];
        int active_buffer;

        /* Priors */
        RegimePrior priors[PARAM_LEARN_MAX_REGIMES];

        /* RNG */
        uint64_t rng[2];
        EntropyBuffer entropy;

        /* State */
        int tick;
        bool structural_break_flag;

        /* Diagnostics */
        uint64_t total_stat_updates;
        uint64_t total_samples_drawn;
        uint64_t total_resets;

        /* Compatibility stubs (not used in lean version) */
        uint64_t samples_skipped_load; /* Always 0 */
        uint64_t ticks_skipped_global; /* Always 0 */
    } ParamLearner;

    /*═══════════════════════════════════════════════════════════════════════════
     * API
     *═══════════════════════════════════════════════════════════════════════════*/

    /* Configuration */
    ParamLearnConfig param_learn_config_defaults(void);

    /* Lifecycle */
    int param_learn_init(ParamLearner *learner,
                         const ParamLearnConfig *config,
                         int n_particles,
                         int n_regimes);
    void param_learn_free(ParamLearner *learner);
    void param_learn_reset(ParamLearner *learner);

    /* Prior specification */
    void param_learn_set_prior(ParamLearner *learner, int regime,
                               param_real mu, param_real phi, param_real sigma);
    void param_learn_broadcast_priors(ParamLearner *learner);

    /* Forgetting */
    void param_learn_set_forgetting(ParamLearner *learner, bool enable, param_real lambda);
    param_real param_learn_get_forgetting_lambda(const ParamLearner *learner, int regime);
    param_real param_learn_get_effective_sample_size(const ParamLearner *learner, int regime);

    /* Per-regime forgetting (compatibility stub - uses global λ) */
    void param_learn_set_regime_forgetting(ParamLearner *learner, int regime, param_real lambda);

    /* Main update (call every tick) */
    void param_learn_update(ParamLearner *learner,
                            const ParticleInfo *particles,
                            int n);

    /* Structural break */
    void param_learn_signal_structural_break(ParamLearner *learner);
    void param_learn_reset_to_priors(ParamLearner *learner);

    /* Resampling */
    void param_learn_apply_resampling(ParamLearner *learner, const int *ancestors, int n);

    /* Get parameters */
    void param_learn_get_params(const ParamLearner *learner,
                                int particle_idx, int regime,
                                RegimeParams *params);

    /* Accessors */
    StorvikSoA *param_learn_get_active_soa(ParamLearner *learner);
    const StorvikSoA *param_learn_get_active_soa_const(const ParamLearner *learner);

    /* Diagnostics */
    void param_learn_print_summary(const ParamLearner *learner);
    void param_learn_print_regime_stats(const ParamLearner *learner, int regime);

#ifdef __cplusplus
}
#endif

#endif /* RBPF_PARAM_LEARN_H */