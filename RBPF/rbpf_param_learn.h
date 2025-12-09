/*
 * ═══════════════════════════════════════════════════════════════════════════
 * RBPF Parameter Learning: Sleeping Storvik Implementation (OPTIMIZED)
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * P99 Optimizations Applied:
 *   - Double-buffered StorvikSoA for pointer-swap resampling (no memcpy)
 *   - HFT intervals as default [50, 20, 5, 1]
 *   - Global tick-skip for 90% duty cycle reduction
 *   - Aligned memory for AVX-512
 *
 * Target: P99 < 25μs (down from 60μs)
 *
 * ═══════════════════════════════════════════════════════════════════════════
 */

#ifndef RBPF_PARAM_LEARN_H
#define RBPF_PARAM_LEARN_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

    /*═══════════════════════════════════════════════════════════════════════════
     * PRECISION & CONSTANTS
     *═══════════════════════════════════════════════════════════════════════════*/

#ifndef PARAM_LEARN_REAL
    typedef double param_real;
#else
typedef PARAM_LEARN_REAL param_real;
#endif

#define PARAM_LEARN_MAX_REGIMES 8
#define PARAM_LEARN_MAX_PARTICLES 1024

/* Memory alignment for AVX-512 */
#define PL_CACHE_LINE 64

/* RNG buffer size - larger = fewer refills */
#define PL_RNG_BUFFER_SIZE 4096

/* Global tick-skip modulo (skip N-1 out of N ticks in calm regimes) */
#define PL_GLOBAL_SKIP_MODULO 10

    /*═══════════════════════════════════════════════════════════════════════════
     * METHOD SELECTION
     *═══════════════════════════════════════════════════════════════════════════*/

    typedef enum
    {
        PARAM_LEARN_SLEEPING_STORVIK, /* Primary: full Bayesian, adaptive sampling */
        PARAM_LEARN_EWSS,             /* Comparison: point estimates only          */
        PARAM_LEARN_FIXED             /* No adaptation: use priors                 */
    } ParamLearnMethod;

    /*═══════════════════════════════════════════════════════════════════════════
     * SAMPLING TRIGGERS
     *═══════════════════════════════════════════════════════════════════════════*/

    typedef enum
    {
        SAMPLE_TRIGGER_NONE = 0,
        SAMPLE_TRIGGER_INTERVAL = 1 << 0,
        SAMPLE_TRIGGER_REGIME_CHANGE = 1 << 1,
        SAMPLE_TRIGGER_STRUCTURAL_BREAK = 1 << 2,
        SAMPLE_TRIGGER_FORCED = 1 << 3,
        SAMPLE_TRIGGER_RESAMPLING = 1 << 4,
        SAMPLE_TRIGGER_FIRST = 1 << 5,
    } SampleTrigger;

    /*═══════════════════════════════════════════════════════════════════════════
     * PRIOR SPECIFICATION
     *═══════════════════════════════════════════════════════════════════════════*/

    typedef struct
    {
        param_real m;
        param_real kappa;
        param_real alpha;
        param_real beta;
        param_real phi;
        param_real sigma_prior;

        /* Precomputed (avoid division on hot path) */
        param_real one_minus_phi;
        param_real one_minus_phi_sq;
        param_real inv_one_minus_phi;
        param_real inv_one_minus_phi_sq;
    } RegimePrior;

    /*═══════════════════════════════════════════════════════════════════════════
     * SOA STORAGE: Double-Buffered for Pointer-Swap Resampling
     *
     * OPTIMIZATION: Instead of copying 7 arrays back after resampling,
     * write to inactive buffer and swap pointers.
     *
     * Layout: array[particle_idx * n_regimes + regime_idx]
     *═══════════════════════════════════════════════════════════════════════════*/

/* Force inline for hot path */
#if defined(__GNUC__) || defined(__clang__)
#define PL_FORCE_INLINE __attribute__((always_inline)) inline
#elif defined(_MSC_VER)
#define PL_FORCE_INLINE __forceinline
#else
#define PL_FORCE_INLINE inline
#endif

    typedef struct
    {
        /* NIG Posterior Hyperparameters (updated every tick) */
        param_real *__restrict m;
        param_real *__restrict kappa;
        param_real *__restrict alpha;
        param_real *__restrict beta;

        /* Cached Samples (updated only when "awake") */
        param_real *__restrict mu_cached;
        param_real *__restrict sigma2_cached;
        param_real *__restrict sigma_cached;

        /* Tracking */
        int *__restrict n_obs;
        int *__restrict ticks_since_sample;
    } StorvikSoA;

    /*═══════════════════════════════════════════════════════════════════════════
     * ENTROPY BUFFER
     *═══════════════════════════════════════════════════════════════════════════*/

    typedef struct
    {
        param_real *normal;
        param_real *uniform;
        int normal_cursor;
        int uniform_cursor;
        int buffer_size;
        uint64_t rng_state[2];
#ifdef PARAM_LEARN_USE_MKL
        void *mkl_stream;
#endif
    } EntropyBuffer;

    /*═══════════════════════════════════════════════════════════════════════════
     * EWSS STATISTICS
     *═══════════════════════════════════════════════════════════════════════════*/

    typedef struct
    {
        param_real sum_z;
        param_real sum_z_sq;
        param_real eff_n;
        param_real mu;
        param_real sigma;
    } EWSSStats;

    /*═══════════════════════════════════════════════════════════════════════════
     * CONFIGURATION
     *═══════════════════════════════════════════════════════════════════════════*/

    typedef struct
    {
        ParamLearnMethod method;

        /* Sleeping intervals by regime (HFT defaults: [50, 20, 5, 1]) */
        int sample_interval[PARAM_LEARN_MAX_REGIMES];

        /* Triggers */
        bool sample_on_regime_change;
        bool sample_on_structural_break;
        bool sample_after_resampling;

        /* Load throttling */
        bool enable_load_throttling;
        param_real load_skip_threshold;

        /* EWSS config */
        param_real ewss_lambda;
        param_real ewss_min_eff_n;

        /* Constraints */
        param_real sigma_floor_mult;
        param_real sigma_ceil_mult;
        param_real mu_drift_max;

        /* Prior strength */
        param_real prior_strength;

        /* RNG */
        uint64_t rng_seed;

        /*─────────────────────────────────────────────────────────────────────────
         * P99 OPTIMIZATION: Global tick-skip
         *
         * When enabled, skip entire param_learn_update() call N-1 out of N ticks
         * UNLESS a trigger condition is met (regime change, structural break).
         *
         * This reduces average latency from 19μs to ~2μs (90% skip rate).
         * P99 remains ~19μs (when we do run), but average drops dramatically.
         *───────────────────────────────────────────────────────────────────────*/
        bool enable_global_tick_skip;
        int global_skip_modulo; /* Run every N ticks (default: 10) */

    } ParamLearnConfig;

    /*═══════════════════════════════════════════════════════════════════════════
     * PARTICLE INFO
     *═══════════════════════════════════════════════════════════════════════════*/

    typedef struct
    {
        int regime;
        int prev_regime;
        param_real ell;
        param_real ell_lag;
        param_real weight;
    } ParticleInfo;

    /*═══════════════════════════════════════════════════════════════════════════
     * OUTPUT PARAMETERS
     *═══════════════════════════════════════════════════════════════════════════*/

    typedef struct
    {
        param_real mu;
        param_real phi;
        param_real sigma;
        param_real sigma2;

        param_real mu_post_mean;
        param_real mu_post_std;
        param_real sigma2_post_mean;
        param_real sigma2_post_std;

        int n_obs;
        int ticks_since_sample;
        SampleTrigger last_trigger;
        param_real confidence;
    } RegimeParams;

    /*═══════════════════════════════════════════════════════════════════════════
     * MAIN LEARNER STRUCTURE (with double-buffer support)
     *═══════════════════════════════════════════════════════════════════════════*/

    typedef struct
    {
        ParamLearnConfig config;
        int n_regimes;
        int n_particles;

        RegimePrior priors[PARAM_LEARN_MAX_REGIMES];

        /*─────────────────────────────────────────────────────────────────────────
         * DOUBLE-BUFFERED STORVIK SOA
         *
         * storvik[0] and storvik[1] are alternating buffers.
         * active_buffer indicates which one is currently live.
         *
         * On resampling: write to inactive buffer, then swap.
         * Eliminates 7× memcpy (2.1μs → 0μs)
         *───────────────────────────────────────────────────────────────────────*/
        StorvikSoA storvik[2];
        int active_buffer; /* 0 or 1 */
        int storvik_total_size;

        /* Scratch for int arrays during resampling (n_obs, ticks_since_sample) */
        int *resample_scratch_int;

        EntropyBuffer entropy;
        EWSSStats ewss[PARAM_LEARN_MAX_REGIMES];
        uint64_t rng[2];

        /* Runtime state */
        int tick;
        bool structural_break_flag;
        param_real current_load;

        /* Global tick-skip state */
        int ticks_since_full_update;
        bool force_next_update; /* Set by triggers to override skip */

        /* Diagnostics */
        uint64_t total_stat_updates;
        uint64_t total_samples_drawn;
        uint64_t samples_skipped_load;
        uint64_t samples_triggered_regime;
        uint64_t samples_triggered_break;
        uint64_t ticks_skipped_global; /* New: count global skips */

    } ParamLearner;

    /*═══════════════════════════════════════════════════════════════════════════
     * API: Configuration Presets
     *═══════════════════════════════════════════════════════════════════════════*/

    /**
     * Default: HFT-optimized sleeping intervals [50, 20, 5, 1]
     *
     * CHANGED from always-awake to sleeping mode for P99 optimization.
     * Use param_learn_config_full_bayesian() if you need every-tick updates.
     */
    ParamLearnConfig param_learn_config_defaults(void);

    /**
     * Sleeping Storvik with moderate intervals.
     */
    ParamLearnConfig param_learn_config_sleeping(void);

    /**
     * Full Bayesian: sample every tick in all regimes.
     * Highest accuracy, highest latency (~45μs).
     */
    ParamLearnConfig param_learn_config_full_bayesian(void);

    /**
     * HFT mode: Aggressive sleeping + global tick-skip.
     * Lowest latency (~5μs average), relies on triggers.
     */
    ParamLearnConfig param_learn_config_hft(void);

    /**
     * EWSS mode for comparison.
     */
    ParamLearnConfig param_learn_config_ewss(void);

    /*═══════════════════════════════════════════════════════════════════════════
     * API: Lifecycle
     *═══════════════════════════════════════════════════════════════════════════*/

    int param_learn_init(ParamLearner *learner,
                         const ParamLearnConfig *config,
                         int n_particles,
                         int n_regimes);

    void param_learn_free(ParamLearner *learner);
    void param_learn_reset(ParamLearner *learner);

    /*═══════════════════════════════════════════════════════════════════════════
     * API: Prior Specification
     *═══════════════════════════════════════════════════════════════════════════*/

    void param_learn_set_prior(ParamLearner *learner, int regime,
                               param_real mu, param_real phi, param_real sigma);

    void param_learn_set_prior_nig(ParamLearner *learner, int regime,
                                   param_real m, param_real kappa,
                                   param_real alpha, param_real beta,
                                   param_real phi);

    void param_learn_broadcast_priors(ParamLearner *learner);

    /*═══════════════════════════════════════════════════════════════════════════
     * API: Main Update
     *═══════════════════════════════════════════════════════════════════════════*/

    void param_learn_update(ParamLearner *learner,
                            const ParticleInfo *particles,
                            int n);

    void param_learn_signal_structural_break(ParamLearner *learner);
    void param_learn_set_load(ParamLearner *learner, param_real load);

    /*═══════════════════════════════════════════════════════════════════════════
     * API: Get Parameters
     *═══════════════════════════════════════════════════════════════════════════*/

    void param_learn_get_params(const ParamLearner *learner,
                                int particle_idx, int regime,
                                RegimeParams *params);

    void param_learn_force_sample(ParamLearner *learner,
                                  int particle_idx, int regime);

    /*═══════════════════════════════════════════════════════════════════════════
     * API: Resampling Support
     *═══════════════════════════════════════════════════════════════════════════*/

    void param_learn_copy_ancestor(ParamLearner *learner,
                                   int dst_particle, int src_particle);

    void param_learn_apply_resampling(ParamLearner *learner,
                                      const int *ancestors, int n);

    /*═══════════════════════════════════════════════════════════════════════════
     * API: Diagnostics
     *═══════════════════════════════════════════════════════════════════════════*/

    void param_learn_print_summary(const ParamLearner *learner);
    void param_learn_print_regime_stats(const ParamLearner *learner, int regime);

    void param_learn_get_regime_summary(const ParamLearner *learner, int regime,
                                        param_real *mu_mean, param_real *mu_std,
                                        param_real *sigma_mean, param_real *sigma_std,
                                        int *total_obs);

    /*═══════════════════════════════════════════════════════════════════════════
     * INLINE HELPER: Get active StorvikSoA (avoids repeated indexing)
     *═══════════════════════════════════════════════════════════════════════════*/

    static PL_FORCE_INLINE StorvikSoA *param_learn_get_active_soa(ParamLearner *learner)
    {
        return &learner->storvik[learner->active_buffer];
    }

    static PL_FORCE_INLINE const StorvikSoA *param_learn_get_active_soa_const(const ParamLearner *learner)
    {
        return &learner->storvik[learner->active_buffer];
    }

#ifdef __cplusplus
}
#endif

#endif /* RBPF_PARAM_LEARN_H */