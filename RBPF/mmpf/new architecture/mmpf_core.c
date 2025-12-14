/**
 * @file mmpf_core.c
 * @brief MMPF-ROCKS Core Implementation
 *
 * Multi-Model Particle Filter for real-time volatility estimation.
 * Three competing hypotheses provide different explanations for market behavior.
 * Bayesian model comparison determines which hypothesis best explains current conditions.
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * ARCHITECTURE: THE DEFENSE STACK
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * Each hypothesis has 6 layers of defense/adaptation:
 *
 *   1. Student-t (ν)        — Elastic absorption of fat tails
 *   2. OCSN                 — Hard cutoff (bump stop), K→0 for extremes
 *   3. Storvik              — Bayesian parameter learning (μ_vol, σ_vol)
 *   4. Swim Lanes           — Hard bounds prevent model collision
 *   5. Adaptive Forgetting  — Controls learning speed (turbo vs stiff)
 *   6. Weight Gate          — Only learn from data you "own" (>10% weight)
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * THE THREE HYPOTHESES
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 *   CALM (The Anchor)
 *     - Student-t ν = 20 (near-Gaussian, stiff spring)
 *     - Swim Lane: μ ∈ [-6.0, -4.5], σ ∈ [0.03, 0.15]
 *     - Learning: PINNED (0.0) — never adapts, provides invariant floor
 *     - Adaptive Forgetting: DISABLED
 *     - Role: Reference point. Rejects volatility → likelihood drops → regime switch
 *
 *   TREND (The Scout)
 *     - Student-t ν = 6 (moderate tails, medium spring)
 *     - Swim Lane: μ ∈ [-5.0, -2.5], σ ∈ [0.08, 0.35]
 *     - Learning: FULL (1.0) — aggressive adaptation
 *     - Adaptive Forgetting: AGGRESSIVE, λ ∈ [0.95, 0.995], COMBINED signal
 *     - Role: Working hypothesis. Tracks gradual regime changes.
 *
 *   CRISIS (Heavy Artillery)
 *     - Student-t ν = 3 (fat tails, soft spring)
 *     - Swim Lane: μ ∈ [-3.0, -0.5], σ ∈ [0.30, 1.20]
 *     - Learning: SLOW (0.1) — preserves crash dynamics memory
 *     - Adaptive Forgetting: RESTRICTED, λ ∈ [0.995, 0.9995], PREDICTIVE only
 *     - Role: Bunker. Expects extremes. Ignores outliers (they're its job!)
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * KEY DESIGN PRINCIPLES
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 *   1. Independence over communication — Models compete, don't coordinate
 *   2. Structural differentiation — ν, φ, bounds are fixed, not learned
 *   3. Bounded adaptation — Swim lanes prevent mode collapse
 *   4. Asymmetric learning — Calm pinned, Trend aggressive, Crisis slow
 *   5. Layered defense — Student-t (elastic) + OCSN (hard cutoff)
 *   6. Signal-appropriate forgetting — Crisis ignores outliers
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * CONSENSUS MECHANISM
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * Models don't communicate — they compete through likelihoods:
 *
 *   w'[k] ∝ w[k] × T[k] × L[k]
 *         (Prior × Transition × Likelihood)
 *
 * The same observation means different things to each model:
 *   - 5% move to Calm (ν=20, low μ): "Impossible!" L ≈ 0
 *   - 5% move to Crisis (ν=3, high μ): "Expected!" L = reasonable
 *
 * Dominant = argmax(weights) = Who explains data best
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 */

#ifdef MMPF_USE_TEST_STUB
#include "rocks_test_stub.c"
#endif

#include "mmpf_internal.h"

/*═══════════════════════════════════════════════════════════════════════════
 * SWIM LANES CONFIGURATION
 *
 * Hard bounds prevent hypothesis encroachment. Each model stays in its lane.
 * Independence is key — models don't push each other around.
 *
 * WHY HARD BOUNDS (not separation constraints):
 *   Separation constraints (Crisis.μ > Trend.μ + Gap) create COUPLING.
 *   If Trend chases a false breakout, it pushes Crisis into "Hyper-Crisis".
 *   When the real crash hits, Crisis is mis-tuned.
 *
 *   Hard bounds preserve independence:
 *   - Trend can be wrong in its lane
 *   - Crisis sits in bunker, unaffected
 *   - When real crash hits, Crisis is exactly where it should be
 *═══════════════════════════════════════════════════════════════════════════*/

typedef struct
{
    rbpf_real_t mu_vol_min;
    rbpf_real_t mu_vol_max;
    rbpf_real_t sigma_vol_min;
    rbpf_real_t sigma_vol_max;
    rbpf_real_t learning_rate_scale; /* 0.0 = pinned, 1.0 = full */
} MMPF_SwimLane;

static const MMPF_SwimLane SWIM_LANES[MMPF_N_MODELS] = {
    /*═══════════════════════════════════════════════════════════════════════
     * MMPF_CALM: The Anchor (Luxury Sedan)
     *
     * Suspension: Stiff (ν=20, near-Gaussian)
     * Position:   Low vol floor
     * Learning:   PINNED — never adapts
     * Purpose:    Invariant reference. "This is what calm looks like."
     *             Rejects volatility → likelihood tanks → regime switch
     *═══════════════════════════════════════════════════════════════════════*/
    {
        .mu_vol_min = RBPF_REAL(-6.0),        /* exp(-6) ≈ 0.25% vol */
        .mu_vol_max = RBPF_REAL(-4.5),        /* exp(-4.5) ≈ 1.1% vol */
        .sigma_vol_min = RBPF_REAL(0.03),     /* Very smooth */
        .sigma_vol_max = RBPF_REAL(0.15),     /* Low vol-of-vol */
        .learning_rate_scale = RBPF_REAL(0.0) /* PINNED — the rock doesn't move */
    },

    /*═══════════════════════════════════════════════════════════════════════
     * MMPF_TREND: The Scout (Sports Car)
     *
     * Suspension: Medium (ν=6, moderate tails)
     * Position:   Middle ground, adaptive
     * Learning:   FULL — aggressive adaptation with turbocharger
     * Purpose:    Working hypothesis. Tracks "Boring Bull" → "Choppy Sideways"
     *             Uses COMBINED adaptive forgetting (Z-score + outliers)
     *═══════════════════════════════════════════════════════════════════════*/
    {
        .mu_vol_min = RBPF_REAL(-5.0),        /* exp(-5) ≈ 0.67% vol */
        .mu_vol_max = RBPF_REAL(-2.5),        /* exp(-2.5) ≈ 8.2% vol */
        .sigma_vol_min = RBPF_REAL(0.08),     /* Moderate responsiveness */
        .sigma_vol_max = RBPF_REAL(0.35),     /* Can get wiggly */
        .learning_rate_scale = RBPF_REAL(1.0) /* FULL — turbocharger engaged */
    },

    /*═══════════════════════════════════════════════════════════════════════
     * MMPF_CRISIS: Heavy Artillery (Rally Car)
     *
     * Suspension: Soft (ν=3, fat tails)
     * Position:   High vol ceiling, explosive σ_vol
     * Learning:   SLOW (0.1) — preserves memory of crash dynamics
     * Purpose:    The bunker. Expects craters. Outliers are its JOB.
     *             Uses PREDICTIVE_SURPRISE only (ignores outlier fraction!)
     *═══════════════════════════════════════════════════════════════════════*/
    {
        .mu_vol_min = RBPF_REAL(-3.0),        /* exp(-3) ≈ 5% vol */
        .mu_vol_max = RBPF_REAL(-0.5),        /* exp(-0.5) ≈ 60% vol (!!) */
        .sigma_vol_min = RBPF_REAL(0.30),     /* MUST be explosive */
        .sigma_vol_max = RBPF_REAL(1.20),     /* Very high vol-of-vol */
        .learning_rate_scale = RBPF_REAL(0.1) /* SLOW — long memory for crashes */
    }};

/* Clamp value to range */
static inline rbpf_real_t clamp_real(rbpf_real_t x, rbpf_real_t lo, rbpf_real_t hi)
{
    if (x < lo)
        return lo;
    if (x > hi)
        return hi;
    return x;
}

/*═══════════════════════════════════════════════════════════════════════════
 * PARTICLE BUFFER IMPLEMENTATION
 *═══════════════════════════════════════════════════════════════════════════*/

MMPF_ParticleBuffer *mmpf_buffer_create(int n_particles, int n_storvik_regimes)
{
    MMPF_ParticleBuffer *buf = (MMPF_ParticleBuffer *)malloc(sizeof(MMPF_ParticleBuffer));
    if (!buf)
        return NULL;

    memset(buf, 0, sizeof(MMPF_ParticleBuffer));
    buf->n_particles = n_particles;
    buf->n_storvik_regimes = n_storvik_regimes;

    size_t n = (size_t)n_particles;
    size_t nr = (size_t)n_storvik_regimes;

    size_t rbpf_float_size = n * sizeof(rbpf_real_t);
    size_t rbpf_int_size = n * sizeof(int);
    size_t storvik_size = n * nr * sizeof(param_real);

    /* Total size with alignment padding */
    size_t total = 0;
    total += (rbpf_float_size + MMPF_ALIGN - 1) & ~(MMPF_ALIGN - 1);
    total += (rbpf_float_size + MMPF_ALIGN - 1) & ~(MMPF_ALIGN - 1);
    total += (rbpf_int_size + MMPF_ALIGN - 1) & ~(MMPF_ALIGN - 1);
    total += (rbpf_float_size + MMPF_ALIGN - 1) & ~(MMPF_ALIGN - 1);
    total += 6 * ((storvik_size + MMPF_ALIGN - 1) & ~(MMPF_ALIGN - 1));

    buf->_memory = mmpf_aligned_alloc(total, MMPF_ALIGN);
    if (!buf->_memory)
    {
        free(buf);
        return NULL;
    }
    buf->_memory_size = total;
    memset(buf->_memory, 0, total);

    /* Assign pointers */
    char *ptr = (char *)buf->_memory;

    buf->mu = (rbpf_real_t *)ptr;
    ptr += (rbpf_float_size + MMPF_ALIGN - 1) & ~(MMPF_ALIGN - 1);

    buf->var = (rbpf_real_t *)ptr;
    ptr += (rbpf_float_size + MMPF_ALIGN - 1) & ~(MMPF_ALIGN - 1);

    buf->ksc_regime = (int *)ptr;
    ptr += (rbpf_int_size + MMPF_ALIGN - 1) & ~(MMPF_ALIGN - 1);

    buf->log_weight = (rbpf_real_t *)ptr;
    ptr += (rbpf_float_size + MMPF_ALIGN - 1) & ~(MMPF_ALIGN - 1);

    buf->storvik_m = (param_real *)ptr;
    ptr += (storvik_size + MMPF_ALIGN - 1) & ~(MMPF_ALIGN - 1);

    buf->storvik_kappa = (param_real *)ptr;
    ptr += (storvik_size + MMPF_ALIGN - 1) & ~(MMPF_ALIGN - 1);

    buf->storvik_alpha = (param_real *)ptr;
    ptr += (storvik_size + MMPF_ALIGN - 1) & ~(MMPF_ALIGN - 1);

    buf->storvik_beta = (param_real *)ptr;
    ptr += (storvik_size + MMPF_ALIGN - 1) & ~(MMPF_ALIGN - 1);

    buf->storvik_mu = (param_real *)ptr;
    ptr += (storvik_size + MMPF_ALIGN - 1) & ~(MMPF_ALIGN - 1);

    buf->storvik_sigma = (param_real *)ptr;

    return buf;
}

void mmpf_buffer_destroy(MMPF_ParticleBuffer *buf)
{
    if (buf)
    {
        if (buf->_memory)
            mmpf_aligned_free(buf->_memory);
        free(buf);
    }
}

void mmpf_buffer_export(MMPF_ParticleBuffer *buf, const RBPF_Extended *ext)
{
    const int n = buf->n_particles;
    const int nr = buf->n_storvik_regimes;
    const RBPF_KSC *rbpf = ext->rbpf;
    const ParamLearner *learner = &ext->storvik;

    memcpy(buf->mu, rbpf->mu, n * sizeof(rbpf_real_t));
    memcpy(buf->var, rbpf->var, n * sizeof(rbpf_real_t));
    memcpy(buf->ksc_regime, rbpf->regime, n * sizeof(int));
    memcpy(buf->log_weight, rbpf->log_weight, n * sizeof(rbpf_real_t));

    const StorvikSoA *soa = param_learn_get_active_soa_const(learner);
    const size_t storvik_size = (size_t)n * nr * sizeof(param_real);

    memcpy(buf->storvik_m, soa->m, storvik_size);
    memcpy(buf->storvik_kappa, soa->kappa, storvik_size);
    memcpy(buf->storvik_alpha, soa->alpha, storvik_size);
    memcpy(buf->storvik_beta, soa->beta, storvik_size);
    memcpy(buf->storvik_mu, soa->mu_cached, storvik_size);
    memcpy(buf->storvik_sigma, soa->sigma_cached, storvik_size);
}

void mmpf_buffer_import(const MMPF_ParticleBuffer *buf, RBPF_Extended *ext)
{
    const int n = buf->n_particles;
    const int nr = buf->n_storvik_regimes;
    RBPF_KSC *rbpf = ext->rbpf;
    ParamLearner *learner = &ext->storvik;

    memcpy(rbpf->mu, buf->mu, n * sizeof(rbpf_real_t));
    memcpy(rbpf->var, buf->var, n * sizeof(rbpf_real_t));
    memcpy(rbpf->regime, buf->ksc_regime, n * sizeof(int));
    memcpy(rbpf->log_weight, buf->log_weight, n * sizeof(rbpf_real_t));

    StorvikSoA *soa = param_learn_get_active_soa(learner);
    const size_t storvik_size = (size_t)n * nr * sizeof(param_real);

    memcpy(soa->m, buf->storvik_m, storvik_size);
    memcpy(soa->kappa, buf->storvik_kappa, storvik_size);
    memcpy(soa->alpha, buf->storvik_alpha, storvik_size);
    memcpy(soa->beta, buf->storvik_beta, storvik_size);
    memcpy(soa->mu_cached, buf->storvik_mu, storvik_size);
    memcpy(soa->sigma_cached, buf->storvik_sigma, storvik_size);
}

void mmpf_buffer_copy_particle(MMPF_ParticleBuffer *dst, int dst_idx,
                               const MMPF_ParticleBuffer *src, int src_idx)
{
    const int nr = src->n_storvik_regimes;

    dst->mu[dst_idx] = src->mu[src_idx];
    dst->var[dst_idx] = src->var[src_idx];
    dst->ksc_regime[dst_idx] = src->ksc_regime[src_idx];
    dst->log_weight[dst_idx] = src->log_weight[src_idx];

    const int src_off = src_idx * nr;
    const int dst_off = dst_idx * nr;

    memcpy(&dst->storvik_m[dst_off], &src->storvik_m[src_off], nr * sizeof(param_real));
    memcpy(&dst->storvik_kappa[dst_off], &src->storvik_kappa[src_off], nr * sizeof(param_real));
    memcpy(&dst->storvik_alpha[dst_off], &src->storvik_alpha[src_off], nr * sizeof(param_real));
    memcpy(&dst->storvik_beta[dst_off], &src->storvik_beta[src_off], nr * sizeof(param_real));
    memcpy(&dst->storvik_mu[dst_off], &src->storvik_mu[src_off], nr * sizeof(param_real));
    memcpy(&dst->storvik_sigma[dst_off], &src->storvik_sigma[src_off], nr * sizeof(param_real));
}

/*═══════════════════════════════════════════════════════════════════════════
 * PARAMETER SYNC: Storvik → RBPF
 *═══════════════════════════════════════════════════════════════════════════*/

void mmpf_sync_parameters(RBPF_Extended *ext)
{
    if (!ext)
        return;

    RBPF_KSC *rbpf = ext->rbpf;
    const ParamLearner *learner = &ext->storvik;
    const StorvikSoA *soa = param_learn_get_active_soa_const(learner);
    const int n = rbpf->n_particles;
    const int n_regimes = learner->n_regimes;

    if (n_regimes != rbpf->n_regimes)
        return;

    const int total = n * n_regimes;

    mmpf_convert_double_to_float(soa->mu_cached, rbpf->particle_mu_vol, total);
    mmpf_convert_double_to_float(soa->sigma_cached, rbpf->particle_sigma_vol, total);

#if defined(__AVX512F__) || defined(__AVX2__) || defined(__SSE2__)
    _mm_sfence();
#endif

    rbpf->use_learned_params = 1;
}

/*═══════════════════════════════════════════════════════════════════════════
 * STORVIK UPDATE FOR HYPOTHESIS
 *═══════════════════════════════════════════════════════════════════════════*/

void mmpf_update_storvik_for_hypothesis(MMPF_ROCKS *mmpf, int k, int resampled)
{
    RBPF_Extended *ext = mmpf->ext[k];
    RBPF_KSC *rbpf = ext->rbpf;
    const MMPF_SwimLane *lane = &SWIM_LANES[k];
    int i, r;

    if (!ext->storvik_initialized)
        return;

    /* Check swim lane — skip if pinned */
    if (lane->learning_rate_scale < RBPF_REAL(0.01))
    {
        /* Pinned model: just update lag buffers, no learning */
        const int n = rbpf->n_particles;
        for (i = 0; i < n; i++)
        {
            ext->ell_lag_buffer[i] = rbpf->mu[i];
            ext->prev_regime[i] = rbpf->regime[i];
        }
        return;
    }

    const int n = rbpf->n_particles;

    if (resampled)
    {
        param_learn_apply_resampling(&ext->storvik, rbpf->indices, n);
    }

    ParticleInfo *info = ext->particle_info;
    const rbpf_real_t *w_norm = rbpf->w_norm;
    const rbpf_real_t *mu = rbpf->mu;
    const int *regime = rbpf->regime;
    rbpf_real_t *ell_lag = ext->ell_lag_buffer;
    int *prev_regime = ext->prev_regime;

    for (i = 0; i < n; i++)
    {
        int parent = resampled ? rbpf->indices[i] : i;
        info[i].regime = regime[i];
        info[i].ell = (param_real)mu[i];
        info[i].weight = (param_real)w_norm[i];
        info[i].ell_lag = (param_real)ell_lag[parent];
        info[i].prev_regime = prev_regime[parent];
    }

    param_learn_update(&ext->storvik, info, n);

    /* Get learned params */
    RegimeParams params;
    param_learn_get_params(&ext->storvik, 0, 0, &params);

    /* Apply swim lane bounds */
    rbpf_real_t mu_vol = clamp_real((rbpf_real_t)params.mu,
                                    lane->mu_vol_min, lane->mu_vol_max);
    rbpf_real_t sigma_vol = clamp_real((rbpf_real_t)params.sigma,
                                       lane->sigma_vol_min, lane->sigma_vol_max);

    /* Apply learning rate scale (interpolate toward learned value) */
    if (lane->learning_rate_scale < RBPF_REAL(1.0))
    {
        rbpf_real_t alpha = lane->learning_rate_scale;
        rbpf_real_t old_mu = rbpf->params[0].mu_vol;
        rbpf_real_t old_sigma = rbpf->params[0].sigma_vol;

        mu_vol = old_mu + alpha * (mu_vol - old_mu);
        sigma_vol = old_sigma + alpha * (sigma_vol - old_sigma);

        /* Re-clamp after interpolation */
        mu_vol = clamp_real(mu_vol, lane->mu_vol_min, lane->mu_vol_max);
        sigma_vol = clamp_real(sigma_vol, lane->sigma_vol_min, lane->sigma_vol_max);
    }

    /* Push bounded params to RBPF */
    for (r = 0; r < rbpf->n_regimes; r++)
    {
        rbpf->params[r].mu_vol = mu_vol;
        rbpf->params[r].sigma_vol = sigma_vol;
    }

    /* Update gated_dynamics for diagnostics */
    mmpf->gated_dynamics[k].mu_vol = (double)mu_vol;
    mmpf->gated_dynamics[k].sigma_eta = (double)sigma_vol;

    /* Update lag buffers */
    for (i = 0; i < n; i++)
    {
        ell_lag[i] = mu[i];
        prev_regime[i] = regime[i];
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * CONFIGURATION DEFAULTS
 *
 * These defaults implement our full architecture:
 *   - Student-t ν: [20, 6, 3] for [Calm, Trend, Crisis]
 *   - φ (persistence): [0.995, 0.95, 0.85]
 *   - Initial μ_vol within swim lane bounds
 *   - OCSN enabled with 5% outlier prior, 150 variance
 *   - Storvik sync enabled
 *   - Adaptive stickiness enabled
 *═══════════════════════════════════════════════════════════════════════════*/

MMPF_Config mmpf_config_defaults(void)
{
    MMPF_Config cfg;
    int k, r;

    memset(&cfg, 0, sizeof(cfg));

    cfg.n_particles = 512;
    cfg.n_ksc_regimes = 4;     /* KSC mixture components */
    cfg.n_storvik_regimes = 4; /* Storvik regime count */

    /*═══════════════════════════════════════════════════════════════════════
     * GLOBAL BASELINE: DISABLED
     *
     * We use swim lanes instead of a shared EWMA baseline.
     * Each hypothesis learns its own level independently.
     *═══════════════════════════════════════════════════════════════════════*/
    cfg.enable_global_baseline = 0;
    cfg.global_mu_vol_init = RBPF_REAL(-4.4);
    cfg.global_mu_vol_alpha = RBPF_REAL(0.95);

    /* Legacy offsets (unused with swim lanes) */
    cfg.mu_vol_offsets[MMPF_CALM] = RBPF_REAL(-0.5);
    cfg.mu_vol_offsets[MMPF_TREND] = RBPF_REAL(0.0);
    cfg.mu_vol_offsets[MMPF_CRISIS] = RBPF_REAL(0.5);

    cfg.baseline_gate_on = RBPF_REAL(0.50);
    cfg.baseline_gate_off = RBPF_REAL(0.40);

    /*═══════════════════════════════════════════════════════════════════════
     * STUDENT-T CONFIGURATION
     *
     * The Spring: Elastic absorption of fat tails
     *   - Calm (ν=20):  Near-Gaussian. "Bumps are impossible" → L tanks
     *   - Trend (ν=6):  Moderate tails. "Some bumps expected"
     *   - Crisis (ν=3): Fat tails. "Craters are normal" → L stays reasonable
     *
     * ν is STRUCTURAL — not learned. This provides permanent differentiation.
     *═══════════════════════════════════════════════════════════════════════*/
    cfg.enable_student_t = 1;
    cfg.hypothesis_nu[MMPF_CALM] = RBPF_REAL(20.0);  /* Stiff spring */
    cfg.hypothesis_nu[MMPF_TREND] = RBPF_REAL(6.0);  /* Medium spring */
    cfg.hypothesis_nu[MMPF_CRISIS] = RBPF_REAL(3.0); /* Soft spring */

    cfg.enable_nu_learning = 0; /* ν is structural, not learned */
    cfg.nu_floor = RBPF_REAL(2.5);
    cfg.nu_ceil = RBPF_REAL(30.0);
    cfg.nu_learning_rate = RBPF_REAL(0.99);

    cfg.enable_gated_learning = 1;
    cfg.gated_learning_threshold = RBPF_REAL(0.0);

    /*═══════════════════════════════════════════════════════════════════════
     * HYPOTHESIS PARAMETERS (within swim lane bounds)
     *
     * ┌──────────┬─────────────────┬─────────────────┬─────────┬────────┐
     * │ Model    │ μ_vol bounds    │ σ_vol bounds    │ φ       │ ν      │
     * ├──────────┼─────────────────┼─────────────────┼─────────┼────────┤
     * │ Calm     │ [-6.0, -4.5]    │ [0.03, 0.15]    │ 0.995   │ 20     │
     * │ Trend    │ [-5.0, -2.5]    │ [0.08, 0.35]    │ 0.95    │ 6      │
     * │ Crisis   │ [-3.0, -0.5]    │ [0.30, 1.20]    │ 0.85    │ 3      │
     * └──────────┴─────────────────┴─────────────────┴─────────┴────────┘
     *
     * φ (persistence) provides additional differentiation:
     *   - Calm (0.995):  Very slow mean reversion → stable, ignores noise
     *   - Trend (0.95):  Medium → tracks gradual changes
     *   - Crisis (0.85): Fast mean reversion → explosive, quick response
     *═══════════════════════════════════════════════════════════════════════*/
    cfg.hypotheses[MMPF_CALM].mu_vol = RBPF_REAL(-5.0);    /* Within [-6.0, -4.5] */
    cfg.hypotheses[MMPF_CALM].phi = RBPF_REAL(0.995);      /* Very persistent */
    cfg.hypotheses[MMPF_CALM].sigma_eta = RBPF_REAL(0.08); /* Within [0.03, 0.15] */
    cfg.hypotheses[MMPF_CALM].nu = cfg.hypothesis_nu[MMPF_CALM];

    cfg.hypotheses[MMPF_TREND].mu_vol = RBPF_REAL(-4.0);    /* Within [-5.0, -2.5] */
    cfg.hypotheses[MMPF_TREND].phi = RBPF_REAL(0.95);       /* Moderate persistence */
    cfg.hypotheses[MMPF_TREND].sigma_eta = RBPF_REAL(0.15); /* Within [0.08, 0.35] */
    cfg.hypotheses[MMPF_TREND].nu = cfg.hypothesis_nu[MMPF_TREND];

    cfg.hypotheses[MMPF_CRISIS].mu_vol = RBPF_REAL(-2.0);    /* Within [-3.0, -0.5] */
    cfg.hypotheses[MMPF_CRISIS].phi = RBPF_REAL(0.85);       /* Fast mean reversion */
    cfg.hypotheses[MMPF_CRISIS].sigma_eta = RBPF_REAL(0.50); /* Within [0.30, 1.20] */
    cfg.hypotheses[MMPF_CRISIS].nu = cfg.hypothesis_nu[MMPF_CRISIS];

    /*═══════════════════════════════════════════════════════════════════════
     * IMM SETTINGS: "Switch, Don't Jump"
     *
     * STICKINESS (Regime persistence):
     *   base_stickiness (0.98): P(stay) in normal conditions
     *   min_stickiness (0.85):  P(stay) floor when outliers high
     *   crisis_exit_boost (0.92): Multiplicative factor for Crisis exit
     *
     * DIRICHLET PRIOR (Regime warming):
     *
     *   Replaces the old heuristic: if (leave < 0.05) leave = 0.05
     *
     *   We use Bayesian smoothing so all regimes stay "warm":
     *
     *                      P_raw × N + α
     *     P_smoothed = ─────────────────────
     *                      N + K × α
     *
     *   transition_prior_alpha (α): Pseudo-counts per regime
     *     - "I believe I've observed α transitions to each regime"
     *     - Higher α → stronger pull toward uniform → more warming
     *
     *   transition_prior_mass (N): Observation mass
     *     - "My stickiness estimate is worth N observations"
     *     - Higher N → trust raw stickiness more → less warming
     *
     *   The natural FLOOR emerges as: α / (N + K×α)
     *
     *   ┌───────┬────────┬─────────┬───────────────────────────────┐
     *   │   α   │   N    │  Floor  │  Interpretation               │
     *   ├───────┼────────┼─────────┼───────────────────────────────┤
     *   │  1.0  │   20   │  4.3%   │  Default (≈ old 5% heuristic) │
     *   │  1.0  │   50   │  1.9%   │  Confident in stickiness      │
     *   │  2.0  │   20   │  7.7%   │  Paranoid — strong warming    │
     *   │  0.5  │   20   │  2.2%   │  Relaxed — minimal warming    │
     *   └───────┴────────┴─────────┴───────────────────────────────┘
     *
     *═══════════════════════════════════════════════════════════════════════*/
    cfg.base_stickiness = RBPF_REAL(0.98);
    cfg.min_stickiness = RBPF_REAL(0.85);
    cfg.crisis_exit_boost = RBPF_REAL(0.92);
    cfg.enable_adaptive_stickiness = 1;

    /* Dirichlet prior for transition matrix smoothing
     * Default: α=1.0, N=20 → floor ≈ 4.3% (all regimes stay warm) */
    cfg.transition_prior_alpha = RBPF_REAL(1.0);
    cfg.transition_prior_mass = RBPF_REAL(20.0);

    cfg.enable_storvik_sync = 1;

    /*═══════════════════════════════════════════════════════════════════════
     * ZERO RETURN HANDLING (HFT Critical)
     *
     * In HFT, price often doesn't move for many ticks (r_t = 0).
     * Policy 1: Use floor at min_log_return_sq ≈ -18 (1bp minimum)
     *═══════════════════════════════════════════════════════════════════════*/
    cfg.zero_return_policy = 1;
    cfg.min_log_return_sq = RBPF_REAL(-18.0);

    /* Initial weights: slight bias toward calm (typical market state) */
    cfg.initial_weights[MMPF_CALM] = RBPF_REAL(0.6);
    cfg.initial_weights[MMPF_TREND] = RBPF_REAL(0.3);
    cfg.initial_weights[MMPF_CRISIS] = RBPF_REAL(0.1);

    /*═══════════════════════════════════════════════════════════════════════
     * OCSN: The Bump Stop
     *
     * Mixture model: P(y) = (1-π)×Normal + π×Normal(0, σ²_outlier)
     * When outlier component wins: K → 0 → state frozen
     *
     * This provides HARD protection that Student-t alone doesn't:
     *   - Student-t (ν=3): 10σ move still updates state (λ dampens but K > 0)
     *   - OCSN: 10σ move → outlier wins → K ≈ 0 → state doesn't move
     *═══════════════════════════════════════════════════════════════════════*/
    cfg.robust_ocsn.enabled = 1;
    for (r = 0; r < PARAM_LEARN_MAX_REGIMES; r++)
    {
        cfg.robust_ocsn.regime[r].prob = RBPF_REAL(0.05);      /* 5% outlier prior */
        cfg.robust_ocsn.regime[r].variance = RBPF_REAL(150.0); /* HUGE → K ≈ 0 */
    }

    cfg.storvik_config = param_learn_config_defaults();
    cfg.rng_seed = 42;

    return cfg;
}

MMPF_Config mmpf_config_hft(void)
{
    MMPF_Config cfg = mmpf_config_defaults();
    cfg.n_particles = 256;
    cfg.storvik_config = param_learn_config_hft();
    return cfg;
}

/*═══════════════════════════════════════════════════════════════════════════
 * IMM MIXING INTERNALS
 *═══════════════════════════════════════════════════════════════════════════*/

static void update_transition_matrix(MMPF_ROCKS *mmpf)
{
    int i, j;

    if (!mmpf->config.enable_adaptive_stickiness)
        return;

    /*═══════════════════════════════════════════════════════════════════════
     * DIRICHLET-SMOOTHED TRANSITION MATRIX
     *
     * PROBLEM: We need all regimes to stay "warm" (never go to 0% probability)
     * so they can respond instantly when conditions change.
     *
     * OLD APPROACH (Heuristic):
     *   if (leave < 0.05) leave = 0.05;  // Magic number, discontinuous kink
     *
     * NEW APPROACH (Bayesian):
     *   Treat transition probabilities as having a Dirichlet prior.
     *   Use Laplace smoothing (additive smoothing) to blend raw estimates
     *   with uniform prior.
     *
     * ─────────────────────────────────────────────────────────────────────
     * THE MATH
     * ─────────────────────────────────────────────────────────────────────
     *
     * Dirichlet distribution: Dir(α₁, α₂, ..., αₖ)
     *
     * With uniform prior (α₁ = α₂ = ... = αₖ = α), the posterior mean is:
     *
     *                    P_raw × N + α
     *   P_smoothed = ─────────────────────
     *                    N + K × α
     *
     * Where:
     *   P_raw = raw transition probability from stickiness logic
     *   N     = "observation mass" (how much we trust our stickiness estimate)
     *   α     = prior pseudo-count (virtual observations per regime)
     *   K     = number of regimes (3)
     *
     * ─────────────────────────────────────────────────────────────────────
     * INTERPRETATION
     * ─────────────────────────────────────────────────────────────────────
     *
     * α = 1.0 means: "Before seeing any data, I believe I've observed
     *                 1 transition to each regime."
     *
     * N = 20 means:  "My current stickiness estimate is worth 20 observations."
     *
     * The FLOOR (minimum transition probability) emerges naturally:
     *
     *                     α                    1.0
     *   Floor = ───────────────── = ─────────────────── ≈ 4.3%
     *            N + K × α           20 + 3 × 1.0
     *
     * ─────────────────────────────────────────────────────────────────────
     * TUNING GUIDE
     * ─────────────────────────────────────────────────────────────────────
     *
     *   α     N      Floor    Effect
     *   ───   ───    ─────    ──────────────────────────────────
     *   1.0   20     4.3%     Default — balanced warming
     *   1.0   50     1.9%     Trust stickiness more, less warming
     *   2.0   20     7.7%     Paranoid — strong warming
     *   0.5   20     2.2%     Confident — minimal warming
     *   1.0   10     7.7%     Uncertain about stickiness
     *
     * ─────────────────────────────────────────────────────────────────────
     * WHY THIS IS BETTER THAN A HARD FLOOR
     * ─────────────────────────────────────────────────────────────────────
     *
     * 1. Smooth: No discontinuous kink in the probability surface
     * 2. Principled: It's a Bayesian prior, not a magic number
     * 3. Interpretable: Parameters have clear meaning (pseudo-counts)
     * 4. Symmetric: Also slightly regularizes the diagonal (stay probability)
     * 5. Scales: Works correctly regardless of K (number of models)
     *
     *═══════════════════════════════════════════════════════════════════════*/

    rbpf_real_t base = mmpf->config.base_stickiness;
    rbpf_real_t min_s = mmpf->config.min_stickiness;
    rbpf_real_t outlier = mmpf->outlier_fraction;

    /* Dirichlet prior parameters */
    rbpf_real_t alpha = mmpf->config.transition_prior_alpha; /* Pseudo-counts */
    rbpf_real_t N_mass = mmpf->config.transition_prior_mass; /* Observation mass */
    rbpf_real_t K = (rbpf_real_t)MMPF_N_MODELS;
    rbpf_real_t denom = N_mass + K * alpha;

    /* Adaptive stickiness: drops when outliers are high (uncertain regime) */
    rbpf_real_t stickiness = base - (base - min_s) * outlier;
    mmpf->current_stickiness = stickiness;

    for (i = 0; i < MMPF_N_MODELS; i++)
    {
        rbpf_real_t stay = stickiness;

        /* Crisis exits faster — don't get stuck in crisis mode */
        if (i == MMPF_CRISIS)
        {
            stay *= mmpf->config.crisis_exit_boost;
        }

        /* Raw leave probability (before smoothing) */
        rbpf_real_t leave = (RBPF_REAL(1.0) - stay) / (MMPF_N_MODELS - 1);

        /* Apply Dirichlet smoothing to each transition probability */
        for (j = 0; j < MMPF_N_MODELS; j++)
        {
            rbpf_real_t raw_prob = (i == j) ? stay : leave;

            /* P_smoothed = (P_raw × N + α) / (N + K × α) */
            mmpf->transition[i][j] = (raw_prob * N_mass + alpha) / denom;
        }
    }
}

static void compute_mixing_weights(MMPF_ROCKS *mmpf)
{
    int target, source;

    for (target = 0; target < MMPF_N_MODELS; target++)
    {
        rbpf_real_t denom = RBPF_REAL(0.0);

        for (source = 0; source < MMPF_N_MODELS; source++)
        {
            rbpf_real_t val = mmpf->transition[source][target] * mmpf->weights[source];
            mmpf->mixing_weights[target][source] = val;
            denom += val;
        }

        if (denom > RBPF_EPS)
        {
            for (source = 0; source < MMPF_N_MODELS; source++)
            {
                mmpf->mixing_weights[target][source] /= denom;
            }
        }
        else
        {
            for (source = 0; source < MMPF_N_MODELS; source++)
            {
                mmpf->mixing_weights[target][source] = RBPF_REAL(1.0) / MMPF_N_MODELS;
            }
        }
    }
}

static void compute_mixing_counts(MMPF_ROCKS *mmpf)
{
    const int n = mmpf->n_particles;
    int target, source;

    for (target = 0; target < MMPF_N_MODELS; target++)
    {
        int total = 0;
        int max_source = 0;
        rbpf_real_t max_weight = mmpf->mixing_weights[target][0];

        for (source = 0; source < MMPF_N_MODELS; source++)
        {
            rbpf_real_t w = mmpf->mixing_weights[target][source];
            int count = (int)(w * n);
            mmpf->mix_counts[target][source] = count;
            total += count;

            if (w > max_weight)
            {
                max_weight = w;
                max_source = source;
            }
        }

        mmpf->mix_counts[target][max_source] += (n - total);
    }
}

static void stratified_resample_from_buffer(
    const MMPF_ParticleBuffer *src,
    int *indices_out,
    int count,
    rbpf_pcg32_t *rng)
{
    const int n_src = src->n_particles;
    int i;

    rbpf_real_t max_log = src->log_weight[0];
    for (i = 1; i < n_src; i++)
    {
        if (src->log_weight[i] > max_log)
            max_log = src->log_weight[i];
    }

    rbpf_real_t sum = RBPF_REAL(0.0);
    rbpf_real_t cumsum[1024];

    for (i = 0; i < n_src; i++)
    {
        sum += rbpf_exp(src->log_weight[i] - max_log);
        cumsum[i] = sum;
    }

    rbpf_real_t step = sum / count;
    rbpf_real_t u = rbpf_pcg32_uniform(rng) * step;
    int src_idx = 0;

    for (i = 0; i < count; i++)
    {
        while (src_idx < n_src - 1 && cumsum[src_idx] < u)
        {
            src_idx++;
        }
        indices_out[i] = src_idx;
        u += step;
    }
}

static void imm_mixing_step(MMPF_ROCKS *mmpf)
{
    int k, target, source, i;
    int indices[1024];

    for (k = 0; k < MMPF_N_MODELS; k++)
    {
        mmpf_buffer_export(mmpf->buffer[k], mmpf->ext[k]);
    }

    for (target = 0; target < MMPF_N_MODELS; target++)
    {
        int p_idx = 0;

        for (source = 0; source < MMPF_N_MODELS; source++)
        {
            int count = mmpf->mix_counts[target][source];
            if (count <= 0)
                continue;

            stratified_resample_from_buffer(
                mmpf->buffer[source], indices, count, &mmpf->rng);

            for (i = 0; i < count; i++)
            {
                mmpf_buffer_copy_particle(
                    mmpf->mixed_buffer[target], p_idx,
                    mmpf->buffer[source], indices[i]);
                mmpf->mixed_buffer[target]->log_weight[p_idx] = RBPF_REAL(0.0);
                p_idx++;
            }
        }
    }

    for (k = 0; k < MMPF_N_MODELS; k++)
    {
        mmpf_buffer_import(mmpf->mixed_buffer[k], mmpf->ext[k]);
    }

    mmpf->imm_mix_count++;
}

/*═══════════════════════════════════════════════════════════════════════════
 * LIFECYCLE
 *═══════════════════════════════════════════════════════════════════════════*/

MMPF_ROCKS *mmpf_create(const MMPF_Config *config)
{
    MMPF_Config cfg = config ? *config : mmpf_config_defaults();
    int k, r, i, j;

    MMPF_ROCKS *mmpf = (MMPF_ROCKS *)malloc(sizeof(MMPF_ROCKS));
    if (!mmpf)
        return NULL;

    memset(mmpf, 0, sizeof(MMPF_ROCKS));
    mmpf->config = cfg;
    mmpf->n_particles = cfg.n_particles;

    rbpf_pcg32_seed(&mmpf->rng, cfg.rng_seed, 1);

    /* Create RBPF_Extended instances */
    for (k = 0; k < MMPF_N_MODELS; k++)
    {
        mmpf->ext[k] = rbpf_ext_create(cfg.n_particles, cfg.n_ksc_regimes, RBPF_PARAM_STORVIK);
        if (!mmpf->ext[k])
        {
            mmpf_destroy(mmpf);
            return NULL;
        }

        /* Allocate Student-t memory if needed */
        if (mmpf->ext[k]->rbpf->lambda == NULL)
        {
            if (rbpf_ksc_alloc_student_t(mmpf->ext[k]->rbpf) != 0)
            {
                fprintf(stderr, "MMPF: Failed to allocate Student-t memory for model %d\n", k);
                mmpf_destroy(mmpf);
                return NULL;
            }
        }

        MMPF_HypothesisParams *hp = &cfg.hypotheses[k];

        for (r = 0; r < cfg.n_ksc_regimes; r++)
        {
            rbpf_ext_set_regime_params(mmpf->ext[k], r,
                                       RBPF_REAL(1.0) - hp->phi,
                                       hp->mu_vol,
                                       hp->sigma_eta);
        }

        /* Configure per-hypothesis OCSN */
        mmpf->ext[k]->robust_ocsn.enabled = cfg.robust_ocsn.enabled;
        for (r = 0; r < cfg.n_ksc_regimes; r++)
        {
            rbpf_real_t scale = (k == MMPF_CALM) ? RBPF_REAL(0.8) : (k == MMPF_TREND) ? RBPF_REAL(1.0)
                                                                                      : RBPF_REAL(1.5);

            mmpf->ext[k]->robust_ocsn.regime[r].prob =
                cfg.robust_ocsn.regime[r].prob * scale;
            mmpf->ext[k]->robust_ocsn.regime[r].variance =
                cfg.robust_ocsn.regime[r].variance * scale;
        }

        rbpf_ext_init(mmpf->ext[k], hp->mu_vol, RBPF_REAL(0.5));

        for (r = 0; r < cfg.n_storvik_regimes; r++)
        {
            param_learn_set_prior(&mmpf->ext[k]->storvik, r,
                                  (param_real)hp->mu_vol,
                                  (param_real)hp->phi,
                                  (param_real)hp->sigma_eta);
        }
        param_learn_broadcast_priors(&mmpf->ext[k]->storvik);

        /* Set per-hypothesis forgetting rates */
        {
            param_real lambda;
            switch (k)
            {
            case MMPF_CALM:
                lambda = 0.999;
                break;
            case MMPF_TREND:
                lambda = 0.997;
                break;
            case MMPF_CRISIS:
                lambda = 0.990;
                break;
            default:
                lambda = 0.997;
                break;
            }
            param_learn_set_forgetting(&mmpf->ext[k]->storvik, true, lambda);
        }

        if (cfg.enable_storvik_sync)
        {
            mmpf_sync_parameters(mmpf->ext[k]);
        }
        else
        {
            mmpf->ext[k]->rbpf->use_learned_params = 0;
        }

        /* Student-t configuration */
        if (cfg.enable_student_t)
        {
            rbpf_ksc_enable_student_t(mmpf->ext[k]->rbpf, hp->nu);
            for (r = 0; r < cfg.n_ksc_regimes; r++)
            {
                rbpf_ksc_set_student_t_nu(mmpf->ext[k]->rbpf, r, hp->nu);
            }
            if (cfg.enable_nu_learning)
            {
                for (r = 0; r < cfg.n_ksc_regimes; r++)
                {
                    rbpf_ksc_enable_nu_learning(mmpf->ext[k]->rbpf, r, cfg.nu_learning_rate);
                }
            }
        }

        /*═══════════════════════════════════════════════════════════════════
         * ADAPTIVE FORGETTING CONFIGURATION (Per-Hypothesis)
         *
         * Swim Lanes control WHERE each model can go (position bounds).
         * Adaptive Forgetting controls HOW FAST it gets there (learning speed).
         *
         * Calm:   DISABLED — the anchor doesn't adapt, infinite memory
         * Trend:  AGGRESSIVE — turbocharger on the scout, rapid adaptation
         * Crisis: RESTRICTED — long memory, ignores outliers (they're its job!)
         *═══════════════════════════════════════════════════════════════════*/
        switch (k)
        {
        case MMPF_CALM:
            /* Calm: The Anchor — DISABLED
             * We want Calm to REJECT volatility (likelihood drops → regime switch)
             * not adapt to it. Effectively infinite memory (λ ≈ 1.0). */
            rbpf_ext_disable_adaptive_forgetting(mmpf->ext[k]);
            break;

        case MMPF_TREND:
            /* Trend: The Scout — AGGRESSIVE
             * Uses COMBINED signal (both Z-score drift and outlier fraction).
             * When market shifts from "Boring Bull" to "Choppy Sideways",
             * Trend must dump old stats fast and learn the new texture. */
            rbpf_ext_enable_adaptive_forgetting_mode(mmpf->ext[k], ADAPT_SIGNAL_COMBINED);

            /* Aggressive bounds: λ ∈ [0.95, 0.995]
             * N_eff ranges from 20 (crisis adaptation) to 200 (stable tracking) */
            rbpf_ext_set_adaptive_bounds(mmpf->ext[k],
                                         RBPF_REAL(0.95),   /* Floor: N_eff ≈ 20 */
                                         RBPF_REAL(0.995)); /* Ceiling: N_eff ≈ 200 */

            /* High steepness — snap to new regime quickly */
            rbpf_ext_set_adaptive_sigmoid(mmpf->ext[k],
                                          RBPF_REAL(2.0),   /* Center: 2σ surprise */
                                          RBPF_REAL(2.5),   /* Steepness: aggressive */
                                          RBPF_REAL(0.10)); /* Max 10% discount */

            /* Fast signal tracking */
            rbpf_ext_set_adaptive_smoothing(mmpf->ext[k],
                                            RBPF_REAL(0.02),  /* Baseline α (slow) */
                                            RBPF_REAL(0.20)); /* Signal α (fast) */
            break;

        case MMPF_CRISIS:
            /* Crisis: The Bunker — RESTRICTED
             * Uses PREDICTIVE_SURPRISE only — ignores outlier fraction!
             * Why? Outliers are Crisis's JOB. High outlier fraction means
             * Crisis is working correctly, not that it should forget.
             *
             * Long memory preserves knowledge of crash dynamics.
             * We don't want Crisis to over-adapt to THIS specific crash
             * and forget the general properties of "market failure". */
            rbpf_ext_enable_adaptive_forgetting_mode(mmpf->ext[k], ADAPT_SIGNAL_PREDICTIVE_SURPRISE);

            /* Very restricted bounds: λ ∈ [0.995, 0.9995]
             * N_eff ranges from 200 to 2000 — always long memory */
            rbpf_ext_set_adaptive_bounds(mmpf->ext[k],
                                         RBPF_REAL(0.995),   /* Floor: N_eff ≈ 200 */
                                         RBPF_REAL(0.9995)); /* Ceiling: N_eff ≈ 2000 */

            /* Very stiff — barely responds to surprise */
            rbpf_ext_set_adaptive_sigmoid(mmpf->ext[k],
                                          RBPF_REAL(3.0),   /* Center: 3σ (high threshold) */
                                          RBPF_REAL(1.0),   /* Steepness: gentle */
                                          RBPF_REAL(0.02)); /* Max 2% discount only */
            break;
        }

        mmpf->learned_nu[k] = hp->nu;
    }

    /* Create particle buffers */
    for (k = 0; k < MMPF_N_MODELS; k++)
    {
        mmpf->buffer[k] = mmpf_buffer_create(cfg.n_particles, cfg.n_storvik_regimes);
        mmpf->mixed_buffer[k] = mmpf_buffer_create(cfg.n_particles, cfg.n_storvik_regimes);

        if (!mmpf->buffer[k] || !mmpf->mixed_buffer[k])
        {
            mmpf_destroy(mmpf);
            return NULL;
        }
    }

    /* Initialize model weights */
    for (k = 0; k < MMPF_N_MODELS; k++)
    {
        mmpf->weights[k] = cfg.initial_weights[k];
        mmpf->log_weights[k] = rbpf_log(cfg.initial_weights[k]);
    }
    mmpf_normalize_weights(mmpf->weights, MMPF_N_MODELS);

    /* Initialize transition matrix */
    mmpf->current_stickiness = cfg.base_stickiness;
    mmpf->outlier_fraction = RBPF_REAL(0.0);
    update_transition_matrix(mmpf);

    /* Initialize regime tracking */
    mmpf->dominant = MMPF_CALM;
    mmpf->prev_dominant = MMPF_CALM;
    mmpf->ticks_in_regime = 0;

    /* Initialize shock state */
    mmpf->shock_active = 0;
    mmpf->process_noise_multiplier = RBPF_REAL(1.0);
    for (i = 0; i < MMPF_N_MODELS; i++)
    {
        for (j = 0; j < MMPF_N_MODELS; j++)
        {
            mmpf->saved_transition[i][j] = RBPF_REAL(0.0);
        }
    }

    /* Initialize cached outputs */
    mmpf->weighted_vol = RBPF_REAL(0.0);
    mmpf->weighted_log_vol = RBPF_REAL(0.0);
    for (k = 0; k < MMPF_N_MODELS; k++)
    {
        RBPF_KSC *rbpf = mmpf->ext[k]->rbpf;
        rbpf_real_t sum_log_vol = RBPF_REAL(0.0);
        for (i = 0; i < rbpf->n_particles; i++)
        {
            sum_log_vol += rbpf->mu[i];
        }
        rbpf_real_t log_vol = sum_log_vol / rbpf->n_particles;
        rbpf_real_t vol = rbpf_exp(log_vol);

        mmpf->model_output[k].log_vol_mean = log_vol;
        mmpf->model_output[k].vol_mean = vol;
        mmpf->model_output[k].ess = (rbpf_real_t)rbpf->n_particles;
        mmpf->model_output[k].outlier_fraction = RBPF_REAL(0.0);

        mmpf->weighted_vol += mmpf->weights[k] * vol;
        mmpf->weighted_log_vol += mmpf->weights[k] * log_vol;
    }
    mmpf->weighted_vol_std = RBPF_REAL(0.0);

    /* Initialize global baseline */
    mmpf->global_mu_vol = cfg.global_mu_vol_init;
    mmpf->prev_weighted_log_vol = cfg.global_mu_vol_init;
    mmpf->baseline_frozen_ticks = 0;

    /* Initialize gated dynamics */
    for (k = 0; k < MMPF_N_MODELS; k++)
    {
        mmpf->gated_dynamics[k].sum_x = 0.0;
        mmpf->gated_dynamics[k].sum_w_mu = 0.0;
        mmpf->gated_dynamics[k].mu_vol = (double)(cfg.global_mu_vol_init + cfg.mu_vol_offsets[k]);
        mmpf->gated_dynamics[k].sum_xy = 0.0;
        mmpf->gated_dynamics[k].sum_xx = 0.0;
        mmpf->gated_dynamics[k].phi = (double)cfg.hypotheses[k].phi;
        mmpf->gated_dynamics[k].sum_resid_sq = 0.0;
        mmpf->gated_dynamics[k].sum_w_sigma = 0.0;
        mmpf->gated_dynamics[k].sigma_eta = (double)cfg.hypotheses[k].sigma_eta;
        mmpf->gated_dynamics[k].prev_state = (double)cfg.global_mu_vol_init;
    }

    return mmpf;
}

void mmpf_destroy(MMPF_ROCKS *mmpf)
{
    int k;

    if (!mmpf)
        return;

    for (k = 0; k < MMPF_N_MODELS; k++)
    {
        if (mmpf->ext[k])
            rbpf_ext_destroy(mmpf->ext[k]);
        if (mmpf->buffer[k])
            mmpf_buffer_destroy(mmpf->buffer[k]);
        if (mmpf->mixed_buffer[k])
            mmpf_buffer_destroy(mmpf->mixed_buffer[k]);
    }

    free(mmpf);
}

void mmpf_reset(MMPF_ROCKS *mmpf, rbpf_real_t initial_vol)
{
    rbpf_real_t log_vol = rbpf_log(initial_vol);
    rbpf_real_t var0 = RBPF_REAL(0.5);
    int k, r, i;

    for (k = 0; k < MMPF_N_MODELS; k++)
    {
        rbpf_ext_init(mmpf->ext[k], log_vol, var0);
        param_learn_reset(&mmpf->ext[k]->storvik);
        param_learn_broadcast_priors(&mmpf->ext[k]->storvik);

        if (mmpf->config.enable_storvik_sync)
        {
            mmpf_sync_parameters(mmpf->ext[k]);
        }
        else
        {
            mmpf->ext[k]->rbpf->use_learned_params = 0;
        }

        if (mmpf->config.enable_student_t && mmpf->config.enable_nu_learning)
        {
            for (r = 0; r < mmpf->ext[k]->rbpf->n_regimes; r++)
            {
                rbpf_ksc_reset_nu_learning(mmpf->ext[k]->rbpf, r);
            }
        }

        mmpf->learned_nu[k] = mmpf->config.hypothesis_nu[k];
    }

    for (k = 0; k < MMPF_N_MODELS; k++)
    {
        mmpf->weights[k] = mmpf->config.initial_weights[k];
        mmpf->log_weights[k] = rbpf_log(mmpf->weights[k]);
    }
    mmpf_normalize_weights(mmpf->weights, MMPF_N_MODELS);

    mmpf->weighted_vol = initial_vol;
    mmpf->weighted_log_vol = log_vol;
    mmpf->weighted_vol_std = RBPF_REAL(0.0);
    for (k = 0; k < MMPF_N_MODELS; k++)
    {
        memset(&mmpf->model_output[k], 0, sizeof(RBPF_KSC_Output));
        mmpf->model_output[k].vol_mean = initial_vol;
        mmpf->model_output[k].log_vol_mean = log_vol;
        mmpf->model_output[k].ess = (rbpf_real_t)mmpf->n_particles;
        mmpf->model_likelihood[k] = RBPF_REAL(1.0);
    }

    mmpf->outlier_fraction = RBPF_REAL(0.0);
    mmpf->current_stickiness = mmpf->config.base_stickiness;
    update_transition_matrix(mmpf);

    mmpf->dominant = MMPF_CALM;
    mmpf->prev_dominant = MMPF_CALM;
    mmpf->ticks_in_regime = 0;

    mmpf->total_steps = 0;
    mmpf->regime_switches = 0;
    mmpf->imm_mix_count = 0;

    mmpf->global_mu_vol = log_vol;
    mmpf->prev_weighted_log_vol = log_vol;
    mmpf->baseline_frozen_ticks = 0;

    if (mmpf->config.enable_global_baseline)
    {
        for (k = 0; k < MMPF_N_MODELS; k++)
        {
            rbpf_real_t mu_k = log_vol + mmpf->config.mu_vol_offsets[k];
            RBPF_Extended *ext = mmpf->ext[k];
            for (r = 0; r < ext->rbpf->n_regimes; r++)
            {
                ext->rbpf->params[r].mu_vol = mu_k;
            }
        }
    }

    for (k = 0; k < MMPF_N_MODELS; k++)
    {
        mmpf->gated_dynamics[k].sum_x = 0.0;
        mmpf->gated_dynamics[k].sum_w_mu = 0.0;
        mmpf->gated_dynamics[k].mu_vol = (double)(log_vol + mmpf->config.mu_vol_offsets[k]);
        mmpf->gated_dynamics[k].sum_xy = 0.0;
        mmpf->gated_dynamics[k].sum_xx = 0.0;
        mmpf->gated_dynamics[k].sum_resid_sq = 0.0;
        mmpf->gated_dynamics[k].sum_w_sigma = 0.0;
        mmpf->gated_dynamics[k].prev_state = (double)log_vol;
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * MAIN STEP FUNCTION
 *═══════════════════════════════════════════════════════════════════════════*/

void mmpf_step(MMPF_ROCKS *mmpf, rbpf_real_t y, MMPF_Output *output)
{
    int k, r, i;
    int skip_update = 0;
    rbpf_real_t y_log;

    /* Global baseline update */
    if (mmpf->config.enable_global_baseline)
    {
        if (mmpf->prev_weighted_log_vol != mmpf->prev_weighted_log_vol)
        {
            mmpf->prev_weighted_log_vol = mmpf->global_mu_vol;
        }

        rbpf_real_t w_crisis = mmpf->weights[MMPF_CRISIS];
        int currently_frozen = (mmpf->baseline_frozen_ticks > 0);
        int should_freeze = (w_crisis > mmpf->config.baseline_gate_on);
        int should_unfreeze = (w_crisis < mmpf->config.baseline_gate_off);

        if (!currently_frozen && should_freeze)
        {
            mmpf->baseline_frozen_ticks = 1;
        }
        else if (currently_frozen && should_unfreeze)
        {
            rbpf_real_t alpha = mmpf->config.global_mu_vol_alpha;
            mmpf->global_mu_vol = alpha * mmpf->global_mu_vol +
                                  (RBPF_REAL(1.0) - alpha) * mmpf->prev_weighted_log_vol;
            mmpf->baseline_frozen_ticks = 0;
        }
        else if (currently_frozen)
        {
            mmpf->baseline_frozen_ticks++;
        }
        else
        {
            rbpf_real_t alpha = mmpf->config.global_mu_vol_alpha;
            mmpf->global_mu_vol = alpha * mmpf->global_mu_vol +
                                  (RBPF_REAL(1.0) - alpha) * mmpf->prev_weighted_log_vol;
        }

        for (k = 0; k < MMPF_N_MODELS; k++)
        {
            rbpf_real_t mu_k = mmpf->global_mu_vol + mmpf->config.mu_vol_offsets[k];
            RBPF_Extended *ext = mmpf->ext[k];
            for (r = 0; r < ext->rbpf->n_regimes; r++)
            {
                ext->rbpf->params[r].mu_vol = mu_k;
            }
            mmpf->gated_dynamics[k].mu_vol = (double)mu_k;
        }
    }

    /* IMM mixing */
    update_transition_matrix(mmpf);
    compute_mixing_weights(mmpf);
    compute_mixing_counts(mmpf);
    imm_mixing_step(mmpf);

    if (mmpf->config.enable_storvik_sync)
    {
        for (k = 0; k < MMPF_N_MODELS; k++)
        {
            mmpf_sync_parameters(mmpf->ext[k]);
        }
    }

    /* Handle zero returns */
    if (rbpf_fabs(y) < RBPF_REAL(1e-10))
    {
        switch (mmpf->config.zero_return_policy)
        {
        case 0:
            skip_update = 1;
            y_log = RBPF_REAL(0.0);
            break;
        case 1:
        default:
            y_log = mmpf->config.min_log_return_sq;
            break;
        }
    }
    else
    {
        y_log = rbpf_log(y * y);
        if (y_log < mmpf->config.min_log_return_sq)
        {
            y_log = mmpf->config.min_log_return_sq;
        }
    }

    /* Step each RBPF */
    for (k = 0; k < MMPF_N_MODELS; k++)
    {
        RBPF_Extended *ext = mmpf->ext[k];
        RBPF_KSC *rbpf = ext->rbpf;
        RBPF_RobustOCSN *ocsn = &ext->robust_ocsn;
        RBPF_KSC_Output *out = &mmpf->model_output[k];
        rbpf_real_t marginal_lik;

        rbpf_ksc_transition(rbpf);
        rbpf_ksc_predict(rbpf);

        if (skip_update)
        {
            marginal_lik = RBPF_REAL(1.0);
            for (i = 0; i < rbpf->n_particles; i++)
            {
                rbpf->mu[i] = rbpf->mu_pred[i];
                rbpf->var[i] = rbpf->var_pred[i];
            }
        }
        else
        {
            if (mmpf->config.enable_student_t && rbpf->student_t_enabled)
            {
                rbpf_real_t nu = mmpf->learned_nu[k];
                marginal_lik = rbpf_ksc_update_student_t_robust_mkl(rbpf, y_log, nu, ocsn);
            }
            else if (ocsn->enabled)
            {
                marginal_lik = rbpf_ksc_update_robust(rbpf, y_log, ocsn);
            }
            else
            {
                marginal_lik = rbpf_ksc_update(rbpf, y_log);
            }
        }

        rbpf_ksc_compute_outputs(rbpf, marginal_lik, out);

        if (!skip_update)
        {
            out->resampled = rbpf_ksc_resample(rbpf);
        }
        else
        {
            out->resampled = 0;
        }

        if (ocsn->enabled && !skip_update)
        {
            out->outlier_fraction = rbpf_ksc_compute_outlier_fraction(rbpf, y_log, ocsn);
        }
        else
        {
            out->outlier_fraction = RBPF_REAL(0.0);
        }

        /* Gated Storvik update with Swim Lanes
         *
         * Swim lanes handle:
         *   - Pinning (learning_rate_scale = 0 for Calm)
         *   - Slow learning (learning_rate_scale = 0.1 for Crisis)
         *   - Full learning (learning_rate_scale = 1.0 for Trend)
         *   - Bounds clamping (mu_vol and sigma_vol ranges)
         *
         * Weight gate still applies — don't learn from data you don't own.
         */
        if (!skip_update && mmpf->config.enable_storvik_sync)
        {
            rbpf_real_t w_k = mmpf->weights[k];
            int passes_weight_gate = (w_k >= RBPF_REAL(0.10));

            if (passes_weight_gate)
            {
                /* Swim lane logic is inside mmpf_update_storvik_for_hypothesis */
                mmpf_update_storvik_for_hypothesis(mmpf, k, out->resampled);
            }
            else
            {
                /* Below weight gate — just update lag buffers */
                const int n = ext->rbpf->n_particles;
                for (i = 0; i < n; i++)
                {
                    ext->ell_lag_buffer[i] = ext->rbpf->mu[i];
                    ext->prev_regime[i] = ext->rbpf->regime[i];
                }
            }
        }

        /* Adaptive Forgetting Update
         *
         * Adjusts Storvik's λ (forgetting factor) based on predictive surprise.
         * Per-hypothesis configuration set in mmpf_create():
         *   - Calm: DISABLED (anchor doesn't adapt)
         *   - Trend: AGGRESSIVE with COMBINED signal
         *   - Crisis: RESTRICTED with PREDICTIVE_SURPRISE only
         *
         * Must be called AFTER Storvik update so λ affects next tick.
         */
        if (!skip_update)
        {
            /* Store outlier fraction for adaptive forgetting's Path B */
            ext->last_outlier_fraction = out->outlier_fraction;

            /* Update adaptive forgetting — uses marginal_lik and dominant regime */
            rbpf_adaptive_forgetting_update(ext, marginal_lik, rbpf->regime[0]);
        }

        mmpf->model_likelihood[k] = marginal_lik;
    }

    /* IMM Bayesian weight update */
    rbpf_real_t c[MMPF_N_MODELS];
    for (int j = 0; j < MMPF_N_MODELS; j++)
    {
        c[j] = RBPF_REAL(0.0);
        for (i = 0; i < MMPF_N_MODELS; i++)
        {
            c[j] += mmpf->weights[i] * mmpf->transition[i][j];
        }
        if (c[j] < RBPF_REAL(1e-6))
            c[j] = RBPF_REAL(1e-6);
    }

    for (k = 0; k < MMPF_N_MODELS; k++)
    {
        mmpf->log_weights[k] = rbpf_log(c[k]) + rbpf_log(mmpf->model_likelihood[k] + RBPF_EPS);
    }

    mmpf_log_to_linear(mmpf->log_weights, mmpf->weights, MMPF_N_MODELS);

    rbpf_real_t lse = mmpf_log_sum_exp(mmpf->log_weights, MMPF_N_MODELS);
    for (k = 0; k < MMPF_N_MODELS; k++)
    {
        mmpf->log_weights[k] -= lse;
    }

    /* Compute weighted outputs */
    mmpf->weighted_vol = RBPF_REAL(0.0);
    mmpf->weighted_log_vol = RBPF_REAL(0.0);

    for (k = 0; k < MMPF_N_MODELS; k++)
    {
        mmpf->weighted_vol += mmpf->weights[k] * mmpf->model_output[k].vol_mean;
        mmpf->weighted_log_vol += mmpf->weights[k] * mmpf->model_output[k].log_vol_mean;
    }

    /* Law of Total Variance */
    rbpf_real_t between_var = RBPF_REAL(0.0);
    for (k = 0; k < MMPF_N_MODELS; k++)
    {
        rbpf_real_t diff = mmpf->model_output[k].vol_mean - mmpf->weighted_vol;
        between_var += mmpf->weights[k] * diff * diff;
    }
    mmpf->between_model_var = between_var;

    rbpf_real_t within_var = RBPF_REAL(0.0);
    for (k = 0; k < MMPF_N_MODELS; k++)
    {
        rbpf_real_t log_vol_var = mmpf->model_output[k].log_vol_var;
        rbpf_real_t vol_mean = mmpf->model_output[k].vol_mean;
        rbpf_real_t model_var = vol_mean * vol_mean * log_vol_var;
        within_var += mmpf->weights[k] * model_var;
    }
    mmpf->within_model_var = within_var;

    mmpf->weighted_vol_std = rbpf_sqrt(between_var + within_var);

    /* Determine dominant */
    int dom = mmpf_argmax(mmpf->weights, MMPF_N_MODELS);
    mmpf->prev_dominant = mmpf->dominant;
    mmpf->dominant = (MMPF_Hypothesis)dom;

    if (mmpf->dominant == mmpf->prev_dominant)
    {
        mmpf->ticks_in_regime++;
    }
    else
    {
        mmpf->ticks_in_regime = 1;
        mmpf->regime_switches++;
    }

    mmpf->outlier_fraction = RBPF_REAL(0.0);
    for (k = 0; k < MMPF_N_MODELS; k++)
    {
        mmpf->outlier_fraction += mmpf->weights[k] * mmpf->model_output[k].outlier_fraction;
    }

    mmpf->total_steps++;
    mmpf->prev_weighted_log_vol = mmpf->weighted_log_vol;

    /* Gated dynamics state tracking */
    if (!skip_update)
    {
        if (mmpf->config.enable_storvik_sync)
        {
            for (k = 0; k < MMPF_N_MODELS; k++)
            {
                rbpf_real_t x_curr = mmpf->model_output[k].log_vol_mean;
                if (x_curr == x_curr)
                {
                    mmpf->gated_dynamics[k].prev_state = (double)x_curr;
                }
            }
        }
        else if (mmpf->config.enable_gated_learning)
        {
            /* Manual WTA learning fallback - abbreviated for brevity */
            int dominant = mmpf_argmax(mmpf->weights, MMPF_N_MODELS);
            for (k = 0; k < MMPF_N_MODELS; k++)
            {
                rbpf_real_t x_curr = mmpf->model_output[k].log_vol_mean;
                if (x_curr == x_curr)
                {
                    mmpf->gated_dynamics[k].prev_state = (double)x_curr;
                }
            }
        }
    }

    /* WTA ν learning */
    if (mmpf->config.enable_student_t && mmpf->config.enable_nu_learning && !skip_update)
    {
        int dominant = (int)mmpf->dominant;
        for (k = 0; k < MMPF_N_MODELS; k++)
        {
            if (k == dominant)
            {
                mmpf->learned_nu[k] = rbpf_ksc_get_nu(mmpf->ext[k]->rbpf, 0);
            }
        }
    }

    /* Fill output */
    if (output)
    {
        output->volatility = mmpf->weighted_vol;
        output->log_volatility = mmpf->weighted_log_vol;
        output->volatility_std = mmpf->weighted_vol_std;
        output->between_model_var = between_var;
        output->within_model_var = within_var;
        output->update_skipped = skip_update;

        for (k = 0; k < MMPF_N_MODELS; k++)
        {
            output->weights[k] = mmpf->weights[k];
            output->model_vol[k] = mmpf->model_output[k].vol_mean;
            output->model_log_vol[k] = mmpf->model_output[k].log_vol_mean;
            output->model_log_vol_var[k] = mmpf->model_output[k].log_vol_var;
            output->model_likelihood[k] = mmpf->model_likelihood[k];
            output->model_ess[k] = mmpf->model_output[k].ess;
        }

        output->dominant = mmpf->dominant;
        output->dominant_prob = mmpf->weights[dom];
        output->outlier_fraction = mmpf->outlier_fraction;
        output->current_stickiness = mmpf->current_stickiness;

        for (i = 0; i < MMPF_N_MODELS; i++)
        {
            for (int j = 0; j < MMPF_N_MODELS; j++)
            {
                output->mixing_weights[i][j] = mmpf->mixing_weights[i][j];
            }
        }

        output->regime_stable = (mmpf->ticks_in_regime >= 10) ? 1 : 0;
        output->ticks_in_regime = mmpf->ticks_in_regime;
        output->global_mu_vol = mmpf->global_mu_vol;
        output->baseline_frozen = (mmpf->baseline_frozen_ticks > 0) ? 1 : 0;

        output->student_t_active = mmpf->config.enable_student_t ? 1 : 0;
        for (k = 0; k < MMPF_N_MODELS; k++)
        {
            output->model_nu[k] = mmpf->learned_nu[k];
            output->model_lambda_mean[k] = mmpf->model_output[k].lambda_mean;
            output->model_nu_effective[k] = mmpf->model_output[k].nu_effective;
        }
    }
}

void mmpf_step_apf(MMPF_ROCKS *mmpf, rbpf_real_t y_current, rbpf_real_t y_next,
                   MMPF_Output *output)
{
    (void)y_next;
    mmpf_step(mmpf, y_current, output);
}