    /**
     * @file rbpf_ksc.c
     * @brief RBPF with Kim-Shephard-Chib (1998) - HOT PATH
     *
     * This file contains performance-critical functions:
     *   - predict()
     *   - update()
     *   - transition()
     *   - resample()
     *   - step()
     *
     * Lifecycle functions are in rbpf_ksc_init.c
     * Output computation is in rbpf_ksc_output.c
     *
     * Key optimizations:
     *   - Zero malloc in hot path (all buffers preallocated)
     *   - Pointer swap instead of memcpy for resampling
     *   - PCG32 RNG (fast, good quality)
     *   - Transition LUT (no cumsum search)
     *   - Regularization after resample (prevents Kalman state degeneracy)
     *   - KL tempering support (deferred weight mode)
     *
     * Latency target: <15μs for 1000 particles (Gaussian)
     *                 <20μs for 1000 particles (Student-t)
     */

    #include "rbpf_ksc.h"
    #include "rbpf_silverman.h"
    #include "rbpf_sprt.h"
    #include "rbpf_fisher_rao.h"
    #include "bocpd.h"
    #include <stdlib.h>
    #include <string.h>
    #include <math.h>
    #include <mkl_vml.h>

    /*═══════════════════════════════════════════════════════════════════════════
    * STUDENT-T COMPILE-TIME SWITCH
    *═══════════════════════════════════════════════════════════════════════════*/

    #ifndef RBPF_ENABLE_STUDENT_T
    #define RBPF_ENABLE_STUDENT_T 1
    #endif

    #if RBPF_ENABLE_STUDENT_T
    extern rbpf_real_t rbpf_ksc_update_student_t(RBPF_KSC *rbpf, rbpf_real_t y);
    #endif

    /*─────────────────────────────────────────────────────────────────────────────
    * OMORI, CHIB, SHEPHARD & NAKAJIMA (2007) MIXTURE PARAMETERS
    *
    * 10-component Gaussian mixture approximation of log(χ²(1)):
    * p(log(ε²)) ≈ Σ_k π_k × N(m_k, v_k²)
    *───────────────────────────────────────────────────────────────────────────*/

    static const rbpf_real_t KSC_PROB[KSC_N_COMPONENTS] = {
        RBPF_REAL(0.00609), RBPF_REAL(0.04775), RBPF_REAL(0.13057), RBPF_REAL(0.20674),
        RBPF_REAL(0.22715), RBPF_REAL(0.18842), RBPF_REAL(0.12047), RBPF_REAL(0.05591),
        RBPF_REAL(0.01575), RBPF_REAL(0.00115)};

    static const rbpf_real_t KSC_MEAN[KSC_N_COMPONENTS] = {
        RBPF_REAL(1.92677), RBPF_REAL(1.34744), RBPF_REAL(0.73504), RBPF_REAL(0.02266),
        RBPF_REAL(-0.85173), RBPF_REAL(-1.97278), RBPF_REAL(-3.46788), RBPF_REAL(-5.55246),
        RBPF_REAL(-8.68384), RBPF_REAL(-14.65000)};

    static const rbpf_real_t KSC_VAR[KSC_N_COMPONENTS] = {
        RBPF_REAL(0.11265), RBPF_REAL(0.17788), RBPF_REAL(0.26768), RBPF_REAL(0.40611),
        RBPF_REAL(0.62699), RBPF_REAL(0.98583), RBPF_REAL(1.57469), RBPF_REAL(2.54498),
        RBPF_REAL(4.16591), RBPF_REAL(7.33342)};

    /* Precomputed: log(π_k) for each Omori (2007) component */
    static const rbpf_real_t KSC_LOG_PROB[KSC_N_COMPONENTS] = {
        RBPF_REAL(-5.101), /* log(0.00609) */
        RBPF_REAL(-3.042), /* log(0.04775) */
        RBPF_REAL(-2.036), /* log(0.13057) */
        RBPF_REAL(-1.577), /* log(0.20674) */
        RBPF_REAL(-1.482), /* log(0.22715) */
        RBPF_REAL(-1.669), /* log(0.18842) */
        RBPF_REAL(-2.116), /* log(0.12047) */
        RBPF_REAL(-2.884), /* log(0.05591) */
        RBPF_REAL(-4.151), /* log(0.01575) */
        RBPF_REAL(-6.768)  /* log(0.00115) */
    };

    /*─────────────────────────────────────────────────────────────────────────────
    * FISHER-RAO MUTATION (Optimized with Small-Angle Approximation)
    *
    * For particles being mutated to a new regime, we interpolate along the
    * Fisher-Rao geodesic. When |Δμ| is small relative to σ, we use a faster
    * linear/geometric approximation instead of the full trig computation.
    *───────────────────────────────────────────────────────────────────────────*/

    static inline void fisher_rao_mutate_fast(
        double mu_particle, double var_particle,
        double mu_regime, double theta_regime, double sigma_regime,
        double *mu_out, double *var_out)
    {
        double sigma_particle = sqrt(var_particle);
        double delta_mu = fabs(mu_regime - mu_particle);

        /* Small-angle approximation threshold: geodesic is nearly vertical */
        if (delta_mu < 0.1 * sigma_particle)
        {
            /* Skip trig - use linear μ blend and geometric σ blend */
            double var_stationary = (sigma_regime * sigma_regime) / (2.0 * theta_regime);
            if (var_stationary < 0.01)
                var_stationary = 0.01;
            if (var_stationary > 10.0)
                var_stationary = 10.0;

            /* Precision-weighted blend parameter */
            double prec_particle = 1.0 / var_particle;
            double prec_regime = 1.0 / var_stationary;
            double t = prec_regime / (prec_particle + prec_regime);

            /* Linear blend for μ, geometric for σ */
            *mu_out = (1.0 - t) * mu_particle + t * mu_regime;
            double sigma_out = pow(sigma_particle, 1.0 - t) * pow(sqrt(var_stationary), t);
            *var_out = sigma_out * sigma_out;
        }
        else
        {
            /* Full geodesic computation (rare path) */
            fisher_rao_mutate(mu_particle, var_particle,
                            mu_regime, theta_regime, sigma_regime,
                            mu_out, var_out);
        }
    }

    /*─────────────────────────────────────────────────────────────────────────────
    * PREDICT STEP
    *
    * ℓ_t = (1-θ)ℓ_{t-1} + θμ + η_t,  η_t ~ N(0, q)
    *
    * Kalman predict:
    *   μ_pred = (1-θ)μ + θμ_vol
    *   P_pred = (1-θ)²P + q
    *───────────────────────────────────────────────────────────────────────────*/

    void rbpf_ksc_predict(RBPF_KSC *rbpf)
    {
        const int n = rbpf->n_particles;
        const RBPF_RegimeParams *params = rbpf->params;
        const int n_regimes = rbpf->n_regimes;

        rbpf_real_t *restrict mu = rbpf->mu;
        rbpf_real_t *restrict var = rbpf->var;
        const int *restrict regime = rbpf->regime;
        rbpf_real_t *restrict mu_pred = rbpf->mu_pred;
        rbpf_real_t *restrict var_pred = rbpf->var_pred;

        /* Per-particle parameters: Set by external learner (Storvik via MMPF) */
        const int use_particles = rbpf->use_learned_params;

        if (use_particles)
        {
            /* Use per-particle learned parameters (Storvik or Liu-West) */
            const rbpf_real_t *particle_mu_vol = rbpf->particle_mu_vol;
            const rbpf_real_t *particle_sigma_vol = rbpf->particle_sigma_vol;

            for (int i = 0; i < n; i++)
            {
                int r = regime[i];
                rbpf_real_t theta = params[r].theta;
                rbpf_real_t omt = RBPF_REAL(1.0) - theta;
                rbpf_real_t omt2 = omt * omt;

                int idx = i * n_regimes + r;
                rbpf_real_t mv = particle_mu_vol[idx];
                rbpf_real_t sigma_vol = particle_sigma_vol[idx];
                rbpf_real_t q = sigma_vol * sigma_vol;

                mu_pred[i] = omt * mu[i] + theta * mv;
                var_pred[i] = omt2 * var[i] + q;
            }
        }
        else
        {
            /* Use global regime parameters (no learning) */
            rbpf_real_t theta_r[RBPF_MAX_REGIMES];
            rbpf_real_t mu_vol_r[RBPF_MAX_REGIMES];
            rbpf_real_t q_r[RBPF_MAX_REGIMES];
            rbpf_real_t one_minus_theta_r[RBPF_MAX_REGIMES];
            rbpf_real_t one_minus_theta_sq_r[RBPF_MAX_REGIMES];

            for (int r = 0; r < n_regimes; r++)
            {
                theta_r[r] = params[r].theta;
                mu_vol_r[r] = params[r].mu_vol;
                q_r[r] = params[r].q;
                one_minus_theta_r[r] = RBPF_REAL(1.0) - theta_r[r];
                one_minus_theta_sq_r[r] = one_minus_theta_r[r] * one_minus_theta_r[r];
            }

            for (int i = 0; i < n; i++)
            {
                int r = regime[i];
                rbpf_real_t omt = one_minus_theta_r[r];
                rbpf_real_t omt2 = one_minus_theta_sq_r[r];
                rbpf_real_t th = theta_r[r];
                rbpf_real_t mv = mu_vol_r[r];
                rbpf_real_t q = q_r[r];

                mu_pred[i] = omt * mu[i] + th * mv;
                var_pred[i] = omt2 * var[i] + q;
            }
        }
    }

    /*─────────────────────────────────────────────────────────────────────────────
    * UPDATE STEP (optimized 10-component Omori mixture Kalman)
    *
    * Observation: y = log(r²) = 2ℓ + log(ε²)
    * Linear: y - m_k = H*ℓ + (log(ε²) - m_k), H = 2
    *
    * KL TEMPERING SUPPORT:
    * When deferred_weight_mode is enabled, log-likelihood increments are stored
    * in log_lik_increment[] but NOT applied to log_weight[]. The caller
    * (Extended layer) then computes KL divergence and applies tempered weights.
    *───────────────────────────────────────────────────────────────────────────*/

    rbpf_real_t rbpf_ksc_update(RBPF_KSC *rbpf, rbpf_real_t y)
    {
        const int n = rbpf->n_particles;
        const rbpf_real_t H = RBPF_REAL(2.0);
        const rbpf_real_t H2 = RBPF_REAL(4.0);
        const rbpf_real_t NEG_HALF = RBPF_REAL(-0.5);

        rbpf_real_t *restrict mu_pred = rbpf->mu_pred;
        rbpf_real_t *restrict var_pred = rbpf->var_pred;
        rbpf_real_t *restrict mu = rbpf->mu;
        rbpf_real_t *restrict var = rbpf->var;
        rbpf_real_t *restrict log_weight = rbpf->log_weight;
        rbpf_real_t *restrict lik_total = rbpf->lik_total;
        rbpf_real_t *restrict mu_accum = rbpf->mu_accum;
        rbpf_real_t *restrict var_accum = rbpf->var_accum;
        rbpf_real_t *restrict log_lik_buf = rbpf->log_lik_buffer;
        rbpf_real_t *restrict max_ll = rbpf->max_log_lik;

        /* Initialize max to very negative */
        for (int i = 0; i < n; i++)
        {
            max_ll[i] = RBPF_REAL(-1e30);
        }

        /* Pass 1: Compute log-likelihoods for all components */
        for (int k = 0; k < KSC_N_COMPONENTS; k++)
        {
            const rbpf_real_t m_k = KSC_MEAN[k];
            const rbpf_real_t v2_k = KSC_VAR[k];
            const rbpf_real_t log_pi_k = KSC_LOG_PROB[k];
            const rbpf_real_t y_adj = y - m_k;

            rbpf_real_t *log_lik_k = log_lik_buf + k * n;

            RBPF_PRAGMA_SIMD
            for (int i = 0; i < n; i++)
            {
                rbpf_real_t innov = y_adj - H * mu_pred[i];
                rbpf_real_t S = H2 * var_pred[i] + v2_k;
                rbpf_real_t innov2_S = innov * innov / S;
                rbpf_real_t log_lik = NEG_HALF * (rbpf_log(S) + innov2_S) + log_pi_k;

                log_lik_k[i] = log_lik;

                if (log_lik > max_ll[i])
                    max_ll[i] = log_lik;
            }
        }

        /* Zero accumulators */
        memset(lik_total, 0, n * sizeof(rbpf_real_t));
        memset(mu_accum, 0, n * sizeof(rbpf_real_t));
        memset(var_accum, 0, n * sizeof(rbpf_real_t));

        /* Pass 2: Compute normalized likelihoods and accumulate */
        for (int k = 0; k < KSC_N_COMPONENTS; k++)
        {
            const rbpf_real_t m_k = KSC_MEAN[k];
            const rbpf_real_t v2_k = KSC_VAR[k];
            const rbpf_real_t y_adj = y - m_k;

            rbpf_real_t *log_lik_k = log_lik_buf + k * n;

            RBPF_PRAGMA_SIMD
            for (int i = 0; i < n; i++)
            {
                rbpf_real_t lik = rbpf_exp(log_lik_k[i] - max_ll[i]);
                lik_total[i] += lik;

                rbpf_real_t innov = y_adj - H * mu_pred[i];
                rbpf_real_t S = H2 * var_pred[i] + v2_k;
                rbpf_real_t K = H * var_pred[i] / S;
                rbpf_real_t mu_k = mu_pred[i] + K * innov;
                rbpf_real_t var_k = (RBPF_REAL(1.0) - K * H) * var_pred[i];

                mu_accum[i] += lik * mu_k;
                var_accum[i] += lik * (var_k + mu_k * mu_k);
            }
        }

        /*═══════════════════════════════════════════════════════════════════════
        * NORMALIZE AND UPDATE WEIGHTS
        *
        * KL Tempering Support:
        * - Always store log-likelihood increment in log_lik_increment[]
        * - Only apply to log_weight[] if NOT in deferred mode
        * - In deferred mode, Extended layer applies tempered weights
        *═══════════════════════════════════════════════════════════════════════*/

        rbpf_real_t total_marginal = RBPF_REAL(0.0);
        rbpf_real_t *log_lik_inc = rbpf->log_lik_increment; /* KL tempering buffer */
        const int deferred = rbpf->deferred_weight_mode;

        for (int i = 0; i < n; i++)
        {
            rbpf_real_t inv_lik = RBPF_REAL(1.0) / (lik_total[i] + RBPF_REAL(1e-30));

            rbpf_real_t mean_final = mu_accum[i] * inv_lik;
            rbpf_real_t E_X2 = var_accum[i] * inv_lik;
            rbpf_real_t var_final = E_X2 - mean_final * mean_final;

            mu[i] = mean_final;
            var[i] = var_final;

            if (var[i] < RBPF_REAL(1e-6))
                var[i] = RBPF_REAL(1e-6);

            /* Compute log-likelihood increment for this particle */
            rbpf_real_t inc = rbpf_log(lik_total[i] + RBPF_REAL(1e-30)) + max_ll[i];

            /* Always store for KL tempering (even if not enabled) */
            log_lik_inc[i] = inc;

            /* Apply immediately only if NOT in deferred mode */
            if (!deferred)
            {
                log_weight[i] += inc;
            }
            /* In deferred mode, caller applies: log_weight += β × log_lik_increment */

            total_marginal += lik_total[i] * rbpf_exp(max_ll[i]);
        }

        return total_marginal / n;
    }

    /*─────────────────────────────────────────────────────────────────────────────
    * REGIME TRANSITION (LUT-based, no cumsum search)
    *───────────────────────────────────────────────────────────────────────────*/

    void rbpf_ksc_transition(RBPF_KSC *rbpf)
    {
        const int n = rbpf->n_particles;
        int *regime = rbpf->regime;
        rbpf_pcg32_t *rng = &rbpf->pcg[0];

        for (int i = 0; i < n; i++)
        {
            int r_old = regime[i];
            rbpf_real_t u = rbpf_pcg32_uniform(rng);
            int lut_idx = (int)(u * RBPF_REAL(1023.0));
            regime[i] = rbpf->trans_lut[r_old][lut_idx];
        }
    }

    /*─────────────────────────────────────────────────────────────────────────────
    * RESAMPLE (systematic + regularization)
    *───────────────────────────────────────────────────────────────────────────*/

    int rbpf_ksc_resample(RBPF_KSC *rbpf)
    {
        const int n = rbpf->n_particles;

        rbpf_real_t *log_weight = rbpf->log_weight;
        rbpf_real_t *w_norm = rbpf->w_norm;
        rbpf_real_t *cumsum = rbpf->cumsum;
        int *indices = rbpf->indices;

        /* Find max log-weight for numerical stability */
        rbpf_real_t max_lw = log_weight[0];
        for (int i = 1; i < n; i++)
        {
            if (log_weight[i] > max_lw)
                max_lw = log_weight[i];
        }

        /* Normalize: w = exp(lw - max) / sum */
        for (int i = 0; i < n; i++)
        {
            w_norm[i] = log_weight[i] - max_lw;
        }
        rbpf_vsExp(n, w_norm, w_norm);

        rbpf_real_t sum_w = rbpf_cblas_asum(n, w_norm, 1);
        if (sum_w < RBPF_REAL(1e-30))
        {
            rbpf_real_t uw = rbpf->uniform_weight;
            for (int i = 0; i < n; i++)
            {
                w_norm[i] = uw;
            }
            sum_w = RBPF_REAL(1.0);
        }
        rbpf_cblas_scal(n, RBPF_REAL(1.0) / sum_w, w_norm, 1);

        /* Compute ESS */
        rbpf_real_t sum_w2 = rbpf_cblas_dot(n, w_norm, 1, w_norm, 1);
        rbpf_real_t ess = RBPF_REAL(1.0) / sum_w2;
        rbpf->last_ess = ess;

        /* Adaptive resampling threshold */
        rbpf_real_t threshold = RBPF_REAL(0.5);

        if (ess > n * threshold)
        {
            return 0;
        }

        /* Cumulative sum */
        cumsum[0] = w_norm[0];
        for (int i = 1; i < n; i++)
        {
            cumsum[i] = cumsum[i - 1] + w_norm[i];
        }

        /* Fused systematic resampling + data copy */
        rbpf_real_t *mu = rbpf->mu;
        rbpf_real_t *var = rbpf->var;
        int *regime = rbpf->regime;
        rbpf_real_t *mu_tmp = rbpf->mu_tmp;
        rbpf_real_t *var_tmp = rbpf->var_tmp;
        int *regime_tmp = rbpf->regime_tmp;

        rbpf_real_t u0 = rbpf_pcg32_uniform(&rbpf->pcg[0]) * rbpf->inv_n;
        int j = 0;
        for (int i = 0; i < n; i++)
        {
            rbpf_real_t u = u0 + (rbpf_real_t)i * rbpf->inv_n;
            while (j < n - 1 && cumsum[j] < u)
                j++;

            indices[i] = j;
            mu_tmp[i] = mu[j];
            var_tmp[i] = var[j];
            regime_tmp[i] = regime[j];
        }

        /* Pointer swap */
        rbpf->mu = mu_tmp;
        rbpf->mu_tmp = mu;
        rbpf->var = var_tmp;
        rbpf->var_tmp = var;
        rbpf->regime = regime_tmp;
        rbpf->regime_tmp = regime;

        /* Reset log-weights */
        memset(rbpf->log_weight, 0, n * sizeof(rbpf_real_t));

        /* Apply regularization (kernel jitter) */
        rbpf_real_t h_mu;

        if (rbpf->use_silverman_bandwidth && rbpf->silverman_scratch != NULL)
        {
            /* NOTE: Use mu_tmp (old data before swap) to match original behavior.
            * After pointer swap, rbpf->mu_tmp holds the pre-resampled distribution
            * which has more diversity for bandwidth estimation. */
    #ifndef RBPF_USE_DOUBLE
            h_mu = rbpf_silverman_bandwidth_f(rbpf->mu_tmp, n, rbpf->silverman_scratch);
    #else
            h_mu = (rbpf_real_t)rbpf_silverman_bandwidth(
                (const double *)rbpf->mu_tmp, n, (double *)rbpf->silverman_scratch);
    #endif

            if (h_mu < RBPF_REAL(0.001))
                h_mu = RBPF_REAL(0.001);
            if (h_mu > RBPF_REAL(0.5))
                h_mu = RBPF_REAL(0.5);

            rbpf->last_silverman_bandwidth = h_mu;
        }
        else
        {
            rbpf_real_t ess_ratio = ess / (rbpf_real_t)n;
            rbpf_real_t scale = rbpf->reg_scale_max -
                                (rbpf->reg_scale_max - rbpf->reg_scale_min) * ess_ratio;
            if (scale < rbpf->reg_scale_min)
                scale = rbpf->reg_scale_min;
            if (scale > rbpf->reg_scale_max)
                scale = rbpf->reg_scale_max;

            h_mu = rbpf->reg_bandwidth_mu * scale;
        }

        rbpf_real_t h_var = rbpf->reg_bandwidth_var;

        /* Generate Gaussian randoms in batch using MKL */
        rbpf_real_t *gauss = rbpf->rng_gaussian;
        RBPF_VSL_RNG_GAUSSIAN(VSL_RNG_METHOD_GAUSSIAN_ICDF, rbpf->mkl_rng[0],
                            2 * n, gauss, RBPF_REAL(0.0), RBPF_REAL(1.0));

        /* Apply jitter */
        mu = rbpf->mu;
        var = rbpf->var;
        regime = rbpf->regime;

        RBPF_PRAGMA_SIMD
        for (int i = 0; i < n; i++)
        {
            mu[i] += h_mu * gauss[i];
            var[i] += h_var * rbpf_fabs(gauss[n + i]);
            if (var[i] < RBPF_REAL(1e-6))
                var[i] = RBPF_REAL(1e-6);
        }

        /* Regime diversity - "Pilot Light" safety net
        *
        * Prevents P(regime=k) = 0 which is an absorbing state in Bayes' rule.
        * Primary regime switching is handled by BOCPD + SPRT ("afterburner"),
        * but we keep 1-2 particles per regime as mathematical insurance.
        *
        * Uses Fisher-Rao geodesic for principled state blending.
        * The blend parameter t is determined by precision weighting:
        *   - Uncertain particle (high var) → large t → teleport toward regime
        *   - Confident particle (low var) → small t → preserve state
        */
        if (rbpf->regime_mutation_prob > RBPF_REAL(0.0))
        {
            int n_regimes = rbpf->n_regimes;
            rbpf_pcg32_t *rng_mut = &rbpf->pcg[0];

            int regime_count[RBPF_MAX_REGIMES] = {0};
            for (int i = 0; i < n; i++)
            {
                regime_count[regime[i]]++;
            }

            int min_count = rbpf->min_particles_per_regime;

            for (int i = 0; i < n; i++)
            {
                int r = regime[i];

                /* Only mutate from over-represented regimes */
                if (regime_count[r] > min_count * 2)
                {
                    if (rbpf_pcg32_uniform(rng_mut) < rbpf->regime_mutation_prob)
                    {
                        /* Find under-represented regime */
                        for (int r_new = 0; r_new < n_regimes; r_new++)
                        {
                            if (regime_count[r_new] < min_count)
                            {
                                regime_count[r]--;
                                regime_count[r_new]++;
                                regime[i] = r_new;

                                /* ═══════════════════════════════════════════════
                                * FISHER-RAO GEODESIC MUTATION (Optimized)
                                *
                                * Uses small-angle approximation when |Δμ| < 0.1σ
                                * to avoid expensive trig functions.
                                * ═══════════════════════════════════════════════*/

                                const RBPF_RegimeParams *p_new = &rbpf->params[r_new];

                                double mu_out, var_out;
                                fisher_rao_mutate_fast(
                                    (double)mu[i],
                                    (double)var[i],
                                    (double)p_new->mu_vol,
                                    (double)p_new->theta,
                                    (double)p_new->sigma_vol,
                                    &mu_out,
                                    &var_out);

                                mu[i] = (rbpf_real_t)mu_out;
                                var[i] = (rbpf_real_t)var_out;

                                break;
                            }
                        }
                    }
                }
            }
        }

        /* Student-t resample */
    #if RBPF_ENABLE_STUDENT_T
        if (rbpf->student_t_enabled)
        {
            extern void rbpf_ksc_resample_student_t(RBPF_KSC * rbpf, const int *indices);
            rbpf_ksc_resample_student_t(rbpf, indices);
        }
    #endif

        return 1;
    }

    /*─────────────────────────────────────────────────────────────────────────────
    * BOCPD HELPER FUNCTIONS (for MMPF integration - not used in single RBPF)
    *───────────────────────────────────────────────────────────────────────────*/

    /* E[log(χ²₁)] = ψ(0.5) + log(2) ≈ -1.27
    * Used to convert y = 2h + log(ε²) back to implied h */
    #define PSI_CORRECTION RBPF_REAL(-1.27036)

    /*─────────────────────────────────────────────────────────────────────────────
    * MAIN UPDATE - THE HOT PATH
    *───────────────────────────────────────────────────────────────────────────*/

    void rbpf_ksc_step(RBPF_KSC *rbpf, rbpf_real_t obs, RBPF_KSC_Output *output)
    {
        /* Transform observation: y = log(r²) */
        rbpf_real_t y;
        if (rbpf_fabs(obs) < RBPF_REAL(1e-10))
        {
            y = RBPF_REAL(-23.0);
        }
        else
        {
            y = rbpf_log(obs * obs);
        }

        /* Initialize output flags */
        output->bocpd_triggered = 0;
        output->regime_changed = 0;

        /* 1. Regime transition */
        rbpf_ksc_transition(rbpf);

        /* 2. Kalman predict */
        rbpf_ksc_predict(rbpf);

        /* 3. Mixture Kalman update */
        rbpf_real_t marginal_lik;

    #if RBPF_ENABLE_STUDENT_T
        if (rbpf->student_t_enabled)
        {
            marginal_lik = rbpf_ksc_update_student_t(rbpf, y);
        }
        else
        {
            marginal_lik = rbpf_ksc_update(rbpf, y);
        }
    #else
        marginal_lik = rbpf_ksc_update(rbpf, y);
    #endif

        /* Store observation for SPRT likelihood computation */
        rbpf->last_y = y;

        /* 4. Compute outputs */
        rbpf_ksc_compute_outputs(rbpf, marginal_lik, output);

        /* 5. Resample if needed */
        output->resampled = rbpf_ksc_resample(rbpf);

        /* Output current regime parameters */
        for (int r = 0; r < rbpf->n_regimes; r++)
        {
            output->learned_mu_vol[r] = rbpf->params[r].mu_vol;
            output->learned_sigma_vol[r] = rbpf->params[r].sigma_vol;
        }
    }