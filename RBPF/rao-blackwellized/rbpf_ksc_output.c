/**
 * @file rbpf_ksc_output.c
 * @brief RBPF-KSC Output Computation
 *
 * This file contains:
 *   - compute_outputs() with all detection logic
 *   - SPRT regime detection
 *   - Fixed-lag smoothing
 *   - Self-aware signals (surprise, entropy, vol ratio)
 *
 * Hot path is in rbpf_ksc.c
 * Lifecycle functions are in rbpf_ksc_init.c
 */

#include "rbpf_ksc.h"
#include "rbpf_sprt.h"
#include "rbpf_dirichlet_transition.h"
#include <string.h>
#include <math.h>
#include <mkl_vml.h>

/*═══════════════════════════════════════════════════════════════════════════
 * STUDENT-T COMPILE-TIME SWITCH
 *═══════════════════════════════════════════════════════════════════════════*/

#ifndef RBPF_ENABLE_STUDENT_T
#define RBPF_ENABLE_STUDENT_T 1
#endif

/*─────────────────────────────────────────────────────────────────────────────
 * COMPUTE OUTPUTS
 *───────────────────────────────────────────────────────────────────────────*/

 /**
 * @brief Rebuild transition LUT from Dirichlet posterior
 * 
 * The RBPF uses a uint8_t[regime][1024] LUT for fast transition sampling.
 * This function rebuilds it from the current Dirichlet posterior.
 */
static void rbpf_rebuild_trans_lut_from_dirichlet(RBPF_KSC *rbpf)
{
    const int n_regimes = rbpf->n_regimes;
    const DirichletTransition *dt = &rbpf->trans_prior;
    
    for (int r = 0; r < n_regimes; r++)
    {
        /* Build cumulative distribution */
        float cumsum[RBPF_MAX_REGIMES];
        cumsum[0] = dt->prob[r][0];
        for (int j = 1; j < n_regimes; j++)
        {
            cumsum[j] = cumsum[j - 1] + dt->prob[r][j];
        }
        
        /* Fill LUT: for each u ∈ [0, 1024), find smallest j where cumsum[j] > u/1024 */
        for (int i = 0; i < 1024; i++)
        {
            float u = (float)i / 1024.0f;
            int next = n_regimes - 1;
            for (int j = 0; j < n_regimes - 1; j++)
            {
                if (u < cumsum[j])
                {
                    next = j;
                    break;
                }
            }
            rbpf->trans_lut[r][i] = (uint8_t)next;
        }
    }
}

void rbpf_ksc_compute_outputs(RBPF_KSC *rbpf, rbpf_real_t marginal_lik,
                              RBPF_KSC_Output *out)
{
    const int n = rbpf->n_particles;
    const int n_regimes = rbpf->n_regimes;

    rbpf_real_t *log_weight = rbpf->log_weight;
    rbpf_real_t *mu = rbpf->mu;
    rbpf_real_t *var = rbpf->var;
    int *regime = rbpf->regime;
    rbpf_real_t *w_norm = rbpf->w_norm;

    /* Normalize weights */
    rbpf_real_t max_lw = log_weight[0];
    for (int i = 1; i < n; i++)
    {
        if (log_weight[i] > max_lw)
            max_lw = log_weight[i];
    }

    for (int i = 0; i < n; i++)
    {
        w_norm[i] = log_weight[i] - max_lw;
    }
    rbpf_vsExp(n, w_norm, w_norm);

    rbpf_real_t sum_w = rbpf_cblas_asum(n, w_norm, 1);
    if (sum_w < RBPF_REAL(1e-30))
        sum_w = RBPF_REAL(1.0);
    rbpf_cblas_scal(n, RBPF_REAL(1.0) / sum_w, w_norm, 1);

    /*========================================================================
     * LOG-VOL MEAN AND VARIANCE (law of total variance)
     *======================================================================*/

    rbpf_real_t log_vol_mean = RBPF_REAL(0.0);
    for (int i = 0; i < n; i++)
    {
        log_vol_mean += w_norm[i] * mu[i];
    }

    rbpf_real_t log_vol_var = RBPF_REAL(0.0);
    for (int i = 0; i < n; i++)
    {
        rbpf_real_t diff = mu[i] - log_vol_mean;
        log_vol_var += w_norm[i] * (var[i] + diff * diff);
    }

    out->log_vol_mean = log_vol_mean;
    out->log_vol_var = log_vol_var;

    /*========================================================================
     * VOL MEAN: TRUE Monte Carlo estimate over particle mixture
     *
     * Each particle i represents a Gaussian: ℓ ~ N(μ_i, σ²_i)
     * For a log-normal: E[exp(ℓ)|particle i] = exp(μ_i + ½σ²_i)
     *
     * True mixture expectation:
     *   E[exp(ℓ)] = Σ_i w_i × E[exp(ℓ)|i] = Σ_i w_i × exp(μ_i + ½var_i)
     *======================================================================*/

    rbpf_real_t vol_mean = RBPF_REAL(0.0);
    for (int i = 0; i < n; i++)
    {
        vol_mean += w_norm[i] * rbpf_exp(mu[i] + RBPF_REAL(0.5) * var[i]);
    }
    out->vol_mean = vol_mean;

    /*========================================================================
     * ESS
     *======================================================================*/

    rbpf_real_t sum_w2 = rbpf_cblas_dot(n, w_norm, 1, w_norm, 1);
    out->ess = RBPF_REAL(1.0) / sum_w2;

    /*========================================================================
     * REGIME PROBABILITIES
     *======================================================================*/

    memset(out->regime_probs, 0, sizeof(out->regime_probs));
    for (int i = 0; i < n; i++)
    {
        out->regime_probs[regime[i]] += w_norm[i];
    }

    /* Dominant regime */
    int dom = 0;
    rbpf_real_t max_prob = out->regime_probs[0];
    for (int r = 1; r < n_regimes; r++)
    {
        if (out->regime_probs[r] > max_prob)
        {
            max_prob = out->regime_probs[r];
            dom = r;
        }
    }
    out->dominant_regime = dom;

    /*========================================================================
     * SPRT REGIME DETECTION + DIRICHLET TRANSITION LEARNING
     *
     * Computes per-regime observation log-likelihoods P(y | regime_k)
     * using the OCSN (2007) log-χ² mixture model, then delegates to
     * sprt_multi_update() for pairwise hypothesis testing.
     *
     * If learning enabled, updates Dirichlet prior on regime changes.
     *======================================================================*/

    RBPF_Detection *det = &rbpf->detection;

    /* Compute per-regime log-likelihoods for SPRT
     * P(y | regime k) using regime center h = μ_vol[k] */
    double log_liks[SPRT_MAX_REGIMES];
    double y_obs = (double)rbpf->last_y;

    for (int r = 0; r < n_regimes; r++)
    {
        double h_regime = (double)rbpf->params[r].mu_vol;
        log_liks[r] = sprt_logchisq_loglik(y_obs, h_regime);
    }

    /* Save current regime before update */
    int old_sprt_regime = det->stable_regime;

    /* Update SPRT module - handles pairwise tests and regime switching */
    int sprt_regime = sprt_multi_update(&rbpf->sprt, log_liks);

    /* ═══════════════════════════════════════════════════════════════════════
     * DIRICHLET TRANSITION LEARNING
     * 
     * Learn from SPRT-confirmed regime changes.
     * This is the ONLY place where we update the transition prior.
     * The prior encodes geometry (nearby regimes more likely) and
     * adapts based on observed market dynamics.
     * ═══════════════════════════════════════════════════════════════════════*/
    if (rbpf->trans_prior_enabled && sprt_regime != old_sprt_regime)
    {
        /* SPRT confirmed a regime change - learn from it */
        dirichlet_transition_update(&rbpf->trans_prior, old_sprt_regime, sprt_regime);
        
        /* Rebuild the LUT so future transitions use learned probabilities */
        rbpf_rebuild_trans_lut_from_dirichlet(rbpf);
    }

    out->smoothed_regime = sprt_regime;
    det->stable_regime = sprt_regime;

    /* Copy SPRT evidence to output for diagnostics */
    sprt_multi_get_evidence(&rbpf->sprt, out->sprt_evidence);
    /*========================================================================
     * SELF-AWARE SIGNALS
     *======================================================================*/

    out->marginal_lik = marginal_lik;
    out->surprise = -rbpf_log(marginal_lik + RBPF_REAL(1e-30));

    /* Regime entropy: -Σ p*log(p) */
    rbpf_real_t entropy = RBPF_REAL(0.0);
    for (int r = 0; r < n_regimes; r++)
    {
        rbpf_real_t p = out->regime_probs[r];
        if (p > RBPF_REAL(1e-10))
        {
            entropy -= p * rbpf_log(p);
        }
    }
    out->regime_entropy = entropy;

    /* Vol ratio (vs EMA) */
    det->vol_ema_short = RBPF_REAL(0.1) * out->vol_mean + RBPF_REAL(0.9) * det->vol_ema_short;
    det->vol_ema_long = RBPF_REAL(0.01) * out->vol_mean + RBPF_REAL(0.99) * det->vol_ema_long;
    out->vol_ratio = det->vol_ema_short / (det->vol_ema_long + RBPF_REAL(1e-10));

    /*========================================================================
     * REGIME CHANGE DETECTION
     *======================================================================*/

    out->regime_changed = 0;
    out->change_type = 0;

    if (det->cooldown > 0)
    {
        det->cooldown--;
    }
    else
    {
        /* Structural: regime flipped with high confidence */
        int structural = (dom != det->prev_regime) && (max_prob > RBPF_REAL(0.7));

        /* Vol shock: >80% increase or >50% decrease */
        int vol_shock = (out->vol_ratio > 1.8f) || (out->vol_ratio < RBPF_REAL(0.5));

        /* Surprise: observation unlikely under model */
        int surprised = (out->surprise > RBPF_REAL(5.0));

        if (structural || vol_shock || surprised)
        {
            out->regime_changed = 1;
            out->change_type = structural ? 1 : (vol_shock ? 2 : 3);
            det->cooldown = 20;
        }
    }

    det->prev_regime = dom;

    /*========================================================================
     * FIXED-LAG SMOOTHING (Dual Output)
     *
     * Store current fast estimates in circular buffer.
     * Output K-lagged estimates for regime confirmation.
     *
     * This provides:
     *   - Fast signal (t):   Immediate reaction to volatility spikes
     *   - Smooth signal (t-K): Stable regime for position sizing
     *======================================================================*/

    const int lag = rbpf->smooth_lag;

    if (lag > 0)
    {
        /* Store current fast estimates at head position */
        RBPF_SmoothEntry *entry = &rbpf->smooth_history[rbpf->smooth_head];
        entry->vol_mean = out->vol_mean;
        entry->log_vol_mean = out->log_vol_mean;
        entry->log_vol_var = out->log_vol_var;
        entry->dominant_regime = out->dominant_regime;
        entry->ess = out->ess;
        entry->valid = 1;

        for (int r = 0; r < n_regimes; r++)
        {
            entry->regime_probs[r] = out->regime_probs[r];
        }

        /* Advance head (circular buffer) */
        rbpf->smooth_head = (rbpf->smooth_head + 1) % RBPF_MAX_SMOOTH_LAG;
        if (rbpf->smooth_count < lag)
        {
            rbpf->smooth_count++;
        }

        /* Output smooth signal if we have enough history */
        if (rbpf->smooth_count >= lag)
        {
            int smooth_idx = (rbpf->smooth_head - lag + RBPF_MAX_SMOOTH_LAG) % RBPF_MAX_SMOOTH_LAG;
            const RBPF_SmoothEntry *smooth_entry = &rbpf->smooth_history[smooth_idx];

            out->smooth_valid = 1;
            out->smooth_lag = lag;
            out->vol_mean_smooth = smooth_entry->vol_mean;
            out->log_vol_mean_smooth = smooth_entry->log_vol_mean;
            out->log_vol_var_smooth = smooth_entry->log_vol_var;
            out->dominant_regime_smooth = smooth_entry->dominant_regime;

            for (int r = 0; r < n_regimes; r++)
            {
                out->regime_probs_smooth[r] = smooth_entry->regime_probs[r];
            }

            rbpf_real_t max_smooth_prob = out->regime_probs_smooth[0];
            for (int r = 1; r < n_regimes; r++)
            {
                if (out->regime_probs_smooth[r] > max_smooth_prob)
                {
                    max_smooth_prob = out->regime_probs_smooth[r];
                }
            }
            out->regime_confidence = max_smooth_prob;
        }
        else
        {
            /* Not enough history yet - output fast signal as fallback */
            out->smooth_valid = 0;
            out->smooth_lag = lag;
            out->vol_mean_smooth = out->vol_mean;
            out->log_vol_mean_smooth = out->log_vol_mean;
            out->log_vol_var_smooth = out->log_vol_var;
            out->dominant_regime_smooth = out->dominant_regime;
            out->regime_confidence = max_prob;

            for (int r = 0; r < n_regimes; r++)
            {
                out->regime_probs_smooth[r] = out->regime_probs[r];
            }
        }
    }
    else
    {
        /* Fixed-lag smoothing disabled - smooth = fast */
        out->smooth_valid = 1;
        out->smooth_lag = 0;
        out->vol_mean_smooth = out->vol_mean;
        out->log_vol_mean_smooth = out->log_vol_mean;
        out->log_vol_var_smooth = out->log_vol_var;
        out->dominant_regime_smooth = out->dominant_regime;
        out->regime_confidence = max_prob;

        for (int r = 0; r < n_regimes; r++)
        {
            out->regime_probs_smooth[r] = out->regime_probs[r];
        }
    }

    /*========================================================================
     * STUDENT-T OUTPUT DIAGNOSTICS
     *======================================================================*/

    out->student_t_active = 0;
    out->lambda_mean = RBPF_REAL(1.0);
    out->lambda_var = RBPF_REAL(0.0);
    out->nu_effective = RBPF_NU_CEIL;

#if RBPF_ENABLE_STUDENT_T
    if (rbpf->student_t_enabled && rbpf->lambda != NULL)
    {
        out->student_t_active = 1;

        /* Compute λ statistics */
        rbpf_real_t sum_lam = RBPF_REAL(0.0);
        rbpf_real_t sum_lam_sq = RBPF_REAL(0.0);

        for (int i = 0; i < n; i++)
        {
            sum_lam += w_norm[i] * rbpf->lambda[i];
            sum_lam_sq += w_norm[i] * rbpf->lambda[i] * rbpf->lambda[i];
        }

        out->lambda_mean = sum_lam;
        out->lambda_var = sum_lam_sq - sum_lam * sum_lam;
        if (out->lambda_var < RBPF_REAL(0.0))
            out->lambda_var = RBPF_REAL(0.0);

        /* Implied ν from observed λ variance: Var[λ] = 2/ν → ν = 2/Var[λ] */
        if (out->lambda_var > RBPF_REAL(0.01))
        {
            out->nu_effective = RBPF_REAL(2.0) / out->lambda_var;
        }
        else
        {
            out->nu_effective = RBPF_NU_CEIL;
        }

        /* Copy learned ν estimates */
        for (int r = 0; r < n_regimes; r++)
        {
            if (rbpf->student_t[r].learn_nu)
            {
                out->learned_nu[r] = rbpf->student_t_stats[r].nu_estimate;
            }
            else
            {
                out->learned_nu[r] = rbpf->student_t[r].nu;
            }
        }
    }
#endif
}