/**
 * @file rbpf_ext_hawkes.c
 * @brief Hawkes Self-Excitation, Robust OCSN, and Asset Presets
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * This file contains:
 *   - Hawkes intensity updates and transition modification
 *   - Robust OCSN (11th mixture component) for outlier handling
 *   - Asset class presets (Equity, FX, Crypto, etc.)
 *
 * Related files:
 *   - rbpf_ksc_param_integration.c  Core lifecycle + step function
 *   - rbpf_ext_smoothed_storvik.c   PARIS fixed-lag smoother
 * ═══════════════════════════════════════════════════════════════════════════
 */

#include "rbpf_ksc_param_integration.h"
#include <string.h>
#include <stdio.h>

/*═══════════════════════════════════════════════════════════════════════════
 * HAWKES INTERNAL HELPERS
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Update Hawkes intensity based on observation
 *
 * λ(t) = μ + (λ(t-1) - μ) × e^(-β_eff) + α × I(|r| > threshold)
 *
 * Call AFTER processing the observation.
 * The updated intensity affects NEXT tick's transitions.
 *
 * ADAPTIVE DECAY:
 *   β_eff = β_base × beta_regime_scale[current_regime]
 *   - R0 (calm): High scale → fast decay → short memory
 *   - R3 (crisis): Low scale → slow decay → long memory
 */
void rbpf_ext_hawkes_update_intensity(RBPF_Extended *ext, rbpf_real_t obs)
{
    if (!ext->hawkes.enabled)
        return;

    RBPF_HawkesState *h = &ext->hawkes;

    /* Store previous for hysteresis detection */
    h->intensity_prev = h->intensity;

    /* Get effective decay rate (adaptive or fixed) */
    rbpf_real_t beta_eff = h->beta;

    if (h->adaptive_beta_enabled && ext->rbpf)
    {
        /* Use dominant regime from particle distribution */
        int regime_counts[RBPF_MAX_REGIMES] = {0};
        const int n = ext->rbpf->n_particles;
        for (int i = 0; i < n; i++)
        {
            int r = ext->rbpf->regime[i];
            if (r >= 0 && r < ext->rbpf->n_regimes)
            {
                regime_counts[r]++;
            }
        }
        int dominant_regime = 0;
        int max_count = 0;
        for (int r = 0; r < ext->rbpf->n_regimes; r++)
        {
            if (regime_counts[r] > max_count)
            {
                max_count = regime_counts[r];
                dominant_regime = r;
            }
        }

        /* Scale beta by regime */
        beta_eff = h->beta * h->beta_regime_scale[dominant_regime];
    }

    /* Exponential decay toward baseline */
    rbpf_real_t decay = rbpf_exp(-beta_eff);
    h->intensity = h->mu + (h->intensity - h->mu) * decay;

    /* Excitation: jump when |return| exceeds threshold */
    rbpf_real_t abs_return = rbpf_fabs(obs);

    if (abs_return > h->threshold)
    {
        /* Scale jump by return magnitude (capped at 3x base alpha) */
        rbpf_real_t magnitude_scale = abs_return / h->threshold;
        if (magnitude_scale > RBPF_REAL(3.0))
        {
            magnitude_scale = RBPF_REAL(3.0);
        }

        h->intensity += h->alpha * magnitude_scale;
    }

    ext->last_hawkes_intensity = h->intensity;
}

/**
 * Apply Hawkes intensity to transition matrix
 *
 * High intensity → boost probability of upward regime transitions
 *
 * Call BEFORE regime transition step, using intensity from previous tick.
 */
void rbpf_ext_hawkes_apply_to_transitions(RBPF_Extended *ext)
{
    if (!ext->hawkes.enabled)
        return;

    RBPF_HawkesState *h = &ext->hawkes;
    const int n = ext->rbpf->n_regimes;

    /* Only modify if intensity significantly above baseline */
    rbpf_real_t excess = h->intensity - h->mu;
    if (excess < h->mu * RBPF_REAL(0.1))
    {
        /* Intensity near baseline - restore original LUT if dirty */
        if (h->lut_dirty)
        {
            rbpf_ksc_build_transition_lut(ext->rbpf, ext->base_trans_matrix);
            h->lut_dirty = 0;
        }
        return;
    }

    /* Copy base matrix */
    rbpf_real_t mod_matrix[RBPF_MAX_REGIMES * RBPF_MAX_REGIMES];
    memcpy(mod_matrix, ext->base_trans_matrix, n * n * sizeof(rbpf_real_t));

    /* Compute boost amount */
    rbpf_real_t boost = excess * h->boost_scale;
    if (boost > h->boost_cap)
        boost = h->boost_cap;

    /* Minimum probability to leave in any cell */
    const rbpf_real_t MIN_PROB = RBPF_REAL(0.02);

    /* Modify transitions: boost probability of moving UP */
    for (int from = 0; from < n - 1; from++)
    {
        rbpf_real_t *row = &mod_matrix[from * n];

        /* Calculate how much we CAN steal (respecting MIN_PROB floor) */
        rbpf_real_t to_redistribute = RBPF_REAL(0.0);

        for (int to = 0; to <= from; to++)
        {
            rbpf_real_t available = row[to] - MIN_PROB;
            if (available <= RBPF_REAL(0.0))
                continue;

            rbpf_real_t steal = row[to] * boost;
            if (steal > available)
                steal = available;

            row[to] -= steal;
            to_redistribute += steal;
        }

        /* Distribute to higher regimes (weighted toward crisis) */
        if (to_redistribute > RBPF_REAL(1e-6))
        {
            rbpf_real_t total_weight = RBPF_REAL(0.0);
            for (int to = from + 1; to < n; to++)
            {
                total_weight += (rbpf_real_t)(to - from);
            }

            if (total_weight > RBPF_REAL(0.0))
            {
                for (int to = from + 1; to < n; to++)
                {
                    rbpf_real_t weight = (rbpf_real_t)(to - from) / total_weight;
                    row[to] += to_redistribute * weight;
                }
            }
        }
    }

    /* Rebuild LUT */
    rbpf_ksc_build_transition_lut(ext->rbpf, mod_matrix);
    h->lut_dirty = 1;
}

/**
 * Restore base transitions (call at end of tick if needed)
 */
void rbpf_ext_hawkes_restore_base_transitions(RBPF_Extended *ext)
{
    if (!ext->hawkes.enabled)
        return;
    if (!ext->hawkes.lut_dirty)
        return;

    rbpf_ksc_build_transition_lut(ext->rbpf, ext->base_trans_matrix);
    ext->hawkes.lut_dirty = 0;
}

/*═══════════════════════════════════════════════════════════════════════════
 * HAWKES API
 *═══════════════════════════════════════════════════════════════════════════*/

void rbpf_ext_enable_hawkes(RBPF_Extended *ext,
                            rbpf_real_t mu, rbpf_real_t alpha,
                            rbpf_real_t beta, rbpf_real_t threshold)
{
    if (!ext)
        return;

    ext->hawkes.enabled = 1;
    ext->hawkes.mu = mu;
    ext->hawkes.alpha = alpha;
    ext->hawkes.beta = beta;
    ext->hawkes.threshold = threshold;
    ext->hawkes.intensity = mu;
    ext->hawkes.intensity_prev = mu;
    ext->hawkes.lut_dirty = 0;

    /* Default boost parameters */
    ext->hawkes.boost_scale = RBPF_REAL(0.1);
    ext->hawkes.boost_cap = RBPF_REAL(0.25);

    /* Enable adaptive beta with sensible regime scales */
    ext->hawkes.adaptive_beta_enabled = 1;
    ext->hawkes.beta_regime_scale[0] = RBPF_REAL(2.0); /* R0: Fast decay */
    ext->hawkes.beta_regime_scale[1] = RBPF_REAL(1.5);
    ext->hawkes.beta_regime_scale[2] = RBPF_REAL(1.0);
    ext->hawkes.beta_regime_scale[3] = RBPF_REAL(0.5); /* R3: Slow decay */
    for (int r = 4; r < RBPF_MAX_REGIMES; r++)
    {
        ext->hawkes.beta_regime_scale[r] = RBPF_REAL(0.5);
    }
}

void rbpf_ext_disable_hawkes(RBPF_Extended *ext)
{
    if (!ext)
        return;

    if (ext->hawkes.lut_dirty)
    {
        rbpf_ksc_build_transition_lut(ext->rbpf, ext->base_trans_matrix);
    }

    ext->hawkes.enabled = 0;
    ext->hawkes.lut_dirty = 0;
}

void rbpf_ext_set_hawkes_boost(RBPF_Extended *ext,
                               rbpf_real_t boost_scale, rbpf_real_t boost_cap)
{
    if (!ext)
        return;
    ext->hawkes.boost_scale = boost_scale;
    ext->hawkes.boost_cap = boost_cap;
}

void rbpf_ext_enable_adaptive_hawkes(RBPF_Extended *ext, int enable)
{
    if (!ext)
        return;
    ext->hawkes.adaptive_beta_enabled = enable;
}

void rbpf_ext_set_hawkes_regime_scale(RBPF_Extended *ext, int regime, rbpf_real_t scale)
{
    if (!ext)
        return;
    if (regime < 0 || regime >= RBPF_MAX_REGIMES)
        return;

    /* Clamp scale to reasonable bounds */
    if (scale < RBPF_REAL(0.1))
        scale = RBPF_REAL(0.1);
    if (scale > RBPF_REAL(5.0))
        scale = RBPF_REAL(5.0);

    ext->hawkes.beta_regime_scale[regime] = scale;
}

rbpf_real_t rbpf_ext_get_hawkes_intensity(const RBPF_Extended *ext)
{
    if (!ext)
        return RBPF_REAL(0.0);
    return ext->last_hawkes_intensity;
}

/*═══════════════════════════════════════════════════════════════════════════
 * ROBUST OCSN API
 *═══════════════════════════════════════════════════════════════════════════*/

void rbpf_ext_enable_robust_ocsn(RBPF_Extended *ext)
{
    if (!ext)
        return;

    ext->robust_ocsn.enabled = 1;

    /* Regime-scaled parameters (3× max OCSN variance = 22 base) */
    ext->robust_ocsn.regime[0].prob = RBPF_REAL(0.010);
    ext->robust_ocsn.regime[0].variance = RBPF_REAL(18.0);

    ext->robust_ocsn.regime[1].prob = RBPF_REAL(0.015);
    ext->robust_ocsn.regime[1].variance = RBPF_REAL(20.0);

    ext->robust_ocsn.regime[2].prob = RBPF_REAL(0.020);
    ext->robust_ocsn.regime[2].variance = RBPF_REAL(24.0);

    ext->robust_ocsn.regime[3].prob = RBPF_REAL(0.025);
    ext->robust_ocsn.regime[3].variance = RBPF_REAL(30.0);

    for (int r = 4; r < RBPF_MAX_REGIMES; r++)
    {
        ext->robust_ocsn.regime[r] = ext->robust_ocsn.regime[3];
    }
}

void rbpf_ext_enable_robust_ocsn_simple(RBPF_Extended *ext,
                                        rbpf_real_t prob, rbpf_real_t variance)
{
    if (!ext)
        return;

    ext->robust_ocsn.enabled = 1;

    for (int r = 0; r < RBPF_MAX_REGIMES; r++)
    {
        ext->robust_ocsn.regime[r].prob = prob;
        ext->robust_ocsn.regime[r].variance = variance;
    }
}

void rbpf_ext_set_outlier_params(RBPF_Extended *ext, int regime,
                                 rbpf_real_t prob, rbpf_real_t variance)
{
    if (!ext)
        return;
    if (regime < 0 || regime >= RBPF_MAX_REGIMES)
        return;

    ext->robust_ocsn.regime[regime].prob = prob;
    ext->robust_ocsn.regime[regime].variance = variance;
}

void rbpf_ext_disable_robust_ocsn(RBPF_Extended *ext)
{
    if (!ext)
        return;
    ext->robust_ocsn.enabled = 0;
}

rbpf_real_t rbpf_ext_get_outlier_fraction(const RBPF_Extended *ext)
{
    if (!ext)
        return RBPF_REAL(0.0);
    return ext->last_outlier_fraction;
}

/*═══════════════════════════════════════════════════════════════════════════
 * ASSET PRESETS
 *═══════════════════════════════════════════════════════════════════════════*/

RBPF_AssetPreset rbpf_ext_get_preset(const RBPF_Extended *ext)
{
    if (!ext)
        return RBPF_PRESET_CUSTOM;
    return ext->current_preset;
}

void rbpf_ext_apply_preset(RBPF_Extended *ext, RBPF_AssetPreset preset)
{
    if (!ext)
        return;

    ext->current_preset = preset;

    switch (preset)
    {

    /*───────────────────────────────────────────────────────────────────────
     * EQUITY INDEX (SPY, QQQ, ES)
     *─────────────────────────────────────────────────────────────────────*/
    case RBPF_PRESET_EQUITY_INDEX:
        rbpf_ext_enable_hawkes(ext,
                               RBPF_REAL(0.05),   /* mu */
                               RBPF_REAL(0.30),   /* alpha */
                               RBPF_REAL(0.10),   /* beta */
                               RBPF_REAL(0.025)); /* threshold: 2.5% */

        ext->hawkes.beta_regime_scale[0] = RBPF_REAL(2.0);
        ext->hawkes.beta_regime_scale[1] = RBPF_REAL(1.5);
        ext->hawkes.beta_regime_scale[2] = RBPF_REAL(1.0);
        ext->hawkes.beta_regime_scale[3] = RBPF_REAL(0.5);

        rbpf_ext_enable_robust_ocsn(ext);
        break;

    /*───────────────────────────────────────────────────────────────────────
     * SINGLE STOCK (AAPL, TSLA, NVDA)
     *─────────────────────────────────────────────────────────────────────*/
    case RBPF_PRESET_SINGLE_STOCK:
        rbpf_ext_enable_hawkes(ext,
                               RBPF_REAL(0.08),
                               RBPF_REAL(0.40),
                               RBPF_REAL(0.12),
                               RBPF_REAL(0.03));

        ext->hawkes.beta_regime_scale[0] = RBPF_REAL(2.5);
        ext->hawkes.beta_regime_scale[1] = RBPF_REAL(1.8);
        ext->hawkes.beta_regime_scale[2] = RBPF_REAL(1.2);
        ext->hawkes.beta_regime_scale[3] = RBPF_REAL(0.6);

        ext->robust_ocsn.enabled = 1;
        ext->robust_ocsn.regime[0].prob = RBPF_REAL(0.015);
        ext->robust_ocsn.regime[0].variance = RBPF_REAL(22.0);
        ext->robust_ocsn.regime[1].prob = RBPF_REAL(0.020);
        ext->robust_ocsn.regime[1].variance = RBPF_REAL(25.0);
        ext->robust_ocsn.regime[2].prob = RBPF_REAL(0.025);
        ext->robust_ocsn.regime[2].variance = RBPF_REAL(30.0);
        ext->robust_ocsn.regime[3].prob = RBPF_REAL(0.035);
        ext->robust_ocsn.regime[3].variance = RBPF_REAL(38.0);
        break;

    /*───────────────────────────────────────────────────────────────────────
     * FX G10 (EUR/USD, USD/JPY, GBP/USD)
     *─────────────────────────────────────────────────────────────────────*/
    case RBPF_PRESET_FX_G10:
        rbpf_ext_enable_hawkes(ext,
                               RBPF_REAL(0.03),
                               RBPF_REAL(0.20),
                               RBPF_REAL(0.15),
                               RBPF_REAL(0.015));

        ext->hawkes.beta_regime_scale[0] = RBPF_REAL(2.5);
        ext->hawkes.beta_regime_scale[1] = RBPF_REAL(2.0);
        ext->hawkes.beta_regime_scale[2] = RBPF_REAL(1.5);
        ext->hawkes.beta_regime_scale[3] = RBPF_REAL(0.8);

        ext->robust_ocsn.enabled = 1;
        ext->robust_ocsn.regime[0].prob = RBPF_REAL(0.008);
        ext->robust_ocsn.regime[0].variance = RBPF_REAL(15.0);
        ext->robust_ocsn.regime[1].prob = RBPF_REAL(0.010);
        ext->robust_ocsn.regime[1].variance = RBPF_REAL(18.0);
        ext->robust_ocsn.regime[2].prob = RBPF_REAL(0.015);
        ext->robust_ocsn.regime[2].variance = RBPF_REAL(22.0);
        ext->robust_ocsn.regime[3].prob = RBPF_REAL(0.020);
        ext->robust_ocsn.regime[3].variance = RBPF_REAL(28.0);
        break;

    /*───────────────────────────────────────────────────────────────────────
     * FX EM (USD/MXN, USD/TRY, USD/ZAR)
     *─────────────────────────────────────────────────────────────────────*/
    case RBPF_PRESET_FX_EM:
        rbpf_ext_enable_hawkes(ext,
                               RBPF_REAL(0.08),
                               RBPF_REAL(0.45),
                               RBPF_REAL(0.08),
                               RBPF_REAL(0.025));

        ext->hawkes.beta_regime_scale[0] = RBPF_REAL(1.8);
        ext->hawkes.beta_regime_scale[1] = RBPF_REAL(1.2);
        ext->hawkes.beta_regime_scale[2] = RBPF_REAL(0.8);
        ext->hawkes.beta_regime_scale[3] = RBPF_REAL(0.4);

        ext->robust_ocsn.enabled = 1;
        ext->robust_ocsn.regime[0].prob = RBPF_REAL(0.015);
        ext->robust_ocsn.regime[0].variance = RBPF_REAL(22.0);
        ext->robust_ocsn.regime[1].prob = RBPF_REAL(0.025);
        ext->robust_ocsn.regime[1].variance = RBPF_REAL(28.0);
        ext->robust_ocsn.regime[2].prob = RBPF_REAL(0.035);
        ext->robust_ocsn.regime[2].variance = RBPF_REAL(35.0);
        ext->robust_ocsn.regime[3].prob = RBPF_REAL(0.045);
        ext->robust_ocsn.regime[3].variance = RBPF_REAL(45.0);
        break;

    /*───────────────────────────────────────────────────────────────────────
     * CRYPTO (BTC, ETH)
     *─────────────────────────────────────────────────────────────────────*/
    case RBPF_PRESET_CRYPTO:
        rbpf_ext_enable_hawkes(ext,
                               RBPF_REAL(0.10),
                               RBPF_REAL(0.50),
                               RBPF_REAL(0.06),
                               RBPF_REAL(0.05));

        ext->hawkes.beta_regime_scale[0] = RBPF_REAL(1.5);
        ext->hawkes.beta_regime_scale[1] = RBPF_REAL(1.0);
        ext->hawkes.beta_regime_scale[2] = RBPF_REAL(0.7);
        ext->hawkes.beta_regime_scale[3] = RBPF_REAL(0.3);

        ext->robust_ocsn.enabled = 1;
        ext->robust_ocsn.regime[0].prob = RBPF_REAL(0.025);
        ext->robust_ocsn.regime[0].variance = RBPF_REAL(30.0);
        ext->robust_ocsn.regime[1].prob = RBPF_REAL(0.035);
        ext->robust_ocsn.regime[1].variance = RBPF_REAL(38.0);
        ext->robust_ocsn.regime[2].prob = RBPF_REAL(0.045);
        ext->robust_ocsn.regime[2].variance = RBPF_REAL(45.0);
        ext->robust_ocsn.regime[3].prob = RBPF_REAL(0.060);
        ext->robust_ocsn.regime[3].variance = RBPF_REAL(50.0);
        break;

    /*───────────────────────────────────────────────────────────────────────
     * COMMODITIES (CL, GC, NG)
     *─────────────────────────────────────────────────────────────────────*/
    case RBPF_PRESET_COMMODITIES:
        rbpf_ext_enable_hawkes(ext,
                               RBPF_REAL(0.06),
                               RBPF_REAL(0.35),
                               RBPF_REAL(0.10),
                               RBPF_REAL(0.03));

        ext->hawkes.beta_regime_scale[0] = RBPF_REAL(2.0);
        ext->hawkes.beta_regime_scale[1] = RBPF_REAL(1.4);
        ext->hawkes.beta_regime_scale[2] = RBPF_REAL(0.9);
        ext->hawkes.beta_regime_scale[3] = RBPF_REAL(0.5);

        ext->robust_ocsn.enabled = 1;
        ext->robust_ocsn.regime[0].prob = RBPF_REAL(0.012);
        ext->robust_ocsn.regime[0].variance = RBPF_REAL(20.0);
        ext->robust_ocsn.regime[1].prob = RBPF_REAL(0.018);
        ext->robust_ocsn.regime[1].variance = RBPF_REAL(25.0);
        ext->robust_ocsn.regime[2].prob = RBPF_REAL(0.025);
        ext->robust_ocsn.regime[2].variance = RBPF_REAL(32.0);
        ext->robust_ocsn.regime[3].prob = RBPF_REAL(0.035);
        ext->robust_ocsn.regime[3].variance = RBPF_REAL(40.0);
        break;

    /*───────────────────────────────────────────────────────────────────────
     * BONDS (ZN, ZB, TLT)
     *─────────────────────────────────────────────────────────────────────*/
    case RBPF_PRESET_BONDS:
        rbpf_ext_enable_hawkes(ext,
                               RBPF_REAL(0.02),
                               RBPF_REAL(0.25),
                               RBPF_REAL(0.18),
                               RBPF_REAL(0.01));

        ext->hawkes.beta_regime_scale[0] = RBPF_REAL(3.0);
        ext->hawkes.beta_regime_scale[1] = RBPF_REAL(2.2);
        ext->hawkes.beta_regime_scale[2] = RBPF_REAL(1.5);
        ext->hawkes.beta_regime_scale[3] = RBPF_REAL(0.8);

        ext->robust_ocsn.enabled = 1;
        ext->robust_ocsn.regime[0].prob = RBPF_REAL(0.005);
        ext->robust_ocsn.regime[0].variance = RBPF_REAL(14.0);
        ext->robust_ocsn.regime[1].prob = RBPF_REAL(0.008);
        ext->robust_ocsn.regime[1].variance = RBPF_REAL(18.0);
        ext->robust_ocsn.regime[2].prob = RBPF_REAL(0.015);
        ext->robust_ocsn.regime[2].variance = RBPF_REAL(24.0);
        ext->robust_ocsn.regime[3].prob = RBPF_REAL(0.025);
        ext->robust_ocsn.regime[3].variance = RBPF_REAL(35.0);
        break;

    case RBPF_PRESET_CUSTOM:
    default:
        /* Leave current settings unchanged */
        break;
    }

    /* Copy remaining regimes from R3 */
    for (int r = 4; r < RBPF_MAX_REGIMES; r++)
    {
        ext->hawkes.beta_regime_scale[r] = ext->hawkes.beta_regime_scale[3];
        ext->robust_ocsn.regime[r] = ext->robust_ocsn.regime[3];
    }
}