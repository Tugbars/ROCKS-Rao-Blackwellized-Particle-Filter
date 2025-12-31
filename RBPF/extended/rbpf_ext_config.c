/**
 * @file rbpf_ext_config.c
 * @brief Configuration and Enable Functions
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * This file contains all setters and enablers:
 *   - rbpf_ext_set_*
 *   - rbpf_ext_enable_*
 *   - rbpf_ext_disable_*
 *   - rbpf_ext_configure_*
 *   - rbpf_ext_signal_*
 *   - rbpf_ext_build_*
 *   - rbpf_ext_reset_*
 *
 * Related files:
 *   - rbpf_ksc_param_integration.c  Core lifecycle + step
 *   - rbpf_ext_diagnostics.c        Getters and print functions
 * ═══════════════════════════════════════════════════════════════════════════
 */

#include "rbpf_ksc_param_integration.h"
#include "rbpf_kl_tempering.h"
#include "hawkes_integrator.h"
#include <string.h>
#include <stdlib.h>

/*═══════════════════════════════════════════════════════════════════════════
 * BASIC CONFIGURATION
 *═══════════════════════════════════════════════════════════════════════════*/

void rbpf_ext_set_regime_params(RBPF_Extended *ext, int regime,
                                rbpf_real_t theta, rbpf_real_t mu_vol,
                                rbpf_real_t sigma_vol)
{
    if (!ext || regime < 0 || regime >= RBPF_MAX_REGIMES)
        return;

    rbpf_ksc_set_regime_params(ext->rbpf, regime, theta, mu_vol, sigma_vol);

    if (ext->storvik_initialized)
    {
        rbpf_real_t phi = RBPF_REAL(1.0) - theta;
        param_learn_set_prior(&ext->storvik, regime, mu_vol, phi, sigma_vol);
    }
}

void rbpf_ext_build_transition_lut(RBPF_Extended *ext, const rbpf_real_t *trans_matrix)
{
    if (!ext)
        return;

    const int n = ext->rbpf->n_regimes;
    memcpy(ext->base_trans_matrix, trans_matrix, n * n * sizeof(rbpf_real_t));
    rbpf_ksc_build_transition_lut(ext->rbpf, trans_matrix);
}

void rbpf_ext_set_storvik_interval(RBPF_Extended *ext, int regime, int interval)
{
    if (!ext || !ext->storvik_initialized)
        return;
    if (regime < 0 || regime >= PARAM_LEARN_MAX_REGIMES)
        return;
    ext->storvik.config.sample_interval[regime] = interval;
}

void rbpf_ext_set_hft_mode(RBPF_Extended *ext, int enable)
{
    if (!ext || !ext->storvik_initialized)
        return;

    if (enable)
    {
        ext->storvik.config.sample_interval[0] = 100;
        ext->storvik.config.sample_interval[1] = 50;
        ext->storvik.config.sample_interval[2] = 20;
        ext->storvik.config.sample_interval[3] = 5;
    }
    else
    {
        for (int r = 0; r < PARAM_LEARN_MAX_REGIMES; r++)
        {
            ext->storvik.config.sample_interval[r] = 1;
        }
    }
}

void rbpf_ext_set_full_update_mode(RBPF_Extended *ext)
{
    if (!ext || !ext->storvik_initialized)
        return;

    for (int r = 0; r < PARAM_LEARN_MAX_REGIMES; r++)
    {
        ext->storvik.config.sample_interval[r] = 1;
    }
    ext->storvik.config.enable_global_tick_skip = false;
    ext->storvik.config.enable_forgetting = true;
    ext->storvik.config.forgetting_lambda = 0.997;
}

void rbpf_ext_signal_structural_break(RBPF_Extended *ext)
{
    if (!ext)
        return;
    ext->structural_break_signaled = 1;
    if (ext->storvik_initialized)
    {
        param_learn_signal_structural_break(&ext->storvik);
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * FORGETTING FACTOR DERIVATION FROM TRANSITION MATRIX
 *═══════════════════════════════════════════════════════════════════════════*/

void rbpf_ext_compute_forgetting_from_transitions(RBPF_Extended *ext, float alpha)
{
    if (!ext || !ext->rbpf || !ext->storvik_initialized)
        return;
    if (alpha <= 0.0f)
        alpha = 3.0f;

    const int nr = ext->rbpf->n_regimes;

    for (int r = 0; r < nr; r++)
    {
        float p_stay = ext->base_trans_matrix[r * nr + r];

        if (p_stay < 0.5f)
            p_stay = 0.5f;
        if (p_stay > 0.999f)
            p_stay = 0.999f;

        float expected_dwell = 1.0f / (1.0f - p_stay);
        float memory_ticks = alpha * expected_dwell;
        float lambda = 1.0f - 1.0f / memory_ticks;

        if (lambda < 0.95f)
            lambda = 0.95f;
        if (lambda > 0.9999f)
            lambda = 0.9999f;

        param_learn_set_regime_forgetting(&ext->storvik, r, lambda);
    }
}

void rbpf_ext_auto_configure_forgetting(RBPF_Extended *ext)
{
    rbpf_ext_compute_forgetting_from_transitions(ext, 3.0f);
}

/*═══════════════════════════════════════════════════════════════════════════
 * TRANSITION LEARNING
 *═══════════════════════════════════════════════════════════════════════════*/

void rbpf_ext_enable_transition_learning(RBPF_Extended *ext, int enable)
{
    if (!ext)
        return;
    ext->trans_learn_enabled = enable;
    if (enable)
        rbpf_ext_reset_transition_counts(ext);
}

void rbpf_ext_configure_transition_learning(RBPF_Extended *ext,
                                            double forgetting,
                                            double prior_diag,
                                            double prior_off,
                                            int update_interval)
{
    if (!ext)
        return;
    ext->trans_forgetting = forgetting;
    ext->trans_prior_diag = prior_diag;
    ext->trans_prior_off = prior_off;
    ext->trans_update_interval = update_interval;
}

void rbpf_ext_reset_transition_counts(RBPF_Extended *ext)
{
    if (!ext)
        return;
    /* No-op: trans_counts removed, PGAS owns Π */
    ext->trans_ticks_since_update = 0;
}

/*═══════════════════════════════════════════════════════════════════════════
 * KL TEMPERING CONFIGURATION
 *═══════════════════════════════════════════════════════════════════════════*/

void rbpf_ext_enable_kl_tempering(RBPF_Extended *ext)
{
    if (!ext)
        return;

    ext->kl_tempering_enabled = 1;
    rbpf_ksc_set_deferred_weight_mode(ext->rbpf, 1);

    /* Allocate if not already done */
    if (!ext->kl_state)
    {
        ext->kl_state = (RBPF_KL_State *)calloc(1, sizeof(RBPF_KL_State));
    }

    if (ext->kl_state)
    {
        rbpf_kl_init(ext->kl_state, ext->rbpf->n_particles);
    }
}

void rbpf_ext_disable_kl_tempering(RBPF_Extended *ext)
{
    if (!ext)
        return;

    ext->kl_tempering_enabled = 0;
    rbpf_ksc_set_deferred_weight_mode(ext->rbpf, 0);
}

/*═══════════════════════════════════════════════════════════════════════════
 * APF KICK CONFIGURATION
 *═══════════════════════════════════════════════════════════════════════════*/

void rbpf_ext_enable_apf_kick(RBPF_Extended *ext, int enable)
{
    if (!ext)
        return;
    ext->apf_kick_enabled = enable;
}

void rbpf_ext_set_apf_surprise_threshold(RBPF_Extended *ext, float threshold)
{
    if (!ext)
        return;
    ext->apf_surprise_threshold = threshold;
}

void rbpf_ext_configure_hawkes(RBPF_Extended *ext, const HawkesIntegratorConfig *cfg)
{
    if (!ext || !cfg)
        return;
    hawkes_integrator_init(&ext->hawkes_integrator, cfg);
}

void rbpf_ext_configure_hawkes_params(RBPF_Extended *ext,
                                      float mu, float alpha, float beta,
                                      float event_threshold)
{
    if (!ext)
        return;
    hawkes_integrator_set_params(&ext->hawkes_integrator, mu, alpha, beta);
    ext->hawkes_integrator.config.hawkes.event_threshold = event_threshold;
}