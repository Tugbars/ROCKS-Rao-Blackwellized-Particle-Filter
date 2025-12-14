/**
 * @file mmpf_api.c
 * @brief MMPF-ROCKS API Implementation
 *
 * Contains:
 *   - Output accessors (get_volatility, get_weights, etc.)
 *   - IMM control (set_stickiness, set_transition_matrix)
 *   - Student-t control
 *   - BOCPD shock mechanism
 *   - Diagnostics and printing
 */

#include "mmpf_internal.h"

/*═══════════════════════════════════════════════════════════════════════════
 * OUTPUT ACCESSORS
 *═══════════════════════════════════════════════════════════════════════════*/

rbpf_real_t mmpf_get_volatility(const MMPF_ROCKS *mmpf)
{
    return mmpf->weighted_vol;
}

rbpf_real_t mmpf_get_log_volatility(const MMPF_ROCKS *mmpf)
{
    return mmpf->weighted_log_vol;
}

rbpf_real_t mmpf_get_volatility_std(const MMPF_ROCKS *mmpf)
{
    return mmpf->weighted_vol_std;
}

MMPF_Hypothesis mmpf_get_dominant(const MMPF_ROCKS *mmpf)
{
    return mmpf->dominant;
}

rbpf_real_t mmpf_get_dominant_probability(const MMPF_ROCKS *mmpf)
{
    return mmpf->weights[mmpf->dominant];
}

void mmpf_get_weights(const MMPF_ROCKS *mmpf, rbpf_real_t *weights)
{
    int k;
    for (k = 0; k < MMPF_N_MODELS; k++)
    {
        weights[k] = mmpf->weights[k];
    }
}

rbpf_real_t mmpf_get_outlier_fraction(const MMPF_ROCKS *mmpf)
{
    return mmpf->outlier_fraction;
}

rbpf_real_t mmpf_get_stickiness(const MMPF_ROCKS *mmpf)
{
    return mmpf->current_stickiness;
}

rbpf_real_t mmpf_get_model_volatility(const MMPF_ROCKS *mmpf, MMPF_Hypothesis model)
{
    return mmpf->model_output[model].vol_mean;
}

rbpf_real_t mmpf_get_model_ess(const MMPF_ROCKS *mmpf, MMPF_Hypothesis model)
{
    return mmpf->model_output[model].ess;
}

const RBPF_Extended *mmpf_get_ext(const MMPF_ROCKS *mmpf, MMPF_Hypothesis model)
{
    return mmpf->ext[model];
}

rbpf_real_t mmpf_get_model_outlier_fraction(const MMPF_ROCKS *mmpf, MMPF_Hypothesis model)
{
    return mmpf->model_output[model].outlier_fraction;
}

rbpf_real_t mmpf_get_global_baseline(const MMPF_ROCKS *mmpf)
{
    return mmpf->global_mu_vol;
}

int mmpf_is_baseline_frozen(const MMPF_ROCKS *mmpf)
{
    return (mmpf->baseline_frozen_ticks > 0) ? 1 : 0;
}

int mmpf_get_baseline_frozen_ticks(const MMPF_ROCKS *mmpf)
{
    return mmpf->baseline_frozen_ticks;
}

rbpf_real_t mmpf_get_model_nu(const MMPF_ROCKS *mmpf, MMPF_Hypothesis model)
{
    return mmpf->learned_nu[model];
}

/*═══════════════════════════════════════════════════════════════════════════
 * IMM CONTROL
 *═══════════════════════════════════════════════════════════════════════════*/

void mmpf_set_transition_matrix(MMPF_ROCKS *mmpf, const rbpf_real_t *transition)
{
    int i, j;
    for (i = 0; i < MMPF_N_MODELS; i++)
    {
        for (j = 0; j < MMPF_N_MODELS; j++)
        {
            mmpf->transition[i][j] = transition[i * MMPF_N_MODELS + j];
        }
    }
}

void mmpf_set_stickiness(MMPF_ROCKS *mmpf, rbpf_real_t base, rbpf_real_t min_s)
{
    mmpf->config.base_stickiness = base;
    mmpf->config.min_stickiness = min_s;
}

void mmpf_set_adaptive_stickiness(MMPF_ROCKS *mmpf, int enable)
{
    mmpf->config.enable_adaptive_stickiness = enable;
}

void mmpf_set_weights(MMPF_ROCKS *mmpf, const rbpf_real_t *weights)
{
    int k;
    for (k = 0; k < MMPF_N_MODELS; k++)
    {
        mmpf->weights[k] = weights[k];
    }
    mmpf_normalize_weights(mmpf->weights, MMPF_N_MODELS);

    for (k = 0; k < MMPF_N_MODELS; k++)
    {
        mmpf->log_weights[k] = rbpf_log(mmpf->weights[k]);
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * STUDENT-T CONTROL
 *═══════════════════════════════════════════════════════════════════════════*/

void mmpf_enable_student_t(MMPF_ROCKS *mmpf, const rbpf_real_t *nu)
{
    int k, r;

    if (!mmpf)
        return;

    mmpf->config.enable_student_t = 1;

    for (k = 0; k < MMPF_N_MODELS; k++)
    {
        rbpf_real_t nu_k = nu ? nu[k] : mmpf->config.hypothesis_nu[k];

        rbpf_ksc_enable_student_t(mmpf->ext[k]->rbpf, nu_k);

        for (r = 0; r < mmpf->ext[k]->rbpf->n_regimes; r++)
        {
            rbpf_ksc_set_student_t_nu(mmpf->ext[k]->rbpf, r, nu_k);
        }

        mmpf->learned_nu[k] = nu_k;
    }
}

void mmpf_disable_student_t(MMPF_ROCKS *mmpf)
{
    int k;

    if (!mmpf)
        return;

    mmpf->config.enable_student_t = 0;

    for (k = 0; k < MMPF_N_MODELS; k++)
    {
        rbpf_ksc_disable_student_t(mmpf->ext[k]->rbpf);
    }
}

void mmpf_set_hypothesis_nu(MMPF_ROCKS *mmpf, MMPF_Hypothesis model, rbpf_real_t nu)
{
    int r;

    if (!mmpf || model < 0 || model >= MMPF_N_MODELS)
        return;

    mmpf->config.hypothesis_nu[model] = nu;
    mmpf->learned_nu[model] = nu;

    for (r = 0; r < mmpf->ext[model]->rbpf->n_regimes; r++)
    {
        rbpf_ksc_set_student_t_nu(mmpf->ext[model]->rbpf, r, nu);
    }
}

void mmpf_set_nu_learning(MMPF_ROCKS *mmpf, int enable, rbpf_real_t learning_rate)
{
    int k, r;

    if (!mmpf)
        return;

    mmpf->config.enable_nu_learning = enable;
    mmpf->config.nu_learning_rate = learning_rate;

    for (k = 0; k < MMPF_N_MODELS; k++)
    {
        if (enable)
        {
            for (r = 0; r < mmpf->ext[k]->rbpf->n_regimes; r++)
            {
                rbpf_ksc_enable_nu_learning(mmpf->ext[k]->rbpf, r, learning_rate);
            }
        }
        else
        {
            for (r = 0; r < mmpf->ext[k]->rbpf->n_regimes; r++)
            {
                rbpf_ksc_disable_nu_learning(mmpf->ext[k]->rbpf, r);
            }
        }
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * BOCPD SHOCK MECHANISM
 *═══════════════════════════════════════════════════════════════════════════*/

void mmpf_inject_shock(MMPF_ROCKS *mmpf)
{
    mmpf_inject_shock_ex(mmpf, RBPF_REAL(50.0));
}

void mmpf_inject_shock_ex(MMPF_ROCKS *mmpf, rbpf_real_t noise_multiplier)
{
    int i, j, k, r;

    if (!mmpf)
        return;
    if (mmpf->shock_active)
        return;

    /* Save current transition matrix */
    for (i = 0; i < MMPF_N_MODELS; i++)
    {
        for (j = 0; j < MMPF_N_MODELS; j++)
        {
            mmpf->saved_transition[i][j] = mmpf->transition[i][j];
        }
    }

    /* Set uniform transitions */
    const rbpf_real_t uniform = RBPF_REAL(1.0) / MMPF_N_MODELS;
    for (i = 0; i < MMPF_N_MODELS; i++)
    {
        for (j = 0; j < MMPF_N_MODELS; j++)
        {
            mmpf->transition[i][j] = uniform;
        }
    }

    /* Boost process noise */
    mmpf->process_noise_multiplier = noise_multiplier;

    for (k = 0; k < MMPF_N_MODELS; k++)
    {
        RBPF_KSC *rbpf = mmpf->ext[k]->rbpf;
        for (r = 0; r < rbpf->n_regimes; r++)
        {
            rbpf->params[r].sigma_vol *= noise_multiplier;
            rbpf->params[r].q *= (noise_multiplier * noise_multiplier);
        }
    }

    mmpf->shock_active = 1;
}

void mmpf_restore_from_shock(MMPF_ROCKS *mmpf)
{
    int i, j, k, r;

    if (!mmpf)
        return;
    if (!mmpf->shock_active)
        return;

    /* Restore saved transition matrix */
    for (i = 0; i < MMPF_N_MODELS; i++)
    {
        for (j = 0; j < MMPF_N_MODELS; j++)
        {
            mmpf->transition[i][j] = mmpf->saved_transition[i][j];
        }
    }

    /* Restore process noise */
    rbpf_real_t inv_multiplier = RBPF_REAL(1.0) / mmpf->process_noise_multiplier;
    rbpf_real_t inv_multiplier_sq = inv_multiplier * inv_multiplier;

    for (k = 0; k < MMPF_N_MODELS; k++)
    {
        RBPF_KSC *rbpf = mmpf->ext[k]->rbpf;
        for (r = 0; r < rbpf->n_regimes; r++)
        {
            rbpf->params[r].sigma_vol *= inv_multiplier;
            rbpf->params[r].q *= inv_multiplier_sq;
        }
    }

    mmpf->process_noise_multiplier = RBPF_REAL(1.0);
    mmpf->shock_active = 0;
}

int mmpf_is_shock_active(const MMPF_ROCKS *mmpf)
{
    return mmpf ? mmpf->shock_active : 0;
}

/*═══════════════════════════════════════════════════════════════════════════
 * DIAGNOSTICS
 *═══════════════════════════════════════════════════════════════════════════*/

void mmpf_print_summary(const MMPF_ROCKS *mmpf)
{
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("MMPF-ROCKS Summary\n");
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("Particles per model: %d\n", mmpf->n_particles);
    printf("Total steps:         %lu\n", (unsigned long)mmpf->total_steps);
    printf("Regime switches:     %lu\n", (unsigned long)mmpf->regime_switches);
    printf("IMM mix count:       %lu\n", (unsigned long)mmpf->imm_mix_count);
    printf("\n");
    printf("Model Weights:\n");
    printf("  Calm:   %.4f (ν=%.1f)\n", (double)mmpf->weights[MMPF_CALM],
           (double)mmpf->learned_nu[MMPF_CALM]);
    printf("  Trend:  %.4f (ν=%.1f)\n", (double)mmpf->weights[MMPF_TREND],
           (double)mmpf->learned_nu[MMPF_TREND]);
    printf("  Crisis: %.4f (ν=%.1f)\n", (double)mmpf->weights[MMPF_CRISIS],
           (double)mmpf->learned_nu[MMPF_CRISIS]);
    printf("\n");
    printf("Dominant: %s (prob=%.4f)\n",
           mmpf->dominant == MMPF_CALM ? "Calm" : mmpf->dominant == MMPF_TREND ? "Trend"
                                                                               : "Crisis",
           (double)mmpf->weights[mmpf->dominant]);
    printf("Ticks in regime: %d\n", mmpf->ticks_in_regime);
    printf("\n");
    printf("Weighted volatility: %.6f\n", (double)mmpf->weighted_vol);
    printf("Outlier fraction:    %.4f\n", (double)mmpf->outlier_fraction);
    printf("Current stickiness:  %.4f\n", (double)mmpf->current_stickiness);
    printf("\n");
    printf("Student-t:           %s\n", mmpf->config.enable_student_t ? "ENABLED" : "disabled");
    printf("ν learning:          %s\n", mmpf->config.enable_nu_learning ? "ENABLED" : "disabled");
    printf("═══════════════════════════════════════════════════════════════════\n");
}

void mmpf_print_output(const MMPF_Output *output)
{
    printf("MMPF Output:\n");
    printf("  Volatility:      %.6f (std=%.6f)\n",
           (double)output->volatility, (double)output->volatility_std);
    printf("  Log-volatility:  %.6f\n", (double)output->log_volatility);
    printf("  Weights:         [%.4f, %.4f, %.4f]\n",
           (double)output->weights[0], (double)output->weights[1], (double)output->weights[2]);
    printf("  Dominant:        %d (prob=%.4f)\n",
           output->dominant, (double)output->dominant_prob);
    printf("  Outlier frac:    %.4f\n", (double)output->outlier_fraction);
    printf("  Stickiness:      %.4f\n", (double)output->current_stickiness);
    printf("  Regime stable:   %d (ticks=%d)\n",
           output->regime_stable, output->ticks_in_regime);
    if (output->student_t_active)
    {
        printf("  Student-t ν:     [%.1f, %.1f, %.1f]\n",
               (double)output->model_nu[0], (double)output->model_nu[1],
               (double)output->model_nu[2]);
    }
}

void mmpf_get_diagnostics(const MMPF_ROCKS *mmpf,
                          uint64_t *total_steps,
                          uint64_t *regime_switches,
                          uint64_t *imm_mix_count)
{
    if (total_steps)
        *total_steps = mmpf->total_steps;
    if (regime_switches)
        *regime_switches = mmpf->regime_switches;
    if (imm_mix_count)
        *imm_mix_count = mmpf->imm_mix_count;
}
