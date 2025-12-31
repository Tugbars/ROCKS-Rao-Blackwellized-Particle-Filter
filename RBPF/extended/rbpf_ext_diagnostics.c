/**
 * @file rbpf_ext_diagnostics.c
 * @brief Getters and Print Functions
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * This file contains all query and diagnostic functions:
 *   - rbpf_ext_get_*
 *   - rbpf_ext_*_enabled (query functions)
 *   - rbpf_ext_print_*
 *
 * Related files:
 *   - rbpf_ksc_param_integration.c  Core lifecycle + step
 *   - rbpf_ext_config.c             Configuration functions
 * ═══════════════════════════════════════════════════════════════════════════
 */

#include "rbpf_ksc_param_integration.h"
#include "rbpf_kl_tempering.h"
#include "hawkes_integrator.h"
#include <stdio.h>
#include <string.h>
#include <inttypes.h>

/*═══════════════════════════════════════════════════════════════════════════
 * TRANSITION PROBABILITY QUERY
 *═══════════════════════════════════════════════════════════════════════════*/

double rbpf_ext_get_transition_prob(const RBPF_Extended *ext, int from, int to)
{
    if (!ext || !ext->rbpf)
        return 0.0;
    if (from < 0 || from >= ext->rbpf->n_regimes)
        return 0.0;
    if (to < 0 || to >= ext->rbpf->n_regimes)
        return 0.0;

    /* Delegate to core RBPF - PGAS owns Π */
    return (double)rbpf_ksc_get_transition_prob(ext->rbpf, from, to);
}

/*═══════════════════════════════════════════════════════════════════════════
 * PARAMETER ACCESS
 *═══════════════════════════════════════════════════════════════════════════*/

void rbpf_ext_get_learned_params(const RBPF_Extended *ext, int regime,
                                 rbpf_real_t *mu_vol, rbpf_real_t *sigma_vol)
{
    if (!ext || regime < 0 || regime >= RBPF_MAX_REGIMES)
    {
        if (mu_vol)
            *mu_vol = RBPF_REAL(-4.6);
        if (sigma_vol)
            *sigma_vol = RBPF_REAL(0.1);
        return;
    }

    switch (ext->param_mode)
    {
    case RBPF_PARAM_STORVIK:
    case RBPF_PARAM_HYBRID:
        if (ext->storvik_initialized)
        {
            RegimeParams params;
            param_learn_get_params(&ext->storvik, 0, regime, &params);
            if (mu_vol)
                *mu_vol = (rbpf_real_t)params.mu;
            if (sigma_vol)
                *sigma_vol = (rbpf_real_t)params.sigma;
        }
        else
        {
            if (mu_vol)
                *mu_vol = ext->rbpf->params[regime].mu_vol;
            if (sigma_vol)
                *sigma_vol = ext->rbpf->params[regime].sigma_vol;
        }
        break;

    default:
        if (mu_vol)
            *mu_vol = ext->rbpf->params[regime].mu_vol;
        if (sigma_vol)
            *sigma_vol = ext->rbpf->params[regime].sigma_vol;
        break;
    }
}

void rbpf_ext_get_storvik_summary(const RBPF_Extended *ext, int regime,
                                  RegimeParams *summary)
{
    if (!ext || !summary || !ext->storvik_initialized)
    {
        if (summary)
            memset(summary, 0, sizeof(RegimeParams));
        return;
    }
    param_learn_get_params(&ext->storvik, 0, regime, summary);
}

void rbpf_ext_get_learning_stats(const RBPF_Extended *ext,
                                 uint64_t *stat_updates,
                                 uint64_t *samples_drawn,
                                 uint64_t *samples_skipped)
{
    if (!ext || !ext->storvik_initialized)
    {
        if (stat_updates)
            *stat_updates = 0;
        if (samples_drawn)
            *samples_drawn = 0;
        if (samples_skipped)
            *samples_skipped = 0;
        return;
    }
    if (stat_updates)
        *stat_updates = ext->storvik.total_stat_updates;
    if (samples_drawn)
        *samples_drawn = ext->storvik.total_samples_drawn;
    if (samples_skipped)
        *samples_skipped = ext->storvik.samples_skipped_load;
}

/*═══════════════════════════════════════════════════════════════════════════
 * KL TEMPERING QUERIES
 *═══════════════════════════════════════════════════════════════════════════*/

int rbpf_ext_kl_tempering_enabled(const RBPF_Extended *ext)
{
    return ext ? ext->kl_tempering_enabled : 0;
}

float rbpf_ext_get_last_beta(const RBPF_Extended *ext)
{
    if (!ext || !ext->kl_state)
        return 1.0f;
    return ext->kl_state->last_beta;
}

float rbpf_ext_get_last_kl(const RBPF_Extended *ext)
{
    if (!ext || !ext->kl_state)
        return 0.0f;
    return ext->kl_state->last_kl;
}

uint64_t rbpf_ext_get_zombie_resets(const RBPF_Extended *ext)
{
    if (!ext || !ext->kl_state)
        return 0;
    return ext->kl_state->zombie_resets;
}

void rbpf_ext_print_kl_diagnostics(const RBPF_Extended *ext)
{
    if (!ext)
    {
        printf("KL Tempering: ext is NULL\n");
        return;
    }

    printf("\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  KL Tempering Diagnostics\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  Enabled:           %s\n", ext->kl_tempering_enabled ? "YES" : "NO");

    if (ext->kl_tempering_enabled && ext->kl_state)
    {
        RBPF_KL_State *kl = ext->kl_state;

        printf("  KL ceiling:        %.4f nats (log N)\n", kl->kl_ceiling);
        printf("  Beta floor:        %.2f\n", kl->beta_floor);
        printf("\n");
        printf("  Last tick:\n");
        printf("    KL divergence:   %.4f nats\n", kl->last_kl);
        printf("    Beta applied:    %.4f\n", kl->last_beta);
        printf("    log(Z_old):      %.4f\n", kl->log_Z_old);
        printf("\n");
        printf("  Counters:\n");
        printf("    Ticks processed: %" PRIu64 "\n", kl->ticks_processed);
        printf("    Hard clamps:     %" PRIu64 " (%.3f%%)\n",
               kl->hard_clamp_count,
               kl->ticks_processed > 0 ? 100.0 * kl->hard_clamp_count / kl->ticks_processed : 0.0);
        printf("    Soft dampens:    %" PRIu64 " (%.3f%%)\n",
               kl->soft_damp_count,
               kl->ticks_processed > 0 ? 100.0 * kl->soft_damp_count / kl->ticks_processed : 0.0);
        printf("    Zombie resets:   %" PRIu64 "\n", kl->zombie_resets);
        printf("\n");
        printf("  Zombie state:\n");
        printf("    Consecutive low: %d / %d\n",
               kl->consecutive_damped_ticks, kl->max_damped_before_reset);
        printf("    Currently zombie: %s\n",
               kl->consecutive_damped_ticks >= kl->max_damped_before_reset ? "YES" : "NO");
        printf("\n");
        printf("  P² Quantile (p95):\n");
        printf("    Warmup:          %s (%" PRIu64 " ticks)\n",
               kl->warmup_complete ? "COMPLETE" : "IN PROGRESS",
               kl->ticks_processed);
        printf("    Current p95:     %.4f nats\n", kl->kl_p95);
        printf("\n");
        printf("  Extremes:\n");
        printf("    Min beta seen:   %.4f\n", kl->min_beta_seen);
        printf("    Max KL seen:     %.4f nats\n", kl->max_kl_seen);
    }
    else if (!ext->kl_tempering_enabled)
    {
        printf("  (Enable with rbpf_ext_enable_kl_tempering())\n");
    }
    else
    {
        printf("  KL state not initialized\n");
    }

    printf("═══════════════════════════════════════════════════════════════\n");
}

/*═══════════════════════════════════════════════════════════════════════════
 * APF KICK QUERIES
 *═══════════════════════════════════════════════════════════════════════════*/

int rbpf_ext_apf_kick_enabled(const RBPF_Extended *ext)
{
    return ext ? ext->apf_kick_enabled : 0;
}

float rbpf_ext_get_apf_surprise_threshold(const RBPF_Extended *ext)
{
    return ext ? ext->apf_surprise_threshold : 0.5f;
}

float rbpf_ext_get_hawkes_intensity(const RBPF_Extended *ext)
{
    if (!ext)
        return 0.0f;
    return hawkes_integrator_get_intensity(&ext->hawkes_integrator);
}

float rbpf_ext_get_hawkes_surprise(const RBPF_Extended *ext)
{
    if (!ext)
        return 0.0f;
    return hawkes_integrator_get_surprise(&ext->hawkes_integrator);
}

int rbpf_ext_hawkes_is_ready(const RBPF_Extended *ext)
{
    if (!ext)
        return 0;
    return hawkes_integrator_is_ready(&ext->hawkes_integrator) ? 1 : 0;
}

void rbpf_ext_print_hawkes_state(const RBPF_Extended *ext)
{
    if (!ext)
        return;
    hawkes_integrator_print_state(&ext->hawkes_integrator);
}

/*═══════════════════════════════════════════════════════════════════════════
 * MAIN DIAGNOSTICS
 *═══════════════════════════════════════════════════════════════════════════*/

void rbpf_ext_print_config(const RBPF_Extended *ext)
{
    if (!ext)
        return;

    printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    printf("║   RBPF-KSC Extended Configuration                            ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    const char *mode_str;
    switch (ext->param_mode)
    {
    case RBPF_PARAM_DISABLED:
        mode_str = "DISABLED";
        break;
    case RBPF_PARAM_LIU_WEST:
        mode_str = "LIU-WEST";
        break;
    case RBPF_PARAM_STORVIK:
        mode_str = "STORVIK";
        break;
    case RBPF_PARAM_HYBRID:
        mode_str = "HYBRID";
        break;
    default:
        mode_str = "UNKNOWN";
        break;
    }

    printf("Parameter Learning: %s\n", mode_str);
    printf("Particles:          %d\n", ext->rbpf->n_particles);
    printf("Regimes:            %d\n", ext->rbpf->n_regimes);

#if defined(__AVX512F__) && !defined(_MSC_VER)
    printf("SIMD:               AVX-512\n");
#elif defined(__AVX2__)
    printf("SIMD:               AVX2\n");
#else
    printf("SIMD:               Scalar\n");
#endif

    if (ext->storvik_initialized)
    {
        printf("\nStorvik Sampling Intervals:\n");
        for (int r = 0; r < ext->rbpf->n_regimes; r++)
        {
            printf("  R%d: every %d ticks\n", r,
                   ext->storvik.config.sample_interval[r]);
        }
    }

    printf("\n  Hawkes Integrator (APF Kick):\n");
    if (ext->apf_kick_enabled)
    {
        printf("    APF Kick:    ENABLED\n");
        printf("    Threshold:   %.2f σ (surprise)\n", ext->apf_surprise_threshold);
        const HawkesIntegrator *hi = &ext->hawkes_integrator;
        printf("    μ (base):    %.4f\n", hi->config.hawkes.mu);
        printf("    α (excite):  %.4f\n", hi->config.hawkes.alpha);
        printf("    β (decay):   %.4f (half-life: %.1f ticks)\n",
               hi->config.hawkes.beta,
               0.693f / (hi->config.hawkes.beta + 1e-6f));
        printf("    Branching:   %.2f\n", hawkes_integrator_get_branching_ratio(hi));
        printf("    Ready:       %s\n", hawkes_integrator_is_ready(hi) ? "YES" : "NO (warmup)");
    }
    else
    {
        printf("    APF Kick:    DISABLED\n");
    }

    printf("\n  Robust OCSN (11th Component):\n");
    if (ext->robust_ocsn.enabled)
    {
        printf("    Enabled:     YES\n");
        for (int r = 0; r < ext->rbpf->n_regimes; r++)
        {
            printf("      R%d: prob=%.1f%%, var=%.1f\n", r,
                   (float)ext->robust_ocsn.regime[r].prob * 100,
                   (float)ext->robust_ocsn.regime[r].variance);
        }
    }
    else
    {
        printf("    Enabled:     NO\n");
    }

    printf("\n  KL Tempering:\n");
    if (ext->kl_tempering_enabled)
    {
        printf("    Enabled:     YES\n");
        if (ext->kl_state)
        {
            printf("    Last β:      %.4f\n", ext->kl_state->last_beta);
            printf("    Last KL:     %.4f nats\n", ext->kl_state->last_kl);
        }
    }
    else
    {
        printf("    Enabled:     NO\n");
    }

    /* Print smoother config */
    rbpf_ext_print_smoother_config(ext);

    printf("\n");
}

void rbpf_ext_print_storvik_stats(const RBPF_Extended *ext, int regime)
{
    if (!ext || !ext->storvik_initialized)
        return;
    printf("\nStorvik Statistics (Regime %d):\n", regime);
    param_learn_print_regime_stats(&ext->storvik, regime);
}