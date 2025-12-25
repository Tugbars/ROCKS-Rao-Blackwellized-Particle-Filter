/**
 * @file oracle_bridge.c
 * @brief Oracle Bridge Implementation
 *
 * Connects Hawkes Trigger → PGAS Oracle → SAEM Blender
 */

#include "oracle_bridge.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/*═══════════════════════════════════════════════════════════════════════════
 * CONFIGURATION
 *═══════════════════════════════════════════════════════════════════════════*/

OracleBridgeConfig oracle_bridge_config_defaults(void)
{
    OracleBridgeConfig cfg;
    memset(&cfg, 0, sizeof(cfg));

    /* PGAS settings */
    cfg.pgas_particles = 256;
    cfg.pgas_sweeps_min = 3;
    cfg.pgas_sweeps_max = 10;
    cfg.pgas_target_accept = 0.15f;

    /* Exponential Weighting (Window Paradox Solution)
     * half-life = ln(2) / lambda ≈ 693 ticks at lambda=0.001
     * This ensures recent data dominates after regime change */
    cfg.recency_lambda = 0.001f;

    /* Dual-gate trigger */
    cfg.use_dual_gate = true;
    cfg.kl_threshold_sigma = 2.0f;

    /* Tempered path */
    cfg.use_tempered_path = true;
    cfg.temper_flip_prob = 0.05f;

    cfg.verbose = false;

    return cfg;
}

/*═══════════════════════════════════════════════════════════════════════════
 * LIFECYCLE
 *═══════════════════════════════════════════════════════════════════════════*/

int oracle_bridge_init(OracleBridge *bridge,
                       const OracleBridgeConfig *cfg,
                       HawkesIntegrator *hawkes,
                       KLTrigger *kl_trigger,
                       SAEMBlender *blender,
                       PGASMKLState *pgas)
{
    if (!bridge)
        return -1;

    memset(bridge, 0, sizeof(*bridge));

    bridge->config = cfg ? *cfg : oracle_bridge_config_defaults();

    /* Store component handles (not owned) */
    bridge->hawkes = hawkes;
    bridge->kl_trigger = kl_trigger; /* Can be NULL to disable dual-gate */
    bridge->blender = blender;
    bridge->pgas = pgas;

    /* Validate components */
    if (!blender || !pgas)
    {
        fprintf(stderr, "OracleBridge: blender and pgas are required\n");
        return -1;
    }

    /* Ensure regime counts match */
    bridge->n_regimes = blender->config.n_regimes;
    if (pgas->K != bridge->n_regimes)
    {
        fprintf(stderr, "OracleBridge: regime mismatch (blender=%d, pgas=%d)\n",
                bridge->n_regimes, pgas->K);
        return -1;
    }

    bridge->last_trigger_tick = -1000; /* Allow immediate first trigger */
    bridge->initialized = true;

    return 0;
}

void oracle_bridge_reset(OracleBridge *bridge)
{
    if (!bridge || !bridge->initialized)
        return;

    bridge->last_hawkes_surprise = 0.0f;
    bridge->last_kl_surprise = 0.0f;
    bridge->last_trigger_tick = -1000;
    bridge->total_oracle_calls = 0;
    bridge->successful_blends = 0;
    bridge->cumulative_kl_change = 0.0f;
}

/*═══════════════════════════════════════════════════════════════════════════
 * TRIGGER CHECK
 *═══════════════════════════════════════════════════════════════════════════*/

OracleTriggerResult oracle_bridge_check_trigger(
    OracleBridge *bridge,
    const HawkesIntegratorResult *hawkes_result,
    float kl_surprise,
    int current_tick)
{

    OracleTriggerResult result;
    memset(&result, 0, sizeof(result));

    if (!bridge || !bridge->initialized || !hawkes_result)
    {
        return result;
    }

    result.hawkes_surprise = hawkes_result->surprise_sigma;
    result.kl_surprise = kl_surprise;
    result.triggered_by_panic = hawkes_result->triggered_by_panic;
    result.ticks_since_last = current_tick - bridge->last_trigger_tick;

    /* Check Hawkes trigger */
    bool hawkes_fired = hawkes_result->should_trigger;

    /* Check KL trigger (if dual-gate enabled) */
    bool kl_fired = true; /* Default: pass if not using dual-gate */
    if (bridge->config.use_dual_gate)
    {
        kl_fired = (kl_surprise >= bridge->config.kl_threshold_sigma);
    }

    /* Absolute panic overrides dual-gate requirement */
    if (hawkes_result->triggered_by_panic)
    {
        result.should_trigger = true;
    }
    else if (bridge->config.use_dual_gate)
    {
        /* Dual-gate: both must fire */
        result.should_trigger = hawkes_fired && kl_fired;
    }
    else
    {
        /* Single-gate: Hawkes only */
        result.should_trigger = hawkes_fired;
    }

    return result;
}

/*═══════════════════════════════════════════════════════════════════════════
 * PGAS → SAEM CONVERSION
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Extract sufficient statistics from PGAS into PGASOutput format
 */
static void pgas_to_saem_output(const PGASMKLState *pgas,
                                float trigger_surprise,
                                PGASOutput *output)
{
    if (!pgas || !output)
        return;

    int K = pgas->K;
    output->n_regimes = K;
    output->n_trajectories = 1; /* Single trajectory from PGAS */
    output->trajectory_length = pgas->T;
    output->trigger_surprise = trigger_surprise;

    /* Copy transition counts as floats
     * PGAS stores as int[K*K], SAEM expects float[K][K] */
    for (int i = 0; i < K; i++)
    {
        for (int j = 0; j < K; j++)
        {
            output->S[i][j] = (float)pgas->n_trans[i * K + j];
        }
    }

    /* MCMC diagnostics */
    output->acceptance_rate = pgas->acceptance_rate;

    /* ESS at final time */
    float ess = pgas_mkl_get_ess(pgas, pgas->T - 1);
    output->ess_fraction = ess / (float)pgas->N;
}

/*═══════════════════════════════════════════════════════════════════════════
 * ORACLE EXECUTION
 *═══════════════════════════════════════════════════════════════════════════*/

OracleRunResult oracle_bridge_run(
    OracleBridge *bridge,
    const int *rbpf_path,
    const double *rbpf_h,
    const double *observations,
    int T,
    float trigger_surprise)
{

    OracleRunResult result;
    memset(&result, 0, sizeof(result));

    if (!bridge || !bridge->initialized || !rbpf_path || !rbpf_h ||
        !observations || T < 2)
    {
        return result;
    }

    PGASMKLState *pgas = bridge->pgas;
    SAEMBlender *blender = bridge->blender;
    int K = bridge->n_regimes;

    bridge->total_oracle_calls++;
    bridge->last_hawkes_surprise = trigger_surprise;

    if (bridge->config.verbose)
    {
        printf("\n[OracleBridge] Running Oracle (trigger=%.2fσ, T=%d)\n",
               trigger_surprise, T);
    }

    /* ═══════════════════════════════════════════════════════════════════
     * STEP 1: Prepare reference path (with optional tempering)
     * ═══════════════════════════════════════════════════════════════════*/

    int *ref_path = (int *)malloc(T * sizeof(int));
    if (!ref_path)
    {
        fprintf(stderr, "OracleBridge: malloc failed\n");
        return result;
    }

    if (bridge->config.use_tempered_path && blender)
    {
        result.temper_flips = saem_blender_temper_path(
            blender, rbpf_path, T, ref_path);

        if (bridge->config.verbose)
        {
            printf("[OracleBridge] Tempered path: %d flips (%.1f%%)\n",
                   result.temper_flips, 100.0f * result.temper_flips / T);
        }
    }
    else
    {
        memcpy(ref_path, rbpf_path, T * sizeof(int));
        result.temper_flips = 0;
    }

    /* ═══════════════════════════════════════════════════════════════════
     * STEP 2: Load data into PGAS
     * ═══════════════════════════════════════════════════════════════════*/

    pgas_mkl_load_observations(pgas, observations, T);
    pgas_mkl_set_reference(pgas, ref_path, rbpf_h, T);

    /* Copy current transition matrix from blender to PGAS */
    float Pi_current[SAEM_MAX_REGIMES * SAEM_MAX_REGIMES];
    saem_blender_get_Pi(blender, Pi_current);

    /* Convert to double for PGAS API */
    double trans_d[PGAS_MKL_MAX_K * PGAS_MKL_MAX_K];
    double mu_vol_d[PGAS_MKL_MAX_K];
    double sigma_vol_d[PGAS_MKL_MAX_K];

    for (int i = 0; i < K * K; i++)
    {
        trans_d[i] = (double)Pi_current[i];
    }
    for (int k = 0; k < K; k++)
    {
        mu_vol_d[k] = (double)pgas->model.mu_vol[k];
        sigma_vol_d[k] = (double)pgas->model.sigma_vol[k];
    }

    pgas_mkl_set_model(pgas, trans_d, mu_vol_d, sigma_vol_d,
                       (double)pgas->model.phi);

    /* Set exponential recency weighting (Window Paradox Solution)
     * This ensures PGAS learns Π_now instead of Π_average after regime change */
    pgas_mkl_set_recency_lambda(pgas, bridge->config.recency_lambda);

    /* ═══════════════════════════════════════════════════════════════════
     * STEP 3: Run PGAS Gibbs sweeps
     * ═══════════════════════════════════════════════════════════════════*/

    float total_accept = 0.0f;
    int sweeps = 0;

    for (sweeps = 0; sweeps < bridge->config.pgas_sweeps_max; sweeps++)
    {
        float accept = pgas_mkl_gibbs_sweep(pgas);
        total_accept += accept;

        if (sweeps >= bridge->config.pgas_sweeps_min - 1 &&
            accept >= bridge->config.pgas_target_accept)
        {
            sweeps++;
            break;
        }
    }

    result.acceptance_rate = total_accept / sweeps;
    result.sweeps_used = sweeps;

    /* Get final ESS */
    float ess = pgas_mkl_get_ess(pgas, T - 1);
    result.ess_fraction = ess / (float)pgas->N;

    if (bridge->config.verbose)
    {
        printf("[OracleBridge] PGAS: %d sweeps, accept=%.3f, ESS=%.1f\n",
               sweeps, result.acceptance_rate, ess);
    }

    /* ═══════════════════════════════════════════════════════════════════
     * STEP 4: Extract sufficient statistics and blend
     * ═══════════════════════════════════════════════════════════════════*/

    PGASOutput oracle_output;
    memset(&oracle_output, 0, sizeof(oracle_output));
    pgas_to_saem_output(pgas, trigger_surprise, &oracle_output);

    SAEMBlendResult blend = saem_blender_blend(blender, &oracle_output);

    result.success = blend.success;
    result.kl_divergence = blend.kl_divergence;
    result.diag_before = blend.diag_avg_before;
    result.diag_after = blend.diag_avg_after;
    result.stickiness_adjusted = blend.stickiness_adjusted;

    if (blend.success)
    {
        bridge->successful_blends++;
        bridge->cumulative_kl_change += blend.kl_divergence;
    }

    if (bridge->config.verbose)
    {
        printf("[OracleBridge] Blend: KL=%.6f, diag %.4f→%.4f%s\n",
               blend.kl_divergence, blend.diag_avg_before, blend.diag_avg_after,
               blend.stickiness_adjusted ? " (κ adjusted)" : "");
    }

    /* Cleanup */
    free(ref_path);

    return result;
}

/*═══════════════════════════════════════════════════════════════════════════
 * GETTERS
 *═══════════════════════════════════════════════════════════════════════════*/

void oracle_bridge_get_Pi(const OracleBridge *bridge, float *Pi_out)
{
    if (!bridge || !bridge->initialized || !bridge->blender || !Pi_out)
        return;
    saem_blender_get_Pi(bridge->blender, Pi_out);
}

void oracle_bridge_get_stats(const OracleBridge *bridge, OracleBridgeStats *stats)
{
    if (!bridge || !stats)
        return;

    memset(stats, 0, sizeof(*stats));

    stats->total_oracle_calls = bridge->total_oracle_calls;
    stats->successful_blends = bridge->successful_blends;

    if (bridge->total_oracle_calls > 0)
    {
        stats->avg_kl_change = bridge->cumulative_kl_change / bridge->total_oracle_calls;
    }

    if (bridge->blender)
    {
        stats->current_gamma = saem_blender_get_gamma(bridge->blender);
        stats->current_avg_diagonal = saem_blender_get_avg_diagonal(bridge->blender);
    }
}

void oracle_bridge_print_state(const OracleBridge *bridge)
{
    if (!bridge || !bridge->initialized)
        return;

    OracleBridgeStats stats;
    oracle_bridge_get_stats(bridge, &stats);

    printf("\n");
    printf("+===========================================================+\n");
    printf("|                  ORACLE BRIDGE STATE                      |\n");
    printf("+===========================================================+\n");
    printf("| Regimes: %d                                               \n", bridge->n_regimes);
    printf("| Oracle calls: %d (successful: %d)                         \n",
           stats.total_oracle_calls, stats.successful_blends);
    printf("| Avg KL change: %.6f                                      \n", stats.avg_kl_change);
    printf("| Current γ: %.4f                                          \n", stats.current_gamma);
    printf("| Avg diagonal: %.4f                                       \n", stats.current_avg_diagonal);
    printf("| Last trigger: tick %d (surprise=%.2fσ)                   \n",
           bridge->last_trigger_tick, bridge->last_hawkes_surprise);
    printf("+-----------------------------------------------------------+\n");
    printf("| Config:                                                   |\n");
    printf("|   Dual-gate: %s (KL threshold=%.1fσ)                     \n",
           bridge->config.use_dual_gate ? "ON" : "OFF",
           bridge->config.kl_threshold_sigma);
    printf("|   Tempered path: %s (flip=%.1f%%)                        \n",
           bridge->config.use_tempered_path ? "ON" : "OFF",
           bridge->config.temper_flip_prob * 100);
    printf("|   PGAS sweeps: %d-%d (target accept=%.2f)                \n",
           bridge->config.pgas_sweeps_min, bridge->config.pgas_sweeps_max,
           bridge->config.pgas_target_accept);
    printf("+===========================================================+\n");

    if (bridge->blender)
    {
        saem_blender_print_Pi(bridge->blender);
    }
}