/**
 * @file oracle_bridge.c
 * @brief Oracle Bridge Implementation - Full Pipeline
 *
 * Flow: Trigger → Scout → PGAS → Confidence → SAEM → Thompson → RBPF
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

    /* Exponential Weighting (Window Paradox Solution) */
    cfg.recency_lambda = 0.001f;

    /* Dual-gate trigger */
    cfg.use_dual_gate = true;
    cfg.kl_threshold_sigma = 2.0f;

    /* Scout sweep */
    cfg.use_scout_sweep = true;
    cfg.scout_sweeps = 5;
    cfg.scout_min_acceptance = 0.10f;
    cfg.scout_min_unique_frac = 0.25f;
    cfg.scout_entropy_skip = 0.1f;

    /* Tempered path */
    cfg.use_tempered_path = true;
    cfg.temper_flip_prob = 0.05f;

    /* Thompson */
    cfg.thompson_exploit_thresh = 500.0f;

    /* Confidence-based γ */
    cfg.gamma_on_regime_change = 0.50f;
    cfg.gamma_on_degeneracy = SAEM_GAMMA_MIN;

    cfg.verbose = false;

    return cfg;
}

/*═══════════════════════════════════════════════════════════════════════════
 * LIFECYCLE
 *═══════════════════════════════════════════════════════════════════════════*/

int oracle_bridge_init_full(OracleBridge *bridge,
                            const OracleBridgeConfig *cfg,
                            HawkesIntegrator *hawkes,
                            KLTrigger *kl_trigger,
                            SAEMBlender *blender,
                            PGASMKLState *pgas,
                            PARISMKLState *paris,
                            ThompsonSampler *thompson)
{
    if (!bridge)
        return -1;

    memset(bridge, 0, sizeof(*bridge));

    bridge->config = cfg ? *cfg : oracle_bridge_config_defaults();

    /* Store component handles (not owned) */
    bridge->hawkes = hawkes;
    bridge->kl_trigger = kl_trigger;
    bridge->blender = blender;
    bridge->pgas = pgas;
    bridge->paris = paris;
    bridge->thompson = thompson;

    /* Validate required components */
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

    /* Optional: validate PARIS if provided */
    if (paris && paris->K != bridge->n_regimes)
    {
        fprintf(stderr, "OracleBridge: regime mismatch (paris=%d)\n", paris->K);
        return -1;
    }

    /* Optional: validate Thompson if provided */
    if (thompson && thompson->config.n_regimes != bridge->n_regimes)
    {
        fprintf(stderr, "OracleBridge: regime mismatch (thompson=%d)\n",
                thompson->config.n_regimes);
        return -1;
    }

    bridge->last_trigger_tick = -1000;
    bridge->initialized = true;

    return 0;
}

/* Backward compatible init */
int oracle_bridge_init(OracleBridge *bridge,
                       const OracleBridgeConfig *cfg,
                       HawkesIntegrator *hawkes,
                       KLTrigger *kl_trigger,
                       SAEMBlender *blender,
                       PGASMKLState *pgas)
{
    return oracle_bridge_init_full(bridge, cfg, hawkes, kl_trigger,
                                   blender, pgas, NULL, NULL);
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
    bridge->scout_skip_count = 0;
    bridge->regime_change_count = 0;
    bridge->degeneracy_count = 0;
    bridge->cumulative_kl_change = 0.0f;
    bridge->last_scout_valid = false;
    bridge->last_scout_skipped_pgas = false;
    memset(&bridge->last_confidence, 0, sizeof(bridge->last_confidence));
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

    bool hawkes_fired = hawkes_result->should_trigger;
    bool kl_fired = true;

    if (bridge->config.use_dual_gate)
    {
        kl_fired = (kl_surprise >= bridge->config.kl_threshold_sigma);
    }

    if (hawkes_result->triggered_by_panic)
    {
        result.should_trigger = true;
    }
    else if (bridge->config.use_dual_gate)
    {
        result.should_trigger = hawkes_fired && kl_fired;
    }
    else
    {
        result.should_trigger = hawkes_fired;
    }

    return result;
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
     * PHASE 1: SCOUT SWEEP (Pre-validation)
     * ═══════════════════════════════════════════════════════════════════*/

    bool should_run_pgas = true;

    if (bridge->config.use_scout_sweep && bridge->paris)
    {
        PARISScoutConfig scout_cfg;
        scout_cfg.n_sweeps = bridge->config.scout_sweeps;
        scout_cfg.min_acceptance = bridge->config.scout_min_acceptance;
        scout_cfg.min_unique_fraction = bridge->config.scout_min_unique_frac;

        PARISScoutResult scout = paris_mkl_scout_sweep(bridge->paris, &scout_cfg);

        result.scout_ran = true;
        result.scout_valid = scout.is_valid;
        result.scout_entropy = scout.entropy;
        result.scout_unique_paths = scout.unique_paths;

        bridge->last_scout_valid = scout.is_valid;

        if (!scout.is_valid)
        {
            /* Scout degenerate - can't trust filter, force PGAS */
            if (bridge->config.verbose)
            {
                printf("[OracleBridge] Scout INVALID: accept=%.2f unique=%d → force PGAS\n",
                       scout.acceptance_rate, scout.unique_paths);
            }
            should_run_pgas = true;
        }
        else if (scout.entropy < bridge->config.scout_entropy_skip)
        {
            /* Scout valid + low entropy → filter confident, skip PGAS */
            if (bridge->config.verbose)
            {
                printf("[OracleBridge] Scout VALID + low entropy (%.3f) → skip PGAS\n",
                       scout.entropy);
            }
            result.scout_skipped_pgas = true;
            bridge->last_scout_skipped_pgas = true;
            bridge->scout_skip_count++;
            should_run_pgas = false;
        }
        else
        {
            if (bridge->config.verbose)
            {
                printf("[OracleBridge] Scout VALID + high entropy (%.3f) → run PGAS\n",
                       scout.entropy);
            }
        }
    }

    if (!should_run_pgas)
    {
        /* Scout allowed skip - return early with success */
        result.success = true;
        return result;
    }

    /* ═══════════════════════════════════════════════════════════════════
     * PHASE 2: PREPARE REFERENCE PATH (with tempering)
     * ═══════════════════════════════════════════════════════════════════*/

    int *ref_path = (int *)malloc(T * sizeof(int));
    int *ref_path_original = (int *)malloc(T * sizeof(int));
    if (!ref_path || !ref_path_original)
    {
        fprintf(stderr, "OracleBridge: malloc failed\n");
        free(ref_path);
        free(ref_path_original);
        return result;
    }

    /* Keep original for divergence check */
    memcpy(ref_path_original, rbpf_path, T * sizeof(int));

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
     * PHASE 3: RUN PGAS
     * ═══════════════════════════════════════════════════════════════════*/

    result.pgas_ran = true;

    pgas_mkl_load_observations(pgas, observations, T);
    pgas_mkl_set_reference(pgas, ref_path, rbpf_h, T);

    /* Copy current Π from blender to PGAS */
    float Pi_current[SAEM_MAX_REGIMES * SAEM_MAX_REGIMES];
    saem_blender_get_Pi(blender, Pi_current);

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
    pgas_mkl_set_recency_lambda(pgas, bridge->config.recency_lambda);

    /* Run Gibbs sweeps */
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

    float ess = pgas_mkl_get_ess(pgas, T - 1);
    result.ess_fraction = ess / (float)pgas->N;

    if (bridge->config.verbose)
    {
        printf("[OracleBridge] PGAS: %d sweeps, accept=%.3f, ESS=%.1f\n",
               sweeps, result.acceptance_rate, ess);
    }

    /* ═══════════════════════════════════════════════════════════════════
     * PHASE 4: PGAS CONFIDENCE → ADAPTIVE γ
     * ═══════════════════════════════════════════════════════════════════*/

    PGASConfidence conf;
    pgas_confidence_compute(pgas, ref_path_original, T, &conf, NULL);

    bridge->last_confidence = conf;
    result.confidence_score = conf.overall_confidence;
    result.path_divergence = conf.path_divergence;
    result.regime_change_detected = conf.regime_change_detected;
    result.degeneracy_detected = !pgas_confidence_usable(&conf);

    float gamma;

    if (conf.regime_change_detected)
    {
        /* Tier-2 reset: market fundamentally changed */
        saem_blender_tier2_reset(blender);
        gamma = bridge->config.gamma_on_regime_change;
        bridge->regime_change_count++;

        if (bridge->config.verbose)
        {
            printf("[OracleBridge] REGIME CHANGE detected → tier2 reset, γ=%.2f\n", gamma);
        }
    }
    else if (!pgas_confidence_usable(&conf))
    {
        /* Degeneracy: PGAS failed to mix properly */
        gamma = bridge->config.gamma_on_degeneracy;
        bridge->degeneracy_count++;

        if (bridge->config.verbose)
        {
            printf("[OracleBridge] DEGENERACY detected → minimal blend, γ=%.3f\n", gamma);
        }
    }
    else
    {
        /* Normal operation: use confidence-derived γ */
        gamma = pgas_confidence_get_gamma(&conf);

        if (bridge->config.verbose)
        {
            printf("[OracleBridge] Confidence=%.3f → γ=%.3f\n",
                   conf.overall_confidence, gamma);
        }
    }

    result.gamma_used = gamma;

    /* ═══════════════════════════════════════════════════════════════════
     * PHASE 5: SAEM BLEND
     * ═══════════════════════════════════════════════════════════════════*/

    /* Extract transition counts from PGAS */
    float S[SAEM_MAX_REGIMES * SAEM_MAX_REGIMES];
    for (int i = 0; i < K; i++)
    {
        for (int j = 0; j < K; j++)
        {
            S[i * K + j] = (float)pgas->n_trans[i * K + j];
        }
    }

    result.diag_before = saem_blender_get_avg_diagonal(blender);

    SAEMBlendResult blend = saem_blender_blend_counts(blender, S, gamma);

    result.success = blend.success;
    result.kl_divergence = blend.kl_divergence;
    result.diag_after = blend.diag_avg_after;
    result.stickiness_adjusted = blend.stickiness_adjusted;

    if (blend.success)
    {
        bridge->successful_blends++;
        bridge->cumulative_kl_change += blend.kl_divergence;
    }

    if (bridge->config.verbose)
    {
        printf("[OracleBridge] Blend: γ=%.3f, KL=%.6f, diag %.4f→%.4f%s\n",
               gamma, blend.kl_divergence, result.diag_before, result.diag_after,
               blend.stickiness_adjusted ? " (κ adjusted)" : "");
    }

    /* ═══════════════════════════════════════════════════════════════════
     * PHASE 6: THOMPSON SAMPLING (if enabled)
     * ═══════════════════════════════════════════════════════════════════*/

    if (bridge->thompson)
    {
        float Q[SAEM_MAX_REGIMES * SAEM_MAX_REGIMES];
        float Pi_thompson[SAEM_MAX_REGIMES * SAEM_MAX_REGIMES];

        saem_blender_get_Q(blender, Q);
        ThompsonSampleResult ts_result = thompson_sampler_sample_flat(
            bridge->thompson, Q, K, Pi_thompson);

        result.thompson_explored = ts_result.explored;

        if (bridge->config.verbose)
        {
            printf("[OracleBridge] Thompson: %s (min_row=%.0f)\n",
                   ts_result.explored ? "EXPLORE" : "EXPLOIT",
                   ts_result.min_row_sum);
        }
    }

    /* Cleanup */
    free(ref_path);
    free(ref_path_original);

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

void oracle_bridge_get_Pi_thompson(OracleBridge *bridge, float *Pi_out)
{
    if (!bridge || !bridge->initialized || !bridge->blender || !Pi_out)
        return;

    int K = bridge->n_regimes;

    if (bridge->thompson)
    {
        float Q[SAEM_MAX_REGIMES * SAEM_MAX_REGIMES];
        saem_blender_get_Q(bridge->blender, Q);
        thompson_sampler_sample_flat(bridge->thompson, Q, K, Pi_out);
    }
    else
    {
        /* No Thompson: return SAEM mean */
        saem_blender_get_Pi(bridge->blender, Pi_out);
    }
}

void oracle_bridge_get_Q(const OracleBridge *bridge, float *Q_out)
{
    if (!bridge || !bridge->initialized || !bridge->blender || !Q_out)
        return;
    saem_blender_get_Q(bridge->blender, Q_out);
}

void oracle_bridge_get_last_confidence(const OracleBridge *bridge, PGASConfidence *conf)
{
    if (!bridge || !conf)
        return;
    *conf = bridge->last_confidence;
}

void oracle_bridge_get_stats(const OracleBridge *bridge, OracleBridgeStats *stats)
{
    if (!bridge || !stats)
        return;

    memset(stats, 0, sizeof(*stats));

    stats->total_oracle_calls = bridge->total_oracle_calls;
    stats->successful_blends = bridge->successful_blends;
    stats->scout_skip_count = bridge->scout_skip_count;
    stats->regime_change_count = bridge->regime_change_count;
    stats->degeneracy_count = bridge->degeneracy_count;

    if (bridge->total_oracle_calls > 0)
    {
        stats->avg_kl_change = bridge->cumulative_kl_change / bridge->total_oracle_calls;
    }

    if (bridge->blender)
    {
        stats->current_gamma = saem_blender_get_gamma(bridge->blender);
        stats->current_avg_diagonal = saem_blender_get_avg_diagonal(bridge->blender);
    }

    if (bridge->thompson)
    {
        stats->thompson_explore_ratio = thompson_sampler_get_explore_ratio(bridge->thompson);
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
    printf("+-----------------------------------------------------------+\n");
    printf("| Statistics:                                               |\n");
    printf("|   Oracle calls:    %d (successful: %d)                    \n",
           stats.total_oracle_calls, stats.successful_blends);
    printf("|   Scout skips:     %d                                     \n", stats.scout_skip_count);
    printf("|   Regime changes:  %d                                     \n", stats.regime_change_count);
    printf("|   Degeneracies:    %d                                     \n", stats.degeneracy_count);
    printf("|   Avg KL change:   %.6f                                  \n", stats.avg_kl_change);
    printf("+-----------------------------------------------------------+\n");
    printf("| Current state:                                            |\n");
    printf("|   γ: %.4f                                                \n", stats.current_gamma);
    printf("|   Avg diagonal: %.4f                                     \n", stats.current_avg_diagonal);
    if (bridge->thompson)
    {
        printf("|   Thompson explore ratio: %.1f%%                         \n",
               stats.thompson_explore_ratio * 100);
    }
    printf("|   Last trigger: tick %d (surprise=%.2fσ)                 \n",
           bridge->last_trigger_tick, bridge->last_hawkes_surprise);
    printf("+-----------------------------------------------------------+\n");
    printf("| Config:                                                   |\n");
    printf("|   Dual-gate: %s (KL threshold=%.1fσ)                     \n",
           bridge->config.use_dual_gate ? "ON" : "OFF",
           bridge->config.kl_threshold_sigma);
    printf("|   Scout sweep: %s                                        \n",
           (bridge->config.use_scout_sweep && bridge->paris) ? "ON" : "OFF");
    printf("|   Tempered path: %s (flip=%.1f%%)                        \n",
           bridge->config.use_tempered_path ? "ON" : "OFF",
           bridge->config.temper_flip_prob * 100);
    printf("|   PGAS sweeps: %d-%d (target accept=%.2f)                \n",
           bridge->config.pgas_sweeps_min, bridge->config.pgas_sweeps_max,
           bridge->config.pgas_target_accept);
    printf("|   Thompson: %s                                           \n",
           bridge->thompson ? "ON" : "OFF");
    printf("+===========================================================+\n");

    /* Last confidence */
    if (bridge->total_oracle_calls > 0)
    {
        printf("| Last PGAS Confidence:                                     |\n");
        printf("|   Overall: %.3f                                          \n",
               bridge->last_confidence.overall_confidence);
        printf("|   Path divergence: %.1f%%                                \n",
               bridge->last_confidence.path_divergence * 100);
        printf("|   ESS ratio: %.3f                                        \n",
               bridge->last_confidence.ess_ratio);
        printf("|   Regime change: %s                                      \n",
               bridge->last_confidence.regime_change_detected ? "YES" : "NO");
        printf("+===========================================================+\n");
    }

    if (bridge->blender)
    {
        saem_blender_print_Pi(bridge->blender);
    }
}