/**
 * @file regime_system.c
 * @brief Complete Regime Trading System Implementation
 *
 * Implements the integration schema:
 *   1. Push observation to PGAS
 *   2. Update Hawkes (crisis detection)
 *   3. Set adaptive λ
 *   4. Hot swap with Hawkes veto
 *   5. RBPF update (stub - to be replaced)
 */

#include "regime_system.h"
#include "pgas_sliding.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/*═══════════════════════════════════════════════════════════════════════════════
 * DEFAULT CONFIGURATION
 *═══════════════════════════════════════════════════════════════════════════════*/

RegimeSystemConfig regime_system_config_defaults(void)
{
    RegimeSystemConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    
    /* Model: 4 regimes, uniform initial Π */
    cfg.K = 4;
    float unif = 1.0f / 4.0f;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            cfg.trans[i * 4 + j] = unif;
        }
        cfg.mu_vol[i] = -1.0 + 0.5 * i;  /* [-1.0, -0.5, 0.0, 0.5] */
        cfg.sigma_vol[i] = 0.1 + 0.05 * i;  /* [0.1, 0.15, 0.2, 0.25] */
    }
    cfg.phi = 0.97;
    
    /* PGAS Oracle */
    cfg.pgas_window_size = 2000;
    cfg.pgas_slide_step = 100;
    cfg.pgas_particles = 256;
    cfg.pgas_sweeps = 3;
    cfg.pgas_prior_alpha = 1.0f;
    cfg.pgas_sticky_kappa = 50.0f;
    cfg.pgas_recency_lambda = 0.001f;
    
    /* RBPF */
    cfg.rbpf_particles = 1024;
    
    /* Hawkes - use responsive config for circuit breaker */
    cfg.hawkes_config = hawkes_integrator_config_responsive();
    
    /* Adaptive forgetting */
    cfg.lambda_steady = 0.995f;  /* Half-life ~140 ticks */
    cfg.lambda_crisis = 0.5f;    /* Half-life ~1.4 ticks */
    cfg.n_eff_reset = 1000.0f;
    
    /* Core affinity */
    cfg.pgas_core_start = 2;
    cfg.pgas_n_cores = 30;
    
    /* RNG */
    cfg.seed = 12345;
    
    return cfg;
}

/*═══════════════════════════════════════════════════════════════════════════════
 * LIFECYCLE
 *═══════════════════════════════════════════════════════════════════════════════*/

RegimeSystem* regime_system_alloc(const RegimeSystemConfig* cfg)
{
    if (!cfg || cfg->K > REGIME_SYSTEM_MAX_K) {
        return NULL;
    }
    
    RegimeSystem* sys = (RegimeSystem*)calloc(1, sizeof(RegimeSystem));
    if (!sys) return NULL;
    
    /* ═══════════════════════════════════════════════════════════════════════
     * ALLOCATE PGAS ORACLE
     * ═══════════════════════════════════════════════════════════════════════*/
    sys->oracle = pgas_oracle_alloc(
        cfg->pgas_window_size,
        cfg->pgas_slide_step,
        cfg->pgas_particles,
        cfg->K,
        cfg->pgas_sweeps,
        cfg->seed
    );
    
    if (!sys->oracle) {
        free(sys);
        return NULL;
    }
    
    /* Configure PGAS model */
    pgas_oracle_set_model(sys->oracle, cfg->trans, cfg->mu_vol, 
                          cfg->sigma_vol, cfg->phi);
    pgas_oracle_set_prior(sys->oracle, cfg->pgas_prior_alpha, 
                          cfg->pgas_sticky_kappa);
    pgas_oracle_set_recency(sys->oracle, cfg->pgas_recency_lambda);
    pgas_oracle_set_affinity(sys->oracle, cfg->pgas_core_start, cfg->pgas_n_cores);
    
    /* ═══════════════════════════════════════════════════════════════════════
     * INITIALIZE HAWKES
     * ═══════════════════════════════════════════════════════════════════════*/
    if (hawkes_integrator_init(&sys->hawkes, &cfg->hawkes_config) != 0) {
        pgas_oracle_free(sys->oracle);
        free(sys);
        return NULL;
    }
    
    /* ═══════════════════════════════════════════════════════════════════════
     * INITIALIZE LOCAL STATE
     * ═══════════════════════════════════════════════════════════════════════*/
    sys->K = cfg->K;
    sys->lambda_steady = cfg->lambda_steady;
    sys->lambda_crisis = cfg->lambda_crisis;
    sys->n_eff_reset = cfg->n_eff_reset;
    sys->lambda = cfg->lambda_steady;
    
    /* Initialize Π to uniform */
    float unif = 1.0f / cfg->K;
    for (int i = 0; i < cfg->K; i++) {
        for (int j = 0; j < cfg->K; j++) {
            sys->Pi[i * cfg->K + j] = unif;
        }
    }
    
    return sys;
}

void regime_system_free(RegimeSystem* sys)
{
    if (!sys) return;
    
    regime_system_stop(sys);
    
    if (sys->oracle) {
        pgas_oracle_free(sys->oracle);
    }
    
    hawkes_integrator_free(&sys->hawkes);
    
    free(sys);
}

int regime_system_start(RegimeSystem* sys)
{
    if (!sys || !sys->oracle) return -1;
    return pgas_oracle_start(sys->oracle);
}

void regime_system_stop(RegimeSystem* sys)
{
    if (!sys || !sys->oracle) return;
    pgas_oracle_stop(sys->oracle);
}

/*═══════════════════════════════════════════════════════════════════════════════
 * CORE TICK UPDATE
 *═══════════════════════════════════════════════════════════════════════════════*/

RegimeSystemTickResult regime_system_tick(RegimeSystem* sys, 
                                           float observation, 
                                           int64_t tick)
{
    RegimeSystemTickResult result;
    memset(&result, 0, sizeof(result));
    
    if (!sys) return result;
    
    sys->total_ticks++;
    
    /* ══════════════════════════════════════════════════════════════════════
     * STEP 0: PUSH OBSERVATION TO PGAS
     *
     * Observation goes into ring buffer. PGAS background thread will
     * extract window, run sweeps, and publish Π when ready.
     * ══════════════════════════════════════════════════════════════════════ */
    pgas_oracle_push(sys->oracle, observation, tick);
    
    /* ══════════════════════════════════════════════════════════════════════
     * STEP 1: HAWKES CIRCUIT BREAKER
     *
     * Updates Hawkes intensity and state machine.
     * Crisis = ARMED or FIRED state, meaning regime structure is breaking.
     * ══════════════════════════════════════════════════════════════════════ */
    HawkesIntegratorResult hawkes_res = hawkes_integrator_update(
        &sys->hawkes, (float)tick, observation);
    
    bool crisis = (hawkes_res.state == HAWKES_TRIG_ARMED ||
                   hawkes_res.state == HAWKES_TRIG_FIRED);
    
    result.crisis = crisis;
    result.hawkes_state = hawkes_res.state;
    result.hawkes_surprise = hawkes_res.surprise_sigma;
    
    if (crisis) {
        sys->crisis_ticks++;
    }
    
    /* ══════════════════════════════════════════════════════════════════════
     * STEP 2: ADAPTIVE FORGETTING (Plasticity Dial)
     *
     * Crisis:  λ=0.5   → half-life ~1.4 ticks (instant adaptation)
     * Normal:  λ=0.995 → half-life ~140 ticks (stable)
     * ══════════════════════════════════════════════════════════════════════ */
    sys->lambda = crisis ? sys->lambda_crisis : sys->lambda_steady;
    result.lambda = sys->lambda;
    
    /* ══════════════════════════════════════════════════════════════════════
     * STEP 3: HOT SWAP WITH HAWKES VETO
     *
     * No threshold checking. No blending.
     * PGAS ready? Swap. Crisis? Veto.
     *
     * When swapping:
     *   - Take PGAS Π as truth (100%, no blend)
     *   - Reset sufficient statistics to lock Storvik
     * ══════════════════════════════════════════════════════════════════════ */
    float pi_new[REGIME_SYSTEM_MAX_K * REGIME_SYSTEM_MAX_K];
    int64_t swap_tick;
    
    if (pgas_oracle_try_hot_swap(sys->oracle, crisis, pi_new, &swap_tick)) {
        /* SWAP: Take PGAS Π as truth */
        memcpy(sys->Pi, pi_new, sys->K * sys->K * sizeof(float));
        
        /* TODO: When RBPF is integrated:
         * reset_sufficient_statistics(sys->rbpf, sys->Pi, sys->n_eff_reset);
         * rbpf_lut_rebuild(sys->rbpf);
         */
        
        result.swapped = true;
        result.swap_tick = swap_tick;
        sys->swaps_performed++;
    } else if (pgas_oracle_ready(sys->oracle) && crisis) {
        /* Swap was vetoed due to crisis */
        sys->swaps_vetoed++;
    }
    
    /* ══════════════════════════════════════════════════════════════════════
     * STEP 4: RBPF UPDATE (STUB)
     *
     * TODO: Replace with actual RBPF step:
     *   rbpf_step(sys->rbpf, observation, &rbpf_output);
     *
     * For now, just return dummy regime info.
     * ══════════════════════════════════════════════════════════════════════ */
    result.dominant_regime = 0;
    for (int k = 0; k < sys->K; k++) {
        result.regime_probs[k] = 1.0f / sys->K;  /* Uniform placeholder */
    }
    
    /* ══════════════════════════════════════════════════════════════════════
     * STEP 5: STRATEGY (STUB)
     *
     * TODO: Replace with actual strategy:
     *   ou_strategy_update(&sys->strategy, &rbpf_output, &signal);
     * ══════════════════════════════════════════════════════════════════════ */
    
    return result;
}

/*═══════════════════════════════════════════════════════════════════════════════
 * QUERIES
 *═══════════════════════════════════════════════════════════════════════════════*/

void regime_system_get_pi(const RegimeSystem* sys, float* pi_out)
{
    if (!sys || !pi_out) return;
    memcpy(pi_out, sys->Pi, sys->K * sys->K * sizeof(float));
}

float regime_system_get_lambda(const RegimeSystem* sys)
{
    return sys ? sys->lambda : 0.0f;
}

bool regime_system_is_crisis(const RegimeSystem* sys)
{
    if (!sys) return false;
    HawkesTriggerState state = hawkes_integrator_get_state(&sys->hawkes);
    return (state == HAWKES_TRIG_ARMED || state == HAWKES_TRIG_FIRED);
}

const PGASOracleState* regime_system_get_oracle(const RegimeSystem* sys)
{
    return sys ? sys->oracle : NULL;
}

const HawkesIntegrator* regime_system_get_hawkes(const RegimeSystem* sys)
{
    return sys ? &sys->hawkes : NULL;
}

/*═══════════════════════════════════════════════════════════════════════════════
 * DIAGNOSTICS
 *═══════════════════════════════════════════════════════════════════════════════*/

void regime_system_print_diagnostics(const RegimeSystem* sys)
{
    if (!sys) return;
    
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║              REGIME SYSTEM DIAGNOSTICS                        ║\n");
    printf("╠═══════════════════════════════════════════════════════════════╣\n");
    printf("║ Regimes:          %d                                          \n", sys->K);
    printf("║ Total ticks:      %lld                                        \n", 
           (long long)sys->total_ticks);
    printf("║ Crisis ticks:     %lld (%.2f%%)                               \n",
           (long long)sys->crisis_ticks,
           sys->total_ticks > 0 ? 100.0 * sys->crisis_ticks / sys->total_ticks : 0.0);
    printf("╠═══════════════════════════════════════════════════════════════╣\n");
    printf("║ HOT SWAP STATISTICS                                           ║\n");
    printf("║   Swaps performed: %lld                                       \n",
           (long long)sys->swaps_performed);
    printf("║   Swaps vetoed:    %lld                                       \n",
           (long long)sys->swaps_vetoed);
    printf("╠═══════════════════════════════════════════════════════════════╣\n");
    printf("║ CURRENT STATE                                                 ║\n");
    printf("║   λ (forgetting):  %.4f                                       \n", sys->lambda);
    printf("║   Crisis mode:     %s                                         \n",
           regime_system_is_crisis(sys) ? "YES" : "NO");
    printf("╠═══════════════════════════════════════════════════════════════╣\n");
    printf("║ CURRENT Π                                                     ║\n");
    for (int i = 0; i < sys->K; i++) {
        printf("║   Row %d: ", i);
        for (int j = 0; j < sys->K; j++) {
            printf("%.3f ", sys->Pi[i * sys->K + j]);
        }
        printf("\n");
    }
    printf("╚═══════════════════════════════════════════════════════════════╝\n");
    
    /* Print sub-component diagnostics */
    printf("\n");
    hawkes_integrator_print_state(&sys->hawkes);
    
    if (sys->oracle) {
        pgas_oracle_print_diagnostics(sys->oracle);
    }
}
