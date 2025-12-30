/**
 * @file rbpf_ksc_param_integration.h
 * @brief RBPF-KSC Extended: Full Integration Layer
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * ARCHITECTURE
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * RBPF_Extended wraps RBPF_KSC with:
 *   1. Storvik online parameter learning (μ_vol, σ_vol per regime)
 *   2. Hawkes self-excitation (jump-sensitive transitions via APF kick)
 *   3. Robust OCSN (11th mixture component for outliers)
 *   4. PARIS smoothed Storvik (fixed-lag backward smoother)
 *   5. Adaptive forgetting (regime-aware λ with P² circuit breaker)
 *   6. Transition learning (online Dirichlet updates)
 *   7. KL tempering (information-geometric weight normalization)
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * HEADER ORGANIZATION
 * ═══════════════════════════════════════════════════════════════════════════
 *
 *   rbpf_ext_types.h    Constants, enums, struct definitions
 *   rbpf_ext_api.h      All function declarations (documented)
 *
 * This header includes both for backward compatibility.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * SOURCE FILE ORGANIZATION
 * ═══════════════════════════════════════════════════════════════════════════
 *
 *   rbpf_ksc_param_integration.c   Core: create, destroy, init, step
 *   rbpf_ext_config.c              Configuration functions
 *   rbpf_ext_diagnostics.c         Getters and print functions
 *   rbpf_ext_hawkes.c              Hawkes + Robust OCSN + Presets
 *   rbpf_ext_smoothed_storvik.c    PARIS fixed-lag smoother
 *   rbpf_adaptive_forgetting.c     Adaptive λ with P² circuit breaker
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * QUICK START
 * ═══════════════════════════════════════════════════════════════════════════
 *
 *   // Create with Storvik learning
 *   RBPF_Extended *ext = rbpf_ext_create(1024, 4, RBPF_PARAM_STORVIK);
 *
 *   // Configure regimes (or use preset)
 *   rbpf_ext_apply_preset(ext, RBPF_PRESET_EQUITY_INDEX);
 *
 *   // Initialize
 *   rbpf_ext_init(ext, -4.6f, 0.5f);
 *
 *   // Process observations
 *   RBPF_KSC_Output output;
 *   for (int t = 0; t < n_obs; t++) {
 *       rbpf_ext_step(ext, returns[t], &output);
 *       printf("vol = %.4f, regime = %d\n", 
 *              output.vol_mean, output.smoothed_regime);
 *   }
 *
 *   // Cleanup
 *   rbpf_ext_destroy(ext);
 *
 * ═══════════════════════════════════════════════════════════════════════════
 */

#ifndef RBPF_KSC_PARAM_INTEGRATION_H
#define RBPF_KSC_PARAM_INTEGRATION_H

/* Type definitions: constants, enums, structs */
#include "rbpf_ext_types.h"

/* Function declarations */
#include "rbpf_ext_api.h"

#endif /* RBPF_KSC_PARAM_INTEGRATION_H */
