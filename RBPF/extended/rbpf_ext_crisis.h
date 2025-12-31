/**
 * @file rbpf_ext_crisis.h
 * @brief Crisis Mode API: PGAS Veto + Emission λ Override
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * Minimal crisis mode coordination:
 *
 *   1. PGAS Veto: Block PGAS injection until it has seen crisis data
 *   2. Emission λ: Override Storvik to fast forgetting (0.95)
 *
 * PGAS handles Π learning. We just coordinate timing.
 *
 * State Machine:
 *   NORMAL ──[enter_crisis]──→ CRISIS ──[exit_crisis]──→ NORMAL
 *
 * ═══════════════════════════════════════════════════════════════════════════
 */

#ifndef RBPF_EXT_CRISIS_H
#define RBPF_EXT_CRISIS_H

#include "rbpf_ext_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/*═══════════════════════════════════════════════════════════════════════════
 * CONSTANTS
 *═══════════════════════════════════════════════════════════════════════════*/

#define EMISSION_LAMBDA_CRISIS    0.95f   /**< Fast forgetting during crisis */
#define CRISIS_PGAS_VETO_TICKS    500     /**< Ticks to veto PGAS after crisis entry */

/*═══════════════════════════════════════════════════════════════════════════
 * STATE TRANSITIONS
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Enter crisis mode (NORMAL → CRISIS)
 *
 * Effects:
 *   - PGAS injection vetoed for 500 ticks
 *   - Storvik emission λ overridden to 0.95
 *
 * Call when: CrisisDetector confirms crisis (SR exceeds threshold)
 *
 * @param ext  RBPF Extended handle
 */
void rbpf_ext_enter_crisis(RBPF_Extended *ext);

/**
 * @brief Exit crisis mode (CRISIS → NORMAL)
 *
 * Effects:
 *   - PGAS injection allowed again
 *   - Storvik emission λ returns to adaptive
 *
 * Call when: CrisisDetector confirms crisis is over
 *
 * @param ext  RBPF Extended handle
 */
void rbpf_ext_exit_crisis(RBPF_Extended *ext);

/*═══════════════════════════════════════════════════════════════════════════
 * QUERIES
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Check if PGAS injection should be vetoed
 *
 * Returns 1 if in CRISIS mode and ticks < CRISIS_PGAS_VETO_TICKS.
 * PGAS needs time to see crisis data before its Π is valid.
 *
 * Usage:
 *   if (pgas_has_new_Pi && !rbpf_ext_should_veto_pgas(ext)) {
 *       rbpf_ksc_update_transition_matrix(rbpf, pgas_Pi);
 *   }
 *
 * @param ext  RBPF Extended handle
 * @return 1 if PGAS should be vetoed, 0 if allowed
 */
int rbpf_ext_should_veto_pgas(const RBPF_Extended *ext);

/**
 * @brief Check if currently in crisis mode
 *
 * @param ext  RBPF Extended handle
 * @return 1 if in crisis, 0 if normal
 */
int rbpf_ext_is_in_crisis(const RBPF_Extended *ext);

/**
 * @brief Get ticks since crisis entry
 *
 * @param ext  RBPF Extended handle
 * @return Ticks since entering crisis mode (0 if not in crisis)
 */
int rbpf_ext_get_crisis_ticks(const RBPF_Extended *ext);

/*═══════════════════════════════════════════════════════════════════════════
 * DIAGNOSTICS
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Print crisis mode state
 *
 * @param ext  RBPF Extended handle
 */
void rbpf_ext_print_crisis_state(const RBPF_Extended *ext);

#ifdef __cplusplus
}
#endif

#endif /* RBPF_EXT_CRISIS_H */