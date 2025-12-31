/**
 * @file rbpf_ext_crisis.c
 * @brief Crisis Mode: PGAS Veto + Emission λ Override
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * Minimal crisis coordination. PGAS handles Π learning.
 * ═══════════════════════════════════════════════════════════════════════════
 */

#include "rbpf_ext_crisis.h"
#include <stdio.h>
#include <inttypes.h>

/*═══════════════════════════════════════════════════════════════════════════
 * ENTER CRISIS
 *═══════════════════════════════════════════════════════════════════════════*/

void rbpf_ext_enter_crisis(RBPF_Extended *ext)
{
    if (!ext) return;
    if (ext->trans_mode == TRANS_MODE_CRISIS) return;  /* Already in crisis */

    ext->trans_mode = TRANS_MODE_CRISIS;
    ext->ticks_in_crisis_mode = 0;

    /* Fast emission λ for Storvik */
    ext->emission_lambda_override_active = 1;
    ext->emission_lambda_override = EMISSION_LAMBDA_CRISIS;

    ext->crisis_entries++;
}

/*═══════════════════════════════════════════════════════════════════════════
 * EXIT CRISIS
 *═══════════════════════════════════════════════════════════════════════════*/

void rbpf_ext_exit_crisis(RBPF_Extended *ext)
{
    if (!ext) return;
    if (ext->trans_mode != TRANS_MODE_CRISIS) return;  /* Not in crisis */

    ext->trans_mode = TRANS_MODE_NORMAL;
    ext->ticks_in_crisis_mode = 0;

    /* Return to adaptive emission λ */
    ext->emission_lambda_override_active = 0;

    ext->crisis_exits++;
}

/*═══════════════════════════════════════════════════════════════════════════
 * PGAS VETO
 *═══════════════════════════════════════════════════════════════════════════*/

int rbpf_ext_should_veto_pgas(const RBPF_Extended *ext)
{
    if (!ext) return 0;

    /* Veto during crisis until PGAS has seen enough crisis data */
    if (ext->trans_mode == TRANS_MODE_CRISIS)
    {
        return (ext->ticks_in_crisis_mode < CRISIS_PGAS_VETO_TICKS);
    }

    return 0;
}

/*═══════════════════════════════════════════════════════════════════════════
 * QUERIES
 *═══════════════════════════════════════════════════════════════════════════*/

int rbpf_ext_is_in_crisis(const RBPF_Extended *ext)
{
    return (ext && ext->trans_mode == TRANS_MODE_CRISIS);
}

int rbpf_ext_get_crisis_ticks(const RBPF_Extended *ext)
{
    if (!ext) return 0;
    if (ext->trans_mode != TRANS_MODE_CRISIS) return 0;
    return ext->ticks_in_crisis_mode;
}

/*═══════════════════════════════════════════════════════════════════════════
 * DIAGNOSTICS
 *═══════════════════════════════════════════════════════════════════════════*/

void rbpf_ext_print_crisis_state(const RBPF_Extended *ext)
{
    if (!ext) return;

    printf("\n");
    printf("┌─────────────────────────────────────────────────────────────┐\n");
    printf("│                    CRISIS MODE STATE                        │\n");
    printf("├─────────────────────────────────────────────────────────────┤\n");
    printf("│  Mode:              %-8s                                │\n",
           ext->trans_mode == TRANS_MODE_CRISIS ? "CRISIS" : "NORMAL");
    
    if (ext->trans_mode == TRANS_MODE_CRISIS)
    {
        printf("│  Ticks in crisis:   %-6d                                  │\n",
               ext->ticks_in_crisis_mode);
        printf("│  PGAS veto:         %-3s (until tick %d)                    │\n",
               ext->ticks_in_crisis_mode < CRISIS_PGAS_VETO_TICKS ? "YES" : "NO",
               CRISIS_PGAS_VETO_TICKS);
    }
    
    printf("│  Emission λ:        %-8s                                │\n",
           ext->emission_lambda_override_active ? "0.95 (fast)" : "adaptive");
    printf("├─────────────────────────────────────────────────────────────┤\n");
    printf("│  Statistics:                                                │\n");
    printf("│    Crisis entries:  %-6" PRIu64 "                                │\n",
           ext->crisis_entries);
    printf("│    Crisis exits:    %-6" PRIu64 "                                │\n",
           ext->crisis_exits);
    printf("└─────────────────────────────────────────────────────────────┘\n");
}