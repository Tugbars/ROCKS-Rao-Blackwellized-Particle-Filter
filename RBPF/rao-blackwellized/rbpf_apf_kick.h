/**
 * @file rbpf_apf_kick.h
 * @brief Auxiliary Particle Filter Look-Ahead Kick for Crisis Transitions
 *
 * When Hawkes intensity spikes, the standard blind transition from Π is too
 * slow - particles take 10-50 ticks to migrate to crisis regime via resampling.
 *
 * APF Kick uses the observation likelihood to guide transitions immediately:
 *   P(r_new | r_old, y) ∝ Π[r_old, r_new] × p(y | r_new)
 *
 * Integration in rbpf_ext_step():
 *
 *   if (ext->hawkes.enabled && ext->hawkes.intensity > ext->hawkes.threshold)
 *       rbpf_ksc_transition_apf(rbpf, y);
 *   else
 *       rbpf_ksc_transition(rbpf);
 */

#ifndef RBPF_APF_KICK_H
#define RBPF_APF_KICK_H

#include "rbpf_ksc.h"

#ifdef __cplusplus
extern "C"
{
#endif

    /**
     * @brief APF-guided regime transition using observation look-ahead
     *
     * Drop-in replacement for rbpf_ksc_transition() during crisis.
     * Weights transition probabilities by p(y | regime).
     *
     * @param rbpf  RBPF state
     * @param y     Current observation y = log(r²)
     */
    void rbpf_ksc_transition_apf(RBPF_KSC *rbpf, rbpf_real_t y);

#ifdef __cplusplus
}
#endif

#endif /* RBPF_APF_KICK_H */