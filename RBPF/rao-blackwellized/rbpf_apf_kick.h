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
 * Integration:
 *   - Extended layer calls rbpf_ksc_step_apf() with crisis_active flag
 *   - When crisis_active=1, APF kick activates
 *   - When crisis_active=0, falls back to fast LUT transition
 */

#ifndef RBPF_APF_KICK_H
#define RBPF_APF_KICK_H

#include "rbpf_ksc.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief APF-guided regime transition using observation look-ahead
 *
 * For each particle, computes likelihood under each candidate regime and
 * samples transitions weighted by: Π[r_old, r] × p(y | r)
 *
 * @param rbpf          RBPF state
 * @param y_lookahead   Current observation y = log(r²) for look-ahead
 */
void rbpf_ksc_transition_apf(RBPF_KSC *rbpf, rbpf_real_t y_lookahead);

/**
 * @brief Hybrid transition: LUT (normal) or APF (crisis)
 *
 * @param rbpf           RBPF state
 * @param y_next         Observation for look-ahead (0 if not available)
 * @param crisis_active  1 if Hawkes detected crisis, 0 for normal operation
 */
void rbpf_ksc_transition_hybrid(RBPF_KSC *rbpf, rbpf_real_t y_next, int crisis_active);

/**
 * @brief Full RBPF step with APF support
 *
 * Replaces rbpf_ksc_step() when APF crisis detection is enabled.
 * The Extended layer passes the crisis_active flag from Hawkes detector.
 *
 * @param rbpf           RBPF state
 * @param obs            Raw observation (return)
 * @param crisis_active  1 if Hawkes triggered crisis, 0 otherwise
 * @param output         Output structure (includes apf_active flag)
 */
void rbpf_ksc_step_apf(RBPF_KSC *rbpf, rbpf_real_t obs, int crisis_active,
                       RBPF_KSC_Output *output);

/**
 * @brief Weight correction for proper importance sampling (optional)
 *
 * APF changes the proposal distribution, requiring weight correction for
 * unbiased estimates. In practice, simplified APF (no correction) works
 * well for regime detection where speed matters more than exactness.
 *
 * @param rbpf           RBPF state
 * @param y_lookahead    Observation used for look-ahead
 */
void rbpf_apf_weight_correction(RBPF_KSC *rbpf, rbpf_real_t y_lookahead);

#ifdef __cplusplus
}
#endif

#endif /* RBPF_APF_KICK_H */
