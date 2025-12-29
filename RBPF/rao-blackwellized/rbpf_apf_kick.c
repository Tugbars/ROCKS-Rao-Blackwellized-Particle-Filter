/**
 * @file rbpf_apf_kick.c
 * @brief Auxiliary Particle Filter Look-Ahead Kick for Crisis Transitions
 *
 * PROBLEM:
 * Standard RBPF transition samples blindly from Π[old→new]. When Hawkes
 * fires, particles take 10-50 ticks to migrate to crisis regime via
 * resampling killing bad particles.
 *
 * SOLUTION:
 * APF Look-Ahead computes p(y_t | regime r) for each candidate regime
 * and weights transitions by this likelihood. Particles immediately
 * jump to regimes that explain the crisis observation.
 *
 * MATH:
 *   Standard:  P(r_new | r_old) = Π[r_old, r_new]
 *   APF Kick:  P(r_new | r_old, y) ∝ Π[r_old, r_new] × p(y | r_new)
 *
 * INTEGRATION:
 * In rbpf_ext_step(), replace:
 *   rbpf_ksc_transition(rbpf);
 * With:
 *   if (ext->hawkes.enabled && ext->hawkes.intensity > ext->hawkes.threshold)
 *       rbpf_ksc_transition_apf(rbpf, y);
 *   else
 *       rbpf_ksc_transition(rbpf);
 */

#include "rbpf_ksc.h"
#include <math.h>

/*═══════════════════════════════════════════════════════════════════════════
 * OMORI MIXTURE CONSTANTS
 *═══════════════════════════════════════════════════════════════════════════*/

#define APF_N_COMPONENTS 10

static const rbpf_real_t APF_KSC_PROB[APF_N_COMPONENTS] = {
    RBPF_REAL(0.00609), RBPF_REAL(0.04775), RBPF_REAL(0.13057), RBPF_REAL(0.20674),
    RBPF_REAL(0.22715), RBPF_REAL(0.18842), RBPF_REAL(0.12047), RBPF_REAL(0.05591),
    RBPF_REAL(0.01575), RBPF_REAL(0.00115)};

static const rbpf_real_t APF_KSC_MEAN[APF_N_COMPONENTS] = {
    RBPF_REAL(1.92677), RBPF_REAL(1.34744), RBPF_REAL(0.73504), RBPF_REAL(0.02266),
    RBPF_REAL(-0.85173), RBPF_REAL(-1.97278), RBPF_REAL(-3.46788), RBPF_REAL(-5.55246),
    RBPF_REAL(-8.68384), RBPF_REAL(-14.65000)};

static const rbpf_real_t APF_KSC_VAR[APF_N_COMPONENTS] = {
    RBPF_REAL(0.11265), RBPF_REAL(0.17788), RBPF_REAL(0.26768), RBPF_REAL(0.40611),
    RBPF_REAL(0.62699), RBPF_REAL(0.98583), RBPF_REAL(1.57469), RBPF_REAL(2.54498),
    RBPF_REAL(4.16591), RBPF_REAL(7.33342)};

/*═══════════════════════════════════════════════════════════════════════════
 * LOOK-AHEAD LIKELIHOOD: p(y | regime)
 *═══════════════════════════════════════════════════════════════════════════*/

static inline rbpf_real_t apf_lookahead_loglik(
    rbpf_real_t y,
    rbpf_real_t mu_particle,
    rbpf_real_t var_particle,
    const RBPF_RegimeParams *regime)
{
    const rbpf_real_t H = RBPF_REAL(2.0);
    const rbpf_real_t H2 = RBPF_REAL(4.0);
    const rbpf_real_t NEG_HALF = RBPF_REAL(-0.5);
    const rbpf_real_t LOG_2PI = RBPF_REAL(1.8378770664093453);

    /* One-step-ahead prediction under target regime */
    rbpf_real_t theta = regime->theta;
    rbpf_real_t omt = RBPF_REAL(1.0) - theta;
    rbpf_real_t mu_pred = omt * mu_particle + theta * regime->mu_vol;
    rbpf_real_t var_pred = omt * omt * var_particle + regime->q;

    if (var_pred < RBPF_REAL(0.01)) var_pred = RBPF_REAL(0.01);
    if (var_pred > RBPF_REAL(10.0)) var_pred = RBPF_REAL(10.0);

    /* Mixture likelihood with log-sum-exp */
    rbpf_real_t max_ll = RBPF_REAL(-1e30);
    rbpf_real_t log_liks[APF_N_COMPONENTS];

    for (int k = 0; k < APF_N_COMPONENTS; k++)
    {
        rbpf_real_t m_k = APF_KSC_MEAN[k];
        rbpf_real_t v2_k = APF_KSC_VAR[k];
        rbpf_real_t pi_k = APF_KSC_PROB[k];

        rbpf_real_t innov = y - (H * mu_pred + m_k);
        rbpf_real_t S = H2 * var_pred + v2_k;
        rbpf_real_t ll = NEG_HALF * (LOG_2PI + rbpf_log(S) + innov * innov / S)
                       + rbpf_log(pi_k);

        log_liks[k] = ll;
        if (ll > max_ll) max_ll = ll;
    }

    rbpf_real_t sum_exp = RBPF_REAL(0.0);
    for (int k = 0; k < APF_N_COMPONENTS; k++)
    {
        sum_exp += rbpf_exp(log_liks[k] - max_ll);
    }

    return max_ll + rbpf_log(sum_exp);
}

/*═══════════════════════════════════════════════════════════════════════════
 * APF TRANSITION: rbpf_ksc_transition_apf()
 *
 * Drop-in replacement for rbpf_ksc_transition() when crisis is active.
 * Weights transition probabilities by observation likelihood.
 *═══════════════════════════════════════════════════════════════════════════*/

void rbpf_ksc_transition_apf(RBPF_KSC *rbpf, rbpf_real_t y)
{
    const int n = rbpf->n_particles;
    const int n_regimes = rbpf->n_regimes;
    const RBPF_RegimeParams *params = rbpf->params;

    int *regime = rbpf->regime;
    rbpf_real_t *mu = rbpf->mu;
    rbpf_real_t *var = rbpf->var;
    rbpf_pcg32_t *rng = &rbpf->pcg[0];

    /*═══════════════════════════════════════════════════════════════════════
     * GET TRANSITION PROBABILITIES FROM LUT OR DIRICHLET
     *═══════════════════════════════════════════════════════════════════════*/
    rbpf_real_t trans_probs[RBPF_MAX_REGIMES][RBPF_MAX_REGIMES];

    if (rbpf->trans_prior_enabled)
    {
        const DirichletTransition *dt = &rbpf->trans_prior;
        for (int from = 0; from < n_regimes; from++)
        {
            for (int to = 0; to < n_regimes; to++)
            {
                trans_probs[from][to] = dt->prob[from][to];
            }
        }
    }
    else
    {
        /* Extract from LUT by counting entries */
        const uint8_t (*lut)[RBPF_LUT_SIZE] = rbpf_lut_acquire_read(&rbpf->trans_lut);
        const rbpf_real_t inv_lut_size = RBPF_REAL(1.0) / RBPF_LUT_SIZE;

        for (int from = 0; from < n_regimes; from++)
        {
            int counts[RBPF_MAX_REGIMES] = {0};
            for (int k = 0; k < RBPF_LUT_SIZE; k++)
            {
                counts[lut[from][k]]++;
            }
            for (int to = 0; to < n_regimes; to++)
            {
                trans_probs[from][to] = (rbpf_real_t)counts[to] * inv_lut_size;
            }
        }
    }

    /*═══════════════════════════════════════════════════════════════════════
     * APF-WEIGHTED TRANSITION FOR EACH PARTICLE
     *═══════════════════════════════════════════════════════════════════════*/
    rbpf_real_t log_liks[RBPF_MAX_REGIMES];
    rbpf_real_t weights[RBPF_MAX_REGIMES];
    rbpf_real_t cumsum[RBPF_MAX_REGIMES];

    for (int i = 0; i < n; i++)
    {
        int r_old = regime[i];
        rbpf_real_t mu_i = mu[i];
        rbpf_real_t var_i = var[i];

        /* Compute look-ahead log-likelihood for each candidate regime */
        rbpf_real_t max_ll = RBPF_REAL(-1e30);
        for (int r = 0; r < n_regimes; r++)
        {
            log_liks[r] = apf_lookahead_loglik(y, mu_i, var_i, &params[r]);
            if (log_liks[r] > max_ll) max_ll = log_liks[r];
        }

        /* Compute APF weights: Π[r_old, r] × exp(loglik[r] - max) */
        rbpf_real_t sum_w = RBPF_REAL(0.0);
        for (int r = 0; r < n_regimes; r++)
        {
            rbpf_real_t lik = rbpf_exp(log_liks[r] - max_ll);
            weights[r] = trans_probs[r_old][r] * lik;
            sum_w += weights[r];
        }

        /* Normalize and build cumsum */
        if (sum_w < RBPF_REAL(1e-30))
        {
            for (int r = 0; r < n_regimes; r++)
            {
                weights[r] = RBPF_REAL(1.0) / n_regimes;
            }
            sum_w = RBPF_REAL(1.0);
        }

        rbpf_real_t inv_sum = RBPF_REAL(1.0) / sum_w;
        cumsum[0] = weights[0] * inv_sum;
        for (int r = 1; r < n_regimes; r++)
        {
            cumsum[r] = cumsum[r - 1] + weights[r] * inv_sum;
        }

        /* Sample from weighted distribution */
        rbpf_real_t u = rbpf_pcg32_uniform(rng);
        int r_new = n_regimes - 1;
        for (int r = 0; r < n_regimes - 1; r++)
        {
            if (u < cumsum[r])
            {
                r_new = r;
                break;
            }
        }

        regime[i] = r_new;
    }
}