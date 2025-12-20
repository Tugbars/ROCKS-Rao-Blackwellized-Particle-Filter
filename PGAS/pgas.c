/**
 * @file pgas.c
 * @brief Particle Gibbs with Ancestor Sampling (PGAS) + PARIS Smoother
 *
 * Implementation of:
 *   - Conditional SMC (CSMC) with fixed reference trajectory
 *   - Ancestor Sampling for improved mixing
 *   - PARIS backward kernel for O(N) smoothing
 */

#include "pgas.h"
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <float.h>

/*═══════════════════════════════════════════════════════════════════════════════
 * CONSTANTS AND HELPERS
 *═══════════════════════════════════════════════════════════════════════════════*/

#define PGAS_EPS 1e-10
#define PGAS_LOG_EPS -23.0259

static inline double maxd(double a, double b) { return a > b ? a : b; }
static inline double mind(double a, double b) { return a < b ? a : b; }

/*═══════════════════════════════════════════════════════════════════════════════
 * RNG (xoroshiro128+)
 *═══════════════════════════════════════════════════════════════════════════════*/

static inline uint64_t rotl(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

static inline uint64_t xoro_next(uint64_t *s) {
    const uint64_t s0 = s[0];
    uint64_t s1 = s[1];
    const uint64_t result = s0 + s1;
    s1 ^= s0;
    s[0] = rotl(s0, 24) ^ s1 ^ (s1 << 16);
    s[1] = rotl(s1, 37);
    return result;
}

static inline double rand_uniform(uint64_t *s) {
    return (xoro_next(s) >> 11) * 0x1.0p-53;
}

static inline double rand_normal(uint64_t *s) {
    /* Box-Muller */
    double u1 = rand_uniform(s);
    double u2 = rand_uniform(s);
    if (u1 < 1e-10) u1 = 1e-10;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

static int sample_categorical(const double *weights, int n, uint64_t *rng) {
    double u = rand_uniform(rng);
    double cumsum = 0.0;
    for (int i = 0; i < n - 1; i++) {
        cumsum += weights[i];
        if (u < cumsum) return i;
    }
    return n - 1;
}

static int sample_categorical_log(const double *log_weights, int n, uint64_t *rng) {
    /* Find max for numerical stability */
    double max_log = log_weights[0];
    for (int i = 1; i < n; i++) {
        if (log_weights[i] > max_log) max_log = log_weights[i];
    }
    
    /* Convert to normalized weights */
    double weights[PGAS_MAX_PARTICLES];
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        weights[i] = exp(log_weights[i] - max_log);
        sum += weights[i];
    }
    
    /* Normalize and sample */
    double inv_sum = 1.0 / sum;
    for (int i = 0; i < n; i++) {
        weights[i] *= inv_sum;
    }
    
    return sample_categorical(weights, n, rng);
}

/*═══════════════════════════════════════════════════════════════════════════════
 * PROBABILITY COMPUTATIONS
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * Log transition probability: log P(s_t = j | s_{t-1} = i)
 */
static inline double log_trans_prob(const PGASModel *m, int from, int to) {
    double p = m->trans[from * m->K + to];
    return (p > PGAS_EPS) ? log(p) : PGAS_LOG_EPS;
}

/**
 * Log emission probability: log P(y_t | s_t, h_t)
 * Stochastic volatility: y_t ~ N(0, exp(h_t))
 */
static inline double log_emission(double y, double h) {
    /* y ~ N(0, exp(h))
     * log p(y|h) = -0.5 * log(2π) - 0.5 * h - 0.5 * y² * exp(-h) */
    double var = exp(h);
    return -0.5 * (log(2.0 * M_PI) + h + y * y / var);
}

/**
 * Log of h transition: h_t | h_{t-1}, s_t ~ N(μ_k + φ(h_{t-1} - μ_k), σ_h²)
 */
static inline double log_h_trans(const PGASModel *m, double h_prev, double h_curr, int regime) {
    double mu = m->mu_vol[regime];
    double mean = mu + m->phi * (h_prev - mu);
    double diff = h_curr - mean;
    double var = m->sigma_h * m->sigma_h;
    return -0.5 * (log(2.0 * M_PI * var) + diff * diff / var);
}

/**
 * Sample h_t | h_{t-1}, s_t
 */
static inline double sample_h(const PGASModel *m, double h_prev, int regime, uint64_t *rng) {
    double mu = m->mu_vol[regime];
    double mean = mu + m->phi * (h_prev - mu);
    return mean + m->sigma_h * rand_normal(rng);
}

/**
 * Sample s_t | s_{t-1} from transition matrix
 */
static inline int sample_regime(const PGASModel *m, int prev_regime, uint64_t *rng) {
    return sample_categorical(&m->trans[prev_regime * m->K], m->K, rng);
}

/*═══════════════════════════════════════════════════════════════════════════════
 * INITIALIZATION
 *═══════════════════════════════════════════════════════════════════════════════*/

void pgas_init(PGASState *pgas, int N, int K, uint64_t seed)
{
    if (!pgas) return;
    
    memset(pgas, 0, sizeof(*pgas));
    
    pgas->N = (N > PGAS_MAX_PARTICLES) ? PGAS_MAX_PARTICLES : N;
    pgas->K = (K > PGAS_MAX_REGIMES) ? PGAS_MAX_REGIMES : K;
    pgas->T = 0;
    pgas->ref_idx = pgas->N - 1;  /* Last particle is reference */
    
    /* Initialize RNG */
    pgas->rng_state[0] = seed;
    pgas->rng_state[1] = seed ^ 0x9E3779B97F4A7C15ULL;
    
    /* Initialize model with uniform transitions */
    pgas->model.K = K;
    double unif = 1.0 / K;
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < K; j++) {
            pgas->model.trans[i * K + j] = unif;
        }
        pgas->model.mu_vol[i] = -1.0 + 0.5 * i;  /* Spread across vol levels */
        pgas->model.sigma_vol[i] = 0.3;
    }
    pgas->model.phi = 0.97;
    pgas->model.sigma_h = 0.15;
    
    /* Copy to prior */
    memcpy(&pgas->model_prior, &pgas->model, sizeof(PGASModel));
}

void pgas_set_model(PGASState *pgas,
                    const double *trans,
                    const double *mu_vol,
                    const double *sigma_vol,
                    double phi,
                    double sigma_h)
{
    if (!pgas) return;
    
    int K = pgas->K;
    
    if (trans) {
        memcpy(pgas->model.trans, trans, K * K * sizeof(double));
    }
    if (mu_vol) {
        memcpy(pgas->model.mu_vol, mu_vol, K * sizeof(double));
    }
    if (sigma_vol) {
        memcpy(pgas->model.sigma_vol, sigma_vol, K * sizeof(double));
    }
    pgas->model.phi = phi;
    pgas->model.sigma_h = sigma_h;
}

void pgas_set_reference(PGASState *pgas,
                        const int *regimes,
                        const double *h,
                        int T)
{
    if (!pgas || !regimes || !h) return;
    
    pgas->T = (T > PGAS_MAX_TIME) ? PGAS_MAX_TIME : T;
    
    memcpy(pgas->ref_regimes, regimes, pgas->T * sizeof(int));
    memcpy(pgas->ref_h, h, pgas->T * sizeof(double));
    
    /* Initialize ancestors to self (will be updated by AS) */
    for (int t = 0; t < pgas->T; t++) {
        pgas->ref_ancestors[t] = pgas->ref_idx;
    }
}

void pgas_load_observations(PGASState *pgas, const BufferSnapshot *snapshot)
{
    if (!pgas || !snapshot) return;
    
    pgas->T = (snapshot->count > PGAS_MAX_TIME) ? PGAS_MAX_TIME : snapshot->count;
    
    memcpy(pgas->observations, snapshot->observations, pgas->T * sizeof(double));
}

/*═══════════════════════════════════════════════════════════════════════════════
 * CSMC FORWARD PASS
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * Initialize particles at t=0
 */
static void csmc_init(PGASState *pgas)
{
    const int N = pgas->N;
    const int K = pgas->K;
    const int ref_idx = pgas->ref_idx;
    
    /* Initialize non-reference particles from prior */
    for (int n = 0; n < N; n++) {
        if (n == ref_idx) {
            /* Reference particle: use conditioned trajectory */
            pgas->regimes[0][n] = pgas->ref_regimes[0];
            pgas->h[0][n] = pgas->ref_h[0];
        } else {
            /* Sample from prior */
            pgas->regimes[0][n] = (int)(rand_uniform(pgas->rng_state) * K);
            pgas->h[0][n] = pgas->model.mu_vol[pgas->regimes[0][n]] + 
                            pgas->model.sigma_h * rand_normal(pgas->rng_state);
        }
        pgas->ancestors[0][n] = n;  /* Self at t=0 */
    }
    
    /* Compute initial weights */
    double y0 = pgas->observations[0];
    double max_log = -DBL_MAX;
    
    for (int n = 0; n < N; n++) {
        pgas->log_weights[0][n] = log_emission(y0, pgas->h[0][n]);
        if (pgas->log_weights[0][n] > max_log) {
            max_log = pgas->log_weights[0][n];
        }
    }
    
    /* Normalize */
    double sum = 0.0;
    for (int n = 0; n < N; n++) {
        pgas->weights[0][n] = exp(pgas->log_weights[0][n] - max_log);
        sum += pgas->weights[0][n];
    }
    double inv_sum = 1.0 / sum;
    for (int n = 0; n < N; n++) {
        pgas->weights[0][n] *= inv_sum;
    }
}

/**
 * Resample particles (except reference)
 */
static void csmc_resample(PGASState *pgas, int t)
{
    const int N = pgas->N;
    const int ref_idx = pgas->ref_idx;
    
    /* Multinomial resampling for non-reference particles */
    for (int n = 0; n < N; n++) {
        if (n == ref_idx) {
            /* Reference keeps its ancestor (will be updated by AS) */
            pgas->ancestors[t][n] = pgas->ref_ancestors[t];
        } else {
            /* Sample ancestor from weights at t-1 */
            pgas->ancestors[t][n] = sample_categorical(pgas->weights[t-1], N, pgas->rng_state);
        }
    }
}

/**
 * Propagate particles forward
 */
static void csmc_propagate(PGASState *pgas, int t)
{
    const int N = pgas->N;
    const int ref_idx = pgas->ref_idx;
    const PGASModel *m = &pgas->model;
    
    for (int n = 0; n < N; n++) {
        int anc = pgas->ancestors[t][n];
        
        if (n == ref_idx) {
            /* Reference particle: use conditioned trajectory */
            pgas->regimes[t][n] = pgas->ref_regimes[t];
            pgas->h[t][n] = pgas->ref_h[t];
        } else {
            /* Sample regime transition */
            int prev_regime = pgas->regimes[t-1][anc];
            pgas->regimes[t][n] = sample_regime(m, prev_regime, pgas->rng_state);
            
            /* Sample h transition */
            double prev_h = pgas->h[t-1][anc];
            pgas->h[t][n] = sample_h(m, prev_h, pgas->regimes[t][n], pgas->rng_state);
        }
    }
}

/**
 * Compute weights after propagation
 */
static void csmc_weight(PGASState *pgas, int t)
{
    const int N = pgas->N;
    double y = pgas->observations[t];
    double max_log = -DBL_MAX;
    
    for (int n = 0; n < N; n++) {
        pgas->log_weights[t][n] = log_emission(y, pgas->h[t][n]);
        if (pgas->log_weights[t][n] > max_log) {
            max_log = pgas->log_weights[t][n];
        }
    }
    
    /* Normalize */
    double sum = 0.0;
    for (int n = 0; n < N; n++) {
        pgas->weights[t][n] = exp(pgas->log_weights[t][n] - max_log);
        sum += pgas->weights[t][n];
    }
    double inv_sum = 1.0 / sum;
    for (int n = 0; n < N; n++) {
        pgas->weights[t][n] *= inv_sum;
    }
}

/**
 * Ancestor Sampling: try to find better ancestor for reference trajectory
 *
 * This is the key innovation that makes PGAS mix.
 * Instead of keeping the reference's ancestor fixed, we sample a new one
 * from the current particle cloud weighted by their ability to "explain"
 * the reference's future path.
 */
static int ancestor_sample(PGASState *pgas, int t)
{
    const int N = pgas->N;
    const PGASModel *m = &pgas->model;
    
    /* Reference state at time t */
    int ref_regime = pgas->ref_regimes[t];
    double ref_h = pgas->ref_h[t];
    
    /* Compute AS weights for each potential ancestor at t-1 */
    double as_log_weights[PGAS_MAX_PARTICLES];
    double max_log = -DBL_MAX;
    
    for (int n = 0; n < N; n++) {
        int prev_regime = pgas->regimes[t-1][n];
        double prev_h = pgas->h[t-1][n];
        
        /* AS weight = w_{t-1}^n × P(s_t^ref | s_{t-1}^n) × P(h_t^ref | h_{t-1}^n, s_t^ref) */
        double log_w = log(pgas->weights[t-1][n] + PGAS_EPS);
        log_w += log_trans_prob(m, prev_regime, ref_regime);
        log_w += log_h_trans(m, prev_h, ref_h, ref_regime);
        
        as_log_weights[n] = log_w;
        if (log_w > max_log) max_log = log_w;
    }
    
    /* Sample new ancestor */
    return sample_categorical_log(as_log_weights, N, pgas->rng_state);
}

/*═══════════════════════════════════════════════════════════════════════════════
 * MAIN CSMC SWEEP
 *═══════════════════════════════════════════════════════════════════════════════*/

double pgas_csmc_sweep(PGASState *pgas)
{
    if (!pgas || pgas->T < 2) return 0.0;
    
    const int T = pgas->T;
    const int ref_idx = pgas->ref_idx;
    
    /* Reset mixing counters */
    pgas->ancestor_proposals = 0;
    pgas->ancestor_accepts = 0;
    
    /* Initialize at t=0 */
    csmc_init(pgas);
    
    /* Forward pass with ancestor sampling */
    for (int t = 1; t < T; t++) {
        /* Resample (except reference) */
        csmc_resample(pgas, t);
        
        /* Ancestor Sampling for reference trajectory */
        int old_ancestor = pgas->ref_ancestors[t];
        int new_ancestor = ancestor_sample(pgas, t);
        
        pgas->ancestor_proposals++;
        if (new_ancestor != old_ancestor) {
            pgas->ancestor_accepts++;
            pgas->ref_ancestors[t] = new_ancestor;
            pgas->ancestors[t][ref_idx] = new_ancestor;
        }
        
        /* Propagate */
        csmc_propagate(pgas, t);
        
        /* Weight */
        csmc_weight(pgas, t);
    }
    
    /* Update reference trajectory from sampled path */
    /* (Sample one trajectory from final weights, update ref) */
    int final_idx = sample_categorical(pgas->weights[T-1], pgas->N, pgas->rng_state);
    
    /* Trace back through ancestors to update reference */
    int idx = final_idx;
    for (int t = T - 1; t >= 0; t--) {
        pgas->ref_regimes[t] = pgas->regimes[t][idx];
        pgas->ref_h[t] = pgas->h[t][idx];
        if (t > 0) {
            idx = pgas->ancestors[t][idx];
        }
    }
    
    /* Compute acceptance rate */
    pgas->acceptance_rate = (pgas->ancestor_proposals > 0) ?
        (double)pgas->ancestor_accepts / pgas->ancestor_proposals : 0.0;
    
    pgas->current_sweep++;
    pgas->total_sweeps++;
    
    return pgas->acceptance_rate;
}

/*═══════════════════════════════════════════════════════════════════════════════
 * ADAPTIVE PGAS
 *═══════════════════════════════════════════════════════════════════════════════*/

PGASResult pgas_run_adaptive(PGASState *pgas)
{
    if (!pgas || pgas->T < 2) return PGAS_INVALID_INPUT;
    
    pgas->current_sweep = 0;
    
    /* Run minimum sweeps */
    for (int s = 0; s < PGAS_MIN_SWEEPS; s++) {
        pgas_csmc_sweep(pgas);
    }
    
    /* Check if already converged */
    if (pgas->acceptance_rate >= PGAS_TARGET_ACCEPTANCE) {
        return PGAS_SUCCESS;
    }
    
    /* Continue until target or max */
    while (pgas->current_sweep < PGAS_MAX_SWEEPS) {
        pgas_csmc_sweep(pgas);
        
        if (pgas->acceptance_rate >= PGAS_TARGET_ACCEPTANCE) {
            return PGAS_SUCCESS;
        }
    }
    
    /* Check abort condition */
    if (pgas->acceptance_rate < PGAS_ABORT_ACCEPTANCE) {
        return PGAS_FAILED_TO_MIX;
    }
    
    /* Between abort and target: still mixing but usable */
    return PGAS_STILL_MIXING;
}

bool pgas_has_converged(const PGASState *pgas)
{
    return pgas && pgas->acceptance_rate >= PGAS_TARGET_ACCEPTANCE;
}

/*═══════════════════════════════════════════════════════════════════════════════
 * PARIS BACKWARD SMOOTHING
 *═══════════════════════════════════════════════════════════════════════════════*/

void paris_backward_smooth(PGASState *pgas)
{
    if (!pgas || pgas->T < 2) return;
    
    const int N = pgas->N;
    const int T = pgas->T;
    const PGASModel *m = &pgas->model;
    
    /* Initialize at final time: smoothed = filtering indices */
    for (int n = 0; n < N; n++) {
        pgas->smoothed_ancestors[T-1][n] = n;
    }
    
    /* Backward pass */
    for (int t = T - 2; t >= 0; t--) {
        for (int n = 0; n < N; n++) {
            /* Current smoothed index at t+1 */
            int idx_next = pgas->smoothed_ancestors[t+1][n];
            int regime_next = pgas->regimes[t+1][idx_next];
            double h_next = pgas->h[t+1][idx_next];
            
            /* Compute backward kernel weights */
            double bw_log_weights[PGAS_MAX_PARTICLES];
            double max_log = -DBL_MAX;
            
            for (int m_idx = 0; m_idx < N; m_idx++) {
                int regime_m = pgas->regimes[t][m_idx];
                double h_m = pgas->h[t][m_idx];
                
                /* Backward weight ∝ w_t^m × P(s_{t+1} | s_t^m) × P(h_{t+1} | h_t^m, s_{t+1}) */
                double log_w = log(pgas->weights[t][m_idx] + PGAS_EPS);
                log_w += log_trans_prob(m, regime_m, regime_next);
                log_w += log_h_trans(m, h_m, h_next, regime_next);
                
                bw_log_weights[m_idx] = log_w;
                if (log_w > max_log) max_log = log_w;
            }
            
            /* Sample from backward kernel */
            pgas->smoothed_ancestors[t][n] = sample_categorical_log(bw_log_weights, N, pgas->rng_state);
        }
    }
    
    pgas->smoothing_done = true;
}

void paris_get_smoothed_particles(const PGASState *pgas,
                                   int t,
                                   int *regimes,
                                   double *h,
                                   double *weights)
{
    if (!pgas || !pgas->smoothing_done || t < 0 || t >= pgas->T) return;
    
    const int N = pgas->N;
    double unif_weight = 1.0 / N;
    
    for (int n = 0; n < N; n++) {
        int idx = pgas->smoothed_ancestors[t][n];
        if (regimes) regimes[n] = pgas->regimes[t][idx];
        if (h) h[n] = pgas->h[t][idx];
        if (weights) weights[n] = unif_weight;  /* Uniform after smoothing */
    }
}

/*═══════════════════════════════════════════════════════════════════════════════
 * LIFEBOAT PACKET GENERATION
 *═══════════════════════════════════════════════════════════════════════════════*/

void pgas_generate_lifeboat(const PGASState *pgas, LifeboatPacket *packet)
{
    if (!pgas || !packet) return;
    
    const int N = pgas->N;
    const int K = pgas->K;
    const int T = pgas->T;
    
    /* Copy model parameters */
    packet->K = K;
    memcpy(packet->trans, pgas->model.trans, K * K * sizeof(double));
    memcpy(packet->mu_vol, pgas->model.mu_vol, K * sizeof(double));
    memcpy(packet->sigma_vol, pgas->model.sigma_vol, K * sizeof(double));
    packet->phi = pgas->model.phi;
    packet->sigma_h = pgas->model.sigma_h;
    
    /* Extract smoothed particles at final time */
    packet->n_particles = N;
    paris_get_smoothed_particles(pgas, T - 1,
                                  packet->final_regimes,
                                  packet->final_h,
                                  packet->final_weights);
    
    /* Diagnostics */
    packet->ancestor_acceptance = pgas->acceptance_rate;
    packet->sweeps_used = pgas->current_sweep;
    packet->buffer_len = T;
}

bool lifeboat_validate(const LifeboatPacket *packet)
{
    if (!packet) return false;
    
    /* Check acceptance rate */
    if (packet->ancestor_acceptance < PGAS_ABORT_ACCEPTANCE) {
        return false;
    }
    
    /* Check transition matrix rows sum to 1 */
    const int K = packet->K;
    for (int i = 0; i < K; i++) {
        double sum = 0.0;
        for (int j = 0; j < K; j++) {
            double p = packet->trans[i * K + j];
            if (p < 0.0 || p > 1.0) return false;
            sum += p;
        }
        if (fabs(sum - 1.0) > 1e-6) return false;
    }
    
    /* Check particle weights sum to 1 */
    double wsum = 0.0;
    for (int n = 0; n < packet->n_particles; n++) {
        if (packet->final_weights[n] < 0.0) return false;
        wsum += packet->final_weights[n];
    }
    if (fabs(wsum - 1.0) > 1e-6) return false;
    
    return true;
}

/*═══════════════════════════════════════════════════════════════════════════════
 * DIAGNOSTICS
 *═══════════════════════════════════════════════════════════════════════════════*/

double pgas_get_acceptance_rate(const PGASState *pgas)
{
    return pgas ? pgas->acceptance_rate : 0.0;
}

int pgas_get_sweep_count(const PGASState *pgas)
{
    return pgas ? pgas->current_sweep : 0;
}

double pgas_get_ess(const PGASState *pgas, int t)
{
    if (!pgas || t < 0 || t >= pgas->T) return 0.0;
    
    const int N = pgas->N;
    double sum_sq = 0.0;
    
    for (int n = 0; n < N; n++) {
        double w = pgas->weights[t][n];
        sum_sq += w * w;
    }
    
    return (sum_sq > PGAS_EPS) ? 1.0 / sum_sq : 0.0;
}

void pgas_print_diagnostics(const PGASState *pgas)
{
    if (!pgas) return;
    
    printf("═══════════════════════════════════════════════════════════\n");
    printf("PGAS DIAGNOSTICS\n");
    printf("═══════════════════════════════════════════════════════════\n");
    printf("Particles:          %d\n", pgas->N);
    printf("Regimes:            %d\n", pgas->K);
    printf("Buffer length:      %d\n", pgas->T);
    printf("Sweeps completed:   %d\n", pgas->current_sweep);
    printf("Ancestor accepts:   %d / %d\n", pgas->ancestor_accepts, pgas->ancestor_proposals);
    printf("Acceptance rate:    %.3f %s\n", pgas->acceptance_rate,
           pgas->acceptance_rate >= PGAS_TARGET_ACCEPTANCE ? "(CONVERGED)" :
           pgas->acceptance_rate >= PGAS_ABORT_ACCEPTANCE ? "(MIXING)" : "(STUCK)");
    printf("Smoothing done:     %s\n", pgas->smoothing_done ? "Yes" : "No");
    
    /* ESS at final time */
    printf("Final ESS:          %.1f / %d\n", pgas_get_ess(pgas, pgas->T - 1), pgas->N);
    printf("═══════════════════════════════════════════════════════════\n");
}
