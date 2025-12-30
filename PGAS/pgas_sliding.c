/**
 * @file pgas_sliding.c
 * @brief Sliding window wrapper for PGAS-MKL
 *
 * Implementation notes:
 *
 * RING BUFFER:
 *   - Power-of-2 capacity for fast modulo (bitwise AND)
 *   - Single producer (tick loop), single consumer (PGAS loop)
 *   - Lock-free using atomic write index
 *
 * WARM START:
 *   After each sweep, we save the reference trajectory. When sliding:
 *   1. Shift ref[S..T-1] → ref[0..T-1-S]  (reuse old portion)
 *   2. Propagate ref[T-S..T-1] forward from ref[T-1-S] using model
 *
 *   This provides PGAS with a good starting trajectory, dramatically
 *   improving mixing (acceptance rate ~30-40% vs ~5-10% cold start).
 *
 * THREAD SAFETY:
 *   - pgas_sliding_push(): Called from tick loop, atomic write
 *   - All other functions: Called from PGAS loop only
 */

#include "pgas_sliding.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#include <mkl.h>
#include <mkl_vsl.h>

/*═══════════════════════════════════════════════════════════════════════════════
 * HELPERS
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * Round up to next power of 2
 */
static int next_power_of_2(int n)
{
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return n + 1;
}

/* Forward declaration for vectorized reference propagation */
static void propagate_reference_forward_vectorized(PGASSlidingState *state,
                                                   int start_t, int end_t);

/*═══════════════════════════════════════════════════════════════════════════════
 * LIFECYCLE
 *═══════════════════════════════════════════════════════════════════════════════*/

PGASSlidingState *pgas_sliding_alloc(int window_size, int slide_step,
                                     int N, int K, uint32_t seed)
{
    if (window_size < 100 || slide_step < 1 || slide_step > window_size)
    {
        return NULL;
    }
    if (K > PGAS_SLIDING_MAX_K)
    {
        return NULL;
    }

    PGASSlidingState *state = (PGASSlidingState *)mkl_calloc(
        1, sizeof(PGASSlidingState), 64);
    if (!state)
        return NULL;

    /* ═══════════════════════════════════════════════════════════════════════
     * ALLOCATE OWNED PGAS STATE
     * ═══════════════════════════════════════════════════════════════════════*/
    state->pgas = pgas_mkl_alloc(N, window_size, K, seed);
    if (!state->pgas)
    {
        mkl_free(state);
        return NULL;
    }

    /* ═══════════════════════════════════════════════════════════════════════
     * RING BUFFER
     *
     * Capacity = window_size * RING_MULT, rounded to power of 2
     * This gives headroom for PGAS to lag behind tick loop
     * ═══════════════════════════════════════════════════════════════════════*/
    int ring_capacity = next_power_of_2(window_size * PGAS_SLIDING_RING_MULT);

    state->obs_ring = (float *)mkl_malloc(ring_capacity * sizeof(float), 64);
    state->tick_ring = (int64_t *)mkl_malloc(ring_capacity * sizeof(int64_t), 64);

    if (!state->obs_ring || !state->tick_ring)
    {
        pgas_sliding_free(state);
        return NULL;
    }

    state->ring_capacity = ring_capacity;
    state->ring_mask = ring_capacity - 1;
    pgas_atomic_store_64(&state->ring_write_idx, 0);
    state->ring_read_idx = 0;

    /* ═══════════════════════════════════════════════════════════════════════
     * WINDOW CONFIG
     * ═══════════════════════════════════════════════════════════════════════*/
    state->window_size = window_size;
    state->slide_step = slide_step;
    state->warmup_size = window_size; /* First window needs full data */

    /* ═══════════════════════════════════════════════════════════════════════
     * OPTIMIZATION: Pre-allocate RNG workspaces for vectorized propagation
     * Align to 64 bytes for AVX-512
     * ═══════════════════════════════════════════════════════════════════════*/
    state->ws_rng_uniform = (float *)mkl_malloc(slide_step * sizeof(float), 64);
    state->ws_rng_normal = (float *)mkl_malloc(slide_step * sizeof(float), 64);

    if (!state->ws_rng_uniform || !state->ws_rng_normal)
    {
        pgas_sliding_free(state);
        return NULL;
    }

    state->has_reference = false;

    /* ═══════════════════════════════════════════════════════════════════════
     * INIT STATE
     * ═══════════════════════════════════════════════════════════════════════*/
    state->window_start_tick = -1;
    state->window_end_tick = -1;
    state->windows_completed = 0;

    memset(state->pi, 0, sizeof(state->pi));
    state->last_acceptance_rate = 0.0f;
    state->last_sweeps_used = 0;
    state->last_window_end_tick = -1;

    state->total_observations = 0;
    state->dropped_observations = 0;
    state->avg_acceptance_rate = 0.0f;

    return state;
}

void pgas_sliding_free(PGASSlidingState *state)
{
    if (!state)
        return;

    if (state->pgas)
    {
        pgas_mkl_free(state->pgas);
    }

    mkl_free(state->obs_ring);
    mkl_free(state->tick_ring);
    mkl_free(state->ws_rng_uniform);
    mkl_free(state->ws_rng_normal);
    mkl_free(state);
}

/*═══════════════════════════════════════════════════════════════════════════════
 * MODEL CONFIGURATION
 *═══════════════════════════════════════════════════════════════════════════════*/

void pgas_sliding_set_model(PGASSlidingState *state,
                            const double *trans,
                            const double *mu_vol,
                            const double *sigma_vol,
                            double phi)
{
    if (!state || !state->pgas)
        return;
    pgas_mkl_set_model(state->pgas, trans, mu_vol, sigma_vol, phi);
}

void pgas_sliding_set_prior(PGASSlidingState *state, float alpha, float kappa)
{
    if (!state || !state->pgas)
        return;
    state->pgas->prior_alpha = alpha;
    state->pgas->sticky_kappa = kappa;
}

void pgas_sliding_set_recency(PGASSlidingState *state, float lambda)
{
    if (!state || !state->pgas)
        return;
    pgas_mkl_set_recency_lambda(state->pgas, lambda);
}

/*═══════════════════════════════════════════════════════════════════════════════
 * OBSERVATION INPUT (Lock-Free)
 *═══════════════════════════════════════════════════════════════════════════════*/

bool pgas_sliding_push(PGASSlidingState *state, float obs, int64_t tick)
{
    if (!state)
        return false;

    /* Get current write position atomically */
    int64_t write_idx = pgas_atomic_load_64(&state->ring_write_idx);

    /* Check for overflow (would overwrite unread data) */
    int64_t available = write_idx - state->ring_read_idx;
    if (available >= state->ring_capacity - 1)
    {
        /* Ring full - drop observation */
        state->dropped_observations++;
        return false;
    }

    /* Write to ring buffer */
    int pos = (int)(write_idx & state->ring_mask);
    state->obs_ring[pos] = obs;
    state->tick_ring[pos] = tick;

    /* Advance write index (release semantics for consumer visibility) */
    pgas_atomic_store_64(&state->ring_write_idx, write_idx + 1);

    state->total_observations++;
    return true;
}

/*═══════════════════════════════════════════════════════════════════════════════
 * WINDOW MANAGEMENT
 *═══════════════════════════════════════════════════════════════════════════════*/

int64_t pgas_sliding_ring_count(const PGASSlidingState *state)
{
    if (!state)
        return 0;
    int64_t write_idx = pgas_atomic_load_64((pgas_atomic_int64 *)&state->ring_write_idx);
    return write_idx - state->ring_read_idx;
}

int64_t pgas_sliding_available(const PGASSlidingState *state)
{
    return pgas_sliding_ring_count(state);
}

bool pgas_sliding_window_ready(const PGASSlidingState *state)
{
    if (!state)
        return false;

    int64_t count = pgas_sliding_ring_count(state);

    if (state->windows_completed == 0)
    {
        /* First window: need full window_size */
        return count >= state->window_size;
    }
    else
    {
        /* Subsequent windows: need slide_step new observations */
        return count >= state->slide_step;
    }
}

/**
 * Extract observations from ring buffer into PGAS observations array
 */
void pgas_sliding_extract_window(PGASSlidingState *state)
{
    if (!state || !state->pgas)
        return;

    const int T = state->window_size;
    const int S = state->slide_step;
    PGASMKLState *pgas = state->pgas;
    float *obs_dst = pgas->observations;

    /* Determine window range in ring buffer */
    int64_t start_idx = state->ring_read_idx;

    if (state->windows_completed == 0)
    {
        /* ═══════════════════════════════════════════════════════════════════
         * COLD START: Full copy of T elements from ring buffer
         * ═══════════════════════════════════════════════════════════════════*/
        for (int t = 0; t < T; t++)
        {
            int ring_pos = (int)((start_idx + t) & state->ring_mask);
            obs_dst[t] = state->obs_ring[ring_pos];
        }

        /* Initialize reference trajectory from observation mean */
        float mean_obs = 0.0f;
        for (int t = 0; t < T; t++)
        {
            mean_obs += obs_dst[t];
        }
        mean_obs /= T;

        for (int t = 0; t < T; t++)
        {
            pgas->ref_regimes[t] = 0;
            pgas->ref_h[t] = mean_obs;
            pgas->ref_ancestors[t] = pgas->ref_idx;
        }
    }
    else
    {
        /* ═══════════════════════════════════════════════════════════════════
         * WARM SLIDE: Differential update (All-in-one)
         *
         * 1. memmove existing obs left by S (reuse T-S elements)
         * 2. Copy only S new elements from ring buffer
         * 3. memmove reference left by S
         * 4. Propagate reference tail (vectorized)
         *
         * This is MUCH faster than full reconstruction.
         * ═══════════════════════════════════════════════════════════════════*/

        /* 1. Shift observations left by S positions */
        memmove(obs_dst, obs_dst + S, (T - S) * sizeof(float));

        /* 2. Copy only the NEW observations from ring (last S positions) */
        for (int t = T - S; t < T; t++)
        {
            int ring_pos = (int)((start_idx + t) & state->ring_mask);
            obs_dst[t] = state->obs_ring[ring_pos];
        }

        /* 3. Shift reference trajectory left by S (IN-PLACE) */
        memmove(pgas->ref_regimes, pgas->ref_regimes + S, (T - S) * sizeof(int));
        memmove(pgas->ref_h, pgas->ref_h + S, (T - S) * sizeof(float));

        /* 4. Propagate reference tail (VECTORIZED) */
        propagate_reference_forward_vectorized(state, T - S, T);
    }

    /* Record window tick range */
    int ring_start = (int)(start_idx & state->ring_mask);
    int ring_end = (int)((start_idx + T - 1) & state->ring_mask);
    state->window_start_tick = state->tick_ring[ring_start];
    state->window_end_tick = state->tick_ring[ring_end];

    pgas->T = T;
}

/*═══════════════════════════════════════════════════════════════════════════════
 * PGAS EXECUTION
 *═══════════════════════════════════════════════════════════════════════════════*/

float pgas_sliding_run_sweeps(PGASSlidingState *state, int n_sweeps)
{
    if (!state || !state->pgas || n_sweeps < 1)
        return 0.0f;

    float total_accept = 0.0f;

    for (int s = 0; s < n_sweeps; s++)
    {
        float accept = pgas_mkl_gibbs_sweep(state->pgas);
        total_accept += accept;
    }

    state->last_acceptance_rate = total_accept / n_sweeps;
    state->last_sweeps_used = n_sweeps;

    /* Update running average */
    float alpha = 0.1f; /* EMA smoothing */
    if (state->windows_completed == 0)
    {
        state->avg_acceptance_rate = state->last_acceptance_rate;
    }
    else
    {
        state->avg_acceptance_rate = alpha * state->last_acceptance_rate +
                                     (1.0f - alpha) * state->avg_acceptance_rate;
    }

    return state->last_acceptance_rate;
}

void pgas_sliding_get_pi(const PGASSlidingState *state, float *pi_out)
{
    if (!state || !state->pgas || !pi_out)
        return;

    const int K = state->pgas->K;
    memcpy(pi_out, state->pgas->model.trans, K * K * sizeof(float));
}

/*═══════════════════════════════════════════════════════════════════════════════
 * WINDOW SLIDING
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * Propagate reference trajectory forward using AR(1) dynamics (VECTORIZED)
 *
 * OPTIMIZATION: Batch generate random numbers instead of scalar MKL calls.
 * This reduces MKL function call overhead by ~90%.
 *
 * Works IN-PLACE on pgas->ref_regimes and pgas->ref_h.
 */
static void propagate_reference_forward_vectorized(PGASSlidingState *state,
                                                   int start_t, int end_t)
{
    if (!state || !state->pgas)
        return;

    PGASMKLState *pgas = state->pgas;
    const PGASMKLModel *m = &pgas->model;
    const int K = pgas->K;
    const int n_needed = end_t - start_t;

    if (n_needed <= 0)
        return;

    /* ═══════════════════════════════════════════════════════════════════════
     * 1. BATCH GENERATE RANDOM NUMBERS
     * ═══════════════════════════════════════════════════════════════════════*/
    VSLStreamStatePtr stream = (VSLStreamStatePtr)pgas->rng.stream;

    /* Generate Uniforms for regime transitions */
    vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, n_needed,
                 state->ws_rng_uniform, 0.0f, 1.0f);

    /* Generate standard Normals for AR(1) noise */
    vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, n_needed,
                  state->ws_rng_normal, 0.0f, 1.0f);

    /* ═══════════════════════════════════════════════════════════════════════
     * 2. PROPAGATE STATE USING BATCHED RANDOM NUMBERS
     *    Work directly on pgas->ref_* buffers (IN-PLACE)
     * ═══════════════════════════════════════════════════════════════════════*/
    int *regimes = pgas->ref_regimes;
    float *h = pgas->ref_h;

    for (int i = 0; i < n_needed; i++)
    {
        const int t = start_t + i;
        const int prev_regime = regimes[t - 1];
        const float prev_h = h[t - 1];

        /* A. Sample Regime Transition */
        const float u = state->ws_rng_uniform[i];
        const float *row = &m->trans[prev_regime * K];
        float cumsum = 0.0f;
        int new_regime = prev_regime; /* Default: stay (stickiness) */

        for (int j = 0; j < K; j++)
        {
            cumsum += row[j];
            if (u < cumsum)
            {
                new_regime = j;
                break;
            }
        }
        regimes[t] = new_regime;

        /* B. AR(1) State Propagation: h_t = μ + φ(h_{t-1} - μ) + σ·ε */
        const float mu_k = m->mu_vol[new_regime];
        const float sigma_k = m->sigma_vol[new_regime];
        const float mean = mu_k + m->phi * (prev_h - mu_k);

        h[t] = mean + sigma_k * state->ws_rng_normal[i];

        /* Set ancestor to self */
        pgas->ref_ancestors[t] = pgas->ref_idx;
    }
}

void pgas_sliding_slide(PGASSlidingState *state)
{
    if (!state || !state->pgas)
        return;

    /* ═══════════════════════════════════════════════════════════════════════
     * SLIDE (Simplified)
     *
     * In the optimized flow, all data movement (obs shift, ref shift,
     * ref propagation) happens in extract_window(). This function just:
     *   1. Advances ring read pointer (commits consumption of S items)
     *   2. Updates output statistics
     *
     * ═══════════════════════════════════════════════════════════════════════*/

    /* Advance ring read index for NEXT window */
    state->ring_read_idx += state->slide_step;

    /* Mark that we have a valid reference for next window */
    state->has_reference = true;

    /* Update output statistics */
    const int K = state->pgas->K;
    memcpy(state->pi, state->pgas->model.trans, K * K * sizeof(float));
    state->last_window_end_tick = state->window_end_tick;

    state->windows_completed++;
}

/*═══════════════════════════════════════════════════════════════════════════════
 * COMBINED ITERATION
 *═══════════════════════════════════════════════════════════════════════════════*/

float pgas_sliding_iterate(PGASSlidingState *state, int n_sweeps, float *pi_out)
{
    if (!pgas_sliding_window_ready(state))
    {
        return -1.0f;
    }

    /* Extract window (copies observations, sets up warm start) */
    pgas_sliding_extract_window(state);

    /* Run PGAS sweeps */
    float accept = pgas_sliding_run_sweeps(state, n_sweeps);

    /* Slide window (saves reference, advances read pointer) */
    pgas_sliding_slide(state);

    /* Output Π */
    if (pi_out)
    {
        pgas_sliding_get_pi(state, pi_out);
    }

    return accept;
}

/*═══════════════════════════════════════════════════════════════════════════════
 * DIAGNOSTICS
 *═══════════════════════════════════════════════════════════════════════════════*/

void pgas_sliding_print_diagnostics(const PGASSlidingState *state)
{
    if (!state)
        return;

    printf("═══════════════════════════════════════════════════════════════\n");
    printf("PGAS SLIDING WINDOW DIAGNOSTICS\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("Window size:        %d\n", state->window_size);
    printf("Slide step:         %d\n", state->slide_step);
    printf("Ring capacity:      %d\n", state->ring_capacity);
    printf("Ring count:         %lld\n", (long long)pgas_sliding_ring_count(state));
    printf("Windows completed:  %d\n", state->windows_completed);
    printf("───────────────────────────────────────────────────────────────\n");
    printf("Total observations: %lld\n", (long long)state->total_observations);
    printf("Dropped (overflow): %lld\n", (long long)state->dropped_observations);
    printf("Has warm reference: %s\n", state->has_reference ? "YES" : "NO");
    printf("───────────────────────────────────────────────────────────────\n");
    printf("Last window ticks:  %lld → %lld\n",
           (long long)state->window_start_tick,
           (long long)state->window_end_tick);
    printf("Last acceptance:    %.3f\n", state->last_acceptance_rate);
    printf("Avg acceptance:     %.3f\n", state->avg_acceptance_rate);
    printf("Last sweeps:        %d\n", state->last_sweeps_used);
    printf("═══════════════════════════════════════════════════════════════\n");

    /* Also print underlying PGAS diagnostics */
    if (state->pgas)
    {
        pgas_mkl_print_diagnostics(state->pgas);
    }
}