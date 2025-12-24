/*═══════════════════════════════════════════════════════════════════════════════
 * PGAS-PARIS Integration Implementation
 *
 * Extends PGAS with PARIS backward smoothing for path degeneracy fix.
 *
 * Key insight: Standard PGAS counts transitions from single reference trajectory
 * which suffers from path degeneracy. PARIS-enhanced PGAS:
 *   1. Runs CSMC forward (standard)
 *   2. Copies particles to PARIS state (float→double conversion)
 *   3. Runs PARIS backward to compute smoothed indices
 *   4. Samples M trajectories from smoothed distribution
 *   5. Counts transitions from ALL trajectories (ensemble)
 *   6. Uses aggregated counts for Dirichlet posterior
 *
 * This gives transition counts that properly represent P(z_{0:T} | y_{1:T}).
 *
 *═══════════════════════════════════════════════════════════════════════════════*/

#include "pgas_paris.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>

#include <mkl.h>
#include <mkl_vsl.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/*═══════════════════════════════════════════════════════════════════════════════
 * LIFECYCLE
 *═══════════════════════════════════════════════════════════════════════════════*/

PGASParisState *pgas_paris_alloc(PGASMKLState *pgas, int n_trajectories)
{
    if (!pgas)
        return NULL;
    if (n_trajectories < 1)
        n_trajectories = PGAS_PARIS_DEFAULT_TRAJECTORIES;
    if (n_trajectories > PGAS_PARIS_MAX_TRAJECTORIES)
    {
        n_trajectories = PGAS_PARIS_MAX_TRAJECTORIES;
    }

    PGASParisState *state = (PGASParisState *)calloc(1, sizeof(PGASParisState));
    if (!state)
        return NULL;

    state->pgas = pgas;
    state->n_trajectories = n_trajectories;
    state->use_ensemble_counts = 1; /* Default: use ensemble */

    /* Allocate PARIS state using paris_mkl_alloc
     * PARIS needs: N particles, T timesteps, K regimes, seed
     * Use a fixed offset seed since PGASMKLState doesn't expose seed */
    state->paris = paris_mkl_alloc(pgas->N, pgas->T, pgas->K, 12345 + (uint32_t)pgas->total_sweeps);
    if (!state->paris)
    {
        free(state);
        return NULL;
    }

    /* Allocate conversion buffers (PGAS float → PARIS double) */
    size_t TN = (size_t)pgas->T * pgas->N;
    state->h_double = (double *)mkl_malloc(TN * sizeof(double), 64);
    state->weights_double = (double *)mkl_malloc(TN * sizeof(double), 64);

    /* OPTIMIZATION: Pre-allocate copy buffers (avoid malloc per sweep) */
    state->regimes_nopad = (int *)mkl_malloc(TN * sizeof(int), 64);
    state->ancestors_nopad = (int *)mkl_malloc(TN * sizeof(int), 64);
    state->h_traj_cache = (float *)mkl_malloc((size_t)n_trajectories * pgas->T * sizeof(float), 64);

    if (!state->h_double || !state->weights_double ||
        !state->regimes_nopad || !state->ancestors_nopad || !state->h_traj_cache)
    {
        if (state->h_double)
            mkl_free(state->h_double);
        if (state->weights_double)
            mkl_free(state->weights_double);
        if (state->regimes_nopad)
            mkl_free(state->regimes_nopad);
        if (state->ancestors_nopad)
            mkl_free(state->ancestors_nopad);
        if (state->h_traj_cache)
            mkl_free(state->h_traj_cache);
        paris_mkl_free(state->paris);
        free(state);
        return NULL;
    }

    /* Allocate trajectory storage */
    state->traj.T = pgas->T;
    state->traj.K = pgas->K;
    state->traj.M = n_trajectories;

    state->traj.trajectories = (int *)mkl_malloc(
        (size_t)n_trajectories * pgas->T * sizeof(int), 64);
    if (!state->traj.trajectories)
    {
        mkl_free(state->h_double);
        mkl_free(state->weights_double);
        mkl_free(state->regimes_nopad);
        mkl_free(state->ancestors_nopad);
        mkl_free(state->h_traj_cache);
        paris_mkl_free(state->paris);
        free(state);
        return NULL;
    }

    memset(state->traj.n_trans, 0, sizeof(state->traj.n_trans));
    memset(state->traj.per_traj_counts, 0, sizeof(state->traj.per_traj_counts));

    return state;
}

void pgas_paris_free(PGASParisState *state)
{
    if (!state)
        return;

    if (state->paris)
    {
        paris_mkl_free(state->paris);
    }
    if (state->h_double)
    {
        mkl_free(state->h_double);
    }
    if (state->weights_double)
    {
        mkl_free(state->weights_double);
    }
    if (state->regimes_nopad)
    {
        mkl_free(state->regimes_nopad);
    }
    if (state->ancestors_nopad)
    {
        mkl_free(state->ancestors_nopad);
    }
    if (state->h_traj_cache)
    {
        mkl_free(state->h_traj_cache);
    }
    if (state->traj.trajectories)
    {
        mkl_free(state->traj.trajectories);
    }

    free(state);
}

/*═══════════════════════════════════════════════════════════════════════════════
 * PARTICLE COPY: PGAS → PARIS (OPTIMIZED)
 *
 * PGAS stores particles as [T × N_padded] with float h and float weights.
 * PARIS load_particles expects [T × N] with double h and double weights.
 *
 * OPTIMIZATIONS:
 *   1. Use pre-allocated buffers (no malloc per sweep)
 *   2. OpenMP parallel for T×N loops
 *   3. Blocked copy for better cache utilization
 *═══════════════════════════════════════════════════════════════════════════════*/

void pgas_paris_copy_particles(PGASParisState *state)
{
    if (!state || !state->pgas || !state->paris)
        return;

    PGASMKLState *pgas = state->pgas;
    const int T = pgas->T;
    const int N = pgas->N;
    const int Np = pgas->N_padded;

    /* Use pre-allocated buffers */
    int *regimes_nopad = state->regimes_nopad;
    int *ancestors_nopad = state->ancestors_nopad;
    double *h_double = state->h_double;
    double *weights_double = state->weights_double;

    /* Parallel copy with stride conversion */
    int t; /* MSVC OpenMP requires loop var declared outside */
#pragma omp parallel for schedule(static) if (T > 100)
    for (t = 0; t < T; t++)
    {
        const float *h_src = &pgas->h[t * Np];
        const float *w_src = &pgas->weights[t * Np];
        const int *r_src = &pgas->regimes[t * Np];
        const int *a_src = &pgas->ancestors[t * Np];

        double *h_dst = &h_double[t * N];
        double *w_dst = &weights_double[t * N];
        int *r_dst = &regimes_nopad[t * N];
        int *a_dst = &ancestors_nopad[t * N];

        for (int n = 0; n < N; n++)
        {
            h_dst[n] = (double)h_src[n];
            w_dst[n] = (double)w_src[n];
            r_dst[n] = r_src[n];
            /* Clamp ancestor to valid range */
            a_dst[n] = (a_src[n] < N) ? a_src[n] : N - 1;
        }
    }

    /* Update PARIS model parameters to match PGAS */
    double trans_d[PGAS_MKL_MAX_REGIMES * PGAS_MKL_MAX_REGIMES];
    double mu_vol_d[PGAS_MKL_MAX_REGIMES];

    for (int i = 0; i < pgas->K * pgas->K; i++)
    {
        trans_d[i] = (double)pgas->model.trans[i];
    }
    for (int k = 0; k < pgas->K; k++)
    {
        mu_vol_d[k] = (double)pgas->model.mu_vol[k];
    }

    paris_mkl_set_model(state->paris, trans_d, mu_vol_d,
                        (double)pgas->model.phi, (double)pgas->model.sigma_h);

    /* Load particles into PARIS */
    paris_mkl_load_particles(state->paris,
                             regimes_nopad,
                             h_double,
                             weights_double,
                             ancestors_nopad,
                             T);
}

/*═══════════════════════════════════════════════════════════════════════════════
 * BACKWARD SMOOTHING
 *═══════════════════════════════════════════════════════════════════════════════*/

void pgas_paris_run_backward(PGASParisState *state)
{
    if (!state || !state->paris)
        return;

    paris_mkl_backward_smooth(state->paris);
}

/*═══════════════════════════════════════════════════════════════════════════════
 * TRAJECTORY SAMPLING (OPTIMIZED)
 *
 * After PARIS backward pass, state->paris->smoothed[t * Np + n] contains
 * the INDEX of the particle at time t for the n-th smoothed trajectory.
 *
 * OPTIMIZATIONS:
 *   1. Hash-based diversity check O(M×T) instead of O(M²×T)
 *   2. Cache h values during extraction for regime stats
 *   3. Direct extraction (avoid paris_mkl_get_trajectory function call)
 *═══════════════════════════════════════════════════════════════════════════════*/

/* FNV-1a hash for fast trajectory comparison */
static inline uint64_t trajectory_hash_fnv1a(const int *traj, int T)
{
    uint64_t hash = 14695981039346656037ULL;

    /* Process 4 ints at a time */
    int t = 0;
    for (; t + 3 < T; t += 4)
    {
        hash ^= (uint64_t)traj[t];
        hash *= 1099511628211ULL;
        hash ^= (uint64_t)traj[t + 1];
        hash *= 1099511628211ULL;
        hash ^= (uint64_t)traj[t + 2];
        hash *= 1099511628211ULL;
        hash ^= (uint64_t)traj[t + 3];
        hash *= 1099511628211ULL;
    }
    for (; t < T; t++)
    {
        hash ^= (uint64_t)traj[t];
        hash *= 1099511628211ULL;
    }
    return hash;
}

void pgas_paris_sample_trajectories(PGASParisState *state)
{
    if (!state || !state->paris)
        return;

    const int T = state->traj.T;
    const int M = state->traj.M;
    const int N = state->pgas->N;
    const int Np_paris = state->paris->N_padded;

    /* Sample M particle indices (evenly spaced for diversity) */
    int stride = (N > M) ? (N / M) : 1;

    /* Hash table for O(M) diversity check */
    uint64_t hashes[PGAS_PARIS_MAX_TRAJECTORIES];

    /* Extract trajectories with h caching */
    for (int m = 0; m < M; m++)
    {
        int particle_idx = (m * stride) % N;

        int *traj_regimes = &state->traj.trajectories[m * T];
        float *traj_h = &state->h_traj_cache[m * T];

        /* Direct extraction - avoid function call overhead */
        for (int t = 0; t < T; t++)
        {
            int smoothed_idx = state->paris->smoothed[t * Np_paris + particle_idx];
            traj_regimes[t] = state->paris->regimes[t * Np_paris + smoothed_idx];
            traj_h[t] = state->paris->h[t * Np_paris + smoothed_idx];
        }

        /* Compute hash */
        hashes[m] = trajectory_hash_fnv1a(traj_regimes, T);
    }

    /* Hash-based diversity: O(M²) worst case but O(M) average
     * Much faster than O(M²×T) element-wise comparison */
    int unique_count = 0;
    for (int m = 0; m < M; m++)
    {
        int is_unique = 1;
        for (int m2 = 0; m2 < m; m2++)
        {
            if (hashes[m] == hashes[m2])
            {
                /* Hash collision - spot check (3 points, not full T) */
                const int *t1 = &state->traj.trajectories[m * T];
                const int *t2 = &state->traj.trajectories[m2 * T];
                if (t1[0] == t2[0] && t1[T - 1] == t2[T - 1] && t1[T / 2] == t2[T / 2])
                {
                    is_unique = 0;
                    break;
                }
            }
        }
        if (is_unique)
            unique_count++;
    }
    state->traj.trajectory_diversity = (float)unique_count / M;
}

/*═══════════════════════════════════════════════════════════════════════════════
 * TRANSITION COUNTING
 *
 * Count transitions from ALL M smoothed trajectories.
 * This gives proper ensemble counts for Dirichlet posterior.
 *═══════════════════════════════════════════════════════════════════════════════*/

void pgas_paris_count_transitions(PGASParisState *state)
{
    if (!state)
        return;

    const int T = state->traj.T;
    const int K = state->traj.K;
    const int M = state->traj.M;

    /* Clear counts */
    memset(state->traj.n_trans, 0, K * K * sizeof(int));
    memset(state->traj.per_traj_counts, 0, sizeof(state->traj.per_traj_counts));

    /* Count transitions from each trajectory */
    for (int m = 0; m < M; m++)
    {
        const int *traj = &state->traj.trajectories[m * T];
        int *per_traj = &state->traj.per_traj_counts[m * K * K];

        for (int t = 1; t < T; t++)
        {
            int from = traj[t - 1];
            int to = traj[t];

            /* Bounds check */
            if (from >= 0 && from < K && to >= 0 && to < K)
            {
                per_traj[from * K + to]++;
                state->traj.n_trans[from * K + to]++;
            }
        }
    }
}

/*═══════════════════════════════════════════════════════════════════════════════
 * TRANSITION MATRIX SAMPLING
 *
 * Sample from Dirichlet posterior using PARIS ensemble EXPECTED counts.
 *
 * CRITICAL FIX: Use average counts across M trajectories, not sum!
 *
 * The Problem (before):
 *   - Summing counts from M=8 trajectories gives 8×(T-1) observations
 *   - This inflates effective data count, drowning out the sticky prior κ
 *   - Result: diagonal collapses, chatter spikes
 *
 * The Fix (Rao-Blackwellization):
 *   - Use expected counts: E[n_ij] = sum(n_ij^m) / M
 *   - This maintains (T-1) effective observations
 *   - Prior κ remains properly weighted
 *   - Lower variance than single-path PGAS (Lindsten et al. 2014)
 *
 * Dirichlet: π_i ~ Dir(α + E[n_i1], ..., α + E[n_iK] + κ·δ_{ij})
 *═══════════════════════════════════════════════════════════════════════════════*/

void pgas_paris_sample_trans_matrix(PGASParisState *state)
{
    if (!state || !state->pgas)
        return;

    PGASMKLState *pgas = state->pgas;
    const int K = pgas->K;
    const float alpha = pgas->prior_alpha;
    const float kappa = pgas->sticky_kappa;
    const int M = state->n_trajectories; /* Number of PARIS trajectories */

    VSLStreamStatePtr stream = (VSLStreamStatePtr)pgas->rng.stream;

    /* Sample each row from Dirichlet */
    for (int i = 0; i < K; i++)
    {
        double gamma_samples[PGAS_MKL_MAX_REGIMES];
        double row_sum = 0.0;

        for (int j = 0; j < K; j++)
        {
            /* CRITICAL: Use EXPECTED count (average across M trajectories)
             * This keeps effective observation count at (T-1), not M×(T-1) */
            double expected_count = (double)state->traj.n_trans[i * K + j] / M;

            /* Dirichlet parameter: α + E[n_ij] + κ·δ_{ij} */
            double dir_alpha = alpha + expected_count;
            if (i == j)
            {
                dir_alpha += kappa; /* Sticky prior */
            }

            /* Sample from Gamma(dir_alpha, 1) */
            vdRngGamma(VSL_RNG_METHOD_GAMMA_GNORM, stream, 1,
                       &gamma_samples[j], dir_alpha, 0.0, 1.0);

            row_sum += gamma_samples[j];
        }

        /* Normalize to get Dirichlet sample */
        for (int j = 0; j < K; j++)
        {
            float p = (float)(gamma_samples[j] / row_sum);

            /* Numerical safety */
            if (p < 1e-8f)
                p = 1e-8f;
            if (p > 1.0f - 1e-8f)
                p = 1.0f - 1e-8f;

            pgas->model.trans[i * K + j] = p;
            pgas->model.log_trans[i * K + j] = logf(p);
        }
    }

    /* Update chatter ratio for adaptive κ
     * Use expected counts (rates cancel, but consistent with above) */
    double total_self = 0.0;
    double total_switch = 0.0;

    for (int i = 0; i < K; i++)
    {
        total_self += (double)state->traj.n_trans[i * K + i] / M;
        for (int j = 0; j < K; j++)
        {
            if (i != j)
            {
                total_switch += (double)state->traj.n_trans[i * K + j] / M;
            }
        }
    }

    /* Expected self-transition rate from current κ */
    float expected_self_rate = kappa / (kappa + K - 1 + K * alpha);
    float actual_self_rate = (total_self + total_switch > 0.001)
                                 ? (float)(total_self / (total_self + total_switch))
                                 : expected_self_rate;

    /* Chatter ratio: actual_switches / expected_switches */
    float expected_switch_rate = 1.0f - expected_self_rate;
    float actual_switch_rate = 1.0f - actual_self_rate;

    if (expected_switch_rate > 0.001f)
    {
        pgas->last_chatter_ratio = actual_switch_rate / expected_switch_rate;
    }

    /* RLS update for adaptive κ (if enabled) */
    if (pgas->adaptive_kappa_enabled)
    {
        /* Smooth chatter ratio with RLS */
        float lambda = pgas->rls_forgetting;
        pgas->rls_chatter_estimate = lambda * pgas->rls_chatter_estimate +
                                     (1.0f - lambda) * pgas->last_chatter_ratio;

        /* Adjust κ based on smoothed chatter
         * Use multiplicative update for scale-invariance */
        float smoothed_chatter = pgas->rls_chatter_estimate;

        /* Adaptive rate based on deviation from target (1.0) */
        float deviation = smoothed_chatter - 1.0f;
        float adapt_rate = 0.02f; /* 2% per sweep */

        if (deviation > 0.05f)
        {
            /* Too many switches → increase κ */
            pgas->sticky_kappa *= (1.0f + adapt_rate * deviation);
        }
        else if (deviation < -0.05f)
        {
            /* Too few switches → decrease κ */
            pgas->sticky_kappa *= (1.0f + adapt_rate * deviation); /* deviation is negative */
        }

        /* Clamp to bounds */
        if (pgas->sticky_kappa < pgas->kappa_min)
        {
            pgas->sticky_kappa = pgas->kappa_min;
        }
        if (pgas->sticky_kappa > pgas->kappa_max)
        {
            pgas->sticky_kappa = pgas->kappa_max;
        }
    }
}

/*═══════════════════════════════════════════════════════════════════════════════
 * FULL GIBBS SWEEP
 *═══════════════════════════════════════════════════════════════════════════════*/

float pgas_paris_gibbs_sweep(PGASParisState *state)
{
    if (!state || !state->pgas || !state->paris)
        return 0.0f;

    /* 1. CSMC forward pass */
    float accept = pgas_mkl_csmc_sweep(state->pgas);

    /* 2. Copy particles to PARIS */
    pgas_paris_copy_particles(state);

    /* 3. PARIS backward smoothing */
    pgas_paris_run_backward(state);

    /* 4. Sample M trajectories */
    pgas_paris_sample_trajectories(state);

    /* 5. Count transitions from ensemble */
    pgas_paris_count_transitions(state);

    /* 6. Sample new Π from Dirichlet (using PARIS counts) */
    pgas_paris_sample_trans_matrix(state);

    state->total_sweeps++;

    return accept;
}

/*═══════════════════════════════════════════════════════════════════════════════
 * RUN MULTIPLE SWEEPS
 *═══════════════════════════════════════════════════════════════════════════════*/

void pgas_paris_run_sweeps(
    PGASParisState *state,
    int n_sweeps,
    int burnin,
    PGASParisSweepCallback callback,
    void *user_data)
{
    if (!state)
        return;

    for (int sweep = 0; sweep < n_sweeps; sweep++)
    {
        pgas_paris_gibbs_sweep(state);

        if (callback && sweep >= burnin)
        {
            callback(state, sweep, user_data);
        }
    }
}

/*═══════════════════════════════════════════════════════════════════════════════
 * DIAGNOSTICS
 *═══════════════════════════════════════════════════════════════════════════════*/

float pgas_paris_get_path_degeneracy(const PGASParisState *state)
{
    if (!state)
        return 1.0f;

    /* Path degeneracy = 1 - trajectory_diversity */
    return 1.0f - state->traj.trajectory_diversity;
}

float pgas_paris_get_trajectory_diversity(const PGASParisState *state)
{
    if (!state)
        return 0.0f;
    return state->traj.trajectory_diversity;
}

void pgas_paris_print_diagnostics(const PGASParisState *state)
{
    if (!state)
        return;

    printf("═══════════════════════════════════════════════════════════\n");
    printf("PGAS-PARIS DIAGNOSTICS\n");
    printf("═══════════════════════════════════════════════════════════\n");
    printf("Total sweeps:         %d\n", state->total_sweeps);
    printf("Trajectories sampled: %d (M)\n", state->n_trajectories);
    printf("Trajectory diversity: %.2f%%\n", state->traj.trajectory_diversity * 100.0f);
    printf("Path degeneracy:      %.2f%%\n", pgas_paris_get_path_degeneracy(state) * 100.0f);
    printf("CSMC acceptance:      %.3f\n", pgas_mkl_get_acceptance_rate(state->pgas));
    printf("Chatter ratio:        %.2f\n", pgas_mkl_get_chatter_ratio(state->pgas));
    printf("Sticky κ:             %.1f\n", pgas_mkl_get_sticky_kappa(state->pgas));

    int K = state->traj.K;
    int M = state->n_trajectories;

    printf("\nPARIS expected transition counts (E[n_ij] = sum/M):\n");
    for (int i = 0; i < K; i++)
    {
        printf("  [");
        for (int j = 0; j < K; j++)
        {
            double expected = (double)state->traj.n_trans[i * K + j] / M;
            printf(" %6.1f", expected);
        }
        printf(" ]\n");
    }

    /* Show total expected transitions (should be ~T-1) */
    double total_expected = 0.0;
    for (int i = 0; i < K * K; i++)
    {
        total_expected += (double)state->traj.n_trans[i] / M;
    }
    printf("  Total expected: %.1f (should be ~T-1 = %d)\n", total_expected, state->pgas->T - 1);

    printf("═══════════════════════════════════════════════════════════\n");
}

/*═══════════════════════════════════════════════════════════════════════════════
 * ENSEMBLE ACCUMULATOR
 *═══════════════════════════════════════════════════════════════════════════════*/

void pgas_paris_ensemble_init(PGASParisEnsemble *ens, int K)
{
    if (!ens)
        return;
    memset(ens, 0, sizeof(PGASParisEnsemble));
    ens->K = K;
}

void pgas_paris_ensemble_accumulate(PGASParisEnsemble *ens, const PGASParisState *state)
{
    if (!ens || !state)
        return;

    float trans[PGAS_MKL_MAX_REGIMES * PGAS_MKL_MAX_REGIMES];
    pgas_paris_get_transitions(state, trans, ens->K);

    ens->n_samples++;

    int K = ens->K;
    for (int i = 0; i < K * K; i++)
    {
        ens->trans_sum[i] += trans[i];
        ens->trans_sum_sq[i] += (double)trans[i] * trans[i];
    }

    float kappa = pgas_paris_get_sticky_kappa(state);
    ens->kappa_sum += kappa;
    ens->kappa_sum_sq += (double)kappa * kappa;
}

void pgas_paris_ensemble_get_mean(const PGASParisEnsemble *ens, float *out)
{
    if (!ens || !out || ens->n_samples == 0)
        return;

    int K = ens->K;
    for (int i = 0; i < K * K; i++)
    {
        out[i] = (float)(ens->trans_sum[i] / ens->n_samples);
    }
}

float pgas_paris_ensemble_get_std(const PGASParisEnsemble *ens, int i, int j)
{
    if (!ens || ens->n_samples < 2)
        return 0.0f;

    int idx = i * ens->K + j;
    double mean = ens->trans_sum[idx] / ens->n_samples;
    double var = (ens->trans_sum_sq[idx] / ens->n_samples) - (mean * mean);
    if (var < 0)
        var = 0;

    return (float)sqrt(var);
}

float pgas_paris_ensemble_get_max_std(const PGASParisEnsemble *ens)
{
    if (!ens)
        return 0.0f;

    float max_std = 0.0f;
    int K = ens->K;

    for (int i = 0; i < K; i++)
    {
        for (int j = 0; j < K; j++)
        {
            float std = pgas_paris_ensemble_get_std(ens, i, j);
            if (std > max_std)
                max_std = std;
        }
    }

    return max_std;
}

int pgas_paris_ensemble_is_converged(const PGASParisEnsemble *ens, float threshold)
{
    if (!ens)
        return 0;
    return pgas_paris_ensemble_get_max_std(ens) < threshold;
}

/*═══════════════════════════════════════════════════════════════════════════════
 * REGIME PARAMETER LEARNING
 *
 * Rao-Blackwellized estimation of μ_vol[k] and σ_vol[k] using PARIS ensemble.
 *═══════════════════════════════════════════════════════════════════════════════*/

void pgas_paris_regime_prior_init(PGASParisRegimePrior *prior, int K)
{
    if (!prior)
        return;

    /* Default μ_vol prior: N(-3.0, 2.0²) - weak prior, typical log-vol range */
    for (int k = 0; k < K && k < PGAS_MKL_MAX_REGIMES; k++)
    {
        prior->mu_prior_mean[k] = -3.0f;
        prior->mu_prior_var[k] = 4.0f; /* 2.0² */
    }

    /* Default σ_vol prior: InvGamma(3, 0.1) - weak prior on emission spread */
    prior->sigma_prior_shape = 3.0f;
    prior->sigma_prior_scale = 0.1f;

    /* Learning disabled by default */
    prior->learn_mu = 0;
    prior->learn_sigma = 0;

    /* Ordering enforced by default (prevents label switching) */
    prior->enforce_ordering = 1;
}

void pgas_paris_set_mu_prior(PGASParisRegimePrior *prior, int k,
                             float mean, float variance)
{
    if (!prior || k < 0 || k >= PGAS_MKL_MAX_REGIMES)
        return;
    prior->mu_prior_mean[k] = mean;
    prior->mu_prior_var[k] = variance;
}

void pgas_paris_collect_regime_stats(PGASParisState *state,
                                     PGASParisRegimeStats *stats)
{
    if (!state || !state->pgas || !stats)
        return;

    PGASMKLState *pgas = state->pgas;
    const int K = pgas->K;
    const int T = pgas->T;
    const int M = state->n_trajectories;
    const float phi = pgas->model.phi;

    stats->K = K;

    /* Zero initialize */
    for (int k = 0; k < K; k++)
    {
        stats->n_k[k] = 0.0;
        stats->sum_h_k[k] = 0.0;
        stats->sum_h_sq_k[k] = 0.0;
        stats->sum_resid_k[k] = 0.0;
        stats->sum_resid_sq_k[k] = 0.0;
    }

    /* Accumulate statistics from cached trajectory data
     * OPTIMIZATION: Use h_traj_cache instead of re-fetching from PARIS */
    for (int m = 0; m < M; m++)
    {
        const int *traj = &state->traj.trajectories[m * T];
        const float *h_cache = &state->h_traj_cache[m * T];

        /* First timestep - no residual */
        int z_0 = traj[0];
        float h_0 = h_cache[0];
        stats->n_k[z_0] += 1.0;
        stats->sum_h_k[z_0] += h_0;
        stats->sum_h_sq_k[z_0] += h_0 * h_0;

        /* Remaining timesteps - with residuals */
        float h_prev = h_0;
        for (int t = 1; t < T; t++)
        {
            int z_t = traj[t];
            float h_t = h_cache[t];

            stats->n_k[z_t] += 1.0;
            stats->sum_h_k[z_t] += h_t;
            stats->sum_h_sq_k[z_t] += h_t * h_t;

            /* Residual: h_t - φ*h_{t-1} */
            float resid = h_t - phi * h_prev;
            stats->sum_resid_k[z_t] += resid;
            stats->sum_resid_sq_k[z_t] += resid * resid;

            h_prev = h_t;
        }
    }

    /* Normalize by M to get expected counts (Rao-Blackwellization) */
    float inv_M = 1.0f / M;
    for (int k = 0; k < K; k++)
    {
        stats->n_k[k] *= inv_M;
        stats->sum_h_k[k] *= inv_M;
        stats->sum_h_sq_k[k] *= inv_M;
        stats->sum_resid_k[k] *= inv_M;
        stats->sum_resid_sq_k[k] *= inv_M;
    }
}

void pgas_paris_sample_mu_vol(PGASParisState *state,
                              const PGASParisRegimeStats *stats,
                              const PGASParisRegimePrior *prior)
{
    if (!state || !state->pgas || !stats || !prior)
        return;
    if (!prior->learn_mu)
        return;

    PGASMKLState *pgas = state->pgas;
    const int K = pgas->K;
    const float phi = pgas->model.phi;
    const float sigma_h = pgas->model.sigma_h;
    const float sigma_h_sq = sigma_h * sigma_h;
    const float one_minus_phi = 1.0f - phi;
    const float one_minus_phi_sq = one_minus_phi * one_minus_phi;

    VSLStreamStatePtr stream = (VSLStreamStatePtr)pgas->rng.stream;

    for (int k = 0; k < K; k++)
    {
        /* Prior parameters */
        float m0 = prior->mu_prior_mean[k];
        float s0_sq = prior->mu_prior_var[k];

        /* Data likelihood contribution
         * From: h_t = μ_k*(1-φ) + φ*h_{t-1} + σ_h*ε
         * Rearranging: residual = h_t - φ*h_{t-1} = μ_k*(1-φ) + σ_h*ε
         * So: E[residual | z=k] = μ_k*(1-φ)
         *     μ_k = E[residual]/(1-φ)
         *
         * The "data" for estimating μ_k is sum_resid_k / (1-φ)
         */
        double n_k = stats->n_k[k];

        if (n_k < 1.0)
        {
            /* No data for this regime, sample from prior */
            float sample;
            vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, stream, 1,
                          &sample, m0, sqrtf(s0_sq));
            pgas->model.mu_vol[k] = sample;
            continue;
        }

        /* Posterior variance:
         * 1/σ²_post = 1/s₀² + n_k*(1-φ)²/σ_h²
         */
        double precision_prior = 1.0 / s0_sq;
        double precision_data = n_k * one_minus_phi_sq / sigma_h_sq;
        double precision_post = precision_prior + precision_data;
        double var_post = 1.0 / precision_post;

        /* Posterior mean:
         * μ_post = σ²_post * (m₀/s₀² + (1-φ)*Σresid_k/σ_h²)
         *
         * Note: sum_resid_k = Σ(h_t - φ*h_{t-1}) for z_t = k
         * We want μ_k such that residual ≈ μ_k*(1-φ)
         * So we use sum_resid_k directly (it's already multiplied by n_k in effect)
         */
        double data_contribution = one_minus_phi * stats->sum_resid_k[k] / sigma_h_sq;
        double prior_contribution = m0 / s0_sq;
        double mean_post = var_post * (prior_contribution + data_contribution);

        /* Sample from posterior */
        float sample;
        vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, stream, 1,
                      &sample, (float)mean_post, sqrtf((float)var_post));

        pgas->model.mu_vol[k] = sample;

        /* Update precomputed mu_shift = μ_k * (1-φ) */
        pgas->model.mu_shift[k] = sample * one_minus_phi;
    }
}

void pgas_paris_sample_sigma_vol(PGASParisState *state,
                                 const PGASParisRegimeStats *stats,
                                 const PGASParisRegimePrior *prior)
{
    if (!state || !state->pgas || !stats || !prior)
        return;
    if (!prior->learn_sigma)
        return;

    PGASMKLState *pgas = state->pgas;
    const int K = pgas->K;
    const float phi = pgas->model.phi;
    const float one_minus_phi = 1.0f - phi;

    VSLStreamStatePtr stream = (VSLStreamStatePtr)pgas->rng.stream;

    for (int k = 0; k < K; k++)
    {
        /* Prior parameters: InvGamma(a, b) */
        float a0 = prior->sigma_prior_shape;
        float b0 = prior->sigma_prior_scale;

        double n_k = stats->n_k[k];

        if (n_k < 2.0)
        {
            /* Not enough data, keep current value */
            continue;
        }

        /* Posterior: InvGamma(a_post, b_post)
         * a_post = a₀ + n_k/2
         * b_post = b₀ + 0.5 * Σ(residual - μ_k*(1-φ))²
         */
        float mu_k = pgas->model.mu_vol[k];
        float target = mu_k * one_minus_phi;

        /* Compute sum of squared deviations from target
         * We have: sum_resid_k = Σ(h_t - φ*h_{t-1})
         *          sum_resid_sq_k = Σ(h_t - φ*h_{t-1})²
         *
         * Σ(resid - target)² = Σresid² - 2*target*Σresid + n*target²
         */
        double ss = stats->sum_resid_sq_k[k] - 2.0 * target * stats->sum_resid_k[k] + n_k * target * target;

        if (ss < 0.0)
            ss = 0.0; /* Numerical safety */

        double a_post = a0 + n_k / 2.0;
        double b_post = b0 + ss / 2.0;

        /* Sample from InvGamma(a_post, b_post)
         * = 1/Gamma(a_post, 1/b_post)
         */
        double gamma_sample;
        vdRngGamma(VSL_RNG_METHOD_GAMMA_GNORM, stream, 1,
                   &gamma_sample, a_post, 0.0, 1.0 / b_post);

        double sigma_sq = 1.0 / gamma_sample;
        float sigma = sqrtf((float)sigma_sq);

        /* Clamp to reasonable range */
        if (sigma < 0.01f)
            sigma = 0.01f;
        if (sigma > 2.0f)
            sigma = 2.0f;

        pgas->model.sigma_vol[k] = sigma;
    }
}

void pgas_paris_enforce_mu_ordering(PGASParisState *state)
{
    if (!state || !state->pgas)
        return;

    PGASMKLState *pgas = state->pgas;
    const int K = pgas->K;

    /* Build permutation that sorts μ_vol in ascending order */
    int perm[PGAS_MKL_MAX_REGIMES];
    for (int k = 0; k < K; k++)
        perm[k] = k;

    /* Simple insertion sort (K is small) */
    for (int i = 1; i < K; i++)
    {
        int key = perm[i];
        float key_mu = pgas->model.mu_vol[key];
        int j = i - 1;

        while (j >= 0 && pgas->model.mu_vol[perm[j]] > key_mu)
        {
            perm[j + 1] = perm[j];
            j--;
        }
        perm[j + 1] = key;
    }

    /* Check if already sorted */
    int already_sorted = 1;
    for (int k = 0; k < K; k++)
    {
        if (perm[k] != k)
        {
            already_sorted = 0;
            break;
        }
    }
    if (already_sorted)
        return;

    /* Apply permutation to model parameters */
    float new_mu_vol[PGAS_MKL_MAX_REGIMES];
    float new_sigma_vol[PGAS_MKL_MAX_REGIMES];
    float new_mu_shift[PGAS_MKL_MAX_REGIMES];
    float new_trans[PGAS_MKL_MAX_REGIMES * PGAS_MKL_MAX_REGIMES];
    float new_log_trans[PGAS_MKL_MAX_REGIMES * PGAS_MKL_MAX_REGIMES];

    /* Permute μ_vol, σ_vol */
    for (int k = 0; k < K; k++)
    {
        new_mu_vol[k] = pgas->model.mu_vol[perm[k]];
        new_sigma_vol[k] = pgas->model.sigma_vol[perm[k]];
        new_mu_shift[k] = pgas->model.mu_shift[perm[k]];
    }

    /* Permute transition matrix: both rows and columns
     * new_trans[i][j] = old_trans[perm[i]][perm[j]]
     */
    for (int i = 0; i < K; i++)
    {
        for (int j = 0; j < K; j++)
        {
            new_trans[i * K + j] = pgas->model.trans[perm[i] * K + perm[j]];
            new_log_trans[i * K + j] = pgas->model.log_trans[perm[i] * K + perm[j]];
        }
    }

    /* Copy back */
    for (int k = 0; k < K; k++)
    {
        pgas->model.mu_vol[k] = new_mu_vol[k];
        pgas->model.sigma_vol[k] = new_sigma_vol[k];
        pgas->model.mu_shift[k] = new_mu_shift[k];
    }
    for (int i = 0; i < K * K; i++)
    {
        pgas->model.trans[i] = new_trans[i];
        pgas->model.log_trans[i] = new_log_trans[i];
    }

    /* Also update log_trans_T (transposed) */
    for (int i = 0; i < K; i++)
    {
        for (int j = 0; j < K; j++)
        {
            pgas->model.log_trans_T[j * K + i] = pgas->model.log_trans[i * K + j];
        }
    }
}

void pgas_paris_get_mu_vol(const PGASParisState *state, float *mu_out, int K)
{
    if (!state || !state->pgas || !mu_out)
        return;
    int K_copy = (K < state->pgas->K) ? K : state->pgas->K;
    for (int k = 0; k < K_copy; k++)
    {
        mu_out[k] = state->pgas->model.mu_vol[k];
    }
}

void pgas_paris_get_sigma_vol(const PGASParisState *state, float *sigma_out, int K)
{
    if (!state || !state->pgas || !sigma_out)
        return;
    int K_copy = (K < state->pgas->K) ? K : state->pgas->K;
    for (int k = 0; k < K_copy; k++)
    {
        sigma_out[k] = state->pgas->model.sigma_vol[k];
    }
}

float pgas_paris_gibbs_sweep_full(PGASParisState *state,
                                  const PGASParisRegimePrior *prior)
{
    if (!state || !state->pgas)
        return 0.0f;

    /* 1. CSMC forward pass */
    float accept = pgas_mkl_csmc_sweep(state->pgas);

    /* 2. Copy particles to PARIS */
    pgas_paris_copy_particles(state);

    /* 3. PARIS backward smoothing */
    pgas_paris_run_backward(state);

    /* 4. Sample trajectories from smoothed distribution */
    pgas_paris_sample_trajectories(state);

    /* 5. Count transitions from ensemble */
    pgas_paris_count_transitions(state);

    /* 6. Sample Π from Dirichlet posterior */
    pgas_paris_sample_trans_matrix(state);

    /* 7. Regime parameter learning (if prior provided and enabled) */
    if (prior && (prior->learn_mu || prior->learn_sigma))
    {
        PGASParisRegimeStats stats;
        pgas_paris_collect_regime_stats(state, &stats);

        /* Sample μ_vol */
        if (prior->learn_mu)
        {
            pgas_paris_sample_mu_vol(state, &stats, prior);
        }

        /* Sample σ_vol */
        if (prior->learn_sigma)
        {
            pgas_paris_sample_sigma_vol(state, &stats, prior);
        }

        /* Enforce ordering to prevent label switching */
        if (prior->enforce_ordering)
        {
            pgas_paris_enforce_mu_ordering(state);
        }
    }

    state->total_sweeps++;

    return accept;
}