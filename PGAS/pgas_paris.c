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

#include <mkl.h>
#include <mkl_vsl.h>

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
    if (!state->h_double || !state->weights_double)
    {
        if (state->h_double)
            mkl_free(state->h_double);
        if (state->weights_double)
            mkl_free(state->weights_double);
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
    if (state->traj.trajectories)
    {
        mkl_free(state->traj.trajectories);
    }

    free(state);
}

/*═══════════════════════════════════════════════════════════════════════════════
 * PARTICLE COPY: PGAS → PARIS
 *
 * PGAS stores particles as [T × N_padded] with float h and float weights.
 * PARIS load_particles expects [T × N] with double h and double weights.
 *
 * We need to:
 *   1. Convert float → double
 *   2. Remove padding (N_padded → N)
 *   3. Convert normalized weights to PARIS expected format
 *═══════════════════════════════════════════════════════════════════════════════*/

void pgas_paris_copy_particles(PGASParisState *state)
{
    if (!state || !state->pgas || !state->paris)
        return;

    PGASMKLState *pgas = state->pgas;
    const int T = pgas->T;
    const int N = pgas->N;
    const int Np = pgas->N_padded;

    /* Copy with stride conversion and float→double */
    for (int t = 0; t < T; t++)
    {
        for (int n = 0; n < N; n++)
        {
            int pgas_idx = t * Np + n;
            int paris_idx = t * N + n;

            state->h_double[paris_idx] = (double)pgas->h[pgas_idx];
            state->weights_double[paris_idx] = (double)pgas->weights[pgas_idx];
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

    /* Load particles into PARIS
     * Note: paris_mkl_load_particles expects:
     *   - regimes [T×N] int
     *   - h [T×N] double
     *   - weights [T×N] double (normalized, will convert to log internally)
     *   - ancestors [T×N] int
     */

    /* Prepare regimes and ancestors without padding */
    int *regimes_nopad = (int *)mkl_malloc((size_t)T * N * sizeof(int), 64);
    int *ancestors_nopad = (int *)mkl_malloc((size_t)T * N * sizeof(int), 64);

    for (int t = 0; t < T; t++)
    {
        for (int n = 0; n < N; n++)
        {
            int pgas_idx = t * Np + n;
            int paris_idx = t * N + n;

            regimes_nopad[paris_idx] = pgas->regimes[pgas_idx];
            ancestors_nopad[paris_idx] = pgas->ancestors[pgas_idx];

            /* Clamp ancestor to valid range (in case of padding artifacts) */
            if (ancestors_nopad[paris_idx] >= N)
            {
                ancestors_nopad[paris_idx] = N - 1;
            }
        }
    }

    paris_mkl_load_particles(state->paris,
                             regimes_nopad,
                             state->h_double,
                             state->weights_double,
                             ancestors_nopad,
                             T);

    mkl_free(regimes_nopad);
    mkl_free(ancestors_nopad);
}

/*═══════════════════════════════════════════════════════════════════════════════
 * BACKWARD SMOOTHING
 *═══════════════════════════════════════════════════════════════════════════════*/

void pgas_paris_backward_smooth(PGASParisState *state)
{
    if (!state || !state->paris)
        return;

    paris_mkl_backward_smooth(state->paris);
}

/*═══════════════════════════════════════════════════════════════════════════════
 * TRAJECTORY SAMPLING
 *
 * After PARIS backward pass, state->paris->smoothed[t * Np + n] contains
 * the INDEX of the particle at time t for the n-th smoothed trajectory.
 *
 * We sample M trajectories by picking M different starting particles
 * and extracting their regime sequences.
 *═══════════════════════════════════════════════════════════════════════════════*/

void pgas_paris_sample_trajectories(PGASParisState *state)
{
    if (!state || !state->paris)
        return;

    const int T = state->traj.T;
    const int M = state->traj.M;
    const int N = state->pgas->N;

    /* Sample M particle indices (evenly spaced for diversity) */
    int stride = (N > M) ? (N / M) : 1;

    for (int m = 0; m < M; m++)
    {
        int particle_idx = (m * stride) % N;

        /* Extract regime trajectory using paris_mkl_get_trajectory */
        int *traj_regimes = &state->traj.trajectories[m * T];
        paris_mkl_get_trajectory(state->paris, particle_idx, traj_regimes, NULL);
    }

    /* Compute trajectory diversity */
    int unique_count = 0;
    for (int m = 0; m < M; m++)
    {
        int is_unique = 1;
        for (int m2 = 0; m2 < m; m2++)
        {
            int same = 1;
            for (int t = 0; t < T && same; t++)
            {
                if (state->traj.trajectories[m * T + t] !=
                    state->traj.trajectories[m2 * T + t])
                {
                    same = 0;
                }
            }
            if (same)
            {
                is_unique = 0;
                break;
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
 * Sample from Dirichlet posterior using PARIS ensemble counts.
 * This replaces pgas_mkl_sample_transitions() when using PARIS.
 *═══════════════════════════════════════════════════════════════════════════════*/

void pgas_paris_sample_trans_matrix(PGASParisState *state)
{
    if (!state || !state->pgas)
        return;

    PGASMKLState *pgas = state->pgas;
    const int K = pgas->K;
    const float alpha = pgas->prior_alpha;
    const float kappa = pgas->sticky_kappa;

    VSLStreamStatePtr stream = (VSLStreamStatePtr)pgas->rng.stream;

    /* Sample each row from Dirichlet */
    for (int i = 0; i < K; i++)
    {
        double gamma_samples[PGAS_MKL_MAX_REGIMES];
        double row_sum = 0.0;

        for (int j = 0; j < K; j++)
        {
            /* Dirichlet parameter: α + n_ij + κ·δ_{ij} */
            double dir_alpha = alpha + (double)state->traj.n_trans[i * K + j];
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
     * Compare PARIS ensemble counts to expected from stickiness prior */
    int total_self = 0;
    int total_switch = 0;

    for (int i = 0; i < K; i++)
    {
        total_self += state->traj.n_trans[i * K + i];
        for (int j = 0; j < K; j++)
        {
            if (i != j)
            {
                total_switch += state->traj.n_trans[i * K + j];
            }
        }
    }

    /* Expected self-transition rate from current κ */
    float expected_self_rate = kappa / (kappa + K - 1 + K * alpha);
    float actual_self_rate = (total_self + total_switch > 0)
                                 ? (float)total_self / (total_self + total_switch)
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

        /* Adjust κ based on smoothed chatter */
        float smoothed_chatter = pgas->rls_chatter_estimate;

        if (smoothed_chatter > 1.05f)
        {
            /* Too many switches → increase κ */
            pgas->sticky_kappa *= (1.0f + pgas->kappa_up_rate);
        }
        else if (smoothed_chatter < 0.95f)
        {
            /* Too few switches → decrease κ */
            pgas->sticky_kappa *= (1.0f - pgas->kappa_down_rate);
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
    pgas_paris_backward_smooth(state);

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
    printf("Trajectories sampled: %d\n", state->n_trajectories);
    printf("Trajectory diversity: %.2f%%\n", state->traj.trajectory_diversity * 100.0f);
    printf("Path degeneracy:      %.2f%%\n", pgas_paris_get_path_degeneracy(state) * 100.0f);
    printf("CSMC acceptance:      %.3f\n", pgas_mkl_get_acceptance_rate(state->pgas));
    printf("Chatter ratio:        %.2f\n", pgas_mkl_get_chatter_ratio(state->pgas));
    printf("Sticky κ:             %.1f\n", pgas_mkl_get_sticky_kappa(state->pgas));

    printf("\nPARIS ensemble transition counts:\n");
    int K = state->traj.K;
    for (int i = 0; i < K; i++)
    {
        printf("  [");
        for (int j = 0; j < K; j++)
        {
            printf(" %5d", state->traj.n_trans[i * K + j]);
        }
        printf(" ]\n");
    }
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