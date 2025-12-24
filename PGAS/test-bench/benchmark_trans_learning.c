/*═══════════════════════════════════════════════════════════════════════════════
 * Transition Matrix Learning Benchmark: PGAS vs PGAS-PARIS
 *
 * Compares speed and accuracy of Π estimation for runtime Lifeboat updates.
 * This is the "fast path" - μ_vol is fixed, only Π is learned.
 *
 * Metrics:
 *   - Time per sweep (μs)
 *   - Π Frobenius distance from ground truth
 *   - Π diagonal accuracy (stickiness)
 *   - Variance across multiple runs
 *
 * Usage:
 *   ./benchmark_trans_learning [sweeps] [seed]
 *
 *═══════════════════════════════════════════════════════════════════════════════*/

#include "pgas_mkl.h"
#include "pgas_paris.h"
#include "mkl_tuning.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include <mkl.h>
#include <mkl_vsl.h>

#ifdef _WIN32
#include <windows.h>
static double get_time_ms(void) {
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart * 1000.0 / (double)freq.QuadPart;
}
#else
#include <sys/time.h>
static double get_time_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}
#endif

/*═══════════════════════════════════════════════════════════════════════════════
 * CONFIGURATION
 *═══════════════════════════════════════════════════════════════════════════════*/

#define DEFAULT_SEED 42
#define DEFAULT_SWEEPS 100
#define T_DATA 2000
#define K_REGIMES 4
#define M_TRAJECTORIES 8
#define N_RUNS 5  /* Multiple runs for variance estimation */

/* Test configurations */
static const int TEST_N[] = {32, 64};
#define N_TEST_CONFIGS 2

/* Ground truth parameters */
static const float TRUE_MU_VOL[4] = {-4.50f, -3.67f, -2.83f, -2.00f};
static const float TRUE_SIGMA_VOL[4] = {0.08f, 0.267f, 0.453f, 0.64f};
static const float TRUE_PHI = 0.97f;
static const float TRUE_SIGMA_H = 0.15f;

/* True transition matrix (sticky) */
static const double TRUE_TRANS[16] = {
    0.95, 0.02, 0.02, 0.01,
    0.02, 0.94, 0.02, 0.02,
    0.02, 0.02, 0.94, 0.02,
    0.01, 0.02, 0.02, 0.95
};

/* OCSN noise sampler */
static float sample_ocsn_noise(VSLStreamStatePtr stream)
{
    static const float Q[10] = {
        0.00609f, 0.04775f, 0.13057f, 0.20674f, 0.22715f,
        0.18842f, 0.12047f, 0.05591f, 0.01575f, 0.00115f
    };
    static const float M[10] = {
        1.92677f, 1.34744f, 0.73504f, 0.02266f, -0.85173f,
        -1.97278f, -3.46788f, -5.55246f, -8.68384f, -14.65000f
    };
    static const float S[10] = {
        0.33563f, 0.42175f, 0.51737f, 0.63728f, 0.79183f,
        0.99289f, 1.25487f, 1.59530f, 2.04106f, 2.70806f
    };
    
    float u;
    vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, 1, &u, 0.0f, 1.0f);
    
    int j = 9;
    float cumsum = 0.0f;
    for (int k = 0; k < 10; k++) {
        cumsum += Q[k];
        if (u < cumsum) { j = k; break; }
    }
    
    float z;
    vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, stream, 1, &z, M[j], S[j]);
    return z;
}

/*═══════════════════════════════════════════════════════════════════════════════
 * SYNTHETIC DATA
 *═══════════════════════════════════════════════════════════════════════════════*/

typedef struct {
    int T;
    int *true_regimes;
    float *true_h;
    float *y;
} SyntheticData;

static void generate_synthetic_data(SyntheticData *data, int T, uint32_t seed)
{
    data->T = T;
    data->true_regimes = (int*)malloc(T * sizeof(int));
    data->true_h = (float*)malloc(T * sizeof(float));
    data->y = (float*)malloc(T * sizeof(float));
    
    VSLStreamStatePtr stream;
    vslNewStream(&stream, VSL_BRNG_MT19937, seed);
    
    data->true_regimes[0] = 1;
    data->true_h[0] = TRUE_MU_VOL[1];
    
    float *uniform = (float*)malloc(T * sizeof(float));
    float *normal = (float*)malloc(T * sizeof(float));
    
    vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, T, uniform, 0.0f, 1.0f);
    vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, stream, T, normal, 0.0f, TRUE_SIGMA_H);
    
    for (int t = 1; t < T; t++) {
        int prev_regime = data->true_regimes[t-1];
        float u = uniform[t];
        float cumsum = 0.0f;
        int new_regime = K_REGIMES - 1;
        
        for (int k = 0; k < K_REGIMES; k++) {
            cumsum += (float)TRUE_TRANS[prev_regime * K_REGIMES + k];
            if (u < cumsum) { new_regime = k; break; }
        }
        data->true_regimes[t] = new_regime;
        
        float mu_k = TRUE_MU_VOL[new_regime];
        data->true_h[t] = mu_k * (1.0f - TRUE_PHI) + TRUE_PHI * data->true_h[t-1] + normal[t];
    }
    
    for (int t = 0; t < T; t++) {
        data->y[t] = data->true_h[t] + sample_ocsn_noise(stream);
    }
    
    free(uniform);
    free(normal);
    vslDeleteStream(&stream);
}

static void free_synthetic_data(SyntheticData *data)
{
    free(data->true_regimes);
    free(data->true_h);
    free(data->y);
}

/*═══════════════════════════════════════════════════════════════════════════════
 * METRICS
 *═══════════════════════════════════════════════════════════════════════════════*/

static double frobenius_distance(const float *A, const double *B, int K)
{
    double sum = 0.0;
    for (int i = 0; i < K * K; i++) {
        double diff = (double)A[i] - B[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

static double avg_diagonal(const float *trans, int K)
{
    double sum = 0.0;
    for (int k = 0; k < K; k++) {
        sum += trans[k * K + k];
    }
    return sum / K;
}

/*═══════════════════════════════════════════════════════════════════════════════
 * BENCHMARK RESULTS
 *═══════════════════════════════════════════════════════════════════════════════*/

typedef struct {
    double time_ms;
    double time_per_sweep_us;
    double frobenius;
    double avg_diag;
    float trans[16];
} RunResult;

typedef struct {
    int N;
    int sweeps;
    
    /* Aggregated over N_RUNS */
    double avg_time_ms;
    double std_time_ms;
    double avg_time_per_sweep_us;
    double avg_frobenius;
    double std_frobenius;
    double avg_diag;
    
    RunResult runs[N_RUNS];
} BenchmarkResult;

/*═══════════════════════════════════════════════════════════════════════════════
 * VANILLA PGAS BENCHMARK
 *═══════════════════════════════════════════════════════════════════════════════*/

static void benchmark_vanilla_pgas(
    const SyntheticData *data,
    int N, int sweeps, uint32_t seed,
    BenchmarkResult *result)
{
    result->N = N;
    result->sweeps = sweeps;
    
    double *obs_d = (double*)malloc(T_DATA * sizeof(double));
    for (int t = 0; t < T_DATA; t++) obs_d[t] = (double)data->y[t];
    
    double init_trans[16];
    for (int i = 0; i < K_REGIMES; i++) {
        for (int j = 0; j < K_REGIMES; j++) {
            init_trans[i * K_REGIMES + j] = (i == j) ? 0.85 : 0.05;
        }
    }
    
    double mu_vol_d[4], sigma_vol_d[4];
    for (int k = 0; k < 4; k++) {
        mu_vol_d[k] = TRUE_MU_VOL[k];  /* Fixed - not learning */
        sigma_vol_d[k] = TRUE_SIGMA_VOL[k];
    }
    
    for (int run = 0; run < N_RUNS; run++) {
        PGASMKLState *pgas = pgas_mkl_alloc(N, T_DATA, K_REGIMES, seed + run * 1000);
        pgas_mkl_set_model(pgas, init_trans, mu_vol_d, sigma_vol_d, TRUE_PHI, TRUE_SIGMA_H);
        pgas_mkl_set_transition_prior(pgas, 1.0f, 50.0f);
        pgas_mkl_load_observations(pgas, obs_d, T_DATA);
        
        /* Initialize reference with true trajectory */
        double *init_h = (double*)malloc(T_DATA * sizeof(double));
        for (int t = 0; t < T_DATA; t++) init_h[t] = data->true_h[t];
        pgas_mkl_set_reference(pgas, data->true_regimes, init_h, T_DATA);
        
        double start = get_time_ms();
        
        for (int s = 0; s < sweeps; s++) {
            pgas_mkl_csmc_sweep(pgas);
            pgas_mkl_sample_transitions(pgas);
        }
        
        double end = get_time_ms();
        
        result->runs[run].time_ms = end - start;
        result->runs[run].time_per_sweep_us = (end - start) * 1000.0 / sweeps;
        result->runs[run].frobenius = frobenius_distance(pgas->model.trans, TRUE_TRANS, K_REGIMES);
        result->runs[run].avg_diag = avg_diagonal(pgas->model.trans, K_REGIMES);
        memcpy(result->runs[run].trans, pgas->model.trans, 16 * sizeof(float));
        
        free(init_h);
        pgas_mkl_free(pgas);
    }
    
    /* Aggregate */
    double sum_time = 0, sum_time_sq = 0;
    double sum_frob = 0, sum_frob_sq = 0;
    double sum_diag = 0;
    
    for (int r = 0; r < N_RUNS; r++) {
        sum_time += result->runs[r].time_ms;
        sum_time_sq += result->runs[r].time_ms * result->runs[r].time_ms;
        sum_frob += result->runs[r].frobenius;
        sum_frob_sq += result->runs[r].frobenius * result->runs[r].frobenius;
        sum_diag += result->runs[r].avg_diag;
    }
    
    result->avg_time_ms = sum_time / N_RUNS;
    result->std_time_ms = sqrt(sum_time_sq / N_RUNS - result->avg_time_ms * result->avg_time_ms);
    result->avg_time_per_sweep_us = result->avg_time_ms * 1000.0 / sweeps;
    result->avg_frobenius = sum_frob / N_RUNS;
    result->std_frobenius = sqrt(sum_frob_sq / N_RUNS - result->avg_frobenius * result->avg_frobenius);
    result->avg_diag = sum_diag / N_RUNS;
    
    free(obs_d);
}

/*═══════════════════════════════════════════════════════════════════════════════
 * PGAS-PARIS BENCHMARK
 *═══════════════════════════════════════════════════════════════════════════════*/

static void benchmark_pgas_paris(
    const SyntheticData *data,
    int N, int sweeps, uint32_t seed,
    BenchmarkResult *result)
{
    result->N = N;
    result->sweeps = sweeps;
    
    double *obs_d = (double*)malloc(T_DATA * sizeof(double));
    for (int t = 0; t < T_DATA; t++) obs_d[t] = (double)data->y[t];
    
    double init_trans[16];
    for (int i = 0; i < K_REGIMES; i++) {
        for (int j = 0; j < K_REGIMES; j++) {
            init_trans[i * K_REGIMES + j] = (i == j) ? 0.85 : 0.05;
        }
    }
    
    double mu_vol_d[4], sigma_vol_d[4];
    for (int k = 0; k < 4; k++) {
        mu_vol_d[k] = TRUE_MU_VOL[k];
        sigma_vol_d[k] = TRUE_SIGMA_VOL[k];
    }
    
    for (int run = 0; run < N_RUNS; run++) {
        PGASMKLState *pgas = pgas_mkl_alloc(N, T_DATA, K_REGIMES, seed + run * 1000);
        pgas_mkl_set_model(pgas, init_trans, mu_vol_d, sigma_vol_d, TRUE_PHI, TRUE_SIGMA_H);
        pgas_mkl_set_transition_prior(pgas, 1.0f, 50.0f);
        pgas_mkl_load_observations(pgas, obs_d, T_DATA);
        
        double *init_h = (double*)malloc(T_DATA * sizeof(double));
        for (int t = 0; t < T_DATA; t++) init_h[t] = data->true_h[t];
        pgas_mkl_set_reference(pgas, data->true_regimes, init_h, T_DATA);
        
        PGASParisState *pp = pgas_paris_alloc(pgas, M_TRAJECTORIES);
        
        double start = get_time_ms();
        
        for (int s = 0; s < sweeps; s++) {
            pgas_paris_gibbs_sweep(pp);  /* CSMC + PARIS backward + Π sampling */
        }
        
        double end = get_time_ms();
        
        result->runs[run].time_ms = end - start;
        result->runs[run].time_per_sweep_us = (end - start) * 1000.0 / sweeps;
        result->runs[run].frobenius = frobenius_distance(pgas->model.trans, TRUE_TRANS, K_REGIMES);
        result->runs[run].avg_diag = avg_diagonal(pgas->model.trans, K_REGIMES);
        memcpy(result->runs[run].trans, pgas->model.trans, 16 * sizeof(float));
        
        pgas_paris_free(pp);
        free(init_h);
        pgas_mkl_free(pgas);
    }
    
    /* Aggregate */
    double sum_time = 0, sum_time_sq = 0;
    double sum_frob = 0, sum_frob_sq = 0;
    double sum_diag = 0;
    
    for (int r = 0; r < N_RUNS; r++) {
        sum_time += result->runs[r].time_ms;
        sum_time_sq += result->runs[r].time_ms * result->runs[r].time_ms;
        sum_frob += result->runs[r].frobenius;
        sum_frob_sq += result->runs[r].frobenius * result->runs[r].frobenius;
        sum_diag += result->runs[r].avg_diag;
    }
    
    result->avg_time_ms = sum_time / N_RUNS;
    result->std_time_ms = sqrt(sum_time_sq / N_RUNS - result->avg_time_ms * result->avg_time_ms);
    result->avg_time_per_sweep_us = result->avg_time_ms * 1000.0 / sweeps;
    result->avg_frobenius = sum_frob / N_RUNS;
    result->std_frobenius = sqrt(sum_frob_sq / N_RUNS - result->avg_frobenius * result->avg_frobenius);
    result->avg_diag = sum_diag / N_RUNS;
    
    free(obs_d);
}

/*═══════════════════════════════════════════════════════════════════════════════
 * MAIN
 *═══════════════════════════════════════════════════════════════════════════════*/

int main(int argc, char **argv)
{
    int sweeps = DEFAULT_SWEEPS;
    uint32_t seed = DEFAULT_SEED;
    
    if (argc > 1) sweeps = atoi(argv[1]);
    if (argc > 2) seed = (uint32_t)atoi(argv[2]);
    
    /* Initialize MKL tuning */
    mkl_tuning_init(8, 1);
    
    printf("\n");
    printf("═══════════════════════════════════════════════════════════════════════════\n");
    printf("  Transition Matrix Learning Benchmark: PGAS vs PGAS-PARIS\n");
    printf("═══════════════════════════════════════════════════════════════════════════\n\n");
    
    printf("Configuration:\n");
    printf("  T (timesteps):         %d\n", T_DATA);
    printf("  K (regimes):           %d\n", K_REGIMES);
    printf("  Sweeps:                %d\n", sweeps);
    printf("  PARIS trajectories:    %d\n", M_TRAJECTORIES);
    printf("  Runs per config:       %d\n", N_RUNS);
    printf("  Seed:                  %u\n", seed);
    
    printf("\nTrue Π diagonal:         [0.95, 0.94, 0.94, 0.95]\n");
    
    /* Generate data once */
    printf("\nGenerating synthetic data...\n");
    SyntheticData data;
    generate_synthetic_data(&data, T_DATA, seed);
    
    /* Results storage */
    BenchmarkResult pgas_results[N_TEST_CONFIGS];
    BenchmarkResult paris_results[N_TEST_CONFIGS];
    
    /* Run benchmarks */
    for (int c = 0; c < N_TEST_CONFIGS; c++) {
        int N = TEST_N[c];
        
        printf("\n───────────────────────────────────────────────────────────────────────────\n");
        printf("  N = %d particles\n", N);
        printf("───────────────────────────────────────────────────────────────────────────\n");
        
        printf("  Running Vanilla PGAS (%d runs)...\n", N_RUNS);
        benchmark_vanilla_pgas(&data, N, sweeps, seed, &pgas_results[c]);
        printf("    Avg time: %.1f ms (%.1f μs/sweep)\n", 
               pgas_results[c].avg_time_ms, pgas_results[c].avg_time_per_sweep_us);
        
        printf("  Running PGAS-PARIS (%d runs)...\n", N_RUNS);
        benchmark_pgas_paris(&data, N, sweeps, seed, &paris_results[c]);
        printf("    Avg time: %.1f ms (%.1f μs/sweep)\n",
               paris_results[c].avg_time_ms, paris_results[c].avg_time_per_sweep_us);
    }
    
    /* Summary table */
    printf("\n");
    printf("═══════════════════════════════════════════════════════════════════════════\n");
    printf("  RESULTS SUMMARY (%d sweeps, %d runs averaged)\n", sweeps, N_RUNS);
    printf("═══════════════════════════════════════════════════════════════════════════\n\n");
    
    printf("┌────────┬─────────────────────────────────────┬─────────────────────────────────────┐\n");
    printf("│   N    │          Vanilla PGAS               │            PGAS-PARIS               │\n");
    printf("├────────┼──────────┬──────────┬───────┬───────┼──────────┬──────────┬───────┬───────┤\n");
    printf("│        │ Time(ms) │  μs/swp  │ Frob  │ Diag  │ Time(ms) │  μs/swp  │ Frob  │ Diag  │\n");
    printf("├────────┼──────────┼──────────┼───────┼───────┼──────────┼──────────┼───────┼───────┤\n");
    
    for (int c = 0; c < N_TEST_CONFIGS; c++) {
        printf("│   %2d   │ %7.1f  │ %7.0f  │ %5.3f │ %5.3f │ %7.1f  │ %7.0f  │ %5.3f │ %5.3f │\n",
               TEST_N[c],
               pgas_results[c].avg_time_ms,
               pgas_results[c].avg_time_per_sweep_us,
               pgas_results[c].avg_frobenius,
               pgas_results[c].avg_diag,
               paris_results[c].avg_time_ms,
               paris_results[c].avg_time_per_sweep_us,
               paris_results[c].avg_frobenius,
               paris_results[c].avg_diag);
    }
    
    printf("└────────┴──────────┴──────────┴───────┴───────┴──────────┴──────────┴───────┴───────┘\n");
    
    /* Speedup and accuracy comparison */
    printf("\n");
    printf("┌────────┬────────────────┬────────────────┬─────────────────────────────────────────┐\n");
    printf("│   N    │ PARIS Slowdown │ Frob Δ (lower  │                 Winner                  │\n");
    printf("│        │   vs PGAS      │   = better)    │                                         │\n");
    printf("├────────┼────────────────┼────────────────┼─────────────────────────────────────────┤\n");
    
    for (int c = 0; c < N_TEST_CONFIGS; c++) {
        double slowdown = paris_results[c].avg_time_ms / pgas_results[c].avg_time_ms;
        double frob_diff = pgas_results[c].avg_frobenius - paris_results[c].avg_frobenius;
        const char *winner = (frob_diff > 0.01) ? "PARIS (more accurate)" :
                             (frob_diff < -0.01) ? "PGAS (more accurate)" : "TIE";
        
        printf("│   %2d   │     %.2fx       │    %+.3f       │ %-39s │\n",
               TEST_N[c], slowdown, frob_diff, winner);
    }
    
    printf("└────────┴────────────────┴────────────────┴─────────────────────────────────────────┘\n");
    
    /* Variance comparison */
    printf("\n");
    printf("Estimation Variance (Frobenius std):\n");
    for (int c = 0; c < N_TEST_CONFIGS; c++) {
        printf("  N=%2d: PGAS=%.4f, PARIS=%.4f  %s\n",
               TEST_N[c],
               pgas_results[c].std_frobenius,
               paris_results[c].std_frobenius,
               paris_results[c].std_frobenius < pgas_results[c].std_frobenius ? 
               "(PARIS more stable)" : "(PGAS more stable)");
    }
    
    /* Best learned Π from PARIS */
    printf("\n");
    printf("Best PARIS Learned Π (N=%d, run 0):\n", TEST_N[0]);
    for (int i = 0; i < K_REGIMES; i++) {
        printf("  [");
        for (int j = 0; j < K_REGIMES; j++) {
            printf(" %5.3f", paris_results[0].runs[0].trans[i * K_REGIMES + j]);
        }
        printf(" ]\n");
    }
    
    printf("\nTrue Π:\n");
    for (int i = 0; i < K_REGIMES; i++) {
        printf("  [");
        for (int j = 0; j < K_REGIMES; j++) {
            printf(" %5.3f", TRUE_TRANS[i * K_REGIMES + j]);
        }
        printf(" ]\n");
    }
    
    /* Recommendations */
    printf("\n═══════════════════════════════════════════════════════════════════════════\n");
    printf("  RECOMMENDATIONS FOR PRODUCTION\n");
    printf("═══════════════════════════════════════════════════════════════════════════\n\n");
    
    int best_n = TEST_N[0];
    double best_paris_time = paris_results[0].avg_time_ms;
    
    for (int c = 1; c < N_TEST_CONFIGS; c++) {
        if (paris_results[c].avg_frobenius < paris_results[0].avg_frobenius * 0.9) {
            /* Significantly better accuracy */
            best_n = TEST_N[c];
            best_paris_time = paris_results[c].avg_time_ms;
        }
    }
    
    printf("  For Lifeboat Π updates (accuracy priority):\n");
    printf("    → Use PGAS-PARIS with N=%d, %d sweeps\n", best_n, sweeps);
    printf("    → Expected time: %.0f ms\n", best_paris_time);
    printf("    → Time per sweep: %.0f μs\n\n", best_paris_time * 1000.0 / sweeps);
    
    printf("  For fast Π updates (speed priority):\n");
    printf("    → Use Vanilla PGAS with N=%d, %d sweeps\n", TEST_N[0], sweeps);
    printf("    → Expected time: %.0f ms\n", pgas_results[0].avg_time_ms);
    printf("    → Time per sweep: %.0f μs\n", pgas_results[0].avg_time_per_sweep_us);
    
    printf("\n═══════════════════════════════════════════════════════════════════════════\n");
    
    /* Cleanup */
    free_synthetic_data(&data);
    mkl_tuning_cleanup();
    
    return 0;
}
