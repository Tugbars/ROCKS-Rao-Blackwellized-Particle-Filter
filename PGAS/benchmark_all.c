/**
 * @file benchmark_all.c
 * @brief Comprehensive benchmark: Original vs Fast vs MKL
 *
 * Compile:
 *   Original/Fast: gcc -O3 -march=native -mavx2 -fopenmp ...
 *   MKL:           gcc -O3 -march=native -mavx2 -fopenmp -I${MKLROOT}/include \
 *                  -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_gnu_thread \
 *                  -lmkl_core -lgomp -lpthread -lm -ldl
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* Include all implementations */
#include "pgas.h"
#include "paris_fast.h"
#include "pgas_mkl.h"
#include "circular_buffer.h"

#define TRIALS 10
#define WARMUP 2

/*═══════════════════════════════════════════════════════════════════════════════
 * TIMING
 *═══════════════════════════════════════════════════════════════════════════════*/

static double get_time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

/*═══════════════════════════════════════════════════════════════════════════════
 * DATA GENERATION
 *═══════════════════════════════════════════════════════════════════════════════*/

typedef struct {
    double *observations;
    int *true_regimes;
    double *true_h;
    double trans[64];
    double mu_vol[8];
    double sigma_vol[8];
    int T, K;
} TestData;

static TestData generate_test_data(int T, int K, uint32_t seed)
{
    TestData data;
    data.T = T;
    data.K = K;
    data.observations = malloc(T * sizeof(double));
    data.true_regimes = malloc(T * sizeof(int));
    data.true_h = malloc(T * sizeof(double));
    
    srand(seed);
    
    /* Sticky transition matrix */
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < K; j++) {
            data.trans[i * K + j] = (i == j) ? 0.95 : 0.05 / (K - 1);
        }
        data.mu_vol[i] = -2.0 + i * 1.0;
        data.sigma_vol[i] = 0.3;
    }
    
    double phi = 0.97;
    double sigma_h = 0.15;
    
    /* Generate trajectory */
    data.true_regimes[0] = 0;
    data.true_h[0] = data.mu_vol[0];
    data.observations[0] = exp(data.true_h[0] / 2.0) * ((double)rand() / RAND_MAX * 2 - 1);
    
    for (int t = 1; t < T; t++) {
        /* Sample regime */
        double u = (double)rand() / RAND_MAX;
        double cumsum = 0;
        int prev = data.true_regimes[t-1];
        data.true_regimes[t] = K - 1;
        for (int j = 0; j < K; j++) {
            cumsum += data.trans[prev * K + j];
            if (u < cumsum) {
                data.true_regimes[t] = j;
                break;
            }
        }
        
        /* Sample h */
        int regime = data.true_regimes[t];
        double mu = data.mu_vol[regime];
        double mean = mu + phi * (data.true_h[t-1] - mu);
        double noise = ((double)rand() / RAND_MAX * 2 - 1) * sigma_h * 1.73;
        data.true_h[t] = mean + noise;
        
        /* Sample observation */
        double vol = exp(data.true_h[t] / 2.0);
        data.observations[t] = vol * ((double)rand() / RAND_MAX * 2 - 1) * 1.73;
    }
    
    return data;
}

static void free_test_data(TestData *data)
{
    free(data->observations);
    free(data->true_regimes);
    free(data->true_h);
}

/*═══════════════════════════════════════════════════════════════════════════════
 * BENCHMARK FUNCTIONS
 *═══════════════════════════════════════════════════════════════════════════════*/

typedef struct {
    double pgas_ms;
    double paris_ms;
    double total_ms;
    float acceptance;
    int sweeps;
} BenchResult;

/**
 * Benchmark Original PGAS + PARIS (from pgas.c)
 */
static BenchResult bench_original(const TestData *data, int N, int trial)
{
    BenchResult result = {0};
    
    PGASState pgas;
    pgas_init(&pgas, N, data->K, trial);
    pgas_set_model(&pgas, data->trans, data->mu_vol, data->sigma_vol, 0.97, 0.15);
    pgas_set_reference(&pgas, data->true_regimes, data->true_h, data->T);
    
    /* Load observations via BufferSnapshot */
    BufferSnapshot snap;
    snap.count = data->T;
    for (int t = 0; t < data->T; t++) {
        snap.observations[t] = data->observations[t];
        snap.tick_ids[t] = t;
    }
    pgas_load_observations(&pgas, &snap);
    
    /* Time PGAS */
    double t0 = get_time_us();
    pgas_run_adaptive(&pgas);
    double t1 = get_time_us();
    result.pgas_ms = (t1 - t0) / 1000.0;
    result.acceptance = pgas.acceptance_rate;
    result.sweeps = pgas.current_sweep;
    
    /* Time PARIS */
    t0 = get_time_us();
    paris_backward_smooth(&pgas);
    t1 = get_time_us();
    result.paris_ms = (t1 - t0) / 1000.0;
    
    result.total_ms = result.pgas_ms + result.paris_ms;
    return result;
}

/**
 * Benchmark Fast PARIS (from paris_fast.c)
 */
static BenchResult bench_fast(const TestData *data, int N, int trial)
{
    BenchResult result = {0};
    
    /* Use Original PGAS (since Fast only optimizes PARIS) */
    PGASState pgas;
    pgas_init(&pgas, N, data->K, trial);
    pgas_set_model(&pgas, data->trans, data->mu_vol, data->sigma_vol, 0.97, 0.15);
    pgas_set_reference(&pgas, data->true_regimes, data->true_h, data->T);
    
    BufferSnapshot snap;
    snap.count = data->T;
    for (int t = 0; t < data->T; t++) {
        snap.observations[t] = data->observations[t];
        snap.tick_ids[t] = t;
    }
    pgas_load_observations(&pgas, &snap);
    
    double t0 = get_time_us();
    pgas_run_adaptive(&pgas);
    double t1 = get_time_us();
    result.pgas_ms = (t1 - t0) / 1000.0;
    result.acceptance = pgas.acceptance_rate;
    result.sweeps = pgas.current_sweep;
    
    /* Prepare data for Fast PARIS */
    int T = data->T;
    int *regimes = malloc(T * N * sizeof(int));
    double *h = malloc(T * N * sizeof(double));
    double *weights = malloc(T * N * sizeof(double));
    int *ancestors = malloc(T * N * sizeof(int));
    
    for (int t = 0; t < T; t++) {
        for (int n = 0; n < N; n++) {
            regimes[t * N + n] = pgas.regimes[t][n];
            h[t * N + n] = pgas.h[t][n];
            weights[t * N + n] = pgas.weights[t][n];
            ancestors[t * N + n] = pgas.ancestors[t][n];
        }
    }
    
    PARISState *paris = paris_alloc(N, T, data->K, trial + 1000);
    paris_set_model(paris, data->trans, data->mu_vol, 0.97, 0.15);
    paris_load_particles(paris, regimes, h, weights, ancestors, T);
    
    t0 = get_time_us();
    paris_backward_smooth_fast(paris);
    t1 = get_time_us();
    result.paris_ms = (t1 - t0) / 1000.0;
    
    paris_free(paris);
    free(regimes);
    free(h);
    free(weights);
    free(ancestors);
    
    result.total_ms = result.pgas_ms + result.paris_ms;
    return result;
}

/**
 * Benchmark MKL PGAS + PARIS
 */
static BenchResult bench_mkl(const TestData *data, int N, int trial)
{
    BenchResult result = {0};
    
    PGASMKLState *state = pgas_mkl_alloc(N, data->T, data->K, trial);
    pgas_mkl_set_model(state, data->trans, data->mu_vol, data->sigma_vol, 0.97, 0.15);
    pgas_mkl_set_reference(state, data->true_regimes, data->true_h, data->T);
    pgas_mkl_load_observations(state, data->observations, data->T);
    
    /* Time PGAS */
    double t0 = get_time_us();
    pgas_mkl_run_adaptive(state);
    double t1 = get_time_us();
    result.pgas_ms = (t1 - t0) / 1000.0;
    result.acceptance = state->acceptance_rate;
    result.sweeps = state->current_sweep;
    
    /* Time PARIS */
    t0 = get_time_us();
    paris_mkl_backward_smooth(state);
    t1 = get_time_us();
    result.paris_ms = (t1 - t0) / 1000.0;
    
    pgas_mkl_free(state);
    
    result.total_ms = result.pgas_ms + result.paris_ms;
    return result;
}

/*═══════════════════════════════════════════════════════════════════════════════
 * MAIN BENCHMARK
 *═══════════════════════════════════════════════════════════════════════════════*/

int main(void)
{
    printf("╔═══════════════════════════════════════════════════════════════════════╗\n");
    printf("║           PGAS + PARIS COMPREHENSIVE BENCHMARK                        ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════════╝\n\n");
    
    #ifdef _OPENMP
    printf("OpenMP: %d threads\n", omp_get_max_threads());
    #else
    printf("OpenMP: DISABLED\n");
    #endif
    
    #ifdef __AVX2__
    printf("AVX2: ENABLED\n");
    #endif
    
    printf("MKL: ENABLED\n");
    printf("\n");
    
    /* Test configurations */
    int configs[][3] = {
        /* T, K, N */
        {100, 4, 50},
        {100, 4, 100},
        {200, 4, 100},
        {300, 4, 100},
        {500, 4, 100},
    };
    int n_configs = sizeof(configs) / sizeof(configs[0]);
    
    printf("┌────────────────────────────────────────────────────────────────────────┐\n");
    printf("│ Config      │ Original      │ Fast          │ MKL           │ Speedup │\n");
    printf("│ T×K×N       │ PGAS  PARIS   │ PGAS  PARIS   │ PGAS  PARIS   │ vs Orig │\n");
    printf("├────────────────────────────────────────────────────────────────────────┤\n");
    
    for (int c = 0; c < n_configs; c++) {
        int T = configs[c][0];
        int K = configs[c][1];
        int N = configs[c][2];
        
        TestData data = generate_test_data(T, K, 12345);
        
        BenchResult orig_sum = {0}, fast_sum = {0}, mkl_sum = {0};
        
        /* Warmup */
        for (int w = 0; w < WARMUP; w++) {
            bench_original(&data, N, w);
            bench_fast(&data, N, w);
            bench_mkl(&data, N, w);
        }
        
        /* Benchmark */
        for (int trial = 0; trial < TRIALS; trial++) {
            BenchResult r;
            
            r = bench_original(&data, N, trial + 100);
            orig_sum.pgas_ms += r.pgas_ms;
            orig_sum.paris_ms += r.paris_ms;
            orig_sum.total_ms += r.total_ms;
            
            r = bench_fast(&data, N, trial + 200);
            fast_sum.pgas_ms += r.pgas_ms;
            fast_sum.paris_ms += r.paris_ms;
            fast_sum.total_ms += r.total_ms;
            
            r = bench_mkl(&data, N, trial + 300);
            mkl_sum.pgas_ms += r.pgas_ms;
            mkl_sum.paris_ms += r.paris_ms;
            mkl_sum.total_ms += r.total_ms;
        }
        
        /* Average */
        double orig_pgas = orig_sum.pgas_ms / TRIALS;
        double orig_paris = orig_sum.paris_ms / TRIALS;
        double orig_total = orig_sum.total_ms / TRIALS;
        
        double fast_pgas = fast_sum.pgas_ms / TRIALS;
        double fast_paris = fast_sum.paris_ms / TRIALS;
        double fast_total = fast_sum.total_ms / TRIALS;
        
        double mkl_pgas = mkl_sum.pgas_ms / TRIALS;
        double mkl_paris = mkl_sum.paris_ms / TRIALS;
        double mkl_total = mkl_sum.total_ms / TRIALS;
        
        double speedup_fast = orig_total / fast_total;
        double speedup_mkl = orig_total / mkl_total;
        
        printf("│ %3d×%d×%-3d   │ %5.1f %6.1f  │ %5.1f %6.1f  │ %5.1f %6.1f  │ F:%.1fx M:%.1fx │\n",
               T, K, N,
               orig_pgas, orig_paris,
               fast_pgas, fast_paris,
               mkl_pgas, mkl_paris,
               speedup_fast, speedup_mkl);
        
        free_test_data(&data);
    }
    
    printf("└────────────────────────────────────────────────────────────────────────┘\n");
    
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════════════╗\n");
    printf("║                           SUMMARY                                     ║\n");
    printf("╠═══════════════════════════════════════════════════════════════════════╣\n");
    printf("║ Original:  Baseline double-precision, scalar                          ║\n");
    printf("║ Fast:      AVX2 + float + Walker's Alias + OpenMP                     ║\n");
    printf("║ MKL:       VML vsExp + VSL RNG + CBLAS + OpenMP                       ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════════╝\n");
    
    return 0;
}
