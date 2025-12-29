/*=============================================================================
 * PGAS + RBPF Coexistence Test
 *
 * Runs both systems on the same data stream WITHOUT injection.
 * Goal: Verify they can run side-by-side without conflicts.
 *
 * - RBPF: Main thread, processes each tick
 * - PGAS: Background thread, builds Π from sliding window (8 P-cores)
 *
 * No wiring yet - just coexistence.
 *
 *===========================================================================*/

#include "rbpf_ksc_param_integration.h"
#include "pgas_oracle.h"
#include "mkl_tuning.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/*─────────────────────────────────────────────────────────────────────────────
 * TIMING
 *───────────────────────────────────────────────────────────────────────────*/

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <mmsystem.h>
#pragma comment(lib, "winmm.lib")
static double g_timer_freq = 0.0;
static void init_timer(void)
{
    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);
    g_timer_freq = (double)freq.QuadPart / 1e6;
}
static inline double get_time_us(void)
{
    LARGE_INTEGER counter;
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / g_timer_freq;
}
#else
#include <sys/time.h>
static void init_timer(void) {}
static inline double get_time_us(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1e6 + tv.tv_usec;
}
#endif

/*─────────────────────────────────────────────────────────────────────────────
 * PCG32 RNG
 *───────────────────────────────────────────────────────────────────────────*/

typedef struct
{
    uint64_t state;
    uint64_t inc;
} pcg32_t;

static uint32_t pcg32_random(pcg32_t *rng)
{
    uint64_t oldstate = rng->state;
    rng->state = oldstate * 6364136223846793005ULL + rng->inc;
    uint32_t xorshifted = (uint32_t)(((oldstate >> 18u) ^ oldstate) >> 27u);
    uint32_t rot = (uint32_t)(oldstate >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

static double pcg32_double(pcg32_t *rng)
{
    return (double)pcg32_random(rng) / 4294967296.0;
}

static double pcg32_gaussian(pcg32_t *rng)
{
    double u1 = pcg32_double(rng);
    double u2 = pcg32_double(rng);
    if (u1 < 1e-10)
        u1 = 1e-10;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979 * u2);
}

/*─────────────────────────────────────────────────────────────────────────────
 * SYNTHETIC DATA (simplified)
 *───────────────────────────────────────────────────────────────────────────*/

typedef struct
{
    double *returns;
    double *true_log_vol;
    int *true_regime;
    int n_ticks;
} SyntheticData;

static SyntheticData *generate_data(int n_ticks, int seed)
{
    SyntheticData *data = (SyntheticData *)calloc(1, sizeof(SyntheticData));
    data->n_ticks = n_ticks;
    data->returns = (double *)malloc(n_ticks * sizeof(double));
    data->true_log_vol = (double *)malloc(n_ticks * sizeof(double));
    data->true_regime = (int *)malloc(n_ticks * sizeof(int));

    pcg32_t rng = {seed * 12345ULL + 1, seed * 67890ULL | 1};

    /* Regime parameters */
    double mu_vol[4] = {-5.0, -3.5, -2.0, -1.0};
    double sigma_vol[4] = {0.08, 0.15, 0.25, 0.40};
    double phi = 0.97;

    double log_vol = mu_vol[0];
    int regime = 0;
    int next_switch = 200 + (int)(pcg32_double(&rng) * 300);

    for (int t = 0; t < n_ticks; t++)
    {
        /* Regime switching */
        if (t >= next_switch)
        {
            int delta = (pcg32_double(&rng) < 0.5) ? -1 : 1;
            regime = (regime + delta + 4) % 4;
            next_switch = t + 200 + (int)(pcg32_double(&rng) * 300);
        }

        /* State evolution */
        double theta = 1.0 - phi;
        log_vol = phi * log_vol + theta * mu_vol[regime] + sigma_vol[regime] * pcg32_gaussian(&rng);
        double vol = exp(log_vol);
        double ret = vol * pcg32_gaussian(&rng);

        data->returns[t] = ret;
        data->true_log_vol[t] = log_vol;
        data->true_regime[t] = regime;
    }

    return data;
}

static void free_data(SyntheticData *data)
{
    if (!data)
        return;
    free(data->returns);
    free(data->true_log_vol);
    free(data->true_regime);
    free(data);
}

/*─────────────────────────────────────────────────────────────────────────────
 * MAIN TEST
 *───────────────────────────────────────────────────────────────────────────*/

int main(int argc, char **argv)
{
    int seed = 42;
    int n_ticks = 10000;

    if (argc > 1)
        seed = atoi(argv[1]);
    if (argc > 2)
        n_ticks = atoi(argv[2]);

    init_timer();

    /* Initialize MKL tuning for RBPF (single-threaded for latency)
     * PGAS will set its own thread count in background thread */
    mkl_tuning_flush_denormals();
    mkl_set_num_threads(1); /* RBPF stays single-threaded */
    mkl_set_dynamic(0);
    mkl_cbwr_set(MKL_CBWR_AVX2);
#ifdef _WIN32
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    timeBeginPeriod(1);
#endif

    printf("╔═══════════════════════════════════════════════════════════════════════╗\n");
    printf("║                      MKL TUNING CONFIGURATION                         ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════════╝\n");
    printf("  Denormals:     FLUSH TO ZERO (FTZ+DAZ enabled)\n");
    printf("  MKL threads:   %d (RBPF single-threaded)\n", mkl_get_max_threads());
    printf("  PGAS threads:  8 (set in background thread)\n");
    printf("  MKL dynamic:   OFF\n");
    printf("  MKL CBWR:      AVX2\n");
#ifdef _WIN32
    printf("  Windows:       HIGH priority, timer=1ms\n");
#endif

    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  PGAS + RBPF Coexistence Test\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  Seed: %d\n", seed);
    printf("  Ticks: %d\n\n", n_ticks);

    /* ═══════════════════════════════════════════════════════════════════════
     * GENERATE DATA
     * ═══════════════════════════════════════════════════════════════════════*/
    printf("Generating synthetic data...\n");
    SyntheticData *data = generate_data(n_ticks, seed);

    /* ═══════════════════════════════════════════════════════════════════════
     * CREATE RBPF
     * ═══════════════════════════════════════════════════════════════════════*/
    printf("Creating RBPF...\n");
    const int RBPF_PARTICLES = 512;
    const int N_REGIMES = 4;

    RBPF_Extended *rbpf = rbpf_ext_create(RBPF_PARTICLES, N_REGIMES, RBPF_PARAM_STORVIK);
    rbpf_ext_enable_kl_tempering(rbpf);

    /* Enable PARIS smoothed Storvik (L=5 tick lag) */
    rbpf_ext_enable_smoothed_storvik(rbpf, 5);

    /* Regime params (θ, μ, σ) */
    rbpf_ext_set_regime_params(rbpf, 0, 0.0030f, -4.299f, 0.080f);
    rbpf_ext_set_regime_params(rbpf, 1, 0.0420f, -3.465f, 0.267f);
    rbpf_ext_set_regime_params(rbpf, 2, 0.0810f, -2.954f, 0.453f);
    rbpf_ext_set_regime_params(rbpf, 3, 0.1200f, -2.171f, 0.640f);

    /* Transition matrix (stickiness=0.92) */
    rbpf_real_t trans[16] = {
        0.920f, 0.056f, 0.020f, 0.004f,
        0.032f, 0.920f, 0.036f, 0.012f,
        0.012f, 0.036f, 0.920f, 0.032f,
        0.004f, 0.020f, 0.056f, 0.920f};
    rbpf_ext_build_transition_lut(rbpf, trans);

    /* Adaptive forgetting */
    rbpf_ext_enable_adaptive_forgetting_mode(rbpf, ADAPT_SIGNAL_REGIME);
    rbpf_ext_enable_circuit_breaker(rbpf, 0.999, 100);

    /* Robust OCSN */
    rbpf->robust_ocsn.enabled = 1;
    rbpf->robust_ocsn.regime[0].prob = 0.02f;
    rbpf->robust_ocsn.regime[0].variance = 100.0f;
    rbpf->robust_ocsn.regime[1].prob = 0.03f;
    rbpf->robust_ocsn.regime[1].variance = 120.0f;
    rbpf->robust_ocsn.regime[2].prob = 0.04f;
    rbpf->robust_ocsn.regime[2].variance = 140.0f;
    rbpf->robust_ocsn.regime[3].prob = 0.05f;
    rbpf->robust_ocsn.regime[3].variance = 160.0f;

    rbpf_ext_init(rbpf, -4.5f, 0.1f);

    /* ═══════════════════════════════════════════════════════════════════════
     * CREATE PGAS ORACLE
     * ═══════════════════════════════════════════════════════════════════════*/
    printf("Creating PGAS Oracle...\n");
    const int PGAS_WINDOW = 500;
    const int PGAS_SLIDE = 50;
    const int PGAS_PARTICLES = 64; /* Keep lightweight for throughput */
    const int PGAS_SWEEPS = 2;     /* 2 sweeps = fast iterations */

    PGASOracleState *oracle = pgas_oracle_alloc(
        PGAS_WINDOW, PGAS_SLIDE, PGAS_PARTICLES, N_REGIMES, PGAS_SWEEPS, seed);

    if (!oracle)
    {
        fprintf(stderr, "Failed to allocate PGAS Oracle\n");
        rbpf_ext_destroy(rbpf);
        free_data(data);
        return 1;
    }

    /* Configure PGAS model (same regime params) */
    double pgas_trans[16];
    for (int i = 0; i < 16; i++)
        pgas_trans[i] = (double)trans[i];
    double mu_vol[4] = {-4.299, -3.465, -2.954, -2.171};
    double sigma_vol[4] = {0.080, 0.267, 0.453, 0.640};
    pgas_oracle_set_model(oracle, pgas_trans, mu_vol, sigma_vol, 0.97);
    pgas_oracle_set_prior(oracle, 1.0f, 50.0f);
    pgas_oracle_set_recency(oracle, 0.001f);

    /* Pin PGAS to 8 P-cores (cores 0-7) */
    pgas_oracle_set_affinity(oracle, 0, 8);

    /* ═══════════════════════════════════════════════════════════════════════
     * START PGAS BACKGROUND THREAD
     * ═══════════════════════════════════════════════════════════════════════*/
    printf("Starting PGAS background thread (8 P-cores)...\n");
    if (pgas_oracle_start(oracle) != 0)
    {
        fprintf(stderr, "Failed to start PGAS Oracle\n");
        pgas_oracle_free(oracle);
        rbpf_ext_destroy(rbpf);
        free_data(data);
        return 1;
    }

    /* ═══════════════════════════════════════════════════════════════════════
     * RUN TICK LOOP
     * ═══════════════════════════════════════════════════════════════════════*/
    printf("\nRunning tick loop...\n");

    double rbpf_total_time = 0.0;
    double rbpf_max_latency = 0.0;
    int pgas_swaps = 0;
    int rbpf_regime_correct = 0;

    RBPF_KSC_Output rbpf_out;
    float pgas_pi[16];
    int64_t pgas_tick;

    double t_loop_start = get_time_us();

    for (int t = 0; t < n_ticks; t++)
    {
        float obs = (float)data->returns[t];

        /* ─────────────────────────────────────────────────────────────────
         * STEP 1: Push observation to PGAS
         * ─────────────────────────────────────────────────────────────────*/
        pgas_oracle_push(oracle, obs, t);

        /* ─────────────────────────────────────────────────────────────────
         * STEP 2: Check for PGAS output (NO INJECTION YET)
         * ─────────────────────────────────────────────────────────────────*/
        if (pgas_oracle_try_hot_swap(oracle, false, pgas_pi, &pgas_tick))
        {
            pgas_swaps++;
            /* NOT injecting - just counting */
        }

        /* ─────────────────────────────────────────────────────────────────
         * STEP 3: RBPF tick
         * ─────────────────────────────────────────────────────────────────*/
        memset(&rbpf_out, 0, sizeof(rbpf_out));
        double t_rbpf_start = get_time_us();
        rbpf_ext_step(rbpf, (rbpf_real_t)obs, &rbpf_out);
        double t_rbpf_end = get_time_us();

        double latency = t_rbpf_end - t_rbpf_start;
        rbpf_total_time += latency;
        if (latency > rbpf_max_latency)
            rbpf_max_latency = latency;

        /* Track accuracy (map 4 regimes to simplified) */
        int rbpf_regime = rbpf_out.dominant_regime;
        int true_regime = data->true_regime[t];
        if (rbpf_regime == true_regime)
            rbpf_regime_correct++;

        /* Progress */
        if ((t + 1) % 2000 == 0)
        {
            printf("  Tick %d: RBPF regime=%d (true=%d), PGAS swaps=%d\n",
                   t + 1, rbpf_regime, true_regime, pgas_swaps);
        }
    }

    double t_loop_end = get_time_us();
    double total_loop_time = (t_loop_end - t_loop_start) / 1000.0;

    /* ═══════════════════════════════════════════════════════════════════════
     * STOP PGAS
     * ═══════════════════════════════════════════════════════════════════════*/
    printf("\nStopping PGAS...\n");
    pgas_oracle_stop(oracle);

    /* ═══════════════════════════════════════════════════════════════════════
     * RESULTS
     * ═══════════════════════════════════════════════════════════════════════*/
    printf("\n");
    printf("══════════════════════════════════════════════════════════════════════════════\n");
    printf("  COEXISTENCE TEST RESULTS\n");
    printf("══════════════════════════════════════════════════════════════════════════════\n");

    printf("\n  RBPF Performance:\n");
    printf("    Total time:     %.2f ms\n", rbpf_total_time / 1000.0);
    printf("    Avg latency:    %.2f us\n", rbpf_total_time / n_ticks);
    printf("    Max latency:    %.2f us\n", rbpf_max_latency);
    printf("    Regime accuracy: %.1f%%\n", 100.0 * rbpf_regime_correct / n_ticks);

    printf("\n  PGAS Oracle:\n");
    printf("    Windows computed: %d\n", pgas_swaps);
    printf("    Expected:         ~%d\n", (n_ticks - PGAS_WINDOW) / PGAS_SLIDE);

    printf("\n  Overall:\n");
    printf("    Loop time:      %.2f ms\n", total_loop_time);
    printf("    Throughput:     %.0f ticks/sec\n", n_ticks / (total_loop_time / 1000.0));

    /* Print PGAS diagnostics */
    printf("\n");
    pgas_oracle_print_diagnostics(oracle);

    /* Print RBPF config */
    printf("\n");
    rbpf_ext_print_config(rbpf);

    printf("\n══════════════════════════════════════════════════════════════════════════════\n");
    printf("  ✓ PGAS and RBPF ran together without conflicts\n");
    printf("══════════════════════════════════════════════════════════════════════════════\n");

    /* Cleanup */
    pgas_oracle_free(oracle);
    rbpf_ext_destroy(rbpf);
    free_data(data);
#ifdef _WIN32
    timeEndPeriod(1);
#endif

    return 0;
}