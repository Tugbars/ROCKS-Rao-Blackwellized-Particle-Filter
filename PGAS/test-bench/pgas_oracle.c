/**
 * @file pgas_oracle.c
 * @brief PGAS Oracle - Background Parameter Server Implementation
 *
 * Threading model:
 *   - Main thread calls pgas_oracle_push() at each tick
 *   - Background thread runs PGAS loop, publishes to channel
 *   - Lock-free communication via atomic double-buffer
 *
 * Core affinity:
 *   - PGAS thread pins OpenMP threads to cores [core_start, core_start + n_cores)
 *   - MKL set to sequential (threading handled by OpenMP)
 *   - Main tick loop should be on cores 0-1
 */

#define _GNU_SOURCE
#include "pgas_oracle.h"
#include "pgas_sliding.h"
#include "pgas_mkl.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <sched.h>
#include <unistd.h>

#include <mkl.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/*═══════════════════════════════════════════════════════════════════════════════
 * HELPERS
 *═══════════════════════════════════════════════════════════════════════════════*/

static inline double get_time_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/**
 * Pin current thread to specific core
 */
static void pin_to_core(int core_id)
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
}

/*═══════════════════════════════════════════════════════════════════════════════
 * LIFECYCLE
 *═══════════════════════════════════════════════════════════════════════════════*/

PGASOracleState* pgas_oracle_alloc(int window_size, int slide_step,
                                    int N, int K, int n_sweeps,
                                    uint32_t seed)
{
    if (K > PGAS_ORACLE_MAX_K) {
        return NULL;
    }
    
    PGASOracleState* oracle = (PGASOracleState*)aligned_alloc(
        PGAS_ORACLE_ALIGN, sizeof(PGASOracleState));
    if (!oracle) return NULL;
    
    memset(oracle, 0, sizeof(PGASOracleState));
    
    /* Allocate sliding window PGAS */
    oracle->sliding = pgas_sliding_alloc(window_size, slide_step, N, K, seed);
    if (!oracle->sliding) {
        free(oracle);
        return NULL;
    }
    
    /* Initialize channel */
    oracle->channel.K = K;
    atomic_store(&oracle->channel.write_idx, 0);
    atomic_store(&oracle->channel.ready, 0);
    oracle->channel.last_consumed_tick = -1;
    
    /* Initialize both buffers to uniform Π */
    float unif = 1.0f / K;
    for (int b = 0; b < 2; b++) {
        for (int i = 0; i < K; i++) {
            for (int j = 0; j < K; j++) {
                oracle->channel.buf[b].Pi[i * K + j] = unif;
            }
        }
        oracle->channel.buf[b].tick = -1;
        oracle->channel.buf[b].acceptance_rate = 0.0f;
        oracle->channel.buf[b].sweeps_used = 0;
        oracle->channel.buf[b].windows_completed = 0;
        oracle->channel.buf[b].avg_acceptance = 0.0f;
    }
    
    /* Configuration defaults */
    oracle->n_sweeps = n_sweeps;
    oracle->core_start = 2;     /* Default: cores 2-31 */
    oracle->n_cores = 30;
    
    /* Thread control */
    atomic_store(&oracle->running, 0);
    atomic_store(&oracle->started, 0);
    
    /* Diagnostics */
    atomic_store(&oracle->iterations, 0);
    atomic_store(&oracle->observations_pushed, 0);
    oracle->avg_acceptance_rate = 0.0f;
    oracle->avg_iteration_time_ms = 0.0f;
    
    return oracle;
}

void pgas_oracle_free(PGASOracleState* oracle)
{
    if (!oracle) return;
    
    /* Stop thread if running */
    pgas_oracle_stop(oracle);
    
    /* Free sliding state */
    if (oracle->sliding) {
        pgas_sliding_free(oracle->sliding);
    }
    
    free(oracle);
}

/*═══════════════════════════════════════════════════════════════════════════════
 * MODEL CONFIGURATION
 *═══════════════════════════════════════════════════════════════════════════════*/

void pgas_oracle_set_model(PGASOracleState* oracle,
                           const double* trans,
                           const double* mu_vol,
                           const double* sigma_vol,
                           double phi)
{
    if (!oracle || !oracle->sliding) return;
    pgas_sliding_set_model(oracle->sliding, trans, mu_vol, sigma_vol, phi);
}

void pgas_oracle_set_prior(PGASOracleState* oracle, float alpha, float kappa)
{
    if (!oracle || !oracle->sliding) return;
    pgas_sliding_set_prior(oracle->sliding, alpha, kappa);
}

void pgas_oracle_set_recency(PGASOracleState* oracle, float lambda)
{
    if (!oracle || !oracle->sliding) return;
    pgas_sliding_set_recency(oracle->sliding, lambda);
}

void pgas_oracle_set_affinity(PGASOracleState* oracle, int core_start, int n_cores)
{
    if (!oracle) return;
    oracle->core_start = core_start;
    oracle->n_cores = n_cores;
}

/*═══════════════════════════════════════════════════════════════════════════════
 * BACKGROUND THREAD
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * Publish Π to double-buffer channel
 */
static void publish_to_channel(PGASOracleState* oracle, int64_t tick, 
                                float acceptance_rate, int sweeps_used)
{
    int K = oracle->channel.K;
    
    /* Get write buffer (opposite of what reader sees) */
    int write_idx = atomic_load(&oracle->channel.write_idx);
    PGASChannelBuffer* buf = &oracle->channel.buf[write_idx];
    
    /* Copy Π from sliding state */
    pgas_sliding_get_pi(oracle->sliding, buf->Pi);
    buf->tick = tick;
    buf->acceptance_rate = acceptance_rate;
    buf->sweeps_used = sweeps_used;
    buf->windows_completed = oracle->sliding->windows_completed;
    buf->avg_acceptance = oracle->avg_acceptance_rate;
    
    /* Flip write index */
    atomic_store(&oracle->channel.write_idx, 1 - write_idx);
    
    /* Signal ready (release semantics) */
    atomic_store_explicit(&oracle->channel.ready, 1, memory_order_release);
}

/**
 * Background PGAS loop
 */
static void* pgas_loop(void* arg)
{
    PGASOracleState* oracle = (PGASOracleState*)arg;
    
    /* ═══════════════════════════════════════════════════════════════════════
     * THREAD SETUP
     * ═══════════════════════════════════════════════════════════════════════*/
    
    /* Disable MKL internal threading (we use OpenMP) */
    mkl_set_num_threads(1);
    
#ifdef _OPENMP
    /* Set OpenMP threads */
    omp_set_num_threads(oracle->n_cores);
    
    /* Pin OpenMP threads to cores */
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        pin_to_core(oracle->core_start + tid);
    }
#else
    /* Single-threaded: pin to first core */
    pin_to_core(oracle->core_start);
#endif
    
    /* Signal that thread has started */
    atomic_store(&oracle->started, 1);
    
    /* ═══════════════════════════════════════════════════════════════════════
     * MAIN LOOP
     * ═══════════════════════════════════════════════════════════════════════*/
    
    while (atomic_load(&oracle->running)) {
        
        /* Check if window is ready */
        if (!pgas_sliding_window_ready(oracle->sliding)) {
            /* Sleep briefly to avoid spinning */
            usleep(100);  /* 100 μs */
            continue;
        }
        
        double t_start = get_time_ms();
        
        /* ═══════════════════════════════════════════════════════════════════
         * PGAS ITERATION
         * ═══════════════════════════════════════════════════════════════════*/
        
        /* Extract window (copies observations, sets up warm start) */
        pgas_sliding_extract_window(oracle->sliding);
        
        /* Run Gibbs sweeps */
        float acceptance = pgas_sliding_run_sweeps(oracle->sliding, oracle->n_sweeps);
        
        /* Get window end tick before sliding */
        int64_t window_end_tick = oracle->sliding->window_end_tick;
        
        /* ═══════════════════════════════════════════════════════════════════
         * VALIDATE & PUBLISH
         * ═══════════════════════════════════════════════════════════════════*/
        
        /* Get Π for validation */
        float pi_temp[PGAS_ORACLE_MAX_K * PGAS_ORACLE_MAX_K];
        pgas_sliding_get_pi(oracle->sliding, pi_temp);
        
        if (pgas_oracle_validate_pi(pi_temp, oracle->channel.K)) {
            /* Publish to channel */
            publish_to_channel(oracle, window_end_tick, acceptance, oracle->n_sweeps);
        } else {
            /* Log error but don't publish stale data */
            fprintf(stderr, "[PGAS Oracle] Invalid Π at tick %lld, skipping publish\n",
                    (long long)window_end_tick);
        }
        
        /* Slide window (saves reference, advances read pointer) */
        pgas_sliding_slide(oracle->sliding);
        
        /* ═══════════════════════════════════════════════════════════════════
         * DIAGNOSTICS UPDATE
         * ═══════════════════════════════════════════════════════════════════*/
        
        double t_end = get_time_ms();
        double iter_time = t_end - t_start;
        
        int64_t iters = atomic_fetch_add(&oracle->iterations, 1) + 1;
        
        /* EMA for timing and acceptance */
        float alpha = (iters < 10) ? (1.0f / iters) : 0.1f;
        oracle->avg_iteration_time_ms = alpha * (float)iter_time + 
                                        (1.0f - alpha) * oracle->avg_iteration_time_ms;
        oracle->avg_acceptance_rate = alpha * acceptance + 
                                      (1.0f - alpha) * oracle->avg_acceptance_rate;
    }
    
    return NULL;
}

/*═══════════════════════════════════════════════════════════════════════════════
 * THREAD CONTROL
 *═══════════════════════════════════════════════════════════════════════════════*/

int pgas_oracle_start(PGASOracleState* oracle)
{
    if (!oracle) return -1;
    
    /* Already running? */
    if (atomic_load(&oracle->running)) {
        return 0;
    }
    
    atomic_store(&oracle->running, 1);
    atomic_store(&oracle->started, 0);
    
    int ret = pthread_create(&oracle->thread, NULL, pgas_loop, oracle);
    if (ret != 0) {
        atomic_store(&oracle->running, 0);
        return ret;
    }
    
    /* Wait for thread to start */
    while (!atomic_load(&oracle->started)) {
        usleep(100);
    }
    
    return 0;
}

void pgas_oracle_stop(PGASOracleState* oracle)
{
    if (!oracle) return;
    
    if (!atomic_load(&oracle->running)) {
        return;
    }
    
    /* Signal stop */
    atomic_store(&oracle->running, 0);
    
    /* Wait for thread */
    pthread_join(oracle->thread, NULL);
}

bool pgas_oracle_is_running(const PGASOracleState* oracle)
{
    return oracle && atomic_load(&oracle->running);
}

/*═══════════════════════════════════════════════════════════════════════════════
 * OBSERVATION INPUT
 *═══════════════════════════════════════════════════════════════════════════════*/

bool pgas_oracle_push(PGASOracleState* oracle, float obs, int64_t tick)
{
    if (!oracle || !oracle->sliding) return false;
    
    bool ok = pgas_sliding_push(oracle->sliding, obs, tick);
    if (ok) {
        atomic_fetch_add(&oracle->observations_pushed, 1);
    }
    return ok;
}

/*═══════════════════════════════════════════════════════════════════════════════
 * CHANNEL ACCESS
 *═══════════════════════════════════════════════════════════════════════════════*/

bool pgas_oracle_ready(const PGASOracleState* oracle)
{
    if (!oracle) return false;
    return atomic_load_explicit(&oracle->channel.ready, memory_order_acquire);
}

bool pgas_oracle_consume(PGASOracleState* oracle, float* pi_out, int64_t* tick_out)
{
    if (!oracle) return false;
    
    /* Check ready flag with acquire semantics */
    if (!atomic_load_explicit(&oracle->channel.ready, memory_order_acquire)) {
        return false;
    }
    
    /* Read from buffer opposite to write_idx */
    int read_idx = 1 - atomic_load(&oracle->channel.write_idx);
    const PGASChannelBuffer* buf = &oracle->channel.buf[read_idx];
    
    /* Copy Π */
    int K = oracle->channel.K;
    if (pi_out) {
        memcpy(pi_out, buf->Pi, K * K * sizeof(float));
    }
    
    if (tick_out) {
        *tick_out = buf->tick;
    }
    
    oracle->channel.last_consumed_tick = buf->tick;
    
    /* Mark as consumed (release semantics) */
    atomic_store_explicit(&oracle->channel.ready, 0, memory_order_release);
    
    return true;
}

void pgas_oracle_peek(const PGASOracleState* oracle,
                      float* acceptance_rate_out,
                      int* sweeps_out)
{
    if (!oracle) return;
    
    int read_idx = 1 - atomic_load(&oracle->channel.write_idx);
    const PGASChannelBuffer* buf = &oracle->channel.buf[read_idx];
    
    if (acceptance_rate_out) {
        *acceptance_rate_out = buf->acceptance_rate;
    }
    if (sweeps_out) {
        *sweeps_out = buf->sweeps_used;
    }
}

/*═══════════════════════════════════════════════════════════════════════════════
 * HOT SWAP INTEGRATION
 *═══════════════════════════════════════════════════════════════════════════════*/

bool pgas_oracle_try_hot_swap(PGASOracleState* oracle,
                              bool crisis,
                              float* pi_out,
                              int64_t* tick_out)
{
    if (!oracle) return false;
    
    /* Check ready flag */
    if (!atomic_load_explicit(&oracle->channel.ready, memory_order_acquire)) {
        return false;
    }
    
    /* ═══════════════════════════════════════════════════════════════════════
     * HAWKES VETO
     *
     * During crisis, PGAS is looking at stale history.
     * Don't swap - let Storvik adapt freely with λ=0.5.
     * ═══════════════════════════════════════════════════════════════════════*/
    if (crisis) {
        /* Mark as read anyway to avoid accumulating stale updates */
        atomic_store_explicit(&oracle->channel.ready, 0, memory_order_release);
        return false;
    }
    
    /* ═══════════════════════════════════════════════════════════════════════
     * HOT SWAP: Take PGAS Π as truth (100%, no blend)
     * ═══════════════════════════════════════════════════════════════════════*/
    return pgas_oracle_consume(oracle, pi_out, tick_out);
}

/*═══════════════════════════════════════════════════════════════════════════════
 * VALIDATION
 *═══════════════════════════════════════════════════════════════════════════════*/

bool pgas_oracle_validate_pi(const float* pi, int K)
{
    if (!pi || K <= 0 || K > PGAS_ORACLE_MAX_K) {
        return false;
    }
    
    for (int i = 0; i < K; i++) {
        float sum = 0.0f;
        for (int j = 0; j < K; j++) {
            float p = pi[i * K + j];
            
            /* Check for NaN/Inf */
            if (!isfinite(p)) {
                return false;
            }
            
            /* Check range */
            if (p < 0.0f || p > 1.0f) {
                return false;
            }
            
            sum += p;
        }
        
        /* Check row sums to 1 */
        if (fabsf(sum - 1.0f) > 0.01f) {
            return false;
        }
    }
    
    return true;
}

/*═══════════════════════════════════════════════════════════════════════════════
 * DIAGNOSTICS
 *═══════════════════════════════════════════════════════════════════════════════*/

int64_t pgas_oracle_get_iterations(const PGASOracleState* oracle)
{
    return oracle ? atomic_load(&oracle->iterations) : 0;
}

int64_t pgas_oracle_get_observations(const PGASOracleState* oracle)
{
    return oracle ? atomic_load(&oracle->observations_pushed) : 0;
}

void pgas_oracle_print_diagnostics(const PGASOracleState* oracle)
{
    if (!oracle) return;
    
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("PGAS ORACLE DIAGNOSTICS\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("Running:            %s\n", atomic_load(&oracle->running) ? "YES" : "NO");
    printf("Core affinity:      %d-%d\n", oracle->core_start, 
           oracle->core_start + oracle->n_cores - 1);
    printf("Sweeps per iter:    %d\n", oracle->n_sweeps);
    printf("───────────────────────────────────────────────────────────────\n");
    printf("Observations:       %lld\n", (long long)atomic_load(&oracle->observations_pushed));
    printf("Iterations:         %lld\n", (long long)atomic_load(&oracle->iterations));
    printf("Avg iter time:      %.2f ms\n", oracle->avg_iteration_time_ms);
    printf("Avg acceptance:     %.3f\n", oracle->avg_acceptance_rate);
    printf("───────────────────────────────────────────────────────────────\n");
    printf("Channel ready:      %s\n", pgas_oracle_ready(oracle) ? "YES" : "NO");
    printf("Last consumed tick: %lld\n", (long long)oracle->channel.last_consumed_tick);
    printf("═══════════════════════════════════════════════════════════════\n");
    
    /* Also print sliding window diagnostics */
    if (oracle->sliding) {
        pgas_sliding_print_diagnostics(oracle->sliding);
    }
}
