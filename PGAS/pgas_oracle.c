/**
 * @file pgas_oracle.c
 * @brief PGAS Oracle - Background Parameter Server Implementation
 *
 * Cross-platform: Windows + POSIX
 */

#include "pgas_oracle.h"
#include "pgas_sliding.h"
#include "pgas_mkl.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#include <mkl.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/*===========================================================================
 * PLATFORM ABSTRACTIONS
 *===========================================================================*/

#ifdef _MSC_VER

/* Windows implementations */
static inline double get_time_ms(void)
{
    static double freq = 0.0;
    LARGE_INTEGER counter, f;
    if (freq == 0.0)
    {
        QueryPerformanceFrequency(&f);
        freq = (double)f.QuadPart / 1000.0;
    }
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / freq;
}

static inline void *aligned_alloc_wrapper(size_t align, size_t size)
{
    return _aligned_malloc(size, align);
}

static inline void aligned_free_wrapper(void *ptr)
{
    _aligned_free(ptr);
}

static inline void thread_sleep_ms(int ms)
{
    Sleep(ms);
}

/* Windows thread function signature */
static DWORD WINAPI pgas_loop_win(LPVOID arg);

#else

/* POSIX implementations */
#define _GNU_SOURCE
#include <time.h>
#include <unistd.h>
#include <sched.h>

static inline double get_time_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static inline void *aligned_alloc_wrapper(size_t align, size_t size)
{
    return aligned_alloc(align, size);
}

static inline void aligned_free_wrapper(void *ptr)
{
    free(ptr);
}

static inline void thread_sleep_ms(int ms)
{
    usleep(ms * 1000);
}

/* Pin current thread to specific core (POSIX only) */
static void pin_to_core(int core_id)
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
}

/* POSIX thread function */
static void *pgas_loop(void *arg);

#endif

/*===========================================================================
 * CHANNEL OPERATIONS
 *===========================================================================*/

static void channel_publish(PGASChannel *channel, const float *Pi, int K,
                            int64_t tick, float acceptance, int sweeps)
{
    /* Write to shadow buffer */
    int write_idx = pgas_atomic_load_32(&channel->write_idx);
    PGASChannelBuffer *buf = &channel->buf[write_idx];

    memcpy(buf->Pi, Pi, K * K * sizeof(float));
    buf->tick = tick;
    buf->acceptance_rate = acceptance;
    buf->sweeps_used = sweeps;
    buf->windows_completed++;

    /* Update running average */
    float alpha = 0.1f;
    buf->avg_acceptance = (1.0f - alpha) * buf->avg_acceptance + alpha * acceptance;

    /* Flip buffer */
    pgas_atomic_store_32(&channel->write_idx, 1 - write_idx);

    /* Signal ready */
    pgas_atomic_fence();
    pgas_atomic_store_32(&channel->ready, 1);
}

/*===========================================================================
 * PGAS WORKER LOOP (shared logic)
 *===========================================================================*/

static void pgas_worker_body(PGASOracleState *oracle)
{
#ifndef _MSC_VER
    /* Set core affinity (POSIX only) */
    if (oracle->core_start >= 0 && oracle->n_cores > 0)
    {
        pin_to_core(oracle->core_start);
    }
#endif

    /* Configure MKL for this thread */
    mkl_set_num_threads(1); /* Sequential MKL, OpenMP handles parallelism */

#ifdef _OPENMP
    if (oracle->n_cores > 0)
    {
        omp_set_num_threads(oracle->n_cores);
    }
#endif

    printf("[PGAS Oracle] Worker started\n");
    pgas_atomic_store_32(&oracle->started, 1);

    float Pi_out[PGAS_ORACLE_MAX_K * PGAS_ORACLE_MAX_K];
    int K = oracle->channel.K;

    while (pgas_atomic_load_32(&oracle->running))
    {
        /* Check if enough data */
        if (!pgas_sliding_window_ready(oracle->sliding))
        {
            thread_sleep_ms(1);
            continue;
        }

        /* Run combined iteration: extract → sweeps → slide */
        double t_start = get_time_ms();

        float acceptance = pgas_sliding_iterate(oracle->sliding, oracle->n_sweeps, Pi_out);

        /* Get tick from sliding state */
        int64_t tick = oracle->sliding->last_window_end_tick;

        double t_end = get_time_ms();
        double elapsed = t_end - t_start;

        /* Update diagnostics */
        float alpha = 0.1f;
        oracle->avg_acceptance_rate = (1.0f - alpha) * oracle->avg_acceptance_rate + alpha * acceptance;
        oracle->avg_iteration_time_ms = (1.0f - alpha) * oracle->avg_iteration_time_ms + alpha * (float)elapsed;

        /* Publish to channel */
        channel_publish(&oracle->channel, Pi_out, K, tick, acceptance, oracle->n_sweeps);

        int64_t iters = pgas_atomic_fetch_add_64(&oracle->iterations, 1) + 1;

        if ((iters % 10) == 0)
        {
            printf("[PGAS] iter=%lld tick=%lld acc=%.3f time=%.1fms\n",
                   (long long)iters, (long long)tick, acceptance, elapsed);
        }
    }

    printf("[PGAS Oracle] Worker stopped\n");
}

/*===========================================================================
 * PLATFORM-SPECIFIC THREAD WRAPPERS
 *===========================================================================*/

#ifdef _MSC_VER

static DWORD WINAPI pgas_loop_win(LPVOID arg)
{
    PGASOracleState *oracle = (PGASOracleState *)arg;
    pgas_worker_body(oracle);
    return 0;
}

#else

static void *pgas_loop(void *arg)
{
    PGASOracleState *oracle = (PGASOracleState *)arg;
    pgas_worker_body(oracle);
    return NULL;
}

#endif

/*===========================================================================
 * LIFECYCLE
 *===========================================================================*/

PGASOracleState *pgas_oracle_alloc(int window_size, int slide_step,
                                   int N, int K, int n_sweeps,
                                   uint32_t seed)
{
    if (K > PGAS_ORACLE_MAX_K)
    {
        return NULL;
    }

    PGASOracleState *oracle = (PGASOracleState *)aligned_alloc_wrapper(
        PGAS_ORACLE_ALIGN, sizeof(PGASOracleState));
    if (!oracle)
        return NULL;

    memset(oracle, 0, sizeof(PGASOracleState));

    /* Allocate sliding window PGAS */
    oracle->sliding = pgas_sliding_alloc(window_size, slide_step, N, K, seed);
    if (!oracle->sliding)
    {
        aligned_free_wrapper(oracle);
        return NULL;
    }

    /* Initialize channel */
    oracle->channel.K = K;
    pgas_atomic_store_32(&oracle->channel.write_idx, 0);
    pgas_atomic_store_32(&oracle->channel.ready, 0);
    oracle->channel.last_consumed_tick = -1;

    /* Initialize both buffers to uniform Pi */
    float unif = 1.0f / K;
    for (int b = 0; b < 2; b++)
    {
        for (int i = 0; i < K; i++)
        {
            for (int j = 0; j < K; j++)
            {
                oracle->channel.buf[b].Pi[i * K + j] = unif;
            }
        }
        oracle->channel.buf[b].tick = -1;
        oracle->channel.buf[b].acceptance_rate = 0.0f;
        oracle->channel.buf[b].sweeps_used = 0;
        oracle->channel.buf[b].windows_completed = 0;
        oracle->channel.buf[b].avg_acceptance = 0.0f;
    }

    /* Thread control */
    pgas_atomic_store_32(&oracle->running, 0);
    pgas_atomic_store_32(&oracle->started, 0);

    /* Configuration */
    oracle->n_sweeps = n_sweeps;
    oracle->core_start = -1;
    oracle->n_cores = 0;

    /* Diagnostics */
    pgas_atomic_store_64(&oracle->iterations, 0);
    pgas_atomic_store_64(&oracle->observations_pushed, 0);
    oracle->avg_acceptance_rate = 0.0f;
    oracle->avg_iteration_time_ms = 0.0f;

    return oracle;
}

void pgas_oracle_free(PGASOracleState *oracle)
{
    if (!oracle)
        return;

    pgas_oracle_stop(oracle);

    if (oracle->sliding)
    {
        pgas_sliding_free(oracle->sliding);
    }

    aligned_free_wrapper(oracle);
}

/*===========================================================================
 * MODEL CONFIGURATION
 *===========================================================================*/

void pgas_oracle_set_model(PGASOracleState *oracle,
                           const double *trans,
                           const double *mu_vol,
                           const double *sigma_vol,
                           double phi)
{
    if (!oracle || !oracle->sliding)
        return;
    pgas_sliding_set_model(oracle->sliding, trans, mu_vol, sigma_vol, phi);
}

void pgas_oracle_set_prior(PGASOracleState *oracle, float alpha, float kappa)
{
    if (!oracle || !oracle->sliding)
        return;
    pgas_sliding_set_prior(oracle->sliding, alpha, kappa);
}

void pgas_oracle_set_recency(PGASOracleState *oracle, float lambda)
{
    if (!oracle || !oracle->sliding)
        return;
    pgas_sliding_set_recency(oracle->sliding, lambda);
}

void pgas_oracle_set_affinity(PGASOracleState *oracle, int core_start, int n_cores)
{
    if (!oracle)
        return;
    oracle->core_start = core_start;
    oracle->n_cores = n_cores;
}

/*===========================================================================
 * THREAD CONTROL
 *===========================================================================*/

int pgas_oracle_start(PGASOracleState *oracle)
{
    if (!oracle || !oracle->sliding)
        return -1;

    if (pgas_atomic_load_32(&oracle->running))
    {
        return -1; /* Already running */
    }

    pgas_atomic_store_32(&oracle->running, 1);
    pgas_atomic_store_32(&oracle->started, 0);

#ifdef _MSC_VER
    oracle->thread = CreateThread(NULL, 0, pgas_loop_win, oracle, 0, NULL);
    if (oracle->thread == NULL)
    {
        pgas_atomic_store_32(&oracle->running, 0);
        return -1;
    }
#else
    int ret = pthread_create(&oracle->thread, NULL, pgas_loop, oracle);
    if (ret != 0)
    {
        pgas_atomic_store_32(&oracle->running, 0);
        return -1;
    }
#endif

    /* Wait for thread to start */
    while (!pgas_atomic_load_32(&oracle->started))
    {
        thread_sleep_ms(1);
    }

    return 0;
}

void pgas_oracle_stop(PGASOracleState *oracle)
{
    if (!oracle)
        return;

    if (!pgas_atomic_load_32(&oracle->running))
    {
        return; /* Not running */
    }

    pgas_atomic_store_32(&oracle->running, 0);

#ifdef _MSC_VER
    WaitForSingleObject(oracle->thread, INFINITE);
    CloseHandle(oracle->thread);
    oracle->thread = NULL;
#else
    pthread_join(oracle->thread, NULL);
#endif
}

bool pgas_oracle_is_running(const PGASOracleState *oracle)
{
    return oracle && pgas_atomic_load_32((pgas_atomic_int32 *)&oracle->running);
}

/*===========================================================================
 * OBSERVATION INPUT
 *===========================================================================*/

bool pgas_oracle_push(PGASOracleState *oracle, float obs, int64_t tick)
{
    if (!oracle || !oracle->sliding)
        return false;

    bool ok = pgas_sliding_push(oracle->sliding, obs, tick);
    if (ok)
    {
        pgas_atomic_fetch_add_64(&oracle->observations_pushed, 1);
    }
    return ok;
}

/*===========================================================================
 * CHANNEL ACCESS
 *===========================================================================*/

bool pgas_oracle_ready(const PGASOracleState *oracle)
{
    if (!oracle)
        return false;
    pgas_atomic_fence();
    return pgas_atomic_load_32((pgas_atomic_int32 *)&oracle->channel.ready) != 0;
}

bool pgas_oracle_consume(PGASOracleState *oracle, float *pi_out, int64_t *tick_out)
{
    if (!oracle || !pi_out)
        return false;

    if (!pgas_atomic_load_32(&oracle->channel.ready))
    {
        return false;
    }

    /* Read from shadow buffer (opposite of write) */
    int read_idx = 1 - pgas_atomic_load_32(&oracle->channel.write_idx);
    PGASChannelBuffer *buf = &oracle->channel.buf[read_idx];

    /* Check for stale data */
    if (buf->tick <= oracle->channel.last_consumed_tick)
    {
        return false;
    }

    /* Copy Pi */
    int K = oracle->channel.K;
    memcpy(pi_out, buf->Pi, K * K * sizeof(float));

    if (tick_out)
    {
        *tick_out = buf->tick;
    }

    oracle->channel.last_consumed_tick = buf->tick;

    /* Mark as consumed */
    pgas_atomic_fence();
    pgas_atomic_store_32(&oracle->channel.ready, 0);

    return true;
}

void pgas_oracle_peek(const PGASOracleState *oracle,
                      float *acceptance_rate_out,
                      int *sweeps_out)
{
    if (!oracle)
        return;

    int read_idx = 1 - pgas_atomic_load_32((pgas_atomic_int32 *)&oracle->channel.write_idx);
    const PGASChannelBuffer *buf = &oracle->channel.buf[read_idx];

    if (acceptance_rate_out)
        *acceptance_rate_out = buf->acceptance_rate;
    if (sweeps_out)
        *sweeps_out = buf->sweeps_used;
}

/*===========================================================================
 * HOT SWAP
 *===========================================================================*/

bool pgas_oracle_try_hot_swap(PGASOracleState *oracle,
                              bool crisis,
                              float *pi_out,
                              int64_t *tick_out)
{
    if (!oracle || !pi_out)
        return false;

    /* Veto if crisis */
    if (crisis)
    {
        return false;
    }

    /* Check ready */
    if (!pgas_atomic_load_32(&oracle->channel.ready))
    {
        return false;
    }

    /* Consume */
    bool consumed = pgas_oracle_consume(oracle, pi_out, tick_out);

    return consumed;
}

/*===========================================================================
 * VALIDATION
 *===========================================================================*/

bool pgas_oracle_validate_pi(const float *pi, int K)
{
    if (!pi || K <= 0)
        return false;

    for (int i = 0; i < K; i++)
    {
        float sum = 0.0f;
        for (int j = 0; j < K; j++)
        {
            float p = pi[i * K + j];
            if (isnan(p) || p < 0.0f || p > 1.0f)
                return false;
            sum += p;
        }
        if (fabsf(sum - 1.0f) > 1e-4f)
            return false;
    }

    return true;
}

/*===========================================================================
 * DIAGNOSTICS
 *===========================================================================*/

int64_t pgas_oracle_get_iterations(const PGASOracleState *oracle)
{
    return oracle ? pgas_atomic_load_64((pgas_atomic_int64 *)&oracle->iterations) : 0;
}

int64_t pgas_oracle_get_observations(const PGASOracleState *oracle)
{
    return oracle ? pgas_atomic_load_64((pgas_atomic_int64 *)&oracle->observations_pushed) : 0;
}

void pgas_oracle_print_diagnostics(const PGASOracleState *oracle)
{
    if (!oracle)
        return;

    printf("\n");
    printf("=== PGAS Oracle Diagnostics ===\n");
    printf("Running:            %s\n", pgas_atomic_load_32((pgas_atomic_int32 *)&oracle->running) ? "YES" : "NO");
    printf("Sweeps/iter:        %d\n", oracle->n_sweeps);
    printf("Core affinity:      %d-%d\n", oracle->core_start,
           oracle->core_start + oracle->n_cores - 1);
    printf("Observations:       %lld\n", (long long)pgas_atomic_load_64((pgas_atomic_int64 *)&oracle->observations_pushed));
    printf("Iterations:         %lld\n", (long long)pgas_atomic_load_64((pgas_atomic_int64 *)&oracle->iterations));
    printf("Avg acceptance:     %.3f\n", oracle->avg_acceptance_rate);
    printf("Avg iter time:      %.1f ms\n", oracle->avg_iteration_time_ms);
    printf("===============================\n");
}