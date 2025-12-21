/**
 * @file lifeboat.c
 * @brief Lifeboat Injection: Replace RBPF particles with PARIS-smoothed cloud
 *
 * Optimizations for 5µs hot-path budget:
 *   1. Atomic state flag for lock-free ready check
 *   2. Pre-allocated worker buffers (no malloc in hot path)
 *   3. Fast-forward catch-up for ticks missed during inference
 *   4. Stickiness validation for transition matrix
 *
 * Thread model:
 *   - Main thread: RBPF updates, atomic ready check, injection
 *   - Worker thread: PGAS+PARIS on pre-allocated buffers
 */

#include "lifeboat.h"
#include "pgas_mkl.h"
#include <mkl.h>

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <inttypes.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/*═══════════════════════════════════════════════════════════════════════════════
 * CROSS-PLATFORM TIMING
 *═══════════════════════════════════════════════════════════════════════════════*/

#ifdef _WIN32
static double get_time_ms(void)
{
    static LARGE_INTEGER freq = {0};
    LARGE_INTEGER counter;
    if (freq.QuadPart == 0)
        QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart * 1000.0 / (double)freq.QuadPart;
}
#else
static double get_time_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}
#endif

/*═══════════════════════════════════════════════════════════════════════════════
 * CROSS-PLATFORM ATOMICS
 *═══════════════════════════════════════════════════════════════════════════════*/

#ifdef _WIN32
static inline void set_thread_state(LifeboatManager *mgr, LifeboatThreadState state)
{
    InterlockedExchange((volatile LONG *)&mgr->thread_state_atomic, (LONG)state);
}

static inline LifeboatThreadState get_thread_state(const LifeboatManager *mgr)
{
    return (LifeboatThreadState)InterlockedCompareExchange(
        (volatile LONG *)&mgr->thread_state_atomic, 0, 0);
}
#else
#include <stdatomic.h>

static inline void set_thread_state(LifeboatManager *mgr, LifeboatThreadState state)
{
    atomic_store((atomic_int *)&mgr->thread_state_atomic, (int)state);
}

static inline LifeboatThreadState get_thread_state(const LifeboatManager *mgr)
{
    return (LifeboatThreadState)atomic_load((const atomic_int *)&mgr->thread_state_atomic);
}
#endif

/*═══════════════════════════════════════════════════════════════════════════════
 * CROSS-PLATFORM THREADING HELPERS
 *═══════════════════════════════════════════════════════════════════════════════*/

#ifdef _WIN32
static inline void lifeboat_mutex_init(lifeboat_mutex_t *mtx)
{
    InitializeCriticalSection(mtx);
}
static inline void lifeboat_mutex_destroy(lifeboat_mutex_t *mtx)
{
    DeleteCriticalSection(mtx);
}
static inline void lifeboat_mutex_lock(lifeboat_mutex_t *mtx)
{
    EnterCriticalSection(mtx);
}
static inline void lifeboat_mutex_unlock(lifeboat_mutex_t *mtx)
{
    LeaveCriticalSection(mtx);
}
static inline void lifeboat_cond_init(lifeboat_cond_t *cond)
{
    InitializeConditionVariable(cond);
}
static inline void lifeboat_cond_destroy(lifeboat_cond_t *cond)
{
    (void)cond; /* No cleanup needed on Windows */
}
static inline void lifeboat_cond_signal(lifeboat_cond_t *cond)
{
    WakeConditionVariable(cond);
}
static inline void lifeboat_cond_wait(lifeboat_cond_t *cond, lifeboat_mutex_t *mtx)
{
    SleepConditionVariableCS(cond, mtx, INFINITE);
}
static inline void lifeboat_thread_create(lifeboat_thread_t *thread,
                                          DWORD(WINAPI *func)(void *), void *arg)
{
    *thread = CreateThread(NULL, 0, func, arg, 0, NULL);
}
static inline void lifeboat_thread_join(lifeboat_thread_t thread)
{
    WaitForSingleObject(thread, INFINITE);
    CloseHandle(thread);
}
#else
static inline void lifeboat_mutex_init(lifeboat_mutex_t *mtx)
{
    pthread_mutex_init(mtx, NULL);
}
static inline void lifeboat_mutex_destroy(lifeboat_mutex_t *mtx)
{
    pthread_mutex_destroy(mtx);
}
static inline void lifeboat_mutex_lock(lifeboat_mutex_t *mtx)
{
    pthread_mutex_lock(mtx);
}
static inline void lifeboat_mutex_unlock(lifeboat_mutex_t *mtx)
{
    pthread_mutex_unlock(mtx);
}
static inline void lifeboat_cond_init(lifeboat_cond_t *cond)
{
    pthread_cond_init(cond, NULL);
}
static inline void lifeboat_cond_destroy(lifeboat_cond_t *cond)
{
    pthread_cond_destroy(cond);
}
static inline void lifeboat_cond_signal(lifeboat_cond_t *cond)
{
    pthread_cond_signal(cond);
}
static inline void lifeboat_cond_wait(lifeboat_cond_t *cond, lifeboat_mutex_t *mtx)
{
    pthread_cond_wait(cond, mtx);
}
static inline void lifeboat_thread_create(lifeboat_thread_t *thread,
                                          void *(*func)(void *), void *arg)
{
    pthread_create(thread, NULL, func, arg);
}
static inline void lifeboat_thread_join(lifeboat_thread_t thread)
{
    pthread_join(thread, NULL);
}
#endif

/*═══════════════════════════════════════════════════════════════════════════════
 * VALIDATION
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * Check transition matrix stickiness (diagonal should be > threshold)
 */
static bool validate_stickiness(const float *trans, int K, float min_diag)
{
    for (int k = 0; k < K; k++)
    {
        if (trans[k * K + k] < min_diag)
        {
            return false;
        }
    }
    return true;
}

/**
 * Check transition matrix rows sum to 1
 */
static bool validate_trans_rows(const float *trans, int K)
{
    for (int i = 0; i < K; i++)
    {
        float sum = 0;
        for (int j = 0; j < K; j++)
        {
            float p = trans[i * K + j];
            if (p < 0 || p > 1)
                return false;
            sum += p;
        }
        if (fabsf(sum - 1.0f) > 1e-4f)
            return false;
    }
    return true;
}

/*═══════════════════════════════════════════════════════════════════════════════
 * WORKER THREAD
 *═══════════════════════════════════════════════════════════════════════════════*/

static LIFEBOAT_THREAD_RETURN lifeboat_worker_thread(void *arg)
{
    LifeboatManager *mgr = (LifeboatManager *)arg;

    /* Allocate PGAS state once (reused across runs) */
    PGASMKLState *pgas = pgas_mkl_alloc(mgr->N, mgr->buffer_size, mgr->K, mgr->seed);

    while (1)
    {
        /* Wait for work or shutdown */
        lifeboat_mutex_lock(&mgr->mutex);
        while (get_thread_state(mgr) == LIFEBOAT_THREAD_IDLE && !mgr->shutdown_requested)
        {
            lifeboat_cond_wait(&mgr->cond_start, &mgr->mutex);
        }

        if (mgr->shutdown_requested)
        {
            lifeboat_mutex_unlock(&mgr->mutex);
            break;
        }

        /* Copy snapshot to pre-allocated worker buffers (NO MALLOC!) */
        int T = mgr->snapshot_count;
        memcpy(mgr->worker_obs, mgr->snapshot_obs, T * sizeof(float));
        memcpy(mgr->worker_ref_regimes, mgr->ref_regimes, T * sizeof(int));
        memcpy(mgr->worker_ref_h, mgr->ref_h, T * sizeof(float));

        uint64_t source_tick = mgr->snapshot_end_tick;

        /* Get model parameters */
        int cloud_idx = mgr->active_cloud;
        LifeboatCloud *cloud = &mgr->clouds[cloud_idx];

        float trans_f[LIFEBOAT_MAX_REGIMES * LIFEBOAT_MAX_REGIMES];
        float mu_vol_f[LIFEBOAT_MAX_REGIMES];
        float sigma_vol_f[LIFEBOAT_MAX_REGIMES];
        float phi_f, sigma_h_f;
        int K = mgr->K;

        memcpy(trans_f, mgr->clouds[0].trans, K * K * sizeof(float));
        memcpy(mu_vol_f, mgr->clouds[0].mu_vol, K * sizeof(float));
        memcpy(sigma_vol_f, mgr->clouds[0].sigma_vol, K * sizeof(float));
        phi_f = mgr->clouds[0].phi;
        sigma_h_f = mgr->clouds[0].sigma_h;

        lifeboat_mutex_unlock(&mgr->mutex);

        /* ═══════════════════════════════════════════════════════════════════
         * CONVERT TO DOUBLE (on pre-allocated buffers)
         * ═══════════════════════════════════════════════════════════════════*/
        for (int t = 0; t < T; t++)
        {
            mgr->worker_obs_d[t] = (double)mgr->worker_obs[t];
            mgr->worker_ref_h_d[t] = (double)mgr->worker_ref_h[t];
        }

        double trans_d[LIFEBOAT_MAX_REGIMES * LIFEBOAT_MAX_REGIMES];
        double mu_vol_d[LIFEBOAT_MAX_REGIMES];
        double sigma_vol_d[LIFEBOAT_MAX_REGIMES];
        for (int i = 0; i < K * K; i++)
        {
            trans_d[i] = (double)trans_f[i];
        }
        for (int i = 0; i < K; i++)
        {
            mu_vol_d[i] = (double)mu_vol_f[i];
            sigma_vol_d[i] = (double)sigma_vol_f[i];
        }

        /* ═══════════════════════════════════════════════════════════════════
         * RUN PGAS + PARIS (outside lock)
         * ═══════════════════════════════════════════════════════════════════*/
        double t_start = get_time_ms();

        pgas_mkl_set_model(pgas, trans_d, mu_vol_d, sigma_vol_d,
                           (double)phi_f, (double)sigma_h_f);
        pgas_mkl_set_reference(pgas, mgr->worker_ref_regimes, mgr->worker_ref_h_d, T);
        pgas_mkl_load_observations(pgas, mgr->worker_obs_d, T);

        int result = pgas_mkl_run_adaptive(pgas);
        pgas_paris_backward_smooth(pgas);

        double t_end = get_time_ms();

        /* ═══════════════════════════════════════════════════════════════════
         * EXTRACT RESULTS (with lock)
         * ═══════════════════════════════════════════════════════════════════*/
        lifeboat_mutex_lock(&mgr->mutex);

        cloud->N = mgr->N;
        cloud->K = K;
        cloud->T = T;
        cloud->source_tick_id = source_tick;
        cloud->compute_time_ms = t_end - t_start;
        cloud->ancestor_acceptance = pgas->acceptance_rate;
        cloud->pgas_sweeps = pgas->current_sweep;

        /* Copy model parameters to cloud */
        memcpy(cloud->trans, trans_f, K * K * sizeof(float));
        memcpy(cloud->mu_vol, mu_vol_f, K * sizeof(float));
        memcpy(cloud->sigma_vol, sigma_vol_f, K * sizeof(float));
        cloud->phi = phi_f;
        cloud->sigma_h = sigma_h_f;

        /* Extract smoothed particles */
        int final_regimes[LIFEBOAT_MAX_PARTICLES];
        float final_h[LIFEBOAT_MAX_PARTICLES];
        pgas_paris_get_smoothed(pgas, T - 1, final_regimes, final_h);

        for (int n = 0; n < mgr->N; n++)
        {
            cloud->regimes[n] = final_regimes[n];
            cloud->h[n] = final_h[n];
            cloud->weights[n] = 1.0f / mgr->N;
        }

        /* Validate cloud */
        bool valid_acceptance = (result != 2) && (pgas->acceptance_rate >= 0.10f);
        bool valid_stickiness = validate_stickiness(trans_f, K, 0.5f);
        bool valid_trans = validate_trans_rows(trans_f, K);

        cloud->valid = valid_acceptance && valid_stickiness && valid_trans;

        /* Update stats */
        mgr->stats.total_compute_time_ms += cloud->compute_time_ms;
        if (cloud->valid)
        {
            mgr->stats.total_injections++;
            float n_inj = (float)mgr->stats.total_injections;
            mgr->stats.avg_acceptance_rate =
                (mgr->stats.avg_acceptance_rate * (n_inj - 1) + cloud->ancestor_acceptance) / n_inj;
        }
        else
        {
            mgr->stats.failed_runs++;
        }
        if (mgr->stats.total_triggers > 0)
        {
            mgr->stats.avg_compute_time_ms =
                mgr->stats.total_compute_time_ms / mgr->stats.total_triggers;
        }

        /* Mark ready (atomic for lock-free check!) */
        mgr->ready_cloud = cloud_idx;
        mgr->active_cloud = 1 - cloud_idx;
        set_thread_state(mgr, LIFEBOAT_THREAD_READY);

        lifeboat_cond_signal(&mgr->cond_done);
        lifeboat_mutex_unlock(&mgr->mutex);
    }

    pgas_mkl_free(pgas);
    return LIFEBOAT_THREAD_RETURN_VALUE;
}

/*═══════════════════════════════════════════════════════════════════════════════
 * LIFECYCLE
 *═══════════════════════════════════════════════════════════════════════════════*/

LifeboatManager *lifeboat_create(int N, int K, int buffer_size, uint32_t seed)
{
    LifeboatManager *mgr = (LifeboatManager *)calloc(1, sizeof(LifeboatManager));
    if (!mgr)
        return NULL;

    mgr->N = N;
    mgr->K = K;
    mgr->buffer_size = buffer_size;
    mgr->seed = seed;

    /* Default configuration */
    mgr->config.enable_periodic = true;
    mgr->config.periodic_interval = 100;
    mgr->config.enable_ess = true;
    mgr->config.ess_threshold = 0.5f;
    mgr->config.enable_kl = false;
    mgr->config.kl_threshold = 1.0f;
    mgr->config.mode = LIFEBOAT_MODE_REPLACE;
    mgr->config.mix_alpha = 0.5f;
    mgr->config.cooldown_ticks = 50;

    /* Allocate main thread buffers */
    mgr->snapshot_obs = (float *)mkl_malloc(buffer_size * sizeof(float), LIFEBOAT_ALIGN);
    mgr->snapshot_ticks = (uint64_t *)mkl_malloc(buffer_size * sizeof(uint64_t), LIFEBOAT_ALIGN);
    mgr->ref_regimes = (int *)mkl_malloc(buffer_size * sizeof(int), LIFEBOAT_ALIGN);
    mgr->ref_h = (float *)mkl_malloc(buffer_size * sizeof(float), LIFEBOAT_ALIGN);

    /* Pre-allocate worker thread buffers (avoid malloc in hot path!) */
    mgr->worker_obs = (float *)mkl_malloc(buffer_size * sizeof(float), LIFEBOAT_ALIGN);
    mgr->worker_obs_d = (double *)mkl_malloc(buffer_size * sizeof(double), LIFEBOAT_ALIGN);
    mgr->worker_ref_regimes = (int *)mkl_malloc(buffer_size * sizeof(int), LIFEBOAT_ALIGN);
    mgr->worker_ref_h = (float *)mkl_malloc(buffer_size * sizeof(float), LIFEBOAT_ALIGN);
    mgr->worker_ref_h_d = (double *)mkl_malloc(buffer_size * sizeof(double), LIFEBOAT_ALIGN);

    /* Initialize clouds */
    for (int c = 0; c < 2; c++)
    {
        mgr->clouds[c].N = N;
        mgr->clouds[c].K = K;
        mgr->clouds[c].valid = false;

        float unif = 1.0f / K;
        for (int i = 0; i < K; i++)
        {
            for (int j = 0; j < K; j++)
            {
                mgr->clouds[c].trans[i * K + j] = unif;
            }
            mgr->clouds[c].mu_vol[i] = -1.0f + 0.5f * i;
            mgr->clouds[c].sigma_vol[i] = 0.3f;
        }
        mgr->clouds[c].phi = 0.97f;
        mgr->clouds[c].sigma_h = 0.15f;
    }

    mgr->active_cloud = 0;
    mgr->ready_cloud = -1;

    /* Initialize synchronization */
    lifeboat_mutex_init(&mgr->mutex);
    lifeboat_cond_init(&mgr->cond_start);
    lifeboat_cond_init(&mgr->cond_done);

    set_thread_state(mgr, LIFEBOAT_THREAD_IDLE);
    mgr->shutdown_requested = false;

    /* Spawn worker thread */
    lifeboat_thread_create(&mgr->worker_thread, lifeboat_worker_thread, mgr);

    return mgr;
}

void lifeboat_destroy(LifeboatManager *mgr)
{
    if (!mgr)
        return;

    /* Signal shutdown */
    lifeboat_mutex_lock(&mgr->mutex);
    mgr->shutdown_requested = true;
    lifeboat_cond_signal(&mgr->cond_start);
    lifeboat_mutex_unlock(&mgr->mutex);

    lifeboat_thread_join(mgr->worker_thread);

    lifeboat_mutex_destroy(&mgr->mutex);
    lifeboat_cond_destroy(&mgr->cond_start);
    lifeboat_cond_destroy(&mgr->cond_done);

    mkl_free(mgr->snapshot_obs);
    mkl_free(mgr->snapshot_ticks);
    mkl_free(mgr->ref_regimes);
    mkl_free(mgr->ref_h);
    mkl_free(mgr->worker_obs);
    mkl_free(mgr->worker_obs_d);
    mkl_free(mgr->worker_ref_regimes);
    mkl_free(mgr->worker_ref_h);
    mkl_free(mgr->worker_ref_h_d);
    free(mgr);
}

void lifeboat_configure(LifeboatManager *mgr, const LifeboatConfig *config)
{
    if (!mgr || !config)
        return;
    lifeboat_mutex_lock(&mgr->mutex);
    mgr->config = *config;
    lifeboat_mutex_unlock(&mgr->mutex);
}

void lifeboat_set_model(LifeboatManager *mgr,
                        const float *trans,
                        const float *mu_vol,
                        const float *sigma_vol,
                        float phi,
                        float sigma_h)
{
    if (!mgr)
        return;
    lifeboat_mutex_lock(&mgr->mutex);
    for (int c = 0; c < 2; c++)
    {
        memcpy(mgr->clouds[c].trans, trans, mgr->K * mgr->K * sizeof(float));
        memcpy(mgr->clouds[c].mu_vol, mu_vol, mgr->K * sizeof(float));
        memcpy(mgr->clouds[c].sigma_vol, sigma_vol, mgr->K * sizeof(float));
        mgr->clouds[c].phi = phi;
        mgr->clouds[c].sigma_h = sigma_h;
    }
    lifeboat_mutex_unlock(&mgr->mutex);
}

/*═══════════════════════════════════════════════════════════════════════════════
 * TRIGGER & CHECK (Lock-free hot path!)
 *═══════════════════════════════════════════════════════════════════════════════*/

LifeboatTrigger lifeboat_check_trigger(LifeboatManager *mgr,
                                       uint64_t tick_id,
                                       float ess,
                                       float kl)
{
    if (!mgr)
        return LIFEBOAT_TRIGGER_NONE;

    /* LOCK-FREE check: is worker already running? */
    LifeboatThreadState state = get_thread_state(mgr);
    if (state == LIFEBOAT_THREAD_RUNNING)
    {
        return LIFEBOAT_TRIGGER_NONE;
    }

    /* Check cooldown (use cached values, slight race is OK) */
    if (tick_id - mgr->last_trigger_tick < (uint64_t)mgr->config.cooldown_ticks)
    {
        return LIFEBOAT_TRIGGER_NONE;
    }

    mgr->current_tick = tick_id;

    LifeboatTrigger trigger = LIFEBOAT_TRIGGER_NONE;

    /* ESS trigger (highest priority) */
    if (mgr->config.enable_ess && ess >= 0)
    {
        float threshold = mgr->config.ess_threshold * mgr->N;
        if (ess < threshold)
        {
            trigger = LIFEBOAT_TRIGGER_ESS;
            mgr->stats.ess_triggers++;
        }
    }

    /* KL trigger */
    if (trigger == LIFEBOAT_TRIGGER_NONE && mgr->config.enable_kl && kl >= 0)
    {
        if (kl > mgr->config.kl_threshold)
        {
            trigger = LIFEBOAT_TRIGGER_KL;
            mgr->stats.kl_triggers++;
        }
    }

    /* Periodic trigger */
    if (trigger == LIFEBOAT_TRIGGER_NONE && mgr->config.enable_periodic)
    {
        if (tick_id - mgr->last_trigger_tick >= (uint64_t)mgr->config.periodic_interval)
        {
            trigger = LIFEBOAT_TRIGGER_PERIODIC;
            mgr->stats.periodic_triggers++;
        }
    }

    if (trigger != LIFEBOAT_TRIGGER_NONE)
    {
        mgr->last_trigger_type = trigger;
    }

    return trigger;
}

bool lifeboat_trigger_manual(LifeboatManager *mgr)
{
    if (!mgr)
        return false;

    if (get_thread_state(mgr) == LIFEBOAT_THREAD_RUNNING)
    {
        return false;
    }

    mgr->last_trigger_type = LIFEBOAT_TRIGGER_MANUAL;
    mgr->stats.manual_triggers++;
    return true;
}

bool lifeboat_start_run(LifeboatManager *mgr,
                        const float *observations,
                        const uint64_t *tick_ids,
                        int count,
                        const int *ref_regimes,
                        const float *ref_h)
{
    if (!mgr || !observations || count < 2)
        return false;

    lifeboat_mutex_lock(&mgr->mutex);

    if (get_thread_state(mgr) == LIFEBOAT_THREAD_RUNNING)
    {
        lifeboat_mutex_unlock(&mgr->mutex);
        return false;
    }

    int T = (count < mgr->buffer_size) ? count : mgr->buffer_size;
    mgr->snapshot_count = T;

    memcpy(mgr->snapshot_obs, observations, T * sizeof(float));
    if (tick_ids)
    {
        memcpy(mgr->snapshot_ticks, tick_ids, T * sizeof(uint64_t));
        mgr->snapshot_start_tick = tick_ids[0];
        mgr->snapshot_end_tick = tick_ids[T - 1];
    }
    else
    {
        mgr->snapshot_start_tick = 0;
        mgr->snapshot_end_tick = T - 1;
    }

    if (ref_regimes && ref_h)
    {
        memcpy(mgr->ref_regimes, ref_regimes, T * sizeof(int));
        memcpy(mgr->ref_h, ref_h, T * sizeof(float));
    }
    else
    {
        for (int t = 0; t < T; t++)
        {
            mgr->ref_regimes[t] = 0;
            mgr->ref_h[t] = mgr->clouds[0].mu_vol[0];
        }
    }

    mgr->last_trigger_tick = mgr->current_tick;
    mgr->stats.total_triggers++;
    set_thread_state(mgr, LIFEBOAT_THREAD_RUNNING);

    lifeboat_cond_signal(&mgr->cond_start);
    lifeboat_mutex_unlock(&mgr->mutex);

    return true;
}

/*═══════════════════════════════════════════════════════════════════════════════
 * LOCK-FREE STATUS CHECKS
 *═══════════════════════════════════════════════════════════════════════════════*/

bool lifeboat_is_ready(const LifeboatManager *mgr)
{
    if (!mgr)
        return false;
    /* Atomic read - safe to call every tick without lock! */
    return get_thread_state(mgr) == LIFEBOAT_THREAD_READY && mgr->ready_cloud >= 0;
}

bool lifeboat_is_running(const LifeboatManager *mgr)
{
    if (!mgr)
        return false;
    return get_thread_state(mgr) == LIFEBOAT_THREAD_RUNNING;
}

uint64_t lifeboat_get_source_tick(const LifeboatManager *mgr)
{
    if (!mgr || mgr->ready_cloud < 0)
        return 0;
    return mgr->clouds[mgr->ready_cloud].source_tick_id;
}

/*═══════════════════════════════════════════════════════════════════════════════
 * INJECTION
 *═══════════════════════════════════════════════════════════════════════════════*/

const LifeboatCloud *lifeboat_get_cloud(const LifeboatManager *mgr)
{
    if (!mgr || mgr->ready_cloud < 0)
        return NULL;
    const LifeboatCloud *cloud = &mgr->clouds[mgr->ready_cloud];
    return cloud->valid ? cloud : NULL;
}

bool lifeboat_inject(LifeboatManager *mgr,
                     int *rbpf_regimes,
                     float *rbpf_h,
                     float *rbpf_weights,
                     uint64_t *source_tick_out,
                     uint64_t *rng_state)
{
    if (!mgr || !rbpf_regimes || !rbpf_h || !rbpf_weights)
        return false;

    lifeboat_mutex_lock(&mgr->mutex);

    if (mgr->ready_cloud < 0)
    {
        lifeboat_mutex_unlock(&mgr->mutex);
        return false;
    }

    const LifeboatCloud *cloud = &mgr->clouds[mgr->ready_cloud];
    if (!cloud->valid)
    {
        lifeboat_mutex_unlock(&mgr->mutex);
        return false;
    }

    /* Return source tick for catch-up */
    if (source_tick_out)
    {
        *source_tick_out = cloud->source_tick_id;
    }

    int N = mgr->N;
    LifeboatMode mode = mgr->config.mode;
    float alpha = mgr->config.mix_alpha;

    switch (mode)
    {
    case LIFEBOAT_MODE_REPLACE:
        memcpy(rbpf_regimes, cloud->regimes, N * sizeof(int));
        memcpy(rbpf_h, cloud->h, N * sizeof(float));
        memcpy(rbpf_weights, cloud->weights, N * sizeof(float));
        break;

    case LIFEBOAT_MODE_MIX:
        for (int n = 0; n < N; n++)
        {
            rbpf_regimes[n] = cloud->regimes[n];
            rbpf_h[n] = (1.0f - alpha) * rbpf_h[n] + alpha * cloud->h[n];
            rbpf_weights[n] = 1.0f / N;
        }
        break;

    case LIFEBOAT_MODE_RESAMPLE:
    {
        float combined_w[2 * LIFEBOAT_MAX_PARTICLES];
        float wsum = 0;

        for (int n = 0; n < N; n++)
        {
            combined_w[n] = (1.0f - alpha) * rbpf_weights[n];
            combined_w[N + n] = alpha * cloud->weights[n];
            wsum += combined_w[n] + combined_w[N + n];
        }

        float inv_wsum = 1.0f / wsum;
        for (int n = 0; n < 2 * N; n++)
        {
            combined_w[n] *= inv_wsum;
        }

        float cdf[2 * LIFEBOAT_MAX_PARTICLES];
        cdf[0] = combined_w[0];
        for (int n = 1; n < 2 * N; n++)
        {
            cdf[n] = cdf[n - 1] + combined_w[n];
        }

        /* Systematic resampling with lock-free RNG */
        float u0;
        if (rng_state)
        {
            u0 = lifeboat_rng_uniform(rng_state) / N;
        }
        else
        {
            u0 = (float)rand() / RAND_MAX / N; /* Fallback */
        }
        int j = 0;

        /* Temp storage for resampling */
        int temp_regimes[LIFEBOAT_MAX_PARTICLES];
        float temp_h[LIFEBOAT_MAX_PARTICLES];
        memcpy(temp_regimes, rbpf_regimes, N * sizeof(int));
        memcpy(temp_h, rbpf_h, N * sizeof(float));

        for (int n = 0; n < N; n++)
        {
            float u = u0 + (float)n / N;
            while (j < 2 * N - 1 && cdf[j] < u)
                j++;

            if (j < N)
            {
                rbpf_regimes[n] = temp_regimes[j];
                rbpf_h[n] = temp_h[j];
            }
            else
            {
                rbpf_regimes[n] = cloud->regimes[j - N];
                rbpf_h[n] = cloud->h[j - N];
            }
            rbpf_weights[n] = 1.0f / N;
        }
    }
    break;
    }

    mgr->last_injection_tick = mgr->current_tick;
    mgr->stats.last_injection_tick = mgr->current_tick;

    lifeboat_mutex_unlock(&mgr->mutex);
    return true;
}

void lifeboat_consume_cloud(LifeboatManager *mgr)
{
    if (!mgr)
        return;
    lifeboat_mutex_lock(&mgr->mutex);
    mgr->ready_cloud = -1;
    set_thread_state(mgr, LIFEBOAT_THREAD_IDLE);
    lifeboat_mutex_unlock(&mgr->mutex);
}

/*═══════════════════════════════════════════════════════════════════════════════
 * FAST-FORWARD (CATCH-UP) — Lock-free RNG version
 *═══════════════════════════════════════════════════════════════════════════════*/

void lifeboat_fast_forward(const LifeboatCloud *cloud,
                           int *regimes,
                           float *h,
                           float *weights,
                           const float *observations,
                           int n_ticks,
                           bool reweight,
                           uint64_t *rng_state)
{
    if (!cloud || !regimes || !h || !weights || n_ticks <= 0 || !rng_state)
        return;

    const int N = cloud->N;
    const int K = cloud->K;
    const float phi = cloud->phi;
    const float sigma_h = cloud->sigma_h;
    const float log_2pi = 1.8378770664f;

    /* For each catch-up tick */
    for (int t = 0; t < n_ticks; t++)
    {
        float y = observations ? observations[t] : 0.0f;
        float y_sq = y * y;

        float log_w_max = -1e30f;
        float log_weights_local[LIFEBOAT_MAX_PARTICLES];

        /* Propagate each particle */
        for (int n = 0; n < N; n++)
        {
            int regime = regimes[n];
            float h_old = h[n];

            /* Sample new regime using xoroshiro (NO GLOBAL LOCK!) */
            float u = lifeboat_rng_uniform(rng_state);
            float cumsum = 0;
            int new_regime = regime;
            for (int j = 0; j < K; j++)
            {
                cumsum += cloud->trans[regime * K + j];
                if (u < cumsum)
                {
                    new_regime = j;
                    break;
                }
            }

            /* AR(1) propagation with Gaussian noise */
            float mu_new = cloud->mu_vol[new_regime];
            float mean = mu_new + phi * (h_old - mu_new);
            float noise = sigma_h * lifeboat_rng_normal(rng_state);
            float h_new = mean + noise;

            regimes[n] = new_regime;
            h[n] = h_new;

            if (reweight && observations)
            {
                /* Compute log-likelihood: log P(y | h) */
                float exp_neg_h = expf(-h_new);
                float log_lik = -0.5f * (log_2pi + h_new + y_sq * exp_neg_h);
                log_weights_local[n] = logf(weights[n] + 1e-30f) + log_lik;

                if (log_weights_local[n] > log_w_max)
                {
                    log_w_max = log_weights_local[n];
                }
            }
        }

        /* Normalize weights if reweighting */
        if (reweight && observations)
        {
            float w_sum = 0;
            for (int n = 0; n < N; n++)
            {
                weights[n] = expf(log_weights_local[n] - log_w_max);
                w_sum += weights[n];
            }
            float inv_sum = 1.0f / w_sum;
            for (int n = 0; n < N; n++)
            {
                weights[n] *= inv_sum;
            }
        }
    }
}

/*═══════════════════════════════════════════════════════════════════════════════
 * DIAGNOSTICS
 *═══════════════════════════════════════════════════════════════════════════════*/

const LifeboatStats *lifeboat_get_stats(const LifeboatManager *mgr)
{
    return mgr ? &mgr->stats : NULL;
}

void lifeboat_reset_stats(LifeboatManager *mgr)
{
    if (!mgr)
        return;
    lifeboat_mutex_lock(&mgr->mutex);
    memset(&mgr->stats, 0, sizeof(LifeboatStats));
    lifeboat_mutex_unlock(&mgr->mutex);
}

void lifeboat_print_diagnostics(const LifeboatManager *mgr)
{
    if (!mgr)
        return;

    const char *state_str;
    LifeboatThreadState state = get_thread_state(mgr);
    switch (state)
    {
    case LIFEBOAT_THREAD_IDLE:
        state_str = "IDLE";
        break;
    case LIFEBOAT_THREAD_RUNNING:
        state_str = "RUNNING";
        break;
    case LIFEBOAT_THREAD_READY:
        state_str = "READY";
        break;
    case LIFEBOAT_THREAD_SHUTDOWN:
        state_str = "SHUTDOWN";
        break;
    default:
        state_str = "UNKNOWN";
    }

    const char *mode_str;
    switch (mgr->config.mode)
    {
    case LIFEBOAT_MODE_REPLACE:
        mode_str = "REPLACE";
        break;
    case LIFEBOAT_MODE_MIX:
        mode_str = "MIX";
        break;
    case LIFEBOAT_MODE_RESAMPLE:
        mode_str = "RESAMPLE";
        break;
    default:
        mode_str = "UNKNOWN";
    }

    printf("═══════════════════════════════════════════════════════════\n");
    printf("LIFEBOAT DIAGNOSTICS (Lock-free optimized)\n");
    printf("═══════════════════════════════════════════════════════════\n");
    printf("Configuration:\n");
    printf("  Particles:        %d\n", mgr->N);
    printf("  Regimes:          %d\n", mgr->K);
    printf("  Buffer size:      %d\n", mgr->buffer_size);
    printf("  Mode:             %s\n", mode_str);
    printf("  Mix alpha:        %.2f\n", mgr->config.mix_alpha);
    printf("  Cooldown:         %d ticks\n", mgr->config.cooldown_ticks);
    printf("\nTriggers:\n");
    printf("  Periodic:         %s (every %d ticks)\n",
           mgr->config.enable_periodic ? "ON" : "OFF", mgr->config.periodic_interval);
    printf("  ESS:              %s (< %.1f%%)\n",
           mgr->config.enable_ess ? "ON" : "OFF", mgr->config.ess_threshold * 100);
    printf("  KL:               %s (> %.2f)\n",
           mgr->config.enable_kl ? "ON" : "OFF", mgr->config.kl_threshold);
    printf("\nState:\n");
    printf("  Thread state:     %s (atomic)\n", state_str);
    printf("  Current tick:     %" PRIu64 "\n", mgr->current_tick);
    printf("  Last trigger:     %" PRIu64 "\n", mgr->last_trigger_tick);
    printf("  Last injection:   %" PRIu64 "\n", mgr->last_injection_tick);
    printf("  Snapshot range:   %" PRIu64 " - %" PRIu64 "\n", mgr->snapshot_start_tick, mgr->snapshot_end_tick);
    printf("  Ready cloud:      %d\n", mgr->ready_cloud);
    printf("\nStatistics:\n");
    printf("  Total triggers:   %" PRIu64 "\n", mgr->stats.total_triggers);
    printf("  Total injections: %" PRIu64 "\n", mgr->stats.total_injections);
    printf("  Failed runs:      %" PRIu64 "\n", mgr->stats.failed_runs);
    printf("  By type:          periodic=%" PRIu64 ", ess=%" PRIu64 ", kl=%" PRIu64 ", manual=%" PRIu64 "\n",
           mgr->stats.periodic_triggers, mgr->stats.ess_triggers,
           mgr->stats.kl_triggers, mgr->stats.manual_triggers);
    printf("  Avg compute time: %.2f ms\n", mgr->stats.avg_compute_time_ms);
    printf("  Avg acceptance:   %.3f\n", mgr->stats.avg_acceptance_rate);
    printf("═══════════════════════════════════════════════════════════\n");
}