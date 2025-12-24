/**
 * @file rbpf_fixed_lag_smoother.c
 * @brief Fixed-Lag PARIS Smoother Implementation
 */

#include "rbpf_fixed_lag_smoother.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#ifdef _WIN32
#include <windows.h>
static double get_time_us(void) {
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart * 1e6 / (double)freq.QuadPart;
}
#else
#include <time.h>
static double get_time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}
#endif

/* MKL for aligned alloc */
#include <mkl.h>

/*═══════════════════════════════════════════════════════════════════════════
 * HELPERS
 *═══════════════════════════════════════════════════════════════════════════*/

/* Pad N to multiple of 16 for SIMD */
static inline int pad_n(int n) {
    return ((n + 15) / 16) * 16;
}

/* Ring buffer index calculation */
static inline int ring_idx(const RBPF_FixedLagSmoother *fls, int offset) {
    /* offset=0 is oldest, offset=count-1 is newest */
    int idx = (fls->head - fls->count + offset + fls->buffer_size) % fls->buffer_size;
    return idx;
}

/*═══════════════════════════════════════════════════════════════════════════
 * LIFECYCLE
 *═══════════════════════════════════════════════════════════════════════════*/

RBPF_FixedLagSmoother *fls_create(int n_particles, int n_regimes, 
                                   int lag, uint32_t rng_seed)
{
    if (n_particles < 1 || n_regimes < 1 || lag < 2 || lag > FLS_MAX_LAG) {
        return NULL;
    }
    
    RBPF_FixedLagSmoother *fls = (RBPF_FixedLagSmoother *)mkl_calloc(
        1, sizeof(RBPF_FixedLagSmoother), FLS_ALIGN);
    if (!fls) return NULL;
    
    fls->lag = lag;
    fls->n_particles = n_particles;
    fls->n_particles_padded = pad_n(n_particles);
    fls->n_regimes = n_regimes;
    fls->buffer_size = lag + 1;  /* Need L+1 for both ell and ell_lag */
    fls->head = 0;
    fls->count = 0;
    fls->enabled = 1;
    
    int N = fls->n_particles;
    int Np = fls->n_particles_padded;
    int buf_size = fls->buffer_size;
    size_t float_buf = (size_t)buf_size * Np * sizeof(float);
    size_t int_buf = (size_t)buf_size * Np * sizeof(int);
    
    /* Allocate ring buffers */
    fls->h_buffer = (float *)mkl_malloc(float_buf, FLS_ALIGN);
    fls->regime_buffer = (int *)mkl_malloc(int_buf, FLS_ALIGN);
    fls->log_weight_buffer = (float *)mkl_malloc(float_buf, FLS_ALIGN);
    fls->ancestor_buffer = (int *)mkl_malloc(int_buf, FLS_ALIGN);
    
    if (!fls->h_buffer || !fls->regime_buffer || 
        !fls->log_weight_buffer || !fls->ancestor_buffer) {
        fls_destroy(fls);
        return NULL;
    }
    
    memset(fls->h_buffer, 0, float_buf);
    memset(fls->regime_buffer, 0, int_buf);
    memset(fls->log_weight_buffer, 0, float_buf);
    memset(fls->ancestor_buffer, 0, int_buf);
    
    /*───────────────────────────────────────────────────────────────────────
     * PER-INSTANCE REORDER WORKSPACE (thread safety for MMPF)
     *
     * CRITICAL: Must be per-instance, not static!
     * Size: buffer_size × N (PARIS uses unpadded N)
     *───────────────────────────────────────────────────────────────────────*/
    size_t ordered_size = (size_t)buf_size * N;
    fls->h_ordered = (double *)mkl_malloc(ordered_size * sizeof(double), FLS_ALIGN);
    fls->w_ordered = (double *)mkl_malloc(ordered_size * sizeof(double), FLS_ALIGN);
    fls->r_ordered = (int *)mkl_malloc(ordered_size * sizeof(int), FLS_ALIGN);
    fls->a_ordered = (int *)mkl_malloc(ordered_size * sizeof(int), FLS_ALIGN);
    
    if (!fls->h_ordered || !fls->w_ordered || 
        !fls->r_ordered || !fls->a_ordered) {
        fls_destroy(fls);
        return NULL;
    }
    
    /* Allocate smoothed output buffers */
    fls->smoothed_h = (float *)mkl_malloc(Np * sizeof(float), FLS_ALIGN);
    fls->smoothed_h_lag = (float *)mkl_malloc(Np * sizeof(float), FLS_ALIGN);
    fls->smoothed_regime = (int *)mkl_malloc(Np * sizeof(int), FLS_ALIGN);
    fls->smoothed_weight = (float *)mkl_malloc(Np * sizeof(float), FLS_ALIGN);
    
    if (!fls->smoothed_h || !fls->smoothed_h_lag || 
        !fls->smoothed_regime || !fls->smoothed_weight) {
        fls_destroy(fls);
        return NULL;
    }
    
    /* Allocate ParticleInfo bridge */
    fls->particle_info = (ParticleInfo *)mkl_malloc(
        N * sizeof(ParticleInfo), FLS_ALIGN);
    if (!fls->particle_info) {
        fls_destroy(fls);
        return NULL;
    }
    
    /* Create PARIS state */
    fls->paris = paris_mkl_alloc(N, buf_size, n_regimes, rng_seed);
    if (!fls->paris) {
        fls_destroy(fls);
        return NULL;
    }
    
    return fls;
}

void fls_destroy(RBPF_FixedLagSmoother *fls)
{
    if (!fls) return;
    
    if (fls->paris) paris_mkl_free(fls->paris);
    
    /* Ring buffers */
    mkl_free(fls->h_buffer);
    mkl_free(fls->regime_buffer);
    mkl_free(fls->log_weight_buffer);
    mkl_free(fls->ancestor_buffer);
    
    /* Per-instance reorder workspace */
    mkl_free(fls->h_ordered);
    mkl_free(fls->w_ordered);
    mkl_free(fls->r_ordered);
    mkl_free(fls->a_ordered);
    
    /* Smoothed output */
    mkl_free(fls->smoothed_h);
    mkl_free(fls->smoothed_h_lag);
    mkl_free(fls->smoothed_regime);
    mkl_free(fls->smoothed_weight);
    mkl_free(fls->particle_info);
    
    mkl_free(fls);
}

void fls_reset(RBPF_FixedLagSmoother *fls)
{
    if (!fls) return;
    
    fls->head = 0;
    fls->count = 0;
    fls->smoothing_calls = 0;
    fls->emergency_flushes = 0;
    fls->ticks_processed = 0;
    fls->total_smooth_time_us = 0;
    fls->last_smooth_time_us = 0;
}

/*═══════════════════════════════════════════════════════════════════════════
 * CONFIGURATION
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Set model parameters for PARIS backward smoothing
 * 
 * ALIGNED WITH RBPF: Uses per-regime sigma_vol[k] for AR process noise.
 * 
 * @param fls        Fixed-lag smoother
 * @param trans      Transition matrix [K×K] row-major
 * @param mu_vol     Per-regime log-vol means [K]
 * @param sigma_vol  Per-regime AR process noise [K]
 * @param phi        AR(1) persistence
 */
void fls_set_model(RBPF_FixedLagSmoother *fls,
                   const double *trans,
                   const double *mu_vol,
                   const double *sigma_vol,
                   double phi)
{
    if (!fls || !fls->paris) return;
    paris_mkl_set_model(fls->paris, trans, mu_vol, sigma_vol, phi);
}

void fls_enable(RBPF_FixedLagSmoother *fls, int enable)
{
    if (fls) fls->enabled = enable ? 1 : 0;
}

/*═══════════════════════════════════════════════════════════════════════════
 * MAIN API
 *═══════════════════════════════════════════════════════════════════════════*/

void fls_push(RBPF_FixedLagSmoother *fls,
              const float *h,
              const int *regimes,
              const float *log_weights,
              const int *ancestors)
{
    if (!fls || !fls->enabled) return;
    if (!h || !regimes || !log_weights) return;
    
    int N = fls->n_particles;
    int Np = fls->n_particles_padded;
    int slot = fls->head;
    
    /* Copy to ring buffer at head position */
    float *h_dst = &fls->h_buffer[slot * Np];
    int *r_dst = &fls->regime_buffer[slot * Np];
    float *w_dst = &fls->log_weight_buffer[slot * Np];
    int *a_dst = &fls->ancestor_buffer[slot * Np];
    
    memcpy(h_dst, h, N * sizeof(float));
    memcpy(r_dst, regimes, N * sizeof(int));
    memcpy(w_dst, log_weights, N * sizeof(float));
    
    if (ancestors) {
        memcpy(a_dst, ancestors, N * sizeof(int));
    } else {
        /* Identity ancestors if not provided */
        for (int i = 0; i < N; i++) {
            a_dst[i] = i;
        }
    }
    
    /* Pad with zeros/identity */
    for (int i = N; i < Np; i++) {
        h_dst[i] = 0.0f;
        r_dst[i] = 0;
        w_dst[i] = -1e30f;  /* -inf for padding */
        a_dst[i] = 0;
    }
    
    /* Advance head */
    fls->head = (fls->head + 1) % fls->buffer_size;
    if (fls->count < fls->buffer_size) {
        fls->count++;
    }
    
    fls->ticks_processed++;
}

int fls_ready(const RBPF_FixedLagSmoother *fls)
{
    if (!fls || !fls->enabled) return 0;
    return (fls->count >= fls->buffer_size) ? 1 : 0;
}

double fls_smooth(RBPF_FixedLagSmoother *fls)
{
    if (!fls || !fls->enabled || !fls_ready(fls)) return 0.0;
    
    double t0 = get_time_us();
    
    int N = fls->n_particles;
    int Np = fls->n_particles_padded;
    int T = fls->buffer_size;
    
    /*───────────────────────────────────────────────────────────────────────
     * STEP 1: Load ring buffer to PARIS in temporal order
     *
     * Ring buffer may wrap around, so we need to reorder.
     * PARIS expects: [t=0, t=1, ..., t=T-1] where t=0 is oldest.
     *
     * Use per-instance workspace (thread-safe for MMPF).
     *───────────────────────────────────────────────────────────────────────*/
    
    double *h_ordered = fls->h_ordered;
    double *w_ordered = fls->w_ordered;
    int *r_ordered = fls->r_ordered;
    int *a_ordered = fls->a_ordered;
    
    /* Reorder from ring buffer to temporal sequence */
    for (int t = 0; t < T; t++) {
        int ring_slot = ring_idx(fls, t);
        
        const float *h_src = &fls->h_buffer[ring_slot * Np];
        const int *r_src = &fls->regime_buffer[ring_slot * Np];
        const float *w_src = &fls->log_weight_buffer[ring_slot * Np];
        const int *a_src = &fls->ancestor_buffer[ring_slot * Np];
        
        double *h_dst = &h_ordered[t * N];
        double *w_dst = &w_ordered[t * N];
        int *r_dst = &r_ordered[t * N];
        int *a_dst = &a_ordered[t * N];
        
        /* Find max log-weight for stable exp */
        float max_lw = w_src[0];
        for (int i = 1; i < N; i++) {
            if (w_src[i] > max_lw) max_lw = w_src[i];
        }
        
        /* Convert: h to double, log-weights to weights */
        double sum_w = 0.0;
        for (int i = 0; i < N; i++) {
            h_dst[i] = (double)h_src[i];
            r_dst[i] = r_src[i];
            w_dst[i] = exp((double)(w_src[i] - max_lw));
            sum_w += w_dst[i];
            a_dst[i] = a_src[i];
        }
        
        /* Normalize weights */
        if (sum_w > 1e-30) {
            double inv_sum = 1.0 / sum_w;
            for (int i = 0; i < N; i++) w_dst[i] *= inv_sum;
        }
    }
    
    /* Load into PARIS */
    paris_mkl_load_particles(fls->paris, 
                              r_ordered, h_ordered, w_ordered, a_ordered, T);
    
    /*───────────────────────────────────────────────────────────────────────
     * STEP 2: PARIS backward smoothing
     *───────────────────────────────────────────────────────────────────────*/
    
    paris_mkl_backward_smooth(fls->paris);
    
    /*───────────────────────────────────────────────────────────────────────
     * STEP 3: Extract smoothed values at t=0 and t=1
     *
     * PARIS gives N backward trajectories. For trajectory n:
     *   smoothed[t][n] = index of particle selected at time t
     *
     * For Storvik update at (conceptual) time t-L, we need:
     *   ell     = h̃_{t-L}   = smoothed h at t=1 (second oldest)
     *   ell_lag = h̃_{t-L-1} = smoothed h at t=0 (oldest)
     *   regime  = regime at t=1
     *
     * Each PARIS trajectory n provides one (ell, ell_lag) sample from
     * the joint smoothing distribution p(h_t, h_{t-1} | y_{1:T}).
     *───────────────────────────────────────────────────────────────────────*/
    
    float *h_lag = fls->smoothed_h_lag;   /* t=0 → h̃_{t-L-1} */
    float *h_cur = fls->smoothed_h;       /* t=1 → h̃_{t-L} */
    int *r_cur = fls->smoothed_regime;    /* t=1 → regime */
    
    paris_mkl_get_smoothed(fls->paris, 0, NULL, h_lag);
    paris_mkl_get_smoothed(fls->paris, 1, r_cur, h_cur);
    
    /* 
     * Weight assignment for Storvik:
     * PARIS backward samples are approximately uniform from the smoothing
     * distribution. Use equal weights.
     */
    float unif_w = 1.0f / (float)N;
    for (int i = 0; i < N; i++) {
        fls->smoothed_weight[i] = unif_w;
    }
    
    /*───────────────────────────────────────────────────────────────────────
     * STEP 4: Build ParticleInfo array for Storvik
     *
     * Each ParticleInfo[i] represents one smoothed trajectory sample.
     * Storvik will update particle i's sufficient stats using this
     * (ell, ell_lag) pair.
     *
     * NOTE: Trajectory i may not have originated from RBPF particle i.
     * This is fine for learning global regime parameters - we're using
     * samples from p(h_t, h_{t-1} | y_{1:T}) to compute E[z] and E[z²]
     * where z = h_t - φ·h_{t-1}.
     *───────────────────────────────────────────────────────────────────────*/
    
    for (int i = 0; i < N; i++) {
        ParticleInfo *p = &fls->particle_info[i];
        p->ell = (param_real)h_cur[i];       /* Smoothed h̃_{t-L} */
        p->ell_lag = (param_real)h_lag[i];   /* Smoothed h̃_{t-L-1} */
        p->regime = r_cur[i];
        p->prev_regime = r_cur[i];           /* No transition tracking needed */
        p->weight = (param_real)fls->smoothed_weight[i];
    }
    
    double elapsed = get_time_us() - t0;
    fls->smoothing_calls++;
    fls->total_smooth_time_us += elapsed;
    fls->last_smooth_time_us = elapsed;
    
    return elapsed;
}

const ParticleInfo *fls_get_particle_info(const RBPF_FixedLagSmoother *fls)
{
    if (!fls || !fls_ready(fls)) return NULL;
    return fls->particle_info;
}

void fls_pop(RBPF_FixedLagSmoother *fls)
{
    if (!fls || fls->count == 0) return;
    
    /* Decrement count - oldest entry is now invalid */
    /* The ring buffer automatically overwrites on next push */
    fls->count--;
}

/*═══════════════════════════════════════════════════════════════════════════
 * CONVENIENCE: Combined smooth + Storvik update
 *═══════════════════════════════════════════════════════════════════════════*/

int fls_update_storvik(RBPF_FixedLagSmoother *fls, ParamLearner *learner)
{
    if (!fls || !learner || !fls_ready(fls)) {
        return 0;
    }
    
    /* Run smoothing */
    fls_smooth(fls);
    
    /* Update Storvik with smoothed particle info */
    const ParticleInfo *info = fls_get_particle_info(fls);
    if (info) {
        param_learn_update(learner, info, fls->n_particles);
    }
    
    /* Pop oldest entry */
    fls_pop(fls);
    
    return 1;
}

/*═══════════════════════════════════════════════════════════════════════════
 * EMERGENCY FLUSH (Circuit Breaker Integration)
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Internal helper: Load partial buffer to PARIS
 */
static void fls_load_partial_to_paris(RBPF_FixedLagSmoother *fls, int T_partial)
{
    int N = fls->n_particles;
    int Np = fls->n_particles_padded;
    
    double *h_ordered = fls->h_ordered;
    double *w_ordered = fls->w_ordered;
    int *r_ordered = fls->r_ordered;
    int *a_ordered = fls->a_ordered;
    
    /* Reorder from ring buffer (only T_partial entries) */
    for (int t = 0; t < T_partial; t++) {
        int ring_slot = ring_idx(fls, t);
        
        const float *h_src = &fls->h_buffer[ring_slot * Np];
        const int *r_src = &fls->regime_buffer[ring_slot * Np];
        const float *w_src = &fls->log_weight_buffer[ring_slot * Np];
        const int *a_src = &fls->ancestor_buffer[ring_slot * Np];
        
        double *h_dst = &h_ordered[t * N];
        double *w_dst = &w_ordered[t * N];
        int *r_dst = &r_ordered[t * N];
        int *a_dst = &a_ordered[t * N];
        
        /* Find max log-weight for stable exp */
        float max_lw = w_src[0];
        for (int i = 1; i < N; i++) {
            if (w_src[i] > max_lw) max_lw = w_src[i];
        }
        
        /* Convert */
        double sum_w = 0.0;
        for (int i = 0; i < N; i++) {
            h_dst[i] = (double)h_src[i];
            r_dst[i] = r_src[i];
            w_dst[i] = exp((double)(w_src[i] - max_lw));
            sum_w += w_dst[i];
            a_dst[i] = a_src[i];
        }
        
        /* Normalize weights */
        if (sum_w > 1e-30) {
            double inv_sum = 1.0 / sum_w;
            for (int i = 0; i < N; i++) w_dst[i] *= inv_sum;
        }
    }
    
    /* Load into PARIS with reduced T */
    paris_mkl_load_particles(fls->paris, 
                              r_ordered, h_ordered, w_ordered, a_ordered, 
                              T_partial);
}

/**
 * Internal helper: Build ParticleInfo from smoothed output at time indices
 */
static void fls_build_particle_info_at(RBPF_FixedLagSmoother *fls, 
                                        int t_lag, int t_cur)
{
    int N = fls->n_particles;
    
    /* Extract from PARIS */
    paris_mkl_get_smoothed(fls->paris, t_lag, NULL, fls->smoothed_h_lag);
    paris_mkl_get_smoothed(fls->paris, t_cur, fls->smoothed_regime, fls->smoothed_h);
    
    /* Build ParticleInfo */
    float unif_w = 1.0f / (float)N;
    for (int i = 0; i < N; i++) {
        ParticleInfo *p = &fls->particle_info[i];
        p->ell = (param_real)fls->smoothed_h[i];
        p->ell_lag = (param_real)fls->smoothed_h_lag[i];
        p->regime = fls->smoothed_regime[i];
        p->prev_regime = fls->smoothed_regime[i];
        p->weight = (param_real)unif_w;
    }
}

int fls_should_flush(const RBPF_FixedLagSmoother *fls)
{
    if (!fls || !fls->enabled) return 0;
    
    /* Need at least 5 ticks for meaningful partial smoothing */
    /* With fewer, the "smoothing" is essentially just filtering */
    return (fls->count >= 5) ? 1 : 0;
}

int fls_emergency_flush(RBPF_FixedLagSmoother *fls, ParamLearner *learner)
{
    if (!fls || !learner) return 0;
    
    /* Need at least 2 entries for (ell, ell_lag) pair */
    if (fls->count < 2) return 0;
    
    double t0 = get_time_us();
    int updates_performed = 0;
    int T_partial = fls->count;
    int N = fls->n_particles;
    
    /*───────────────────────────────────────────────────────────────────────
     * STEP 1: Load partial buffer to PARIS
     *
     * Even with only 10-20 ticks, we get SOME smoothing benefit.
     * Better than waiting and letting Storvik use stale params.
     *───────────────────────────────────────────────────────────────────────*/
    
    fls_load_partial_to_paris(fls, T_partial);
    
    /*───────────────────────────────────────────────────────────────────────
     * STEP 2: PARIS backward smoothing on partial buffer
     *───────────────────────────────────────────────────────────────────────*/
    
    paris_mkl_backward_smooth(fls->paris);
    
    /*───────────────────────────────────────────────────────────────────────
     * STEP 3: Extract ALL available smoothed (ell, ell_lag) pairs
     *
     * Unlike normal operation (one update per tick), we process
     * the entire partial buffer at once.
     *
     * For T_partial entries, we get (T_partial - 1) z values:
     *   z_1 = h̃_1 - φ·h̃_0
     *   z_2 = h̃_2 - φ·h̃_1
     *   ...
     *   z_{T-1} = h̃_{T-1} - φ·h̃_{T-2}
     *───────────────────────────────────────────────────────────────────────*/
    
    for (int t = 1; t < T_partial; t++)
    {
        fls_build_particle_info_at(fls, t - 1, t);
        param_learn_update(learner, fls->particle_info, N);
        updates_performed++;
    }
    
    /*───────────────────────────────────────────────────────────────────────
     * STEP 4: Reset buffer to start fresh epoch
     *
     * The structural break invalidates the old regime. Starting fresh
     * ensures Storvik learns from post-break data only.
     *───────────────────────────────────────────────────────────────────────*/
    
    fls->head = 0;
    fls->count = 0;
    
    /* Signal Storvik that a structural break occurred */
    param_learn_signal_structural_break(learner);
    
    /* Update diagnostics */
    double elapsed = get_time_us() - t0;
    fls->emergency_flushes++;
    fls->smoothing_calls++;  /* Count the PARIS backward pass */
    fls->total_smooth_time_us += elapsed;
    fls->last_smooth_time_us = elapsed;
    
    return updates_performed;
}

/*═══════════════════════════════════════════════════════════════════════════
 * DIAGNOSTICS
 *═══════════════════════════════════════════════════════════════════════════*/

void fls_get_stats(const RBPF_FixedLagSmoother *fls,
                   uint64_t *smoothing_calls,
                   uint64_t *ticks_processed,
                   double *avg_smooth_time_us)
{
    if (!fls) return;
    
    if (smoothing_calls) *smoothing_calls = fls->smoothing_calls;
    if (ticks_processed) *ticks_processed = fls->ticks_processed;
    if (avg_smooth_time_us) {
        if (fls->smoothing_calls > 0) {
            *avg_smooth_time_us = fls->total_smooth_time_us / fls->smoothing_calls;
        } else {
            *avg_smooth_time_us = 0.0;
        }
    }
}

void fls_print_info(const RBPF_FixedLagSmoother *fls)
{
    if (!fls) return;
    
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║              Fixed-Lag Smoother (PARIS)                      ║\n");
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    printf("║ Enabled:     %s                                              ║\n",
           fls->enabled ? "YES" : "NO ");
    printf("║ Lag:         %d ticks                                        ║\n",
           fls->lag);
    printf("║ Particles:   %d (padded: %d)                                 ║\n",
           fls->n_particles, fls->n_particles_padded);
    printf("║ Regimes:     %d                                              ║\n",
           fls->n_regimes);
    printf("║ Buffer:      %d / %d entries                                 ║\n",
           fls->count, fls->buffer_size);
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    printf("║ Statistics:                                                  ║\n");
    printf("║   Ticks processed:     %12llu                        ║\n",
           (unsigned long long)fls->ticks_processed);
    printf("║   Smoothing calls:     %12llu                        ║\n",
           (unsigned long long)fls->smoothing_calls);
    printf("║   Emergency flushes:   %12llu                        ║\n",
           (unsigned long long)fls->emergency_flushes);
    
    if (fls->smoothing_calls > 0) {
        double avg = fls->total_smooth_time_us / fls->smoothing_calls;
        printf("║   Avg smooth time:     %11.2f μs                       ║\n", avg);
        printf("║   Last smooth time:    %11.2f μs                       ║\n", 
               fls->last_smooth_time_us);
    }
    printf("╚══════════════════════════════════════════════════════════════╝\n");
}
