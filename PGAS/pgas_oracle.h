/**
 * @file pgas_oracle.h
 * @brief PGAS Oracle - Background Parameter Server for RBPF
 *
 * Architecture from Integration Schema:
 *
 *   PGAS = Parameter Server. Provides the physics (Pi).
 *   RBPF = Game Client. Plays the game with given physics.
 *   Hawkes = Circuit Breaker. Disconnects during crisis.
 */

#ifndef PGAS_ORACLE_H
#define PGAS_ORACLE_H

#include <stdint.h>
#include <stdbool.h>
#include "pgas_compat.h"

/*===========================================================================
 * PLATFORM-SPECIFIC THREADING
 *===========================================================================*/

#ifdef _MSC_VER
/* Windows threading */
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
typedef HANDLE pgas_thread_t;
typedef CRITICAL_SECTION pgas_mutex_t;
typedef CONDITION_VARIABLE pgas_cond_t;

#define PGAS_ALIGN(x) __declspec(align(x))
#else
/* POSIX threading */
#include <pthread.h>
typedef pthread_t pgas_thread_t;
typedef pthread_mutex_t pgas_mutex_t;
typedef pthread_cond_t pgas_cond_t;

#define PGAS_ALIGN(x) __attribute__((aligned(x)))
#endif

#ifdef __cplusplus
extern "C"
{
#endif

    /*===========================================================================
     * FORWARD DECLARATIONS
     *===========================================================================*/

    typedef struct PGASSlidingState PGASSlidingState;

    /*===========================================================================
     * CONFIGURATION
     *===========================================================================*/

#define PGAS_ORACLE_MAX_K 8
#define PGAS_ORACLE_ALIGN 64

    /*===========================================================================
     * DOUBLE BUFFER CHANNEL (PGAS -> RBPF)
     *===========================================================================*/

    typedef struct
    {
        float Pi[PGAS_ORACLE_MAX_K * PGAS_ORACLE_MAX_K];
        int64_t tick;          /* Window end tick */
        float acceptance_rate; /* Mixing diagnostic */
        int sweeps_used;
        int windows_completed; /* How many windows PGAS has processed */
        float avg_acceptance;  /* Running average acceptance rate */
    } PGASChannelBuffer;

    typedef struct
    {
        pgas_atomic_int32 write_idx; /* Which buffer PGAS is writing to */
        pgas_atomic_int32 ready;     /* New data available for consumption */

#ifdef _MSC_VER
        __declspec(align(64)) PGASChannelBuffer buf[2];
#else
    PGASChannelBuffer buf[2] __attribute__((aligned(64)));
#endif

        /* Metadata */
        int K;                      /* Regime count */
        int64_t last_consumed_tick; /* Last tick consumed by RBPF */

    } PGASChannel;

    /*===========================================================================
     * PGAS ORACLE STATE
     *===========================================================================*/

    typedef struct PGASOracleState
    {
        /* OWNED COMPONENTS */
        PGASSlidingState *sliding; /* Sliding window PGAS */
        PGASChannel channel;       /* Double-buffer output */

        /* THREAD CONTROL */
        pgas_thread_t thread;
        pgas_atomic_int32 running; /* 0 = stop, 1 = run */
        pgas_atomic_int32 started; /* Thread has started */

        /* CONFIGURATION */
        int n_sweeps;   /* Gibbs sweeps per iteration */
        int core_start; /* First core for affinity (e.g., 2) */
        int n_cores;    /* Number of cores to use (e.g., 30) */

        /* DIAGNOSTICS */
        pgas_atomic_int64 iterations; /* Windows processed */
        pgas_atomic_int64 observations_pushed;
        float avg_acceptance_rate;
        float avg_iteration_time_ms;

    } PGASOracleState;

    /*===========================================================================
     * LIFECYCLE
     *===========================================================================*/

    PGASOracleState *pgas_oracle_alloc(int window_size, int slide_step,
                                       int N, int K, int n_sweeps,
                                       uint32_t seed);

    void pgas_oracle_free(PGASOracleState *oracle);

    /*===========================================================================
     * MODEL CONFIGURATION
     *===========================================================================*/

    void pgas_oracle_set_model(PGASOracleState *oracle,
                               const double *trans,
                               const double *mu_vol,
                               const double *sigma_vol,
                               double phi);

    void pgas_oracle_set_prior(PGASOracleState *oracle,
                               float alpha, float kappa);

    void pgas_oracle_set_recency(PGASOracleState *oracle, float lambda);

    void pgas_oracle_set_affinity(PGASOracleState *oracle,
                                  int core_start, int n_cores);

    /*===========================================================================
     * THREAD CONTROL
     *===========================================================================*/

    int pgas_oracle_start(PGASOracleState *oracle);
    void pgas_oracle_stop(PGASOracleState *oracle);
    bool pgas_oracle_is_running(const PGASOracleState *oracle);

    /*===========================================================================
     * OBSERVATION INPUT (Tick Loop)
     *===========================================================================*/

    bool pgas_oracle_push(PGASOracleState *oracle, float obs, int64_t tick);

    /*===========================================================================
     * CHANNEL ACCESS (Tick Loop)
     *===========================================================================*/

    bool pgas_oracle_ready(const PGASOracleState *oracle);
    bool pgas_oracle_consume(PGASOracleState *oracle, float *pi_out, int64_t *tick_out);
    void pgas_oracle_peek(const PGASOracleState *oracle,
                          float *acceptance_rate_out,
                          int *sweeps_out);

    /*===========================================================================
     * HOT SWAP INTEGRATION (Tick Loop)
     *===========================================================================*/

    bool pgas_oracle_try_hot_swap(PGASOracleState *oracle,
                                  bool crisis,
                                  float *pi_out,
                                  int64_t *tick_out);

    /*===========================================================================
     * VALIDATION & DIAGNOSTICS
     *===========================================================================*/

    bool pgas_oracle_validate_pi(const float *pi, int K);
    int64_t pgas_oracle_get_iterations(const PGASOracleState *oracle);
    int64_t pgas_oracle_get_observations(const PGASOracleState *oracle);
    void pgas_oracle_print_diagnostics(const PGASOracleState *oracle);

    /*===========================================================================
     * BOUNDED-MEMORY BAYESIAN ESTIMATION
     *===========================================================================*/

    /**
     * @brief Set memory decay for accumulated counts
     *
     * Controls how inactive rows erode over time.
     *
     * @param oracle  Oracle state
     * @param decay   Per-window decay factor (0.99 default)
     * @param floor   Minimum count floor (0.5 default)
     */
    void pgas_oracle_set_memory_decay(PGASOracleState *oracle, float decay, float floor);

    /**
     * @brief Set maximum inertia for accumulated counts (Inertia Clamping)
     *
     * Controls how "heavy" history can get before new data is ignored:
     *   - 200 = Very Agile (20 ticks = 10% impact)
     *   - 300 = Agile (20 ticks = 6.7% impact) [RECOMMENDED]
     *   - 500 = Balanced (20 ticks = 4% impact)
     *
     * @param oracle       Oracle state
     * @param max_inertia  Maximum row sum for accumulated counts
     */
    void pgas_oracle_set_max_inertia(PGASOracleState *oracle, float max_inertia);

    /**
     * @brief Enable/disable adaptive kappa (anti-chattering)
     *
     * When enabled, sticky_kappa is dynamically adjusted based on observed
     * chatter rate using RLS estimation. Reduces regime flipping noise.
     *
     * @param oracle   Oracle state
     * @param enabled  1 = enabled, 0 = disabled (default)
     */
    void pgas_oracle_set_adaptive_kappa(PGASOracleState *oracle, int enabled);

    /*===========================================================================
     * ADAPTIVE SLIDE CONTROL
     *===========================================================================*/

    /**
     * @brief Enable/disable adaptive slide based on SR
     *
     * When enabled, slide interval adapts based on SR statistic:
     *   - SR low  → slide = slide_normal (save compute)
     *   - SR high → slide = slide_fast (faster Π updates)
     *
     * @param oracle   Oracle state
     * @param enabled  1 = enabled (default), 0 = use fixed slide_step
     */
    void pgas_oracle_set_adaptive_slide(PGASOracleState *oracle, int enabled);

    /**
     * @brief Set SR thresholds for adaptive slide modes
     *
     * @param oracle       Oracle state
     * @param sr_elevated  SR threshold for elevated mode (default: 1.0)
     * @param sr_fast      SR threshold for fast mode (default: 3.0)
     */
    void pgas_oracle_set_sr_thresholds(PGASOracleState *oracle,
                                       float sr_elevated, float sr_fast);

    /**
     * @brief Update adaptive slide based on current SR statistic
     *
     * Call this from the main loop whenever you have a new SR value.
     * Typically from Hawkes integrator: hawkes_integrator_get_cumulative_residual()
     *
     * @param oracle   Oracle state
     * @param sr_stat  Current SR statistic
     */
    void pgas_oracle_update_sr(PGASOracleState *oracle, float sr_stat);

    /**
     * @brief Get current effective slide interval
     *
     * @return Current slide being used (may differ from configured slide_step
     *         if adaptive slide is enabled)
     */
    int pgas_oracle_get_effective_slide(const PGASOracleState *oracle);

#ifdef __cplusplus
}
#endif

#endif /* PGAS_ORACLE_H */