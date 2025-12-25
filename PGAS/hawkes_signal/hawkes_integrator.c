/**
 * @file hawkes_integrator.c
 * @brief Hawkes Process Oracle Trigger - MKL Implementation
 *
 * Self-contained Hawkes process with Oracle trigger logic.
 * Uses Intel MKL for vectorized kernel computation.
 */

#include "hawkes_integrator.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

/* Intel MKL for vectorized math */
#include <mkl.h>

/*═══════════════════════════════════════════════════════════════════════════
 * INTERNAL HELPERS
 *═══════════════════════════════════════════════════════════════════════════*/

static inline float maxf(float a, float b) { return a > b ? a : b; }
static inline float minf(float a, float b) { return a < b ? a : b; }
static inline float clampf(float x, float lo, float hi)
{
    return x < lo ? lo : (x > hi ? hi : x);
}

/*═══════════════════════════════════════════════════════════════════════════
 * HAWKES INTENSITY COMPUTATION (MKL-accelerated)
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * Compute current intensity λ(t) using MKL vectorization
 *
 * λ(t) = μ + α × Σᵢ exp(-β × (t - tᵢ)) × mᵢ
 *
 * For n events, this is:
 *   1. Compute dt[i] = t - event_times[i]
 *   2. Compute arg[i] = -β × dt[i]
 *   3. Compute kernel[i] = exp(arg[i])      <- MKL vmsExp
 *   4. Compute weighted[i] = kernel[i] × marks[i]
 *   5. Sum weighted and scale by α
 */
static float compute_intensity_mkl(HawkesIntegrator *integ, float time)
{
    const HawkesParams *p = &integ->config.hawkes;
    int n = integ->event_count;

    if (n == 0)
    {
        return p->mu;
    }

    /* Use scratch buffers (already aligned) */
    float *dt = integ->scratch_dt;
    float *kernel = integ->scratch_kernel;

    /* Step 1: Compute dt = time - event_times */
    /* Using cblas for simplicity; could also use vsLinearFrac */
    for (int i = 0; i < n; i++)
    {
        int idx = (integ->event_head - n + i + HAWKES_MAX_EVENTS) % HAWKES_MAX_EVENTS;
        dt[i] = time - integ->event_times[idx];
    }

    /* Step 2: Compute -β × dt in-place */
    cblas_sscal(n, -p->beta, dt, 1);

    /* Step 3: Compute exp(-β × dt) using MKL */
    /* vmsExp is the VML function for single-precision exp */
    /* VML_HA = High Accuracy mode */
    vmsExp(n, dt, kernel, VML_HA);

    /* Step 4 & 5: Weighted sum with marks */
    float excitation = 0.0f;
    int active_events = 0;

    for (int i = 0; i < n; i++)
    {
        if (kernel[i] > HAWKES_KERNEL_CUTOFF)
        {
            int idx = (integ->event_head - n + i + HAWKES_MAX_EVENTS) % HAWKES_MAX_EVENTS;
            excitation += kernel[i] * integ->event_marks[idx];
            active_events++;
        }
    }

    /* Prune old events if all kernels are below cutoff */
    /* (Simple heuristic: if oldest event's kernel is tiny, we can reduce count) */
    while (integ->event_count > 0)
    {
        int oldest_idx = (integ->event_head - integ->event_count + HAWKES_MAX_EVENTS) % HAWKES_MAX_EVENTS;
        float oldest_dt = time - integ->event_times[oldest_idx];
        float oldest_kernel = expf(-p->beta * oldest_dt);
        if (oldest_kernel < HAWKES_KERNEL_CUTOFF)
        {
            integ->event_count--;
        }
        else
        {
            break;
        }
    }

    /* Compute final intensity */
    float lambda = p->mu + p->alpha * excitation;

    return clampf(lambda, HAWKES_MIN_INTENSITY, HAWKES_MAX_INTENSITY);
}

/**
 * Predict intensity at future time (no new events)
 */
static float predict_intensity_mkl(const HawkesIntegrator *integ, float future_time)
{
    const HawkesParams *p = &integ->config.hawkes;
    int n = integ->event_count;

    if (n == 0)
    {
        return p->mu;
    }

    /* Can't use scratch buffers (const), compute inline */
    float excitation = 0.0f;

    for (int i = 0; i < n; i++)
    {
        int idx = (integ->event_head - n + i + HAWKES_MAX_EVENTS) % HAWKES_MAX_EVENTS;
        float dt = future_time - integ->event_times[idx];
        if (dt > 0)
        {
            float kernel = expf(-p->beta * dt);
            if (kernel > HAWKES_KERNEL_CUTOFF)
            {
                excitation += kernel * integ->event_marks[idx];
            }
        }
    }

    float lambda = p->mu + p->alpha * excitation;
    return clampf(lambda, HAWKES_MIN_INTENSITY, HAWKES_MAX_INTENSITY);
}

/**
 * Add event to buffer
 */
static void add_event(HawkesIntegrator *integ, float time, float mark)
{
    integ->event_times[integ->event_head] = time;
    integ->event_marks[integ->event_head] = mark;

    integ->event_head = (integ->event_head + 1) % HAWKES_MAX_EVENTS;
    if (integ->event_count < HAWKES_MAX_EVENTS)
    {
        integ->event_count++;
    }
    integ->total_events++;
}

/*═══════════════════════════════════════════════════════════════════════════
 * DEFAULT CONFIGURATIONS
 *═══════════════════════════════════════════════════════════════════════════*/

HawkesIntegratorConfig hawkes_integrator_config_defaults(void)
{
    HawkesIntegratorConfig cfg;
    memset(&cfg, 0, sizeof(cfg));

    /* Hawkes parameters */
    cfg.hawkes.mu = 0.05f;              /* Baseline: 5% chance per tick */
    cfg.hawkes.alpha = 0.5f;            /* Excitation strength */
    cfg.hawkes.beta = 0.1f;             /* Decay: half-life ≈ 7 ticks */
    cfg.hawkes.event_threshold = 0.02f; /* 2% return triggers event */

    /* Validate branching ratio */
    float n = cfg.hawkes.alpha / cfg.hawkes.beta;
    if (n >= 1.0f)
    {
        /* Ensure subcritical */
        cfg.hawkes.alpha = cfg.hawkes.beta * 0.9f;
    }

    /* Integration window */
    cfg.window_size = 64;

    /* Variance tracking */
    cfg.ema_alpha = 0.01f;
    cfg.sigma_floor = 0.001f;
    cfg.sigma_cap = 1.0f;

    /* Cumulative residual */
    cfg.residual_decay = 0.05f;
    cfg.residual_threshold = 0.1f;

    /* Hysteresis */
    cfg.high_water_mark = 2.5f;
    cfg.low_water_mark = 1.5f;
    cfg.min_ticks_armed = 3;

    /* Absolute panic */
    cfg.absolute_panic_intensity = 2.0f;
    cfg.instant_spike_multiplier = 1.5f;
    cfg.use_absolute_panic = true;

    /* Refractory */
    cfg.refractory_ticks = 500;

    /* Warmup */
    cfg.warmup_ticks = 200;

    return cfg;
}

HawkesIntegratorConfig hawkes_integrator_config_responsive(void)
{
    HawkesIntegratorConfig cfg = hawkes_integrator_config_defaults();

    /* Faster decay for quicker response */
    cfg.hawkes.beta = 0.15f;
    cfg.hawkes.alpha = cfg.hawkes.beta * 0.8f; /* Keep n < 1 */

    /* Smaller window */
    cfg.window_size = 32;

    /* Faster EMA */
    cfg.ema_alpha = 0.02f;

    /* Lower thresholds */
    cfg.high_water_mark = 2.0f;
    cfg.low_water_mark = 1.2f;
    cfg.min_ticks_armed = 2;

    /* Shorter refractory */
    cfg.refractory_ticks = 200;

    /* Lower panic threshold */
    cfg.absolute_panic_intensity = 1.5f;
    cfg.instant_spike_multiplier = 1.5f;

    return cfg;
}

HawkesIntegratorConfig hawkes_integrator_config_conservative(void)
{
    HawkesIntegratorConfig cfg = hawkes_integrator_config_defaults();

    /* Slower decay */
    cfg.hawkes.beta = 0.05f;
    cfg.hawkes.alpha = cfg.hawkes.beta * 0.7f;

    /* Larger window */
    cfg.window_size = 128;

    /* Slower EMA */
    cfg.ema_alpha = 0.005f;

    /* Higher thresholds */
    cfg.high_water_mark = 3.0f;
    cfg.low_water_mark = 2.0f;
    cfg.min_ticks_armed = 5;

    /* Longer refractory */
    cfg.refractory_ticks = 1000;

    /* Higher panic threshold */
    cfg.absolute_panic_intensity = 3.0f;
    cfg.instant_spike_multiplier = 1.5f;

    return cfg;
}

/*═══════════════════════════════════════════════════════════════════════════
 * LIFECYCLE
 *═══════════════════════════════════════════════════════════════════════════*/

int hawkes_integrator_init(HawkesIntegrator *integ,
                           const HawkesIntegratorConfig *cfg)
{
    if (!integ)
        return -1;

    memset(integ, 0, sizeof(*integ));

    /* Set config */
    integ->config = cfg ? *cfg : hawkes_integrator_config_defaults();

    /* Validate window size */
    if (integ->config.window_size > HAWKES_INTEG_MAX_WINDOW)
    {
        integ->config.window_size = HAWKES_INTEG_MAX_WINDOW;
    }
    if (integ->config.window_size < 1)
    {
        integ->config.window_size = 1;
    }

    /* Validate branching ratio */
    float n = integ->config.hawkes.alpha / (integ->config.hawkes.beta + HAWKES_EPSILON);
    if (n >= 1.0f)
    {
        fprintf(stderr, "WARNING: Hawkes branching ratio %.2f >= 1, clamping alpha\n", n);
        integ->config.hawkes.alpha = integ->config.hawkes.beta * 0.9f;
    }

    /* Initialize variance tracking */
    integ->lambda_ema = integ->config.hawkes.mu;
    integ->lambda_var_ema = 0.01f;
    integ->lambda_sigma = sqrtf(integ->lambda_var_ema + HAWKES_EPSILON);

    /* Initialize state machine */
    integ->trigger_state = HAWKES_TRIG_IDLE;
    integ->ticks_since_trigger = integ->config.refractory_ticks;

    return 0;
}

void hawkes_integrator_reset(HawkesIntegrator *integ)
{
    if (!integ)
        return;

    HawkesIntegratorConfig cfg = integ->config;

    memset(integ, 0, sizeof(*integ));
    integ->config = cfg;

    integ->lambda_ema = integ->config.hawkes.mu;
    integ->lambda_var_ema = 0.01f;
    integ->lambda_sigma = sqrtf(integ->lambda_var_ema + HAWKES_EPSILON);
    integ->trigger_state = HAWKES_TRIG_IDLE;
    integ->ticks_since_trigger = integ->config.refractory_ticks;
}

void hawkes_integrator_free(HawkesIntegrator *integ)
{
    if (integ)
    {
        memset(integ, 0, sizeof(*integ));
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * CORE UPDATE
 *═══════════════════════════════════════════════════════════════════════════*/

HawkesIntegratorResult hawkes_integrator_update(HawkesIntegrator *integ,
                                                float time,
                                                float obs_return)
{
    HawkesIntegratorResult result;
    memset(&result, 0, sizeof(result));

    if (!integ)
        return result;

    const HawkesIntegratorConfig *cfg = &integ->config;
    const HawkesParams *hp = &cfg->hawkes;
    float abs_ret = fabsf(obs_return);

    /* ═══════════════════════════════════════════════════════════════════
     * STEP 1: Check if this tick triggers an event
     * ═══════════════════════════════════════════════════════════════════ */
    if (abs_ret > hp->event_threshold)
    {
        add_event(integ, time, abs_ret);
    }

    /* ═══════════════════════════════════════════════════════════════════
     * STEP 2: Compute intensity using MKL
     * ═══════════════════════════════════════════════════════════════════ */
    integ->current_time = time;
    float lambda = compute_intensity_mkl(integ, time);
    integ->current_intensity = lambda;
    result.instantaneous_intensity = lambda;
    result.active_events = integ->event_count;

    /* ═══════════════════════════════════════════════════════════════════
     * STEP 3: Update integration window (O(1) sliding average)
     * ═══════════════════════════════════════════════════════════════════ */
    int ws = cfg->window_size;

    if (integ->window_count < ws)
    {
        integ->intensity_window[integ->window_head] = lambda;
        integ->intensity_sum += lambda;
        integ->window_count++;
    }
    else
    {
        float old_lambda = integ->intensity_window[integ->window_head];
        integ->intensity_sum -= old_lambda;
        integ->intensity_sum += lambda;
        integ->intensity_window[integ->window_head] = lambda;
    }

    integ->window_head = (integ->window_head + 1) % ws;

    float avg_lambda = integ->intensity_sum / (float)integ->window_count;
    integ->current_avg_intensity = avg_lambda;
    result.integrated_intensity = avg_lambda;

    /* ═══════════════════════════════════════════════════════════════════
     * STEP 4: Update variance tracking (Welford-style EMA)
     * ═══════════════════════════════════════════════════════════════════ */
    float alpha = cfg->ema_alpha;
    float delta = avg_lambda - integ->lambda_ema;

    integ->lambda_ema += alpha * delta;

    float delta2 = avg_lambda - integ->lambda_ema;
    integ->lambda_var_ema = (1.0f - alpha) * integ->lambda_var_ema +
                            alpha * delta * delta2;

    float sigma = sqrtf(integ->lambda_var_ema + HAWKES_EPSILON);
    sigma = maxf(sigma, maxf(cfg->sigma_floor, HAWKES_MIN_SIGMA));
    float sigma_capped = minf(sigma, cfg->sigma_cap);
    integ->lambda_sigma = sigma_capped;

    float surprise = delta / (sigma_capped + HAWKES_EPSILON);
    integ->current_surprise_sigma = surprise;
    result.surprise_sigma = surprise;
    result.baseline_ema = integ->lambda_ema;
    result.sigma = sigma_capped;

    /* ═══════════════════════════════════════════════════════════════════
     * STEP 5: Update cumulative residual (spike vs plateau)
     * ═══════════════════════════════════════════════════════════════════ */
    float predicted = integ->last_predicted;
    if (predicted < HAWKES_EPSILON)
    {
        predicted = integ->lambda_ema;
    }

    float residual = lambda - predicted;

    integ->residual_integral *= (1.0f - cfg->residual_decay);
    integ->residual_integral += residual;

    integ->current_residual = integ->residual_integral;
    result.residual = integ->residual_integral;

    /* Predict next tick's intensity */
    integ->last_predicted = predict_intensity_mkl(integ, time + 1.0f);

    /* ═══════════════════════════════════════════════════════════════════
     * STEP 6: Update counters
     * ═══════════════════════════════════════════════════════════════════ */
    integ->total_ticks++;
    integ->ticks_since_trigger++;

    if (!integ->warmed_up && integ->total_ticks >= cfg->warmup_ticks)
    {
        integ->warmed_up = true;
    }

    bool in_refractory = (integ->ticks_since_trigger < cfg->refractory_ticks);
    result.is_valid = integ->warmed_up && !in_refractory;

    /* ═══════════════════════════════════════════════════════════════════
     * STEP 7: Check absolute panic
     * ═══════════════════════════════════════════════════════════════════ */
    bool panic_trigger = false;
    if (cfg->use_absolute_panic && result.is_valid)
    {
        float instant_threshold = cfg->absolute_panic_intensity *
                                  cfg->instant_spike_multiplier;
        if (avg_lambda > cfg->absolute_panic_intensity ||
            lambda > instant_threshold)
        {
            panic_trigger = true;
        }
    }

    /* ═══════════════════════════════════════════════════════════════════
     * STEP 8: Hysteresis state machine
     * ═══════════════════════════════════════════════════════════════════ */
    result.should_trigger = false;
    result.triggered_by_panic = false;
    integ->trigger_fired_this_tick = false;

    if (!result.is_valid)
    {
        if (in_refractory)
        {
            integ->trigger_state = HAWKES_TRIG_REFRACTORY;
        }
        result.state = integ->trigger_state;
        result.ticks_armed = integ->ticks_armed;
        return result;
    }

    switch (integ->trigger_state)
    {

    case HAWKES_TRIG_IDLE:
        if (panic_trigger)
        {
            result.should_trigger = true;
            result.triggered_by_panic = true;
            integ->trigger_state = HAWKES_TRIG_FIRED;
            integ->triggers_by_panic++;
        }
        else if (surprise > cfg->high_water_mark)
        {
            integ->trigger_state = HAWKES_TRIG_ARMED;
            integ->ticks_armed = 1;
        }
        break;

    case HAWKES_TRIG_ARMED:
        integ->ticks_armed++;

        if (panic_trigger)
        {
            result.should_trigger = true;
            result.triggered_by_panic = true;
            integ->trigger_state = HAWKES_TRIG_FIRED;
            integ->triggers_by_panic++;
            integ->sum_armed_duration += integ->ticks_armed;
        }
        else if (surprise < cfg->low_water_mark)
        {
            integ->trigger_state = HAWKES_TRIG_IDLE;
            integ->ticks_armed = 0;
            integ->disarms_by_low_water++;
        }
        else if (integ->ticks_armed >= cfg->min_ticks_armed &&
                 integ->residual_integral > cfg->residual_threshold)
        {
            result.should_trigger = true;
            integ->trigger_state = HAWKES_TRIG_FIRED;
            integ->triggers_by_hysteresis++;
            integ->sum_armed_duration += integ->ticks_armed;
        }
        break;

    case HAWKES_TRIG_FIRED:
        integ->trigger_state = HAWKES_TRIG_REFRACTORY;
        integ->ticks_since_trigger = 0;
        integ->total_triggers++;
        integ->ticks_armed = 0;
        integ->trigger_fired_this_tick = true;
        break;

    case HAWKES_TRIG_REFRACTORY:
        if (integ->ticks_since_trigger >= cfg->refractory_ticks)
        {
            integ->trigger_state = HAWKES_TRIG_IDLE;
        }
        break;
    }

    result.state = integ->trigger_state;
    result.ticks_armed = integ->ticks_armed;

    return result;
}

int hawkes_integrator_update_batch(HawkesIntegrator *integ,
                                   const float *returns,
                                   int n,
                                   HawkesIntegratorResult *out_results)
{
    if (!integ || !returns || n <= 0)
        return 0;

    int trigger_count = 0;
    float time = integ->current_time;

    for (int i = 0; i < n; i++)
    {
        time += 1.0f;
        HawkesIntegratorResult res = hawkes_integrator_update(integ, time, returns[i]);

        if (out_results)
        {
            out_results[i] = res;
        }

        if (res.should_trigger)
        {
            trigger_count++;
        }
    }

    return trigger_count;
}

/*═══════════════════════════════════════════════════════════════════════════
 * QUERIES
 *═══════════════════════════════════════════════════════════════════════════*/

float hawkes_integrator_get_surprise(const HawkesIntegrator *integ)
{
    return integ ? integ->current_surprise_sigma : 0.0f;
}

float hawkes_integrator_get_intensity(const HawkesIntegrator *integ)
{
    return integ ? integ->current_avg_intensity : 0.0f;
}

float hawkes_integrator_get_instant_intensity(const HawkesIntegrator *integ)
{
    return integ ? integ->current_intensity : 0.0f;
}

HawkesTriggerState hawkes_integrator_get_state(const HawkesIntegrator *integ)
{
    return integ ? integ->trigger_state : HAWKES_TRIG_IDLE;
}

bool hawkes_integrator_is_ready(const HawkesIntegrator *integ)
{
    if (!integ)
        return false;
    return integ->warmed_up &&
           (integ->ticks_since_trigger >= integ->config.refractory_ticks);
}

float hawkes_integrator_get_branching_ratio(const HawkesIntegrator *integ)
{
    if (!integ)
        return 0.0f;
    return integ->config.hawkes.alpha / (integ->config.hawkes.beta + HAWKES_EPSILON);
}

float hawkes_integrator_get_half_life(const HawkesIntegrator *integ)
{
    if (!integ)
        return 0.0f;
    return 0.693147f / (integ->config.hawkes.beta + HAWKES_EPSILON);
}

float hawkes_integrator_predict(const HawkesIntegrator *integ, float future_time)
{
    if (!integ)
        return 0.0f;
    return predict_intensity_mkl(integ, future_time);
}

/*═══════════════════════════════════════════════════════════════════════════
 * MANUAL CONTROL
 *═══════════════════════════════════════════════════════════════════════════*/

void hawkes_integrator_force_trigger(HawkesIntegrator *integ)
{
    if (!integ)
        return;

    integ->trigger_state = HAWKES_TRIG_REFRACTORY;
    integ->ticks_since_trigger = 0;
    integ->total_triggers++;
    integ->ticks_armed = 0;
    integ->trigger_fired_this_tick = true;
}

void hawkes_integrator_clear_refractory(HawkesIntegrator *integ)
{
    if (!integ)
        return;

    integ->ticks_since_trigger = integ->config.refractory_ticks;
    if (integ->trigger_state == HAWKES_TRIG_REFRACTORY)
    {
        integ->trigger_state = HAWKES_TRIG_IDLE;
    }
}

void hawkes_integrator_set_thresholds(HawkesIntegrator *integ,
                                      float high_water,
                                      float low_water)
{
    if (!integ)
        return;

    if (high_water <= low_water)
    {
        high_water = low_water + 0.5f;
    }

    integ->config.high_water_mark = high_water;
    integ->config.low_water_mark = low_water;
}

void hawkes_integrator_set_params(HawkesIntegrator *integ,
                                  float mu,
                                  float alpha,
                                  float beta)
{
    if (!integ)
        return;

    /* Ensure subcritical */
    if (alpha >= beta)
    {
        alpha = beta * 0.9f;
    }

    integ->config.hawkes.mu = maxf(mu, HAWKES_MIN_INTENSITY);
    integ->config.hawkes.alpha = alpha;
    integ->config.hawkes.beta = maxf(beta, HAWKES_EPSILON);
}

/*═══════════════════════════════════════════════════════════════════════════
 * DIAGNOSTICS
 *═══════════════════════════════════════════════════════════════════════════*/

static const char *state_name(HawkesTriggerState s)
{
    switch (s)
    {
    case HAWKES_TRIG_IDLE:
        return "IDLE";
    case HAWKES_TRIG_ARMED:
        return "ARMED";
    case HAWKES_TRIG_FIRED:
        return "FIRED";
    case HAWKES_TRIG_REFRACTORY:
        return "REFRACTORY";
    default:
        return "UNKNOWN";
    }
}

void hawkes_integrator_print_state(const HawkesIntegrator *integ)
{
    if (!integ)
        return;

    printf("\n");
    printf("+===========================================================+\n");
    printf("|              HAWKES INTEGRATOR STATE                      |\n");
    printf("+===========================================================+\n");
    printf("| Tick: %d  Warmed up: %s                                   \n",
           integ->total_ticks, integ->warmed_up ? "YES" : "NO");
    printf("| State: %-12s  Ticks armed: %d                          \n",
           state_name(integ->trigger_state), integ->ticks_armed);
    printf("+-----------------------------------------------------------+\n");
    printf("| Hawkes Engine:                                            |\n");
    printf("|   μ=%.3f  α=%.3f  β=%.3f  n=%.2f  t½=%.1f                 \n",
           integ->config.hawkes.mu, integ->config.hawkes.alpha,
           integ->config.hawkes.beta,
           hawkes_integrator_get_branching_ratio(integ),
           hawkes_integrator_get_half_life(integ));
    printf("|   Events: %d active, %d total                              \n",
           integ->event_count, integ->total_events);
    printf("+-----------------------------------------------------------+\n");
    printf("| Intensity:                                                |\n");
    printf("|   Instantaneous: %.4f                                     \n",
           integ->current_intensity);
    printf("|   Integrated:    %.4f (window=%d)                         \n",
           integ->current_avg_intensity, integ->config.window_size);
    printf("+-----------------------------------------------------------+\n");
    printf("| Variance Tracking:                                        |\n");
    printf("|   μ_λ (baseline): %.4f                                    \n",
           integ->lambda_ema);
    printf("|   σ_λ (capped):   %.4f                                    \n",
           integ->lambda_sigma);
    printf("|   Surprise:       %.2f σ                                  \n",
           integ->current_surprise_sigma);
    printf("+-----------------------------------------------------------+\n");
    printf("| Residual: %.4f (threshold=%.2f)                           \n",
           integ->residual_integral, integ->config.residual_threshold);
    printf("+-----------------------------------------------------------+\n");
    printf("| Refractory: %d / %d ticks                                  \n",
           integ->ticks_since_trigger, integ->config.refractory_ticks);
    printf("+-----------------------------------------------------------+\n");
    printf("| Statistics:                                               |\n");
    printf("|   Triggers: %d (hysteresis=%d, panic=%d)                  \n",
           integ->total_triggers, integ->triggers_by_hysteresis,
           integ->triggers_by_panic);
    printf("|   Low-water disarms: %d                                   \n",
           integ->disarms_by_low_water);
    printf("+===========================================================+\n");
}

void hawkes_integrator_print_config(const HawkesIntegratorConfig *cfg)
{
    if (!cfg)
        return;

    float n = cfg->hawkes.alpha / (cfg->hawkes.beta + HAWKES_EPSILON);
    float half_life = 0.693147f / (cfg->hawkes.beta + HAWKES_EPSILON);

    printf("\n");
    printf("+===========================================================+\n");
    printf("|              HAWKES INTEGRATOR CONFIG                     |\n");
    printf("+===========================================================+\n");
    printf("| Hawkes Parameters:                                        |\n");
    printf("|   μ (baseline):    %.4f                                   \n", cfg->hawkes.mu);
    printf("|   α (excitation):  %.4f                                   \n", cfg->hawkes.alpha);
    printf("|   β (decay):       %.4f                                   \n", cfg->hawkes.beta);
    printf("|   n = α/β:         %.3f %s                                \n",
           n, n < 1.0f ? "(subcritical)" : "(WARNING: supercritical!)");
    printf("|   Half-life:       %.1f ticks                             \n", half_life);
    printf("|   Event threshold: %.4f                                   \n", cfg->hawkes.event_threshold);
    printf("+-----------------------------------------------------------+\n");
    printf("| Window size:      %d ticks                                \n", cfg->window_size);
    printf("| EMA alpha:        %.4f                                    \n", cfg->ema_alpha);
    printf("| Sigma floor/cap:  %.4f / %.4f                             \n",
           cfg->sigma_floor, cfg->sigma_cap);
    printf("+-----------------------------------------------------------+\n");
    printf("| Residual decay:   %.4f  threshold: %.4f                   \n",
           cfg->residual_decay, cfg->residual_threshold);
    printf("+-----------------------------------------------------------+\n");
    printf("| Hysteresis:                                               |\n");
    printf("|   High water: %.2f σ                                      \n", cfg->high_water_mark);
    printf("|   Low water:  %.2f σ                                      \n", cfg->low_water_mark);
    printf("|   Min ticks armed: %d                                     \n", cfg->min_ticks_armed);
    printf("+-----------------------------------------------------------+\n");
    printf("| Absolute panic: %s                                        \n",
           cfg->use_absolute_panic ? "ON" : "OFF");
    printf("|   Avg threshold:     %.2f                                 \n",
           cfg->absolute_panic_intensity);
    printf("|   Instant threshold: %.2f (%.1fx)                         \n",
           cfg->absolute_panic_intensity * cfg->instant_spike_multiplier,
           cfg->instant_spike_multiplier);
    printf("+-----------------------------------------------------------+\n");
    printf("| Refractory: %d ticks  Warmup: %d ticks                    \n",
           cfg->refractory_ticks, cfg->warmup_ticks);
    printf("+===========================================================+\n");
}

void hawkes_integrator_get_stats(const HawkesIntegrator *integ,
                                 HawkesIntegratorStats *stats)
{
    if (!integ || !stats)
        return;

    memset(stats, 0, sizeof(*stats));

    stats->total_triggers = integ->total_triggers;
    stats->triggers_hysteresis = integ->triggers_by_hysteresis;
    stats->triggers_panic = integ->triggers_by_panic;
    stats->disarms_low_water = integ->disarms_by_low_water;
    stats->total_events = integ->total_events;

    if (integ->total_ticks > 0)
    {
        stats->trigger_rate = (float)integ->total_triggers * 1000.0f /
                              (float)integ->total_ticks;
        stats->event_rate = (float)integ->total_events * 1000.0f /
                            (float)integ->total_ticks;
    }

    if (integ->total_triggers > 0)
    {
        stats->avg_armed_duration = integ->sum_armed_duration /
                                    (float)integ->total_triggers;
    }
}