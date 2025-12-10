/**
 * @file rbpf_adaptive_forgetting.c
 * @brief Adaptive forgetting based on predictive surprise
 *
 * Implements West & Harrison's Bayesian Intervention Analysis:
 *   - Monitor: Track one-step-ahead forecast error (predictive surprise)
 *   - Intervene: When error spikes, reduce λ to accelerate adaptation
 *
 * The key insight: Low likelihood means the current parameter set θ is
 * unlikely to have generated this observation. We should "forget" old
 * sufficient statistics faster to adapt to the new regime.
 *
 * Reference: West & Harrison (1997) "Bayesian Forecasting and Dynamic Models"
 */

#include "rbpf_ksc_param_integration.h"
#include "rbpf_param_learn.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

/*============================================================================
 * CONSTANTS
 *============================================================================*/

/* Default regime λ values (GARCH-inspired persistence) */
static const rbpf_real_t DEFAULT_LAMBDA_REGIME[RBPF_MAX_REGIMES] = {
    RBPF_REAL(0.999), /* R0 (Calm): Very long memory, N_eff ≈ 1000 */
    RBPF_REAL(0.998), /* R1 (Normal): Long memory, N_eff ≈ 500 */
    RBPF_REAL(0.995), /* R2 (Elevated): Medium memory, N_eff ≈ 200 */
    RBPF_REAL(0.990), /* R3 (Crisis): Shorter memory, N_eff ≈ 100 */
    RBPF_REAL(0.985), /* R4+: Even shorter for extreme regimes */
    RBPF_REAL(0.985),
    RBPF_REAL(0.985),
    RBPF_REAL(0.985)};

/* Sigmoid defaults */
#define DEFAULT_SIGMOID_CENTER RBPF_REAL(2.0)    /* 2σ for half-response */
#define DEFAULT_SIGMOID_STEEPNESS RBPF_REAL(2.0) /* Moderate sharpness */
#define DEFAULT_MAX_DISCOUNT RBPF_REAL(0.12)     /* Max 12% reduction */

/* Bounds */
#define DEFAULT_LAMBDA_FLOOR RBPF_REAL(0.980)    /* N_eff ≈ 50 minimum */
#define DEFAULT_LAMBDA_CEILING RBPF_REAL(0.9995) /* N_eff ≈ 2000 maximum */

/* EMA parameters */
#define DEFAULT_BASELINE_ALPHA RBPF_REAL(0.02) /* Very slow (τ ≈ 50) */
#define DEFAULT_SIGNAL_ALPHA RBPF_REAL(0.15)   /* Faster reaction */

/* Cooldown */
#define DEFAULT_COOLDOWN_TICKS 10

/* Intervention threshold (z-score above which we count as intervention) */
#define INTERVENTION_THRESHOLD RBPF_REAL(1.5)

/*============================================================================
 * INITIALIZATION
 *============================================================================*/

/**
 * Initialize adaptive forgetting to defaults
 *
 * Called internally by rbpf_ext_create(). Sets up default values
 * but leaves disabled until explicitly enabled.
 */
void rbpf_adaptive_forgetting_init(RBPF_AdaptiveForgetting *af)
{
    if (!af)
        return;

    memset(af, 0, sizeof(*af));

    af->enabled = 0;
    af->signal_source = ADAPT_SIGNAL_COMBINED; /* Recommended default */

    /* Regime baselines */
    for (int r = 0; r < RBPF_MAX_REGIMES; r++)
    {
        af->lambda_per_regime[r] = DEFAULT_LAMBDA_REGIME[r];
    }

    /* Surprise tracking - initialize to reasonable "calm market" values */
    af->surprise_baseline = RBPF_REAL(5.0); /* Typical -log(p) in normal conditions */
    af->surprise_var = RBPF_REAL(1.0);      /* Unit variance initially */
    af->surprise_ema_alpha = DEFAULT_BASELINE_ALPHA;

    af->signal_ema = RBPF_REAL(0.0);
    af->signal_ema_alpha = DEFAULT_SIGNAL_ALPHA;

    /* Sigmoid parameters */
    af->sigmoid_center = DEFAULT_SIGMOID_CENTER;
    af->sigmoid_steepness = DEFAULT_SIGMOID_STEEPNESS;
    af->max_discount = DEFAULT_MAX_DISCOUNT;

    /* Bounds */
    af->lambda_floor = DEFAULT_LAMBDA_FLOOR;
    af->lambda_ceiling = DEFAULT_LAMBDA_CEILING;

    /* Cooldown */
    af->cooldown_ticks = DEFAULT_COOLDOWN_TICKS;
    af->cooldown_remaining = 0;

    /* Output */
    af->lambda_current = RBPF_REAL(0.998); /* Safe default */
    af->surprise_current = RBPF_REAL(0.0);
    af->surprise_zscore = RBPF_REAL(0.0);
    af->discount_applied = RBPF_REAL(0.0);

    /* Statistics */
    af->interventions = 0;
    af->max_surprise_seen = RBPF_REAL(0.0);
}

/*============================================================================
 * CORE UPDATE FUNCTION
 *============================================================================*/

/**
 * Sigmoid function for continuous intervention
 *
 * Maps z-score to discount amount [0, max_discount]
 */
static inline rbpf_real_t sigmoid_discount(
    const RBPF_AdaptiveForgetting *af,
    rbpf_real_t z)
{
    if (z < RBPF_REAL(-5.0))
        return RBPF_REAL(0.0); /* Avoid exp overflow */
    if (z > RBPF_REAL(10.0))
        return af->max_discount; /* Saturate */

    rbpf_real_t x = -af->sigmoid_steepness * (z - af->sigmoid_center);
    return af->max_discount / (RBPF_REAL(1.0) + rbpf_exp(x));
}

/**
 * Main update function - call AFTER rbpf_ksc_update()
 *
 * Uses the marginal likelihood from the update step to compute surprise
 * and adjust the forgetting factor.
 *
 * @param ext              Extended RBPF handle
 * @param marginal_lik     Marginal likelihood from rbpf_ksc_update()
 * @param dominant_regime  Current dominant regime (for baseline λ)
 */
void rbpf_adaptive_forgetting_update(
    RBPF_Extended *ext,
    rbpf_real_t marginal_lik,
    int dominant_regime)
{
    if (!ext)
        return;

    RBPF_AdaptiveForgetting *af = &ext->adaptive_forgetting;

    if (!af->enabled)
    {
        /* When disabled, use fixed λ from Storvik config */
        return;
    }

    /*═══════════════════════════════════════════════════════════════════════
     * STEP 1: Compute Effective Surprise
     *
     * Two signal sources that BOTH indicate model misspecification:
     *
     * A) Predictive Surprise = -log p(y_t | y_{1:t-1})
     *    High when: Data doesn't fit the model at all
     *
     * B) Structural Surprise = f(outlier_fraction)
     *    High when: Data only fits because we're calling it an outlier
     *
     * The "Signal Starvation" Problem:
     *   When Robust OCSN absorbs a shock, marginal likelihood stays decent
     *   (because the outlier component explains it), so predictive surprise
     *   stays low. But the model is still "cheating" by using the crutch.
     *
     * Solution: Take MAX of both signals. We adapt if EITHER:
     *   - Data fits poorly (drift) → predictive surprise triggers
     *   - Data fits via outlier (shock) → structural surprise triggers
     *═══════════════════════════════════════════════════════════════════════*/

    rbpf_real_t pred_surprise = RBPF_REAL(0.0);
    rbpf_real_t struct_surprise = RBPF_REAL(0.0);
    rbpf_real_t surprise;

    /* A) Predictive Surprise from marginal likelihood */
    if (af->signal_source != ADAPT_SIGNAL_OUTLIER_FRAC)
    {
        if (marginal_lik < RBPF_REAL(1e-30))
        {
            marginal_lik = RBPF_REAL(1e-30); /* Avoid log(0) */
        }
        pred_surprise = -rbpf_log(marginal_lik);
    }

    /* B) Structural Surprise from outlier usage */
    /* Scale: 0% outlier → 0, 50% outlier → 2.5, 100% outlier → 5.0 */
    /* This is calibrated so heavy outlier usage (~50%+) triggers intervention */
    struct_surprise = ext->last_outlier_fraction * RBPF_REAL(5.0);

    /* Combine signals based on mode */
    switch (af->signal_source)
    {
    case ADAPT_SIGNAL_REGIME:
        /* Pure regime-based, no surprise adaptation */
        surprise = RBPF_REAL(0.0);
        break;

    case ADAPT_SIGNAL_OUTLIER_FRAC:
        /* Only structural surprise (for testing) */
        surprise = struct_surprise * RBPF_REAL(4.0); /* Scale up for standalone use */
        break;

    case ADAPT_SIGNAL_PREDICTIVE_SURPRISE:
        /* Only predictive surprise (original behavior, for comparison) */
        surprise = pred_surprise;
        break;

    case ADAPT_SIGNAL_COMBINED:
    default:
        /* ADDITIVE: Predictive + Outlier penalty
         *
         * Key insight: When OCSN absorbs a shock, pred_surprise stays low
         * (because the model "explains" it via the outlier component).
         * But we should PENALIZE relying on the crutch!
         *
         * Formula: surprise = pred_surprise + α × outlier_fraction²
         *
         * The quadratic term:
         *   - 10% outlier → +0.1 (negligible, normal market noise)
         *   - 30% outlier → +0.9 (moderate penalty, elevated vol)
         *   - 50% outlier → +2.5 (strong penalty, regime break likely)
         *   - 80% outlier → +6.4 (massive penalty, clear structural change)
         *
         * This ensures heavy outlier usage triggers adaptation even when
         * the marginal likelihood looks "fine" because OCSN fit well.
         */
        {
            rbpf_real_t outlier_penalty = ext->last_outlier_fraction *
                                          ext->last_outlier_fraction *
                                          RBPF_REAL(10.0);
            surprise = pred_surprise + outlier_penalty;
        }
        break;
    }

    af->surprise_current = surprise;

    /* Track maximum for diagnostics */
    if (surprise > af->max_surprise_seen)
    {
        af->max_surprise_seen = surprise;
    }

    /*═══════════════════════════════════════════════════════════════════════
     * STEP 2: Update Baseline Statistics
     *
     * We track what "normal" surprise looks like so we can detect anomalies.
     * Use slow EMA to avoid chasing regime changes.
     *═══════════════════════════════════════════════════════════════════════*/

    rbpf_real_t alpha_base = af->surprise_ema_alpha;

    /* Only update baseline when NOT in intervention mode */
    if (af->cooldown_remaining == 0 && af->surprise_zscore < RBPF_REAL(1.0))
    {
        /* Update baseline mean */
        af->surprise_baseline = alpha_base * surprise +
                                (RBPF_REAL(1.0) - alpha_base) * af->surprise_baseline;

        /* Update baseline variance (for z-score calculation) */
        rbpf_real_t delta = surprise - af->surprise_baseline;
        af->surprise_var = alpha_base * (delta * delta) +
                           (RBPF_REAL(1.0) - alpha_base) * af->surprise_var;

        /* Floor variance to prevent division issues */
        if (af->surprise_var < RBPF_REAL(0.01))
        {
            af->surprise_var = RBPF_REAL(0.01);
        }
    }

    /*═══════════════════════════════════════════════════════════════════════
     * STEP 3: Compute Surprise Z-Score
     *
     * z = (surprise - baseline) / sqrt(variance)
     *
     * High z-score indicates observation is unusual given recent history.
     *═══════════════════════════════════════════════════════════════════════*/

    rbpf_real_t surprise_std = rbpf_sqrt(af->surprise_var);
    af->surprise_zscore = (surprise - af->surprise_baseline) / (surprise_std + RBPF_REAL(1e-6));

    /* Smooth the signal for stability */
    af->signal_ema = af->signal_ema_alpha * af->surprise_zscore +
                     (RBPF_REAL(1.0) - af->signal_ema_alpha) * af->signal_ema;

    /*═══════════════════════════════════════════════════════════════════════
     * STEP 4: Compute Adaptive λ Based on Signal Source
     *═══════════════════════════════════════════════════════════════════════*/

    rbpf_real_t lambda;

    switch (af->signal_source)
    {
    case ADAPT_SIGNAL_REGIME:
        /* Pure regime baseline (no surprise modulation) */
        lambda = af->lambda_per_regime[dominant_regime];
        af->discount_applied = RBPF_REAL(0.0);
        break;

    case ADAPT_SIGNAL_OUTLIER_FRAC:
    case ADAPT_SIGNAL_PREDICTIVE_SURPRISE:
        /* Pure surprise-based (no regime baseline) */
        {
            rbpf_real_t base_lambda = RBPF_REAL(0.998); /* Fixed base */
            rbpf_real_t discount = sigmoid_discount(af, af->signal_ema);
            lambda = base_lambda * (RBPF_REAL(1.0) - discount);
            af->discount_applied = discount;
        }
        break;

    case ADAPT_SIGNAL_COMBINED:
    default:
        /* RECOMMENDED: Regime baseline × surprise modifier */
        {
            rbpf_real_t base_lambda = af->lambda_per_regime[dominant_regime];
            rbpf_real_t discount = sigmoid_discount(af, af->signal_ema);
            lambda = base_lambda * (RBPF_REAL(1.0) - discount);
            af->discount_applied = discount;
        }
        break;
    }

    /*═══════════════════════════════════════════════════════════════════════
     * STEP 5: Apply Cooldown
     *
     * After intervention, hold low λ for a few ticks to allow adaptation.
     * This prevents oscillation between high/low λ.
     *═══════════════════════════════════════════════════════════════════════*/

    if (af->cooldown_remaining > 0)
    {
        /* Keep previous (low) λ during cooldown */
        lambda = af->lambda_current;
        af->cooldown_remaining--;
    }
    else if (af->surprise_zscore > INTERVENTION_THRESHOLD)
    {
        /* Start cooldown on significant intervention */
        af->cooldown_remaining = af->cooldown_ticks;
        af->interventions++;
    }

    /*═══════════════════════════════════════════════════════════════════════
     * STEP 6: Apply Bounds and Store Result
     *═══════════════════════════════════════════════════════════════════════*/

    if (lambda < af->lambda_floor)
        lambda = af->lambda_floor;
    if (lambda > af->lambda_ceiling)
        lambda = af->lambda_ceiling;

    af->lambda_current = lambda;

    /*═══════════════════════════════════════════════════════════════════════
     * STEP 7: Push to Storvik Parameter Learner
     *
     * The whole point: update the forgetting factor used by sufficient stats.
     *═══════════════════════════════════════════════════════════════════════*/

    if (ext->storvik_initialized)
    {
        param_learn_set_forgetting(&ext->storvik, 1, lambda);
    }
}

/*============================================================================
 * API FUNCTIONS
 *============================================================================*/

void rbpf_ext_enable_adaptive_forgetting(RBPF_Extended *ext)
{
    if (!ext)
        return;

    RBPF_AdaptiveForgetting *af = &ext->adaptive_forgetting;

    /* Initialize to defaults if not already */
    rbpf_adaptive_forgetting_init(af);

    af->enabled = 1;
    af->signal_source = ADAPT_SIGNAL_COMBINED;

    /* Also enable Storvik forgetting as the underlying mechanism */
    if (ext->storvik_initialized)
    {
        param_learn_set_forgetting(&ext->storvik, 1, af->lambda_per_regime[1]); /* Start at R1 */
    }
}

void rbpf_ext_enable_adaptive_forgetting_mode(RBPF_Extended *ext, RBPF_AdaptSignal signal)
{
    if (!ext)
        return;

    rbpf_ext_enable_adaptive_forgetting(ext);
    ext->adaptive_forgetting.signal_source = signal;
}

void rbpf_ext_disable_adaptive_forgetting(RBPF_Extended *ext)
{
    if (!ext)
        return;
    ext->adaptive_forgetting.enabled = 0;
}

void rbpf_ext_set_regime_lambda(RBPF_Extended *ext, int regime, rbpf_real_t lambda)
{
    if (!ext || regime < 0 || regime >= RBPF_MAX_REGIMES)
        return;

    /* Clamp to reasonable range */
    if (lambda < RBPF_REAL(0.9))
        lambda = RBPF_REAL(0.9);
    if (lambda > RBPF_REAL(0.9999))
        lambda = RBPF_REAL(0.9999);

    ext->adaptive_forgetting.lambda_per_regime[regime] = lambda;
}

void rbpf_ext_set_adaptive_sigmoid(RBPF_Extended *ext,
                                   rbpf_real_t center,
                                   rbpf_real_t steepness,
                                   rbpf_real_t max_discount)
{
    if (!ext)
        return;

    RBPF_AdaptiveForgetting *af = &ext->adaptive_forgetting;

    af->sigmoid_center = center;
    af->sigmoid_steepness = steepness;

    /* Clamp max_discount to prevent excessive forgetting */
    if (max_discount > RBPF_REAL(0.5))
        max_discount = RBPF_REAL(0.5);
    if (max_discount < RBPF_REAL(0.01))
        max_discount = RBPF_REAL(0.01);
    af->max_discount = max_discount;
}

void rbpf_ext_set_adaptive_bounds(RBPF_Extended *ext,
                                  rbpf_real_t floor,
                                  rbpf_real_t ceiling)
{
    if (!ext)
        return;

    /* Ensure floor < ceiling and both in valid range */
    if (floor < RBPF_REAL(0.8))
        floor = RBPF_REAL(0.8);
    if (ceiling > RBPF_REAL(0.9999))
        ceiling = RBPF_REAL(0.9999);
    if (floor >= ceiling)
        floor = ceiling - RBPF_REAL(0.01);

    ext->adaptive_forgetting.lambda_floor = floor;
    ext->adaptive_forgetting.lambda_ceiling = ceiling;
}

void rbpf_ext_set_adaptive_smoothing(RBPF_Extended *ext,
                                     rbpf_real_t baseline_alpha,
                                     rbpf_real_t signal_alpha)
{
    if (!ext)
        return;

    /* Clamp to reasonable range (0.001, 0.5) */
    if (baseline_alpha < RBPF_REAL(0.001))
        baseline_alpha = RBPF_REAL(0.001);
    if (baseline_alpha > RBPF_REAL(0.5))
        baseline_alpha = RBPF_REAL(0.5);
    if (signal_alpha < RBPF_REAL(0.001))
        signal_alpha = RBPF_REAL(0.001);
    if (signal_alpha > RBPF_REAL(0.5))
        signal_alpha = RBPF_REAL(0.5);

    ext->adaptive_forgetting.surprise_ema_alpha = baseline_alpha;
    ext->adaptive_forgetting.signal_ema_alpha = signal_alpha;
}

void rbpf_ext_set_adaptive_cooldown(RBPF_Extended *ext, int ticks)
{
    if (!ext)
        return;

    if (ticks < 0)
        ticks = 0;
    if (ticks > 100)
        ticks = 100;

    ext->adaptive_forgetting.cooldown_ticks = ticks;
}

rbpf_real_t rbpf_ext_get_current_lambda(const RBPF_Extended *ext)
{
    if (!ext)
        return RBPF_REAL(0.998);
    return ext->adaptive_forgetting.lambda_current;
}

rbpf_real_t rbpf_ext_get_surprise_zscore(const RBPF_Extended *ext)
{
    if (!ext)
        return RBPF_REAL(0.0);
    return ext->adaptive_forgetting.surprise_zscore;
}

void rbpf_ext_get_adaptive_stats(const RBPF_Extended *ext,
                                 uint64_t *interventions,
                                 rbpf_real_t *current_lambda,
                                 rbpf_real_t *max_surprise)
{
    if (!ext)
        return;

    const RBPF_AdaptiveForgetting *af = &ext->adaptive_forgetting;

    if (interventions)
        *interventions = af->interventions;
    if (current_lambda)
        *current_lambda = af->lambda_current;
    if (max_surprise)
        *max_surprise = af->max_surprise_seen;
}

/*============================================================================
 * DIAGNOSTICS
 *============================================================================*/

void rbpf_ext_print_adaptive_config(const RBPF_Extended *ext)
{
    if (!ext)
        return;

    const RBPF_AdaptiveForgetting *af = &ext->adaptive_forgetting;

    static const char *signal_names[] = {
        "REGIME", "OUTLIER_FRAC", "PREDICTIVE_SURPRISE", "COMBINED"};

    printf("\n┌─────────────────────────────────────────────────────────────┐\n");
    printf("│             Adaptive Forgetting Configuration               │\n");
    printf("├─────────────────────────────────────────────────────────────┤\n");

    if (!af->enabled)
    {
        printf("│  Status: DISABLED                                           │\n");
        printf("└─────────────────────────────────────────────────────────────┘\n");
        return;
    }

    printf("│  Status:        ENABLED                                     │\n");
    printf("│  Signal Source: %-20s                      │\n", signal_names[af->signal_source]);
    printf("├─────────────────────────────────────────────────────────────┤\n");
    printf("│  Regime Baselines (λ → N_eff):                              │\n");
    for (int r = 0; r < 4; r++)
    {
        rbpf_real_t lam = af->lambda_per_regime[r];
        int n_eff = (int)(RBPF_REAL(1.0) / (RBPF_REAL(1.0) - lam));
        printf("│    R%d: λ=%.4f → N_eff≈%4d                                │\n",
               r, (float)lam, n_eff);
    }
    printf("├─────────────────────────────────────────────────────────────┤\n");
    printf("│  Sigmoid Response:                                          │\n");
    printf("│    Center (z):     %.2f                                     │\n", (float)af->sigmoid_center);
    printf("│    Steepness:      %.2f                                     │\n", (float)af->sigmoid_steepness);
    printf("│    Max Discount:   %.1f%%                                    │\n", (float)af->max_discount * 100);
    printf("├─────────────────────────────────────────────────────────────┤\n");
    printf("│  Bounds:                                                    │\n");
    printf("│    Floor:   λ=%.3f (N_eff≈%d)                             │\n",
           (float)af->lambda_floor, (int)(1.0f / (1.0f - af->lambda_floor)));
    printf("│    Ceiling: λ=%.4f (N_eff≈%d)                            │\n",
           (float)af->lambda_ceiling, (int)(1.0f / (1.0f - af->lambda_ceiling)));
    printf("├─────────────────────────────────────────────────────────────┤\n");
    printf("│  Smoothing:                                                 │\n");
    printf("│    Baseline α: %.3f (τ≈%.0f ticks)                         │\n",
           (float)af->surprise_ema_alpha, 1.0f / (float)af->surprise_ema_alpha);
    printf("│    Signal α:   %.3f (τ≈%.0f ticks)                         │\n",
           (float)af->signal_ema_alpha, 1.0f / (float)af->signal_ema_alpha);
    printf("│    Cooldown:   %d ticks                                     │\n", af->cooldown_ticks);
    printf("├─────────────────────────────────────────────────────────────┤\n");
    printf("│  Current State:                                             │\n");
    printf("│    λ_current:      %.4f                                    │\n", (float)af->lambda_current);
    printf("│    Surprise z:     %+.2f                                    │\n", (float)af->surprise_zscore);
    printf("│    Discount:       %.1f%%                                    │\n", (float)af->discount_applied * 100);
    printf("│    Cooldown left:  %d                                       │\n", af->cooldown_remaining);
    printf("├─────────────────────────────────────────────────────────────┤\n");
    printf("│  Statistics:                                                │\n");
    printf("│    Interventions:  %llu                                     │\n", (unsigned long long)af->interventions);
    printf("│    Max Surprise:   %.2f                                    │\n", (float)af->max_surprise_seen);
    printf("│    Baseline μ:     %.2f                                    │\n", (float)af->surprise_baseline);
    printf("│    Baseline σ:     %.2f                                    │\n", (float)rbpf_sqrt(af->surprise_var));
    printf("└─────────────────────────────────────────────────────────────┘\n");
}