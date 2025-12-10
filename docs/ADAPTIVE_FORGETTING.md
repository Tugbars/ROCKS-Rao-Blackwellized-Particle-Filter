# Adaptive Forgetting with Predictive Surprise

## Overview

The RBPF now supports **adaptive forgetting** — a dynamic adjustment of the forgetting factor λ based on how well the model predicts incoming observations. This implements West & Harrison's Bayesian Intervention Analysis, enabling the filter to automatically accelerate adaptation during regime changes while maintaining stability during normal operation.

## The Problem

Fixed forgetting factors face an impossible trade-off:

| λ Value | N_eff | Calm Markets | Regime Breaks |
|---------|-------|--------------|---------------|
| 0.999 | ~1000 | ✅ Stable, accurate | ❌ Slow adaptation (100+ ticks) |
| 0.990 | ~100 | ⚠️ Noisy estimates | ✅ Fast adaptation (~10 ticks) |
| 0.980 | ~50 | ❌ Unstable | ✅ Very fast |

**Solution**: Dynamically adjust λ based on **predictive surprise** — when the model is surprised by an observation, forget faster.

## Theoretical Basis

### West & Harrison's Intervention Analysis (1997)

The approach is grounded in Bayesian forecasting theory:

1. **Monitor**: Track the one-step-ahead forecast error (predictive surprise)
2. **Intervene**: When error spikes, reduce λ to "reset" the posterior
3. **Restore**: Return to baseline λ once adaptation completes

Our innovation: **continuous sigmoid intervention** instead of binary reset.

### Predictive Surprise

```
Surprise_t = -log p(y_t | y_{1:t-1})
```

- **Low surprise**: Observation matches model predictions → maintain long memory
- **High surprise**: Observation unlikely under current parameters → forget faster

This naturally distinguishes:
- **Transient outliers**: Single spike, surprise doesn't persist → Robust OCSN handles it
- **Regime breaks**: Surprise stays elevated → adaptive forgetting kicks in
- **Gradual drift**: Likelihood consistently low → gentle acceleration

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      OBSERVATION (return)                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  RBPF Kalman Update                                             │
│  → marginal_lik = p(y_t | y_{1:t-1})                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  ADAPTIVE FORGETTING                                            │
│                                                                 │
│  1. Compute surprise: S = -log(marginal_lik)                   │
│  2. Update baseline: μ_S ← EMA(S), σ_S ← EMA(|S - μ_S|)       │
│  3. Z-score: z = (S - μ_S) / σ_S                               │
│  4. Sigmoid discount: d = max_discount / (1 + exp(-k(z - c)))  │
│  5. Final λ: λ_t = λ_regime × (1 - d)                          │
│  6. Apply bounds: λ_t = max(λ_floor, λ_t)                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STORVIK UPDATE with λ_t                                        │
│  N_eff ≈ 1/(1-λ_t)                                             │
└─────────────────────────────────────────────────────────────────┘
```

## Signal Modes

| Mode | Formula | Best For |
|------|---------|----------|
| `ADAPT_SIGNAL_REGIME` | λ = λ_per_regime[r] | Baseline only, no modulation |
| `ADAPT_SIGNAL_OUTLIER_FRAC` | λ = f(outlier_ema) | When not using Robust OCSN |
| `ADAPT_SIGNAL_PREDICTIVE_SURPRISE` | λ = f(-log p(y)) | Pure surprise-based |
| **`ADAPT_SIGNAL_COMBINED`** | λ = λ_regime × (1 - sigmoid(z)) | **Recommended** |

### Why COMBINED Mode?

The combined mode captures two effects:

1. **Regime baseline**: Different volatility regimes have different persistence
   - Calm (R0): Long memory (λ = 0.999) — parameters drift slowly
   - Crisis (R3): Short memory (λ = 0.990) — rapid parameter changes

2. **Surprise modulation**: Uncertainty triggers faster forgetting
   - Normal operation: z ≈ 0 → no discount → λ = λ_regime
   - Regime break: z > 2 → discount kicks in → λ drops

## Usage

### Basic Setup

```c
RBPF_Extended *ext = rbpf_ext_create(256, 4, RBPF_PARAM_STORVIK);

// Enable the full stack
rbpf_ext_enable_robust_ocsn(ext);           // Fat-tail protection
rbpf_ext_enable_adaptive_forgetting(ext);   // Predictive surprise (COMBINED mode)

// Initialize and run
rbpf_ext_init(ext, -3.5f, 1.0f);
rbpf_ext_step(ext, return_value, &output);
```

### Custom Configuration

```c
// Per-regime baseline λ
rbpf_ext_set_regime_lambda(ext, 0, 0.9995f);  // R0: N_eff ≈ 2000
rbpf_ext_set_regime_lambda(ext, 1, 0.998f);   // R1: N_eff ≈ 500
rbpf_ext_set_regime_lambda(ext, 2, 0.995f);   // R2: N_eff ≈ 200
rbpf_ext_set_regime_lambda(ext, 3, 0.985f);   // R3: N_eff ≈ 67

// Sigmoid response curve
rbpf_ext_set_adaptive_sigmoid(ext,
    2.0f,    // center: 50% response at z = 2
    2.0f,    // steepness: moderate transition
    0.15f);  // max_discount: up to 15% reduction

// Safety bounds
rbpf_ext_set_adaptive_bounds(ext,
    0.975f,   // floor: N_eff ≥ 40
    0.9995f); // ceiling: N_eff ≤ 2000

// Smoothing
rbpf_ext_set_adaptive_smoothing(ext,
    0.02f,   // baseline_alpha: slow baseline tracking (τ ≈ 50)
    0.15f);  // signal_alpha: faster reaction (τ ≈ 7)

// Cooldown after intervention
rbpf_ext_set_adaptive_cooldown(ext, 15);  // Hold low λ for 15 ticks
```

### Diagnostics

```c
// Current state
printf("λ = %.4f\n", rbpf_ext_get_current_lambda(ext));
printf("Surprise z = %.2f\n", rbpf_ext_get_surprise_zscore(ext));

// Statistics
uint64_t interventions;
rbpf_real_t lambda, max_surprise;
rbpf_ext_get_adaptive_stats(ext, &interventions, &lambda, &max_surprise);

// Full config dump
rbpf_ext_print_adaptive_config(ext);
```

## Default Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lambda_per_regime[0]` | 0.999 | Calm: N_eff ≈ 1000 |
| `lambda_per_regime[1]` | 0.998 | Normal: N_eff ≈ 500 |
| `lambda_per_regime[2]` | 0.995 | Elevated: N_eff ≈ 200 |
| `lambda_per_regime[3]` | 0.990 | Crisis: N_eff ≈ 100 |
| `sigmoid_center` | 2.0 | 50% response at 2σ surprise |
| `sigmoid_steepness` | 2.0 | Moderate transition sharpness |
| `max_discount` | 0.12 | Maximum 12% λ reduction |
| `lambda_floor` | 0.980 | Never below (N_eff ≥ 50) |
| `lambda_ceiling` | 0.9995 | Never above (N_eff ≤ 2000) |
| `surprise_ema_alpha` | 0.02 | Baseline adaptation (τ ≈ 50) |
| `signal_ema_alpha` | 0.15 | Signal smoothing (τ ≈ 7) |
| `cooldown_ticks` | 10 | Post-intervention hold |

## Sigmoid Response Curve

```
  discount
     ↑
 0.12├─────────────────────────────────●●●●●
     │                              ●●
     │                            ●●
     │                          ●●
 0.06├─────────────────────────●─────────────  (50% at z=2)
     │                       ●●
     │                     ●●
     │                  ●●●
 0.00├●●●●●●●●●●●●●●●●───────────────────────
     └─────┴─────┴─────┴─────┴─────┴─────┴──→ surprise z-score
          -1     0     1     2     3     4
                             ↑
                          center
```

## Interaction with Other Components

### Robust OCSN

Adaptive forgetting and Robust OCSN are **complementary**:

| Component | Handles | Mechanism |
|-----------|---------|-----------|
| Robust OCSN | Transient fat-tails | Absorbs outlier into 11th component |
| Adaptive Forgetting | Regime breaks | Accelerates parameter adaptation |

When a true outlier hits:
1. Robust OCSN assigns high probability to outlier component
2. Marginal likelihood stays reasonable (outlier explained)
3. Surprise z-score doesn't spike
4. λ remains at baseline

When a regime break occurs:
1. Robust OCSN can't fully explain the shift
2. Marginal likelihood drops persistently
3. Surprise z-score rises over multiple ticks
4. λ decreases → faster adaptation

### Hawkes Process

Hawkes modifies **transition probabilities**, adaptive forgetting modifies **parameter learning rate**. They operate on orthogonal dimensions and combine naturally.

## Example: Regime Break Response

```
Time    Event              Surprise_z    λ         N_eff
────────────────────────────────────────────────────────
t=100   Normal             0.1          0.998      500
t=101   Normal             -0.2         0.998      500
t=102   BREAK STARTS       2.1          0.986      71    ← Sigmoid kicks in
t=103   High vol           2.8          0.982      56    ← Deeper discount
t=104   High vol           2.5          0.984      63
t=105   Adapting...        1.8          0.990      100   ← Cooldown holds
...
t=115   New regime stable  0.3          0.995      200   ← Returns to baseline
```

## Files

| File | Description |
|------|-------------|
| `rbpf_ksc_param_integration.h` | Struct definitions, API declarations |
| `rbpf_adaptive_forgetting.c` | Core implementation |
| `rbpf_ksc_param_integration_ext_impl.c` | Integration into `rbpf_ext_step()` |

## References

1. West, M. & Harrison, J. (1997). *Bayesian Forecasting and Dynamic Models*. Springer.
2. Bernstein, D. (1987). *Adaptive forgetting in recursive least squares*. IEEE TAC.
3. Kim, S., Shephard, N., & Chib, S. (1998). *Stochastic volatility: Likelihood inference and comparison with ARCH models*. Review of Economic Studies.
