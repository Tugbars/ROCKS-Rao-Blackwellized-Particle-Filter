# RBPF Tail Handling Plan: Hawkes + Robust OCSN

## Executive Summary

Two complementary extensions to handle fat-tailed market returns in the RBPF regime-switching filter:

| Component | Problem Solved | Mechanism |
|-----------|---------------|-----------|
| **Hawkes Self-Excitation** | Volatility clustering (shocks breed shocks) | Modifies regime transition probabilities |
| **Robust OCSN (11th Component)** | Particle collapse during extreme moves | Adds outlier likelihood term |

These are **independent and complementary**—enable one, both, or neither.

---

## Problem Statement

### 1. Standard OCSN Fails on Extreme Returns

The Kim, Shephard & Chib (1998) OCSN approximates log(χ²(1)) with 10 Gaussians. This works well for typical returns but fails catastrophically during market crashes:

```
Standard OCSN variance range: 0.11 to 7.33
8σ move in log-space: y ≈ -23 + 16 = -7 (assuming h_t ≈ -4)

For y = -7 with h_t = -4:
  Innovation = y - 2*h_t = -7 - (-8) = 1
  But if crash: y = 2 (from 10% daily move)
  Innovation = 2 - (-8) = 10 → likelihood ≈ 0
```

**Result**: All particles get near-zero weight → effective sample size collapses → filter diverges.

### 2. Regime Transitions Don't Capture Clustering

The standard Markov transition matrix assumes:
- P(regime_t | regime_{t-1}) is constant
- No memory of recent shocks

**Reality**: After a large move, probability of another large move increases (volatility clustering). The filter should "expect" elevated volatility to persist.

---

## Solution 1: Robust OCSN (The 11th Component)

### Concept

Add a wide Gaussian "safety valve" to the OCSN mixture:

```
P(y_t | h_t) = (1 - π_outlier) × P_OCSN(y_t | h_t) + π_outlier × P_broad(y_t | h_t)
```

Where:
- `P_OCSN` = standard 10-component mixture
- `P_broad` = N(2·h_t, σ²_outlier) with large variance
- `π_outlier` = small probability (1-3%)

### Why It Works

During a crash (e.g., -15% daily return):

| Component | Likelihood | Contribution |
|-----------|------------|--------------|
| OCSN (10 components) | ≈ 0 | Negligible |
| Outlier (11th) | Small but non-zero | Dominates |

The outlier component ensures particles survive with non-zero weight.

### Kalman Update Behavior

The outlier component has high variance → low Kalman gain:

```
K_outlier = H × P_pred / (H² × P_pred + σ²_outlier)
         ≈ 2 × 0.1 / (4 × 0.1 + 25)
         ≈ 0.008  (vs typical K ≈ 0.1-0.3)
```

**Effect**: When observation is explained by outlier component, the state update is gentle. This prevents a single crash from spiking h_t to unrealistic levels.

### Per-Regime Parameters

Different regimes have different tail expectations:

| Regime | Outlier Prob | Outlier Variance | Rationale |
|--------|--------------|------------------|-----------|
| R0 (Calm) | 1.0% | 15 | Rare outliers, moderate tolerance |
| R1 (Mild) | 1.5% | 18 | Slightly more common |
| R2 (Elevated) | 2.0% | 22 | Stress period |
| R3 (Crisis) | 3.0% | 30 | Expect fat tails |

---

## Solution 2: Hawkes Self-Exciting Process

### Concept

Model volatility clustering as a self-exciting point process:

```
λ(t) = μ + Σᵢ α × exp(-β × (t - tᵢ))
```

Where:
- `λ(t)` = intensity (expected shock rate)
- `μ` = baseline intensity
- `α` = jump size when shock occurs
- `β` = decay rate
- `tᵢ` = times of past shocks (|return| > threshold)

### Integration with RBPF

High Hawkes intensity → boost upward regime transitions:

```
Before tick t:
  1. Check λ(t) from previous tick
  2. If λ(t) > μ × 1.1:
     - Compute boost = (λ(t) - μ) × boost_scale
     - Modify transition matrix: increase P(R_i → R_j) for j > i
     - Rebuild LUT
  3. Run regime transition with modified LUT
  
After tick t:
  4. Update λ(t+1) based on current observation
  5. Restore base transition LUT
```

### Parameter Interpretation

| Parameter | Typical Value | Interpretation |
|-----------|---------------|----------------|
| μ | 0.05 | ~5% baseline chance of being in "excited" state |
| α | 0.30 | Each shock adds 0.3 to intensity |
| β | 0.10 | Half-life ≈ 7 ticks (0.693/β) |
| threshold | 3% | Returns > 3% trigger excitation |

### Transition Modification Logic

```c
/* Compute boost from excess intensity */
float excess = intensity - mu;
float boost = min(excess * boost_scale, boost_cap);

/* For each row (source regime) */
for (from = 0; from < n_regimes - 1; from++) {
    /* Steal from self and lower transitions */
    to_redistribute = row[from] * boost;
    row[from] -= to_redistribute;
    
    /* Give to higher regimes (weighted toward crisis) */
    for (to = from + 1; to < n_regimes; to++) {
        row[to] += to_redistribute * weight[to];
    }
}
```

---

## Combined Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        rbpf_ext_step()                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ 1. HAWKES: Apply intensity to transition LUT             │   │
│  │    (uses λ from previous tick)                           │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              ↓                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ 2. REGIME TRANSITION: Sample new regime                  │   │
│  │    P(R_t | R_{t-1}, λ_{t-1})                             │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              ↓                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ 3. KALMAN PREDICT: h_t|t-1 = φ·h_{t-1} + μ_vol          │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              ↓                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ 4. ROBUST OCSN UPDATE: 10 + 1 component mixture         │   │
│  │    P(y_t | h_t) = (1-π)·OCSN + π·Outlier                │   │
│  │    → Particle weights updated                            │   │
│  │    → State h_t updated (gentle if outlier dominates)     │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              ↓                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ 5. RESAMPLE: If ESS < threshold                          │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              ↓                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ 6. HAWKES: Update intensity for next tick                │   │
│  │    λ_t = μ + (λ_{t-1} - μ)·e^{-β} + α·I(|r_t|>thresh)   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              ↓                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ 7. STORVIK/LIU-WEST: Parameter learning                  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component Roles (Clarification)

| Component | What It Modifies | When It Acts |
|-----------|------------------|--------------|
| **Hawkes** | P(regime_t \| regime_{t-1}, history) | Before transition step |
| **Robust OCSN** | P(y_t \| h_t) | During Kalman update |
| **RBPF Core** | P(h_t \| y_{1:t}, regime_t) | Always |

**Key Insight**: These are independent mechanisms:
- **Hawkes says**: "We just had a shock, so stay in / move to high-vol regime"
- **Robust OCSN says**: "Given our regime, this extreme return is plausible"

Without Hawkes, the filter may correctly identify a crash as R3 but then immediately transition back to R0. Without Robust OCSN, the filter may fail to survive the crash at all (particle collapse).

---

## API Usage

### Quick Start (Recommended)

```c
RBPF_Extended *ext = rbpf_ext_create(512, 4, RBPF_PARAM_STORVIK);

/* Configure and initialize as usual */
rbpf_ext_set_regime_params(ext, ...);
rbpf_ext_build_transition_lut(ext, trans_matrix);

/* Apply preset BEFORE init */
rbpf_ext_apply_preset(ext, RBPF_PRESET_EQUITY_INDEX);

rbpf_ext_init(ext, -4.0f, 0.1f);

/* Run - extensions handled automatically */
rbpf_ext_step(ext, returns[t], &output);
```

### Manual Configuration

```c
RBPF_Extended *ext = rbpf_ext_create(512, 4, RBPF_PARAM_STORVIK);

/* Configure and initialize as usual */
rbpf_ext_set_regime_params(ext, ...);
rbpf_ext_build_transition_lut(ext, trans_matrix);
rbpf_ext_init(ext, -4.0f, 0.1f);

/* Enable Hawkes */
rbpf_ext_enable_hawkes(ext,
    0.05f,   /* mu: baseline intensity */
    0.30f,   /* alpha: jump size */
    0.10f,   /* beta: decay rate */
    0.03f);  /* threshold: 3% return */

/* Enable Robust OCSN (uses default per-regime params) */
rbpf_ext_enable_robust_ocsn(ext);
```

### Custom Configuration

```c
/* Hawkes boost behavior */
rbpf_ext_set_hawkes_boost(ext,
    0.15f,   /* boost_scale */
    0.30f);  /* boost_cap */

/* Per-regime outlier params */
rbpf_ext_set_outlier_params(ext, 3,  /* R3 (crisis) */
    0.05f,   /* 5% outlier probability */
    40.0f);  /* variance 40 */
```

### Diagnostics

```c
RBPF_KSC_Output output;
rbpf_ext_step(ext, returns[t], &output);

float intensity = rbpf_ext_get_hawkes_intensity(ext);
float outlier_frac = rbpf_ext_get_outlier_fraction(ext);

if (outlier_frac > 0.5f) {
    printf("OUTLIER: %.0f%% of likelihood from 11th component\n",
           outlier_frac * 100);
}
```

### A/B Testing

```c
/* Run with both enabled */
run_backtest(ext, data, "both");

/* Disable Hawkes, keep OCSN */
rbpf_ext_disable_hawkes(ext);
run_backtest(ext, data, "ocsn_only");

/* Disable OCSN, re-enable Hawkes */
rbpf_ext_enable_hawkes(ext, 0.05f, 0.3f, 0.1f, 0.03f);
rbpf_ext_disable_robust_ocsn(ext);
run_backtest(ext, data, "hawkes_only");

/* Disable both (baseline) */
rbpf_ext_disable_hawkes(ext);
run_backtest(ext, data, "baseline");
```

---

## Expected Benefits

### Robust OCSN

| Metric | Without | With | Improvement |
|--------|---------|------|-------------|
| Particle survival (crash) | 10-20% | 80-95% | 4-5× |
| ESS during crisis | < 50 | > 200 | 4× |
| h_t spike after crash | +5 to +8 | +2 to +3 | Realistic |
| False R3 exits | Common | Rare | Stability |

### Hawkes

| Metric | Without | With | Improvement |
|--------|---------|------|-------------|
| R3 persistence (actual crisis) | 3-5 ticks | 8-15 ticks | Realistic |
| Whipsaw (R3→R0→R3) | Common | Rare | Smoother |
| Transition lag after shock | 2-3 ticks | 0-1 ticks | Faster |

### Combined

- **Flash crash scenario**: Hawkes keeps filter in R3, OCSN keeps particles alive
- **Recovery scenario**: Hawkes decays, allowing natural transition back to calm
- **Normal markets**: Both components dormant, no overhead

---

## Implementation Files

| File | Contents |
|------|----------|
| `rbpf_ksc_param_integration_header_additions.h` | Struct definitions, API declarations |
| `rbpf_ksc_update_robust.c` | 11th component likelihood function |
| `rbpf_ksc_param_integration_ext_impl.c` | Hawkes logic, modified `rbpf_ext_step()` |
| `example_hawkes_robust_ocsn.c` | Usage example |

---

## Tuning Guidelines

### Asset Class Presets (Recommended Starting Points)

Use `rbpf_ext_apply_preset()` for pre-tuned parameters:

```c
rbpf_ext_apply_preset(ext, RBPF_PRESET_EQUITY_INDEX);
```

| Preset | Assets | Tail Behavior | Hawkes Threshold | Outlier Var (R3) |
|--------|--------|---------------|------------------|------------------|
| `EQUITY_INDEX` | SPY, QQQ, ES | Moderate | 2.5% | 30 |
| `SINGLE_STOCK` | AAPL, TSLA | Fat | 3.0% | 38 |
| `FX_G10` | EUR/USD, USD/JPY | Thin | 1.5% | 28 |
| `FX_EM` | USD/MXN, USD/TRY | Very Fat | 2.5% | 45 |
| `CRYPTO` | BTC, ETH | Extreme | 5.0% | 50 |
| `COMMODITIES` | CL, GC | Moderate-Fat | 3.0% | 40 |
| `BONDS` | ZN, ZB | Very Thin | 1.0% | 35 |
| `CUSTOM` | User-defined | — | — | — |

### Adaptive Hawkes Decay (Prevents Phantom Regime)

The decay rate `β` varies by current regime:

```
β_effective = β_base × scale[regime]
```

| Regime | Default Scale | Effective Half-Life | Purpose |
|--------|---------------|---------------------|---------|
| R0 (Calm) | 2.0× | ~3.5 ticks | Fast recovery after flash crash |
| R1 (Mild) | 1.5× | ~4.6 ticks | Moderate recovery |
| R2 (Elevated) | 1.0× | ~7 ticks | Base rate |
| R3 (Crisis) | 0.5× | ~14 ticks | Crisis persists |

**Hysteresis Effect**: This creates positive feedback—once in R3, intensity decays slowly, keeping us in crisis longer. This is desirable for volatility trend-following.

```c
/* Enable/disable adaptive decay */
rbpf_ext_enable_adaptive_hawkes(ext, 1);  /* ON */
rbpf_ext_enable_adaptive_hawkes(ext, 0);  /* OFF (fixed β) */

/* Custom per-regime scales */
rbpf_ext_set_hawkes_regime_scale(ext, 0, 3.0f);  /* R0: Very fast decay */
rbpf_ext_set_hawkes_regime_scale(ext, 3, 0.3f);  /* R3: Very slow decay */
```

### Hawkes Parameters

| Parameter | Low | Medium | High | Notes |
|-----------|-----|--------|------|-------|
| μ (baseline) | 0.02 | 0.05 | 0.10 | Higher = more sensitive |
| α (jump) | 0.10 | 0.30 | 0.50 | Higher = stronger clustering |
| β (decay) | 0.05 | 0.10 | 0.20 | Higher = faster decay |
| threshold | 1% | 3% | 5% | Lower = more triggers |
| boost_scale | 0.05 | 0.10 | 0.20 | Higher = stronger effect |
| boost_cap | 0.10 | 0.25 | 0.40 | Safety limit |

**Rule of thumb**: Start with medium values, adjust based on:
- Too many false R3 entries → increase threshold, decrease α
- Too slow R3 entry → decrease threshold, increase α
- R3 too sticky → increase β
- R3 exits too fast → decrease β

### Robust OCSN Parameters (Tightened Bounds)

**Critical**: Outlier variance must be bounded to preserve signal.

| Variance | Effect | Kalman Gain K |
|----------|--------|---------------|
| < 15 | Competes with OCSN | ~0.007 (distorts) |
| **18-30** | **Sweet spot** | ~0.004-0.005 (gentle) |
| > 50 | Signal suppression | < 0.002 (ignores crash) |

**Recommended values** (~3× max OCSN variance of 7.33):

| Regime | Prob Range | Variance | Rationale |
|--------|------------|----------|-----------|
| R0 | 0.8% - 1.5% | 15-20 | Tighter, rare outliers |
| R1 | 1.0% - 2.0% | 18-22 | Moderate |
| R2 | 1.5% - 2.5% | 22-28 | Elevated tolerance |
| R3 | 2.0% - 3.5% | 28-35 | Crisis, but bounded |

---

## Known Risks and Mitigations

### 1. Signal Suppression (Robust OCSN)

**Risk**: Large outlier variance → K≈0 → h_t ignores crash magnitude

**Mitigation**:
- Variance bounded to ~3× OCSN max (18-30)
- Non-zero K ensures *some* state update during outliers
- Regime transition (R0→R3) shifts μ_vol baseline independently

**Residual**: First tick of crash may have muted h_t update. Accept this tradeoff for particle survival.

### 2. Phantom Regime (Hawkes)

**Risk**: Slow β decay → stuck in R3 after flash crash → false positives

**Mitigation**:
- Adaptive β: R0 decays 2× faster than R3
- Fast recovery in calm regimes prevents "sticky" intensity
- Hysteresis is intentional for true crises

**Residual**: Tuning β scales is asset-dependent. Use presets as starting points.

### 3. Theoretical Impurity

**Risk**: Time-varying transition matrix breaks pure Bayesian framework

**Acknowledgment**: 
- This is now a "hybrid SV-GARCH" model
- `marginal_likelihood` output is not valid for rigorous model comparison
- But predictive density (what matters for trading) is improved

**Mitigation**: Document as hybrid model. Accept for practical use.

### 4. Parameter Explosion

**Risk**: 8+ new hyperparameters → overfitting risk

**Mitigation**:
- Asset class presets reduce tuning burden
- Sensible defaults work for most instruments
- Most params have natural interpretations (threshold = "big move")

**Residual**: Fine-tuning still required for exotic instruments.

---

## Testing Plan

1. **Unit tests**: Verify Hawkes intensity math, OCSN likelihood computation
2. **Synthetic crash**: 500-tick scenario with embedded crash, verify particle survival
3. **Regime accuracy**: Compare confusion matrices with/without extensions
4. **Historical replay**: 2008 crisis, COVID crash, flash crashes
5. **A/B backtest**: Trading performance with/without extensions

---

## References

- Kim, Shephard & Chib (1998): Original OCSN approximation
- Hawkes (1971): Self-exciting point processes
- Omori, Chib, Shephard & Nakajima (2007): 10-component mixture (NOT Omori's seismic law)
- Embrechts et al. (2011): Hawkes processes in finance

---

## Planned Enhancements

The following extensions are well-established in empirical finance and control theory. They are documented here for future implementation.

### 1. Leverage Effect (Asymmetry)

**Source**: Black (1976), Nelson (1991) — EGARCH/GJR-GARCH literature

**The Stylized Fact**: Negative returns increase volatility more than positive returns of the same magnitude. This "panic > euphoria" asymmetry is one of the most robust findings in financial econometrics. It explains options skew and is why the VIX spikes on down moves but barely reacts to up moves.

**Current Gap**: Our volatility dynamics are symmetric:
```
h_t = φ·h_{t-1} + μ_vol + σ_vol·ε_t
```

**Proposed Fix**: Add GJR-GARCH style asymmetry in the predict step:
```c
/* Leverage effect: negative returns boost volatility more */
float leverage_term = 0.0f;
if (prev_return < 0.0f) {
    leverage_term = rho * fabsf(prev_return);  /* rho ≈ 0.5-1.0 */
}
h_pred = phi * h_prev + mu_vol + leverage_term;
```

**Implementation Location**: `rbpf_ksc_predict()` or regime-specific dynamics

**Risk if Omitted**: Systematically underestimate downside volatility risk

---

### 2. Adaptive Parameter Forgetting (Discount Factor)

**Source**: J.P. Morgan RiskMetrics (1996), West & Harrison (1997) — Bayesian forecasting with discounting

**The Problem**: Financial time series are non-stationary. Parameters that defined the 2008 crisis are not the same as 2025. Without forgetting, Storvik sufficient statistics accumulate forever:
- Posteriors become too tight (overconfidence)
- Adaptability dies (can't respond to regime drift)
- Parameters converge to "grand average" of all history

**Current Gap**: Storvik stats accumulate without decay:
```c
soa->sum_x[idx] += new_x;
soa->sum_x2[idx] += new_x * new_x;
soa->n_obs[idx] += 1.0;
```

**Proposed Fix**: Apply exponential forgetting (discount factor λ ≈ 0.97-0.995):
```c
/* Discount old information before accumulating new */
const double lambda = 0.995;  /* ~200 tick half-life */

soa->sum_x[idx] *= lambda;
soa->sum_x2[idx] *= lambda;
soa->n_obs[idx] *= lambda;  /* Effective sample size decays */

/* Then accumulate new observation */
soa->sum_x[idx] += new_x;
soa->sum_x2[idx] += new_x * new_x;
soa->n_obs[idx] += 1.0;
```

**Implementation Location**: `param_learn_update()` in Storvik learner

**Tuning**: 
- λ = 0.99: Half-life ~69 ticks (fast adaptation)
- λ = 0.995: Half-life ~139 ticks (moderate)
- λ = 0.999: Half-life ~693 ticks (slow, stable)

**Risk if Omitted**: Model fossilizes over time, unable to adapt to new market regimes

---

### 3. Defibrillator (Divergence Watchdog)

**Source**: Thrun et al. (2005) — Probabilistic Robotics, "Kidnapped Robot Problem"

**Status**: ✅ **IMPLEMENTED** — See `rbpf_watchdog.h` and `rbpf_watchdog.c`

**The Problem**: In production, particle filters can diverge due to:
- Model mismatch (reality deviates from assumptions)
- Black swan events (observations outside training distribution)
- Numerical issues (underflow, overflow)

When this happens:
- ESS collapses (all weight on one particle)
- Weights become NaN/Inf
- State estimates diverge from reality
- Trading system acts on garbage

**Failure Modes Detected**:

| Code | Trigger | Severity |
|------|---------|----------|
| `ESS_COLLAPSE` | ESS < 5% of N | SOFT |
| `NAN_WEIGHT` | NaN in weights | CRITICAL |
| `INF_WEIGHT` | Inf in weights | CRITICAL |
| `ZERO_WEIGHTS` | Sum of weights ≈ 0 | MEDIUM |
| `LIK_COLLAPSE` | log_lik < -1e6 | MEDIUM |
| `LIK_NAN` | NaN in likelihood | HARD |
| `STATE_DIVERGENCE` | \|h - h_implied\| > 8σ | MEDIUM |
| `STATE_NAN` | NaN in state | CRITICAL |
| `STATE_EXPLOSION` | \|h\| > 10 | HARD |
| `REGIME_DEADLOCK` | Stuck in R0 during crisis | SOFT |

**Reset Severity Levels**:

| Severity | Action |
|----------|--------|
| SOFT | Re-diversify particles, add jitter |
| MEDIUM | Reset state distribution, keep Storvik params |
| HARD | Full reset including state |
| CRITICAL | Full reset, signal upstream, consider halt |

**Key Features**:
- **Cooldown**: Minimum ticks between resets (prevents reset storms)
- **Circuit Breaker**: Max resets per session (e.g., 10), then halt
- **Callback**: Notify upstream systems (risk manager, execution)
- **Event Logging**: Full state capture before reset for post-mortem

**Usage**:

```c
/* Initialize */
RBPF_Watchdog watchdog;
rbpf_watchdog_init(&watchdog);

/* Set callback for trading system */
rbpf_watchdog_set_callback(&watchdog, on_reset_callback, &trading_ctx);

/* In main loop */
rbpf_ext_step(ext, obs, &output);

RBPF_ResetReason reason = rbpf_watchdog_check(ext, &watchdog, obs);
if (reason != RESET_NONE) {
    RBPF_ResetEvent event;
    rbpf_watchdog_reset(ext, &watchdog, reason, &event);
}

/* Start of trading day */
rbpf_watchdog_new_session(&watchdog);
```

**Files**:
- `rbpf_watchdog.h` — Header with structs and API
- `rbpf_watchdog.c` — Implementation
- `example_watchdog_integration.c` — Usage examples

---

### Provenance Summary

| Strategy | Source/Authority | Role in Finance |
|----------|------------------|-----------------|
| Robust OCSN | Omori et al. (2007) + Robust Statistics | Handling fat tails (kurtosis) |
| Hawkes Process | Hawkes (1971); Embrechts (2011) | Modeling clustering (aftershocks) |
| **Leverage Effect** | Black (1976); Nelson (1991) | Modeling skew (panic > euphoria) |
| **Adaptive Forgetting** | RiskMetrics (1996); West & Harrison (1997) | Handling regime drift (non-stationarity) |
| **Defibrillator** | Thrun (2005) — Probabilistic Robotics | Handling numerical failure (production safety) |

---

### Implementation Priority

| Enhancement | Priority | Status | Dependencies |
|-------------|----------|--------|--------------|
| Defibrillator | **P0 (Critical)** | ✅ **DONE** | None |
| Adaptive Forgetting | P1 (High) | Planned | Storvik internals |
| Leverage Effect | P2 (Medium) | Planned | Predict step |

**Next Steps**: 
1. Integrate watchdog into your build (add `rbpf_watchdog.c`)
2. Add watchdog check to `rbpf_ext_step()` or call manually
3. Implement Adaptive Forgetting for long-running systems
4. Add Leverage Effect for improved downside risk estimation

---

## Implementation Notes

### 1. Transition Matrix Normalization

When Hawkes modifies the transition matrix, rows must remain valid probability distributions (sum to 1.0). The implementation:

- Enforces a **minimum probability floor** (2%) for any cell
- Only "steals" probability that's available above the floor
- Redistributes exactly what was stolen (conservation)
- Includes debug assertions to verify row sums

```c
const rbpf_real_t MIN_PROB = RBPF_REAL(0.02);

for (int to = 0; to <= from; to++) {
    rbpf_real_t available = row[to] - MIN_PROB;
    if (available <= 0) continue;
    
    rbpf_real_t steal = fminf(row[to] * boost, available);
    row[to] -= steal;
    to_redistribute += steal;
}
/* to_redistribute is then added to higher regimes */
```

### 2. Log-Sum-Exp Numerical Stability

The robust OCSN computes likelihoods that can differ by orders of magnitude. We use the log-sum-exp trick with a **shared maximum** across all 11 components:

```
PASS 1:  Compute log-lik for components 1-10, track max_ll[i]
PASS 1b: Compute log-lik for component 11, update max_ll[i]
PASS 2:  Accumulate exp(log_lik_k - max_ll[i]) for k=1..10
PASS 3:  Accumulate exp(outlier_ll - max_ll[i])

Final:   log_weight += log(lik_total) + max_ll
```

This ensures we never compute `exp(large_positive)` or `exp(large_negative)`.

### 3. Threshold Consistency

Hawkes uses **raw returns** for the excitation threshold, not log-squared returns:

| Threshold Type | Check | Example (3% threshold) |
|---------------|-------|------------------------|
| Raw return | `|obs| > 0.03` | `-0.05 → triggered` |
| Log-squared | `y > log(0.03²)` | `y > -7.0 → triggered` |

We use raw returns because:
- More intuitive for parameter tuning ("3% move triggers excitation")
- Consistent with how traders think about moves
- The observation `obs` is already available before the log transform
