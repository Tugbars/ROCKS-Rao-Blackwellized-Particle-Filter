# MMPF Hybrid Architecture

## A Hierarchical Mixture of Experts with Timescale Separation

### Overview

The Multi-Model Particle Filter (MMPF) implements a principled approximation of hierarchical Bayesian inference, designed for real-time volatility estimation in high-frequency trading. It achieves **~150μs latency** while maintaining the statistical rigor of much more expensive joint inference methods.

The architecture separates **level** (where is volatility?) from **identity** (which regime?) from **dynamics** (how does it behave?), solving the fundamental tension between adaptation and discrimination.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SLOW LOOP (~daily)                       │
│                                                             │
│    EWMA Baseline: μ_baseline ← α·μ_baseline + (1-α)·vol    │
│                         (α = 0.98)                          │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│               STRUCTURAL LAYER (fixed)                      │
│                                                             │
│    Calm   = μ_baseline - 1.5    (0.22× baseline)           │
│    Trend  = μ_baseline + 0.0    (1.00× baseline)           │
│    Crisis = μ_baseline + 2.0    (7.39× baseline)           │
│                                                             │
│         Separation GUARANTEED by fixed offsets              │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    FAST LOOP (~150μs)                       │
│                                                             │
│    ┌──────────┐    ┌──────────┐    ┌──────────┐            │
│    │   RBPF   │    │   RBPF   │    │   RBPF   │            │
│    │   Calm   │    │  Trend   │    │  Crisis  │◄──┐        │
│    │  φ=0.98  │    │  φ=0.95  │    │  φ=0.80  │   │        │
│    └────┬─────┘    └────┬─────┘    └────┬─────┘   │        │
│         │               │               │         │        │
│         └───────────────┼───────────────┘    PANIC DRIFT   │
│                         │                   (fat tails)    │
│                         ▼                         │        │
│              IMM Weight Update ───────────────────┘        │
│         (Bayesian model comparison)                        │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              WINNER-TAKES-ALL LEARNING                      │
│                                                             │
│    Dominant hypothesis (max weight) updates:                │
│      • φ (mean-reversion speed)                            │
│      • σ_η (innovation volatility)                         │
│                                                             │
│    Non-winners: FROZEN (preserves structural memory)        │
└─────────────────────────────────────────────────────────────┘
```

---

## Scientific Foundations

### 1. Interacting Multiple Model (IMM)

**Source:** Bar-Shalom & Li (1988), control theory

**Original use:** Tracking targets that switch behaviors (aircraft maneuvering)

**Our use:** Tracking markets that switch regimes (calm → trending → crisis)

The IMM framework provides:
- Parallel hypothesis evaluation
- Soft switching via probability weights
- Markov transition dynamics between models

### 2. Timescale Separation ("Climate vs Weather")

**Source:** Slow Feature Analysis, Hierarchical Bayesian Models

**Principle:** Complex systems have fast variables (daily returns) and slow variables (secular volatility trends)

**Our use:** 
- **Slow:** EWMA baseline tracks the "climate" (market volatility level over weeks/months)
- **Fast:** RBPFs track the "weather" (daily fluctuations around baseline)

This solves the **non-stationarity problem**: volatility isn't stationary, but deviations from a slowly-moving baseline can be.

### 3. Gated / Winner-Takes-All Learning

**Source:** Competitive Learning (Kohonen, 1982), Mixture of Experts (Jacobs, Jordan, Nowlan, Hinton, 1991)

**Principle:** Experts should only train on data they explain well, preventing **catastrophic forgetting**

**Our use:** 
- Only the dominant hypothesis updates its dynamics each tick
- Crisis "remembers" how to handle crashes even after years of calm
- Prevents all experts from converging to the same behavior

### 4. Identifiability Constraints

**Source:** Mixture model theory, Bayesian model selection

**Problem:** Without constraints, mixture components suffer **mode collapse** (converge to identical parameters)

**Our use:**
- Fixed offsets guarantee `Calm < Trend < Crisis` always
- Level (μ_vol) is NOT learned per-hypothesis
- Only dynamics (φ, σ_η) adapt, preserving structural separation

---

## The Discovery: Why Separation Matters

Through empirical testing, we discovered that **level** and **identity** cannot be learned together:

| Approach | Result | Problem |
|----------|--------|---------|
| Storvik sync (learn μ_vol per-particle) | Convergence | All hypotheses drift to same level |
| Gated μ_vol learning (WTA) | Identity theft | Calm learns Trend's data, becomes Trend |
| **Fixed offsets + WTA dynamics** | **Success** | Level locked, behavior adapts |

**The insight:** A hypothesis's identity IS its level. If Calm learns that vol is 1.0%, it's no longer Calm—it's become Trend. The μ_vol parameter must be structural, not adaptive.

---

## Computational Complexity

**Full Hierarchical Bayesian inference:**
- MCMC over joint posterior of (baseline, offsets, dynamics, states)
- Cost: O(minutes) per tick
- Impractical for real-time trading

**Our factorized approximation:**

| Component | Operation | Complexity |
|-----------|-----------|------------|
| Baseline | EWMA update | O(1) |
| Reanchoring | μ_vol = baseline + offset | O(K) where K=3 |
| Kalman predict/update | Per particle | O(N) |
| Weight normalization | Likelihood ratio | O(K) |
| WTA Learning | Sufficient statistics | O(1) |

**Total: ~150μs** — three orders of magnitude faster than joint inference

---

## Configuration

```c
/* Hybrid Architecture Configuration */
cfg.enable_global_baseline = 1;       // EWMA controls μ_vol
cfg.global_mu_vol_alpha = 0.98;       // ~34 tick half-life
cfg.mu_vol_offsets[CALM]   = -1.5;    // exp(-1.5) = 0.22×
cfg.mu_vol_offsets[TREND]  =  0.0;    // 1.00×
cfg.mu_vol_offsets[CRISIS] = +2.0;    // exp(2.0) = 7.39×
cfg.enable_gated_learning = 1;        // WTA for φ, σ_η
cfg.enable_storvik_sync = 0;          // No per-particle μ_vol learning
```

---

## Results (SPY 5-Year Daily)

```
Final state:
  Global baseline: -4.969 (0.70% vol)

Per-hypothesis (μ_vol from baseline, φ/σ_η from WTA learning):
  Calm:   μ_vol=-6.469 (0.16%), φ=0.993, σ_η=0.093
  Trend:  μ_vol=-4.969 (0.70%), φ=0.800, σ_η=0.102
  Crisis: μ_vol=-2.969 (5.14%), φ=0.800, σ_η=0.420

Performance:
  Latency:    147.78 μs/tick average
  Throughput: 6,767 ticks/sec
  Switches:   78 regime transitions
  Distribution: Calm 37.9%, Trend 61.0%, Crisis 1.1%
```

---

## Key Equations

**Baseline update (Fair Weather Gate):**
```
if w_crisis < 0.50:
    μ_baseline ← α·μ_baseline + (1-α)·weighted_log_vol
else:
    μ_baseline ← frozen (protect from crisis spikes)
```

**Hypothesis anchoring:**
```
μ_vol[k] = μ_baseline + offset[k]
```

**WTA learning gate:**
```
winner = argmax(weights)
if k == winner:
    accumulate sufficient statistics for φ, σ_η
else:
    freeze (preserve structural memory)
```

**Panic Drift (adaptive Crisis ceiling):**
```
if w_crisis > 0.5 AND y_log > crisis_anchor:
    gap = y_log - crisis_anchor
    crisis_drift += 0.2 × gap
else:
    crisis_drift *= 0.95  (decay)

crisis_drift = min(crisis_drift, 1.5)  (cap)
crisis_anchor_effective = baseline + 2.0 + crisis_drift
```

---

## Panic Drift: Adaptive Fat Tail Capture

### The Problem

Fixed offsets guarantee discrimination but create a ceiling:
- Crisis anchor = Baseline + 2.0 = exp(+2.0) × baseline ≈ 7.4× baseline
- If baseline is 0.7% vol → Crisis cap at ~5.2% vol
- But actual crash vol can hit 10-20% → filter under-estimates

### The Solution

**Panic Drift** temporarily boosts the Crisis ceiling during extreme events:

1. **Trigger**: w_crisis > 50% AND observation exceeds Crisis anchor
2. **Accumulate**: Drift grows proportional to how much observation exceeds anchor
3. **Decay**: When crisis passes, drift decays back to zero (95% per tick)
4. **Cap**: Maximum drift of +1.5 (Crisis can go from +2.0 to +3.5 offset)

### Why This Works

```
Normal times:
  Crisis = Baseline + 2.0 (fixed, guarantees discrimination)

Crisis detected with extreme observation:
  Crisis = Baseline + 2.0 + drift (adaptive ceiling)
  
Crisis passes:
  drift → 0 (reverts to structural default)
```

This is the "Adrenaline" mechanism: emergency override of structural constraints that auto-reverts when danger passes.

### Configuration

```c
cfg.enable_panic_drift = 1;
cfg.panic_drift_threshold = 0.50;  /* Activate when w_crisis > 50% */
cfg.panic_drift_rate = 0.20;       /* Accumulation speed */
cfg.panic_drift_decay = 0.95;      /* Decay when not in panic */
cfg.panic_drift_max = 1.50;        /* Max boost to Crisis offset */
```

---

## Summary

The MMPF Hybrid Architecture achieves robust regime detection by:

1. **Decoupling level from identity** — baseline sets WHERE, offsets set WHO
2. **Decoupling identity from dynamics** — μ_vol is structural, φ/σ_η are learned
3. **Protecting specialist knowledge** — WTA prevents catastrophic forgetting
4. **Tracking non-stationarity** — EWMA baseline adapts to secular drift
5. **Capturing fat tails** — Panic Drift lifts Crisis ceiling during extreme events

This is principled engineering: a real-time approximation of hierarchical Bayesian inference that runs in microseconds while preserving essential statistical structure.

---

## References

- Bar-Shalom, Y., & Li, X. R. (1988). *Estimation and Tracking: Principles, Techniques, and Software*
- Jacobs, R. A., Jordan, M. I., Nowlan, S. J., & Hinton, G. E. (1991). Adaptive Mixtures of Local Experts. *Neural Computation*
- Kim, S., Shephard, N., & Chib, S. (1998). Stochastic Volatility: Likelihood Inference and Comparison with ARCH Models. *Review of Economic Studies*
- Kohonen, T. (1982). Self-organized formation of topologically correct feature maps. *Biological Cybernetics*
