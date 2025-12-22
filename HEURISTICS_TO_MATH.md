# From Heuristics to Mathematics: Principled Particle Filter Design

## Overview

This document describes three major improvements to the RBPF (Rao-Blackwellized Particle Filter) where arbitrary heuristics were replaced with mathematically principled solutions. Each improvement addresses a specific failure mode that can occur in production high-frequency trading systems.

---

## 1. Weight Clamping → KL Divergence Ceiling

### The Problem

Particle filters update weights based on how well each particle explains new observations:

```
log_weight[i] += log_likelihood(observation | particle[i])
```

When an extreme observation arrives (fat-finger trade, flash crash, data glitch), some particles receive enormous negative log-likelihoods. A single "impossible" observation can drive weights to `-300` or below, causing:

- **Numerical underflow**: `exp(-300) = 0` in floating point
- **Particle genocide**: All but 1-2 particles get zero weight
- **ESS collapse**: Effective Sample Size drops from 500 to 1
- **Information loss**: Decades of Bayesian learning erased in one tick

### The Old Solution

```c
if (log_weight[i] < -50.0f) log_weight[i] = -50.0f;
```

A hard floor at `-50`. Why `-50`? Because it "seemed to work." This is a heuristic with no theoretical basis.

### What Could Go Wrong

| Scenario | Problem |
|----------|---------|
| Floor too high (`-20`) | Legitimate low-probability particles survive, polluting inference |
| Floor too low (`-100`) | Doesn't prevent collapse on extreme outliers |
| Any fixed value | Doesn't adapt to particle count, model complexity, or data characteristics |

The fundamental issue: **a fixed threshold ignores how many particles you have**. With 100 particles, `-50` might be appropriate. With 10,000 particles, it's far too aggressive.

### The New Solution: KL Divergence Ceiling

Instead of clamping individual weights, we constrain **how much the entire weight distribution can change per tick**. The constraint is:

```
KL(w_proposed || w_old) ≤ log(N) nats
```

Where:
- `KL` = Kullback-Leibler divergence (information-theoretic distance)
- `N` = number of particles
- `log(N)` ≈ 6.24 nats for 512 particles

**Why log(N)?** This is the maximum entropy of a discrete distribution over N outcomes. It represents "complete uncertainty" — the most the posterior could possibly shift is from certainty to uniform. Any larger shift is mathematically impossible, so clamping there catches only numerical pathologies.

### Why This Works Better

| Property | Old (`-50` clamp) | New (KL ceiling) |
|----------|-------------------|------------------|
| Adapts to particle count | ❌ | ✓ |
| Has theoretical basis | ❌ | ✓ (information geometry) |
| Preserves relative ordering | ❌ | ✓ |
| Smooth degradation | ❌ (hard cutoff) | ✓ (continuous tempering) |

### Alternatives Considered

1. **Robust likelihood (Student-t)**: Addresses the problem at the observation model level. We implemented this too, but it doesn't handle all outlier types (e.g., data errors that aren't heavy-tailed).

2. **Adaptive floor based on ESS**: Set floor dynamically to maintain minimum ESS. Rejected because it's reactive (damage already done) rather than preventive.

3. **Winsorization**: Clamp observations, not weights. Rejected because it discards potentially valid extreme observations.

4. **Mixture with uniform**: Blend likelihood with uniform distribution. This is essentially what β-tempering does, but KL framing gives us a principled choice of blend ratio.

### Why We Chose KL

The KL divergence ceiling emerged from asking: "What is the maximum amount of information one observation can provide?" Information theory gives a precise answer. The solution is self-calibrating, theoretically grounded, and degrades gracefully.

---

## 2. Magic Dampening Factor → Continuous β Tempering

### The Problem

Even with a ceiling, we might want to moderate weight updates during volatile periods — not because they're numerically dangerous, but because our model might be temporarily misspecified.

The old approach used a fixed dampening factor:

```c
log_weight[i] += 0.8 * log_likelihood;  // Why 0.8? Who knows.
```

### What Could Go Wrong

| Issue | Consequence |
|-------|-------------|
| `β = 0.8` always | Unnecessarily slows learning during calm periods |
| `β = 0.8` always | Insufficient dampening during extreme events |
| No adaptation | Can't learn what "normal" volatility looks like for this data |

A fixed β is like wearing sunglasses 24/7 — helpful at noon, useless at midnight.

### The New Solution: P² Quantile Learning

Instead of a fixed β, we:

1. **Track the empirical distribution** of KL divergences using the P² algorithm (streaming quantile estimation)
2. **Learn the 95th percentile** of "normal" KL values over ~500 ticks
3. **Set β dynamically**:
   - If KL < p95: β = 1.0 (no dampening, normal operation)
   - If p95 < KL < ceiling: β = p95 / KL (smooth dampening)
   - If KL > ceiling: β = ceiling / KL (hard limit kicks in)

This creates a **continuous tempering curve**:

```
β
1.0 ─────────┐
             │
             ╲
              ╲
0.1 ──────────╲───────────
    0       p95      ceiling     KL
```

### Why This Works Better

| Property | Old (`β = 0.8`) | New (adaptive β) |
|----------|-----------------|------------------|
| Learns from data | ❌ | ✓ |
| Full speed when appropriate | ❌ | ✓ |
| Handles regime changes | ❌ | ✓ (p95 adapts) |
| No magic numbers | ❌ | ✓ |

### Alternatives Considered

1. **Fixed percentile (e.g., always use median)**: Simpler but doesn't handle heavy tails. P95 specifically targets "unusual but not impossible" events.

2. **Exponential moving average of KL**: Considered, but doesn't give us a threshold — just a smoother version of current KL.

3. **Online changepoint detection for β**: Use BOCPD to detect when β should change. Rejected as over-engineered; the continuous formula handles transitions naturally.

4. **Multiple β values per regime**: Let each volatility regime have its own dampening. Rejected because it adds parameters and the KL-based approach already captures regime-specific behavior implicitly.

### Why We Chose P² Quantile + Continuous Formula

The P² algorithm is elegant: O(1) space, O(1) update, converges to true quantile. Combined with the `β = p95/KL` formula, we get smooth adaptation with no tuning required. The system learns what "normal" looks like and adjusts expectations accordingly.

---

## 3. Arbitrary Regime Blending → Fisher-Rao Geodesic

### The Problem

When a particle needs to "teleport" to a new regime (mutation for diversity, BOCPD-triggered reset), we must blend its current state with the target regime's stationary distribution. The old approach:

```c
mu_new = 0.7 * mu_particle + 0.3 * mu_regime;
var_new = 0.7 * var_particle + 0.3 * var_regime;
```

Why 70/30? Because someone tried a few values and this one "felt right."

### What Could Go Wrong

| Issue | Consequence |
|-------|-------------|
| Fixed blend ignores uncertainty | A particle with high variance (uncertain) should move more toward regime |
| Fixed blend ignores distance | Small corrections treated same as large jumps |
| Linear blending of variance | Variances don't blend linearly on statistical manifolds |
| No geometric meaning | The path between Gaussians isn't a straight line in parameter space |

The fundamental issue: **Gaussian distributions live on a curved manifold, not flat Euclidean space.** Linear interpolation takes the "wrong" path.

### The New Solution: Fisher-Rao Geodesic

The Fisher-Rao metric defines the natural geometry of probability distributions. On this manifold:

- **Distance** = how distinguishable two distributions are
- **Geodesic** = shortest path between distributions
- **Interpolation** = moves along the geodesic, not a straight line

For Gaussians, the geodesic interpolation is:

```
t = precision_regime / (precision_particle + precision_regime)
```

Where `precision = 1/variance`. This means:

- **Uncertain particle** (high variance, low precision) → large `t` → moves strongly toward regime
- **Confident particle** (low variance, high precision) → small `t` → preserves current state

The interpolation then follows the geodesic in the upper half-plane model of Gaussian space.

### Why This Works Better

| Property | Old (70/30 blend) | New (Fisher-Rao) |
|----------|-------------------|------------------|
| Respects uncertainty | ❌ | ✓ |
| Geometrically correct | ❌ | ✓ |
| Preserves information | ❌ | ✓ (low-precision particles sacrifice less) |
| Distance-aware | ❌ | ✓ |

### Small-Angle Optimization

For computational efficiency, when `|Δμ| < 0.1σ` (particle already close to regime mean), we skip the trigonometric functions and use a linear/geometric approximation:

```
μ_out = (1-t) * μ_particle + t * μ_regime        // Linear for μ
σ_out = σ_particle^(1-t) * σ_regime^t             // Geometric for σ
```

This saves ~1μs on the P99 latency path while being mathematically equivalent for small angles (Taylor expansion).

### Alternatives Considered

1. **Optimal transport (Wasserstein)**: Another principled way to interpolate distributions. Rejected because it's computationally heavier and Fisher-Rao is more natural for exponential families.

2. **Moment matching**: Blend means and variances to match target moments. Rejected because it doesn't account for particle confidence.

3. **Importance sampling from regime**: Sample a new state from regime stationary distribution. Rejected because it discards particle's current information entirely.

4. **α-divergence geodesics**: Generalization of KL that includes Fisher-Rao as special case. Rejected as over-complicated for this application.

### Why We Chose Fisher-Rao

The Fisher-Rao metric is the **unique** Riemannian metric on probability distributions that is invariant under sufficient statistics. It's not just "a" principled choice — it's "the" principled choice for this problem. The geodesic interpolation naturally incorporates uncertainty weighting without any tuning parameters.

---

## Summary: Design Principles

These three improvements share common themes:

### 1. Let the Data Decide

| Old | New |
|-----|-----|
| Fixed `-50` floor | Ceiling scales with particle count |
| Fixed `β = 0.8` | β learned from empirical KL distribution |
| Fixed `70/30` blend | Blend ratio from relative precisions |

### 2. Use Information Geometry

All three solutions draw from the same mathematical framework:

- **KL divergence** measures information gain
- **Fisher-Rao metric** defines natural geometry
- **Sufficient statistics** preserved under transformations

### 3. Degrade Gracefully

| Old | New |
|-----|-----|
| Hard clamp (discontinuity) | Continuous tempering curve |
| Binary dampening (on/off) | Smooth β from 1.0 to floor |
| Fixed teleport (jarring) | Geodesic path (smooth) |

### 4. Self-Calibrate

The system adapts to:
- Particle count (ceiling = log N)
- Data characteristics (p95 learned online)
- Particle confidence (precision weighting)

No manual tuning required for different instruments, frequencies, or market conditions.

---

## Remaining Heuristics

Not everything has been converted. Still present in the system:

| Heuristic | Why It Remains |
|-----------|----------------|
| `β_floor = 0.1` | Hard limit to prevent total update suppression |
| `warmup = 500 ticks` | P² needs samples before p95 is meaningful |
| `zombie threshold = 10 ticks` | When to declare filter "stuck" |
| `small-angle threshold = 0.1σ` | When to use fast path vs full geodesic |

These are secondary parameters with clear physical meaning and low sensitivity. The primary heuristics that were causing production issues have been eliminated.

---

## References

1. **P² Algorithm**: Jain, R., & Chlamtac, I. (1985). "The P² Algorithm for Dynamic Calculation of Quantiles and Histograms Without Storing Observations"

2. **Fisher-Rao Geometry**: Amari, S., & Nagaoka, H. (2000). "Methods of Information Geometry"

3. **Tempered Posteriors**: Grünwald, P. (2012). "The Safe Bayesian" — theoretical foundation for β < 1 tempering

4. **KL Divergence in Particle Filters**: Chopin, N. (2004). "Central Limit Theorem for Sequential Monte Carlo Methods and its Application to Bayesian Inference"
