# MMPF-ROCKS: Architecture & Strategic Decisions

## Executive Summary

MMPF-ROCKS is a Multi-Model Particle Filter for real-time volatility estimation in HFT applications. It runs three competing hypotheses (Calm, Trend, Crisis) that provide different explanations for observed market behavior. The system uses Bayesian model comparison to determine which hypothesis best explains current conditions.

**Core Principle:** Models don't communicate — they compete through likelihoods.

---

## The Problem: Why Single-Model Filters Fail

### The Gaussian Assumption

Traditional volatility filters assume Gaussian (normal) observations:

| Move | Gaussian Probability | Reality |
|------|---------------------|---------|
| 3σ   | 1 in 370            | Weekly  |
| 5σ   | 1 in 3.5 million    | Monthly |
| 10σ  | 1 in 10²³           | Yearly  |

**The Panic Response:**

When a Gaussian model sees a 5% move (≈5σ):
1. Calculates likelihood: L ≈ 0.000000001
2. Thinks: "This is impossible given my σ estimate!"
3. Violently spikes σ estimate to explain the "impossible" observation
4. Result: Jumpy, nervous volatility tracker that ruins Kelly sizing

### The Solution: Fat Tails + Multiple Hypotheses

Student-t distribution with multiple competing models:
- **Polynomial decay** ($x^{-(\nu+1)}$) instead of exponential ($e^{-x^2}$)
- Different models for different market regimes
- Bayesian model comparison picks the winner

---

## The Suspension System Analogy

The filter is like a car's adaptive suspension system:

| Component | Filter Element | Mechanism | Handles |
|-----------|----------------|-----------|---------|
| **Rigid steel rod** | Gaussian | None | Nothing — transmits all shock |
| **Spring** | Student-t (ν) | Elastic absorption | Potholes (3-5σ) |
| **Bump stop** | OCSN | Hard cutoff | Craters (10σ+) |
| **Adaptive controller** | Storvik | Learns road type | Tunes spring stiffness & ride height |
| **Swim lanes** | μ_vol bounds | Position limits | Prevents model collision |
| **Turbocharger** | Adaptive Forgetting | Learning speed | How fast to adapt |

---

## The Full Defense Stack

```
┌─────────────────────────────────────────────────────────────────┐
│                        OBSERVATION (y)                          │
└─────────────────────────┬───────────────────────────────────────┘
                          │
         ┌────────────────┼────────────────┐
         │                │                │
         ▼                ▼                ▼
    ┌─────────┐      ┌─────────┐      ┌─────────┐
    │  CALM   │      │  TREND  │      │ CRISIS  │
    │  ν=20   │      │  ν=6    │      │  ν=3    │
    └────┬────┘      └────┬────┘      └────┬────┘
         │                │                │
    ┌────┴────┐      ┌────┴────┐      ┌────┴────┐
    │Student-t│      │Student-t│      │Student-t│  ← Elastic absorption
    │ Spring  │      │ Spring  │      │ Spring  │
    └────┬────┘      └────┬────┘      └────┬────┘
         │                │                │
    ┌────┴────┐      ┌────┴────┐      ┌────┴────┐
    │  OCSN   │      │  OCSN   │      │  OCSN   │  ← Bump stop (K→0)
    │ Shield  │      │ Shield  │      │ Shield  │
    └────┬────┘      └────┬────┘      └────┬────┘
         │                │                │
    ┌────┴────┐      ┌────┴────┐      ┌────┴────┐
    │ Storvik │      │ Storvik │      │ Storvik │  ← Parameter learning
    │ PINNED  │      │ LEARNS  │      │ SLOW    │
    └────┬────┘      └────┬────┘      └────┬────┘
         │                │                │
    ┌────┴────┐      ┌────┴────┐      ┌────┴────┐
    │Swim Lane│      │Swim Lane│      │Swim Lane│  ← Position bounds
    │[-6,-4.5]│      │[-5,-2.5]│      │[-3,-0.5]│
    └────┬────┘      └────┬────┘      └────┬────┘
         │                │                │
    ┌────┴────┐      ┌────┴────┐      ┌────┴────┐
    │Adaptive │      │Adaptive │      │Adaptive │  ← Learning speed
    │DISABLED │      │TURBO    │      │STIFF    │
    │         │      │λ∈[.95,  │      │λ∈[.995, │
    │         │      │   .995] │      │  .9995] │
    └────┬────┘      └────┬────┘      └────┬────┘
         │                │                │
         └────────────────┼────────────────┘
                          ▼
                  ┌───────────────┐
                  │  IMM Weights  │
                  │ (Likelihood)  │
                  └───────────────┘
```

---

## Component Details

### 1. Student-t Distribution (The Spring)

**What it does:** Provides elastic absorption proportional to bump size.

**The math:**
- Gaussian tails decay exponentially: $e^{-x^2}$
- Student-t tails decay polynomially: $x^{-(\nu+1)}$

**Per-hypothesis tuning:**

| Model | ν | Behavior | Analogy |
|-------|---|----------|---------|
| Calm | 20 | Near-Gaussian, rejects fat tails | Stiff spring |
| Trend | 6 | Moderate tails, accepts some outliers | Medium spring |
| Crisis | 3 | Heavy tails, expects extremes | Soft spring |

**Effect on likelihood:**

| Move | Calm (ν=20) | Trend (ν=6) | Crisis (ν=3) |
|------|-------------|-------------|--------------|
| 3σ   | "Unusual"   | "Normal"    | "Tiny"       |
| 5σ   | "Impossible"| "Unlikely"  | "Expected"   |
| 10σ  | L ≈ 0       | L = low     | L = OK       |

---

### 2. OCSN (The Bump Stop)

**Outlier-Contaminated Scale Normal** — a mixture model:

```
P(y) = (1-π) × Normal(y | KSC) + π × Normal(y | 0, σ²_outlier)
```

Where π ≈ 5% and σ²_outlier = 150 (huge).

**Why Student-t alone isn't enough:**

| Move | Student-t λ | K reduction | State movement |
|------|-------------|-------------|----------------|
| 3σ   | ~0.5        | 2×          | Dampened |
| 5σ   | ~0.1        | 10×         | Heavily dampened |
| 10σ  | ~0.02       | 50×         | Still moves! |
| 10σ + OCSN | N/A   | K ≈ 0       | **Frozen** |

**Key insight:** Student-t controls LIKELIHOOD (model comparison). OCSN controls KALMAN GAIN (state protection). You need both.

---

### 3. Storvik Parameter Learning (The Adaptive Controller)

**What it learns:**

| Parameter | Suspension Analog | Effect |
|-----------|-------------------|--------|
| μ_vol | Ride height | Neutral position — baseline bumpiness |
| σ_vol | Spring stiffness | Responsiveness to new data |

**The feedback loop:**

```
Road bumps (observations)
    ↓
Spring absorbs (Student-t dampens)
    ↓
Bump stop catches extremes (OCSN protects)
    ↓
Controller measures ride quality (Storvik tracks sufficient stats)
    ↓
Adjusts height/stiffness (updates μ_vol, σ_vol)
    ↓
Better tuned for next bump
```

---

### 4. Swim Lanes (Position Bounds)

**The Problem:** Without bounds, models can drift toward each other ("mode collapse").

**The Solution:** Hard bounds that prevent encroachment.

```c
typedef struct {
    rbpf_real_t mu_vol_min;
    rbpf_real_t mu_vol_max;
    rbpf_real_t sigma_vol_min;
    rbpf_real_t sigma_vol_max;
    rbpf_real_t learning_rate_scale;
} MMPF_SwimLane;

static const MMPF_SwimLane SWIM_LANES[3] = {
    /* CALM: The Anchor */
    { -6.0, -4.5, 0.03, 0.15, 0.0 },   /* Pinned */
    
    /* TREND: The Scout */
    { -5.0, -2.5, 0.08, 0.35, 1.0 },   /* Full learning */
    
    /* CRISIS: Heavy Artillery */
    { -3.0, -0.5, 0.30, 1.20, 0.1 },   /* Slow learning */
};
```

**Why hard bounds beat separation constraints:**

Separation constraints (e.g., "Crisis.μ > Trend.μ + Gap") create **coupling**:

```
Tick 1:    Trend.μ = -4.0, Crisis.μ = -3.0  (Gap = 1.0) ✓

Tick 100:  Trend chases false breakout → Trend.μ = -3.2
           Separation constraint kicks in
           Crisis.μ pushed to -2.7 (into "Hyper-Crisis" territory)

Tick 150:  REAL CRASH arrives
           But Crisis is tuned for apocalypse, not standard crash
           Detection delayed
```

**Hard bounds preserve independence:** Trend can be wrong in its lane. Crisis sits in its bunker, unaffected. When the real crash hits, Crisis is exactly where it should be.

---

### 5. Adaptive Forgetting (Learning Speed Control)

**The Storvik forgetting factor λ controls effective sample size:**

$$N_{eff} = \frac{1}{1 - \lambda}$$

| λ | N_eff | Memory |
|---|-------|--------|
| 0.999 | 1000 | Very long |
| 0.995 | 200 | Medium |
| 0.95 | 20 | Short |

**Per-hypothesis configuration:**

| Model | Status | Signal Source | λ Bounds | Why |
|-------|--------|---------------|----------|-----|
| Calm | DISABLED | N/A | N/A | Anchor doesn't adapt |
| Trend | AGGRESSIVE | COMBINED | [0.95, 0.995] | Scout needs to snap to new regimes |
| Crisis | RESTRICTED | PREDICTIVE only | [0.995, 0.9995] | Long memory, outliers are its job |

**Critical insight for Crisis:**

Crisis uses `ADAPT_SIGNAL_PREDICTIVE_SURPRISE` — it **ignores outlier fraction**.

Why? Outliers are Crisis's JOB. High outlier fraction means Crisis is working correctly, absorbing the fat tails. It shouldn't interpret that as "I'm failing, forget faster."

---

## The Consensus Mechanism

**Models don't communicate — they compete through likelihoods.**

```
Observation: y = 5% move

┌─────────────────────────────────────────────────────────────────┐
│                    INDEPENDENT LIKELIHOOD                        │
├─────────────┬─────────────┬─────────────────────────────────────┤
│    CALM     │    TREND    │    CRISIS                           │
│   ν=20      │    ν=6      │    ν=3                              │
│   μ=-5.0    │    μ=-4.0   │    μ=-2.0                           │
├─────────────┼─────────────┼─────────────────────────────────────┤
│ "5% move?   │ "5% move?   │ "5% move?                           │
│  At my vol  │  At my vol  │  At my vol level                    │
│  that's 8σ! │  that's 4σ. │  that's 1.5σ.                       │
│  Impossible"│  Unlikely"  │  Expected!"                         │
├─────────────┼─────────────┼─────────────────────────────────────┤
│  L = 0.0001 │  L = 0.02   │  L = 0.15                           │
└─────────────┴─────────────┴─────────────────────────────────────┘
                            │
                            ▼
              ┌─────────────────────────────┐
              │     IMM WEIGHT UPDATE       │
              │                             │
              │  w'[k] ∝ w[k] × T[k] × L[k] │
              │                             │
              │  Prior × Transition × Lik   │
              └─────────────────────────────┘
                            │
                            ▼
              ┌─────────────────────────────┐
              │  Calm:   5% → 0.5%          │
              │  Trend: 30% → 15%           │
              │  Crisis: 65% → 84.5%        │
              │                             │
              │  DOMINANT: CRISIS           │
              └─────────────────────────────┘
```

**The same observation means different things to each model:**

| Observation | Calm sees | Trend sees | Crisis sees |
|-------------|-----------|------------|-------------|
| 0.5% move | Normal (1σ) | Small (0.5σ) | Tiny (0.2σ) |
| 2% move | Large (4σ) | Moderate (2σ) | Normal (0.8σ) |
| 5% move | Impossible (10σ) | Unlikely (4σ) | Expected (1.5σ) |

**Regime detection = Who explains data best.** The model that says "YES, this is my territory" most convincingly (highest likelihood) wins.

---

## The Over-Adaptation Problem

### The Failure Mode

Without proper safeguards, Storvik learning + Student-t absorption can prevent regime switches by "normalizing" what should be crisis events.

```
Tick 1-100:   Normal market, Trend tracking well
              Trend μ_vol = -4.0

Tick 101:     3σ move arrives
              Trend (ν=6): "Bit unusual, but within my tails"
              Trend Storvik: "Bumping μ_vol to -3.8"
              
Tick 102:     Another 3σ move
              Trend: "I expected higher vol, this fits"
              Trend Storvik: "μ_vol now -3.6"
              
Tick 103:     5σ move — REAL CRISIS STARTING
              Trend (adapted): "I've been tracking this rise"
              Crisis: "Finally!"
              
              Result: Crisis gains weight, but not decisively
              Regime switch is delayed
```

### The Solution: Swim Lanes + Adaptive Forgetting

1. **Swim lanes** prevent Trend from entering Crisis's territory
2. **ν is structural** — Trend's ν=6 can never match Crisis's ν=3 for fat tails
3. **Calm is pinned** — provides invariant floor reference
4. **Crisis learns slowly** — stays rally-tuned, doesn't over-adapt

---

## Configuration Summary

| Component | Calm | Trend | Crisis |
|-----------|------|-------|--------|
| **Student-t ν** | 20 (stiff) | 6 (medium) | 3 (soft) |
| **OCSN** | Enabled | Enabled | Enabled |
| **Swim Lane μ** | [-6.0, -4.5] | [-5.0, -2.5] | [-3.0, -0.5] |
| **Swim Lane σ** | [0.03, 0.15] | [0.08, 0.35] | [0.30, 1.20] |
| **Learning Rate Scale** | 0.0 (pinned) | 1.0 (full) | 0.1 (slow) |
| **Adaptive Forgetting** | DISABLED | COMBINED, aggressive | PREDICTIVE only, stiff |
| **λ bounds** | N/A | [0.95, 0.995] | [0.995, 0.9995] |
| **φ (persistence)** | 0.995 | 0.95 | 0.85 |

---

## The Three Cars

| Car | Hypothesis | Suspension | Learning | Home Terrain |
|-----|------------|------------|----------|--------------|
| **Luxury sedan** | Calm | Stiff (ν=20) | Pinned | Smooth highway |
| **Sports car** | Trend | Medium (ν=6) | Aggressive | Winding roads |
| **Rally car** | Crisis | Soft (ν=3) | Slow | Rough terrain, craters |

**Bayesian model comparison** picks which car's assessment of "current road conditions" is most believable at each tick.

---

## File Structure

```
mmpf/
├── mmpf_rocks.h          # Public API
├── mmpf_internal.h       # Internal helpers, swim lane config
├── mmpf_core.c           # Lifecycle, step, IMM mixing, Storvik
├── mmpf_api.c            # Accessors, control, shock, diagnostics
```

---

## Key Design Principles

1. **Independence over communication** — Models compete, don't coordinate
2. **Structural differentiation** — ν, φ, σ bounds are fixed, not learned
3. **Bounded adaptation** — Swim lanes prevent mode collapse
4. **Asymmetric learning** — Calm pinned, Trend aggressive, Crisis slow
5. **Layered defense** — Student-t (elastic) + OCSN (hard cutoff)
6. **Signal-appropriate forgetting** — Crisis ignores outliers (they're its job)

---

## Summary Table: What Each Component Prevents

| Component | Without It | With It |
|-----------|-----------|---------|
| Student-t | Violent state jumps on outliers | Smooth elastic absorption |
| OCSN | State corrupted by flash crashes | Hard protection cutoff |
| Storvik | Fixed params, can't adapt to regime | Learns road conditions |
| Swim Lanes | Models drift toward each other | Preserved separation |
| Adaptive Forgetting | Fixed learning speed | Turbo for Trend, stiff for Crisis |
| Weight Gate | Learn from data you don't own | Only dominant learns |
| Pinned Calm | No stable reference floor | Invariant anchor |
