# MMPF-ROCKS: The Adaptive Suspension Analogy

## The Problem We're Solving

A volatility filter needs to handle three scenarios gracefully:

1. **Normal bumps** (1-2σ) — Routine market noise
2. **Potholes** (3-5σ) — Rare but regular outliers  
3. **Craters** (10σ+) — Flash crashes, black swans

A naive Gaussian filter treats all bumps as signal, causing violent state estimates that destroy Kelly sizing.

---

## The Suspension System

| Component | Filter Element | Mechanism | Handles |
|-----------|----------------|-----------|---------|
| **Rigid steel rod** | Gaussian | None | Nothing — transmits all shock |
| **Spring** | Student-t (ν) | Elastic absorption | Potholes (3-5σ) |
| **Bump stop** | OCSN | Hard cutoff | Craters (10σ+) |
| **Adaptive controller** | Storvik | Learns road type | Tunes spring stiffness & ride height |

---

## Component Details

### 1. The Spring — Student-t (ν)

**What it does:** Provides elastic absorption proportional to bump size.

**The math:**
- Gaussian tails decay exponentially: $e^{-x^2}$
- Student-t tails decay polynomially: $x^{-(\nu+1)}$

**The effect:**

| Move | Gaussian likelihood | Student-t (ν=5) likelihood |
|------|---------------------|----------------------------|
| 3σ   | 0.003               | 0.02                       |
| 5σ   | 0.0000006           | 0.005                      |
| 10σ  | ~0                  | 0.0008                     |

**Analogy:** A 5σ move compresses the spring significantly, but smoothly. The chassis (state estimate) moves, but dampened — not violently.

**Per-hypothesis tuning:**
- Calm (ν=20): Stiff spring — "bumps are rare, track precisely"
- Trend (ν=6): Medium spring — "some bumps expected"
- Crisis (ν=3): Soft spring — "expecting craters constantly"

---

### 2. The Bump Stop — OCSN

**What it does:** Hard limit that disconnects wheel from chassis during catastrophic events.

**The math:**
```
P(y) = (1-π) × Normal(y | KSC mixture) + π × Normal(y | 0, σ²_outlier)
```
Where π ≈ 5% and σ²_outlier = 150 (huge).

**The effect:**

When the outlier component "wins" the mixture:
- Effective observation variance → huge  
- Kalman gain K → 0
- State doesn't update

**Analogy:** The spring can only compress so far. The bump stop is the hard rubber block that prevents bottoming out. When you hit a meteor crater, the bump stop says "NOPE" and the chassis stays level.

**Why Student-t isn't enough:**

Even with ν=3, a 10σ observation has *some* likelihood, so the Kalman update still moves the state. The λ scaling helps but doesn't provide a hard cutoff.

| Move | Student-t λ | K reduction | State movement |
|------|-------------|-------------|----------------|
| 3σ   | ~0.5        | 2×          | Dampened |
| 5σ   | ~0.1        | 10×         | Heavily dampened |
| 10σ  | ~0.02       | 50×         | Still moves! |
| 10σ + OCSN | N/A   | K ≈ 0       | Frozen |

---

### 3. The Adaptive Controller — Storvik

**What it does:** Learns road conditions and adjusts the suspension.

**Parameters learned:**

| Parameter | Suspension analog | Effect |
|-----------|-------------------|--------|
| μ_vol     | Ride height       | Neutral position — "expect this baseline bumpiness" |
| σ_vol     | Spring stiffness  | Responsiveness — "how quickly should I adapt?" |

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

## The Three Cars (MMPF Hypotheses)

| Car | Hypothesis | Suspension | Learning rate | Home terrain |
|-----|------------|------------|---------------|--------------|
| Luxury sedan | Calm | Stiff (ν=20) | None (pinned) | Smooth highway |
| Sports car | Trend | Medium (ν=6) | Medium (λ=0.997) | Winding roads |
| Rally car | Crisis | Soft (ν=3) | Fast (λ=0.990) | Rough terrain |

**Bayesian model comparison** picks which car's assessment of "current road conditions" is most believable at each tick.

---

## The Complete Picture

```
                    ┌─────────────────┐
                    │   Observation   │
                    │    (y = bump)   │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
         ┌────────┐    ┌────────┐    ┌────────┐
         │  Calm  │    │ Trend  │    │ Crisis │
         │  ν=20  │    │  ν=6   │    │  ν=3   │
         └───┬────┘    └───┬────┘    └───┬────┘
             │             │             │
             ▼             ▼             ▼
        ┌─────────┐   ┌─────────┐   ┌─────────┐
        │Student-t│   │Student-t│   │Student-t│
        │ Spring  │   │ Spring  │   │ Spring  │
        └────┬────┘   └────┬────┘   └────┬────┘
             │             │             │
             ▼             ▼             ▼
        ┌─────────┐   ┌─────────┐   ┌─────────┐
        │  OCSN   │   │  OCSN   │   │  OCSN   │
        │Bump Stop│   │Bump Stop│   │Bump Stop│
        └────┬────┘   └────┬────┘   └────┬────┘
             │             │             │
             ▼             ▼             ▼
        ┌─────────┐   ┌─────────┐   ┌─────────┐
        │ Storvik │   │ Storvik │   │ Storvik │
        │(PINNED) │   │(learns) │   │(learns) │
        └────┬────┘   └────┬────┘   └────┬────┘
             │             │             │
             └──────────┬──────────────┘
                        │
                        ▼
                ┌───────────────┐
                │  IMM Weights  │
                │ (which car?)  │
                └───────────────┘
```

---

## Summary

| Component | Without it | With it |
|-----------|-----------|---------|
| Student-t | Violent state jumps on outliers | Smooth elastic absorption |
| OCSN | State corrupted by flash crashes | Hard protection cutoff |
| Storvik | Fixed params, can't adapt to regime | Learns road conditions |
| Gating | Models drift toward each other | Preserves specialization |
| Pinned Calm | No stable reference floor | Invariant anchor |

The system provides:
- **Elasticity** (Student-t) — smooth response to rare-but-regular outliers
- **Safety** (OCSN) — hard cutoff against catastrophic events  
- **Adaptation** (Storvik) — learns the current road type
- **Specialization** (Gating) — preserves each car's home-terrain tuning
