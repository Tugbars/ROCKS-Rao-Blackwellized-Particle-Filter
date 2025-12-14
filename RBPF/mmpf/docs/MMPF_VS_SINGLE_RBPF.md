# MMPF vs Single RBPF: The PWM Effect and Architectural Tradeoffs

## The Observation

Single RBPF tracks volatility more smoothly than MMPF, despite MMPF being the "more sophisticated" system.

## The PWM (Pulse Width Modulation) Analogy

### Single RBPF: The "Blender"

- Doesn't decide on a regime - particles are distributed across regime space
- When true vol is between "Calm" and "Medium", particles naturally split 50/50
- Like PWM in an LED dimmer: rapid flickering creates perfect "50% brightness"
- **Result:** Smooth, accurate vol tracking. Noisy, flickering regime signal.

### MMPF/IMM: The "Decider"

- Forces particles into 3 discrete buckets with fixed μ_vol anchors
- Sticky transitions (98% stay) prevent flickering, force commitment
- When true vol falls in the "gap" between models, IMM struggles
- **Result:** Clean, stable regime signal. Steppy, less accurate vol tracking.

## The Discretization Tax

```
True vol = Calm + 0.7 × (Trend - Calm)

Single RBPF:  Particles naturally distribute to match
              → Perfect interpolation

MMPF:         Must express as weights [0.3, 0.7, 0.0]
              But each model anchored at fixed μ_vol
              → Interpolating between discrete points, not continuous surface
```

## Why MMPF Exists (The Warmth Advantage)

Single RBPF at crisis onset:
```
Tick 0:   Particles have calm dynamics (φ=0.95, σ_η=0.15)
Tick 50:  Still adapting...
Tick 100: Finally learned crisis dynamics
```

MMPF at crisis onset (with BOCPD):
```
Tick 0:   BOCPD fires → weights go 33/33/33
          Crisis-RBPF already has correct physics (φ=0.85, σ_η=0.50)
          Crisis-RBPF: "This is exactly what I expected"
Tick 1:   Crisis wins. Done.
```

**The warmth is in the dynamics (φ, σ_η, μ_vol), not the state (particles).**

Crisis-RBPF has been running with crisis-appropriate physics the whole time. When a shock hits, that worldview immediately produces the best likelihood.

## The Problem: MMPF Without BOCPD

Without BOCPD to snap the weights:
- Sticky transitions mean ~100 ticks to switch regimes
- During transition, weights are confused (33/33/33)
- Weighted averaging dilutes the best estimate
- The "warmth" advantage is wasted

**MMPF without BOCPD < Single RBPF < MMPF with BOCPD**

## The Two-Headed Solution

Since BOCPD handles fast detection, MMPF can optimize for tracking:

| Output | Method | Purpose |
|--------|--------|---------|
| **Detection** | BOCPD delta signal | Fast "something broke!" alert |
| **Regime** | Sticky IMM weights | Stable regime classification |
| **Tracking** | Softened likelihood blend | Smooth σ estimate |

## Fixes for Better Tracking

### 1. Close the Gaps (Overlapping Distributions)

Current offsets might be too far apart:
```
Calm:   μ_base - 1.0
Trend:  μ_base
Crisis: μ_base + 1.5
```

If models are "too confident" (low σ_η), they don't overlap. Trend should be the "bridge" with wider variance.

### 2. Relax Stickiness (With BOCPD as backup)

Current transition matrix:
```
[0.98  0.01  0.01]
[0.01  0.98  0.01]
[0.01  0.01  0.98]
```

Try 0.90-0.95 on diagonal. Allows more blending for smooth tracking. BOCPD handles the fast detection anyway.

### 3. Softened Tracking Output

Instead of:
```c
σ = w_calm × σ_calm + w_trend × σ_trend + w_crisis × σ_crisis
```

Use raw likelihoods for an "instantaneous blend" that ignores sticky transitions just for the σ estimate.

## The Tradeoff Summary

| Aspect | Single RBPF | MMPF (no BOCPD) | MMPF (with BOCPD) |
|--------|-------------|-----------------|-------------------|
| Vol tracking | ✓ Smooth | ✗ Steppy | ✓ Smooth (relaxed) |
| Regime signal | ✗ Flickery | ✓ Stable | ✓ Stable |
| Crisis response | ✗ Slow (~100 ticks) | ✗ Slow (~100 ticks) | ✓ Fast (~1-2 ticks) |
| Adaptation | Continuous | Discrete buckets | Discrete + instant switch |

## Bottom Line

> Single RBPF is "over-fitting" regime switching (flickering) to achieve perfect tracking.
> 
> MMPF is "under-fitting" tracking to achieve robust detection.
>
> BOCPD lets you have both: fast detection externally, smooth tracking internally.

The sports car (MMPF) needs its turbocharger (BOCPD) connected.
