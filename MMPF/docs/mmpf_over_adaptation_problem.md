# The Over-Adaptation Problem

## The Failure Mode

**Question:** Can Storvik learning + Student-t absorption prevent regime switches by "normalizing" what should be crisis events?

**Answer:** Yes, this is a real concern.

---

## The Scenario

### Setup
- Trend hypothesis at 30% weight during moderately volatile period
- Trend is learning (weight > 10%, passes gate)
- Trend has ν=6 (moderate fat tails)

### The Slow Boil

```
Tick 1-100:   Normal market, Trend tracking well
              Trend μ_vol = -4.0 (≈1.8% vol)

Tick 101:     3σ move arrives
              Trend (ν=6): "Bit unusual, but within my tails"
              Trend Storvik: "Bumping μ_vol to -3.8"
              Trend likelihood: decent
              
Tick 102:     Another 3σ move
              Trend: "I expected higher vol, this fits"
              Trend Storvik: "μ_vol now -3.6"
              Trend likelihood: still good
              
Tick 103:     4σ move
              Trend: "My updated μ_vol handles this"
              Trend Storvik: "μ_vol now -3.4"
              Crisis: "Hey, this looks like my territory!"
              But Trend is ALSO explaining it...
              
Tick 104:     5σ move — REAL CRISIS STARTING
              Trend (adapted): "I've been tracking this rise"
              Crisis: "Finally!"
              
              Result: Crisis gains weight, but not decisively
              Regime switch is delayed or muted
```

### The Problem

Trend's Storvik learning "chased" the volatility increase, allowing Trend to explain observations that should have triggered a Crisis switch.

**Trend stole Crisis's thunder by adapting too aggressively.**

---

## Why ν Alone Doesn't Fully Solve This

### What ν Does Protect

The degrees of freedom (ν) is **not learned** — it's structural. So even if Trend raises its μ_vol:

| Model | ν | 10σ move likelihood |
|-------|---|---------------------|
| Calm (pinned) | 20 | ~0 (thin tails reject completely) |
| Trend (adapted μ_vol) | 6 | Low (moderate tails) |
| Crisis | 3 | OK (fat tails accept) |

A truly extreme observation will still favor Crisis because Trend's ν=6 can't match Crisis's ν=3 tail thickness.

### What ν Doesn't Protect

The **transition zone** between regimes. For 3-5σ moves:

- Trend's ν=6 gives reasonable likelihood
- Trend's adapted μ_vol makes the observation look "expected"
- Crisis's advantage is reduced

**Result:** Blurred boundary between Trend and Crisis. Slower, less decisive regime switches.

---

## The Specific Vulnerabilities

### 1. Trend Encroaching on Crisis Territory

During a prolonged high-vol period where Trend keeps >10% weight:

```
Initial:        Trend μ_vol = -4.0    Crisis μ_vol = -3.0
                       ↓ learning
After 100 ticks: Trend μ_vol = -3.2    Crisis μ_vol = -3.0
                       
                 Trend has drifted INTO Crisis's territory
```

### 2. Crisis Softening During Extended Calm

If Crisis somehow maintains >10% weight during calm (maybe due to min_mixing_prob):

```
Initial:        Crisis μ_vol = -3.0 (tuned for craters)
                       ↓ learning from calm data
After drift:    Crisis μ_vol = -3.8 (now expects smaller bumps)

Next flash crash: Crisis is no longer "rally-tuned"
```

### 3. The "Goldilocks" Trap

Trend can become a "jack of all trades":
- Learns high μ_vol during volatile periods
- Has moderate ν=6 that handles some fat tails
- Steals observations from both Calm and Crisis

---

## Potential Solutions

### Option 1: Pin Crisis Too

Only Trend learns. Calm and Crisis are fixed floor/ceiling.

```c
int is_pinned = (k == MMPF_CALM || k == MMPF_CRISIS);

if (passes_weight_gate && !is_pinned) {
    mmpf_update_storvik_for_hypothesis(...);
}
```

**Pros:**
- Crisis always "rally-tuned" for craters
- Clear, fixed regime boundaries
- Simplest solution

**Cons:**
- Crisis can't adapt to different types of crises
- μ_vol floor/ceiling are arbitrary constants

---

### Option 2: μ_vol Bounds Per Hypothesis

Each hypothesis has a bounded operating range:

```c
/* In Storvik update or sync */
switch (k) {
    case MMPF_CALM:
        /* μ_vol ∈ [-6.0, -4.5] — must stay low */
        if (mu_vol > -4.5) mu_vol = -4.5;
        if (mu_vol < -6.0) mu_vol = -6.0;
        break;
        
    case MMPF_TREND:
        /* μ_vol ∈ [-5.0, -2.5] — free to roam middle */
        if (mu_vol > -2.5) mu_vol = -2.5;
        if (mu_vol < -5.0) mu_vol = -5.0;
        break;
        
    case MMPF_CRISIS:
        /* μ_vol ∈ [-3.5, -1.5] — must stay high */
        if (mu_vol > -1.5) mu_vol = -1.5;
        if (mu_vol < -3.5) mu_vol = -3.5;
        break;
}
```

**Pros:**
- Each hypothesis can adapt within its territory
- Prevents Trend from encroaching on Crisis
- Clear structural separation

**Cons:**
- More hyperparameters (6 bounds)
- Bounds are somewhat arbitrary
- What if true vol is outside all bounds?

---

### Option 3: Asymmetric Learning Rates

Trend learns slowly upward, quickly downward:

```c
if (k == MMPF_TREND) {
    if (new_mu_vol > old_mu_vol) {
        /* Slow to increase — don't chase Crisis */
        lambda = 0.999;  /* N_eff ≈ 1000 */
    } else {
        /* Quick to decrease — return to Trend territory */
        lambda = 0.990;  /* N_eff ≈ 100 */
    }
}
```

**Pros:**
- Trend can't chase Crisis quickly
- Trend naturally "snaps back" to its home range

**Cons:**
- Asymmetry is a hack
- Doesn't address Crisis softening during calm

---

### Option 4: Weight-Gated Learning Rate

Stronger weight = faster learning. Weak weight = frozen.

Currently binary: learn if w > 10%, else frozen.

Could be continuous:

```c
/* Learning rate scales with weight */
effective_lambda = base_lambda + (1 - base_lambda) * (1 - w_k);

/* w_k = 0.50 → fast learning (confident) */
/* w_k = 0.15 → slow learning (uncertain) */
/* w_k = 0.08 → frozen (below gate) */
```

**Pros:**
- Gradual transition, not binary
- Dominant model adapts fastest
- Losing models naturally drift slower

**Cons:**
- More complexity
- Losing models still drift (just slower)

---

### Option 5: Separation Constraint

Enforce minimum μ_vol gap between hypotheses:

```c
/* After all Storvik updates */
const rbpf_real_t min_gap = 0.5;

/* Sort by μ_vol: Calm < Trend < Crisis */
if (trend_mu - calm_mu < min_gap) {
    trend_mu = calm_mu + min_gap;
}
if (crisis_mu - trend_mu < min_gap) {
    crisis_mu = trend_mu + min_gap;
}
```

**Pros:**
- Guarantees regime separation
- Hypotheses can still adapt, just not collide

**Cons:**
- Which one gets pushed?
- Could cause oscillations

---

## Recommendation

**Start with Option 1 (Pin Crisis) + Option 2 (Bounds on Trend):**

```c
/* Calm: PINNED — the floor/anchor */
/* Crisis: PINNED — the ceiling, rally-tuned */
/* Trend: LEARNS within bounds [-5.0, -2.5] */

int is_pinned = (k == MMPF_CALM || k == MMPF_CRISIS);

if (passes_weight_gate && !is_pinned) {
    mmpf_update_storvik_for_hypothesis(mmpf, k, resampled);
    
    /* Clamp Trend to its territory */
    if (mmpf->gated_dynamics[k].mu_vol > -2.5) 
        mmpf->gated_dynamics[k].mu_vol = -2.5;
    if (mmpf->gated_dynamics[k].mu_vol < -5.0) 
        mmpf->gated_dynamics[k].mu_vol = -5.0;
}
```

**Rationale:**

1. **Calm pinned** — Provides invariant "low vol" reference
2. **Crisis pinned** — Always rally-tuned, ready for craters
3. **Trend learns** — The "working hypothesis" that tracks regime changes
4. **Trend bounded** — Can't encroach on Calm or Crisis territory

This gives you:
- Clear regime boundaries (physics)
- One adaptive model (Trend) to track gradual changes
- Two fixed anchors (Calm/Crisis) for extreme conditions
- ν still provides tail-thickness discrimination

---

## Open Questions

1. **What if true vol exceeds Crisis's fixed μ_vol?**
   - Crisis's fat tails (ν=3) should still give reasonable likelihood
   - But state estimate may lag reality

2. **Should Crisis learn a little, with tight bounds?**
   - e.g., μ_vol ∈ [-3.5, -2.5] with slow learning (λ=0.999)
   - Gives some adaptability without losing rally-tuning

3. **Is the 10% weight gate the right threshold?**
   - Lower (5%): More hypotheses learn, more drift risk
   - Higher (20%): Only dominant learns, maybe too restrictive

4. **Should σ_vol also be bounded/pinned?**
   - Currently only μ_vol discussed
   - σ_vol affects responsiveness, not level
