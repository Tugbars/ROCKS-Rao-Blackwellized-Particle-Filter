# Self-Driving Transition Matrix Strategies

## Overview

In regime-switching stochastic volatility models, the transition matrix **A** governs how the hidden regime evolves:

```
P(r_t = j | r_{t-1} = i) = A[i][j]
```

A well-tuned transition matrix is critical for:
- **Regime detection accuracy** — wrong A causes lag or false alarms
- **Volatility estimation** — wrong A propagates through Kalman updates
- **Position sizing** — Kelly criterion depends on regime confidence

This document surveys four approaches to learning/adapting **A** online, ranging from simple heuristics to fully Bayesian nonparametrics.

---

## Strategy Comparison

| Strategy | Heuristics | Principled | Compute | When to Use |
|----------|------------|------------|---------|-------------|
| **Fixed Matrix** | 1 (stickiness) | ❌ | O(1) | Known dynamics, single asset |
| **Discounted Dirichlet (Events)** | 3 | 🟡 | O(R²) on events | Multi-asset, need adaptation |
| **Soft Dirichlet (ξ)** | 1 | ✅ | O(R²) per tick | Production, unknown dynamics |
| **Sticky-HDP** | 0-2 | ✅✅ | O(T·K²) | Research, structure discovery |

---

## 1. Fixed Transition Matrix

### Description

Hand-tuned transition probabilities based on domain knowledge or backtesting.

```c
rbpf_real_t trans[16] = {
    0.920f, 0.056f, 0.020f, 0.004f,  /* From R0 (calm) */
    0.032f, 0.920f, 0.036f, 0.012f,  /* From R1 (mild) */
    0.012f, 0.036f, 0.920f, 0.032f,  /* From R2 (elevated) */
    0.004f, 0.020f, 0.056f, 0.920f   /* From R3 (crisis) */
};
rbpf_ksc_build_transition_lut(rbpf, trans);
```

### Properties

| Aspect | Value |
|--------|-------|
| Tuning parameters | 1 (diagonal stickiness) |
| Adapts to data | ❌ No |
| Theoretical basis | None (empirical) |
| Compute cost | O(1) at init |

### Design Pattern

```
A[i][j] ∝ exp(-|μ_i - μ_j| / scale) × (1 + κ·I(i=j))
```

Where:
- `μ_i` = mean log-vol of regime i
- `scale` = controls geometry influence
- `κ` = stickiness bonus for self-transitions

### When to Use

✅ Single asset with stable dynamics  
✅ Backtested regime structure  
✅ Maximum speed (no per-tick updates)  
❌ Unknown or changing market dynamics  
❌ Multi-asset deployment  

### Implementation Status

**✅ Implemented** in `rbpf_ksc_build_transition_lut()`

---

## 2. Discounted Dirichlet with Event Updates

### Description

Bayesian learning of transition probabilities using Dirichlet priors, updated on discrete regime-change events detected by SPRT.

```c
/* On SPRT-confirmed transition from regime i to j */
if (sprt_regime != old_sprt_regime) {
    dirichlet_transition_update(&dt, old_regime, new_regime);
}
```

### The Model

Each row of **A** has a Dirichlet prior:

```
A[i,:] ~ Dirichlet(α[i,0], α[i,1], ..., α[i,R-1])
```

**Update rule (on events):**
```
α[i][j] ← γ·α[i][j] + I(transition i→j)
```

**Posterior mean:**
```
E[A[i][j]] = α[i][j] / Σ_k α[i][k]
```

### Properties

| Aspect | Value |
|--------|-------|
| Tuning parameters | 3 (stickiness, distance_scale, γ) |
| Adapts to data | ✅ Yes (on events) |
| Theoretical basis | Bayesian (Dirichlet-Multinomial) |
| Compute cost | O(R²) per event |

### Problems Identified

1. **Requires SPRT** — adds dependency and potential lag
2. **"What counts as transition?"** — heuristic decision
3. **Three interacting knobs** — hard to tune
4. **Choppy periods** — forces hard switches when uncertain

### Experimental Results

| Config | Slow Trend | Choppy | Sudden Crisis |
|--------|------------|--------|---------------|
| No Dirichlet | 56.5% | 44.6% | 69.6% |
| Dirichlet (stickiness=30) | 36.9% | 45.6% | 59.4% |
| Dirichlet (stickiness=10) | 31.1% | 31.7% | 56.8% |

**Verdict:** Hurt performance on synthetic data with known structure.

### When to Use

✅ Sparse, well-defined regime changes  
✅ When SPRT is already in the pipeline  
❌ Choppy markets with uncertain regimes  
❌ When you need principled single-knob control  

### Implementation Status

**✅ Implemented** in `rbpf_dirichlet_transition.h`

```c
dirichlet_transition_init_geometric(&dt, n_regimes, mu_vol, 30.0f, 1.0f, 0.999f);
dirichlet_transition_update(&dt, from_regime, to_regime);
```

---

## 3. Soft Dirichlet with ξ Updates (Recommended)

### Description

Bayesian learning using the **exact posterior transition responsibility** ξ_t(i,j) every tick, instead of waiting for discrete events.

```c
/* Every tick: soft update with Bayes-consistent responsibility */
soft_dirichlet_update(&dt, regime_probs, regime_liks, trans_matrix);
```

### The Math

**Joint transition posterior:**
```
ξ̃_t(i,j) = p_{t-1}(i) × A[i][j] × ℓ_t(j)
ξ_t(i,j) = ξ̃_t(i,j) / Σ_{i',j'} ξ̃_t(i',j')
```

Where:
- `p_{t-1}(i)` = regime probability at t-1
- `A[i][j]` = current transition matrix
- `ℓ_t(j)` = observation likelihood under regime j

**Update rule:**
```
α[i][j] ← γ·α[i][j] + κ·ξ_t(i,j)
```

**Row ESS capping:**
```
ESS_i = Σ_j α[i][j]
if (ESS_i > ESS_max):
    α[i,:] *= ESS_max / ESS_i
```

### Properties

| Aspect | Value |
|--------|-------|
| Tuning parameters | **1** (ESS_max) |
| Adapts to data | ✅ Yes (every tick) |
| Theoretical basis | ✅ Bayes-consistent |
| Compute cost | O(R²) per tick |

### Why This Eliminates Heuristics

| Problem | Event-Based | Soft ξ |
|---------|-------------|--------|
| "What is a transition?" | SPRT threshold | **Eliminated** |
| Choppy periods | Forced hard switch | Fractional updates |
| Tuning | 3 interacting knobs | **1 intuitive knob** |
| SPRT dependency | Required | **None** |

### Single-Knob Configuration

```c
/* ESS_max = effective memory in ticks */
dt->ess_max = 200.0f;  /* ← THE ONLY KNOB */
dt->gamma = 1.0f;      /* No decay (ESS cap handles it) */
dt->kappa = 1.0f;      /* Full ξ contribution */
```

**Interpretation:** "I want ~200 ticks of effective memory" → `ESS_max = 200`

### When to Use

✅ Production systems  
✅ Unknown or changing dynamics  
✅ Choppy markets with uncertain regimes  
✅ When you want principled single-knob control  
❌ When transitions are truly discrete and rare  

### Implementation Status

**⏳ Designed, not yet implemented**

```c
typedef struct {
    int n_regimes;
    float alpha[MAX_R][MAX_R];
    float prob[MAX_R][MAX_R];
    float gamma;
    float kappa;
    float ess_max;
    float prev_regime_probs[MAX_R];
} SoftDirichletTransition;

void soft_dirichlet_update(SoftDirichletTransition *dt,
                           const float *regime_probs,
                           const float *regime_liks,
                           const float *trans_matrix);
```

---

## 4. Sticky HDP-HMM with Beam Sampling

### Description

Nonparametric Bayesian model that learns:
- Number of regimes (K) from data
- Transition structure via hierarchical Dirichlet Process
- Stickiness (κ) can be learned or fixed

```c
StickyHDP *hdp = sticky_hdp_create(32, 1000);
sticky_hdp_set_stickiness(hdp, 50.0);

for (int t = 0; t < T; t++) {
    sticky_hdp_observe(hdp, y[t]);
    if (t % 100 == 0) {
        sticky_hdp_beam_sweep(hdp, 3);
    }
}
```

### The Model

```
β ~ GEM(γ)                              /* Global state distribution */
π_k ~ DP(α + κ, (α·β + κ·δ_k)/(α + κ)) /* Transition from state k */
θ_k ~ H                                 /* Emission parameters */
s_t | s_{t-1} ~ π_{s_{t-1}}            /* State sequence */
y_t | s_t ~ F(θ_{s_t})                 /* Observations */
```

### Beam Sampling

Introduces auxiliary "slice" variables to limit active states:

```
u_t | s_{t-1}, s_t ~ Uniform(0, π_{s_{t-1}, s_t})
Active states: A_t = {k : π_{s_{t-1}, k} > u_t}
```

Typically |A_t| ≈ 3-8, making inference tractable.

### Properties

| Aspect | Value |
|--------|-------|
| Tuning parameters | 0-2 (γ, κ can be learned) |
| Learns # regimes | ✅ Yes |
| Theoretical basis | ✅✅ Full Bayesian |
| Compute cost | O(T·K²) per sweep (~1ms) |

### MKL Acceleration Points

| Operation | MKL Function | Speedup |
|-----------|--------------|---------|
| Log-sum-exp | `vdExp` + `cblas_dasum` | 3-5× |
| Forward filter | `cblas_dgemv` | 2-4× |
| Stick-breaking | `vdRngBeta` | 2-3× |
| Slice sampling | `vdRngUniform` | 2× |

### When to Use

✅ Research / regime discovery  
✅ Unknown number of regimes  
✅ Learning stickiness from data  
✅ Offline analysis  
⚠️ Online use (run every N ticks)  
❌ Ultra-low latency HFT (<100μs)  

### RBPF Integration

```c
/* Export learned structure to RBPF */
double trans[16], mu_vol[4], sigma_vol[4];
sticky_hdp_export_to_rbpf(hdp, 4, trans, mu_vol, sigma_vol, NULL);
rbpf_ksc_build_transition_lut(rbpf, trans);

/* Import RBPF estimates to warm-start HDP */
sticky_hdp_import_from_rbpf(hdp, 4, regime_seq, T, mu_vol, sigma_vol);
```

### Implementation Status

**✅ Implemented** in `sticky_hdp_beam.h/c`

---

## Decision Tree

```
                    ┌─────────────────────────────────────┐
                    │ Do you know the regime structure?   │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
                   YES                            NO
                    │                             │
        ┌───────────┴───────────┐     ┌──────────┴──────────┐
        │ Is it stable over     │     │ Do you need online   │
        │ your trading horizon? │     │ adaptation?          │
        └───────────┬───────────┘     └──────────┬──────────┘
                    │                             │
         ┌──────────┴──────────┐       ┌─────────┴─────────┐
         ▼                     ▼       ▼                   ▼
        YES                    NO     YES                  NO
         │                     │       │                   │
         ▼                     │       ▼                   ▼
   ┌───────────┐               │  ┌─────────────┐    ┌───────────┐
   │  FIXED    │               │  │ SOFT ξ      │    │ STICKY    │
   │  MATRIX   │               │  │ DIRICHLET   │    │ HDP       │
   └───────────┘               │  └─────────────┘    └───────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ How often do true   │
                    │ regime changes      │
                    │ occur?              │
                    └──────────┬──────────┘
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
                 RARELY              FREQUENTLY
                    │                     │
                    ▼                     ▼
            ┌───────────────┐     ┌─────────────┐
            │ EVENT-BASED   │     │ SOFT ξ      │
            │ DIRICHLET     │     │ DIRICHLET   │
            └───────────────┘     └─────────────┘
```

---

## Recommended Strategy

For **production trading systems** with unknown or changing dynamics:

### Primary: Soft Dirichlet with ξ Updates

```c
SoftDirichletTransition dt;
soft_dirichlet_init(&dt, n_regimes, 200.0f);  /* ESS_max = 200 */

/* In hot loop */
soft_dirichlet_update(&dt, regime_probs, regime_liks, trans_matrix);
rebuild_transition_lut_from_soft_dirichlet(rbpf, &dt);
```

**Why:**
- One intuitive knob (ESS_max)
- Bayes-consistent (no heuristics)
- O(R²) per tick (~16 ops for R=4)
- Handles choppy periods gracefully

### Secondary: Sticky-HDP for Discovery

```c
/* Offline: discover regime structure */
StickyHDP *hdp = sticky_hdp_create(32, 10000);
sticky_hdp_set_observations(hdp, historical_y, T);
sticky_hdp_beam_sweep(hdp, 100);  /* MCMC */

/* Export to production */
sticky_hdp_export_to_rbpf(hdp, 4, trans, mu_vol, sigma_vol, NULL);
```

**Why:**
- Learns K from data
- Discovers transition structure
- Provides principled initialization for Soft Dirichlet

---

## Summary Table

| Strategy | File | Status | Knobs | Use Case |
|----------|------|--------|-------|----------|
| Fixed Matrix | `rbpf_ksc.c` | ✅ Done | 1 | Stable dynamics |
| Event Dirichlet | `rbpf_dirichlet_transition.h` | ✅ Done | 3 | Rare transitions |
| **Soft ξ Dirichlet** | `soft_dirichlet_transition.h` | ⏳ TODO | **1** | **Production** |
| Sticky-HDP | `sticky_hdp_beam.h/c` | ✅ Done | 0-2 | Discovery |

---

## Next Steps

1. **Implement Soft Dirichlet** — ~100 lines, high impact
2. **Test on synthetic data** — compare to fixed matrix baseline
3. **Test on real data** — where adaptive should shine
4. **Periodic HDP refresh** — run offline, export to Soft Dirichlet

---

## References

- Fox, E. B., et al. (2011). "A Sticky HDP-HMM with Application to Speaker Diarization"
- Van Gael, J., et al. (2008). "Beam Sampling for the Infinite Hidden Markov Model"
- Teh, Y. W., et al. (2006). "Hierarchical Dirichlet Processes"
- Murphy, K. P. (2012). "Machine Learning: A Probabilistic Perspective" — Ch. 17
