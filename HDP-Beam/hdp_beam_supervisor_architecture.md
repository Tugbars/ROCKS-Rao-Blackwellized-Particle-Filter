# HDP-BEAM Supervisor Architecture

## Executive Summary

This document describes the hybrid architecture for real-time regime detection in high-frequency trading, combining:

- **HDP-BEAM** (Sticky HDP-HMM with Beam Sampling): Offline structure discovery
- **Discounted Dirichlet RBPF**: Online tick-by-tick filtering

The key insight: the transition matrix is not a static property of the asset, but a **latent time-varying process itself**.

---

## 1. The Fundamental Friction

### Why Not Pure HDP-BEAM in the Hot Path?

| Approach | Latency | Knowledge |
|----------|---------|-----------|
| **HDP-BEAM** | ~200μs/sweep, needs 10+ sweeps | Full posterior over structure |
| **RBPF** | ~5μs/tick | Point estimates, fixed structure |

Swapping RBPF for pure HDP-BEAM would be like replacing a racing engine with a scientific laboratory. You gain "knowledge" at the cost of **responsiveness**.

### The Solution: Supervisor Hybrid

```
┌─────────────────────────────────────────────────────────────────┐
│                    HDP-BEAM (Architect)                         │
│              Low-frequency structure discovery                   │
│                                                                  │
│  Outputs:                                                        │
│    • Number of regimes K                                         │
│    • Transition mask (impossible transitions)                    │
│    • Stickiness priors per regime                                │
│    • μ-ordered emission parameters                               │
└──────────────────────────┬──────────────────────────────────────┘
                           │ Reset priors (every ~5000 ticks)
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              Discounted Dirichlet RBPF (Builder)                │
│                    Per-tick: ~5μs                                │
│                                                                  │
│  Responsibilities:                                               │
│    • Soft ξ_t(i,j) updates from particle mass                   │
│    • Adaptive γ_t driven by regime entropy                       │
│    • Real-time regime probability estimates                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Soft Transition Updates

### The Problem with Hard Updates

Current approach: Update Dirichlet counts only on "confirmed" SPRT flips.

```c
/* Hard update - too slow for chaotic markets */
if (sprt_confirmed_transition) {
    alpha[from][to] += 1.0;
}
```

**Issue**: Markets don't wait for statistical confirmation. By the time SPRT confirms, the regime may have changed again.

### The Solution: Learn from Ambiguity

In a Rao-Blackwellized context, the probability of transition i→j at time t is:

$$\xi_{t}(i,j) = P(r_{t-1}=i, r_t=j \mid y_{1:t})$$

This can be approximated from particle weights:

$$\xi_{t}^{(n)}(i,j) = w_{t-1}^{(n)} \cdot A_{ij} \cdot p(y_t \mid r_t=j)$$

### Implementation

```c
/**
 * Soft Dirichlet update with fractional mass
 * 
 * Instead of waiting for confirmed transitions, accumulate
 * the probability mass of transitions across all particles.
 */
void dirichlet_transition_update_soft(DirichletTransition *dt,
                                       int from, int to,
                                       float mass)
{
    if (from < 0 || from >= dt->n_regimes) return;
    if (to < 0 || to >= dt->n_regimes) return;
    if (mass <= 0.0f) return;
    
    /* Apply forgetting to entire row */
    float *row = &dt->alpha[from * dt->n_regimes];
    for (int j = 0; j < dt->n_regimes; j++) {
        row[j] = dt->alpha_prior[from * dt->n_regimes + j] +
                 dt->gamma * (row[j] - dt->alpha_prior[from * dt->n_regimes + j]);
    }
    
    /* Add soft mass */
    row[to] += mass;
    
    /* Recompute transition probabilities */
    dirichlet_recompute_row(dt, from);
}
```

### Per-Tick Integration

```c
/* After RBPF resampling step */
void accumulate_soft_transitions(RBPF *rbpf, DirichletTransition *dt)
{
    for (int n = 0; n < rbpf->n_particles; n++) {
        int i = rbpf->prev_regime[n];
        int j = rbpf->curr_regime[n];
        float xi_ij = rbpf->weights[n];  /* Normalized weight */
        
        dirichlet_transition_update_soft(dt, i, j, xi_ij);
    }
}
```

---

## 3. Adaptive Forgetting (γ_t)

### The Insight

The forgetting factor γ should not be constant. When the market is confused (high regime entropy), old transition probabilities are **liabilities**, not assets.

### Entropy-Driven γ

Define regime entropy:

$$H_t = -\sum_{k} P(r_t = k) \log P(r_t = k)$$

Scale forgetting factor:

$$\gamma_t = \gamma_{\text{base}} - \eta \cdot H_t$$

| Market State | Entropy | γ | Effect |
|--------------|---------|---|--------|
| **Stable** (one regime dominates) | Low (~0.3) | 0.999 | Sticky, long memory |
| **Chaotic** (uniform uncertainty) | High (~1.4) | 0.95 | Fast adaptation |

### Implementation

```c
/**
 * Update forgetting factor based on regime entropy
 * 
 * @param dt        Dirichlet transition struct
 * @param entropy   Current regime entropy H_t
 * @param eta       Sensitivity (0.02-0.05 typical)
 */
void dirichlet_adapt_forgetting(DirichletTransition *dt,
                                 float entropy,
                                 float eta)
{
    /* Base gamma for stable markets */
    const float gamma_base = 0.999f;
    const float gamma_min = 0.95f;
    const float gamma_max = 0.9995f;
    
    /* Reduce gamma when entropy is high */
    float gamma_t = gamma_base - eta * entropy;
    
    /* Clamp to safe range */
    dt->gamma = fmaxf(gamma_min, fminf(gamma_max, gamma_t));
}
```

### Integration with RBPF Output

```c
/* In main tick loop */
void process_tick(RBPF *rbpf, DirichletTransition *dt, float y)
{
    /* Run RBPF step */
    RBPF_Output out;
    rbpf_step(rbpf, y, &out);
    
    /* Adapt forgetting based on uncertainty */
    dirichlet_adapt_forgetting(dt, out.regime_entropy, 0.03f);
    
    /* Accumulate soft transitions */
    accumulate_soft_transitions(rbpf, dt);
}
```

---

## 4. HDP-BEAM as the "Refinery"

### The Complementary Views

| Component | Temporal View | Strength |
|-----------|---------------|----------|
| **RBPF** | Keyhole (one tick) | Speed, responsiveness |
| **HDP-BEAM** | Panoramic (2000 ticks) | Structure, smoothing |

### What HDP-BEAM Provides

1. **Number of Regimes K**
   - Discovers if market has 3, 4, or 5 distinct volatility levels
   - RBPF doesn't ask "how many?" - it's told

2. **Transition Mask**
   - "S0→S4 is physically impossible" (can't jump from calm to crisis)
   - Encodes market microstructure constraints

3. **Stickiness Priors**
   - Per-regime self-transition bias
   - Crisis regimes are stickier than trend regimes

4. **Emission Parameters**
   - μ-ordered: Regime 0 = calm, Regime K-1 = crisis
   - Warm-starts RBPF after structural changes

### Synchronization Protocol

```c
/**
 * Periodic HDP-BEAM refinement (every ~5000 ticks or 1 hour)
 */
void hdp_refine_rbpf(StickyHDP *hdp, RBPF *rbpf, DirichletTransition *dt)
{
    /* 1. Run HDP-BEAM inference */
    sticky_hdp_blocked_gibbs(hdp, 3, 50);  /* 3 sweeps, block=50 */
    
    /* 2. Reorder states by μ (prevent label switching) */
    sticky_hdp_reorder_by_mu(hdp);
    
    /* 3. Export structure to RBPF */
    int K = hdp->K;
    double trans[16], mu_vol[4], sigma_vol[4], theta[4];
    sticky_hdp_export_to_rbpf(hdp, K, trans, mu_vol, sigma_vol, theta);
    
    /* 4. Reset Dirichlet priors */
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < K; j++) {
            float hdp_trans = (float)trans[i * K + j];
            
            /* Convert HDP posterior to Dirichlet prior */
            float alpha_ij = 1.0f + hdp_trans * 10.0f;  /* Scale factor */
            dt->alpha_prior[i * K + j] = alpha_ij;
            dt->alpha[i * K + j] = alpha_ij;
        }
    }
    
    /* 5. Update RBPF emission parameters */
    for (int k = 0; k < K; k++) {
        rbpf->mu_vol[k] = (float)mu_vol[k];
        rbpf->sigma_vol[k] = (float)sigma_vol[k];
    }
    
    /* 6. Extract stickiness for adaptive forgetting */
    for (int k = 0; k < K; k++) {
        dt->stickiness[k] = (float)sticky_hdp_get_stickiness_prob(hdp, k);
    }
}
```

---

## 5. Complete Integration Example

```c
typedef struct {
    StickyHDP *hdp;           /* Offline structure discovery */
    RBPF *rbpf;               /* Online filtering */
    DirichletTransition *dt;  /* Adaptive transitions */
    
    int ticks_since_refine;
    int refine_interval;      /* 5000 ticks typical */
} HybridFilter;

void hybrid_filter_tick(HybridFilter *hf, float y)
{
    /* 1. Feed observation to HDP (for later batch processing) */
    sticky_hdp_observe(hf->hdp, (double)y);
    
    /* 2. Run RBPF tick */
    RBPF_Output out;
    rbpf_step(hf->rbpf, y, &out);
    
    /* 3. Adaptive forgetting */
    dirichlet_adapt_forgetting(hf->dt, out.regime_entropy, 0.03f);
    
    /* 4. Soft transition update */
    accumulate_soft_transitions(hf->rbpf, hf->dt);
    
    /* 5. Periodic HDP refinement */
    hf->ticks_since_refine++;
    if (hf->ticks_since_refine >= hf->refine_interval) {
        
        /* Check if HDP should actually run (adaptive trigger) */
        if (sticky_hdp_should_sweep(hf->hdp)) {
            hdp_refine_rbpf(hf->hdp, hf->rbpf, hf->dt);
            hf->ticks_since_refine = 0;
        }
    }
}
```

---

## 6. Performance Budget

| Component | Frequency | Latency | % of 1ms Budget |
|-----------|-----------|---------|-----------------|
| RBPF step | Every tick | 5 μs | 0.5% |
| Soft Dirichlet update | Every tick | 0.5 μs | 0.05% |
| Entropy-driven γ | Every tick | 0.1 μs | 0.01% |
| HDP-BEAM (3 sweeps) | Every 5000 ticks | 500 μs | 0% (amortized) |

**Total per-tick overhead: ~6 μs (0.6% of 1ms budget)**

---

## 7. Summary

| Aspect | RBPF Alone | Hybrid Architecture |
|--------|------------|---------------------|
| Structure discovery | Manual, fixed K | Automatic via HDP |
| Transition learning | Hard SPRT flips | Soft particle mass |
| Forgetting factor | Constant γ | Entropy-adaptive γ_t |
| Regime count | Static | Dynamic (birth/merge) |
| Label consistency | Risk of switching | μ-ordered guarantee |

### The Key Principles

1. **Let HDP-BEAM be the Architect**: It discovers "what exists"
2. **Let RBPF be the Builder**: It estimates "what's happening now"
3. **Soft updates capture ambiguity**: Don't wait for certainty
4. **Entropy drives forgetting**: Flush memory when confused
5. **μ-ordering prevents label switching**: Regime 0 is always "calm"

---

## 8. Files Reference

| File | Purpose |
|------|---------|
| `sticky_hdp_beam.h/c` | HDP-BEAM implementation |
| `soft_dirichlet_transition.h/c` | Adaptive Dirichlet transitions |
| `rbpf_ksc.h/c` | Core RBPF filtering |
| `rbpf_ksc_param_integration.c` | RBPF ↔ Dirichlet wiring |

---

## 9. Future Extensions

1. **Parallel HDP-BEAM**: Run on separate thread, async updates
2. **Multi-asset structure sharing**: Common regime structure across correlated assets
3. **Hawkes-modulated transitions**: Event clustering affects transition rates
4. **Online K adaptation**: Birth/merge in hot path (with care)
