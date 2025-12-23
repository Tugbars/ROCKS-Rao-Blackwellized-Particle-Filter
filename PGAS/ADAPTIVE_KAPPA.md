# Adaptive Kappa: Self-Tuning Stickiness Prior for PGAS

## Overview

PGAS (Particle Gibbs with Ancestor Sampling) requires a **sticky prior** to prevent "chatter" — false regime transitions caused by observation noise overwhelming regime separation.

The sticky prior is controlled by **κ (kappa)**, which determines how strongly the model believes regimes persist:

| κ Value | Expected Diagonal | Behavior |
|---------|-------------------|----------|
| 20 | ~0.85 | Fast switching |
| 100 | ~0.95 | Normal stickiness |
| 500 | ~0.99 | Very sticky |

**Problem**: We don't know the true stickiness ahead of time.

**Solution**: Use the **chatter ratio** as feedback to auto-tune κ.

---

## The Chatter Ratio

After each Gibbs sweep, we compute:

```
chatter_ratio = observed_off_diagonal_transitions / expected_off_diagonal_transitions
```

Where:
- `observed` = actual regime switches in the reference trajectory
- `expected` = what the current κ predicts

### Interpretation

| Chatter Ratio | Meaning | Action |
|---------------|---------|--------|
| > 1.5 | PGAS sees more switching than prior expects | Increase κ (more sticky) |
| 0.5 - 1.5 | Prior matches data | Keep κ |
| < 0.5 | Prior suppressing real transitions | Decrease κ (less sticky) |

---

## Why Not Online EM?

Online EM (Cappé & Moulines 2009) updates the **full transition matrix** with a decaying learning rate:

```
A_{t+1} = A_t + γ_t * (sufficient_stats - A_t)
```

### Problems with Online EM

1. **Gets "arrogant"**: Early estimates lock in as γ_t → 0
2. **Self-reinforcing bias**: Wrong θ → wrong E-step → confirms wrong θ
3. **No recovery**: Once stuck, hard to escape without reset
4. **K² parameters**: All 16 transition values adapt independently

### Why Chatter Ratio Works Better

| Property | Online EM | Chatter Ratio |
|----------|-----------|---------------|
| What adapts | θ (the answer) | κ (regularization strength) |
| DOF | K² = 16 | 1 |
| Can lock in wrong answer | Yes | No |
| Recovery from error | Hard | Automatic |
| Structure preserved | No | Yes (sticky diagonal) |

**Key insight**: We adapt *how much to trust the prior*, not the answer itself.

---

## Mathematical Formulation

### Dirichlet Posterior

For each row i of the transition matrix:

```
π_i ~ Dirichlet(α + n_{i1}, α + n_{i2}, ..., α + n_{iK} + κ·I(j=i))
```

Where:
- α = symmetric prior (default 1.0)
- n_{ij} = transition counts from data
- κ = sticky bonus (self-transition)

### Expected Diagonal

```
E[A_{ii}] ≈ (κ + α + n_{ii}) / (κ + K·α + Σ_j n_{ij})
```

For large κ: E[A_{ii}] → 1 (always stay)
For κ → 0: E[A_{ii}] → data-driven

### Chatter Ratio Derivation

```
expected_diag = (κ + α) / (κ + K·α)
expected_off_diag_rate = 1 - expected_diag
expected_off_diag_count = (T - 1) × expected_off_diag_rate

chatter_ratio = observed_off_diag_count / expected_off_diag_count
```

This is essentially a **likelihood ratio** for the stickiness hypothesis.

---

## Adaptation Algorithm

```c
if (adaptive_kappa_enabled) {
    float ratio = last_chatter_ratio;
    
    if (ratio > 1.5f) {
        // Too much chatter: increase stickiness
        kappa *= 1.0f + adapt_rate * (ratio - 1.0f);
    }
    else if (ratio < 0.5f) {
        // Prior too strong: decrease stickiness
        kappa *= 1.0f - adapt_rate * (1.0f - 2.0f * ratio);
    }
    // else: ratio in [0.5, 1.5] → no change
    
    // Clamp to bounds
    kappa = clamp(kappa, kappa_min, kappa_max);
}
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `kappa_min` | 20.0 | Lower bound (prevents instability) |
| `kappa_max` | 500.0 | Upper bound (prevents over-regularization) |
| `adapt_rate` | 0.2 | Speed of adaptation (0.0-1.0) |

---

## API Usage

### Basic (Adaptive Disabled)

```c
PGASMKLState *pgas = pgas_mkl_alloc(N, T, K, seed);
pgas_mkl_set_transition_prior(pgas, 1.0f, 100.0f);  // α=1, κ=100

// Chatter ratio is ALWAYS computed (useful for diagnostics)
for (int i = 0; i < n_sweeps; i++) {
    pgas_mkl_gibbs_sweep(pgas);
    printf("Chatter: %.2fx\n", pgas_mkl_get_chatter_ratio(pgas));
}
```

### With Adaptive Kappa

```c
PGASMKLState *pgas = pgas_mkl_alloc(N, T, K, seed);
pgas_mkl_set_transition_prior(pgas, 1.0f, 100.0f);  // Initial κ

// Enable adaptation
pgas_mkl_enable_adaptive_kappa(pgas, 1);
pgas_mkl_configure_adaptive_kappa(pgas, 
    50.0f,   // kappa_min
    300.0f,  // kappa_max  
    0.15f);  // adapt_rate

for (int i = 0; i < n_sweeps; i++) {
    pgas_mkl_gibbs_sweep(pgas);
    printf("κ=%.1f, chatter=%.2fx\n", 
           pgas_mkl_get_sticky_kappa(pgas),
           pgas_mkl_get_chatter_ratio(pgas));
}
```

### In Lifeboat (Slow Loop)

```c
void lifeboat_on_pgas_complete(LifeboatState *state) {
    float kappa = pgas_mkl_get_sticky_kappa(state->pgas);
    float chatter = pgas_mkl_get_chatter_ratio(state->pgas);
    
    // Log for monitoring
    log_metric("pgas.kappa", kappa);
    log_metric("pgas.chatter_ratio", chatter);
    
    // κ self-tunes to match actual market stickiness
    // No manual intervention needed
}
```

---

## Validation Results

### With Fixed κ=100

| True Diagonal | Learned | Error | Result |
|---------------|---------|-------|--------|
| 0.95 | 0.95-0.96 | < 1% | ✓ PASS |
| 0.90 | 0.95-0.97 | 5-7% | ✗ FAIL (prior too strong) |

### Lesson

- κ=100 works for true diagonal ≈ 0.95
- κ=100 is too strong for true diagonal ≈ 0.90
- Adaptive κ can correct this automatically

---

## Comparison to Literature

### Online EM (Cappé & Moulines 2009)

- Updates full transition matrix online
- Decaying learning rate γ_t
- Asymptotic convergence guarantees
- **Problem**: Gets stuck in local optima

### Streaming Variational Bayes

- Similar to Online EM but with natural gradients
- Still adapts full parameter vector
- **Problem**: Same arrogance/lock-in issues

### Our Approach: Online Hyperparameter Tuning

- Adapts regularization strength (κ), not parameters
- Structure preserved (sticky diagonal)
- Bounded, robust, interpretable
- **Trade-off**: Can't learn asymmetric transitions

---

## When to Use What

| Scenario | Recommendation |
|----------|----------------|
| Fixed market regime structure | Adaptive κ ✓ |
| New regimes may emerge | Need HDP-HMM |
| Asymmetric transitions (A→B ≠ B→A) | Need full Online EM or fixed matrix |
| Production robustness critical | Adaptive κ ✓ |
| Research/exploration | Try both, compare |

---

## Implementation Notes

### Chatter Ratio Always Computed

Even with `adaptive_kappa_enabled = 0`, the chatter ratio is computed every sweep. This allows monitoring without adaptation.

### Bounds Are Critical

The [κ_min, κ_max] bounds prevent pathological states:
- κ → 0: complete chatter, sampler unstable
- κ → ∞: prior dominates, learns nothing

### Adapt Rate Tuning

| Rate | Behavior |
|------|----------|
| 0.05 | Very slow, stable |
| 0.2 | Default, balanced |
| 0.5 | Fast, may oscillate |

For HFT: slower is safer (0.1-0.2)

---

## Summary

**The Problem**: PGAS needs a stickiness prior, but we don't know the true stickiness.

**The Solution**: Use chatter ratio as feedback to auto-tune κ.

**Why It Works**:
1. Single DOF (κ) instead of K²
2. Bounded — can't diverge
3. Self-correcting — detects prior-data mismatch
4. Structure preserved — always sticky diagonal

**Key Insight**: Adapt *how much to trust the prior*, not the answer itself. This is robust where Online EM is brittle.
