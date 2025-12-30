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
| > 1.5 | Data has MORE switches than prior expects | DECREASE κ (prior too sticky) |
| 0.5 - 1.5 | Prior matches data | Keep κ |
| < 0.5 | Data has FEWER switches than prior expects | INCREASE κ (prior not sticky enough) |

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

### The "Adaptive MCMC Death Spiral" Problem

Naive moment-matching has a **circular dependency**:
1. High κ → PGAS produces sticky trajectory
2. Moment-matching sees sticky trajectory → thinks κ should be high
3. κ stays high → repeat!

The prior is essentially **validating itself**.

### Solution: Chatter-Corrected Moment Matching

Use the **chatter ratio** as a negative feedback signal to break the loop:

```c
// 1. Expected switch rate from current prior
expected_diag = (κ + α) / (κ + K*α);
expected_switch_rate = 1 - expected_diag;

// 2. Observed switch rate from PGAS counts
observed_switch_rate = off_diag_count / total_count;

// 3. Chatter Ratio (with Laplace smoothing)
chatter = (observed + 0.5/N) / (expected + 0.5/N);

// 4. Corrected Target Diagonal
target_switch_rate = expected_switch_rate * chatter;
target_diag = 1 - target_switch_rate;

// 5. Oracle κ from target diagonal
kappa_oracle = α * (target_diag * K - 1) / (1 - target_diag);

// 6. Log-Space Momentum Update
log_kappa_new = 0.7 * log(kappa) + 0.3 * log(kappa_oracle);
kappa = exp(log_kappa_new);
```

### Why This Works

| Scenario | Chatter | Effect |
|----------|---------|--------|
| Data fights prior (more switches) | > 1 | target_diag ↓ → κ ↓ |
| Data matches prior | ≈ 1 | κ stable |
| Data calmer than prior | < 1 | target_diag ↑ → κ ↑ |

**The feedback is negative**: if PGAS sees excess switching, κ decreases to accommodate.

### Stabilization: RLS Smoothing

Single-sweep chatter is **extremely noisy** due to OCSN observation noise (σ≈2.2). Without smoothing, κ can swing from 30 → 300 in a few sweeps.

**Solution: Recursive Least Squares (RLS) with Forgetting Factor**

RLS is a Kalman filter for tracking a slowly-varying parameter:

```c
// RLS state: estimate θ, variance P

// Kalman gain (adaptive)
float K = P / (λ + P);

// Update estimate
θ = θ + K * (raw_chatter - θ);

// Update variance (with forgetting)
P = (1/λ) * (P - K * P);
```

**Why RLS > EMA:**

| Aspect | EMA | RLS |
|--------|-----|-----|
| Gain | Fixed | **Adaptive** (based on P) |
| Early phase | Over-trusts noisy data | **High P → cautious** |
| Steady state | Fixed response | **Low P → stable** |
| Theory | Heuristic | **Optimal** for tracking |

**Forgetting Factor λ:**
- λ = 0.95 → ~20 sweep effective window (faster)
- λ = 0.97 → ~33 sweep effective window (default)
- λ = 0.99 → ~100 sweep effective window (slower)

```c
// Configure RLS forgetting factor
pgas_mkl_set_rls_forgetting(pgas, 0.97f);

// Monitor RLS variance (for diagnostics)
float P = pgas_mkl_get_rls_variance(pgas);
// Low P → confident, high P → still adapting
```

### Log-Space Momentum Update

Updates in log-space ensure **symmetric steps**:

| Linear (bad) | Log-space (good) |
|--------------|------------------|
| κ=100→50: moves 50 | κ=100→50: ratio 0.5 |
| κ=100→200: moves 100 | κ=100→200: ratio 2.0 |

The momentum factor (0.7) also satisfies **diminishing adaptation** for ergodicity (Andrieu & Moulines 2006).

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `kappa_min` | 20.0 | Lower bound (prevents instability) |
| `kappa_max` | 500.0 | Upper bound (prevents over-regularization) |
| `rls_forgetting` | 0.97 | λ: forgetting factor (~33 sweep window) |
| `rls_variance` | 1.0 | P: initial estimation uncertainty |
| `momentum` | 0.8 | Log-space κ update momentum (internal) |
| `chatter_min` | 0.3 | Minimum chatter (prevents upward spiral) |
| `chatter_max` | 3.0 | Maximum chatter (prevents downward spiral) |
| Laplace | 0.5/N | Numerical stability (internal) |

### Literature Reference

**Andrieu & Moulines (2006)** "On the Ergodicity of Adaptive MCMC Algorithms"
- Proves adaptive samplers require "diminishing adaptation" to converge
- Our momentum factor acts as a proxy for this requirement

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
    0.3f,    // up_rate (fast DECREASE when chatter high)
    0.1f);   // down_rate (slow INCREASE when chatter low)

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

---

## Production Considerations

### A. Chatter-Corrected Moment Matching with RLS (IMPLEMENTED)

Breaks the "Adaptive MCMC Death Spiral" where the prior validates itself.

**Key Insight**: The chatter ratio is an **unbiased error signal**:
- chatter > 1: Data has more switches than prior expects → κ should decrease
- chatter < 1: Data has fewer switches than prior expects → κ should increase

**Smoothing**: Recursive Least Squares (RLS) with forgetting factor
- More principled than EMA (optimal for tracking non-stationary signals)
- Adaptive gain based on estimation uncertainty P
- High P → trust new data more; Low P → trust estimate more

**Implementation:**
```c
// 1. Compute raw chatter
float raw_chatter = observed_switch_rate / expected_switch_rate;

// 2. RLS update (Kalman filter for scalar)
float K = P / (lambda + P);              // Adaptive gain
theta = theta + K * (raw_chatter - theta); // Update estimate
P = (1/lambda) * (P - K * P);            // Update variance

// 3. Use RLS-smoothed chatter
float chatter = clamp(theta, 0.3f, 3.0f);

// 4. Corrected target from smoothed chatter
float target_diag = 1 - expected_switch_rate * chatter;

// 5. Oracle κ from corrected target
float kappa_oracle = alpha * (target_diag * K - 1) / (1 - target_diag);

// 6. Log-space momentum update
float log_kappa = 0.8f * logf(kappa) + 0.2f * logf(kappa_oracle);
kappa = expf(log_kappa);
```

**Reference**: Ljung & Söderström, "Theory and Practice of Recursive Identification"

---

### B. Transition Matrix Handoff to RBPF

The RBPF runs on the fast tick loop. Don't hand off raw sampled transitions from a single Gibbs sweep — too noisy.

**Recommendation**: Use exponential moving average for RBPF's transition matrix:

```c
// In lifeboat handoff
float eta = 0.1f;  // Smoothing factor
for (int i = 0; i < K*K; i++) {
    Pi_RBPF[i] = (1.0f - eta) * Pi_RBPF[i] + eta * Pi_learned[i];
}
```

**Benefits**:
- Single outlier sweep doesn't destabilize RBPF
- RBPF "worldview" remains smooth
- Gradual adaptation to regime changes

**Tuning η**:
| η | Behavior |
|---|----------|
| 0.05 | Very smooth, slow to adapt |
| 0.1 | Balanced (recommended) |
| 0.3 | Responsive but may oscillate |

---

### C. Regime "Ghosting" — A Feature, Not a Bug

In HFT, some regimes may not be visited for thousands of ticks (e.g., "Limit Up/Down").

**Observation**: `n_trans` only updates for regimes present in `ref_regimes`. If a regime isn't visited, its row drifts back toward the prior.

**Why this is good**:
- "Refreshes" model's openness to rare events
- Prevents over-fitting to recent calm periods
- When rare regime finally appears, prior provides reasonable starting point

**Caution**: If a regime is never visited during PGAS window, the learned row will be dominated by prior. This is correct Bayesian behavior but may surprise if not expected.

---

### D. "Arrogance" Protection Checklist

To ensure this doesn't get stuck like Online EM:

| Protection | Implementation | Why |
|------------|----------------|-----|
| **κ_min ≥ 20** | `kappa_min = 20.0f` | Never become un-sticky, or RBPF loses mind on microstructure noise |
| **κ_max ≤ 500** | `kappa_max = 500.0f` | Never become so sticky that real transitions are suppressed |
| **Lifeboat validation** | `lifeboat_mkl_validate()` | If PGAS fails to converge (low acceptance), don't update RBPF — stay with last "known good" |
| **Bounded adaptation** | Rate ∈ [0.05, 0.3] | Prevent wild swings in κ |

```c
// In lifeboat handoff
if (lifeboat_mkl_validate(&packet) && packet.ancestor_acceptance > 0.5f) {
    // Good PGAS run — update RBPF
    update_rbpf_transitions(rbpf, &packet);
} else {
    // Bad run — keep previous parameters
    log_warning("PGAS validation failed, keeping previous transitions");
}
```

---

### E. Monitoring Metrics

Log these in production:

| Metric | Healthy Range | Alert If |
|--------|---------------|----------|
| `kappa` | 50 - 200 | < 30 or > 400 |
| `chatter_ratio` | 0.7 - 1.3 | < 0.3 or > 2.0 |
| `acceptance_rate` | 0.6 - 0.95 | < 0.4 |
| `regime_coverage` | 3-4 regimes visited | < 2 regimes |

```c
void log_pgas_health(PGASMKLState *pgas) {
    log_metric("pgas.kappa", pgas_mkl_get_sticky_kappa(pgas));
    log_metric("pgas.chatter", pgas_mkl_get_chatter_ratio(pgas));
    log_metric("pgas.acceptance", pgas_mkl_get_acceptance_rate(pgas));
}
```

---

### F. Recovery Procedures

If PGAS gets into a bad state:

| Symptom | Cause | Fix |
|---------|-------|-----|
| κ stuck at min (20) | Prior too weak for noise | Increase kappa_min to 50 |
| κ stuck at max (500) | Prior too strong OR regime change | Reset κ to 100, increase window |
| Acceptance < 0.3 | Model mismatch | Check mu_vol, phi, sigma_h |
| Chatter > 3.0 | Observation noise spike | Temporary — wait for recovery |
| Single regime dominates | Market in one state | Normal — will recover when market moves |

**Emergency Reset**:
```c
void pgas_emergency_reset(PGASMKLState *pgas) {
    pgas->sticky_kappa = 100.0f;  // Reset to default
    pgas->last_chatter_ratio = 1.0f;
    // Re-initialize reference trajectory
    pgas_mkl_csmc_sweep(pgas);
}
```

---

## Future Enhancements

### Regime-Specific κ (Not Yet Implemented)
Different regimes may have different stickiness:
- Crisis regime: very sticky (hard to exit)
- Normal regime: moderately sticky
- Transition regime: less sticky

Would require K separate κ values instead of one global.

### Chatter Ratio EMA (Not Yet Implemented)
Smooth chatter ratio over multiple sweeps to reduce noise:
```c
chatter_ema = 0.9f * chatter_ema + 0.1f * current_chatter;
// Adapt based on EMA, not instantaneous value
```

### Transition Matrix EMA for RBPF Handoff (Recommended)
Don't hand raw samples to RBPF — use running mean:
```c
float eta = 0.1f;
for (int i = 0; i < K*K; i++) {
    Pi_RBPF[i] = (1.0f - eta) * Pi_RBPF[i] + eta * Pi_learned[i];
}
```
This belongs in the Lifeboat layer, not PGAS core.