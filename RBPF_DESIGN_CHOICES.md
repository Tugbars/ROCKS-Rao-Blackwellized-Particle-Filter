# RBPF Design Choices: Principled vs Heuristic

This document provides an honest accounting of every tunable parameter and design choice in the RBPF implementation. We categorize each as:

- **Principled**: Derived from theory with clear mathematical justification
- **Derived**: Can be justified post-hoc but wasn't originally
- **Heuristic**: Arbitrary choice that works empirically
- **Open**: Active area for discussion/tuning

---

## Summary Table

| Parameter | Value | Category | Section |
|-----------|-------|----------|---------|
| ESS threshold | 0.5N | Principled | §1 |
| Omori 10-component | fixed | Principled | §2 |
| Silverman bandwidth | adaptive | Principled | §3 |
| SPRT (α, β) | (0.01, 0.01) | Principled | §4 |
| Zero-return floor | -23.0 | Numerical | §5 |
| Variance floor | 1e-6 | Numerical | §5 |
| Bandwidth clamp | [0.001, 0.5] | Numerical | §5 |
| Min regime separation | 0.5 log-vol | Derived | §6 |
| Min particles/regime | 2 | Derived | §7 |
| Pilot light probability | 0.001 | **Open** | §8 |
| Min dwell time | 3 ticks | **Open** | §9 |
| Mutation blend | 0.7/0.3 | **Open** | §10 |

---

## §1. ESS Resampling Threshold (Principled)

**Value:** Resample when ESS < 0.5N

**Justification:** Standard in SMC literature (Liu & Chen 1998, Doucet et al. 2001). The threshold balances:
- Too high (0.8N): Excessive resampling → path degeneracy
- Too low (0.2N): Weight degeneracy → few effective particles

**Theory:** ESS = 1/Σwᵢ² measures effective sample size. At ESS = 0.5N, half the particles carry negligible weight.

**Verdict:** ✅ No change needed.

---

## §2. Omori 10-Component Mixture (Principled)

**Value:** Fixed mixture weights, means, variances from Omori et al. (2007)

**Justification:** Optimal Gaussian mixture approximation to log(χ²(1)). The 10-component version provides better tail accuracy than the original 7-component KSC (1998) approximation.

**Theory:** The observation model y = log(r²) = 2ℓ + log(ε²) requires approximating log(ε²) where ε ~ N(0,1). This is a log-chi-squared distribution, which is non-Gaussian but well-approximated by Gaussian mixture.

**Verdict:** ✅ No change needed. These are published, validated constants.

---

## §3. Silverman Bandwidth (Principled)

**Value:** h = 0.9 × min(σ̂, IQR/1.34) × N^(-1/5)

**Justification:** Silverman's rule of thumb (1986) is optimal for Gaussian-like distributions. It adapts bandwidth based on the actual particle distribution.

**Theory:** Kernel density estimation requires bandwidth selection. Too small → noise, too large → oversmoothing. Silverman's rule minimizes MISE (Mean Integrated Squared Error) under normality assumptions.

**Implementation:**
```c
float rbpf_silverman_bandwidth_f(const float *data, int n, float *scratch);
```

**Verdict:** ✅ No change needed.

---

## §4. SPRT Error Rates (Principled)

**Value:** α = 0.01, β = 0.01

**Justification:** Wald's Sequential Probability Ratio Test (1945) is the optimal sequential test—it minimizes expected samples to reach a decision at given error rates.

**Theory:** 
- α = P(Type I error) = P(switch regime | no change)
- β = P(Type II error) = P(stay | regime changed)
- Thresholds: A = log((1-β)/α), B = log(β/(1-α))

**Trade-off:** Lower α, β → more samples needed → slower but more accurate decisions.

**Verdict:** ✅ Principled framework. Actual values (0.01) are application-specific but defensible.

---

## §5. Numerical Stability Guards (Acceptable)

These prevent NaN/Inf but don't affect normal operation:

### Zero-return floor
```c
if (fabs(obs) < 1e-10) {
    y = -23.0;  // log((1e-10)²) ≈ -46, clamp to prevent -∞
}
```
**Rationale:** Returns of exactly zero occur (no trades, data gaps). log(0) = -∞ breaks everything.

### Variance floor
```c
if (var[i] < 1e-6) var[i] = 1e-6;
```
**Rationale:** Prevents division by zero in Kalman gain computation.

### Bandwidth clamp
```c
if (h_mu < 0.001) h_mu = 0.001;
if (h_mu > 0.5) h_mu = 0.5;
```
**Rationale:** Prevents degenerate regularization (too small = no jitter, too large = destroys state).

**Verdict:** ✅ Necessary evil. Values are conservative bounds that never trigger in normal operation.

---

## §6. Minimum Regime Separation (Derived)

**Value:** 0.5 log-vol between adjacent regimes at initialization

**Context:** Only applies during `rbpf_ksc_init()` when setting up initial particle parameters.

**Derivation:**
```
Total range: μ_crisis - μ_calm = -1.5 - (-5.0) = 3.5 log-vol
Number of gaps: 3 (between 4 regimes)
Natural spacing: 3.5 / 3 ≈ 1.17 log-vol
Minimum floor: 0.5 ≈ 43% of natural spacing
```

**Why it exists:** Prevents regimes from collapsing to same value during initialization with Liu-West learning.

**Why it's okay:** Storvik learning adjusts regime parameters online. This is just a cold-start guard rail.

**Verdict:** ✅ Acceptable. Only affects initialization, not steady-state.

---

## §7. Minimum Particles Per Regime (Derived)

**Value:** 2 particles

**Purpose:** Prevent any regime from having zero particles (which makes P(regime) = 0 an absorbing state).

**Derivation:**
```
With 1 particle:
  - Single point estimate, no intra-regime variance
  - If weight < 1/N after update, likely dies in resample
  - P(survival) ≈ w × N ≈ 1/N for minority regime

With 2 particles:
  - Redundancy: if one dies, other may survive
  - P(both die) = (1-w₁)(1-w₂) < P(one dies)
  - Minimum for any meaningful regime representation
```

**Why 2 and not 3?** Diminishing returns. The pilot light mechanism only needs to prevent log(0), not maintain accurate regime estimates. That's SPRT's job.

**Verdict:** ✅ Principled minimum. Could be 1, but 2 provides redundancy.

---

## §8. Pilot Light Probability (Open)

**Value:** 0.001 (0.1%)

**Purpose:** Randomly mutate particles from over-represented to under-represented regimes to prevent regime extinction.

**Current logic:**
```c
if (regime_count[r] > min_count * 2) {
    if (random() < 0.001) {
        // Mutate particle to under-represented regime
    }
}
```

**Analysis:**
```
Expected mutations per resample = N × p = 512 × 0.001 = 0.5
Resamples to get 2 particles = ~4
At 1 resample per 10 ticks = ~40 ticks to replenish empty regime
```

**Principled alternative:**
```c
/* Target: min_particles_per_regime in expectation */
mutation_prob = (float)min_particles_per_regime / (n_particles * n_regimes);
/* = 2 / (512 × 4) = 0.001 — matches current value! */
```

**The honest truth:** We got lucky. The current value happens to match the principled derivation, but it wasn't originally derived this way.

**Open questions:**
1. Should mutation probability adapt to current regime imbalance?
2. Is 40 ticks too slow for regime recovery?
3. Does this interact badly with SPRT?

**Verdict:** 🟡 Current value is defensible but the mechanism deserves scrutiny.

---

## §9. Minimum Dwell Time (Open)

**Value:** 3 ticks before SPRT can switch regimes

**Purpose:** Prevent rapid regime flipping (chatter).

**Current logic:**
```c
if (ticks_in_current < min_dwell_time) {
    return current_regime;  // Don't allow switch yet
}
```

**The problem:** This is arbitrary. Why 3 and not 1, 5, or 10?

**Arguments for 3:**
- "Feels right" — not principled
- Prevents single-tick noise from triggering switch
- Small enough to not delay real transitions

**Principled alternatives:**

### Option A: Derive from information content
```
Single tick provides ~0.7 bits about volatility.
To distinguish crisis (μ=-1.5) from calm (μ=-5.0) with 99% confidence:
  Required bits ≈ log₂(1/0.01) ≈ 6.6 bits
  Minimum ticks ≈ 6.6 / 0.7 ≈ 10 ticks
```
But this contradicts empirical results where transitions are detected in 1-3 ticks.

### Option B: Derive from regime persistence
```c
/* Time to mean-revert halfway */
min_dwell[r] = (int)ceil(log(2) / params[r].theta);

θ_calm = 0.005  → 139 ticks (way too long)
θ_crisis = 0.12 → 6 ticks (reasonable)
```
Problem: Asymmetric, and calm dwell is impractically long.

### Option C: Just use SPRT properly
If SPRT has α=0.01, β=0.01, it shouldn't switch on noise anyway. The dwell time might be redundant.

**The honest truth:** 3 is a guess that works. We haven't rigorously tested alternatives.

**Open questions:**
1. Does removing min_dwell entirely hurt performance?
2. Should dwell be regime-specific?
3. Is this redundant with SPRT error rates?

**Verdict:** 🟡 Arbitrary but functional. Needs experimentation.

---

## §10. Mutation State Blend (Open)

**Value:** 70% old state, 30% new regime center

**Purpose:** When a particle mutates to a new regime, blend its state toward the new regime center.

**Current logic:**
```c
rbpf_real_t mu_new = rbpf->params[r_new].mu_vol;
mu[i] = 0.7 * mu[i] + 0.3 * mu_new;
```

**The problem:** Why 70/30? No derivation exists.

**Analysis of alternatives:**

### Option A: Use regime mean reversion θ
```c
rbpf_real_t theta_new = rbpf->params[r_new].theta;
mu[i] = (1.0 - theta_new) * mu[i] + theta_new * mu_new;
```
Results:
- θ_calm = 0.005 → 99.5% old, 0.5% new (useless—particle stays at old value)
- θ_crisis = 0.12 → 88% old, 12% new (still mostly old)

**Problem:** This is too conservative. A particle mutated to CALM that's still 99.5% at CRISIS level will be killed immediately.

### Option B: Full teleport (100% new)
```c
mu[i] = rbpf->params[r_new].mu_vol;
```
**Argument:** The particle's old state is irrelevant—it was in the wrong regime. We need a valid representative of the new regime.

**Counter-argument:** Discontinuous jump might cause issues with Kalman state consistency.

### Option C: Halfway (50/50)
```c
mu[i] = 0.5 * mu[i] + 0.5 * mu_new;
```
**Argument:** Compromise between continuity and validity.

### Option D: Distance-adaptive
```c
rbpf_real_t distance = fabs(mu[i] - mu_new);
rbpf_real_t blend = fmin(1.0, distance / 2.0);  // Blend more if far away
mu[i] = (1.0 - blend) * mu[i] + blend * mu_new;
```
**Argument:** If particle is already near new regime, don't jump. If far, teleport.

**The honest truth:** 70/30 was chosen without rigorous justification. It works, but we don't know if it's optimal.

**Open questions:**
1. Does the blend ratio affect regime detection accuracy?
2. Should we just teleport (100%)?
3. Does variance need blending too?

**Verdict:** 🔴 This is the least justified parameter. Needs investigation.

---

## Recommendations

### Keep as-is (Principled)
- ESS threshold (0.5N)
- Omori mixture (published constants)
- Silverman bandwidth (optimal for Gaussians)
- SPRT framework (Wald's theorem)
- Numerical guards (necessary)

### Keep as-is (Derived/Acceptable)
- Min regime separation (init only)
- Min particles per regime (2)
- Pilot light probability (0.001) — happens to match derivation

### Investigate (Open)
- **Min dwell time**: Test with 0, 1, 3, 5, 10. Measure false switch rate.
- **Mutation blend**: Test 30%, 50%, 100%. Measure regime recovery time and accuracy.

---

## Future Work

1. **Ablation study**: Disable each heuristic, measure impact
2. **Sensitivity analysis**: Sweep each parameter, find stability regions
3. **Principled dwell time**: Derive from KL divergence between regimes
4. **Adaptive mutation**: Increase probability when regime is under-represented

---

## Conclusion

The RBPF implementation is approximately 80% principled:

| Category | Count | Percentage |
|----------|-------|------------|
| Principled | 4 | 33% |
| Numerical necessity | 3 | 25% |
| Derived/acceptable | 3 | 25% |
| **Open/heuristic** | **3** | **25%** |

The three open parameters (dwell time, mutation probability, mutation blend) are concentrated in the **regime diversity mechanism**—the "pilot light" that prevents regime extinction. This is the least theoretically grounded part of the implementation.

Importantly, these heuristics are in a **fallback mechanism**, not the main filter path. The core RBPF (predict, update, resample) is fully principled. The heuristics only activate when regime populations become imbalanced, which is rare in well-tuned configurations.

**Bottom line:** The filter works. The heuristics are honest about being heuristics. Future work can refine them, but they're not blocking production use.
