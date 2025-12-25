# Oracle Integration Plan: PGAS-RBPF Coordination

## Executive Summary

This document describes a principled approach for integrating a **Particle Gibbs with Ancestor Sampling (PGAS)** oracle with a real-time **Rao-Blackwellized Particle Filter (RBPF)** for adaptive regime-switching volatility estimation in HFT environments.

The core challenge: **When and how should the slow, accurate PGAS update the fast, approximate RBPF's transition matrix Π?**

### Design Consensus

After evaluating multiple approaches (fixed thresholds, SMC-SA gradients, information bottleneck, RLS blending), we converged on:

| Component | Approach | Rationale |
|-----------|----------|-----------|
| **Trigger 1** | Hawkes (Lead) | External early warning before filter confusion |
| **Trigger 2** | KL Divergence (Truth) | Internal ground truth of filter state |
| **Thresholds** | Variance-based (z × σ) | Self-calibrating, no magic numbers |
| **Handoff** | SAEM (Sufficient Statistics) | Respects Dirichlet structure, automatic simplex |
| **Control** | Two knobs (z, α) | Statistically meaningful parameters |

### Critical Pitfalls Addressed

| Pitfall | Risk | Robust Mitigation | Literature |
|---------|------|-------------------|------------|
| Confirmation Bias | Filter+Oracle reinforce errors | Tempered path injection (5% flips) | Chopin & Papaspiliopoulos (2020) |
| Count Overwhelment | SAEM ballast prevents adaptation | Dual-conditioned reset gate | Särkkä (2013) |
| Latency Spike | Scout pollutes RBPF cache | Pin to separate cores | — |
| Resampling Noise | False KL triggers | Grace window requirement | — |
| Dead Zone | Particle death on rare transitions | Uniform escape hatch | — |

### Final Tier Upgrades

| Upgrade | Problem | Solution |
|---------|---------|----------|
| Exponential Weighting | Lagged average after regime shift | Recency-weighted likelihood in CSMC |
| Mixing Validation | Degenerate scout false confidence | Acceptance rate + unique path check |
| Dual-Gate + σ Cap | Oracle blind during chaos | Variance cap + absolute panic threshold |
| Innovation Partial Reset | SAEM can't track phase shifts | Three-tier reset hierarchy |

### Multi-Armed Bandit Integration

| MAB Concept | Implementation | Literature |
|-------------|---------------|------------|
| UCB | z×σ_H threshold for triggers | Auer et al. (2002) |
| Thompson Sampling | Π ~ Dirichlet(Q) for handoff | Agrawal & Goyal (2012) |
| Restless Bandit | Tempered injection + SAEM reset | Whittle (1988) |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PRODUCTION PIPELINE                               │
│                                                                             │
│    FAST PATH (every tick, <5µs)         SLOW PATH (triggered, ~100ms)      │
│    ┌─────────────────────────┐          ┌─────────────────────────┐        │
│    │         RBPF            │          │      PGAS ORACLE        │        │
│    │  • Track h_t, regime_t  │◄─────────│  • Learn Π from data    │        │
│    │  • Use current Π        │   Π_new  │  • Ensemble statistics  │        │
│    │  • Export MAP path      │──────────▶  • Confidence metrics   │        │
│    └─────────────────────────┘  warmup  └─────────────────────────┘        │
│              │                                     ▲                        │
│              │ observations                        │                        │
│              ▼                                     │ trigger                │
│    ┌─────────────────────────┐                     │                        │
│    │    HAWKES PROCESS       │─────────────────────┘                        │
│    │  • Model event clusters │                                              │
│    │  • Detect anomalies     │                                              │
│    └─────────────────────────┘                                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## The Problem with Naive Triggering

### Why Not Trigger on Intensity Spikes?

A Hawkes process models **self-exciting** event dynamics—spikes are *expected* behavior, not anomalies.

```
    Intensity λ(t)
        │
        │      ┌──┐
        │      │  │   ← Spike (expected under Hawkes)
        │      │  │
        │  ────┘  └────────────   Decay follows kernel
        │
        └──────────────────────▶ time
        
        ✗ Triggering here causes "Parameter Jitter"
```

### The Correct Signal: Persistent Deviation

```
    Intensity λ(t)
        │
        │      ┌──┐
        │      │  │   Hawkes predicts
        │      │  │   this decay ────┐
        │      │  └──┐               │
        │      │     └──┐            │
        │      │        └──────      ▼
        │      │                     
        │      │  ████████████████   ← Observed stays elevated
        │      │  ████████████████     (structural change!)
        │      │  ████████████████
        └──────┴─────────────────────▶ time
        
        Cumulative Residual = ∫ (observed - λ_predicted) dt
        
        ✓ Trigger when residual persists beyond kernel decay time
```

---

## Three-Layer Oracle Activation

### Layer 1: Detection (Hawkes Integrator)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         INTEGRATED INTENSITY                            │
│                                                                         │
│   Instead of:  if λ(t) > threshold → trigger  ❌                       │
│                                                                         │
│   Use:         if ∫λ(t)dt over window > high_water_mark                │
│                AND refractory_period elapsed                            │
│                → trigger SCOUT SWEEP  ✓                                 │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Hysteresis Thresholds:                                                │
│                                                                         │
│       ──────────────── High Water Mark (trigger scout) ────────────    │
│                                                                         │
│                  ████                                                   │
│              ████████████                                               │
│          ████████████████████                                           │
│       ███████████████████████████                                       │
│                                                                         │
│       ──────────────── Low Water Mark (stay active) ───────────────    │
│                                                                         │
│   • High water: Start oracle investigation                              │
│   • Low water: Keep oracle active while elevated                        │
│   • Below low water: Return to monitoring mode                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Layer 2: Verification (Scout Sweep)

Before committing expensive PGAS resources, run a **lightweight verification**:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           SCOUT SWEEP                                   │
│                                                                         │
│   1. Run 5-10 PARIS backward smoothing sweeps (fast, ~5ms)             │
│                                                                         │
│   2. Compute Latent State Entropy:                                      │
│                                                                         │
│      H = -Σ p(regime_t) log p(regime_t)                                │
│                                                                         │
│   3. Compare to baseline entropy:                                       │
│                                                                         │
│      ΔH = |H_scout - H_baseline|                                       │
│                                                                         │
│   4. Decision:                                                          │
│                                                                         │
│      if ΔH > entropy_threshold:                                        │
│          → Regime structure changed, run FULL PGAS                     │
│      else:                                                              │
│          → Transient noise, return to monitoring                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Layer 3: Handoff (SAEM Blender)

When PGAS produces a new Π, blend it into RBPF using **Stochastic Approximation EM**:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        SAEM PARAMETER BLENDING                          │
│                                                                         │
│   Key insight: Update sufficient statistics, not point estimates        │
│                                                                         │
│   PGAS outputs: S_oracle[i,j] = transition counts i→j                  │
│                                                                         │
│   SAEM update:                                                          │
│      Q_t[i,j] = (1 - γ) × Q_{t-1}[i,j] + γ × S_oracle[i,j]            │
│                                                                         │
│   Posterior mean (automatic simplex constraint!):                       │
│      Π[i,j] = Q_t[i,j] / Σ_k Q_t[i,k]                                  │
│                                                                         │
│   Adaptive γ based on:                                                  │
│   ┌─────────────────┬──────────────────────────────────────────────┐   │
│   │ Signal          │ Effect on γ                                  │   │
│   ├─────────────────┼──────────────────────────────────────────────┤   │
│   │ High acceptance │ ↑ Trust oracle more (γ → 0.3)               │   │
│   │ Low acceptance  │ ↓ Trust oracle less (γ → 0.05)              │   │
│   │ High diversity  │ ↑ Confident estimate                         │   │
│   │ Large surprise  │ ↑ Market changed, adapt faster               │   │
│   │ Robbins-Monro   │ γ decays as 1/√t over time                   │   │
│   └─────────────────┴──────────────────────────────────────────────┘   │
│                                                                         │
│   Safety Rails:                                                         │
│   • Floor enforcement: Π_ij ≥ 1e-5 (escape hatch for rare events)     │
│   • Automatic normalization: Dirichlet posterior is on simplex         │
│   • γ bounds: 0.02 ≤ γ ≤ 0.5                                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Parameter Handoff: Why SAEM over RLS

### The Problem with RLS

Recursive Least Squares treats transition matrix elements as independent scalars:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           RLS LIMITATIONS                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   What PGAS outputs:                                                    │
│   ┌─────────────────────────────────────────────────────┐              │
│   │  • 500 ensemble samples of Π                        │              │
│   │  • Sufficient statistics: n_trans[i,j] = counts     │              │
│   │  • Full posterior uncertainty                        │              │
│   └─────────────────────────────────────────────────────┘              │
│                                                                         │
│   What RLS uses:                                                        │
│   ┌─────────────────────────────────────────────────────┐              │
│   │  • Point estimate E[Π] only                         │              │
│   │  • Treats Π_ij as 16 independent scalars           │              │
│   │  • Ignores: rows must sum to 1                      │              │
│   │  • Ignores: ensemble uncertainty                    │              │
│   └─────────────────────────────────────────────────────┘              │
│                                                                         │
│   Problem: "Throws away" the rich Bayesian information                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Alternatives Considered

| Method | Approach | Constraint Handling | Uncertainty | Verdict |
|--------|----------|---------------------|-------------|---------|
| **RLS** | Blend Π directly | Post-hoc projection | Covariance P only | Baseline |
| **BLS** | Batch optimization | Post-hoc projection | None | Incremental |
| **RLS-KF** | Kalman on vec(Π) | Post-hoc projection | Full covariance | Awkward |
| **SAEM** | Blend counts | **Automatic** | **Full posterior** | **Winner** |

### Why SAEM Wins

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         SAEM ADVANTAGES                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   1. NATURAL FIT                                                        │
│      PGAS already outputs n_trans[i,j] counts                          │
│      SAEM directly uses these - no conversion needed                   │
│                                                                         │
│   2. RESPECTS CONSTRAINTS                                               │
│      Dirichlet posterior is defined on probability simplex             │
│      Rows automatically sum to 1                                        │
│                                                                         │
│   3. FULL BAYESIAN                                                      │
│      We get uncertainty quantification, not just point estimate        │
│      Can compute credible intervals if needed                          │
│                                                                         │
│   4. SIMPLE IMPLEMENTATION                                              │
│      Just weighted average of sufficient statistics                    │
│      No matrix inversions, no projections                              │
│                                                                         │
│   5. UNIFIED WITH TRIGGERS                                              │
│      γ uses same confidence signals (acceptance, entropy surprise)     │
│      Same z-score controls both triggers and blending                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### SAEM Implementation

```c
typedef struct {
    // Accumulated sufficient statistics (pseudo-counts)
    float Q[PGAS_MAX_K * PGAS_MAX_K];
    
    // Prior (encodes stickiness)
    float alpha_prior;
    float kappa_prior;
    
    // Adaptive step size
    float gamma_base;
    float gamma_min;
    int   update_count;
} SAEMBlender;

void saem_init(SAEMBlender *saem, int K, float alpha, float kappa) {
    saem->alpha_prior = alpha;
    saem->kappa_prior = kappa;
    saem->gamma_base = 0.3f;
    saem->gamma_min = 0.05f;
    saem->update_count = 0;
    
    // Initialize Q with prior pseudo-counts
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < K; j++) {
            float prior_count = alpha;
            if (i == j) prior_count += kappa;
            saem->Q[i * K + j] = prior_count;
        }
    }
}

void saem_update(SAEMBlender *saem,
                 const int *S_oracle,     // Transition counts from PGAS
                 int K,
                 float confidence,        // Oracle confidence (0-1)
                 float *Pi_out) {         // Output: updated Π
    
    saem->update_count++;
    
    // Adaptive step size: decays over time, scaled by confidence
    float gamma = saem->gamma_base * confidence / sqrtf((float)saem->update_count);
    if (gamma < saem->gamma_min) gamma = saem->gamma_min;
    
    // SAEM update: blend sufficient statistics
    for (int i = 0; i < K * K; i++) {
        saem->Q[i] = (1.0f - gamma) * saem->Q[i] + gamma * (float)S_oracle[i];
    }
    
    // Compute posterior mean (Dirichlet)
    for (int i = 0; i < K; i++) {
        float row_sum = 0.0f;
        for (int j = 0; j < K; j++) {
            row_sum += saem->Q[i * K + j];
        }
        
        float inv_sum = 1.0f / row_sum;
        for (int j = 0; j < K; j++) {
            Pi_out[i * K + j] = saem->Q[i * K + j] * inv_sum;
        }
    }
    
    // Enforce floor (safety)
    enforce_transition_floor(Pi_out, K);
}
```

### Adaptive γ: Unified with Trigger System

The SAEM step size uses the **same uncertainty signals** as the trigger system:

```c
float compute_saem_gamma(SAEMBlender *saem, 
                         float acceptance_rate,
                         float trajectory_diversity,
                         float entropy_surprise,
                         float z_score) {
    
    // Base decay (Robbins-Monro)
    float gamma = saem->gamma_base / sqrtf((float)saem->update_count + 1.0f);
    
    // Confidence scaling
    float confidence = 1.0f;
    
    if (acceptance_rate < 0.15f) confidence *= 0.5f;      // Oracle struggled
    if (trajectory_diversity < 0.5f) confidence *= 0.7f; // Poor mixing
    
    // Entropy surprise scaling (uses same z-score as triggers!)
    float surprise_scale = 1.0f + 0.5f * (entropy_surprise - z_score);
    if (surprise_scale < 0.5f) surprise_scale = 0.5f;
    if (surprise_scale > 2.0f) surprise_scale = 2.0f;
    
    gamma *= confidence * surprise_scale;
    
    // Clamp
    if (gamma < saem->gamma_min) gamma = saem->gamma_min;
    if (gamma > 0.5f) gamma = 0.5f;  // Never trust oracle 100%
    
    return gamma;
}
```

### SAEM Integration with Two Knobs

The SAEM blender naturally integrates with our two-knob philosophy:

| Knob | Effect on SAEM |
|------|----------------|
| **z_score** | Scales entropy_surprise threshold for γ adjustment |
| **α_variance** | Controls how fast we update confidence estimates |

```c
// Derived from knobs
cfg.saem_gamma_base = 0.3f;                              // Fixed base
cfg.saem_gamma_min = 0.02f + 0.08f * (1.0f - memory);   // Memory → slower min
cfg.saem_surprise_scale = z_score;                       // Same z as triggers
```

---

## Complete Flow Diagram

```
                                TICK ARRIVES
                                     │
                                     ▼
                        ┌────────────────────────┐
                        │   RBPF processes tick  │
                        │   (always, <5µs)       │
                        └────────────┬───────────┘
                                     │
                                     ▼
                        ┌────────────────────────┐
                        │  Hawkes updates λ(t)   │
                        └────────────┬───────────┘
                                     │
                                     ▼
                        ┌────────────────────────┐
                  No    │  ∫λdt > high_water?    │
              ┌─────────│  AND refractory OK?    │
              │         └────────────┬───────────┘
              │                      │ Yes
              │                      ▼
              │         ┌────────────────────────┐
              │         │    SCOUT SWEEP         │
              │         │    (5-10 PARIS)        │
              │         └────────────┬───────────┘
              │                      │
              │                      ▼
              │         ┌────────────────────────┐
              │    No   │   ΔH > threshold?      │
              │◄────────│   (entropy changed)    │
              │         └────────────┬───────────┘
              │                      │ Yes
              │                      ▼
              │         ┌────────────────────────┐
              │         │      FULL PGAS         │
              │         │  (300 burn + 500 samp) │
              │         └────────────┬───────────┘
              │                      │
              │                      ▼
              │         ┌────────────────────────┐
              │         │     RLS BLENDER        │
              │         │  Π = Π + G(Π_new - Π)  │
              │         └────────────┬───────────┘
              │                      │
              │                      ▼
              │         ┌────────────────────────┐
              │         │   Update RBPF's Π      │
              │         │   Reset refractory     │
              │         └────────────┬───────────┘
              │                      │
              └──────────────────────┘
                                     │
                                     ▼
                              NEXT TICK
```

---

## Comparison of Triggering Strategies

| Strategy | Signal Source | Robustness | Accuracy | Latency |
|----------|---------------|------------|----------|---------|
| **Fixed Clock** | Timer | High (consistent) | Low (lags shifts) | Fixed |
| **ESS Trigger** | RBPF confusion | Medium | Medium | Reactive |
| **Hawkes Spike** | Instant intensity | Low (overreacts) | High (fast) | Immediate |
| **Integrated Hawkes** | Sustained activity | **High** | **High** | Adaptive |

**Recommendation:** Use **Integrated Hawkes** as primary trigger with ESS as backup.

---

## Safeguards Against Common Failures

### 1. Information Feedback Loops

**Risk:** PGAS initialized with biased RBPF path → confirms existing errors.

**Fix:** Blind initialization for periodic updates:
```c
// Use observation-based initialization, not RBPF path
init_reference_from_observations(obs, T, K, mu_vol, phi, ref_regimes, ref_h);
```

### 2. Weight Shock on Π Update

**Risk:** Sudden Π change → particle weight collapse.

**Fix:** RLS blending with confidence-weighted gain (see Layer 3).

### 3. Zero-Probability Traps

**Risk:** PGAS learns Π_ij = 0 → RBPF can never transition to regime j.

**Fix:** Enforce floor:
```c
#define TRANS_FLOOR 1e-5f
if (trans[i*K + j] < TRANS_FLOOR) {
    trans[i*K + j] = TRANS_FLOOR;
}
// Then renormalize row
```

### 4. Parameter Jitter

**Risk:** Frequent PGAS triggers → unstable Π estimates.

**Fix:** Refractory period (500+ ticks) + hysteresis thresholds.

### 5. κ (Stickiness) Drift

**Risk:** Adaptive κ becomes extreme → RBPF fossilized or jittery.

**Fix:** Learn **ratios** from PGAS, control **absolute stickiness** manually:
```c
// Extract relative off-diagonal structure from PGAS
// But impose target average diagonal (e.g., 0.92-0.95)
extract_ratio_structure(pgas_trans, rbpf_trans, K, target_diag);
```

---

## Critical Pitfalls and Mitigations

### Pitfall 1: Confirmation Bias Loop (Information Leakage)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      THE DELUSIONAL STABILITY TRAP                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   RBPF stuck in wrong regime                                            │
│        │                                                                │
│        ▼                                                                │
│   Feeds incorrect MAP path to PGAS as warm start                       │
│        │                                                                │
│        ▼                                                                │
│   PGAS learns Π that "justifies" the error                             │
│        │                                                                │
│        ▼                                                                │
│   SAEM blends this into RBPF                                           │
│        │                                                                │
│        ▼                                                                │
│   RBPF becomes MORE confident in wrong regime                          │
│        │                                                                │
│        └──────────▶ Feedback loop of mutual delusion                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Risk:** Filter and Oracle agree on incorrect market state. Both reinforce each other's errors.

**Robust Solution: Tempered Path Injection**

Instead of periodic blind sweeps (which waste compute on high-variance cold starts), we use **tempered reference paths** that inject controlled exploration noise.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        TEMPERED PATH INJECTION                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Every PGAS run:                                                       │
│      1. Take RBPF MAP path (good starting point, low variance)         │
│      2. Inject 5% random regime flips (forced exploration)             │
│      3. PGAS samples around this "what-if" scenario                    │
│                                                                         │
│   Example (T=100, K=4):                                                │
│                                                                         │
│   RBPF MAP:     [0,0,0,0,0,1,1,1,1,1,2,2,2,2,2,...]                    │
│                          ↓ 5% random flips                             │
│   Tempered:     [0,0,3,0,0,1,1,2,1,1,2,2,2,0,2,...]                    │
│                      ↑       ↑           ↑                              │
│                   forced exploration points                             │
│                                                                         │
│   Result: Oracle always considers "what if RBPF is wrong here?"        │
│           Continuous exploration with LOW VARIANCE                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Literature:** Chopin & Papaspiliopoulos (2020) show that adding artificial "jitter" to the reference path ensures the sampler explores the full posterior density. This technique is also used in Iterated Filtering (IF2) for robust parameter tracking.

**Implementation:**

```c
typedef struct {
    float flip_probability;      // 0.05 = 5% of timesteps get random regime
    uint64_t seed;
} TemperedPathConfig;

void temper_reference_path(const int *rbpf_regimes,    // Input: RBPF MAP
                           int *tempered_regimes,       // Output: Jittered path
                           int T,
                           int K,
                           TemperedPathConfig *cfg) {
    
    // Fast RNG (xoroshiro128+)
    uint64_t s0 = cfg->seed;
    uint64_t s1 = cfg->seed ^ 0x9E3779B97F4A7C15ULL;
    
    for (int t = 0; t < T; t++) {
        // Generate random float in [0,1)
        uint64_t x = s0 + s1;
        s1 ^= s0;
        s0 = ((s0 << 55) | (s0 >> 9)) ^ s1 ^ (s1 << 14);
        s1 = (s1 << 36) | (s1 >> 28);
        float u = (float)(x >> 40) * 0x1.0p-24f;
        
        if (u < cfg->flip_probability) {
            // Random regime flip (explore alternative)
            int random_k = (int)(u / cfg->flip_probability * K) % K;
            tempered_regimes[t] = random_k;
        } else {
            // Keep RBPF regime (exploit knowledge)
            tempered_regimes[t] = rbpf_regimes[t];
        }
    }
    
    cfg->seed = s0;  // Update seed for next call
}
```

**Why Tempered > Blind:**

| Aspect | Blind Sweep | Tempered Injection |
|--------|-------------|-------------------|
| Frequency | Every 5th run | Every run |
| Starting point | Cold (observations only) | Warm (RBPF + jitter) |
| Variance | High (burns sweeps) | Low (controlled) |
| Exploration | Rare, expensive | Continuous, cheap |

---

### Pitfall 2: SAEM Count Overwhelment (The Ballast Trap)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         THE FOSSILIZED FILTER                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   50,000 ticks in "Calm" regime                                        │
│        │                                                                │
│        ▼                                                                │
│   Q[Calm→Calm] accumulates to ~45,000                                  │
│   Q[Calm→Crisis] stays at ~50                                          │
│        │                                                                │
│        ▼                                                                │
│   Flash Crash occurs! Oracle correctly identifies Crisis               │
│        │                                                                │
│        ▼                                                                │
│   SAEM update: Q_new = (1-γ)×45000 + γ×500                            │
│                      = 44550 + 150 = 44700                             │
│        │                                                                │
│        ▼                                                                │
│   Π[Calm→Calm] moves from 0.989 to 0.988                               │
│   (Effectively no change - ballast too heavy!)                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Risk:** Historical count mass prevents rapid adaptation to regime changes.

**Robust Solution: Dual-Conditioned Reset Gate**

Standard SAEM uses step-size γ_t = 1/t which eventually shrinks to zero—a failure point in HFT. The robust solution uses **conditioned covariance resetting** triggered only when BOTH signals confirm a structural break.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      DUAL-CONDITIONED RESET GATE                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Single trigger (fragile):                                            │
│      if (KL > 5σ) → RESET Q                                            │
│      Problem: KL can spike from resampling noise alone                 │
│                                                                         │
│   Dual-conditioned (robust):                                           │
│      if (KL > 5σ AND Hawkes > 5σ) → RESET Q                            │
│                                                                         │
│   Logic:                                                                │
│   • KL spike alone: Filter confused, maybe just resampling noise      │
│   • Hawkes spike alone: Market activity, maybe normal clustering      │
│   • BOTH spike: Market changed AND filter confused → STRUCTURAL BREAK │
│                                                                         │
│   This applies our dual-trigger philosophy to the reset decision!      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Literature:** Särkkä (2013) establishes that Covariance Resetting is the only principled way to keep recursive filters from becoming "overly confident" in historical estimates during structural breaks. The dual-conditioning prevents false resets from single noisy signals.

**Implementation:**

```c
typedef struct {
    float Q[PGAS_MAX_K * PGAS_MAX_K];
    float alpha_prior;
    float kappa_prior;
    float Q_max;                   // Cap to prevent fossilization
    
    // Dual reset thresholds
    float kl_reset_threshold;      // e.g., 5.0 (5σ)
    float hawkes_reset_threshold;  // e.g., 5.0 (5σ)
    
    // Reset state machine
    bool reset_armed;              // True if one condition met
    int reset_arm_ticks;           // How long armed
    int reset_arm_timeout;         // Disarm if second doesn't fire (e.g., 50 ticks)
} SAEMBlenderRobust;

typedef struct {
    float kl_sigma;       // KL surprise in σ units
    float hawkes_sigma;   // Hawkes surprise in σ units
} SurpriseMetrics;

void saem_update_robust(SAEMBlenderRobust *saem,
                        const int *S_oracle,
                        int K,
                        float gamma,
                        SurpriseMetrics surprise,
                        float *Pi_out) {
    
    bool kl_extreme = (surprise.kl_sigma > saem->kl_reset_threshold);
    bool hawkes_extreme = (surprise.hawkes_sigma > saem->hawkes_reset_threshold);
    
    // ═══════════════════════════════════════════════════════════════════
    // DUAL-CONDITIONED RESET GATE
    // ═══════════════════════════════════════════════════════════════════
    
    if (kl_extreme && hawkes_extreme) {
        // STRUCTURAL BREAK CONFIRMED: Both signals agree
        // Reset Q to prior + current oracle counts ("Day 0" event)
        for (int i = 0; i < K; i++) {
            for (int j = 0; j < K; j++) {
                float prior = (i == j) ? saem->kappa_prior : saem->alpha_prior;
                saem->Q[i * K + j] = prior + (float)S_oracle[i * K + j];
            }
        }
        gamma = 0.5f;  // High gamma for immediate adaptation
        saem->reset_armed = false;
        
    } else if (kl_extreme || hawkes_extreme) {
        // ONE signal extreme: Arm the reset, wait for confirmation
        if (!saem->reset_armed) {
            saem->reset_armed = true;
            saem->reset_arm_ticks = 0;
        } else {
            saem->reset_arm_ticks++;
            if (saem->reset_arm_ticks > saem->reset_arm_timeout) {
                // Timeout: disarm (was probably noise)
                saem->reset_armed = false;
            }
        }
        // Normal update (don't reset yet)
        
    } else {
        // Both normal: disarm and normal update
        saem->reset_armed = false;
    }
    
    // ═══════════════════════════════════════════════════════════════════
    // STANDARD SAEM UPDATE (with Q_max cap)
    // ═══════════════════════════════════════════════════════════════════
    
    for (int i = 0; i < K * K; i++) {
        saem->Q[i] = (1.0f - gamma) * saem->Q[i] + gamma * (float)S_oracle[i];
        if (saem->Q[i] > saem->Q_max) saem->Q[i] = saem->Q_max;
    }
    
    // Compute posterior mean (Dirichlet)
    for (int i = 0; i < K; i++) {
        float row_sum = 0.0f;
        for (int j = 0; j < K; j++) {
            row_sum += saem->Q[i * K + j];
        }
        float inv_sum = 1.0f / row_sum;
        for (int j = 0; j < K; j++) {
            Pi_out[i * K + j] = saem->Q[i * K + j] * inv_sum;
        }
    }
    
    // Enforce floor + escape hatch
    apply_escape_hatch(Pi_out, K);
}
```

**Why Dual-Conditioned > Single-Trigger:**

| Aspect | Single Trigger | Dual-Conditioned |
|--------|---------------|------------------|
| Trigger | KL or Hawkes alone | KL AND Hawkes |
| False reset risk | High | Low |
| Structural break detection | Noisy | Confirmed |
| Literature | Ad-hoc | Särkkä (2013) |

---

### Pitfall 3: Scout Sweep Latency Spike

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      THE CACHE THRASHING PROBLEM                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Normal tick:     RBPF processes in 2µs (hot cache)                   │
│                                                                         │
│   Scout triggers:  5ms PARIS sweep runs on same core                   │
│                    │                                                    │
│                    ├── Evicts RBPF data from L1/L2 cache               │
│                    ├── Consumes memory bandwidth                        │
│                    └── Next RBPF tick: cache miss → 50µs!              │
│                                                                         │
│   Result:  99.9th percentile latency spikes from 2µs to 5000µs        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Risk:** Scout sweeps near high-water mark cause CPU cache pollution, creating latency jitter.

**Mitigation:** Dedicated Verification Core

```c
// Pin RBPF to core 0, Scout to core 2 (separate L2 cache)

#ifdef _WIN32
void pin_to_core(int core_id) {
    SetThreadAffinityMask(GetCurrentThread(), 1ULL << core_id);
}
#else
void pin_to_core(int core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
}
#endif

// In RBPF main thread init:
pin_to_core(0);  // RBPF on core 0

// In Scout/Oracle worker thread:
pin_to_core(2);  // Scout on core 2 (different L2 cache domain)
```

**Architecture recommendation:**
```
Core 0: RBPF hot path (isolated, never preempted)
Core 1: Hawkes intensity tracking
Core 2: Scout sweeps + PGAS oracle
Core 3: SAEM blending + logging
```

---

### Pitfall 4: Entropy Resampling Noise False Alarms

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    THE BAD DICE ROLL PROBLEM                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   At N=32 particles, entropy is noisy:                                 │
│                                                                         │
│   Tick 1000: Resample happens to pick diverse particles → H = 2.1     │
│   Tick 1001: Resample happens to pick clustered particles → H = 1.3   │
│                                                                         │
│   ΔH = 0.8 nats in ONE TICK (purely from resampling luck!)            │
│                                                                         │
│   If threshold τ = 0.5:                                                │
│        → False trigger! Scout sweep for no reason.                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Risk:** KL divergence spikes from resampling luck, not structural change.

**Mitigation:** Grace Window + Consecutive Check Requirement

```c
typedef struct {
    // ... existing fields ...
    
    // Grace window for false alarm prevention
    int   grace_window;           // Consecutive checks required (e.g., 5)
    int   consecutive_elevated;   // Counter
    float elevated_threshold;     // What counts as "elevated" (e.g., 0.7 × τ)
} AdaptiveKLTrigger;

KLTriggerResult adaptive_kl_update_with_grace(AdaptiveKLTrigger *trigger,
                                               float kl_divergence,
                                               float current_entropy) {
    KLTriggerResult result = {0};
    
    // ... compute τ as before ...
    float tau = trigger->z_score * sqrtf(trigger->H_var_ema);
    
    // Check if elevated (even if below full threshold)
    float elevated_tau = trigger->elevated_threshold * tau;
    
    if (kl_divergence > elevated_tau) {
        trigger->consecutive_elevated++;
    } else {
        trigger->consecutive_elevated = 0;  // Reset on any non-elevated tick
    }
    
    // Only trigger if:
    // 1. Above full threshold, AND
    // 2. Elevated for grace_window consecutive checks
    result.should_trigger = (kl_divergence > tau) && 
                            (trigger->consecutive_elevated >= trigger->grace_window);
    
    // Populate diagnostics
    result.kl_observed = kl_divergence;
    result.threshold = tau;
    result.consecutive = trigger->consecutive_elevated;
    
    return result;
}
```

**Recommended:** `grace_window = 5`, `elevated_threshold = 0.7`.

---

### Pitfall 5: Transition Matrix Dead Zone (Zero-Probability Trap)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      THE PARTICLE DEATH SPIRAL                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   PGAS window: 2000 ticks, all in Calm regime                          │
│   Oracle output: S[Calm→Crisis] = 0 (no transitions observed)         │
│                                                                         │
│   After SAEM:  Π[Calm→Crisis] → floor (1e-5)                           │
│                                                                         │
│   SUDDEN FLASH CRASH:                                                   │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │  All particles in Calm                                         │   │
│   │  Likelihood of Crisis observation under Calm: ~0               │   │
│   │  Weight update: w *= P(obs|Calm) ≈ 0                          │   │
│   │  All weights → 0                                               │   │
│   │  ESS → 0                                                       │   │
│   │  TOTAL FILTER COLLAPSE                                         │   │
│   └────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Risk:** Zero transition probability prevents particles from escaping to correct regime.

**Mitigation:** Global Escape Hatch (Uniform Blending)

```c
#define ESCAPE_HATCH_WEIGHT 0.01f   // 1% uniform mixture
#define TRANS_FLOOR 1e-5f

void apply_escape_hatch(float *Pi, int K) {
    float uniform = 1.0f / (float)K;
    float alpha = ESCAPE_HATCH_WEIGHT;
    
    for (int i = 0; i < K; i++) {
        float row_sum = 0.0f;
        
        for (int j = 0; j < K; j++) {
            // Blend with uniform: always some probability of any transition
            Pi[i * K + j] = (1.0f - alpha) * Pi[i * K + j] + alpha * uniform;
            
            // Also enforce hard floor
            if (Pi[i * K + j] < TRANS_FLOOR) {
                Pi[i * K + j] = TRANS_FLOOR;
            }
            
            row_sum += Pi[i * K + j];
        }
        
        // Renormalize row
        float inv_sum = 1.0f / row_sum;
        for (int j = 0; j < K; j++) {
            Pi[i * K + j] *= inv_sum;
        }
    }
}

// Apply after every SAEM update:
saem_update(saem, S_oracle, K, gamma, Pi_out);
apply_escape_hatch(Pi_out, K);
```

**Interpretation:** 
- 99% of transition probability follows learned Π
- 1% is "escape hatch" allowing jumps to any regime
- Ensures particles can always reach Crisis even if never observed in training window

---

## Pitfall Summary

| Pitfall | Symptom | Robust Mitigation | Literature | Key Parameter |
|---------|---------|-------------------|------------|---------------|
| **Confirmation Bias** | Filter+Oracle agree on wrong state | Tempered path injection (5% flips) | Chopin & Papaspiliopoulos (2020) | `flip_prob = 0.05` |
| **Count Overwhelment** | Π barely moves after regime change | Dual-conditioned reset (KL AND Hawkes) | Särkkä (2013) | `reset_threshold = 5σ` |
| **Latency Spike** | P99 latency jumps during scouts | Pin threads to separate cores | — | Core affinity |
| **Resampling Noise** | False KL triggers | Grace window requirement | — | `grace_window = 5` |
| **Dead Zone** | Particle death on rare transition | Uniform escape hatch blend | — | `escape_weight = 0.01` |

---

## Final Tier Upgrades

These upgrades address subtle failure modes that emerge in production HFT environments.

### Upgrade 1: Exponentially Weighted PGAS (Window Paradox)

**The Problem: Temporal Misalignment**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         THE LAGGED AVERAGE TRAP                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Window: T = 2000 ticks                                               │
│                                                                         │
│   tick 0        tick 1000       tick 2000                              │
│   │              │               │                                      │
│   ▼              ▼               ▼                                      │
│   ├──────────────┼───────────────┤                                      │
│   │    CALM      │    CRISIS     │                                      │
│   │   Π_calm     │   Π_crisis    │                                      │
│                  ↑                                                       │
│            Regime shift                                                 │
│                                                                         │
│   PGAS output: Π_learned ≈ 0.5 × Π_calm + 0.5 × Π_crisis               │
│                                                                         │
│   Problem: "Average" describes NEITHER regime correctly!               │
│            RBPF needs Π_crisis NOW, not Π_average                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Solution: Recency-Weighted Likelihood**

In the CSMC forward pass, weight likelihoods so recent ticks have higher importance:

```c
typedef struct {
    float recency_lambda;    // Decay rate (e.g., 0.001 = half-life ~700 ticks)
} ExponentialWeightConfig;

float compute_recency_weight(int t, int T, float lambda) {
    // Recent ticks get higher weight
    // w(t) = exp(-λ × (T - t))
    // At t=T (most recent): w = 1.0
    // At t=0 (oldest): w = exp(-λT) ≈ 0.14 for λ=0.001, T=2000
    return expf(-lambda * (float)(T - t));
}

// Modified likelihood accumulation in CSMC
void csmc_forward_weighted(PGASMKLState *state, 
                           const float *obs,
                           int T, int N,
                           float lambda) {
    for (int t = 0; t < T; t++) {
        float recency = compute_recency_weight(t, T, lambda);
        
        // Weight this timestep's contribution
        for (int n = 0; n < N; n++) {
            float log_lik = compute_log_likelihood(obs[t], state->particles[n]);
            state->log_weights[n] += recency * log_lik;  // Recency-weighted
        }
        
        // Resampling, propagation, etc.
        // ...
    }
}
```

**Trade-off Analysis:**

| Aspect | Uniform Window | Exponential Window |
|--------|---------------|-------------------|
| Theory | Valid PGAS posterior | Approximate (biased toward recent) |
| Regime shift | Lagged average | Prioritizes "Now" |
| Statistical power | Full T ticks | Effective ~T/2 ticks |
| Non-stationary data | Poor | **Good** |

**Recommended:** `recency_lambda = 0.001` (half-life ≈ 700 ticks)

---

### Upgrade 2: Mixing-Aware Scout Verification

**The Problem: Degenerate Scout**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      THE DEGENERATE SCOUT TRAP                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   PARIS with N=32, 10 sweeps:                                          │
│                                                                         │
│   Sweep 1:  [path A] [path A] [path A] ... (all same ancestor)        │
│   Sweep 2:  [path A] [path A] [path A] ...                             │
│   ...                                                                   │
│   Sweep 10: [path A] [path A] [path A] ...                             │
│                                                                         │
│   Acceptance rate: 2% (barely any new paths accepted)                  │
│   Entropy: LOW (all paths identical)                                   │
│                                                                         │
│   Scout verdict: "Filter is CONFIDENT" ← WRONG!                        │
│   Reality: Sampler is DEGENERATE, not confident                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Solution: Validate Scout Mixing Before Trusting Entropy**

```c
typedef struct {
    float entropy;
    float acceptance_rate;
    int   unique_paths;
    bool  is_valid;
} ScoutResult;

ScoutResult scout_sweep_with_validation(PARISState *paris,
                                        const float *obs,
                                        int T, int N, int K,
                                        int n_sweeps,
                                        float min_acceptance,
                                        float min_unique_fraction) {
    ScoutResult result = {0};
    
    int total_proposals = 0;
    int total_accepts = 0;
    
    for (int sweep = 0; sweep < n_sweeps; sweep++) {
        int accepts = paris_backward_sweep(paris, obs, T, N, K);
        total_accepts += accepts;
        total_proposals += T;  // One proposal per timestep
    }
    
    result.acceptance_rate = (float)total_accepts / (float)total_proposals;
    result.entropy = compute_path_entropy(paris);
    result.unique_paths = count_unique_paths(paris, T);
    
    // ═══════════════════════════════════════════════════════════════════
    // VALIDITY CHECK: Is the scout's opinion trustworthy?
    // ═══════════════════════════════════════════════════════════════════
    
    if (result.acceptance_rate < min_acceptance) {
        // Sampler stuck - can't explore path space
        result.is_valid = false;
    } else if (result.unique_paths < (int)(min_unique_fraction * N)) {
        // Path collapse - apparent confidence is actually degeneracy
        result.is_valid = false;
    } else {
        result.is_valid = true;
    }
    
    return result;
}

// In trigger logic:
ScoutResult scout = scout_sweep_with_validation(paris, obs, T, N, K, 
                                                 10,     // sweeps
                                                 0.10f,  // min 10% acceptance
                                                 0.25f); // min 25% unique paths

if (!scout.is_valid) {
    // Scout degenerate - can't trust entropy signal
    // Conservative action: Force full PGAS anyway
    trigger_reason = TRIGGER_INVALID_SCOUT;
    trigger_full_pgas = true;
} else {
    // Scout valid - use entropy signal normally
    trigger_full_pgas = (scout.entropy > entropy_threshold);
}
```

**Recommended:** `min_acceptance = 0.10`, `min_unique_fraction = 0.25`

---

### Upgrade 3: Dual-Gate Trigger (Variance + Absolute Panic)

**The Problem: Variance Explosion Causes Blindness**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      THE BLIND ORACLE PARADOX                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Normal market:                                                        │
│      σ_H = 0.1 nats                                                    │
│      Threshold = 2.5 × 0.1 = 0.25 nats                                 │
│      → Oracle triggers on KL > 0.25 ✓                                  │
│                                                                         │
│   Flash crash (chaos):                                                  │
│      σ_H EXPLODES to 2.0 nats (SMC struggling)                         │
│      Threshold = 2.5 × 2.0 = 5.0 nats                                  │
│      → Oracle needs KL > 5.0 to trigger                                │
│      → Actual KL is "only" 3.0 nats                                    │
│      → NO TRIGGER despite obvious market chaos!                        │
│                                                                         │
│   The oracle goes BLIND exactly when needed most!                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Solution: Dual-Gate with σ Cap and Absolute Panic**

```c
typedef struct {
    // Variance-based (relative) gate
    float z_score;
    float H_var_ema;
    float sigma_H_max;            // CAP: Never let σ exceed this (e.g., 1.0 nats)
    
    // Absolute panic gate (safety net)
    float absolute_panic_kl;      // e.g., 2.0 nats
    float absolute_panic_hawkes;  // e.g., 10× baseline intensity
    
} DualGateTrigger;

typedef struct {
    bool  should_trigger;
    bool  relative_fired;         // Which gate triggered?
    bool  absolute_fired;
    float threshold_used;
} DualGateResult;

DualGateResult dual_gate_check(DualGateTrigger *trigger,
                               float kl_divergence,
                               float hawkes_integrated,
                               float hawkes_baseline) {
    DualGateResult result = {0};
    
    // ═══════════════════════════════════════════════════════════════════
    // GATE 1: Relative (variance-based) with σ CAP
    // ═══════════════════════════════════════════════════════════════════
    float sigma_H = sqrtf(trigger->H_var_ema);
    
    // CRITICAL: Cap σ to prevent blindness during chaos
    if (sigma_H > trigger->sigma_H_max) {
        sigma_H = trigger->sigma_H_max;
    }
    
    float relative_threshold = trigger->z_score * sigma_H;
    result.threshold_used = relative_threshold;
    result.relative_fired = (kl_divergence > relative_threshold);
    
    // ═══════════════════════════════════════════════════════════════════
    // GATE 2: Absolute panic (safety net - always responsive)
    // ═══════════════════════════════════════════════════════════════════
    bool kl_panic = (kl_divergence > trigger->absolute_panic_kl);
    bool hawkes_panic = (hawkes_integrated > trigger->absolute_panic_hawkes * hawkes_baseline);
    
    result.absolute_fired = kl_panic || hawkes_panic;
    
    // ═══════════════════════════════════════════════════════════════════
    // EITHER gate fires → trigger
    // ═══════════════════════════════════════════════════════════════════
    result.should_trigger = result.relative_fired || result.absolute_fired;
    
    return result;
}
```

**Why σ Cap is Essential:**

```
Without cap:  σ_H = 5.0 → threshold = 2.5 × 5.0 = 12.5 nats (blind!)
With cap:     σ_H = min(5.0, 1.0) = 1.0 → threshold = 2.5 nats (responsive)
```

**Recommended:** `sigma_H_max = 1.0`, `absolute_panic_kl = 2.0`, `absolute_panic_hawkes = 10.0`

---

### Upgrade 4: Innovation-Targeted Partial Reset

**The Problem: SAEM Can't Track Phase Shifts**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   THE SMOOTH BLENDER VS PHASE SHIFT                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   SAEM smooth blending:                                                │
│      Q_new = (1 - γ) × Q_old + γ × S_oracle                           │
│                                                                         │
│   Works for: Gradual drift (Π evolves slowly)                          │
│   Fails for: Phase shift (Π jumps discontinuously)                    │
│                                                                         │
│   Example phase shift:                                                  │
│      Π_rbpf[Calm→Calm] = 0.95                                          │
│      Π_oracle[Calm→Calm] = 0.70  (25% absolute jump!)                  │
│                                                                         │
│   With γ=0.1:                                                          │
│      Update 1: 0.95 → 0.925                                            │
│      Update 2: 0.925 → 0.9025                                          │
│      ...                                                                │
│      Need ~30 updates to close gap!                                    │
│                                                                         │
│   Market doesn't wait 30 oracle cycles!                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Solution: Three-Tier Reset Hierarchy**

```c
typedef struct {
    float Q[PGAS_MAX_K * PGAS_MAX_K];
    float alpha_prior;
    float kappa_prior;
    float Q_max;
    
    // Innovation tracking
    float innovation_history[100];
    int   innovation_idx;
    float innovation_p99;           // 99th percentile (computed online)
    
    // Partial reset config
    float partial_reset_fraction;   // 0.5 = forget 50% of history
    
    // Dual-conditioned reset thresholds
    float reset_kl_threshold;
    float reset_hawkes_threshold;
    
} SAEMBlenderFinal;

float compute_innovation(const float *Pi_oracle, const float *Pi_rbpf, int K) {
    float norm = 0.0f;
    for (int i = 0; i < K * K; i++) {
        float diff = Pi_oracle[i] - Pi_rbpf[i];
        norm += diff * diff;
    }
    return sqrtf(norm);  // Frobenius norm
}

void saem_update_final(SAEMBlenderFinal *saem,
                       const int *S_oracle,
                       const float *Pi_oracle,
                       const float *Pi_rbpf,
                       int K,
                       float gamma,
                       SurpriseMetrics surprise,
                       float *Pi_out) {
    
    // Compute innovation
    float innovation = compute_innovation(Pi_oracle, Pi_rbpf, K);
    
    // Update innovation history (rolling buffer for P99)
    saem->innovation_history[saem->innovation_idx] = innovation;
    saem->innovation_idx = (saem->innovation_idx + 1) % 100;
    saem->innovation_p99 = compute_percentile_99(saem->innovation_history, 100);
    
    bool kl_extreme = (surprise.kl_sigma > saem->reset_kl_threshold);
    bool hawkes_extreme = (surprise.hawkes_sigma > saem->reset_hawkes_threshold);
    
    // ═══════════════════════════════════════════════════════════════════
    // TIER 3: FULL RESET (structural break - both signals confirm)
    // ═══════════════════════════════════════════════════════════════════
    if (kl_extreme && hawkes_extreme) {
        // Catastrophic reset to prior + current counts
        for (int i = 0; i < K; i++) {
            for (int j = 0; j < K; j++) {
                float prior = (i == j) ? saem->kappa_prior : saem->alpha_prior;
                saem->Q[i * K + j] = prior + (float)S_oracle[i * K + j];
            }
        }
        gamma = 0.5f;  // Immediate high-weight update
    }
    // ═══════════════════════════════════════════════════════════════════
    // TIER 2: PARTIAL RESET (phase shift - large innovation)
    // ═══════════════════════════════════════════════════════════════════
    else if (innovation > saem->innovation_p99) {
        // Forget portion of history to enable faster tracking
        float forget = saem->partial_reset_fraction;  // 0.5
        
        for (int i = 0; i < K * K; i++) {
            float prior = ((i % (K+1)) == 0) ? saem->kappa_prior : saem->alpha_prior;
            saem->Q[i] = (1.0f - forget) * saem->Q[i] + forget * prior;
        }
        gamma = fminf(gamma * 2.0f, 0.4f);  // Boost gamma
    }
    // ═══════════════════════════════════════════════════════════════════
    // TIER 1: NORMAL UPDATE (gradual drift)
    // ═══════════════════════════════════════════════════════════════════
    
    // Standard SAEM with Q_max cap
    for (int i = 0; i < K * K; i++) {
        saem->Q[i] = (1.0f - gamma) * saem->Q[i] + gamma * (float)S_oracle[i];
        if (saem->Q[i] > saem->Q_max) saem->Q[i] = saem->Q_max;
    }
    
    // Compute posterior mean
    for (int i = 0; i < K; i++) {
        float row_sum = 0.0f;
        for (int j = 0; j < K; j++) {
            row_sum += saem->Q[i * K + j];
        }
        float inv_sum = 1.0f / row_sum;
        for (int j = 0; j < K; j++) {
            Pi_out[i * K + j] = saem->Q[i * K + j] * inv_sum;
        }
    }
    
    // Apply escape hatch
    apply_escape_hatch(Pi_out, K);
}
```

**Three-Tier Reset Hierarchy:**

| Tier | Trigger | Action | Use Case |
|------|---------|--------|----------|
| **Tier 1: Normal** | Innovation < P90 | Standard γ blend | Gradual drift |
| **Tier 2: Partial** | Innovation > P99 | Forget 50% of Q, boost γ×2 | Phase shift |
| **Tier 3: Full** | KL AND Hawkes > 5σ | Reset Q to prior | Structural break |

**Recommended:** `partial_reset_fraction = 0.5`, innovation tracked over 100 samples

---

## Final Tier Summary

| Upgrade | Problem Solved | Key Parameter |
|---------|---------------|---------------|
| **Exponential Weighting** | Lagged average after regime shift | `recency_lambda = 0.001` |
| **Mixing Validation** | Degenerate scout false confidence | `min_acceptance = 0.10` |
| **Dual-Gate + σ Cap** | Oracle blindness during chaos | `sigma_H_max = 1.0`, `panic_kl = 2.0` |
| **Innovation Partial Reset** | SAEM can't track phase shifts | `partial_reset = 0.5` |

---

## Multi-Armed Bandit Framing

The Oracle integration problem is fundamentally a **Restless Bandit** problem. This framing reveals principled solutions from the MAB literature.

### The Analogy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ORACLE AS RESTLESS BANDIT                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Classic MAB:                                                             │
│      Arms = {Π_1, Π_2, ..., Π_k}                                           │
│      Pull arm → observe reward → update belief                             │
│      Goal: Minimize regret over T pulls                                    │
│                                                                             │
│   Our System:                                                               │
│      "Arm" = Current Π estimate                                            │
│      "Pull" = Run RBPF with this Π                                         │
│      "Reward" = Predictive log-likelihood (negative entropy)               │
│      "Restless" = True Π drifts independently of our actions               │
│                                                                             │
│   Key insight: We're solving a BAYESIAN BANDIT problem!                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### UCB Connection (Already Implemented)

Our z×σ_H threshold is functionally equivalent to the Upper Confidence Bound algorithm:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        UCB vs OUR THRESHOLD                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Classic UCB (for arm selection):                                         │
│                                                                             │
│      Score(arm) = μ̂(arm) + c × √(log(t) / n(arm))                         │
│                   ───────   ─────────────────────                          │
│                   exploit        explore bonus                              │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Our Trigger (for oracle activation):                                     │
│                                                                             │
│      Threshold = z × σ_H                                                   │
│                  ─   ───                                                    │
│                  confidence   uncertainty                                   │
│                                                                             │
│      Trigger if: KL > Threshold                                            │
│                                                                             │
│   This is UCB! High uncertainty (σ_H) raises the bar,                      │
│   requiring larger KL to trigger (exploit current Π longer).               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Thompson Sampling for Π (New Addition)

Instead of passing a point-estimate Π from PGAS to RBPF, we sample from the Dirichlet posterior.

**Literature:** Agrawal & Goyal (2012) establishes Thompson Sampling as nearly optimal for balancing exploration in unknown environments.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                 THOMPSON SAMPLING: AUTOMATIC EXPLORATION                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Low counts (Q[i,j] ~ 10):                                                │
│                                                                             │
│   Sample 1: Π[0→0] = 0.72    Sample 2: Π[0→0] = 0.91    (high variance)   │
│   Sample 3: Π[0→0] = 0.85    Sample 4: Π[0→0] = 0.68                       │
│                                                                             │
│   → RBPF sees different Π each oracle cycle                                │
│   → Naturally explores parameter space                                     │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   High counts (Q[i,j] ~ 1000):                                             │
│                                                                             │
│   Sample 1: Π[0→0] = 0.923   Sample 2: Π[0→0] = 0.921   (low variance)    │
│   Sample 3: Π[0→0] = 0.924   Sample 4: Π[0→0] = 0.922                      │
│                                                                             │
│   → RBPF sees stable Π                                                     │
│   → Exploits learned structure                                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Why Thompson Sampling is elegant:**

| Situation | Q Counts | Thompson Behavior |
|-----------|----------|-------------------|
| High uncertainty | Low (~10) | Wide variance → **Exploration** |
| High confidence | High (~1000) | Tight around mean → **Exploitation** |
| After reset | Prior only | Maximum exploration |

### Two Levels of Exploration

Tempered Injection and Thompson Sampling are **complementary**, not substitutes:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TWO LEVELS OF EXPLORATION                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Level 1: PATH EXPLORATION (Tempered Injection)                           │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  RBPF MAP → 5% flips → PGAS explores "what-if" regime sequences    │  │
│   │                                                                     │  │
│   │  Purpose: Prevent PGAS from rubber-stamping RBPF's mistakes        │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   Level 2: PARAMETER EXPLORATION (Thompson Sampling)                       │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  Q matrix → Sample Π* ~ Dirichlet(Q) → RBPF uses Π*                │  │
│   │                                                                     │  │
│   │  Purpose: Automatic explore/exploit based on uncertainty            │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   Combined: Full Bayesian exploration of both paths AND parameters         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Thompson Sampling Implementation

```c
// Gamma sampling via Marsaglia-Tsang method
float sample_gamma(RNG *rng, float alpha, float beta) {
    if (alpha < 1.0f) {
        // Ahrens-Dieter for alpha < 1
        float u = rng_uniform(rng);
        return sample_gamma(rng, alpha + 1.0f, beta) * powf(u, 1.0f / alpha);
    }
    
    float d = alpha - 1.0f/3.0f;
    float c = 1.0f / sqrtf(9.0f * d);
    
    while (1) {
        float x, v;
        do {
            x = rng_normal(rng);
            v = 1.0f + c * x;
        } while (v <= 0.0f);
        
        v = v * v * v;
        float u = rng_uniform(rng);
        
        if (u < 1.0f - 0.0331f * (x*x) * (x*x)) return d * v / beta;
        if (logf(u) < 0.5f * x * x + d * (1.0f - v + logf(v))) return d * v / beta;
    }
}

void thompson_sample_transition_matrix(const float *Q, 
                                        int K,
                                        float temperature,
                                        RNG *rng,
                                        float *Pi_out) {
    for (int i = 0; i < K; i++) {
        float row[PGAS_MAX_K];
        float row_sum = 0.0f;
        
        for (int j = 0; j < K; j++) {
            // Temperature-adjusted concentration
            // temp=1.0: pure Thompson
            // temp<1.0: sharper (more exploitative)
            // temp>1.0: flatter (more explorative)
            float alpha = Q[i * K + j] * temperature + (1.0f - temperature);
            if (alpha < 0.01f) alpha = 0.01f;  // Safety floor
            
            row[j] = sample_gamma(rng, alpha, 1.0f);
            row_sum += row[j];
        }
        
        // Normalize to get Dirichlet sample
        float inv_sum = 1.0f / row_sum;
        for (int j = 0; j < K; j++) {
            Pi_out[i * K + j] = row[j] * inv_sum;
        }
    }
    
    // Apply escape hatch
    apply_escape_hatch(Pi_out, K);
}
```

### Adaptive Thompson vs Mean Selection

```c
typedef enum {
    HANDOFF_MEAN,           // Posterior mean (exploitation)
    HANDOFF_THOMPSON,       // Thompson sampling (exploration)
    HANDOFF_ADAPTIVE        // Auto-select based on uncertainty
} HandoffMode;

void oracle_handoff(SAEMBlenderFinal *saem, 
                    int K, 
                    HandoffMode mode,
                    RNG *rng, 
                    float *Pi_rbpf) {
    
    if (mode == HANDOFF_ADAPTIVE) {
        // Compute effective sample size (minimum row sum)
        float min_row_sum = FLT_MAX;
        for (int i = 0; i < K; i++) {
            float row_sum = 0.0f;
            for (int j = 0; j < K; j++) {
                row_sum += saem->Q[i * K + j];
            }
            if (row_sum < min_row_sum) min_row_sum = row_sum;
        }
        
        // Low counts → Thompson (explore)
        // High counts → Mean (exploit)
        if (min_row_sum < 500.0f) {
            mode = HANDOFF_THOMPSON;
        } else {
            mode = HANDOFF_MEAN;
        }
    }
    
    if (mode == HANDOFF_THOMPSON) {
        thompson_sample_transition_matrix(saem->Q, K, 
                                          saem->thompson_temperature, 
                                          rng, Pi_rbpf);
    } else {
        compute_posterior_mean(saem->Q, K, Pi_rbpf);
    }
}
```

### When to Use Each Mode

| Situation | Recommended Mode | Rationale |
|-----------|-----------------|-----------|
| Early learning | Thompson | Explore parameter space |
| Stable market | Mean | Exploit learned Π |
| Post-reset | Thompson | High uncertainty → explore |
| High-frequency updates | Mean | Reduce variance in RBPF |
| Low-frequency updates | Thompson | Each update should explore |
| Structural break detected | Thompson | Need to find new Π quickly |

### MAB Summary

| MAB Concept | Our Implementation | Status |
|-------------|-------------------|--------|
| **UCB** | z×σ_H threshold for triggers | ✓ Already implemented |
| **Thompson Sampling** | Sample Π ~ Dirichlet(Q) for handoff | ✓ **Added** |
| **Restless Bandit** | Tempered injection + SAEM reset | ✓ Already implemented |
| **Explore/Exploit** | Adaptive handoff mode selection | ✓ **Added** |

---

## Configuration Parameters

```c
typedef struct {
    // ═══════════════════════════════════════════════
    // LAYER 1: HAWKES DETECTION
    // ═══════════════════════════════════════════════
    int   integration_window;     // 50-100 ticks
    float high_water_mark;        // Trigger scout (e.g., 2.5σ)
    float low_water_mark;         // Stay active (e.g., 1.5σ)
    int   refractory_period;      // 500 ticks minimum
    
    // ═══════════════════════════════════════════════
    // LAYER 2: SCOUT VERIFICATION  
    // ═══════════════════════════════════════════════
    int   scout_sweeps;           // 5-10 PARIS sweeps
    float entropy_threshold;      // Trigger full PGAS (e.g., 0.3 nats)
    int   grace_window;           // Consecutive elevated checks (e.g., 5)
    float elevated_threshold;     // What counts as "elevated" (e.g., 0.7)
    
    // Mixing validation (Final Tier)
    float scout_min_acceptance;   // 0.10 (10% minimum acceptance rate)
    float scout_min_unique_frac;  // 0.25 (25% unique paths required)
    
    // ═══════════════════════════════════════════════
    // LAYER 3: FULL PGAS
    // ═══════════════════════════════════════════════
    int   pgas_burnin;            // 300 sweeps
    int   pgas_samples;           // 500 sweeps
    int   pgas_window_default;    // 2000 ticks
    int   pgas_window_min;        // 500 ticks (post-jump)
    int   pgas_window_max;        // 8000 ticks (stable)
    
    // Tempered path injection (Chopin & Papaspiliopoulos 2020)
    float tempered_flip_prob;     // 0.05 (5% random regime flips)
    
    // Exponential weighting (Final Tier)
    float recency_lambda;         // 0.001 (half-life ~700 ticks)
    
    // ═══════════════════════════════════════════════
    // LAYER 4: TRIGGER GATES
    // ═══════════════════════════════════════════════
    
    // Dual-gate: Relative threshold
    float z_score;                // 2.5 (confidence level)
    float sigma_H_max;            // 1.0 (cap to prevent blindness)
    
    // Dual-gate: Absolute panic (Final Tier)
    float absolute_panic_kl;      // 2.0 nats
    float absolute_panic_hawkes;  // 10.0 (× baseline intensity)
    
    // ═══════════════════════════════════════════════
    // LAYER 5: SAEM BLENDING
    // ═══════════════════════════════════════════════
    float saem_gamma_base;        // 0.3 base step size
    float saem_gamma_min;         // 0.02 minimum step size
    float saem_Q_max;             // 10000 max count per cell (prevent ballast)
    
    // Dual-conditioned reset gate (Särkkä 2013)
    float reset_kl_threshold;     // 5.0 (5σ KL surprise)
    float reset_hawkes_threshold; // 5.0 (5σ Hawkes surprise)
    int   reset_arm_timeout;      // 50 ticks (disarm if second signal doesn't fire)
    
    // Innovation-targeted partial reset (Final Tier)
    float partial_reset_fraction; // 0.5 (forget 50% on phase shift)
    int   innovation_history_len; // 100 (samples for P99 calculation)
    
    // ═══════════════════════════════════════════════
    // LAYER 6: HANDOFF MODE (MAB / Thompson Sampling)
    // ═══════════════════════════════════════════════
    int   handoff_mode;           // 0=Mean, 1=Thompson, 2=Adaptive
    float thompson_temperature;   // 1.0 (pure Thompson), <1.0 (exploitative)
    float thompson_threshold;     // 500 (min row sum for adaptive switch)
    
    // ═══════════════════════════════════════════════
    // SAFETY RAILS
    // ═══════════════════════════════════════════════
    float trans_floor;            // 1e-5 minimum probability
    float escape_hatch_weight;    // 0.01 (1% uniform blend)
    float kappa_max;              // 300 maximum stickiness
    float target_avg_diagonal;    // 0.92-0.95 for reactivity
    
    // ═══════════════════════════════════════════════
    // THREADING (Pitfall 3 mitigation)
    // ═══════════════════════════════════════════════
    int   rbpf_core;              // Pin RBPF to this core (e.g., 0)
    int   scout_core;             // Pin Scout/PGAS to this core (e.g., 2)
    
} OracleConfig;
```

---

## Implementation Modules

| Module | File | Purpose |
|--------|------|---------|
| `HawkesIntegrator` | `hawkes_trigger.h/c` | Integrated intensity + hysteresis |
| `AdaptiveKLTrigger` | `kl_trigger.h/c` | Variance-based KL trigger + grace window |
| `DualGateTrigger` | `dual_gate.h/c` | Variance + absolute panic + σ cap |
| `ScoutSweepValidated` | `scout_sweep.h/c` | Mixing-aware entropy verification |
| `TemperedPath` | `tempered_path.h/c` | Reference path jitter (5% flips) |
| `ExponentialCSMC` | `pgas_mkl.h/c` | Recency-weighted likelihood |
| `SAEMBlenderFinal` | `saem_blender.h/c` | Three-tier reset (normal/partial/full) |
| `ThompsonSampler` | `thompson_sampler.h/c` | Dirichlet sampling for Π exploration |
| `AdaptiveHandoff` | `handoff.h/c` | Mean vs Thompson mode selection |
| `EscapeHatch` | `transition_safety.h/c` | Uniform blend + floor enforcement |
| `OracleScheduler` | `oracle_scheduler.h/c` | Main coordinator |
| `TransitionLearner` | `pgas_learner.h/c` | PGAS wrapper for Π learning |

---

## Implementation Order

**Phase 1: Core Triggers**
1. **`HawkesIntegrator`** — Integrated intensity with hysteresis thresholds
2. **`AdaptiveKLTrigger`** — Variance-based KL trigger with grace window
3. **`DualGateTrigger`** — Combine relative + absolute panic + σ cap

**Phase 2: Scout Verification**
4. **`ScoutSweepValidated`** — Mixing-aware entropy with acceptance/uniqueness checks
5. **`TemperedPath`** — Reference path jitter for exploration (5% flips)

**Phase 3: PGAS Enhancements**
6. **`ExponentialCSMC`** — Recency-weighted likelihood in forward pass
7. **`SAEMBlenderFinal`** — Three-tier reset hierarchy (normal/partial/full)
8. **`EscapeHatch`** — Uniform blend + transition floor enforcement

**Phase 4: MAB / Thompson Sampling**
9. **`ThompsonSampler`** — Dirichlet sampling via Marsaglia-Tsang gamma
10. **`AdaptiveHandoff`** — Mode selection based on effective sample size

**Phase 5: Integration**
11. **`OracleScheduler`** — Main coordinator tying components together
12. **Core pinning** — Thread affinity for RBPF/Scout isolation
13. **Integration test** — Synthetic data with known regime changes + phase shifts
14. **Production integration** — Connect to live RBPF

---

## Literature References

| Concept | Source | Application |
|---------|--------|-------------|
| Warm Start Initialization | Lindsten et al. (2014) "Particle Gibbs with Ancestor Sampling" | RBPF MAP path → PGAS reference |
| Bayesian Ensemble Averaging | Schön et al. (2015) "Sequential Monte Carlo Methods" | E[Π] from PGAS samples → RBPF |
| **Tempered Path Injection** | Chopin & Papaspiliopoulos (2020) "An Introduction to SMC" | Reference path jitter for exploration |
| **Covariance Resetting** | Särkkä (2013) "Bayesian Filtering and Smoothing" | Dual-conditioned reset gate |
| Markov-Modulated Hawkes | Liniger (2009), Pierre (2025) | Intensity triggering for regime detection |
| Jump-Markov Regularization | Doucet et al. (2001) "Sequential Monte Carlo Methods in Practice" | Particle migration during Π shifts |
| RLS with Forgetting | Haykin (2002) "Adaptive Filter Theory" | Baseline parameter handoff |
| Information-Theoretic Triggers | Kullback & Leibler (1951), Cover & Thomas (2006) | KL divergence for model mismatch |
| Variance-Based Thresholds | Chopin (2004) "Central Limit Theorem for SMC" | Self-calibrating trigger thresholds |
| Stochastic Approximation | Robbins & Monro (1951), Polyak & Juditsky (1992) | Adaptive step size γ decay |
| **SAEM Algorithm** | Delyon et al. (1999) "Convergence of a Stochastic Approximation Version of the EM Algorithm" | Sufficient statistics blending |
| Dirichlet-Multinomial | Gelman et al. (2013) "Bayesian Data Analysis" | Conjugate prior for transition matrix |
| Iterated Filtering (IF2) | Ionides et al. (2015) | Parameter exploration via artificial noise |
| **Exponential Forgetting** | Ljung & Söderström (1983) "Theory and Practice of Recursive Identification" | Recency-weighted estimation |
| **SMC Mixing Diagnostics** | Andrieu et al. (2010) "Particle Markov Chain Monte Carlo Methods" | Acceptance rate validation |
| **Innovation-Based Adaptation** | Mehra (1972) "Approaches to Adaptive Filtering" | Phase shift detection via innovation |
| **Thompson Sampling** | Agrawal & Goyal (2012) "Analysis of Thompson Sampling for the Multi-Armed Bandit Problem" | Dirichlet sampling for explore/exploit |
| **UCB Algorithm** | Auer et al. (2002) "Finite-time Analysis of the Multiarmed Bandit Problem" | Uncertainty bonus in triggers |
| **Restless Bandits** | Whittle (1988) "Restless Bandits: Activity Allocation in a Changing World" | Non-stationary parameter tracking |
| **Gamma Sampling** | Marsaglia & Tsang (2000) "A Simple Method for Generating Gamma Variables" | Efficient Dirichlet sampling |

---

## Success Criteria

| Metric | Target | Measurement |
|--------|--------|-------------|
| False trigger rate | < 5% | Scout sweeps that don't lead to full PGAS |
| Regime detection lag | < 30 ticks | Time from true change to RBPF adaptation |
| Π estimation error | < 0.10 Frobenius | PGAS vs ground truth |
| Hot path latency | < 5µs | RBPF tick processing time |
| Oracle latency | < 200ms | Full PGAS sweep time |

---

## Dual-Trigger Philosophy: Lead + Truth

The system uses two complementary triggers with distinct failure modes:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│                    HAWKES (Lead)              KL (Truth)                    │
│                         │                         │                         │
│                         │                         │                         │
│    "Something is        │                         │    "The filter IS       │
│     happening"          │                         │     confused"           │
│                         │                         │                         │
│    External signal      │                         │    Internal signal      │
│    (market physics)     │                         │    (filter state)       │
│                         │                         │                         │
│    ┌────────────────────┴─────────────────────────┴───────────────────┐    │
│    │                                                                   │    │
│    │                         EITHER                                    │    │
│    │                           │                                       │    │
│    │                           ▼                                       │    │
│    │                     SCOUT SWEEP                                   │    │
│    │                           │                                       │    │
│    │                           ▼                                       │    │
│    │                      FULL PGAS                                    │    │
│    │                                                                   │    │
│    └───────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Why Two Triggers?

| Trigger | Strength | Weakness | Failure Mode |
|---------|----------|----------|--------------|
| **Hawkes** | Early warning | Needs market physics | False positive on normal spike |
| **KL** | Ground truth | Reactive (lag) | Already confused when it fires |

**Combined:** Hawkes gives you a head start, KL gives you certainty. Neither can fool you alone.

### Trigger Comparison

| Strategy | Signal Source | Robustness | Accuracy |
|----------|---------------|------------|----------|
| Fixed Clock | Timer | High (consistent) | Low (lags shifts) |
| ESS Trigger | Internal RBPF Confusion | Medium | Medium |
| Hawkes Spike | Instant Market Intensity | Low (overreacts) | High (speed) |
| **Integrated Hawkes** | Sustained Market Activity | **High** | **High** |
| **KL Divergence** | Filter vs Smoother | **High** | **High** |

---

## Self-Calibrating Thresholds (Variance-Based)

### The Problem with Fixed Thresholds

A fixed threshold (e.g., KL > 0.3 nats) is a heuristic that doesn't account for the inherent noise in Monte Carlo estimates:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        FIXED THRESHOLD (problematic)                        │
│                                                                             │
│   KL                                                                        │
│    │                                                                        │
│    │  ─────────────────────────────── τ = 0.3 (magic number)               │
│    │         ╱╲      ╱╲                                                     │
│    │    ╱╲  ╱  ╲ ╱╲ ╱  ╲   ╱╲                                              │
│    │───╱──╲╱────╳──╳────╲─╱──╲──────────                                   │
│    │                                                                        │
│    └────────────────────────────────────────────────────────────────────▶  │
│                                                                             │
│   Problem: Triggers on noise during volatile periods                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### The Solution: Variance-Adaptive Thresholds

The entropy H(t) is a Monte Carlo estimate with its own variance. Use this variance to define the threshold:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       ADAPTIVE THRESHOLD (principled)                       │
│                                                                             │
│   KL                                                                        │
│    │         ┌─────────────┐                                               │
│    │  ───────┤  τ = z·σ_H  ├─────────   (widens during volatility)        │
│    │         └──╱╲─────────┘                                               │
│    │    ╱╲     ╱  ╲ ╱╲                     ╱╲                               │
│    │───╱──╲───╱────╳──╲────────────────╳─╱──╲──                            │
│    │                                   ↑                                    │
│    └───────────────────────────────────│────────────────────────────────▶  │
│                                        │                                    │
│   Only this one triggers (structural)  │                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Mathematical Foundation

The particle entropy is a random variable:

```
H_t = -Σ w_t^i × ln(w_t^i)
```

Because H_t is estimated from random samples, it has asymptotic variance. The threshold becomes:

```
τ_t = z × √Var(H)_t
```

Where z is the confidence level (2.0 = 95%, 2.5 = 99%, 3.0 = 99.7%).

### Self-Calibration Property

| Market State | N_eff | σ_H | Threshold τ | Behavior |
|--------------|-------|-----|-------------|----------|
| **Calm** | High | Low | Tight | Sensitive to small structural changes |
| **Volatile** | Low | High | Wide | Ignores noise from poor particle health |
| **Crisis** | Collapsed | Very High | Maximum | Only triggers on catastrophic mismatch |

**Key insight:** The filter reports its own uncertainty. When confident, we listen to small signals. When struggling, we ignore noise.

### Implementation

```c
typedef struct {
    // Entropy tracking
    float H_ema;              // Smoothed entropy estimate
    float H_var_ema;          // Smoothed variance of entropy
    int   warmup_ticks;       // Ticks to accumulate before triggering
    int   tick_count;
    
    // Confidence level
    float z_score;            // 2.0 = 95%, 2.5 = 99%, 3.0 = 99.7%
    
    // Bounds (safety)
    float tau_floor;          // Minimum threshold (prevent hyper-sensitivity)
    float tau_ceiling;        // Maximum threshold (prevent blindness)
} AdaptiveKLTrigger;

KLTriggerResult adaptive_kl_update(AdaptiveKLTrigger *trigger,
                                    float kl_divergence,
                                    float current_entropy) {
    // EMA update for entropy mean and variance
    float alpha = 0.02f;  // Slow adaptation
    
    float H_delta = current_entropy - trigger->H_ema;
    trigger->H_ema += alpha * H_delta;
    
    // Variance: E[(H - μ)²]
    float var_sample = H_delta * H_delta;
    trigger->H_var_ema += alpha * (var_sample - trigger->H_var_ema);
    
    // Compute adaptive threshold
    float sigma_H = sqrtf(trigger->H_var_ema);
    float tau = trigger->z_score * sigma_H;
    
    // Clamp to safety bounds
    if (tau < trigger->tau_floor) tau = trigger->tau_floor;
    if (tau > trigger->tau_ceiling) tau = trigger->tau_ceiling;
    
    // Trigger decision
    result.should_trigger = (kl_divergence > tau);
    return result;
}
```

---

## Unified Threshold System

All triggers use the same variance-based approach:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         UNIFIED TRIGGER SYSTEM                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   HAWKES (Lead)                        KL (Truth)                          │
│   ─────────────                        ──────────                          │
│   Threshold: ∫λdt > μ_λ + z·σ_λ        Threshold: KL > z·σ_H               │
│              ↑                                     ↑                        │
│              │                                     │                        │
│              └──── Both derived from ─────────────┘                        │
│                    their own variance                                       │
│                                                                             │
│   No magic numbers. Self-calibrating.                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

| Trigger | Threshold | Meaning |
|---------|-----------|---------|
| **Hawkes** | `z × σ_λ` | Integrated intensity beyond normal clustering |
| **KL** | `z × σ_H` | Divergence beyond normal filter noise |
| **Scout entropy** | `z × σ_ΔH` | Entropy change beyond sweep variance |

**All three use the same z-score.** One knob controls sensitivity across the entire system.

---

## The Two Knobs: Simplified Control

The entire Oracle system reduces to two statistically meaningful parameters:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         KNOB 1: CONFIDENCE (z-score)                        │
│                                                                             │
│   "How many sigma before we believe it's real?"                            │
│                                                                             │
│   Stable ◄────────────────────────────────────────────────────► Reactive   │
│      │                                                               │      │
│   z=3.0                           z=2.5                           z=2.0    │
│   99.7%                            99%                             95%     │
│   Fewer triggers                                            More triggers  │
│   Risk: Stale Π                                             Risk: Jitter   │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                         KNOB 2: MEMORY (α for variance EMA)                 │
│                                                                             │
│   "How fast do we update our uncertainty estimate?"                        │
│                                                                             │
│   Long ◄──────────────────────────────────────────────────────► Short      │
│      │                                                               │      │
│   α=0.001                         α=0.01                         α=0.05   │
│   ~1000 tick memory               ~100 tick memory              ~20 ticks  │
│   Stable variance                                           Fast-adapting  │
│   Risk: Lag                                                  Risk: Noise   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Configuration

```c
typedef struct {
    // ═══════════════════════════════════════════════
    // KNOB 1: CONFIDENCE (z-score for all triggers)
    // ═══════════════════════════════════════════════
    // 2.0 = 95% confidence (more triggers, faster reaction)
    // 2.5 = 99% confidence (balanced)
    // 3.0 = 99.7% confidence (fewer triggers, stable)
    float z_score;
    
    // ═══════════════════════════════════════════════
    // KNOB 2: MEMORY (EMA alpha for variance estimation)
    // ═══════════════════════════════════════════════
    // 0.05 = fast adaptation (recent ~20 ticks dominate)
    // 0.01 = medium adaptation (recent ~100 ticks dominate)
    // 0.001 = slow adaptation (recent ~1000 ticks dominate)
    float alpha_variance;
    
} OracleKnobs;
```

### Production Presets

```c
// Conservative: Stable params, trust historical data
OracleKnobs PRESET_CONSERVATIVE = { .z_score = 3.0f, .alpha_variance = 0.005f };

// Balanced: Default for most markets
OracleKnobs PRESET_BALANCED = { .z_score = 2.5f, .alpha_variance = 0.01f };

// Aggressive: React fast, short memory (volatile markets)
OracleKnobs PRESET_AGGRESSIVE = { .z_score = 2.0f, .alpha_variance = 0.02f };

// Crisis mode: Maximum reactivity
OracleKnobs PRESET_CRISIS = { .z_score = 1.5f, .alpha_variance = 0.05f };
```

### Derived Parameters

All other parameters are derived from these two knobs:

```c
OracleConfig oracle_config_from_knobs(OracleKnobs knobs) {
    OracleConfig cfg;
    
    // All thresholds use the same z-score
    cfg.z_score = knobs.z_score;
    
    // All variance trackers use the same alpha
    cfg.alpha_variance = knobs.alpha_variance;
    
    // Derived: Memory-related parameters
    float memory = 1.0f / (knobs.alpha_variance * 100.0f);  // 0-1 scale
    cfg.rls_lambda         = 0.90f + 0.09f * memory;        // 0.90 - 0.99
    cfg.pgas_window        = (int)(500 + 7500 * memory);    // 500 - 8000
    cfg.refractory_period  = (int)(200 + 800 * memory);     // 200 - 1000
    
    // Fixed safety rails
    cfg.trans_floor        = 1e-5f;
    cfg.kappa_max          = 300.0f;
    cfg.tau_floor          = 0.05f;
    cfg.tau_ceiling        = 1.0f;
    
    return cfg;
}
```

---

## Why This Approach is "Heuristic-Free"

| Component | Before (Heuristic) | After (Principled) |
|-----------|-------------------|-------------------|
| **KL threshold** | 0.3 nats (magic) | z × σ_H (self-calibrating) |
| **Hawkes threshold** | 2.5σ (arbitrary) | z × σ_λ (same z as KL) |
| **Scout entropy** | 0.5 nats (guess) | z × σ_ΔH (consistent) |
| **Sensitivity** | Multiple thresholds | Single z-score |
| **Memory** | Multiple time constants | Single α |

### Comparison of Tuning Requirements

| Component | Mathematical Basis | Remaining "Heuristic" |
|-----------|-------------------|----------------------|
| KL Trigger | Information Entropy | **z-score choice** (statistically meaningful) |
| Hawkes Trigger | Point Process Theory | **z-score choice** (same as KL) |
| Scout Sweep | Bayesian Verification | **z-score choice** (consistent) |
| RLS Blender | Least Squares | **α choice** (memory horizon) |

**Result:** Two knobs with clear statistical interpretation. No magic numbers.

---

## Alternatives Considered

### Option 1: SMC-SA (Stochastic Gradient)

```
PGAS outputs: ∇Π = ∂log p(y|Π) / ∂Π   (score function gradient)
RBPF updates: Π_t = Π_{t-1} + γ_t × ∇Π  (Robbins-Monro step)
```

| Aspect | Assessment |
|--------|------------|
| Theory | Optimal asymptotic convergence |
| Gradient computation | Requires Fisher score from PGAS path |
| Stability | **Gradients can explode** with sticky matrices |
| Step size | Critical tuning (γ too big → oscillation) |

**Verdict:** Too fragile for production HFT. Debugging gradient explosions under time pressure is impractical.

### Option 2: Variational Bridge (KL Only)

Monitor KL divergence between filtered and smoothed distributions continuously.

| Aspect | Assessment |
|--------|------------|
| Theory | Information-theoretic optimality |
| Computation | PARIS scout needed frequently |
| Signal quality | Ground truth of filter state |
| Independence | Self-contained (no market model) |

**Verdict:** Good as "Truth" signal, but lacks early warning. Filter is already confused when KL fires.

### Option 3: Hawkes Only

Trigger based on integrated Hawkes intensity exceeding threshold.

| Aspect | Assessment |
|--------|------------|
| Theory | Point process for event clustering |
| Early warning | **Excellent** - sees market change first |
| False positives | Can trigger on normal spikes |
| Domain knowledge | Needs α, β calibration |

**Verdict:** Good as "Lead" signal, but can be fooled by transient spikes.

### Chosen: Dual-Trigger (Hawkes + KL)

Combine both signals:
- **Hawkes:** Early warning (external market signal)
- **KL:** Ground truth (internal filter state)

**Why this wins:**
1. Complementary failure modes - neither can fool you alone
2. Self-calibrating thresholds (same z-score for both)
3. Two knobs instead of many magic numbers

---

---

## Final Robust Architecture Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      PRODUCTION-GRADE ORACLE INTEGRATION                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  DUAL-GATE TRIGGERS (Lead + Truth + Panic):                                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Hawkes (Lead)  ──┬──▶ Scout ──▶ PGAS ──▶ SAEM ──▶ Handoff ──▶ Π  │   │
│  │  KL (Truth)     ──┤     │                           │              │   │
│  │  Absolute Panic ──┘     │                           ▼              │   │
│  │                         ▼                    ┌─────────────┐       │   │
│  │  Thresholds:   Relative: z × min(σ_H, σ_max) │  Thompson   │       │   │
│  │                Absolute: KL > 2.0 nats       │  Sampling   │       │   │
│  │  Grace window: 5 consecutive elevated checks │  (adaptive) │       │   │
│  │                                              └─────────────┘       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  SCOUT VERIFICATION (Mixing-Aware):                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  10 PARIS sweeps ──▶ Check acceptance ≥ 10%                        │   │
│  │                  ──▶ Check unique paths ≥ 25%                       │   │
│  │                  ──▶ If invalid: force full PGAS anyway            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  PGAS REFERENCE PATH (Chopin & Papaspiliopoulos 2020):                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  RBPF MAP path ──▶ Tempered Injection (5% flips) ──▶ PGAS ref     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  EXPONENTIAL CSMC (Non-Stationary Adaptation):                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Likelihood weighted by recency: w(t) = exp(-λ(T-t))               │   │
│  │  λ = 0.001 → half-life ~700 ticks → prioritizes "Now"              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  SAEM THREE-TIER RESET (Delyon 1999 + Särkkä 2013):                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Tier 1 (Normal):   γ blend              [Innovation < P90]        │   │
│  │  Tier 2 (Partial):  Forget 50%, γ×2      [Innovation > P99]        │   │
│  │  Tier 3 (Full):     Reset to prior       [KL AND Hawkes > 5σ]      │   │
│  │                                                                     │   │
│  │  Safeguards: Q_max cap, escape hatch (1%), trans floor (1e-5)      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  THOMPSON SAMPLING HANDOFF (Agrawal & Goyal 2012):                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Low counts (< 500):  Π* ~ Dirichlet(Q)  [Explore]                 │   │
│  │  High counts (≥ 500): Π* = E[Q]          [Exploit]                 │   │
│  │                                                                     │   │
│  │  Automatic explore/exploit without explicit scheduling              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  THREADING:                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Core 0: RBPF hot path (isolated, never preempted)                 │   │
│  │  Core 2: Scout sweeps + PGAS oracle (separate L2 cache)            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  TWO PRIMARY KNOBS: z_score (confidence), α_variance (memory)              │
│  SAFETY OVERRIDES: σ_H cap, absolute panic thresholds                      │
│  MAB FRAMEWORK: UCB triggers, Thompson handoff, restless adaptation        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Appendix: Hawkes Process Primer

A **Hawkes process** models event arrivals where past events increase the probability of future events:

```
λ(t) = μ + Σ α × exp(-β × (t - t_i))
       ↑       ↑           ↑
    baseline  excitation   decay
              strength     rate
```

**Key insight:** Spikes are *expected* under Hawkes. The anomaly is when intensity **doesn't decay** as predicted—signaling a structural regime change rather than normal clustering.

```
Normal Hawkes behavior:          Structural change:
                                 
    │  ╱╲                           │  ╱╲
    │ ╱  ╲                          │ ╱  ╲____________________
    │╱    ╲____                     │╱                        
    └────────────▶                  └────────────▶
    
    Spike + decay = expected        Spike + plateau = anomaly
    → Don't trigger                 → Trigger PGAS
```

---

*Document version: 1.7*  
*Last updated: December 2025*  
*Status: Production-ready — Dual-trigger + SAEM + Final Tier + MAB/Thompson Sampling*