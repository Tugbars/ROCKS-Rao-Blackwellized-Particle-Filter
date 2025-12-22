# Volatility Filter Architecture
## RBPF + PARIS + PGAS + Lifeboat

---

## Executive Summary

A regime-switching stochastic volatility filter with two threads:

| Thread | Components | Frequency | What it learns |
|--------|------------|-----------|----------------|
| **Live** | RBPF → PARIS → Kelly | Every tick (~1ms) | States (h, r), emission params (μ_vol, σ_vol) |
| **Background** | PGAS → Lifeboat | Periodic (~50-500ms) | Transition matrix Π, validates K |

**The key insight**: RBPF adapts everything it can online. PGAS handles what RBPF cannot adapt: the transition structure Π and regime count K.

---

## The Model

```
Observation:  y_t = exp(h_t/2) · ε_t           ε_t ~ N(0,1)
Volatility:   h_t = μ[r_t] + φ·(h_{t-1} - μ[r_{t-1}]) + σ·η_t
Regime:       r_t ~ Categorical(Π[r_{t-1}, :])
```

| Symbol | Meaning | Learned by |
|--------|---------|------------|
| h_t | Log-volatility (continuous state) | RBPF particles |
| r_t | Regime index (discrete state) | RBPF particles |
| μ[k] | Mean log-vol for regime k | Storvik (live) |
| σ[k] | Vol-of-vol for regime k | Storvik (live) |
| φ | AR(1) persistence | Fixed or Storvik |
| Π | K×K transition matrix | PGAS (background) |
| K | Number of regimes | PGAS validates |

---

## Part I: Live Thread

### 1.1 RBPF (Forward Filter)

**Purpose**: Estimate current state given past observations.

**Output**: Filtering distribution P(h_t, r_t | y_{1:t})

**What it does each tick**:
```
1. PREDICT: Propagate particles through transition
   r_t^n ~ Categorical(Π[r_{t-1}^n, :])
   h_t^n ~ AR(1) dynamics given r_t^n

2. UPDATE: Reweight by observation likelihood
   w_t^n ∝ w_{t-1}^n × P(y_t | h_t^n)

3. RESAMPLE: If ESS < threshold
```

**Problem it has**: Path degeneracy — all particles trace back to few ancestors over time.

---

### 1.2 PARIS (Backward Smoother)

**Purpose**: Fix path degeneracy by resampling ancestors using future information.

**Output**: Smoothing distribution P(h_t, r_t | y_{1:T})

**What it does**:
```
For t = T-1 down to 0:
    For each particle n at time t+1:
        Resample ancestor at time t using backward weights:
        w_backward(i) ∝ w_t^i × P(r_{t+1} | r_t^i) × P(h_{t+1} | h_t^i)
```

**Critical**: PARIS does NOT change particle values. It only RESELECTS which trajectories are kept.

**Runs with RBPF**: Every tick on live thread, not in background.

---

### 1.3 Storvik (Parameter Learning — Live)

**Purpose**: Learn emission parameters (μ_vol, σ_vol) per regime online.

**Method**: Sufficient statistics with NIG conjugacy.

**What it tracks per particle per regime**:
```c
StorvikSoA {
    m[particle][regime];      // Posterior mean of μ
    kappa[particle][regime];  // Precision count  
    alpha[particle][regime];  // Shape for σ²
    beta[particle][regime];   // Rate for σ²
}
```

**Key feature**: Exponential forgetting prevents fossilization.
- λ = 0.997 → N_eff ≈ 333 ticks memory
- Crisis regimes can forget faster than calm regimes

**What Storvik provides**:
- Point estimates: μ_vol[k], σ_vol[k]
- Posterior uncertainty: σ(μ), σ(σ²) — useful for Lifeboat validation

---

### 1.4 Silverman Bandwidth

**Purpose**: Data-driven kernel bandwidth for post-resample jitter.

**Why needed**: After resampling, particles are duplicated. Jitter prevents collapse.

**Formula**: 
```
h = 0.9 × min(σ, IQR/1.34) × N^{-1/5}
```

**Adaptive**: Responds to particle spread automatically.

---

### 1.5 Fisher-Rao Geodesic

**Purpose**: Principled state blending when particle transitions to new regime.

**Problem it solves**: When particle moves from regime 0 to regime 3, how much should state "teleport" vs "persist"?

**Solution**: Precision-weighted interpolation on Gaussian manifold.
```
t = σ²_target / (σ²_particle + σ²_target)
```
- Uncertain particle → large t → teleport toward regime mean
- Confident particle → small t → preserve current state

---

### 1.6 SPRT (Regime Detection)

**Purpose**: Statistically principled regime switching with error control.

**Method**: Sequential Probability Ratio Test — accumulates log-likelihood ratios until decisive.

**What it provides**:
```c
double evidence[K];     // Per-regime confidence [0,1]
int current_regime;     // SPRT's best estimate
int ticks_in_current;   // Dwell time (prevents chatter)
```

**Why not BOCPD**: Tried it, triggers too fast or too slow. SPRT with pairwise tests is more reliable.

---

### 1.7 What RBPF Holds FIXED

| Parameter | Status | Why |
|-----------|--------|-----|
| Π (transition matrix) | FIXED | Cannot adapt online — needs PGAS |
| K (regime count) | FIXED | Structural — needs PGAS validation |
| Regime definitions | FIXED | What each regime "means" |

This is why we need PGAS.

---

## Part II: Background Thread

### 2.1 PGAS (Parameter Learning — Background)

**Purpose**: Learn transition matrix Π via MCMC. Validate regime structure.

**Full name**: Particle Gibbs with Ancestor Sampling

**What it does**:
```
For M iterations (M = 50-200):
    1. Run conditional SMC with reference trajectory locked
    2. Ancestor sampling allows reference to reconnect
    3. Sample new trajectory from final particles
    4. Update Π from transition counts
```

**Key outputs**:
- Learned Π matrix
- Particle cloud consistent with learned Π (the "lifeboat")
- Evidence about whether K is appropriate

**Frequency**: Periodic, triggered by health metrics or heartbeat.

---

### 2.2 PGAS Triggers

**When to run PGAS** (health-based, not uncertainty-based):

| Trigger | What it detects | Self-correcting? |
|---------|-----------------|------------------|
| ESS collapse | Particles dying | ✓ Yes |
| Log-likelihood drop | Model failing predictions | ✓ Yes |
| Regime dominance | One regime >98% for too long | ✓ Yes |
| Heartbeat | Stiffening, dead zones | ✓ Yes (forced) |

**Why health-based**: Uncertainty-based triggers create feedback loops. Health triggers detect "something is broken" and are self-correcting.

---

### 2.3 What PGAS Produces

```c
LifeboatPacket {
    // New parameters
    float trans[K][K];           // Learned transition matrix
    
    // Particle cloud consistent with new params
    Particle particles[N];       // The "lifeboat"
    float weights[N];
    
    // Buffered observations during PGAS run
    float buffered_obs[T_buffer];
    int buffer_len;
}
```

**Critical**: Parameters + Particles are a PAIR. You must swap both together.

---

## Part III: Lifeboat (Handoff Gate)

### 3.1 Purpose

**Safely inject PGAS-learned Π into live RBPF without causing particle collapse.**

**The danger without Lifeboat**:
```
Old particles sampled under old Π
Just swap Π?
→ Old particles are "impossible" under new Π
→ Weights collapse to zero
→ ESS = 1
→ FILTER IS DEAD
```

**Lifeboat solution**: Swap BOTH Π and particles atomically.

---

### 3.2 The Protocol

```
1. VALIDATE     — Is PGAS output trustworthy?
2. ALIGN        — Solve label switching (regime permutation)
3. INJECT       — Replace particles with lifeboat cloud
4. CATCH-UP     — Process buffered ticks to reach present
5. RESET VI     — Re-anchor transition learning from new Π
```

---

### 3.3 Validation Gate

**What it checks**:

| Check | Purpose |
|-------|---------|
| Stickiness stability | Π diagonal didn't crash (>30% drop) |
| K change bounded | At most ±1 regime change |
| μ ordering preserved | Regimes still sorted by volatility level |
| Row sums valid | Π rows sum to 1.0 |
| Regime anchors | Regime 0 still "calm", regime K-1 still "crisis" |

**Decisions**:
```c
typedef enum {
    LIFEBOAT_DISCARD,   // Not worth the disruption
    LIFEBOAT_MIX,       // Gradual blend
    LIFEBOAT_REPLACE,   // Immediate swap
    LIFEBOAT_ALERT      // Something suspicious
} LifeboatDecision;
```

---

### 3.4 Label Alignment

**The problem**: PGAS can return same solution with permuted regime labels.

```
Live RBPF:
  Regime 0: μ=0.1 (low vol)
  Regime 1: μ=0.8 (high vol)

PGAS output (labels swapped):
  Regime 0: μ=0.8 (high vol)  ← WRONG LABEL
  Regime 1: μ=0.1 (low vol)   ← WRONG LABEL
```

**Solution**: Sort both by μ, create permutation map, apply to Π and particles.

---

### 3.5 Catch-Up

**Problem**: During PGAS run, ticks keep arriving.

**Solution**: Buffer ticks, then fast-forward through them after injection.

**Protection**: 
- Buffer cap (never grow unbounded)
- Time-bounded catch-up (don't block forever)
- Subsampling if buffer too large

---

## Part IV: What Each Component Learns

### Summary Table

| Component | Thread | What it learns | How |
|-----------|--------|----------------|-----|
| **RBPF** | Live | h_t, r_t (states) | Particle filtering |
| **PARIS** | Live | Trajectory ancestry | Backward resampling |
| **Storvik** | Live | μ_vol[k], σ_vol[k] | NIG sufficient stats |
| **Silverman** | Live | Kernel bandwidth | IQR of particles |
| **SPRT** | Live | Regime evidence | LLR accumulation |
| **PGAS** | Background | Π (transitions) | MCMC sampling |
| **Lifeboat** | Handoff | — | Validation + injection |

### The Division of Labor

```
RBPF adapts:
  ✓ Particle states (h, r)
  ✓ Emission parameters (μ_vol, σ_vol via Storvik)
  ✓ Kernel bandwidth (via Silverman)
  ✓ Regime detection smoothing (via SPRT)

RBPF holds FIXED:
  ✗ Transition matrix Π
  ✗ Number of regimes K
  ✗ Regime definitions

PGAS provides:
  ✓ Learned Π
  ✓ Validation of K
  ✓ Fresh particle cloud consistent with Π
```

---

## Part V: Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                     LIVE THREAD (every tick)                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Market Tick                                                        │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐          │
│  │  RBPF   │───►│ PARIS   │───►│ Storvik │───►│  SPRT   │          │
│  │ Forward │    │Backward │    │ (μ,σ)   │    │(regime) │          │
│  └────┬────┘    └─────────┘    └─────────┘    └────┬────┘          │
│       │                                            │                │
│       ▼                                            ▼                │
│  Silverman ──► Resample ──► Fisher-Rao      Kelly Sizing           │
│  (bandwidth)   (+ jitter)   (mutation)           │                 │
│                                                   ▼                 │
│                                              Trade Signal           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              │ (Π is FIXED here)
                              │
                              │ Lifeboat injection
                              │ (when PGAS ready)
                              │
┌─────────────────────────────│───────────────────────────────────────┐
│                     BACKGROUND THREAD (periodic)                    │
├─────────────────────────────│───────────────────────────────────────┤
│                             │                                       │
│  Buffered Ticks ◄───────────┘                                       │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────┐         ┌─────────────────────────────┐               │
│  │  PGAS   │────────►│    Lifeboat Packet          │               │
│  │ (learns │         │  • New Π                    │               │
│  │   Π)    │         │  • Particle cloud           │               │
│  └─────────┘         │  • Buffered observations    │               │
│                      └──────────────┬──────────────┘               │
│                                     │                               │
│                                     ▼                               │
│                      ┌──────────────────────────────┐              │
│                      │     Validation Gate          │              │
│                      │  • Label alignment           │              │
│                      │  • Stability checks          │              │
│                      │  • Anchor validation         │              │
│                      └──────────────┬───────────────┘              │
│                                     │                               │
│                                     ▼                               │
│                      DISCARD / MIX / REPLACE                        │
│                                     │                               │
│                                     ▼                               │
│                      Inject to Live RBPF ──────────────────────────►│
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Part VI: Failure Modes & Mitigations

### Primary Failures

| Failure | Cause | Mitigation |
|---------|-------|------------|
| **Hot Swap Shock** | Old particles incompatible with new Π | Lifeboat: swap Π + particles together |
| **Poor MCMC Mixing** | PGAS returns unconverged sample | Validation gate + ensemble checking |
| **Semantic Drift** | Regime meanings shift over time | μ-ordering + anchor validation |

### Secondary Failures

| Failure | Cause | Mitigation |
|---------|-------|------------|
| **Golden Sample Fallacy** | Single PGAS sample may be suboptimal | Ensemble mean of last N sweeps |
| **Threshold Dead Zones** | Absolute thresholds miss slow degradation | Adaptive thresholds + heartbeat trigger |
| **Catch-Up Bottleneck** | Buffer grows faster than processing | Buffer cap + time-bounded catch-up |
| **Learning Rate Stiffening** | Online VI stops responding as ρ→0 | Floor on ρ + heartbeat reset |

---

## Part VI-B: Skeptical Analysis (Critical Risks)

### Risk 1: Golden Sample Fallacy (PGAS)

**Problem**: PGAS produces a distribution over Π, but Lifeboat injects ONE sample. If that MCMC iteration landed in a local mode or noisy outlier, the entire HFT execution is pinned to a sub-optimal structural model for ~500ms.

**Fix**: Ensemble Mean
```c
// Instead of single sample:
PG_Update lifeboat = last_pgas_sample;

// Use ensemble mean of last N converged sweeps:
PG_Update lifeboat = pg_ensemble_get_mean(&ensemble, K);

// Only inject if ensemble variance is low (chain has mixed)
if (!pg_ensemble_is_stable(&ensemble, K)) {
    return LIFEBOAT_DISCARD;  // Wait for more sweeps
}
```

### Risk 2: Temporal Aliasing (Catch-Up Jitter)

**Problem**: During ~3ms PGAS run, ticks buffer. After injection, we "slam" the new filter with backlog as fast as possible. This creates a **Liquidity Mirage** — the Live Thread is blind during catch-up, and by the time it catches up, the market regime may have reverted. Induces **Phase Lag** in signals.

**Fix**: Iterative Fisher-Rao Drift (not hard swap + fast-forward)
```c
// BAD: Hard swap then catch-up
lifeboat_inject(rbpf, pkt);           // Teleport to new state
lifeboat_catchup(rbpf, pkt);          // Slam through buffer

// GOOD: Gradual blend across the gap
int gap = pkt->buffer_len;
for (int i = 0; i < gap; i++) {
    float blend_t = (float)(i + 1) / gap;  // 0→1 over buffer
    
    // Fisher-Rao interpolate between live and lifeboat state
    fisher_rao_geodesic(
        live_mu, live_var,
        lb_mu, lb_var,
        blend_t,
        &blended_mu, &blended_var
    );
    
    // Step with blended state
    rbpf_step_with_state(rbpf, pkt->buffered_obs[i], blended_mu, blended_var);
}
```

### Risk 3: Numerical Fossilization (Storvik) — PRIORITY #1

**Problem**: Storvik uses exponential forgetting (λ=0.997), but sufficient statistics are sensitive to initial conditions. If RBPF spends time in a **False Regime** before PGAS corrects it, Storvik's (m, κ, α, β) get poisoned by garbage data. Even with forgetting, κ can grow large enough that the model becomes **stiff** — stops responding to volatility spikes. Self-fulfilling prophecy of stale estimates.

**The "Arrogant Model" Failure:**
```
Summer: Calm market, κ grows to 10⁵ in regime 0
September: VIX doubles, market enters crisis

But: New tick disagreeing with μ has weight 1/100,000
Result: Model stays in "Calm Mode" while market burns
Symptom: surprise spikes, log_vol_mean stays flat
```

**Why this is #1 Priority:**
- Only item that prevents "Silent Blowup"
- Causes slow P&L bleed, extremely hard to diagnose
- Gets worse over time (κ compounds)
- Low λ (more forgetting) adds constant noise — not the solution
- Need REACTIVE reset, not constant decay

**Fix**: Hard Reset Trigger — restore elasticity when SPRT contradicts Storvik

```c
typedef struct {
    float surprise_ema;           // Smoothed surprise metric
    float surprise_threshold;     // Trigger level (e.g., 3σ above baseline)
    int consecutive_surprises;    // Count of high-surprise ticks
    int surprise_patience;        // Ticks before reset (e.g., 5)
} FossilizationDetector;

void storvik_check_fossilization(ParamLearner *learner, 
                                  const SPRT_Multi *sprt,
                                  FossilizationDetector *detector,
                                  float current_surprise) {
    // Update surprise EMA
    const float alpha = 0.05f;
    detector->surprise_ema = (1 - alpha) * detector->surprise_ema + alpha * current_surprise;
    
    // Check for sustained high surprise (model failing)
    if (current_surprise > detector->surprise_threshold) {
        detector->consecutive_surprises++;
    } else {
        detector->consecutive_surprises = 0;
    }
    
    // Don't reset on single spike — wait for pattern
    if (detector->consecutive_surprises < detector->surprise_patience) {
        return;
    }
    
    // Find which regime SPRT thinks we're in vs which Storvik thinks
    int sprt_regime = sprt_multi_get_dominant(sprt);
    
    // Check if Storvik's estimate for that regime is way off
    float sprt_implied_vol = sprt->regime_vol_estimate[sprt_regime];
    float storvik_mean = storvik_get_mu(learner, sprt_regime);
    float storvik_sigma = storvik_get_sigma_mu(learner, sprt_regime);
    
    float z_disagreement = fabs(sprt_implied_vol - storvik_mean) / storvik_sigma;
    
    if (z_disagreement > 3.0f) {
        // 3σ disagreement = Storvik is fossilized
        storvik_partial_reset(learner, sprt_regime, 0.3f);
        detector->consecutive_surprises = 0;
        
        log_warning("Storvik reset: regime %d, z=%.2f, κ was %.0f",
                    sprt_regime, z_disagreement, 
                    storvik_get_kappa(learner, sprt_regime));
    }
}

void storvik_partial_reset(ParamLearner *learner, int regime, float keep_ratio) {
    StorvikSoA *soa = param_learn_get_active_soa(learner);
    const RegimePrior *prior = &learner->priors[regime];
    
    for (int n = 0; n < learner->n_particles; n++) {
        int idx = n * learner->n_regimes + regime;
        
        // Reduce precision counts (regain elasticity)
        // κ: 100000 → 30000 (still informed, but flexible)
        soa->kappa[idx] = prior->kappa + keep_ratio * (soa->kappa[idx] - prior->kappa);
        
        // α, β: blend back toward prior
        soa->alpha[idx] = prior->alpha + keep_ratio * (soa->alpha[idx] - prior->alpha);
        soa->beta[idx] = prior->beta + keep_ratio * (soa->beta[idx] - prior->beta);
        
        // Keep m (mean) — we trust the location, just not the certainty
    }
    
    learner->fossilization_resets++;
}
```

**Key insight:** Keep `m` (the mean estimate), reset `κ` (the certainty). We trust WHERE Storvik thinks volatility is, we just restore its ability to MOVE.

### Risk 4: Semantic Drift (Anchor Violation)

**Problem**: μ-ordering solves label switching but assumes "Low Vol" found by PGAS is the same "Low Vol" RBPF tracks. In systemic crisis, ALL regimes shift up. The "Low Vol" slot (μ₀) might represent what used to be "High Vol". Sorting preserves order but not meaning.

**Fix**: Physical Anchor Validation
```c
typedef struct {
    float mu_calm_ceiling;     // Regime 0 must have μ below this
    float mu_crisis_floor;     // Regime K-1 must have μ above this
    float baseline_vol;        // 1-year baseline (physical anchor)
} RegimeAnchors;

LifeboatDecision validate_anchors(const PG_Update *update, const RegimeAnchors *anchors) {
    // Regime 0 should be "calm" relative to baseline
    if (update->mu_vol[0] > anchors->mu_calm_ceiling) {
        // "Low vol" is actually high — systemic crisis?
        return LIFEBOAT_ALERT;  // Don't auto-replace, flag for review
    }
    
    // Regime K-1 should be "crisis-capable"
    if (update->mu_vol[update->K - 1] < anchors->mu_crisis_floor) {
        // Highest regime is too low — missing crisis mode
        return LIFEBOAT_ALERT;
    }
    
    // Check if entire structure has drifted from baseline
    float mean_mu = 0;
    for (int k = 0; k < update->K; k++) {
        mean_mu += update->mu_vol[k];
    }
    mean_mu /= update->K;
    
    if (fabs(mean_mu - anchors->baseline_vol) > 1.0) {
        // Entire regime structure has shifted significantly
        return LIFEBOAT_ALERT;
    }
    
    return LIFEBOAT_REPLACE;  // Anchors valid
}
```

### Summary: Skeptical Gaps

| Feature | Claimed Benefit | Skeptical Counter-Point |
|---------|-----------------|------------------------|
| Fisher-Rao | Principled blending | Overkill if Δμ is small; adds latency |
| Lifeboat | Prevents collapse | If catch-up buffer is large, swap is already stale |
| PGAS | Learns Π | MCMC convergence never guaranteed in 200 iterations |
| Storvik | Online μ,σ learning | κ fossilization if stuck in wrong regime |
| μ-ordering | Solves label switching | Preserves order, not meaning |

---

## Part VII: Key Design Decisions

### Why PARIS runs on Live Thread (not with PGAS)

PARIS fixes path degeneracy every tick. Without it:
- After 100 ticks, all particles trace to ~1-2 ancestors
- Historical state estimates become meaningless
- Smoothed volatility output is garbage

PARIS must run every tick to maintain trajectory diversity.

### Why Storvik runs on Live Thread (not with PGAS)

Emission parameters (μ_vol, σ_vol) change continuously. Waiting for PGAS would mean:
- Stale parameters for hundreds of ticks
- Missed volatility regime shifts
- Poor filtering accuracy

Storvik's sufficient statistics enable per-tick updates with O(1) cost.

### Why Π requires PGAS (cannot adapt online)

Transition matrix Π is discrete structure:
- Dirichlet-Multinomial conjugacy helps but...
- Online updates create feedback loops
- No ground truth for "correct" transition
- Needs full trajectory to estimate properly

PGAS with MCMC provides proper posterior over Π.

### Why Lifeboat (not just parameter swap)

Parameters and particles are coupled:
- Particles were sampled under old Π
- New Π makes old particles "impossible"
- Weight collapse → ESS → 1 → filter death

Must swap both atomically + catch-up to present.

---

## Part VIII: Implementation Status

### Implemented ✓

| Component | Status |
|-----------|--------|
| RBPF core | ✓ Complete |
| PARIS backward | ✓ Complete |
| Storvik parameter learning | ✓ Complete |
| Silverman bandwidth | ✓ Complete |
| Fisher-Rao geodesic | ✓ Complete |
| SPRT regime detection | ✓ Complete |
| PGAS/PARIS smoother | ✓ Complete |
| Online VI (header) | ✓ Complete |

### In Progress

| Component | Status | Notes |
|-----------|--------|-------|
| Lifeboat validation | Partial | Has fixed thresholds — needs uncertainty-aware redesign |
| Online VI integration | **Priority** | Header exists, needs integration for Var[π_ij] |
| PGAS triggers | Partial | Need adaptive thresholds + heartbeat |
| PGAS ensemble | Not started | Required to fix Golden Sample Fallacy |
| Storvik reset trigger | Not started | Required to fix Numerical Fossilization |

### TODO

#### Implementation Order (Revised)

| Order | Task | Goal | Complexity |
|-------|------|------|------------|
| **1** | **Storvik Hard Reset** | Elasticity — restore ability to learn when SPRT contradicts Storvik | Medium |
| **2** | Expose Storvik σ_μ | Context — turns DISCARD/ACCEPT into statistical z-score | Trivial |
| **3** | Fisher-Rao Continuous Drift | Smoothness — eliminates jerky MIX/REPLACE jumps | Low |
| **4** | Dynamic Shadow Filter | Evidence — "Race to Certainty" not fixed window | Medium |
| **5** | Online VI Integration | Precision — Var[π_ij] for Mahalanobis gate | Medium |

**Why Storvik Hard Reset is #1:**
- Only item that prevents "Silent Blowup"
- Fossilization causes slow P&L bleed, hard to diagnose
- Will hit you on Summer→September transition when κ is huge from calm period
- Model becomes "arrogant" — deaf to market with κ at 10⁵

#### Critical: Zero-Heuristic Validation

| Task | Kills Heuristic | Component |
|------|-----------------|-----------|
| **Shadow Filter (Evidence Accumulator)** | `KL > 0.50` threshold | New module using `rbpf_ksc_step` |
| **Expose Storvik σ_μ** | `DELTA_MU < 0.02` threshold | Already computed, just expose |
| **Continuous Fisher-Rao injection** | `MIX/REPLACE` modes | Already built, wire it up |
| **SPRT confusion → merge trigger** | `MU_COLLISION` threshold | SPRT exists, add merge hint |

#### High Priority (Skeptical Fixes)

| Task | Risk Addressed | Impact |
|------|----------------|--------|
| **PGAS Ensemble Mean** | Golden Sample Fallacy | Prevents injection of outlier MCMC sample |
| **Storvik Hard Reset Trigger** | Numerical Fossilization | Restores elasticity when SPRT contradicts Storvik |
| **Iterative Fisher-Rao Catch-Up** | Temporal Aliasing | Eliminates phase lag from hard swap |
| **Physical Anchor Validation** | Semantic Drift | Detects systemic regime shift vs label swap |

#### High Priority (Uncertainty-Aware Validation)

| Task | Purpose | Impact |
|------|---------|--------|
| **Integrate Online VI** | Track Var[π_ij] on live thread | Enables Mahalanobis distance for Π validation |
| **PGAS Ensemble Variance** | Track uncertainty in learned Π | Weighted validation thresholds |
| **Replace fixed thresholds** | Remove magic numbers | χ²-based decision with proper DoF |

#### Medium Priority

| Task | Purpose |
|------|---------|
| Expose Storvik σ_μ | Diagnostics, not validation (Storvik is live) |
| Heartbeat trigger | Prevents stiffening in dead zones |
| Continuous blend mode | Smoother injection via Fisher-Rao |
| Buffer cap tuning | Prevent catch-up bottleneck |

#### Validation Gate Redesign: Zero-Heuristic Architecture

**Current (heuristic):**
```c
if (kl_divergence < 0.02f) return LIFEBOAT_DISCARD;
if (kl_divergence > 0.50f) return LIFEBOAT_REPLACE;
```

**Target (principled):** Replace ALL fixed thresholds with live statistical tests.

---

**1. Kill `LV_THRESH_DISCARD_DELTA_MU` → Storvik Z-Score**

```c
// OLD: if (delta_mu < 0.02f) DISCARD
// NEW: Use Storvik posterior uncertainty

float sigma_mu_k = sqrt(storvik->beta[k] / (storvik->alpha[k] * storvik->kappa[k]));
float z_score = fabs(lb_mu[k] - live_mu[k]) / sigma_mu_k;

// z < 2: within 95% CI → noise, not signal
// z > 3: beyond 99% CI → structural shift
```

---

**2. Kill `LV_THRESH_REPLACE_KL` → Dynamic Shadow Filter ("Race to Certainty")**

**Problem with fixed window:** 20 ticks = 1ms at 20kHz (acceptable), but 20ms at 1kHz (unacceptable). Volatility spikes happen in first 5-10ms.

**Solution:** Sequential test. Decide when evidence is decisive, not after arbitrary window.

```c
typedef struct {
    float log_bf;             // Accumulated log(P(y|lifeboat) / P(y|live))
    float drift_t;            // Current Fisher-Rao blend parameter
    int ticks_accumulated;
    
    // Thresholds (Jeffreys scale: log(3)=1.1, log(10)=2.3, log(100)=4.6)
    float log_bf_decisive;    // e.g., 4.6 (BF > 100)
    float log_bf_reject;      // e.g., -4.6 (BF < 1/100)
    
    // Shadow state
    RBPF_State lifeboat_shadow;
    bool active;
} LifeboatEvidenceAccumulator;

void shadow_filter_init(LifeboatEvidenceAccumulator *acc, 
                        const LifeboatPacket *pkt,
                        float initial_z_score) {
    acc->log_bf = 0.0f;
    acc->ticks_accumulated = 0;
    acc->active = true;
    
    // Start drift immediately based on z-score
    // z=2 → t=0.1, z=3 → t=0.3, z=5 → t=0.6
    acc->drift_t = fminf(0.8f, (initial_z_score - 1.5f) * 0.15f);
    acc->drift_t = fmaxf(0.0f, acc->drift_t);
    
    // Copy lifeboat state for shadow filtering
    memcpy(&acc->lifeboat_shadow, &pkt->rbpf_state, sizeof(RBPF_State));
}

typedef enum {
    SHADOW_CONTINUE,      // Keep accumulating, maintain current drift
    SHADOW_ACCELERATE,    // Evidence strong, increase drift speed
    SHADOW_ABORT,         // Evidence against, reverse drift
    SHADOW_COMPLETE       // Fully transitioned
} ShadowDecision;

ShadowDecision shadow_filter_step(LifeboatEvidenceAccumulator *acc, 
                                   RBPF *live_rbpf, 
                                   float y_t) {
    if (!acc->active) return SHADOW_COMPLETE;
    
    // Step both models
    float ll_live = rbpf_ksc_get_marginal_lik(live_rbpf);
    float ll_shadow = rbpf_ksc_step_shadow(&acc->lifeboat_shadow, y_t);
    
    // Accumulate evidence
    acc->log_bf += (ll_shadow - ll_live);
    acc->ticks_accumulated++;
    
    // Sequential decision
    if (acc->log_bf > acc->log_bf_decisive) {
        // Strong evidence FOR lifeboat — accelerate drift
        acc->drift_t = fminf(1.0f, acc->drift_t + 0.3f);
        
        if (acc->drift_t >= 1.0f) {
            acc->active = false;
            return SHADOW_COMPLETE;
        }
        return SHADOW_ACCELERATE;
    }
    
    if (acc->log_bf < acc->log_bf_reject) {
        // Strong evidence AGAINST lifeboat — abort
        acc->drift_t = 0.0f;
        acc->active = false;
        return SHADOW_ABORT;
    }
    
    // Inconclusive — continue at current rate
    return SHADOW_CONTINUE;
}
```

**The "Race to Certainty" Flow:**

```
Lifeboat arrives
    │
    ├── IMMEDIATELY: Compute z-score from Storvik σ_μ
    │                Start Fisher-Rao drift at t = f(z_score)
    │
    └── EACH TICK (parallel):
            │
            ├── Apply current drift: fisher_rao_geodesic(..., drift_t, ...)
            │
            ├── Update Shadow Filter: log_bf += ll_shadow - ll_live
            │
            └── Adjust speed:
                    log_bf > +4.6  → accelerate (t += 0.3)
                    log_bf < -4.6  → abort (t = 0, reverse drift)
                    else           → maintain current t
```

**Why this works:**
- Flash crash: Shadow Filter decides in 2-3 ticks, instant transition
- Slow drift: May take 50+ ticks, gradual blend
- No dead time: Always moving, speed determined by evidence
- No fixed threshold: Bayes Factor has natural interpretation

---

**3. Kill `MIX/REPLACE` modes → Continuous Fisher-Rao Drift**

**Unified with Shadow Filter:** Not a separate step. Fisher-Rao drift happens every tick, Shadow Filter controls the speed.

```c
// OLD: discrete choice
if (decision == LIFEBOAT_MIX) { blend(0.5); }
if (decision == LIFEBOAT_REPLACE) { swap(); }

// NEW: continuous drift, speed from evidence
void lifeboat_tick(RBPF *rbpf, LifeboatEvidenceAccumulator *acc, float y_t) {
    // 1. Update evidence
    ShadowDecision sd = shadow_filter_step(acc, rbpf, y_t);
    
    // 2. Apply drift at current speed
    if (acc->drift_t > 0.0f) {
        for (int n = 0; n < rbpf->n_particles; n++) {
            fisher_rao_geodesic(
                rbpf->particles[n].h, rbpf->particles[n].var,
                acc->lifeboat_shadow.particles[n].h, 
                acc->lifeboat_shadow.particles[n].var,
                acc->drift_t,  // Speed controlled by Shadow Filter
                &rbpf->particles[n].h, &rbpf->particles[n].var
            );
        }
    }
    
    // 3. Handle abort (reverse drift)
    if (sd == SHADOW_ABORT) {
        // Lifeboat was wrong, snap back to pure live state
        // (already there since drift_t = 0)
    }
}
```

**Benefit:** Eliminates "Jerk" (third derivative of position) in Kelly sizing. Smooth transitions save more in slippage than almost any other optimization.

---

**4. Kill `LV_THRESH_MU_COLLISION` → SPRT Regime Merging**

```c
// OLD: if (mu_separation < 0.1f) ABORT
// NEW: Use SPRT pairwise decisiveness

void check_regime_overlap(SPRT_Multi *sprt, int K) {
    for (int i = 0; i < K; i++) {
        for (int j = i + 1; j < K; j++) {
            // How often does SPRT confuse regime i and j?
            float confusion_rate = sprt->pairwise_indecision[i][j];
            
            if (confusion_rate > 0.3f) {
                // Regimes i,j are not distinguishable
                // This isn't a failure — K is too high
                trigger_pgas_with_merge_hint(i, j);
            }
        }
    }
}
```

**Why this works:** Collision isn't an error, it's information that K should decrease.

---

**Summary: Heuristic-Free Transformation**

| Heuristic | Killed By | Component Used | Status |
|-----------|-----------|----------------|--------|
| `DELTA_MU < 0.02` | Z-score from Storvik σ_μ | Storvik NIG | TODO #2 |
| `KL > 0.50` | Dynamic Shadow Filter (Race to Certainty) | RBPF marginal_lik | TODO #4 |
| `MIX/REPLACE` | Continuous geodesic drift | Fisher-Rao | TODO #3 |
| `MU_COLLISION` | SPRT pairwise confusion → merge | SPRT | Future |
| `JITTER_PCT` | Already killed | Silverman IQR | ✓ Done |
| (Fossilization) | Hard Reset on SPRT contradiction | Storvik + SPRT | TODO #1 |

**New Modules Required:**
1. `FossilizationDetector` — monitors surprise, triggers Storvik reset
2. `LifeboatEvidenceAccumulator` — Shadow Filter with dynamic window
3. `storvik_get_sigma_mu()` — expose posterior uncertainty (trivial)

**The Unified Flow:**

```
LIVE THREAD (every tick):
    │
    ├── RBPF step → marginal_lik, surprise
    │
    ├── Fossilization check:
    │   └── if (surprise high + SPRT disagrees) → partial reset κ
    │
    ├── If Lifeboat pending:
    │   ├── Shadow Filter: log_bf += ll_shadow - ll_live
    │   ├── Adjust drift_t based on evidence
    │   └── Apply Fisher-Rao geodesic at current drift_t
    │
    └── Output: vol_mean, regime_probs, kelly_fraction


BACKGROUND THREAD (periodic):
    │
    ├── PGAS sweep (learns Π)
    │
    ├── Build Lifeboat packet
    │
    └── Signal Live Thread:
        ├── Compute initial z-score from Storvik σ_μ
        ├── Start drift_t = f(z_score)
        └── Activate Shadow Filter
```

**No fixed thresholds. No discrete modes. Evidence drives everything.**

---

## Appendix: Component Interfaces

### RBPF Output
```c
RBPF_KSC_Output {
    float vol_mean;           // E[exp(h)]
    float log_vol_mean;       // E[h]
    float log_vol_var;        // Var[h]
    
    int dominant_regime;
    float regime_probs[K];
    
    float ess;
    float surprise;           // -log P(y|model)
}
```

### Storvik Interface
```c
// Already tracked:
float m[particle][regime];     // Posterior mean of μ
float kappa[particle][regime]; // Precision

// Should expose:
float sigma_mu = sqrt(beta / (alpha * kappa));  // Posterior uncertainty
```

### Online VI Interface
```c
// Mean transition matrix (for RBPF)
void online_vi_get_mean(OnlineVI *vi, double *trans);

// Variance (for Kelly sizing)
void online_vi_get_variance(OnlineVI *vi, double *var);

// Per-row entropy (for triggers)
void online_vi_get_row_entropy(OnlineVI *vi, double *H);
```

### Lifeboat Interface
```c
LifeboatDecision lifeboat_validate(
    const LiveRBPFState *live,
    const LifeboatPacket *lb,
    LifeboatDivergence *div
);

void lifeboat_inject(
    RBPF *rbpf,
    const LifeboatPacket *pkt,
    LifeboatMode mode  // REPLACE or MIX
);
```

---

## Appendix: Equations Reference

### Storvik NIG Posterior
```
κ_n = κ₀ + n
m_n = (κ₀m₀ + Σz_i) / κ_n
α_n = α₀ + n/2
β_n = β₀ + ½Σ(z_i - m_n)² + κ₀n(m_n - m₀)²/(2κ_n)

Posterior uncertainty on μ:
σ(μ) = √(β_n / (α_n × κ_n))
```

### Online VI Natural Gradient
```
α̃_ij ← (1 - ρ_t) × α̃_ij + ρ_t × (α_prior + ξ_ij)

Where:
ξ_ij = P(s_{t-1}=i, s_t=j | y_{1:t})
ρ_t = (τ + t)^{-κ}  (Robbins-Monro)

Posterior variance:
Var[π_ij] = α̃_ij(α̃_i0 - α̃_ij) / (α̃_i0² × (α̃_i0 + 1))
```

### Fisher-Rao Geodesic
```
t = σ²_target / (σ²_particle + σ²_target)
μ_blend = (1-t)μ_particle + t×μ_target
σ²_blend = (1-t)σ²_particle + t×σ²_target
```

---


Our consensus on **PARIS** (PARticle-based, Rapid Incremental Smoother) within your specific HFT stack is that it serves as the **"Trajectory Repair Kit"** for the Live Thread.

While the RBPF forward filter tells you where the market is *now*, PARIS ensures that the **history** of how we got here remains diverse and statistically valid for the background thread.

Here is the breakdown of the consensus we reached regarding its role, implementation, and integration:

### 1. The Operational Consensus: Where it fits

PARIS does not run in the background; it runs on the **Live Thread**.

* **Frequency:** Every market tick (Incremental Smoothing).
* **The Problem it Solves:** Path Degeneracy. Without PARIS, your particle histories collapse into a single ancestor within ~100 ticks. If that one ancestor is "wrong" or a noisy outlier, your background PGAS thread will learn a "garbage" transition matrix.
* **The Solution:** PARIS "repairs" the ancestry pointers at every step by looking at the immediate future () to re-select better ancestors for time .

### 2. The Computational Consensus:  vs 

Though the original Olsson & Westerborn paper highlights an  version via rejection sampling, our consensus for your **i9-14900KF / MKL** build is:

* **Stick to  with MKL batching.**
* **Why:** Rejection sampling involves heavy branching logic which kills the instruction pipeline on modern CPUs. For your target , a vectorized MKL kernel that computes the full  backward weight matrix is actually **faster** and has **zero jitter** compared to the "more efficient"  algorithm.

### 3. Integration Consensus: Why we need it for your stack

| Component | Why it needs PARIS |
| --- | --- |
| **Storvik (Live)** | Updates parameters () using smoothed paths, which reduces the "filtering bias" and prevents the model from chasing noise. |
| **Kelly Sizing** | Smoothed regime probabilities are more stable than filtered ones. This prevents your position sizing from "chattering" during micro-spikes. |
| **PGAS (Background)** | PARIS provides the **Reference Trajectory**. This is the single most important role. It ensures that when PGAS wakes up to update the transition matrix (), it is working with a high-quality, diverse set of paths. |
| **Lifeboat** | The Lifeboat relies on a "Consistent Particle Cloud." PARIS ensures that cloud has a long, valid history so the handoff doesn't cause a weight collapse. |

### 4. The "Handshake" with PGAS

Our consensus on the data flow is:

1. **Live Thread:** Runs RBPF + PARIS every tick.
2. **Buffer:** The Live Thread stores a window of these *smoothed* particles ().
3. **Trigger:** When PGAS is triggered, it grabs this  buffer.
4. **Inference:** Because PARIS has kept the history diverse, PGAS can efficiently sample new transition matrices () without getting stuck in local modes.

### 5. Summary Table

| Feature | Consensus Value |
| --- | --- |
| **Target Latency** | **Sub-5µs** (Incremental overhead) |
| **Complexity** |  (AVX-512/MKL Vectorized) |
| **Smoothing Lag** | **1-Tick** (Immediate retrospective repair) |
| **Primary Goal** | Prevent Path Collapse for the PGAS thread |


*Last updated: December 2025*