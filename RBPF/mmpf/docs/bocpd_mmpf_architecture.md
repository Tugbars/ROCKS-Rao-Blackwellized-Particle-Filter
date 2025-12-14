# BOCPD + MMPF: Sidecar Architecture for Regime Detection

## The Problem

We have two conflicting needs:

1. **Fast detection**: When the market regime changes (calm → crisis), we need to know immediately
2. **Accurate tracking**: We need stable, accurate volatility estimates that don't jump around on noise

One system can't do both well. A sensitive detector gives false alarms. A stable tracker is slow to adapt.

## The Solution: Sidecar Architecture

Run two systems in parallel:

```
                    ┌─────────────┐
                    │   BOCPD     │  ← Lightweight watchdog
                    │  (detector) │     "Did something break?"
                    └──────┬──────┘
                           │ shock signal (rare)
                           ▼
┌──────────┐        ┌─────────────────────────────────────┐
│  Market  │───────▶│              MMPF                   │
│   Data   │        │  ┌───────┐ ┌───────┐ ┌───────┐     │
└──────────┘        │  │ RBPF  │ │ RBPF  │ │ RBPF  │     │
                    │  │ Calm  │ │ Trend │ │Crisis │     │
                    │  │  σ₁   │ │  σ₂   │ │  σ₃   │     │
                    │  └───┬───┘ └───┬───┘ └───┬───┘     │
                    │      └────┬────┴────┬────┘         │
                    │           ▼         ▼              │
                    │    regime weights + weighted σ     │
                    └──────────────┬──────────────────────┘
                                   │
                                   ▼
                            ┌──────────┐
                            │  Kelly   │
                            │  Sizing  │
                            └──────────┘
```

**BOCPD** (Bayesian Online Changepoint Detection):
- Runs first, every tick
- Asks: "Is the current regime still plausible?"
- Lightweight, sensitive
- Outputs: probability mass at short run lengths

**MMPF** (Multiple Model Particle Filter):
- IMM (Interacting Multiple Model) layer
- Runs 3 parallel RBPFs with different structural hypotheses
- Sticky transitions (98% stay in same regime) for stability
- Outputs: regime weights [P(Calm), P(Trend), P(Crisis)]

**RBPF** (Rao-Blackwellized Particle Filter):
- The actual volatility tracker (one per regime hypothesis)
- Uses KSC 10-component mixture for log-χ² likelihood
- Handles fat tails correctly (won't dismiss 5σ as impossible)
- Each RBPF has different μ_vol anchor (Calm=-4.83, Trend=-4.20, Crisis=-3.35)
- Outputs: σ estimate under that hypothesis

```
MMPF
 ├── RBPF[Calm]   (φ=0.98, low noise)   → σ_calm
 ├── RBPF[Trend]  (φ=0.95, med noise)   → σ_trend
 └── RBPF[Crisis] (φ=0.80, high noise)  → σ_crisis
         │
         ▼
    σ_final = w_calm·σ_calm + w_trend·σ_trend + w_crisis·σ_crisis
```

## How They Talk

BOCPD doesn't tell MMPF *which* regime we're in. It just says "something changed."

```
Normal tick:
  BOCPD: "Run length 847, everything normal"
  MMPF:  Uses sticky transitions (98% stay), updates smoothly

Shock tick:
  BOCPD: "Run length collapsed to 0! Changepoint detected!"
  MMPF:  Receives shock signal →
         - Transitions become uniform (33% to each regime)
         - Process noise boosted 50×
         - Particles spread out, explore all regimes
         - Likelihood determines winner in ONE tick
         - Back to normal sticky mode
```

## Why BOCPD Works

It tracks a probability distribution over "run length" — how long since the last changepoint.

```
Normal observation (1-2σ):
  Old run length stays plausible
  Mass stays where it is

Extreme observation (6-8σ):
  Old run length becomes impossible
  Mass teleports to r=0 (new regime)
```

The **delta detector** watches for this mass movement:

```
delta = P(r < 15)_today - P(r < 15)_yesterday
```

When delta spikes (z-score > 3σ), fire the shock.

## Why 3 RBPFs Instead of 1?

A single RBPF with multiple regimes has to **adapt** when dynamics change. Particles are stuck with their current assumptions and slowly drift to new values.

```
Single RBPF at crisis onset:
  Tick 0:   Particles at μ=-4.5, φ=0.98 (calm dynamics, wrong)
  Tick 50:  Particles slowly drifting, still confused
  Tick 100: Finally adapted to crisis
```

With 3 parallel RBPFs, each assumes different **dynamics** (not just different μ_vol):

| Hypothesis | φ (persistence) | σ_eta (vol noise) | Meaning |
|------------|-----------------|-------------------|---------|
| Calm   | 0.98 | 0.10 | Vol shocks die fast |
| Trend  | 0.95 | 0.20 | Vol moves persist |
| Crisis | 0.80 | 0.50 | Vol is explosive |

The Crisis-RBPF is **always running** with crisis-appropriate dynamics. When crisis hits:

```
MMPF at crisis onset:
  Tick 0:  Crisis-RBPF already has good estimates (it was warm)
           Just upweight Crisis from 10% → 80%
  Tick 1:  Done
```

**No adaptation. No particle drift. Just reweight.**

All the IMM math is just the rigorous justification. The practical win is simple:

> 1 RBPF has to adapt → slow
> 3 RBPFs: one is already warm → instant

## Why Not Just Use BOCPD Alone?

BOCPD knows *when* something changed, not *what* changed to or *what σ is*.

| System | Knows When | Knows Which Regime | Knows σ |
|--------|------------|-------------------|---------|
| BOCPD  | ✓          | ✗                 | ✗       |
| RBPF   | ✗          | ✗                 | ✓       |
| MMPF   | Slow       | ✓                 | ✓ (weighted) |
| All 3  | ✓ (fast)   | ✓                 | ✓       |

**Who does what:**
- **BOCPD**: "Something broke at tick 847" (detection)
- **MMPF**: "We're 73% Crisis, 20% Trend, 7% Calm" (classification)  
- **RBPF**: "σ = 4.2% under the Crisis hypothesis" (estimation)

## Why Not Just Use MMPF Alone?

MMPF's sticky transitions (98% stay) make it stable but slow:

```
Regime change at t=0:
  t=0:   Calm 98%, Crisis 2%
  t=10:  Calm 85%, Crisis 15%
  t=50:  Calm 40%, Crisis 60%
  t=100: Calm 10%, Crisis 90%
```

~100 ticks to fully switch. Too slow for trading.

With BOCPD shock:

```
Regime change at t=0:
  t=0:   BOCPD fires shock
         Calm 33%, Trend 33%, Crisis 33%  (uniform)
         Likelihood scores each hypothesis
  t=1:   Calm 5%, Trend 10%, Crisis 85%   (winner emerges)
  t=2:   Back to sticky mode, Crisis dominates
```

~1-2 ticks to switch. Fast enough.

## The Delta Detector (No Hand-Tuned Thresholds)

The naive approach:
```c
if (delta > 0.3) shock();  // Where does 0.3 come from? 🤷
```

The Storvik approach:
```c
// Learn what "normal" delta looks like
storvik_update(&detector, delta);

// Fire on 3σ deviation from learned distribution
z_score = (delta - learned_mean) / learned_std;
if (z_score > 3.0) shock();
```

The threshold calibrates itself. No magic numbers.

## Power-Law Hazard

BOCPD needs a "hazard function" — prior probability of changepoint.

**Constant hazard** (H = 1/λ):
- Assumes geometric regime durations
- Wrong for finance (regimes have heavy-tailed durations)

**Power-law hazard** (H(r) = α/(r+1)):
- New regimes are fragile (high hazard when r is small)
- Old regimes are stable (low hazard when r is large)
- Matches empirical regime durations

```
r=0:   H = 0.80  (new regime, very fragile)
r=10:  H = 0.07  (stabilizing)
r=100: H = 0.008 (entrenched, hard to break)
```

## Failure Modes & Mitigations

| Failure | Consequence | Mitigation |
|---------|-------------|------------|
| BOCPD false positive | One wasted tick, uniform exploration | Likelihood corrects immediately |
| BOCPD false negative | Slow regime switch (~100 ticks) | HMM anchor prevents drift |
| Repeated false positives | MMPF never settles, noisy σ | Cooldown period between shocks |
| Gradual drift (boiling frog) | BOCPD misses slow changes | Drift detector on posterior mean |

## The Full Stack

```
Timescale        Component        Question                    Output
──────────────────────────────────────────────────────────────────────────
Daily            ICEEMDAN         "Is there signal?"          GO/NO-GO
Hours            HMM              "What do regimes look like?" θ_anchor for priors
Tick (1st)       BOCPD            "Did it break?"             shock signal
Tick (2nd)       MMPF             "Which regime?"             [w_calm, w_trend, w_crisis]
                  └─ RBPF ×3      "What's σ under each?"      σ_calm, σ_trend, σ_crisis
Tick (3rd)       Kelly            "How much to bet?"          position size
```

## Code Pattern

```c
// Initialize
bocpd_hazard_init_power_law(&hazard, 0.8, 1024);
bocpd_init_with_hazard(&bocpd, &hazard, prior);
bocpd_delta_init(&delta, 100);
mmpf = mmpf_create(&config);

// Each tick
bocpd_step(&bocpd, observation);
double d = bocpd_delta_update(&delta, bocpd.r, bocpd.active_len, 0.995);

if (bocpd_delta_check(&delta, 3.0)) {
    mmpf_inject_shock(mmpf);
    mmpf_step(mmpf, observation, &output);
    mmpf_restore_from_shock(mmpf);
} else {
    mmpf_step(mmpf, observation, &output);
}

// Use output
double sigma = output.volatility;
double kelly_fraction = compute_kelly(mu, sigma, regime_weights);
```

## Why This Works for Trading

1. **σ estimation is bulletproof**: RBPF + KSC mixture handles fat tails correctly
2. **Regime detection is fast**: BOCPD shock cuts latency from ~100 to ~2 ticks
3. **No feedback loops**: BOCPD is purely external watchdog, no circular dependencies
4. **Self-calibrating**: Storvik learns thresholds, no hand-tuning
5. **Robust to errors**: False positives waste one tick, likelihood corrects

## OCSN: Per-RBPF Outlier Handling

Each RBPF needs its **own OCSN** (Outlier Component Selection Network). Why?

Same observation, different interpretations:

| Hypothesis | μ_vol anchor | 5% move | Verdict |
|------------|--------------|---------|---------|
| Calm   | -4.83 (0.8% vol) | 6σ | Outlier! |
| Crisis | -3.35 (3.5% vol) | 1.4σ | Normal |

With shared OCSN, you get mixed signals. Crisis-RBPF sees normal data, but shared OCSN screams "outlier" because Calm-RBPF is confused.

**Solution:** Use `RBPF_Extended` which bundles RBPF + Storvik + OCSN per instance:

```c
MMPF
 ├── RBPF_Extended[Calm]   (own OCSN, judges outliers under calm assumption)
 ├── RBPF_Extended[Trend]  (own OCSN, judges outliers under trend assumption)
 └── RBPF_Extended[Crisis] (own OCSN, judges outliers under crisis assumption)
```

OCSN provides ~10× better tail handling. Each hypothesis needs to judge "is this an outlier **in my world**" independently.

## Storvik Learning: Climate vs Weather

### The Icarus Paradox

If each RBPF independently learns μ_vol from the same data, they all converge to the global mean:

```
μ_calm   → μ_global
μ_trend  → μ_global  
μ_crisis → μ_global
```

Result: Identical likelihoods. Weights freeze at 33/33/33. Discrimination dies.

### The Solution: Global Baseline + Fixed Offsets + Gated Dynamics

```
                 ┌──────────────────┐
                 │   Global μ_base  │ ← EWMA on IMM weighted output
                 │   (slow drift)   │
                 └────────┬─────────┘
                          │
         ┌────────────────┼────────────────┐
         │                │                │
         ▼                ▼                ▼
   ┌───────────┐    ┌───────────┐    ┌───────────┐
   │   CALM    │    │   TREND   │    │  CRISIS   │
   │ μ=base-1.0│    │ μ=base    │    │ μ=base+1.5│
   │ φ,σ_η     │    │ φ,σ_η     │    │ φ,σ_η     │
   │ (gated)   │    │ (gated)   │    │ (gated)   │
   └─────┬─────┘    └─────┬─────┘    └─────┬─────┘
         │                │                │
         └────────────────┼────────────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │   IMM Weights    │
                 │  (discrimination)│
                 └──────────────────┘
```

**What gets learned where:**

| Parameter | Source | Mechanism | Why |
|-----------|--------|-----------|-----|
| μ_vol | Global + Offset | EWMA + constant | Defines "where" regime sits. Fixed offset guarantees discrimination. |
| φ | Gated Storvik | Weight by w_regime | Defines "how" shocks decay. Crisis learns fast reversion, Calm learns persistence. |
| σ_η | Gated Storvik | Weight by w_regime | Defines "how" vol wiggles. Crisis learns erratic, Calm learns smooth. |

### Why Gated Learning Works

**The "Structural Memory" Effect:**

```
2020 (Crisis):  Crisis RBPF dominant (w≈0.9)
                → Learns: φ=0.85, σ_η=0.5 (fast reversion, wild swings)
                
2021-2023:      Calm RBPF dominant (w≈0.9)
                → Crisis RBPF frozen at φ=0.85, σ_η=0.5
                → Crisis state tracks data (badly), but params preserved
                
2024 (Shock):   Data jumps to crisis levels
                → Crisis model IMMEDIATELY has correct dynamics
                → No warm-up period needed
```

The frozen Crisis model is like a fire extinguisher — you don't want it "learning" from years of no fires.

### State vs Parameters

Critical distinction:

| What | Updates When | Why |
|------|-------------|-----|
| **State** (x_t, particles) | Every tick | Must track current vol level for likelihood |
| **Parameters** (φ, σ_η) | Only when w_regime > threshold | Preserve regime-specific dynamics |

During calm periods:
- Crisis state tracks data (produces terrible likelihood, which is correct!)
- Crisis params frozen (φ, σ_η preserved from last crisis)
- Crisis weight → 0 (as it should be)

When crisis hits:
- Data matches Crisis hypothesis
- Crisis likelihood spikes instantly
- Crisis has correct dynamics immediately (no adaptation needed)

### Gated Sufficient Statistics Update

Don't use hard threshold (too choppy). Weight by regime probability:

```c
// Standard Storvik update:
S_t = S_{t-1} + SuffStat(y_t, x_t)

// Gated Storvik update:
S_t = S_{t-1} + w_regime * SuffStat(y_t, x_t)
```

When w ≈ 1.0: Full learning, model adapts to "now"
When w ≈ 0.0: Sufficient stats freeze, parameters stay constant
When w ≈ 0.3: Slow drift, acknowledges data without overhauling worldview

### Implementation

```c
typedef struct {
    /* Sufficient statistics for φ (AR coefficient) */
    double sum_xy;      // Σ w * x_{t-1} * x_t
    double sum_xx;      // Σ w * x_{t-1}²
    
    /* Sufficient statistics for σ_η (innovation variance) */
    double sum_resid_sq;  // Σ w * (x_t - φ*x_{t-1} - (1-φ)*μ)²
    double sum_weight;    // Σ w (effective sample size)
    
    /* Current estimates */
    double phi;
    double sigma_eta;
} GatedDynamicsLearner;

void gated_dynamics_update(GatedDynamicsLearner *learner,
                           double x_prev, double x_curr,
                           double mu_anchor, double regime_weight)
{
    double centered_prev = x_prev - mu_anchor;
    double centered_curr = x_curr - mu_anchor;
    
    /* Accumulate weighted sufficient statistics */
    learner->sum_xy += regime_weight * centered_prev * centered_curr;
    learner->sum_xx += regime_weight * centered_prev * centered_prev;
    
    double predicted = learner->phi * centered_prev;
    double residual = centered_curr - predicted;
    learner->sum_resid_sq += regime_weight * residual * residual;
    learner->sum_weight += regime_weight;
    
    /* Batch update when enough weight accumulated */
    if (learner->sum_weight > 10.0) {
        /* φ = Σxy / Σxx (weighted OLS) */
        learner->phi = learner->sum_xy / (learner->sum_xx + 1e-10);
        learner->phi = fmax(0.5, fmin(0.999, learner->phi));
        
        /* σ_η² = Σresid² / Σw */
        learner->sigma_eta = sqrt(learner->sum_resid_sq / learner->sum_weight);
        
        /* Exponential forgetting */
        double forget = 0.99;
        learner->sum_xy *= forget;
        learner->sum_xx *= forget;
        learner->sum_resid_sq *= forget;
        learner->sum_weight *= forget;
    }
}
```

### Main Loop Integration

```c
void mmpf_step(MMPF_ROCKS *mmpf, rbpf_real_t ret, MMPF_Output *out) {
    
    /* 1. Update global baseline (slow EWMA on previous output) */
    mmpf->global_mu_vol = 0.999 * mmpf->global_mu_vol 
                        + 0.001 * mmpf->prev_weighted_log_vol;
    
    /* 2. Reanchor each hypothesis (μ = global + offset) */
    for (int k = 0; k < MMPF_N_MODELS; k++) {
        rbpf_real_t mu_k = mmpf->global_mu_vol + mmpf->config.mu_offsets[k];
        rbpf_ext_set_mu_vol(mmpf->ext[k], mu_k);
        rbpf_ext_set_phi(mmpf->ext[k], mmpf->dynamics[k].phi);
        rbpf_ext_set_sigma_eta(mmpf->ext[k], mmpf->dynamics[k].sigma_eta);
    }
    
    /* 3. Run IMM step (state estimation + likelihood) */
    // ... existing logic ...
    
    /* 4. Gated parameter learning (AFTER getting weights) */
    rbpf_real_t weights[MMPF_N_MODELS];
    mmpf_get_weights(mmpf, weights);
    
    for (int k = 0; k < MMPF_N_MODELS; k++) {
        rbpf_real_t x_curr = mmpf->ext[k]->rbpf->mean_state;
        rbpf_real_t x_prev = mmpf->prev_state[k];
        rbpf_real_t mu_k = mmpf->global_mu_vol + mmpf->config.mu_offsets[k];
        
        gated_dynamics_update(&mmpf->dynamics[k], 
                              x_prev, x_curr, mu_k, weights[k]);
        mmpf->prev_state[k] = x_curr;
    }
    
    mmpf->prev_weighted_log_vol = out->log_volatility;
}
```

### Why This Can't Converge

| Parameter | Calm Learns From | Crisis Learns From | Converge? |
|-----------|------------------|-------------------|-----------|
| μ_vol | N/A (fixed offset) | N/A (fixed offset) | **No** (structural) |
| φ | Calm data only | Crisis data only | **No** (different data) |
| σ_η | Calm data only | Crisis data only | **No** (different data) |

Each model learns from its own regime's data:
- **Calm** sees smooth, persistent data → learns φ→0.98, σ_η→0.10
- **Crisis** sees volatile, mean-reverting data → learns φ→0.85, σ_η→0.50

They CAN'T converge because they're trained on fundamentally different distributions.

### BOCPD Integration: Spread Widening

When BOCPD fires, temporarily widen the μ_vol offsets to explore more aggressively:

```c
void mmpf_on_changepoint(MMPF_ROCKS *mmpf) {
    /* Normal offsets: [-1.0, 0.0, +1.5]
     * Crisis exploration: [-1.5, 0.0, +2.5]
     */
    mmpf->offset_scale = 1.5;
    mmpf->offset_decay_ticks = 50;
}

// In mmpf_step:
rbpf_real_t offset = mmpf->config.mu_offsets[k] * mmpf->offset_scale;
if (mmpf->offset_scale > 1.0) {
    mmpf->offset_scale *= 0.98;  // Decay back to normal
}
```

This lets IMM "cast a wider net" right after a structural break, then tighten back to normal.

### Summary

| Component | Role | Adapts? |
|-----------|------|---------|
| Global μ_base | Secular drift (decade-scale) | Yes, slow EWMA |
| μ_vol offsets | Regime identity | **No** (constants) |
| φ per-regime | Shock decay dynamics | Yes, gated Storvik |
| σ_η per-regime | Vol-of-vol dynamics | Yes, gated Storvik |
| State (particles) | Current vol level | Yes, every tick |

**Adaptation to the decade. Discrimination of the day. Exploration on breaks.**

## Bottom Line

> Separate the "something changed" detector from the "what is it" tracker.
> 
> BOCPD is the smoke alarm. MMPF is the fire investigator.
> 
> The smoke alarm doesn't need to know what's burning. It just needs to wake everyone up fast.