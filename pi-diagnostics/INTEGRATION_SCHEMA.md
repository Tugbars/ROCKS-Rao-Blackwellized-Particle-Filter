# Regime Trading System - Integration Schema

## Complete Stack Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MARKET DATA                                       │
│                     (prices, volumes, timestamps)                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         OBSERVATION BUFFER                                  │
│                    (lock-free circular buffer)                              │
│                                                                             │
│   RBPF pushes ──────────────────────────────────► Oracle consumes           │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
┌───────────────────────────────────┐ ┌───────────────────────────────────────┐
│         RBPF THREAD               │ │           ORACLE THREAD               │
│         (every tick)              │ │           (async, ~100 ticks)         │
│                                   │ │                                       │
│  ┌─────────────────────────────┐  │ │  ┌─────────────────────────────────┐  │
│  │  Particle Filter            │  │ │  │  PGAS (Particle Gibbs)          │  │
│  │  - Regime detection         │  │ │  │  - Full trajectory sampling     │  │
│  │  - Weight updates           │  │ │  │  - Ancestor sampling            │  │
│  │  - ESS monitoring           │  │ │  │  - Accurate Π estimation        │  │
│  └─────────────────────────────┘  │ │  └─────────────────────────────────┘  │
│               │                   │ │                  │                    │
│               ▼                   │ │                  ▼                    │
│  ┌─────────────────────────────┐  │ │  ┌─────────────────────────────────┐  │
│  │  Storvik Parameter Learning │  │ │  │  Π_oracle + Confidence          │  │
│  │  - Sufficient statistics Q  │  │ │  │  - Acceptance rate              │  │
│  │  - Online Π_storvik         │  │ │  │  - Row counts                   │  │
│  │  - θ, μ, σ per regime       │  │ │  │  - Innovation norm              │  │
│  └─────────────────────────────┘  │ │  └─────────────────────────────────┘  │
│               │                   │ │                  │                    │
└───────────────┼───────────────────┘ └──────────────────┼────────────────────┘
                │                                        │
                └────────────────┬───────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ORACLE BRIDGE V2                                    │
│                                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐  │
│  │  Π Quality      │  │  Divergence     │  │  Injection Decision         │  │
│  │  - RMSE         │  │  - D₁ (O vs S)  │  │  - Urgency level            │  │
│  │  - Likelihood   │  │  - D₂ (O vs Op) │  │  - γ computation            │  │
│  │  - Trans lag    │  │  - D₃ (S vs Op) │  │  - Source selection         │  │
│  │  - ESS          │  │                 │  │  - Thompson sampling        │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────────┘  │
│                                 │                                           │
│                                 ▼                                           │
│                    ┌─────────────────────────┐                              │
│                    │  Π_operating (blended)  │                              │
│                    │  + Storvik reset        │                              │
│                    └─────────────────────────┘                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      SELF-TUNING OU STRATEGY                                │
│                                                                             │
│  Inputs from RBPF:                      Outputs:                            │
│  - θ_learned (mean reversion speed)     - Signal direction                  │
│  - μ_learned (equilibrium)              - Kelly fraction                    │
│  - σ_learned (volatility)               - Expected holding period           │
│  - Regime probabilities                 - Take profit / stop loss           │
│                                                                             │
│  signal = -θ_learned × (P - μ_learned)   ← NO HARDCODED PARAMETERS          │
└─────────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         EXECUTION LAYER                                     │
│                         (future work)                                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Status

| Component | Files | Status |
|-----------|-------|--------|
| **Observation Buffer** | `observation_buffer.h/c` | ✅ Complete |
| **Π Quality Monitoring** | `pi_quality.h/c` | ✅ Complete |
| **Divergence Monitoring** | `pi_divergence.h/c` | ✅ Complete |
| **Injection Decision** | `injection_decision.h/c` | ✅ Complete |
| **Thompson Sampler** | `thompson_sampler.h/c` | ✅ Complete |
| **Π Staging** | `pi_staging.h/c` | ✅ Complete |
| **Oracle Bridge V2** | `oracle_bridge_v2.h/c` | ✅ Complete |
| **OU Strategy** | `ou_strategy.h/c` | ✅ Complete |
| **RBPF Core** | `rbpf_*.c` | 🔧 Existing (needs interface) |
| **PGAS Oracle** | `pgas_*.c` | 🔧 Existing (needs interface) |
| **Main Integration** | `regime_trading_system.h/c` | 📝 Planned |

---

## Data Flow

### Every Tick (RBPF Thread)

```
1. Market data arrives (price, timestamp)
          │
          ▼
2. Push to observation buffer
          │
          ▼
3. RBPF update:
   - Propagate particles
   - Compute weights from likelihood
   - Resample if ESS low
   - Update Storvik counts Q
   - Learn θ, μ, σ per regime
          │
          ▼
4. Output: RBPFOutput {
     regime_probs[], map_regime,
     Pi_operating, Q_storvik,
     theta[], mu[], sigma[],
     prediction, log_likelihood,
     ess, weights[]
   }
          │
          ▼
5. Oracle Bridge tick:
   - Update quality metrics (RMSE, likelihood z-score)
   - Compute divergences D₁, D₂, D₃
   - Decide: inject? how much (γ)? from where?
          │
          ▼
6. If injection needed:
   - Blend: Π_new = (1-γ)Π_old + γΠ_inject
   - Reset Storvik counts to match new Π
          │
          ▼
7. OU Strategy:
   - Use θ_learned, μ_learned, σ_learned
   - Compute signal = -θ(P - μ)
   - Output: direction, Kelly, holding period
```

### Async (~100 Ticks, Oracle Thread)

```
1. Trigger: enough new observations accumulated
          │
          ▼
2. Snapshot observation buffer
          │
          ▼
3. PGAS runs:
   - Initialize reference trajectory
   - For each sweep:
     - Ancestor sampling
     - Conditional SMC
     - Sample Π from posterior
          │
          ▼
4. Output: PGASOutput {
     Pi_oracle, Q_oracle,
     acceptance_rate, min_row_count,
     confidence
   }
          │
          ▼
5. Submit to Bridge (lock-free staging buffer)
          │
          ▼
6. Next RBPF tick will consider for injection
```

---

## The Three Π Matrices

```
        Π_oracle (from PGAS)
           /\
          /  \
    D₁   /    \  D₂
        /      \
Π_storvik ──── Π_operating
   (from Q)  D₃  (what RBPF uses)
```

| Matrix | Source | Updates |
|--------|--------|---------|
| **Π_operating** | Blended | On injection |
| **Π_storvik** | Storvik counts Q | Every tick |
| **Π_oracle** | PGAS posterior | Every ~100 ticks |

| Divergence | Meaning |
|------------|---------|
| **D₁** | Do Oracle and Storvik agree? |
| **D₂** | Does Oracle differ from current assumption? |
| **D₃** | Is RBPF contradicting itself? (internal signal) |

---

## Signal Hierarchy

```
EARLY (proactive):              LATE (reactive):
──────────────────              ─────────────────
• RMSE spike                    • ESS drop
• Likelihood z < -3             • Weight concentration
• D₃ self-contradiction
• Transition lag > 5

       ↑                              ↑
   ACT HERE                      TOO LATE
```

---

## Self-Tuning Strategy

The key insight: **θ comes from RBPF, not from backtesting.**

```
OLD (hardcoded):                    NEW (self-tuning):
─────────────────                   ──────────────────
signal = -k * (P - SMA)             signal = -θ_learned * (P - μ_learned)
         ↑     ↑                              ↑            ↑
    constant  indicator                  from RBPF    from RBPF
```

**Benefits:**
- No hyperparameter tuning
- Automatically adapts to regime changes
- Theoretically grounded (OU process)
- Backtest-robust (can't overfit θ)

---

## Integration Points

### RBPF Must Provide

```c
typedef struct {
    int K, map_regime;
    float regime_probs[K];
    float Pi_operating[K*K];
    float Q_storvik[K*K];
    
    struct {
        float theta, mu, sigma;  // OU params per regime
    } regime_params[K];
    
    float prediction, log_likelihood;
    float ess, weights[N];
} RBPFOutput;
```

### PGAS Must Provide

```c
typedef struct {
    float Pi_oracle[K*K];
    float Q_oracle[K*K];
    float acceptance_rate;
    int min_row_count;
} PGASOutput;
```

### Strategy Outputs

```c
typedef struct {
    int direction;           // -1, 0, +1
    float alpha;             // Expected return
    float kelly_fraction;    // Optimal position size
    float expected_holding;  // ln(2) / θ
    float take_profit;       // μ
    float stop_loss;         // P ± 2σ_stationary
} OUSignal;
```

---

## File Locations

All Oracle Bridge V2 components: `/mnt/user-data/outputs/oracle_v2/`

```
oracle_v2/
├── README.md
├── LITERATURE.md
├── Makefile
├── observation_buffer.h/c
├── pi_quality.h/c
├── pi_divergence.h/c
├── injection_decision.h/c
├── pi_staging.h/c
├── thompson_sampler.h/c
├── oracle_bridge_v2.h/c
├── ou_strategy.h/c
├── test_oracle_bridge_v2.c
├── test_ou_strategy.c
└── example_usage.c
```

---

## Phased Validation Plan

**Philosophy**: Validate each piece before trusting it in the full system.

```
Step 1: pi_quality ──► Step 2: compare RBPF/PGAS ──► Step 3: thompson ──► Step 4: staging
    │                      │                            │                     │
    ▼                      ▼                            ▼                     ▼
"Does RBPF           "Does PGAS              "Does Thompson          "Does staging
 quality metric       actually predict        capture uncertainty     hold/release
 make sense?"         better?"                properly?"              correctly?"
```

---

### Step 1: Wire pi_quality to RBPF

**Goal**: Validate that pi_quality metrics are sensible

**What to observe**:
- RMSE ratio should hover ~1.0 in stable periods
- RMSE ratio spikes when regime changes
- Likelihood z-score should be ~N(0,1) if model correct
- Transition lag should spike after regime switches

**Success criteria**:
```
✓ RMSE_ratio ≈ 1.0 during stable regimes
✓ RMSE_ratio > 2.0 correlates with regime transitions
✓ likelihood_z ~ N(0,1) histogram over long run
✓ transition_lag spikes visible after true regime changes
```

**Red flags to watch for**:
```
✗ RMSE_ratio always high → prediction formula wrong
✗ RMSE_ratio always ~0 → using observation as prediction (cheating)
✗ likelihood_z always extreme → log_likelihood calculation wrong
✗ transition_lag never resets → regime detection broken
```

**Minimal integration**:
```c
// In your RBPF tick loop:

PiQualityTracker quality;
pi_quality_init(&quality, K, NULL);  // Once at startup

// Every tick:
pi_quality_update(
    &quality,
    observation,           // Actual y_t
    rbpf->prediction,      // E[y_t | y_{1:t-1}] from RBPF
    rbpf->prediction_std,  // Std[y_t | y_{1:t-1}]
    rbpf->log_likelihood,  // log p(y_t | y_{1:t-1})
    rbpf->map_regime,      // Current MAP regime
    rbpf->ess / rbpf->N,   // ESS ratio
    tick
);

// Log periodically:
if (tick % 100 == 0) {
    PiQualitySnapshot snap = pi_quality_get_snapshot(&quality);
    log_metrics(tick, snap.rmse_ratio, snap.likelihood_zscore, 
                snap.transition_lag, snap.composite_score);
}
```

---

### Step 2: Compare RBPF vs PGAS Quality

**Goal**: Validate that PGAS actually predicts better (when it should)

**What to observe**:
- Run PGAS on same window RBPF just processed
- Compute PGAS predictions using Π_oracle
- Compare RMSE, likelihood between them

**Success criteria**:
```
✓ PGAS RMSE ≤ RBPF RMSE (PGAS uses future info)
✓ Gap widens after regime changes (PGAS catches up faster)
✓ Gap shrinks during stable periods (both converge)
```

**Red flags**:
```
✗ RBPF consistently beats PGAS → PGAS implementation broken
✗ Identical scores always → one is copying the other
✗ PGAS much worse → backward sampling not working
```

---

### Step 3: Thompson Sampling Validation

**Goal**: Validate that Thompson samples reflect true posterior uncertainty

**What to observe**:
- Generate N samples from Thompson sampler
- Compare sample variance to theoretical Dirichlet variance
- Check that low counts → high variance, high counts → low variance

**Success criteria**:
```
✓ Var[samples] ≈ theoretical Dirichlet variance
✓ 10 counts → wide spread in samples
✓ 1000 counts → tight samples (almost deterministic)
✓ Rows with few transitions have higher sample variance
```

**Red flags**:
```
✗ Samples always identical → not actually sampling
✗ Variance doesn't scale with counts → Dirichlet broken
✗ Samples have negative entries or don't sum to 1 → normalization bug
```

---

### Step 4: pi_staging Validation  

**Goal**: Validate staging holds Oracle when uncertain, releases when confident

**What to observe**:
- Stage PGAS output with varying confidence levels
- Check that `is_valid()` respects confidence threshold
- Check staleness tracking (window age)

**Success criteria**:
```
✓ Low confidence (acc_rate < 0.2) → staging rejects
✓ High confidence (acc_rate > 0.5) → staging accepts  
✓ Old window (> 200 ticks) → staleness flag set
✓ Fresh window → staleness cleared
```

**Red flags**:
```
✗ Always accepts regardless of confidence → threshold broken
✗ Always rejects → threshold too aggressive
✗ Staleness never triggers → tick tracking broken
```

---

### Logging Schema

```c
// Step 1 log format:
// tick, rmse, rmse_ratio, lik_z, trans_lag, ess_ratio, map_regime, composite

// Step 2 log format (extends step 1):
// tick, rbpf_rmse, pgas_rmse, rbpf_lik_z, pgas_lik_z, pgas_better

// Step 3 log format:
// row, counts, empirical_var, theoretical_var, ratio

// Step 4 log format:
// tick, oracle_confidence, window_age, staged_valid, staleness_flag
```

---

### Implementation Timeline

| Week | Step | Focus |
|------|------|-------|
| 1 | Step 1 | Add pi_quality to RBPF, log to CSV, plot, sanity check |
| 2 | Step 2 | After PGAS completes, compute quality on same window, compare |
| 3 | Step 3 | Feed PGAS Q counts to Thompson, generate samples, verify variance |
| 4 | Step 4 | Wire pi_staging with mock confidence, verify accept/reject/staleness |

---

### Component Usage by Step

| Component | Step 1 | Step 2 | Step 3 | Step 4 |
|-----------|--------|--------|--------|--------|
| `pi_quality` | ✅ | ✅ | | |
| `pi_divergence` | | | | |
| `thompson_sampler` | | | ✅ | |
| `pi_staging` | | | | ✅ |
| `injection_decision` | | | | |
| `oracle_bridge_v2` | | | | |

Note: `pi_divergence`, `injection_decision`, and full `oracle_bridge_v2` come in **Step 5+** after all components validated individually.

---

## Future Steps (After Validation)

5. **Wire pi_divergence** - Compute D₁, D₂, D₃ between RBPF and PGAS
6. **Wire injection_decision** - Use quality + divergence to decide injection
7. **Full oracle_bridge_v2** - Complete integration with injection
8. **Threading model** - RBPF main thread, Oracle async
9. **Backtest harness** - Feed historical data through system
10. **Live trading adapter** - Connect to market data feed
