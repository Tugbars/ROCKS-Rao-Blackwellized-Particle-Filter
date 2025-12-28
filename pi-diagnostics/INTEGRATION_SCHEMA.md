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

## Threading Architecture

### The Problem

```
┌─────────────────────────────────────────────────────────────────────┐
│                          32+ CORE MACHINE                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   RBPF (2 threads)              PGAS (30 threads)                  │
│   ┌─────┐ ┌─────┐               ┌─────────────────────────┐        │
│   │ T0  │ │ T1  │               │ T0 T1 T2 ... T29        │        │
│   │     │ │     │               │                         │        │
│   │particles    │               │ parallel sweeps         │        │
│   │split        │               │                         │        │
│   └─────┘ └─────┘               └─────────────────────────┘        │
│       │     │                              │                        │
│       └──┬──┘                              │                        │
│          │                                 │                        │
│          ▼                                 ▼                        │
│   ┌─────────────┐                 ┌─────────────────┐              │
│   │ MKL calls   │                 │ MKL calls       │              │
│   │ (small)     │                 │ (per sweep)     │              │
│   └─────────────┘                 └─────────────────┘              │
│                                                                     │
│   Question: How to isolate RBPF's MKL from PGAS's MKL?             │
└─────────────────────────────────────────────────────────────────────┘
```

### Solution: MKL Sequential + Our Threading

```c
/* CRITICAL: At program startup */
mkl_set_threading_layer(MKL_THREADING_SEQUENTIAL);
```

**What this means:**

| Setting | Effect |
|---------|--------|
| `MKL_THREADING_SEQUENTIAL` | MKL never spawns internal threads |
| Your 30 threads call MKL | 30 parallel MKL operations |
| Each gets hand-tuned kernel | ✅ AVX-512, cache-optimized |

**Without this setting:**
```
30 PGAS threads × 30 MKL internal threads = 900 threads (chaos)
```

**With this setting:**
```
30 PGAS threads × 1 MKL thread each = 30 hand-tuned parallel ops (correct)
```

### MKL Functions Used

**RBPF (`rbpf_ksc.c`):**
```c
/* VML - Thread-safe */
vsExp() / vdExp()          // Batch exp()
vsLn() / vdLn()            // Batch log()

/* Level-1 BLAS - Thread-safe */
cblas_sasum()              // Sum of absolute values
cblas_sscal()              // Scale vector  
cblas_sdot()               // Dot product

/* VSL RNG - Per-stream required */
vsRngGaussian()            // Batch normal samples
```

**PGAS (`pgas_mkl.c`):**
```c
/* VML - Thread-safe */
vsExp(), vsLn()

/* Level-1 BLAS - Thread-safe */
cblas_isamax(), cblas_sasum(), cblas_sscal(), cblas_sdot()

/* VSL RNG - Per-stream required */
vsRngGaussian(), vsRngUniform(), vsRngGamma(), viRngUniform()
```

**All thread-safe except VSL RNG** - each thread needs its own `VSLStreamStatePtr`.

### PGAS Already Has Per-Thread Resources

```c
/* From pgas_mkl.c - already implemented! */
state->thread_rng_streams[i]     // ✅ Up to PGAS_MKL_MAX_THREADS

state->thread_ws[i].log_bw       // ✅ Per-thread workspace
state->thread_ws[i].bw
state->thread_ws[i].workspace
state->thread_ws[i].cumsum
```

### Parallelism Strategy: Parallel Sweeps

```
┌─────────────────────────────────────────────────────────────────────┐
│  PARALLEL SWEEPS (Independent Gibbs chains)                         │
│                                                                     │
│  Thread 0 ──► sweep 0, 30, 60...  ──► Π sample 0                   │
│  Thread 1 ──► sweep 1, 31, 61...  ──► Π sample 1                   │
│  Thread 2 ──► sweep 2, 32, 62...  ──► Π sample 2                   │
│  ...                                                                │
│  Thread 29 ──► sweep 29, 59, 89... ──► Π sample 29                 │
│                                                                     │
│  Each thread: own RNG stream, own workspace, own PGAS state copy   │
│  Result: 30 independent Π samples → aggregate for final estimate   │
└─────────────────────────────────────────────────────────────────────┘
```

### Implementation Pattern

```c
typedef struct {
    PGASMKLState* states[32];      /* Per-thread PGAS state */
    float Pi_samples[32][64];      /* Per-thread Π samples (K*K max) */
    int n_threads;
    int K;
} PGASParallel;

void pgas_parallel_run(PGASParallel* pp, int sweeps_per_thread)
{
    const int n_threads = pp->n_threads;
    const int K = pp->K;
    
    #pragma omp parallel num_threads(n_threads)
    {
        int tid = omp_get_thread_num();
        PGASMKLState* my_state = pp->states[tid];
        
        /* Each thread runs independent sweeps */
        for (int s = 0; s < sweeps_per_thread; s++) {
            pgas_mkl_gibbs_sweep(my_state);
        }
        
        /* Store final Π sample */
        memcpy(pp->Pi_samples[tid], my_state->model.trans, 
               K * K * sizeof(float));
    }
}

void pgas_parallel_aggregate(PGASParallel* pp, float* Pi_mean, float* Pi_var)
{
    /* Average 30 Π samples → Pi_mean */
    /* Compute variance → Pi_var (for confidence) */
}
```

### Complete Threading Picture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         SYSTEM                                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   RBPF (2 pthreads)              PGAS (30 OpenMP threads)          │
│   Cores 0-1                      Cores 2-31                        │
│                                                                     │
│   ┌─────────────────┐            ┌─────────────────────────────┐   │
│   │ pthread 0       │            │ #pragma omp parallel        │   │
│   │ particles 0-127 │            │ num_threads(30)             │   │
│   │ mkl_rng[0]      │            │ {                           │   │
│   │                 │            │   int tid = get_thread_num  │   │
│   │ pthread 1       │            │   PGASMKLState* my = [tid]  │   │
│   │ particles 128+  │            │                             │   │
│   │ mkl_rng[1]      │            │   for (s = 0; s < 3; s++)   │   │
│   └─────────────────┘            │     gibbs_sweep(my)         │   │
│          │                       │                             │   │
│          │                       │   // Each thread: own RNG   │   │
│          ▼                       │   // own workspace          │   │
│   vsRngGaussian(rng[0])          │   // own PGAS state         │   │
│   vsRngGaussian(rng[1])          │ }                           │   │
│   cblas_sdot()                   │                             │   │
│   vsExp()                        │ // Aggregate 30 Π samples   │   │
│                                  └─────────────────────────────┘   │
│                                             │                       │
│                                             ▼                       │
│                                    vsRngGaussian(stream_tid)       │
│                                    vsRngGamma(stream_tid)          │
│                                    cblas_sdot()                    │
│                                    vsExp()                         │
│                                                                     │
│   All MKL calls use hand-tuned AVX-512 kernels                     │
│   No MKL internal threads (MKL_THREADING_SEQUENTIAL)               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Threading Summary

| Aspect | RBPF | PGAS |
|--------|------|------|
| Threading model | 2 pthreads | 30 OpenMP threads |
| Parallelism type | Split particles | Independent sweeps |
| RNG streams | 2 VSL streams (`mkl_rng[0..1]`) | 30 VSL streams (per-thread state) |
| Workspaces | Per-thread | Per-thread (already in code) |
| MKL mode | Sequential (no internal threads) | Sequential (no internal threads) |
| Core affinity | Cores 0-1 | Cores 2-31 |

### Startup Code

```c
void init_system(void) {
    /* CRITICAL: Disable MKL internal threading */
    mkl_set_threading_layer(MKL_THREADING_SEQUENTIAL);
    
    /* Verify */
    printf("MKL threads: %d (should be 1)\n", mkl_get_max_threads());
    
    /* Set OpenMP for PGAS */
    omp_set_num_threads(30);
    
    /* Optional: Core affinity */
    // export OMP_PLACES="{2:30}"
    // export OMP_PROC_BIND=close
}
```

---

## Future Steps (After Validation)

5. **Wire pi_divergence** - Compute D₁, D₂, D₃ between RBPF and PGAS
6. **Wire injection_decision** - Use quality + divergence to decide injection
7. **Full oracle_bridge_v2** - Complete integration with injection
8. **Threading model** - RBPF main thread, Oracle async
9. **Backtest harness** - Feed historical data through system
10. **Live trading adapter** - Connect to market data feed