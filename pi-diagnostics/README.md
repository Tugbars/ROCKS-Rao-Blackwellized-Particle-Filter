# Oracle Bridge V2

## Clean Dual-Loop Architecture for RBPF Π Correction

### The Core Insight

**ESS is a late symptom. By the time ESS drops, the damage is done.**

```
Π wrong → Bad predictions → Weight skew → ESS drops → Collapse
   ↑                                                      ↑
ROOT CAUSE                                           TOO LATE
(act here)                                          (act here?)
```

This system monitors **Π accuracy directly** through its downstream effects:
- Prediction RMSE
- Transition detection lag
- Marginal likelihood
- Self-contradiction (D₃)

## Architecture

```
                      Observation Buffer
                    (Lock-Free Circular)
                            │
              ┌─────────────┴─────────────┐
              │                           │
              ▼                           ▼
   ┌────────────────────┐     ┌────────────────────┐
   │    RBPF Thread     │     │   Oracle Thread    │
   │    (Consumer)      │     │   (Producer)       │
   │                    │     │                    │
   │  - Uses Π          │     │  - Owns ref path   │
   │  - Filters z,h     │◄────│  - Full Gibbs      │
   │  - Monitors quality│  Π  │  - Learns Π        │
   │                    │     │                    │
   └────────────────────┘     └────────────────────┘
```

## The Three Π Matrices

```
        Π_oracle
           /\
          /  \
    D₁   /    \  D₂
        /      \
Π_storvik ──── Π_operating
           D₃
```

| Matrix | Description |
|--------|-------------|
| Π_operating | What RBPF uses for predictions |
| Π_storvik | What RBPF learns online (from counts) |
| Π_oracle | What PGAS suggests (ground truth) |

| Divergence | Meaning |
|------------|---------|
| D₁ = KL(Π_oracle ‖ Π_storvik) | Oracle vs RBPF learning |
| D₂ = KL(Π_oracle ‖ Π_operating) | Oracle vs RBPF assumption |
| D₃ = KL(Π_storvik ‖ Π_operating) | **RBPF self-contradiction** |

**D₃ is the hidden gold:** RBPF can detect its own internal conflict without Oracle!

## Signal Hierarchy

### Early Signals (Act Immediately)
1. **RMSE spike** - predictions went bad → Π is wrong NOW
2. **Likelihood z-score < -3** - data "impossible" under model
3. **D₃ high** - RBPF's learning contradicts its assumption
4. **Transition lag increasing** - Π diagonal too sticky

### Medium Signals (Prepare for Action)
5. **Weight variance high** - particles disagree
6. **Hawkes intensity high** - market active

### Late Signals (Damage Done)
7. **ESS dropping** - Π has been wrong for a while
8. **Max weight > 50%** - particle monopoly

## Files

| File | Purpose |
|------|---------|
| `observation_buffer.h/c` | Lock-free circular buffer for observations |
| `pi_quality.h/c` | Direct Π quality metrics (RMSE, lag, likelihood) |
| `pi_divergence.h/c` | Three-way divergence monitoring (D₁, D₂, D₃) |
| `injection_decision.h/c` | Urgency computation + injection decision |
| `pi_staging.h/c` | Lock-free Π handoff from Oracle to RBPF |
| `thompson_sampler.h/c` | Dirichlet sampling for variance shots |
| `oracle_bridge_v2.h/c` | Main coordinator |

## Usage

### 1. Initialize

```c
ObservationBuffer obs_buffer;
obs_buffer_init(&obs_buffer);

OracleBridgeV2 bridge;
OracleBridgeConfig config = oracle_bridge_config_defaults(K);

oracle_bridge_init(&bridge, &config, &obs_buffer, Pi_initial);
```

### 2. RBPF Thread (Every Tick)

```c
// Push observation
obs_buffer_push(&obs_buffer, observation, tick);

// Your RBPF update
rbpf_update(&rbpf, observation);

// Update bridge with Storvik counts
oracle_bridge_update_storvik(&bridge, rbpf.Q_storvik);

// Get injection decision
InjectionDecision dec = oracle_bridge_rbpf_tick(
    &bridge,
    observation,
    rbpf.prediction,
    rbpf.log_likelihood,
    rbpf.map_regime,
    rbpf.weights,
    rbpf.N,
    tick);

// Apply if needed
if (dec.should_inject) {
    oracle_bridge_apply_injection(&bridge, &dec);
    
    // Update your RBPF's Π
    memcpy(rbpf.Pi, oracle_bridge_get_pi(&bridge), K * K * sizeof(float));
    memcpy(rbpf.Q, oracle_bridge_get_storvik(&bridge), K * K * sizeof(float));
}
```

### 3. Oracle Thread (Async)

```c
// After PGAS completes
oracle_bridge_submit_oracle(
    &bridge,
    Pi_oracle,           // Learned Π
    Q_oracle,            // Sufficient stats
    acceptance_rate,     // For confidence
    min_row_count,       // For confidence
    sweeps_used,
    window_start,
    window_end);
```

## Injection Decision Matrix

```
                    D₃ (Storvik vs Operating)
                LOW                    HIGH
           (self-consistent)    (self-contradicting)
      ┌──────────────────┬──────────────────┐
  LOW │   ALL AGREE      │   RBPF DRIFTING  │
D₁,D₂ │   Do nothing     │   Inject Storvik │
      ├──────────────────┼──────────────────┤
 HIGH │   ORACLE SEES    │   ORACLE +       │
D₁,D₂ │   SOMETHING NEW  │   STORVIK AGREE  │
      │   Check Oracle   │   STRONGEST      │
      │   confidence     │   signal! High γ │
      └──────────────────┴──────────────────┘
```

## γ Decision Matrix

```
                    Oracle Confidence
               LOW (<0.35)   MED (0.35-0.6)   HIGH (>0.6)
         ┌─────────────┬──────────────┬──────────────┐
NONE     │   0.05      │    0.10      │   0.15       │
         ├─────────────┼──────────────┼──────────────┤
LOW      │   0.08      │    0.15      │   0.25       │
         ├─────────────┼──────────────┼──────────────┤
MEDIUM   │   0.12      │    0.22      │   0.35       │
         ├─────────────┼──────────────┼──────────────┤
HIGH     │   0.18      │    0.30      │   0.45       │
         ├─────────────┼──────────────┼──────────────┤
EMERGENCY│   0.25†     │    0.40†     │   0.55†      │
         └─────────────┴──────────────┴──────────────┘
         † = use Thompson sample (variance shot)
```

## Thompson Sampling (Variance Shot)

When RBPF is critical and Oracle confidence is low, we don't just inject the Oracle's point estimate. We sample from the posterior to inject variance (exploration):

```c
// Each row: Π_i ~ Dir(Q_i + α)
thompson_sample_with_confidence(
    Q_oracle,
    prior_alpha,
    K,
    oracle_confidence,  // Blend between mean and sample
    Pi_inject,
    &rng);
```

## Building

```bash
make           # Build library
make test      # Run tests
make example   # Run integration example
make clean     # Clean
```

## Key Design Decisions

1. **PGAS is independent** - generates own trajectories, no RBPF dependency
2. **Three Π matrices** - Operating, Storvik, Oracle
3. **D₃ self-contradiction** - RBPF can self-diagnose without Oracle
4. **Urgency hierarchy** - ESS > D₃+D₁ > D₃ alone > D₂ > Hawkes+KL
5. **Simple blend** - `(1-γ)Π_old + γΠ_new` with Storvik reset
6. **Thompson variance shot** - when RBPF critical and Oracle uncertain
7. **Lock-free buffers** - observations (circular) and Π staging (double-buffer)

## Storvik Reset ("Lobotomy")

Critical: After injection, we must reset Storvik counts to match new Π:

```c
// If we only update Π_operating but leave Q_storvik alone,
// Storvik will pull Π back to old value in next tick!

for (int i = 0; i < K; i++) {
    for (int j = 0; j < K; j++) {
        Q_storvik[i * K + j] = Pi_operating[i * K + j] * effective_count;
    }
}
```

This gives Storvik a "clean slate" to start learning from the new Π.
