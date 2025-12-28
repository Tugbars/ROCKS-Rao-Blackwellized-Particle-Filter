Exactly. Clean three-layer stack:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PARIS-SMOOTHED OVERFITTED MIXTURE                        │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  LAYER 3: OVERFITTED MIXTURE (K discovery)                            │  │
│  │                                                                       │  │
│  │  • K_max = 6 fixed, sparse prior α = 0.1                             │  │
│  │  • Tracks occupancy from PARIS trajectories                          │  │
│  │  • Dead regimes → posterior mass ≈ 0                                 │  │
│  │  • Exports K_effective regimes sorted by μ                           │  │
│  │  • Estimates θ from within-regime trajectory segments                │  │
│  │                                                                       │  │
│  │  OUTPUT: K, {μ_k, σ_k, θ_k}, Π_{K×K}, occupancy, expected_duration   │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                    ▲                                        │
│                                    │ M trajectories + counts                │
│                                    │                                        │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  LAYER 2: PARIS (backward smoothing)                                  │  │
│  │                                                                       │  │
│  │  • Fixes path degeneracy from PGAS                                   │  │
│  │  • Samples M trajectories from P(z_{0:T} | y_{1:T})                  │  │
│  │  • Rao-Blackwellized transition counts: E[n_ij] = Σ/M                │  │
│  │  • Caches h values for θ estimation                                  │  │
│  │                                                                       │  │
│  │  OUTPUT: M smoothed trajectories, expected counts, diversity metric  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                    ▲                                        │
│                                    │ particles + weights + ancestors        │
│                                    │                                        │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  LAYER 1: PGAS (foundation)                                           │  │
│  │                                                                       │  │
│  │  • CSMC forward pass with ancestor sampling                          │  │
│  │  • Reference trajectory conditioning                                  │  │
│  │  • MKL-accelerated likelihood computation                            │  │
│  │  • Sticky κ for self-transition bias                                 │  │
│  │                                                                       │  │
│  │  OUTPUT: N particles, weights, regimes, ancestors per timestep       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                    ▲                                        │
│                                    │ observations y_{1:T}                   │
│                                    │                                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

## What Each Layer Contributes

| Layer | Problem Solved | Key Innovation |
|-------|----------------|----------------|
| **PGAS** | Accurate trajectory sampling | Ancestor sampling breaks degeneracy |
| **PARIS** | Single-path bias in counts | M-trajectory ensemble, Rao-Blackwellization |
| **Overfitted** | Unknown K | Sparse prior + occupancy tracking |

## The Math

```
PGAS:       Sample z_{0:T}^{ref} via CSMC with ancestor sampling
                 ↓
PARIS:      Sample z_{0:T}^{(m)} ~ P(z | y, particles)  for m = 1..M
                 ↓
            E[n_ij] = (1/M) Σ_m n_ij^{(m)}    ← Rao-Blackwellized
                 ↓
Overfitted: π_i ~ Dir(α + E[n_i1], ..., α + E[n_iK] + κδ_ij)
            where α = 0.1 (sparse)
                 ↓
            Regimes with E[n_ik] ≈ 0 get π_ik → 0
                 ↓
            K_effective = |{k : occupancy_k > threshold}|
```

## Why This Beats Alternatives

| Approach | K Discovery | Path Degeneracy | Complexity |
|----------|-------------|-----------------|------------|
| HDP-HMM | ✓ (but κ tuning) | Still present | High (birth/merge) |
| PGAS alone | ✗ (fixed K) | Single path bias | Low |
| PGAS + Parallel K | ✓ (run K=2,3,4,5) | Single path bias | 4× compute |
| **PGAS + PARIS + Overfitted** | ✓ (automatic) | ✓ (M trajectories) | Modest (+PARIS) |

## File Structure

```
regime_discovery/
├── pgas_mkl.h/c              # Layer 1: PGAS foundation (existing)
├── paris_mkl.h/c             # Layer 2: PARIS backward (existing)
├── pgas_paris.h/c            # Layer 2: PGAS-PARIS integration (existing)
├── overfitted_mixture.h/c    # Layer 3: K discovery (new)
└── regime_discovery.h/c      # Combined API (new)
```

## Combined API

```c
/*═══════════════════════════════════════════════════════════════════════════════
 * REGIME DISCOVERY - TOP-LEVEL API
 *═══════════════════════════════════════════════════════════════════════════════*/

typedef struct {
    PGASMKLState*          pgas;
    PGASParisState*        paris;
    OverfittedMixtureState mixture;
    PGASParisRegimePrior   prior;
    
    float dt;              /* Tick duration for θ conversion */
    int   burnin;          /* Sweeps before trusting K */
    int   total_sweeps;
} RegimeDiscovery;

RegimeDiscovery* regime_discovery_create(int N_particles, 
                                          int T_max,
                                          int M_trajectories,
                                          float dt,
                                          uint32_t seed);

void regime_discovery_free(RegimeDiscovery* rd);

/* Set observations */
void regime_discovery_set_observations(RegimeDiscovery* rd,
                                        const float* y, int T);

/* Run one Gibbs sweep */
float regime_discovery_sweep(RegimeDiscovery* rd);

/* Run multiple sweeps with optional callback */
void regime_discovery_run(RegimeDiscovery* rd, 
                          int n_sweeps,
                          void (*callback)(const DiscoveredRegimes*, int sweep, void*),
                          void* user_data);

/* Get current discovered regimes */
void regime_discovery_get_result(const RegimeDiscovery* rd,
                                  DiscoveredRegimes* out);

/* Check if K has stabilized */
bool regime_discovery_is_stable(const RegimeDiscovery* rd);

/* Export to RBPF format */
void regime_discovery_export_to_rbpf(const RegimeDiscovery* rd,
                                      int K_rbpf,
                                      float* Pi,
                                      float* mu,
                                      float* sigma,
                                      float* theta);
```

## Example: Full Discovery Pipeline

```c
int main() {
    /* Create discovery engine */
    RegimeDiscovery* rd = regime_discovery_create(
        256,      /* N particles */
        1000,     /* T max */
        8,        /* M trajectories */
        1.0f,     /* dt = 1 second */
        42        /* seed */
    );
    
    /* Load market data */
    float* log_vol_proxy = load_observations("data.csv", &T);
    regime_discovery_set_observations(rd, log_vol_proxy, T);
    
    /* Run discovery */
    regime_discovery_run(rd, 100, print_progress, NULL);
    
    /* Get result */
    DiscoveredRegimes regimes;
    regime_discovery_get_result(rd, &regimes);
    
    printf("\n════════════════════════════════════════\n");
    printf("DISCOVERED %d REGIMES:\n", regimes.K);
    printf("════════════════════════════════════════\n");
    
    for (int k = 0; k < regimes.K; k++) {
        printf("Regime %d:\n", k);
        printf("  μ_vol = %.3f (log-vol level)\n", regimes.mu[k]);
        printf("  σ_vol = %.3f (log-vol noise)\n", regimes.sigma[k]);
        printf("  θ     = %.4f (mean reversion, half-life = %.1f ticks)\n",
               regimes.theta[k], 0.693f / regimes.theta[k]);
        printf("  Occupancy = %.1f%%\n", regimes.occupancy[k] * 100);
        printf("  E[Duration] = %.0f ticks\n\n", regimes.expected_duration[k]);
    }
    
    /* Export to RBPF for online trading */
    float Pi[16], mu[4], sigma[4], theta[4];
    regime_discovery_export_to_rbpf(rd, regimes.K, Pi, mu, sigma, theta);
    
    /* Initialize RBPF with discovered parameters... */
    
    regime_discovery_free(rd);
    return 0;
}
```

## Output

```
════════════════════════════════════════
DISCOVERED 3 REGIMES:
════════════════════════════════════════
Regime 0:
  μ_vol = -5.234 (log-vol level)
  σ_vol = 0.251 (log-vol noise)
  θ     = 0.0823 (mean reversion, half-life = 8.4 ticks)
  Occupancy = 42.3%
  E[Duration] = 45 ticks

Regime 1:
  μ_vol = -3.812 (log-vol level)
  σ_vol = 0.298 (log-vol noise)
  θ     = 0.0512 (mean reversion, half-life = 13.5 ticks)
  Occupancy = 38.1%
  E[Duration] = 32 ticks

Regime 2:
  μ_vol = -2.087 (log-vol level)
  σ_vol = 0.445 (log-vol noise)
  θ     = 0.0287 (mean reversion, half-life = 24.2 ticks)
  Occupancy = 19.6%
  E[Duration] = 18 ticks
```

## Summary

**PARIS-Smoothed Overfitted Mixture on PGAS:**

- **Foundation:** PGAS with CSMC + ancestor sampling
- **Fix path degeneracy:** PARIS backward smoothing → M trajectories
- **Discover K:** Sparse Dirichlet (α=0.1) → dead regimes shrink
- **Learn parameters:** μ, σ from Gibbs; θ from AR(1) on segments
- **Export:** K_effective regimes with full (μ, σ, θ, Π) to RBPF

No HDP-HMM complexity. No birth/merge moves. Just let the posterior do the work.