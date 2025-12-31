# Recursive PGAS: Discounted Dirichlet Counts

## Problem Statement

### The Amnesia Problem

Standard sliding-window PGAS is **structurally amnesiac**:

```
Window 1: [tick 0-499]   → counts crisis transitions → learns Π
Window 2: [tick 500-999] → crisis gone, counts reset → forgets crisis
```

When a short crisis (10-50 ticks) leaves the window, all evidence of it vanishes. The system "snaps back" to calm-dominated transition estimates.

**Observed symptoms:**
- Slow Trend accuracy: +35.6% (PGAS helps ✓)
- Sudden Crisis accuracy: -16.8% (PGAS hurts ✗)
- Crisis Persistence accuracy: -7.7% (PGAS hurts ✗)

PGAS learns that "crisis entry is rare" (statistically correct) but this blocks RBPF from detecting real crises (operationally wrong).

### Root Cause

PGAS conditions on a **finite trajectory** of length T. When the window slides:
- States before t-T are integrated out
- Sufficient statistics are lost
- Transition learning becomes IID across windows

This is **not fixable inside PGAS**. Memory must live outside the window.

---

## Solution: Discounted Dirichlet Counts

### Core Idea

Maintain **persistent sufficient statistics** that accumulate across windows with exponential decay:

```
C_new = ρ × C_old + N_this_window
```

Where:
- `C` = accumulated transition counts (persists forever)
- `ρ` = forgetting factor (0.99-0.999)
- `N` = counts from current PGAS window

Then sample transitions from the accumulated history:

```
Π[i,:] ~ Dirichlet(α + C[i,:] + κ×I[i==j])
```

### Why This Works

| Property | Why It Matters |
|----------|----------------|
| Counts, not probabilities | Preserves distribution sharpness |
| ρ decay | Recent data dominates, no hard cutoff |
| Persistent C | Crisis evidence survives window slides |
| Dirichlet sampling | Guaranteed valid stochastic matrix |

### Mathematical Justification

This is **not** ad-hoc blending. It is:

> "Discounted Bayesian updating of Dirichlet natural parameters"

The Dirichlet distribution is conjugate to the multinomial. Counts are the natural parameters. Exponential forgetting in count space is the principled way to handle non-stationary transition dynamics.

---

## Implementation

### Data Structures

```c
typedef struct {
    /* ... existing fields ... */
    
    /* Recursive PGAS */
    float accumulated_counts[K * K];  /* Persistent history */
    float discount_rho;               /* Forgetting factor */
} PGASMKLState;
```

### Core Algorithm

```c
void pgas_mkl_sample_transitions(PGASMKLState *state)
{
    /* STEP 1: Count transitions from THIS window */
    int n_trans[K * K] = {0};
    for (t = 1; t < T; t++) {
        n_trans[from * K + to]++;
    }

    /* STEP 2: Recursive accumulation */
    for (i = 0; i < K * K; i++) {
        accumulated_counts[i] = ρ × accumulated_counts[i] + n_trans[i];
    }

    /* STEP 3: Sample Dirichlet from accumulated counts */
    for (i = 0; i < K; i++) {
        for (j = 0; j < K; j++) {
            α[j] = prior_alpha 
                 + accumulated_counts[i * K + j]
                 + sticky_kappa × (i == j);
            
            /* Ensure crisis entry is never blocked */
            if (j == CRISIS && α[j] < crisis_entry_prior) {
                α[j] = crisis_entry_prior;
            }
        }
        Π[i,:] ~ Dirichlet(α);
    }
}
```

### Enhancement: Expected Counts from PARIS

**Current approach:** Count from single sampled trajectory (reference path)
```c
n_trans[ref_regimes[t-1] * K + ref_regimes[t]]++;
```

**Better approach:** Use PARIS smoothed distribution to get expected counts
```c
/* After PARIS backward smoothing */
for (t = 1; t < T; t++) {
    for (n = 0; n < N; n++) {
        int from = smoothed_regimes[t-1][n];
        int to = smoothed_regimes[t][n];
        expected_counts[from * K + to] += weights[n];
    }
}
```

**Why expected counts are better:**
- Single-path counts have high variance
- PARIS provides N smoothed trajectories
- Expected counts = weighted average over all particles
- Reduces noise, often makes RBPF blending unnecessary

**Trade-off:** Slightly more computation, but PARIS is already running.
```

### Parameter Guidelines

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `discount_rho` | 0.995 | Half-life ≈ 138 iterations |
| `prior_alpha` | 1.0 | Base regularization (prevents zeros) |
| `sticky_kappa` | 50.0 | Diagonal stickiness (anti-chatter) |
| `crisis_entry_prior` | 5.0 | Minimum mass for calm→crisis (optional) |

**⚠️ Important: ρ is calibrated in PGAS update units, not ticks**

If PGAS slides every 50 ticks:
```
ρ = 0.995, half-life = 138 iterations × 50 ticks = 6,900 ticks
ρ = 0.99,  half-life = 69 iterations × 50 ticks  = 3,450 ticks
```

**Half-life calculation:**
```
half_life_iterations = ln(0.5) / ln(ρ)
half_life_ticks = half_life_iterations × slide_step

ρ = 0.99  → half-life ≈ 69 iterations
ρ = 0.995 → half-life ≈ 138 iterations
ρ = 0.999 → half-life ≈ 693 iterations
```

**Steady-state count mass:**
```
C_∞ = N_avg / (1 - ρ)

ρ = 0.995, N_avg = 30 → C_∞ ≈ 6000 per cell
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      PGAS Oracle                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. PGAS CSMC Sweep                                         │
│     └─> Samples trajectory z[1:T] in current window         │
│                                                             │
│  2. Count Transitions                                       │
│     └─> N[i→j] from sampled trajectory                      │
│                                                             │
│  3. Recursive Accumulation                                  │
│     └─> C = ρ × C + N                                       │
│                                                             │
│  4. Sample Π from Dirichlet(α + C + κI)                     │
│     └─> Thompson sampling (preserves rare transitions)      │
│                                                             │
│  5. Inject Π into RBPF                                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Information Flow

```
PGAS Window (local)     Accumulated Counts (global)      RBPF (online)
      │                         │                            │
      │   N[i,j]                │                            │
      ├────────────────────────>│                            │
      │                         │                            │
      │                    C = ρC + N                        │
      │                         │                            │
      │                         │   Π ~ Dir(α + C)           │
      │                         ├───────────────────────────>│
      │                         │                            │
```

---

## What We Keep vs Remove

### Keep

| Component | Purpose |
|-----------|---------|
| PGAS CSMC | Core trajectory inference |
| PARIS smoothing | Expected counts (lower variance) |
| Dirichlet sampling | Valid stochastic matrix |
| Sticky kappa | Anti-chatter |
| Adaptive kappa (optional) | Matches observed switching rate |

### Remove / Replace

| Old | New | Reason |
|-----|-----|--------|
| Per-window counts | Accumulated counts | Memory across windows |
| Hard count reset | ρ decay | Smooth forgetting |
| Fixed window learning | Recursive learning | Non-amnesiac |

### Tuning Constraint: Crisis Entry Prior

Even with Thompson sampling, ensure calm→crisis is never effectively blocked:

```c
/* After computing α from accumulated counts */
const float CRISIS_ENTRY_FLOOR = 5.0f;  /* Minimum pseudo-counts */

for (int i = 0; i < K - 1; i++) {  /* Non-crisis rows */
    int crisis_col = (K - 1);
    if (alpha[i * K + crisis_col] < CRISIS_ENTRY_FLOOR) {
        alpha[i * K + crisis_col] = CRISIS_ENTRY_FLOOR;
    }
}
```

**Why this matters:**
- If accumulated counts for calm→crisis are near zero
- And prior_alpha is small (e.g., 1.0)
- Thompson sampling will almost never sample crisis entry
- Floor ensures P(→crisis) ≥ FLOOR / (row_sum) ≈ 1-5%

---

## Theoretical Backing

### Literature Support

1. **Bazzi et al. 2017** - Time-Varying Transition Probabilities (TVTP)
2. **Diebold & Filardo 1994** - Covariate-driven transitions
3. **Andrieu et al. 2010** - Particle MCMC with adaptive transitions

### Key Insight from Literature

> "A single matrix learned from a window must compromise between calm and crisis. Don't force that."

Our solution: Don't learn from one window. Accumulate evidence across all windows with decay.

### Bayesian Interpretation

The update rule:
```
C_new = ρ × C_old + N
```

Is equivalent to:
```
posterior ∝ likelihood × discounted_prior
```

Where the "discounted prior" is yesterday's posterior, geometrically down-weighted.

---

## Future Extensions (If Needed)

### 1. SR Signal Injection

Add crisis signal as "virtual counts":

```c
if (j == CRISIS && sr_signal > threshold) {
    α[j] += f(sr_signal);  // Virtual evidence for crisis
}
```

### 2. RBPF-Side Blending

If injection causes jitter:

```c
Π_RBPF = (1 - λ) × Π_RBPF + λ × Π_PGAS
```

But try without blending first—accumulated counts already provide smoothing.

### 3. Dynamic Dirichlet Evolution

More principled time-varying model:

```
Π[i,:,t] | Π[i,:,t-1] ~ Dirichlet(κ × Π[i,:,t-1])
```

### 4. Two Time-Scale Counts

Separate fast/slow memory:

```c
C_fast = ρ_fast × C_fast + N   // ρ = 0.95, reacts quickly
C_slow = ρ_slow × C_slow + N   // ρ = 0.999, long memory

C_combined = w × C_fast + (1-w) × C_slow
```

---

## Success Criteria

| Metric | Target | Rationale |
|--------|--------|-----------|
| Slow Trend accuracy | ≥ 90% | Maintain PGAS learning benefit |
| Sudden Crisis accuracy | ≥ 70% | Restore crisis detection |
| Overall accuracy | ≥ 63% | Match or beat baseline |

---

## Summary

**Problem:** Windowed PGAS forgets crises when they leave the window.

**Solution:** Accumulate transition counts across windows with exponential decay.

**Implementation:** `C = ρ × C + N`, then `Π ~ Dirichlet(α + C)`.

**Why it's principled:** Discounted Bayesian updating in natural parameter space.

**Status:** Implemented in `pgas_mkl.c`. Ready for testing.
