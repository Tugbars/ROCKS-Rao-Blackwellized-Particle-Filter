# PGAS Oracle + RBPF Integration Plan v2.0

## Overview

Hybrid architecture where **PGAS provides structure (Π)** and **Storvik tracks state (μ, σ)**.

**New in v2.0:** Hawkes-gated crisis management with physics-based exit detection and adaptive Storvik transition learning during crisis.

```
┌─────────────────────────────────────────────────────────────────┐
│                        TICK LOOP (Core 0-1)                     │
│                                                                 │
│   Observation ──→ Hawkes ──→ RBPF ──→ Output                   │
│                     │          ↑                                │
│                     │    ┌─────┴─────┐                         │
│                   Veto?  │  Mailbox  │ (Lock-free)             │
│                     │    └─────┬─────┘                         │
│                     ↓          │                                │
│              [Crisis: Storvik] [Calm: Inject Π]                │
└─────────────────────────────────────────────────────────────────┘
                                 ↑
                                 │ Publishes Π every 50 ticks
┌─────────────────────────────────────────────────────────────────┐
│                     PGAS ORACLE (Cores 2-7)                     │
│                                                                 │
│   Ring Buffer ──→ Sliding Window ──→ CSMC ──→ Dirichlet ──→ Π  │
│      (T=500)         (S=50)        (N=64)                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component Configuration

### PGAS Sliding Window

```c
const int WINDOW_SIZE = 500;   // T - ticks of history for Π estimation
const int SLIDE_STEP  = 50;    // S - publish frequency
const int N_PARTICLES = 64;    // Particle count (sweet spot)
const int N_SWEEPS    = 2;     // CSMC sweeps per window
const int K           = 3;     // Regimes: CALM, TREND, CRISIS
```

**Performance:** ~7ms per window, 500ms wall time at 100Hz

### MKL Threading

```c
mkl_tuning_init(8, 0);  // 8 P-cores, quiet mode
```

**Finding:** 8 threads faster than 32 for N=64 particles (better granularity).

### Dirichlet Prior

```c
const float PRIOR_ALPHA   = 1.0f;   // Base concentration
const float STICKY_KAPPA  = 50.0f;  // Diagonal boost (stickiness)
const float RECENCY_LAMBDA = 0.002f; // Exponential decay
```

---

## The "Split Brain" Reset Strategy

**Core Insight:** PGAS dictates the *rules* (Π), Storvik tracks the *score* (μ, σ).

| Parameter | Type | Owner | Reset on Inject? | Reason |
|-----------|------|-------|------------------|--------|
| Π (transition matrix) | Structure | PGAS | ✓ **YES** | Storvik can't learn in 50 ticks |
| μ (price level) | State | Storvik | ✗ **NO** | Tracks continuously, needs history |
| σ (volatility) | State | Storvik | ✗ **NO** | Tracks continuously, needs history |

### Why This Works

- **Π needs ~500+ ticks** to see enough regime transitions
- **μ/σ adapt in ~10-20 ticks** to local price movements
- Resetting μ/σ would "blind" the filter, causing 20-tick re-learning lag
- Resetting Π prevents Storvik from drifting on noise

### Implementation

```c
void inject_physics_only(RBPF* rbpf, const float* Pi_pgas, float N_eff) {
    
    /* ═══════════════════════════════════════════════════════════════
     * 1. RESET STRUCTURE (Transition Matrix)
     *    Storvik's short-term transition memory is statistically weak.
     *    We overwrite it with PGAS's estimate.
     * ═══════════════════════════════════════════════════════════════*/
    for (int i = 0; i < K; i++) {
        float row_total = 0.0f;
        
        for (int j = 0; j < K; j++) {
            float new_count = Pi_pgas[i * K + j] * N_eff;
            rbpf->counts_N_ij[i][j] = new_count;
            row_total += new_count;
        }
        rbpf->counts_N_i[i] = row_total;
    }
    
    /* Rebuild LUTs so next resampling uses new probabilities */
    rbpf_lut_rebuild_transitions(rbpf);
    
    /* ═══════════════════════════════════════════════════════════════
     * 2. PRESERVE STATE (Emission Statistics) - DO NOT TOUCH
     *    Storvik needs continuous history for smooth price tracking.
     * ═══════════════════════════════════════════════════════════════*/
    // rbpf->stats_sum_y[k]      <-- KEEP
    // rbpf->stats_sum_y2[k]     <-- KEEP  
    // rbpf->stats_count_obs[k]  <-- KEEP
}
```

### N_eff Scaling

```c
// Tie confidence to PGAS window size
float N_eff = (float)PGAS_WINDOW_SIZE * 2.0f;  // T=500 → N_eff=1000
```

---

## Hawkes Crisis Detection

### Timing Hierarchy

Each component has a specific reaction time and role:

| Component | Latency | Role | Action |
|-----------|---------|------|--------|
| **Hawkes** | <10ms | Flash detection | Veto injection, manage crisis state |
| **RBPF** | 10ms | State tracking | Track μ/σ tick-by-tick |
| **PGAS** | 500ms | Historian | Analyze past, publish Π |

### Key Principle

> **PGAS should never be the first responder. It's the historian.**

- **Onset (0-100ms):** Hawkes sees intensity spike instantly, vetoes PGAS
- **Duration (100ms+):** RBPF + Storvik ride the wave with local learning
- **Aftermath (500ms+):** PGAS analyzes crash data, updates Π for "new normal"

---

## Crisis State Machine (v2.0)

The old refractory-based approach (500-tick timeout) is replaced with **physics-based exit detection**.

### State Diagram

```
                        ┌─────────────────────────────────────────┐
                        │                                         │
                        ▼                                         │
    IDLE ──[onset]──→ ARMED ──[confirm]──→ CRISIS_ACTIVE ────────┘
     ▲                                          │      (> T ticks: PGAS adapted)
     │                                          │
     │                                          │ [intensity < baseline
     │                                          │  AND surprise < 1.5σ
     │                                          │  for N consecutive ticks]
     │                                          ▼
     └────────────────────────────────── RECOVERING
```

### State Definitions

| State | PGAS Veto | λ | Storvik Π Learning | Description |
|-------|-----------|---|-------------------|-------------|
| **IDLE** | No | N/A | OFF | Peace time, PGAS owns Π |
| **ARMED** | Yes | 0.95 | SHADOW | Probation period, learning in parallel |
| **CRISIS_ACTIVE** (< T) | Yes | 0.60 | ON | Early crisis, PGAS Π stale |
| **CRISIS_ACTIVE** (≥ T) | No | 0.95 | ON | Deep crisis, PGAS adapted |
| **RECOVERING** | Yes | 0.90 | OFF | Transitional, confirming exit |

---

## Shadow Mode (False Alarm Protection)

### The Problem: Amnesia Cost

When Hawkes triggers, immediately resetting Storvik to weak prior (N=2) is dangerous:

**Scenario:** Fat finger trade—one bad tick, then back to normal.

**Cost without protection:**
- You wiped your high-quality PGAS prior (N=1000)
- For ~50 ticks, filter runs on "Amnesia Mode" with weak uniform matrix
- Small secondary ripples cause erratic regime switching
- You threw away the "Peace Map" too fast

### The Solution: Probation Period

Use ARMED state as a **shadow mode**—learn in parallel, don't commit until crisis confirms.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              IDLE                                       │
│   Using: PGAS Π (N_eff = 1000)                                         │
│   Storvik: OFF                                                          │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │ Hawkes onset detected
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         ARMED (Shadow Mode)                             │
│                                                                         │
│   ┌─────────────┐    ┌──────────────────────────────────┐              │
│   │ PGAS Buffer │    │ Shadow Storvik (N=2, λ=0.6)      │              │
│   │ (frozen)    │    │ Learning in parallel, NOT USED   │              │
│   └─────────────┘    └──────────────────────────────────┘              │
│                                                                         │
│   RBPF still uses: PGAS Π                                              │
│   Duration: 3-5 ticks                                                   │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
            [Crisis confirms]              [Blip - returns to calm]
                    │                               │
                    ▼                               ▼
┌───────────────────────────────┐   ┌───────────────────────────────────┐
│       CRISIS_ACTIVE           │   │            IDLE                   │
│                               │   │                                   │
│ Commit: Shadow → Active       │   │ Restore: PGAS Buffer → Active    │
│ Discard: PGAS Buffer          │   │ Discard: Shadow Storvik           │
│ Use: Storvik Π (already warm) │   │ Use: PGAS Π (never lost it)       │
└───────────────────────────────┘   └───────────────────────────────────┘
```

### Data Structures

```c
typedef struct {
    /* ... existing RBPF ... */
    
    /* Primary transition matrix (always used by filter) */
    float trans_counts[K][K];
    float trans_row_sums[K];
    
    /* Shadow Storvik (runs during ARMED, not used until confirmed) */
    float shadow_counts[K][K];
    float shadow_row_sums[K];
    bool  shadow_active;
    
    /* PGAS buffer (frozen at ARMED entry) */
    float pgas_buffer_counts[K][K];
    float pgas_buffer_row_sums[K];
    bool  pgas_buffered;
    
    /* Crisis state */
    bool  storvik_learning_active;  /* True only after CRISIS_ACTIVE confirmed */
    
} RBPF;
```

### Enter Shadow Mode (IDLE → ARMED)

```c
void rbpf_enter_shadow_mode(RBPF *rbpf) {
    /* 1. Buffer current PGAS matrix (don't touch primary) */
    memcpy(rbpf->pgas_buffer_counts, rbpf->trans_counts, sizeof(rbpf->trans_counts));
    memcpy(rbpf->pgas_buffer_row_sums, rbpf->trans_row_sums, sizeof(rbpf->trans_row_sums));
    rbpf->pgas_buffered = true;
    
    /* 2. Initialize shadow Storvik with weak prior */
    const float weak_prior = 2.0f;
    for (int i = 0; i < K; i++) {
        rbpf->shadow_row_sums[i] = 0.0f;
        for (int j = 0; j < K; j++) {
            rbpf->shadow_counts[i][j] = weak_prior;
            rbpf->shadow_row_sums[i] += weak_prior;
        }
    }
    rbpf->shadow_active = true;
    
    /* 3. Primary matrix UNCHANGED - filter still uses PGAS Π */
    /* rbpf->storvik_learning_active stays false */
}
```

### Update Shadow (During ARMED)

```c
void rbpf_shadow_update(RBPF *rbpf, float lambda) {
    if (!rbpf->shadow_active) return;
    
    /* Update shadow counts in parallel - NOT used by filter yet */
    for (int i = 0; i < K; i++) {
        rbpf->shadow_row_sums[i] = 0.0f;
        for (int j = 0; j < K; j++) {
            rbpf->shadow_counts[i][j] *= lambda;
            rbpf->shadow_row_sums[i] += rbpf->shadow_counts[i][j];
        }
    }
    
    /* Count transitions from particle cloud into SHADOW */
    for (int p = 0; p < N_PARTICLES; p++) {
        int from = rbpf->particles[p].prev_regime;
        int to   = rbpf->particles[p].regime;
        float w  = rbpf->particles[p].weight;
        
        rbpf->shadow_counts[from][to] += w;
        rbpf->shadow_row_sums[from] += w;
    }
    
    /* Note: We do NOT rebuild primary LUT here */
}
```

### Confirm Crisis (ARMED → CRISIS_ACTIVE)

```c
void rbpf_confirm_crisis(RBPF *rbpf) {
    /* Shadow has been learning for 3-5 ticks - it's warm */
    
    /* 1. Promote shadow to primary */
    memcpy(rbpf->trans_counts, rbpf->shadow_counts, sizeof(rbpf->trans_counts));
    memcpy(rbpf->trans_row_sums, rbpf->shadow_row_sums, sizeof(rbpf->trans_row_sums));
    
    /* 2. Rebuild LUT with crisis-adapted matrix */
    rbpf_lut_rebuild_transitions(rbpf);
    
    /* 3. Activate ongoing learning */
    rbpf->storvik_learning_active = true;
    rbpf->shadow_active = false;
    
    /* 4. PGAS buffer no longer needed */
    rbpf->pgas_buffered = false;
}
```

### False Alarm Recovery (ARMED → IDLE)

```c
void rbpf_abort_crisis(RBPF *rbpf) {
    /* It was a blip - restore PGAS matrix */
    
    if (rbpf->pgas_buffered) {
        /* 1. Restore buffered PGAS matrix */
        memcpy(rbpf->trans_counts, rbpf->pgas_buffer_counts, sizeof(rbpf->trans_counts));
        memcpy(rbpf->trans_row_sums, rbpf->pgas_buffer_row_sums, sizeof(rbpf->trans_row_sums));
        
        /* 2. Rebuild LUT with original PGAS matrix */
        rbpf_lut_rebuild_transitions(rbpf);
    }
    
    /* 3. Discard shadow - it learned noise */
    rbpf->shadow_active = false;
    rbpf->pgas_buffered = false;
    rbpf->storvik_learning_active = false;
}
```

### The Payoff

| Scenario | Old Behavior | New Behavior |
|----------|--------------|--------------|
| **True Crisis** | Wipe immediately, 3-tick warmup | Shadow warms during ARMED, instant switch |
| **Fat Finger** | Wipe, run on N=2 for 50 ticks, degraded | Shadow runs 3 ticks, discard, restore PGAS |
| **Stability** | False alarms cause 50-tick degradation | False alarms cost nothing |

**Shadow mode makes false alarms free.** The filter never stops using the PGAS matrix until crisis is confirmed.

---

### Physics-Based Exit Detection

A crisis ends when the market's **Temperature** (Intensity) and **Sparks** (Surprise) both return to baseline.

```c
/* Two signals must agree */

/* 1. MACRO: Is the fire out? */
float intensity_threshold = lambda_ema + exit_intensity_margin * lambda_sigma;
bool macro_ok = (current_avg_intensity < intensity_threshold);

/* 2. MICRO: Are the sparks gone? */
bool micro_ok = (current_surprise_sigma < exit_surprise_threshold);  // e.g., 1.5σ

/* 3. TIME HYSTERESIS: Confirm it's not eye of the storm */
if (macro_ok && micro_ok) {
    ticks_below_threshold++;
    if (ticks_below_threshold >= exit_confirmation_ticks) {
        // Peace declared
        trigger_state = HAWKES_TRIG_IDLE;
    }
} else {
    ticks_below_threshold = 0;  // Reset counter
}
```

**Why both signals?**

- **Intensity alone:** Could exit during "eye of the storm" (temporary pause)
- **Surprise alone:** Intensity lags, could miss that background heat remains
- **Both together:** Confirms genuine normalization

---

## Baseline Drift Fix (Frozen Baseline)

### The Problem

Exit detection uses `intensity < baseline`. But during a long crisis, the baseline EMA drifts up to meet the crisis intensity.

**Scenario:** 2000-tick structural crash.

**Without freeze:**
```
Tick    0: λ_ema = 0.05, crisis intensity = 0.50
Tick  500: λ_ema drifts to 0.20
Tick 1000: λ_ema drifts to 0.35
Tick 1500: λ_ema drifts to 0.42
Tick 1800: intensity drops to 0.40
           Exit triggered! (0.40 < 0.42 + margin)
           BUT: 0.40 is still 8x the pre-crisis baseline!
```

**Risk:** False exit because "Normal" drifted up to match crisis.

### The Solution: Freeze Baseline at Crisis Entry

```c
typedef struct {
    /* ... existing Hawkes ... */
    
    /* Frozen baseline for exit detection */
    float frozen_baseline;        /* λ_ema at crisis entry */
    float frozen_sigma;           /* σ_ema at crisis entry */
    bool  baseline_frozen;
    
} HawkesIntegrator;
```

### Freeze on Crisis Confirmation (ARMED → CRISIS_ACTIVE)

```c
case HAWKES_TRIG_ARMED:
    /* ... existing ARMED logic ... */
    
    /* When transitioning ARMED → CRISIS_ACTIVE */
    if (should_confirm_crisis) {
        /* FREEZE the baseline before it gets polluted */
        integ->frozen_baseline = integ->lambda_ema;
        integ->frozen_sigma = integ->lambda_sigma;
        integ->baseline_frozen = true;
        
        integ->trigger_state = HAWKES_TRIG_CRISIS_ACTIVE;
    }
    break;
```

### Exit Detection Uses Frozen Baseline

```c
case HAWKES_TRIG_CRISIS_ACTIVE:
    integ->ticks_in_crisis++;
    
    /* ═══════════════════════════════════════════════════════════════
     * VARIANCE TRACKING: Update live EMA but DON'T use for exit
     * (Keep updating for diagnostics and deep crisis monitoring)
     * ═══════════════════════════════════════════════════════════════*/
    float delta = avg_lambda - integ->lambda_ema;
    integ->lambda_ema += alpha * delta;
    /* ... etc ... */
    
    /* ═══════════════════════════════════════════════════════════════
     * EXIT DETECTION: Use FROZEN baseline, not drifted one
     * ═══════════════════════════════════════════════════════════════*/
    float exit_baseline = integ->baseline_frozen ? 
                          integ->frozen_baseline : 
                          integ->lambda_ema;
    
    float exit_sigma = integ->baseline_frozen ? 
                       integ->frozen_sigma : 
                       integ->lambda_sigma;
    
    float intensity_threshold = exit_baseline + 
                                cfg->exit_intensity_margin * exit_sigma;
    
    bool macro_ok = (integ->current_avg_intensity < intensity_threshold);
    bool micro_ok = (integ->current_surprise_sigma < cfg->exit_surprise_threshold);
    
    /* ... rest of exit logic ... */
    break;
```

### Unfreeze on Exit

```c
case HAWKES_TRIG_RECOVERING:
    /* ... */
    if (confirmed_exit) {
        integ->baseline_frozen = false;
        integ->trigger_state = HAWKES_TRIG_IDLE;
    }
    break;
```

### With Freeze (Correct Behavior)

```
Tick    0: frozen_baseline = 0.05, crisis intensity = 0.50
Tick 1800: intensity drops to 0.40
           No exit (0.40 > 0.05 + margin)
Tick 2200: intensity drops to 0.08
           Exit triggered (0.08 < 0.05 + margin) ✓
```

**Key Insight:** Compare against the **pre-crisis peace**, not the drifted war baseline.

---

## Label Switching Fix (Sort Before Inject)

### The Problem: MCMC Label Ambiguity

PGAS uses arbitrary labels (0, 1, 2) for regimes. MCMC can swap them:
- PGAS thinks: "Regime 0 = High Vol, Regime 2 = Low Vol"
- RBPF thinks: "Regime 0 = Low Vol, Regime 2 = High Vol"

### When It Kills You: Deep Crisis Injection (Tick 500)

**Phase 1: The Drift (Tick 0-499)**
```
RBPF: Tracking crash using Regime 2 (High Vol)
PGAS: Accidentally swaps labels
      Calculates: "Regime 0 is High Vol, Regime 2 is Low Vol"
      Learns: Regime 0 is sticky (Crisis), Regime 2 is jumpy (Calm)
```

**Phase 2: The Collision (Tick 500)**
```
Action: inject_physics_only(rbpf, pgas_matrix)
Upload: RBPF transition rules overwritten with PGAS map

Logic Bomb:
  - RBPF particles sitting in Regime 2 (thinking it's Crisis)
  - NEW matrix says: "Regime 2 is Low Vol. Not sticky. Leave."
```

**Phase 3: The Crash (Tick 501)**
```
Particles: Read new rules, see low persistence in Regime 2
Action: Migrate to Regime 0 (which matrix says is sticky)
Result: Filter loses confidence at the worst moment
        Stickiness applied to wrong index
        Crisis tracking destabilizes
```

### The Solution: Canonical Order by Volatility

**Convention:** Index 0 = Lowest σ, Index K-1 = Highest σ

```
CALM = 0, TREND = 1, CRISIS = 2 (always sorted by σ)
```

### PGAS Must Export Emission Parameters

```c
typedef struct {
    float Pi[K * K];          /* Transition matrix */
    float sigma_vol[K];       /* Emission σ per regime - REQUIRED */
    float mu_vol[K];          /* Emission μ per regime (optional) */
    int64_t tick;
    /* ... */
} PGASChannelBuffer;
```

### Label Permutation Functions

```c
typedef struct {
    int pgas_to_canonical[K];   /* map[pgas_idx] = canonical_idx */
    int canonical_to_pgas[K];   /* inverse */
} LabelPermutation;

LabelPermutation compute_label_permutation(const float *pgas_sigma, int K) {
    LabelPermutation perm;
    
    /* Create index array */
    int indices[K];
    for (int i = 0; i < K; i++) indices[i] = i;
    
    /* Sort indices by sigma (ascending: low vol = 0, high vol = K-1) */
    for (int i = 0; i < K - 1; i++) {
        for (int j = i + 1; j < K; j++) {
            if (pgas_sigma[indices[i]] > pgas_sigma[indices[j]]) {
                int tmp = indices[i];
                indices[i] = indices[j];
                indices[j] = tmp;
            }
        }
    }
    
    /* indices[canonical] = pgas_idx */
    /* So: pgas_to_canonical[pgas_idx] = canonical */
    for (int canonical = 0; canonical < K; canonical++) {
        int pgas_idx = indices[canonical];
        perm.canonical_to_pgas[canonical] = pgas_idx;
        perm.pgas_to_canonical[pgas_idx] = canonical;
    }
    
    return perm;
}

void apply_permutation_to_matrix(const float *raw_Pi, 
                                  const LabelPermutation *perm,
                                  float *sorted_Pi, 
                                  int K) {
    /* 
     * raw_Pi[i][j] = P(pgas_i → pgas_j)
     * sorted_Pi[a][b] = P(canonical_a → canonical_b)
     * 
     * canonical_a corresponds to pgas_i = perm->canonical_to_pgas[a]
     * canonical_b corresponds to pgas_j = perm->canonical_to_pgas[b]
     */
    for (int a = 0; a < K; a++) {
        int i = perm->canonical_to_pgas[a];
        for (int b = 0; b < K; b++) {
            int j = perm->canonical_to_pgas[b];
            sorted_Pi[a * K + b] = raw_Pi[i * K + j];
        }
    }
}
```

### Updated Injection Function (Label-Safe)

```c
void inject_physics_only(RBPF *rbpf, 
                          const float *Pi_pgas_raw, 
                          const float *sigma_pgas,
                          float N_eff,
                          int K) {
    
    /* ═══════════════════════════════════════════════════════════════
     * 1. COMPUTE LABEL PERMUTATION
     *    PGAS labels may not match canonical order (sorted by σ)
     * ═══════════════════════════════════════════════════════════════*/
    LabelPermutation perm = compute_label_permutation(sigma_pgas, K);
    
    /* ═══════════════════════════════════════════════════════════════
     * 2. APPLY PERMUTATION TO TRANSITION MATRIX
     * ═══════════════════════════════════════════════════════════════*/
    float Pi_sorted[K * K];
    apply_permutation_to_matrix(Pi_pgas_raw, &perm, Pi_sorted, K);
    
    /* ═══════════════════════════════════════════════════════════════
     * 3. INJECT SORTED MATRIX (now label-safe)
     * ═══════════════════════════════════════════════════════════════*/
    for (int i = 0; i < K; i++) {
        float row_total = 0.0f;
        
        for (int j = 0; j < K; j++) {
            float new_count = Pi_sorted[i * K + j] * N_eff;
            rbpf->counts_N_ij[i][j] = new_count;
            row_total += new_count;
        }
        rbpf->counts_N_i[i] = row_total;
    }
    
    rbpf_lut_rebuild_transitions(rbpf);
}
```

### RBPF Must Also Maintain Canonical Order

During Storvik learning, emission σ estimates can drift. Periodically verify:

```c
void rbpf_enforce_canonical_order(RBPF *rbpf) {
    /* Check if current emission σ estimates are out of order */
    float sigma[K];
    for (int k = 0; k < K; k++) {
        sigma[k] = rbpf_get_regime_sigma(rbpf, k);
    }
    
    /* If σ[0] > σ[1] or σ[1] > σ[2], we have label drift */
    bool needs_reorder = false;
    for (int k = 0; k < K - 1; k++) {
        if (sigma[k] > sigma[k + 1]) {
            needs_reorder = true;
            break;
        }
    }
    
    if (needs_reorder) {
        /* Compute permutation and apply to:
         * - Particle regime assignments
         * - Transition counts
         * - Emission sufficient statistics
         */
        rbpf_relabel_regimes(rbpf);
    }
}
```

### When to Check

```c
/* In tick loop, periodically verify order (every 100 ticks) */
if (sys->tick % 100 == 0) {
    rbpf_enforce_canonical_order(sys->rbpf);
}

/* ALWAYS check before PGAS injection */
void inject_physics_only(...) {
    /* 1. Sort PGAS output */
    /* 2. Verify RBPF is still canonical */
    rbpf_enforce_canonical_order(rbpf);
    /* 3. Inject */
}
```

### Summary: Both Systems Must Agree on Labels

| System | Responsibility |
|--------|----------------|
| **PGAS** | Export σ with Π, caller sorts before use |
| **RBPF** | Maintain canonical order, relabel if drifted |
| **Injection** | Sort PGAS Π, verify RBPF order, then inject |

---

## Storvik Transition Learning During Crisis

### The Problem: Blind Pilot

If PGAS is vetoed and Storvik isn't learning Π:
- Filter uses stale Π from pre-crisis
- No adaptation to crisis dynamics
- Jittery behavior as filter misreads regime persistence

### The Solution: Emergency Autopilot

Storvik learns Π **only during crisis** with aggressive forgetting.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           PEACE TIME                                    │
│                                                                         │
│   PGAS ───[Π]───→ RBPF                                                 │
│                    │                                                    │
│   Storvik Π: OFF   └───→ Track μ/σ only                                │
└─────────────────────────────────────────────────────────────────────────┘
                                │
                                │ Hawkes triggers
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          CRISIS MODE                                    │
│                                                                         │
│   PGAS: VETOED (stale Π)                                               │
│                                                                         │
│   Storvik ───[Π]───→ RBPF                                              │
│      │                  │                                               │
│      │                  └───→ Track μ/σ (emission stats)               │
│      │                                                                  │
│      └───→ Learn Π locally with λ=0.6                                  │
│            "Is this crash sticky or volatile?"                          │
└─────────────────────────────────────────────────────────────────────────┘
                                │
                                │ Hawkes detects exit (Physics)
                                │ OR crisis > T ticks (PGAS adapted)
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           RECOVERY                                      │
│                                                                         │
│   PGAS ───[Π]───→ RBPF    (fresh crisis-aware Π if deep crisis)       │
│                                                                         │
│   Storvik Π: OFF                                                        │
│   Discard crisis counts, restore to PGAS-driven                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Why NOT Background Storvik?

**The Baggage Problem:**

If Storvik runs during peace, it accumulates massive counts supporting peace dynamics:
```
After 1 hour of peace: N_calm→calm ≈ 10,000
```

When crisis hits, switching to this Storvik gives you the **same rigidity as PGAS**—10,000 votes saying "Don't Switch."

### The Fresh Start Protocol

**Reset (Melt) Storvik counts at crisis entry:**

```c
void rbpf_enter_crisis_mode(RBPF *rbpf) {
    /* Initialize with WEAK prior - allows instant adaptation */
    const float weak_prior = 2.0f;
    
    for (int i = 0; i < K; i++) {
        rbpf->trans_row_sums[i] = 0.0f;
        for (int j = 0; j < K; j++) {
            rbpf->trans_counts[i][j] = weak_prior;
            rbpf->trans_row_sums[i] += weak_prior;
        }
    }
    
    rbpf->storvik_learning_active = true;
    rbpf_lut_rebuild_transitions(rbpf);
}
```

### Flash Learning Math

With weak prior (N = 2.0) and aggressive decay (λ = 0.6):

**Tick 0 (Crisis Entry):**
```
N_stay = 2.0, N_switch = 2.0
P(switch) = 50%
```

**Tick 1 (First crisis observation):**
```
After decay:     N_stay = 1.2,  N_switch = 1.2
After APF kick:  N_switch += 1.0 → 2.2
P(switch) = 2.2 / 3.4 = 65%
```

**Tick 2:**
```
After decay:     N_stay = 0.72,  N_switch = 1.32
After APF kick:  N_switch += 1.0 → 2.32
P(switch) = 2.32 / 3.04 = 76%
```

**Tick 3:**
```
After decay:     N_stay = 0.43,  N_switch = 1.39
After APF kick:  N_switch += 1.0 → 2.39
P(switch) = 2.39 / 2.82 = 85%
```

**Result:** Transition matrix adapts in **2-3 ticks**, not 50.

### Why So Fast?

| Learning Target | Speed | Reason |
|-----------------|-------|--------|
| **Π (transitions)** | 2-3 ticks | Discrete choice, APF kick dominates |
| **μ, σ (emissions)** | 30-50 ticks | Continuous estimation, needs samples |

The APF likelihood ratio does the heavy lifting:
```
L(obs | Crisis, σ=0.22) >> L(obs | Calm, σ=0.007)
```

Particle weights collapse onto crisis particles instantly. Storvik just counts where the weight went.

### Adaptive Crisis Behavior

The filter **discovers** what kind of crisis it's in:

**Flashy Crisis (GME squeeze):**
- Rapid regime switching observed
- Storvik learns: Low persistence, high transition probability
- Filter stays nimble

**Structural Crisis (2008):**
- Persistent crisis state observed
- Storvik learns: P(Stay|Crisis) → 0.95
- Filter locks in, stops looking for false exits

---

## The λ Ladder

| State | λ | Applies To | Effect |
|-------|---|------------|--------|
| **PEACE** | N/A | Nothing | PGAS owns Π, no local counting |
| **ARMED** | 0.95 | (transition counts if enabled) | Something brewing |
| **CRISIS (< T)** | 0.60 | Transition counts N_ij | Forget pre-crisis fast |
| **CRISIS (≥ T)** | 0.95 | Transition counts N_ij | Stabilize, Storvik has learned |
| **RECOVERING** | 0.90 | (transition counts if still active) | Cautious return |

---

## PGAS Veto Decision Function

```c
typedef struct {
    bool   should_veto;         /* Block PGAS injection? */
    bool   is_deep_crisis;      /* Crisis > T ticks (PGAS adapted) */
    bool   shadow_mode;         /* ARMED: run shadow learning */
    bool   storvik_learning;    /* Should Storvik learn Π? (primary) */
    float  recommended_lambda;
} HawkesPGASDecision;

HawkesPGASDecision hawkes_get_pgas_decision(const HawkesIntegrator *integ, 
                                             int pgas_window_T) {
    HawkesPGASDecision d = {0};
    
    if (!integ) {
        d.recommended_lambda = 0.995f;
        return d;
    }
    
    switch (integ->trigger_state) {
        
        case HAWKES_TRIG_IDLE:
            /* Peace - business as usual */
            d.should_veto = false;
            d.is_deep_crisis = false;
            d.shadow_mode = false;
            d.storvik_learning = false;
            d.recommended_lambda = 0.995f;
            break;
            
        case HAWKES_TRIG_ARMED:
            /* SHADOW MODE: Veto PGAS, run shadow Storvik, but don't switch yet */
            d.should_veto = true;
            d.is_deep_crisis = false;
            d.shadow_mode = true;           /* Shadow learning active */
            d.storvik_learning = false;     /* Primary still uses PGAS */
            d.recommended_lambda = 0.60f;   /* Shadow uses aggressive decay */
            break;
            
        case HAWKES_TRIG_CRISIS_ACTIVE:
            if (integ->ticks_in_crisis >= pgas_window_T) {
                /* DEEP CRISIS: PGAS has full window of crisis data.
                 * Its Π is NOW ACCURATE. INJECT IT.
                 * But also keep Storvik learning for local texture. */
                d.should_veto = false;
                d.is_deep_crisis = true;
                d.shadow_mode = false;
                d.storvik_learning = true;
                d.recommended_lambda = 0.95f;
            } else {
                /* EARLY CRISIS: PGAS Π is stale.
                 * VETO. Storvik takes over. */
                d.should_veto = true;
                d.is_deep_crisis = false;
                d.shadow_mode = false;
                d.storvik_learning = true;
                d.recommended_lambda = 0.60f;
            }
            break;
            
        case HAWKES_TRIG_RECOVERING:
            /* Transitional - be cautious, stop learning */
            d.should_veto = true;
            d.is_deep_crisis = false;
            d.shadow_mode = false;
            d.storvik_learning = false;
            d.recommended_lambda = 0.90f;
            break;
    }
    
    return d;
}
```

---

## Tick Update Logic (v2.0)

```c
void tick_update(System* sys, float obs) {
    
    /* ═══════════════════════════════════════════════════════════════
     * 1. UPDATE HAWKES (The Guardian)
     * ═══════════════════════════════════════════════════════════════*/
    HawkesIntegratorResult hawkes_result = hawkes_integrator_update(
        &sys->hawkes, sys->tick, obs);
    
    HawkesState prev_state = sys->prev_hawkes_state;
    HawkesState curr_state = hawkes_result.state;
    
    /* ═══════════════════════════════════════════════════════════════
     * 2. GET PGAS DECISION
     * ═══════════════════════════════════════════════════════════════*/
    HawkesPGASDecision decision = hawkes_get_pgas_decision(
        &sys->hawkes, PGAS_WINDOW_SIZE);
    
    /* ═══════════════════════════════════════════════════════════════
     * 3. HANDLE STATE TRANSITIONS
     * ═══════════════════════════════════════════════════════════════*/
    
    /* IDLE → ARMED: Enter shadow mode (buffer PGAS, start shadow learning) */
    if (prev_state == HAWKES_TRIG_IDLE && curr_state == HAWKES_TRIG_ARMED) {
        rbpf_enter_shadow_mode(sys->rbpf);
    }
    
    /* ARMED → CRISIS_ACTIVE: Confirm crisis, promote shadow to primary */
    if (prev_state == HAWKES_TRIG_ARMED && curr_state == HAWKES_TRIG_CRISIS_ACTIVE) {
        rbpf_confirm_crisis(sys->rbpf);
    }
    
    /* ARMED → IDLE: False alarm, restore PGAS buffer, discard shadow */
    if (prev_state == HAWKES_TRIG_ARMED && curr_state == HAWKES_TRIG_IDLE) {
        rbpf_abort_crisis(sys->rbpf);
    }
    
    /* CRISIS_ACTIVE → RECOVERING or IDLE: Exit crisis */
    if (prev_state == HAWKES_TRIG_CRISIS_ACTIVE && 
        (curr_state == HAWKES_TRIG_RECOVERING || curr_state == HAWKES_TRIG_IDLE)) {
        rbpf_exit_crisis_mode(sys->rbpf);
    }
    
    /* ═══════════════════════════════════════════════════════════════
     * 4. UPDATE SHADOW (if in shadow mode)
     * ═══════════════════════════════════════════════════════════════*/
    if (decision.shadow_mode) {
        rbpf_shadow_update(sys->rbpf, decision.recommended_lambda);
    }
    
    /* ═══════════════════════════════════════════════════════════════
     * 5. PGAS INJECTION (if not vetoed) - LABEL-SAFE
     * ═══════════════════════════════════════════════════════════════*/
    if (atomic_load(&sys->mailbox.updated) && !decision.should_veto) {
        /* Enforce canonical order before injection */
        rbpf_enforce_canonical_order(sys->rbpf);
        
        /* Inject with label permutation (sorts by σ) */
        inject_physics_only(sys->rbpf, 
                            sys->mailbox.Pi, 
                            sys->mailbox.sigma_vol,  /* PGAS must export σ */
                            1000.0f,
                            K);
        
        atomic_store(&sys->mailbox.updated, false);
        
        if (decision.is_deep_crisis) {
            /* Log: PGAS caught up to crisis */
        }
    }
    
    /* ═══════════════════════════════════════════════════════════════
     * 6. PERIODIC CANONICAL ORDER CHECK (every 100 ticks)
     * ═══════════════════════════════════════════════════════════════*/
    if (sys->tick % 100 == 0) {
        rbpf_enforce_canonical_order(sys->rbpf);
    }
    
    /* ═══════════════════════════════════════════════════════════════
     * 7. RUN FILTER
     * ═══════════════════════════════════════════════════════════════*/
    rbpf_step(sys->rbpf, obs, decision.storvik_learning, decision.recommended_lambda);
    
    sys->prev_hawkes_state = curr_state;
}
```

---

## Storvik RBPF Extensions

### Data Structures

```c
typedef struct {
    /* ... existing RBPF ... */
    
    /* Storvik transition learning (only active during crisis) */
    float trans_counts[K][K];      /* N_ij pseudo-counts */
    float trans_row_sums[K];       /* N_i for normalization */
    bool  storvik_learning_active;
    
} RBPF;
```

### Crisis Mode Entry

```c
void rbpf_enter_crisis_mode(RBPF *rbpf) {
    /* Melt to weak uniform prior - allows instant adaptation */
    const float weak_prior = 2.0f;
    
    for (int i = 0; i < K; i++) {
        rbpf->trans_row_sums[i] = 0.0f;
        for (int j = 0; j < K; j++) {
            rbpf->trans_counts[i][j] = weak_prior;
            rbpf->trans_row_sums[i] += weak_prior;
        }
    }
    
    rbpf->storvik_learning_active = true;
    rbpf_lut_rebuild_transitions(rbpf);
}
```

### Crisis Mode Exit

```c
void rbpf_exit_crisis_mode(RBPF *rbpf) {
    /* Stop local learning, PGAS will inject on next update */
    rbpf->storvik_learning_active = false;
    
    /* Zero out counts so next PGAS inject is clean */
    memset(rbpf->trans_counts, 0, sizeof(rbpf->trans_counts));
    memset(rbpf->trans_row_sums, 0, sizeof(rbpf->trans_row_sums));
}
```

### Transition Update (During Crisis)

```c
void rbpf_storvik_update_transitions(RBPF *rbpf, float lambda) {
    if (!rbpf->storvik_learning_active) return;
    
    /* 1. Decay existing counts */
    for (int i = 0; i < K; i++) {
        rbpf->trans_row_sums[i] = 0.0f;
        for (int j = 0; j < K; j++) {
            rbpf->trans_counts[i][j] *= lambda;
            rbpf->trans_row_sums[i] += rbpf->trans_counts[i][j];
        }
    }
    
    /* 2. Count transitions from particle cloud */
    for (int p = 0; p < N_PARTICLES; p++) {
        int from = rbpf->particles[p].prev_regime;
        int to   = rbpf->particles[p].regime;
        float w  = rbpf->particles[p].weight;
        
        rbpf->trans_counts[from][to] += w;
        rbpf->trans_row_sums[from] += w;
    }
    
    /* 3. Rebuild LUT for next resampling */
    rbpf_lut_rebuild_transitions(rbpf);
}
```

### Step Function

```c
void rbpf_step(RBPF *rbpf, float obs, bool crisis_learning, float lambda) {
    
    /* ... standard APF steps: propagate, weight, resample ... */
    
    /* Storvik: always update emission stats (μ, σ) */
    rbpf_storvik_update_emissions(rbpf);
    
    /* Storvik: update transitions ONLY during crisis */
    if (crisis_learning) {
        rbpf_storvik_update_transitions(rbpf, lambda);
    }
}
```

---

## Hawkes Configuration (v2.0)

### New Fields

```c
typedef struct {
    /* ... existing Hawkes config ... */
    
    /* Physics-based exit detection */
    bool  use_adaptive_exit;
    float exit_intensity_margin;    /* σ above baseline to consider "elevated" (0.5) */
    float exit_surprise_threshold;  /* Max surprise to consider "boring" (1.5σ) */
    int   exit_confirmation_ticks;  /* Consecutive calm ticks required (20) */
    
    /* Crisis tracking */
    int   min_crisis_ticks;         /* Minimum crisis duration before exit allowed (10) */
    
} HawkesIntegratorConfig;
```

### New State Fields

```c
typedef struct {
    /* ... existing Hawkes state ... */
    
    /* Crisis tracking */
    int   ticks_in_crisis;          /* Duration in CRISIS_ACTIVE */
    int   ticks_below_threshold;    /* Consecutive calm ticks */
    int   crisis_exits;             /* Lifetime count */
    
} HawkesIntegrator;
```

---

## System Lifecycle Summary

| Phase | Indicator | Decision | PGAS | Storvik Π | λ |
|-------|-----------|----------|------|-----------|---|
| **Normal** | Intensity ≈ Baseline | Peace | INJECT | OFF | N/A |
| **Crash Start** | Intensity Spikes | War Declared | VETO | MELT + ON | 0.60 |
| **Crash Pause** | Returns small, Intensity High | Eye of Storm | VETO | ON | 0.60 |
| **Deep Crash** | Duration > T ticks | New Normal | INJECT | ON | 0.95 |
| **Recovery** | Intensity < Baseline | Peace Declared | VETO (briefly) | OFF | 0.90 |

---

## What Happens Inside Crisis

### Tick-by-Tick Adaptation

```
Tick  0 (Entry):    Storvik reset to N=2.0 uniform
Tick  1 (Kick):     P(Crisis→Crisis) → 65%  (learning!)
Tick  3:            P(Crisis→Crisis) → 85%  (locked in)
Tick 50:            P(Crisis→Crisis) → 95%  (converged)
...
Tick 500:           PGAS publishes crisis-aware Π
                    Deep crisis mode - both sources agree
```

### Flash Crash vs Structural Crash

**Flash Crash (60 ticks):**
```
Tick  0: Enter crisis, Storvik reset
Tick  3: Storvik learns switching pattern
Tick 40: Intensity drops
Tick 60: Exit criteria met, return to IDLE
         PGAS never caught up - didn't need to
```

**Structural Crash (2000 ticks):**
```
Tick    0: Enter crisis, Storvik reset
Tick    3: Storvik learns persistence
Tick  500: PGAS publishes crisis Π - INJECT
Tick 1000: Both PGAS and Storvik agree on dynamics
Tick 1800: Intensity drops
Tick 1850: Exit criteria met, return to IDLE
```

---

## Performance Summary

| Config | Time/Window | Notes |
|--------|-------------|-------|
| N=128, 3 sweeps, 32 threads | 21.5ms | Baseline |
| N=64, 2 sweeps, 32 threads | 12.8ms | -40% |
| N=64, 2 sweeps, 8 threads | **7.2ms** | **Optimal** |
| Warm window (best case) | 2.96ms | memmove optimization |

**Budget at 100Hz, S=50:**
```
Wall time available: 500ms
Processing time:     7ms
Headroom:            493ms (98.6% idle)
```

---

## Files Reference

| File | Location | Purpose |
|------|----------|---------|
| `pgas_sliding.h` | PGAS/ | Sliding window wrapper |
| `pgas_sliding.c` | PGAS/ | Ring buffer, warm start, differential update |
| `pgas_compat.h` | PGAS/ | Cross-platform atomics (MSVC/GCC) |
| `pgas_mkl.h` | PGAS/ | Core CSMC + Dirichlet |
| `hawkes_integrator.h` | Hawkes/ | Crisis detection header |
| `hawkes_integrator.c` | Hawkes/ | Hawkes process + state machine |
| `test_hawkes_timing.c` | test-bench/ | Hawkes validation test |

---

## Summary

| What | Value |
|------|-------|
| PGAS Window (T) | 500 ticks |
| Slide Step (S) | 50 ticks |
| Particles (N) | 64 |
| Sweeps | 2 |
| Threads | 8 P-cores |
| Processing | ~7ms |
| Update Frequency | 500ms at 100Hz |
| **Crisis Entry** | Hawkes onset detection |
| **Crisis Exit** | Physics-based (frozen baseline + surprise) |
| **Shadow Mode** | ARMED state, 3-5 tick probation |
| **Storvik Learning** | Active during CRISIS_ACTIVE only |
| **Weak Prior** | N = 2.0 |
| **Crisis λ** | 0.60 (early), 0.95 (deep) |
| **Label Safety** | Sort by σ before injection |
| **Canonical Order** | Regime 0 = Low σ, Regime K-1 = High σ |

**Mantras:**

> PGAS writes the rulebook (Π). Storvik keeps the score (μ, σ). Don't wipe the scoreboard.

> Crisis ends when Physics says so, not when the clock runs out.

> Storvik is the Emergency Autopilot—give it amnesia so it can learn the current crash, not remember the past peace.

> Shadow mode makes false alarms free. Never wipe the PGAS matrix until crisis is confirmed.

> Compare against the pre-crisis peace, not the drifted war baseline. Freeze λ_ema at crisis entry.

> Labels are arbitrary to MCMC. Sort by σ before crossing the PGAS→RBPF bridge.

---

## Silent Killers: The Three Fixes

These are bugs that appear to work until the exact moment they fail catastrophically.

| Problem | When It Kills | Symptom | Fix |
|---------|---------------|---------|-----|
| **False Alarm Amnesia** | IDLE→ARMED on fat finger | 50-tick degradation, erratic switching | Shadow Mode: buffer PGAS, learn in parallel, commit only if confirmed |
| **Baseline Drift** | 2000-tick structural crash | False exit at tick 1800 | Freeze λ_ema at crisis entry, compare against pre-crisis baseline |
| **Label Switching** | Deep Crisis injection at tick 500 | Filter destabilizes at worst moment | Sort PGAS Π by σ before injection, maintain canonical order |

### Why They're Silent

- **False Alarm:** System "works" but with degraded performance. Hard to detect in testing.
- **Baseline Drift:** Only manifests in very long crises. Short tests pass.
- **Label Switching:** MCMC labels are random. Works 80% of the time by luck.

### Why They're Killers

- **False Alarm:** Every false positive costs 50 ticks of instability.
- **Baseline Drift:** Exit during ongoing crash = catastrophic P&L.
- **Label Switching:** Matrix applied to wrong regimes = inverted behavior.

---

## Implementation Checklist

- [ ] **Hawkes Exit Detection**
  - [ ] Add `ticks_in_crisis` tracking
  - [ ] Add `ticks_below_threshold` counter
  - [ ] Implement physics-based exit (intensity + surprise)
  - [ ] Add `HAWKES_TRIG_RECOVERING` state
  - [ ] Add `frozen_baseline` and `frozen_sigma` fields
  - [ ] Freeze baseline on crisis confirmation
  - [ ] Use frozen baseline for exit detection
  - [ ] Test exit timing accuracy
  - [ ] Test no false exit during 2000-tick structural crash

- [ ] **Shadow Mode (False Alarm Protection)**
  - [ ] Add `shadow_counts[K][K]` to RBPF
  - [ ] Add `pgas_buffer_counts[K][K]` to RBPF
  - [ ] Implement `rbpf_enter_shadow_mode()` - buffer PGAS, init shadow
  - [ ] Implement `rbpf_shadow_update()` - learn in parallel
  - [ ] Implement `rbpf_confirm_crisis()` - promote shadow to primary
  - [ ] Implement `rbpf_abort_crisis()` - restore PGAS buffer
  - [ ] Add `shadow_mode` flag to `HawkesPGASDecision`
  - [ ] Test false alarm recovery (restore PGAS in 3 ticks)
  - [ ] Test true crisis transition (shadow already warm)

- [ ] **Storvik Transition Learning**
  - [ ] Add `trans_counts[K][K]` to RBPF
  - [ ] Implement `rbpf_enter_crisis_mode()` with weak prior
  - [ ] Implement `rbpf_exit_crisis_mode()`
  - [ ] Implement `rbpf_storvik_update_transitions()`
  - [ ] Test 2-3 tick learning convergence

- [ ] **Label Switching Fix**
  - [ ] Add `sigma_vol[K]` to `PGASChannelBuffer`
  - [ ] PGAS exports σ with every Π publication
  - [ ] Implement `compute_label_permutation()`
  - [ ] Implement `apply_permutation_to_matrix()`
  - [ ] Update `inject_physics_only()` to sort before inject
  - [ ] Implement `rbpf_enforce_canonical_order()`
  - [ ] Implement `rbpf_relabel_regimes()`
  - [ ] Add periodic canonical order check (every 100 ticks)
  - [ ] Test deep crisis injection with swapped PGAS labels

- [ ] **Integration**
  - [ ] Implement `hawkes_get_pgas_decision()` with shadow_mode
  - [ ] Wire state transitions in tick loop (IDLE↔ARMED↔CRISIS↔RECOVERING)
  - [ ] Connect λ ladder to RBPF
  - [ ] Track `prev_hawkes_state` for transition detection
  - [ ] Test full system on synthetic data

- [ ] **Validation**
  - [ ] Flash crash detection and exit
  - [ ] Structural crash persistence learning
  - [ ] Deep crisis PGAS injection (with label sort)
  - [ ] Recovery without false re-entry
  - [ ] Fat finger false alarm (restore in 3 ticks)
  - [ ] Baseline drift resistance (2000-tick test)
  - [ ] Label switching resistance (force PGAS swap, verify no crash)
