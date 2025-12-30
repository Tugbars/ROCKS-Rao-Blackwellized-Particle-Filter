# PGAS Oracle + RBPF Integration Plan

## Overview

Hybrid architecture where **PGAS provides structure (Π)** and **Storvik tracks state (μ, σ)**.

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
│              [Crisis: Skip] [Calm: Inject Π]                   │
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

## Timing Hierarchy

Each component has a specific reaction time and role:

| Component | Latency | Role | Action |
|-----------|---------|------|--------|
| **Hawkes** | <10ms | Flash detection | Veto injection, drop λ |
| **RBPF** | 10ms | State tracking | Track μ/σ tick-by-tick |
| **PGAS** | 500ms | Historian | Analyze past, publish Π |

### Key Principle

> **PGAS should never be the first responder. It's the historian.**

- **Onset (0-100ms):** Hawkes sees volume spike instantly, vetoes PGAS, drops λ
- **Duration (100ms+):** RBPF rides the wave with low λ
- **Aftermath (500ms+):** PGAS analyzes crash data, updates Π for "new normal"

---

## Tick Update Logic

```c
void tick_update(System* sys, float obs) {
    
    /* ═══════════════════════════════════════════════════════════════
     * 1. UPDATE HAWKES (The Veto Guardian)
     * ═══════════════════════════════════════════════════════════════*/
    float intensity = hawkes_update(&sys->hawkes, obs);
    bool is_crisis = (intensity > CRISIS_THRESHOLD);
    
    /* ═══════════════════════════════════════════════════════════════
     * 2. CHECK MAILBOX (PGAS Oracle Output)
     * ═══════════════════════════════════════════════════════════════*/
    if (atomic_load(&sys->mailbox.updated)) {
        
        /* GATE: Only inject if NOT in crisis */
        if (!is_crisis) {
            
            /* HOT SWAP: Inject new physics */
            inject_physics_only(sys->rbpf, 
                               sys->mailbox.Pi, 
                               1000.0f);
            
            /* ACK: Mark as consumed */
            atomic_store(&sys->mailbox.updated, false);
        }
        /* If is_crisis: ignore update, keep last known good Π */
    }
    
    /* ═══════════════════════════════════════════════════════════════
     * 3. ADAPTIVE FORGETTING
     * ═══════════════════════════════════════════════════════════════*/
    sys->rbpf->lambda = is_crisis ? 0.60f : 0.995f;
    
    /* ═══════════════════════════════════════════════════════════════
     * 4. RUN FILTER
     * ═══════════════════════════════════════════════════════════════*/
    rbpf_step(sys->rbpf, obs);
}
```

---

## Why We Removed the KL Gate

Original design had: `if (KL(rbpf_pi, pgas_pi) > threshold) → inject`

**Removed because:**

1. **Redundancy:** At S=50, memcpy(Π) costs ~100ns. No performance penalty.
2. **False Negatives:** Small but critical updates (1% crash prob shift) might be blocked.
3. **Simplicity:** Hawkes veto is the only gate that matters. Fewer branches = fewer bugs.

---

## What Happens Inside 50 Ticks?

Since Storvik can't "learn" Π in 50 ticks, what is it doing?

**Micro-Adaptation:**

```
Tick  0 (Reset):  Π[0→1] = 0.05  (PGAS Baseline)
Tick 10 (Jump):   Π[0→1] → 0.051 (Storvik increments count)
Tick 20 (Jump):   Π[0→1] → 0.052 (Another increment)
Tick 50 (Reset):  Π[0→1] = 0.05  (PGAS snaps it back)
```

**Why this is good:**
- Filter has "local opinion" that captures volatility clustering
- Hard reset prevents drift into hallucination
- Best of both worlds: local reactivity + global stability

---

## S=50 is the Goldilocks Zone

| S | Wall Time | Micro-Adaptation | Risk |
|---|-----------|------------------|------|
| 10 | 100ms | Almost none | Over-resetting, just fixed-param filter |
| **50** | **500ms** | **Balanced** | **Sweet spot** |
| 500 | 5000ms | Too much | Drift, overfitting noise |

---

## Files Reference

| File | Location | Purpose |
|------|----------|---------|
| `pgas_sliding.h` | PGAS/ | Sliding window wrapper |
| `pgas_sliding.c` | PGAS/ | Ring buffer, warm start, differential update |
| `pgas_compat.h` | PGAS/ | Cross-platform atomics (MSVC/GCC) |
| `pgas_mkl.h` | PGAS/ | Core CSMC + Dirichlet |
| `test_pgas_sliding.c` | PGAS/test-bench/ | Validation test |

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
| Reset | Π counts only |
| Preserve | μ/σ emission stats |
| Crisis Gate | Hawkes veto |

**Mantra:**
> PGAS writes the rulebook (Π). Storvik keeps the score (μ, σ). Don't wipe the scoreboard.
