# Crisis Mode + Transition Learning Design

## Consensus Document v1.0

---

## Executive Summary

This document describes the integration between **CrisisDetector** and **RBPF** for handling regime crises. The core insight is separation of concerns:

| Component | Learns | During Crisis |
|-----------|--------|---------------|
| **Storvik** | μ, σ (emissions) | Fast λ override (0.95) |
| **Trans Learning** | Π (transitions) | Shadow → Blend → Fast learning |
| **PGAS** | Π, μ, σ (everything) | VETOED until caught up (500+ ticks) |
| **CrisisDetector** | Nothing - just detects | Owns the state machine |

---

## Current State (What Exists)

| Module | Status | What It Does |
|--------|--------|--------------|
| **Storvik** (`rbpf_param_learn.c`) | ✓ Exists | Per-particle μ, σ learning |
| **Trans Learning** (`RBPF_Extended`) | ✓ Exists | Global `trans_counts[K][K]`, slow λ=0.995 |
| **PGAS Injection** | ✓ Exists | `rbpf_ksc_update_transition_matrix_threadsafe()` |
| **Adaptive Forgetting** | ✓ Exists | Controls Storvik λ based on surprise |
| **Circuit Breaker** | ✓ Exists | P² tail detection, emergency λ |
| **CrisisDetector** | ✓ Exists | SR + Hawkes + State Machine (in Crisis_Detection/) |
| **Shadow Mode** | ✗ NEW | Parallel transition learning during ALERT |
| **pgas_buffer** | ✗ NEW | Snapshot for rollback on false alarm |
| **Crisis λ Override** | ✗ NEW | Force fast λ during confirmed crisis |
| **CrisisDetector→RBPF Wiring** | ✗ NEW | API for crisis mode control |

---

## The Cold Start Problem

### Why This Architecture Exists

PGAS learns from what it observes. In a market that's mostly calm:

```
Historical:  [calm calm calm calm ... 10000 ticks of calm]
PGAS knows:  Π[0][0] = 0.999  (excellent)
             Π[3][3] = prior  (never seen crisis!)
             μ₃, σ₃ = prior   (never seen crisis!)
```

When crisis hits for the first time:
- PGAS has **no data** on crisis dynamics
- Only prior knowledge (theoretical, not learned)
- Online learning must fill the gap

### Learning Timeline

```
Day 1:  [calm calm calm calm calm calm ...]
        └─── PGAS learns Π[0][0] only, crisis params = prior

Day 2:  [calm calm CRISIS! CRISIS CRISIS calm ...]
        └─── First crisis: flying on priors + real-time learning
        └─── After crisis: NOW have real estimates

Day 3:  [calm calm CRISIS! CRISIS ...]
        └─── Second crisis: have learned priors from Day 2
        └─── Fast λ adapts if this crisis differs
```

---

## Three-Buffer Design for Transitions

### The Buffers

| Buffer | Purpose | When Updated |
|--------|---------|--------------|
| `trans_counts[K][K]` | **Active** - used for LUT sampling | Always |
| `shadow_counts[K][K]` | **Parallel learning** during ALERT | ALERT phase only |
| `pgas_buffer[K][K]` | **Snapshot** for rollback | Saved on ALERT entry |

### State Machine Flow

```
IDLE ──[Hawkes/SR rising]──→ ALERT ──[SR confirms]──→ CRISIS ──[SR_down]──→ RECOVERING
  ▲                            │                         │                      │
  │                            │ [SR rejects]            │ [SR_up re-triggers]  │
  │                            ▼                         ▼                      │
  │                          IDLE                     CRISIS                    │
  │                      (restore pgas_buffer)                                  │
  │                                                                             │
  └─────────────────────────────[exit confirmed]────────────────────────────────┘
```

### Buffer Operations Per Phase

```
═══════════════════════════════════════════════════════════════════════════════
                              IDLE (Normal Operation)
═══════════════════════════════════════════════════════════════════════════════
trans_counts:    Learning (slow λ = 0.995)
shadow_counts:   Unused
pgas_buffer:     Unused
PGAS:            Injecting every 50 ticks
Storvik:         Learning μ, σ (adaptive λ from surprise)


═══════════════════════════════════════════════════════════════════════════════
                              ALERT (Shadow Mode)
═══════════════════════════════════════════════════════════════════════════════
On Entry:
  1. Copy trans_counts → pgas_buffer (snapshot for rollback)
  2. Copy trans_counts → shadow_counts (start parallel learning)

During ALERT:
  trans_counts:    FROZEN (don't corrupt with uncertain data)
  shadow_counts:   Learning (fast λ = 0.95)
  pgas_buffer:     Frozen (backup)
  PGAS:            VETOED (don't inject)
  Storvik:         Learning μ, σ (still adaptive λ - not confirmed yet)


═══════════════════════════════════════════════════════════════════════════════
                              CRISIS (Confirmed)
═══════════════════════════════════════════════════════════════════════════════
On Entry (SR confirms):
  1. Blend: trans_counts = (1-β) × pgas_buffer + β × shadow_counts
     where β = f(SR strength) ∈ [0.05, 0.95]
  2. Clear shadow_counts (job done)

During CRISIS:
  trans_counts:    Learning (fast λ = 0.95)
  shadow_counts:   Cleared (not needed - we're committed)
  pgas_buffer:     Keep (PGAS may catch up)
  PGAS:            VETOED until ticks_in_crisis > 500
  Storvik:         Learning μ, σ (OVERRIDE λ = 0.95)

After PGAS Catches Up (ticks > 500):
  PGAS:            Can inject again (has seen crisis data)


═══════════════════════════════════════════════════════════════════════════════
                              FALSE ALARM (ALERT → IDLE)
═══════════════════════════════════════════════════════════════════════════════
On SR Rejection:
  1. Restore: trans_counts = pgas_buffer
  2. Clear shadow_counts
  3. Clear pgas_buffer

Result: trans_counts unchanged from before ALERT


═══════════════════════════════════════════════════════════════════════════════
                              RECOVERING (Exit Check)
═══════════════════════════════════════════════════════════════════════════════
trans_counts:    Learning (slow down λ → 0.995)
shadow_counts:   Unused
pgas_buffer:     Cleared
PGAS:            Can inject
Storvik:         Back to adaptive λ

If SR_up re-triggers → back to CRISIS
If exit confirmed → IDLE
```

---

## Convex Blending (ALERT → CRISIS)

### The Problem

With only 64 particles and ~10-50 ticks of ALERT phase, shadow might "hallucinate" structure from noise. Direct copy is risky.

### The Solution

Blend based on SR confidence:

```c
float excess = log_sr - log_H;
float beta = 0.5f + 0.45f * tanhf(excess);  // Range [0.05, 0.95]

// Blend: Π_new = (1-β) × Π_PGAS + β × Π_shadow
for (int i = 0; i < K; i++) {
    for (int j = 0; j < K; j++) {
        trans_counts[i][j] = 
            (1.0f - beta) * pgas_buffer[i][j] +
            beta * shadow_counts[i][j];
    }
}
```

| SR Strength | β | Result |
|-------------|---|--------|
| Barely crosses (log_sr ≈ log_H) | ~0.5 | 50% PGAS, 50% shadow - conservative |
| Strong (log_sr = log_H + 2) | ~0.85 | 15% PGAS, 85% shadow - confident |
| Overwhelming (log_sr = log_H + 4) | ~0.95 | 5% PGAS, 95% shadow - very confident |

### Why Blend Works

- **PGAS has hindsight:** 500-tick backward sampling, structural knowledge
- **Shadow is reactive:** Current observations, may be noisy
- **Confidence-weighted:** Uncertain? Trust PGAS more. Certain? Trust shadow more.

---

## Storvik: Emissions Only

### What Storvik Learns

| Parameter | Type | Storvik? |
|-----------|------|----------|
| μ (regime means) | Emission | ✓ Yes, per-particle |
| σ (regime vols) | Emission | ✓ Yes, per-particle |
| Π (transitions) | Structure | ✗ No - Trans Learning handles this |

### Storvik During Crisis

Storvik **never stops learning**. The question is how fast:

| Phase | Storvik λ | N_eff | Meaning |
|-------|-----------|-------|---------|
| IDLE | Adaptive (~0.998) | ~500 | Remember last 500 ticks |
| ALERT | Adaptive (~0.998) | ~500 | Not confirmed yet, stay cautious |
| CRISIS | **OVERRIDE (0.95)** | ~20 | Recent observations dominate |
| RECOVERING | Back to adaptive | ~500 | Slowing down |

### Why Override During Crisis?

CrisisDetector knows we're in crisis **before** RBPF's particle distribution shifts:

```
CrisisDetector:  IDLE ──→ ALERT ──→ CRISIS
                          │
                          ▼ SR confirms
RBPF particles:  [R0 R0 R0 R0] ─────────→ [R3 R3 R3 R3]
                               (takes time to shift)
```

Waiting for `dominant_regime == R3` wastes adaptation time. Override forces immediate fast learning.

---

## Integration with Existing Adaptive Forgetting

### Current Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    ADAPTIVE FORGETTING                          │
│  Inputs: surprise, outlier_fraction, regime, circuit_breaker   │
│  Output: λ_current → pushed to Storvik                         │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │  STORVIK (μ, σ)        │  ← Uses adaptive λ
              └────────────────────────┘
```

### With Crisis Override

```
┌─────────────────────────────────────────────────────────────────┐
│                    ADAPTIVE FORGETTING                          │
│  Inputs: surprise, outlier_fraction, regime, circuit_breaker   │
│  Output: λ_adaptive                                            │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │  CRISIS OVERRIDE?      │
              │  If CRISIS: λ = 0.95   │
              │  Else: λ = λ_adaptive  │
              └────────────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │  STORVIK (μ, σ)        │
              └────────────────────────┘
```

### Circuit Breaker Interaction

**Disable circuit breaker during known crisis.** During crisis, high surprise is expected - we don't want spurious resets.

```c
// In adaptive forgetting update:
if (crisis_mode == CRISIS_ACTIVE || crisis_mode == CRISIS_RECOVERING) {
    // Skip circuit breaker check - high surprise is expected
    af->enable_circuit_breaker = 0;  // Temporarily disable
} else {
    af->enable_circuit_breaker = 1;  // Re-enable in normal operation
}
```

---

## Proposed API

### New Types

```c
typedef enum {
    TRANS_MODE_NORMAL = 0,    // PGAS owns Π, slow learning
    TRANS_MODE_SHADOW,        // ALERT: parallel learning active
    TRANS_MODE_CRISIS,        // Storvik owns Π, fast learning
    TRANS_MODE_RECOVERING     // Checking exit
} TransitionMode;
```

### New Fields in RBPF_Extended

```c
// Crisis mode state
TransitionMode trans_mode;
double shadow_counts[RBPF_MAX_REGIMES][RBPF_MAX_REGIMES];
double pgas_buffer[RBPF_MAX_REGIMES][RBPF_MAX_REGIMES];
double trans_forgetting_crisis;    // 0.95 (faster than normal 0.995)
int ticks_in_crisis;

// λ override
int crisis_lambda_override_active;
double crisis_lambda_value;        // 0.95
```

### New Functions (Called by CrisisDetector)

```c
// Enter shadow mode (IDLE → ALERT)
void rbpf_ext_enter_shadow_mode(RBPF_Extended *ext);
// - Copy trans_counts → pgas_buffer
// - Copy trans_counts → shadow_counts
// - trans_mode = TRANS_MODE_SHADOW
// - emission λ: still adaptive

// Update shadow during ALERT (called every tick)
void rbpf_ext_shadow_update(RBPF_Extended *ext, int from_regime, int to_regime);
// - Update shadow_counts with fast λ
// - trans_counts stays frozen

// Confirm crisis (ALERT → CRISIS)
void rbpf_ext_confirm_crisis(RBPF_Extended *ext, float log_sr, float log_H);
// - Compute β from SR strength
// - Blend: trans_counts = (1-β)*pgas_buffer + β*shadow_counts
// - Clear shadow_counts
// - trans_mode = TRANS_MODE_CRISIS
// - Activate emission λ override (0.95)
// - Rebuild LUT

// Abort crisis (ALERT → IDLE, false alarm)
void rbpf_ext_abort_crisis(RBPF_Extended *ext);
// - Restore: trans_counts = pgas_buffer
// - Clear shadow_counts, pgas_buffer
// - trans_mode = TRANS_MODE_NORMAL

// Begin recovery (CRISIS → RECOVERING)
void rbpf_ext_begin_recovery(RBPF_Extended *ext);
// - trans_mode = TRANS_MODE_RECOVERING
// - Deactivate emission λ override
// - Slow down trans λ

// Confirm exit (RECOVERING → IDLE)
void rbpf_ext_exit_crisis(RBPF_Extended *ext);
// - trans_mode = TRANS_MODE_NORMAL
// - Clear pgas_buffer

// Re-trigger (RECOVERING → CRISIS)
void rbpf_ext_retrigger_crisis(RBPF_Extended *ext);
// - trans_mode = TRANS_MODE_CRISIS
// - Reactivate emission λ override

// Query current mode
TransitionMode rbpf_ext_get_trans_mode(const RBPF_Extended *ext);

// Check if PGAS should be vetoed
int rbpf_ext_should_veto_pgas(const RBPF_Extended *ext);
```

---

## LUT Rebuild Frequency

| Mode | Rebuild LUT Every |
|------|-------------------|
| NORMAL | 100 ticks (current `trans_update_interval`) |
| SHADOW | Don't rebuild (trans_counts frozen) |
| CRISIS | **Every tick** (fast adaptation needed) |
| RECOVERING | 50 ticks |

---

## PGAS Veto Logic

```c
int rbpf_ext_should_veto_pgas(const RBPF_Extended *ext) {
    switch (ext->trans_mode) {
        case TRANS_MODE_SHADOW:
            return 1;  // Always veto during ALERT
            
        case TRANS_MODE_CRISIS:
            // Veto until PGAS has seen crisis data
            return (ext->ticks_in_crisis < 500);
            
        case TRANS_MODE_RECOVERING:
            return 1;  // Veto during recovery check
            
        case TRANS_MODE_NORMAL:
        default:
            return 0;  // Allow injection
    }
}
```

---

## Integration in Tick Loop

```c
void tick_update(System *sys, float obs) {
    
    // ... existing code ...
    
    // Check for PGAS output
    if (pgas_oracle_try_hot_swap(oracle, pgas_pi, &pgas_tick)) {
        
        // Veto check
        if (!rbpf_ext_should_veto_pgas(sys->rbpf_ext)) {
            rbpf_ksc_update_transition_matrix_threadsafe(
                sys->rbpf_ext->rbpf, pgas_pi);
        }
    }
    
    // CrisisDetector state machine (owns the transitions)
    switch (sys->crisis.state) {
        case CRISIS_IDLE:
            if (hawkes_armed || log_sr_up > 0.5 * log_H) {
                rbpf_ext_enter_shadow_mode(sys->rbpf_ext);
                sys->crisis.state = CRISIS_ALERT;
            }
            break;
            
        case CRISIS_ALERT:
            // Update shadow with current transition
            rbpf_ext_shadow_update(sys->rbpf_ext, prev_regime, curr_regime);
            
            if (log_sr_up > log_H) {
                rbpf_ext_confirm_crisis(sys->rbpf_ext, log_sr_up, log_H);
                sys->crisis.state = CRISIS_ACTIVE;
            } else if (log_sr_up < 0 && ticks_in_state > 5) {
                rbpf_ext_abort_crisis(sys->rbpf_ext);
                sys->crisis.state = CRISIS_IDLE;
            }
            break;
            
        case CRISIS_ACTIVE:
            // Fast transition learning happens in rbpf_ext_step
            
            if (log_sr_down > LOG_H_EXIT) {
                rbpf_ext_begin_recovery(sys->rbpf_ext);
                sys->crisis.state = CRISIS_RECOVERING;
            }
            break;
            
        case CRISIS_RECOVERING:
            if (ticks_in_state >= 10 && 
                log_sr_down > LOG_H_EXIT && 
                log_sr_up < 0.5 * log_H) {
                rbpf_ext_exit_crisis(sys->rbpf_ext);
                sys->crisis.state = CRISIS_IDLE;
            } else if (log_sr_up > 0.7 * log_H) {
                rbpf_ext_retrigger_crisis(sys->rbpf_ext);
                sys->crisis.state = CRISIS_ACTIVE;
            }
            break;
    }
    
    // ... rest of tick loop ...
}
```

---

## Configuration Constants

```c
// Forgetting factors
#define TRANS_LAMBDA_NORMAL    0.995   // N_eff ≈ 200
#define TRANS_LAMBDA_CRISIS    0.95    // N_eff ≈ 20
#define EMISSION_LAMBDA_CRISIS 0.95    // N_eff ≈ 20

// SR thresholds
#define LOG_H_BASE             5.0     // log(148) for entry
#define LOG_H_EXIT             3.0     // log(20) for exit

// Timing
#define PGAS_WINDOW_T          500     // PGAS window size
#define LUT_REBUILD_CRISIS     1       // Every tick during crisis
#define LUT_REBUILD_NORMAL     100     // Every 100 ticks normally
```

---

## Summary

### Who Learns What

| Parameter | Learner | λ Normal | λ Crisis |
|-----------|---------|----------|----------|
| μ (means) | Storvik | Adaptive (~0.998) | Override (0.95) |
| σ (vols) | Storvik | Adaptive (~0.998) | Override (0.95) |
| Π (transitions) | Trans Learning | 0.995 | 0.95 |

### Who Owns What

| Responsibility | Owner |
|----------------|-------|
| State machine (IDLE/ALERT/CRISIS/RECOVERING) | CrisisDetector |
| Shadow mode buffers | RBPF_Extended |
| Emission learning | Storvik (inside RBPF_Extended) |
| Transition learning | Trans Learning (inside RBPF_Extended) |
| Structure estimation | PGAS (background thread) |

### Key Design Decisions

1. **CrisisDetector owns state machine** - RBPF provides primitives
2. **Global transitions** - single `trans_counts[K][K]`, not per-particle
3. **Shadow for transitions only** - Storvik has built-in diversity
4. **Blend, don't discard** - PGAS historical knowledge is valuable
5. **Override emission λ during crisis** - CrisisDetector knows before RBPF
6. **Disable circuit breaker during crisis** - High surprise is expected

---

## Implementation Checklist

- [ ] Add `TransitionMode` enum to RBPF header
- [ ] Add `shadow_counts`, `pgas_buffer` arrays to `RBPF_Extended`
- [ ] Add crisis λ override fields to `RBPF_Extended`
- [ ] Implement `rbpf_ext_enter_shadow_mode()`
- [ ] Implement `rbpf_ext_shadow_update()`
- [ ] Implement `rbpf_ext_confirm_crisis()` with convex blending
- [ ] Implement `rbpf_ext_abort_crisis()`
- [ ] Implement `rbpf_ext_begin_recovery()`
- [ ] Implement `rbpf_ext_exit_crisis()`
- [ ] Implement `rbpf_ext_retrigger_crisis()`
- [ ] Implement `rbpf_ext_should_veto_pgas()`
- [ ] Modify `update_transition_counts_optimized()` to respect trans_mode
- [ ] Modify `rebuild_transition_lut()` for crisis frequency
- [ ] Wire CrisisDetector to RBPF API
- [ ] Test: false alarm rollback
- [ ] Test: confirmed crisis blend
- [ ] Test: PGAS veto and catch-up
- [ ] Test: emission λ override effect
