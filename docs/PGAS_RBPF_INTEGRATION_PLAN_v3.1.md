# PGAS Oracle + RBPF Integration Plan v3.1

## Overview

Hybrid architecture where **PGAS provides structure (Π)** and **Storvik tracks state (μ, σ)**.

**New in v3.1:** Fixes from review — log-space SR, winsorization, explicit Hawkes events, convex blending, full parameter canonicalization, dual SR for principled exit detection.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           TICK LOOP (Core 0-1)                          │
│                                                                         │
│   Observation ──┬──→ Event Detector ──→ Hawkes (on events only)        │
│                 │                              │                        │
│                 │                              ▼ Intensity Signal       │
│                 │                                                       │
│                 └──→ Dual SR ──→ log_sr_up (peace→crisis)              │
│                            └──→ log_sr_down (crisis→peace)             │
│                                       │                                 │
│                                       ▼ Likelihood Signal               │
│                              ┌────────────────┐                        │
│                              │ Decision Logic │                        │
│                              │ (Dual Signal)  │                        │
│                              └───────┬────────┘                        │
│                                      │                                  │
│                    ┌─────────────────┼─────────────────┐               │
│                    ▼                 ▼                 ▼               │
│              [IDLE: PGAS]    [ALERT: Shadow]   [CRISIS: Storvik]       │
│                    │                                   │                │
│                    └──────────► RBPF ◄────────────────┘                │
└─────────────────────────────────────────────────────────────────────────┘
                                   ↑
                                   │ Publishes Π + σ + μ every 50 ticks
┌─────────────────────────────────────────────────────────────────────────┐
│                        PGAS ORACLE (Cores 2-7)                          │
│                                                                         │
│   Ring Buffer ──→ Sliding Window ──→ CSMC ──→ Dirichlet ──→ Π,σ,μ     │
│      (T=500)         (S=50)        (N=64)                              │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## The Core Philosophy: Two-Stage Detonator

We separate **Speed** (Alerting) from **Truth** (Validation).

### Stage 1: The Alarm (Hawkes Process on Events)

| Aspect | Description |
|--------|-------------|
| **Role** | Motion Sensor — monitors clustering of meaningful events |
| **Input** | Explicit events (quantile breaks, volume spikes), NOT raw ticks |
| **Speed** | <10ms reaction time |
| **Action** | ARM the system, veto PGAS, start shadow mode |

### Stage 2: The Judge (Dual Shiryaev-Roberts)

| Aspect | Description |
|--------|-------------|
| **Role** | Validator — accumulates likelihood evidence |
| **SR_up** | Tests peace → crisis (entry) |
| **SR_down** | Tests crisis → peace (exit) |
| **Why Principled** | Proper hypothesis test in both directions |

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

### SR Configuration

```c
const float LOG_H_BASE     = 5.0f;   // log(148) ≈ 148:1 odds for entry
const float LOG_H_EXIT     = 3.0f;   // log(20) ≈ 20:1 odds for exit
const float SIGMA_MULTIPLE = 3.0f;   // MDE: crisis = 3× peace volatility
const float WINSORIZE_CAP  = 10.0f;  // Cap |z| to prevent outlier domination
const float STUDENT_NU     = 6.0f;   // Degrees of freedom if using Student-t
```

### Hawkes Event Configuration

```c
const float EVENT_QUANTILE    = 0.90f;  // |r| > 90th percentile
const float VOLUME_MULTIPLE   = 2.0f;   // Volume > 2× EMA
const float IMBALANCE_MULTIPLE = 2.0f;  // |Imbalance| > 2× EMA
```

---

## The "Split Brain" Reset Strategy

**Core Insight:** PGAS dictates the *rules* (Π), Storvik tracks the *score* (μ, σ).

| Parameter | Type | Owner | Reset on Inject? | Reason |
|-----------|------|-------|------------------|--------|
| Π (transition matrix) | Structure | PGAS | ✓ **YES** | Storvik can't learn structure in 50 ticks |
| μ (regime means) | State | Storvik | ✗ **NO** | Tracks continuously |
| σ (regime vols) | State | Storvik | ✗ **NO** | Tracks continuously |

---

## Explicit Hawkes Events

### The Problem

At fixed 100Hz, Hawkes on raw returns is just a fancy volatility EMA. It loses independent value vs SR.

### The Solution: Define Meaningful Events

```c
typedef struct {
    /* Rolling statistics for event detection */
    float return_quantile_90;   /* 90th percentile of |r| over last 500 ticks */
    float volume_ema;           /* EMA of volume */
    float imbalance_ema;        /* EMA of |bid-ask imbalance| */
    
    /* Ring buffer for quantile estimation */
    float return_buffer[500];
    int   buffer_idx;
    
} EventDetector;

void event_detector_update(EventDetector *evt, float obs) {
    /* Update ring buffer */
    evt->return_buffer[evt->buffer_idx] = fabsf(obs);
    evt->buffer_idx = (evt->buffer_idx + 1) % 500;
    
    /* Update quantile estimate (could use P² or similar) */
    evt->return_quantile_90 = estimate_quantile(evt->return_buffer, 500, 0.90f);
}

bool is_hawkes_event(EventDetector *evt, float obs, float volume, float imbalance) {
    /*
     * Event = significant return OR volume spike OR imbalance spike
     * This gives Hawkes independent value from SR
     */
    
    bool return_event = (fabsf(obs) > evt->return_quantile_90);
    bool volume_event = (volume > VOLUME_MULTIPLE * evt->volume_ema);
    bool imbalance_event = (fabsf(imbalance) > IMBALANCE_MULTIPLE * evt->imbalance_ema);
    
    return return_event || volume_event || imbalance_event;
}
```

### Hawkes Updates on Events Only

```c
void hawkes_conditional_update(HawkesIntegrator *h, bool is_event, int64_t tick) {
    /* Decay always happens (time passes) */
    hawkes_decay(h, tick);
    
    /* Excitation only on meaningful events */
    if (is_event) {
        hawkes_add_event(h, tick);
    }
}
```

**Why this matters:** Hawkes now detects clustering of *meaningful* events, not just high volatility. This provides independent information from SR.

---

## Log-Space Shiryaev-Roberts

### The Bug in v3.0

We said "H_BASE = 5.0 means e^5 ≈ 148:1 odds" but compared `sr_stat > H` where `sr_stat` was linear R_t. Threshold of 5 ≠ threshold of 148.

### The Fix: Everything in Log-Space

```c
typedef struct {
    /* Log-space SR statistics */
    float log_sr_up;        /* log(R_t) for peace → crisis */
    float log_sr_down;      /* log(R_t) for crisis → peace */
    
    /* Baseline parameters */
    float sigma_peace;      /* From PGAS calm regime */
    float sigma_crisis;     /* Learned during crisis (EMA) */
    
    /* Adaptive threshold components */
    float log_H_base;       /* 5.0 = log(148) */
    int   ticks_since_crisis;
    int   recent_false_alarms;
    
    /* State */
    SystemState state;
    int ticks_in_state;
    
    /* Frozen baseline for exit */
    float frozen_sigma_peace;
    bool  baseline_frozen;
    
} CrisisDetector;
```

### Log-Likelihood Ratio (Winsorized)

```c
float compute_log_lr_winsorized(float obs, float sigma_h0, float sigma_h1) {
    /*
     * Compute log(p(obs|H1) / p(obs|H0)) with winsorization
     * 
     * H0: obs ~ N(0, sigma_h0)
     * H1: obs ~ N(0, sigma_h1)
     */
    
    /* Winsorize to prevent single outlier domination */
    float z_h0 = obs / sigma_h0;
    z_h0 = fmaxf(fminf(z_h0, WINSORIZE_CAP), -WINSORIZE_CAP);
    
    float z_h1 = obs / sigma_h1;
    z_h1 = fmaxf(fminf(z_h1, WINSORIZE_CAP), -WINSORIZE_CAP);
    
    /* Gaussian log-likelihood ratio */
    float log_LR = 0.5f * (z_h0 * z_h0 - z_h1 * z_h1)
                 + logf(sigma_h0 / sigma_h1);
    
    return log_LR;
}
```

### Alternative: Student-t (More Robust)

```c
float compute_log_lr_student_t(float obs, float sigma_h0, float sigma_h1, float nu) {
    /*
     * Student-t log-likelihood ratio
     * More robust to heavy tails than Gaussian
     * nu = 5-8 typical for financial returns
     */
    
    float z_h0 = obs / sigma_h0;
    float z_h1 = obs / sigma_h1;
    
    /* Student-t: log p(z|nu) ∝ -(nu+1)/2 * log(1 + z²/nu) */
    float log_LR = 0.5f * (nu + 1.0f) * (
        logf(1.0f + z_h0 * z_h0 / nu) - 
        logf(1.0f + z_h1 * z_h1 / nu)
    ) + logf(sigma_h0 / sigma_h1);
    
    return log_LR;
}
```

### SR Update in Log-Space

```c
void update_log_sr(float *log_sr, float log_LR) {
    /*
     * SR recursion: R_t = LR_t × (1 + R_{t-1})
     * In log-space: log(R_t) = log(LR_t) + log(1 + R_{t-1})
     *                        = log(LR_t) + log(1 + exp(log_sr))
     */
    
    float log_one_plus_R = log1pf(expf(*log_sr));
    *log_sr = log_LR + log_one_plus_R;
    
    /* Clamp to prevent overflow (e^20 ≈ 500M is plenty) */
    if (*log_sr > 20.0f) *log_sr = 20.0f;
    if (*log_sr < -20.0f) *log_sr = -20.0f;
}
```

---

## Dual SR: Principled Entry AND Exit

### The Problem with Single SR

v3.0 used SR for entry but ad-hoc "intensity + SR < 1" for exit. This isn't principled.

### The Solution: Two Hypothesis Tests

| Statistic | Null (H0) | Alternative (H1) | Purpose |
|-----------|-----------|------------------|---------|
| **log_sr_up** | Peace (σ_peace) | Crisis (3×σ_peace) | Detect crisis entry |
| **log_sr_down** | Crisis (σ_crisis) | Peace (σ_peace) | Detect crisis exit |

### Implementation

```c
void update_dual_sr(CrisisDetector *det, float obs) {
    
    switch (det->state) {
        
        case STATE_IDLE:
        case STATE_ALERT:
            /* SR_up active: testing peace → crisis */
            {
                float sigma_h0 = det->sigma_peace;
                float sigma_h1 = SIGMA_MULTIPLE * det->sigma_peace;
                float log_LR = compute_log_lr_winsorized(obs, sigma_h0, sigma_h1);
                update_log_sr(&det->log_sr_up, log_LR);
            }
            /* SR_down inactive */
            det->log_sr_down = 0.0f;
            break;
            
        case STATE_CRISIS_ACTIVE:
            /* Learn crisis sigma (EMA of |obs|, with warmup) */
            if (det->ticks_in_state < 10) {
                /* Warmup: initialize sigma_crisis */
                det->sigma_crisis = 0.9f * det->sigma_crisis + 0.1f * fabsf(obs);
            } else {
                /* Steady state: slower adaptation */
                det->sigma_crisis = 0.99f * det->sigma_crisis + 0.01f * fabsf(obs);
            }
            
            /* SR_down active: testing crisis → peace */
            {
                float sigma_h0 = det->sigma_crisis;
                float sigma_h1 = det->frozen_sigma_peace;
                float log_LR = compute_log_lr_winsorized(obs, sigma_h0, sigma_h1);
                update_log_sr(&det->log_sr_down, log_LR);
            }
            /* SR_up reset */
            det->log_sr_up = 0.0f;
            break;
            
        case STATE_RECOVERING:
            /* Both active for hysteresis */
            {
                /* SR_up: could crisis re-trigger? */
                float sigma_h0_up = det->frozen_sigma_peace;
                float sigma_h1_up = SIGMA_MULTIPLE * det->frozen_sigma_peace;
                float log_LR_up = compute_log_lr_winsorized(obs, sigma_h0_up, sigma_h1_up);
                update_log_sr(&det->log_sr_up, log_LR_up);
                
                /* SR_down: is exit confirmed? */
                float sigma_h0_down = det->sigma_crisis;
                float sigma_h1_down = det->frozen_sigma_peace;
                float log_LR_down = compute_log_lr_winsorized(obs, sigma_h0_down, sigma_h1_down);
                update_log_sr(&det->log_sr_down, log_LR_down);
            }
            break;
    }
}
```

---

## Adaptive Threshold (Log-Space)

```c
float compute_adaptive_log_threshold(CrisisDetector *det) {
    float log_H = det->log_H_base;  /* 5.0 = log(148) */
    
    /* Inertia: longer peace → higher bar (additive in log-space) */
    log_H += logf(1.0f + det->ticks_since_crisis / 1000.0f);
    
    /* Wolf penalty: recent false alarms → higher bar */
    log_H += 2.0f * fminf(det->recent_false_alarms, 3);
    
    return log_H;
}
```

| Component | Formula | Effect |
|-----------|---------|--------|
| **Base** | 5.0 | 148:1 odds minimum |
| **Inertia** | +log(1 + ticks/1000) | Long peace → harder to trigger |
| **Wolf** | +2.0 per false alarm | Recent lies → trust less |

---

## Crisis State Machine (v3.1)

### State Diagram

```
                              SR_up rejected (log_sr_up < 0)
                    ┌──────────────────────────────────────┐
                    │                                      │
                    ▼                                      │
    IDLE ──[Hawkes OR log_sr_up > 0.5H]──→ ALERT ──[log_sr_up > H]──→ CRISIS
     ▲                                                           │
     │                                                           │
     │         [log_sr_down > H_exit                             │
     │          for 10 consecutive ticks]                        │
     │                                                           ▼
     └───────────────────────────────────────────────────── RECOVERING
                                                                 │
                         [log_sr_up > 0.7H: false exit] ─────────┘
```

### State Definitions

| State | PGAS Veto | Storvik Π | SR_up | SR_down | Description |
|-------|-----------|-----------|-------|---------|-------------|
| **IDLE** | No | OFF | Accumulating | Inactive | Peace, PGAS owns Π |
| **ALERT** | Yes | SHADOW | Accumulating | Inactive | Probation, validation |
| **CRISIS_ACTIVE** | Yes (< T) / No (≥ T) | ON | Reset | Accumulating | Confirmed crisis |
| **RECOVERING** | Yes | OFF | Accumulating | Accumulating | Checking exit |

---

## Shadow Mode with Convex Blending

### The Problem

With 64 particles and λ=0.6, shadow Π can hallucinate structure from noise. Direct copy is risky.

### The Solution: Blend Based on SR Strength

```c
void rbpf_confirm_crisis_blended(RBPF *rbpf, float log_sr, float log_H) {
    /*
     * β = blend factor, increases with SR strength
     * 
     * At threshold (log_sr = log_H):     β ≈ 0.5
     * Well above (log_sr = log_H + 3):   β → 0.95
     * 
     * Π_new = (1-β) × Π_PGAS + β × Π_shadow
     */
    
    float excess = log_sr - log_H;
    float beta = 0.5f + 0.45f * tanhf(excess);  /* Smooth [0.05, 0.95] */
    
    /* Blend transition counts */
    for (int i = 0; i < K; i++) {
        rbpf->trans_row_sums[i] = 0.0f;
        for (int j = 0; j < K; j++) {
            rbpf->trans_counts[i][j] = 
                (1.0f - beta) * rbpf->pgas_buffer_counts[i][j] +
                beta * rbpf->shadow_counts[i][j];
            rbpf->trans_row_sums[i] += rbpf->trans_counts[i][j];
        }
    }
    
    rbpf_lut_rebuild_transitions(rbpf);
    
    rbpf->storvik_learning_active = true;
    rbpf->shadow_active = false;
    /* Keep pgas_buffer — might need it if crisis is short */
}
```

### Why This Works

| SR Strength | β | Result |
|-------------|---|--------|
| Barely crosses (log_sr ≈ log_H) | 0.5 | 50% PGAS, 50% shadow — conservative |
| Strongly crosses (log_sr = log_H + 2) | 0.85 | Mostly shadow — confident |
| Overwhelmingly (log_sr = log_H + 4) | 0.95 | Almost pure shadow — very confident |

Prevents "shadow hallucination becomes primary structure."

---

## Full Parameter Canonicalization

### The Problem

We sort Π by σ, but μ associations can break if not sorted together.

### The Solution: Sort Everything

```c
typedef struct {
    float Pi[K * K];        /* Transition matrix */
    float sigma_vol[K];     /* Emission volatilities — REQUIRED */
    float mu_vol[K];        /* Emission means — REQUIRED */
    float ar_coef[K];       /* AR coefficients if applicable */
    int64_t tick;
} PGASChannelBuffer;

void inject_physics_canonicalized(RBPF *rbpf, 
                                   const PGASChannelBuffer *pgas,
                                   float N_eff) {
    
    /* 1. Compute permutation by σ (ascending: calm=0, crisis=K-1) */
    LabelPermutation perm = compute_label_permutation(pgas->sigma_vol, K);
    
    /* 2. Apply permutation to ALL regime-specific parameters */
    float Pi_sorted[K * K];
    float sigma_sorted[K];
    float mu_sorted[K];
    
    for (int canonical = 0; canonical < K; canonical++) {
        int pgas_idx = perm.canonical_to_pgas[canonical];
        sigma_sorted[canonical] = pgas->sigma_vol[pgas_idx];
        mu_sorted[canonical] = pgas->mu_vol[pgas_idx];
    }
    
    /* Apply permutation to Π */
    for (int a = 0; a < K; a++) {
        int i = perm.canonical_to_pgas[a];
        for (int b = 0; b < K; b++) {
            int j = perm.canonical_to_pgas[b];
            Pi_sorted[a * K + b] = pgas->Pi[i * K + j];
        }
    }
    
    /* 3. Inject transition matrix */
    for (int i = 0; i < K; i++) {
        rbpf->trans_row_sums[i] = 0.0f;
        for (int j = 0; j < K; j++) {
            float new_count = Pi_sorted[i * K + j] * N_eff;
            rbpf->trans_counts[i][j] = new_count;
            rbpf->trans_row_sums[i] += new_count;
        }
    }
    
    rbpf_lut_rebuild_transitions(rbpf);
    
    /* 4. Optionally update emission parameter priors (if RBPF tracks them) */
    /* rbpf_update_emission_priors(rbpf, mu_sorted, sigma_sorted); */
}
```

---

## Baseline Management

### Freeze on Crisis Confirmation

```c
void freeze_baseline(CrisisDetector *det) {
    det->frozen_sigma_peace = det->sigma_peace;
    det->baseline_frozen = true;
    
    /* Initialize sigma_crisis for SR_down */
    det->sigma_crisis = SIGMA_MULTIPLE * det->sigma_peace;
}

void unfreeze_baseline(CrisisDetector *det) {
    det->baseline_frozen = false;
    det->sigma_crisis = 0.0f;
}
```

---

## Complete Tick Loop (v3.1)

```c
void tick_update(System *sys, float obs, float volume, float imbalance) {
    
    /* ═══════════════════════════════════════════════════════════════
     * 1. UPDATE EVENT DETECTOR
     * ═══════════════════════════════════════════════════════════════*/
    event_detector_update(&sys->evt, obs);
    bool is_event = is_hawkes_event(&sys->evt, obs, volume, imbalance);
    
    /* ═══════════════════════════════════════════════════════════════
     * 2. UPDATE HAWKES (on events only)
     * ═══════════════════════════════════════════════════════════════*/
    hawkes_conditional_update(&sys->det.hawkes, is_event, sys->tick);
    bool hawkes_armed = (sys->det.hawkes.state == HAWKES_ARMED);
    
    /* ═══════════════════════════════════════════════════════════════
     * 3. UPDATE DUAL SR (always, ~3 operations)
     * ═══════════════════════════════════════════════════════════════*/
    update_dual_sr(&sys->det, obs);
    
    /* ═══════════════════════════════════════════════════════════════
     * 4. STATE MACHINE (Dual-Signal Decision)
     * ═══════════════════════════════════════════════════════════════*/
    float log_H = compute_adaptive_log_threshold(&sys->det);
    
    switch (sys->det.state) {
        
        case STATE_IDLE:
            sys->det.ticks_since_crisis++;
            
            /* Entry: Hawkes fires OR SR accumulating */
            if (hawkes_armed || sys->det.log_sr_up > 0.5f * log_H) {
                sys->det.state = STATE_ALERT;
                rbpf_enter_shadow_mode(sys->rbpf);
                sys->det.ticks_in_state = 0;
            }
            break;
            
        case STATE_ALERT:
            sys->det.ticks_in_state++;
            
            /* Update shadow in parallel */
            rbpf_shadow_update(sys->rbpf, 0.6f);
            
            if (sys->det.log_sr_up > log_H) {
                /* SR confirmed — commit with blending */
                sys->det.state = STATE_CRISIS_ACTIVE;
                rbpf_confirm_crisis_blended(sys->rbpf, sys->det.log_sr_up, log_H);
                freeze_baseline(&sys->det);
                sys->det.log_sr_up = 0.0f;
                sys->det.ticks_since_crisis = 0;
                sys->det.ticks_in_state = 0;
            }
            else if (sys->det.log_sr_up < 0.0f && sys->det.ticks_in_state > 5) {
                /* SR rejected — false alarm */
                sys->det.state = STATE_IDLE;
                rbpf_abort_crisis(sys->rbpf);
                sys->det.recent_false_alarms++;
                sys->det.log_sr_up = 0.0f;
                sys->det.ticks_in_state = 0;
            }
            break;
            
        case STATE_CRISIS_ACTIVE:
            sys->det.ticks_in_state++;
            
            /* Exit detection via SR_down */
            if (sys->det.log_sr_down > LOG_H_EXIT) {
                sys->det.state = STATE_RECOVERING;
                sys->det.ticks_in_state = 0;
            }
            break;
            
        case STATE_RECOVERING:
            sys->det.ticks_in_state++;
            
            /* Confirm exit: SR_down sustained, SR_up not re-triggering */
            if (sys->det.ticks_in_state >= 10 && 
                sys->det.log_sr_down > LOG_H_EXIT &&
                sys->det.log_sr_up < 0.5f * log_H) {
                /* Clean exit */
                sys->det.state = STATE_IDLE;
                rbpf_exit_crisis_mode(sys->rbpf);
                unfreeze_baseline(&sys->det);
                sys->det.log_sr_up = 0.0f;
                sys->det.log_sr_down = 0.0f;
                sys->det.recent_false_alarms = 0;  /* Reset wolf penalty */
                sys->det.ticks_in_state = 0;
            }
            else if (sys->det.log_sr_up > 0.7f * log_H) {
                /* False exit — crisis re-triggering */
                sys->det.state = STATE_CRISIS_ACTIVE;
                sys->det.log_sr_down = 0.0f;
                sys->det.ticks_in_state = 0;
            }
            break;
    }
    
    /* ═══════════════════════════════════════════════════════════════
     * 5. STORVIK UPDATE (during crisis)
     * ═══════════════════════════════════════════════════════════════*/
    if (sys->det.state == STATE_CRISIS_ACTIVE) {
        float lambda = (sys->det.ticks_in_state < PGAS_WINDOW_T) ? 0.60f : 0.95f;
        rbpf_storvik_update_transitions(sys->rbpf, lambda);
    }
    
    /* ═══════════════════════════════════════════════════════════════
     * 6. PGAS INJECTION (if not vetoed)
     * ═══════════════════════════════════════════════════════════════*/
    bool should_veto = (sys->det.state == STATE_ALERT ||
                        sys->det.state == STATE_RECOVERING ||
                        (sys->det.state == STATE_CRISIS_ACTIVE && 
                         sys->det.ticks_in_state < PGAS_WINDOW_T));
    
    if (!should_veto && atomic_load(&sys->mailbox.updated)) {
        rbpf_enforce_canonical_order(sys->rbpf);
        inject_physics_canonicalized(sys->rbpf, &sys->mailbox, 1000.0f);
        atomic_store(&sys->mailbox.updated, false);
        
        /* Update sigma_peace for SR computation */
        sys->det.sigma_peace = sys->mailbox.sigma_vol[0];  /* Calm regime */
    }
    
    /* ═══════════════════════════════════════════════════════════════
     * 7. RUN RBPF STEP
     * ═══════════════════════════════════════════════════════════════*/
    rbpf_step(sys->rbpf, obs);
}
```

---

## Decision Matrix

| Scenario | Hawkes | SR_up | SR_down | Outcome |
|----------|--------|-------|---------|---------|
| **Flash crash** | Events cluster | Explodes | — | Fast confirm, high β blend |
| **Slow bleed** | No events | Accumulates | — | SR alone triggers ALERT→CRISIS |
| **Fat finger** | 1 event | Spikes then negative | — | SR rejects, restore PGAS |
| **Choppy market** | Many events | Oscillates | — | Wolf penalty rises, threshold rises |
| **Deep crisis** | — | Reset | Low | PGAS catches up at tick 500 |
| **Clean exit** | — | Low | Rises | SR_down confirms, clean exit |
| **False exit** | — | Rising | — | SR_up re-triggers, back to crisis |

---

## Performance

| Component | Per-Tick Cost |
|-----------|---------------|
| Event detection | ~500ns |
| Hawkes update (conditional) | ~200ns (avg, only on events ~1μs) |
| Dual SR (2× log1p, 2× exp) | ~300ns |
| Adaptive threshold (1× log) | ~50ns |
| Shadow update (during ALERT) | ~5μs |
| Convex blend (on confirm) | ~10μs (rare) |

**Total tick cost:** ~1-2μs typical, ~6μs during ALERT.

---

## Summary

| What | Value |
|------|-------|
| **Hawkes Input** | Explicit events (quantile, volume, imbalance) |
| **SR Representation** | Log-space (log_sr_up, log_sr_down) |
| **Outlier Protection** | Winsorize at \|z\| ≤ 10 (or Student-t ν=6) |
| **Entry Test** | SR_up: peace → crisis (3σ MDE) |
| **Exit Test** | SR_down: crisis → peace |
| **log_H_base (entry)** | 5.0 = log(148) ≈ 148:1 odds |
| **log_H_exit** | 3.0 = log(20) ≈ 20:1 odds |
| **Shadow Promotion** | Convex blend β = f(SR strength) |
| **Parameter Injection** | Full canonicalization (Π, σ, μ) |

---

## Fixes from v3.0

| Issue | v3.0 | v3.1 |
|-------|------|------|
| **A) Outliers** | Raw Gaussian LR | Winsorized \|z\| ≤ 10 |
| **B) Log-space bug** | Linear R_t vs log threshold | Everything in log-space |
| **C) Hawkes events** | Every tick | Explicit events only |
| **D) Shadow hallucination** | Direct copy | Convex blend by SR strength |
| **E) μ canonicalization** | Only Π, σ sorted | Full parameter bundle |
| **F) Exit logic** | Ad-hoc intensity + SR<1 | Principled SR_down test |

---

## Mantras

> **Hawkes sees events, not ticks.** Define meaningful events or Hawkes is just a fancy EMA.

> **Log-space or disaster.** R_t overflows. Thresholds in odds ratios only make sense in log.

> **Winsorize the outliers.** One 15σ tick shouldn't dominate a week of evidence.

> **Blend, don't copy.** Shadow Π with 64 particles and λ=0.6 can hallucinate.

> **Canonicalize everything.** Π, σ, μ must all use the same label permutation.

> **Two tests, two directions.** SR_up for entry, SR_down for exit. No ad-hoc hysteresis.

> **Inertia protects history.** Long peace → high bar to wipe.

> **Wolf penalty silences noise.** Recent lies → trust less.

---

## Implementation Checklist

- [ ] **Event Detection**
  - [ ] Implement rolling quantile for returns
  - [ ] Add volume EMA
  - [ ] Add imbalance EMA
  - [ ] Implement `is_hawkes_event()`
  - [ ] Wire Hawkes to event stream

- [ ] **Log-Space SR**
  - [ ] Replace `sr_stat` with `log_sr_up`, `log_sr_down`
  - [ ] Implement `compute_log_lr_winsorized()`
  - [ ] Implement `update_log_sr()` with log1p
  - [ ] Update all threshold comparisons to log-space
  - [ ] Test numerical stability

- [ ] **Dual SR**
  - [ ] Implement state-dependent SR updates
  - [ ] Add `sigma_crisis` learning during CRISIS_ACTIVE
  - [ ] Implement exit detection via SR_down
  - [ ] Implement RECOVERING with both SR active
  - [ ] Test hysteresis behavior

- [ ] **Convex Blending**
  - [ ] Implement `rbpf_confirm_crisis_blended()`
  - [ ] Compute β from SR strength
  - [ ] Test blend behavior at various SR levels

- [ ] **Full Canonicalization**
  - [ ] PGAS exports μ_vol[K] with every publication
  - [ ] Apply permutation to μ, σ, Π together
  - [ ] Update `inject_physics_canonicalized()`

- [ ] **Adaptive Threshold**
  - [ ] Implement log-space threshold computation
  - [ ] Add inertia term
  - [ ] Add wolf penalty
  - [ ] Reset wolf on clean exit

- [ ] **State Machine**
  - [ ] IDLE with dual entry paths
  - [ ] ALERT with SR validation
  - [ ] CRISIS_ACTIVE with SR_down exit detection
  - [ ] RECOVERING with dual SR hysteresis

- [ ] **Validation**
  - [ ] Flash crash (events + SR_up fast)
  - [ ] Slow bleed (SR_up alone)
  - [ ] Fat finger (SR rejects)
  - [ ] Choppy market (wolf penalty)
  - [ ] Clean exit (SR_down confirms)
  - [ ] False exit (SR_up re-triggers)
  - [ ] Outlier robustness (15σ tick)
  - [ ] Log-space numerical stability
