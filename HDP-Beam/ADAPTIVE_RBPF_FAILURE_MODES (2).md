# Adaptive RBPF: Failure Modes and Mitigations

## Executive Summary

The adaptive RBPF architecture uses Particle Gibbs (PG) on a background thread to discover and refine regime structure while RBPF filters online. This document analyzes:

- **3 Primary Failure Modes** (structural): Hot Swap Shock, Poor MCMC Mixing, Semantic Drift
- **5 Secondary Failure Modes** (operational): Golden Sample Fallacy, Threshold Dead Zones, Fast-Forward Bottleneck, Arithmetic Divergence, Learning Rate Stiffening

Each failure mode includes root cause analysis, detection metrics, and mitigation strategies with implementation code.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DUAL-THREAD DESIGN                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Main Thread (per tick, ~5μs):                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  [RBPF step] → [Online VI update] → [Check for PG update]   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│         │                                      ↑                    │
│         │ trigger (health-based)               │ publish            │
│         ▼                                      │                    │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              Lock-Free SPSC Channel                         │   │
│  └─────────────────────────────────────────────────────────────┘   │
│         │                                      ↑                    │
│         ▼                                      │                    │
│  PG Thread (background, ~1-5ms per sweep):                         │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  [CSMC sweep] → [Validate] → [Publish if valid]             │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Table of Contents

**Primary Failure Modes**
1. [Hot Swap Shock](#1-hot-swap-shock)
2. [Poor MCMC Mixing](#2-poor-mcmc-mixing)
3. [Semantic Drift](#3-semantic-drift)

**Core Components**
4. [PG Trigger Design](#4-pg-trigger-design)
5. [The Lifeboat Protocol](#5-the-lifeboat-protocol)

**Secondary Failure Modes**
6. [Golden Sample Fallacy](#6-golden-sample-fallacy)
7. [Threshold Dead Zones](#7-threshold-dead-zones)
8. [Fast-Forward Bottleneck](#8-fast-forward-bottleneck)
9. [Arithmetic vs Geometric Divergence](#9-arithmetic-vs-geometric-divergence)
10. [Learning Rate Stiffening](#10-learning-rate-stiffening)

**Operations**
11. [Stress Testing Checklist](#11-stress-testing-checklist)
12. [Implementation Checklist](#12-implementation-checklist)

---

## 1. Hot Swap Shock

### The Problem

When PG updates the model structure, RBPF particles represent beliefs conditioned on the **old** model. The particles exist in an incompatible state space.

```
Before PG:  particles valid under θ_old, K_old
After PG:   θ_new, K_new (possibly K changed)
Result:     P(old particles | new model) ≈ 0
            ESS → 0, filter dies
```

### Why Re-weighting Fails

Standard importance re-weighting requires:

```
w_new = w_old × P(history | θ_new) / P(history | θ_old)
```

If K changed (regime birth/death), the state space dimensions don't match. The ratio is undefined.

### The Solution: Lifeboat Protocol

Don't salvage old particles. **Replace them** with fresh particles from PG's internal CSMC.

See [Section 5: The Lifeboat Protocol](#5-the-lifeboat-protocol) for full specification.

### Quick Reference

| Approach | Viability | Notes |
|----------|-----------|-------|
| Re-weight old particles | ❌ Fails | P(old \| new) ≈ 0 |
| Relabel indices only | ⚠️ Partial | Only works if K unchanged |
| **Inject CSMC particles** | ✅ Robust | Bypasses incompatibility |

---

## 2. Poor MCMC Mixing

### The Problem

Even Particle Gibbs can return samples that haven't fully converged. A single sweep starting from the previous state might not traverse the posterior adequately during structural changes.

```
Risk: PG returns a sample that is WORSE than the current model
      because the chain hasn't mixed properly.
```

### Symptoms of Poor Mixing

- Stickiness changes drastically (>30%) in one update
- K changes by more than ±1
- μ ordering violated after update
- Transition rows don't sum to 1.0

### Solution A: Validation Gate

Reject updates that look pathological:

```c
typedef struct {
    double max_stickiness_drop;    /* e.g., 0.30 (30%) */
    int max_K_change;              /* e.g., 1 */
    double min_row_sum;            /* e.g., 0.999 */
    double max_row_sum;            /* e.g., 1.001 */
} ValidationConfig;

bool validate_pg_update(RBPF *rbpf, PG_Update *update, ValidationConfig *cfg)
{
    /* Rule 1: Stickiness can't crash */
    for (int k = 0; k < update->K && k < rbpf->K; k++) {
        double old_stick = rbpf->trans[k * rbpf->K + k];
        double new_stick = update->trans[k * update->K + k];
        
        if (old_stick > 0.5 && new_stick < old_stick * (1 - cfg->max_stickiness_drop)) {
            return false;  /* Reject: stickiness crashed */
        }
    }
    
    /* Rule 2: Bounded K changes */
    if (abs(update->K - rbpf->K) > cfg->max_K_change) {
        return false;  /* Reject: too many regimes added/removed */
    }
    
    /* Rule 3: μ must be ordered */
    for (int k = 0; k < update->K - 1; k++) {
        if (update->mu_vol[k] >= update->mu_vol[k + 1]) {
            return false;  /* Reject: ordering violated */
        }
    }
    
    /* Rule 4: Rows must sum to 1 */
    for (int i = 0; i < update->K; i++) {
        double sum = 0;
        for (int j = 0; j < update->K; j++) {
            sum += update->trans[i * update->K + j];
        }
        if (sum < cfg->min_row_sum || sum > cfg->max_row_sum) {
            return false;  /* Reject: invalid transition matrix */
        }
    }
    
    return true;  /* Accept */
}
```

### Solution B: Tempering

Instead of instant swap, blend old and new parameters over 5-10 ticks:

```c
typedef struct {
    double trans_current[K_MAX * K_MAX];
    double trans_target[K_MAX * K_MAX];
    double mu_current[K_MAX];
    double mu_target[K_MAX];
    double sigma_current[K_MAX];
    double sigma_target[K_MAX];
    int blend_remaining;
    int blend_total;
} TemperedTransition;

void start_tempered_blend(TemperedTransition *tt, 
                          const PG_Update *update,
                          int blend_ticks)
{
    memcpy(tt->trans_target, update->trans, K_MAX * K_MAX * sizeof(double));
    memcpy(tt->mu_target, update->mu_vol, K_MAX * sizeof(double));
    memcpy(tt->sigma_target, update->sigma_vol, K_MAX * sizeof(double));
    tt->blend_remaining = blend_ticks;
    tt->blend_total = blend_ticks;
}

void apply_tempered_step(TemperedTransition *tt, RBPF *rbpf)
{
    if (tt->blend_remaining <= 0) return;
    
    double alpha = 1.0 / tt->blend_remaining;  /* Increasing weight to target */
    
    for (int i = 0; i < rbpf->K * rbpf->K; i++) {
        tt->trans_current[i] = (1 - alpha) * tt->trans_current[i] 
                              + alpha * tt->trans_target[i];
    }
    
    for (int k = 0; k < rbpf->K; k++) {
        tt->mu_current[k] = (1 - alpha) * tt->mu_current[k] 
                           + alpha * tt->mu_target[k];
        tt->sigma_current[k] = (1 - alpha) * tt->sigma_current[k] 
                              + alpha * tt->sigma_target[k];
    }
    
    /* Copy blended values to RBPF */
    memcpy(rbpf->trans, tt->trans_current, rbpf->K * rbpf->K * sizeof(double));
    memcpy(rbpf->mu_vol, tt->mu_current, rbpf->K * sizeof(double));
    memcpy(rbpf->sigma_vol, tt->sigma_current, rbpf->K * sizeof(double));
    
    tt->blend_remaining--;
}
```

### Why Tempering Helps

```
Without tempering:
  Tick 100: trans = [0.90, 0.10]
  Tick 101: trans = [0.65, 0.35]  ← 28% jump, particles shocked

With tempering (5 ticks):
  Tick 100: trans = [0.90, 0.10]
  Tick 101: trans = [0.85, 0.15]
  Tick 102: trans = [0.80, 0.20]
  Tick 103: trans = [0.75, 0.25]
  Tick 104: trans = [0.70, 0.30]
  Tick 105: trans = [0.65, 0.35]  ← Gradual, particles adapt
```

---

## 3. Semantic Drift

### The Problem

In a fixed system, "Regime 2 = High Vol" is invariant. In an adaptive system where K can change, the meaning of regime indices can drift.

```
Scenario:
  - Market calms significantly over weeks
  - PG merges old "High Vol" into "Medium Vol"
  - PG creates new "Ultra-Low Vol" as Regime 0
  - Previous "Regime 1" is now "Regime 2"
  - Any hardcoded regime-specific logic breaks
```

### Solution A: μ-Ordering (Already Implemented)

States are always sorted by μ (volatility level):

```
Regime 0:   lowest μ   (calmest)
Regime K-1: highest μ  (crisis)
```

This preserves **ordinal** meaning: "higher regime index = higher volatility."

### Solution B: Label Alignment

Before injecting PG particles, align new regime labels to minimize disruption:

```c
/* Cost matrix: C[i][j] = distance from old regime i to new regime j */
void compute_alignment_cost(double *C, int K_old, int K_new,
                            const double *mu_old, const double *sigma_old,
                            const double *mu_new, const double *sigma_new)
{
    for (int i = 0; i < K_old; i++) {
        for (int j = 0; j < K_new; j++) {
            double d_mu = fabs(mu_old[i] - mu_new[j]);
            double d_sigma = fabs(sigma_old[i] - sigma_new[j]);
            C[i * K_new + j] = d_mu + 0.5 * d_sigma;
        }
    }
}

/* Greedy assignment (O(K²), sufficient for K ≤ 8) */
void greedy_alignment(const double *C, int K, int *old_to_new)
{
    bool used[K_MAX] = {false};
    
    for (int i = 0; i < K; i++) {
        double best_cost = DBL_MAX;
        int best_j = 0;
        
        for (int j = 0; j < K; j++) {
            if (!used[j] && C[i * K + j] < best_cost) {
                best_cost = C[i * K + j];
                best_j = j;
            }
        }
        
        old_to_new[i] = best_j;
        used[best_j] = true;
    }
}
```

### Solution C: Anchor Endpoints

Fix the semantic meaning of extreme regimes with hard priors:

```c
typedef struct {
    double mu_calm_ceiling;    /* Regime 0 must have μ below this */
    double mu_crisis_floor;    /* Regime K-1 must have μ above this */
    bool enforce;
} RegimeAnchors;

void init_anchors(RegimeAnchors *anchors)
{
    /* Based on domain knowledge:
     * μ = -2.5 corresponds to ~10% annualized vol (calm)
     * μ = +1.0 corresponds to ~50% annualized vol (crisis)
     */
    anchors->mu_calm_ceiling = -2.0;
    anchors->mu_crisis_floor = 0.5;
    anchors->enforce = true;
}

bool validate_anchors(const PG_Update *update, const RegimeAnchors *anchors)
{
    if (!anchors->enforce) return true;
    
    /* Regime 0 must be "calm" */
    if (update->mu_vol[0] > anchors->mu_calm_ceiling) {
        return false;  /* Lowest regime drifted too high */
    }
    
    /* Regime K-1 must be "crisis-capable" */
    if (update->mu_vol[update->K - 1] < anchors->mu_crisis_floor) {
        return false;  /* Highest regime drifted too low */
    }
    
    return true;
}
```

### When Anchor Validation Fails

| Violation | Interpretation | Action |
|-----------|----------------|--------|
| Regime 0 too high | Market has no calm period | Consider forcing K+1, inject calm regime |
| Regime K-1 too low | Market has no crisis mode | Accept (safer to underestimate crisis) |
| Both | Major structural shift | Reject update, keep current model |

---

## 4. PG Trigger Design

### The Wrong Approach (Why We Don't Use It)

Triggering on **uncertainty** (entropy, KL divergence) creates feedback loops:

```
VI uncertain → trigger PG → PG adjusts → VI recalibrates → 
VI uncertain again → trigger PG → ...
```

This was a concern in earlier designs but **does not apply** when PG runs on its own thread with health-based triggers.

### The Right Approach: Health-Based Triggers

Trigger on **failure**, not **uncertainty**:

| Trigger | Meaning | Creates Loop? |
|---------|---------|---------------|
| **ESS collapse** | Particles degenerating | ❌ No — real problem |
| **Log-likelihood drop** | Model failing predictions | ❌ No — real problem |
| **Regime starvation** | One regime dominates too long | ❌ No — structural issue |

These detect "something is broken" rather than "I'm unsure."

### Implementation

```c
typedef struct {
    /*─────────────────────────────────────────────────────────────────
     * ESS-based trigger: particles dying
     *─────────────────────────────────────────────────────────────────*/
    double ess_floor;              /* Trigger if ESS < this (e.g., N/4) */
    int ess_failures;              /* Consecutive ticks below floor */
    int ess_patience;              /* Trigger after this many (e.g., 10) */
    
    /*─────────────────────────────────────────────────────────────────
     * Log-likelihood trigger: model predictions failing
     *─────────────────────────────────────────────────────────────────*/
    double loglik_ema;             /* Exponential moving average */
    double loglik_baseline;        /* Expected log-lik when healthy */
    double loglik_threshold;       /* Trigger if below (e.g., baseline - 2σ) */
    int loglik_failures;           /* Consecutive failures */
    int loglik_patience;           /* Trigger after this many (e.g., 20) */
    
    /*─────────────────────────────────────────────────────────────────
     * Regime dominance trigger: structure stale
     *─────────────────────────────────────────────────────────────────*/
    int regime_counts[K_MAX];      /* Visits in recent window */
    int window_position;           /* Circular buffer index */
    int window_size;               /* e.g., 500 ticks */
    double dominance_threshold;    /* Trigger if one regime > 98% */
    
    /*─────────────────────────────────────────────────────────────────
     * Cooldown: prevent rapid re-triggering
     *─────────────────────────────────────────────────────────────────*/
    int cooldown_remaining;
    int cooldown_ticks;            /* e.g., 100 ticks */
    
    /*─────────────────────────────────────────────────────────────────
     * Statistics
     *─────────────────────────────────────────────────────────────────*/
    uint64_t total_triggers;
    uint64_t ess_triggers;
    uint64_t loglik_triggers;
    uint64_t dominance_triggers;
    
} PG_Trigger;

void pg_trigger_init(PG_Trigger *trig, int n_particles)
{
    memset(trig, 0, sizeof(*trig));
    
    /* ESS config */
    trig->ess_floor = n_particles / 4.0;
    trig->ess_patience = 10;
    
    /* Log-likelihood config */
    trig->loglik_baseline = -2.0;  /* Calibrate from data */
    trig->loglik_threshold = -5.0; /* 3σ below baseline */
    trig->loglik_patience = 20;
    
    /* Dominance config */
    trig->window_size = 500;
    trig->dominance_threshold = 0.98;
    
    /* Cooldown */
    trig->cooldown_ticks = 100;
}

void pg_trigger_update_dominance(PG_Trigger *trig, int regime, int K)
{
    /* Circular buffer update */
    trig->regime_counts[regime]++;
    trig->window_position++;
    
    if (trig->window_position >= trig->window_size) {
        /* Reset counts periodically (simple approach) */
        memset(trig->regime_counts, 0, sizeof(trig->regime_counts));
        trig->window_position = 0;
    }
}

bool pg_trigger_check_dominance(PG_Trigger *trig, int K)
{
    if (trig->window_position < trig->window_size / 2) {
        return false;  /* Not enough data yet */
    }
    
    int total = 0;
    int max_count = 0;
    
    for (int k = 0; k < K; k++) {
        total += trig->regime_counts[k];
        if (trig->regime_counts[k] > max_count) {
            max_count = trig->regime_counts[k];
        }
    }
    
    if (total == 0) return false;
    
    double dominance = (double)max_count / total;
    return dominance > trig->dominance_threshold;
}

bool should_trigger_pg(PG_Trigger *trig, RBPF *rbpf, double log_lik)
{
    /* Respect cooldown */
    if (trig->cooldown_remaining > 0) {
        trig->cooldown_remaining--;
        return false;
    }
    
    bool triggered = false;
    
    /*─────────────────────────────────────────────────────────────────
     * Trigger 1: ESS collapse (particles dying)
     *─────────────────────────────────────────────────────────────────*/
    double ess = rbpf_get_ess(rbpf);
    
    if (ess < trig->ess_floor) {
        trig->ess_failures++;
        if (trig->ess_failures >= trig->ess_patience) {
            trig->ess_triggers++;
            triggered = true;
        }
    } else {
        trig->ess_failures = 0;
    }
    
    /*─────────────────────────────────────────────────────────────────
     * Trigger 2: Log-likelihood collapse (model wrong)
     *─────────────────────────────────────────────────────────────────*/
    const double alpha = 0.05;
    trig->loglik_ema = (1 - alpha) * trig->loglik_ema + alpha * log_lik;
    
    if (log_lik < trig->loglik_threshold) {
        trig->loglik_failures++;
        if (trig->loglik_failures >= trig->loglik_patience) {
            trig->loglik_triggers++;
            triggered = true;
        }
    } else {
        trig->loglik_failures = 0;
    }
    
    /*─────────────────────────────────────────────────────────────────
     * Trigger 3: Regime dominance (structure stale)
     *─────────────────────────────────────────────────────────────────*/
    int current_regime = rbpf_get_map_regime(rbpf);
    pg_trigger_update_dominance(trig, current_regime, rbpf->K);
    
    if (pg_trigger_check_dominance(trig, rbpf->K)) {
        trig->dominance_triggers++;
        triggered = true;
    }
    
    /*─────────────────────────────────────────────────────────────────
     * Apply cooldown if triggered
     *─────────────────────────────────────────────────────────────────*/
    if (triggered) {
        trig->cooldown_remaining = trig->cooldown_ticks;
        trig->total_triggers++;
        
        /* Reset failure counters */
        trig->ess_failures = 0;
        trig->loglik_failures = 0;
    }
    
    return triggered;
}
```

### Trigger Summary

| Trigger | Detects | Patience | Typical Threshold |
|---------|---------|----------|-------------------|
| ESS | Particle degeneracy | 10 ticks | ESS < N/4 |
| Log-likelihood | Prediction failure | 20 ticks | 3σ below baseline |
| Dominance | Stale structure | 500 tick window | >98% one regime |

### Why This Doesn't Loop

```
Old (bad) - uncertainty-based:
  Uncertainty → Trigger → Model changes → Uncertainty persists → Trigger → ...
  
New (good) - health-based:
  ESS collapse → Trigger → Lifeboat injects fresh particles → ESS recovers → No trigger
  Log-lik drop → Trigger → PG finds better model → Log-lik recovers → No trigger
  Dominance   → Trigger → PG refines structure → Dominance breaks → No trigger
```

Each trigger is **self-correcting**: fixing the problem removes the trigger condition.

---

## 5. The Lifeboat Protocol

### Core Concept: Trajectory Injection

When PG produces a new model, don't try to salvage old particles. **Replace them** with fresh particles from PG's internal Conditional SMC.

```
Old approach (fails):
  Old particles ──[re-weight]──► Adapted particles
                        ↑
                  P(old | new) ≈ 0
                        ↓
                  Weight collapse

Lifeboat approach (works):
  PG's CSMC particles ──[inject]──► Fresh RBPF particles
  Buffered observations ──[fast-forward]──► Caught up to present
```

### Step 1: Label Alignment

Before injection, align new regime labels to minimize semantic disruption:

```c
void align_labels(PG_Update *update, const RBPF *rbpf)
{
    if (update->K != rbpf->K) {
        /* K changed: just use μ-ordering (already done by PG) */
        return;
    }
    
    /* Same K: find best alignment to old regimes */
    double cost[K_MAX * K_MAX];
    int perm[K_MAX];
    
    compute_alignment_cost(cost, rbpf->K, update->K,
                           rbpf->mu_vol, rbpf->sigma_vol,
                           update->mu_vol, update->sigma_vol);
    
    greedy_alignment(cost, update->K, perm);
    apply_permutation(update, perm);
}

void apply_permutation(PG_Update *update, const int *perm)
{
    double trans_temp[K_MAX * K_MAX];
    double mu_temp[K_MAX], sigma_temp[K_MAX];
    int K = update->K;
    
    /* Permute emissions */
    for (int k = 0; k < K; k++) {
        mu_temp[k] = update->mu_vol[perm[k]];
        sigma_temp[k] = update->sigma_vol[perm[k]];
    }
    memcpy(update->mu_vol, mu_temp, K * sizeof(double));
    memcpy(update->sigma_vol, sigma_temp, K * sizeof(double));
    
    /* Permute transition matrix: π'[i][j] = π[perm[i]][perm[j]] */
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < K; j++) {
            trans_temp[i * K + j] = update->trans[perm[i] * K + perm[j]];
        }
    }
    memcpy(update->trans, trans_temp, K * K * sizeof(double));
    
    /* Permute CSMC particles' regime labels */
    for (int n = 0; n < update->n_csmc_particles; n++) {
        int old_regime = update->csmc_particles[n].regime;
        update->csmc_particles[n].regime = perm[old_regime];
    }
}
```

### Step 2: Particle Injection

Extract particles from PG's CSMC and inject into RBPF:

```c
typedef struct {
    /* Timing */
    int t_trigger;              /* When PG started */
    int t_now;                  /* Current wall-clock tick */
    
    /* Buffered observations during PG */
    double *buffered_obs;
    int buffer_len;
    int buffer_capacity;
    
    /* CSMC particles from Particle Gibbs */
    RBPFParticle *csmc_particles;
    double *csmc_weights;
    int n_csmc;
    
    /* New model parameters */
    PG_Update update;
    
} LifeboatPacket;

void lifeboat_inject(RBPF *rbpf, LifeboatPacket *pkt)
{
    const int N = rbpf->n_particles;
    const int M = pkt->n_csmc;
    
    /*─────────────────────────────────────────────────────────────────
     * Step A: Multinomial resample CSMC particles to fill RBPF
     *─────────────────────────────────────────────────────────────────*/
    int indices[RBPF_MAX_PARTICLES];
    multinomial_resample(pkt->csmc_weights, M, indices, N);
    
    for (int n = 0; n < N; n++) {
        rbpf->particles[n] = pkt->csmc_particles[indices[n]];
        rbpf->weights[n] = 1.0 / N;  /* Uniform after resampling */
    }
    
    /*─────────────────────────────────────────────────────────────────
     * Step B: Jitter to prevent degeneracy
     *─────────────────────────────────────────────────────────────────*/
    for (int n = 0; n < N; n++) {
        /* Small noise on continuous state (log-vol) */
        rbpf->particles[n].h += 0.01 * randn();
    }
    
    /*─────────────────────────────────────────────────────────────────
     * Step C: Update RBPF model parameters
     *─────────────────────────────────────────────────────────────────*/
    rbpf->K = pkt->update.K;
    memcpy(rbpf->trans, pkt->update.trans, rbpf->K * rbpf->K * sizeof(double));
    memcpy(rbpf->mu_vol, pkt->update.mu_vol, rbpf->K * sizeof(double));
    memcpy(rbpf->sigma_vol, pkt->update.sigma_vol, rbpf->K * sizeof(double));
}
```

### Step 3: Fast-Forward Catch-Up

Process buffered observations to synchronize with current time:

```c
void lifeboat_catchup(RBPF *rbpf, LifeboatPacket *pkt)
{
    for (int i = 0; i < pkt->buffer_len; i++) {
        double y = pkt->buffered_obs[i];
        
        /* Standard RBPF step */
        rbpf_step_internal(rbpf, y);
        
        /* NO Kelly sizing during catch-up */
        /* NO external signals during catch-up */
    }
    
    pkt->buffer_len = 0;
}
```

### Complete Protocol

```c
typedef enum {
    LIFEBOAT_SUCCESS,
    LIFEBOAT_REJECTED_VALIDATION,
    LIFEBOAT_REJECTED_ANCHORS,
} LifeboatResult;

LifeboatResult execute_lifeboat(RBPF *rbpf, OnlineVI *vi, 
                                 LifeboatPacket *pkt,
                                 ValidationConfig *val_cfg,
                                 RegimeAnchors *anchors)
{
    /*─────────────────────────────────────────────────────────────────
     * Phase 1: Validate update quality
     *─────────────────────────────────────────────────────────────────*/
    if (!validate_pg_update(rbpf, &pkt->update, val_cfg)) {
        pkt->buffer_len = 0;  /* Discard buffer */
        return LIFEBOAT_REJECTED_VALIDATION;
    }
    
    /*─────────────────────────────────────────────────────────────────
     * Phase 2: Validate regime anchors
     *─────────────────────────────────────────────────────────────────*/
    if (!validate_anchors(&pkt->update, anchors)) {
        pkt->buffer_len = 0;
        return LIFEBOAT_REJECTED_ANCHORS;
    }
    
    /*─────────────────────────────────────────────────────────────────
     * Phase 3: Label alignment
     *─────────────────────────────────────────────────────────────────*/
    align_labels(&pkt->update, rbpf);
    
    /*─────────────────────────────────────────────────────────────────
     * Phase 4: Inject particles
     *─────────────────────────────────────────────────────────────────*/
    lifeboat_inject(rbpf, pkt);
    
    /*─────────────────────────────────────────────────────────────────
     * Phase 5: Fast-forward through buffer
     *─────────────────────────────────────────────────────────────────*/
    lifeboat_catchup(rbpf, pkt);
    
    /*─────────────────────────────────────────────────────────────────
     * Phase 6: Reset Online VI
     *─────────────────────────────────────────────────────────────────*/
    online_vi_reset_from_hdp(vi, pkt->update.trans, 10.0);
    
    return LIFEBOAT_SUCCESS;
}
```

### Why Lifeboat Works

| Approach | Problem | Result |
|----------|---------|--------|
| Re-weight | P(old history \| new model) ≈ 0 | ESS → 0 |
| Relabel only | State space mismatch if K changed | Partial fix |
| **Lifeboat** | Fresh particles + fast-forward | ESS preserved |

Key insight: **Don't evaluate new model on old history.** Transplant the history that *generated* the new model.

---

## 6. Golden Sample Fallacy

### The Problem

The Lifeboat Protocol assumes that the Particle Gibbs sample is an "oracle" superior to the current RBPF state. This is not always true.

```
Risk: PG sweep returns a sample that has not reached the stationary
      distribution (poor mixing). The Validation Gate catches "broken"
      updates but cannot detect "suboptimal" updates.

Consequence: Lifeboat REPLACES rather than blends. A single mediocre
             PG sample forces the entire filter into a local mode that
             may be harder to recover from than the original state.
```

### Why Validation Gate Is Insufficient

The Validation Gate checks for pathological updates:
- Stickiness crash > 30%
- K change > 1
- μ ordering violated
- Row sums ≠ 1

But a sample can pass all these checks and still be:
- A low-probability trajectory from the posterior
- Biased toward a local mode
- Not representative of the true posterior mean

### Solution: Ensemble Validation

Don't trust a single sample. Run multiple PG sweeps and check consistency:

```c
typedef struct {
    double trans_samples[PG_ENSEMBLE_SIZE][K_MAX * K_MAX];
    double mu_samples[PG_ENSEMBLE_SIZE][K_MAX];
    double sigma_samples[PG_ENSEMBLE_SIZE][K_MAX];
    int n_samples;
    int current_idx;
    
    /* Thresholds */
    double trans_variance_threshold;  /* Reject if variance too high */
    double mu_variance_threshold;
} PG_Ensemble;

void pg_ensemble_init(PG_Ensemble *ens)
{
    memset(ens, 0, sizeof(*ens));
    ens->trans_variance_threshold = 0.01;  /* Calibrate empirically */
    ens->mu_variance_threshold = 0.05;
}

void pg_ensemble_add_sample(PG_Ensemble *ens, const PG_Update *update)
{
    int idx = ens->current_idx % PG_ENSEMBLE_SIZE;
    
    memcpy(ens->trans_samples[idx], update->trans, 
           update->K * update->K * sizeof(double));
    memcpy(ens->mu_samples[idx], update->mu_vol, 
           update->K * sizeof(double));
    memcpy(ens->sigma_samples[idx], update->sigma_vol, 
           update->K * sizeof(double));
    
    ens->current_idx++;
    if (ens->n_samples < PG_ENSEMBLE_SIZE) {
        ens->n_samples++;
    }
}

bool pg_ensemble_is_stable(const PG_Ensemble *ens, int K)
{
    if (ens->n_samples < 3) {
        return false;  /* Not enough samples to judge */
    }
    
    /*─────────────────────────────────────────────────────────────────
     * Check variance of transition matrix entries
     *─────────────────────────────────────────────────────────────────*/
    double trans_var_sum = 0;
    
    for (int i = 0; i < K * K; i++) {
        /* Compute mean */
        double mean = 0;
        for (int s = 0; s < ens->n_samples; s++) {
            mean += ens->trans_samples[s][i];
        }
        mean /= ens->n_samples;
        
        /* Compute variance */
        double var = 0;
        for (int s = 0; s < ens->n_samples; s++) {
            double diff = ens->trans_samples[s][i] - mean;
            var += diff * diff;
        }
        trans_var_sum += var / ens->n_samples;
    }
    
    if (trans_var_sum / (K * K) > ens->trans_variance_threshold) {
        return false;  /* Transition samples too variable */
    }
    
    /*─────────────────────────────────────────────────────────────────
     * Check variance of emission means
     *─────────────────────────────────────────────────────────────────*/
    double mu_var_sum = 0;
    
    for (int k = 0; k < K; k++) {
        double mean = 0;
        for (int s = 0; s < ens->n_samples; s++) {
            mean += ens->mu_samples[s][k];
        }
        mean /= ens->n_samples;
        
        double var = 0;
        for (int s = 0; s < ens->n_samples; s++) {
            double diff = ens->mu_samples[s][k] - mean;
            var += diff * diff;
        }
        mu_var_sum += var / ens->n_samples;
    }
    
    if (mu_var_sum / K > ens->mu_variance_threshold) {
        return false;  /* Emission samples too variable */
    }
    
    return true;  /* Samples are consistent → chain has mixed */
}

PG_Update pg_ensemble_get_mean(const PG_Ensemble *ens, int K)
{
    PG_Update mean_update;
    mean_update.K = K;
    
    /* Average transition matrix */
    for (int i = 0; i < K * K; i++) {
        double sum = 0;
        for (int s = 0; s < ens->n_samples; s++) {
            sum += ens->trans_samples[s][i];
        }
        mean_update.trans[i] = sum / ens->n_samples;
    }
    
    /* Normalize rows */
    for (int i = 0; i < K; i++) {
        double row_sum = 0;
        for (int j = 0; j < K; j++) {
            row_sum += mean_update.trans[i * K + j];
        }
        for (int j = 0; j < K; j++) {
            mean_update.trans[i * K + j] /= row_sum;
        }
    }
    
    /* Average emissions */
    for (int k = 0; k < K; k++) {
        double mu_sum = 0, sigma_sum = 0;
        for (int s = 0; s < ens->n_samples; s++) {
            mu_sum += ens->mu_samples[s][k];
            sigma_sum += ens->sigma_samples[s][k];
        }
        mean_update.mu_vol[k] = mu_sum / ens->n_samples;
        mean_update.sigma_vol[k] = sigma_sum / ens->n_samples;
    }
    
    return mean_update;
}
```

### Updated PG Thread Logic

```c
void pg_thread_main(PG_Context *ctx)
{
    PG_Ensemble ensemble;
    pg_ensemble_init(&ensemble);
    
    while (!ctx->shutdown) {
        /* Wait for trigger or heartbeat */
        wait_for_trigger(ctx);
        
        /* Run multiple sweeps */
        for (int sweep = 0; sweep < PG_ENSEMBLE_SIZE; sweep++) {
            PG_Update sample = run_pg_sweep(ctx);
            pg_ensemble_add_sample(&ensemble, &sample);
        }
        
        /* Check if chain has mixed */
        if (!pg_ensemble_is_stable(&ensemble, ctx->K)) {
            log_warning("PG ensemble unstable, running more sweeps");
            
            /* Run additional sweeps */
            for (int sweep = 0; sweep < PG_ENSEMBLE_SIZE; sweep++) {
                PG_Update sample = run_pg_sweep(ctx);
                pg_ensemble_add_sample(&ensemble, &sample);
            }
            
            if (!pg_ensemble_is_stable(&ensemble, ctx->K)) {
                log_warning("PG still unstable, skipping update");
                continue;  /* Don't publish */
            }
        }
        
        /* Publish ensemble mean, not single sample */
        PG_Update mean_update = pg_ensemble_get_mean(&ensemble, ctx->K);
        publish_pg_update(ctx->channel, &mean_update);
    }
}
```

### Key Insight

```
Single sample:   High variance, may be in local mode
Ensemble mean:   Lower variance, closer to true posterior mean
```

If successive sweeps disagree → chain hasn't mixed → wait for more sweeps or skip update.

---

## 7. Threshold Dead Zones

### The Problem

The health-based triggers use absolute thresholds:
- ESS < N/4
- Log-likelihood < -5.0
- Dominance > 98%

These are "magic numbers" that may not hold across different market regimes or asset classes.

```
Failure Mode: "Dead Zones"

The model performs poorly (e.g., slightly late signals that lose money)
but health metrics stay JUST ABOVE the trigger floors.

Example:
  ESS hovers at N/4 + 1 (just above threshold)
  Log-lik hovers at -4.9 (just above -5.0)
  → No trigger fires
  → Model slowly degrades
  → P&L suffers
```

### Opposite Risk: Thrashing

```
If thresholds are too tight:
  ESS drops to N/3 → Trigger!
  Lifeboat injects → ESS recovers to N/2
  ... 50 ticks later ...
  ESS drops to N/3 → Trigger!
  
  → System constantly in "transition" mode
  → Kelly = 0 too often
  → Miss trading opportunities
```

### Solution: Adaptive Thresholds

Use **relative** thresholds based on recent history, not absolute values:

```c
typedef struct {
    /*─────────────────────────────────────────────────────────────────
     * Rolling Baselines (exponential moving averages)
     *─────────────────────────────────────────────────────────────────*/
    double ess_ema;              /* Recent average ESS */
    double loglik_ema;           /* Recent average log-likelihood */
    double loglik_var_ema;       /* Variance of log-likelihood */
    
    /*─────────────────────────────────────────────────────────────────
     * Relative Trigger Parameters
     *─────────────────────────────────────────────────────────────────*/
    double ess_drop_ratio;       /* Trigger if ESS < ratio × baseline */
    double loglik_sigma_mult;    /* Trigger if log-lik < baseline - mult×σ */
    
    /*─────────────────────────────────────────────────────────────────
     * Absolute Floors (safety nets)
     *─────────────────────────────────────────────────────────────────*/
    double ess_absolute_floor;   /* Never ignore ESS below this */
    double loglik_absolute_floor;/* Never ignore log-lik below this */
    
    /*─────────────────────────────────────────────────────────────────
     * Adaptation Rate
     *─────────────────────────────────────────────────────────────────*/
    double alpha;                /* EMA decay (e.g., 0.01 = slow) */
    
    /*─────────────────────────────────────────────────────────────────
     * State
     *─────────────────────────────────────────────────────────────────*/
    bool initialized;
    uint64_t n_updates;
    
} AdaptiveThresholds;

void adaptive_thresholds_init(AdaptiveThresholds *th, int n_particles)
{
    memset(th, 0, sizeof(*th));
    
    /* Relative parameters */
    th->ess_drop_ratio = 0.5;      /* Trigger if ESS drops to 50% of baseline */
    th->loglik_sigma_mult = 2.5;   /* Trigger if log-lik is 2.5σ below baseline */
    
    /* Absolute floors */
    th->ess_absolute_floor = n_particles / 10.0;  /* Never below 10% */
    th->loglik_absolute_floor = -10.0;            /* Catastrophic failure */
    
    /* Adaptation */
    th->alpha = 0.01;  /* Slow adaptation to baseline */
    
    th->initialized = false;
}

void adaptive_thresholds_update(AdaptiveThresholds *th, double ess, double log_lik)
{
    if (!th->initialized) {
        /* Bootstrap with first observation */
        th->ess_ema = ess;
        th->loglik_ema = log_lik;
        th->loglik_var_ema = 1.0;  /* Initial variance estimate */
        th->initialized = true;
        return;
    }
    
    double alpha = th->alpha;
    
    /* Update ESS baseline */
    th->ess_ema = (1 - alpha) * th->ess_ema + alpha * ess;
    
    /* Update log-likelihood baseline and variance */
    double delta = log_lik - th->loglik_ema;
    th->loglik_ema = (1 - alpha) * th->loglik_ema + alpha * log_lik;
    th->loglik_var_ema = (1 - alpha) * th->loglik_var_ema + alpha * delta * delta;
    
    th->n_updates++;
}

typedef struct {
    bool triggered;
    bool ess_trigger;
    bool loglik_trigger;
    bool absolute_trigger;
    double ess_ratio;       /* Current ESS / baseline */
    double loglik_zscore;   /* (current - baseline) / σ */
} TriggerResult;

TriggerResult adaptive_thresholds_check(const AdaptiveThresholds *th, 
                                         double ess, double log_lik)
{
    TriggerResult result = {0};
    
    if (!th->initialized || th->n_updates < 100) {
        /* Not enough data for adaptive thresholds */
        return result;
    }
    
    /*─────────────────────────────────────────────────────────────────
     * Check relative ESS drop
     *─────────────────────────────────────────────────────────────────*/
    result.ess_ratio = ess / th->ess_ema;
    
    if (result.ess_ratio < th->ess_drop_ratio) {
        result.ess_trigger = true;
        result.triggered = true;
    }
    
    /*─────────────────────────────────────────────────────────────────
     * Check relative log-likelihood drop
     *─────────────────────────────────────────────────────────────────*/
    double sigma = sqrt(th->loglik_var_ema);
    if (sigma < 0.01) sigma = 0.01;  /* Prevent division by zero */
    
    result.loglik_zscore = (log_lik - th->loglik_ema) / sigma;
    
    if (result.loglik_zscore < -th->loglik_sigma_mult) {
        result.loglik_trigger = true;
        result.triggered = true;
    }
    
    /*─────────────────────────────────────────────────────────────────
     * Check absolute floors (safety nets)
     *─────────────────────────────────────────────────────────────────*/
    if (ess < th->ess_absolute_floor || log_lik < th->loglik_absolute_floor) {
        result.absolute_trigger = true;
        result.triggered = true;
    }
    
    return result;
}
```

### Why Relative Thresholds Work

```
Scenario: Low-volatility market

Absolute threshold:
  ESS baseline: 900 (out of 1000)
  ESS floor: 250 (N/4)
  → ESS can drop 72% before trigger
  → Dead zone is HUGE

Relative threshold:
  ESS baseline: 900
  Trigger ratio: 0.5
  → Trigger at 450
  → Dead zone is proportional to baseline
```

```
Scenario: High-volatility market

Absolute threshold:
  ESS baseline: 400 (particles struggling)
  ESS floor: 250 (N/4)
  → Only 37% headroom before trigger
  → May trigger too often

Relative threshold:
  ESS baseline: 400
  Trigger ratio: 0.5
  → Trigger at 200
  → Consistent behavior regardless of baseline
```

---

## 8. Fast-Forward Bottleneck

### The Problem

When a Lifeboat update is published, the RBPF must "fast-forward" through buffered observations to catch up to the present.

```
Timeline:
  t=0:    PG triggered, starts sweep
  t=0:    Main thread buffers incoming observations
  ...
  t=5ms:  PG completes, publishes update
  t=5ms:  Buffer contains 100 observations (at 20k ticks/sec)
  t=5ms:  RBPF must process 100 ticks to catch up
  
Problem: During fast-forward, NEW ticks keep arriving
```

### Failure Mode: Perpetual Catch-Up

```
During high-volatility events:
  - PG is most likely to trigger (ESS collapse)
  - Tick density is highest (market moving fast)
  - Buffer grows faster than catch-up can process
  
  Fast-forward takes 2ms
  100 new ticks arrive during those 2ms
  → Filter never catches up
  → Always processing stale data
```

### Resource Contention

```
Both threads on same CPU socket:
  - PG thread: cache-heavy CSMC sweeps
  - Main thread: RBPF + fast-forward
  
Competition for:
  - L3 cache
  - Memory bandwidth
  - CPU cycles (if hyperthreading)
  
→ Main thread's 5μs path may slow to 10μs+
```

### Solution A: Buffer Cap with Drop Policy

Never let buffer grow unbounded. Drop oldest observations when full:

```c
typedef struct {
    double *observations;
    int capacity;
    int head;       /* Write position */
    int tail;       /* Read position */
    int count;
    
    /* Statistics */
    uint64_t total_dropped;
    uint64_t total_buffered;
} CircularBuffer;

void buffer_init(CircularBuffer *buf, int capacity)
{
    buf->observations = aligned_alloc(64, capacity * sizeof(double));
    buf->capacity = capacity;
    buf->head = 0;
    buf->tail = 0;
    buf->count = 0;
    buf->total_dropped = 0;
    buf->total_buffered = 0;
}

void buffer_push(CircularBuffer *buf, double y)
{
    if (buf->count >= buf->capacity) {
        /* Buffer full: drop oldest */
        buf->tail = (buf->tail + 1) % buf->capacity;
        buf->count--;
        buf->total_dropped++;
    }
    
    buf->observations[buf->head] = y;
    buf->head = (buf->head + 1) % buf->capacity;
    buf->count++;
    buf->total_buffered++;
}

double buffer_pop(CircularBuffer *buf)
{
    if (buf->count == 0) return NAN;
    
    double y = buf->observations[buf->tail];
    buf->tail = (buf->tail + 1) % buf->capacity;
    buf->count--;
    return y;
}

#define BUFFER_CAPACITY 200  /* Never more than 200 ticks buffered */
```

### Solution B: Subsampled Catch-Up

If buffer is large, process every Nth observation:

```c
void lifeboat_catchup_subsampled(RBPF *rbpf, CircularBuffer *buf)
{
    int n = buf->count;
    
    if (n == 0) return;
    
    /* Determine subsampling rate */
    int target_processed = 50;  /* Process at most 50 ticks */
    int skip = (n > target_processed) ? n / target_processed : 1;
    
    int idx = 0;
    while (buf->count > 0) {
        double y = buffer_pop(buf);
        
        if (idx % skip == 0) {
            rbpf_step_internal(rbpf, y);
        }
        /* Else: skip this observation */
        
        idx++;
    }
}
```

### Solution C: Time-Bounded Catch-Up

Process as many as possible within a time budget:

```c
typedef struct {
    int max_catchup_us;     /* Time budget for catch-up (e.g., 500μs) */
    int min_ticks;          /* Always process at least this many */
    int processed;          /* Stats: how many processed */
    int discarded;          /* Stats: how many discarded */
} CatchupConfig;

void lifeboat_catchup_bounded(RBPF *rbpf, CircularBuffer *buf, CatchupConfig *cfg)
{
    double t_start = get_time_us();
    cfg->processed = 0;
    cfg->discarded = 0;
    
    while (buf->count > 0) {
        double y = buffer_pop(buf);
        rbpf_step_internal(rbpf, y);
        cfg->processed++;
        
        /* Check time budget after minimum ticks */
        if (cfg->processed >= cfg->min_ticks) {
            double elapsed = get_time_us() - t_start;
            if (elapsed > cfg->max_catchup_us) {
                /* Time's up: discard remaining */
                cfg->discarded = buf->count;
                buf->count = 0;
                buf->head = buf->tail = 0;
                break;
            }
        }
    }
    
    if (cfg->discarded > 0) {
        log_warning("Catch-up timeout: processed %d, discarded %d",
                    cfg->processed, cfg->discarded);
    }
}
```

### Solution D: CPU Affinity

Pin threads to different cores to reduce contention:

```c
#define _GNU_SOURCE
#include <pthread.h>
#include <sched.h>

void pin_to_core(int core_id)
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
}

/* In main thread initialization */
pin_to_core(0);  /* Main thread on core 0 */

/* In PG thread initialization */
pin_to_core(2);  /* PG thread on core 2 (different physical core) */
```

### Recommended Strategy

```c
CatchupConfig default_catchup = {
    .max_catchup_us = 500,   /* 500μs budget */
    .min_ticks = 20,         /* Always process at least 20 */
};

/* Combined approach */
void lifeboat_catchup_robust(RBPF *rbpf, CircularBuffer *buf, CatchupConfig *cfg)
{
    /* If buffer is huge, subsample first */
    if (buf->count > 100) {
        lifeboat_catchup_subsampled(rbpf, buf);
    } else {
        lifeboat_catchup_bounded(rbpf, buf, cfg);
    }
}
```

---

## 9. Arithmetic vs Geometric Divergence

### The Problem

The Online VI module uses **arithmetic means** for computing responsibilities:

```c
/* Current implementation */
double xi_ij = prev_probs[i] * vi->mean[i][j] * regime_liks[j];

/* Where vi->mean[i][j] = α_ij / Σ_k α_ik */
```

But true Variational Bayes requires the **geometric mean** (via digamma function):

```
E[π_ij] = α_ij / Σ_k α_ik                    (arithmetic - WRONG)
exp(E[log π_ij]) = exp(ψ(α_ij) - ψ(Σ_k α_ik))  (geometric - CORRECT)
```

### Why This Matters

For a Dirichlet(α) distribution:
- E[x] ≠ exp(E[log x])
- The difference grows when α is small (high uncertainty)

```
Example: Dirichlet(2, 2, 2, 2) for K=4

Arithmetic mean: E[π_ij] = 0.25
Geometric mean:  exp(ψ(2) - ψ(8)) ≈ 0.227

Difference: 10%
```

### The Drift Risk

Over thousands of ticks:
- Small errors accumulate
- VI converges to different stationary point than PG expects
- VI may signal high entropy when model is actually stable
- Or low entropy when model is actually uncertain

### Solution: Use Digamma for ξ Computation

```c
/* Digamma function (already implemented in online_vi_transition.c) */
static double digamma(double x);

/* Precompute geometric means when stats are recomputed */
static void recompute_stats(OnlineVI *vi)
{
    if (!vi->stats_dirty) return;
    
    const int K = vi->K;
    
    for (int i = 0; i < K; i++) {
        double alpha_0 = vi->alpha_sum[i];
        double psi_alpha_0 = digamma(alpha_0);
        
        for (int j = 0; j < K; j++) {
            double a_ij = vi->alpha[i][j];
            
            /* Arithmetic mean (for backward compatibility) */
            vi->mean[i][j] = a_ij / alpha_0;
            
            /* Geometric mean (for correct VI updates) */
            vi->log_mean[i][j] = digamma(a_ij) - psi_alpha_0;
            vi->geom_mean[i][j] = exp(vi->log_mean[i][j]);
            
            /* Variance (unchanged) */
            vi->var[i][j] = a_ij * (alpha_0 - a_ij) / 
                           (alpha_0 * alpha_0 * (alpha_0 + 1.0));
        }
    }
    
    vi->stats_dirty = false;
}

/* Updated ξ computation using geometric mean */
void online_vi_update_correct(OnlineVI *vi,
                               const double *regime_probs,
                               const double *regime_liks)
{
    /* ... initialization check ... */
    
    recompute_stats(vi);
    
    const int K = vi->K;
    double xi[K][K];
    double Z = 0.0;
    
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < K; j++) {
            /* USE GEOMETRIC MEAN, NOT ARITHMETIC */
            double xi_ij = vi->prev_probs[i] * vi->geom_mean[i][j] * regime_liks[j];
            xi[i][j] = xi_ij;
            Z += xi_ij;
        }
    }
    
    /* ... rest of update unchanged ... */
}
```

### Performance Impact

```
Digamma: ~10-20 ns per call
K = 4:   16 entries × 20 ns = 320 ns per stats recompute
Stats recompute: only when dirty (after each update)

Total overhead: ~320 ns per tick
Original VI update: ~1300 ns per tick
New total: ~1600 ns per tick

Acceptable: still well under 5μs budget
```

### Monitoring Drift

Add diagnostic to detect divergence between VI and PG:

```c
double compute_vi_pg_divergence(const OnlineVI *vi, const PG_Update *pg)
{
    /* KL divergence between VI mean and PG posterior */
    double kl = 0;
    int K = vi->K;
    
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < K; j++) {
            double p = vi->mean[i][j] + 1e-10;
            double q = pg->trans[i * K + j] + 1e-10;
            kl += p * log(p / q);
        }
    }
    
    return kl;
}

/* In Lifeboat Protocol */
void log_vi_pg_drift(const OnlineVI *vi, const PG_Update *pg)
{
    double kl = compute_vi_pg_divergence(vi, pg);
    
    if (kl > 0.1) {
        log_warning("VI-PG divergence: KL = %.4f (high)", kl);
    }
    
    /* Track over time for debugging */
    metrics_record("vi_pg_kl", kl);
}
```

---

## 10. Learning Rate Stiffening

### The Problem

The Robbins-Monro schedule guarantees convergence but causes "stiffening":

```
ρ_t = ρ_0 × (τ + t)^{-κ}

Where:
  ρ_0 = 1.0 (base rate)
  τ = 64 (delay)
  κ = 0.7 (decay exponent)

After 10,000 ticks:
  ρ = 1.0 × (64 + 10000)^{-0.7} ≈ 0.0013

After 100,000 ticks:
  ρ = 1.0 × (64 + 100000)^{-0.7} ≈ 0.00025
```

### Consequence

The VI becomes **unresponsive** to new data:

```
Before stiffening:
  New transition observed → α updates significantly → mean shifts

After stiffening:
  New transition observed → α barely changes → mean frozen
  
If market structure changes but triggers don't fire:
  → VI cannot adapt via fast path
  → Model stuck in obsolete state
  → P&L degradation
```

### Current Mitigation: Reset on Lifeboat

```c
void lifeboat_reset_vi(OnlineVI *vi, const PG_Update *update)
{
    online_vi_reset_from_hdp(vi, update->trans, 10.0);
    
    /* Also reset t to restore learning rate */
    vi->t = 0;
    vi->rho = vi->rho_base;
}
```

**Problem:** If triggers never fire, reset never happens.

### Solution A: Heartbeat Trigger

Force a PG sweep every N ticks regardless of health metrics:

```c
typedef struct {
    /* ... existing trigger fields ... */
    
    /* Heartbeat: force PG periodically */
    int heartbeat_interval;      /* e.g., 10000 ticks */
    int ticks_since_last_pg;
    
} PG_Trigger;

bool should_trigger_pg(PG_Trigger *trig, RBPF *rbpf, double log_lik)
{
    trig->ticks_since_last_pg++;
    
    /*─────────────────────────────────────────────────────────────────
     * Heartbeat: force refresh even if everything looks healthy
     *─────────────────────────────────────────────────────────────────*/
    if (trig->ticks_since_last_pg >= trig->heartbeat_interval) {
        trig->ticks_since_last_pg = 0;
        trig->heartbeat_triggers++;
        return true;
    }
    
    /* ... existing health triggers ... */
}
```

### Solution B: Learning Rate Floor

Never let ρ drop below a minimum:

```c
void online_vi_update_learning_rate(OnlineVI *vi)
{
    vi->t++;
    
    /* Robbins-Monro schedule */
    vi->rho = vi->rho_base * pow(vi->rho_tau + (double)vi->t, -vi->rho_kappa);
    
    /* FLOOR: never below minimum */
    if (vi->rho < vi->rho_min) {
        vi->rho = vi->rho_min;
    }
}

/* Default configuration */
void online_vi_init(OnlineVI *vi, int K)
{
    /* ... */
    vi->rho_min = 0.005;  /* Floor at 0.5% learning rate */
    /* ... */
}
```

### Solution C: Responsiveness Monitor

Detect when VI stops responding and force refresh:

```c
typedef struct {
    double xi_history[100][K_MAX][K_MAX];  /* Recent ξ values */
    int history_idx;
    int history_count;
    double responsiveness_threshold;
} ResponsivenessMonitor;

double compute_responsiveness(ResponsivenessMonitor *mon, const OnlineVI *vi)
{
    if (mon->history_count < 10) return 1.0;  /* Not enough data */
    
    /* Compare recent ξ to older ξ */
    int recent_idx = (mon->history_idx - 1 + 100) % 100;
    int old_idx = (mon->history_idx - 10 + 100) % 100;
    
    double diff = 0;
    for (int i = 0; i < vi->K; i++) {
        for (int j = 0; j < vi->K; j++) {
            diff += fabs(mon->xi_history[recent_idx][i][j] - 
                        mon->xi_history[old_idx][i][j]);
        }
    }
    
    return diff;  /* Low diff = VI is frozen */
}

void update_responsiveness(ResponsivenessMonitor *mon, const OnlineVI *vi)
{
    /* Store current ξ */
    for (int i = 0; i < vi->K; i++) {
        for (int j = 0; j < vi->K; j++) {
            mon->xi_history[mon->history_idx][i][j] = vi->last_xi[i][j];
        }
    }
    
    mon->history_idx = (mon->history_idx + 1) % 100;
    if (mon->history_count < 100) mon->history_count++;
}

bool vi_is_frozen(ResponsivenessMonitor *mon, const OnlineVI *vi)
{
    double resp = compute_responsiveness(mon, vi);
    return resp < mon->responsiveness_threshold;
}
```

### Solution D: Adaptive Learning Rate (Safer Version)

Instead of Beam controlling ρ (which causes feedback loops), use a **self-resetting** mechanism:

```c
void online_vi_update_learning_rate_adaptive(OnlineVI *vi)
{
    vi->t++;
    
    /* Base Robbins-Monro */
    double base_rho = vi->rho_base * pow(vi->rho_tau + (double)vi->t, -vi->rho_kappa);
    
    /* If ρ is very low and we haven't had a reset recently,
       gradually increase it back up */
    if (base_rho < vi->rho_min && vi->ticks_since_reset > 1000) {
        /* Slow recovery toward minimum usable rate */
        vi->rho = vi->rho_min + 0.001 * (vi->ticks_since_reset - 1000) / 1000;
        vi->rho = fmin(vi->rho, 0.05);  /* Cap at 5% */
    } else {
        vi->rho = fmax(base_rho, vi->rho_min);
    }
    
    vi->ticks_since_reset++;
}

void online_vi_reset_learning_rate(OnlineVI *vi)
{
    vi->t = 0;
    vi->rho = vi->rho_base;
    vi->ticks_since_reset = 0;
}
```

### Recommended Combination

```c
/* Use all three mitigations */

/* 1. Floor on learning rate */
vi->rho_min = 0.005;

/* 2. Heartbeat trigger */
trig->heartbeat_interval = 10000;  /* Force PG every 10k ticks */

/* 3. Responsiveness monitor as additional trigger */
if (vi_is_frozen(&resp_mon, vi)) {
    trigger_pg();
}
```

---

## 11. Stress Testing Checklist

### Critical Metrics to Monitor

| Weakness | Metric | Warning Threshold | Critical Threshold |
|----------|--------|-------------------|-------------------|
| **PG Mixing** | Variance across ensemble | >0.01 per entry | >0.05 per entry |
| **Trigger Lag** | Time from market shift to Lifeboat | >100 ticks | >500 ticks |
| **Buffer Growth** | Max buffer depth during catch-up | >100 ticks | >200 ticks |
| **Numerical Drift** | KL(VI mean ∥ PG posterior) | >0.05 | >0.2 |
| **Stiffening** | Current ρ value | <0.005 | <0.001 |
| **Responsiveness** | ξ change over 10 ticks | <0.001 | <0.0001 |

### Test Scenarios

```c
typedef struct {
    const char *name;
    const char *description;
    void (*setup)(TestContext *ctx);
    void (*run)(TestContext *ctx);
    void (*validate)(TestContext *ctx, TestResult *result);
} StressTest;

StressTest stress_tests[] = {
    {
        .name = "rapid_regime_switching",
        .description = "Market switches regimes every 50 ticks",
        .setup = setup_rapid_switching,
        .run = run_n_ticks,
        .validate = validate_tracking_accuracy,
    },
    {
        .name = "gradual_drift",
        .description = "Regime parameters drift slowly over 10k ticks",
        .setup = setup_gradual_drift,
        .run = run_n_ticks,
        .validate = validate_adaptation,
    },
    {
        .name = "sudden_structure_change",
        .description = "K changes from 3 to 5 at tick 5000",
        .setup = setup_structure_change,
        .run = run_n_ticks,
        .validate = validate_lifeboat_success,
    },
    {
        .name = "high_frequency_burst",
        .description = "1000 ticks arrive in 50ms during PG sweep",
        .setup = setup_hf_burst,
        .run = run_burst,
        .validate = validate_buffer_handling,
    },
    {
        .name = "dead_zone_scenario",
        .description = "Model slightly wrong but above thresholds",
        .setup = setup_dead_zone,
        .run = run_n_ticks,
        .validate = validate_heartbeat_fires,
    },
    {
        .name = "vi_stiffening",
        .description = "Run 100k ticks without PG, check VI responsiveness",
        .setup = setup_long_run,
        .run = run_n_ticks,
        .validate = validate_vi_not_frozen,
    },
};
```

### Diagnostic Dashboard

```c
typedef struct {
    /* PG Health */
    double pg_ensemble_variance;
    int pg_sweeps_since_stable;
    
    /* Trigger Health */
    double current_ess_ratio;
    double current_loglik_zscore;
    int ticks_since_last_trigger;
    
    /* Buffer Health */
    int current_buffer_depth;
    int max_buffer_depth_session;
    int total_observations_dropped;
    
    /* VI Health */
    double current_rho;
    double vi_responsiveness;
    double vi_pg_kl_divergence;
    
    /* Timing */
    double avg_tick_latency_us;
    double max_tick_latency_us;
    double avg_catchup_latency_us;
    
} DiagnosticsDashboard;

void print_diagnostics(const DiagnosticsDashboard *d)
{
    printf("=== RBPF Diagnostics ===\n");
    printf("PG ensemble variance:     %.4f %s\n", 
           d->pg_ensemble_variance,
           d->pg_ensemble_variance > 0.01 ? "⚠️" : "✓");
    printf("Current ρ:                %.4f %s\n",
           d->current_rho,
           d->current_rho < 0.005 ? "⚠️" : "✓");
    printf("VI responsiveness:        %.4f %s\n",
           d->vi_responsiveness,
           d->vi_responsiveness < 0.001 ? "⚠️" : "✓");
    printf("VI-PG KL divergence:      %.4f %s\n",
           d->vi_pg_kl_divergence,
           d->vi_pg_kl_divergence > 0.1 ? "⚠️" : "✓");
    printf("Buffer depth:             %d/%d %s\n",
           d->current_buffer_depth, BUFFER_CAPACITY,
           d->current_buffer_depth > 100 ? "⚠️" : "✓");
    printf("Ticks since trigger:      %d %s\n",
           d->ticks_since_last_trigger,
           d->ticks_since_last_trigger > 10000 ? "⚠️" : "✓");
    printf("Avg tick latency:         %.1f μs %s\n",
           d->avg_tick_latency_us,
           d->avg_tick_latency_us > 10 ? "⚠️" : "✓");
}
```

---

## 12. Implementation Checklist

### Week 1: Core Safety

| Task | File | Priority |
|------|------|----------|
| Lifeboat inject + catchup | `lifeboat_protocol.c` | 🔴 Critical |
| Validation gate | `pg_validation.c` | 🔴 Critical |
| PG trigger (ESS, log-lik, dominance) | `pg_trigger.c` | 🔴 Critical |
| Label alignment (greedy) | `label_alignment.c` | 🔴 Critical |
| Buffer with cap + drop policy | `circular_buffer.c` | 🔴 Critical |

### Week 2: Stability

| Task | File | Priority |
|------|------|----------|
| Tempering blend | `tempered_transition.c` | 🟡 High |
| Regime anchors | `regime_anchors.c` | 🟡 High |
| Lock-free SPSC channel | `pg_channel.c` | 🟡 High |
| PG ensemble validation | `pg_ensemble.c` | 🟡 High |
| Adaptive thresholds | `adaptive_thresholds.c` | 🟡 High |

### Week 3: Robustness

| Task | File | Priority |
|------|------|----------|
| Digamma for geometric mean | `online_vi_transition.c` | 🟡 High |
| Learning rate floor + heartbeat | `pg_trigger.c` | 🟡 High |
| Time-bounded catch-up | `lifeboat_protocol.c` | 🟡 High |
| VI responsiveness monitor | `responsiveness_monitor.c` | 🟡 High |

### Week 4: Polish

| Task | File | Priority |
|------|------|----------|
| Diagnostics dashboard | `diagnostics.c` | 🟢 Medium |
| Stress test suite | `test_stress.c` | 🟢 Medium |
| CPU affinity configuration | `thread_config.c` | 🟢 Medium |
| Latency monitoring | `latency_monitor.c` | 🟢 Medium |

---

## Summary

### Primary Failure Modes (Structural)

| # | Failure Mode | Cause | Solution |
|---|--------------|-------|----------|
| 1 | **Hot Swap Shock** | Old particles incompatible with new model | Lifeboat Protocol |
| 2 | **Poor MCMC Mixing** | PG returns unconverged sample | Validation + Tempering |
| 3 | **Semantic Drift** | Regime meanings shift over time | Anchors + Alignment |

### Secondary Failure Modes (Operational)

| # | Failure Mode | Cause | Solution |
|---|--------------|-------|----------|
| 4 | **Golden Sample Fallacy** | Single PG sample may be suboptimal | Ensemble validation |
| 5 | **Threshold Dead Zones** | Absolute thresholds don't adapt | Adaptive thresholds + heartbeat |
| 6 | **Fast-Forward Bottleneck** | Buffer grows faster than catch-up | Buffer cap + time-bounded catch-up |
| 7 | **Arithmetic Divergence** | VI uses wrong mean type | Digamma for geometric mean |
| 8 | **Learning Rate Stiffening** | ρ→0 as t→∞ | Floor + heartbeat + responsiveness |

### PG Triggers (Health-Based, Self-Correcting)

| Trigger | Detects | Self-Correcting? |
|---------|---------|------------------|
| ESS collapse | Particle death | ✅ Yes |
| Log-likelihood | Prediction failure | ✅ Yes |
| Regime dominance | Stale structure | ✅ Yes |
| **Heartbeat** | Stiffening/dead zones | ✅ Yes (forced) |
| **Responsiveness** | Frozen VI | ✅ Yes |

### The Lifeboat Protocol

```
Ensemble Check → Validate → Align Labels → Inject CSMC → Jitter → Bounded Catch-Up → Reset VI
```

### Monitoring Checklist

| Metric | Warning | Critical |
|--------|---------|----------|
| PG ensemble variance | >0.01 | >0.05 |
| Buffer depth | >100 | >200 |
| VI-PG KL divergence | >0.05 | >0.2 |
| Learning rate ρ | <0.005 | <0.001 |
| VI responsiveness | <0.001 | <0.0001 |
| Ticks since trigger | >5000 | >10000 |

---

## References

1. Andrieu, C., Doucet, A., & Holenstein, R. (2010). *Particle Markov Chain Monte Carlo Methods*. JRSS-B.
2. Lindsten, F., Jordan, M. I., & Schön, T. B. (2014). *Particle Gibbs with Ancestor Sampling*. JMLR.
3. Chopin, N., & Singh, S. S. (2015). *On Particle Gibbs Sampling*. Bernoulli.
