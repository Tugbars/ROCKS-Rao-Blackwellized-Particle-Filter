# MMPF Adaptive Architecture Roadmap

## The Vision: From Heuristics to Self-Driving

Transform MMPF from a **statically-tuned filter** into a **self-calibrating engine** that:
- Tracks volatility shocks instantly (not with lag)
- Discovers regime boundaries automatically (not hard-coded)
- Adapts to any asset without manual tuning
- Recovers gracefully from false positives

---

## Table of Contents

1. [The Problem Statement](#1-the-problem-statement)
2. [Component Overview](#2-component-overview)
3. [Component 1: Silverman's Rule](#3-silvermans-rule)
4. [Component 2: SPRT Detection](#4-sprt-detection)
5. [Component 3: MCMC Move](#5-mcmc-move)
6. [Component 4: Online EM](#6-online-em)
7. [Component 5: Entropy Lock](#7-entropy-lock)
8. [Integration Architecture](#8-integration-architecture)
9. [Implementation Order](#9-implementation-order)
10. [Validation Strategy](#10-validation-strategy)
11. [Risk Analysis](#11-risk-analysis)

---

## 1. The Problem Statement

### Current State: Death by Heuristics

The current MMPF relies on **magic numbers** that work for specific assets in specific time periods:

| Heuristic | Current Value | Problem |
|-----------|---------------|---------|
| Swim Lanes | `Calm=[-5.5, -3.5]` | Fails when asset volatility profile changes |
| Shock Response | `noise *= 50` | "Spray and pray" — may overshoot or undershoot |
| Recovery Timer | `lockout = 20 ticks` | Time ≠ Information; crashes settle in 2 ticks, regime shifts take 50 |
| Jitter Scale | `0.5 * std(weights)` | Arbitrary; causes particle collapse in edge cases |
| Regime Detection | Counter + hysteresis | Ad-hoc; no statistical foundation |

### Evidence: Stage 1 Tuner Results

```
Per-Scenario Accuracy:
  Flash:         94.1%  ✓ (transient, lands in swim lanes)
  Choppy:        95.8%  ✓ (transient, lands in swim lanes)
  Trend:         76.7%  ~ (borderline)
  Calm:          26.0%  ✗ (sustained, swim lane mismatch)
  Recovery:       6.3%  ✗ (sustained, swim lane mismatch)
  Crisis:         0.3%  ✗ (sustained, OUTSIDE swim lanes)
  CrisisPersist:  0.0%  ✗ (sustained, OUTSIDE swim lanes)
```

**Diagnosis**: MMPF excels at transient events but fails at sustained regimes because:
1. Particles are **imprisoned** in swim lanes
2. Shock response is **too slow** (shark fin lag)
3. No mechanism to **discover** where regimes actually live

### The Goal

| Metric | Current | Target |
|--------|---------|--------|
| Crisis Accuracy | 0.3% | >80% |
| Shock Tracking Lag | 15-25 ticks | 1-2 ticks |
| Asset Retuning | Manual, monthly | Automatic, continuous |
| False Positive Recovery | 20+ ticks | 3-5 ticks |

---

## 2. Component Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        DATA FLOW (Per Tick)                             │
└─────────────────────────────────────────────────────────────────────────┘

  y_t (observation) arrives
         │
         ▼
  ┌──────────────────┐
  │   BOCPD / SPRT   │  ◄── Component 2: Principled shock detection
  │   Is this a      │
  │   changepoint?   │
  └────────┬─────────┘
           │
     ┌─────┴─────┐
     │ Yes       │ No
     ▼           │
  ┌──────────────────┐    │
  │   MCMC Move      │    │  ◄── Component 3: Particle teleportation
  │   • Teleport μ   │    │
  │   • Reset var    │    │
  │   • Flatten Π    │    │
  └────────┬─────────┘    │
           │              │
           ▼              ▼
  ┌──────────────────────────────────────┐
  │        Standard MMPF Step            │
  │   1. Predict (model physics)         │
  │   2. Update (likelihood weights)     │
  │   3. Resample + Silverman Jitter     │  ◄── Component 1: Density-based jitter
  │   4. IMM (model probabilities)       │
  └────────┬─────────────────────────────┘
           │
           ▼
  ┌──────────────────┐
  │   Online EM      │  ◄── Component 4: Regime discovery
  │   Update μ_k     │
  │   for each       │
  │   cluster        │
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │   Entropy Lock   │  ◄── Component 5: Stability detection
  │   Restore Π if   │
  │   particles      │
  │   converged      │
  └──────────────────┘
```

### Component Summary

| # | Component | Replaces | Key Insight |
|---|-----------|----------|-------------|
| 1 | **Silverman's Rule** | Fixed jitter scalar | Jitter should scale with local particle density |
| 2 | **SPRT Detection** | Counter/hysteresis | Use Wald's sequential test for statistical rigor |
| 3 | **MCMC Move** | Blind noise boost | Climb likelihood gradient, don't spray randomly |
| 4 | **Online EM** | Static swim lanes | Discover regime centers from data |
| 5 | **Entropy Lock** | Fixed timer | Unlock when information stabilizes, not after N ticks |

---

## 3. Silverman's Rule

### Location
`rbpf_ksc.c` → resampling/jitter section

### The Problem

After resampling, particles cluster at high-weight locations. Without jitter, they collapse to identical values (degeneracy). Current jitter:

```c
jitter = 0.5 * std(weights) * randn();  // Arbitrary scale
```

**Failure modes:**
- Too small → particle collapse, filter death
- Too large → particles scattered, accuracy loss
- Fixed scale → can't adapt to varying particle density

### The Solution: Silverman's Rule of Thumb

From kernel density estimation theory, optimal bandwidth for Gaussian kernel:

```
h = 0.9 × min(σ, IQR/1.34) × N^(-1/5)
```

Where:
- σ = standard deviation of particle states
- IQR = interquartile range (robust to outliers)
- N = number of particles

**Intuition**: Jitter should be proportional to the **spread** of particles and inversely proportional to the **count**. Dense regions get small jitter; sparse regions get large jitter.

### Implementation Sketch

```c
double silverman_bandwidth(const double *x, int n) {
    /* Compute std */
    double mean = 0.0, var = 0.0;
    for (int i = 0; i < n; i++) mean += x[i];
    mean /= n;
    for (int i = 0; i < n; i++) var += (x[i] - mean) * (x[i] - mean);
    double sigma = sqrt(var / (n - 1));
    
    /* Compute IQR (need sorted copy or approximate) */
    /* For speed, use P² quantile estimators or approximate */
    double iqr = approx_iqr(x, n);
    
    /* Silverman's rule */
    double scale = fmin(sigma, iqr / 1.34);
    double h = 0.9 * scale * pow((double)n, -0.2);
    
    return h;
}

/* In resampling: */
double h = silverman_bandwidth(rbpf->mu, n_particles);
for (int i = 0; i < n_particles; i++) {
    rbpf->mu[i] += h * randn();
}
```

### What It Prevents
- Particle collapse after resampling
- Over-smoothing in concentrated distributions
- Under-smoothing in spread distributions

### Interaction with MCMC
After MCMC teleports particles, they're clustered at the shock level. Silverman ensures appropriate spread without destroying the location information.

---

## 4. SPRT Detection

### Location
New file: `rbpf_sprt.c` / `rbpf_sprt.h`

### The Problem

Current regime switch detection uses ad-hoc counters:

```c
if (crisis_weight > 0.6) crisis_counter++;
if (crisis_counter > 5) switch_to_crisis();
```

**Problems:**
- Threshold (0.6) is arbitrary
- Counter length (5) is arbitrary  
- No statistical foundation
- Can't quantify confidence

### The Solution: Wald's Sequential Probability Ratio Test

SPRT is the **optimal** sequential test — it minimizes expected samples to reach a decision at given error rates.

**Setup:**
- H₀: We're in regime A (e.g., Calm)
- H₁: We're in regime B (e.g., Crisis)
- At each tick, compute log-likelihood ratio:

```
Λ_t = Λ_{t-1} + log(P(y_t | H₁) / P(y_t | H₀))
```

**Decision:**
- If Λ_t > A: Accept H₁ (switch to Crisis)
- If Λ_t < B: Accept H₀ (stay in Calm)  
- Otherwise: Continue sampling

Where A and B are derived from desired error rates (α, β):
```
A ≈ log((1-β)/α)
B ≈ log(β/(1-α))
```

### Implementation Sketch

```c
typedef struct {
    double log_ratio;       /* Cumulative log-likelihood ratio */
    double threshold_high;  /* A: Accept H1 */
    double threshold_low;   /* B: Accept H0 */
    int current_hypothesis; /* 0 = H0, 1 = H1 */
} SPRT_Detector;

void sprt_init(SPRT_Detector *s, double alpha, double beta) {
    s->log_ratio = 0.0;
    s->threshold_high = log((1.0 - beta) / alpha);
    s->threshold_low = log(beta / (1.0 - alpha));
    s->current_hypothesis = 0;
}

int sprt_update(SPRT_Detector *s, double ll_h1, double ll_h0) {
    s->log_ratio += ll_h1 - ll_h0;
    
    if (s->log_ratio > s->threshold_high) {
        s->log_ratio = 0.0;  /* Reset for next test */
        s->current_hypothesis = 1;
        return 1;  /* Switch to H1 */
    }
    
    if (s->log_ratio < s->threshold_low) {
        s->log_ratio = 0.0;
        s->current_hypothesis = 0;
        return -1; /* Confirm H0 */
    }
    
    return 0; /* Continue */
}
```

### What It Prevents
- Arbitrary threshold tuning
- Premature switches on noise
- Delayed switches on real regime changes

### Relationship to BOCPD

| Aspect | BOCPD | SPRT |
|--------|-------|------|
| Model | Full Bayesian posterior | Frequentist likelihood ratio |
| Output | P(changepoint) | Binary decision |
| Memory | O(max_run) | O(1) |
| Use case | "How likely is a changepoint?" | "Should I switch NOW?" |

**Recommendation**: Use BOCPD for shock detection (MCMC trigger) and SPRT for regime labeling (Calm↔Crisis switching). They serve different purposes.

---

## 5. MCMC Move

### Location
New file: `mmpf_mcmc.c` / `mmpf_mcmc.h`

### The Problem: The Shark Fin

When a flash crash occurs, true volatility teleports from 1% to 80% instantly. But particles **walk** via state transitions:

```
h_{t+1} = μ + φ(h_t - μ) + η
```

Even with φ=0.99, walking from h=-5 to h=0 takes ~50 ticks. This creates the "shark fin" lag:

```
True Vol:     ─────┐
                   │████████████████████
                   │
Estimated:    ─────────╱
                      ╱  ← "Shark fin" ramp
                     ╱     (RMSE disaster)
                    ╱
              ─────╱
```

### The Solution: Particle Teleportation

When BOCPD fires, pause the filter and run Metropolis-Hastings:

```
For each particle:
    For k = 1 to MCMC_STEPS:
        h_proposal = h_current + noise
        
        α = P(y_t | h_proposal) / P(y_t | h_current)
        
        if rand() < α:
            h_current = h_proposal
```

Particles **climb the likelihood gradient** and land at the new truth:

```
True Vol:     ─────┐
                   │████████████████████
                   │
Estimated:    ─────┤  ← Instant tracking
                   │     (RMSE minimal)
              ─────┘
```

### Key Design Decisions

**1. What state is moved?**
- `rbpf->mu[i]` — the Kalman filter mean (log-volatility estimate)

**2. Are swim lanes enforced?**
- **NO**. During MCMC, particles are free to explore all of volatility space. The data is the only truth.

**3. What about variance?**
- Reset to 1.0 (high uncertainty). Prevents overconfidence after teleport.

**4. What about transition matrix?**
- Flattened to uniform. All regime switches equally likely during shock.

**5. Won't all models converge to the same place?**
- Yes, temporarily. But the **predict step** immediately separates them:
  - Calm (φ=0.9, μ=-5): Pulls hard toward low vol
  - Crisis (φ=0.99, μ=-2): Stays at high vol
  
  The model that matches subsequent data wins.

### Implementation Sketch (AVX2 + MKL)

```c
void mmpf_inject_shock_mcmc(MMPF_ROCKS *mmpf, double y_log_sq) {
    /* 1. Generate ALL random numbers upfront (MKL batch) */
    int total_rng = N_PARTICLES * MCMC_STEPS * N_MODELS;
    vdRngGaussian(stream, total_rng, rng_gauss, 0.0, STEP_SIZE);
    vdRngUniform(stream, total_rng, rng_uniform, 0.0, 1.0);
    vdLn(total_rng, rng_uniform, rng_uniform);  /* Pre-compute log for acceptance */
    
    /* 2. MCMC for each model's particles */
    for (int k = 0; k < N_MODELS; k++) {
        double *mu = mmpf->ext[k]->rbpf->mu;
        
        /* AVX2: Process 4 particles at a time */
        for (int i = 0; i < N_PARTICLES; i += 4) {
            __m256d x = _mm256_loadu_pd(&mu[i]);
            __m256d score = likelihood_avx(x, y_log_sq);
            
            for (int step = 0; step < MCMC_STEPS; step++) {
                __m256d noise = _mm256_loadu_pd(&rng_gauss[idx]);
                __m256d log_u = _mm256_loadu_pd(&rng_uniform[idx]);
                
                __m256d x_prop = _mm256_add_pd(x, noise);
                __m256d score_prop = likelihood_avx(x_prop, y_log_sq);
                
                __m256d log_alpha = _mm256_sub_pd(score_prop, score);
                __m256d mask = _mm256_cmp_pd(log_alpha, log_u, _CMP_GE_OQ);
                
                x = _mm256_blendv_pd(x, x_prop, mask);
                score = _mm256_blendv_pd(score, score_prop, mask);
            }
            
            _mm256_storeu_pd(&mu[i], x);
        }
        
        /* Reset variances */
        for (int i = 0; i < N_PARTICLES; i++) {
            mmpf->ext[k]->rbpf->var[i] = 1.0;
        }
    }
    
    /* 3. Flatten transition matrix */
    for (int i = 0; i < N_MODELS; i++) {
        for (int j = 0; j < N_MODELS; j++) {
            mmpf->saved_transition[i][j] = mmpf->transition[i][j];
            mmpf->transition[i][j] = 1.0 / N_MODELS;
        }
    }
    
    mmpf->shock_active = 1;
}
```

### What It Prevents
- Shark fin lag (the #1 RMSE contributor)
- Particle death on extreme shocks
- Slow regime switching

### The "Double-Counting" Defense

**Concern**: MCMC uses y_t, then Update uses y_t again. Isn't this data incest?

**Defense**: In a structural break, the prior P(x_t | x_{t-1}) is **invalid**. The market teleported; the bridge from the past is broken. MCMC is not "double-counting" — it's **re-initializing** based on likelihood alone, since the prior is useless.

The variance reset to 1.0 prevents overconfidence. We know the location but not the precision.

---

## 6. Online EM

### Location
New file: `mmpf_online_em.c` / `mmpf_online_em.h`

### The Problem: Static Swim Lanes

Current regime definitions are hard-coded:

```c
cfg->swim_lanes[MMPF_CALM].mu_vol_min = -5.5;
cfg->swim_lanes[MMPF_CALM].mu_vol_max = -3.5;
cfg->swim_lanes[MMPF_CRISIS].mu_vol_min = -2.5;
cfg->swim_lanes[MMPF_CRISIS].mu_vol_max = 0.0;
```

**Problems:**
- Bitcoin 2020 ≠ Bitcoin 2024
- Eurodollar ≠ Crude Oil
- Requires manual retuning every few months

### The Solution: Gaussian Mixture Model with Online EM

Model log-volatility as a mixture of 3 Gaussians:

```
P(y) = π₀ N(y|μ₀,σ₀²) + π₁ N(y|μ₁,σ₁²) + π₂ N(y|μ₂,σ₂²)
       └── Calm ──┘    └── Trend ──┘    └── Crisis ──┘
```

Online EM learns {μ_k, σ_k², π_k} from streaming data:

**E-Step**: Compute responsibility (which cluster owns this observation?)
```
γ_k ∝ π_k × N(y | μ_k, σ_k²)
```

**M-Step**: Update parameters via stochastic approximation
```
μ_k ← μ_k + η × γ_k × (y - μ_k) / π_k
σ_k² ← (1-η) × σ_k² + η × γ_k × (y - μ_k)²
π_k ← (1-η) × π_k + η × γ_k
```

### The Ordering Constraint

EM doesn't care about labels. Cluster 0 might become "Crisis" if it drifts upward. Fix by enforcing:

```
μ₀ < μ₁ < μ₂
```

After each update, bubble-sort the clusters. This ensures:
- Cluster 0 → Always lowest μ → Assigned to CALM
- Cluster 2 → Always highest μ → Assigned to CRISIS

### Implementation Sketch

```c
typedef struct {
    double mu[3];      /* Cluster centers */
    double var[3];     /* Cluster variances */
    double pi[3];      /* Cluster weights (prior probabilities) */
    double eta;        /* Learning rate */
} MMPF_OnlineEM;

void mmpf_online_em_update(MMPF_OnlineEM *em, double y) {
    double gamma[3], sum = 0.0;
    
    /* E-Step: Responsibilities */
    for (int k = 0; k < 3; k++) {
        gamma[k] = em->pi[k] * gaussian_pdf(y, em->mu[k], em->var[k]);
        sum += gamma[k];
    }
    for (int k = 0; k < 3; k++) gamma[k] /= sum;
    
    /* M-Step: Parameter updates */
    for (int k = 0; k < 3; k++) {
        double eta_k = em->eta * gamma[k] / (em->pi[k] + 1e-10);
        
        em->pi[k] = (1 - em->eta) * em->pi[k] + em->eta * gamma[k];
        em->pi[k] = fmax(em->pi[k], 0.05);  /* Floor: prevent cluster death */
        
        double delta = y - em->mu[k];
        em->mu[k] += eta_k * delta;
        em->var[k] = (1 - eta_k) * em->var[k] + eta_k * delta * delta;
        em->var[k] = fmax(em->var[k], 0.1);  /* Floor: prevent singularity */
    }
    
    /* Enforce ordering: μ₀ < μ₁ < μ₂ */
    bubble_sort_clusters(em);
}
```

### Integration with MMPF

After EM update, feed learned centers into hypothesis configuration:

```c
/* In mmpf_step(), after EM update: */
mmpf->config.hypotheses[MMPF_CALM].mu_vol = mmpf->online_em.mu[0];
mmpf->config.hypotheses[MMPF_TREND].mu_vol = mmpf->online_em.mu[1];
mmpf->config.hypotheses[MMPF_CRISIS].mu_vol = mmpf->online_em.mu[2];
```

### What It Prevents
- Manual per-asset tuning
- Drift-induced accuracy decay
- Secular regime changes going unnoticed

### Cold Start

Initialize with tuned values from Stage 1:

```c
void mmpf_online_em_init(MMPF_OnlineEM *em) {
    em->mu[0] = -4.5;   /* Calm (from tuner) */
    em->mu[1] = -3.0;   /* Trend */
    em->mu[2] = -1.25;  /* Crisis (from tuner) */
    
    em->var[0] = em->var[1] = em->var[2] = 1.0;
    em->pi[0] = 0.5; em->pi[1] = 0.3; em->pi[2] = 0.2;
    
    em->eta = 0.001;  /* ~1000 tick memory */
}
```

This ensures reasonable behavior from tick 1, while allowing adaptation over time.

---

## 7. Entropy Lock

### Location
New file: `mmpf_entropy.c` / `mmpf_entropy.h`

### The Problem: Fixed Recovery Timer

After shock injection, we currently wait a fixed number of ticks:

```c
if (ticks_since_shock > 20) {
    restore_transition_matrix();
}
```

**Problems:**
- Flash crashes settle in 2-3 ticks; waiting 20 wastes accuracy
- True regime shifts need 50+ ticks; unlocking at 20 causes flicker
- Time ≠ Information

### The Solution: Entropy-Based Stability Detection

Shannon entropy measures disorder in the particle distribution:

```
H = -Σ wᵢ log(wᵢ)
```

**High entropy**: Particles disagree; filter is uncertain → Keep transitions uniform
**Low entropy**: Particles converged; filter is confident → Restore sticky transitions

Normalized entropy (H / log(N)) gives a 0-1 scale:
- H_norm ≈ 1.0: Maximum uncertainty (uniform distribution)
- H_norm ≈ 0.0: Maximum certainty (one particle dominates)

### The Stability Criterion

Track **change** in entropy, not absolute value:

```c
delta_H = |H_current - H_previous|
delta_H_ema = 0.3 * delta_H + 0.7 * delta_H_ema_prev

if (delta_H_ema < threshold) {
    /* Information has stabilized; restore transitions */
}
```

**Intuition**: After a shock, entropy fluctuates wildly as particles compete. When fluctuation dies down, we've reached equilibrium.

### Implementation Sketch

```c
double mmpf_calculate_entropy(MMPF_ROCKS *mmpf) {
    double H = 0.0;
    int N = 0;
    
    for (int k = 0; k < N_MODELS; k++) {
        double model_w = mmpf->weights[k];
        for (int i = 0; i < mmpf->n_particles; i++) {
            double w = model_w * mmpf->ext[k]->rbpf->w_norm[i];
            w = fmax(w, 1e-12);  /* Avoid log(0) */
            H -= w * log(w);
            N++;
        }
    }
    
    return H / log(N);  /* Normalized to [0, 1] */
}

int mmpf_check_stability(MMPF_ROCKS *mmpf) {
    double H = mmpf_calculate_entropy(mmpf);
    double delta = fabs(H - mmpf->entropy_prev);
    
    mmpf->entropy_ema = 0.3 * delta + 0.7 * mmpf->entropy_ema;
    mmpf->entropy_prev = H;
    mmpf->ticks_since_shock++;
    
    /* Safety bounds */
    if (mmpf->ticks_since_shock < MIN_SHOCK_DURATION) return 0;
    if (mmpf->ticks_since_shock > MAX_SHOCK_DURATION) return 1;
    
    /* Thermodynamic criterion */
    return (mmpf->entropy_ema < STABILITY_THRESHOLD);
}
```

### What It Prevents
- Premature unlock (flicker on regime shifts)
- Delayed unlock (accuracy loss on flash crashes)
- Arbitrary timer tuning

### Alternative: Two-Level Entropy

Instead of combining all weights, track separately:

1. **Model entropy**: Uncertainty about which regime
   ```c
   H_model = -Σ model_weight[k] * log(model_weight[k])  /* Just 3 terms */
   ```

2. **Particle entropy**: Uncertainty within each model
   ```c
   H_particle[k] = -Σ w_norm[i] * log(w_norm[i])
   ```

Unlock when **both** stabilize. This prevents edge cases where models are certain but particles are scattered (or vice versa).

---

## 8. Integration Architecture

### Modified MMPF_ROCKS Struct

```c
typedef struct MMPF_ROCKS {
    /* === Existing Fields === */
    MMPF_Config config;
    MMPF_Extended *ext[MMPF_N_MODELS];
    rbpf_real_t weights[MMPF_N_MODELS];
    rbpf_real_t transition[MMPF_N_MODELS][MMPF_N_MODELS];
    /* ... */
    
    /* === New: Adaptive Components === */
    
    /* Online EM: Regime Discovery */
    MMPF_OnlineEM online_em;
    
    /* Entropy Lock: Stability Detection */
    double entropy_prev;
    double entropy_ema;
    int ticks_since_shock;
    double stability_threshold;
    int min_shock_duration;
    int max_shock_duration;
    
    /* Shock State */
    int shock_active;
    rbpf_real_t saved_transition[MMPF_N_MODELS][MMPF_N_MODELS];
    
    /* MCMC Scratch (avoid malloc in hot path) */
    double *mcmc_rng_gauss;
    double *mcmc_rng_uniform;
    int mcmc_rng_capacity;
    
} MMPF_ROCKS;
```

### Modified mmpf_step() Flow

```c
void mmpf_step(MMPF_ROCKS *mmpf, rbpf_real_t y, MMPF_Output *out) {
    
    /* 1. Check for shock (BOCPD runs externally, passes flag) */
    if (mmpf->pending_shock) {
        double y_log_sq = (fabs(y) > 1e-10) ? log(y * y) : -20.0;
        mmpf_inject_shock_mcmc(mmpf, y_log_sq);
        mmpf->pending_shock = 0;
    }
    
    /* 2. Standard MMPF update */
    for (int k = 0; k < MMPF_N_MODELS; k++) {
        /* Predict */
        mmpf_predict_model(mmpf, k);
        
        /* Update (likelihood weighting) */
        mmpf_update_model(mmpf, k, y);
        
        /* Resample with Silverman jitter */
        double bandwidth = silverman_bandwidth(mmpf->ext[k]->rbpf->mu, 
                                                mmpf->n_particles);
        mmpf_resample_model(mmpf, k, bandwidth);
    }
    
    /* 3. IMM update (model probabilities) */
    mmpf_imm_update(mmpf);
    
    /* 4. Online EM update (regime discovery) */
    double y_log_vol = mmpf_get_weighted_log_vol(mmpf);
    mmpf_online_em_update(&mmpf->online_em, y_log_vol);
    
    /* Apply learned centers to hypotheses */
    for (int k = 0; k < MMPF_N_MODELS; k++) {
        mmpf->config.hypotheses[k].mu_vol = mmpf->online_em.mu[k];
    }
    
    /* 5. Check recovery (if in shock state) */
    if (mmpf->shock_active && mmpf_check_stability(mmpf)) {
        mmpf_restore_from_shock(mmpf);
    }
    
    /* 6. Populate output */
    mmpf_populate_output(mmpf, out);
}
```

---

## 9. Implementation Order

```
Phase 1: Foundation (rbpf_ksc.c)
├── 1.1 Silverman's Rule
│   └── Replace fixed jitter with density-based bandwidth
└── 1.2 SPRT Detector
    └── New file: rbpf_sprt.c

Phase 2: The Big Win (mmpf_mcmc.c)
├── 2.1 MCMC Move implementation
│   ├── AVX2 likelihood evaluation
│   ├── MKL batch RNG
│   └── Branchless accept/reject
└── 2.2 Integration with mmpf_inject_shock()

Phase 3: Validation
├── 3.1 Test Silverman + MCMC + existing BOCPD
├── 3.2 Run Stage 1 scenarios
│   └── Target: Crisis accuracy 0% → 80%+
└── 3.3 Benchmark latency impact

Phase 4: Regime Discovery (mmpf_online_em.c)
├── 4.1 Online EM implementation
├── 4.2 Ordering constraint (bubble sort)
├── 4.3 Integration with hypothesis config
└── 4.4 Cold start with tuned values

Phase 5: Smart Recovery (mmpf_entropy.c)
├── 5.1 Entropy calculation
├── 5.2 Stability detection
└── 5.3 Replace fixed timer lockout

Phase 6: Full Integration
├── 6.1 End-to-end testing
├── 6.2 Multi-asset validation (BTC, ETH, ES, CL)
└── 6.3 Performance optimization
```

### Dependency Graph

```
Silverman ──────────────────────────────┐
     │                                  │
     │ (foundation for                  │ (jitter after
     │  particle health)                │  MCMC teleport)
     ▼                                  │
   SPRT ─────────┐                      │
     │           │                      │
     │ (optional │                      │
     │  trigger) │                      │
     ▼           ▼                      │
  BOCPD ──────► MCMC Move ◄─────────────┘
                 │
                 │ (reduces dependence
                 │  on swim lanes)
                 ▼
            Online EM ──────────────────┐
                 │                      │
                 │ (slower adaptation,  │
                 │  long-term)          │
                 ▼                      │
          Entropy Lock ◄────────────────┘
                 │     (stability depends on
                 │      converged EM)
                 ▼
            Full Stack
```

---

## 10. Validation Strategy

### Per-Component Tests

| Component | Test | Success Criterion |
|-----------|------|-------------------|
| Silverman | Particle variance after resample | σ² ≈ Silverman bandwidth² |
| SPRT | Synthetic regime switches | Detection within 5 ticks |
| MCMC | Flash crash scenario | Lag < 3 ticks (was 15-25) |
| Online EM | Drifting asset simulation | Centers track within 100 ticks |
| Entropy | Flash vs regime shift | Different unlock times |

### Integration Tests

| Scenario | Current | Target |
|----------|---------|--------|
| Flash (transient) | 94% | >95% |
| Choppy (transient) | 96% | >95% |
| Crisis (sustained) | 0.3% | >80% |
| CrisisPersist | 0% | >80% |
| Calm (sustained) | 26% | >70% |

### Multi-Asset Validation

Test on 4 assets without re-tuning:
1. **BTC/USD** — Crypto, high vol
2. **ETH/USD** — Crypto, correlated with BTC
3. **ES** (S&P 500 futures) — Equity, moderate vol
4. **CL** (Crude Oil) — Commodity, different dynamics

**Success**: Same code, same hyperparameters, all assets >70% accuracy.

### Latency Budget

| Operation | Budget | Notes |
|-----------|--------|-------|
| Normal tick | <200 μs | Current baseline |
| MCMC shock | <500 μs | 5 steps × 768 particles |
| Online EM | <10 μs | O(1) per tick |
| Entropy | <20 μs | Sum over all particles |

---

## 11. Risk Analysis

### Risk 1: Feedback Loop Instability

**Concern**: Multiple adaptive components might fight each other.

**Mitigation**:
- MCMC is triggered rarely (only on BOCPD fire)
- Online EM adapts slowly (η = 0.001)
- Entropy lock has safety bounds (min/max duration)
- Test each component in isolation first

### Risk 2: Cold Start Degradation

**Concern**: Online EM needs warmup; early ticks might be worse.

**Mitigation**:
- Initialize with Stage 1 tuned values
- EM adapts slowly; initial values dominate for ~1000 ticks
- Silverman and MCMC don't need warmup

### Risk 3: MCMC Computational Cost

**Concern**: 5 steps × 768 particles × likelihood evaluation

**Mitigation**:
- AVX2 processes 4 particles simultaneously
- MKL batch RNG (one call for all random numbers)
- Pre-computed log(uniform) for acceptance
- Budget: <500 μs, acceptable for shock response

### Risk 4: False Positive Cascade

**Concern**: BOCPD false positive → MCMC teleport → filter disrupted

**Mitigation**:
- Variance reset to 1.0 gives high Kalman gain
- Filter snaps back on next normal tick
- Cost: ~3-5 ticks of wide confidence intervals
- Entropy lock prevents premature re-lock

### Risk 5: Online EM Cluster Death

**Concern**: If π_k → 0, cluster never recovers

**Mitigation**:
- Floor: `pi[k] = fmax(pi[k], 0.05)`
- Ensures all 3 clusters remain viable
- Trade-off: Slight bias toward uniform prior

---

## Summary

| Component | Impact | Complexity | Priority |
|-----------|--------|------------|----------|
| **MCMC Move** | 🔴 Critical | High | 1 |
| **Silverman** | 🟡 Important | Low | 1 |
| **Online EM** | 🟡 Important | Medium | 2 |
| **Entropy Lock** | 🟢 Nice-to-have | Medium | 3 |
| **SPRT** | 🟢 Nice-to-have | Medium | 3 |

**The Killer Feature**: MCMC Move eliminates the shark fin lag, which is the single largest source of RMSE. Everything else is polish.

**The Force Multiplier**: Online EM removes the need for per-asset tuning, enabling true asset-agnostic deployment.

---

## Appendix: File Structure

```
RBPF/
├── rao-blackwellized/
│   └── rbpf_ksc.c          # Modified: Add Silverman's Rule
├── mmpf/
│   ├── mmpf_core.c         # Modified: Integrate new components
│   ├── mmpf_api.c          # Modified: New API functions
│   ├── mmpf_rocks.h        # Modified: New struct fields
│   ├── mmpf_internal.h     # Modified: New internal types
│   ├── mmpf_mcmc.c         # NEW: MCMC move implementation
│   ├── mmpf_online_em.c    # NEW: Online EM implementation
│   └── mmpf_entropy.c      # NEW: Entropy lock implementation
├── sprt/
│   ├── rbpf_sprt.c         # NEW: SPRT detector
│   └── rbpf_sprt.h         # NEW: SPRT header
└── test/
    ├── test_silverman.c    # NEW: Silverman validation
    ├── test_mcmc.c         # NEW: MCMC validation
    ├── test_online_em.c    # NEW: Online EM validation
    └── test_full_stack.c   # NEW: Integration test
```
