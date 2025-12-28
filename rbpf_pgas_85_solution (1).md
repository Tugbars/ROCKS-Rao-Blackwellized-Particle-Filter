# RBPF + PGAS Coordination: The 85% Solution

## Overview

A simplified but principled approach to coordinating RBPF with PGAS for online volatility estimation. This design prioritizes:

- **Simplicity**: Fixed update interval, no complex triggers
- **Correctness**: SAEM for proper Bayesian blending
- **Adaptivity**: Thompson sampling for explore/exploit

```
┌─────────────────────────────────────────────────────────────────┐
│                        ARCHITECTURE                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Every tick:                                                   │
│   ┌─────────┐                                                   │
│   │  RBPF   │──► h_t, s_t (latent state estimate)              │
│   └────┬────┘                                                   │
│        │                                                        │
│        │ stores y_t                                             │
│        ▼                                                        │
│   ┌─────────┐                                                   │
│   │ Buffer  │ (ring buffer, last 2000 ticks)                   │
│   └────┬────┘                                                   │
│        │                                                        │
│        │ every 500 ticks                                        │
│        ▼                                                        │
│   ┌─────────┐                                                   │
│   │  PGAS   │──► transition counts S[i,j]                      │
│   └────┬────┘                                                   │
│        │                                                        │
│        ▼                                                        │
│   ┌─────────┐                                                   │
│   │  SAEM   │──► Q = (1-γ)Q + γS (sufficient stats)           │
│   └────┬────┘                                                   │
│        │                                                        │
│        ▼                                                        │
│   ┌─────────┐                                                   │
│   │Thompson │──► Π (sample or mean, based on confidence)       │
│   └────┬────┘                                                   │
│        │                                                        │
│        │ update                                                 │
│        ▼                                                        │
│   ┌─────────┐                                                   │
│   │  RBPF   │ (uses new Π)                                     │
│   └─────────┘                                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Why This Design

### What We Skip (Complexity)

| Feature | Full Solution | 85% Solution | Why Skip |
|---------|---------------|--------------|----------|
| Dual-gate triggers | Hawkes AND KL | Fixed interval | Simpler, debuggable |
| Scout sweep | PARIS pre-validation | None | Adds latency |
| Confidence-based γ | Adaptive | Fixed γ=0.3 | One less thing to tune |
| Three-tier reset | Normal/Partial/Full | None | Add if gets stuck |
| Tempered injection | 5% random flips | None | Add if confirmation bias |

### What We Keep (Value)

| Feature | Why Keep |
|---------|----------|
| SAEM blending | Proper Bayesian, constraints automatic |
| Thompson sampling | Principled explore/exploit |
| Sufficient statistics | No label switching issues |
| Ring buffer | Clean, bounded memory |

---

## Components

### 1. Ring Buffer

Stores last N observations for PGAS window.

```c
typedef struct {
    float *data;
    int capacity;
    int count;
    int head;
} RingBuffer;

void ring_buffer_init(RingBuffer *rb, int capacity) {
    rb->data = (float *)aligned_alloc(64, capacity * sizeof(float));
    rb->capacity = capacity;
    rb->count = 0;
    rb->head = 0;
}

void ring_buffer_push(RingBuffer *rb, float y) {
    rb->data[rb->head] = y;
    rb->head = (rb->head + 1) % rb->capacity;
    if (rb->count < rb->capacity) rb->count++;
}

/* Copy to contiguous array for PGAS */
void ring_buffer_copy_to(const RingBuffer *rb, float *out, int *len) {
    *len = rb->count;
    if (rb->count < rb->capacity) {
        /* Not wrapped yet */
        memcpy(out, rb->data, rb->count * sizeof(float));
    } else {
        /* Wrapped: copy tail then head */
        int tail_len = rb->capacity - rb->head;
        memcpy(out, rb->data + rb->head, tail_len * sizeof(float));
        memcpy(out + tail_len, rb->data, rb->head * sizeof(float));
    }
}

void ring_buffer_free(RingBuffer *rb) {
    free(rb->data);
}
```

### 2. SAEM Blender

Blends PGAS transition counts into sufficient statistics.

```c
#define K 4  /* Number of regimes */

typedef struct {
    float Q[K * K];         /* Sufficient statistics (pseudo-counts) */
    float alpha_prior;      /* Dirichlet prior per cell */
    float kappa_prior;      /* Extra prior on diagonal (stickiness) */
    float gamma;            /* Blend rate */
} SAEMBlender;

void saem_init(SAEMBlender *saem, float alpha, float kappa, float gamma) {
    saem->alpha_prior = alpha;
    saem->kappa_prior = kappa;
    saem->gamma = gamma;
    
    /* Initialize Q with prior */
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < K; j++) {
            float prior = alpha + ((i == j) ? kappa : 0.0f);
            saem->Q[i * K + j] = prior;
        }
    }
}

void saem_update(SAEMBlender *saem, const int *pgas_counts) {
    /*
     * SAEM update rule:
     *   Q_new = (1 - γ) * Q_old + γ * S
     * 
     * Where S is the transition counts from PGAS
     */
    for (int i = 0; i < K * K; i++) {
        saem->Q[i] = (1.0f - saem->gamma) * saem->Q[i] 
                   + saem->gamma * (float)pgas_counts[i];
    }
}

void saem_get_posterior_mean(const SAEMBlender *saem, float *Pi) {
    /* Dirichlet posterior mean: Π[i,j] = Q[i,j] / Σ_j Q[i,j] */
    for (int i = 0; i < K; i++) {
        float row_sum = 0.0f;
        for (int j = 0; j < K; j++) {
            row_sum += saem->Q[i * K + j];
        }
        float inv = 1.0f / row_sum;
        for (int j = 0; j < K; j++) {
            Pi[i * K + j] = saem->Q[i * K + j] * inv;
        }
    }
}

float saem_get_min_row_sum(const SAEMBlender *saem) {
    /* Used to decide explore vs exploit */
    float min_sum = 1e30f;
    for (int i = 0; i < K; i++) {
        float row_sum = 0.0f;
        for (int j = 0; j < K; j++) {
            row_sum += saem->Q[i * K + j];
        }
        if (row_sum < min_sum) min_sum = row_sum;
    }
    return min_sum;
}
```

### 3. Thompson Sampler

Samples Π from Dirichlet posterior or uses mean based on confidence.

```c
typedef struct {
    float exploit_threshold;  /* When to stop exploring */
    uint64_t rng_state;       /* RNG state */
} ThompsonSampler;

void thompson_init(ThompsonSampler *ts, float exploit_threshold, uint64_t seed) {
    ts->exploit_threshold = exploit_threshold;
    ts->rng_state = seed;
}

/* Fast RNG */
static inline float thompson_randf(ThompsonSampler *ts) {
    ts->rng_state ^= ts->rng_state << 13;
    ts->rng_state ^= ts->rng_state >> 7;
    ts->rng_state ^= ts->rng_state << 17;
    return (float)(ts->rng_state & 0xFFFFFF) / (float)0x1000000;
}

/* Gamma sample via Marsaglia-Tsang */
static float sample_gamma(ThompsonSampler *ts, float alpha) {
    if (alpha < 1.0f) {
        float u = thompson_randf(ts);
        return sample_gamma(ts, alpha + 1.0f) * powf(u, 1.0f / alpha);
    }
    
    float d = alpha - 1.0f / 3.0f;
    float c = 1.0f / sqrtf(9.0f * d);
    
    while (1) {
        float x, v;
        do {
            /* Box-Muller for normal */
            float u1 = thompson_randf(ts) + 1e-10f;
            float u2 = thompson_randf(ts);
            x = sqrtf(-2.0f * logf(u1)) * cosf(6.28318530718f * u2);
            v = 1.0f + c * x;
        } while (v <= 0.0f);
        
        v = v * v * v;
        float u = thompson_randf(ts);
        
        if (u < 1.0f - 0.0331f * (x * x) * (x * x)) return d * v;
        if (logf(u) < 0.5f * x * x + d * (1.0f - v + logf(v))) return d * v;
    }
}

void thompson_sample_Pi(ThompsonSampler *ts, const SAEMBlender *saem, float *Pi) {
    float min_row_sum = saem_get_min_row_sum(saem);
    
    if (min_row_sum >= ts->exploit_threshold) {
        /* 
         * EXPLOIT: High confidence, use posterior mean 
         * Variance is low, sampling would just add noise
         */
        saem_get_posterior_mean(saem, Pi);
    } else {
        /* 
         * EXPLORE: Low confidence, sample from Dirichlet
         * Each row is sampled independently
         */
        for (int i = 0; i < K; i++) {
            float row[K];
            float row_sum = 0.0f;
            
            for (int j = 0; j < K; j++) {
                /* Sample from Gamma(Q[i,j], 1) */
                row[j] = sample_gamma(ts, saem->Q[i * K + j]);
                row_sum += row[j];
            }
            
            /* Normalize to get Dirichlet sample */
            float inv = 1.0f / row_sum;
            for (int j = 0; j < K; j++) {
                Pi[i * K + j] = row[j] * inv;
            }
        }
    }
}
```

### 4. PGAS Interface

Wrapper for your existing PGAS implementation.

```c
typedef struct {
    /* Your existing PGAS state */
    void *pgas_state;
    
    /* Config */
    int n_sweeps;
    int window_size;
} PGASRunner;

void pgas_runner_init(PGASRunner *runner, int window_size, int n_sweeps,
                      const float *mu, const float *phi, const float *sigma) {
    runner->window_size = window_size;
    runner->n_sweeps = n_sweeps;
    
    /* Initialize your PGAS - adapt to your actual API */
    // runner->pgas_state = pgas_mkl_alloc(...);
    // pgas_mkl_init(runner->pgas_state);
}

void pgas_runner_run(PGASRunner *runner, const float *y, int T,
                     int *trans_counts_out) {
    /*
     * Run PGAS on observation window
     * Output: transition counts S[i,j] = # of i→j transitions
     * 
     * Adapt this to your actual PGAS API
     */
    
    // pgas_mkl_set_observations(runner->pgas_state, y, T);
    // 
    // for (int sweep = 0; sweep < runner->n_sweeps; sweep++) {
    //     pgas_mkl_sweep(runner->pgas_state);
    // }
    // 
    // /* Extract counts from final path */
    // const int *path = pgas_mkl_get_path(runner->pgas_state);
    // 
    // memset(trans_counts_out, 0, K * K * sizeof(int));
    // for (int t = 1; t < T; t++) {
    //     trans_counts_out[path[t-1] * K + path[t]]++;
    // }
}

void pgas_runner_free(PGASRunner *runner) {
    /* Free your PGAS state */
    // pgas_mkl_free(runner->pgas_state);
}
```

---

## Main Coordination Loop

```c
typedef struct {
    /* Components */
    RingBuffer buffer;
    SAEMBlender saem;
    ThompsonSampler thompson;
    PGASRunner pgas;
    
    /* Current estimate */
    float Pi[K * K];
    
    /* Config */
    int pgas_interval;      /* Run PGAS every N ticks */
    int warmup_ticks;       /* Wait before first PGAS */
    
    /* State */
    int tick_count;
} Coordinator;

void coordinator_init(Coordinator *coord,
                      int buffer_size,
                      int pgas_interval,
                      int pgas_sweeps,
                      const float *mu,
                      const float *phi,
                      const float *sigma) {
    
    /* Ring buffer */
    ring_buffer_init(&coord->buffer, buffer_size);
    
    /* SAEM: alpha=1.0 (weak prior), kappa=10.0 (sticky), gamma=0.3 */
    saem_init(&coord->saem, 1.0f, 10.0f, 0.3f);
    
    /* Thompson: exploit when row_sum > 500 */
    thompson_init(&coord->thompson, 500.0f, 12345);
    
    /* PGAS */
    pgas_runner_init(&coord->pgas, buffer_size, pgas_sweeps, mu, phi, sigma);
    
    /* Config */
    coord->pgas_interval = pgas_interval;
    coord->warmup_ticks = buffer_size;  /* Wait for buffer to fill */
    coord->tick_count = 0;
    
    /* Initial Π: sticky diagonal */
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < K; j++) {
            coord->Pi[i * K + j] = (i == j) ? 0.9f : 0.1f / (K - 1);
        }
    }
}

/*
 * Call this every tick
 * Returns: pointer to current Π for RBPF to use
 */
const float* coordinator_tick(Coordinator *coord, float y) {
    
    /* 1. Store observation */
    ring_buffer_push(&coord->buffer, y);
    coord->tick_count++;
    
    /* 2. Check if time to run PGAS */
    bool run_pgas = (coord->tick_count >= coord->warmup_ticks) &&
                    (coord->tick_count % coord->pgas_interval == 0);
    
    if (run_pgas) {
        /* 2a. Copy buffer to contiguous array */
        float y_window[coord->buffer.capacity];
        int window_len;
        ring_buffer_copy_to(&coord->buffer, y_window, &window_len);
        
        /* 2b. Run PGAS */
        int trans_counts[K * K];
        pgas_runner_run(&coord->pgas, y_window, window_len, trans_counts);
        
        /* 2c. SAEM blend */
        saem_update(&coord->saem, trans_counts);
        
        /* 2d. Thompson sample/exploit */
        thompson_sample_Pi(&coord->thompson, &coord->saem, coord->Pi);
    }
    
    return coord->Pi;
}

void coordinator_free(Coordinator *coord) {
    ring_buffer_free(&coord->buffer);
    pgas_runner_free(&coord->pgas);
}
```

---

## Usage with RBPF

```c
int main(void) {
    /* True parameters (for data generation / PGAS emission model) */
    float mu[K]    = {-5.0f, -3.5f, -1.5f, -4.0f};
    float phi[K]   = {0.995f, 0.95f, 0.85f, 0.97f};
    float sigma[K] = {0.08f, 0.20f, 0.50f, 0.15f};
    
    /* Initialize coordinator */
    Coordinator coord;
    coordinator_init(&coord,
                     2000,    /* buffer_size */
                     500,     /* pgas_interval */
                     10,      /* pgas_sweeps */
                     mu, phi, sigma);
    
    /* Initialize RBPF (your existing code) */
    RBPF rbpf;
    rbpf_init(&rbpf, ...);
    
    /* Main loop */
    for (int t = 0; t < T; t++) {
        float y = get_observation(t);
        
        /* 1. Coordinator updates Π */
        const float *Pi = coordinator_tick(&coord, y);
        
        /* 2. RBPF uses current Π */
        rbpf_set_transition_matrix(&rbpf, Pi);  /* or however you inject Π */
        rbpf_step(&rbpf, y);
        
        /* 3. Get estimates */
        float h_est = rbpf_get_h(&rbpf);
        int s_est = rbpf_get_regime(&rbpf);
    }
    
    /* Cleanup */
    coordinator_free(&coord);
    rbpf_free(&rbpf);
    
    return 0;
}
```

---

## Configuration

### Default Parameters

```c
/* SAEM */
#define SAEM_ALPHA      1.0f    /* Weak Dirichlet prior */
#define SAEM_KAPPA      10.0f   /* Stickiness prior on diagonal */
#define SAEM_GAMMA      0.3f    /* Blend rate: 0=ignore PGAS, 1=replace */

/* Thompson */
#define EXPLOIT_THRESHOLD 500.0f /* Row sum before switching to exploit */

/* Coordinator */
#define BUFFER_SIZE     2000    /* PGAS window (ticks) */
#define PGAS_INTERVAL   500     /* Run PGAS every N ticks */
#define PGAS_SWEEPS     10      /* Gibbs sweeps per PGAS run */
```

### Tuning Guide

| Parameter | Lower | Higher | Trade-off |
|-----------|-------|--------|-----------|
| `gamma` | Stable Π, slow adapt | Fast adapt, noisy | Responsiveness vs stability |
| `exploit_threshold` | Explore longer | Exploit sooner | Learning vs efficiency |
| `pgas_interval` | More responsive | Less compute | Latency vs accuracy |
| `buffer_size` | Recent data only | More context | Recency vs statistics |
| `pgas_sweeps` | Faster, noisier | Slower, cleaner | Latency vs quality |

---

## What This Gives You

### Correctness

- **SAEM**: Proper Bayesian blending on simplex
- **Sufficient statistics**: Rows always sum to 1
- **Thompson**: Principled exploration early, exploitation later

### Simplicity

- **Fixed interval**: Predictable, easy to debug
- **No triggers**: One less failure mode
- **Clean separation**: RBPF knows nothing about PGAS

### Performance

- **RBPF**: Every tick, unaffected by PGAS
- **PGAS**: Only every 500 ticks, ~160μs when it runs
- **Worst case**: Normal tick + PGAS = still under budget

---

## When to Upgrade

| Symptom | Add This |
|---------|----------|
| PGAS runs during stable periods (wasted) | Dual-gate triggers |
| Gets stuck after structural break | Three-tier reset |
| Confirmation bias (PGAS agrees with wrong RBPF) | Tempered injection |
| PGAS quality varies wildly | Confidence-based γ |
| PGAS fails silently | Scout pre-validation |

**Start simple. Add complexity only where it fails.**

---

## Testing Checklist

### 1. Basic Sanity

```c
/* Does SAEM converge to true Π with perfect counts? */
int true_counts[K * K] = {...};  /* From known data */
for (int i = 0; i < 100; i++) {
    saem_update(&saem, true_counts);
}
/* Check: saem.Q should approach true_counts structure */
```

### 2. Thompson Behavior

```c
/* Early: Should sample (explore) */
assert(saem_get_min_row_sum(&saem) < 500.0f);
/* Π should vary each call */

/* Late: Should use mean (exploit) */
/* Run many updates... */
assert(saem_get_min_row_sum(&saem) >= 500.0f);
/* Π should be stable */
```

### 3. End-to-End

```c
/* Generate synthetic data with known regimes */
/* Run coordinator + RBPF */
/* Compare estimated regimes vs true */
/* Compare estimated h vs true h */
/* Report RMSE and accuracy */
```

---

## Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                     THE 85% SOLUTION                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   RBPF          Fast, every tick                                │
│      ↑                                                          │
│      │ Π                                                        │
│      │                                                          │
│   Thompson      Explore (sample) or Exploit (mean)              │
│      ↑                                                          │
│      │ Q                                                        │
│      │                                                          │
│   SAEM          Blend sufficient statistics                     │
│      ↑                                                          │
│      │ counts                                                   │
│      │                                                          │
│   PGAS          Every 500 ticks, learns transitions             │
│      ↑                                                          │
│      │ observations                                             │
│      │                                                          │
│   Buffer        Ring buffer, last 2000 ticks                    │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Simple. Correct. Debuggable.                                  │
│   Add complexity only where it fails.                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

*Document version: 1.0*
*Last updated: December 2025*
