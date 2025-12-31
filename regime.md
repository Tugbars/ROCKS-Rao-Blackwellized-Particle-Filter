Below is the **clean, consolidated conclusion** of all recommendations, without detours, contradictions, or alternatives.

This is the **most robust, production-grade way** to combine **PGAS** and **RBPF** for stochastic volatility with rare, short crises.

---

# Final Recommendation: **PGAS-as-Teacher + RBPF-as-Worker**

## Core problem you observed

* Sliding / online PGAS is **structurally amnesiac**
* Short crises vanish once they leave the window
* Feeding window-learned transition matrices directly into RBPF makes the system **too calm**

This is not a bug. It is a **fundamental limitation of windowed smoothing**.

---

## Key principle (non-negotiable)

> **PGAS must not be responsible for long-term memory.**
> **RBPF must not relearn global structure online.**

Memory must live **between them**.

---

## Architecture (the correct one)

### 1. **RBPF (online, fast)**

* Runs at every tick
* Tracks:

  * latent volatility (h_t)
  * regime probabilities (p(s_t))
* Reacts immediately to innovations
* Uses a *slowly evolving* transition matrix (\Pi^{RBPF})

RBPF **never learns transitions itself**.

---

### 2. **PGAS (slow tick / batch)**

* Runs every (B) ticks (e.g. 50–500) or on triggers
* Operates on a sliding window
* Produces:

  * smoothed regime paths
  * reliable transition evidence inside the window

PGAS **never carries long-term memory**.

---

### 3. **Persistent Transition Memory (the missing piece)**

Maintain global, decayed transition sufficient statistics:

[
C_{ij} \leftarrow \rho,C_{ij} + w,N^{\text{PGAS}}_{ij}
]

Where:

* (N^{\text{PGAS}}_{ij}): transition counts or expected counts from PGAS
* (\rho): slow decay (much longer than window length)
* (w): confidence weight (acceptance rate / ESS based)

This memory:

* never resets
* never slides
* decays smoothly
* preserves short crises

**This is what fixes amnesia.**

---

## Transition matrix construction (robust choice)

From (C):

* Use **Dirichlet posterior** with:

  * diagonal stickiness
  * non-zero calm→crisis prior

Then:

### **Thompson sample** (strongly recommended)

[
\Pi^{new}*{i\cdot} \sim \text{Dirichlet}(\alpha*{i\cdot} + C_{i\cdot})
]

Why:

* avoids brittle posterior means
* preserves rare but plausible transitions
* cheap and conjugate

Posterior mean also works **only if counts are large and priors strong**.

---

## Injection into RBPF (do NOT overwrite)

Use **plain convex blending (not SAEM)**:

[
\Pi^{RBPF}
\leftarrow
(1-\lambda_t),\Pi^{RBPF}
+
\lambda_t,\Pi^{new}
]

* (\lambda_t) small by default (stability)
* increase (\lambda_t) only when **surprise evidence** appears:

  * predictive likelihood drop
  * ESS collapse
  * volatility innovation spike

This prevents:

* oscillations
* calm snap-back
* policy jitter

**SAEM blending is NOT recommended** (wrong abstraction for nonstationary dynamics).

---

## What is necessary vs optional

### Necessary (must have)

1. Persistent decayed transition counts
2. Sticky + nonzero entry Dirichlet prior

### Strongly recommended

3. Thompson sampling from the posterior
4. Light blending/gating into RBPF

### Optional refinements

* Weight window contribution by PGAS acceptance rate
* Dual-timescale counts (fast + slow memory)
* Expected counts instead of single path

---

## Why this is the most robust solution

* Short crises leave a lasting footprint
* Calm periods do not erase crisis memory
* PGAS remains fast and local
* RBPF remains reactive and stable
* No SMC², no recursive PGAS, no EM convergence traps

This architecture is:

* statistically principled
* computationally feasible
* production-stable
* **S-tier**

---

## Final bottom line

> **Use PGAS only to extract transition evidence.**
> **Store that evidence in persistent, decayed counts.**
> **Sample transitions with Thompson sampling.**
> **Blend them slowly into an online RBPF.**

If you implement exactly this, the “PGAS is offline or amnesiac” problem disappears—without changing your core models or increasing computational order.
