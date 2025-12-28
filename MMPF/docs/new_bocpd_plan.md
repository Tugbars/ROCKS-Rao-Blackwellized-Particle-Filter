Here is the strategic summary for your documentation.

# BOCPD Strategy: The "Sleepy Watchdog" Problem

## The Question

**"Can we make BOCPD hazard-rate-less?"**

Technically, yes. We can treat the hazard rate $H$ (probability of a changepoint) as a learnable parameter using a Beta-Bernoulli conjugate prior. As we observe run lengths, we update our belief about how often regimes break.

## The Verdict: DO NOT DO THIS.

For High-Frequency Trading (HFT) risk detection, learning the hazard rate introduces a critical failure mode known as the **"Sleepy Watchdog."**

## The Failure Mode

1.  **The Setup:** A long, stable bull market occurs (e.g., 3 months of low volatility).
2.  **The Learning:** The model observes thousands of ticks with no structure breaks.
3.  **The Adaptation:** The learned hazard rate $H$ asymptotically approaches zero. The model concludes: *"Structure breaks are effectively impossible."*
4.  **The Crash:** A sudden crisis hits.
5.  **The Lag:** Because the prior belief in stability is now massive, the model requires overwhelming evidence to trigger a changepoint. The "watchdog" has fallen asleep and reacts too late.

In HFT, we need **Constant Vigilance**. The alarm must be just as sensitive after a year of calm as it is during a storm.

## The Superior Alternatives

Instead of learning $H$, use one of these robust approaches to remove arbitrary tuning without losing sensitivity.

### Option 1: Scale-Free Power Laws (Recommended)

Use a hazard function that decays with run length but maintains a "fat tail" of probability.

$$H(r) = \frac{1}{r}$$

  * **Why:** It respects the fractal nature of financial time series. Regimes do not have a characteristic length; they follow a power law.
  * **Implementation:** Already available in `bocpd.c` via `bocpd_hazard_init_power_law`.

### Option 2: The Bayesian Grid (Maximum Accuracy)

Run 3 lightweight BOCPD instances in parallel with fixed, distinct timescales and average their shock probabilities.

```c
/* The "Omniscient" Detector */
double shock_prob = 0.0;

/* 1. The Nervous Scout (Fast regimes) */
bocpd_step(b_fast, y);   // Hazard = 1/50
shock_prob += 0.33 * b_fast->short_mass;

/* 2. The Normal Observer (Medium regimes) */
bocpd_step(b_med, y);    // Hazard = 1/200
shock_prob += 0.33 * b_med->short_mass;

/* 3. The Deep Structurer (Long regimes) */
bocpd_step(b_slow, y);   // Hazard = 1/1000
shock_prob += 0.33 * b_slow->short_mass;
```

**Why this wins:**

  * Catches **Micro-Fractures** (via `b_fast`) that signal flash crashes.
  * Catches **Macro-Shifts** (via `b_slow`) that happen gradually.
  * **Robust:** No single parameter choice can blind the system.

## Summary

For Sharpe 1.5+ performance, **avoid online learning for the hazard rate.** Use the **Power Law** or **Grid** method to ensure your risk detector never becomes complacent.