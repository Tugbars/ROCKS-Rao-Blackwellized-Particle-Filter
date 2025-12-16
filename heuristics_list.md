###**💀 The Graveyard (Eliminated Heuristics)***These were arbitrary rules or "magic numbers" that have been replaced by rigorous math.*

| Feature | What it was | Why it was bad | What replaced it |
| --- | --- | --- | --- |
| **EWMA "Wagon"** | A separate EWMA filter forcing the particle filter to a baseline. | It ignored the particle filter's own optimal estimate. | **Online EM (MAP)**: The filter learns its own baseline from the data it "owns." |
| **"Blind Spray" Shock** | Multiplying process noise by 50\times when a shock is detected. | It relied on random chance (diffusion) to find the new volatility level. Slow and laggy. | **MCMC Teleportation**: Uses Metropolis-Hastings to statistically "jump" to the likelihood peak instantly. |
| **"Panic Drift"** | Adding an arbitrary constant to variance when returns were large. | It broke the Kalman update logic to force wider confidence intervals. | **Student-t (\nu)**: Heavy tails are now baked into the probability model (y_{shifted}). |
| **"Jaws of Life"** | Hard `if` statements forcing model means apart (`if A > B then A = B - 0.5`). | It created discontinuities and artificial boundaries. | **Bayesian Priors (\kappa)**: Soft gravitational pull prevents mode collapse mathematically. |
| **Liu-West Kernel** | Shrinking parameters via kernel density estimation. | It introduces "loss of information" (over-smoothing) every time you resample. | **Storvik Filter**: Updates sufficient statistics exactly. No information loss. |
| **Drift Clamping** | Hard limits on how far a particle could drift from the mean. | prevented the model from discovering new structural regimes. | **Moving Leash**: Updating the Storvik Prior allows the particles to migrate naturally. |

---

###**🛡️ The Survivors (Remaining Heuristics)***These are "Structural Priors" or "Numerical Safeguards." They exist because we run on finite computers with finite data, not because the math is wrong.*

| Feature | The Heuristic | Justification (Why it stays) |
| --- | --- | --- |
| **Mixing Floor** | `min_prob = 0.005` (0.5%) | **Numerical Safety (Lindstone Smoothing).** If a probability hits absolute 0.0, the hypothesis dies forever. We keep it barely alive so it can resurrect if the market changes. |
| **OCSN Variance** | `var = 150.0` | **Uninformative Robust Prior.** We cannot learn the variance of a rare outlier component from sparse data. We define it a priori as "Very Large" to neutralize the Kalman Gain during glitches. |
| **Entropy Lock** | `delta < 0.02` | **Convergence Proxy.** There is no analytical formula for "when a particle cloud has settled." We use an EMA threshold as a practical engineering stop condition for the MCMC loop. |
| **Zero Returns** | `min_log_sq = -18.0` | **Singularity Avoidance.** \log(0) = -\infty. We model the market as continuous, but prices are discrete. We must clamp zero returns to the "resolution limit" of the exchange (1 tick). |
| **Resampling** | `ESS < N/2` | **Efficiency Heuristic.** Standard particle filter practice. Resampling at every step adds noise; resampling too rarely causes degeneracy. N/2 is the industry standard balance. |
| **Detection Dwell** | `min_dwell = 3` | **Hysteresis.** Prevents the model from "flickering" between regimes (Calm -> Crisis -> Calm) on a single tick. Enforces a minimum structural duration. |

###**The Final Verdict*** **Before:** ~70% Heuristic, 30% Math. (A system that "worked" because it was tweaked to death).
* **Now:** ~5% Heuristic, 95% Math.

The remaining heuristics are **necessary boundary conditions** for running statistical inference on real-time hardware. 