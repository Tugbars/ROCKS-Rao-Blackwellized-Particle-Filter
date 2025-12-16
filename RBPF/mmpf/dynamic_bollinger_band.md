# The Dynamic Bollinger Architecture (Structural Hierarchical Estimation)

This document explains the **Structural Hierarchical Estimation (SHE)** logic used in the MMPF Online EM module. This architecture replaces independent parameter learning with a geometrically constrained "Fleet" model.

## 1. The Core Concept

In standard adaptive filters, models (Calm, Trend, Crisis) learn independently. This leads to **Mode Collapse**: when data is ambiguous, models drift toward the average and become identical.

**The Structural Solution:** Instead of learning 3 independent volatility levels ($\mu_0, \mu_1, \mu_2$), we learn **2 structural variables** that define the entire system's geometry:

1.  **Base Level ($B_t$):** The "Center of Gravity" for market volatility.
2.  **Spread ($S_t$):** The structural distance between regimes.

The regimes are effectively **Dynamic Bollinger Bands** around the Base Level:

$$
\begin{aligned}
\mu_{\text{CALM}}   &= B_t - 1.0 \times S_t \quad (\text{Lower Band}) \\
\mu_{\text{TREND}}  &= B_t \quad \quad \quad \quad \quad (\text{Moving Average}) \\
\mu_{\text{CRISIS}} &= B_t + 1.5 \times S_t \quad (\text{Upper Band})
\end{aligned}
$$



[Image of Bollinger Bands]


## 2. Asymmetric Physics ("Fast to Fear, Slow to Forget")

The robustness comes from **Asymmetric Learning Rates**. Financial markets exhibit "Volatility Clustering" and "Leverage Effects"—panic hits instantly, but calm returns slowly.

We model this by making the **Spread ($S_t$)** behave like a ratchet mechanism:

### A. Expansion (The Shock)
When high-volatility data hits, the gradient suggests widening the spread ($\nabla S > 0$).
* **Learning Rate:** High ($0.05$)
* **Result:** The "Crisis Band" expands **instantly** to capture the shock. The fleet "puffs up" to catch the outlier.

### B. Contraction (The Decay)
When the market is calm, the gradient suggests shrinking the spread ($\nabla S < 0$).
* **Learning Rate:** Ultra-Low ($0.0005$)
* **Result:** The fleet contracts **very slowly**.
* **Benefit:** Even after months of calm, the Crisis model remains deployed far "out of the money" ($S_t \ge 0.8$). It does not collapse into the Calm model. It stands sentry.

## 3. The Algorithm (Online SGD)

The solver minimizes the weighted squared error between the *Weighted Posterior* (Truth) and the *Structural Hypothesis* (Prediction).

```c
/* 1. Calculate Error Gradients */
Error = y_log_sq - (Base + Coeff[k] * Spread);

/* 2. Update Base Level (Symmetric Drift) */
Base += 0.02 * Gradient_Base;

/* 3. Update Spread (Asymmetric Breathing) */
if (Gradient_Spread > 0) {
    /* Market is exploding -> Expand FAST */
    Spread += 0.05 * Gradient_Spread;
} else {
    /* Market is boring -> Contract SLOWLY */
    Spread += 0.0005 * Gradient_Spread;
}

/* 4. Enforce "Hard Deck" */
if (Spread < 0.8) Spread = 0.8; /* Never let the fleet collapse */