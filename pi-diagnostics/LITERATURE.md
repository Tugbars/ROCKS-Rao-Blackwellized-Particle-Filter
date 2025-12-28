# Literature Foundations for Oracle Bridge V2

## The Core Insight

**Our approach:** Monitor Π accuracy through prediction quality (RMSE, likelihood, transition lag), not just ESS.

This is grounded in several established bodies of literature:

---

## 1. Score-Driven Models (GAS)

**Key Papers:**
- Creal, D., Koopman, S.J., Lucas, A. (2013). "Generalized Autoregressive Score Models with Applications." *Journal of Applied Econometrics*, 28(5):777-795.
- Harvey, A.C. (2013). *Dynamic Models for Volatility and Heavy Tails*. Cambridge University Press.
- Blasques, F., Koopman, S.J., Lucas, A. (2015). "Information-theoretic optimality of observation-driven time series models." *Biometrika*, 102(2):325-343.

**Key Insight:**
> "By taking a scaled (local density) score step as a driving mechanism, the time-varying parameter automatically reduces its one-step-ahead **prediction error** at the current observation with respect to current parameter values."

**Relevance to Our Work:**
- GAS models update parameters using the **score of the likelihood** - essentially the prediction error
- Blasques et al. (2015) prove this is **optimal in minimizing KL divergence**
- Our approach: When prediction error spikes → Π is wrong → inject correction
- This is precisely the GAS philosophy applied to transition matrices

**The GAS Update Equation:**
```
f_{t+1} = ω + β·f_t + α·S(f_t)·∂log p(y_t|f_t)/∂f_t
                      └────────────────────────────┘
                           scaled score = prediction error signal
```

---

## 2. Particle Filter Degeneracy Literature

**Key Papers:**
- Doucet, A., de Freitas, N., Gordon, N. (2001). *Sequential Monte Carlo Methods in Practice*. Springer.
- Li, T., Bolic, M., Djuric, P.M. (2015). "Resampling Methods for Particle Filtering." *IEEE Signal Processing Magazine*.
- "Fight sample degeneracy and impoverishment in particle filters" (2014). *Expert Systems with Applications*.

**Key Insight:**
> "The consequences of the degeneracy problem are (i) almost all computational effort will be put into computations related to particles that have negligible or no contribution to the overall estimate and (ii) the number of effective particles is only one... Performance is poor since the particle filter will **diverge**."

**The Causal Chain (from literature):**
```
Wrong model parameters → Bad importance weights → Weight concentration → ESS drops → Collapse
```

**Our Extension:**
We monitor the **upstream** signals (prediction error, likelihood) rather than the **downstream** symptom (ESS).

---

## 3. Storvik Filter & Sufficient Statistics

**Key Papers:**
- Storvik, G. (2002). "Particle filters for state space models with the presence of unknown static parameters." *IEEE Trans. Signal Processing*, 50:281-289.
- Carvalho, C.M., Johannes, M.S., Lopes, H.F., Polson, N.G. (2010). "Particle Learning and Smoothing." *Statistical Science*, 25(1):88-106.
- Johannes, M.S., Polson, N.G. (2006). "Exact Particle Filtering and Parameter Learning."

**Key Insight:**
> "The approach is based on marginalizing the static parameters out of the posterior distribution... tracking sufficient statistics can be seen as replacing the sequential estimation of fixed parameters by the sequential updating of a low-dimensional vector of deterministic states."

**Relevance to Our Work:**
- Storvik's Q counts are our sufficient statistics for Π
- When Q (what RBPF learns) diverges from Π_operating (what RBPF assumes) → **D₃ self-contradiction**
- This is a pure internal signal requiring no Oracle

---

## 4. PGAS (Particle Gibbs with Ancestor Sampling)

**Key Papers:**
- Andrieu, C., Doucet, A., Holenstein, R. (2010). "Particle Markov chain Monte Carlo methods." *JRSS-B*, 72(3):269-342.
- Lindsten, F., Jordan, M.I., Schön, T.B. (2014). "Particle Gibbs with Ancestor Sampling." *JMLR*, 15:2145-2184.

**Key Insight:**
> "The ancestor sampling procedure enables fast mixing of the PGAS kernel even when using seemingly few particles... This is important as it can significantly reduce the computational burden."

**Relevance to Our Work:**
- PGAS provides the "Oracle" - a slower but more accurate Π estimator
- It samples from the **joint smoothing distribution** p(x_{1:T}, θ | y_{1:T})
- Unlike RBPF's filtering distribution, this uses **future** information
- PGAS Π is our ground truth for correcting RBPF's online estimate

---

## 5. Bayesian Model Criticism / Predictive Checks

**Key Papers:**
- Gelman, A., Meng, X.L., Stern, H. (1996). "Posterior predictive assessment of model fitness via realized discrepancies." *Statistica Sinica*, 6:733-807.
- Box, G.E.P. (1980). "Sampling and Bayes' inference in scientific modelling and robustness." *JRSS-A*, 143(4):383-430.

**Key Insight:**
> "If the model is correct, the data should look plausible under the posterior predictive distribution."

**Relevance to Our Work:**
- If Π is correct → predictions should be good → RMSE low, likelihood high
- If Π is wrong → predictions fail → RMSE spikes, likelihood drops
- We use prediction quality as a **model criticism** signal

---

## 6. Online Change Detection (BOCPD Connection)

**Key Papers:**
- Adams, R.P., MacKay, D.J.C. (2007). "Bayesian Online Changepoint Detection." arXiv:0710.3742.

**Key Insight:**
> BOCPD uses **predictive probability** p(x_t | x_{1:t-1}) to detect when the model no longer fits.

**Relevance to Our Work:**
- When likelihood drops sharply → regime may have changed
- Our "transition lag" metric measures how long until RBPF catches up
- High lag → Π diagonal is too sticky → need injection

---

## 7. Thompson Sampling (Variance Injection)

**Key Papers:**
- Thompson, W.R. (1933). "On the likelihood that one unknown probability exceeds another in view of the evidence of two samples." *Biometrika*, 25(3-4):285-294.
- Russo, D.J., Van Roy, B., Kazerouni, A., Osband, I., Wen, Z. (2018). "A Tutorial on Thompson Sampling." *Foundations and Trends in Machine Learning*, 11(1):1-96.

**Key Insight:**
> "Thompson Sampling maintains uncertainty and explores by sampling from the posterior."

**Relevance to Our Work:**
- When Oracle confidence is low but RBPF is critical → sample from Dirichlet posterior
- This injects **variance** rather than a potentially wrong point estimate
- Exploration under uncertainty

---

## Synthesis: Our Contribution

| Literature | What They Established | Our Extension |
|------------|----------------------|---------------|
| **GAS** | Score (prediction error) drives parameter updates | Use prediction error to trigger Π injection |
| **PF Degeneracy** | ESS measures particle health | ESS is late symptom; monitor upstream |
| **Storvik** | Sufficient statistics for online learning | Use Q divergence (D₃) as self-diagnosis |
| **PGAS** | Accurate offline Π estimation | PGAS as "Oracle" providing ground truth |
| **Model Criticism** | Prediction failures indicate model problems | RMSE/likelihood as early Π quality signals |
| **Thompson** | Sample from posterior under uncertainty | Variance shots when Oracle uncertain |

---

## The Complete Signal Hierarchy (Literature-Grounded)

```
┌─────────────────────────────────────────────────────────────────────┐
│  EARLY SIGNALS (Score-driven / Model Criticism)                     │
├─────────────────────────────────────────────────────────────────────┤
│  • Prediction RMSE spike [GAS, Model Criticism]                     │
│  • Likelihood z-score drop [BOCPD, Model Criticism]                 │
│  • D₃ self-contradiction [Storvik]                                  │
│  • Transition lag [BOCPD]                                           │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  CORRECTION MECHANISM                                               │
├─────────────────────────────────────────────────────────────────────┤
│  • PGAS Oracle provides Π_oracle [PMCMC, PGAS]                      │
│  • Three-way divergence diagnosis [Novel: D₁, D₂, D₃]               │
│  • γ from urgency × confidence [Score-driven intuition]             │
│  • Thompson sample when uncertain [Thompson Sampling]               │
│  • Storvik reset after injection [Storvik]                          │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  LATE SIGNALS (Standard PF literature)                              │
├─────────────────────────────────────────────────────────────────────┤
│  • ESS drop [Doucet et al., Gordon et al.]                          │
│  • Weight concentration [PF Degeneracy literature]                  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Key References (BibTeX)

```bibtex
@article{creal2013gas,
  title={Generalized autoregressive score models with applications},
  author={Creal, Drew and Koopman, Siem Jan and Lucas, Andr{\'e}},
  journal={Journal of Applied Econometrics},
  volume={28},
  number={5},
  pages={777--795},
  year={2013}
}

@article{storvik2002particle,
  title={Particle filters for state-space models with the presence of unknown static parameters},
  author={Storvik, Geir},
  journal={IEEE Transactions on signal Processing},
  volume={50},
  number={2},
  pages={281--289},
  year={2002}
}

@article{andrieu2010particle,
  title={Particle markov chain monte carlo methods},
  author={Andrieu, Christophe and Doucet, Arnaud and Holenstein, Roman},
  journal={Journal of the Royal Statistical Society: Series B},
  volume={72},
  number={3},
  pages={269--342},
  year={2010}
}

@article{lindsten2014pgas,
  title={Particle gibbs with ancestor sampling},
  author={Lindsten, Fredrik and Jordan, Michael I and Sch{\"o}n, Thomas B},
  journal={The Journal of Machine Learning Research},
  volume={15},
  number={1},
  pages={2145--2184},
  year={2014}
}

@article{blasques2015optimal,
  title={Information-theoretic optimality of observation-driven time series models for continuous responses},
  author={Blasques, Francisco and Koopman, Siem Jan and Lucas, Andr{\'e}},
  journal={Biometrika},
  volume={102},
  number={2},
  pages={325--343},
  year={2015}
}

@article{carvalho2010particle,
  title={Particle learning and smoothing},
  author={Carvalho, Carlos M and Johannes, Michael S and Lopes, Hedibert F and Polson, Nicholas G},
  journal={Statistical Science},
  volume={25},
  number={1},
  pages={88--106},
  year={2010}
}

@article{adams2007bocpd,
  title={Bayesian online changepoint detection},
  author={Adams, Ryan Prescott and MacKay, David JC},
  journal={arXiv preprint arXiv:0710.3742},
  year={2007}
}
```

---

## Summary

Our Oracle Bridge V2 architecture synthesizes:

1. **GAS philosophy**: Prediction error drives parameter correction
2. **Storvik's insight**: Sufficient statistics enable online learning + self-diagnosis
3. **PGAS**: Provides "Oracle" ground truth from smoothing distribution
4. **Model criticism**: Prediction quality signals model adequacy
5. **Thompson sampling**: Exploration under uncertainty

The novel contribution is the **three-way divergence monitoring** (D₁, D₂, D₃) and the **hierarchical signal framework** that prioritizes early prediction-based signals over late ESS-based symptoms.
