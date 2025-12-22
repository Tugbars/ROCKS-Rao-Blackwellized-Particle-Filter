"""
Volatility Model Comparison

Compare RBPF against standard volatility models:
- GARCH(1,1)
- EWMA
- Rolling Realized Volatility
- Constant (baseline)

Metrics:
- RMSE, MAE, QLIKE
- Mincer-Zarnowitz regression
- Diebold-Mariano test

Usage:
    from compare import ModelComparison
    
    comp = ModelComparison(returns, dates)
    comp.add_rbpf(rbpf_vols, name="RBPF")
    comp.add_ewma(halflife=20)
    comp.add_garch()
    comp.add_realized(window=20)
    
    results = comp.evaluate()
    comp.plot_comparison()
    comp.print_summary()
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import warnings

# Optional dependencies
try:
    from arch import arch_model
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False
    warnings.warn("arch package not installed. GARCH models unavailable. pip install arch")

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# =============================================================================
# Metrics
# =============================================================================

def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    """Root Mean Squared Error."""
    valid = ~(np.isnan(pred) | np.isnan(true))
    return np.sqrt(np.mean((pred[valid] - true[valid]) ** 2))


def mae(pred: np.ndarray, true: np.ndarray) -> float:
    """Mean Absolute Error."""
    valid = ~(np.isnan(pred) | np.isnan(true))
    return np.mean(np.abs(pred[valid] - true[valid]))


def qlike(pred: np.ndarray, true: np.ndarray) -> float:
    """
    QLIKE loss function (quasi-likelihood).
    
    QLIKE = E[log(σ²_pred) + σ²_true / σ²_pred]
    
    Standard metric for volatility forecasting.
    Lower is better.
    """
    valid = ~(np.isnan(pred) | np.isnan(true))
    pred_v = pred[valid] ** 2  # Variance
    true_v = true[valid] ** 2
    
    # Avoid log(0)
    pred_v = np.clip(pred_v, 1e-10, None)
    
    return np.mean(np.log(pred_v) + true_v / pred_v)


def mincer_zarnowitz(pred: np.ndarray, true: np.ndarray) -> Dict[str, float]:
    """
    Mincer-Zarnowitz regression.
    
    σ²_true = α + β * σ²_pred + ε
    
    Good forecast: α ≈ 0, β ≈ 1
    Returns α, β, R², and p-value for β=1 test.
    """
    valid = ~(np.isnan(pred) | np.isnan(true))
    x = pred[valid] ** 2
    y = true[valid] ** 2
    
    n = len(x)
    if n < 10:
        return {'alpha': np.nan, 'beta': np.nan, 'r2': np.nan, 'p_value': np.nan}
    
    # OLS regression
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    beta = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
    alpha = y_mean - beta * x_mean
    
    # R²
    y_pred = alpha + beta * x
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    
    # Standard error of beta (for hypothesis test)
    se_beta = np.sqrt(ss_res / (n - 2) / np.sum((x - x_mean) ** 2))
    
    # t-test for H0: β = 1
    t_stat = (beta - 1) / se_beta if se_beta > 0 else np.inf
    
    if HAS_SCIPY:
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-2))
    else:
        p_value = np.nan
    
    return {
        'alpha': alpha,
        'beta': beta,
        'r2': r2,
        'p_value': p_value,  # p-value for β ≠ 1
    }


def diebold_mariano(
    loss1: np.ndarray,
    loss2: np.ndarray,
    h: int = 1,
) -> Tuple[float, float]:
    """
    Diebold-Mariano test for equal predictive accuracy.
    
    Parameters
    ----------
    loss1, loss2 : np.ndarray
        Loss series for two models (e.g., squared errors)
    h : int
        Forecast horizon
    
    Returns
    -------
    dm_stat : float
        DM test statistic
    p_value : float
        Two-sided p-value
    """
    d = loss1 - loss2
    valid = ~np.isnan(d)
    d = d[valid]
    n = len(d)
    
    if n < 10:
        return np.nan, np.nan
    
    # Mean loss differential
    d_mean = np.mean(d)
    
    # HAC variance estimate (Newey-West with h-1 lags)
    gamma_0 = np.var(d, ddof=1)
    gamma = gamma_0
    
    for k in range(1, h):
        if k < n:
            gamma += 2 * (1 - k / h) * np.cov(d[k:], d[:-k])[0, 1]
    
    # DM statistic
    se = np.sqrt(gamma / n)
    dm_stat = d_mean / se if se > 0 else 0
    
    if HAS_SCIPY:
        p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    else:
        p_value = np.nan
    
    return dm_stat, p_value


# =============================================================================
# Volatility Models
# =============================================================================

def compute_ewma(
    returns: np.ndarray,
    halflife: int = 20,
) -> np.ndarray:
    """Exponentially Weighted Moving Average volatility."""
    alpha = 1 - np.exp(-np.log(2) / halflife)
    
    n = len(returns)
    var = np.zeros(n)
    var[0] = returns[0] ** 2
    
    for i in range(1, n):
        var[i] = alpha * returns[i] ** 2 + (1 - alpha) * var[i - 1]
    
    return np.sqrt(var)


def compute_realized_vol(
    returns: np.ndarray,
    window: int = 20,
) -> np.ndarray:
    """Rolling realized volatility."""
    n = len(returns)
    rv = np.full(n, np.nan)
    
    for i in range(window - 1, n):
        rv[i] = np.std(returns[i - window + 1:i + 1], ddof=1)
    
    return rv


def compute_garch(
    returns: np.ndarray,
    p: int = 1,
    q: int = 1,
) -> np.ndarray:
    """
    GARCH(p,q) conditional volatility.
    
    Requires 'arch' package.
    """
    if not HAS_ARCH:
        raise ImportError("arch package required for GARCH. pip install arch")
    
    # Scale returns for numerical stability
    scale = 100
    scaled_returns = returns * scale
    
    # Fit model
    model = arch_model(scaled_returns, vol='Garch', p=p, q=q, rescale=False)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.fit(disp='off')
    
    # Get conditional volatility
    cond_vol = result.conditional_volatility / scale
    
    return cond_vol


def compute_constant(returns: np.ndarray) -> np.ndarray:
    """Constant volatility (unconditional std)."""
    return np.full(len(returns), np.std(returns))


# =============================================================================
# Model Comparison Class
# =============================================================================

@dataclass
class ModelResult:
    """Results for a single model."""
    name: str
    volatility: np.ndarray
    rmse: float = np.nan
    mae: float = np.nan
    qlike: float = np.nan
    mz_alpha: float = np.nan
    mz_beta: float = np.nan
    mz_r2: float = np.nan
    mz_pvalue: float = np.nan
    
    def to_dict(self) -> dict:
        return {
            'Model': self.name,
            'RMSE': self.rmse,
            'MAE': self.mae,
            'QLIKE': self.qlike,
            'MZ_α': self.mz_alpha,
            'MZ_β': self.mz_beta,
            'MZ_R²': self.mz_r2,
            'MZ_p': self.mz_pvalue,
        }


class ModelComparison:
    """
    Compare multiple volatility models.
    
    Example
    -------
    >>> comp = ModelComparison(returns, dates)
    >>> comp.add_rbpf(rbpf_vols)
    >>> comp.add_ewma(halflife=20)
    >>> comp.add_garch()
    >>> comp.add_realized(window=20)
    >>> 
    >>> results = comp.evaluate()
    >>> comp.print_summary()
    >>> comp.plot_comparison()
    """
    
    def __init__(
        self,
        returns: np.ndarray,
        dates: Optional[pd.DatetimeIndex] = None,
        realized_vol: Optional[np.ndarray] = None,
        realized_window: int = 20,
    ):
        self.returns = np.asarray(returns)
        self.dates = dates
        self.n = len(returns)
        
        # Compute realized vol as ground truth if not provided
        if realized_vol is not None:
            self.realized_vol = np.asarray(realized_vol)
        else:
            self.realized_vol = compute_realized_vol(returns, realized_window)
        
        self.models: Dict[str, ModelResult] = {}
    
    def add_model(self, name: str, volatility: np.ndarray):
        """Add a custom model's volatility estimates."""
        if len(volatility) != self.n:
            raise ValueError(f"Expected {self.n} values, got {len(volatility)}")
        self.models[name] = ModelResult(name=name, volatility=np.asarray(volatility))
    
    def add_rbpf(self, volatility: np.ndarray, name: str = "RBPF"):
        """Add RBPF volatility estimates."""
        self.add_model(name, volatility)
    
    def add_ewma(self, halflife: int = 20, name: Optional[str] = None):
        """Add EWMA model."""
        if name is None:
            name = f"EWMA({halflife})"
        vol = compute_ewma(self.returns, halflife)
        self.add_model(name, vol)
    
    def add_garch(self, p: int = 1, q: int = 1, name: Optional[str] = None):
        """Add GARCH model."""
        if not HAS_ARCH:
            warnings.warn("GARCH unavailable (arch package not installed)")
            return
        
        if name is None:
            name = f"GARCH({p},{q})"
        
        try:
            vol = compute_garch(self.returns, p, q)
            self.add_model(name, vol)
        except Exception as e:
            warnings.warn(f"GARCH fitting failed: {e}")
    
    def add_realized(self, window: int = 20, name: Optional[str] = None):
        """Add rolling realized volatility as a model."""
        if name is None:
            name = f"RV({window})"
        vol = compute_realized_vol(self.returns, window)
        self.add_model(name, vol)
    
    def add_constant(self, name: str = "Constant"):
        """Add constant volatility baseline."""
        vol = compute_constant(self.returns)
        self.add_model(name, vol)
    
    def evaluate(self, warmup: int = 50) -> pd.DataFrame:
        """
        Evaluate all models against realized volatility.
        
        Parameters
        ----------
        warmup : int
            Skip first N observations (for realized vol warmup)
        
        Returns
        -------
        results : pd.DataFrame
            Comparison table
        """
        rv = self.realized_vol
        
        for name, model in self.models.items():
            vol = model.volatility
            
            # Compute metrics (skip warmup and NaN)
            valid = np.arange(warmup, self.n)
            valid = valid[~(np.isnan(rv[valid]) | np.isnan(vol[valid]))]
            
            if len(valid) < 10:
                warnings.warn(f"Too few valid points for {name}")
                continue
            
            v = vol[valid]
            r = rv[valid]
            
            model.rmse = rmse(v, r)
            model.mae = mae(v, r)
            model.qlike = qlike(v, r)
            
            mz = mincer_zarnowitz(v, r)
            model.mz_alpha = mz['alpha']
            model.mz_beta = mz['beta']
            model.mz_r2 = mz['r2']
            model.mz_pvalue = mz['p_value']
        
        # Build results DataFrame
        results = pd.DataFrame([m.to_dict() for m in self.models.values()])
        results = results.set_index('Model')
        
        return results
    
    def diebold_mariano_matrix(self, loss: str = 'se', warmup: int = 50) -> pd.DataFrame:
        """
        Compute pairwise Diebold-Mariano tests.
        
        Parameters
        ----------
        loss : str
            Loss function: 'se' (squared error) or 'qlike'
        warmup : int
            Skip first N observations
        
        Returns
        -------
        dm_matrix : pd.DataFrame
            Matrix of DM test p-values
        """
        rv = self.realized_vol
        names = list(self.models.keys())
        n_models = len(names)
        
        # Compute losses
        losses = {}
        for name, model in self.models.items():
            vol = model.volatility
            valid = np.arange(warmup, self.n)
            
            if loss == 'se':
                losses[name] = (vol - rv) ** 2
            elif loss == 'qlike':
                vol_sq = np.clip(vol ** 2, 1e-10, None)
                rv_sq = rv ** 2
                losses[name] = np.log(vol_sq) + rv_sq / vol_sq
            else:
                raise ValueError(f"Unknown loss: {loss}")
        
        # Pairwise DM tests
        dm_matrix = pd.DataFrame(index=names, columns=names, dtype=float)
        
        for i, name1 in enumerate(names):
            for j, name2 in enumerate(names):
                if i == j:
                    dm_matrix.loc[name1, name2] = np.nan
                else:
                    _, pval = diebold_mariano(losses[name1], losses[name2])
                    dm_matrix.loc[name1, name2] = pval
        
        return dm_matrix
    
    def print_summary(self):
        """Print comparison summary."""
        results = self.evaluate()
        
        print("\n" + "=" * 75)
        print("  VOLATILITY MODEL COMPARISON")
        print("=" * 75)
        print(f"\n  Observations: {self.n}")
        print(f"  Models: {len(self.models)}")
        
        print("\n  METRICS (lower is better for RMSE, MAE, QLIKE)")
        print("  " + "-" * 70)
        
        # Format table
        fmt = "  {:<15} {:>10} {:>10} {:>10} {:>8} {:>8} {:>8}"
        print(fmt.format("Model", "RMSE", "MAE", "QLIKE", "MZ_β", "MZ_R²", "MZ_p"))
        print("  " + "-" * 70)
        
        for name, model in self.models.items():
            print(fmt.format(
                name[:15],
                f"{model.rmse:.6f}",
                f"{model.mae:.6f}",
                f"{model.qlike:.4f}",
                f"{model.mz_beta:.3f}",
                f"{model.mz_r2:.3f}",
                f"{model.mz_pvalue:.3f}" if not np.isnan(model.mz_pvalue) else "N/A"
            ))
        
        print("  " + "-" * 70)
        
        # Find best
        best_rmse = min(self.models.values(), key=lambda m: m.rmse)
        best_qlike = min(self.models.values(), key=lambda m: m.qlike if not np.isnan(m.qlike) else np.inf)
        
        print(f"\n  Best RMSE:  {best_rmse.name}")
        print(f"  Best QLIKE: {best_qlike.name}")
        
        # MZ interpretation
        print("\n  MINCER-ZARNOWITZ INTERPRETATION:")
        print("    β ≈ 1, α ≈ 0: Unbiased forecast")
        print("    β < 1: Underreacts to vol changes")
        print("    β > 1: Overreacts to vol changes")
        print("    p < 0.05: Reject H₀: β=1 (biased)")
    
    def plot_comparison(
        self,
        figsize: Tuple[int, int] = (14, 10),
        start: Optional[int] = None,
        end: Optional[int] = None,
    ):
        """
        Plot volatility comparison.
        
        Parameters
        ----------
        figsize : tuple
            Figure size
        start, end : int, optional
            Slice of data to plot
        """
        start = start or 0
        end = end or self.n
        
        x = self.dates[start:end] if self.dates is not None else np.arange(start, end)
        rv = self.realized_vol[start:end]
        
        n_models = len(self.models)
        colors = plt.cm.tab10(np.linspace(0, 1, n_models))
        
        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
        
        # Top: All volatilities
        ax = axes[0]
        ax.plot(x, rv, 'k-', linewidth=2, alpha=0.7, label='Realized')
        
        for (name, model), color in zip(self.models.items(), colors):
            vol = model.volatility[start:end]
            ax.plot(x, vol, linewidth=1.2, color=color, alpha=0.8, label=name)
        
        ax.set_ylabel('Volatility')
        ax.set_title('Volatility Estimates vs Realized')
        ax.legend(loc='upper right', ncol=2)
        ax.grid(True, alpha=0.3)
        
        # Bottom: Errors
        ax = axes[1]
        
        for (name, model), color in zip(self.models.items(), colors):
            vol = model.volatility[start:end]
            error = vol - rv
            ax.plot(x, error, linewidth=1, color=color, alpha=0.7, label=name)
        
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_ylabel('Error (Est - Realized)')
        ax.set_title('Forecast Errors')
        ax.legend(loc='upper right', ncol=2)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_scatter(self, figsize: Tuple[int, int] = (14, 4)):
        """Plot predicted vs realized scatter for each model."""
        n_models = len(self.models)
        fig, axes = plt.subplots(1, n_models, figsize=figsize)
        
        if n_models == 1:
            axes = [axes]
        
        rv = self.realized_vol
        
        for ax, (name, model) in zip(axes, self.models.items()):
            vol = model.volatility
            valid = ~(np.isnan(vol) | np.isnan(rv))
            
            ax.scatter(vol[valid], rv[valid], alpha=0.3, s=10)
            
            # 45-degree line
            lim = max(np.nanmax(vol), np.nanmax(rv))
            ax.plot([0, lim], [0, lim], 'r--', linewidth=1, label='Perfect')
            
            # MZ regression line
            mz = mincer_zarnowitz(vol, rv)
            x_line = np.linspace(0, lim, 100)
            y_line = mz['alpha'] + mz['beta'] * x_line ** 2
            y_line = np.sqrt(np.clip(y_line, 0, None))
            ax.plot(x_line, y_line, 'g-', linewidth=1, 
                    label=f"MZ: β={mz['beta']:.2f}")
            
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Realized')
            ax.set_title(f'{name}\nRMSE={model.rmse:.5f}')
            ax.legend(loc='upper left', fontsize=8)
            ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.show()


# =============================================================================
# Quick Comparison Function
# =============================================================================

def quick_compare(
    returns: np.ndarray,
    rbpf_vols: np.ndarray,
    dates: Optional[pd.DatetimeIndex] = None,
    show_plot: bool = True,
) -> pd.DataFrame:
    """
    Quick comparison of RBPF against standard models.
    
    Parameters
    ----------
    returns : np.ndarray
        Return series
    rbpf_vols : np.ndarray
        RBPF volatility estimates
    dates : pd.DatetimeIndex, optional
        Date index for plotting
    show_plot : bool
        Whether to show comparison plot
    
    Returns
    -------
    results : pd.DataFrame
        Comparison table
    """
    comp = ModelComparison(returns, dates)
    comp.add_rbpf(rbpf_vols)
    comp.add_ewma(halflife=10)
    comp.add_ewma(halflife=20)
    comp.add_ewma(halflife=60)
    
    if HAS_ARCH:
        comp.add_garch(1, 1)
    
    comp.add_constant()
    
    comp.print_summary()
    
    if show_plot:
        comp.plot_comparison()
    
    return comp.evaluate()


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == "__main__":
    print("Model Comparison Test")
    print("=" * 65)
    
    # Generate synthetic data
    np.random.seed(42)
    n = 500
    
    # Time-varying volatility
    vol_true = 0.01 + 0.005 * np.sin(np.linspace(0, 4*np.pi, n))
    vol_true[200:250] *= 3  # Crisis period
    
    returns = vol_true * np.random.randn(n)
    
    # Simulate RBPF output (with some noise around true vol)
    rbpf_vols = vol_true * (1 + 0.1 * np.random.randn(n))
    rbpf_vols = np.clip(rbpf_vols, 0.001, None)
    
    print(f"Data: {n} observations")
    
    # Run comparison
    results = quick_compare(returns, rbpf_vols, show_plot=False)
    
    print("\nResults DataFrame:")
    print(results)
    
    print("\nDone!")
