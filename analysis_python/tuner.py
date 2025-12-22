"""
RBPF Parameter Tuner

Port of rbpf_tuner.c for real market data.

8 knobs → 32 derived params:
  1. mu_calm      - Log-vol mean in calm regime
  2. mu_crisis    - Log-vol mean in crisis regime  
  3. sigma_calm   - Vol-of-vol in calm regime
  4. sigma_ratio  - σ_crisis / σ_calm
  5. theta_calm   - Mean reversion speed in calm
  6. theta_ratio  - θ_crisis / θ_calm
  7. stickiness   - Diagonal of transition matrix
  8. lambda_calm  - Forgetting factor in calm

Usage:
    from tuner import GridSearchTuner, quick_tune
    from data import fetch_returns, compute_realized_vol
    
    # Fetch data
    returns, dates = fetch_returns("SPY", period="2y")
    realized_vol = compute_realized_vol(returns, window=20, annualize=False)
    
    # Quick tune
    best = quick_tune(returns, realized_vol)
    print(best)
    
    # Full grid search
    tuner = GridSearchTuner(returns, realized_vol)
    results = tuner.search()
    tuner.print_best()
    tuner.save_csv("tuning_results.csv")
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import warnings

# Local imports
from rbpf import RBPF, ParamMode, AdaptSignal


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TunerConfig:
    """8-knob parameterization"""
    mu_calm: float = -4.5
    mu_crisis: float = -2.0
    sigma_calm: float = 0.08
    sigma_ratio: float = 8.0
    theta_calm: float = 0.003
    theta_ratio: float = 40.0
    stickiness: float = 0.92
    lambda_calm: float = 0.999
    
    def to_dict(self) -> dict:
        return {
            'mu_calm': self.mu_calm,
            'mu_crisis': self.mu_crisis,
            'sigma_calm': self.sigma_calm,
            'sigma_ratio': self.sigma_ratio,
            'theta_calm': self.theta_calm,
            'theta_ratio': self.theta_ratio,
            'stickiness': self.stickiness,
            'lambda_calm': self.lambda_calm,
        }
    
    def derive_params(self) -> Dict[str, np.ndarray]:
        """Derive full 4-regime parameters from 8 knobs."""
        # Interpolate μ_vol
        mu = np.array([
            self.mu_calm,
            self.mu_calm + (self.mu_crisis - self.mu_calm) * 0.33,
            self.mu_calm + (self.mu_crisis - self.mu_calm) * 0.67,
            self.mu_crisis
        ])
        
        # Interpolate σ_vol
        sigma_crisis = self.sigma_calm * self.sigma_ratio
        sigma = np.array([
            self.sigma_calm,
            self.sigma_calm + (sigma_crisis - self.sigma_calm) * 0.33,
            self.sigma_calm + (sigma_crisis - self.sigma_calm) * 0.67,
            sigma_crisis
        ])
        
        # Interpolate θ
        theta_crisis = self.theta_calm * self.theta_ratio
        theta = np.array([
            self.theta_calm,
            self.theta_calm + (theta_crisis - self.theta_calm) * 0.33,
            self.theta_calm + (theta_crisis - self.theta_calm) * 0.67,
            theta_crisis
        ])
        
        # Interpolate λ
        lambda_crisis = max(self.lambda_calm - 0.006, 0.990)
        lam = np.array([
            self.lambda_calm,
            self.lambda_calm - (self.lambda_calm - lambda_crisis) * 0.33,
            self.lambda_calm - (self.lambda_calm - lambda_crisis) * 0.67,
            lambda_crisis
        ])
        
        # Transition matrix
        s = self.stickiness
        leak = 1.0 - s
        trans = np.array([
            [s, leak * 0.70, leak * 0.25, leak * 0.05],
            [leak * 0.40, s, leak * 0.45, leak * 0.15],
            [leak * 0.15, leak * 0.45, s, leak * 0.40],
            [leak * 0.05, leak * 0.25, leak * 0.70, s]
        ], dtype=np.float32)
        
        # Normalize rows
        trans = trans / trans.sum(axis=1, keepdims=True)
        
        return {
            'mu': mu,
            'sigma': sigma,
            'theta': theta,
            'lambda': lam,
            'transition': trans
        }


@dataclass 
class TunerMetrics:
    """Evaluation metrics"""
    vol_rmse: float = np.inf
    vol_mae: float = np.inf
    vol_corr: float = 0.0
    log_vol_rmse: float = np.inf
    mean_surprise: float = np.inf
    mean_ess: float = 0.0
    min_ess: float = 0.0
    resample_rate: float = 0.0
    regime_entropy: float = 0.0      # Low = confident, high = uncertain
    regime_switches: int = 0         # Too many = unstable
    crisis_fraction: float = 0.0     # Fraction in regime 3
    
    def to_dict(self) -> dict:
        return {
            'vol_rmse': self.vol_rmse,
            'vol_mae': self.vol_mae,
            'vol_corr': self.vol_corr,
            'log_vol_rmse': self.log_vol_rmse,
            'mean_surprise': self.mean_surprise,
            'mean_ess': self.mean_ess,
            'min_ess': self.min_ess,
            'resample_rate': self.resample_rate,
            'regime_entropy': self.regime_entropy,
            'regime_switches': self.regime_switches,
            'crisis_fraction': self.crisis_fraction,
        }


@dataclass
class TunerResult:
    """Single tuning result"""
    config: TunerConfig
    metrics: TunerMetrics
    score: float = np.inf
    
    def to_dict(self) -> dict:
        return {**self.config.to_dict(), **self.metrics.to_dict(), 'score': self.score}


# =============================================================================
# Default Grids
# =============================================================================

DEFAULT_GRID = {
    'mu_calm': [-5.0, -4.5, -4.0],
    'mu_crisis': [-2.5, -2.0, -1.5],
    'sigma_calm': [0.06, 0.08, 0.10],
    'sigma_ratio': [4.0, 6.0, 8.0],
    'theta_calm': [0.003, 0.005, 0.008],
    'theta_ratio': [15.0, 25.0, 40.0],
    'stickiness': [0.90, 0.92, 0.95],
    'lambda_calm': [0.998, 0.999, 0.9995],
}

QUICK_GRID = {
    'mu_calm': [-5.0, -4.5],
    'mu_crisis': [-2.0, -1.5],
    'sigma_calm': [0.06, 0.08],
    'sigma_ratio': [6.0, 8.0],
    'theta_calm': [0.003, 0.005],
    'theta_ratio': [25.0, 40.0],
    'stickiness': [0.92, 0.95],
    'lambda_calm': [0.998, 0.999],
}

# Focused grid after initial search
FINE_GRID = {
    'mu_calm': [-4.75, -4.5, -4.25],
    'mu_crisis': [-2.25, -2.0, -1.75],
    'sigma_calm': [0.07, 0.08, 0.09],
    'sigma_ratio': [7.0, 8.0, 9.0],
    'theta_calm': [0.002, 0.003, 0.004],
    'theta_ratio': [35.0, 40.0, 45.0],
    'stickiness': [0.91, 0.92, 0.93],
    'lambda_calm': [0.9985, 0.999, 0.9995],
}


# =============================================================================
# Core Evaluation
# =============================================================================

def apply_config(rbpf: RBPF, config: TunerConfig):
    """Apply 8-knob config to RBPF instance."""
    params = config.derive_params()
    
    # Set regime params
    for r in range(4):
        rbpf.set_regime_params(r, params['theta'][r], params['mu'][r], params['sigma'][r])
    
    # Set transition matrix
    rbpf.set_transition_matrix(params['transition'])
    
    # Set forgetting factors
    for r in range(4):
        rbpf.set_regime_lambda(r, params['lambda'][r])


def run_config(
    returns: np.ndarray,
    realized_vol: np.ndarray,
    config: TunerConfig,
    n_particles: int = 256,
    warmup: int = 50,
) -> TunerMetrics:
    """
    Run RBPF with given config and compute metrics.
    
    Parameters
    ----------
    returns : np.ndarray
        Return series
    realized_vol : np.ndarray
        Realized volatility for comparison (same length as returns)
    config : TunerConfig
        8-knob configuration
    n_particles : int
        Number of particles
    warmup : int
        Warmup period to skip in metrics
    
    Returns
    -------
    metrics : TunerMetrics
    """
    n = len(returns)
    
    # Create RBPF
    rbpf = RBPF(n_particles=n_particles, n_regimes=4, mode=ParamMode.STORVIK)
    
    # Apply config
    apply_config(rbpf, config)
    
    # Enable features
    rbpf.enable_adaptive_forgetting(AdaptSignal.REGIME)
    rbpf.enable_circuit_breaker(0.999, 100)
    rbpf.enable_ocsn()
    for r in range(4):
        rbpf.set_ocsn_params(r, 0.02 + r * 0.01, 100 + r * 20)
    
    # Initialize
    rbpf.init(mu0=config.mu_calm, var0=0.1)
    
    # Storage
    vol_est = np.zeros(n)
    log_vol_est = np.zeros(n)
    surprises = np.zeros(n)
    ess_vals = np.zeros(n)
    resampled = np.zeros(n, dtype=bool)
    regimes = np.zeros(n, dtype=int)
    regime_probs = np.zeros((n, 4))
    
    # Run
    for t in range(n):
        rbpf.step(returns[t])
        vol_est[t] = rbpf.vol_mean
        log_vol_est[t] = rbpf.log_vol_mean
        surprises[t] = rbpf.surprise
        ess_vals[t] = rbpf.ess
        resampled[t] = rbpf.resampled
        regimes[t] = rbpf.dominant_regime
        regime_probs[t] = rbpf.regime_probs
    
    # Compute metrics (skip warmup, skip NaN in realized_vol)
    valid = np.arange(warmup, n)
    valid = valid[~np.isnan(realized_vol[valid])]
    
    if len(valid) == 0:
        return TunerMetrics()
    
    vol_true = realized_vol[valid]
    vol_pred = vol_est[valid]
    
    # Vol metrics
    vol_rmse = np.sqrt(np.mean((vol_pred - vol_true) ** 2))
    vol_mae = np.mean(np.abs(vol_pred - vol_true))
    vol_corr = np.corrcoef(vol_pred, vol_true)[0, 1] if len(valid) > 2 else 0.0
    
    # Log-vol RMSE (compare to log of realized)
    log_vol_true = np.log(vol_true + 1e-10)
    log_vol_pred = log_vol_est[valid]
    log_vol_rmse = np.sqrt(np.mean((log_vol_pred - log_vol_true) ** 2))
    
    # Surprise (predictive fit)
    mean_surprise = np.mean(surprises[valid])
    
    # ESS health
    mean_ess = np.mean(ess_vals[valid])
    min_ess = np.min(ess_vals[valid])
    resample_rate = np.mean(resampled[valid])
    
    # Regime stability
    regime_switches = np.sum(np.diff(regimes[valid]) != 0)
    crisis_fraction = np.mean(regimes[valid] == 3)
    
    # Regime entropy (average entropy of regime probabilities)
    probs = regime_probs[valid]
    probs = np.clip(probs, 1e-10, 1.0)  # Avoid log(0)
    entropy = -np.sum(probs * np.log(probs), axis=1)
    regime_entropy = np.mean(entropy)
    
    return TunerMetrics(
        vol_rmse=vol_rmse,
        vol_mae=vol_mae,
        vol_corr=vol_corr,
        log_vol_rmse=log_vol_rmse,
        mean_surprise=mean_surprise,
        mean_ess=mean_ess,
        min_ess=min_ess,
        resample_rate=resample_rate,
        regime_entropy=regime_entropy,
        regime_switches=regime_switches,
        crisis_fraction=crisis_fraction,
    )


def compute_score(
    metrics: TunerMetrics,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Compute overall score (lower is better).
    
    Default weighting:
    - 40% vol_rmse (primary goal)
    - 25% mean_surprise (predictive fit)
    - 15% vol_corr (inverse - higher is better)
    - 10% regime stability
    - 10% ESS health
    """
    if weights is None:
        weights = {
            'vol_rmse': 0.40,
            'surprise': 0.25,
            'corr': 0.15,
            'stability': 0.10,
            'ess': 0.10,
        }
    
    # Normalize each component to ~[0, 1]
    norm_rmse = metrics.vol_rmse / 0.02      # 2% vol = 1.0
    norm_surprise = metrics.mean_surprise / 5.0  # 5 nats = 1.0
    norm_corr = 1.0 - metrics.vol_corr       # Invert (higher corr = better)
    norm_stability = metrics.regime_switches / 100  # 100 switches = 1.0
    norm_ess = 1.0 - (metrics.mean_ess / 256)  # Invert (higher ESS = better)
    
    score = (
        weights.get('vol_rmse', 0) * norm_rmse +
        weights.get('surprise', 0) * norm_surprise +
        weights.get('corr', 0) * norm_corr +
        weights.get('stability', 0) * norm_stability +
        weights.get('ess', 0) * norm_ess
    )
    
    return score


# =============================================================================
# Grid Search Tuner
# =============================================================================

class GridSearchTuner:
    """
    Grid search over 8-knob parameter space.
    
    Example
    -------
    >>> tuner = GridSearchTuner(returns, realized_vol)
    >>> results = tuner.search()
    >>> tuner.print_best()
    >>> tuner.save_csv("results.csv")
    """
    
    def __init__(
        self,
        returns: np.ndarray,
        realized_vol: np.ndarray,
        grid: Optional[Dict] = None,
        n_particles: int = 256,
        warmup: int = 50,
        score_weights: Optional[Dict[str, float]] = None,
    ):
        self.returns = np.asarray(returns, dtype=np.float32)
        self.realized_vol = np.asarray(realized_vol, dtype=np.float32)
        self.grid = grid or DEFAULT_GRID
        self.n_particles = n_particles
        self.warmup = warmup
        self.score_weights = score_weights
        
        self.results: List[TunerResult] = []
        self.best_rmse: Optional[TunerResult] = None
        self.best_corr: Optional[TunerResult] = None
        self.best_balanced: Optional[TunerResult] = None
    
    def _build_configs(self) -> List[TunerConfig]:
        """Build all configurations from grid."""
        from itertools import product
        
        keys = list(self.grid.keys())
        values = [self.grid[k] for k in keys]
        
        configs = []
        for combo in product(*values):
            cfg_dict = dict(zip(keys, combo))
            configs.append(TunerConfig(**cfg_dict))
        
        return configs
    
    def search(self, parallel: bool = True, max_workers: Optional[int] = None) -> List[TunerResult]:
        """
        Run grid search.
        
        Parameters
        ----------
        parallel : bool
            Use multiprocessing (default: True)
        max_workers : int, optional
            Number of parallel workers
        
        Returns
        -------
        results : List[TunerResult]
            All results sorted by score
        """
        configs = self._build_configs()
        n_configs = len(configs)
        
        print(f"Grid Search: {n_configs} configurations")
        print(f"  Data: {len(self.returns)} ticks")
        print(f"  Particles: {self.n_particles}")
        
        start = time.time()
        self.results = []
        
        if parallel and n_configs > 1:
            # Parallel execution
            # Note: Can't pickle RBPF easily, so we use a wrapper function
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {}
                for i, cfg in enumerate(configs):
                    future = executor.submit(
                        _run_config_wrapper,
                        self.returns, self.realized_vol, cfg,
                        self.n_particles, self.warmup, self.score_weights
                    )
                    futures[future] = i
                
                for i, future in enumerate(as_completed(futures)):
                    result = future.result()
                    self.results.append(result)
                    
                    if (i + 1) % 50 == 0 or i == n_configs - 1:
                        print(f"  Progress: {i+1}/{n_configs} ({100*(i+1)/n_configs:.0f}%)")
        else:
            # Sequential
            for i, cfg in enumerate(configs):
                metrics = run_config(self.returns, self.realized_vol, cfg,
                                     self.n_particles, self.warmup)
                score = compute_score(metrics, self.score_weights)
                self.results.append(TunerResult(cfg, metrics, score))
                
                if (i + 1) % 50 == 0 or i == n_configs - 1:
                    print(f"  Progress: {i+1}/{n_configs} ({100*(i+1)/n_configs:.0f}%)")
        
        elapsed = time.time() - start
        print(f"\nCompleted in {elapsed:.1f}s ({1000*elapsed/n_configs:.1f}ms/config)")
        
        # Sort by score
        self.results.sort(key=lambda r: r.score)
        
        # Find bests
        self.best_balanced = self.results[0]
        self.best_rmse = min(self.results, key=lambda r: r.metrics.vol_rmse)
        self.best_corr = max(self.results, key=lambda r: r.metrics.vol_corr)
        
        return self.results
    
    def print_best(self):
        """Print best configurations."""
        if not self.results:
            print("No results. Run search() first.")
            return
        
        print("\n" + "=" * 65)
        print("  BEST FOR VOL RMSE")
        print("=" * 65)
        _print_result(self.best_rmse)
        
        print("\n" + "=" * 65)
        print("  BEST FOR CORRELATION")
        print("=" * 65)
        _print_result(self.best_corr)
        
        print("\n" + "=" * 65)
        print("  BEST BALANCED")
        print("=" * 65)
        _print_result(self.best_balanced)
        
        print("\n" + "=" * 65)
        print("  RECOMMENDED PYTHON CODE")
        print("=" * 65)
        _print_python_code(self.best_balanced)
    
    def save_csv(self, filepath: str):
        """Save all results to CSV."""
        if not self.results:
            print("No results. Run search() first.")
            return
        
        df = pd.DataFrame([r.to_dict() for r in self.results])
        df.to_csv(filepath, index=False)
        print(f"Saved {len(self.results)} results to {filepath}")
    
    def get_dataframe(self) -> pd.DataFrame:
        """Get results as DataFrame."""
        return pd.DataFrame([r.to_dict() for r in self.results])


def _run_config_wrapper(returns, realized_vol, config, n_particles, warmup, score_weights):
    """Wrapper for parallel execution."""
    metrics = run_config(returns, realized_vol, config, n_particles, warmup)
    score = compute_score(metrics, score_weights)
    return TunerResult(config, metrics, score)


def _print_result(result: TunerResult):
    """Print single result."""
    cfg = result.config
    m = result.metrics
    
    print(f"  Config:")
    print(f"    μ_calm={cfg.mu_calm:.2f}  μ_crisis={cfg.mu_crisis:.2f}")
    print(f"    σ_calm={cfg.sigma_calm:.3f}  σ_ratio={cfg.sigma_ratio:.1f}")
    print(f"    θ_calm={cfg.theta_calm:.4f}  θ_ratio={cfg.theta_ratio:.1f}")
    print(f"    stickiness={cfg.stickiness:.2f}  λ_calm={cfg.lambda_calm:.4f}")
    print(f"  ─────────────────────────────")
    print(f"    Vol RMSE:     {m.vol_rmse:.6f}")
    print(f"    Vol Corr:     {m.vol_corr:.4f}")
    print(f"    Log-Vol RMSE: {m.log_vol_rmse:.4f}")
    print(f"    Mean Surprise:{m.mean_surprise:.2f}")
    print(f"    Mean ESS:     {m.mean_ess:.1f}")
    print(f"    Regime Switches: {m.regime_switches}")
    print(f"    Crisis Frac:  {m.crisis_fraction:.1%}")
    print(f"    Score:        {result.score:.4f}")


def _print_python_code(result: TunerResult):
    """Print Python code to reproduce config."""
    cfg = result.config
    params = cfg.derive_params()
    
    print(f"""
# Regime params (theta, mu_vol, sigma_vol)
rbpf.set_regime_params(0, {params['theta'][0]:.4f}, {params['mu'][0]:.2f}, {params['sigma'][0]:.3f})  # Calm
rbpf.set_regime_params(1, {params['theta'][1]:.4f}, {params['mu'][1]:.2f}, {params['sigma'][1]:.3f})  # Mild
rbpf.set_regime_params(2, {params['theta'][2]:.4f}, {params['mu'][2]:.2f}, {params['sigma'][2]:.3f})  # Trend
rbpf.set_regime_params(3, {params['theta'][3]:.4f}, {params['mu'][3]:.2f}, {params['sigma'][3]:.3f})  # Crisis

# Transition matrix (stickiness={cfg.stickiness:.2f})
trans = np.array([
    [{params['transition'][0,0]:.3f}, {params['transition'][0,1]:.3f}, {params['transition'][0,2]:.3f}, {params['transition'][0,3]:.3f}],
    [{params['transition'][1,0]:.3f}, {params['transition'][1,1]:.3f}, {params['transition'][1,2]:.3f}, {params['transition'][1,3]:.3f}],
    [{params['transition'][2,0]:.3f}, {params['transition'][2,1]:.3f}, {params['transition'][2,2]:.3f}, {params['transition'][2,3]:.3f}],
    [{params['transition'][3,0]:.3f}, {params['transition'][3,1]:.3f}, {params['transition'][3,2]:.3f}, {params['transition'][3,3]:.3f}]
], dtype=np.float32)
rbpf.set_transition_matrix(trans)

# Forgetting λ per regime
rbpf.set_regime_lambda(0, {params['lambda'][0]:.4f})
rbpf.set_regime_lambda(1, {params['lambda'][1]:.4f})
rbpf.set_regime_lambda(2, {params['lambda'][2]:.4f})
rbpf.set_regime_lambda(3, {params['lambda'][3]:.4f})
""")


# =============================================================================
# Quick Tune (Convenience Function)
# =============================================================================

def quick_tune(
    returns: np.ndarray,
    realized_vol: np.ndarray,
    verbose: bool = True,
) -> TunerResult:
    """
    Quick parameter tuning with reduced grid.
    
    Returns best balanced configuration.
    """
    tuner = GridSearchTuner(returns, realized_vol, grid=QUICK_GRID, n_particles=256)
    tuner.search(parallel=False)  # Quick grid is small enough
    
    if verbose:
        tuner.print_best()
    
    return tuner.best_balanced


def fine_tune(
    returns: np.ndarray,
    realized_vol: np.ndarray,
    center: Optional[TunerConfig] = None,
    verbose: bool = True,
) -> TunerResult:
    """
    Fine-grained tuning around a center point.
    
    Use after quick_tune to refine.
    """
    if center is not None:
        # Build grid around center point
        grid = {
            'mu_calm': [center.mu_calm - 0.25, center.mu_calm, center.mu_calm + 0.25],
            'mu_crisis': [center.mu_crisis - 0.25, center.mu_crisis, center.mu_crisis + 0.25],
            'sigma_calm': [center.sigma_calm - 0.01, center.sigma_calm, center.sigma_calm + 0.01],
            'sigma_ratio': [center.sigma_ratio - 1, center.sigma_ratio, center.sigma_ratio + 1],
            'theta_calm': [center.theta_calm * 0.8, center.theta_calm, center.theta_calm * 1.2],
            'theta_ratio': [center.theta_ratio - 5, center.theta_ratio, center.theta_ratio + 5],
            'stickiness': [center.stickiness - 0.01, center.stickiness, center.stickiness + 0.01],
            'lambda_calm': [center.lambda_calm - 0.0005, center.lambda_calm, center.lambda_calm + 0.0005],
        }
    else:
        grid = FINE_GRID
    
    tuner = GridSearchTuner(returns, realized_vol, grid=grid, n_particles=256)
    tuner.search(parallel=False)
    
    if verbose:
        tuner.print_best()
    
    return tuner.best_balanced


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == "__main__":
    print("RBPF Tuner Test")
    print("=" * 65)
    
    # Generate synthetic data for testing
    np.random.seed(42)
    n = 1000
    vol = 0.01 + 0.005 * np.sin(np.linspace(0, 4*np.pi, n))  # Time-varying vol
    returns = vol * np.random.randn(n)
    
    # Realized vol (20-day rolling)
    realized_vol = np.full(n, np.nan)
    for i in range(19, n):
        realized_vol[i] = np.std(returns[i-19:i+1])
    
    print(f"Data: {n} ticks")
    print(f"Realized vol range: [{np.nanmin(realized_vol):.4f}, {np.nanmax(realized_vol):.4f}]")
    
    # Quick tune
    print("\nRunning quick tune...")
    best = quick_tune(returns, realized_vol)
    
    print("\nDone!")
