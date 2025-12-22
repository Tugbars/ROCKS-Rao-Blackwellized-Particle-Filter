"""
RBPF Python Wrapper (ctypes)

Rao-Blackwellized Particle Filter for Stochastic Volatility

Usage:
    from rbpf import RBPF, ParamMode, AdaptSignal
    
    # Create filter
    rbpf = RBPF(n_particles=512, n_regimes=4, mode=ParamMode.STORVIK)
    
    # Configure regimes (theta, mu_vol, sigma_vol)
    rbpf.set_regime_params(0, 0.003, -4.5, 0.08)   # Calm
    rbpf.set_regime_params(1, 0.042, -3.67, 0.267) # Mild  
    rbpf.set_regime_params(2, 0.081, -2.83, 0.453) # Trend
    rbpf.set_regime_params(3, 0.120, -2.0, 0.64)   # Crisis
    
    # Set transition matrix (4x4 row-major)
    rbpf.set_transition_matrix(trans_matrix)
    
    # Initialize
    rbpf.init(mu0=-4.5, var0=0.1)
    
    # Process ticks
    for ret in returns:
        rbpf.step(ret)
        print(f"Vol: {rbpf.vol_mean:.4f}, Regime: {rbpf.dominant_regime}")
"""

import ctypes
import numpy as np
import os
import sys
from pathlib import Path
from enum import IntEnum
from typing import Tuple, Optional, List, Dict
import platform

# =============================================================================
# Windows MKL DLL paths (must be set BEFORE loading the library)
# =============================================================================

if sys.platform == "win32":
    mkl_paths = [
        r"C:\Program Files (x86)\Intel\oneAPI\mkl\2025.0\bin",
        r"C:\Program Files (x86)\Intel\oneAPI\mkl\2025.1\bin",
        r"C:\Program Files (x86)\Intel\oneAPI\mkl\2025.2\bin",
        r"C:\Program Files (x86)\Intel\oneAPI\mkl\2025.3\bin",
        r"C:\Program Files (x86)\Intel\oneAPI\mkl\latest\bin",
        r"C:\Program Files (x86)\Intel\oneAPI\compiler\2025.0\bin",
        r"C:\Program Files (x86)\Intel\oneAPI\compiler\2025.1\bin",
        r"C:\Program Files (x86)\Intel\oneAPI\compiler\2025.2\bin",
        r"C:\Program Files (x86)\Intel\oneAPI\compiler\2025.3\bin",
        r"C:\Program Files (x86)\Intel\oneAPI\compiler\latest\bin",
    ]
    for p in mkl_paths:
        if os.path.exists(p):
            os.add_dll_directory(p)

# =============================================================================
# Enums
# =============================================================================

class ParamMode(IntEnum):
    """Parameter learning mode"""
    DISABLED = 0
    LIU_WEST = 1
    STORVIK = 2
    HYBRID = 3

class AdaptSignal(IntEnum):
    """Adaptive forgetting signal source"""
    REGIME = 0              # Regime-only (baseline λ)
    OUTLIER_FRAC = 1        # Outlier fraction only
    SURPRISE = 2            # Predictive surprise only
    COMBINED = 3            # Max of outlier + surprise (recommended)

# =============================================================================
# Constants
# =============================================================================

RBPF_MAX_REGIMES = 8
SPRT_MAX_REGIMES = 8

# =============================================================================
# ctypes Structure Definitions
# =============================================================================

class RBPF_KSC_Output(ctypes.Structure):
    """Output structure from rbpf_ext_step / rbpf_ksc_step"""
    _fields_ = [
        # Fast signal (t)
        ("vol_mean", ctypes.c_float),
        ("log_vol_mean", ctypes.c_float),
        ("log_vol_var", ctypes.c_float),
        ("ess", ctypes.c_float),
        
        # Regime
        ("regime_probs", ctypes.c_float * RBPF_MAX_REGIMES),
        ("dominant_regime", ctypes.c_int),
        ("smoothed_regime", ctypes.c_int),
        
        # Self-aware signals
        ("marginal_lik", ctypes.c_float),
        ("surprise", ctypes.c_float),
        ("vol_ratio", ctypes.c_float),
        ("regime_entropy", ctypes.c_float),
        
        # Detection flags
        ("regime_changed", ctypes.c_int),
        ("change_type", ctypes.c_int),
        
        # Smooth signal (t-K)
        ("smooth_valid", ctypes.c_int),
        ("smooth_lag", ctypes.c_int),
        ("vol_mean_smooth", ctypes.c_float),
        ("log_vol_mean_smooth", ctypes.c_float),
        ("log_vol_var_smooth", ctypes.c_float),
        ("regime_probs_smooth", ctypes.c_float * RBPF_MAX_REGIMES),
        ("dominant_regime_smooth", ctypes.c_int),
        ("regime_confidence", ctypes.c_float),
        
        # Learned parameters
        ("learned_mu_vol", ctypes.c_float * RBPF_MAX_REGIMES),
        ("learned_sigma_vol", ctypes.c_float * RBPF_MAX_REGIMES),
        
        # Diagnostics
        ("resampled", ctypes.c_int),
        ("apf_triggered", ctypes.c_int),
        ("outlier_fraction", ctypes.c_float),
        
        # Student-t diagnostics
        ("lambda_mean", ctypes.c_float),
        ("lambda_var", ctypes.c_float),
        ("nu_effective", ctypes.c_float),
        ("learned_nu", ctypes.c_float * RBPF_MAX_REGIMES),
        ("student_t_active", ctypes.c_int),
        
        # SPRT evidence
        ("sprt_evidence", ctypes.c_double * SPRT_MAX_REGIMES),
        
        # BOCPD diagnostics
        ("bocpd_triggered", ctypes.c_int),
        ("bocpd_map_runlength", ctypes.c_size_t),
        ("bocpd_p_changepoint", ctypes.c_float),
        
        # Transition learning
        ("trans_learning_active", ctypes.c_int),
        ("trans_learned_this_tick", ctypes.c_int),
        ("trans_stickiness", ctypes.c_float * RBPF_MAX_REGIMES),
    ]

# =============================================================================
# Library Loading
# =============================================================================

def _load_library() -> ctypes.CDLL:
    """Load the RBPF shared library (rocks.dll)."""
    
    system = platform.system()
    if system == "Windows":
        lib_names = ["rocks.dll", "librocks.dll", "RBPF.dll"]
    elif system == "Darwin":
        lib_names = ["librocks.dylib", "librocks.so"]
    else:
        lib_names = ["librocks.so"]
    
    this_dir = Path(__file__).parent
    root_dir = this_dir.parent
    
    search_paths = [
        this_dir,
        this_dir / "lib",
        root_dir / "build" / "RBPF" / "Release",
        root_dir / "build" / "RBPF" / "Debug",
        root_dir / "build" / "Release",
        root_dir / "build" / "Debug",
        root_dir / "build",
        root_dir,
        Path.cwd(),
        Path.cwd() / "build" / "RBPF" / "Release",
        Path.cwd() / "build" / "Release",
    ]
    
    errors = []
    for search_path in search_paths:
        for lib_name in lib_names:
            lib_path = search_path / lib_name
            if lib_path.exists():
                try:
                    return ctypes.CDLL(str(lib_path))
                except OSError as e:
                    errors.append(f"{lib_path}: {e}")
                    continue
    
    # Try system paths
    for lib_name in lib_names:
        try:
            return ctypes.CDLL(lib_name)
        except OSError as e:
            errors.append(f"{lib_name} (system): {e}")
            continue
    
    error_msg = f"Could not find/load RBPF library.\n"
    error_msg += f"Searched for {lib_names} in {[str(p) for p in search_paths]}.\n"
    if errors:
        error_msg += "Load errors:\n" + "\n".join(f"  - {e}" for e in errors)
        error_msg += "\n\nMake sure Intel MKL runtime DLLs are in PATH."
    raise RuntimeError(error_msg)

# Load library
_lib = _load_library()

# =============================================================================
# Function Signatures
# =============================================================================

# Opaque handle (RBPF_Extended*)
_ExtPtr = ctypes.c_void_p

# --- Lifecycle ---
_lib.rbpf_ext_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
_lib.rbpf_ext_create.restype = _ExtPtr

_lib.rbpf_ext_destroy.argtypes = [_ExtPtr]
_lib.rbpf_ext_destroy.restype = None

_lib.rbpf_ext_init.argtypes = [_ExtPtr, ctypes.c_float, ctypes.c_float]
_lib.rbpf_ext_init.restype = None

# --- Step ---
_lib.rbpf_ext_step.argtypes = [_ExtPtr, ctypes.c_float, ctypes.POINTER(RBPF_KSC_Output)]
_lib.rbpf_ext_step.restype = None

# --- Configuration ---
_lib.rbpf_ext_set_regime_params.argtypes = [_ExtPtr, ctypes.c_int, 
                                             ctypes.c_float, ctypes.c_float, ctypes.c_float]
_lib.rbpf_ext_set_regime_params.restype = None

_lib.rbpf_ext_build_transition_lut.argtypes = [_ExtPtr, ctypes.POINTER(ctypes.c_float)]
_lib.rbpf_ext_build_transition_lut.restype = None

# --- Robust OCSN ---
_lib.rbpf_ext_enable_robust_ocsn.argtypes = [_ExtPtr]
_lib.rbpf_ext_enable_robust_ocsn.restype = None

_lib.rbpf_ext_disable_robust_ocsn.argtypes = [_ExtPtr]
_lib.rbpf_ext_disable_robust_ocsn.restype = None

_lib.rbpf_ext_set_outlier_params.argtypes = [_ExtPtr, ctypes.c_int, 
                                              ctypes.c_float, ctypes.c_float]
_lib.rbpf_ext_set_outlier_params.restype = None

_lib.rbpf_ext_get_outlier_fraction.argtypes = [_ExtPtr]
_lib.rbpf_ext_get_outlier_fraction.restype = ctypes.c_float

# --- Adaptive Forgetting ---
_lib.rbpf_ext_enable_adaptive_forgetting.argtypes = [_ExtPtr]
_lib.rbpf_ext_enable_adaptive_forgetting.restype = None

_lib.rbpf_ext_enable_adaptive_forgetting_mode.argtypes = [_ExtPtr, ctypes.c_int]
_lib.rbpf_ext_enable_adaptive_forgetting_mode.restype = None

_lib.rbpf_ext_disable_adaptive_forgetting.argtypes = [_ExtPtr]
_lib.rbpf_ext_disable_adaptive_forgetting.restype = None

_lib.rbpf_ext_set_regime_lambda.argtypes = [_ExtPtr, ctypes.c_int, ctypes.c_float]
_lib.rbpf_ext_set_regime_lambda.restype = None

_lib.rbpf_ext_get_current_lambda.argtypes = [_ExtPtr]
_lib.rbpf_ext_get_current_lambda.restype = ctypes.c_float

# --- Circuit Breaker ---
_lib.rbpf_ext_enable_circuit_breaker.argtypes = [_ExtPtr, ctypes.c_double, ctypes.c_int]
_lib.rbpf_ext_enable_circuit_breaker.restype = None

_lib.rbpf_ext_disable_circuit_breaker.argtypes = [_ExtPtr]
_lib.rbpf_ext_disable_circuit_breaker.restype = None

_lib.rbpf_ext_get_circuit_breaker_trips.argtypes = [_ExtPtr]
_lib.rbpf_ext_get_circuit_breaker_trips.restype = ctypes.c_uint64

_lib.rbpf_ext_structural_break_detected.argtypes = [_ExtPtr]
_lib.rbpf_ext_structural_break_detected.restype = ctypes.c_int

# --- Smoothed Storvik (PARIS) ---
_lib.rbpf_ext_enable_smoothed_storvik.argtypes = [_ExtPtr, ctypes.c_int]
_lib.rbpf_ext_enable_smoothed_storvik.restype = ctypes.c_int

_lib.rbpf_ext_disable_smoothed_storvik.argtypes = [_ExtPtr]
_lib.rbpf_ext_disable_smoothed_storvik.restype = None

_lib.rbpf_ext_is_smoothed_storvik_enabled.argtypes = [_ExtPtr]
_lib.rbpf_ext_is_smoothed_storvik_enabled.restype = ctypes.c_int

# --- Parameter Access ---
_lib.rbpf_ext_get_learned_params.argtypes = [_ExtPtr, ctypes.c_int,
                                              ctypes.POINTER(ctypes.c_float),
                                              ctypes.POINTER(ctypes.c_float)]
_lib.rbpf_ext_get_learned_params.restype = None

# --- Diagnostics ---
_lib.rbpf_ext_print_config.argtypes = [_ExtPtr]
_lib.rbpf_ext_print_config.restype = None

# =============================================================================
# Python Wrapper Class
# =============================================================================

class RBPF:
    """
    Rao-Blackwellized Particle Filter for Stochastic Volatility
    
    Features:
    - Storvik online parameter learning (μ_vol, σ_vol per regime)
    - Robust OCSN (11th mixture component for outliers)
    - Adaptive forgetting (regime-aware λ)
    - Circuit breaker (P² quantile detector)
    - PARIS smoothed Storvik (optional)
    
    Parameters
    ----------
    n_particles : int
        Number of particles (default: 512)
    n_regimes : int
        Number of volatility regimes (default: 4)
    mode : ParamMode
        Parameter learning mode (default: STORVIK)
    
    Examples
    --------
    >>> from rbpf import RBPF, ParamMode
    >>> import numpy as np
    >>> 
    >>> # Create filter
    >>> rbpf = RBPF(n_particles=512, n_regimes=4)
    >>> 
    >>> # Configure regimes
    >>> rbpf.set_regime_params(0, theta=0.003, mu_vol=-4.5, sigma_vol=0.08)
    >>> rbpf.set_regime_params(1, theta=0.042, mu_vol=-3.67, sigma_vol=0.267)
    >>> rbpf.set_regime_params(2, theta=0.081, mu_vol=-2.83, sigma_vol=0.453)
    >>> rbpf.set_regime_params(3, theta=0.120, mu_vol=-2.0, sigma_vol=0.64)
    >>> 
    >>> # Set transition matrix
    >>> trans = np.array([
    ...     [0.92, 0.056, 0.020, 0.004],
    ...     [0.032, 0.92, 0.036, 0.012],
    ...     [0.012, 0.036, 0.92, 0.032],
    ...     [0.004, 0.020, 0.056, 0.92]
    >>> ], dtype=np.float32)
    >>> rbpf.set_transition_matrix(trans)
    >>> 
    >>> # Initialize and run
    >>> rbpf.init(mu0=-4.5, var0=0.1)
    >>> for ret in returns:
    ...     rbpf.step(ret)
    ...     print(f"Vol: {rbpf.vol_mean:.4f}")
    """
    
    def __init__(self, n_particles: int = 512, n_regimes: int = 4, 
                 mode: ParamMode = ParamMode.STORVIK):
        self._n_particles = n_particles
        self._n_regimes = n_regimes
        self._mode = mode
        
        self._handle = _lib.rbpf_ext_create(n_particles, n_regimes, int(mode))
        if not self._handle:
            raise RuntimeError("Failed to create RBPF instance")
        
        # Output structure (reused each tick)
        self._output = RBPF_KSC_Output()
        self._tick_count = 0
    
    def __del__(self):
        if hasattr(self, '_handle') and self._handle:
            _lib.rbpf_ext_destroy(self._handle)
    
    # =========================================================================
    # Lifecycle
    # =========================================================================
    
    def init(self, mu0: float = -4.5, var0: float = 0.1):
        """
        Initialize filter state.
        
        Parameters
        ----------
        mu0 : float
            Initial log-volatility mean (default: -4.5 ≈ 1.1% daily vol)
        var0 : float
            Initial log-volatility variance (default: 0.1)
        """
        _lib.rbpf_ext_init(self._handle, mu0, var0)
        self._tick_count = 0
    
    def step(self, obs: float):
        """
        Process one observation (tick).
        
        After calling, access results via properties:
        - rbpf.vol_mean, rbpf.log_vol_mean
        - rbpf.dominant_regime, rbpf.regime_probs
        - rbpf.ess, rbpf.surprise, etc.
        
        Parameters
        ----------
        obs : float
            Raw return (NOT log-squared, the filter transforms it)
        """
        _lib.rbpf_ext_step(self._handle, obs, ctypes.byref(self._output))
        self._tick_count += 1
    
    # =========================================================================
    # Configuration
    # =========================================================================
    
    def set_regime_params(self, regime: int, theta: float, mu_vol: float, sigma_vol: float):
        """
        Set parameters for a volatility regime.
        
        Parameters
        ----------
        regime : int
            Regime index [0, n_regimes)
        theta : float
            Mean reversion speed (e.g., 0.003 for calm, 0.12 for crisis)
        mu_vol : float
            Long-run mean of log-volatility (e.g., -4.5 for calm, -2.0 for crisis)
        sigma_vol : float
            Vol-of-vol (e.g., 0.08 for calm, 0.64 for crisis)
        """
        if regime < 0 or regime >= self._n_regimes:
            raise ValueError(f"regime must be in [0, {self._n_regimes})")
        _lib.rbpf_ext_set_regime_params(self._handle, regime, theta, mu_vol, sigma_vol)
    
    def set_transition_matrix(self, trans: np.ndarray):
        """
        Set the K×K transition matrix.
        
        Parameters
        ----------
        trans : np.ndarray
            Transition probabilities, shape (K, K), row-major.
            trans[i, j] = P(regime_t = j | regime_{t-1} = i)
        """
        trans = np.ascontiguousarray(trans, dtype=np.float32).flatten()
        expected = self._n_regimes * self._n_regimes
        if len(trans) != expected:
            raise ValueError(f"Expected {expected} elements, got {len(trans)}")
        trans_ptr = trans.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        _lib.rbpf_ext_build_transition_lut(self._handle, trans_ptr)
    
    # =========================================================================
    # Robust OCSN (Outlier Handling)
    # =========================================================================
    
    def enable_ocsn(self):
        """Enable Robust OCSN (11th mixture component for outliers)."""
        _lib.rbpf_ext_enable_robust_ocsn(self._handle)
    
    def disable_ocsn(self):
        """Disable Robust OCSN."""
        _lib.rbpf_ext_disable_robust_ocsn(self._handle)
    
    def set_ocsn_params(self, regime: int, prob: float, variance: float):
        """
        Set per-regime outlier parameters.
        
        Parameters
        ----------
        regime : int
            Regime index
        prob : float
            Outlier probability (e.g., 0.02 = 2%)
        variance : float
            Outlier variance (e.g., 100.0)
        """
        _lib.rbpf_ext_set_outlier_params(self._handle, regime, prob, variance)
    
    # =========================================================================
    # Adaptive Forgetting
    # =========================================================================
    
    def enable_adaptive_forgetting(self, mode: AdaptSignal = AdaptSignal.REGIME):
        """
        Enable adaptive forgetting.
        
        Parameters
        ----------
        mode : AdaptSignal
            Signal source for forgetting factor modulation.
            REGIME: Use regime-specific baseline λ only
            COMBINED: Modulate λ based on surprise (recommended)
        """
        _lib.rbpf_ext_enable_adaptive_forgetting_mode(self._handle, int(mode))
    
    def disable_adaptive_forgetting(self):
        """Disable adaptive forgetting."""
        _lib.rbpf_ext_disable_adaptive_forgetting(self._handle)
    
    def set_regime_lambda(self, regime: int, lambda_val: float):
        """
        Set per-regime forgetting factor.
        
        Parameters
        ----------
        regime : int
            Regime index
        lambda_val : float
            Forgetting factor (e.g., 0.999 for calm, 0.993 for crisis)
        """
        _lib.rbpf_ext_set_regime_lambda(self._handle, regime, lambda_val)
    
    # =========================================================================
    # Circuit Breaker
    # =========================================================================
    
    def enable_circuit_breaker(self, quantile: float = 0.999, warmup: int = 100):
        """
        Enable P² circuit breaker.
        
        Triggers on extreme surprises, signaling structural breaks.
        
        Parameters
        ----------
        quantile : float
            Trigger percentile (default: 0.999 = 99.9th percentile)
        warmup : int
            Warmup ticks before activation (default: 100)
        """
        _lib.rbpf_ext_enable_circuit_breaker(self._handle, quantile, warmup)
    
    def disable_circuit_breaker(self):
        """Disable circuit breaker."""
        _lib.rbpf_ext_disable_circuit_breaker(self._handle)
    
    @property
    def circuit_breaker_trips(self) -> int:
        """Number of times circuit breaker has tripped."""
        return _lib.rbpf_ext_get_circuit_breaker_trips(self._handle)
    
    @property
    def structural_break_detected(self) -> bool:
        """True if structural break detected this tick."""
        return bool(_lib.rbpf_ext_structural_break_detected(self._handle))
    
    # =========================================================================
    # Smoothed Storvik (PARIS)
    # =========================================================================
    
    def enable_smoothed_storvik(self, lag: int = 50) -> bool:
        """
        Enable PARIS-smoothed Storvik parameter learning.
        
        Parameters
        ----------
        lag : int
            Smoothing lag L (default: 50)
        
        Returns
        -------
        success : bool
        """
        return _lib.rbpf_ext_enable_smoothed_storvik(self._handle, lag) == 0
    
    def disable_smoothed_storvik(self):
        """Disable smoothed Storvik."""
        _lib.rbpf_ext_disable_smoothed_storvik(self._handle)
    
    @property
    def smoothed_storvik_enabled(self) -> bool:
        """True if smoothed Storvik is enabled."""
        return bool(_lib.rbpf_ext_is_smoothed_storvik_enabled(self._handle))
    
    # =========================================================================
    # Output Properties (call after step())
    # =========================================================================
    
    @property
    def vol_mean(self) -> float:
        """E[exp(ℓ)] - expected volatility."""
        return self._output.vol_mean
    
    @property
    def log_vol_mean(self) -> float:
        """E[ℓ] - expected log-volatility."""
        return self._output.log_vol_mean
    
    @property
    def log_vol_var(self) -> float:
        """Var[ℓ] - log-volatility variance."""
        return self._output.log_vol_var
    
    @property
    def vol_std(self) -> float:
        """Std[ℓ] - log-volatility standard deviation."""
        return np.sqrt(self._output.log_vol_var)
    
    @property
    def dominant_regime(self) -> int:
        """Most probable regime."""
        return self._output.dominant_regime
    
    @property
    def regime_probs(self) -> np.ndarray:
        """Regime probabilities [n_regimes]."""
        return np.array(self._output.regime_probs[:self._n_regimes])
    
    @property
    def ess(self) -> float:
        """Effective sample size."""
        return self._output.ess
    
    @property
    def surprise(self) -> float:
        """Predictive surprise: -log(marginal_lik)."""
        return self._output.surprise
    
    @property
    def marginal_lik(self) -> float:
        """Marginal likelihood p(y_t | y_{1:t-1})."""
        return self._output.marginal_lik
    
    @property
    def outlier_fraction(self) -> float:
        """Weighted fraction of particles using outlier component."""
        return self._output.outlier_fraction
    
    @property
    def resampled(self) -> bool:
        """True if resampling occurred this tick."""
        return bool(self._output.resampled)
    
    @property
    def regime_entropy(self) -> float:
        """Regime probability entropy (uncertainty measure)."""
        return self._output.regime_entropy
    
    @property
    def regime_changed(self) -> bool:
        """True if regime changed this tick."""
        return bool(self._output.regime_changed)
    
    # Learned parameters
    @property
    def learned_mu_vol(self) -> np.ndarray:
        """Learned μ_vol per regime [n_regimes]."""
        return np.array(self._output.learned_mu_vol[:self._n_regimes])
    
    @property
    def learned_sigma_vol(self) -> np.ndarray:
        """Learned σ_vol per regime [n_regimes]."""
        return np.array(self._output.learned_sigma_vol[:self._n_regimes])
    
    # Smooth signal (if enabled)
    @property
    def smooth_valid(self) -> bool:
        """True if smooth signal is valid (after lag warmup)."""
        return bool(self._output.smooth_valid)
    
    @property
    def vol_mean_smooth(self) -> float:
        """Smoothed volatility (t-K)."""
        return self._output.vol_mean_smooth
    
    @property
    def dominant_regime_smooth(self) -> int:
        """Smoothed dominant regime (t-K)."""
        return self._output.dominant_regime_smooth
    
    # =========================================================================
    # Utility
    # =========================================================================
    
    @property
    def n_particles(self) -> int:
        """Number of particles."""
        return self._n_particles
    
    @property
    def n_regimes(self) -> int:
        """Number of regimes."""
        return self._n_regimes
    
    @property
    def tick_count(self) -> int:
        """Number of ticks processed."""
        return self._tick_count
    
    @property
    def current_lambda(self) -> float:
        """Current forgetting factor."""
        return _lib.rbpf_ext_get_current_lambda(self._handle)
    
    def get_learned_params(self, regime: int) -> Tuple[float, float]:
        """
        Get learned parameters for a regime.
        
        Returns
        -------
        mu_vol, sigma_vol : Tuple[float, float]
        """
        mu = ctypes.c_float()
        sigma = ctypes.c_float()
        _lib.rbpf_ext_get_learned_params(self._handle, regime, 
                                          ctypes.byref(mu), ctypes.byref(sigma))
        return mu.value, sigma.value
    
    def print_config(self):
        """Print current configuration (debug)."""
        _lib.rbpf_ext_print_config(self._handle)
    
    def get_output_dict(self) -> Dict:
        """
        Get all output fields as a dictionary.
        
        Useful for logging or DataFrame construction.
        """
        return {
            'tick': self._tick_count,
            'vol_mean': self.vol_mean,
            'log_vol_mean': self.log_vol_mean,
            'log_vol_var': self.log_vol_var,
            'dominant_regime': self.dominant_regime,
            'regime_probs': self.regime_probs.tolist(),
            'ess': self.ess,
            'surprise': self.surprise,
            'outlier_fraction': self.outlier_fraction,
            'resampled': self.resampled,
            'regime_changed': self.regime_changed,
            'learned_mu_vol': self.learned_mu_vol.tolist(),
            'learned_sigma_vol': self.learned_sigma_vol.tolist(),
            'current_lambda': self.current_lambda,
        }


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == "__main__":
    print("RBPF Python Wrapper Test")
    print("=" * 60)
    
    # Create filter
    rbpf = RBPF(n_particles=512, n_regimes=4, mode=ParamMode.STORVIK)
    print(f"Created RBPF: {rbpf.n_particles} particles, {rbpf.n_regimes} regimes")
    
    # Configure regimes (from test_mmpf_comparison.c)
    rbpf.set_regime_params(0, 0.0030, -4.50, 0.080)  # Calm
    rbpf.set_regime_params(1, 0.0420, -3.67, 0.267)  # Mild
    rbpf.set_regime_params(2, 0.0810, -2.83, 0.453)  # Trend
    rbpf.set_regime_params(3, 0.1200, -2.00, 0.640)  # Crisis
    
    # Transition matrix (stickiness=0.92)
    trans = np.array([
        [0.920, 0.056, 0.020, 0.004],
        [0.032, 0.920, 0.036, 0.012],
        [0.012, 0.036, 0.920, 0.032],
        [0.004, 0.020, 0.056, 0.920]
    ], dtype=np.float32)
    rbpf.set_transition_matrix(trans)
    
    # Enable features
    rbpf.enable_adaptive_forgetting(AdaptSignal.REGIME)
    rbpf.enable_circuit_breaker(0.999, 100)
    rbpf.enable_ocsn()
    rbpf.set_ocsn_params(0, 0.02, 100.0)
    rbpf.set_ocsn_params(1, 0.03, 120.0)
    rbpf.set_ocsn_params(2, 0.04, 140.0)
    rbpf.set_ocsn_params(3, 0.05, 160.0)
    
    # Initialize
    rbpf.init(mu0=-4.5, var0=0.1)
    print("Initialized")
    
    # Generate synthetic returns
    np.random.seed(42)
    n_ticks = 1000
    vol = 0.01  # 1% daily vol
    returns = vol * np.random.randn(n_ticks)
    
    # Add a volatility spike
    returns[500:520] *= 5.0
    
    print(f"\nProcessing {n_ticks} ticks...")
    
    # Process
    for t, ret in enumerate(returns):
        rbpf.step(ret)
        
        if t % 200 == 0 or (500 <= t <= 520):
            print(f"t={t:4d}: vol={rbpf.vol_mean:.4f}, regime={rbpf.dominant_regime}, "
                  f"ess={rbpf.ess:.1f}, surprise={rbpf.surprise:.2f}")
    
    print(f"\nFinal state:")
    print(f"  Tick count: {rbpf.tick_count}")
    print(f"  Circuit breaker trips: {rbpf.circuit_breaker_trips}")
    print(f"  Learned μ_vol: {rbpf.learned_mu_vol}")
    print(f"  Learned σ_vol: {rbpf.learned_sigma_vol}")
    print("\nDone!")
