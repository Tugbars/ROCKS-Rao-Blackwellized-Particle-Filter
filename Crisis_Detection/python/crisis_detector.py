"""
crisis_detector.py - Python bindings for Crisis Detector

Usage:
    from crisis_detector import CrisisDetector, CrisisState
    
    cd = CrisisDetector()
    
    for t, ret in enumerate(returns):
        state = cd.update(ret, t)
        if state == CrisisState.ACTIVE:
            print(f"Crisis at tick {t}!")
    
    # Get full history for plotting
    df = cd.to_dataframe()
"""

import ctypes
import numpy as np
from pathlib import Path
from enum import IntEnum
from dataclasses import dataclass
from typing import Optional, List, Tuple
import os
import sys

# ═══════════════════════════════════════════════════════════════════════════════
# WINDOWS MKL DLL PATHS (must be set BEFORE loading the library)
# ═══════════════════════════════════════════════════════════════════════════════

if sys.platform == "win32":
    mkl_paths = [
        r"C:\Program Files (x86)\Intel\oneAPI\mkl\latest\bin",
        r"C:\Program Files (x86)\Intel\oneAPI\mkl\latest\redist\intel64",
        r"C:\Program Files (x86)\Intel\oneAPI\compiler\latest\bin",
        r"C:\Program Files\Intel\oneAPI\mkl\latest\bin",
        r"C:\Program Files\Intel\oneAPI\mkl\latest\redist\intel64",
        r"C:\Program Files\Intel\oneAPI\compiler\latest\bin",
    ]
    for p in mkl_paths:
        if os.path.exists(p):
            try:
                os.add_dll_directory(p)
            except (OSError, AttributeError):
                pass  # add_dll_directory not available on older Python

# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════════

class CrisisState(IntEnum):
    IDLE = 0
    ALERT = 1
    ACTIVE = 2
    RECOVERING = 3

class HawkesTriggerState(IntEnum):
    IDLE = 0
    ARMED = 1
    FIRED = 2
    REFRACTORY = 3

# ═══════════════════════════════════════════════════════════════════════════════
# C STRUCTURES (must match crisis_detector.h)
# ═══════════════════════════════════════════════════════════════════════════════

class EventDetectorConfig(ctypes.Structure):
    _fields_ = [
        ("return_quantile", ctypes.c_double),
        ("volume_multiple", ctypes.c_double),
        ("imbalance_multiple", ctypes.c_double),
        ("ema_alpha", ctypes.c_double),
        ("warmup_ticks", ctypes.c_int),
    ]

class HawkesParams(ctypes.Structure):
    _fields_ = [
        ("mu", ctypes.c_float),
        ("alpha", ctypes.c_float),
        ("beta", ctypes.c_float),
        ("event_threshold", ctypes.c_float),
    ]

class HawkesIntegratorConfig(ctypes.Structure):
    _fields_ = [
        ("hawkes", HawkesParams),
        ("window_size", ctypes.c_int),
        ("ema_alpha", ctypes.c_float),
        ("sigma_floor", ctypes.c_float),
        ("sigma_cap", ctypes.c_float),
        ("residual_decay", ctypes.c_float),
        ("residual_threshold", ctypes.c_float),
        ("high_water_mark", ctypes.c_float),
        ("low_water_mark", ctypes.c_float),
        ("min_ticks_armed", ctypes.c_int),
        ("absolute_panic_intensity", ctypes.c_float),
        ("instant_spike_multiplier", ctypes.c_float),
        ("use_absolute_panic", ctypes.c_bool),
        ("refractory_ticks", ctypes.c_int),
        ("warmup_ticks", ctypes.c_int),
    ]

class SRConfig(ctypes.Structure):
    _fields_ = [
        ("sigma_multiple", ctypes.c_float),
        ("winsorize_cap", ctypes.c_float),
        ("log_sr_clamp", ctypes.c_float),
        ("student_nu", ctypes.c_float),
    ]

class AdaptiveThresholdConfig(ctypes.Structure):
    _fields_ = [
        ("log_H_base", ctypes.c_float),
        ("inertia_scale", ctypes.c_float),
        ("wolf_penalty", ctypes.c_float),
        ("max_wolf_count", ctypes.c_int),
    ]

class CrisisDetectorConfig(ctypes.Structure):
    _fields_ = [
        ("event_cfg", EventDetectorConfig),
        ("hawkes_cfg", HawkesIntegratorConfig),
        ("sr_cfg", SRConfig),
        ("threshold_cfg", AdaptiveThresholdConfig),
        ("sr_alert_fraction", ctypes.c_float),
        ("sr_exit_threshold", ctypes.c_float),
        ("sr_reentry_fraction", ctypes.c_float),
        ("confirmation_ticks", ctypes.c_int),
        ("min_hold_active", ctypes.c_int),
        ("cooldown_ticks", ctypes.c_int),
        ("nuclear_override", ctypes.c_float),
        ("sigma_crisis_ema_alpha", ctypes.c_float),
        ("initial_sigma_peace", ctypes.c_float),
        ("warmup_ticks", ctypes.c_int),
        # Robust warmup (v3)
        ("use_robust_warmup", ctypes.c_bool),
        ("warmup_outlier_k", ctypes.c_float),
        ("warmup_winsorize_k", ctypes.c_float),
        ("warmup_min_clean", ctypes.c_int),
        # Sanity anchor (v3.1)
        ("max_peace_sigma", ctypes.c_float),
    ]

# Opaque structure - we just need a pointer
class CrisisDetectorOpaque(ctypes.Structure):
    _fields_ = [("_opaque", ctypes.c_byte * 65536)]  # Large enough

# ═══════════════════════════════════════════════════════════════════════════════
# LIBRARY LOADER
# ═══════════════════════════════════════════════════════════════════════════════

def _load_library():
    """Load the shared library from various possible locations."""
    import platform
    
    # Platform-specific library name
    if platform.system() == "Windows":
        lib_names = ["libcrisis.dll", "crisis.dll"]
    elif platform.system() == "Darwin":
        lib_names = ["libcrisis.dylib", "libcrisis.so"]
    else:
        lib_names = ["libcrisis.so"]
    
    # Search paths
    search_dirs = [
        Path(__file__).parent,
        Path.cwd(),
        Path.cwd() / "Release",
        Path.cwd() / "Debug",
    ]
    
    # Build full search paths
    search_paths = []
    for dir_path in search_dirs:
        for lib_name in lib_names:
            search_paths.append(dir_path / lib_name)
    
    # Try each path
    for path in search_paths:
        if path.exists():
            try:
                return ctypes.CDLL(str(path))
            except OSError as e:
                print(f"Warning: Found {path} but failed to load: {e}")
                continue
    
    # Try system path
    for lib_name in lib_names:
        try:
            return ctypes.CDLL(lib_name)
        except OSError:
            pass
    
    raise RuntimeError(
        f"Could not find crisis detection library.\n"
        f"Searched: {[str(p) for p in search_paths]}\n"
        f"On Windows: place libcrisis.dll in the same folder as this script.\n"
        f"On Linux: compile with: gcc -O3 -fPIC -shared crisis_detector.c "
        f"event_detector.c sr_detector.c hawkes_integrator.c -lm -o libcrisis.so"
    )

_lib = None

def _get_lib():
    global _lib
    if _lib is None:
        _lib = _load_library()
        _setup_functions(_lib)
    return _lib

def _setup_functions(lib):
    """Set up function signatures."""
    # crisis_detector_config_default
    lib.crisis_detector_config_default.argtypes = []
    lib.crisis_detector_config_default.restype = CrisisDetectorConfig
    
    # crisis_detector_init
    lib.crisis_detector_init.argtypes = [
        ctypes.POINTER(CrisisDetectorOpaque),
        ctypes.POINTER(CrisisDetectorConfig)
    ]
    lib.crisis_detector_init.restype = ctypes.c_int
    
    # crisis_detector_reset
    lib.crisis_detector_reset.argtypes = [ctypes.POINTER(CrisisDetectorOpaque)]
    lib.crisis_detector_reset.restype = None
    
    # crisis_detector_free
    lib.crisis_detector_free.argtypes = [ctypes.POINTER(CrisisDetectorOpaque)]
    lib.crisis_detector_free.restype = None
    
    # crisis_detector_update
    lib.crisis_detector_update.argtypes = [
        ctypes.POINTER(CrisisDetectorOpaque),
        ctypes.c_float,
        ctypes.c_int64
    ]
    lib.crisis_detector_update.restype = ctypes.c_int
    
    # crisis_detector_get_sigma_peace
    lib.crisis_detector_get_sigma_peace.argtypes = [ctypes.POINTER(CrisisDetectorOpaque)]
    lib.crisis_detector_get_sigma_peace.restype = ctypes.c_float
    
    # crisis_detector_set_sigma_peace
    lib.crisis_detector_set_sigma_peace.argtypes = [
        ctypes.POINTER(CrisisDetectorOpaque),
        ctypes.c_float
    ]
    lib.crisis_detector_set_sigma_peace.restype = None

# ═══════════════════════════════════════════════════════════════════════════════
# RESULT DATACLASS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TickResult:
    """Result from a single tick update."""
    tick: int
    obs: float
    state: CrisisState
    log_sr_up: float
    log_sr_down: float
    log_H: float
    hawkes_intensity: float
    hawkes_surprise: float
    sigma_peace: float
    sigma_crisis: float
    is_event: bool

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class CrisisDetector:
    """
    Python wrapper for the C crisis detector.
    
    Example:
        cd = CrisisDetector()
        
        # Process returns
        for t, ret in enumerate(returns):
            state = cd.update(ret, t)
        
        # Get results as DataFrame
        df = cd.to_dataframe()
        
        # Plot
        cd.plot()
    """
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize crisis detector.
        
        Args:
            config: Optional dict with config overrides:
                - confirmation_ticks: int
                - min_hold_active: int
                - cooldown_ticks: int
                - nuclear_override: float
                - warmup_ticks: int
                - initial_sigma_peace: float
                - use_robust_warmup: bool (default True)
                - warmup_outlier_k: float (default 3.0)
                - warmup_winsorize_k: float (default 4.0)
                - warmup_min_clean: int (default 50)
                - max_peace_sigma: float (sanity anchor, default 0.015)
        """
        self._lib = _get_lib()
        self._detector = CrisisDetectorOpaque()
        
        # Get default config
        cfg = self._lib.crisis_detector_config_default()
        
        # Apply overrides
        if config:
            for key, value in config.items():
                if hasattr(cfg, key):
                    setattr(cfg, key, value)
        
        # Initialize
        ret = self._lib.crisis_detector_init(ctypes.byref(self._detector), ctypes.byref(cfg))
        if ret != 0:
            raise RuntimeError("Failed to initialize crisis detector")
        
        # History for plotting
        self._history: List[TickResult] = []
        self._observations: List[float] = []
    
    def __del__(self):
        if hasattr(self, '_lib') and hasattr(self, '_detector'):
            self._lib.crisis_detector_free(ctypes.byref(self._detector))
    
    def reset(self):
        """Reset detector state."""
        self._lib.crisis_detector_reset(ctypes.byref(self._detector))
        self._history.clear()
        self._observations.clear()
    
    def update(self, obs: float, tick: int) -> CrisisState:
        """
        Process one observation.
        
        Args:
            obs: Return observation
            tick: Tick number
            
        Returns:
            Current crisis state
        """
        state_int = self._lib.crisis_detector_update(
            ctypes.byref(self._detector),
            ctypes.c_float(obs),
            ctypes.c_int64(tick)
        )
        state = CrisisState(state_int)
        
        # Read internal state for history
        # We need to access the struct fields directly
        # This is a simplified version - for full access we'd need to expose more C functions
        sigma_peace = self._lib.crisis_detector_get_sigma_peace(ctypes.byref(self._detector))
        
        result = TickResult(
            tick=tick,
            obs=obs,
            state=state,
            log_sr_up=0.0,  # Would need C accessor
            log_sr_down=0.0,
            log_H=0.0,
            hawkes_intensity=0.0,
            hawkes_surprise=0.0,
            sigma_peace=sigma_peace,
            sigma_crisis=0.0,
            is_event=False,
        )
        
        self._history.append(result)
        self._observations.append(obs)
        
        return state
    
    def process_batch(self, returns: np.ndarray) -> np.ndarray:
        """
        Process a batch of returns.
        
        Args:
            returns: Array of returns
            
        Returns:
            Array of states (as integers)
        """
        states = np.zeros(len(returns), dtype=np.int32)
        for t, ret in enumerate(returns):
            states[t] = self.update(float(ret), t)
        return states
    
    def set_sigma_peace(self, sigma: float):
        """Set peace sigma (from PGAS)."""
        self._lib.crisis_detector_set_sigma_peace(
            ctypes.byref(self._detector),
            ctypes.c_float(sigma)
        )
    
    @property
    def sigma_peace(self) -> float:
        """Get current peace sigma."""
        return self._lib.crisis_detector_get_sigma_peace(ctypes.byref(self._detector))
    
    def to_dataframe(self):
        """Convert history to pandas DataFrame."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas required for to_dataframe()")
        
        return pd.DataFrame([
            {
                'tick': r.tick,
                'obs': r.obs,
                'state': r.state.value,
                'state_name': r.state.name,
                'sigma_peace': r.sigma_peace,
            }
            for r in self._history
        ])
    
    def get_crisis_regions(self) -> List[Tuple[int, int]]:
        """Get list of (start, end) tick pairs for crisis regions."""
        regions = []
        in_crisis = False
        start = 0
        
        for r in self._history:
            if r.state in (CrisisState.ACTIVE, CrisisState.RECOVERING):
                if not in_crisis:
                    start = r.tick
                    in_crisis = True
            else:
                if in_crisis:
                    regions.append((start, r.tick))
                    in_crisis = False
        
        # Handle case where we end in crisis
        if in_crisis and self._history:
            regions.append((start, self._history[-1].tick))
        
        return regions
    
    def plot(self, figsize=(14, 8), title="Crisis Detection"):
        """
        Plot returns with crisis regions highlighted.
        
        Args:
            figsize: Figure size tuple
            title: Plot title
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            raise ImportError("matplotlib required for plot()")
        
        if not self._history:
            print("No data to plot. Run update() first.")
            return
        
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
        
        ticks = [r.tick for r in self._history]
        obs = [r.obs for r in self._history]
        states = [r.state.value for r in self._history]
        
        # Plot 1: Returns with crisis regions
        ax1 = axes[0]
        ax1.plot(ticks, obs, 'b-', linewidth=0.5, alpha=0.7, label='Returns')
        
        # Highlight crisis regions
        crisis_regions = self.get_crisis_regions()
        for start, end in crisis_regions:
            ax1.axvspan(start, end, alpha=0.3, color='red', label='Crisis')
        
        ax1.set_ylabel('Returns')
        ax1.set_title(title)
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Cumulative returns
        ax2 = axes[1]
        cum_returns = np.cumsum(obs)
        ax2.plot(ticks, cum_returns, 'g-', linewidth=1, label='Cumulative Returns')
        
        for start, end in crisis_regions:
            ax2.axvspan(start, end, alpha=0.3, color='red')
        
        ax2.set_ylabel('Cumulative Returns')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: State
        ax3 = axes[2]
        colors = {0: 'green', 1: 'yellow', 2: 'red', 3: 'orange'}
        state_colors = [colors[s] for s in states]
        ax3.scatter(ticks, states, c=state_colors, s=1, alpha=0.5)
        ax3.set_ylabel('State')
        ax3.set_xlabel('Tick')
        ax3.set_yticks([0, 1, 2, 3])
        ax3.set_yticklabels(['IDLE', 'ALERT', 'ACTIVE', 'RECOVERING'])
        ax3.grid(True, alpha=0.3)
        
        # Legend
        patches = [
            mpatches.Patch(color='green', label='IDLE'),
            mpatches.Patch(color='yellow', label='ALERT'),
            mpatches.Patch(color='red', label='ACTIVE'),
            mpatches.Patch(color='orange', label='RECOVERING'),
        ]
        ax3.legend(handles=patches, loc='upper right')
        
        plt.tight_layout()
        return fig, axes


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_market(n_ticks: int, sigma_peace: float = 0.01, 
                    crisis_start: int = None, crisis_duration: int = 100,
                    sigma_crisis: float = 0.05, seed: int = None) -> np.ndarray:
    """
    Simulate market returns with optional crisis period.
    
    Args:
        n_ticks: Number of ticks
        sigma_peace: Normal volatility
        crisis_start: Tick to start crisis (None for no crisis)
        crisis_duration: Length of crisis
        sigma_crisis: Crisis volatility
        seed: Random seed
        
    Returns:
        Array of returns
    """
    if seed is not None:
        np.random.seed(seed)
    
    returns = np.random.randn(n_ticks) * sigma_peace
    
    if crisis_start is not None:
        crisis_end = min(crisis_start + crisis_duration, n_ticks)
        returns[crisis_start:crisis_end] = np.random.randn(crisis_end - crisis_start) * sigma_crisis
    
    return returns


def demo():
    """Run a quick demo."""
    print("Crisis Detector Demo")
    print("=" * 50)
    
    # Simulate market with flash crash
    returns = simulate_market(
        n_ticks=2000,
        sigma_peace=0.01,
        crisis_start=500,
        crisis_duration=150,
        sigma_crisis=0.05,
        seed=42
    )
    
    # Run detector
    cd = CrisisDetector()
    states = cd.process_batch(returns)
    
    # Summary
    unique, counts = np.unique(states, return_counts=True)
    print("\nState distribution:")
    for s, c in zip(unique, counts):
        print(f"  {CrisisState(s).name}: {c} ticks ({100*c/len(states):.1f}%)")
    
    # Crisis regions
    regions = cd.get_crisis_regions()
    print(f"\nCrisis regions detected: {len(regions)}")
    for start, end in regions:
        print(f"  Ticks {start} - {end} ({end - start} ticks)")
    
    print("\nTrue crisis: ticks 500 - 650")
    
    # Plot if matplotlib available
    try:
        cd.plot()
        import matplotlib.pyplot as plt
        plt.savefig('crisis_demo.png', dpi=150)
        print("\nPlot saved to crisis_demo.png")
    except ImportError:
        print("\nInstall matplotlib to see plots: pip install matplotlib")


if __name__ == "__main__":
    demo()