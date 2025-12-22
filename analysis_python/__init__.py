"""
RBPF Python Analysis Tools

Modules:
    rbpf      - Core filter wrapper (ctypes)
    data      - yFinance data fetching
    tuner     - Parameter optimization
    compare   - Model comparison

Example:
    from rbpf import RBPF, ParamMode
    from data import fetch_returns, compute_realized_vol
    from tuner import quick_tune
    from compare import quick_compare
"""

from .rbpf import RBPF, ParamMode, AdaptSignal
from .data import fetch_returns, fetch_ohlcv, compute_realized_vol
from .tuner import GridSearchTuner, quick_tune, TunerConfig
from .compare import ModelComparison, quick_compare

__all__ = [
    # Core
    'RBPF', 'ParamMode', 'AdaptSignal',
    # Data
    'fetch_returns', 'fetch_ohlcv', 'compute_realized_vol',
    # Tuner
    'GridSearchTuner', 'quick_tune', 'TunerConfig',
    # Compare
    'ModelComparison', 'quick_compare',
]
