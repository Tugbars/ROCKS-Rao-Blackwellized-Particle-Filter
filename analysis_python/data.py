"""
RBPF Data Utilities

Fetches market data from yFinance and preprocesses for RBPF consumption.

Usage:
    from data import fetch_returns, fetch_ohlcv, compute_realized_vol
    
    # Simple: get returns ready for RBPF
    returns, dates = fetch_returns("SPY", period="1y")
    
    # Full OHLCV with metadata
    df = fetch_ohlcv("SPY", period="2y", interval="1d")
    
    # Realized volatility for comparison
    rv = compute_realized_vol(returns, window=20)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Union
from datetime import datetime, timedelta
import warnings

try:
    import yfinance as yf
except ImportError:
    raise ImportError("yfinance required: pip install yfinance")


# =============================================================================
# Core Data Fetching
# =============================================================================

def fetch_ohlcv(
    ticker: str,
    period: str = "1y",
    interval: str = "1d",
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch OHLCV data from yFinance.
    
    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol (e.g., "SPY", "AAPL", "BTC-USD")
    period : str
        Data period: "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"
    interval : str
        Data interval: "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo"
    start : str, optional
        Start date (YYYY-MM-DD), overrides period
    end : str, optional
        End date (YYYY-MM-DD)
    
    Returns
    -------
    df : pd.DataFrame
        DataFrame with columns: Open, High, Low, Close, Volume, Adj Close
        Index is DatetimeIndex
    
    Examples
    --------
    >>> df = fetch_ohlcv("SPY", period="1y")
    >>> df = fetch_ohlcv("AAPL", start="2023-01-01", end="2024-01-01")
    >>> df = fetch_ohlcv("BTC-USD", period="6mo", interval="1h")
    """
    tk = yf.Ticker(ticker)
    
    if start is not None:
        df = tk.history(start=start, end=end, interval=interval)
    else:
        df = tk.history(period=period, interval=interval)
    
    if df.empty:
        raise ValueError(f"No data returned for {ticker}")
    
    # Clean up
    df = df.dropna(subset=['Close'])
    
    # Store ticker as attribute
    df.attrs['ticker'] = ticker
    df.attrs['interval'] = interval
    
    return df


def fetch_returns(
    ticker: str,
    period: str = "1y",
    interval: str = "1d",
    log_returns: bool = False,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    """
    Fetch returns ready for RBPF consumption.
    
    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol
    period : str
        Data period (default: "1y")
    interval : str
        Data interval (default: "1d")
    log_returns : bool
        If True, compute log returns. If False, simple returns (default).
    start, end : str, optional
        Date range (overrides period)
    
    Returns
    -------
    returns : np.ndarray
        Array of returns (float32 for RBPF compatibility)
    dates : pd.DatetimeIndex
        Corresponding dates
    
    Examples
    --------
    >>> returns, dates = fetch_returns("SPY", period="1y")
    >>> for ret in returns:
    ...     rbpf.step(ret)
    """
    df = fetch_ohlcv(ticker, period=period, interval=interval, start=start, end=end)
    
    prices = df['Close'].values
    
    if log_returns:
        # Log returns: r_t = log(P_t / P_{t-1})
        returns = np.diff(np.log(prices))
    else:
        # Simple returns: r_t = (P_t - P_{t-1}) / P_{t-1}
        returns = np.diff(prices) / prices[:-1]
    
    # Drop NaN/Inf
    valid = np.isfinite(returns)
    if not valid.all():
        n_invalid = (~valid).sum()
        warnings.warn(f"Dropped {n_invalid} invalid returns")
        returns = returns[valid]
        dates = df.index[1:][valid]
    else:
        dates = df.index[1:]
    
    return returns.astype(np.float32), dates


def fetch_multi(
    tickers: List[str],
    period: str = "1y",
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Fetch adjusted close prices for multiple tickers.
    
    Returns DataFrame with tickers as columns.
    """
    dfs = []
    for ticker in tickers:
        try:
            df = fetch_ohlcv(ticker, period=period, interval=interval)
            dfs.append(df['Close'].rename(ticker))
        except Exception as e:
            warnings.warn(f"Failed to fetch {ticker}: {e}")
    
    return pd.concat(dfs, axis=1).dropna()


# =============================================================================
# Realized Volatility
# =============================================================================

def compute_realized_vol(
    returns: np.ndarray,
    window: int = 20,
    annualize: bool = True,
    trading_days: int = 252,
) -> np.ndarray:
    """
    Compute rolling realized volatility.
    
    Parameters
    ----------
    returns : np.ndarray
        Array of returns
    window : int
        Rolling window size (default: 20)
    annualize : bool
        If True, annualize volatility (default: True)
    trading_days : int
        Trading days per year for annualization (default: 252)
    
    Returns
    -------
    rv : np.ndarray
        Rolling realized volatility (same length as returns, NaN for warmup)
    """
    n = len(returns)
    rv = np.full(n, np.nan)
    
    for i in range(window - 1, n):
        rv[i] = np.std(returns[i - window + 1:i + 1], ddof=1)
    
    if annualize:
        rv *= np.sqrt(trading_days)
    
    return rv


def compute_realized_vol_exp(
    returns: np.ndarray,
    halflife: int = 20,
    annualize: bool = True,
    trading_days: int = 252,
) -> np.ndarray:
    """
    Compute exponentially weighted realized volatility.
    
    Uses EWMA variance, which is what RBPF is effectively estimating.
    
    Parameters
    ----------
    returns : np.ndarray
        Array of returns
    halflife : int
        Halflife in periods (default: 20)
    annualize : bool
        If True, annualize (default: True)
    trading_days : int
        Trading days per year (default: 252)
    
    Returns
    -------
    rv : np.ndarray
        EWMA volatility
    """
    alpha = 1 - np.exp(-np.log(2) / halflife)
    
    n = len(returns)
    var = np.zeros(n)
    var[0] = returns[0] ** 2
    
    for i in range(1, n):
        var[i] = alpha * returns[i] ** 2 + (1 - alpha) * var[i - 1]
    
    rv = np.sqrt(var)
    
    if annualize:
        rv *= np.sqrt(trading_days)
    
    return rv


# =============================================================================
# Data Preprocessing
# =============================================================================

def winsorize_returns(
    returns: np.ndarray,
    sigma: float = 5.0,
) -> np.ndarray:
    """
    Winsorize extreme returns to ±N sigma.
    
    Useful for backtesting to prevent outliers from dominating.
    
    Parameters
    ----------
    returns : np.ndarray
        Raw returns
    sigma : float
        Clip at ±sigma standard deviations (default: 5.0)
    
    Returns
    -------
    clipped : np.ndarray
        Winsorized returns
    """
    std = np.std(returns)
    mean = np.mean(returns)
    lower = mean - sigma * std
    upper = mean + sigma * std
    return np.clip(returns, lower, upper)


def detect_outliers(
    returns: np.ndarray,
    sigma: float = 5.0,
) -> np.ndarray:
    """
    Detect outlier returns.
    
    Returns boolean mask where True = outlier.
    """
    std = np.std(returns)
    mean = np.mean(returns)
    z_scores = np.abs((returns - mean) / std)
    return z_scores > sigma


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add useful features to OHLCV DataFrame.
    
    Adds:
    - returns: Simple returns
    - log_returns: Log returns
    - realized_vol_20: 20-day rolling volatility
    - realized_vol_60: 60-day rolling volatility
    - range_pct: (High - Low) / Close (intraday range)
    - gap: Overnight gap (Open / prev Close - 1)
    """
    df = df.copy()
    
    # Returns
    df['returns'] = df['Close'].pct_change()
    df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Realized volatility
    df['realized_vol_20'] = df['returns'].rolling(20).std() * np.sqrt(252)
    df['realized_vol_60'] = df['returns'].rolling(60).std() * np.sqrt(252)
    
    # Intraday range
    df['range_pct'] = (df['High'] - df['Low']) / df['Close']
    
    # Overnight gap
    df['gap'] = df['Open'] / df['Close'].shift(1) - 1
    
    return df


# =============================================================================
# CSV I/O
# =============================================================================

def save_returns_csv(
    ticker: str,
    returns: np.ndarray,
    dates: pd.DatetimeIndex,
    filepath: str,
):
    """Save returns to CSV for reproducibility."""
    df = pd.DataFrame({
        'date': dates,
        'return': returns,
    })
    df.to_csv(filepath, index=False)
    print(f"Saved {len(returns)} returns to {filepath}")


def load_returns_csv(filepath: str) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    """Load returns from CSV."""
    df = pd.read_csv(filepath, parse_dates=['date'])
    returns = df['return'].values.astype(np.float32)
    dates = pd.DatetimeIndex(df['date'])
    return returns, dates


def save_rbpf_results(
    results: List[dict],
    filepath: str,
):
    """
    Save RBPF results to CSV.
    
    Parameters
    ----------
    results : List[dict]
        List of dicts from rbpf.get_output_dict()
    filepath : str
        Output CSV path
    """
    df = pd.DataFrame(results)
    
    # Flatten list columns
    for col in ['regime_probs', 'learned_mu_vol', 'learned_sigma_vol']:
        if col in df.columns:
            # Expand list to multiple columns
            expanded = pd.DataFrame(df[col].tolist())
            expanded.columns = [f"{col}_{i}" for i in range(len(expanded.columns))]
            df = pd.concat([df.drop(columns=[col]), expanded], axis=1)
    
    df.to_csv(filepath, index=False)
    print(f"Saved {len(results)} ticks to {filepath}")


# =============================================================================
# Quick Info
# =============================================================================

def ticker_info(ticker: str) -> dict:
    """Get basic info about a ticker."""
    tk = yf.Ticker(ticker)
    info = tk.info
    return {
        'symbol': info.get('symbol', ticker),
        'name': info.get('shortName', info.get('longName', '')),
        'currency': info.get('currency', ''),
        'exchange': info.get('exchange', ''),
        'market_cap': info.get('marketCap', 0),
        'sector': info.get('sector', ''),
        'industry': info.get('industry', ''),
    }


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == "__main__":
    print("RBPF Data Utilities Test")
    print("=" * 60)
    
    # Fetch SPY
    ticker = "SPY"
    print(f"\nFetching {ticker}...")
    
    returns, dates = fetch_returns(ticker, period="6mo")
    print(f"  Returns: {len(returns)}")
    print(f"  Date range: {dates[0].date()} to {dates[-1].date()}")
    print(f"  Mean: {returns.mean():.6f}")
    print(f"  Std: {returns.std():.6f}")
    print(f"  Min: {returns.min():.6f}")
    print(f"  Max: {returns.max():.6f}")
    
    # Realized vol
    rv = compute_realized_vol(returns, window=20, annualize=True)
    print(f"\n  Realized Vol (20d, annualized):")
    print(f"    Current: {rv[-1]:.2%}")
    print(f"    Mean: {np.nanmean(rv):.2%}")
    
    # Outliers
    outliers = detect_outliers(returns, sigma=3.0)
    print(f"\n  Outliers (>3σ): {outliers.sum()}")
    
    # Full OHLCV with features
    df = fetch_ohlcv(ticker, period="6mo")
    df = add_features(df)
    print(f"\n  DataFrame shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    
    print("\nDone!")
