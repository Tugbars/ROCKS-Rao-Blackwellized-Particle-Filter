# MMPF Real Data Testing - Quick Reference

## Pipeline Overview

```
┌─────────────────┐     ┌─────────────────────┐     ┌──────────────────────┐
│ fetch_market_   │────▶│ test_mmpf_real_     │────▶│ visualize_mmpf_      │
│ data.py         │     │ data.exe            │     │ results.py           │
│                 │     │                     │     │                      │
│ Downloads SPY   │     │ Runs MMPF filter    │     │ Generates plots      │
│ from yfinance   │     │ Outputs CSV         │     │ and statistics       │
└─────────────────┘     └─────────────────────┘     └──────────────────────┘
```

## Step 1: Fetch Real Market Data

```bash
python fetch_market_data.py --ticker SPY
```

**Output:** `mmpf_test_data/` folder with:
- `spy_5year.csv` - 5 years daily data
- `spy_covid_2020.csv` - COVID crash period
- `spy_bear_2022.csv` - 2022 bear market
- `spy_2023_2024.csv` - Recent period + SVB crisis
- `spy_intraday_1m.csv` - Last 7 days, 1-minute bars

**Options:**
```bash
--ticker QQQ          # Different ticker
--multi               # Fetch multiple assets (SPY, QQQ, TLT, GLD, BTC, etc.)
--output ./my_data    # Custom output directory
```

## Step 2: Run MMPF Test

```bash
test_mmpf_real_data.exe mmpf_test_data/spy_5year.csv output_spy_5year.csv --no-learning
```

**Options:**
```bash
--no-learning         # Disable Storvik sync (RECOMMENDED for hypothesis discrimination)
--particles 1024      # More particles (default: 512)
--intraday            # Use intraday volatility scaling
```

**Output CSV columns:**
```
timestamp, return, vol, log_vol, vol_std, w_calm, w_trend, w_crisis, 
dominant, outlier_frac, ess_min, latency_us, event
```

## Step 3: Visualize Results

```bash
python visualize_mmpf_results.py output_spy_5year.csv --output-dir plots
```

**Output:** `plots/` folder with:
- `output_spy_5year_overview.png` - 5-panel overview
- `output_spy_5year_latency.png` - Latency histogram
- `output_spy_5year_COVID_crash.png` - Event-specific analysis (auto-generated)
- `output_spy_5year_Fed_hike_2022.png`
- etc.

**Options:**
```bash
--output-dir ./my_plots   # Custom output directory
--event COVID_crash       # Focus on specific event only
--no-show                 # Don't display interactive plots
```

## Quick Commands

**Full pipeline (5-year SPY):**
```bash
python fetch_market_data.py --ticker SPY
test_mmpf_real_data.exe mmpf_test_data/spy_5year.csv output_spy_5year.csv --no-learning
python visualize_mmpf_results.py output_spy_5year.csv --output-dir plots
```

**COVID crash only:**
```bash
test_mmpf_real_data.exe mmpf_test_data/spy_covid_2020.csv output_covid.csv --no-learning
python visualize_mmpf_results.py output_covid.csv --output-dir plots
```

**Intraday data:**
```bash
test_mmpf_real_data.exe mmpf_test_data/spy_intraday_1m.csv output_intraday.csv --no-learning --intraday
python visualize_mmpf_results.py output_intraday.csv --output-dir plots
```

**High particle count (more accurate, slower):**
```bash
test_mmpf_real_data.exe mmpf_test_data/spy_5year.csv output_hq.csv --no-learning --particles 2048
```

## Visualization Panels

The overview plot shows 5 panels:

1. **Price with Regime Overlay** - Cumulative returns, background colored by dominant regime
2. **Volatility** - MMPF estimate (blue) vs realized vol (red), with ±1σ band
3. **Regime Probabilities** - Stacked area: Calm (green), Trend (orange), Crisis (red)
4. **Outlier Detection** - OCSN outlier fraction (purple)
5. **Filter Health** - Minimum ESS across models (should stay >20)

## Known Volatility Events (auto-labeled)

| Event | Period | Description |
|-------|--------|-------------|
| COVID_crash | Feb 20 - Mar 23, 2020 | Initial crash |
| COVID_recovery | Mar 24 - Apr 30, 2020 | V-shaped bounce |
| Fed_hike_2022 | Jan - Jun 2022 | Rate hike cycle |
| Oct_2022_bottom | Oct 2022 | Bear market bottom |
| SVB_crisis | Mar 8-15, 2023 | Banking crisis |
| Aug_2024_selloff | Aug 1-10, 2024 | Yen carry unwind |

## Tips

- **Use `--no-learning`** for tests - Storvik sync causes hypothesis convergence
- **Check ESS panel** - If ESS drops below 20, filter may be collapsing
- **Outlier spikes** should align with large returns
- **Regime transitions** should be smooth, not noisy flickering
