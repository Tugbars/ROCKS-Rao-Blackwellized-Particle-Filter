#!/usr/bin/env python3
"""
RBPF Performance Visualization

Generates publication-quality figures from rbpf_viz_data.csv

Usage:
    python rbpf_viz_plots.py [csv_file]

Output:
    - rbpf_fig1_volatility_tracking.png
    - rbpf_fig2_regime_detection.png
    - rbpf_fig3_flash_crash.png
    - rbpf_fig4_crisis_persistence.png
    - rbpf_fig5_all_scenarios.png
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import sys

# Style configuration
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10

# Colors
COLORS = {
    'true_vol': '#2c3e50',      # Dark blue-gray
    'est_vol': '#3498db',       # Bright blue
    'calm': '#27ae60',          # Green
    'trend': '#f39c12',         # Orange
    'crisis': '#e74c3c',        # Red
    'outlier': '#9b59b6',       # Purple
    'error_band': '#bdc3c7',    # Light gray
}

HYPO_COLORS = [COLORS['calm'], COLORS['trend'], COLORS['crisis']]
HYPO_NAMES = ['CALM', 'TREND', 'CRISIS']

SCENARIOS = [
    ('Extended_Calm', 0, 1500),
    ('Slow_Trend', 1500, 2500),
    ('Sudden_Crisis', 2500, 3000),
    ('Crisis_Persist', 3000, 4000),
    ('Recovery', 4000, 5200),
    ('Flash_Crash', 5200, 5700),
    ('Choppy', 5700, 8000),
]


def load_data(csv_file='rbpf_viz_data.csv'):
    """Load and prepare the CSV data."""
    df = pd.read_csv(csv_file)
    return df


def add_scenario_shading(ax, df, alpha=0.1):
    """Add light background shading for each scenario."""
    colors = ['#ecf0f1', '#bdc3c7'] * 4
    for i, (name, start, end) in enumerate(SCENARIOS):
        ax.axvspan(start, end, alpha=alpha, color=colors[i % 2], zorder=0)
        # Add scenario label at top
        mid = (start + end) / 2
        ax.text(mid, ax.get_ylim()[1] * 0.98, name.replace('_', '\n'),
                ha='center', va='top', fontsize=7, alpha=0.7)


def fig1_volatility_tracking(df, save=True):
    """Panel 1: True vs Estimated Volatility across all 8000 ticks."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # Top: Volatility
    ax1 = axes[0]
    ax1.plot(df['tick'], df['true_vol'] * 100, color=COLORS['true_vol'], 
             linewidth=0.8, label='True σ', alpha=0.8)
    ax1.plot(df['tick'], df['est_vol'] * 100, color=COLORS['est_vol'], 
             linewidth=0.8, label='Estimated σ', alpha=0.8)
    
    # Mark outliers
    outliers = df[df['is_outlier'] == 1]
    ax1.scatter(outliers['tick'], outliers['true_vol'] * 100, 
                color=COLORS['outlier'], s=50, marker='x', zorder=5, label='Outlier')
    
    ax1.set_ylabel('Volatility (%)')
    ax1.set_title('RBPF Volatility Tracking (Best Vol RMSE Config)', fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.set_ylim(0, None)
    add_scenario_shading(ax1, df, alpha=0.08)
    
    # Bottom: Log-volatility with uncertainty band
    ax2 = axes[1]
    ax2.fill_between(df['tick'], 
                     df['est_log_vol'] - 2*df['est_log_vol_std'],
                     df['est_log_vol'] + 2*df['est_log_vol_std'],
                     color=COLORS['error_band'], alpha=0.5, label='±2σ band')
    ax2.plot(df['tick'], df['true_log_vol'], color=COLORS['true_vol'], 
             linewidth=0.8, label='True log(σ)', alpha=0.8)
    ax2.plot(df['tick'], df['est_log_vol'], color=COLORS['est_vol'], 
             linewidth=0.8, label='Estimated log(σ)', alpha=0.8)
    
    ax2.set_xlabel('Tick')
    ax2.set_ylabel('Log-Volatility')
    ax2.legend(loc='upper right')
    add_scenario_shading(ax2, df, alpha=0.08)
    
    plt.tight_layout()
    if save:
        plt.savefig('rbpf_fig1_volatility_tracking.png', bbox_inches='tight')
        print("Saved: rbpf_fig1_volatility_tracking.png")
    return fig


def fig2_regime_detection(df, save=True):
    """Panel 2: True vs Estimated Regime (Hypothesis)."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    
    # Top: True regime
    ax1 = axes[0]
    for hypo in range(3):
        mask = df['true_hypo'] == hypo
        ax1.fill_between(df['tick'], 0, 1, where=mask, 
                        color=HYPO_COLORS[hypo], alpha=0.7, label=HYPO_NAMES[hypo])
    ax1.set_ylabel('True')
    ax1.set_yticks([])
    ax1.set_title('Regime Detection: True vs Estimated', fontweight='bold')
    ax1.legend(loc='upper right', ncol=3)
    
    # Middle: Estimated regime
    ax2 = axes[1]
    for hypo in range(3):
        mask = df['est_hypo'] == hypo
        ax2.fill_between(df['tick'], 0, 1, where=mask,
                        color=HYPO_COLORS[hypo], alpha=0.7)
    ax2.set_ylabel('Estimated')
    ax2.set_yticks([])
    
    # Bottom: Regime probabilities
    ax3 = axes[2]
    # Stack the regime probabilities (combining R0+R1 for CALM)
    calm_prob = df['regime_prob_0'] + df['regime_prob_1']
    trend_prob = df['regime_prob_2']
    crisis_prob = df['regime_prob_3']
    
    ax3.fill_between(df['tick'], 0, calm_prob, color=HYPO_COLORS[0], alpha=0.7, label='P(CALM)')
    ax3.fill_between(df['tick'], calm_prob, calm_prob + trend_prob, 
                     color=HYPO_COLORS[1], alpha=0.7, label='P(TREND)')
    ax3.fill_between(df['tick'], calm_prob + trend_prob, 1, 
                     color=HYPO_COLORS[2], alpha=0.7, label='P(CRISIS)')
    
    ax3.set_xlabel('Tick')
    ax3.set_ylabel('Probability')
    ax3.set_ylim(0, 1)
    ax3.legend(loc='upper right', ncol=3)
    
    plt.tight_layout()
    if save:
        plt.savefig('rbpf_fig2_regime_detection.png', bbox_inches='tight')
        print("Saved: rbpf_fig2_regime_detection.png")
    return fig


def fig3_flash_crash(df, save=True):
    """Panel 3: Zoom on Flash Crash scenario (5200-5700)."""
    zoom = df[(df['tick'] >= 5150) & (df['tick'] <= 5750)].copy()
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    
    # Top: Returns with outlier marked
    ax1 = axes[0]
    ax1.plot(zoom['tick'], zoom['return'] * 100, color='#34495e', linewidth=0.8)
    outliers = zoom[zoom['is_outlier'] == 1]
    ax1.scatter(outliers['tick'], outliers['return'] * 100, 
                color=COLORS['outlier'], s=100, marker='v', zorder=5, label='12σ Outlier')
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Return (%)')
    ax1.set_title('Flash Crash: CALM → CRISIS (60 ticks) → CALM', fontweight='bold')
    ax1.legend()
    
    # Flash crash zone
    ax1.axvspan(5350, 5410, alpha=0.2, color=COLORS['crisis'], label='Crisis Zone')
    
    # Middle: Volatility
    ax2 = axes[1]
    ax2.plot(zoom['tick'], zoom['true_vol'] * 100, color=COLORS['true_vol'], 
             linewidth=1.5, label='True σ')
    ax2.plot(zoom['tick'], zoom['est_vol'] * 100, color=COLORS['est_vol'], 
             linewidth=1.5, label='Estimated σ')
    ax2.axvspan(5350, 5410, alpha=0.2, color=COLORS['crisis'])
    ax2.set_ylabel('Volatility (%)')
    ax2.legend()
    
    # Bottom: Regime probabilities
    ax3 = axes[2]
    calm_prob = zoom['regime_prob_0'] + zoom['regime_prob_1']
    crisis_prob = zoom['regime_prob_3']
    
    ax3.plot(zoom['tick'], calm_prob, color=HYPO_COLORS[0], linewidth=2, label='P(CALM)')
    ax3.plot(zoom['tick'], crisis_prob, color=HYPO_COLORS[2], linewidth=2, label='P(CRISIS)')
    ax3.axvspan(5350, 5410, alpha=0.2, color=COLORS['crisis'])
    ax3.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Tick')
    ax3.set_ylabel('Probability')
    ax3.set_ylim(0, 1)
    ax3.legend()
    
    plt.tight_layout()
    if save:
        plt.savefig('rbpf_fig3_flash_crash.png', bbox_inches='tight')
        print("Saved: rbpf_fig3_flash_crash.png")
    return fig


def fig4_crisis_persistence(df, save=True):
    """Panel 4: Zoom on Crisis Persistence with extreme outlier."""
    zoom = df[(df['tick'] >= 2900) & (df['tick'] <= 4100)].copy()
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    
    # Top: Returns
    ax1 = axes[0]
    ax1.plot(zoom['tick'], zoom['return'] * 100, color='#34495e', linewidth=0.5, alpha=0.8)
    outliers = zoom[zoom['is_outlier'] == 1]
    ax1.scatter(outliers['tick'], outliers['return'] * 100,
                color=COLORS['outlier'], s=80, marker='v', zorder=5)
    
    # Label the 15σ extreme
    extreme = zoom[zoom['outlier_sigma'] == 15.0]
    if len(extreme) > 0:
        ax1.annotate('15σ', (extreme['tick'].values[0], extreme['return'].values[0] * 100),
                    xytext=(10, 10), textcoords='offset points', fontsize=9,
                    arrowprops=dict(arrowstyle='->', color=COLORS['outlier']))
    
    ax1.set_ylabel('Return (%)')
    ax1.set_title('Crisis Persistence: Sustained CRISIS with Extreme Outliers', fontweight='bold')
    
    # Middle: Volatility
    ax2 = axes[1]
    ax2.plot(zoom['tick'], zoom['true_vol'] * 100, color=COLORS['true_vol'],
             linewidth=1.5, label='True σ')
    ax2.plot(zoom['tick'], zoom['est_vol'] * 100, color=COLORS['est_vol'],
             linewidth=1.5, label='Estimated σ')
    ax2.set_ylabel('Volatility (%)')
    ax2.legend()
    
    # Bottom: ESS health
    ax3 = axes[2]
    ax3.plot(zoom['tick'], zoom['ess'], color='#16a085', linewidth=1)
    ax3.axhline(256 * 0.1, color='red', linestyle='--', alpha=0.7, label='10% threshold')
    ax3.scatter(outliers['tick'], zoom.loc[outliers.index, 'ess'],
                color=COLORS['outlier'], s=50, marker='x', zorder=5)
    ax3.set_xlabel('Tick')
    ax3.set_ylabel('ESS')
    ax3.legend()
    
    plt.tight_layout()
    if save:
        plt.savefig('rbpf_fig4_crisis_persistence.png', bbox_inches='tight')
        print("Saved: rbpf_fig4_crisis_persistence.png")
    return fig


def fig5_summary_dashboard(df, save=True):
    """Panel 5: Summary dashboard with key metrics."""
    fig = plt.figure(figsize=(16, 10))
    
    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Full volatility track (top row, spans all columns)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df['tick'], df['true_vol'] * 100, color=COLORS['true_vol'],
             linewidth=0.6, label='True σ', alpha=0.8)
    ax1.plot(df['tick'], df['est_vol'] * 100, color=COLORS['est_vol'],
             linewidth=0.6, label='Est σ', alpha=0.8)
    outliers = df[df['is_outlier'] == 1]
    ax1.scatter(outliers['tick'], outliers['true_vol'] * 100,
                color=COLORS['outlier'], s=30, marker='x', zorder=5)
    ax1.set_ylabel('Volatility (%)')
    ax1.set_title('Rao-Blackwellized Particle Filter: Volatility Tracking', fontweight='bold', fontsize=14)
    ax1.legend(loc='upper right')
    
    # Add scenario labels
    for name, start, end in SCENARIOS:
        mid = (start + end) / 2
        ax1.axvline(start, color='gray', linestyle=':', alpha=0.3)
    
    # 2. Per-scenario accuracy (middle left)
    ax2 = fig.add_subplot(gs[1, 0])
    scenario_acc = []
    scenario_names = []
    for name, start, end in SCENARIOS:
        mask = (df['tick'] >= start) & (df['tick'] < end)
        acc = df.loc[mask, 'hypo_correct'].mean() * 100
        scenario_acc.append(acc)
        scenario_names.append(name.replace('_', '\n'))
    
    bars = ax2.bar(range(len(scenario_acc)), scenario_acc, color=[
        COLORS['calm'], COLORS['calm'], COLORS['crisis'],
        COLORS['crisis'], COLORS['trend'], COLORS['crisis'], COLORS['trend']
    ], alpha=0.8)
    ax2.set_xticks(range(len(scenario_names)))
    ax2.set_xticklabels(scenario_names, fontsize=8)
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Per-Scenario Accuracy', fontweight='bold')
    ax2.axhline(50, color='gray', linestyle='--', alpha=0.5)
    
    # 3. ESS distribution (middle center)
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.hist(df['ess'], bins=50, color='#16a085', alpha=0.7, edgecolor='white')
    ax3.axvline(df['ess'].mean(), color='red', linestyle='--', label=f"Mean: {df['ess'].mean():.0f}")
    ax3.axvline(df['ess'].min(), color='orange', linestyle=':', label=f"Min: {df['ess'].min():.0f}")
    ax3.set_xlabel('ESS')
    ax3.set_ylabel('Count')
    ax3.set_title('ESS Distribution', fontweight='bold')
    ax3.legend(fontsize=8)
    
    # 4. Error distribution (middle right)
    ax4 = fig.add_subplot(gs[1, 2])
    error = df['est_log_vol'] - df['true_log_vol']
    ax4.hist(error, bins=50, color='#3498db', alpha=0.7, edgecolor='white')
    ax4.axvline(0, color='black', linestyle='-', linewidth=2)
    ax4.axvline(error.mean(), color='red', linestyle='--', label=f"Mean: {error.mean():.3f}")
    ax4.set_xlabel('Log-Vol Error')
    ax4.set_ylabel('Count')
    ax4.set_title('Estimation Error Distribution', fontweight='bold')
    ax4.legend(fontsize=8)
    
    # 5. Regime stacked area (bottom, spans all columns)
    ax5 = fig.add_subplot(gs[2, :])
    calm_prob = df['regime_prob_0'] + df['regime_prob_1']
    trend_prob = df['regime_prob_2']
    crisis_prob = df['regime_prob_3']
    
    ax5.fill_between(df['tick'], 0, calm_prob, color=HYPO_COLORS[0], alpha=0.7, label='P(CALM)')
    ax5.fill_between(df['tick'], calm_prob, calm_prob + trend_prob,
                     color=HYPO_COLORS[1], alpha=0.7, label='P(TREND)')
    ax5.fill_between(df['tick'], calm_prob + trend_prob, 1,
                     color=HYPO_COLORS[2], alpha=0.7, label='P(CRISIS)')
    ax5.set_xlabel('Tick')
    ax5.set_ylabel('Probability')
    ax5.set_title('Regime Probability Evolution', fontweight='bold')
    ax5.legend(loc='upper right', ncol=3)
    ax5.set_ylim(0, 1)
    
    # Add metrics text box
    rmse = np.sqrt(((df['est_log_vol'] - df['true_log_vol'])**2).mean())
    accuracy = df['hypo_correct'].mean() * 100
    
    textstr = f'Log-Vol RMSE: {rmse:.4f}\nHypothesis Acc: {accuracy:.1f}%\nMin ESS: {df["ess"].min():.0f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax1.text(0.02, 0.95, textstr, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    if save:
        plt.savefig('rbpf_fig5_summary_dashboard.png', bbox_inches='tight')
        print("Saved: rbpf_fig5_summary_dashboard.png")
    return fig


def main():
    csv_file = sys.argv[1] if len(sys.argv) > 1 else 'rbpf_viz_data.csv'
    
    print(f"Loading {csv_file}...")
    df = load_data(csv_file)
    print(f"  Loaded {len(df)} ticks\n")
    
    print("Generating figures...")
    fig1_volatility_tracking(df)
    fig2_regime_detection(df)
    fig3_flash_crash(df)
    fig4_crisis_persistence(df)
    fig5_summary_dashboard(df)
    
    print("\nDone! All figures saved.")


if __name__ == '__main__':
    main()
