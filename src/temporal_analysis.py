"""
Temporal Trend Analysis Module
- Average shares by weekday
- Weekend vs weekday comparison
- Trends over timedelta (if available)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

FIGURES_DIR = "figures"


def analyze_weekday_shares(df, target='shares'):
    """Compute and plot average shares by weekday."""
    os.makedirs(FIGURES_DIR, exist_ok=True)

    weekday_cols = [c for c in df.columns if 'weekday_is_' in c]
    if not weekday_cols:
        print("No weekday columns found. Skipping weekday analysis.")
        return

    day_order = ['monday', 'tuesday', 'wednesday', 'thursday',
                 'friday', 'saturday', 'sunday']

    results = {}
    for col in weekday_cols:
        day = col.replace('weekday_is_', '').strip()
        day_data = df[df[col] == 1][target]
        results[day.capitalize()] = {
            'mean': day_data.mean(),
            'median': day_data.median(),
            'count': len(day_data)
        }

    # Sort by day order
    ordered = sorted(results.keys(), key=lambda d: day_order.index(d.lower())
                     if d.lower() in day_order else 99)

    means = [results[d]['mean'] for d in ordered]
    medians = [results[d]['median'] for d in ordered]
    counts = [results[d]['count'] for d in ordered]

    # Print summary
    print("\n--- Average Shares by Weekday ---")
    print(f"{'Day':<12} {'Mean':>10} {'Median':>10} {'Count':>8}")
    print("-" * 44)
    for d in ordered:
        r = results[d]
        print(f"{d:<12} {r['mean']:>10.2f} {r['median']:>10.2f} {r['count']:>8}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2',
              '#59a14f', '#edc948', '#b07aa1']

    ax1.bar(ordered, means, color=colors, edgecolor='black', alpha=0.8)
    ax1.set_xlabel('Weekday')
    ax1.set_ylabel(f'Mean {target}')
    ax1.set_title('Mean Shares by Weekday', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.tick_params(axis='x', rotation=45)

    ax2.bar(ordered, medians, color=colors, edgecolor='black', alpha=0.8)
    ax2.set_xlabel('Weekday')
    ax2.set_ylabel(f'Median {target}')
    ax2.set_title('Median Shares by Weekday', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig_weekday_shares.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def analyze_weekend_vs_weekday(df, target='shares'):
    """Compare weekend vs weekday popularity."""
    os.makedirs(FIGURES_DIR, exist_ok=True)

    if 'is_weekend' not in df.columns:
        print("No 'is_weekend' column found. Skipping.")
        return

    groups = df.groupby('is_weekend')[target].agg(['mean', 'median', 'count'])
    groups.index = ['Weekday', 'Weekend']
    print("\n--- Weekend vs Weekday ---")
    print(groups.to_string())

    fig, ax = plt.subplots(figsize=(8, 5))
    x = ['Weekday', 'Weekend']
    bars = ax.bar(x, groups['mean'], color=['steelblue', 'coral'],
                  edgecolor='black', alpha=0.8)
    ax.set_ylabel(f'Mean {target}')
    ax.set_title('Mean Shares: Weekday vs Weekend', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, val in zip(bars, groups['mean']):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                f'{val:.0f}', ha='center', fontweight='bold')

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig_weekend_vs_weekday.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def analyze_timedelta_trend(df, target='shares'):
    """If timedelta exists, plot moving-average trend of shares over time."""
    os.makedirs(FIGURES_DIR, exist_ok=True)

    if 'timedelta' not in df.columns:
        print("No 'timedelta' column. Skipping time trend analysis.")
        return

    # Sort by timedelta and compute moving average
    sorted_df = df.sort_values('timedelta')
    window = max(len(sorted_df) // 50, 10)
    sorted_df['shares_ma'] = sorted_df[target].rolling(window=window).mean()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(sorted_df['timedelta'], sorted_df['shares_ma'],
            color='teal', linewidth=1.5)
    ax.set_xlabel('Days since first article (timedelta)')
    ax.set_ylabel(f'Moving Avg {target} (window={window})')
    ax.set_title('Popularity Trend Over Time', fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig_timedelta_trend.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def run_temporal_analysis(df, target='shares'):
    """Full temporal analysis pipeline."""
    print("\n" + "=" * 60)
    print("  TEMPORAL TREND ANALYSIS")
    print("=" * 60)

    analyze_weekday_shares(df, target)
    analyze_weekend_vs_weekday(df, target)
    analyze_timedelta_trend(df, target)
