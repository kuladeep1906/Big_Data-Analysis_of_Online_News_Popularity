# Exploratory Data Analysis (EDA) 

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

FIGURES_DIR = "figures"


def _save(fig, filename):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def compute_metrics(df):
    """Compute Range, Mean, Mode, Std, and Median."""
    metrics_dict = {}
    for col in df.columns:
        col_data = df[col]
        is_numeric = pd.api.types.is_numeric_dtype(col_data)
        metrics_dict[col] = {
            'Range': f"[{col_data.min()}, {col_data.max()}]" if is_numeric else "N/A",
            'Mean': f"{col_data.mean():.4f}" if is_numeric else "N/A",
            'Std': f"{col_data.std():.4f}" if is_numeric else "N/A",
            'Median': f"{col_data.median():.4f}" if is_numeric else "N/A",
        }
        try:
            mode_result = col_data.mode()
            if len(mode_result) > 0:
                mode_value = mode_result.min()
                metrics_dict[col]['Mode'] = (
                    f"{mode_value:.4f}" if isinstance(mode_value, float) else str(mode_value)
                )
            else:
                metrics_dict[col]['Mode'] = "N/A"
        except Exception:
            metrics_dict[col]['Mode'] = "N/A"

    return pd.DataFrame(metrics_dict)


def make_eda_plots(df):
    """Generate and save the 5 standard EDA plots + log-transform"""
    sns.set_style("whitegrid")
    target_col = 'shares'

    # Plot 1: Histogram (clipped to 99th percentile)
    fig, ax = plt.subplots(figsize=(10, 6))
    col_data = df[target_col].dropna()
    p99 = col_data.quantile(0.99)
    col_clipped = col_data[col_data <= p99]
    ax.hist(col_clipped, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(col_clipped.mean(), color='red', linestyle='--', linewidth=1.5,
               label=f'Mean: {col_clipped.mean():.0f}')
    ax.axvline(col_clipped.median(), color='orange', linestyle='-', linewidth=1.5,
               label=f'Median: {col_clipped.median():.0f}')
    ax.set_xlabel(target_col, fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Figure 1: Histogram of {target_col} (clipped to 99th percentile)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, 'fig1_hist_shares.png')

    # Plot 2: Box plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(df[target_col].dropna(), vert=True, patch_artist=True,
               boxprops=dict(facecolor='lightcoral', alpha=0.7),
               medianprops=dict(color='darkred', linewidth=2))
    ax.set_ylabel(target_col, fontsize=12)
    ax.set_title(f'Figure 2: Box Plot of {target_col}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    _save(fig, 'fig2_boxplot_shares.png')

    # Plot 3: Correlation heatmap
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    fig, ax = plt.subplots(figsize=(12, 10))
    top_cols = (df[numeric_cols].var().sort_values(ascending=False).head(20).index.tolist()
                if len(numeric_cols) > 20 else numeric_cols)
    sns.heatmap(df[top_cols].corr(), annot=False, cmap='coolwarm',
                center=0, square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title('Figure 3: Correlation Heatmap (Top 20 by Variance)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save(fig, 'fig3_correlation_heatmap.png')

    # Plot 4: Scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df['n_tokens_content'], df[target_col], alpha=0.6,
               c='seagreen', edgecolors='black', s=20)
    ax.set_xlabel('n_tokens_content', fontsize=12)
    ax.set_ylabel(target_col, fontsize=12)
    ax.set_title('Figure 4: Scatter Plot - n_tokens_content vs shares',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, 'fig4_scatter_plot.png')

    # Plot 5: Bar chart - avg shares by weekday
    fig, ax = plt.subplots(figsize=(10, 6))
    weekday_cols = [c for c in df.columns if 'weekday_is_' in c.lower()]
    if weekday_cols:
        names, avgs = [], []
        for wcol in weekday_cols:
            day = wcol.replace('weekday_is_', '').capitalize()
            day_data = df[df[wcol] == 1][target_col]
            names.append(day)
            avgs.append(day_data.mean() if len(day_data) > 0 else 0)
        ax.bar(names, avgs, color='mediumpurple', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Weekday', fontsize=12)
        ax.set_ylabel(f'Average {target_col}', fontsize=12)
        ax.set_title(f'Figure 5: Average {target_col} by Weekday',
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45)
    plt.tight_layout()
    _save(fig, 'fig5_bar_chart.png')

    # Exp Plot: Raw shares vs log(1+shares)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    shares = df[target_col].dropna()
    p99 = shares.quantile(0.99)
    shares_clipped = shares[shares <= p99]

    ax1.hist(shares_clipped, bins=60, color='#e74c3c', edgecolor='black', alpha=0.8)
    ax1.axvline(shares_clipped.mean(), color='navy', linestyle='--', linewidth=1.5,
                label=f'Mean: {shares_clipped.mean():.0f}')
    ax1.axvline(shares_clipped.median(), color='gold', linestyle='-', linewidth=1.5,
                label=f'Median: {shares_clipped.median():.0f}')
    ax1.set_title('Raw shares (clipped to 99th pct)\n(heavily right-skewed)', fontweight='bold')
    ax1.set_xlabel('shares')
    ax1.set_ylabel('Frequency')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    log_shares = np.log1p(shares)
    ax2.hist(log_shares, bins=60, color='#2ecc71', edgecolor='black', alpha=0.8)
    ax2.axvline(log_shares.mean(), color='navy', linestyle='--', linewidth=1.5,
                label=f'Mean: {log_shares.mean():.2f}')
    ax2.axvline(log_shares.median(), color='gold', linestyle='-', linewidth=1.5,
                label=f'Median: {log_shares.median():.2f}')
    ax2.set_title('log(1 + shares)\n(approximately normal)', fontweight='bold')
    ax2.set_xlabel('log(1 + shares)')
    ax2.set_ylabel('Frequency')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Target Variable: Before vs After Log-Transform', fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save(fig, 'exp_fig1_shares_vs_log_shares.png')


def run_eda(df):
    print("  PHASE 1, STEP 2: EXPLORATORY DATA ANALYSIS (EDA)")
  
    # Analysis table
    print("\n  >> Statistical Summary Table")
    metrics_df = compute_metrics(df)
    print(metrics_df.to_string())
    print()

    # Target variable summary
    if 'shares' in df.columns:
        print("  >> Target Variable Summary (shares)")
        print(f"     Mean:   {df['shares'].mean():.2f}")
        print(f"     Median: {df['shares'].median():.2f}")
        print(f"     Std:    {df['shares'].std():.2f}")
        print(f"     Skew:   {df['shares'].skew():.2f}")
        print(f"     Kurt:   {df['shares'].kurtosis():.2f}")

    # Generate plots
    print("\n  >> Generating EDA Visualizations (6 plots)")
    make_eda_plots(df)
    print("  EDA complete.\n")
