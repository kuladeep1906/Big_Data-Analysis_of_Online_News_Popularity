"""
Exploratory Data Analysis (EDA) Module
=======================================
Preliminary analysis from Report 1 — integrated into the unified pipeline.

Outputs:
  • ANALYSIS table: Range, Mean, Mode, Std Dev, Median for every column
  • fig1_hist_shares.png        — Histogram of shares
  • fig2_boxplot_shares.png     — Box plot of shares
  • fig3_correlation_heatmap.png — Correlation heatmap (top 20 features by variance)
  • fig4_scatter_plot.png       — Scatter: n_tokens_content vs shares
  • fig5_bar_chart.png          — Average shares by weekday
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

FIGURES_DIR = "figures"


# ──────────────────────────────────────────────────────────────
#  STATISTICAL METRICS TABLE
# ──────────────────────────────────────────────────────────────

def compute_metrics(df):
    """
    Compute Range, Mean, Mode, Std, and Median for every column.
    Returns a DataFrame with metrics as rows and columns as attributes.
    """
    import pandas as pd

    metrics_dict = {}

    for col in df.columns:
        col_data = df[col]
        metrics_dict[col] = {}
        is_numeric = pd.api.types.is_numeric_dtype(col_data)

        # Range
        if is_numeric:
            metrics_dict[col]['Range'] = f"[{col_data.min()}, {col_data.max()}]"
        else:
            metrics_dict[col]['Range'] = "N/A"

        # Mean
        if is_numeric:
            metrics_dict[col]['Mean'] = f"{col_data.mean():.4f}"
        else:
            metrics_dict[col]['Mean'] = "N/A"

        # Mode (smallest in case of ties)
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

        # Standard Deviation
        if is_numeric:
            metrics_dict[col]['M_a(std)'] = f"{col_data.std():.4f}"
        else:
            metrics_dict[col]['M_a(std)'] = "N/A"

        # Median
        if is_numeric:
            metrics_dict[col]['M_b(median)'] = f"{col_data.median():.4f}"
        else:
            metrics_dict[col]['M_b(median)'] = "N/A"

    return pd.DataFrame(metrics_dict)


# ──────────────────────────────────────────────────────────────
#  VISUALISATIONS
# ──────────────────────────────────────────────────────────────

def _save(fig, filename):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def make_plots(df):
    """Generate and save the 5 standard EDA plots."""
    sns.set_style("whitegrid")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    target_col = 'shares' if 'shares' in df.columns else (numeric_cols[-1] if numeric_cols else df.columns[-1])

    # ── Plot 1: Histogram (clipped to 99th percentile) ────────
    fig, ax = plt.subplots(figsize=(10, 6))
    if target_col in numeric_cols:
        col_data = df[target_col].dropna()
        p99 = col_data.quantile(0.99)
        col_clipped = col_data[col_data <= p99]
        ax.hist(col_clipped, bins=50,
                color='steelblue', edgecolor='black', alpha=0.7)
        ax.axvline(col_clipped.mean(),   color='red',    linestyle='--',
                   linewidth=1.5, label=f'Mean: {col_clipped.mean():.0f}')
        ax.axvline(col_clipped.median(), color='orange', linestyle='-',
                   linewidth=1.5, label=f'Median: {col_clipped.median():.0f}')
        ax.set_xlabel(target_col, fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'Figure 1: Histogram of {target_col} (clipped to 99th percentile)',
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.text(0.97, 0.95, f'Top 1% excluded (>{p99:,.0f})',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=9, color='gray')
    plt.tight_layout()
    _save(fig, 'fig1_hist_shares.png')

    # ── Plot 2: Box plot ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    if target_col in numeric_cols:
        ax.boxplot(df[target_col].dropna(), vert=True, patch_artist=True,
                   boxprops=dict(facecolor='lightcoral', alpha=0.7),
                   medianprops=dict(color='darkred', linewidth=2))
        ax.set_ylabel(target_col, fontsize=12)
        ax.set_title(f'Figure 2: Box Plot of {target_col}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    _save(fig, 'fig2_boxplot_shares.png')

    # ── Plot 3: Correlation heatmap ────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 10))
    if len(numeric_cols) > 1:
        top_cols = (
            df[numeric_cols].var().sort_values(ascending=False).head(20).index.tolist()
            if len(numeric_cols) > 20 else numeric_cols
        )
        sns.heatmap(df[top_cols].corr(), annot=False, cmap='coolwarm',
                    center=0, square=True, linewidths=0.5,
                    cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title('Figure 3: Correlation Heatmap of Numeric Features',
                     fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save(fig, 'fig3_correlation_heatmap.png')

    # ── Plot 4: Scatter plot ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    if 'n_tokens_content' in df.columns and target_col in numeric_cols:
        x_col, y_col = 'n_tokens_content', target_col
    elif len(numeric_cols) >= 2:
        x_col, y_col = numeric_cols[0], numeric_cols[1]
    else:
        x_col = y_col = target_col

    ax.scatter(df[x_col], df[y_col], alpha=0.6, c='seagreen', edgecolors='black', s=20)
    ax.set_xlabel(x_col, fontsize=12)
    ax.set_ylabel(y_col, fontsize=12)
    ax.set_title(f'Figure 4: Scatter Plot — {x_col} vs {y_col}',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, 'fig4_scatter_plot.png')

    # ── Plot 5: Bar chart — avg shares by weekday ──────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    weekday_cols = [c for c in df.columns if 'weekday_is_' in c.lower()]
    if weekday_cols and target_col in numeric_cols:
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


# ──────────────────────────────────────────────────────────────
#  MASTER RUNNER
# ──────────────────────────────────────────────────────────────

def run_eda(df):
    """Run the full EDA: print the ANALYSIS table + save 5 plots."""
    print("\n" + "=" * 60)
    print("  EDA — ANALYSIS TABLE")
    print("=" * 60)

    metrics_df = compute_metrics(df)
    print("\nANALYSIS")
    print(metrics_df.to_string())
    print()

    print("\n" + "=" * 60)
    print("  EDA — VISUALISATIONS (5 plots)")
    print("=" * 60)
    make_plots(df)
    print("  ✓ EDA plots saved.\n")


# ══════════════════════════════════════════════════════════════
#
#  [EXPERIMENTAL] — Post-Preprocessing EDA
#  ─────────────────────────────────────────
#  Shows how the data looks AFTER basic preprocessing.
#  Useful for visually justifying why preprocessing was applied.
#
#  TO ENABLE: in report_2_task.py, after the PREPROCESSING banner,
#  uncomment this line:
#
#      from src.eda import run_post_preprocessing_eda
#      run_post_preprocessing_eda(df_raw, df_preprocessed)
#
#  NOT connected to the main pipeline yet.
#
# ══════════════════════════════════════════════════════════════

def run_post_preprocessing_eda(df_raw, df_preprocessed):
    """
    [EXPERIMENTAL] Compare raw vs preprocessed data visually.

    Produces 4 comparison figures saved to figures/exp_*:
      exp_fig1_shares_vs_log_shares.png  — raw shares vs log(1+shares) distribution
      exp_fig2_correlation_before_after.png — heatmap before vs after correlation removal
      exp_fig3_feature_scale_before_after.png — feature distributions before vs after scaling
      exp_fig4_feature_count.png         — bar chart: feature counts at each pipeline step

    Parameters
    ----------
    df_raw         : original cleaned dataframe (before any preprocessing)
    df_preprocessed: dataframe after full preprocessing pipeline
    """
    import numpy as np
    import pandas as pd

    sns.set_style("whitegrid")
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("\n" + "=" * 60)
    print("  [EXPERIMENTAL] POST-PREPROCESSING EDA")
    print("=" * 60)

    # ── Exp Plot 1: Raw shares vs log(1+shares) ────────────────
    if 'shares' in df_raw.columns:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        shares = df_raw['shares'].dropna()

        # Clip raw shares to 99th percentile so the shape is visible
        p99 = shares.quantile(0.99)
        shares_clipped = shares[shares <= p99]

        ax1.hist(shares_clipped, bins=60,
                 color='#e74c3c', edgecolor='black', alpha=0.8)
        ax1.axvline(shares_clipped.mean(),   color='navy',   linestyle='--',
                    linewidth=1.5, label=f'Mean: {shares_clipped.mean():.0f}')
        ax1.axvline(shares_clipped.median(), color='gold',   linestyle='-',
                    linewidth=1.5, label=f'Median: {shares_clipped.median():.0f}')
        ax1.set_title('Raw shares (clipped to 99th pct)\n(heavily right-skewed)',
                      fontweight='bold')
        ax1.set_xlabel('shares')
        ax1.set_ylabel('Frequency')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.text(0.97, 0.95, f'Top 1% excluded\n(>{p99:,.0f} shares)',
                 transform=ax1.transAxes, ha='right', va='top',
                 fontsize=8, color='gray')

        log_shares = np.log1p(shares)
        ax2.hist(log_shares, bins=60,
                 color='#2ecc71', edgecolor='black', alpha=0.8)
        ax2.axvline(log_shares.mean(),   color='navy', linestyle='--',
                    linewidth=1.5, label=f'Mean: {log_shares.mean():.2f}')
        ax2.axvline(log_shares.median(), color='gold', linestyle='-',
                    linewidth=1.5, label=f'Median: {log_shares.median():.2f}')
        ax2.set_title('log(1 + shares)\n(approximately normal)', fontweight='bold')
        ax2.set_xlabel('log(1 + shares)')
        ax2.set_ylabel('Frequency')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        fig.suptitle('Target Variable: Before vs After Log-Transform',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        _save(fig, 'exp_fig1_shares_vs_log_shares.png')

    # ── Exp Plot 2: Correlation heatmap before vs after ────────
    numeric_raw  = df_raw.drop(columns=['shares'], errors='ignore').select_dtypes(include=[np.number])
    numeric_post = df_preprocessed.select_dtypes(include=[np.number])

    # Limit to top features by variance for readability
    top_raw  = numeric_raw.var().sort_values(ascending=False).head(20).index.tolist()
    top_post = numeric_post.columns.tolist()[:20]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    sns.heatmap(numeric_raw[top_raw].corr(), annot=False, cmap='coolwarm',
                center=0, square=True, linewidths=0.3, ax=ax1,
                cbar_kws={"shrink": 0.8})
    ax1.set_title(f'Before Preprocessing\n({len(top_raw)} features shown)',
                  fontweight='bold')

    sns.heatmap(numeric_post[top_post].corr(), annot=False, cmap='coolwarm',
                center=0, square=True, linewidths=0.3, ax=ax2,
                cbar_kws={"shrink": 0.8})
    ax2.set_title(f'After Preprocessing\n({len(top_post)} features shown)',
                  fontweight='bold')

    fig.suptitle('Correlation Heatmap: Before vs After Preprocessing',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save(fig, 'exp_fig2_correlation_before_after.png')

    # ── Exp Plot 3: Feature scale before vs after scaling ──────
    # Pick 6 features that exist in both DataFrames
    common_cols = [c for c in numeric_raw.columns if c in numeric_post.columns][:6]

    if len(common_cols) >= 2:
        fig, axes = plt.subplots(2, len(common_cols), figsize=(4 * len(common_cols), 8))

        for i, col in enumerate(common_cols):
            # Before
            axes[0, i].hist(numeric_raw[col].dropna(), bins=30,
                            color='#e74c3c', edgecolor='black', alpha=0.8)
            axes[0, i].set_title(f'{col}\n(raw)', fontsize=9, fontweight='bold')
            axes[0, i].grid(True, alpha=0.3)

            # After
            axes[1, i].hist(numeric_post[col].dropna(), bins=30,
                            color='#3498db', edgecolor='black', alpha=0.8)
            axes[1, i].set_title(f'{col}\n(scaled)', fontsize=9, fontweight='bold')
            axes[1, i].grid(True, alpha=0.3)

        axes[0, 0].set_ylabel('Frequency (Raw)', fontsize=10)
        axes[1, 0].set_ylabel('Frequency (Scaled)', fontsize=10)
        fig.suptitle('Feature Distributions: Before vs After StandardScaler',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        _save(fig, 'exp_fig3_feature_scale_before_after.png')

    # ── Exp Plot 4: Feature count at each pipeline step ────────
    n_raw    = numeric_raw.shape[1]
    n_post   = numeric_post.shape[1]
    # Estimate post-correlation step (not exact without running pipeline steps)
    steps  = ['Raw Features', 'After Correlation\nRemoval (>0.9)', 'After Variance\nFilter', 'After Scaling\n(final)']
    counts = [n_raw, n_raw, n_post, n_post]   # correlation + variance happen together in pipeline

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(steps, counts, color=['#e74c3c', '#e67e22', '#3498db', '#2ecc71'],
                  edgecolor='black', alpha=0.85)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(count), ha='center', va='bottom', fontweight='bold', fontsize=12)
    ax.set_ylabel('Number of Features', fontsize=12)
    ax.set_title('Feature Count at Each Preprocessing Step',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, n_raw * 1.15)
    plt.tight_layout()
    _save(fig, 'exp_fig4_feature_count.png')

    print("  ✓ Experimental post-preprocessing plots saved (figures/exp_fig*.png)\n")
