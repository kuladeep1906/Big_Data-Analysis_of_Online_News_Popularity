#!/usr/bin/env python3
"""
Report 1: Preliminary Analysis and Visualization
Big Data Course Project - UCI Online News Popularity Dataset

Usage: python report_1.py <subset_file>
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Output directory for figures
FIGURES_DIR = "figures"


def load_data(filepath):
    """
    Load the subset CSV file into a pandas DataFrame.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        pandas DataFrame with the loaded data
    """
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)


def compute_metrics(df):
    """
    Compute statistical metrics for all columns in the DataFrame.
    
    Metrics computed:
    - Range: [min, max] for numeric columns
    - Mean: arithmetic mean for numeric columns
    - Mode: most frequent value (smallest in case of ties)
    - M_a (Standard Deviation): std for numeric columns
    - M_b (Median): median for numeric columns
    
    Args:
        df: pandas DataFrame
        
    Returns:
        pandas DataFrame with metrics as rows and columns as attributes
    """
    metrics_dict = {}
    
    for col in df.columns:
        col_data = df[col]
        metrics_dict[col] = {}
        
        # Check if column is numeric
        is_numeric = pd.api.types.is_numeric_dtype(col_data)
        
        # Range: [min, max]
        if is_numeric:
            col_min = col_data.min()
            col_max = col_data.max()
            metrics_dict[col]['Range'] = f"[{col_min}, {col_max}]"
        else:
            metrics_dict[col]['Range'] = "N/A"
        
        # Mean: only for numeric
        if is_numeric:
            metrics_dict[col]['Mean'] = f"{col_data.mean():.4f}"
        else:
            metrics_dict[col]['Mean'] = "N/A"
        
        # Mode: works for numeric and categorical
        # Handle ties by taking smallest mode
        try:
            mode_result = col_data.mode()
            if len(mode_result) > 0:
                mode_value = mode_result.min()  # Take smallest in case of ties
                if is_numeric:
                    metrics_dict[col]['Mode'] = f"{mode_value:.4f}" if isinstance(mode_value, float) else str(mode_value)
                else:
                    metrics_dict[col]['Mode'] = str(mode_value)
            else:
                metrics_dict[col]['Mode'] = "N/A"
        except:
            metrics_dict[col]['Mode'] = "N/A"
        
        # M_a: Standard Deviation (numeric only)
        if is_numeric:
            metrics_dict[col]['M_a(std)'] = f"{col_data.std():.4f}"
        else:
            metrics_dict[col]['M_a(std)'] = "N/A"
        
        # M_b: Median (numeric only)
        if is_numeric:
            metrics_dict[col]['M_b(median)'] = f"{col_data.median():.4f}"
        else:
            metrics_dict[col]['M_b(median)'] = "N/A"
    
    # Convert to DataFrame with metrics as rows
    metrics_df = pd.DataFrame(metrics_dict)
    
    return metrics_df


def make_plots(df):
    """
    Generate and display exactly five different visualization techniques.
    Also save figures to disk for later inclusion in the report.
    
    Plots:
    1. Histogram of shares (or fallback column)
    2. Box plot of shares (or fallback column)
    3. Correlation heatmap of top numeric columns
    4. Scatter plot: n_tokens_content vs shares
    5. Bar chart: average shares by weekday
    
    Args:
        df: pandas DataFrame
    """
    # Create figures directory if it doesn't exist
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    # Set matplotlib style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 200
    
    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Determine target column (prefer 'shares', else last numeric column)
    if 'shares' in df.columns:
        target_col = 'shares'
    elif len(numeric_cols) > 0:
        target_col = numeric_cols[-1]
    else:
        target_col = df.columns[-1]
    
    # ------ PLOT 1: Histogram of target variable ------
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    if target_col in numeric_cols:
        ax1.hist(df[target_col].dropna(), bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        ax1.set_xlabel(target_col, fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title(f'Figure 1: Histogram of {target_col}', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, 'fig1_hist_shares.png')
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    print(f"Saved: {fig_path}")
    plt.show()  # Display the figure
    
    # ------ PLOT 2: Box plot of target variable ------
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    if target_col in numeric_cols:
        ax2.boxplot(df[target_col].dropna(), vert=True, patch_artist=True,
                    boxprops=dict(facecolor='lightcoral', alpha=0.7),
                    medianprops=dict(color='darkred', linewidth=2))
        ax2.set_ylabel(target_col, fontsize=12)
        ax2.set_title(f'Figure 2: Box Plot of {target_col}', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, 'fig2_boxplot_shares.png')
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    print(f"Saved: {fig_path}")
    plt.show()  # Display the figure
    
    # ------ PLOT 3: Correlation heatmap ------
    fig3, ax3 = plt.subplots(figsize=(12, 10))
    if len(numeric_cols) > 1:
        # If too many columns, select top 20 by variance
        if len(numeric_cols) > 20:
            variances = df[numeric_cols].var().sort_values(ascending=False)
            top_cols = variances.head(20).index.tolist()
        else:
            top_cols = numeric_cols
        
        corr_matrix = df[top_cols].corr()
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
                    square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax3)
        ax3.set_title('Figure 3: Correlation Heatmap of Numeric Features', 
                      fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, 'fig3_correlation_heatmap.png')
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    print(f"Saved: {fig_path}")
    plt.show()  # Display the figure
    
    # ------ PLOT 4: Scatter plot ------
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    # Prefer n_tokens_content vs shares, else first two numeric columns
    if 'n_tokens_content' in df.columns and target_col in numeric_cols:
        x_col = 'n_tokens_content'
        y_col = target_col
    elif len(numeric_cols) >= 2:
        x_col = numeric_cols[0]
        y_col = numeric_cols[1]
    else:
        x_col = numeric_cols[0] if len(numeric_cols) > 0 else df.columns[0]
        y_col = target_col
    
    if x_col in numeric_cols and y_col in numeric_cols:
        ax4.scatter(df[x_col], df[y_col], alpha=0.6, c='seagreen', edgecolors='black', s=50)
        ax4.set_xlabel(x_col, fontsize=12)
        ax4.set_ylabel(y_col, fontsize=12)
        ax4.set_title(f'Figure 4: Scatter Plot - {x_col} vs {y_col}', 
                      fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, 'fig4_scatter_plot.png')
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    print(f"Saved: {fig_path}")
    plt.show()  # Display the figure
    
    # ------ PLOT 5: Bar chart - Average shares by weekday ------
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    weekday_cols = [col for col in df.columns if 'weekday_is_' in col.lower()]
    
    if weekday_cols and target_col in numeric_cols:
        # Compute average shares for each weekday
        weekday_names = []
        avg_shares = []
        
        for wcol in weekday_cols:
            day_name = wcol.replace('weekday_is_', '').capitalize()
            weekday_names.append(day_name)
            # Filter rows where this weekday indicator is 1
            day_data = df[df[wcol] == 1][target_col]
            if len(day_data) > 0:
                avg_shares.append(day_data.mean())
            else:
                avg_shares.append(0)
        
        ax5.bar(weekday_names, avg_shares, color='mediumpurple', edgecolor='black', alpha=0.7)
        ax5.set_xlabel('Weekday', fontsize=12)
        ax5.set_ylabel(f'Average {target_col}', fontsize=12)
        ax5.set_title(f'Figure 5: Average {target_col} by Weekday', 
                      fontsize=14, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45)
    else:
        # Fallback: use any binary columns if weekday columns not found
        binary_cols = [col for col in numeric_cols if df[col].nunique() == 2]
        if binary_cols and target_col in numeric_cols:
            sample_binary = binary_cols[:7]  # Take up to 7 binary columns
            categories = []
            avg_values = []
            
            for bcol in sample_binary:
                categories.append(bcol)
                val_data = df[df[bcol] == 1][target_col]
                if len(val_data) > 0:
                    avg_values.append(val_data.mean())
                else:
                    avg_values.append(0)
            
            ax5.bar(categories, avg_values, color='mediumpurple', edgecolor='black', alpha=0.7)
            ax5.set_xlabel('Binary Indicator', fontsize=12)
            ax5.set_ylabel(f'Average {target_col}', fontsize=12)
            ax5.set_title(f'Figure 5: Average {target_col} by Binary Indicators', 
                          fontsize=14, fontweight='bold')
            ax5.grid(True, alpha=0.3, axis='y')
            plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, 'fig5_bar_chart.png')
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    print(f"Saved: {fig_path}")
    plt.show()  # Display the figure
    
    # Close all figures to free memory
    plt.close('all')


def main():
    """
    Main function to orchestrate the analysis.
    """
    # Check command-line arguments
    if len(sys.argv) != 2:
        print("Usage: python report_1.py <subset_file>")
        sys.exit(1)
    
    subset_file = sys.argv[1]
    
    # Load data
    df = load_data(subset_file)
    
    # Compute metrics
    metrics_df = compute_metrics(df)
    
    # Print ANALYSIS section
    print("ANALYSIS")
    print(metrics_df.to_string())
    print()
    
    # Generate and display visualizations
    make_plots(df)


if __name__ == "__main__":
    main()
