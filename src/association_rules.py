"""
Association Rule Mining Module
- Uses binary indicator columns (already 0/1) + binarized target
- Apply Apriori algorithm via mlxtend
- Extract rules where high popularity is the consequent

Strategy: Use binary indicator columns from the dataset directly (no binning needed,
they are already 0/1). This avoids combinatorial explosion from discretizing
continuous features into many one-hot columns.
"""

import os
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules

FIGURES_DIR = "figures"


def build_binary_matrix(df, target='shares'):
    """
    Build a boolean DataFrame of binary indicator columns + binarized target.
    Binary columns (0/1) are used as-is — no discretization needed.
    Target is binarized using the median.
    """
    # Identify binary indicator columns (value set is subset of {0, 1})
    binary_cols = [c for c in df.columns if c != target and
                   df[c].dropna().isin([0, 1]).all()]

    if not binary_cols:
        print("No binary indicator columns found.")
        return pd.DataFrame()

    df_bin = df[binary_cols].copy().astype(bool)

    # Binarize target
    if target in df.columns:
        median_val = df[target].median()
        df_bin[f'{target}_high'] = df[target] >= median_val

    print(f"Using {len(df_bin.columns)} binary columns for rule mining")
    return df_bin


def run_apriori(df_bin, min_support=0.1, min_confidence=0.4, min_lift=1.05):
    """Run Apriori and extract association rules."""
    print(f"Running Apriori (min_support={min_support}, "
          f"min_confidence={min_confidence}, min_lift={min_lift})...")
    try:
        freq_itemsets = apriori(df_bin, min_support=min_support, use_colnames=True,
                                max_len=4)
    except Exception as e:
        print(f"Apriori error: {e}")
        return pd.DataFrame()

    print(f"Found {len(freq_itemsets)} frequent itemsets")

    if len(freq_itemsets) == 0:
        print("No frequent itemsets found.")
        return pd.DataFrame()

    rules = association_rules(freq_itemsets, metric="confidence",
                              min_threshold=min_confidence, num_itemsets=len(freq_itemsets))
    rules = rules[rules['lift'] >= min_lift]
    rules = rules.sort_values('lift', ascending=False)

    print(f"Found {len(rules)} association rules (lift >= {min_lift})")
    return rules


def filter_popularity_rules(rules, target_item='shares_high'):
    """Filter rules where the consequent contains the popularity indicator."""
    mask = rules['consequents'].apply(lambda x: target_item in str(x))
    filtered = rules[mask].copy()
    print(f"Rules with '{target_item}' as consequent: {len(filtered)}")
    return filtered


def run_association_rules(df, target='shares'):
    """Full association rule mining pipeline."""
    print("\n" + "=" * 60)
    print("  ASSOCIATION RULE MINING")
    print("=" * 60)

    df_bin = build_binary_matrix(df, target=target)
    if df_bin.empty:
        print("Skipping association rule mining (no binary columns found).")
        return pd.DataFrame()

    # Adaptive min_support: lower threshold for larger datasets
    n_rows = len(df_bin)
    min_support = max(0.05, 50 / n_rows)   # Reduced from 0.15 to 0.05 to find rules in full dataset
    print(f"Adaptive min_support: {min_support:.3f} (dataset size: {n_rows})")

    rules = run_apriori(df_bin, min_support=min_support, min_confidence=0.3, min_lift=1.01)

    display_cols = ['antecedents', 'consequents', 'support', 'confidence', 'lift']

    if len(rules) > 0:
        print("\n--- Top 10 Rules (by Lift) ---")
        print(rules[display_cols].head(10).to_string(index=False))

        pop_rules = filter_popularity_rules(rules, f'{target}_high')
        if len(pop_rules) > 0:
            print("\n--- Top Popularity Rules ---")
            print(pop_rules[display_cols].head(10).to_string(index=False))

        # Generate plot
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 6))
        sc = ax.scatter(rules['support'], rules['confidence'], c=rules['lift'], cmap='viridis', alpha=0.8, s=100)
        plt.colorbar(sc, label='Lift')
        ax.set_title('Association Rules: Support vs Confidence', fontsize=14, fontweight='bold')
        ax.set_xlabel('Support', fontsize=12)
        ax.set_ylabel('Confidence', fontsize=12)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig_path = os.path.join(FIGURES_DIR, 'fig_association_rules.png')
        plt.savefig(fig_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"  Saved plot: {fig_path}")
    else:
        print("\nNo association rules found to plot above the thresholds.")

    return rules

