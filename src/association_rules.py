"""
Association Rule Mining Task Module (Phase 3)
==============================================
TASK 4: Identifying Content Patterns Associated with High Engagement
  - Type: Unsupervised - Association Rule Mining
  - Input: Binary feature indicators (weekend, channel type, high/low polarity, etc.)
  - Output: Association rules with support, confidence, lift
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules

FIGURES_DIR = "figures"
RESULTS_DIR = "results"


def _save(fig, filename):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def build_binary_matrix(df, target='shares'):
    """
    Build a boolean DataFrame of binary indicator columns + binarized target.
    Binary columns (0/1) are used as-is.
    Target is binarized using the median.
    """
    binary_cols = [c for c in df.columns if c != target and
                   df[c].dropna().isin([0, 1]).all()]

    if not binary_cols:
        print("  No binary indicator columns found.")
        return pd.DataFrame()

    df_bin = df[binary_cols].copy().astype(bool)

    # Binarize target
    if target in df.columns:
        median_val = df[target].median()
        df_bin[f'{target}_high'] = df[target] >= median_val

    print(f"     Using {len(df_bin.columns)} binary columns for rule mining")
    return df_bin


def run_task4_association_rules(df_full, target='shares'):
    """
    TASK 4: Identifying Content Patterns Associated with High Engagement
    Type: Unsupervised - Association Rule Mining
    """
    # Count binary columns for display
    binary_cols = [c for c in df_full.columns if c != target and
                   df_full[c].dropna().isin([0, 1]).all()]

    print("\n" + "=" * 70)
    print("  TASK 4: Identifying Content Patterns Associated with High Engagement")
    print("  Real-World Question: What combinations of features tend to appear")
    print("    together in high-share articles?")
    print("  Type: Unsupervised - Association Rule Mining")
    print(f"  Input: {len(binary_cols)} binary feature indicators + binarized target")
    print("  Output: Association rules with support, confidence, lift")
    print("=" * 70)

    df_bin = build_binary_matrix(df_full, target=target)
    if df_bin.empty:
        print("  Skipping association rule mining (no binary columns found).")
        return pd.DataFrame()

    # Adaptive min_support
    n_rows = len(df_bin)
    min_support = max(0.05, 50 / n_rows)
    min_confidence = 0.3
    min_lift = 1.01

    print(f"\n     Parameters: min_support={min_support:.3f}, min_confidence={min_confidence}, min_lift={min_lift}")
    print(f"     Dataset size: {n_rows} rows")

    # Run Apriori
    print("     Running Apriori algorithm...")
    try:
        freq_itemsets = apriori(df_bin, min_support=min_support,
                                use_colnames=True, max_len=4)
    except Exception as e:
        print(f"     Apriori error: {e}")
        return pd.DataFrame()

    print(f"     Found {len(freq_itemsets)} frequent itemsets")

    if len(freq_itemsets) == 0:
        print("     No frequent itemsets found.")
        return pd.DataFrame()

    rules = association_rules(freq_itemsets, metric="confidence",
                              min_threshold=min_confidence,
                              num_itemsets=len(freq_itemsets))
    rules = rules[rules['lift'] >= min_lift]
    rules = rules.sort_values('lift', ascending=False)

    print(f"     Found {len(rules)} association rules (lift >= {min_lift})")

    display_cols = ['antecedents', 'consequents', 'support', 'confidence', 'lift']

    if len(rules) > 0:
        print("\n     --- Top 10 Rules (by Lift) ---")
        print(rules[display_cols].head(10).to_string(index=False))

        # Filter popularity rules
        pop_rules = rules[rules['consequents'].apply(lambda x: f'{target}_high' in str(x))]
        print(f"\n     Rules with '{target}_high' as consequent: {len(pop_rules)}")

        if len(pop_rules) > 0:
            print("\n     --- Top Popularity Rules ---")
            print(pop_rules[display_cols].head(10).to_string(index=False))

            # Actionable insights
            print("\n     Actionable Insights:")
            for _, row in pop_rules.head(5).iterrows():
                antecedents = ', '.join(str(s) for s in row['antecedents'])
                print(f"       IF {antecedents}")
                print(f"       THEN article is likely to be popular")
                print(f"       (confidence: {row['confidence']:.2f}, lift: {row['lift']:.2f})")
                print()

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        sc = ax.scatter(rules['support'], rules['confidence'],
                        c=rules['lift'], cmap='viridis', alpha=0.8, s=100)
        plt.colorbar(sc, label='Lift')
        ax.set_title('Task 4: Association Rules - Support vs Confidence',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Support', fontsize=12)
        ax.set_ylabel('Confidence', fontsize=12)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        _save(fig, 'fig_task4_association_rules.png')

        # Save rules to CSV
        os.makedirs(RESULTS_DIR, exist_ok=True)
        rules_export = rules[display_cols].head(50).copy()
        rules_export['antecedents'] = rules_export['antecedents'].apply(lambda x: ', '.join(str(s) for s in x))
        rules_export['consequents'] = rules_export['consequents'].apply(lambda x: ', '.join(str(s) for s in x))
        rules_export.to_csv(os.path.join(RESULTS_DIR, 'task4_association_rules.csv'), index=False)
        print(f"  Saved: {os.path.join(RESULTS_DIR, 'task4_association_rules.csv')}")
    else:
        print("\n     No association rules found above the thresholds.")

    print("\n  TASK 4 COMPLETE")
    print("=" * 70)

    return rules
