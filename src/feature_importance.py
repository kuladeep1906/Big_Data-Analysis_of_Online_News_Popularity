# Preliminary Feature Analysis & Selection Module 

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

FIGURES_DIR = "figures"
SEED = 42

def _save(fig, filename):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def correlation_analysis(df, target='shares', threshold=0.85):
  
   #  Step 1: Remove one feature from each highly correlated pair.
    print("\n  >> Step 1: Correlation Analysis")
    print(f"     Threshold: |r| > {threshold}")

    X = df.drop(columns=[target], errors='ignore')
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

    if to_drop:
        print(f"     Found {len(to_drop)} highly correlated features to drop:")
        for col in to_drop:
            # Find what it's correlated with
            corr_partner = upper.index[upper[col] > threshold].tolist()
            partner_str = ', '.join(corr_partner[:3])
            max_corr = upper[col].max()
            print(f"       - {col} (max |r| = {max_corr:.3f} with {partner_str})")
        df = df.drop(columns=to_drop)
    else:
        print("     No highly correlated feature pairs found.")

    print(f"     Features remaining after correlation removal: {len(df.columns) - 1}")

    # Plot correlation heatmap after removal
    X_after = df.drop(columns=[target], errors='ignore')
    top_cols = X_after.var().sort_values(ascending=False).head(25).index.tolist()

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(X_after[top_cols].corr(), annot=False, cmap='coolwarm',
                center=0, square=True, linewidths=0.3, ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_title('Correlation Heatmap After Removing Highly Correlated Pairs',
                 fontweight='bold', fontsize=13)
    plt.tight_layout()
    _save(fig, 'fig_correlation_after_removal.png')

    return df, to_drop


def random_forest_importance(df, target='shares'):
    
   #  Step 2: Train Random Forest and compute Gini importance.
    print("\n  >> Step 2: Random Forest Feature Importance (Gini)")

    X = df.drop(columns=[target], errors='ignore')
    median_val = df[target].median()
    y = (df[target] >= median_val).astype(int)

    feature_names = list(X.columns)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=SEED, stratify=y
    )

    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
    rf.fit(X_train, y_train)

    # Get importances
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    # Print all features ranked
    print(f"\n     All {len(sorted_features)} features ranked by Gini importance:")
    print(f"     {'Rank':<6} {'Feature':<45} {'Importance':<12} {'Cumulative %'}")
    print(f"     {'─'*6} {'─'*45} {'─'*12} {'─'*12}")

    cumulative = 0.0
    for i, (feat, imp) in enumerate(zip(sorted_features, sorted_importances), 1):
        cumulative += imp
        print(f"     {i:>4}.  {feat:<45s} {imp:.6f}     {cumulative:.1%}")

    return sorted_features, sorted_importances, rf


def select_top_features(sorted_features, sorted_importances, cumulative_threshold=0.90):
    
    # Step 3: Select top N features using cumulative importance threshold.
    print(f"\n  >> Step 3: Feature Selection (cumulative importance >= {cumulative_threshold:.0%})")
    cumulative = np.cumsum(sorted_importances)
    n_selected = int(np.argmax(cumulative >= cumulative_threshold) + 1)

    # Ensure at least 10 and at most 35 features
    n_selected = max(10, min(n_selected, 35))

    selected_features = sorted_features[:n_selected]
    selected_importances = sorted_importances[:n_selected]
    total_importance = selected_importances.sum()

    print(f"\n     *** SELECTED {n_selected} FEATURES (covering {total_importance:.1%} of total importance) ***")
    for i, (feat, imp) in enumerate(zip(selected_features, selected_importances), 1):
        print(f"       {i:>2}. {feat:<45s} {imp:.6f}")
    return selected_features, n_selected

def plot_feature_importance(sorted_features, sorted_importances, n_selected):
    """Plot feature importance with selected features highlighted."""
    top_n = min(30, len(sorted_features))
    features = sorted_features[:top_n]
    importances = sorted_importances[:top_n]

    colors = ['#2ecc71' if i < n_selected else '#bdc3c7' for i in range(top_n)]

    fig, ax = plt.subplots(figsize=(12, 10))
    y_pos = range(len(features))
    ax.barh(y_pos, importances[::-1], color=colors[::-1], edgecolor='black', alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features[::-1], fontsize=9)
    ax.set_xlabel('Gini Importance', fontsize=12)
    ax.set_title(f'Feature Importance (Green = Selected Top {n_selected})',
                 fontweight='bold', fontsize=13)
    ax.grid(True, alpha=0.3, axis='x')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', edgecolor='black', label=f'Selected ({n_selected})'),
        Patch(facecolor='#bdc3c7', edgecolor='black', label='Not selected'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

    plt.tight_layout()
    _save(fig, 'fig_feature_selection.png')

    # Also plot cumulative importance curve
    cumulative = np.cumsum(sorted_importances)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(sorted_importances) + 1), cumulative, 'b-o', markersize=3)
    ax.axhline(y=0.90, color='red', linestyle='--', alpha=0.7, label='90% threshold')
    ax.axvline(x=n_selected, color='green', linestyle='--', alpha=0.7,
               label=f'N={n_selected} selected')
    ax.set_xlabel('Number of Features', fontsize=12)
    ax.set_ylabel('Cumulative Importance', fontsize=12)
    ax.set_title('Cumulative Feature Importance - Elbow Analysis', fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, 'fig_cumulative_importance.png')


def compute_permutation_importance(rf_model, df, target='shares', top_n=20):
    """Compute permutation importance for validation (secondary method)."""
    print("\n  >> Permutation Importance (validation)")

    X = df.drop(columns=[target], errors='ignore')
    median_val = df[target].median()
    y = (df[target] >= median_val).astype(int)
    feature_names = list(X.columns)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=SEED, stratify=y
    )

    print("     Computing permutation importance (this may take a moment)...")
    result = permutation_importance(rf_model, X_test, y_test,
                                    n_repeats=10, random_state=SEED, n_jobs=-1)
    indices = np.argsort(result.importances_mean)[::-1][:top_n]

    top_features = [feature_names[i] for i in indices]
    top_importances = result.importances_mean[indices]

    print(f"\n     Top {top_n} Features (Permutation Importance):")
    for i, (feat, imp) in enumerate(zip(top_features, top_importances), 1):
        print(f"       {i:>2}. {feat:<45s} {imp:.6f}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = range(len(top_features))
    ax.barh(y_pos, top_importances[::-1], color='teal', edgecolor='black', alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features[::-1], fontsize=9)
    ax.set_xlabel('Permutation Importance', fontsize=12)
    ax.set_title(f'Top {top_n} Features (Permutation Importance)', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    _save(fig, 'fig_importance_permutation.png')

    return top_features


def run_feature_selection(df, target='shares'):

  #  Full preliminary feature analysis and selection pipeline.

    print("  PHASE 2, STEP 3: PRELIMINARY FEATURE ANALYSIS & SELECTION")

    # Step 1: Correlation analysis
    df, dropped_corr = correlation_analysis(df, target, threshold=0.85)

    # Step 2: Random Forest importance
    sorted_features, sorted_importances, rf_model = random_forest_importance(df, target)

    # Step 3: Select top N 
    selected_features, n_selected = select_top_features(
        sorted_features, sorted_importances, cumulative_threshold=0.90
    )

    # Plot importance
    plot_feature_importance(sorted_features, sorted_importances, n_selected)

    # Permutation importance (validation)
    compute_permutation_importance(rf_model, df, target)

    # Build reduced DataFrame
    keep_cols = selected_features + [target]
    df_reduced = df[keep_cols].copy()

    print(f"\n  >> FEATURE SELECTION COMPLETE")
    print(f"     Original features: {len(df.columns) - 1}")
    print(f"     After correlation removal: {len(df.columns) - 1 - len(dropped_corr)}")
    print(f"     Selected features: {n_selected}")
    print(f"     Reduced DataFrame shape: {df_reduced.shape}")

    return selected_features, df_reduced, n_selected, df
