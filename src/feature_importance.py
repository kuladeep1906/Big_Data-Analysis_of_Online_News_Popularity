"""
Feature Importance Module
- Random Forest Gini importance
- Permutation importance
- Horizontal bar chart of top features
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

FIGURES_DIR = "figures"


def compute_rf_importance(X_train, y_train, feature_names, top_n=20):
    """Train a Random Forest and extract Gini importances."""
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    print("\n--- Random Forest Feature Importance (Gini) ---")
    for i, (feat, imp) in enumerate(zip(top_features, top_importances), 1):
        print(f"  {i:>2}. {feat:<40s} {imp:.4f}")

    return top_features, top_importances, rf


def compute_permutation_importance(model, X_test, y_test, feature_names, top_n=20):
    """Compute permutation importance on the test set."""
    print("\nComputing permutation importance (this may take a moment)...")
    result = permutation_importance(model, X_test, y_test, n_repeats=10,
                                    random_state=42, n_jobs=-1)
    indices = np.argsort(result.importances_mean)[::-1][:top_n]

    top_features = [feature_names[i] for i in indices]
    top_importances = result.importances_mean[indices]

    print("\n--- Permutation Importance ---")
    for i, (feat, imp) in enumerate(zip(top_features, top_importances), 1):
        print(f"  {i:>2}. {feat:<40s} {imp:.4f}")

    return top_features, top_importances


def plot_feature_importance(features, importances, title, filename):
    """Horizontal bar chart of feature importances."""
    os.makedirs(FIGURES_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = range(len(features))
    ax.barh(y_pos, importances[::-1], color='teal', edgecolor='black', alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features[::-1], fontsize=9)
    ax.set_xlabel('Importance')
    ax.set_title(title, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def run_feature_importance(df, target='shares'):
    """Full feature importance pipeline."""
    print("\n" + "=" * 60)
    print("  FEATURE IMPORTANCE")
    print("=" * 60)

    X = df.drop(columns=[target], errors='ignore')
    median_val = df[target].median()
    y = (df[target] >= median_val).astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # Gini importance
    gini_feats, gini_imps, rf_model = compute_rf_importance(
        X_train, y_train, feature_names
    )
    plot_feature_importance(gini_feats, gini_imps,
                            'Top 20 Features (Gini Importance)',
                            'fig_importance_gini.png')

    # Permutation importance
    perm_feats, perm_imps = compute_permutation_importance(
        rf_model, X_test, y_test, feature_names
    )
    plot_feature_importance(perm_feats, perm_imps,
                            'Top 20 Features (Permutation Importance)',
                            'fig_importance_permutation.png')
