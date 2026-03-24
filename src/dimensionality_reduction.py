"""
Dimensionality Reduction Module (Phase 2, Step 4)
==================================================
Professor's Feedback #8: Apply PCA/t-SNE AFTER feature selection,
on the selected N features only (NOT all 58).

This module:
  - PCA with explained variance analysis on selected features
  - t-SNE for 2D visualization on selected features
  - 2D scatter plots colored by popularity class
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

FIGURES_DIR = "figures"


def _save(fig, filename):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def run_pca_analysis(X_scaled, labels, n_features):
    """Run PCA: explained variance plot + 2D scatter."""
    # Explained variance analysis
    n_comp = min(n_features, X_scaled.shape[1])
    pca_full = PCA(n_components=n_comp)
    pca_full.fit(X_scaled)
    cumulative = np.cumsum(pca_full.explained_variance_ratio_)

    # How many components for 90%
    n_90 = int(np.argmax(cumulative >= 0.90) + 1)
    print(f"\n     PCA on {n_features} selected features:")
    print(f"     Components needed for 90% variance: {n_90}")
    print(f"     PC1 explains: {pca_full.explained_variance_ratio_[0]:.1%}")
    print(f"     PC2 explains: {pca_full.explained_variance_ratio_[1]:.1%}")

    # Plot explained variance
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(1, n_comp + 1), pca_full.explained_variance_ratio_,
           alpha=0.6, color='steelblue', label='Individual')
    ax.plot(range(1, n_comp + 1), cumulative, 'ro-', label='Cumulative')
    ax.axhline(y=0.90, color='gray', linestyle='--', alpha=0.7, label='90% threshold')
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Explained Variance Ratio')
    ax.set_title(f'PCA Explained Variance ({n_features} selected features)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, 'fig_pca_variance.png')

    # 2D PCA scatter
    pca_2d = PCA(n_components=2)
    X_pca = pca_2d.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='coolwarm',
                         alpha=0.4, s=8, edgecolors='none')
    ax.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%} variance)')
    ax.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%} variance)')
    ax.set_title(f'PCA 2D Projection ({n_features} selected features)', fontweight='bold')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Popularity (0=Low, 1=High)')
    plt.tight_layout()
    _save(fig, 'fig_pca_2d.png')

    return X_pca


def run_tsne_analysis(X_scaled, labels, perplexity=30, sample_size=5000):
    """Run t-SNE on a sample and plot 2D scatter."""
    # t-SNE is slow on large datasets, so sample
    if len(X_scaled) > sample_size:
        idx = np.random.RandomState(42).choice(len(X_scaled), sample_size, replace=False)
        X_sample = X_scaled[idx]
        labels_sample = labels.values[idx] if hasattr(labels, 'values') else labels[idx]
    else:
        X_sample = X_scaled
        labels_sample = labels

    print(f"\n     Running t-SNE on {len(X_sample)} samples (perplexity={perplexity})...")
    tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=1000,
                random_state=42, init='pca', learning_rate='auto')
    X_tsne = tsne.fit_transform(X_sample)

    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels_sample, cmap='coolwarm',
                         alpha=0.4, s=8, edgecolors='none')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title('t-SNE 2D Projection (selected features)', fontweight='bold')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Popularity (0=Low, 1=High)')
    plt.tight_layout()
    _save(fig, 'fig_tsne_2d.png')

    return X_tsne


def run_dimensionality_reduction(df_reduced, selected_features, target='shares'):
    """
    Full dimensionality reduction pipeline on selected features only.

    Parameters
    ----------
    df_reduced : pd.DataFrame
        DataFrame with selected features + target column.
    selected_features : list of str
        List of selected feature names.
    target : str
        Target column name.
    """
    n_features = len(selected_features)

    print("=" * 70)
    print("  PHASE 2, STEP 4: DIMENSIONALITY REDUCTION (PCA + t-SNE)")
    print(f"  Applied on {n_features} SELECTED features (NOT all original features)")
    print("=" * 70)

    X = df_reduced[selected_features]
    median_val = df_reduced[target].median()
    labels = (df_reduced[target] >= median_val).astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    X_pca = run_pca_analysis(X_scaled, labels, n_features)

    # t-SNE
    X_tsne = run_tsne_analysis(X_scaled, labels)

    print("\n  Dimensionality reduction complete.")
    print("=" * 70)

    return X_pca
