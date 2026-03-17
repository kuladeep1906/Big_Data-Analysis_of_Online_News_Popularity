"""
Dimensionality Reduction Module
- PCA (linear) with explained variance analysis
- t-SNE (non-linear) for visualization
- 2D scatter plots colored by popularity class
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

FIGURES_DIR = "figures"


def run_pca(X, n_components=2):
    """Run PCA and return transformed data + model."""
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    return X_pca, pca


def plot_explained_variance(X, max_components=20):
    """Plot cumulative explained variance."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    n_comp = min(max_components, X.shape[1])
    pca = PCA(n_components=n_comp)
    pca.fit(X)

    cumulative = np.cumsum(pca.explained_variance_ratio_)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(1, n_comp + 1), pca.explained_variance_ratio_,
           alpha=0.6, color='steelblue', label='Individual')
    ax.plot(range(1, n_comp + 1), cumulative, 'ro-', label='Cumulative')
    ax.axhline(y=0.90, color='gray', linestyle='--', alpha=0.7, label='90% threshold')
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Explained Variance Ratio')
    ax.set_title('PCA — Explained Variance', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = os.path.join(FIGURES_DIR, 'fig_pca_variance.png')
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

    # How many components for 90%
    n_90 = np.argmax(cumulative >= 0.90) + 1
    print(f"Components needed for 90% variance: {n_90}")
    return pca


def plot_pca_2d(X_pca, labels, title="PCA 2D Projection"):
    """Plot 2D PCA scatter colored by popularity labels."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='coolwarm',
                         alpha=0.4, s=8, edgecolors='none')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title(title, fontweight='bold')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Popularity (0=Low, 1=High)')
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig_pca_2d.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_tsne_2d(X, labels, perplexity=30, n_iter=1000, sample_size=5000):
    """Run t-SNE on a sample and plot 2D scatter."""
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # t-SNE is slow on large datasets, so sample
    if len(X) > sample_size:
        idx = np.random.RandomState(42).choice(len(X), sample_size, replace=False)
        X_sample = X[idx]
        labels_sample = labels.values[idx] if hasattr(labels, 'values') else labels[idx]
    else:
        X_sample = X
        labels_sample = labels

    print(f"\nRunning t-SNE on {len(X_sample)} samples (perplexity={perplexity})...")
    tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=n_iter,
                random_state=42, init='pca', learning_rate='auto')
    X_tsne = tsne.fit_transform(X_sample)

    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels_sample, cmap='coolwarm',
                         alpha=0.4, s=8, edgecolors='none')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title('t-SNE 2D Projection', fontweight='bold')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Popularity (0=Low, 1=High)')
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig_tsne_2d.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def run_dimensionality_reduction(df, target='shares'):
    """Full dimensionality reduction pipeline."""
    print("\n" + "=" * 60)
    print("  DIMENSIONALITY REDUCTION")
    print("=" * 60)

    X = df.drop(columns=[target], errors='ignore')
    median_val = df[target].median()
    labels = (df[target] >= median_val).astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Explained variance analysis
    pca_model = plot_explained_variance(X_scaled)

    # PCA 2D
    X_pca, _ = run_pca(X_scaled, n_components=2)
    plot_pca_2d(X_pca, labels)

    # t-SNE 2D
    plot_tsne_2d(X_scaled, labels)

    return X_pca
