"""
Clustering Module
- K-Means with Elbow Method & Silhouette Score
- DBSCAN
- 2D PCA scatter of clusters
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

FIGURES_DIR = "figures"


def find_optimal_k(X, k_range=range(2, 11)):
    """Run K-Means for different k values and return inertia + silhouette scores."""
    inertias, sil_scores = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X, labels, sample_size=5000))
    return list(k_range), inertias, sil_scores


def plot_elbow(k_range, inertias, sil_scores):
    """Plot the elbow curve and silhouette scores."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(k_range, inertias, 'bo-')
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Inertia')
    ax1.set_title('Elbow Method', fontweight='bold')
    ax1.grid(True, alpha=0.3)

    ax2.plot(k_range, sil_scores, 'ro-')
    ax2.set_xlabel('Number of Clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Analysis', fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig_clustering_elbow.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def run_kmeans(X, n_clusters=4):
    """Run K-Means and return labels."""
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    sil = silhouette_score(X, labels, sample_size=5000)
    print(f"\nK-Means (k={n_clusters}): Silhouette Score = {sil:.4f}")
    return labels, km


def run_dbscan(X, eps=1.5, min_samples=5):
    """Run DBSCAN and return labels."""
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    print(f"\nDBSCAN: {n_clusters} clusters found, {n_noise} noise points")
    if n_clusters > 1:
        mask = labels != -1
        sil = silhouette_score(X[mask], labels[mask], sample_size=min(5000, mask.sum()))
        print(f"DBSCAN Silhouette Score (excl. noise): {sil:.4f}")
    return labels


def plot_clusters_2d(X, labels, title="Cluster Visualization", filename="fig_clusters_2d.png"):
    """Project to 2D via PCA and plot clusters."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='tab10',
                         alpha=0.5, s=10, edgecolors='none')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
    ax.set_title(title, fontweight='bold')
    plt.colorbar(scatter, ax=ax, label='Cluster')
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def run_clustering(df, target='shares'):
    """Full clustering pipeline."""
    print("\n" + "=" * 60)
    print("  CLUSTERING")
    print("=" * 60)

    X = df.drop(columns=[target], errors='ignore')
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Elbow method
    k_range, inertias, sil_scores = find_optimal_k(X_scaled)
    plot_elbow(k_range, inertias, sil_scores)

    # Best k from silhouette
    best_k = k_range[np.argmax(sil_scores)]
    print(f"\nBest k by silhouette: {best_k}")

    # K-Means with best k
    km_labels, km_model = run_kmeans(X_scaled, best_k)
    plot_clusters_2d(X_scaled, km_labels, f"K-Means (k={best_k})", "fig_kmeans_clusters.png")

    # DBSCAN
    db_labels = run_dbscan(X_scaled, eps=3.0, min_samples=10)
    plot_clusters_2d(X_scaled, db_labels, "DBSCAN Clusters", "fig_dbscan_clusters.png")

    return km_labels, db_labels
