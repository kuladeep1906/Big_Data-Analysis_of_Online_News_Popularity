"""
Clustering Task Module (Phase 3)
=================================
TASK 3: Discovering Natural Groupings of News Articles
  - Type: Unsupervised - Clustering
  - Input: Top N selected features (scaled)
  - Output: Cluster labels (K-Means, DBSCAN)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

FIGURES_DIR = "figures"


def _save(fig, filename):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


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
    _save(fig, 'fig_task3_clustering_elbow.png')


def plot_clusters_2d(X, labels, title, filename):
    """Project to 2D via PCA and plot clusters."""
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
    _save(fig, filename)


def profile_clusters(df_reduced, labels, selected_features, target='shares'):
    """Profile each cluster by computing mean feature values."""
    df_temp = df_reduced[selected_features].copy()
    df_temp['cluster'] = labels
    df_temp['shares'] = df_reduced[target].values

    print("\n     Cluster Profiling:")
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    for c in range(n_clusters):
        mask = df_temp['cluster'] == c
        cluster_data = df_temp[mask]
        print(f"\n     Cluster {c} ({mask.sum()} articles, {mask.sum()/len(df_temp)*100:.1f}%):")
        print(f"       Mean shares: {cluster_data['shares'].mean():.0f}")
        print(f"       Median shares: {cluster_data['shares'].median():.0f}")

        # Show top 5 distinguishing features (highest mean relative to overall)
        overall_mean = df_temp[selected_features].mean()
        cluster_mean = cluster_data[selected_features].mean()
        ratio = (cluster_mean / (overall_mean + 1e-10)).sort_values(ascending=False)
        print(f"       Top distinguishing features (ratio to overall mean):")
        for feat in ratio.head(5).index:
            print(f"         {feat}: {cluster_mean[feat]:.4f} (overall: {overall_mean[feat]:.4f}, ratio: {ratio[feat]:.2f}x)")


def run_task3_clustering(df_reduced, selected_features, target='shares'):
    """
    TASK 3: Discovering Natural Groupings of News Articles
    Type: Unsupervised - Clustering
    """
    n_features = len(selected_features)

    print("\n" + "=" * 70)
    print("  TASK 3: Discovering Natural Groupings of News Articles")
    print("  Real-World Question: Are there distinct types of articles in the dataset?")
    print("  Type: Unsupervised - Clustering")
    print(f"  Input: {n_features} selected features (scaled)")
    print("  Output: Cluster labels (K-Means, DBSCAN)")
    print("=" * 70)

    X = df_reduced[selected_features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Elbow Method ──
    print("\n  Running elbow method (k=2..10)...")
    k_range, inertias, sil_scores = find_optimal_k(X_scaled)
    plot_elbow(k_range, inertias, sil_scores)

    best_k = k_range[np.argmax(sil_scores)]
    print(f"\n     Best k by silhouette: {best_k}")

    # ── K-Means with best k ──
    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    km_labels = km.fit_predict(X_scaled)
    km_sil = silhouette_score(X_scaled, km_labels, sample_size=5000)
    print(f"\n     K-Means (k={best_k}): Silhouette Score = {km_sil:.4f}")

    plot_clusters_2d(X_scaled, km_labels,
                     f'Task 3: K-Means Clusters (k={best_k})',
                     'fig_task3_kmeans_clusters.png')

    # Profile clusters
    profile_clusters(df_reduced, km_labels, selected_features, target)

    # ── DBSCAN ──
    print("\n  Running DBSCAN...")
    db = DBSCAN(eps=3.0, min_samples=10)
    db_labels = db.fit_predict(X_scaled)
    n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    n_noise = (db_labels == -1).sum()
    print(f"\n     DBSCAN: {n_clusters} clusters found, {n_noise} noise points")

    if n_clusters > 1:
        mask = db_labels != -1
        db_sil = silhouette_score(X_scaled[mask], db_labels[mask],
                                   sample_size=min(5000, mask.sum()))
        print(f"     DBSCAN Silhouette Score (excl. noise): {db_sil:.4f}")

    plot_clusters_2d(X_scaled, db_labels,
                     'Task 3: DBSCAN Clusters',
                     'fig_task3_dbscan_clusters.png')

    print(f"\n     Best Clustering Method: K-Means (k={best_k}, Silhouette={km_sil:.4f})")
    print("\n  TASK 3 COMPLETE")
    print("=" * 70)

    return km_labels, db_labels
