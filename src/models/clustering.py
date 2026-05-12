"""K-Means clustering for customer segmentation.

Performs unsupervised learning to identify distinct customer segments
based on behavioral patterns. Uses multiple evaluation metrics (Silhouette,
Davies-Bouldin, Calinski-Harabasz) to select optimal cluster count.
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

from config import OUTPUT_DIR, RANDOM_STATE, KMEANS_K_RANGE, KMEANS_SAMPLE_SIZE, KMEANS_N_INIT


def run_kmeans(X_pca: np.ndarray, feat: pd.DataFrame, output_dir: Path = OUTPUT_DIR) -> pd.DataFrame:
    """Perform K-means clustering and segment users.
    
    Evaluates multiple k values using clustering metrics, selects optimal k,
    and labels users into interpretable business segments.
    
    Args:
        X_pca: PCA-reduced feature matrix from fit_pca
        feat: User feature dataframe with original data
        output_dir: Directory for saving results and visualizations
        
    Returns:
        Updated feature dataframe with 'cluster' and 'segment' columns
    """
    if X_pca.shape[0] == 0:
        raise ValueError("X_pca cannot be empty")
    if len(feat) == 0:
        raise ValueError("feat dataframe cannot be empty")
    
    print("\n" + "=" * 60)
    print("[5] K-Means Customer Segmentation")
    print("=" * 60)

    Xs         = _subsample(X_pca)
    scan_stats = _scan_k_values(Xs)
    optimal_k  = _pick_optimal_k(scan_stats)
    _save_selection_plot(scan_stats, optimal_k, output_dir)

    feat = _fit_and_label(X_pca, feat, optimal_k)

    path = Path(output_dir) / "user_segments.csv"
    feat.to_csv(path, index=False)
    print(f"\n  Segments saved → {path}")
    return feat


def _subsample(X_pca: np.ndarray) -> np.ndarray:
    """Subsample data for efficient k-means scanning.
    
    Args:
        X_pca: PCA-transformed feature matrix
        
    Returns:
        Subsampled data (up to KMEANS_SAMPLE_SIZE rows)
    """
    n   = min(KMEANS_SAMPLE_SIZE, len(X_pca))
    idx = np.random.RandomState(RANDOM_STATE).choice(len(X_pca), n, replace=False)
    return X_pca[idx]


def _scan_k_values(Xs: np.ndarray) -> pd.DataFrame:
    """Evaluate clustering metrics for different k values.
    
    Computes silhouette, Davies-Bouldin, and Calinski-Harabasz scores
    for each k in KMEANS_K_RANGE.
    
    Args:
        Xs: Subsampled PCA features
        
    Returns:
        DataFrame with k and evaluation metrics
    """
    records = []
    for k in KMEANS_K_RANGE:
        km     = KMeans(n_clusters=k, init="k-means++", n_init=5, random_state=RANDOM_STATE)
        labels = km.fit_predict(Xs)
        records.append({
            "k":          k,
            "inertia":    km.inertia_,
            "silhouette": silhouette_score(Xs, labels, sample_size=10_000, random_state=RANDOM_STATE),
            "db":         davies_bouldin_score(Xs, labels),
            "ch":         calinski_harabasz_score(Xs, labels),
        })
        r = records[-1]
        print(f"  K={k}  inertia={r['inertia']:>12,.0f}  sil={r['silhouette']:.4f}  DB={r['db']:.4f}  CH={r['ch']:,.0f}")
    return pd.DataFrame(records)


def _pick_optimal_k(stats: pd.DataFrame) -> int:
    """Select optimal number of clusters based on silhouette score.
    
    Args:
        stats: DataFrame with clustering metrics for each k
        
    Returns:
        Optimal k value
    """
    best = stats.loc[stats["silhouette"].idxmax()]
    print(f"\n  → Optimal K = {int(best['k'])}  (Silhouette = {best['silhouette']:.4f}, DB = {best['db']:.4f})")
    return int(best["k"])


def _save_selection_plot(stats: pd.DataFrame, optimal_k: int, output_dir: Path) -> None:
    """Save clustering evaluation metrics visualization.
    
    Creates 4-panel plot showing elbow curve, silhouette, Davies-Bouldin,
    and Calinski-Harabasz scores with optimal k highlighted.
    
    Args:
        stats: DataFrame with clustering metrics
        optimal_k: Selected optimal k value
        output_dir: Directory for saving plot
    """
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for ax, (col, title) in zip(axes.flatten(), [
            ("inertia",    "Elbow (Inertia)"),
            ("silhouette", "Silhouette ↑"),
            ("db",         "Davies-Bouldin ↓"),
            ("ch",         "Calinski-Harabasz ↑"),
        ]):
            ax.plot(stats["k"], stats[col], marker="o", lw=2, color="#2E5596")
            ax.axvline(optimal_k, ls="--", color="#ED7D31", label=f"K={optimal_k}")
            ax.set_title(title)
            ax.set_xlabel("K")
            ax.legend()
            ax.grid(alpha=0.3)
        plt.suptitle("K-Means Cluster Selection", fontsize=13, y=1.01)
        plt.tight_layout()
        path = Path(output_dir) / "kmeans_selection.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Plot saved → {path}")
    except Exception as e:
        warnings.warn(f"Could not save K-means selection plot: {str(e)}")


def _fit_and_label(X_pca: np.ndarray, feat: pd.DataFrame, optimal_k: int) -> pd.DataFrame:
    """Fit final K-means model and assign business segment labels.
    
    Args:
        X_pca: Full PCA-transformed feature matrix
        feat: Original feature dataframe
        optimal_k: Number of clusters
        
    Returns:
        DataFrame with 'cluster' and 'segment' columns added
    """
    km = KMeans(n_clusters=optimal_k, init="k-means++", n_init=KMEANS_N_INIT, random_state=RANDOM_STATE)
    feat = feat.copy()
    feat["cluster"] = km.fit_predict(X_pca)

    profile = feat.groupby("cluster")[
        ["total_views", "total_carts", "unique_items", "recency_days", "session_count", "purchased"]
    ].mean()

    feat["segment"] = feat["cluster"].map(_assign_labels(profile))

    print("\n  Cluster profiles:")
    print(profile.round(3).to_string())
    print("\n  Segment distribution:")
    print(feat["segment"].value_counts().to_string())
    return feat


def _assign_labels(profile: pd.DataFrame) -> dict:
    """Assign interpretable business segment names to clusters.
    
    Heuristic-based labeling using purchase rates and browsing behavior:
    - High-Value Buyers: >50% conversion rate
    - Cart Abandoners: above-median cart rate but <10% conversion
    - Active Browsers: above-median views
    - Inactive Users: default category
    
    Args:
        profile: Cluster mean profiles
        
    Returns:
        Mapping of cluster index to segment name
    """
    labels = {}
    for c in profile.index:
        row = profile.loc[c]
        if row["purchased"] > 0.50:
            labels[c] = "High-Value Buyers"
        elif row["total_carts"] > profile["total_carts"].median() and row["purchased"] < 0.10:
            labels[c] = "Cart Abandoners"
        elif row["total_views"] > profile["total_views"].median():
            labels[c] = "Active Browsers"
        else:
            labels[c] = "Inactive Users"
    return labels
