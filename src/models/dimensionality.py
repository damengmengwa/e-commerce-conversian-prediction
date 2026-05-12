"""PCA dimensionality reduction module.

Applies Principal Component Analysis to reduce feature space while 
retaining specified variance threshold, improving model efficiency.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA

from config import OUTPUT_DIR, PCA_VARIANCE_THRESHOLD, RANDOM_STATE


def fit_pca(
    X_scaled: np.ndarray,
    output_dir: Path = OUTPUT_DIR,
    variance_threshold: float = PCA_VARIANCE_THRESHOLD,
) -> tuple:
    """Apply PCA for dimensionality reduction.
    
    Args:
        X_scaled: Standardized feature matrix (n_samples, n_features)
        output_dir: Directory for saving visualizations
        variance_threshold: Target cumulative explained variance (0-1)
        
    Returns:
        Tuple of (fitted PCA model, transformed data)
        
    Raises:
        ValueError: If X_scaled has invalid shape or variance_threshold is invalid
    """
    if not isinstance(X_scaled, np.ndarray):
        raise ValueError("X_scaled must be a numpy array")
    if X_scaled.ndim != 2:
        raise ValueError(f"X_scaled must be 2-dimensional, got shape {X_scaled.shape}")
    if not (0 < variance_threshold < 1):
        raise ValueError(f"variance_threshold must be between 0 and 1, got {variance_threshold}")
    
    print("\n" + "=" * 60)
    print("[4] PCA Dimensionality Reduction")
    print("=" * 60)

    pca   = PCA(n_components=variance_threshold, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    n     = X_pca.shape[1]

    print(f"  Input features      : {X_scaled.shape[1]}")
    print(f"  Components retained : {n}")
    print(f"  Variance explained  : {pca.explained_variance_ratio_.sum() * 100:.1f}%")

    for i, v in enumerate(pca.explained_variance_ratio_):
        print(f"    PC{i + 1}: {v * 100:5.1f}%  {'█' * int(v * 200)}")

    _save_plots(pca, n, output_dir)
    return pca, X_pca


def _save_plots(pca: PCA, n: int, output_dir: Path) -> None:
    """Save PCA variance and cumulative variance plots.
    
    Args:
        pca: Fitted PCA model
        n: Number of components retained
        output_dir: Directory for saving plots
    """
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].bar(range(1, n + 1), pca.explained_variance_ratio_ * 100, color="#2E5596", edgecolor="white")
        axes[0].set_xlabel("Principal Component")
        axes[0].set_ylabel("Explained Variance (%)")
        axes[0].set_title("Scree Plot")
        axes[0].grid(alpha=0.3, axis='y')

        axes[1].plot(range(1, n + 1), cumvar, marker="o", color="#2E5596", lw=2)
        axes[1].axhline(90, color="gray", ls="--", label="90% threshold")
        axes[1].set_xlabel("Number of Components")
        axes[1].set_ylabel("Cumulative Explained Variance (%)")
        axes[1].set_title("Cumulative PCA Variance")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        path = Path(output_dir) / "pca_variance.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Plot saved → {path}")
    except Exception as e:
        print(f"  [WARN] Could not save PCA plots: {str(e)}")
