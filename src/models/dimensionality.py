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
    cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].bar(range(1, n + 1), pca.explained_variance_ratio_ * 100, color="#2E5596", edgecolor="white")
    axes[0].set_xlabel("Principal Component")
    axes[0].set_ylabel("Explained Variance (%)")
    axes[0].set_title("Scree Plot")

    axes[1].plot(range(1, n + 1), cumvar, marker="o", color="#2E5596", lw=2)
    axes[1].axhline(90, color="gray", ls="--", label="90% threshold")
    axes[1].set_xlabel("Number of Components")
    axes[1].set_ylabel("Cumulative Explained Variance (%)")
    axes[1].set_title("Cumulative PCA Variance")
    axes[1].legend()

    plt.tight_layout()
    path = Path(output_dir) / "pca_variance.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Plot saved → {path}")
