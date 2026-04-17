import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import LinearRegression

from config import OUTPUT_DIR, RANDOM_STATE


def validate_features(feat: pd.DataFrame, feature_cols: list, output_dir: Path = OUTPUT_DIR) -> None:
    print("\n" + "=" * 60)
    print("[3] Data Validation  (correlation + VIF)")
    print("=" * 60)

    X = feat[feature_cols].copy()
    print(X.describe().round(3).to_string())

    _plot_correlation_heatmap(X, output_dir)
    _report_high_correlations(X)
    _compute_vif(X)


def _plot_correlation_heatmap(X: pd.DataFrame, output_dir: Path) -> None:
    corr = X.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(
        corr, mask=mask, annot=False,
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Feature Correlation Matrix", fontsize=14, pad=10)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()

    path = Path(output_dir) / "feature_correlation.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n  Correlation heatmap → {path}")


def _report_high_correlations(X: pd.DataFrame, threshold: float = 0.85) -> None:
    corr = X.corr()
    high = [
        (corr.columns[i], corr.columns[j], round(corr.iloc[i, j], 3))
        for i in range(len(corr.columns))
        for j in range(i + 1, len(corr.columns))
        if abs(corr.iloc[i, j]) > threshold
    ]
    if high:
        print(f"\n  Highly correlated pairs (|r| > {threshold}):")
        for a, b, r in high:
            print(f"    {a:30s}  ↔  {b:30s}  r = {r}")
    else:
        print(f"\n  No pairs above |r| = {threshold}.")


def _compute_vif(X: pd.DataFrame) -> None:
    print("\n  VIF (each feature regressed on the rest):")
    Xv  = X.values
    idx = np.random.RandomState(RANDOM_STATE).choice(len(Xv), min(30_000, len(Xv)), replace=False)

    vif_rows = []
    for i, col in enumerate(X.columns):
        r2  = LinearRegression().fit(np.delete(Xv[idx], i, axis=1), Xv[idx, i]).score(
                  np.delete(Xv[idx], i, axis=1), Xv[idx, i])
        vif_rows.append({"Feature": col, "VIF": round(1.0 / (1.0 - r2 + 1e-10), 2)})

    vif_df = pd.DataFrame(vif_rows).sort_values("VIF", ascending=False).reset_index(drop=True)
    print(vif_df.to_string(index=False))

    n_high = len(vif_df[vif_df["VIF"] > 10])
    if n_high:
        print(f"\n  [WARN] {n_high} features with VIF > 10 — PCA will handle multicollinearity.")
    else:
        print("\n  All VIF values < 10.")
