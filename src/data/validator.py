"""Feature validation and multicollinearity analysis.

Detects highly correlated features and computes VIF (Variance Inflation Factor)
to diagnose multicollinearity issues in the feature set.
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import LinearRegression

from config import OUTPUT_DIR, RANDOM_STATE


def validate_features(feat: pd.DataFrame, feature_cols: list, output_dir: Path = OUTPUT_DIR) -> None:
    """Validate features for quality, correlation, and multicollinearity.
    
    Generates correlation heatmap, identifies highly correlated pairs,
    and computes VIF statistics to assess feature redundancy.
    
    Args:
        feat: Feature dataframe containing all columns
        feature_cols: List of feature column names to validate
        output_dir: Directory for saving plots and reports
        
    Raises:
        ValueError: If feature_cols is empty or contains invalid column names
    """
    if not feature_cols:
        raise ValueError("feature_cols cannot be empty")
    
    missing = set(feature_cols) - set(feat.columns)
    if missing:
        raise ValueError(f"Missing columns in dataframe: {missing}")
    
    print("\n" + "=" * 60)
    print("[3] Data Validation  (correlation + VIF)")
    print("=" * 60)

    X = feat[feature_cols].copy()
    
    # Check for missing values
    n_missing = X.isnull().sum().sum()
    if n_missing > 0:
        warnings.warn(f"Found {n_missing} null values in features; they should be handled before validation")
    
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
    """Report pairs of features with high correlation.
    
    Args:
        X: Feature dataframe
        threshold: Correlation coefficient threshold (default 0.85)
    """
    if X.shape[1] < 2:
        print("\n  Cannot compute correlations with fewer than 2 features.")
        return
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr = X.corr(numeric_only=True)
    
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
    """Compute Variance Inflation Factor for each feature.
    
    Measures how much the variance of a regression coefficient is inflated
    due to multicollinearity. VIF > 10 indicates high multicollinearity.
    
    Args:
        X: Feature dataframe
    """
    print("\n  VIF (each feature regressed on the rest):")
    
    if X.shape[1] < 2:
        print("  Cannot compute VIF with fewer than 2 features.")
        return
    
    Xv  = X.values
    sample_size = min(30_000, len(Xv))
    idx = np.random.RandomState(RANDOM_STATE).choice(len(Xv), sample_size, replace=False)

    vif_rows = []
    try:
        for i, col in enumerate(X.columns):
            X_excl = np.delete(Xv[idx], i, axis=1)  # Exclude feature i
            y_col = Xv[idx, i]
            
            r2 = LinearRegression().fit(X_excl, y_col).score(X_excl, y_col)
            vif = round(1.0 / (1.0 - r2 + 1e-10), 2)
            vif_rows.append({"Feature": col, "VIF": vif})
    except Exception as e:
        warnings.warn(f"Error computing VIF: {str(e)}")
        return

    vif_df = pd.DataFrame(vif_rows).sort_values("VIF", ascending=False).reset_index(drop=True)
    print(vif_df.to_string(index=False))

    n_high = len(vif_df[vif_df["VIF"] > 10])
    if n_high:
        print(f"\n  [WARN] {n_high} features with VIF > 10 — PCA will handle multicollinearity.")
    else:
        print("\n  All VIF values < 10.")
