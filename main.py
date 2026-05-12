"""E-Commerce Conversion Prediction & Segmentation Pipeline.

End-to-end workflow:
1. Load and clean raw event data (bot removal, deduplication)
2. Engineer 29 behavioral, temporal, and engagement features
3. Validate feature quality and handle missing values
4. Dimensionality reduction (PCA)
5. User segmentation (K-means clustering)
6. Conversion prediction with multiple models (LR, RF, XGB, Stacking)
7. Model evaluation and interpretability analysis (SHAP)

Output: Predictions, model comparisons, and segmentation results saved to outputs/
"""
import warnings
import matplotlib
matplotlib.use("Agg")
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler

from config import OUTPUT_DIR, LEAKY_COLS
from src.data import load_events, validate_features
from src.features import engineer_features
from src.models import fit_pca, run_kmeans, run_prediction


def main():
    """Execute the complete e-commerce prediction and segmentation pipeline."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("\n" + "=" * 60)
    print("  E-Commerce Conversion Prediction & Segmentation")
    print("  Spring 2026 — Team: Abhishek Jha | Tanmay Ranaware | Yumen Ren")
    print("=" * 60)

    # Step 1: Load and clean data
    events = load_events()
    
    # Step 2: Feature engineering
    feat   = engineer_features(events)

    # Step 3: Prepare feature set (exclude leaky columns and target)
    leaky_and_tgt = set(LEAKY_COLS + ["purchased"])
    feature_cols  = [c for c in feat.columns if c not in {"visitorid"} | leaky_and_tgt]

    # Step 4: Validate features
    validate_features(feat, feature_cols, OUTPUT_DIR)

    # Step 5: Standardize and reduce dimensionality
    X_scaled = StandardScaler().fit_transform(feat[feature_cols].values)
    pca, X_pca = fit_pca(X_scaled, OUTPUT_DIR)

    # Step 6: Segment users via clustering
    feat    = run_kmeans(X_pca, feat, OUTPUT_DIR)
    
    # Step 7: Predict conversion with multiple models
    results = run_prediction(feat, X_pca, pca, feature_cols, OUTPUT_DIR)

    print(f"\n{'=' * 60}")
    print(f"  Done — outputs in {OUTPUT_DIR}/")
    print(f"{'=' * 60}\n")
    return results


if __name__ == "__main__":
    main()
