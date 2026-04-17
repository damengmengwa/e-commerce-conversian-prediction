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
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("\n" + "=" * 60)
    print("  E-Commerce Conversion Prediction & Segmentation")
    print("  Spring 2026 — Team: Jha | Ranaware | Ren")
    print("=" * 60)

    events = load_events()
    feat   = engineer_features(events)

    leaky_and_tgt = set(LEAKY_COLS + ["purchased"])
    feature_cols  = [c for c in feat.columns if c not in {"visitorid"} | leaky_and_tgt]

    validate_features(feat, feature_cols, OUTPUT_DIR)

    X_scaled = StandardScaler().fit_transform(feat[feature_cols].values)
    pca, X_pca = fit_pca(X_scaled, OUTPUT_DIR)

    feat    = run_kmeans(X_pca, feat, OUTPUT_DIR)
    results = run_prediction(feat, X_pca, pca, feature_cols, OUTPUT_DIR)

    print(f"\n{'=' * 60}")
    print(f"  Done — outputs in {OUTPUT_DIR}/")
    print(f"{'=' * 60}\n")
    return results


if __name__ == "__main__":
    main()
