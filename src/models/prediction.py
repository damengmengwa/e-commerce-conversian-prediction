import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import xgboost as xgb

from config import OUTPUT_DIR, RANDOM_STATE, NEG_SAMPLE, TEST_SIZE, RF_PARAM_DIST, XGB_PARAM_DIST
from src.models.tuning import tune_model
from src.evaluation.metrics import evaluate_model, mcnemar_pairwise, save_results_table
from src.visualization.plots import plot_all_curves, plot_rf_importance


def run_prediction(feat, X_pca, pca, feature_names, output_dir=OUTPUT_DIR):
    print("\n" + "=" * 60)
    print("[6] Conversion Prediction")
    print("=" * 60)

    X_tr, X_te, y_tr, y_te, X_tr_s, y_tr_s = _prepare_data(feat, X_pca)
    pc_names = [f"PC{i+1}" for i in range(X_pca.shape[1])]
    results, preds = [], {}

    print("\n  --- Logistic Regression (Baseline) ---")
    evaluate_model(
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
        "Logistic Regression",
        X_tr_s, y_tr_s, X_te, y_te, results, preds, pc_names, output_dir,
    )

    print("\n  --- Random Forest (Tuned) ---")
    rf_best  = tune_model(
        RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
        RF_PARAM_DIST, X_tr_s, y_tr_s, "Random Forest",
    )
    rf_model = evaluate_model(
        rf_best, "Random Forest",
        X_tr_s, y_tr_s, X_te, y_te, results, preds, pc_names, output_dir,
    )
    plot_rf_importance(rf_model, pca, feature_names, X_pca.shape[1], output_dir)

    print("\n  --- XGBoost (Tuned) ---")
    spw      = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    xgb_best = tune_model(
        xgb.XGBClassifier(scale_pos_weight=spw, eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1),
        XGB_PARAM_DIST, X_tr_s, y_tr_s, "XGBoost",
    )
    evaluate_model(
        xgb_best, "XGBoost",
        X_tr_s, y_tr_s, X_te, y_te, results, preds, pc_names, output_dir,
    )

    print("\n  --- Stacking Ensemble (RF + XGB → LR) ---")
    stack = StackingClassifier(
        estimators=[("rf", rf_model), ("xgb", xgb_best)],
        final_estimator=LogisticRegression(max_iter=500, random_state=RANDOM_STATE),
        cv=3, n_jobs=-1,
    )
    evaluate_model(
        stack, "Stacking Ensemble",
        X_tr_s, y_tr_s, X_te, y_te, results, preds, pc_names, output_dir,
    )

    plot_all_curves(preds, y_te, output_dir)
    mcnemar_pairwise(preds, y_te)

    return save_results_table(results, output_dir)


def _prepare_data(feat, X_pca):
    y       = feat["purchased"].values
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.random.RandomState(RANDOM_STATE).choice(np.where(y == 0)[0], NEG_SAMPLE, replace=False)
    idx     = np.concatenate([pos_idx, neg_idx])
    X2, y2  = X_pca[idx], y[idx]

    X_tr, X_te, y_tr, y_te = train_test_split(X2, y2, test_size=TEST_SIZE, stratify=y2, random_state=RANDOM_STATE)
    X_tr_s, y_tr_s = SMOTE(random_state=RANDOM_STATE, k_neighbors=5).fit_resample(X_tr, y_tr)

    print(f"  Train (raw)   : {y_tr.sum():,} pos / {len(y_tr):,} total")
    print(f"  Train (SMOTE) : {y_tr_s.sum():,} pos / {len(y_tr_s):,} total")
    print(f"  Test          : {y_te.sum():,} pos / {len(y_te):,} total")

    return X_tr, X_te, y_tr, y_te, X_tr_s, y_tr_s
