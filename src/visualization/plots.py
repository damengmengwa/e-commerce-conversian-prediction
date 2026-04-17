import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.calibration import calibration_curve
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.model_selection import StratifiedKFold, learning_curve

from config import RANDOM_STATE, SHAP_SAMPLE_SIZE

COLORS = ["#2E5596", "#ED7D31", "#70AD47", "#9B59B6"]


def plot_roc_pr(name, y_te, yprob, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    RocCurveDisplay.from_predictions(y_te, yprob, ax=axes[0], name=name, color=COLORS[0])
    axes[0].set_title(f"ROC Curve — {name}")
    PrecisionRecallDisplay.from_predictions(y_te, yprob, ax=axes[1], name=name, color=COLORS[1])
    axes[1].set_title(f"Precision-Recall — {name}")
    _save(fig, f"curves_{_slug(name)}.png", output_dir)


def plot_all_curves(preds_store, y_te, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for (name, (_, yprob, _)), col in zip(preds_store.items(), COLORS):
        RocCurveDisplay.from_predictions(y_te, yprob, ax=axes[0], name=name, color=col)
        PrecisionRecallDisplay.from_predictions(y_te, yprob, ax=axes[1], name=name, color=col)
    axes[0].set_title("ROC Curves — All Models")
    axes[1].set_title("Precision-Recall Curves — All Models")
    _save(fig, "curves_comparison.png", output_dir)


def plot_confusion_matrix(name, y_te, yp, output_dir):
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_te, yp, display_labels=["No Purchase", "Purchase"], cmap="Blues", ax=ax
    )
    ax.set_title(f"Confusion Matrix — {name}")
    _save(fig, f"cm_{_slug(name)}.png", output_dir)


def plot_calibration(name, y_te, yprob, brier, output_dir):
    prob_true, prob_pred = calibration_curve(y_te, yprob, n_bins=10)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(prob_pred, prob_true, marker="o", lw=2, color=COLORS[0], label=name)
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.set_title(f"Calibration — {name}\n(Brier = {brier:.4f})")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.legend()
    _save(fig, f"calibration_{_slug(name)}.png", output_dir)


def plot_learning_curve(model, name, X_tr, y_tr, output_dir):
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    train_sizes, tr_sc, val_sc = learning_curve(
        model, X_tr, y_tr,
        train_sizes=np.linspace(0.1, 1.0, 8),
        cv=cv, scoring="roc_auc", n_jobs=-1,
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(train_sizes, tr_sc.mean(1) - tr_sc.std(1), tr_sc.mean(1) + tr_sc.std(1), alpha=0.15, color=COLORS[0])
    ax.fill_between(train_sizes, val_sc.mean(1) - val_sc.std(1), val_sc.mean(1) + val_sc.std(1), alpha=0.15, color=COLORS[1])
    ax.plot(train_sizes, tr_sc.mean(1), "o-", color=COLORS[0], label="Train AUC")
    ax.plot(train_sizes, val_sc.mean(1), "s-", color=COLORS[1], label="Val AUC")
    ax.set_title(f"Learning Curves — {name}")
    ax.set_xlabel("Training Size")
    ax.set_ylabel("AUC-ROC")
    ax.legend()
    _save(fig, f"learning_curve_{_slug(name)}.png", output_dir)


def plot_threshold_sweep(name, thresholds, f1s, fb2s, best_f1_t, best_fb_t, output_dir):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(thresholds, f1s,  color=COLORS[0], label=f"F1  (best t={best_f1_t:.2f})")
    ax.plot(thresholds, fb2s, color=COLORS[1], label=f"Fβ2 (best t={best_fb_t:.2f})")
    ax.axvline(best_f1_t, ls="--", color=COLORS[0], alpha=0.6)
    ax.axvline(best_fb_t, ls="--", color=COLORS[1], alpha=0.6)
    ax.set_title(f"Threshold Optimisation — {name}")
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Score")
    ax.legend()
    _save(fig, f"threshold_{_slug(name)}.png", output_dir)


def plot_shap(model, name, X_te, pc_names, output_dir):
    try:
        import shap
    except ImportError:
        print(f"    [WARN] shap not installed — skipping for {name}")
        return

    if not hasattr(model, "estimators_") and not hasattr(model, "get_booster"):
        return

    try:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_te[:min(SHAP_SAMPLE_SIZE, len(X_te))])
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        mean_abs = np.abs(shap_vals).mean(0)
        labels   = pc_names[:len(mean_abs)]
        order    = np.argsort(mean_abs)[::-1]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(range(len(order)), mean_abs[order], color=COLORS[0], edgecolor="white")
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([labels[i] for i in order], fontsize=9)
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(f"SHAP Global Importance — {name}")
        _save(fig, f"shap_{_slug(name)}.png", output_dir)
    except Exception as exc:
        print(f"    [WARN] SHAP failed for {name}: {exc}")


def plot_rf_importance(rf_model, pca, feature_names, n_pcs, output_dir):
    imp_pca = pd.Series(
        rf_model.feature_importances_,
        index=[f"PC{i+1}" for i in range(n_pcs)],
    ).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    imp_pca.plot(kind="bar", color=COLORS[0], ax=ax)
    ax.set_title("RF Feature Importance (PCA Components)")
    ax.set_ylabel("Importance")
    _save(fig, "rf_feature_importance.png", output_dir)

    orig_imp = np.abs(pca.components_).T.dot(rf_model.feature_importances_)
    orig_s   = pd.Series(orig_imp, index=feature_names).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    orig_s.head(20).plot(kind="bar", color=COLORS[1], ax=ax)
    ax.set_title("Top-20 Original Features (RF via PCA back-projection)")
    ax.set_ylabel("Importance (approx.)")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    _save(fig, "rf_orig_feature_importance.png", output_dir)

    print(f"\n  Top-10 original features:\n{orig_s.head(10).round(4).to_string()}")


def _slug(name: str) -> str:
    return name.lower().replace(" ", "_")


def _save(fig, fname: str, output_dir) -> None:
    fig.tight_layout()
    path = Path(output_dir) / fname
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"    Saved → {path}")
