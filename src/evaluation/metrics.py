"""Model evaluation metrics and comparison utilities.

Provides comprehensive model evaluation including cross-validation,
performance metrics, threshold optimization, and statistical significance testing.
"""
import itertools
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    matthews_corrcoef, cohen_kappa_score, brier_score_loss,
    classification_report,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from statsmodels.stats.contingency_tables import mcnemar as _mcnemar
from sklearn.metrics import fbeta_score
from config import CV_FOLDS, RANDOM_STATE, THRESHOLD_SWEEP_STEPS, FBETA_BETA
from src.visualization.plots import (
    plot_roc_pr, plot_confusion_matrix, plot_calibration,
    plot_learning_curve, plot_threshold_sweep, plot_shap,
)


def evaluate_model(model, name, X_tr_s, y_tr_s, X_te, y_te,
                   results, preds_store, pc_names, output_dir):
    """Evaluate model performance with comprehensive metrics.
    
    Computes cross-validation AUC, test metrics (AUC-ROC, F1, Precision, Recall, MCC, Kappa, Brier),
    generates visualizations (ROC/PR curves, confusion matrix, calibration, learning curves, SHAP),
    and optimizes classification thresholds.
    
    Args:
        model: Scikit-learn classifier with predict_proba method
        name: Model name for reporting
        X_tr_s: Training features (scaled)
        y_tr_s: Training labels
        X_te: Test features
        y_te: Test labels
        results: List to append results dict to
        preds_store: Dict to store predictions for pairwise comparison
        pc_names: List of feature names for SHAP analysis
        output_dir: Directory for saving outputs
        
    Returns:
        Fitted model
        
    Raises:
        ValueError: If model doesn't support predict_proba
    """
    if not hasattr(model, 'predict_proba'):
        raise ValueError(f"Model {name} must support predict_proba method")
    
    if len(y_te) < 2 or len(np.unique(y_te)) < 2:
        warnings.warn(f"Test set for {name} has insufficient samples or single class")
    
    skf     = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    try:
        cv_aucs = cross_val_score(model, X_tr_s, y_tr_s, cv=skf, scoring="roc_auc", n_jobs=-1)
    except Exception as e:
        warnings.warn(f"Cross-validation failed for {name}: {str(e)}")
        cv_aucs = np.array([np.nan])
    
    print(f"\n  [{name}] CV-{CV_FOLDS} AUC: {cv_aucs.round(3)}  mean={cv_aucs.mean():.4f} ± {cv_aucs.std():.4f}")

    model.fit(X_tr_s, y_tr_s)
    yprob = model.predict_proba(X_te)[:, 1]
    yp    = model.predict(X_te)

    m = {
        "Model":     name,
        "CV_AUC":    round(float(cv_aucs.mean()), 4),
        "AUC-ROC":   round(roc_auc_score(y_te, yprob), 4),
        "F1":        round(f1_score(y_te, yp), 4),
        "Precision": round(precision_score(y_te, yp), 4),
        "Recall":    round(recall_score(y_te, yp), 4),
        "MCC":       round(matthews_corrcoef(y_te, yp), 4),
        "Kappa":     round(cohen_kappa_score(y_te, yp), 4),
        "Brier":     round(brier_score_loss(y_te, yprob), 4),
    }
    results.append(m)
    preds_store[name] = (model, yprob, yp)

    for key in ["AUC-ROC", "F1", "Precision", "Recall", "MCC", "Kappa", "Brier"]:
        print(f"    {key:<12}: {m[key]}")
    print(classification_report(y_te, yp, target_names=["No Purchase", "Purchase"]))

    best_f1_t, best_fb_t = _threshold_sweep(name, y_te, yprob, output_dir)
    print(f"    Optimal F1  threshold : {best_f1_t:.2f}")
    print(f"    Optimal Fβ2 threshold : {best_fb_t:.2f}")

    plot_roc_pr(name, y_te, yprob, output_dir)
    plot_confusion_matrix(name, y_te, yp, output_dir)
    plot_calibration(name, y_te, yprob, m["Brier"], output_dir)
    plot_learning_curve(model, name, X_tr_s, y_tr_s, output_dir)
    plot_shap(model, name, X_te, pc_names, output_dir)

    return model


def _threshold_sweep(name, y_te, yprob, output_dir):
    """Find optimal probability thresholds for F1 and Fβ2 scores.
    
    Args:
        name: Model name for plotting
        y_te: Test labels
        yprob: Predicted probabilities
        output_dir: Directory for saving plots
        
    Returns:
        Tuple of (optimal F1 threshold, optimal Fβ2 threshold)
    """
    thresholds = np.linspace(0.01, 0.99, THRESHOLD_SWEEP_STEPS)
    f1s  = [f1_score(y_te, (yprob >= t).astype(int), zero_division=0) for t in thresholds]
    fb2s = [fbeta_score(y_te, (yprob >= t).astype(int), beta=FBETA_BETA, zero_division=0) for t in thresholds]
    best_f1_t = thresholds[int(np.argmax(f1s))]
    best_fb_t = thresholds[int(np.argmax(fb2s))]
    plot_threshold_sweep(name, thresholds, f1s, fb2s, best_f1_t, best_fb_t, output_dir)
    return best_f1_t, best_fb_t


def mcnemar_pairwise(preds_store: dict, y_te: np.ndarray) -> None:
    """Perform pairwise McNemar's test for statistical significance.
    
    McNemar's test evaluates whether two classifiers have significantly different
    error rates. Null hypothesis: both models have equal error rates.
    
    Args:
        preds_store: Dict mapping model names to (model, yprob, yp) tuples
        y_te: True labels
        
    Reports:
        Chi-square statistic, p-value, and significance level for each pair
    """
    print("\n  McNemar Pairwise Significance Tests:")
    for a, b in itertools.combinations(preds_store.keys(), 2):
        yp_a, yp_b = preds_store[a][2], preds_store[b][2]
        a_right, b_right = yp_a == y_te, yp_b == y_te
        ct  = np.array([
            [np.sum( a_right &  b_right), np.sum( a_right & ~b_right)],
            [np.sum(~a_right &  b_right), np.sum(~a_right & ~b_right)],
        ])
        res = _mcnemar(ct, exact=False, correction=True)
        sig = "***" if res.pvalue < 0.001 else "**" if res.pvalue < 0.01 else "*" if res.pvalue < 0.05 else "n.s."
        print(f"    {a:22s}  vs  {b:22s}  χ²={res.statistic:7.2f}  p={res.pvalue:.4f}  {sig}")


def save_results_table(results: list, output_dir) -> pd.DataFrame:
    """Save model evaluation results to CSV and display formatted table.
    
    Args:
        results: List of result dicts (one per model)
        output_dir: Directory for saving output CSV
        
    Returns:
        Results DataFrame
    """
    df   = pd.DataFrame(results)
    path = output_dir / "model_results.csv"
    df.to_csv(path, index=False)
    print(f"\n  Results saved → {path}")
    print(df.to_string(index=False))
    return df
