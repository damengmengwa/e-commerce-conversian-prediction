"""Visualization and plotting utilities.

Functions:
- plot_roc_pr: ROC and Precision-Recall curves
- plot_all_curves: Combined metric curves
- plot_confusion_matrix: Confusion matrix heatmap
- plot_calibration: Model calibration plot
- plot_learning_curve: Learning curves by training set size
- plot_threshold_sweep: F1 and Fβ2 score vs classification threshold
- plot_shap: SHAP force plots for model interpretability
- plot_rf_importance: Random Forest feature importance
"""
from .plots import (
    plot_roc_pr,
    plot_all_curves,
    plot_confusion_matrix,
    plot_calibration,
    plot_learning_curve,
    plot_threshold_sweep,
    plot_shap,
    plot_rf_importance,
)

__all__ = [
    "plot_roc_pr",
    "plot_all_curves",
    "plot_confusion_matrix",
    "plot_calibration",
    "plot_learning_curve",
    "plot_threshold_sweep",
    "plot_shap",
    "plot_rf_importance",
]
