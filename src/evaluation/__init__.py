"""Model evaluation and metrics module.

Functions:
- evaluate_model: Comprehensive model evaluation with cross-validation and metrics
- mcnemar_pairwise: Statistical significance testing between classifiers
- save_results_table: Save evaluation results to CSV
"""
from .metrics import evaluate_model, mcnemar_pairwise, save_results_table

__all__ = ["evaluate_model", "mcnemar_pairwise", "save_results_table"]
