"""ML models for dimensionality reduction, clustering, and prediction.

Functions:
- fit_pca: Apply PCA for dimensionality reduction
- run_kmeans: Perform K-means user segmentation
- run_prediction: Train and evaluate multiple classifiers
- tune_model: Hyperparameter tuning with RandomizedSearch
"""
from .dimensionality import fit_pca
from .clustering import run_kmeans
from .prediction import run_prediction
from .tuning import tune_model

__all__ = ["fit_pca", "run_kmeans", "run_prediction", "tune_model"]
