"""Configuration module for e-commerce conversion prediction pipeline.

Centralizes all hyperparameters, paths, and model configurations
used throughout the project for easy adjustment and reproducibility.
"""
from pathlib import Path

# ==================== Paths ====================
ROOT_DIR   = Path(__file__).parent
DATA_DIR   = ROOT_DIR / "src" / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"

# ==================== Reproducibility ====================
RANDOM_STATE = 42  # Fixed seed for deterministic results

# ==================== Data Cleaning ====================
BOT_THRESH          = 200     # max events/user/day before flagging as bot
SESSION_GAP_SECONDS = 1800    # 30-min inactivity gap defines a new session

# ==================== Dimensionality Reduction ====================
PCA_VARIANCE_THRESHOLD = 0.90  # Retain 90% of variance

# ==================== Clustering ====================
KMEANS_K_RANGE     = range(2, 9)      # Range of k values to evaluate
KMEANS_SAMPLE_SIZE = 50_000           # Sample size for k-means (computational efficiency)
KMEANS_N_INIT      = 10               # Number of initializations

# ==================== Model Training & Evaluation ====================
NEG_SAMPLE = 150_000   # negative samples to subsample; all positives kept
TEST_SIZE  = 0.20      # train-test split ratio
CV_FOLDS   = 5         # cross-validation folds

# Columns that directly encode the target — excluded from feature set
# (These would cause data leakage if used during training)
LEAKY_COLS = ["total_trans", "c2p", "trans_rate"]

# ==================== Hyperparameter Tuning ====================
HP_ITER = 20   # RandomizedSearchCV iterations per model

RF_PARAM_DIST = {
    "n_estimators":      [100, 200, 300],
    "max_depth":         [6, 8, 10, None],
    "min_samples_split": [2, 5, 10],
    "max_features":      ["sqrt", "log2"],
}

XGB_PARAM_DIST = {
    "n_estimators":     [100, 200, 300],
    "max_depth":        [4, 5, 6, 7],
    "learning_rate":    [0.05, 0.10, 0.15, 0.20],
    "subsample":        [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9],
    "reg_alpha":        [0, 0.1, 0.5],
    "reg_lambda":       [1, 1.5, 2],
}

# ==================== Model Evaluation ====================
THRESHOLD_SWEEP_STEPS = 200  # Steps for probability threshold optimization
FBETA_BETA            = 2    # F-beta score: β=2 emphasizes recall (catch more converters)
SHAP_SAMPLE_SIZE      = 2000 # Sample size for SHAP feature importance analysis
