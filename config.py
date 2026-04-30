from pathlib import Path

ROOT_DIR   = Path(__file__).parent
DATA_DIR   = ROOT_DIR / "src" / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"

RANDOM_STATE = 42

BOT_THRESH          = 200     # max events/user/day before flagging as bot
SESSION_GAP_SECONDS = 1800    # 30-min inactivity gap defines a new session

PCA_VARIANCE_THRESHOLD = 0.90

KMEANS_K_RANGE     = range(2, 9)
KMEANS_SAMPLE_SIZE = 50_000
KMEANS_N_INIT      = 10

NEG_SAMPLE = 150_000   # negatives to subsample; all positives are kept
TEST_SIZE  = 0.20
CV_FOLDS   = 5

# Columns that directly encode the target — excluded before modelling
LEAKY_COLS = ["total_trans", "c2p", "trans_rate"]

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

THRESHOLD_SWEEP_STEPS = 200
FBETA_BETA            = 2
SHAP_SAMPLE_SIZE      = 2000
