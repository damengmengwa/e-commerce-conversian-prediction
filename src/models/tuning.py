"""Hyperparameter tuning using randomized search.

Provides automated hyperparameter optimization with cross-validation
and performance reporting.
"""
import warnings
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from config import HP_ITER, RANDOM_STATE


def tune_model(model, param_dist: dict, X_tr, y_tr, name: str = "") -> object:
    """Tune model hyperparameters using randomized search.
    
    Performs RandomizedSearchCV with stratified k-fold cross-validation
    on the training set, optimizing for ROC-AUC score.
    
    Args:
        model: Scikit-learn estimator to tune
        param_dist: Dictionary of parameter distributions for RandomizedSearchCV
        X_tr: Training features
        y_tr: Training labels
        name: Model name for reporting
        
    Returns:
        Best estimator from the search (fitted on full training data)
        
    Raises:
        ValueError: If param_dist is empty or invalid
    """
    if not param_dist:
        raise ValueError("param_dist cannot be empty")
    
    if not hasattr(model, 'fit') or not hasattr(model, 'predict'):
        raise ValueError(f"Model must implement fit and predict methods")
    
    if len(y_tr) < 4:
        warnings.warn(f"Training set has only {len(y_tr)} samples; cross-validation may be unreliable")
    
    try:
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        search = RandomizedSearchCV(
            model,
            param_distributions=param_dist,
            n_iter=HP_ITER,
            scoring="roc_auc",
            cv=cv,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=0,
            refit=True,
        )
        search.fit(X_tr, y_tr)
        print(f"  [{name}] Best CV-3 AUC : {search.best_score_:.4f}")
        print(f"  [{name}] Best params   : {search.best_params_}")
        return search.best_estimator_
    except Exception as e:
        warnings.warn(f"Hyperparameter tuning failed for {name}: {str(e)}. Returning untrained model.")
        return model
