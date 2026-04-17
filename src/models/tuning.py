from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from config import HP_ITER, RANDOM_STATE


def tune_model(model, param_dist: dict, X_tr, y_tr, name: str = ""):
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
