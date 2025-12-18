import optuna
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
import numpy as np

param_grid = {
    # 'learning_rate': ('uniform', 0.001, 0.3),
    'max_depth': ('int', 3, 20),
    'n_estimators': ('int', 50, 500),
}


def tune_optuna(param_space, model, X, y, n_trials=40, cv=3, random_state=305):
    def suggest(trial, name, spec):
        t = spec[0]
        if t == 'uniform':
            return trial.suggest_float(name, spec[1], spec[2])
        if t == 'int':
            return trial.suggest_int(name, spec[1], spec[2])
        if t == 'choice':
            return trial.suggest_categorical(name, spec[1])
        raise ValueError(f"Unknown spec type: {t}")

    def objective(trial):
        params = {k: suggest(trial, k, spec) for k, spec in param_space.items()}
        mdl = clone(model)
        try:
            mdl.set_params(**params)
        except Exception:
            # some estimators may not accept certain params; let fit raise if needed
            pass
        scores = cross_val_score(mdl, X, y, cv=cv)
        return float(np.mean(scores))

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_score = float(study.best_value)
    return best_params, best_score, study


# Example usage (same as original):
from xgboost import XGBClassifier
from pandas import read_csv
from numpy import ravel

clf = XGBClassifier(enable_categorical=True)

X = read_csv("data/cleaned_features.csv")
categorical = ['COW', 'MAR', 'OCCP', 'POBP', 'RELP', 'SEX', 'RAC1P']
X[categorical] = X[categorical].astype("category").apply(lambda s: s.cat.codes)
y = ravel(read_csv("data/alt_acsincome_ca_labels_85.csv"))

best_params, best_score, study = tune_optuna(param_grid, clf, X, y, n_trials=40, cv=3)
with open("param_opt_logs", "a", encoding="utf-8") as f:
    f.write(f"{best_params} {best_score}\n")