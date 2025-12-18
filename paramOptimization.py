import optuna
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
import numpy as np

# Adeboost param grid
# param_grid = {
#     'learning_rate': ('choice', [0.01, 0.05, 0.1]),
#     'max_depth': {'type':'int', 'low':1, 'high':18, 'step':2},
#     'n_estimators': {'type':'int', 'low':50, 'high':450, 'step':10},
    # 'learning_rate': ("choice", [0.01, 0.05, 0.1, 0.3]),
    # 'max_depth': ("choice", [1, 3, 6, 10, 12, 15, 20]),
    # 'n_estimators': ("choice", [50, 100, 200, 300, 400]),
# }

# XGBoost param grid
# param_grid = {
#     'learning_rate': ('choice', [0.01, 0.05, 0.1, 0.3]),
#     'max_depth': {'type':'int', 'low':1, 'high':18, 'step':2},
#     'n_estimators': {'type':'int', 'low':50, 'high':450, 'step':10},
# }

# RandomForest param grid
param_grid = {
    # 'max_depth': {'type':'int', 'low':3, 'high':18, 'step':3},
    # 'n_estimators': {'type':'int', 'low':50, 'high':450, 'step':10},
    'max_depth': ("choice", [3, 6, 10, 12, 15, 20]),
    'n_estimators': ("choice", [50, 100, 200, 300, 400]),
}


def tune_optuna(param_space, model, X, y, n_trials=40, cv=3, random_state=305):
    def suggest(trial, name, spec):
        if isinstance(spec, tuple):
            # old-style tuple ('choice', [...])
            t = spec[0]
            if t == 'choice':
                return trial.suggest_categorical(name, spec[1])
            raise ValueError(f"Unsupported tuple type: {t}")
        elif isinstance(spec, dict):
            t = spec.get('type')
            if t == 'int':
                return trial.suggest_int(name, spec['low'], spec['high'], step=spec.get('step', 1))
            if t == 'float':
                return trial.suggest_float(name, spec['low'], spec['high'], log=spec.get('log', False))
            if t == 'choice':
                return trial.suggest_categorical(name, spec['values'])
        raise ValueError(f"Unknown spec format: {spec}")


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

from sys import argv

if len(argv) > 1:
    match argv[1]:
        case "rf":
            from sklearn.ensemble import RandomForestClassifier
            clf = RandomForestClassifier()
        case "ab":
            from sklearn.ensemble import AdaBoostClassifier
            clf = AdaBoostClassifier()
        case _:
            from xgboost import XGBClassifier
            clf = XGBClassifier(enable_categorical=True)
        
else:
    from xgboost import XGBClassifier
    clf = XGBClassifier(enable_categorical=True)

from pandas import read_csv
from numpy import ravel

X = read_csv("data/cleaned_features.csv")
categorical = ['COW', 'MAR', 'OCCP', 'POBP', 'RELP', 'SEX', 'RAC1P']
X[categorical] = X[categorical].astype("category").apply(lambda s: s.cat.codes)
y = ravel(read_csv("data/alt_acsincome_ca_labels_85.csv"))

best_params, best_score, study = tune_optuna(param_grid, clf, X, y, n_trials=20, cv=3)
with open("param_opt_logs", "a", encoding="utf-8") as f:
    f.write(f"{best_params} {best_score} {clf.__class__.__name__}\n")