from utils import ModelTester
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from numpy import random
from matplotlib import pyplot as plt


categorical = ['COW', 'MAR', 'OCCP', 'POBP', 'RELP', 'SEX', 'RAC1P']
params = {'learning_rate': 0.05, 'max_depth': 6, 'n_estimators': 400}

clf = XGBClassifier(enable_categorical=True, **params)

def permutation_feature_importance(clf):
    scores = []
    tester = ModelTester(clf)
    tester.test_model()
    test_score = tester.test_score
    for column in tester.X_test.keys():
        permutated_X_test = tester.X_test.copy()
        permutated_X_test[column] = permutated_X_test[column].sample(frac=1).reset_index(drop=True)
        scores.append(tester.score_model(x_test=permutated_X_test))
        permutated_X_test = tester.X_test.copy()
    plt.figure(figsize=(15,10))
    importances = [test_score - score for score in scores]
    sorted_indices = sorted(range(len(importances)), key=lambda k: importances[k], reverse=True)
    importances = [importances[i] for i in sorted_indices]
    plt.bar([tester.X_test.keys()[i] for i in sorted_indices], importances)
    plt.xlabel("Decrease in accuracy score after permutation")
    plt.title(f"Permutation Feature Importance for {clf.__class__.__name__}")
    plt.savefig(f"images/{clf.__class__.__name__}/permutation_feature_importance_default.png")

permutation_feature_importance(clf)