from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from utils import ModelTester

clf_xgb = XGBClassifier(enable_categorical=True)
clf_rf = RandomForestClassifier()
clf_ab = AdaBoostClassifier()

for clf in [clf_xgb, clf_rf, clf_ab]:
    tester = ModelTester(clf)
    tester.test_model()