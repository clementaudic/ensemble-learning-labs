from xgboost import XGBClassifier
from utils import ModelTester

clf = XGBClassifier(enable_categorical=True)
tester = ModelTester(clf)
tester.test_model()