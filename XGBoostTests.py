from xgboost import XGBClassifier
from utils import ModelTester
params = {'learning_rate': 0.05, 'max_depth': 6, 'n_estimators': 400}

clf = XGBClassifier(enable_categorical=True, **params)
tester = ModelTester(clf)
tester.test_model(file_name_prefix="optimized_", generate_files=True)