from sklearn.ensemble import AdaBoostClassifier
from utils import ModelTester
params = {'learning_rate': 1.7592424690135926, 'n_estimators': 210}
clf = AdaBoostClassifier(**params)
tester = ModelTester(clf)
tester.test_model(file_name_prefix="optimized_", generate_files=True)