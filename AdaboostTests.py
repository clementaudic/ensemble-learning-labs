from sklearn.ensemble import AdaBoostClassifier
from utils import ModelTester
params = {'learning_rate': 0.2156400132705045, 'n_estimators': 450}
clf = AdaBoostClassifier(**params)
tester = ModelTester(clf)
tester.test_model(file_name_prefix="optimized_", generate_files=True)