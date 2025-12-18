from sklearn.ensemble import RandomForestClassifier
from utils import ModelTester
params = {'max_depth': 15, 'n_estimators': 300}

clf = RandomForestClassifier(**params)
tester = ModelTester(clf)
tester.test_model(file_name_prefix="optimized_", generate_files=True)