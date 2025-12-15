from sklearn.ensemble import RandomForestClassifier
from utils import ModelTester

clf = RandomForestClassifier()
tester = ModelTester(clf)
tester.test_model()