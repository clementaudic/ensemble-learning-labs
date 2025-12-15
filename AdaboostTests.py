from sklearn.ensemble import AdaBoostClassifier
from utils import ModelTester

clf = AdaBoostClassifier()
tester = ModelTester(clf)
tester.test_model()