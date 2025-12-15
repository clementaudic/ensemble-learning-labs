from lime import lime_tabular
from utils import ModelTester
from xgboost import XGBClassifier

clf = XGBClassifier(enable_categorical=True)
tester = ModelTester(clf)
tester.test_model()
explainer = lime_tabular.LimeTabularExplainer(training_data=tester.X_train.values,
                                              feature_names=tester.X_train.keys(),
                                              class_names=[str(c) for c in tester.model.classes_],
                                              mode='classification')
explainer.explain_instance(data_row=tester.X_test.iloc[0].values,
                           predict_fn=tester.model.predict_proba)