from lime import lime_tabular
from utils import ModelTester
from xgboost import XGBClassifier
from matplotlib import pyplot as plt
params = {'learning_rate': 0.05, 'max_depth': 6, 'n_estimators': 400}

clf = XGBClassifier(enable_categorical=True, **params)
tester = ModelTester(clf)
tester.test_model()
explainer = lime_tabular.LimeTabularExplainer(training_data=tester.X_train.values,
                                              feature_names=tester.X_train.keys(),
                                              class_names=[str(c) for c in tester.model.classes_],
                                              mode='classification')
explainer.explain_instance(data_row=tester.X_test.iloc[0].values,
                           predict_fn=tester.model.predict_proba).as_pyplot_figure()
plt.tight_layout()
plt.savefig(f"images/lime/default.png")