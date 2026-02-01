from lime import lime_tabular
from XGBoostTests import tester
from matplotlib import pyplot as plt

explainer = lime_tabular.LimeTabularExplainer(training_data=tester.X_train.values,
                                              feature_names=tester.X_train.keys(),
                                              class_names=[str(c) for c in tester.model.classes_],
                                              mode='classification')
explainer.explain_instance(data_row=tester.X_test.iloc[[0]].values[0],
                           predict_fn=tester.model.predict_proba).as_pyplot_figure()
plt.title("LIME explanation for first test instance (class 0)")
plt.tight_layout()
plt.savefig(f"images/lime/default.png")