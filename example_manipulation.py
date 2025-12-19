from XGBoostTests import tester
from pandas import read_csv
from numpy import ravel
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix, classification_report
from matplotlib import pyplot as plt
import shap
from lime import lime_tabular

dataset = tester.X_test.iloc[[0]].copy()
dataset["WKHP"] = 40
preds = tester.model.predict(dataset)
print(f"Prediction with WKHP=40: {preds}")
shap_explainer = shap.Explainer(tester.model,tester.X_train)
shap_values = shap_explainer(dataset)
shap.waterfall_plot(shap_values[0], show=False)
plt.tight_layout()
plt.savefig("images/shap/manipulated_waterfall_plot.png")
plt.close()

explainer = lime_tabular.LimeTabularExplainer(training_data=tester.X_train.values,
                                              feature_names=tester.X_train.keys(),
                                                class_names=[str(c) for c in tester.model.classes_],
                                                mode='classification')
exp = explainer.explain_instance(data_row=dataset.iloc[0].values,
                                 predict_fn=tester.model.predict_proba)
fig = exp.as_pyplot_figure()
plt.tight_layout()
plt.savefig(f"images/lime/manipulated_WKHP_40.png")