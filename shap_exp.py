import shap
from utils import ModelTester
from xgboost import XGBClassifier
from matplotlib import pyplot as plt
from pickle import dump, load

old=False

clf = XGBClassifier(enable_categorical=True)
tester = ModelTester(clf, dataset="old" if old else None)
try :
    shap_values = load(open( "shap_values.pkl" if old else "shap_values_new.pkl", "rb"))
except FileNotFoundError:
    tester.test_model()
    shap_explainer = shap.Explainer(tester.model, tester.X_train)
    shap_values = shap_explainer(tester.X_test)
    dump(shap_values, open("shap_values.pkl" if old else "shap_values_new.pkl", "wb"))
shap.summary_plot(shap_values, tester.X_test, show=False, cmap='turbo')
plt.savefig(f"images/shap/summary_default.png" if old else f"images/shap/summary_default_new.png")