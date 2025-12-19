import shap
from pickle import load
from matplotlib.pyplot import savefig, tight_layout


with open("shap_values_new.pkl", "rb") as f:
    shap_values = load(f)
shap.plots.waterfall(shap_values[0], show=False)
tight_layout()
savefig("images/shap/waterfall_plot.png")