import shap
from utils import ModelTester
from xgboost import XGBClassifier
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from pickle import dump, load

# XGBoost params resulting in a 0.818586305586955 test score
params = {'learning_rate': 0.05, 'max_depth': 6, 'n_estimators': 400}

# old=True corresponds to the alt_acsincome_ca_features_85 dataset whereas old=False corresponds to the acsincome_ne_allfeatures_with_POBP_and_OCCP_regrouped dataset
old=False

clf = XGBClassifier(enable_categorical=True,**params)
tester = ModelTester(clf, dataset="old" if old else None)

try :
    shap_values = load(open( "shap_values.pkl" if old else "shap_values_new.pkl", "rb"))
except FileNotFoundError:
    tester.test_model()
    # shap.Explainer automatically selected shap.Explainer (not the brute-force ExactExplainer), because it detected that the model is XGBoost (tree-based).
    shap_explainer = shap.Explainer(tester.model,tester.X_train)
    shap_values = shap_explainer(tester.X_test)
    dump(shap_values, open("shap_values.pkl" if old else "shap_values_new.pkl", "wb"))

def shap_plot(feature_to_plot=""):
    # By default, the SHAP plot contains all features
    if feature_to_plot == "":
        shap.summary_plot(
            shap_values.values,
            tester.X_test,
            show=False,
            cmap="turbo"
        )
    else:
        feature_idx = list(tester.X_test.columns).index(feature_to_plot)
        label_names = tester.category_map

        label_names[feature_to_plot]
        cmap = plt.colormaps["tab20"]

        legend_elements = [
            Patch(facecolor=cmap(i / 20), label=f"{label}") for i,label in label_names[feature_to_plot].items()
        ]

        shap.summary_plot(
            shap_values.values[:, feature_idx].reshape(-1, 1),
            tester.X_test[[feature_to_plot]],
            show=False,
            cmap=cmap,
            color_bar=False # Hide the color bar, as it would be redundant with the pyplot legend
        )

        plt.legend(
            handles=legend_elements,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.5)
        )

    # Use bbox_inches="tight" instead of plt.tight_layout() as the latter does no not work well when legend are too large (for instance, with feature_to_plot = "COW", it results in "UserWarning: Tight layout not applied. The bottom and top margins cannot be made large enough to accommodate all axes decorations.")
    plt.savefig(f"images/shap/summary_default{'' if old else '_new'}{'' if feature_to_plot == '' else '_' + feature_to_plot}.png",bbox_inches="tight")

shap_plot("POBP")