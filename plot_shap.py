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
        label_names = {
            "MAR": ["Married","Widowed","Divorced","Separated","Never married or under 15 yo"],
            "COW": ["N/A","Employee of a private for-profit company or business","Employee of a private not-for-profit, tax-exempt, or charitable organization","Local government employee (city, county, etc.)","State government employee","Federal government employee","Self-employed in own not incorporated business, professional practice, or farm","Self-employed in own incorporated business, professional practice or farm","Working without pay in family business or farm","Unemployed and last worked 5 years ago or earlier or never worked"],
            "RELP": ["Reference person","Husband/wife","Biological son or daughter","Adopted son or daughter","Stepson or stepdaughter","Brother or sister","Father or mother","Grandchild","Parent-in-law","Son-in-law or daughter-in-law","Other relative","Roomer or boarder","Housemate or roommate","Unmarried partner","Foster child","Other nonrelative","Institutionalized group quarters population","Noninstitutionalized group quarters population"],
            "SEX": ["Male","Female"],
            "RAC1P": ["White alone","Black or African American alone","American Indian alone","Alaska Native alone","American Indian and Alaska Native tribes specified, or American Indian or Alaska Native, not specified and no other races","Asian alone","Native Hawaiian and Other Pacific Islander alone","Some Other Race alone","Two or More Races"],
            "SCHL": ["N/A (less than 3 yo)","No schooling completed","Nursery school/preschool","Kindergarten","Grade 1","Grade 2","Grade 3","Grade 4","Grade 5","Grade 6","Grade 7","Grade 8","Grade 9","Grade 10","Grade 11","12th Grade - no diploma","Regular high school diploma","GED or alternative credential","Some college but less than 1 year","1 or more years of college credit but no degree","Associate’s degree","Bachelor’s degree","Master’s degree","Professional degree beyond a bachelor’s degree","Doctorate degree"]
        }

        num_categories = len(label_names[feature_to_plot])
        cmap = plt.colormaps["tab20"]

        legend_elements = [
            Patch(facecolor=cmap(i / 20), label=f"{label_names[feature_to_plot][i]}") for i in range(num_categories)
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

shap_plot("SCHL")