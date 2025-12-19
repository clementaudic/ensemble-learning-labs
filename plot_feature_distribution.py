from pandas import read_csv
import matplotlib.pyplot as plt

features_df = read_csv("data/alt_acsincome_ca_features_85.csv")

def plot_features_distribution(feature_name):
    feature_count = features_df[feature_name].value_counts()

    plt.figure(figsize=(12, 6))
    feature_count.plot(kind="bar")
    plt.title(f"Distribution of the {feature_name} feature in the dataset")
    plt.ylabel(feature_name)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

# plot_features_distribution("OCCP")
# plot_features_distribution("POBP")