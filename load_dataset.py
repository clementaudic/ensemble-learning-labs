from pandas import read_csv
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

data = read_csv('data/alt_acsincome_ca_features_85.csv')

categorical = ['COW', 'MAR', 'OCCP', 'POBP', 'RELP', 'SEX', 'RAC1P']

from sklearn.preprocessing import OneHotEncoder
import os

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

data_encoded = encoder.fit_transform(data[categorical])

# scatter_matrix(data[data.columns.difference(categorical)], figsize=(25, 25))
# plt.savefig('scatter_matrix.png')

os.makedirs('distributions', exist_ok=True)
from visualize import plot_distribution, plot_correlation_with_labels

numeric_cols = data.columns.difference(categorical)
for col in numeric_cols:
    plot_distribution(data, col, output_path=f'distributions/kde_{col}.png')
    plot_correlation_with_labels(data, col, output_path=f'distributions/corr_{col}.png')
    plt.figure(figsize=(20, 10))
    plt.hist(data[col], bins=30, color='C0', edgecolor='black', alpha=0.7)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(f'distributions/dist_{col}.png')
    plt.close()

for col in categorical:
    plot_distribution(data, col, output_path=f'distributions/kde_{col}.png')
    plot_correlation_with_labels(data, col, output_path=f'distributions/corr_{col}.png')
    plt.figure(figsize=(20, 10))
    counts = data[col].value_counts().sort_index()
    counts.sort_values(inplace=True)
    counts.plot(kind='bar', color='C1', edgecolor='black', alpha=0.8)
    plt.title(f'Category counts for {col}')
    xmax = len(counts)
    if xmax <= 50:
        plt.xticks(range(0, xmax))
    if xmax > 100:
        plt.xticks(range(0, xmax, 20))
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(f'distributions/dist_{col}.png')
    plt.close()

