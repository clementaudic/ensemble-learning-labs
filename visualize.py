from seaborn import kdeplot, heatmap
import matplotlib.pyplot as plt

def plot_distribution(data, column, output_path):
    plt.figure(figsize=(10, 6))
    kdeplot(data[column], fill=True, color='blue', alpha=0.5)
    plt.title(f'Distribution of {column} with KDE')
    plt.xlabel(column)
    plt.ylabel('Density')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close() 

def plot_correlation_with_labels(data, labels, output_path="correlation.png"):
    plt.figure(figsize=(12, 10))
    corr = data.corr()
    heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', cbar=True,
             xticklabels=data.columns,
             yticklabels=data.columns)
    plt.title('Correlation Matrix with Labels')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()