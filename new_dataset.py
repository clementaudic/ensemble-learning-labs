from XGBoostTests import tester
from pandas import read_csv
from numpy import ravel
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix, classification_report
from matplotlib import pyplot as plt

categorical = ['COW', 'MAR', 'OCCP', 'POBP', 'RELP', 'SEX', 'RAC1P']

n=6
dataset_name = f"features_split_{n}_with_POBP_and_OCCP_regrouped.csv"
labels_name = f"labels_split_{n}.csv"
dataset = read_csv(f"data/{dataset_name}")
labels = ravel(read_csv(f"data/{labels_name}"))

# dataset= read_csv("data/acsincome_co_allfeatures_with_POBP_and_OCCP_regrouped.csv")
dataset[categorical] = dataset[categorical].astype("category").apply(lambda s: s.cat.codes)
# labels = ravel(read_csv("data/acsincome_co_label.csv"))

preds = tester.model.predict(dataset)
accuracy = accuracy_score(labels, preds)
print(f"Accuracy: {accuracy}")

cm = confusion_matrix(labels, preds)
disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.savefig(f"images/tests/confusion_matrix_test_{n}.png")
# plt.savefig("images/colorado/confusion_matrix_test.png")
plt.close()
