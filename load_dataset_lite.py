from sklearn.model_selection import train_test_split
from pandas import read_csv

features = read_csv("alt_acsincome_ca_features_85.csv")
labels = read_csv("alt_acsincome_ca_labels_85.csv")

features = features.drop("OCCP",axis=1)

X_train, X_val, y_train, y_val = train_test_split(features,labels,test_size=0.2,shuffle=True,random_state=42)