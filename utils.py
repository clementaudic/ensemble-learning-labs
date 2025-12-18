from numpy import ravel
from pandas import read_csv, DataFrame, concat
from os import makedirs, path
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from time import time

categorical = ['COW', 'MAR', 'OCCP', 'POBP', 'RELP', 'SEX', 'RAC1P']

class ModelTester():
    def __init__(self, model, dataset=None):
        self.model = model
            
        output_dir = path.join("images", model.__class__.__name__)
        makedirs(output_dir, exist_ok=True)
        match dataset:
            case "old":
                self.X = read_csv("data/alt_acsincome_ca_features_85.csv")
            case "POBP":
                self.X = read_csv("data/cleaned_features_POBP.csv")
            case _:
                self.X = read_csv("data/cleaned_features.csv")
            
        self.y = ravel(read_csv("data/alt_acsincome_ca_labels_85.csv"))
        if model.__class__.__name__ == "XGBClassifier":
            self.X[categorical] = self.X[categorical].astype("category").apply(lambda s: s.cat.codes)
        else:
            encoder = OneHotEncoder(sparse_output=False)
            encoded = encoder.fit_transform(self.X[categorical])
            self.X = concat([self.X.drop(columns=categorical), DataFrame(encoded, index=self.X.index, columns=encoder.get_feature_names_out(categorical))], axis=1)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.25, random_state=305)
        self.score = None

    def set_model(self, model):
        self.model = model
        self.predictions = None
        self.score = None

        if model.__class__.__name__ == "XGBClassifier":
            self.X[categorical] = self.X[categorical].astype("category")
        else:
            encoder = OneHotEncoder(sparse_output=False)
            encoded = encoder.fit_transform(self.X[categorical])
            self.X = concat([self.X.drop(columns=categorical), DataFrame(encoded, index=self.X.index, columns=encoder.get_feature_names_out(categorical))], axis=1)


    def predict(self,x_test=None, force = False):
        if not hasattr(self, 'predictions') or self.predictions is None or force:
            self.predictions = self.model.predict(x_test if x_test is not None else self.X_test)
        return self.predictions

    def test_model(self, in_logs=False, file_name_prefix=""):
        start_time = time()
        self.model.fit(self.X_train, self.y_train)
        self.train_time = time() - start_time
        start_time = time()
        self.predict()
        self.test_time = time() - start_time
        self.score_model()
        self.train_score = accuracy_score(self.y_train, self.model.predict(self.X_train))
        print(f"{self.model.__class__.__name__} || Test score: {self.test_score} | Train score: {self.train_score} | Training time: {self.train_time:.2f} seconds || Test time: {self.test_time:.2f} seconds")
        if in_logs:
            with open("logs", "a") as f:
                f.write(f"{self.model.__class__.__name__} || Test score: {self.test_score} | Train score: {self.train_score} | Training time: {self.train_time:.2f} seconds || Test time: {self.test_time:.2f} seconds\n")
        self.save_confusion_matrix(file_name_prefix=file_name_prefix)
        self.save_confusion_matrix(train=True, file_name_prefix=file_name_prefix)
        self.save_classification_report(file_name_prefix=file_name_prefix)
        return self.test_score
    
    def score_model(self, x_test=None):
        preds = self.predict(x_test=x_test if x_test is not None else self.X_test, force=True)
        self.test_score = accuracy_score(self.y_test, preds)
        return self.test_score
    
    def save_confusion_matrix(self, train=False, file_name_prefix=""):
        if train:
            y_pred = self.model.predict(self.X_train)
            cm = confusion_matrix(self.y_train, y_pred)
        else:
            y_pred = self.model.predict(self.X_test)
            cm = confusion_matrix(self.y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm)
        disp.plot()
        plt.savefig(path.join("images", self.model.__class__.__name__, f"{file_name_prefix}confusion_matrix.png" if not train else f"{file_name_prefix}confusion_matrix_train.png"))
        plt.close()

    def save_classification_report(self, file_name_prefix=""):
        y_pred = self.model.predict(self.X_test)
        report = classification_report(self.y_test, y_pred)
        with open(path.join("images", self.model.__class__.__name__, f"{file_name_prefix}classification_report.txt"), "w") as f:
            f.write(str(report))