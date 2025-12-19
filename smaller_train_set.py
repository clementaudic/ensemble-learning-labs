from utils import ModelTester
from xgboost import XGBClassifier

params = {'learning_rate': 0.05, 'max_depth': 6, 'n_estimators': 400}

test_scores = []

train_set_sizes = []
trainings_times = []

for i in [0.5, 0.2, 0.1, 0.05, 0.01, 0.001]:
    clf = XGBClassifier(**params)
    tester = ModelTester(clf, reduce_train=i)

    tester.test_model()
    test_scores.append((tester.test_score))
    train_set_sizes.append(tester.X_train.shape[0])
    trainings_times.append(tester.train_time)

from matplotlib import pyplot as plt

plt.xscale("log")
plt.plot(train_set_sizes,test_scores, marker='o')
plt.xticks(train_set_sizes, train_set_sizes)
plt.gca().invert_xaxis() 
plt.xlabel("Training set size")
plt.ylabel("Test Accuracy")
plt.title("Impact of training set size on XGBoost test accuracy")
plt.savefig("images/XGBClassifier/smaller_train_set.png")
plt.close()

plt.plot(train_set_sizes,trainings_times, marker='o')
plt.gca().invert_xaxis() 
plt.xlabel("Training set size")
plt.ylabel("Training Time (seconds)")
plt.title("Impact of training set size on XGBoost training time")
plt.savefig("images/XGBClassifier/smaller_train_set_time.png")