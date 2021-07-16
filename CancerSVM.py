#Predicting whether a tumor is malignant or benign using SVMs

import sklearn
from sklearn import datasets, svm, metrics
from sklearn.neighbors import KNeighborsClassifier
cancer = datasets.load_breast_cancer()
# print(cancer.feature_names)
# print(cancer.target_names)

X = cancer.data
y = cancer.target
sumAcc = 0
rangeNum = 10
for i in range(rangeNum):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=.1)

    # print(x_train, " ", y_train)
    classes = ['malignant', 'benign']

    clf = svm.SVC(kernel="linear", C=3)
    clf.fit(x_train, y_train)

    yPred = clf.predict(x_test)
    acc = metrics.accuracy_score(y_test, yPred)
    print(acc)
    sumAcc += acc

averageAcc = sumAcc / rangeNum
print("Average Accuracy: ", averageAcc)