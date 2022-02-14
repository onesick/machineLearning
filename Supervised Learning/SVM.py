import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
import time

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import reciprocal, uniform

class SVM():
    def __init__(self):
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("linear_svc", LinearSVC(C=1, loss="hinge")),
        ])

    def setPolyKernal(self):
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=5))
        ])

    def setGaussianKernal(self):
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("svm_clf", SVC(kernel="rbf", gamma=5, C=0.001))
        ])

    def fit(self, X_train, y_train):
        start_time = time.time()
        self.model.fit(X_train, y_train)
        end_time = time.time()
        print('\n\nFitting Training Set: {:.4f} seconds'.format(end_time-start_time))

    def predict(self, X_test):
        return self.model.predict(X_test)    

    def findBestParams(self, X_train, y_train, param_distributions = {"gamma": reciprocal(0.001, 0.1), "C": uniform(1, 10)}):
        rnd_search_cv = RandomizedSearchCV(self.model[1], param_distributions, n_iter=10, verbose=2, cv=3)
        rnd_search_cv.fit(X_train, y_train)

        return rnd_search_cv