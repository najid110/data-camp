import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator


class Classifier(BaseEstimator):
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)

    def fit(self, X, Y):
        self.model.fit(X, Y)

    def predict_proba(self, X):
        res = self.model.predict_proba(X)
        return res
