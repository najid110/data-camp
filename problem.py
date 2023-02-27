import os
import string
from glob import glob

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ShuffleSplit

import rampwf as rw
from rampwf.score_types.base import BaseScoreType


problem_title = "predicting-league-of-legends-winner"

_prediction_label_names = [1, 2]

Predictions = rw.prediction_types.make_multiclass(label_names=_prediction_label_names)

workflow = rw.workflows.Classifier()


class Accuracy(BaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name="accuracy_score", precision=4):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        acc = accuracy_score(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1))
        return acc


score_types = [
    Accuracy(name="accuracy_score")
]


def get_train_data(path="./"):
    train = pd.read_csv(os.path.join(path, "data", "train", 'train.csv'))
    X_train, y_train = train.drop(['winner'], axis=1), train['winner']
    X_train = X_train.reset_index()
    return X_train.values, y_train.values


def get_test_data(path="./"):
    test = pd.read_csv(os.path.join(path, "data", "test", "test.csv"))
    X_test, y_test = test.drop(['winner'], axis=1), test['winner']
    X_test = X_test.reset_index()
    return X_test.values, y_test.values


def get_cv(X, y):
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=1)
    return cv.split(X, y)
