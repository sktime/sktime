# -*- coding: utf-8 -*-
"""Classifier Examples: some use case examples for building and assessing classifiers.

This will become a note book once complete.
"""

__author__ = ["TonyBagnall"]

import numpy as np

from sklearn.ensemble import RandomForestClassifier
#from sktime.classification.hybrid import HIVECOTEV2
from sktime.classification.interval_based import TimeSeriesForestClassifier

def build_classifiers():
    """Examples of building a classifier.

    1. Directly from 2D numpy arrays.
    2. Directly from 3D numpy arrays.
    3. From a nested pandas.
    4. From a baked in dataset.
    5. From any UCR/UEA dataset downloaded from timeseriesclassification.com.
    """
    # Create an array
    # Random forest, rocket and HC2.
    randf = RandomForestClassifier()
    trainX, train_y, testX, test_y = make_toy_problem()
    randf.fit(trainX, train_y)
    print("Fit complete")
    print(" rand f acc = ", randf.score(testX, test_y))
    hc2 = TimeSeriesForestClassifier(n_estimators=3)
    hc2.fit(trainX, train_y)
    print(" rand f acc = ", hc2.score(testX, test_y))


def make_toy_problem():
    """Makes a toy classification problem out of numpy arrays."""
    X_train_class1 = np.random.uniform(-1, 1, size=(20, 100))
    y_train_class1 = np.zeros(20,)
    X_train_class2 = np.random.uniform(-0.9, 1.1, size=(20, 100))
    y_train_class2 = np.ones(20,)
    X_train = np.concatenate((X_train_class1, X_train_class2), axis=0)
    y_train = np.concatenate((y_train_class1, y_train_class2))

    X_test_class1 = np.random.uniform(-1, 1, size=(20, 100))
    y_test_class1 = np.zeros(20,)
    X_test_class2 = np.random.uniform(-0.9, 1.1, size=(20, 100))
    y_test_class2 = np.ones(20,)
    X_test = np.concatenate((X_test_class1, X_test_class2), axis=0)
    y_test = np.concatenate((y_test_class1, y_test_class2))
    return X_train, y_train, X_test, y_test


def compare_classifiers():
    """Build pipeline classifiers and compare to published results."""
    # Data set list

    # Define Transformer pipeline

    # fit and score for each dataset

    # Pull down accuracies

    # Draw CD diagram

build_classifiers()
