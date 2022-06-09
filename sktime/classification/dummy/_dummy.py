# -*- coding: utf-8 -*-
"""Dummy time series classifier."""

__author__ = ["ZiyaoWei"]
__all__ = ["DummyClassifier"]

import numpy as np
from sklearn.dummy import DummyClassifier as SklearnDummyClassifier

from sktime.classification.base import BaseClassifier


class DummyClassifier(BaseClassifier):
    """DummyClassifier makes predictions that ignore the input features.

    This classifier serves as a simple baseline to compare against other more
    complex classifiers, similar to sklearn.dummy.DummyClassifier.
    """

    _tags = {
        "X_inner_mtype": "nested_univ",
        "capability:missing_values": True,
        "capability:unequal_length": True,
    }

    def __init__(self, strategy="prior", random_state=None, constant=None):
        self.strategy = strategy
        self.random_state = random_state
        self.constant = constant
        self.sklearn_dummy_classifier = SklearnDummyClassifier(
            strategy=strategy, random_state=random_state, constant=constant
        )
        super(DummyClassifier, self).__init__()

    def _fit(self, X, y):
        """Fit the dummy classifier.

        Parameters
        ----------
        X : sktime-format pandas dataframe with shape(n,d),
        or numpy ndarray with shape(n,d,m)
        y : array-like, shape = [n_instances] - the class labels

        Returns
        -------
        self : reference to self.
        """
        self.sklearn_dummy_classifier.fit(np.zeros(X.shape), y)
        return self

    def _predict(self, X) -> np.ndarray:
        """Predict labels for sequences in X.

        Parameters
        ----------
        X : sktime-format pandas dataframe or array-like, shape (n, d)

        Returns
        -------
        y : predictions of labels for X, np.ndarray
        """
        return self.sklearn_dummy_classifier.predict(np.zeros(X.shape))

    def _predict_proba(self, X) -> np.ndarray:
        """Predicts labels probabilities for sequences in X.

        Parameters
        ----------
        X : sktime-format pandas dataframe or array-like, shape (n, d)

        Returns
        -------
        y : predictions of probabilities for class values of X, np.ndarray
        """
        return self.sklearn_dummy_classifier.predict_proba(np.zeros(X.shape))
