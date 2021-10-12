# -*- coding: utf-8 -*-
"""Martrix Profile classifier.

Pipeline classifier using the Matrix Profile transformer and an estimator.
"""

__author__ = ["MatthewMiddlehurst"]
__all__ = ["MatrixProfileClassifier"]

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from sktime.base._base import _clone_estimator
from sktime.classification.base import BaseClassifier
from sktime.transformations.panel.matrix_profile import MatrixProfile


class MatrixProfileClassifier(BaseClassifier):
    """Martrix Profile (MP) classifier.

    This classifier simply transforms the input data using the MatrixProfile [1]
    transformer and builds a provided estimator using the transformed data.

    Parameters
    ----------
    subsequence_length : int, default=10
        The subsequence length for the MatrixProfile transformer.
    estimator : sklearn classifier, default=None
        An sklearn estimator to be built using the transformed data. Defaults to a
        1-nearest neighbour classifier.
    n_jobs : int, default=1
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``-1`` means using all processors.
    random_state : int or None, default=None
        Seed for random, integer.

    Attributes
    ----------
    n_classes_ : int
        Number of classes. Extracted from the data.
    classes_ : ndarray of shape (n_classes)
        Holds the label for each class.

    See Also
    --------
    MatrixProfile

    References
    ----------
    .. [1] Yeh, Chin-Chia Michael, et al. "Time series joins, motifs, discords and
        shapelets: a unifying view that exploits the matrix profile." Data Mining and
        Knowledge Discovery 32.1 (2018): 83-123.
        https://link.springer.com/article/10.1007/s10618-017-0519-9

    Examples
    --------
    >>> from sktime.classification.feature_based import MatrixProfileClassifier
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train", return_X_y=True)
    >>> X_test, y_test = load_unit_test(split="test", return_X_y=True)
    >>> clf = MatrixProfileClassifier()
    >>> clf.fit(X_train, y_train)
    MatrixProfileClassifier(...)
    >>> y_pred = clf.predict(X_test)
    """

    _tags = {
        "capability:multivariate": False,
        "capability:unequal_length": False,
        "capability:missing_values": False,
        "capability:train_estimate": False,
        "capability:contractable": False,
    }

    def __init__(
        self,
        subsequence_length=10,
        estimator=None,
        n_jobs=1,
        random_state=None,
    ):
        self.subsequence_length = subsequence_length
        self.estimator = estimator

        self.n_jobs = n_jobs
        self.random_state = random_state

        self._transformer = None
        self._estimator = None
        self.n_classes_ = 0
        self.classes_ = []

        super(MatrixProfileClassifier, self).__init__()

    def _fit(self, X, y):
        self.classes_ = np.unique(y)
        self.n_classes_ = self.classes_.shape[0]

        self._transformer = MatrixProfile(m=self.subsequence_length)
        self._estimator = _clone_estimator(
            KNeighborsClassifier(n_neighbors=1)
            if self.estimator is None
            else self.estimator,
            self.random_state,
        )

        m = getattr(self._estimator, "n_jobs", None)
        if m is not None:
            self._estimator.n_jobs = self.n_jobs

        X_t = self._transformer.fit_transform(X, y)
        self._estimator.fit(X_t, y)

    def _predict(self, X):
        return self._estimator.predict(self._transformer.transform(X))

    def _predict_proba(self, X):
        m = getattr(self._estimator, "predict_proba", None)
        if callable(m):
            return self._estimator.predict_proba(self._transformer.transform(X))
        else:
            dists = np.zeros((X.shape[0], self.n_classes_))
            preds = self._estimator.predict(self._transformer.transform(X))
            for i in range(0, X.shape[0]):
                dists[i, np.where(self.classes_ == preds[i])] = 1
            return dists
