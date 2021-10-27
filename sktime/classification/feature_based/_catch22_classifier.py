# -*- coding: utf-8 -*-
"""Catch22 Classifier.

Pipeline classifier using the Catch22 transformer and an estimator.
"""

__author__ = ["Matthew Middlehurst", "RavenRudi"]
__all__ = ["Catch22Classifier"]

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.multiclass import class_distribution

from sktime.base._base import _clone_estimator
from sktime.classification.base import BaseClassifier
from sktime.transformations.panel.catch22 import Catch22


class Catch22Classifier(BaseClassifier):
    """Canonical Time-series Characteristics (catch22) classifier.

    This classifier simply transforms the input data using the Catch22 [1]
    transformer and builds a provided estimator using the transformed data.

    Parameters
    ----------
    outlier_norm : bool, default=False
        Normalise each series during the two outlier catch22 features, which can take a
        while to process for large values
    estimator : sklearn classifier, default=None
        An sklearn estimator to be built using the transformed data. Defaults to a
        Random Forest with 200 trees.
    n_jobs : int, default=1
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``-1`` means using all processors.
    random_state : int or None, default=None
        Seed for random, integer.

    Attributes
    ----------
    n_classes : int
        Number of classes. Extracted from the data.
    classes_ : ndarray of shape (n_classes)
        Holds the label for each class.

    See Also
    --------
    Catch22

    Notes
    -----
    Authors `catch22ForestClassifier <https://github.com/chlubba/sktime-catch22>`_.

    For the Java version, see `tsml <https://github.com/uea-machine-learning/tsml/blob
    /master/src/main/java/tsml/classifiers/hybrids/Catch22Classifier.java>`_.

    References
    ----------
    .. [1] Lubba, Carl H., et al. "catch22: Canonical time-series characteristics."
        Data Mining and Knowledge Discovery 33.6 (2019): 1821-1852.
        https://link.springer.com/article/10.1007/s10618-019-00647-x

    Examples
    --------
    >>> from sktime.classification.feature_based import Catch22Classifier
    >>> from sktime.datasets import load_italy_power_demand
    >>> X_train, y_train = load_italy_power_demand(split="train", return_X_y=True)
    >>> X_test, y_test = load_italy_power_demand(split="test", return_X_y=True)
    >>> clf = Catch22Classifier()
    >>> clf.fit(X_train, y_train)
    Catch22Classifier(...)
    >>> y_pred = clf.predict(X_test)
    """

    # Capability tags
    _tags = {
        "capability:multivariate": True,
        "capability:unequal_length": False,
        "capability:missing_values": False,
        "capability:train_estimate": False,
        "capability:contractable": False,
    }

    def __init__(
        self,
        outlier_norm=False,
        estimator=None,
        n_jobs=1,
        random_state=None,
    ):
        self.outlier_norm = outlier_norm
        self.estimator = estimator

        self.n_jobs = n_jobs
        self.random_state = random_state

        self._transformer = None
        self._estimator = None
        self.n_classes = 0
        self.classes_ = []
        super(Catch22Classifier, self).__init__()

    def _fit(self, X, y):
        """Fit an estimator using transformed data from the Catch22 transformer.

        Parameters
        ----------
        X : nested pandas DataFrame of shape [n_instances, n_dims]
            Nested dataframe with univariate time-series in cells.
        y : array-like, shape = [n_instances] The class labels.

        Returns
        -------
        self : object
        """
        self.classes_ = class_distribution(np.asarray(y).reshape(-1, 1))[0][0]
        self.n_classes = np.unique(y).shape[0]

        self._transformer = Catch22(outlier_norm=self.outlier_norm)
        self._estimator = _clone_estimator(
            RandomForestClassifier(n_estimators=200)
            if self.estimator is None
            else self.estimator,
            self.random_state,
        )

        m = getattr(self._estimator, "n_jobs", None)
        if m is not None:
            self._estimator.n_jobs = self.n_jobs

        X_t = self._transformer.fit_transform(X, y)
        X_t = np.nan_to_num(X_t, False, 0, 0, 0)
        self._estimator.fit(X_t, y)

        return self

    def _predict(self, X):
        """Predict class values of n_instances in X.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_instances, n_dims)

        Returns
        -------
        preds : np.ndarray of shape (n, 1)
            Predicted class.
        """
        X_t = self._transformer.transform(X)
        X_t = np.nan_to_num(X_t, False, 0, 0, 0)
        return self._estimator.predict(X_t)

    def _predict_proba(self, X):
        """Predict class probabilities for n_instances in X.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_instances, n_dims)

        Returns
        -------
        predicted_probs : array of shape (n_instances, n_classes)
            Predicted probability of each class.
        """
        X_t = self._transformer.transform(X)
        X_t = np.nan_to_num(X_t, False, 0, 0, 0)

        m = getattr(self._estimator, "predict_proba", None)
        if callable(m):
            return self._estimator.predict_proba(X_t)
        else:
            dists = np.zeros((X.shape[0], self.n_classes))
            preds = self._estimator.predict(X_t)
            for i in range(0, X.shape[0]):
                dists[i, np.where(self.classes_ == preds[i])] = 1
            return dists
