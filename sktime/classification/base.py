# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""
Abstract base class for time series classifiers.

    class name: BaseClassifier

Defining methods:
    fitting         - fit(self, X, y)
    predicting      - predict(self, X)
                    - predict_proba(self, X)

Inspection methods:
    hyper-parameter inspection  - get_params()
    fitted parameter inspection - get_fitted_params()

State:
    fitted model/strategy   - by convention, any attributes ending in "_"
    fitted state flag       - is_fitted (property)
    fitted state inspection - check_is_fitted()
"""

__all__ = [
    "BaseClassifier",
]
__author__ = ["mloning", "fkiraly", "TonyBagnall", "MatthewMiddlehurst"]

import numpy as np
import time

from sktime.base import BaseEstimator
from sktime.datatypes._panel._convert import(
    from_3d_numpy_to_nested,
    from_nested_to_3d_numpy,
)
from sktime.datatypes._panel._check import is_nested_dataframe
from sktime.utils.validation import check_n_jobs
from sktime.utils.validation.panel import (
    check_X,
    check_X_y,
    check_classifier_input,
    get_data_characteristics,
)


class BaseClassifier(BaseEstimator):
    """Abstract base class for time series classifiers.

    The base classifier specifies the methods and method signatures that all
    classifiers have to implement.
    """

    _tags = {
        "coerce-X-to-numpy": True,
        "coerce-X-to-pandas": False,
        "capability:multivariate": False,
        "capability:unequal_length": False,
        "capability:missing_values": False,
        "capability:train_estimate": False,
        "capability:contractable": False,
        "capability:multithreading": False,
    }

    def __init__(self):
        self.classes_ = []
        self.n_classes_ = 0
        self.fit_time_ = 0
        self._class_dictionary = {}
        self._threads_to_use = 1
        super(BaseClassifier, self).__init__()

    def fit(self, X, y):
        """Fit time series classifier to training data.

        Parameters
        ----------
        X : 2D np.array (univariate, equal length series) of shape = [n_instances,
        series_length]
            or 3D np.array (any number of dimensions, equal length series) of shape =
            [n_instances,n_dimensions,series_length]
            or pd.DataFrame with each column a dimension, each cell a pd.Series (any
            number of dimensions, equal or unequal length series)
        y : 1D np.array of shape =  [n_instances] - the class labels.

        Returns
        -------
        self :
            Reference to self.

        Notes
        -----
        Changes state by creating a fitted model that updates attributes
        ending in "_" and sets is_fitted flag to True.
        """
        start = int(round(time.time() * 1000))
        coerce_to_numpy = self.get_tag("coerce-X-to-numpy")
        coerce_to_pandas = self.get_tag("coerce-X-to-pandas")
        allow_multivariate = self.get_tag("capability:multivariate")

        check_classifier_input(X)
        missing, multivariate, unequal = get_data_characteristics(X)
        check_capabilities(self, missing, multivariate, unequal)
        #convert X and y if necessary, only if equal length
        X, y = check_X_y(
            X,
            y,
            coerce_to_numpy=coerce_to_numpy,
            coerce_to_pandas=coerce_to_pandas,
            enforce_univariate=not allow_multivariate,
        )

        multithread = self.get_tag("capability:multithreading")
        if multithread:
            try:
                self._threads_to_use = check_n_jobs(self.n_jobs)
            except NameError:
                raise AttributeError(
                    "self.n_jobs must be set if capability:multithreading is True"
                )

        self.classes_ = np.unique(y)
        self.n_classes_ = self.classes_.shape[0]
        for index, classVal in enumerate(self.classes_):
            self._class_dictionary[classVal] = index

        self._fit(X, y)

        # this should happen last
        self._is_fitted = True

        fit_time_ = int(round(time.time() * 1000)) - start
        return self

    def predict(self, X) -> np.array:
        """Predicts labels for sequences in X.

        Parameters
        ----------
        X : 2D np.array (univariate, equal length series) of shape = [n_instances,
        series_length]
            or 3D np.array (any number of dimensions, equal length series) of shape =
            [n_instances,n_dimensions,series_length]
            or pd.DataFrame with each column a dimension, each cell a pd.Series (any
            number of dimensions, equal or unequal length series)

        Returns
        -------
        y : 1D np.array of shape =  [n_instances] - predicted class labels
        """
        self.check_is_fitted()

        coerce_to_numpy = self.get_tag("coerce-X-to-numpy")
        coerce_to_pandas = self.get_tag("coerce-X-to-pandas")
        allow_multivariate = self.get_tag("capability:multivariate")
        X = check_X(
            X,
            coerce_to_numpy=coerce_to_numpy,
            coerce_to_pandas=coerce_to_pandas,
            enforce_univariate=not allow_multivariate,
        )

        return self._predict(X)

    def predict_proba(self, X) -> np.array:
        """Predicts labels probabilities for sequences in X.

        Parameters
        ----------
        X : 2D np.array (univariate, equal length series) of shape = [n_instances,
        series_length]
            or 3D np.array (any number of dimensions, equal length series) of shape =
            [n_instances,n_dimensions,series_length]
            or pd.DataFrame with each column a dimension, each cell a pd.Series (any
            number of dimensions, equal or unequal length series)

        Returns
        -------
        y : 2D array of shape =  [n_instances, n_classes] - estimated class
        probabilities
        """
        self.check_is_fitted()

        coerce_to_numpy = self.get_tag("coerce-X-to-numpy")
        coerce_to_pandas = self.get_tag("coerce-X-to-pandas")
        allow_multivariate = self.get_tag("capability:multivariate")
        X = check_X(
            X,
            coerce_to_numpy=coerce_to_numpy,
            coerce_to_pandas=coerce_to_pandas,
            enforce_univariate=not allow_multivariate,
        )

        return self._predict_proba(X)

    def score(self, X, y) -> float:
        """Scores predicted labels against ground truth labels on X.

        Parameters
        ----------
        X : 2D np.array (univariate, equal length series) of shape = [n_instances,
        series_length]
            or 3D np.array (any number of dimensions, equal length series) of shape =
            [n_instances,n_dimensions,series_length]
            or pd.DataFrame with each column a dimension, each cell a pd.Series (any
            number of dimensions, equal or unequal length series)
        y : array-like, shape =  [n_instances] - actual class labels

        Returns
        -------
        float, accuracy score of predict(X) vs y
        """
        from sklearn.metrics import accuracy_score

        return accuracy_score(y, self.predict(X), normalize=True)

    def _fit(self, X, y):
        """Fit time series classifier to training data.

        Abstract method, must be implemented.

        Parameters
        ----------
        X : 3D np.array, array-like or sparse matrix
                of shape = [n_instances,n_dimensions,series_length]
                or shape = [n_instances,series_length]
            or pd.DataFrame with each column a dimension, each cell a pd.Series
        y : array-like, shape = [n_instances] - the class labels

        Returns
        -------
        self :
            Reference to self.

        Notes
        -----
        Changes state by creating a fitted model that updates attributes
        ending in "_" and sets is_fitted flag to True.
        """
        raise NotImplementedError(
            "_fit is a protected abstract method, it must be implemented."
        )

    def _predict(self, X) -> np.array:
        """Predicts labels for sequences in X.

        Abstract method, must be implemented.

        Parameters
        ----------
        X : 3D np.array, array-like or sparse matrix
                of shape = [n_instances,n_dimensions,series_length]
                or shape = [n_instances,series_length]
            or pd.DataFrame with each column a dimension, each cell a pd.Series

        Returns
        -------
        y : array-like, shape =  [n_instances] - predicted class labels
        """
        raise NotImplementedError(
            "_predict is a protected abstract method, it must be implemented."
        )

    def _predict_proba(self, X) -> np.ndarray:
        """Predicts labels probabilities for sequences in X.

        Default behaviour is to call _predict and set the predicted class probability
        to 1, other class probabilities to 0. Override if better estimates are
        obtainable.

        Parameters
        ----------
        X : 3D np.array, array-like or sparse matrix
                of shape = [n_instances,n_dimensions,series_length]
                or shape = [n_instances,series_length]
            or pd.DataFrame with each column a dimension, each cell a pd.Series

        Returns
        -------
        y : array-like, shape =  [n_instances, n_classes] - estimated probabilities
        of class membership.
        """
        dists = np.zeros((X.shape[0], self.n_classes_))
        preds = self._predict(X)
        for i in range(0, X.shape[0]):
            dists[i, self._class_dictionary[preds[i]]] = 1

        return dists


def check_capabilities(self, missing, multivariate, unequal):
    """Check wether this classifier can handle the data characteristics.
    Attributes
    ----------
    missing : boolean, does the data passed to fit contain missing values?
    multivariate : boolean, does the data passed to fit contain missing values?
    unequal : boolea, do the time series passed to fit have variable lengths?

    Raises
    ------
    ValueError if the capabilities in self._tags do not handle the data.

    """
    allow_multivariate = self.get_tag("capability:multivariate")
    allow_missing = self.get_tag("capability:missing_values")
    allow_unequal = self.get_tag("capability:missing_values")
    if missing and not allow_missing:
        raise ValueError("The data has missing values, this classifier cannot handle "
                         "missing values")
    if multivariate and not allow_multivariate:
        raise ValueError("The data is multivariate, this classifier cannot handle "
                         "multivariate time serries")
    if unequal and not allow_unequal:
        raise ValueError("The data has unequal length series, this classifier cannot "
                         "handle unequal length series")


def convert_data(self, X):
    """Convert equal length series from pandas to numpy or vice versa.

    Parameters
    ----------
    self : this classifier
    X : pd.DataFrame or np.array
        Input data

    Returns
    -------
    X : pd.DataFrame or np.array
        Checked and possibly converted input data

    Raises
    ------
    ValueError
        If X is invalid input data
    """
    convert_to_numpy = self.get_tag("coerce-X-to-numpy")
    convert_to_pandas = self.get_tag("coerce-X-to-pandas")
    if convert_to_numpy and convert_to_pandas:
        raise ValueError("Tag error: cannot set both coerce-X-to-numpy and "
                         "coerce-X-to-pandas to be true.")
    # check pd.DataFrame
    if isinstance(X, pd.DataFrame):
        if not is_nested_dataframe(X):
            raise ValueError(
                "If passed as a pd.DataFrame, X must be a nested "
                "pd.DataFrame, with pd.Series or np.arrays inside cells."
            )
        # convert pd.DataFrame
        if convert_to_numpy:
            X = from_nested_to_3d_numpy(X)
    elsif isinstance(X,np.ndarray):
    # Temporary fix to insist on 3D numpy. For univariate problems, most classifiers
    # simply convert back to 2D. This squashing should be done here, but touches a
    # lot of files, so will get this to work first.
        if X.ndims == 2:

        X=X
    return X
