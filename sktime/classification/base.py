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
import pandas as pd
import time

from sktime.base import BaseEstimator
from sktime.utils.data_io import make_multi_index_dataframe
from sktime.utils.validation import check_n_jobs
from sktime.utils.validation.panel import check_X, check_X_y

def check_classifier_input(
        X,
        y,
        enforce_min_instances=1,
        enforce_min_series_length=1,
):
    """Check wether input X and y are valid formats with minimum data. Raises a
    ValueError if the input is not valid.

    Parameters
    ----------
    X : check whether a pd.DataFrame or np.ndarray
    y : check whether a pd.Series or np.array
    enforce_min_instances : int, optional (default=1)
        check there are a minimum number of instances.
    enforce_min_series_length : int, optional (default=1)
        Enforce minimum series length for input ndarray (i.e. fixed length problems)

    Raises
    ------
    ValueError
        If y or X is invalid input data type, or not enough data
    """
    # Check y
    if not isinstance(y, (pd.Series, np.ndarray)):
        raise ValueError(
            f"y must be a np.array or a pd.Series, "
            f"but found type: {type(y)}"
        )
    # Check X
    if not isinstance(X, (pd.DataFrame, np.ndarray)):
        raise ValueError(
            f"x must be either a pd.Series or a np.ndarray, "
            f"but found type: {type(X)}"
        )
    # Check size of X and y match and minimum data input
    n_cases = X.shape[0]
    n_labels = y.shape[0]
    if isinstance(X, np.ndarray):
            if not (X.ndim is 2 or X.ndim is 3):
                raise ValueError(
                    f"x is an np.ndarray, which means it must be 2 or 3 dimensional"
                    f"but found to be: {n_dims}"
                )
            if X.ndim is 2 and X.shape[1] < enforce_min_series_length:
                raise ValueError(
                    f"x is a 2D np.ndarray, equal length series are length {n_dims}"
                    f"but the minimum is  {enforce_min_series_length}"
                )
            if X.ndim is 3 and X.shape[2] < enforce_min_series_length:
                raise ValueError(
                    f"x is a 2D np.ndarray, equal length series are length {n_dims}"
                    f"but the minimum is  {enforce_min_series_length}"
                )

    else:
        if X.shape[1] is 0:
            raise ValueError(
                f"x is an pd.pandas with no data (num columns == 0)."
            )
    if n_cases < enforce_min_instances:
        raise ValueError(
            f"Minimum number of cases required is {enforce_min_instances} but X "
            f"has : {n_cases}"
        )
    if n_cases != n_labels:
        raise ValueError(
            f"Mismatch in number of cases. Number in X = {n_cases} nos in y = "
            f"{n_labels}"
        )

def test_check_classifier_input():
    """test for valid estimator format.
    1. Test correct: X: np.array of 2 and 3 dimensions vs y:np.array and np.Series
    2. Test correct: X: pd.DataFrame with 1 and 3 cols vs y:np.array and np.Series
    3. Test incorrect: X: np.array of 4 dimensions vs y:np.array
    4. Test incorrect: X: np.array of 3 dimensions vs y:List
    5. Test incorrect: mismatch in length
    6. Test incorrect: too small or too short
    """
# 1. Test correct: X: np.array of 2 and 3 dimensions vs y:np.array and np.Series
    test_X1 = np.random.uniform(-1, 1, size=(5, 10))
    test_X2 = np.random.uniform(-1, 1, size=(5, 2, 10))
    test_y1 = np.random.randint(0, 1, size=5)
    test_y2 = pd.Series(np.random.randn(5))
    check_classifier_input(test_X1, test_y1)
    check_classifier_input(test_X2, test_y1)
    check_classifier_input(test_X1, test_y2)
    check_classifier_input(test_X2, test_y2)
# # 2. Test correct: X: pd.DataFrame with 1 and 3 cols vs y:np.array and np.Series
    test_X3 = make_multi_index_dataframe(n_instances=5, n_columns=1, n_timepoints=10)
    test_X4 = make_multi_index_dataframe(n_instances=5, n_columns=3, n_timepoints=10)

    check_classifier_input(test_X3, test_y1)
#    check_classifier_input(test_X4, test_y1)
#    check_classifier_input(test_X3, test_y2)
#    check_classifier_input(test_X4, test_y2)
# # 3. Test incorrect: X: np.array of 4 dimensions vs y:np.array
#     test_X5 = np.random.uniform(-1, 1, size=(5, 3, 10))
#     check_classifier_input(test_X5, test_y1)
# # 4. Test incorrect: X: np.array of 4 dimensions vs y:List
#     test_y3 = [1, 2, 3, 4, 5]
#     check_classifier_input(test_X5, test_y3)
# # 5. Test incorrect: mismatch in length
#     test_y4 = np.random.randint(0, 1, size=6)
#     check_classifier_input(test_X1, test_y4)
# # 6. Test incorrect: too small or too short
#     check_classifier_input(test_X1, test_y1, enforce_min_instances=6)
#     check_classifier_input(test_X1, test_y1, enforce_min_series_length=11)


test_check_classifier_input()



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
