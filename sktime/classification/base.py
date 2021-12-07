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

import time

import numpy as np
import pandas as pd

from sktime.base import BaseEstimator
from sktime.datatypes._panel._check import is_nested_dataframe
from sktime.datatypes._panel._convert import (
    from_3d_numpy_to_nested,
    from_nested_to_3d_numpy,
)
from sktime.utils.validation import check_n_jobs


class BaseClassifier(BaseEstimator):
    """Abstract base class for time series classifiers.

    The base classifier specifies the methods and method signatures that all
    classifiers have to implement. Attributes with a underscore suffix are set in the
    method fit.
    #TODO: Make _fit and _predict abstract

    Attributes
    ----------
    classes_            : ndarray of class labels, possibly strings
    n_classes_          : integer, number of classes (length of classes_)
    fit_time_           : integer, time (in milliseconds) for fit to run.
    _class_dictionary   : dictionary mapping classes_ onto integers 0...n_classes_-1.
    _threads_to_use     : number of threads to use in fit.
    """

    _tags = {
        "convert_X_to_numpy": True,
        "convert_X_to_dataframe": False,
        "convert_y_to_numpy": True,
        "convert_y_to_series": False,
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
        # Check the data is either numpy arrays or pandas dataframes
        _check_classifier_input(X, y)
        # Query the data for characteristics
        missing, multivariate, unequal = _get_data_characteristics(X)
        # Check this classifier can handle characteristics
        self._check_capabilities(missing, multivariate, unequal)
        # Convert data as dictated by the classifier tags
        X = self._convert_X(X)
        y = self._convert_y(y)
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
        self.fit_time_ = int(round(time.time() * 1000)) - start
        # this should happen last
        self._is_fitted = True
        return self

    def predict(self, X) -> np.ndarray:
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

        # Check the data is either numpy arrays or pandas dataframes
        _check_classifier_input(X)
        # Query the data for characteristics
        missing, multivariate, unequal = _get_data_characteristics(X)
        # Check this classifier can handle characteristics
        self._check_capabilities(missing, multivariate, unequal)
        # Convert data as dictated by the classifier tags
        X = self._convert_X(X)

        return self._predict(X)

    def predict_proba(self, X) -> np.ndarray:
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

        # Check the data is either numpy arrays or pandas dataframes
        # TODO: add parameters for min instances and min length
        _check_classifier_input(X)
        # Query the data for characteristics
        missing, multivariate, unequal = _get_data_characteristics(X)
        # Check this classifier can handle characteristics
        self._check_capabilities(missing, multivariate, unequal)
        # Convert data as dictated by the classifier tags
        X = self._convert_X(X)

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

    def _predict(self, X) -> np.ndarray:
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

    def _check_capabilities(self, missing, multivariate, unequal):
        """Check whether this classifier can handle the data characteristics.

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
        allow_unequal = self.get_tag("capability:unequal_length")
        if missing and not allow_missing:
            raise ValueError(
                "The data has missing values, this classifier cannot handle missing "
                "values"
            )
        if multivariate and not allow_multivariate:
            # this error message could be more informative, but it is for backward
            # compatibility with the testing functions
            raise ValueError(
                "X must be univariate, this classifier cannot deal with "
                "multivariate input."
            )
        if unequal and not allow_unequal:
            raise ValueError(
                "The data has unequal length series, this classifier cannot handle "
                "unequal length series"
            )

    def _convert_X(self, X):
        """Convert equal length series from DataFrame to numpy array or vice versa.

        Parameters
        ----------
        self : this classifier
        X : pd.DataFrame or np.ndarray. Input attribute data

        Returns
        -------
        X : pd.DataFrame or np.array
            Checked and possibly converted input data
        """
        convert_to_numpy = self.get_tag("convert_X_to_numpy")
        convert_to_pandas = self.get_tag("convert_X_to_dataframe")
        if convert_to_numpy and convert_to_pandas:
            raise ValueError(
                "Tag error: cannot set both convert_X_to_numpy and "
                "convert_X_to_dataframe to be true."
            )
        # convert pd.DataFrame
        if isinstance(X, np.ndarray):
            # Temporary fix to insist on 3D numpy. For univariate problems,
            # most classifiers simply convert back to 2D. This squeezing should be
            # done here, but touches a lot of files, so will get this to work first.
            if X.ndim == 2:
                X = X.reshape(X.shape[0], 1, X.shape[1])

        if convert_to_numpy:
            if isinstance(X, pd.DataFrame):
                X = from_nested_to_3d_numpy(X)
        elif convert_to_pandas:
            if isinstance(X, np.ndarray):
                X = from_3d_numpy_to_nested(X)
        return X

    def _convert_y(self, y):
        """Convert y into a pd.Series or an np.array, depending on convert tags.

        y is the target variable.

        Parameters
        ----------
        self : this classifier
        y : np.array or pd.Series.

        Returns
        -------
        y: pd.Series or np.ndarray
        """
        if isinstance(y, pd.Series):
            if self.get_tag("convert_y_to_numpy"):
                y = pd.Series.to_numpy(y)
        elif isinstance(y, np.ndarray):
            if self.get_tag("convert_y_to_series"):
                y = pd.Series(y)
        return y


def _check_classifier_input(
    X,
    y=None,
    enforce_min_instances=1,
    enforce_min_series_length=1,
):
    """Check whether input X and y are valid formats with minimum data.

    Raises a ValueError if the input is not valid.

    Arguments
    ---------
    X : check whether a pd.DataFrame or np.ndarray
    y : check whether a pd.Series or np.array
    enforce_min_instances : int, optional (default=1)
        check there are a minimum number of instances.
    enforce_min_series_length : int, optional (default=1)
        Enforce minimum series length for input ndarray (i.e. fixed length problems)

    Raises
    ------
    ValueError
        If y or X is invalid input data type, or there is not enough data
    """
    # Check X
    if not isinstance(X, (pd.DataFrame, np.ndarray)):
        raise ValueError(
            f"X must be either a pd.DataFrame or a np.ndarray, "
            f"but found type: {type(X)}"
        )
    n_cases = X.shape[0]
    if isinstance(X, np.ndarray):
        if not (X.ndim == 2 or X.ndim == 3):
            raise ValueError(
                f"x is an np.ndarray, which means it must be 2 or 3 dimensional"
                f"but found to be: {X.ndim}"
            )
        if X.ndim == 2 and X.shape[1] < enforce_min_series_length:
            raise ValueError(
                f"Series length below the minimum, equal length series are length"
                f" {X.shape[1]}"
                f"but the minimum is  {enforce_min_series_length}"
            )
        if X.ndim == 3 and X.shape[2] < enforce_min_series_length:
            raise ValueError(
                f"Series length below the minimum, equal length series are length"
                f" {X.shape[2]}"
                f"but the minimum is  {enforce_min_series_length}"
            )
    else:
        if X.shape[1] == 0:
            raise ValueError("x is a pd.DataFrame with no data (num columns == 0).")
    if n_cases < enforce_min_instances:
        raise ValueError(
            f"Minimum number of cases required is {enforce_min_instances} but X "
            f"has : {n_cases}"
        )
    if isinstance(X, pd.DataFrame):
        if not is_nested_dataframe(X):
            raise ValueError(
                "If passed as a pd.DataFrame, X must be a nested "
                "pd.DataFrame, with pd.Series or np.arrays inside cells."
            )
    # Check y if passed
    if y is not None:
        # Check y valid input
        if not isinstance(y, (pd.Series, np.ndarray)):
            raise ValueError(
                f"y must be a np.array or a pd.Series, but found type: {type(y)}"
            )
        # Check matching number of labels
        n_labels = y.shape[0]
        if n_cases != n_labels:
            raise ValueError(
                f"Mismatch in number of cases. Number in X = {n_cases} nos in y = "
                f"{n_labels}"
            )
        if isinstance(y, np.ndarray):
            if y.ndim > 1:
                raise ValueError(
                    f"y must be 1-dimensional but is in fact " f"{y.ndim} dimensional"
                )


def _get_data_characteristics(X):
    """Query the data to find its characteristics for classifier capability check.

    This is for checking array input where we assume series are equal length.
    classifiers can take either ndarray or pandas input, and the checks are different
    for each. For ndarrays, it checks:
        a) whether x contains missing values;
        b) whether x is multivariate.
    for pandas it checks
        a) whether x contains unequal length series;
        a) whether x contains missing values;
        b) whether x is multivariate.

    Parameters
    ----------
    X : pd.pandas containing pd.Series or np.ndarray of either 2 or 3 dimensions.

    Returns
    -------
    three boolean data characteristics: missing, multivariate and unequal
    """
    if isinstance(X, np.ndarray):
        missing = _has_nans(X)
        if X.ndim == 3 and X.shape[1] > 1:
            multivariate = True
        else:
            multivariate = False
        return missing, multivariate, False
    else:
        missing = _nested_dataframe_has_nans(X)
        cols = len(X.columns)
        if cols > 1:
            multivariate = True
        else:
            multivariate = False
        unequal = _nested_dataframe_has_unequal(X)
        return missing, multivariate, unequal


def _nested_dataframe_has_unequal(X: pd.DataFrame) -> bool:
    """Check whether an input nested DataFrame of Series has unequal length series.

    Parameters
    ----------
    X : pd.DataFrame where each cell is a pd.Series

    Returns
    -------
    True if x contains any NaNs, False otherwise.
    """
    rows = len(X)
    cols = len(X.columns)
    s = X.iloc[0, 0]
    length = len(s)

    for i in range(0, rows):
        for j in range(0, cols):
            s = X.iloc[i, j]
            temp = len(s)
            if temp != length:
                return True
    return False


def _nested_dataframe_has_nans(X: pd.DataFrame) -> bool:
    """Check whether an input pandas of Series has nans.

    Parameters
    ----------
    X : pd.DataFrame where each cell is a pd.Series

    Returns
    -------
    True if x contains any NaNs, False otherwise.
    """
    cases = len(X)
    dimensions = len(X.columns)
    for i in range(0, cases):
        for j in range(0, dimensions):
            s = X.iloc[i, j]
            for k in range(0, s.size):
                if pd.isna(s[k]):
                    return True
    return False


# @njit(cache=True)
def _has_nans(x: np.ndarray) -> bool:
    """Check whether an input numpy array has nans.

    Parameters
    ----------
    X : np.ndarray of either 2 or 3 dimensions.

    Returns
    -------
    True if x contains any NaNs, False otherwise.
    """
    # 2D
    if x.ndim == 2:
        for i in range(0, x.shape[0]):
            for j in range(0, x.shape[1]):
                if np.isnan(x[i][j]):
                    return True
    elif x.ndim == 3:
        for i in range(0, x.shape[0]):
            for j in range(0, x.shape[1]):
                for k in range(0, x.shape[2]):
                    if np.isnan(x[i][j][k]):
                        return True
    else:
        raise ValueError(f"Expected array of two or three dimensions, got {x.ndim}")
    return False
