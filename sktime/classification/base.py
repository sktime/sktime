# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""
Abstract base class for time series classifiers.

    class name: BaseClassifier

Defining methods:
    fitting         - fit(self, X, y)
    predicting      - predict(self, X)
                    - predict_proba(self, X)

Inherited inspection methods:
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
from sktime.datatypes import check_is_scitype, convert_to
from sktime.utils.validation import check_n_jobs


class BaseClassifier(BaseEstimator):
    """Abstract base class for time series classifiers.

    The base classifier specifies the methods and method signatures that all
    classifiers have to implement. Attributes with a underscore suffix are set in the
    method fit.
    #TODO: Make _fit and _predict abstract

    Parameters
    ----------
    classes_            : ndarray of class labels, possibly strings
    n_classes_          : integer, number of classes (length of classes_)
    fit_time_           : integer, time (in milliseconds) for fit to run.
    _class_dictionary   : dictionary mapping classes_ onto integers 0...n_classes_-1.
    _threads_to_use     : number of threads to use in fit as determined by n_jobs.
    """

    _tags = {
        "X_inner_mtype": "numpy3D",  # which type do _fit/_predict, support for X?
        #    it should be either "numpy3D" or "nested_univ" (nested pd.DataFrame)
        "capability:multivariate": False,
        "capability:unequal_length": False,
        "capability:missing_values": False,
        "capability:train_estimate": False,
        "capability:contractable": False,
        "capability:early_prediction": False,
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
        X : 3D np.array (any number of dimensions, equal length series)
                of shape [n_instances, n_dimensions, series_length]
            or 2D np.array (univariate, equal length series)
                of shape [n_instances, series_length]
            or pd.DataFrame with each column a dimension, each cell a pd.Series
                (any number of dimensions, equal or unequal length series)
            or of any other supported Panel mtype
                for list of mtypes, see datatypes.SCITYPE_REGISTER
                for specifications, see examples/AA_datatypes_and_datasets.ipynb
        y : 1D np.array of int, of shape [n_instances] - class labels for fitting
            indices correspond to instance indices in X

        Returns
        -------
        self : Reference to self.

        Notes
        -----
        Changes state by creating a fitted model that updates attributes
        ending in "_" and sets is_fitted flag to True.
        """
        start = int(round(time.time() * 1000))
        # convenience conversions to allow user flexibility:
        # if X is 2D array, convert to 3D, if y is Series, convert to numpy
        X, y = _internal_convert(X, y)
        X_metadata = _check_classifier_input(X, y)
        missing = X_metadata["has_nans"]
        multivariate = not X_metadata["is_univariate"]
        unequal = not X_metadata["is_equal_length"]
        # Check this classifier can handle characteristics
        self._check_capabilities(missing, multivariate, unequal)
        # Convert data as dictated by the classifier tags
        X = self._convert_X(X)
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
        self._class_dictionary = {}
        for index, class_val in enumerate(self.classes_):
            self._class_dictionary[class_val] = index
        self._fit(X, y)
        self.fit_time_ = int(round(time.time() * 1000)) - start
        # this should happen last
        self._is_fitted = True
        return self

    def predict(self, X) -> np.ndarray:
        """Predicts labels for sequences in X.

        Parameters
        ----------
        X : 3D np.array (any number of dimensions, equal length series)
                of shape [n_instances, n_dimensions, series_length]
            or 2D np.array (univariate, equal length series)
                of shape [n_instances, series_length]
            or pd.DataFrame with each column a dimension, each cell a pd.Series
                (any number of dimensions, equal or unequal length series)
            or of any other supported Panel mtype
                for list of mtypes, see datatypes.SCITYPE_REGISTER
                for specifications, see examples/AA_datatypes_and_datasets.ipynb

        Returns
        -------
        y : 1D np.array of int, of shape [n_instances] - predicted class labels
            indices correspond to instance indices in X
        """
        self.check_is_fitted()

        # boilerplate input checks for predict-like methods
        X = self._check_convert_X_for_predict(X)

        return self._predict(X)

    def predict_proba(self, X) -> np.ndarray:
        """Predicts labels probabilities for sequences in X.

        Parameters
        ----------
        X : 3D np.array (any number of dimensions, equal length series)
                of shape [n_instances, n_dimensions, series_length]
            or 2D np.array (univariate, equal length series)
                of shape [n_instances, series_length]
            or pd.DataFrame with each column a dimension, each cell a pd.Series
                (any number of dimensions, equal or unequal length series)
            or of any other supported Panel mtype
                for list of mtypes, see datatypes.SCITYPE_REGISTER
                for specifications, see examples/AA_datatypes_and_datasets.ipynb

        Returns
        -------
        y : 2D array of shape [n_instances, n_classes] - predicted class probabilities
            1st dimension indices correspond to instance indices in X
            2nd dimension indices correspond to possible labels (integers)
            (i, j)-th entry is predictive probability that i-th instance is of class j
        """
        self.check_is_fitted()

        # boilerplate input checks for predict-like methods
        X = self._check_convert_X_for_predict(X)
        return self._predict_proba(X)

    def score(self, X, y) -> float:
        """Scores predicted labels against ground truth labels on X.

        Parameters
        ----------
        X : 3D np.array (any number of dimensions, equal length series)
                of shape [n_instances, n_dimensions, series_length]
            or 2D np.array (univariate, equal length series)
                of shape [n_instances, series_length]
            or pd.DataFrame with each column a dimension, each cell a pd.Series
                (any number of dimensions, equal or unequal length series)
            or of any other supported Panel mtype
                for list of mtypes, see datatypes.SCITYPE_REGISTER
                for specifications, see examples/AA_datatypes_and_datasets.ipynb
        y : 1D np.ndarray of int, of shape [n_instances] - class labels (ground truth)
            indices correspond to instance indices in X

        Returns
        -------
        float, accuracy score of predict(X) vs y
        """
        from sklearn.metrics import accuracy_score

        self.check_is_fitted()

        return accuracy_score(y, self.predict(X), normalize=True)

    def _check_convert_X_for_predict(self, X):
        """Input checks, capability checks, repeated in all predict/score methods.

        Parameters
        ----------
        X : any object (to check/convert)
            should be of a supported Panel mtype or 2D numpy.ndarray

        Returns
        -------
        X: an object of a supported Panel mtype, numpy3D if X was a 2D numpy.ndarray

        Raises
        ------
        ValueError if X is of invalid input data type, or there is not enough data
        ValueError if the capabilities in self._tags do not handle the data.
        """
        X = _internal_convert(X)
        X_metadata = _check_classifier_input(X)
        missing = X_metadata["has_nans"]
        multivariate = not X_metadata["is_univariate"]
        unequal = not X_metadata["is_equal_length"]
        # Check this classifier can handle characteristics
        self._check_capabilities(missing, multivariate, unequal)
        # Convert data as dictated by the classifier tags
        X = self._convert_X(X)

        return X

    def _fit(self, X, y):
        """Fit time series classifier to training data.

        Abstract method, must be implemented.

        Parameters
        ----------
        X : guaranteed to be of a type in self.get_tag("X_inner_mtype")
            if self.get_tag("X_inner_mtype") = "numpy3D":
                3D np.ndarray of shape = [n_instances, n_dimensions, series_length]
            if self.get_tag("X_inner_mtype") = "nested_univ":
                pd.DataFrame with each column a dimension, each cell a pd.Series
            for list of other mtypes, see datatypes.SCITYPE_REGISTER
            for specifications, see examples/AA_datatypes_and_datasets.ipynb
        y : 1D np.array of int, of shape [n_instances] - class labels for fitting
            indices correspond to instance indices in X
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
        X : guaranteed to be of a type in self.get_tag("X_inner_mtype")
            if self.get_tag("X_inner_mtype") = "numpy3D":
                3D np.ndarray of shape = [n_instances, n_dimensions, series_length]
            if self.get_tag("X_inner_mtype") = "nested_univ":
                pd.DataFrame with each column a dimension, each cell a pd.Series
            for list of other mtypes, see datatypes.SCITYPE_REGISTER
            for specifications, see examples/AA_datatypes_and_datasets.ipynb

        Returns
        -------
        y : 1D np.array of int, of shape [n_instances] - predicted class labels
            indices correspond to instance indices in X
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
        X : guaranteed to be of a type in self.get_tag("X_inner_mtype")
            if self.get_tag("X_inner_mtype") = "numpy3D":
                3D np.ndarray of shape = [n_instances, n_dimensions, series_length]
            if self.get_tag("X_inner_mtype") = "nested_univ":
                pd.DataFrame with each column a dimension, each cell a pd.Series
            for list of other mtypes, see datatypes.SCITYPE_REGISTER
            for specifications, see examples/AA_datatypes_and_datasets.ipynb

        Returns
        -------
        y : 2D array of shape [n_instances, n_classes] - predicted class probabilities
            1st dimension indices correspond to instance indices in X
            2nd dimension indices correspond to possible labels (integers)
            (i, j)-th entry is predictive probability that i-th instance is of class j
        """
        dists = np.zeros((X.shape[0], self.n_classes_))
        preds = self._predict(X)
        for i in range(0, X.shape[0]):
            dists[i, self._class_dictionary[preds[i]]] = 1

        return dists

    def _check_capabilities(self, missing, multivariate, unequal):
        """Check whether this classifier can handle the data characteristics.

        Parameters
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
        X : input X converted to type in "X_inner_mtype" tag
                usually a pd.DataFrame (nested) or 3D np.ndarray
            Checked and possibly converted input data
        """
        inner_type = self.get_tag("X_inner_mtype")
        # convert pd.DataFrame
        X = convert_to(
            X,
            to_type=inner_type,
            as_scitype="Panel",
        )
        return X


def _check_classifier_input(
    X,
    y=None,
    enforce_min_instances=1,
):
    """Check whether input X and y are valid formats with minimum data.

    Raises a ValueError if the input is not valid.

    Parameters
    ----------
    X : check whether conformant with any sktime Panel mtype specification
    y : check whether a pd.Series or np.array
    enforce_min_instances : int, optional (default=1)
        check there are a minimum number of instances.

    Returns
    -------
    metadata : dict with metadata for X returned by datatypes.check_is_scitype

    Raises
    ------
    ValueError
        If y or X is invalid input data type, or there is not enough data
    """
    # Check X is valid input type and recover the data characteristics
    X_valid, _, X_metadata = check_is_scitype(X, scitype="Panel", return_metadata=True)
    if not X_valid:
        raise TypeError(
            f"X is not of a supported input data type."
            f"X must be in a supported mtype format for Panel, found {type(X)}"
            f"Use datatypes.check_is_mtype to check conformance with specifications."
        )
    n_cases = X_metadata["n_instances"]
    if n_cases < enforce_min_instances:
        raise ValueError(
            f"Minimum number of cases required is {enforce_min_instances} but X "
            f"has : {n_cases}"
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
    return X_metadata


def _internal_convert(X, y=None):
    """Convert X and y if necessary as a user convenience.

    Convert X to a 3D numpy array if already a 2D and convert y into an 1D numpy
    array if passed as a Series.

    Parameters
    ----------
    X : an object of a supported Panel mtype, or 2D numpy.ndarray
    y : np.ndarray or pd.Series

    Returns
    -------
    X: an object of a supported Panel mtype, numpy3D if X was a 2D numpy.ndarray
    y: np.ndarray
    """
    if isinstance(X, np.ndarray):
        # Temporary fix to insist on 3D numpy. For univariate problems,
        # most classifiers simply convert back to 2D. This squeezing should be
        # done here, but touches a lot of files, so will get this to work first.
        if X.ndim == 2:
            X = X.reshape(X.shape[0], 1, X.shape[1])
    if y is not None and isinstance(y, pd.Series):
        # y should be a numpy array, although we allow Series for user convenience
        y = pd.Series.to_numpy(y)
    if y is None:
        return X
    return X, y
