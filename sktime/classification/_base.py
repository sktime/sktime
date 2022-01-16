# -*- coding: utf-8 -*-
"""
Abstract base class for time series classifiers.

    class name: BaseClassifier. Note the term Panel used here and throughout
    sktime simply refers to a collection of time series, for example a training set,
    where each series is assumed to be independent of the others.

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
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from sktime.base import BaseEstimator
from sktime.datatypes import check_is_scitype, convert_to
from sktime.utils.validation import check_n_jobs


class BaseClassifier(BaseEstimator, ABC):
    """Abstract base class for time series classifiers.

    The base classifier specifies the methods and method signatures that all
    classifiers have to implement. Attributes with a underscore suffix are set in the
    method fit.

    Parameters
    ----------
    classes_            : ndarray of class labels, possibly strings
    n_classes_          : integer, number of classes (length of classes_)
    fit_time_           : integer, time (in milliseconds) for fit to run.
    _class_dictionary   : dictionary mapping classes_ onto integers 0...n_classes_-1.
    _threads_to_use     : number of threads to use in fit as determined by n_jobs.
    _series_length      : length of series if they are all the same length, otherwise 0.
    _X_metadata         : dictionary, the data characteristics of X passed to fit.
                            The keys for this hash table are given in function check_is
                            in datatypes._check.py
    """

    _tags = {
        "X_inner_mtype": "numpy3D",  # which type do _fit/_predict accept, usually
        # this is either "numpy3D" or "nested_univ" (nested pd.DataFrame). Other
        # types are allowable, see datatypes/panel/_registry.py for options.
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
        self._series_length = 0
        self._X_metadata = {}
        super(BaseClassifier, self).__init__()

    def fit(self, X, y) -> BaseEstimator:
        """Fit time series classifier to training data.

        Parameters
        ----------
        X : np.ndarray (2d or 3d array of shape (n_instances, series_length) or shape
            (n_instances, n_dimensions, series_length)) or pd.DataFrame (where each
            column is a dimension, each cell is a pd.Series (any number of dimensions,
            equal or unequal length series))
            or of any other supported Panel data structures, for list of
            supported, see MTYPE_REGISTER_PANEL in datatypes/panel/_registry.py
        y : 1D np.array, of shape [n_instances] - class labels for fitting
            indices correspond to instance indices in X

        Returns
        -------
        self :  Fitted estimator.

        Notes
        -----
        Converts X to type self.get_tag("X_inner_mtype"), creates a fitted model by
        calling _fit that updates attributes ending in "_" and sets is_fitted to True.
        """
        start = int(round(time.time() * 1000))
        # convenience conversions to allow user flexibility:
        # if X is 2D array, convert to 3D, if y is Series, convert to 1D-numpy
        X, y = _internal_convert(X, y)
        #  Get data characteristics
        self._X_metadata = _check_classifier_input(X, y)
        missing = self._X_metadata["has_nans"]
        multivariate = not self._X_metadata["is_univariate"]
        unequal = not self._X_metadata["is_equal_length"]
        # Check this classifier can handle these three characteristics
        self._check_capabilities(missing, multivariate, unequal)
        # Convert data as dictated by the classifier tag self.get_tag("X_inner_mtype")
        X = self._convert_X(X)
        # Store series length for checking in predict
        if not unequal:
            self._series_length = self._find_series_length(X)
        else:
            self._series_length = 0

        if self.get_tag("capability:multithreading"):
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
        self._is_fitted = True
        return self

    def predict(self, X) -> np.ndarray:
        """Predicts labels for sequences in X.

        Parameters
        ----------
        X : np.ndarray (2d or 3d array of shape (n_instances, series_length) or shape
            (n_instances, n_dimensions, series_length)) or pd.DataFrame (where each
            column is a dimension, each cell is a pd.Series (any number of dimensions,
            equal or unequal length series))
            or of any other supported Panel data structures, for list of
            supported, see MTYPE_REGISTER_PANEL in datatypes/panel/_registry.py

        Returns
        -------
        y : 1D np.array, of shape [n_instances] - predicted class labels
            indices correspond to instance indices in X
        """
        self.check_is_fitted()

        # input checks for predict-like methods
        X = self._check_convert_X_for_predict(X)
        # If the input have to be the same length as the training data
        if not self.get_tag("capability:unequal_length") and not self.get_tag(
            "capability:early_prediction"
        ):
            # check input is the same length as train data
            length = self._find_series_length(X)
            if length != self._series_length:
                raise ValueError(
                    "Error in predict: input series different length to "
                    "training series, but tags dictate they must be the "
                    "same."
                )
        return self._predict(X)

    def predict_proba(self, X) -> np.ndarray:
        """Predicts labels probabilities for sequences in X.

        Parameters
        ----------
        X : np.ndarray (2d or 3d array of shape (n_instances, series_length) or shape
            (n_instances, n_dimensions, series_length)) or pd.DataFrame (where each
            column is a dimension, each cell is a pd.Series (any number of dimensions,
            equal or unequal length series))
            or of any other supported Panel data structures, for list of
            supported, see MTYPE_REGISTER_PANEL in datatypes/panel/_registry.py

        Returns
        -------
        y : 2D array of shape [n_instances, n_classes] - predicted class probabilities
            y[i, j] is estimated probability that i-th instance is of class j
        """
        self.check_is_fitted()

        # input checks for predict and predict_proba
        X = self._check_convert_X_for_predict(X)
        return self._predict_proba(X)

    def score(self, X, y) -> float:
        """Scores predicted labels against ground truth labels on X.

        Parameters
        ----------
        X : np.ndarray (2d or 3d array of shape (n_instances, series_length) or shape
            (n_instances, n_dimensions, series_length)) or pd.DataFrame (where each
            column is a dimension, each cell is a pd.Series (any number of dimensions,
            equal or unequal length series))
            or of any other supported Panel data structures, for list of
            supported, see MTYPE_REGISTER_PANEL in datatypes/panel/_registry.py
        y : 1D np.ndarray, of shape [n_instances] - class labels (ground truth)
            indices correspond to instance indices in X

        Returns
        -------
        float, accuracy score of predict(X) vs y
        """
        self.check_is_fitted()
        return accuracy_score(y, self.predict(X), normalize=True)

    def _check_convert_X_for_predict(self, X):
        """Input checks, capability checks, repeated in all predict/score methods.

        Parameters
        ----------
        X : any object (to check/convert)
            should be of a type compatible with fit

        Returns
        -------
        X: an object of a type

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
        # Check data the same length
        # Convert data as dictated by the classifier tags
        X = self._convert_X(X)

        return X

    @abstractmethod
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
        self : Reference to self.

        Notes
        -----
        Changes state by creating a fitted model that updates attributes
        ending in "_" and sets is_fitted flag to True.
        """
        ...

    @abstractmethod
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
        ...

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
            y[i, j] is estimated probability that i-th instance is of class j
        """
        dists = np.zeros((X.shape[0], self.n_classes_))
        preds = self._predict(X)
        for i in range(0, X.shape[0]):
            dists[i, self._class_dictionary[preds[i]]] = 1

        return dists

    def _check_capabilities(self, missing, multivariate, unequal) -> object:
        """Check whether this classifier can handle the data characteristics.

        Parameters
        ----------
        missing : boolean, does the data passed to fit contain missing values?
        multivariate : boolean, does the data passed to fit contain missing values?
        unequal : boolean, do the time series passed to fit have variable lengths?

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
        X : input X converted to type self.get_tag("X_inner_mtype") tag
                usually a pd.DataFrame (tag nested_univ) or 3D np.ndarray (numpy3D)
            Checked and possibly converted input data
        """
        inner_type = self.get_tag("X_inner_mtype")
        # If data is unequal length and the classifier can handle it,
        # but the inner type is numpy, we should *not* convert
        # this use case is NOT yet allowed for
        X = convert_to(
            X,
            to_type=inner_type,
            as_scitype="Panel",
        )
        return X

    def _find_series_length(self, X):
        """Find the series length for a fixed length input series.

        Parameters
        ----------
        X: input data of type self.get_tag("X_inner_mtype"), assumed to be equal length

        Returns
        -------
        int, length of the first series in X

        Raises
        ------
        ValueError if X is not of type np.ndarray or pd.DataFrame
        """
        if isinstance(X, pd.DataFrame):
            return X.iloc(0, 0).size
        elif isinstance(X, np.ndarray):
            if X.ndim == 2:
                return X.shape[1]
            else:
                return X.shape[2]
        else:
            raise ValueError(
                f"Unable to detect the length of series from X, "
                f"it is not an np.ndarray or pd.DataFrame, it is "
                f"{type(X)}."
            )


def _check_classifier_input(
    X,
    y=None,
    enforce_min_instances=1,
):
    """Check whether input X and y are valid formats with minimum data.

    Raises a ValueError if the input is not valid.

    Parameters
    ----------
    X : check whether X is any sktime Panel mtype specification
    y : check whether a pd.Series or np.array
    enforce_min_instances : int, optional (default=1)
        check there are a minimum number of instances.

    Returns
    -------
    metadata : dictionary with metadata for X returned by check_is_scitype in
    datatypes._check.py. The keys for this hash table are given in function check_is in
    datatypes._check.py

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
