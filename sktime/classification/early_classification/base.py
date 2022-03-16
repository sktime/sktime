# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""
Abstract base class for time series classifiers.

    class name: BaseEarlyClassifier

Defining methods:
    fitting         - fit(self, X, y)
    predicting      - predict(self, X, state_info=None)
                    - predict_proba(self, X, state_info=None)

Inherited inspection methods:
    hyper-parameter inspection  - get_params()
    fitted parameter inspection - get_fitted_params()

State:
    fitted model/strategy   - by convention, any attributes ending in "_"
    fitted state flag       - is_fitted (property)
    fitted state inspection - check_is_fitted()
"""

__all__ = [
    "BaseEarlyClassifier",
]
__author__ = ["mloning", "fkiraly", "TonyBagnall", "MatthewMiddlehurst"]

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from sktime.base import BaseEstimator
from sktime.classification import BaseClassifier


class BaseEarlyClassifier(BaseEstimator, ABC):
    """Abstract base class for early time series classifiers.

    The base classifier specifies the methods and method signatures that all
    early classifiers have to implement. Attributes with a underscore suffix are set in
    the method fit.

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
        "capability:early_prediction": True,
        "capability:multithreading": False,
    }

    def __init__(self):
        self.classes_ = []
        self.n_classes_ = 0
        self.fit_time_ = 0
        self._class_dictionary = {}
        self._threads_to_use = 1

        super(BaseEarlyClassifier, self).__init__()

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
        fit = BaseClassifier.fit
        return fit(self, X, y)

    def predict(self, X, state_info=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        state_info : List or None
            A List containing the state info for each decision in X. contains
            information for future decisions on the data. Inputs should be None or an
            empty List for the first decision made, the returned List new_state_info for
            subsequent decisions.
            If no state_info is provided and the input series_length is greater than
            the first classification_points time stamp, all previous time stamps are
            considered up to the input series_length and the class value for the first
            safe prediction is returned.

        Returns
        -------
        y : 1D np.array of int, of shape [n_instances] - predicted class labels
            indices correspond to instance indices in X
        decisions : 1D bool array
            An array of booleans, containing the decision of whether a prediction is
            safe to use or not. Returned if return_safety_decisions is True.
        new_state_info : 2D int array
            An array containing the state info for each decision in X, contains
            information for future decisions on the data. Returned if
            return_safety_decisions is True.
            Each row contains information for a case from the latest decision on its
            safety.
        """
        self.check_is_fitted()

        # boilerplate input checks for predict-like methods
        X = self._check_convert_X_for_predict(X)

        return self._predict(X, state_info=state_info)

    def predict_proba(
        self, X, state_info=None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Decide on the safety of an early classification.

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
        state_info : List or None
            A List containing the state info for each decision in X. contains
            information for future decisions on the data. Inputs should be None or an
            empty List for the first decision made, the returned List new_state_info for
            subsequent decisions.
            If no state_info is provided and the input series_length is greater than
            the first classification_points time stamp, all previous time stamps are
            considered up to the input series_length and the probabilities for the first
            safe prediction are returned.

        Returns
        -------
        y : 2D array of shape [n_instances, n_classes] - predicted class probabilities
            1st dimension indices correspond to instance indices in X
            2nd dimension indices correspond to possible labels (integers)
            (i, j)-th entry is predictive probability that i-th instance is of class j
        decisions : 1D bool array
            An array of booleans, containing the decision of whether a prediction is
            safe to use or not. Returned if return_safety_decisions is True.
        new_state_info : 2D int array
            An array containing the state info for each decision in X, contains
            information for future decisions on the data. Returned if
            return_safety_decisions is True.
            Each row contains information for a case from the latest decision on its
            safety.
        """
        self.check_is_fitted()

        # boilerplate input checks for predict-like methods
        X = self._check_convert_X_for_predict(X)

        return self._predict_proba(X, state_info=state_info)

    def score(self, X, y) -> Tuple[float, float, float]:
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
        Tuple of floats, harmonic mean, accuracy and earliness scores of predict(X) vs y
        """
        self.check_is_fitted()

        # boilerplate input checks for predict-like methods
        X = self._check_convert_X_for_predict(X)

        return self._score(X, y)

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
        self :
            Reference to self.

        Notes
        -----
        Changes state by creating a fitted model that updates attributes
        ending in "_" and sets is_fitted flag to True.
        """
        ...

    @abstractmethod
    def _predict(self, X, state_info=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        state_info : List or None
            A List containing the state info for each decision in X. contains
            information for future decisions on the data. Inputs should be None or an
            empty List for the first decision made, the returned List new_state_info for
            subsequent decisions.
            If no state_info is provided and the input series_length is greater than
            the first classification_points time stamp, all previous time stamps are
            considered up to the input series_length and the class value for the first
            safe prediction is returned.

        Returns
        -------
        y : 1D np.array of int, of shape [n_instances] - predicted class labels
            indices correspond to instance indices in X
        decisions : 1D bool array
            An array of booleans, containing the decision of whether a prediction is
            safe to use or not. Returned if return_safety_decisions is True.
        new_state_info : 2D int array
            An array containing the state info for each decision in X, contains
            information for future decisions on the data. Returned if
            return_safety_decisions is True.
            Each row contains information for a case from the latest decision on its
            safety.
        """
        ...

    def _predict_proba(
        self, X, state_info=None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        state_info : List or None
            A List containing the state info for each decision in X. contains
            information for future decisions on the data. Inputs should be None or an
            empty List for the first decision made, the returned List new_state_info for
            subsequent decisions.
            If no state_info is provided and the input series_length is greater than
            the first classification_points time stamp, all previous time stamps are
            considered up to the input series_length and the probabilities for the first
            safe prediction are returned.

        Returns
        -------
        y : 2D array of shape [n_instances, n_classes] - predicted class probabilities
            1st dimension indices correspond to instance indices in X
            2nd dimension indices correspond to possible labels (integers)
            (i, j)-th entry is predictive probability that i-th instance is of class j
        decisions : 1D bool array
            An array of booleans, containing the decision of whether a prediction is
            safe to use or not. Returned if return_safety_decisions is True.
        new_state_info : 2D int array
            An array containing the state info for each decision in X, contains
            information for future decisions on the data. Returned if
            return_safety_decisions is True.
            Each row contains information for a case from the latest decision on its
            safety.
        """
        dists = np.zeros((X.shape[0], self.n_classes_))
        preds, decisions, state_info = self._predict(X, state_info=state_info)
        for i in range(0, X.shape[0]):
            dists[i, self._class_dictionary[preds[i]]] = 1

        return dists, decisions, state_info

    @abstractmethod
    def _score(self, X, y) -> Tuple[float, float, float]:
        """Scores predicted labels against ground truth labels on X.

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
        Tuple of floats, harmonic mean, accuracy and earliness scores of predict(X) vs y
        """
        ...

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
        _check_convert_X_for_predict = BaseClassifier._check_convert_X_for_predict
        return _check_convert_X_for_predict(self, X)

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
        _check_capabilities = BaseClassifier._check_capabilities
        return _check_capabilities(self, missing, multivariate, unequal)

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
        _convert_X = BaseClassifier._convert_X
        return _convert_X(self, X)
