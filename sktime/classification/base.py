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
        "X_inner_mtype": "numpy3D",  # which types do _fit/_predict, support for X?
        "y_inner_mtype": "numpy1D",  # which types do _fit/_predict, support for y?
        # note: y_inner_mtype is always assumped to be numpy1D for now
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
        X : Panel, any of the supported formats (default=None)
                usually 3D numpy array, pd-multiindex, or nested pd.DataFrame
                if 3D numpy, of shape [n_instances, n_dimensions, series_length]
            additionally, can be 2D numpy array of shape [n_instances, series_length]
            panel of time series to train classifier on
        y : Table, univariate, any of the supported formats (default=None)
                usually, 1D np.array of shape = [n_instances]
            class labels for the series in X

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

        # input checks and minor coercions on X, y
        X_inner, y_inner = self._check_X_y(X=X, y=y)

        multithread = self.get_tag("capability:multithreading")
        if multithread:
            try:
                self._threads_to_use = check_n_jobs(self.n_jobs)
            except NameError:
                raise AttributeError(
                    "self.n_jobs must be set if capability:multithreading is True"
                )

        # for this we assume that y_inner_mtype is always numpy
        # this paragraph needs be changed if we want to support pd.DataFrame internally
        self.classes_ = np.unique(y)
        self.n_classes_ = self.classes_.shape[0]
        for index, classVal in enumerate(self.classes_):
            self._class_dictionary[classVal] = index

        self._fit(X_inner, y_inner)
        self.fit_time_ = int(round(time.time() * 1000)) - start

        # this should happen last
        self._is_fitted = True
        return self

    def predict(self, X) -> np.ndarray:
        """Predicts labels for sequences in X.

        Parameters
        ----------
        X : Panel, any of the supported formats (default=None)
                usually 3D numpy array, pd-multiindex, or nested pd.DataFrame
                if 3D numpy, of shape [n_instances, n_dimensions, series_length]
            additionally, can be 2D numpy array of shape [n_instances, series_length]
            panel of time series to classify

        Returns
        -------
        y : 1D np.array of shape =  [n_instances] - predicted class labels
        """
        self.check_is_fitted()

        # input checks and minor coercions on X, y
        X_inner = self._check_X(X=X)

        return self._predict(X_inner)

    def predict_proba(self, X) -> np.ndarray:
        """Predicts labels probabilities for sequences in X.

        Parameters
        ----------
        X : Panel, any of the supported formats (default=None)
                usually 3D numpy array, pd-multiindex, or nested pd.DataFrame
                if 3D numpy, of shape [n_instances, n_dimensions, series_length]
            additionally, can be 2D numpy array of shape [n_instances, series_length]
            panel of time series to classify

        Returns
        -------
        y : array-like, shape =  [n_instances, n_classes]
            estimated probabilities of class membership
            (i,j)-the element is predictive probability
                of i-th instance of X to be contained in class j
        """
        self.check_is_fitted()

        # input checks and minor coercions on X, y
        X_inner = self._check_X(X=X)

        return self._predict_proba(X_inner)

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
        X : Panel of mtype X_inner_mtype
            usually: 3D np.array, array-like or sparse matrix
                of shape = [n_instances,n_dimensions,series_length]
            or pd.DataFrame with each column a dimension, each cell a pd.Series
        y : Table of mtype y_iner_mtype
            usually: array-like, shape = [n_instances] - the class labels

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
        X : Panel of mtype X_inner_mtype
            usually: 3D np.array, array-like or sparse matrix
                of shape = [n_instances,n_dimensions,series_length]
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
        X : Panel of mtype X_inner_mtype
            usually: 3D np.array, array-like or sparse matrix
                of shape = [n_instances,n_dimensions,series_length]
            or pd.DataFrame with each column a dimension, each cell a pd.Series

        Returns
        -------
        y : array-like, shape =  [n_instances, n_classes]
            estimated probabilities of class membership
            (i,j)-the element is predictive probability
                of i-th instance of X to be contained in class j
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

    def _check_X_y(self, X=None, y=None):
        """Check and coerce X/y for fit/predict/update functions.

        Parameters
        ----------
        X : Panel, any of the supported formats (default=None)
                usually 3D numpy array, pd-multiindex, or nested pd.DataFrame
            panel of time series to classify
        y : Table, univariate, any of the supported formats (default=None)
            class labels for the series in X

        Returns
        -------
        X_inner : Series compatible with self.get_tag("X_inner_mtype") format
            converted/coerced version of y, mtype determined by "X_inner_mtype" tag
            None if X was None
        y_inner : Series compatible with self.get_tag("y_inner_mtype") format
            converted/coerced version of y, mtype determined by "y_inner_mtype" tag
            None if y was None

        Raises
        ------
        TypeError if y or X is not one of the permissible Series mtypes
        TypeError if y is not compatible with self.get_tag("scitype:y")
            if tag value is "univariate", y must be univariate
            if tag value is "multivariate", y must be bi- or higher-variate
            if tag vaule is "both", y can be either
        TypeError if self.get_tag("X-y-must-have-same-index") is True
            and the index set of X is not a super-set of the index set of y
        """
        # check inputs X/y to be compatible with classifier specification
        #################################################################

        # is X of Panel scitype?
        X_valid, _, X_metadata = check_is_scitype(
            X, scitype="Panel", return_metadata=True
        )
        if not X_valid:
            raise TypeError(
                "X is not of a supported Panel mtype. "
                "Use datatypes.check_is_mtype to check conformance with specification."
            )

        # is y of Table scitype and univariate?
        y_valid, _, y_metadata = check_is_scitype(
            y, scitype="Table", return_metadata=True
        )
        if not y_valid:
            raise TypeError(
                "X is not of a supported Panel mtype. "
                "Use datatypes.check_is_mtype to check conformance with specification."
            )
        if not y_metadata["is_univariate"]:
            raise ValueError("label container y must be univariate")

        # do X/y have the same number of instances, and at least 1 instance?
        n_instances = X_metadata["n_instances"]
        n_labels = y_metadata["n_instances"]

        if n_instances != n_labels:
            raise ValueError(
                f"Mismatch in number of cases. "
                f"Number instances in X = {n_instances}, but"
                f"number labels in y = {n_labels}"
            )

        if n_instances == 0:
            raise ValueError("Found zero instances in X, there should be at least 1.")

        # Can this classifier can handle data characteristics?
        self._check_capabilities(
            X_metadata["has_nans"],
            not X_metadata["is_univariate"],
            not X_metadata["is_equally_spaced"],
        )

        # convert X & y to supported inner type, if necessary
        #####################################################

        # convert X and y to a supported internal mtype
        #  it X/y mtype is already supported, no conversion takes place
        #  if X/y is None, then no conversion takes place (returns None)
        X_inner_mtype = self.get_tag("X_inner_mtype")
        X_inner = convert_to(
            X,
            to_type=X_inner_mtype,
            as_scitype="Series",  # we are dealing with series
        )
        y_inner_mtype = self.get_tag("y_inner_mtype")
        y_inner = convert_to(
            y,
            to_type=y_inner_mtype,
            as_scitype="Series",  # we are dealing with series
            store=self._converter_store_y,
        )

        return X_inner, y_inner

    def _check_X(self, X=None):
        """Shorthand for _check_X_y with one argument X, see _check_X_y."""
        return self._check_X_y(X=X)[0]
