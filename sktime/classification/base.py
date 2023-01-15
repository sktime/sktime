# -*- coding: utf-8 -*-
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
from abc import ABC, abstractmethod
from warnings import warn

import numpy as np
import pandas as pd

from sktime.base import BaseEstimator
from sktime.datatypes import check_is_scitype, convert_to
from sktime.utils.sklearn import is_sklearn_transformer
from sktime.utils.validation import check_n_jobs
from sktime.utils.validation._dependencies import _check_estimator_deps


class BaseClassifier(BaseEstimator, ABC):
    """Abstract base class for time series classifiers.

    The base classifier specifies the methods and method signatures that all
    classifiers have to implement. Attributes with an underscore suffix are set in the
    method fit.

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
        "capability:multithreading": False,
        "python_version": None,  # PEP 440 python version specifier to limit versions
    }

    def __init__(self):
        # reserved attributes written to in fit
        self.classes_ = []  # classes seen in y, unique labels
        self.n_classes_ = 0  # number of unique classes in y
        self.fit_time_ = 0  # time elapsed in last fit call
        self._class_dictionary = {}
        self._threads_to_use = 1
        self._X_metadata = []  # metadata/properties of X seen in fit

        # required for compatability with some sklearn interfaces
        # i.e. CalibratedClassifierCV
        self._estimator_type = "classifier"

        super(BaseClassifier, self).__init__()
        _check_estimator_deps(self)

    def __rmul__(self, other):
        """Magic * method, return concatenated ClassifierPipeline, transformers on left.

        Overloaded multiplication operation for classifiers. Implemented for `other`
        being a transformer, otherwise returns `NotImplemented`.

        Parameters
        ----------
        other: `sktime` transformer, must inherit from BaseTransformer
            otherwise, `NotImplemented` is returned

        Returns
        -------
        ClassifierPipeline object, concatenation of `other` (first) with `self` (last).
        """
        from sktime.classification.compose import ClassifierPipeline
        from sktime.transformations.base import BaseTransformer
        from sktime.transformations.compose import TransformerPipeline
        from sktime.transformations.series.adapt import TabularToSeriesAdaptor

        # behaviour is implemented only if other inherits from BaseTransformer
        #  in that case, distinctions arise from whether self or other is a pipeline
        #  todo: this can probably be simplified further with "zero length" pipelines
        if isinstance(other, BaseTransformer):
            # ClassifierPipeline already has the dunder method defined
            if isinstance(self, ClassifierPipeline):
                return other * self
            # if other is a TransformerPipeline but self is not, first unwrap it
            elif isinstance(other, TransformerPipeline):
                return ClassifierPipeline(classifier=self, transformers=other.steps)
            # if neither self nor other are a pipeline, construct a ClassifierPipeline
            else:
                return ClassifierPipeline(classifier=self, transformers=[other])
        elif is_sklearn_transformer(other):
            return TabularToSeriesAdaptor(other) * self
        else:
            return NotImplemented

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
        # reset estimator at the start of fit
        self.reset()

        start = int(round(time.time() * 1000))
        # convenience conversions to allow user flexibility:
        # if X is 2D array, convert to 3D, if y is Series, convert to numpy
        X, y = self._internal_convert(X, y)
        X_metadata = self._check_classifier_input(X, y)
        missing = X_metadata["has_nans"]
        multivariate = not X_metadata["is_univariate"]
        unequal = not X_metadata["is_equal_length"]
        self._X_metadata = X_metadata

        # Check this classifier can handle characteristics
        self._check_capabilities(missing, multivariate, unequal)

        # remember class labels
        self.classes_ = np.unique(y)
        self.n_classes_ = self.classes_.shape[0]
        self._class_dictionary = {}
        for index, class_val in enumerate(self.classes_):
            self._class_dictionary[class_val] = index

        # escape early and do not fit if only one class label has been seen
        #   in this case, we later predict the single class label seen
        if len(self.classes_) == 1:
            self.fit_time_ = int(round(time.time() * 1000)) - start
            self._is_fitted = True
            return self

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

        # pass coerced and checked data to inner _fit
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

        # handle the single-class-label case
        if len(self._class_dictionary) == 1:
            return self._single_class_y_pred(X, method="predict")

        # call internal _predict_proba
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

        # handle the single-class-label case
        if len(self._class_dictionary) == 1:
            return self._single_class_y_pred(X, method="predict_proba")

        # call internal _predict_proba
        return self._predict_proba(X)

    def fit_predict(self, X, y, cv=None, change_state=True) -> np.ndarray:
        """Fit and predict labels for sequences in X.

        Convenience method to produce in-sample predictions and
        cross-validated out-of-sample predictions.

        Writes to self, if change_state=True:
            Sets self.is_fitted to True.
            Sets fitted model attributes ending in "_".

        Does not update state if change_state=False.

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
        cv : None, int, or sklearn cross-validation object, optional, default=None
            None : predictions are in-sample, equivalent to fit(X, y).predict(X)
            cv : predictions are equivalent to fit(X_train, y_train).predict(X_test)
                where multiple X_train, y_train, X_test are obtained from cv folds
                returned y is union over all test fold predictions
                cv test folds must be non-intersecting
            int : equivalent to cv=KFold(cv, shuffle=True, random_state=x),
                i.e., k-fold cross-validation predictions out-of-sample
                random_state x is taken from self if exists, otherwise x=None
        change_state : bool, optional (default=True)
            if False, will not change the state of the classifier,
                i.e., fit/predict sequence is run with a copy, self does not change
            if True, will fit self to the full X and y,
                end state will be equivalent to running fit(X, y)

        Returns
        -------
        y : 1D np.array of int, of shape [n_instances] - predicted class labels
            indices correspond to instance indices in X
            if cv is passed, -1 indicates entries not seen in union of test sets
        """
        return self._fit_predict_boilerplate(
            X=X, y=y, cv=cv, change_state=change_state, method="predict"
        )

    def _fit_predict_boilerplate(self, X, y, cv, change_state, method):
        """Boilerplate logic for fit_predict and fit_predict_proba."""
        from sklearn.model_selection import KFold

        if isinstance(cv, int):
            random_state = getattr(self, "random_state", None)
            cv = KFold(cv, random_state=random_state, shuffle=True)

        if change_state:
            self.reset()
            est = self
        else:
            est = self.clone()

        if cv is None:
            return getattr(est.fit(X, y), method)(X)
        elif change_state:
            self.fit(X, y)

        # we now know that cv is an sklearn splitter
        X, y = self._internal_convert(X, y)
        X_metadata = self._check_classifier_input(X, y)
        missing = X_metadata["has_nans"]
        multivariate = not X_metadata["is_univariate"]
        unequal = not X_metadata["is_equal_length"]
        # Check this classifier can handle characteristics
        self._check_capabilities(missing, multivariate, unequal)

        # handle single class case
        if len(self._class_dictionary) == 1:
            return self._single_class_y_pred(X)

        # Convert data to format easily useable for applying cv
        if isinstance(X, np.ndarray):
            X = convert_to(
                X,
                to_type="numpy3D",
                as_scitype="Panel",
                store_behaviour="freeze",
            )
        else:
            X = convert_to(
                X,
                to_type="nested_univ",
                as_scitype="Panel",
                store_behaviour="freeze",
            )

        if method == "predict_proba":
            y_pred = np.empty([len(y), len(np.unique(y))])
        else:
            y_pred = np.empty_like(y)
        y_pred[:] = -1
        if isinstance(X, np.ndarray):
            for tr_idx, tt_idx in cv.split(X):
                X_train = X[tr_idx]
                X_test = X[tt_idx]
                y_train = y[tr_idx]
                fitted_est = self.clone().fit(X_train, y_train)
                y_pred[tt_idx] = getattr(fitted_est, method)(X_test)
        else:
            for tr_idx, tt_idx in cv.split(X):
                X_train = X.iloc[tr_idx]
                X_test = X.iloc[tt_idx]
                y_train = y[tr_idx]
                fitted_est = self.clone().fit(X_train, y_train)
                y_pred[tt_idx] = getattr(fitted_est, method)(X_test)

        return y_pred

    def fit_predict_proba(self, X, y, cv=None, change_state=True) -> np.ndarray:
        """Fit and predict labels probabilities for sequences in X.

        Convenience method to produce in-sample predictions and
        cross-validated out-of-sample predictions.

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
        cv : None, int, or sklearn cross-validation object, optional, default=None
            None : predictions are in-sample, equivalent to fit(X, y).predict(X)
            cv : predictions are equivalent to fit(X_train, y_train).predict(X_test)
                where multiple X_train, y_train, X_test are obtained from cv folds
                returned y is union over all test fold predictions
                cv test folds must be non-intersecting
            int : equivalent to cv=Kfold(int), i.e., k-fold cross-validation predictions
        change_state : bool, optional (default=True)
            if False, will not change the state of the classifier,
                i.e., fit/predict sequence is run with a copy, self does not change
            if True, will fit self to the full X and y,
                end state will be equivalent to running fit(X, y)

        Returns
        -------
        y : 2D array of shape [n_instances, n_classes] - predicted class probabilities
            1st dimension indices correspond to instance indices in X
            2nd dimension indices correspond to possible labels (integers)
            (i, j)-th entry is predictive probability that i-th instance is of class j
        """
        return self._fit_predict_boilerplate(
            X=X, y=y, cv=cv, change_state=change_state, method="predict_proba"
        )

    def _single_class_y_pred(self, X, method="predict"):
        """Handle the prediction case where only single class label was seen in fit."""
        _, _, X_meta = check_is_scitype(X, scitype="Panel", return_metadata=True)
        n_instances = X_meta["n_instances"]
        if method == "predict":
            return np.repeat(list(self._class_dictionary.keys()), n_instances)
        else:  # method == "predict_proba"
            return np.repeat([[1]], n_instances, axis=0)

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

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            For classifiers, a "default" set of parameters should be provided for
            general testing, and a "results_comparison" set for comparing against
            previously recorded results if the general set does not produce suitable
            probabilities to compare against.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`.
        """
        return super().get_test_params(parameter_set=parameter_set)

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
        y_proba = self._predict_proba(X)
        y_pred = y_proba.argmax(axis=1)

        return y_pred

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
        preds = self._predict(X)
        n_pred = len(preds)
        dists = np.zeros((n_pred, self.n_classes_))
        for i in range(n_pred):
            dists[i, self._class_dictionary[preds[i]]] = 1

        return dists

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
        X = self._internal_convert(X)
        X_metadata = self._check_classifier_input(X)
        missing = X_metadata["has_nans"]
        multivariate = not X_metadata["is_univariate"]
        unequal = not X_metadata["is_equal_length"]
        # Check this classifier can handle characteristics
        self._check_capabilities(missing, multivariate, unequal)
        # Convert data as dictated by the classifier tags
        X = self._convert_X(X)

        return X

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

        self_name = type(self).__name__

        # identify problems, mismatch of capability and inputs
        problems = []
        if missing and not allow_missing:
            problems += ["missing values"]
        if multivariate and not allow_multivariate:
            problems += ["multivariate series"]
        if unequal and not allow_unequal:
            problems += ["unequal length series"]

        # construct error message
        problems_and = " and ".join(problems)
        problems_or = " or ".join(problems)
        msg = (
            f"Data seen by {self_name} instance has {problems_and}, "
            f"but this {self_name} instance cannot handle {problems_or}. "
            f"Calls with {problems_or} may result in error or unreliable results."
        )

        # raise exception or warning with message
        # if self is composite, raise a warning, since passing could be fine
        #   see discussion in PR 2366 why
        if len(problems) > 0:
            if self.is_composite():
                warn(msg)
            else:
                raise ValueError(msg)

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

    def _check_classifier_input(self, X, y=None, enforce_min_instances=1):
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
        X_valid, _, X_metadata = check_is_scitype(
            X, scitype="Panel", return_metadata=True
        )
        if not X_valid:
            raise TypeError(
                f"X is not of a supported input data type."
                f"X must be in a supported mtype format for Panel, found {type(X)}"
                f"Use datatypes.check_is_mtype to check conformance "
                "with specifications."
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
                        f"np.ndarray y must be 1-dimensional, "
                        f"but found {y.ndim} dimensions"
                    )
            # warn if only a single class label is seen
            # this should not raise exception since this can occur by train subsampling
            if len(np.unique(y)) == 1:
                warn(
                    "only single class label seen in y passed to "
                    f"fit of classifier {type(self).__name__}"
                )

        return X_metadata

    def _internal_convert(self, X, y=None):
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
