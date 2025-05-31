"""Abstract base class for time series classifiers.

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
__author__ = ["mloning", "fkiraly", "TonyBagnall", "MatthewMiddlehurst", "ksharma6"]

import time

import numpy as np

from sktime.base import BasePanelMixin
from sktime.datatypes import VectorizedDF, check_is_scitype
from sktime.utils.dependencies import _check_estimator_deps
from sktime.utils.sklearn import is_sklearn_transformer
from sktime.utils.validation import check_n_jobs


class BaseClassifier(BasePanelMixin):
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
        "object_type": "classifier",  # type of object
        "X_inner_mtype": "numpy3D",  # which type do _fit/_predict, support for X?
        "y_inner_mtype": "numpy1D",  # which type do _fit/_predict, support for y?
        #    it should be either "numpy3D" or "nested_univ" (nested pd.DataFrame)
        "capability:multioutput": False,  # whether classifier supports multioutput
        "capability:multivariate": False,
        "capability:unequal_length": False,
        "capability:missing_values": False,
        "capability:train_estimate": False,
        "capability:feature_importance": False,
        "capability:contractable": False,
        "capability:multithreading": False,
        "capability:categorical_in_X": True,
        "capability:predict_proba": False,
        "python_version": None,  # PEP 440 python version specifier to limit versions
        "requires_cython": False,  # whether C compiler is required in env, e.g., gcc
        "authors": "sktime developers",  # author(s) of the object
        "maintainers": "sktime developers",  # current maintainer(s) of the object
    }

    # convenience constant to control which metadata of input data
    # are regularly retrieved in input checks
    METADATA_REQ_IN_CHECKS = [
        "n_instances",
        "has_nans",
        "is_univariate",
        "is_equal_length",
        "feature_kind",
    ]

    # attribute name where vectorized estimators are stored
    VECTORIZATION_ATTR = "classifiers_"  # e.g., classifiers_, regressors_

    # used in error messages
    TASK = "classification"  # e.g., classification, regression
    EST_TYPE = "classifier"  # e.g., classifier, regressor
    EST_TYPE_PLURAL = "classifiers"  # e.g., classifiers, regressors

    def __init__(self):
        # reserved attributes written to in fit
        self.classes_ = []  # classes seen in y, unique labels
        self.n_classes_ = 0  # number of unique classes in y
        self.fit_time_ = 0  # time elapsed in last fit call
        self._class_dictionary = {}
        self._threads_to_use = 1
        self._X_metadata = []  # metadata/properties of X seen in fit

        # required for compatibility with some sklearn interfaces
        # i.e. CalibratedClassifierCV
        self._estimator_type = "classifier"
        self._is_vectorized = False
        self._converter_store_y = {}

        super().__init__()
        _check_estimator_deps(self)

    def __rmul__(self, other):
        """Magic * method, return concatenated ClassifierPipeline, transformers on left.

        Overloaded multiplication operation for classifiers. Implemented for ``other``
        being a transformer, otherwise returns ``NotImplemented``.

        Parameters
        ----------
        other: ``sktime`` transformer, must inherit from BaseTransformer
            otherwise, ``NotImplemented`` is returned

        Returns
        -------
        ClassifierPipeline object, concatenation of ``other`` (first) with ``self``
        (last).
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

    def __or__(self, other):
        """Magic | method, return MultiplexClassifier.

        Implemented for `other` being either a MultiplexClassifier or a classifier.

        Parameters
        ----------
        other: `sktime` classifier or sktime MultiplexClassifier

        Returns
        -------
        MultiplexClassifier object
        """
        from sktime.classification.compose import MultiplexClassifier

        if isinstance(other, MultiplexClassifier) or isinstance(other, BaseClassifier):
            multiplex_self = MultiplexClassifier([self])
            return multiplex_self | other
        else:
            return NotImplemented

    def fit(self, X, y):
        """
        Fit time series classifier to training data.

        State change:
            Changes state to "fitted".

        Writes to self:
            Sets self.is_fitted to True.
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        X : sktime compatible time series panel data container of Panel scitype
            time series to fit the estimator to.

            Can be in any :term:`mtype` of ``Panel`` :term:`scitype`, for instance:

            * pd-multiindex: pd.DataFrame with columns = variables,
              index = pd.MultiIndex with first level = instance indices,
              second level = time indices
            * numpy3D: 3D np.array (any number of dimensions, equal length series)
              of shape [n_instances, n_dimensions, series_length]
            * or of any other supported ``Panel`` :term:`mtype`

            for list of mtypes, see ``datatypes.SCITYPE_REGISTER``

            for specifications, see ``examples/AA_datatypes_and_datasets.ipynb``

            Not all estimators support panels with multivariate or unequal length
            series, see the :ref:`tag reference <panel_tags>` for details.

        y : sktime compatible tabular data container, Table scitype
            1D iterable, of shape [n_instances]
            or 2D iterable, of shape [n_instances, n_dimensions]
            class labels for fitting
            0-th indices correspond to instance indices in X
            1-st indices (if applicable) correspond to multioutput vector indices in X
            supported sktime types: np.ndarray (1D, 2D), pd.Series, pd.DataFrame

        Returns
        -------
        self : Reference to self.
        """
        # reset estimator at the start of fit
        self.reset()

        # fit timer start
        start = int(round(time.time() * 1000))

        # check and convert y for multioutput vectorization
        y, y_metadata, y_inner_mtype = self._check_y(y, return_to_mtype=True)
        self._y_metadata = y_metadata
        self._y_inner_mtype = y_inner_mtype
        self._is_vectorized = isinstance(y, VectorizedDF)

        if self._is_vectorized:
            self._vectorize("fit", X=X, y=y)
            # fit timer end
            self.fit_time_ = int(round(time.time() * 1000)) - start
            # this should happen last: fitted state is set to True
            self._is_fitted = True
            return self

        # no vectorization needed, proceed with normal fit

        # convenience conversions to allow user flexibility:
        # if X is 2D array, convert to 3D, if y is Series, convert to numpy
        X, y = self._internal_convert(X, y)
        X_metadata = self._check_input(
            X, y, return_metadata=self.METADATA_REQ_IN_CHECKS
        )
        X_mtype = X_metadata["mtype"]
        self._X_metadata = X_metadata

        # Check this classifier can handle characteristics
        self._check_capabilities(X_metadata)

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
        X = self._convert_X(X, X_mtype)
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

        # this should happen last: fitted state is set to True
        self._is_fitted = True
        return self

    def predict(self, X):
        """Predicts labels for sequences in X.

        Parameters
        ----------
        X : sktime compatible time series panel data container of Panel scitype
            time series to predict labels for.

            Can be in any :term:`mtype` of ``Panel`` :term:`scitype`, for instance:

            * pd-multiindex: pd.DataFrame with columns = variables,
              index = pd.MultiIndex with first level = instance indices,
              second level = time indices
            * numpy3D: 3D np.array (any number of dimensions, equal length series)
              of shape [n_instances, n_dimensions, series_length]
            * or of any other supported ``Panel`` :term:`mtype`

            for list of mtypes, see ``datatypes.SCITYPE_REGISTER``

            for specifications, see ``examples/AA_datatypes_and_datasets.ipynb``

            Not all estimators support panels with multivariate or unequal length
            series, see the :ref:`tag reference <panel_tags>` for details.

        Returns
        -------
        y_pred : sktime compatible tabular data container, of Table :term:`scitype`
            predicted class labels

            1D iterable, of shape [n_instances],
            or 2D iterable, of shape [n_instances, n_dimensions].

            0-th indices correspond to instance indices in X,
            1-st indices (if applicable) correspond to multioutput vector indices in X.

            1D np.npdarray, if y univariate (one dimension);
            otherwise, same type as y passed in fit
        """
        self.check_is_fitted()

        # enter vectorized mode if needed
        if self._is_vectorized:
            return self._vectorize("predict", X=X)

        # boilerplate input checks for predict-like methods
        X = self._check_convert_X_for_predict(X)

        # handle the single-class-label case
        if len(self._class_dictionary) == 1:
            return self._single_class_y_pred(X, method="predict")

        # call internal _predict, convert output
        y_pred_inner = self._predict(X)
        y_pred = self._convert_output_y(y_pred_inner)
        return y_pred

    def predict_proba(self, X):
        """Predicts labels probabilities for sequences in X.

        Parameters
        ----------
        X : sktime compatible time series panel data container of Panel scitype
            time series to predict labels for.

            Can be in any :term:`mtype` of ``Panel`` :term:`scitype`, for instance:

            * pd-multiindex: pd.DataFrame with columns = variables,
              index = pd.MultiIndex with first level = instance indices,
              second level = time indices
            * numpy3D: 3D np.array (any number of dimensions, equal length series)
              of shape [n_instances, n_dimensions, series_length]
            * or of any other supported ``Panel`` :term:`mtype`

            for list of mtypes, see ``datatypes.SCITYPE_REGISTER``

            for specifications, see ``examples/AA_datatypes_and_datasets.ipynb``

            Not all estimators support panels with multivariate or unequal length
            series, see the :ref:`tag reference <panel_tags>` for details.

        Returns
        -------
        y_pred : 2D np.array of int, of shape [n_instances, n_classes]
            predicted class label probabilities
            0-th indices correspond to instance indices in X
            1-st indices correspond to class index, in same order as in self.classes_
            entries are predictive class probabilities, summing to 1
        """
        self.check_is_fitted()

        # enter vectorized mode if needed
        if self._is_vectorized:
            return self._vectorize("predict_proba", X=X)

        self.check_is_fitted()

        # boilerplate input checks for predict-like methods
        X = self._check_convert_X_for_predict(X)

        # handle the single-class-label case
        if len(self._class_dictionary) == 1:
            return self._single_class_y_pred(X, method="predict_proba")

        # call internal _predict_proba
        return self._predict_proba(X)

    def fit_predict(self, X, y, cv=None, change_state=True):
        """Fit and predict labels for sequences in X.

        Convenience method to produce in-sample predictions and
        cross-validated out-of-sample predictions.

        Writes to self, if change_state=True:
            Sets self.is_fitted to True.
            Sets fitted model attributes ending in "_".

        Does not update state if change_state=False.

        Parameters
        ----------
        X : sktime compatible time series panel data container of Panel scitype
            time series to fit to and predict labels for.

            Can be in any :term:`mtype` of ``Panel`` :term:`scitype`, for instance:

            * pd-multiindex: pd.DataFrame with columns = variables,
              index = pd.MultiIndex with first level = instance indices,
              second level = time indices
            * numpy3D: 3D np.array (any number of dimensions, equal length series)
              of shape [n_instances, n_dimensions, series_length]
            * or of any other supported ``Panel`` :term:`mtype`

            for list of mtypes, see ``datatypes.SCITYPE_REGISTER``

            for specifications, see ``examples/AA_datatypes_and_datasets.ipynb``

            Not all estimators support panels with multivariate or unequal length
            series, see the :ref:`tag reference <panel_tags>` for details.

        y : sktime compatible tabular data container, Table scitype
            1D iterable, of shape [n_instances]
            or 2D iterable, of shape [n_instances, n_dimensions]
            class labels for fitting
            0-th indices correspond to instance indices in X
            1-st indices (if applicable) correspond to multioutput vector indices in X
            supported sktime types: np.ndarray (1D, 2D), pd.Series, pd.DataFrame

        cv : None, int, or sklearn cross-validation object, optional, default=None

            * None : predictions are in-sample, equivalent to ``fit(X, y).predict(X)``
            * cv : predictions are equivalent to
              ``fit(X_train, y_train).predict(X_test)``, where multiple
              ``X_train``, ``y_train``, ``X_test`` are obtained from ``cv`` folds.
              returned ``y`` is union over all test fold predictions,
              ``cv`` test folds must be non-intersecting
            * int : equivalent to ``cv=KFold(cv, shuffle=True, random_state=x)``,
              i.e., k-fold cross-validation predictions out-of-sample, and where
              ``random_state`` ``x`` is taken from ``self`` if exists,
              otherwise ``x=None``

        change_state : bool, optional (default=True)

            * if False, will not change the state of the classifier,
              i.e., fit/predict sequence is run with a copy, self does not change
            * if True, will fit self to the full X and y,
              end state will be equivalent to running fit(X, y)

        Returns
        -------
        y_pred : sktime compatible tabular data container, of Table :term:`scitype`
            predicted class labels

            1D iterable, of shape [n_instances],
            or 2D iterable, of shape [n_instances, n_dimensions].

            0-th indices correspond to instance indices in X,
            1-st indices (if applicable) correspond to multioutput vector indices in X.

            1D np.npdarray, if y univariate (one dimension);
            otherwise, same type as y passed in fit
        """
        return self._fit_predict_boilerplate(
            X=X, y=y, cv=cv, change_state=change_state, method="predict"
        )

    def fit_predict_proba(self, X, y, cv=None, change_state=True):
        """Fit and predict labels probabilities for sequences in X.

        Convenience method to produce in-sample predictions and
        cross-validated out-of-sample predictions.

        Writes to self, if change_state=True:
            Sets self.is_fitted to True.
            Sets fitted model attributes ending in "_".

        Does not update state if change_state=False.

        Parameters
        ----------
        X : sktime compatible time series panel data container of Panel scitype
            time series to fit to and predict labels for.

            Can be in any :term:`mtype` of ``Panel`` :term:`scitype`, for instance:

            * pd-multiindex: pd.DataFrame with columns = variables,
              index = pd.MultiIndex with first level = instance indices,
              second level = time indices
            * numpy3D: 3D np.array (any number of dimensions, equal length series)
              of shape [n_instances, n_dimensions, series_length]
            * or of any other supported ``Panel`` :term:`mtype`

            for list of mtypes, see ``datatypes.SCITYPE_REGISTER``

            for specifications, see ``examples/AA_datatypes_and_datasets.ipynb``

            Not all estimators support panels with multivariate or unequal length
            series, see the :ref:`tag reference <panel_tags>` for details.

        y : sktime compatible tabular data container, Table scitype
            1D iterable, of shape [n_instances]
            or 2D iterable, of shape [n_instances, n_dimensions]
            class labels for fitting
            0-th indices correspond to instance indices in X
            1-st indices (if applicable) correspond to multioutput vector indices in X
            supported sktime types: np.ndarray (1D, 2D), pd.Series, pd.DataFrame

        cv : None, int, or sklearn cross-validation object, optional, default=None

            * None : predictions are in-sample, equivalent to ``fit(X, y).predict(X)``
            * cv : predictions are equivalent to
              ``fit(X_train, y_train).predict(X_test)``, where multiple
              ``X_train``, ``y_train``, ``X_test`` are obtained from ``cv`` folds.
              returned ``y`` is union over all test fold predictions,
              ``cv`` test folds must be non-intersecting
            * int : equivalent to ``cv=KFold(cv, shuffle=True, random_state=x)``,
              i.e., k-fold cross-validation predictions out-of-sample, and where
              ``random_state`` ``x`` is taken from ``self`` if exists,
              otherwise ``x=None``

        change_state : bool, optional (default=True)

            * if False, will not change the state of the classifier,
              i.e., fit/predict sequence is run with a copy, self does not change
            * if True, will fit self to the full X and y,
              end state will be equivalent to running fit(X, y)

        Returns
        -------
        y_pred : 2D np.array of int, of shape [n_instances, n_classes]
            predicted class label probabilities
            0-th indices correspond to instance indices in X
            1-st indices correspond to class index, in same order as in self.classes_
            entries are predictive class probabilities, summing to 1
        """
        return self._fit_predict_boilerplate(
            X=X, y=y, cv=cv, change_state=change_state, method="predict_proba"
        )

    def _single_class_y_pred(self, X, method="predict"):
        """Handle the prediction case where only single class label was seen in fit."""
        X_meta_required = ["n_instances"]
        _, _, X_meta = check_is_scitype(
            X, scitype="Panel", return_metadata=X_meta_required
        )
        n_instances = X_meta["n_instances"]
        if method == "predict":
            return np.repeat(list(self._class_dictionary.keys()), n_instances)
        else:  # method == "predict_proba"
            return np.repeat([[1]], n_instances, axis=0)

    def score(self, X, y) -> float:
        """Scores predicted labels against ground truth labels on X.

        Parameters
        ----------
        X : sktime compatible time series panel data container of Panel scitype
            time series to score predicted labels for.

            Can be in any :term:`mtype` of ``Panel`` :term:`scitype`, for instance:

            * pd-multiindex: pd.DataFrame with columns = variables,
              index = pd.MultiIndex with first level = instance indices,
              second level = time indices
            * numpy3D: 3D np.array (any number of dimensions, equal length series)
              of shape [n_instances, n_dimensions, series_length]
            * or of any other supported ``Panel`` :term:`mtype`

            for list of mtypes, see ``datatypes.SCITYPE_REGISTER``

            for specifications, see ``examples/AA_datatypes_and_datasets.ipynb``

            Not all estimators support panels with multivariate or unequal length
            series, see the :ref:`tag reference <panel_tags>` for details.

        y : sktime compatible tabular data container, Table scitype
            1D iterable, of shape [n_instances]
            or 2D iterable, of shape [n_instances, n_dimensions]
            class labels for fitting
            0-th indices correspond to instance indices in X
            1-st indices (if applicable) correspond to multioutput vector indices in X
            supported sktime types: np.ndarray (1D, 2D), pd.Series, pd.DataFrame

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
            special parameters are defined for a value, will return ``"default"`` set.
            For classifiers, a "default" set of parameters should be provided for
            general testing, and a "results_comparison" set for comparing against
            previously recorded results if the general set does not produce suitable
            probabilities to compare against.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``.
        """
        return super().get_test_params(parameter_set=parameter_set)

    def _fit(self, X, y):
        """Fit time series classifier to training data.

        Abstract method, must be implemented.

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        X : guaranteed to be of a type in self.get_tag("X_inner_mtype")
            if self.get_tag("X_inner_mtype") = "numpy3D":
            3D np.ndarray of shape = [n_instances, n_dimensions, series_length]
            if self.get_tag("X_inner_mtype") = "pd-multiindex:":
            pd.DataFrame with columns = variables,
            index = pd.MultiIndex with first level = instance indices,
            second level = time indices
            for list of other mtypes, see datatypes.SCITYPE_REGISTER
            for specifications, see examples/AA_datatypes_and_datasets.ipynb
        y : guaranteed to be of a type in self.get_tag("y_inner_mtype")
            1D iterable, of shape [n_instances]
            or 2D iterable, of shape [n_instances, n_dimensions]
            class labels for fitting
            if self.get_tag("capaility:multioutput") = False, guaranteed to be 1D
            if self.get_tag("capaility:multioutput") = True, guaranteed to be 2D

        Returns
        -------
        self : Reference to self.
        """
        raise NotImplementedError("abstract method")

    def _predict(self, X) -> np.ndarray:
        """Predicts labels for sequences in X.

        Abstract method, must be implemented.

        Parameters
        ----------
        X : guaranteed to be of a type in self.get_tag("X_inner_mtype")
            if self.get_tag("X_inner_mtype") = "numpy3D":
            3D np.ndarray of shape = [n_instances, n_dimensions, series_length]
            if self.get_tag("X_inner_mtype") = "pd-multiindex:":
            pd.DataFrame with columns = variables,
            index = pd.MultiIndex with first level = instance indices,
            second level = time indices
            for list of other mtypes, see datatypes.SCITYPE_REGISTER
            for specifications, see examples/AA_datatypes_and_datasets.ipynb

        Returns
        -------
        y : should be of mtype in self.get_tag("y_inner_mtype")
            1D iterable, of shape [n_instances]
            or 2D iterable, of shape [n_instances, n_dimensions]
            predicted class labels
            indices correspond to instance indices in X
            if self.get_tag("capaility:multioutput") = False, should be 1D
            if self.get_tag("capaility:multioutput") = True, should be 2D
        """
        y_proba = self._predict_proba(X)
        y_pred = y_proba.argmax(axis=1)

        # restore class labels if they were not originally integer
        y_pred = self.classes_[y_pred]

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
            if self.get_tag("X_inner_mtype") = "pd-multiindex:":
            pd.DataFrame with columns = variables,
            index = pd.MultiIndex with first level = instance indices,
            second level = time indices
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
