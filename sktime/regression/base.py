# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Abstract base class for time series regressors.

    class name: BaseRegressor

Defining methods:
    fitting         - fit(self, X, y)
    predicting      - predict(self, X)

Inherited inspection methods:
    hyper-parameter inspection  - get_params()
    fitted parameter inspection - get_fitted_params()

State:
    fitted model/strategy   - by convention, any attributes ending in "_"
    fitted state flag       - is_fitted (property)
    fitted state inspection - check_is_fitted()
"""

__all__ = [
    "BaseRegressor",
]
__author__ = ["mloning", "fkiraly", "ksharma6"]

import time

import numpy as np

from sktime.base import BasePanelMixin
from sktime.datatypes import VectorizedDF
from sktime.utils.sklearn import is_sklearn_transformer
from sktime.utils.validation import check_n_jobs


class BaseRegressor(BasePanelMixin):
    """Abstract base class for time series regressors.

    The base regressor specifies the methods and method signatures that all
    regressors have to implement. Attributes with a underscore suffix are set in the
    method fit.

    Parameters
    ----------
    fit_time_           : integer, time (in milliseconds) for fit to run.
    _class_dictionary   : dictionary mapping classes_ onto integers 0...n_classes_-1.
    _threads_to_use     : number of threads to use in fit as determined by n_jobs.
    """

    _tags = {
        "object_type": "regressor",  # type of object
        "X_inner_mtype": "numpy3D",  # which type do _fit/_predict, support for X?
        "y_inner_mtype": "numpy1D",  # which type do _fit/_predict, support for y?
        #    it should be either "numpy3D" or "nested_univ" (nested pd.DataFrame)
        "capability:multioutput": False,  # whether regressor supports multioutput
        "capability:multivariate": False,
        "capability:unequal_length": False,
        "capability:missing_values": False,
        "capability:train_estimate": False,
        "capability:contractable": False,
        "capability:multithreading": False,
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
    VECTORIZATION_ATTR = "regressors_"  # e.g., classifiers_, regressors_

    # used in error messages
    TASK = "regression"  # e.g., classification, regression
    EST_TYPE = "regressor"  # e.g., classifier, regressor
    EST_TYPE_PLURAL = "regressors"  # e.g., classifiers, regressors

    def __init__(self):
        self.fit_time_ = 0
        self._class_dictionary = {}
        self._threads_to_use = 1
        self._X_metadata = {}

        # required for compatibility with some sklearn interfaces
        # i.e. CalibratedRegressorCV
        self._estimator_type = "regressor"
        self._is_vectorized = False
        self._is_timed = False
        self._converter_store_y = {}

        super().__init__()

    def __rmul__(self, other):
        """Magic * method, return concatenated RegressorPipeline, transformers on left.

        Overloaded multiplication operation for regressors. Implemented for ``other``
        being a transformer, otherwise returns ``NotImplemented``.

        Parameters
        ----------
        other: ``sktime`` transformer, must inherit from BaseTransformer
            otherwise, ``NotImplemented`` is returned

        Returns
        -------
        RegressorPipeline object, concatenation of ``other`` (first) with ``self``
        (last).
        """
        from sktime.regression.compose import RegressorPipeline
        from sktime.transformations.base import BaseTransformer
        from sktime.transformations.compose import TransformerPipeline
        from sktime.transformations.series.adapt import TabularToSeriesAdaptor

        # behaviour is implemented only if other inherits from BaseTransformer
        #  in that case, distinctions arise from whether self or other is a pipeline
        #  todo: this can probably be simplified further with "zero length" pipelines
        if isinstance(other, BaseTransformer):
            # RegressorPipeline already has the dunder method defined
            if isinstance(self, RegressorPipeline):
                return other * self
            # if other is a TransformerPipeline but self is not, first unwrap it
            elif isinstance(other, TransformerPipeline):
                return RegressorPipeline(regressor=self, transformers=other.steps)
            # if neither self nor other are a pipeline, construct a RegressorPipeline
            else:
                return RegressorPipeline(regressor=self, transformers=[other])
        elif is_sklearn_transformer(other):
            return TabularToSeriesAdaptor(other) * self
        else:
            return NotImplemented

    def __or__(self, other):
        """Magic | method, return MultiplexRegressor.

        Implemented for `other` being either a MultiplexRegressor or a regressor.

        Parameters
        ----------
        other: `sktime` regressor or sktime MultiplexRegressor

        Returns
        -------
        MultiplexRegressor object
        """
        from sktime.regression.compose import MultiplexRegressor

        if isinstance(other, MultiplexRegressor) or isinstance(other, BaseRegressor):
            multiplex_self = MultiplexRegressor([self])
            return multiplex_self | other
        else:
            return NotImplemented

    def fit(self, X, y):
        """Fit time series regressor to training data.

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

        Notes
        -----
        Changes state by creating a fitted model that updates attributes
        ending in "_" and sets is_fitted flag to True.
        """
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

        # y float coercion
        if y is not None and isinstance(y, np.ndarray):
            y = y.astype("float")

        # input checks
        X_metadata = self._check_input(
            X, y, return_metadata=self.METADATA_REQ_IN_CHECKS
        )
        self._X_metadata = X_metadata
        X_mtype = X_metadata["mtype"]

        # Check this regressor can handle characteristics
        self._check_capabilities(X_metadata)

        # Convert data as dictated by the regressor tags
        X = self._convert_X(X, X_mtype)
        multithread = self.get_tag("capability:multithreading")
        if multithread:
            try:
                self._threads_to_use = check_n_jobs(self.n_jobs)
            except NameError:
                raise AttributeError(
                    "self.n_jobs must be set if capability:multithreading is True"
                )

        self._fit(X, y)
        self.fit_time_ = int(round(time.time() * 1000)) - start

        # this should happen last: fitted state is set to True
        self._is_fitted = True
        return self

    def predict(self, X) -> np.ndarray:
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
            predicted regression labels

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

        # call internal _predict, convert output
        y_pred_inner = self._predict(X)
        y_pred = self._convert_output_y(y_pred_inner)
        return y_pred

    def score(self, X, y, multioutput="uniform_average") -> float:
        """Scores predicted labels against ground truth labels on X.

        Parameters
        ----------
        X : sktime compatible time series panel data container, Panel scitype, e.g.,
            pd-multiindex: pd.DataFrame with columns = variables,
            index = pd.MultiIndex with first level = instance indices,
            second level = time indices
            numpy3D: 3D np.array (any number of dimensions, equal length series)
            of shape [n_instances, n_dimensions, series_length]
            or of any other supported Panel mtype
            for list of mtypes, see datatypes.SCITYPE_REGISTER
            for specifications, see examples/AA_datatypes_and_datasets.ipynb
        y : 2D np.array of int, of shape [n_instances, n_dimensions] - regression labels
            for fitting indices correspond to instance indices in X
            or 1D np.array of int, of shape [n_instances] - regression labels for
            fitting indices correspond to instance indices in X
        multioutput : str, optional (default="uniform_average")
            {"raw_values", "uniform_average", "variance_weighted"}, array-like of shape
            (n_outputs,) or None, default="uniform_average".
            Defines aggregating of multiple output scores. Array-like value defines
            weights used to average scores.

        Returns
        -------
        float (default) or 1D np.array of float
            R-squared score of predict(X) vs y
            float if multioutput="uniform_average" or "variance_weighted,
            or y is univariate;
            1D np.array if multioutput="raw_values" and y is multivariate
        """
        from sklearn.metrics import r2_score

        self.check_is_fitted()

        return r2_score(y, self.predict(X), multioutput=multioutput)

    def _fit(self, X, y):
        """Fit time series regressor to training data.

        Abstract method, must be implemented.

        Parameters
        ----------
        X : guaranteed to be of a type in self.get_tag("X_inner_mtype")
            if self.get_tag("X_inner_mtype") = "numpy3D":
            3D np.ndarray of shape = [n_instances, n_dimensions, series_length]
            if self.get_tag("X_inner_mtype") = "pd-multiindex":
            pd.DataFrame with columns = variables,
            index = pd.MultiIndex with first level = instance indices,
            second level = time indices
            if self.get_tag("X_inner_mtype") = "nested_univ":
            pd.DataFrame with each column a dimension, each cell a pd.Series
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

        Notes
        -----
        Changes state by creating a fitted model that updates attributes ending in "_"
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
            if self.get_tag("X_inner_mtype") = "nested_univ":
                pd.DataFrame with each column a dimension, each cell a pd.Series
            for list of other mtypes, see datatypes.SCITYPE_REGISTER
            for specifications, see examples/AA_datatypes_and_datasets.ipynb

        Returns
        -------
        y : 1D np.array of float, of shape [n_instances] - predicted regression labels
            indices correspond to instance indices in X
        """
        raise NotImplementedError("abstract method")
