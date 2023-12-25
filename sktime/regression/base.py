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
__author__ = ["mloning", "fkiraly"]

import time
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from sktime.base import BaseEstimator
from sktime.datatypes import (
    MTYPE_LIST_PANEL,
    MTYPE_LIST_TABLE,
    VectorizedDF,
    check_is_error_msg,
    check_is_scitype,
    convert,
)
from sktime.utils.sklearn import is_sklearn_transformer
from sktime.utils.validation import check_n_jobs
from sktime.utils.warnings import warn


class BaseRegressor(BaseEstimator, ABC):
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
    }

    # convenience constant to control which metadata of input data
    # are regularly retrieved in input checks
    METADATA_REQ_IN_CHECKS = [
        "n_instances",
        "has_nans",
        "is_univariate",
        "is_equal_length",
    ]

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

        Overloaded multiplication operation for regressors. Implemented for `other`
        being a transformer, otherwise returns `NotImplemented`.

        Parameters
        ----------
        other: `sktime` transformer, must inherit from BaseTransformer
            otherwise, `NotImplemented` is returned

        Returns
        -------
        RegressorPipeline object, concatenation of `other` (first) with `self` (last).
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

    def fit(self, X, y):
        """Fit time series regressor to training data.

        State change:
            Changes state to "fitted".

        Writes to self:
            Sets self.is_fitted to True.
            Sets fitted model attributes ending in "_".

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
        y, y_metadata = self._check_y(y)
        self._y_metadata = y_metadata
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
        X, y = _internal_convert(X, y)
        X_metadata = _check_regressor_input(
            X, y, return_metadata=self.METADATA_REQ_IN_CHECKS
        )
        self._X_metadata = X_metadata
        missing = X_metadata["has_nans"]
        multivariate = not X_metadata["is_univariate"]
        unequal = not X_metadata["is_equal_length"]
        X_mtype = X_metadata["mtype"]
        self._X_metadata = X_metadata

        # Check this regressor can handle characteristics
        self._check_capabilities(missing, multivariate, unequal)

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
        X : sktime compatible time series panel data container, Panel scitype, e.g.,
            pd-multiindex: pd.DataFrame with columns = variables,
            index = pd.MultiIndex with first level = instance indices,
            second level = time indices
            numpy3D: 3D np.array (any number of dimensions, equal length series)
            of shape [n_instances, n_dimensions, series_length]
            or of any other supported Panel mtype
            for list of mtypes, see datatypes.SCITYPE_REGISTER
            for specifications, see examples/AA_datatypes_and_datasets.ipynb

        Returns
        -------
        y_pred : sktime compatible tabular data container, Table scitype
            1D iterable, of shape [n_instances]
            or 2D iterable, of shape [n_instances, n_dimensions]
            predicted class labels
            0-th indices correspond to instance indices in X
            1-st indices (if applicable) correspond to multioutput vector indices in X
            1D np.npdarray, if y univariate (one dimension)
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
        float, R-squared score of predict(X) vs y
        """
        from sklearn.metrics import r2_score

        self.check_is_fitted()

        return r2_score(y, self.predict(X), normalize=True, multioutput=multioutput)

    def _vectorize(self, methodname, **kwargs):
        """Vectorized/iterated loop over method of BaseClassifier.

        Uses regressors_ attribute to store one regressor per loop index.
        """
        y = kwargs.get("y")
        X = kwargs.get("X")
        if X is not None:
            kwargs.pop("X")
        if y is not None:
            kwargs.pop("y")
            self._y_vec = y
        regressors_ = self._y_vec.vectorize_est(
            self,
            method="clone",
        )
        if methodname == "fit":
            self.regressors_ = self._y_vec.vectorize_est(
                regressors_,
                method=methodname,
                args={"y": y},
                X=X,
            )
            return self
        else:
            regressors_ = self.regressors_
            y_pred = self._y_vec.vectorize_est(
                regressors_,
                method=methodname,
                # return_type="list",
                X=X,
                args={"y": y} if y is not None else {},
                **kwargs,  # contains X inside
            )
            y_pred = pd.DataFrame(
                {str(i): y_pred[col].values[0] for i, col in enumerate(y_pred.columns)}
            )
            return y_pred
        # add code for score

    @abstractmethod
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
        y : 1D np.array of float, of shape [n_instances] - predicted regression labels
            indices correspond to instance indices in X
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
        X = _internal_convert(X)
        X_metadata = _check_regressor_input(
            X, return_metadata=self.METADATA_REQ_IN_CHECKS
        )
        missing = X_metadata["has_nans"]
        multivariate = not X_metadata["is_univariate"]
        unequal = not X_metadata["is_equal_length"]
        X_mtype = X_metadata["mtype"]
        # Check this classifier can handle characteristics
        self._check_capabilities(missing, multivariate, unequal)
        # Convert data as dictated by the classifier tags
        X = self._convert_X(X, X_mtype=X_mtype)

        return X

    def _check_y(self, y=None):
        """Check and coerce X/y for fit/transform functions.

        Parameters
        ----------
        y : pd.DataFrame, pd.Series or np.ndarray

        Returns
        -------
        y : object of sktime compatible time series type
            can be Series, Panel, Hierarchical
        y_metadata : dict
            metadata of y, retured by check_is_scitype
        """
        if y is None:
            return None

        capa_multioutput = self.get_tag("capability:multioutput")
        y_inner_mtype = self.get_tag("y_inner_mtype")

        y_valid, y_msg, y_metadata = check_is_scitype(
            y, "Table", return_metadata=["is_univariate"]
        )

        if not y_valid:
            allowed_msg = (
                f"In regression, y must be of a supported type, "
                f"for instance 1D or 2D numpy arrays, pd.DataFrame, or pd.Series. "
                f"Allowed compatible mtype format specifications are:"
                f" {MTYPE_LIST_TABLE} ."
            )
            check_is_error_msg(
                y_msg, var_name="y", allowed_msg=allowed_msg, raise_exception=True
            )

        y_uni = y_metadata["is_univariate"]
        y_mtype = y_metadata["mtype"]

        requires_vectorization = not capa_multioutput and not y_uni

        if requires_vectorization:
            y_df = convert(
                y,
                from_type=y_mtype,
                to_type="pd_DataFrame_Table",
                as_scitype="Table",
                store=self._converter_store_y,
            )
            y_vec = VectorizedDF([y_df], iterate_cols=True)
            return y_vec, y_metadata

        y_inner = convert(
            y,
            from_type=y_mtype,
            to_type=y_inner_mtype,
            as_scitype="Table",
            store=self._converter_store_y,
        )

        return y_inner, y_metadata

    def _convert_output_y(self, y):
        """Convert output y to original format.

        Parameters
        ----------
        y : np.ndarray or pd.DataFrame

        Returns
        -------
        y : np.ndarray or pd.DataFrame
        """
        # for consistency with legacy behaviour:
        # output is coerced to numpy1D in case of univariate output
        if not self._y_metadata["is_univariate"]:
            output_mtype = self._y_metadata["mtype"]
            converter_store = self._converter_store_y
        else:
            output_mtype = "numpy1D"
            converter_store = None

        y = convert(
            y,
            from_type=self.get_tag("y_inner_mtype"),
            to_type=output_mtype,
            as_scitype="Table",
            store=converter_store,
            store_behaviour="freeze",
        )
        return y

    def _check_capabilities(self, missing, multivariate, unequal):
        """Check whether this regressor can handle the data characteristics.

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
                warn(msg, obj=self)
            else:
                raise ValueError(msg)

    def _convert_X(self, X, X_mtype):
        """Convert equal length series from DataFrame to numpy array or vice versa.

        Parameters
        ----------
        X : input data for the classifier
        X_mtype : str, a Panel mtype string, e.g., "pd_multiindex", "numpy3D"

        Returns
        -------
        X : input X converted to type in "X_inner_mtype" tag
            usually a pd.DataFrame (nested) or 3D np.ndarray
            Checked and possibly converted input data
        """
        inner_type = self.get_tag("X_inner_mtype")
        # convert pd.DataFrame
        X = convert(
            X,
            from_type=X_mtype,
            to_type=inner_type,
            as_scitype="Panel",
        )
        return X


def _check_regressor_input(
    X,
    y=None,
    enforce_min_instances=1,
    return_metadata=True,
):
    """Check whether input X and y are valid formats with minimum data.

    Raises a ValueError if the input is not valid.

    Parameters
    ----------
    X : check whether conformant with any sktime Panel mtype specification
    y : check whether a pd.Series or np.array
    enforce_min_instances : int, optional (default=1)
        check there are a minimum number of instances.
    return_metadata : bool, str, or list of str
        metadata fields to return with X_metadata, input to check_is_scitype

    Returns
    -------
    metadata : dict with metadata for X returned by datatypes.check_is_scitype

    Raises
    ------
    ValueError
        If y or X is invalid input data type, or there is not enough data
    """
    # Check X is valid input type and recover the data characteristics
    X_valid, msg, X_metadata = check_is_scitype(
        X, scitype="Panel", return_metadata=return_metadata
    )
    # raise informative error message if X is in wrong format
    allowed_msg = (
        f"Allowed scitypes for regressors are Panel mtypes, "
        f"for instance a pandas.DataFrame with MultiIndex and last(-1) "
        f"level an sktime compatible time index. "
        f"Allowed compatible mtype format specifications are: {MTYPE_LIST_PANEL} ."
    )
    if not X_valid:
        check_is_error_msg(
            msg, var_name="X", allowed_msg=allowed_msg, raise_exception=True
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
        if not isinstance(y, (pd.Series, pd.DataFrame, np.ndarray)):
            raise ValueError(
                "y must be a np.array or a pd.Series or pd.DataFrame, but found ",
                f"type: {type(y)}",
            )
        # Check matching number of labels
        n_labels = y.shape[0]
        if n_cases != n_labels:
            raise ValueError(
                f"Mismatch in number of cases. Number in X = {n_cases} nos in y = "
                f"{n_labels}"
            )
        if isinstance(y, np.ndarray):
            if y.ndim > 2:
                raise ValueError(
                    f"np.ndarray y must be 1-dimensional or 2-dimensional, "
                    f"but found {y.ndim} dimensions"
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
        # most regressors simply convert back to 2D. This squeezing should be
        # done here, but touches a lot of files, so will get this to work first.
        if X.ndim == 2:
            X = X.reshape(X.shape[0], 1, X.shape[1])
    if y is not None and isinstance(y, pd.Series):
        # y should be a numpy array, although we allow Series for user convenience
        y = pd.Series.to_numpy(y)
    if y is not None and isinstance(y, np.ndarray):
        y = y.astype("float")
    if y is None:
        return X
    return X, y
