# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Base class template for parameter estimator scitype.

    class name: BaseParamFitter

Scitype defining methods:
    fitting                - fit(X, y(optional))
    updating               - update(X, y=None)
    get fitted parameters  - get_fitted_params() -> dict

Inspection methods:
    hyper-parameter inspection  - get_params()

State:
    fitted model/strategy   - by convention, any attributes ending in "_"
    fitted state flag       - is_fitted (property)
    fitted state inspection - check_is_fitted()
"""

__author__ = ["fkiraly", "satvshr"]

__all__ = ["BaseParamFitter"]

import numpy as np

from sktime.base import BaseEstimator
from sktime.datatypes import (
    check_is_scitype,
    convert,
    scitype_to_mtype,
    update_data,
)
from sktime.datatypes._dtypekind import DtypeKind
from sktime.utils.adapters._safe_call import _safe_call
from sktime.utils.dependencies import _check_estimator_deps
from sktime.utils.sklearn import is_sklearn_transformer
from sktime.utils.warnings import warn


def _coerce_to_list(obj):
    """Return [obj] if obj is not a list, otherwise obj."""
    if not isinstance(obj, list):
        return [obj]
    else:
        return obj


class BaseParamFitter(BaseEstimator):
    """Base parameter fitting estimator class.

    The base parameter fitter specifies the methods and method signatures that all
    parameter fitter have to implement.

    Specific implementations of these methods is deferred to concrete instances.

    Parameters
    ----------
    None

    Attributes
    ----------
    _is_fitted : bool
        Flag indicating whether the estimator has been fitted.
    _X : sktime compatible container
        Stored feature data from fit and update.
    _y : array-like
        Stored target data from fit and update.
    """

    # default tag values - these typically make the "safest" assumption
    _tags = {
        "object_type": "param_est",  # type of object
        "X_inner_mtype": "pd.DataFrame",  # which types do _fit/_predict support for X?
        "y_inner_mtype": [
            "pd.DataFrame",
            "pd.Series",
        ],  # which types do _fit/_predict support for y?
        "scitype:X": "Series",  # which X scitypes are supported natively?
        "scitype:y": "Series",  # which y scitypes are supported natively?
        "capability:missing_values": False,  # can estimator handle missing data?
        "capability:multivariate": False,  # can estimator handle multivariate data?
        "requires_y": False,  # does estimator require y?
        "capability:pairwise": False,  # can handle pairwise parameter estimation?
        "python_version": None,  # PEP 440 python version specifier to limit versions
        "python_dependencies": None,  # string or str list of pkg soft dependencies
        "authors": "sktime developers",  # author(s) of the object
        "maintainers": "sktime developers",  # current maintainer(s) of the object
    }

    def __init__(self):
        self._is_fitted = False

        self._X = None
        self._y = None

        super().__init__()
        _check_estimator_deps(self)

    def __mul__(self, other):
        """Magic * method, for estimators on the right.

        Overloaded multiplication operation for parameter fitters.
        Implemented for ``other`` being:

        * a forecaster, results in ``PluginParamsForecaster``
        * a transformer, results in ``PluginParamsTransformer``
        * otherwise returns `NotImplemented`.

        Parameters
        ----------
        other: `sktime` estimator, must be one of the types specified above
            otherwise, `NotImplemented` is returned

        Returns
        -------
        one of the plugin estimator objects,
        concatenation of `self` (first) with `other` (last).
        """
        from sktime.forecasting.base import BaseForecaster
        from sktime.param_est.plugin import (
            PluginParamsForecaster,
            PluginParamsTransformer,
        )
        from sktime.transformations.base import BaseTransformer

        if isinstance(other, BaseForecaster):
            return PluginParamsForecaster(param_est=self, forecaster=other)
        elif isinstance(other, BaseTransformer):
            return PluginParamsTransformer(param_est=self, transformer=other)
        else:
            return NotImplemented

    def __rmul__(self, other):
        """Magic * method, return concatenated ParamFitterPipeline, trafos on left.

        Overloaded multiplication operation for parameter fitters.
        Implemented for ``other`` being a transformer,
        otherwise returns `NotImplemented`.

        Parameters
        ----------
        other: `sktime` transformer, must inherit from BaseTransformer
            otherwise, `NotImplemented` is returned

        Returns
        -------
        BaseParamFitter object, concatenation of `other` (first) with `self` (last).
        """
        from sktime.param_est.compose import ParamFitterPipeline
        from sktime.transformations.base import BaseTransformer
        from sktime.transformations.compose import TransformerPipeline
        from sktime.transformations.series.adapt import TabularToSeriesAdaptor

        # behaviour is implemented only if other inherits from BaseTransformer
        #  in that case, distinctions arise from whether self or other is a pipeline
        #  todo: this can probably be simplified further with "zero length" pipelines
        if isinstance(other, BaseTransformer):
            # ClassifierPipeline already has the dunder method defined
            if isinstance(self, ParamFitterPipeline):
                return other * self
            # if other is a TransformerPipeline but self is not, first unwrap it
            elif isinstance(other, TransformerPipeline):
                return ParamFitterPipeline(param_est=self, transformers=other.steps)
            # if neither self nor other are a pipeline, construct a ClassifierPipeline
            else:
                return ParamFitterPipeline(param_est=self, transformers=[other])
        elif is_sklearn_transformer(other):
            return TabularToSeriesAdaptor(other) * self
        else:
            return NotImplemented

    def _validate_data(self, ALLOWED_SCITYPES, FORBIDDEN_MTYPES, data, var_name="data"):
        """Validate input data (X or y)."""
        # Prepare a message about allowed mtypes
        for scitype in ALLOWED_SCITYPES:
            mtypes = set(scitype_to_mtype(scitype))
            mtypes = list(mtypes.difference(FORBIDDEN_MTYPES))
            mtypes_msg = f'"For {scitype} scitype: {mtypes}. '

        data_valid, _, data_metadata = check_is_scitype(
            data,
            scitype=ALLOWED_SCITYPES,
            return_metadata=["feature_kind"],
            var_name=var_name,
        )

        msg = (
            f"{var_name} must be in an sktime compatible format, "
            f"of scitypes {ALLOWED_SCITYPES}, for example a pandas.DataFrame with "
            "an sktime compatible time index. See the data format tutorial for "
            "more details. "
        )

        if not data_valid:
            raise TypeError(msg + mtypes_msg)

        if DtypeKind.CATEGORICAL in data_metadata["feature_kind"]:
            raise TypeError(
                "Parameter estimators do not support categorical features "
                f" in {var_name}. "
            )

        data_mtype = data_metadata["mtype"]
        data_inner_mtype = _coerce_to_list(self.get_tag(f"{var_name}_inner_mtype"))
        data_scitype = data_metadata["scitype"]

        return convert(
            data,
            from_type=data_mtype,
            to_type=data_inner_mtype,
            as_scitype=data_scitype,
        )

    def fit(self, X, y=None):
        """Fit estimator and estimate parameters.

        State change:
            Changes state to "fitted".

        Writes to self:
            Sets self._is_fitted flag to True.
            Writes `X` to self._X.
            Writes `y` to self._y if y is not None.
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        X : time series in sktime compatible data container format
                Time series to which to fit the forecaster in the update.
            y can be in one of the following formats, must be same scitype as in fit:
            Series scitype: pd.Series, pd.DataFrame, or np.ndarray (1D or 2D)
            Panel scitype: pd.DataFrame with 2-level row MultiIndex,
                3D np.ndarray, list of Series pd.DataFrame, or nested pd.DataFrame
            Hierarchical scitype: pd.DataFrame with 3 or more level row MultiIndex
            For further details:
                on usage, see forecasting tutorial examples/01_forecasting.ipynb
                on specification of formats, examples/AA_datatypes_and_datasets.ipynb

        Returns
        -------
        self : Reference to self.
        """
        # check X is not None
        assert X is not None, "X cannot be None, but found None"

        # if fit is called, estimator is reset, including fitted state
        self.reset()

        # Check and convert X and y according to estimator requirements
        X_inner, y_inner = self._check_X_y(X=X, y=y)

        # set internal X to the new X
        self._update_y_X(y_inner, X_inner)

        # checks and conversions complete, pass to inner fit
        #####################################################
        _safe_call(self._fit, args=(), kwargs={"X": X_inner, "y": y_inner})

        # this should happen last
        self._is_fitted = True

        return self

    def update(self, X, y=None):
        """Update fitted parameters on more data.

        If no estimator-specific update method has been implemented,
        default fall-back is fitting to all observed data so far
        State required:
            Requires state to be "fitted".
        Accesses in self:
            Fitted model attributes ending in "_".
            Pointers to seen data, self._X
            self._is_fitted
            model attributes ending in "_".
        Writes to self:
            Update self._X with `X`, by appending rows.
            Updates fitted model attributes ending in "_".

        Parameters
        ----------
        X : time series in sktime compatible data container format
                Time series to which to fit the forecaster in the update.
            y can be in one of the following formats, must be same scitype as in fit:
            Series scitype: pd.Series, pd.DataFrame, or np.ndarray (1D or 2D)
            Panel scitype: pd.DataFrame with 2-level row MultiIndex,
                3D np.ndarray, list of Series pd.DataFrame, or nested pd.DataFrame
            Hierarchical scitype: pd.DataFrame with 3 or more level row MultiIndex
            For further details:
                on usage, see forecasting tutorial examples/01_forecasting.ipynb
                on specification of formats, examples/AA_datatypes_and_datasets.ipynb

        Returns
        -------
        self : reference to self
        """
        self.check_is_fitted()

        if X is None or (hasattr(X, "__len__") and len(X) == 0):
            warn(
                f"empty y passed to update of {self}, no update was carried out",
                obj=self,
            )
            return self

        if y is not None:
            # Validate and convert the new X and y data
            X_inner, y_inner = self._check_X_y(X=X, y=y)
            self._update_y_X(y_inner, X_inner)
        else:
            # Update only the target data
            X_inner = self._check_X(X=X)
            self._update_y_X(None, X_inner)

        # Pass the checked and converted data to the estimator-specific _update logic
        _safe_call(self._fit, args=(), kwargs={"X": X_inner, "y": y_inner})

        return self

    def _check_X_y(self, X=None, y=None):
        """Check and coerce X and y for fit/update functions.

        Parameters
        ----------
        X : time series in sktime compatible data container format
                Time series to which to fit the forecaster in the update.
            X can be in one of the following formats, must be same scitype as in fit:
            Series scitype: pd.Series, pd.DataFrame, or np.ndarray (1D or 2D)
            Panel scitype: pd.DataFrame with 2-level row MultiIndex,
                3D np.ndarray, list of Series pd.DataFrame, or nested pd.DataFrame
            Hierarchical scitype: pd.DataFrame with 3 or more level row MultiIndex
            For further details:
                on usage, see forecasting tutorial examples/01_forecasting.ipynb
                on specification of formats, examples/AA_datatypes_and_datasets.ipynb

        y : time series in sktime compatible data container format, optional
            (default=None) Target values for parameter estimation to be validated and
            converted. y can be in one of the following formats,
            must be same scitype as in fit:
            Series scitype: pd.Series, pd.DataFrame, or np.ndarray (1D or 2D)
            Panel scitype: pd.DataFrame with 2-level row MultiIndex,
                3D np.ndarray, list of Series pd.DataFrame, or nested pd.DataFrame
            Hierarchical scitype: pd.DataFrame with 3 or more level row MultiIndex
            For further details:
                on usage, see forecasting tutorial examples/01_forecasting.ipynb
                on specification of formats, examples/AA_datatypes_and_datasets.ipynb

        Returns
        -------
        X_inner : Series, Panel, or Hierarchical object
                compatible with self.get_tag("X_inner_mtype") format
            Case 1: self.get_tag("X_inner_mtype") supports scitype of X, then
                converted/coerced version of X, mtype determined by "X_inner_mtype" tag
            Case 2: None if X was None

        y_inner : Series, Panel, or Hierarchical object or None
                compatible with self.get_tag("y_inner_mtype") format
            Case 1: self.get_tag("y_inner_mtype") supports scitype of y, then
                converted/coerced version of y, mtype determined by "y_inner_mtype" tag
            Case 2: None if y was None or if the estimator is pairwise

        Raises
        ------
        TypeError
            If X or y is not one of the permissible Series mtypes
            If X or y is of a different scitype as self.get_tag("scitype:X")
            or self.get_tag("scitype:y")
        ValueError
            If y is provided for an estimator that does not support target values.
            If the lengths of X and y do not match.
            TODO: Throw a ValueError for pairwise estimators, if X is not square.
        """
        X_inner, y_inner = None, None
        if X is None and y is None:
            return None, None

        if y is not None and self.get_tag("requires_y") is False:
            raise ValueError(
                f"{type(self).__name__} does not require y, but y was passed."
            )

        def _check_missing(obj):
            """Raise an error if obj contains missing values (if not supported)."""
            if isinstance(obj, np.ndarray):
                has_nan = np.isnan(obj).any()
            else:
                has_nan = obj.isnull().values.any()

            if has_nan and not self.get_tag("capability:missing_values"):
                msg = (
                    f"{type(self).__name__} cannot handle missing data (nans), "
                    f"but the passed data contained missing values."
                )
                raise ValueError(msg)

        # Process y if provided
        if y is not None:
            ALLOWED_SCITYPES = [_coerce_to_list(self.get_tag("scitype:y"))]
            FORBIDDEN_MTYPES = []
            y_inner = self._validate_data(ALLOWED_SCITYPES, FORBIDDEN_MTYPES, y, "y")

        if X is not None:
            ALLOWED_SCITYPES = _coerce_to_list(self.get_tag("scitype:X"))
            FORBIDDEN_MTYPES = ["numpyflat", "pd-wide"]
            X_inner = self._validate_data(ALLOWED_SCITYPES, FORBIDDEN_MTYPES, X, "X")

        # If both X and y are provided, check that their lengths match.
        if y_inner is not None and len(y_inner) != len(X_inner):
            raise ValueError(
                f"Length of X ({len(X_inner)}) and y ({len(y_inner)}) must match."
            )

        return X_inner, y_inner

    def _check_X(self, X=None):
        """Validate and coerce feature data X.

        This is a convenience method that calls _check_X_y with only X provided.

        Parameters
        ----------
        X : sktime-compatible container, optional (default=None)
            Feature data to be validated and converted.

        Returns
        -------
        X_inner : object or None
            The converted feature data in the format specified by the estimator.
        """
        return self._check_X_y(X=X)[0]

    def _update_y_X(self, y, X):
        """Update internal memory of seen training data.

        Accesses in self:
        _X : only if exists, then assumed same type as X and same cols
            these assumptions should be guaranteed by calls
        _y : only if exists, then assumed same type as y

        Writes to self:
        _X : same type as X - new rows from X are added to current _X
            if _X does not exist, stores X as _X
        _y : same type as y - new rows from y are added to current _y
            if _y does not exist, stores y as _y

        _X is guaranteed to be one of mtypes:
            pd.DataFrame, pd.Series, np.ndarray, pd-multiindex, numpy3D,
            pd_multiindex_hier
        _y is guaranteed to be one of mtypes:
            pd.Series, pd.DataFrame, np.ndarray (1D or 2D), pd-multiindex,
            numpy3D, pd_multiindex_hier

        Parameters
        ----------
        X : time series in sktime compatible data container format
                Time series to which to fit the forecaster in the update.
            X can be in one of the following formats, must be same scitype as in fit:
            Series scitype: pd.Series, pd.DataFrame, or np.ndarray (1D or 2D)
            Panel scitype: pd.DataFrame with 2-level row MultiIndex,
                3D np.ndarray, list of Series pd.DataFrame, or nested pd.DataFrame
            Hierarchical scitype: pd.DataFrame with 3 or more level row MultiIndex
            For further details:
                on usage, see forecasting tutorial examples/01_forecasting.ipynb
                on specification of formats, examples/AA_datatypes_and_datasets.ipynb

        y : time series in sktime compatible data container format, optional
            (default=None) Target values to update the internal memory.
            y can be in one of the following formats, must be same scitype as in fit:
            Series scitype: pd.Series, pd.DataFrame, or np.ndarray (1D or 2D)
            Panel scitype: pd.DataFrame with 2-level row MultiIndex,
                3D np.ndarray, list of Series pd.DataFrame, or nested pd.DataFrame
            Hierarchical scitype: pd.DataFrame with 3 or more level row MultiIndex
            For further details:
                on usage, see forecasting tutorial examples/01_forecasting.ipynb
                on specification of formats, examples/AA_datatypes_and_datasets.ipynb
        """
        if y is not None and self.get_config()["remember_data"]:
            if not hasattr(self, "_y") or self._y is None or not self.is_fitted:
                self._y = y
            else:
                self._y = update_data(self._y, y)

        if X is not None:
            if not hasattr(self, "_X") or self._X is None or not self.is_fitted:
                self._X = X
            else:
                self._X = update_data(self._X, X)

    def _update_X(self, X):
        """Update internal memory with new feature data X.

        Parameters
        ----------
        X : sktime-compatible container
            New feature data to be remembered.
        """
        self._update_y_X(None, X)

    def _fit(self, X, y=None):
        """Fit estimator and estimate parameters.

        private _fit containing the core logic, called from fit

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        X : guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Time series to which to fit the estimator.
        y : array-like, optional (default=None)
            Target values in the format specified by the estimators "y_inner_mtype" tag.

        Returns
        -------
        self : reference to self
        """
        raise NotImplementedError("abstract method")

    def _update(self, X, y):
        """Update fitted parameters on more data.

        private _update containing the core logic, called from update

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"

        Writes to self:
            Sets fitted model attributes ending in "_"

        Parameters
        ----------
        X : guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Time series with which to update the estimator.
        y : array-like, optional (default=None)
            New target data in the format specified by "y_inner_mtype".

        Returns
        -------
        self : reference to self
        """
        warn(
            f"NotImplementedWarning: {self.__class__.__name__} "
            f"does not have a custom `update` method implemented. "
            f"{self.__class__.__name__} will be refit each time "
            f"`update` is called.",
            obj=self,
        )
        # refit with updated data, not only passed data
        self.fit(X=self._X, y=self._y)
        # todo: should probably be self._fit, not self.fit
        # but looping to self.fit for now to avoid interface break

        return self
