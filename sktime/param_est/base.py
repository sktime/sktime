# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Base class template for parameter estimator scitype.

    class name: BaseParamFitter

Scitype defining methods:
    fitting                - fit(X)
    updating               - update(X)
    get fitted parameters  - get_fitted_params() -> dict

Inspection methods:
    hyper-parameter inspection  - get_params()

State:
    fitted model/strategy   - by convention, any attributes ending in "_"
    fitted state flag       - is_fitted (property)
    fitted state inspection - check_is_fitted()
"""

__author__ = ["fkiraly"]

__all__ = ["BaseParamFitter"]

from sktime.base import BaseEstimator
from sktime.datatypes import (
    VectorizedDF,
    check_is_scitype,
    convert,
    scitype_to_mtype,
    update_data,
)
from sktime.utils.sklearn import is_sklearn_transformer
from sktime.utils.validation._dependencies import _check_estimator_deps
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
    """

    # default tag values - these typically make the "safest" assumption
    _tags = {
        "object_type": "param_est",  # type of object
        "X_inner_mtype": "pd.DataFrame",  # which types do _fit/_predict, support for X?
        "scitype:X": "Series",  # which X scitypes are supported natively?
        "capability:missing_values": False,  # can estimator handle missing data?
        "capability:multivariate": False,  # can estimator handle multivariate data?
        "python_version": None,  # PEP 440 python version specifier to limit versions
        "python_dependencies": None,  # string or str list of pkg soft dependencies
    }

    def __init__(self):
        self._is_fitted = False

        self._X = None

        super().__init__()
        _check_estimator_deps(self)

    def __rmul__(self, other):
        """Magic * method, return concatenated ParamFitterPipeline, trafos on left.

        Overloaded multiplication operation for classifiers. Implemented for `other`
        being a transformer, otherwise returns `NotImplemented`.

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

    def fit(self, X):
        """Fit estimator and estimate parameters.

        State change:
            Changes state to "fitted".

        Writes to self:
            Sets self._is_fitted flag to True.
            Writes `X` to self._X.
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

        # check and convert X/y
        X_inner = self._check_X(X=X)

        # set internal X to the new X
        self._update_X(X_inner)

        # checks and conversions complete, pass to inner fit
        #####################################################
        self._fit(X=X_inner)

        # this should happen last
        self._is_fitted = True

        return self

    def update(self, X):
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

        # input checks and minor coercions on X, y
        X_inner = self._check_X(X=X)

        # update internal X with the new X
        self._update_X(X_inner)

        # checks and conversions complete, pass to inner update
        self._update(X=X_inner)

        return self

    def _check_X(self, X=None):
        """Check and coerce X for fit/update functions.

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
        X_inner : Series, Panel, or Hierarchical object
                compatible with self.get_tag("X_inner_mtype") format
            Case 1: self.get_tag("X_inner_mtype") supports scitype of X, then
                converted/coerced version of X, mtype determined by "X_inner_mtype" tag
            Case 2: None if X was None

        Raises
        ------
        TypeError if X is not one of the permissible Series mtypes
        TypeError if X is of a different scitype as self.get_tag("scitype:X")
        """
        if X is None:
            return None

        X_inner_mtype = _coerce_to_list(self.get_tag("X_inner_mtype"))
        # X_inner_scitype = mtype_to_scitype(X_inner_mtype, return_unique=True)

        ALLOWED_SCITYPES = _coerce_to_list(self.get_tag("scitype:X"))
        FORBIDDEN_MTYPES = ["numpyflat", "pd-wide"]

        for scitype in ALLOWED_SCITYPES:
            mtypes = set(scitype_to_mtype(scitype))
            mtypes = list(mtypes.difference(FORBIDDEN_MTYPES))
            mtypes_msg = f'"For {scitype} scitype: {mtypes}. '

        # checking X
        X_valid, _, X_metadata = check_is_scitype(
            X, scitype=ALLOWED_SCITYPES, return_metadata=[], var_name="X"
        )
        msg = (
            "X must be in an sktime compatible format, "
            f"of scitypes {ALLOWED_SCITYPES}, "
            "for instance a pandas.DataFrame with sktime compatible time indices, "
            "or with MultiIndex and last(-1) level an sktime compatible time index."
            " See data format tutorial examples/AA_datatypes_and_datasets.ipynb,"
            "If you think X is already in an sktime supported input format, "
            "run sktime.datatypes.check_raise(X, mtype) to diagnose the error, "
            "where mtype is the string of the type specification you want for y. "
            "Possible mtype specification strings are as follows. "
        )
        if not X_valid:
            raise TypeError(msg + mtypes_msg)
        X_scitype = X_metadata["scitype"]
        X_mtype = X_metadata["mtype"]
        # end checking X

        # converts X, converts None to None if X is None
        X_inner = convert(
            X,
            from_type=X_mtype,
            to_type=X_inner_mtype,
            as_scitype=X_scitype,
        )

        return X_inner

    def _update_X(self, X):
        """Update internal memory of seen training data.

        Accesses in self:
        _X : only if exists, then assumed same type as X and same cols
            these assumptions should be guaranteed by calls

        Writes to self:
        _X : same type as X - new rows from X are added to current _X
            if _X does not exist, stores X as _X

        _X is guaranteed to be one of mtypes:
            pd.DataFrame, pd.Series, np.ndarray, pd-multiindex, numpy3D,
            pd_multiindex_hier

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
        """
        if X is not None:
            # unwrap X if VectorizedDF
            if isinstance(X, VectorizedDF):
                X = X.X_multiindex
            # if _X does not exist yet, initialize it with X
            if not hasattr(self, "_X") or self._X is None or not self.is_fitted:
                self._X = X
            else:
                self._X = update_data(self._X, X)

    def _fit(self, X):
        """Fit estimator and estimate parameters.

        private _fit containing the core logic, called from fit

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        X : guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Time series to which to fit the estimator.

        Returns
        -------
        self : reference to self
        """
        raise NotImplementedError("abstract method")

    def _update(self, X):
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

        Returns
        -------
        self : reference to self
        """
        # default to re-fitting if update is not implemented
        warn(
            f"NotImplementedWarning: {self.__class__.__name__} "
            f"does not have a custom `update` method implemented. "
            f"{self.__class__.__name__} will be refit each time "
            f"`update` is called.",
            obj=self,
        )
        # refit with updated data, not only passed data
        self.fit(X=self._X)
        # todo: should probably be self._fit, not self.fit
        # but looping to self.fit for now to avoid interface break

        return self

    def _get_fitted_params(self):
        """Get fitted parameters.

        private _get_fitted_params, called from get_fitted_params

        State required:
            Requires state to be "fitted".

        Returns
        -------
        fitted_params : dict
        """
        # default retrieves all self attributes ending in "_"
        # and returns them with keys that have the "_" removed
        fitted_params = [attr for attr in dir(self) if attr.endswith("_")]
        fitted_params = [x for x in fitted_params if not x.startswith("_")]
        fitted_param_dict = {p[:-1]: getattr(self, p) for p in fitted_params}

        return fitted_param_dict
