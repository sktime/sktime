# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Base class template for parameter estimator scitype.

    class name: BaseParamFitter

Scitype defining methods:
    fitting                - fit(X, y=None)
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

from sktime.base import BaseEstimator
from sktime.datatypes import (
    VectorizedDF,
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
        "y_inner_mtype": "pd.DataFrame",  # which types do _fit/_predict support for y?
        "scitype:X": "Series",  # which X scitypes are supported natively?
        "scitype:y": "Series",  # which y scitypes are supported natively?
        "capability:missing_values": False,  # can estimator handle missing data?
        "capability:multivariate": False,  # can estimator handle multivariate data?
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
        X : time series in ``sktime`` compatible data container format.
            Time series to which to fit the parameter estimator.

            Individual data formats in ``sktime`` are so-called :term:`mtype`
            specifications, each mtype implements an abstract :term:`scitype`.

            * ``Series`` scitype = individual time series, vanilla forecasting.
              ``pd.DataFrame``, ``pd.Series``, or ``np.ndarray`` (1D or 2D)

            * ``Panel`` scitype = collection of time series, global/panel forecasting.
              ``pd.DataFrame`` with 2-level row ``MultiIndex`` ``(instance, time)``,
              ``3D np.ndarray`` ``(instance, variable, time)``,
              ``list`` of ``Series`` typed ``pd.DataFrame``

            * ``Hierarchical`` scitype = hierarchical collection, for
              hierarchical forecasting. ``pd.DataFrame`` with 3 or more level row
              ``MultiIndex`` ``(hierarchy_1, ..., hierarchy_n, time)``

            Whether the estimator supports panel or hierarchical data is determined
            by the scitype tags ``scitype:X`` and ``scitype:y``.

            For further details on data format, see glossary on :term:`mtype`.

        y : time series in ``sktime`` compatible data container format.
            Second time series to which to fit the parameter estimator.

            Only required if the estimator is a pairwise estimator,
            i.e., if the tag ``capability:pairwise`` is True.

            The input is ignored otherwise, and will not throw an exception.

        Returns
        -------
        self : Reference to self.
        """
        # check X is not None
        assert X is not None, "X cannot be None, but found None"

        # if fit is called, estimator is reset, including fitted state
        self.reset()

        # check and convert X/y
        X_inner, y_inner = self._check_X_y(X=X, y=y)

        # set internal X to the new X
        self._update_X_y(X_inner, y_inner)

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
        X : time series in ``sktime`` compatible data container format.
            Time series to which to fit the parameter estimator.

            Individual data formats in ``sktime`` are so-called :term:`mtype`
            specifications, each mtype implements an abstract :term:`scitype`.

            * ``Series`` scitype = individual time series, vanilla forecasting.
              ``pd.DataFrame``, ``pd.Series``, or ``np.ndarray`` (1D or 2D)

            * ``Panel`` scitype = collection of time series, global/panel forecasting.
              ``pd.DataFrame`` with 2-level row ``MultiIndex`` ``(instance, time)``,
              ``3D np.ndarray`` ``(instance, variable, time)``,
              ``list`` of ``Series`` typed ``pd.DataFrame``

            * ``Hierarchical`` scitype = hierarchical collection, for
              hierarchical forecasting. ``pd.DataFrame`` with 3 or more level row
              ``MultiIndex`` ``(hierarchy_1, ..., hierarchy_n, time)``

            Whether the estimator supports panel or hierarchical data is determined
            by the scitype tags ``scitype:X`` and ``scitype:y``.

            For further details on data format, see glossary on :term:`mtype`.

        y : time series in ``sktime`` compatible data container format.
            Second time series to which to fit the parameter estimator.

            Only required if the estimator is a pairwise estimator,
            i.e., if the tag ``capability:pairwise`` is True.

            The input is ignored otherwise, and will not throw an exception.

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
        X_inner, y_inner = self._check_X_y(X=X, y=y)

        # update internal X, y with the new X, y
        self._update_X_y(X_inner, y_inner)

        # checks and conversions complete, pass to inner update
        _safe_call(self._update, args=(), kwargs={"X": X_inner, "y": y_inner})

        return self

    def _check_X_y(self, X=None, y=None):
        """Check and coerce X and y for fit/update functions.

        Parameters
        ----------
        X : time series in sktime compatible data container format
            Time series to check.

        y : time series in sktime compatible data container format, optional
            Second time series to check.

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
            Case 2: None if y was None or if the estimator is not pairwise

        Raises
        ------
        TypeError
            If X or y is not one of the permissible Series mtypes
            If X or y is of a different scitype as self.get_tag("scitype:X")
            or self.get_tag("scitype:y")
        """
        X_inner = self._validate_data(X, var_name="X")
        y_inner = self._validate_data(y, var_name="y")
        return X_inner, y_inner

    def _validate_data(self, data, var_name="data"):
        """Validate input data (X or y)."""
        if data is None:
            return None

        ALLOWED_SCITYPES = _coerce_to_list(self.get_tag(f"scitype:{var_name}"))
        FORBIDDEN_MTYPES = ["numpyflat", "pd-wide"]

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
            "an sktime compatible time index."
            " See data format tutorial examples/AA_datatypes_and_datasets.ipynb,"
            "If you think X is already in an sktime supported input format, "
            "run sktime.datatypes.check_raise(X, mtype) to diagnose the error, "
            "where mtype is the string of the type specification you want for X. "
            "Possible mtype specification strings are as follows. "
        )

        if not data_valid:
            raise TypeError(msg + mtypes_msg)

        if DtypeKind.CATEGORICAL in data_metadata["feature_kind"]:
            raise TypeError(
                "Parameter estimators do not support categorical features "
                f" in {var_name}. "
            )

        data_scitype = data_metadata["scitype"]
        data_mtype = data_metadata["mtype"]
        data_inner_mtype = _coerce_to_list(self.get_tag(f"{var_name}_inner_mtype"))

        return convert(
            data,
            from_type=data_mtype,
            to_type=data_inner_mtype,
            as_scitype=data_scitype,
        )

    def _update_X_y(self, X, y):
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

        _X is guaranteed to be one of mtypes in the tag "X_inner_mtype"
        _y is guaranteed to be one of mtypes in the tag "y_inner_mtype"

        Parameters
        ----------
        X : time series in ``sktime`` compatible data container format.
            Time series to which to fit the parameter estimator.

            Individual data formats in ``sktime`` are so-called :term:`mtype`
            specifications, each mtype implements an abstract :term:`scitype`.

            * ``Series`` scitype = individual time series, vanilla forecasting.
              ``pd.DataFrame``, ``pd.Series``, or ``np.ndarray`` (1D or 2D)

            * ``Panel`` scitype = collection of time series, global/panel forecasting.
              ``pd.DataFrame`` with 2-level row ``MultiIndex`` ``(instance, time)``,
              ``3D np.ndarray`` ``(instance, variable, time)``,
              ``list`` of ``Series`` typed ``pd.DataFrame``

            * ``Hierarchical`` scitype = hierarchical collection, for
              hierarchical forecasting. ``pd.DataFrame`` with 3 or more level row
              ``MultiIndex`` ``(hierarchy_1, ..., hierarchy_n, time)``

            Whether the estimator supports panel or hierarchical data is determined
            by the scitype tags ``scitype:X`` and ``scitype:y``.

            For further details on data format, see glossary on :term:`mtype`.

        y : time series in ``sktime`` compatible data container format.
            Second time series to which to fit the parameter estimator.

            Only required if the estimator is a pairwise estimator,
            i.e., if the tag ``capability:pairwise`` is True.
        """
        self._update_data(X, "_X")
        self._update_data(y, "_y")

    def _update_data(self, data, self_data):
        """Update internal memory of seen training data.

        Updates attribute in self_data with data.
        """
        X = data
        if X is not None:
            # unwrap X if VectorizedDF
            if isinstance(X, VectorizedDF):
                X = X.X_multiindex
            # if _X does not exist yet, initialize it with X
            if not hasattr(self, "_X") or self._X is None or not self.is_fitted:
                setattr(self, self_data, X)
            else:
                self_X = getattr(self, self_data)
                setattr(self, self_data, update_data(self_X, X))

    def _fit(self, X, y=None):
        """Fit estimator and estimate parameters.

        private _fit containing the core logic, called from fit

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        X : guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Time series to which to fit the estimator.
        y : guaranteed to be of a type in self.get_tag("y_inner_mtype")
            Second time series to which to fit the estimator.
            None if estimator is not pairwise.

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
        y : guaranteed to be of a type in self.get_tag("y_inner_mtype")
            Second time series with which to update the estimator.
            None if estimator is not pairwise.

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
        self.fit(X=self._X, y=self._y)
        # todo: should probably be self._fit, not self.fit
        # but looping to self.fit for now to avoid interface break

        return self
