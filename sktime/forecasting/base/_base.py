# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Base class template for forecaster scitype.

    class name: BaseForecaster

Scitype defining methods:
    fitting            - fit(y, X=None, fh=None)
    forecasting        - predict(fh=None, X=None)
    updating           - update(y, X=None, update_params=True)

Convenience methods:
    fit&forecast       - fit_predict(y, X=None, fh=None)
    update&forecast    - update_predict(cv=None, X=None, update_params=True)
    forecast residuals - predict_residuals(y, X=None, fh=None)
    forecast scores    - score(y, X=None, fh=None)

Optional, special capability methods (check capability tags if available):
    forecast intervals    - predict_interval(fh=None, X=None, coverage=0.90)
    forecast quantiles    - predict_quantiles(fh=None, X=None, alpha=[0.05, 0.95])
    forecast variance     - predict_var(fh=None, X=None, cov=False)
    distribution forecast - predict_proba(fh=None, X=None, marginal=True)

Inspection methods:
    hyper-parameter inspection  - get_params()
    fitted parameter inspection - get_fitted_params()
    current ForecastingHorizon  - fh
    current cutoff              - cutoff

State:
    fitted model/strategy   - by convention, any attributes ending in "_"
    fitted state flag       - is_fitted (property)
    fitted state inspection - check_is_fitted()
"""

__author__ = ["mloning", "big-o", "fkiraly", "sveameyer13", "miraep8", "ciaran-g"]

__all__ = ["BaseForecaster", "_BaseGlobalForecaster"]

from copy import deepcopy
from itertools import product

import numpy as np
import pandas as pd

from sktime.base import BaseEstimator
from sktime.base._proba import _PredictProbaMixin
from sktime.datatypes import (
    VectorizedDF,
    check_is_error_msg,
    check_is_scitype,
    convert_to,
    get_cutoff,
    mtype_to_scitype,
    scitype_to_mtype,
    update_data,
)
from sktime.datatypes._dtypekind import DtypeKind
from sktime.forecasting.base._fh import ForecastingHorizon
from sktime.utils.datetime import _shift
from sktime.utils.dependencies import _check_estimator_deps, _check_soft_dependencies
from sktime.utils.validation.forecasting import check_alpha, check_cv, check_fh, check_X
from sktime.utils.validation.series import check_equal_time_index
from sktime.utils.warnings import warn

DEFAULT_ALPHA = 0.05


def _coerce_to_list(obj):
    """Return [obj] if obj is not a list, otherwise obj."""
    if not isinstance(obj, list):
        return [obj]
    else:
        return obj


class BaseForecaster(_PredictProbaMixin, BaseEstimator):
    """Base forecaster template class.

    The base forecaster specifies the methods and method signatures that all forecasters
    have to implement.

    Specific implementations of these methods is deferred to concrete forecasters.
    """

    # default tag values - these typically make the "safest" assumption
    # for more extensive documentation, see extension_templates/forecasting.py
    _tags = {
        # packaging info
        # --------------
        "authors": "sktime developers",  # author(s) of the object
        "maintainers": "sktime developers",  # current maintainer(s) of the object
        "python_version": None,  # PEP 440 python version specifier to limit versions
        "python_dependencies": None,  # str or list of str, package soft dependencies
        # estimator type
        # --------------
        "object_type": "forecaster",  # type of object
        "scitype:y": "univariate",  # which y are fine? univariate/multivariate/both
        "ignores-exogeneous-X": False,  # does estimator ignore the exogeneous X?
        "capability:insample": True,  # can the estimator make in-sample predictions?
        "capability:pred_int": False,  # can the estimator produce prediction intervals?
        "capability:pred_int:insample": True,  # if yes, also for in-sample horizons?
        "handles-missing-data": False,  # can estimator handle missing data?
        "y_inner_mtype": "pd.Series",  # which types do _fit/_predict, support for y?
        "X_inner_mtype": "pd.DataFrame",  # which types do _fit/_predict, support for X?
        "requires-fh-in-fit": True,  # is forecasting horizon already required in fit?
        "X-y-must-have-same-index": True,  # can estimator handle different X/y index?
        "enforce_index_type": None,  # index type that needs to be enforced in X/y
        "fit_is_empty": False,  # is fit empty and can be skipped?
        "capability:categorical_in_X": False,
        # does the forecaster natively support categorical in exogeneous X?
    }

    # configs and default config values
    # see set_config documentation for details
    _config = {
        "backend:parallel": None,  # parallelization backend for broadcasting
        #  {None, "dask", "loky", "multiprocessing", "threading"}
        #  None: no parallelization
        #  "loky", "multiprocessing" and "threading": uses `joblib` Parallel loops
        #  "joblib": uses custom joblib backend, set via `joblib_backend` tag
        #  "dask": uses `dask`, requires `dask` package in environment
        #  "ray": uses `ray`, requires `ray` package in environment
        "backend:parallel:params": None,  # params for parallelization backend
        "remember_data": True,  # whether to remember data in fit - self._X, self._y
    }

    _config_doc = {
        "remember_data": """
        remember_data : bool, default=True
            whether self._X and self._y are stored in fit, and updated
            in update. If True, self._X and self._y are stored and updated.
            If False, self._X and self._y are not stored and updated.
            This reduces serialization size when using save,
            but the update will default to "do nothing" rather than
            "refit to all data seen".
        """,
    }

    def __init__(self):
        self._is_fitted = False

        self._y = None
        self._X = None

        # forecasting horizon
        self._fh = None
        self._cutoff = None  # reference point for relative fh

        self._converter_store_y = dict()  # storage dictionary for in/output conversion

        super().__init__()
        _check_estimator_deps(self)

    def __mul__(self, other):
        """Magic * method, return (right) concatenated TransformedTargetForecaster.

        Implemented for ``other`` being a transformer, otherwise returns
        ``NotImplemented``.

        Parameters
        ----------
        other: ``sktime`` transformer, must inherit from BaseTransformer
            otherwise, ``NotImplemented`` is returned

        Returns
        -------
        TransformedTargetForecaster object,
            concatenation of ``self`` (first) with ``other`` (last).
            not nested, contains only non-TransformerPipeline ``sktime`` transformers
        """
        from sktime.forecasting.compose import TransformedTargetForecaster
        from sktime.transformations.base import BaseTransformer
        from sktime.transformations.series.adapt import TabularToSeriesAdaptor
        from sktime.utils.sklearn import is_sklearn_transformer

        # we wrap self in a pipeline, and concatenate with the other
        #   the TransformedTargetForecaster does the rest, e.g., dispatch on other
        if isinstance(other, BaseTransformer):
            self_as_pipeline = TransformedTargetForecaster(steps=[self])
            return self_as_pipeline * other
        elif is_sklearn_transformer(other):
            return self * TabularToSeriesAdaptor(other)
        else:
            return NotImplemented

    def __rmul__(self, other):
        """Magic * method, return (left) concatenated TransformedTargetForecaster.

        Implemented for ``other`` being a transformer, otherwise returns
        ``NotImplemented``.

        Parameters
        ----------
        other: ``sktime`` transformer, must inherit from BaseTransformer
            otherwise, ``NotImplemented`` is returned

        Returns
        -------
        TransformedTargetForecaster object,
            concatenation of ``other`` (first) with ``self`` (last).
            not nested, contains only non-TransformerPipeline ``sktime`` steps
        """
        from sktime.forecasting.compose import TransformedTargetForecaster
        from sktime.transformations.base import BaseTransformer
        from sktime.transformations.series.adapt import TabularToSeriesAdaptor
        from sktime.utils.sklearn import is_sklearn_transformer

        # we wrap self in a pipeline, and concatenate with the other
        #   the TransformedTargetForecaster does the rest, e.g., dispatch on other
        if isinstance(other, BaseTransformer):
            self_as_pipeline = TransformedTargetForecaster(steps=[self])
            return other * self_as_pipeline
        elif is_sklearn_transformer(other):
            return TabularToSeriesAdaptor(other) * self
        else:
            return NotImplemented

    def __rpow__(self, other):
        """Magic ** method, return (left) concatenated ForecastingPipeline.

        Implemented for ``other`` being a transformer, otherwise returns
        ``NotImplemented``.

        Parameters
        ----------
        other: ``sktime`` transformer, must inherit from BaseTransformer
            otherwise, ``NotImplemented`` is returned

        Returns
        -------
        TransformedTargetForecaster object,
            concatenation of ``other`` (first) with ``self`` (last).
            not nested, contains only non-TransformerPipeline ``sktime`` steps
        """
        from sktime.forecasting.compose import ForecastingPipeline
        from sktime.transformations.base import BaseTransformer
        from sktime.transformations.series.adapt import TabularToSeriesAdaptor
        from sktime.utils.sklearn import is_sklearn_transformer

        # we wrap self in a pipeline, and concatenate with the other
        #   the ForecastingPipeline does the rest, e.g., dispatch on other
        if isinstance(other, BaseTransformer):
            self_as_pipeline = ForecastingPipeline(steps=[self])
            return other**self_as_pipeline
        elif is_sklearn_transformer(other):
            return TabularToSeriesAdaptor(other) ** self
        else:
            return NotImplemented

    def __or__(self, other):
        """Magic | method, return MultiplexForecaster.

        Implemented for ``other`` being either a MultiplexForecaster or a forecaster.

        Parameters
        ----------
        other: ``sktime`` forecaster or sktime MultiplexForecaster

        Returns
        -------
        MultiplexForecaster object
        """
        from sktime.forecasting.compose import MultiplexForecaster

        if isinstance(other, MultiplexForecaster) or isinstance(other, BaseForecaster):
            multiplex_self = MultiplexForecaster([self])
            return multiplex_self | other
        else:
            return NotImplemented

    def __getitem__(self, key):
        """Magic [...] method, return forecaster with subsetted data.

        First index does subsetting of exogeneous input data.
        Second index does subsetting of the forecast (but not of endogeneous data).

        Keys must be valid inputs for ``columns`` in ``ColumnSelect``.

        Parameters
        ----------
        key: valid input for ``columns`` in ``ColumnSelect``, or pair thereof
            keys can also be a :-slice, in which case it is considered as not passed

        Returns
        -------
        the following composite pipeline object:
            ColumnSelect(columns1) ** self * ColumnSelect(columns2)
            where ``columns1`` is first or only item in ``key``, and ``columns2`` is the
            last
            if only one item is passed in ``key``, only ``columns1`` is applied to input
        """
        from sktime.transformations.series.subset import ColumnSelect

        def is_noneslice(obj):
            res = isinstance(obj, slice)
            res = res and obj.start is None and obj.stop is None and obj.step is None
            return res

        if isinstance(key, tuple):
            if not len(key) == 2:
                raise ValueError(
                    "there should be one or two keys when calling [] or getitem, "
                    "of a forecaster, "
                    "e.g., mytrafo[key], or mytrafo[key1, key2]. "
                    f"But {self.__class__.__name__} instance got tuple"
                    f" with {len(key)} keys."
                )
            columns1 = key[0]
            columns2 = key[1]
            if is_noneslice(columns1) and is_noneslice(columns2):
                return self
            elif is_noneslice(columns2):
                return ColumnSelect(columns1) ** self
            elif is_noneslice(columns1):
                return self * ColumnSelect(columns2)
            else:
                return ColumnSelect(columns1) ** self * ColumnSelect(columns2)
        else:
            return ColumnSelect(key) ** self

    def fit(self, y, X=None, fh=None):
        """Fit forecaster to training data.

        State change:
            Changes state to "fitted".

        Writes to self:

            * Sets fitted model attributes ending in "_", fitted attributes are
              inspectable via ``get_fitted_params``.
            * Sets ``self.is_fitted`` flag to ``True``.
            * Sets ``self.cutoff`` to last index seen in ``y``.
            * Stores ``fh`` to ``self.fh`` if ``fh`` is passed.

        Parameters
        ----------
        y : time series in ``sktime`` compatible data container format.
            Time series to which to fit the forecaster.

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

            For further details on data format, see glossary on :term:`mtype`.
            For usage, see forecasting tutorial ``examples/01_forecasting.ipynb``

        fh : int, list, pd.Index coercible, or ``ForecastingHorizon``, default=None
            The forecasting horizon encoding the time stamps to forecast at.
            If ``self.get_tag("requires-fh-in-fit")`` is ``True``,
            must be passed in ``fit``, not optional

        X : time series in ``sktime`` compatible format, optional (default=None).
            Exogeneous time series to fit the model to.
            Should be of same :term:`scitype` (``Series``, ``Panel``,
            or ``Hierarchical``) as ``y``.
            If ``self.get_tag("X-y-must-have-same-index")``,
            ``X.index`` must contain ``y.index``.

        Returns
        -------
        self : Reference to self.
        """
        # check y is not None
        assert y is not None, "y cannot be None, but found None"

        # if fit is called, estimator is reset, including fitted state
        self.reset()

        # check and convert X/y
        X_inner, y_inner = self._check_X_y(X=X, y=y)

        # set internal X/y to the new X/y
        # this also updates cutoff from y
        self._update_y_X(y_inner, X_inner)

        # check forecasting horizon and coerce to ForecastingHorizon object
        fh = self._check_fh(fh)

        # checks and conversions complete, pass to inner fit
        #####################################################
        vectorization_needed = isinstance(y_inner, VectorizedDF)
        self._is_vectorized = vectorization_needed
        # we call the ordinary _fit if no looping/vectorization needed
        if not vectorization_needed:
            self._fit(y=y_inner, X=X_inner, fh=fh)
        else:
            # otherwise we call the vectorized version of fit
            self._vectorize("fit", y=y_inner, X=X_inner, fh=fh)

        # this should happen last
        self._is_fitted = True

        return self

    def predict(self, fh=None, X=None):
        """Forecast time series at future horizon.

        State required:
            Requires state to be "fitted", i.e., ``self.is_fitted=True``.

        Accesses in self:

            * Fitted model attributes ending in "_".
            * ``self.cutoff``, ``self.is_fitted``

        Writes to self:
            Stores ``fh`` to ``self.fh`` if ``fh`` is passed and has not been passed
            previously.

        Parameters
        ----------
        fh : int, list, pd.Index coercible, or ``ForecastingHorizon``, default=None
            The forecasting horizon encoding the time stamps to forecast at.
            Should not be passed if has already been passed in ``fit``.
            If has not been passed in fit, must be passed, not optional

            If fh is not None and not of type ForecastingHorizon it is coerced to
            ForecastingHorizon via a call to _check_fh. In particular,
            if fh is of type pd.Index it is coerced via
            ForecastingHorizon(fh, is_relative=False)

        X : time series in ``sktime`` compatible format, optional (default=None)
            Exogeneous time series to use in prediction.
            Should be of same scitype (``Series``, ``Panel``, or ``Hierarchical``)
            as ``y`` in ``fit``.
            If ``self.get_tag("X-y-must-have-same-index")``,
            ``X.index`` must contain ``fh`` index reference.

        Returns
        -------
        y_pred : time series in sktime compatible data container format
            Point forecasts at ``fh``, with same index as ``fh``.
            ``y_pred`` has same type as the ``y`` that has been passed most recently:
            ``Series``, ``Panel``, ``Hierarchical`` scitype, same format (see above)
        """
        # handle inputs
        self.check_is_fitted()

        # input check and conversion for X
        X_inner = self._check_X(X=X)

        # check fh and coerce to ForecastingHorizon, if not already passed in fit
        fh = self._check_fh(fh)

        # we call the ordinary _predict if no looping/vectorization needed
        if not self._is_vectorized:
            y_pred = self._predict(fh=fh, X=X_inner)
        else:
            # otherwise we call the vectorized version of predict
            y_pred = self._vectorize("predict", X=X_inner, fh=fh)

        # convert to output mtype, identical with last y mtype seen
        y_out = convert_to(
            y_pred,
            self._y_metadata["mtype"],
            store=self._converter_store_y,
            store_behaviour="freeze",
        )

        return y_out

    def fit_predict(self, y, X=None, fh=None, X_pred=None):
        """Fit and forecast time series at future horizon.

        Same as ``fit(y, X, fh).predict(X_pred)``.
        If ``X_pred`` is not passed, same as
        ``fit(y, fh, X).predict(X)``.

        State change:
            Changes state to "fitted".

        Writes to self:

            * Sets fitted model attributes ending in "_", fitted attributes are
              inspectable via ``get_fitted_params``.
            * Sets ``self.is_fitted`` flag to ``True``.
            * Sets ``self.cutoff`` to last index seen in ``y``.
            * Stores ``fh`` to ``self.fh``.

        Parameters
        ----------
        y : time series in sktime compatible data container format
            Time series to which to fit the forecaster.

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

            For further details on data format, see glossary on :term:`mtype`.
            For usage, see forecasting tutorial ``examples/01_forecasting.ipynb``

        fh : int, list, pd.Index coercible, or ``ForecastingHorizon`` (not optional)
            The forecasting horizon encoding the time stamps to forecast at.

            If fh is not None and not of type ForecastingHorizon it is coerced to
            ForecastingHorizon via a call to _check_fh. In particular, if fh is
            of type pd.Index it is coerced via ForecastingHorizon(fh, is_relative=False)

        X : time series in ``sktime`` compatible format, optional (default=None).
            Exogeneous time series to fit the model to.
            Should be of same :term:`scitype` (``Series``, ``Panel``,
            or ``Hierarchical``) as ``y``.
            If ``self.get_tag("X-y-must-have-same-index")``,
            ``X.index`` must contain ``y.index``.

        X_pred : time series in sktime compatible format, optional (default=None)
            Exogeneous time series to use in prediction.
            If passed, will be used in predict instead of X.
            Should be of same scitype (``Series``, ``Panel``, or ``Hierarchical``)
            as ``y`` in ``fit``.
            If ``self.get_tag("X-y-must-have-same-index")``,
            ``X.index`` must contain ``fh`` index reference.

        Returns
        -------
        y_pred : time series in sktime compatible data container format
            Point forecasts at ``fh``, with same index as ``fh``.
            ``y_pred`` has same type as the ``y`` that has been passed most recently:
            ``Series``, ``Panel``, ``Hierarchical`` scitype, same format (see above)
        """
        # if X_pred is passed, run fit/predict with different X
        if X_pred is not None:
            return self.fit(y=y, X=X, fh=fh).predict(X=X_pred)
        # otherwise, we use the same X for fit and predict
        # below code carries out conversion and checks for X only once

        # if fit is called, fitted state is re-set
        self._is_fitted = False

        # check and convert X/y
        X_inner, y_inner = self._check_X_y(X=X, y=y)

        # set internal X/y to the new X/y
        # this also updates cutoff from y
        self._update_y_X(y_inner, X_inner)

        # check fh and coerce to ForecastingHorizon
        fh = self._check_fh(fh)

        # apply fit and then predict
        vectorization_needed = isinstance(y_inner, VectorizedDF)
        self._is_vectorized = vectorization_needed
        # we call the ordinary _fit if no looping/vectorization needed
        if not vectorization_needed:
            self._fit(y=y_inner, X=X_inner, fh=fh)
        else:
            # otherwise we call the vectorized version of fit
            self._vectorize("fit", y=y_inner, X=X_inner, fh=fh)

        self._is_fitted = True
        # call the public predict to avoid duplicating output conversions
        #  input conversions are skipped since we are using X_inner
        return self.predict(fh=fh, X=X_inner)

    def predict_quantiles(self, fh=None, X=None, alpha=None):
        """Compute/return quantile forecasts.

        If ``alpha`` is iterable, multiple quantiles will be calculated.

        State required:
            Requires state to be "fitted", i.e., ``self.is_fitted=True``.

        Accesses in self:

            * Fitted model attributes ending in "_".
            * ``self.cutoff``, ``self.is_fitted``

        Writes to self:
            Stores ``fh`` to ``self.fh`` if ``fh`` is passed and has not been passed
            previously.

        Parameters
        ----------
        fh : int, list, pd.Index coercible, or ``ForecastingHorizon``, default=None
            The forecasting horizon encoding the time stamps to forecast at.
            Should not be passed if has already been passed in ``fit``.
            If has not been passed in fit, must be passed, not optional


            If ``fh`` is not None and not of type ``ForecastingHorizon``,
            it is coerced to ``ForecastingHorizon`` internally (via ``_check_fh``).

            * if ``fh`` is ``int`` or array-like of ``int``, it is interpreted as
              relative horizon, and coerced to a
              relative ``ForecastingHorizon(fh, is_relative=True)``.
            * if ``fh`` is of type ``pd.Index``, it is interpreted
              as an absolute horizon, and coerced
              to an absolute ``ForecastingHorizon(fh, is_relative=False)``.


        X : time series in ``sktime`` compatible format, optional (default=None)
            Exogeneous time series to use in prediction.
            Should be of same scitype (``Series``, ``Panel``, or ``Hierarchical``)
            as ``y`` in ``fit``.
            If ``self.get_tag("X-y-must-have-same-index")``,
            ``X.index`` must contain ``fh`` index reference.

        alpha : float or list of float of unique values, optional (default=[0.05, 0.95])
            A probability or list of, at which quantile forecasts are computed.

        Returns
        -------
        quantiles : pd.DataFrame
            Column has multi-index: first level is variable name from y in fit,
                second level being the values of alpha passed to the function.
            Row index is fh, with additional (upper) levels equal to instance levels,
                    from y seen in fit, if y seen in fit was Panel or Hierarchical.
            Entries are quantile forecasts, for var in col index,
                at quantile probability in second col index, for the row index.
        """
        if not self.get_tag("capability:pred_int"):
            raise NotImplementedError(
                f"{self.__class__.__name__} does not have the capability to return "
                "quantile predictions. If you "
                "think this estimator should have the capability, please open "
                "an issue on sktime."
            )
        self.check_is_fitted()

        # input checks and conversions

        # check fh and coerce to ForecastingHorizon, if not already passed in fit
        fh = self._check_fh(fh, pred_int=True)

        # default alpha
        if alpha is None:
            alpha = [0.05, 0.95]
        # check alpha and coerce to list
        alpha = check_alpha(alpha, name="alpha")

        # input check and conversion for X
        X_inner = self._check_X(X=X)

        # we call the ordinary _predict_quantiles if no looping/vectorization needed
        if not self._is_vectorized:
            quantiles = self._predict_quantiles(fh=fh, X=X_inner, alpha=alpha)
        else:
            # otherwise we call the vectorized version of predict_quantiles
            quantiles = self._vectorize(
                "predict_quantiles",
                fh=fh,
                X=X_inner,
                alpha=alpha,
            )

        return quantiles

    def predict_interval(self, fh=None, X=None, coverage=0.90):
        """Compute/return prediction interval forecasts.

        If ``coverage`` is iterable, multiple intervals will be calculated.

        State required:
            Requires state to be "fitted", i.e., ``self.is_fitted=True``.

        Accesses in self:

            * Fitted model attributes ending in "_".
            * ``self.cutoff``, ``self.is_fitted``

        Writes to self:
            Stores ``fh`` to ``self.fh`` if ``fh`` is passed and has not been passed
            previously.

        Parameters
        ----------
        fh : int, list, pd.Index coercible, or ``ForecastingHorizon``, default=None
            The forecasting horizon encoding the time stamps to forecast at.
            Should not be passed if has already been passed in ``fit``.
            If has not been passed in fit, must be passed, not optional

            If ``fh`` is not None and not of type ``ForecastingHorizon``,
            it is coerced to ``ForecastingHorizon`` internally (via ``_check_fh``).

            * if ``fh`` is ``int`` or array-like of ``int``, it is interpreted as
              relative horizon, and coerced to a
              relative ``ForecastingHorizon(fh, is_relative=True)``.
            * if ``fh`` is of type ``pd.Index``, it is interpreted
              as an absolute horizon, and coerced
              to an absolute ``ForecastingHorizon(fh, is_relative=False)``.

        X : time series in ``sktime`` compatible format, optional (default=None)
            Exogeneous time series to use in prediction.
            Should be of same scitype (``Series``, ``Panel``, or ``Hierarchical``)
            as ``y`` in ``fit``.
            If ``self.get_tag("X-y-must-have-same-index")``,
            ``X.index`` must contain ``fh`` index reference.

        coverage : float or list of float of unique values, optional (default=0.90)
           nominal coverage(s) of predictive interval(s)

        Returns
        -------
        pred_int : pd.DataFrame
            Column has multi-index: first level is variable name from y in fit,
                second level coverage fractions for which intervals were computed.
                    in the same order as in input ``coverage``.
                Third level is string "lower" or "upper", for lower/upper interval end.
            Row index is fh, with additional (upper) levels equal to instance levels,
                from y seen in fit, if y seen in fit was Panel or Hierarchical.
            Entries are forecasts of lower/upper interval end,
                for var in col index, at nominal coverage in second col index,
                lower/upper depending on third col index, for the row index.
                Upper/lower interval end forecasts are equivalent to
                quantile forecasts at alpha = 0.5 - c/2, 0.5 + c/2 for c in coverage.
        """
        if not self.get_tag("capability:pred_int"):
            raise NotImplementedError(
                f"{self.__class__.__name__} does not have the capability to return "
                "prediction intervals. If you "
                "think this estimator should have the capability, please open "
                "an issue on sktime."
            )
        self.check_is_fitted()

        # input checks and conversions

        # check fh and coerce to ForecastingHorizon, if not already passed in fit
        fh = self._check_fh(fh, pred_int=True)

        # check alpha and coerce to list
        coverage = check_alpha(coverage, name="coverage")

        # check and convert X
        X_inner = self._check_X(X=X)

        # we call the ordinary _predict_interval if no looping/vectorization needed
        if not self._is_vectorized:
            pred_int = self._predict_interval(fh=fh, X=X_inner, coverage=coverage)
        else:
            # otherwise we call the vectorized version of predict_interval
            pred_int = self._vectorize(
                "predict_interval",
                fh=fh,
                X=X_inner,
                coverage=coverage,
            )

        return pred_int

    def predict_var(self, fh=None, X=None, cov=False):
        """Compute/return variance forecasts.

        State required:
            Requires state to be "fitted", i.e., ``self.is_fitted=True``.

        Accesses in self:

            * Fitted model attributes ending in "_".
            * ``self.cutoff``, ``self.is_fitted``

        Writes to self:
            Stores ``fh`` to ``self.fh`` if ``fh`` is passed and has not been passed
            previously.

        Parameters
        ----------
        fh : int, list, pd.Index coercible, or ``ForecastingHorizon``, default=None
            The forecasting horizon encoding the time stamps to forecast at.
            Should not be passed if has already been passed in ``fit``.
            If has not been passed in fit, must be passed, not optional


            If ``fh`` is not None and not of type ``ForecastingHorizon``,
            it is coerced to ``ForecastingHorizon`` internally (via ``_check_fh``).

            * if ``fh`` is ``int`` or array-like of ``int``, it is interpreted as
              relative horizon, and coerced to a
              relative ``ForecastingHorizon(fh, is_relative=True)``.
            * if ``fh`` is of type ``pd.Index``, it is interpreted
              as an absolute horizon, and coerced
              to an absolute ``ForecastingHorizon(fh, is_relative=False)``.


        X : time series in ``sktime`` compatible format, optional (default=None)
            Exogeneous time series to use in prediction.
            Should be of same scitype (``Series``, ``Panel``, or ``Hierarchical``)
            as ``y`` in ``fit``.
            If ``self.get_tag("X-y-must-have-same-index")``,
            ``X.index`` must contain ``fh`` index reference.

        cov : bool, optional (default=False)
            if True, computes covariance matrix forecast.
            if False, computes marginal variance forecasts.

        Returns
        -------
        pred_var : pd.DataFrame, format dependent on ``cov`` variable
            If cov=False:
                Column names are exactly those of ``y`` passed in ``fit``/``update``.
                    For nameless formats, column index will be a RangeIndex.
                Row index is fh, with additional levels equal to instance levels,
                    from y seen in fit, if y seen in fit was Panel or Hierarchical.
                Entries are variance forecasts, for var in col index.
                A variance forecast for given variable and fh index is a predicted
                    variance for that variable and index, given observed data.
            If cov=True:
                Column index is a multiindex: 1st level is variable names (as above)
                    2nd level is fh.
                Row index is fh, with additional levels equal to instance levels,
                    from y seen in fit, if y seen in fit was Panel or Hierarchical.
                Entries are (co-)variance forecasts, for var in col index, and
                    covariance between time index in row and col.
                Note: no covariance forecasts are returned between different variables.
        """
        if not self.get_tag("capability:pred_int"):
            raise NotImplementedError(
                f"{self.__class__.__name__} does not have the capability to return "
                "variance predictions. If you "
                "think this estimator should have the capability, please open "
                "an issue on sktime."
            )
        self.check_is_fitted()

        # input checks and conversions

        # check fh and coerce to ForecastingHorizon, if not already passed in fit
        fh = self._check_fh(fh, pred_int=True)

        # check and convert X
        X_inner = self._check_X(X=X)

        # we call the ordinary _predict_interval if no looping/vectorization needed
        if not self._is_vectorized:
            pred_var = self._predict_var(fh=fh, X=X_inner, cov=cov)
        else:
            # otherwise we call the vectorized version of predict_interval
            pred_var = self._vectorize("predict_var", fh=fh, X=X_inner, cov=cov)

        return pred_var

    def predict_proba(self, fh=None, X=None, marginal=True):
        """Compute/return fully probabilistic forecasts.

        Note:

        * currently only implemented for Series (non-panel, non-hierarchical) y.
        * requires ``skpro`` installed for the distribution objects returned.

        State required:
            Requires state to be "fitted", i.e., ``self.is_fitted=True``.

        Accesses in self:

            * Fitted model attributes ending in "_".
            * ``self.cutoff``, ``self.is_fitted``

        Writes to self:
            Stores ``fh`` to ``self.fh`` if ``fh`` is passed and has not been passed
            previously.

        Parameters
        ----------
        fh : int, list, pd.Index coercible, or ``ForecastingHorizon``, default=None
            The forecasting horizon encoding the time stamps to forecast at.
            Should not be passed if has already been passed in ``fit``.
            If has not been passed in fit, must be passed, not optional

            If ``fh`` is not None and not of type ``ForecastingHorizon``,
            it is coerced to ``ForecastingHorizon`` internally (via ``_check_fh``).

            * if ``fh`` is ``int`` or array-like of ``int``, it is interpreted as
              relative horizon, and coerced to a
              relative ``ForecastingHorizon(fh, is_relative=True)``.
            * if ``fh`` is of type ``pd.Index``, it is interpreted
              as an absolute horizon, and coerced
              to an absolute ``ForecastingHorizon(fh, is_relative=False)``.

        X : time series in ``sktime`` compatible format, optional (default=None)
            Exogeneous time series to use in prediction.
            Should be of same scitype (``Series``, ``Panel``, or ``Hierarchical``)
            as ``y`` in ``fit``.
            If ``self.get_tag("X-y-must-have-same-index")``,
            ``X.index`` must contain ``fh`` index reference.

        marginal : bool, optional (default=True)
            whether returned distribution is marginal by time index

        Returns
        -------
        pred_dist : skpro BaseDistribution
            predictive distribution
            if marginal=True, will be marginal distribution by time point
            if marginal=False and implemented by method, will be joint
        """
        if not self.get_tag("capability:pred_int"):
            raise NotImplementedError(
                f"{self.__class__.__name__} does not have the capability to return "
                "fully probabilistic predictions. If you "
                "think this estimator should have the capability, please open "
                "an issue on sktime."
            )
        self.check_is_fitted()

        if hasattr(self, "_is_vectorized") and self._is_vectorized:
            raise NotImplementedError(
                "automated vectorization for predict_proba is not implemented"
            )

        # predict_proba requires skpro to provide the distribution object returns
        # "silent" exception for user convenience is Normal distribution,
        # which has a minimal implementation living in sktime
        # this is not signposted to users though, to avoid too high reliance
        msg = (
            "Forecasters' predict_proba requires "
            "skpro to be present in the python environment, "
            "for distribution objects to represent distributional forecasts. "
            "To silence this message, ensure skpro is installed in the environment "
            "when calling forecasters' predict_proba."
        )
        non_default_pred_proba = self._has_implementation_of("_predict_proba")
        skpro_present = _check_soft_dependencies("skpro", severity="none")

        if not non_default_pred_proba and not skpro_present:
            warn(msg, obj=self, stacklevel=2)

        if non_default_pred_proba and not skpro_present:
            raise ImportError(msg)

        # input checks and conversions

        # check fh and coerce to ForecastingHorizon, if not already passed in fit
        fh = self._check_fh(fh, pred_int=True)

        # check and convert X
        X_inner = self._check_X(X=X)

        # pass to inner _predict_proba
        pred_dist = self._predict_proba(fh=fh, X=X_inner, marginal=marginal)

        return pred_dist

    def update(self, y, X=None, update_params=True):
        """Update cutoff value and, optionally, fitted parameters.

        If no estimator-specific update method has been implemented,
        default fall-back is as follows:

            * ``update_params=True``: fitting to all observed data so far
            * ``update_params=False``: updates cutoff and remembers data only

        State required:
            Requires state to be "fitted", i.e., ``self.is_fitted=True``.

        Accesses in self:

            * Fitted model attributes ending in "_".
            * ``self.cutoff``, ``self.is_fitted``

        Writes to self:

            * Updates ``self.cutoff`` to latest index seen in ``y``.
            * If ``update_params=True``, updates fitted model attributes ending in "_".

        Parameters
        ----------
        y : time series in ``sktime`` compatible data container format.
            Time series with which to update the forecaster.

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

            For further details on data format, see glossary on :term:`mtype`.
            For usage, see forecasting tutorial ``examples/01_forecasting.ipynb``

        X : time series in ``sktime`` compatible format, optional (default=None).
            Exogeneous time series to update the model fit with
            Should be of same :term:`scitype` (``Series``, ``Panel``,
            or ``Hierarchical``) as ``y``.
            If ``self.get_tag("X-y-must-have-same-index")``,
            ``X.index`` must contain ``y.index``.

        update_params : bool, optional (default=True)
            whether model parameters should be updated.
            If ``False``, only the cutoff is updated, model parameters
            (e.g., coefficients) are not updated.

        Returns
        -------
        self : reference to self
        """
        self.check_is_fitted()

        if y is None or (hasattr(y, "__len__") and len(y) == 0):
            warn(
                f"empty y passed to update of {self}, no update was carried out",
                obj=self,
            )
            return self

        # input checks and minor coercions on X, y
        X_inner, y_inner = self._check_X_y(X=X, y=y)

        # update internal X/y with the new X/y
        # this also updates cutoff from y
        self._update_y_X(y_inner, X_inner)

        # checks and conversions complete, pass to inner fit
        if not self._is_vectorized:
            self._update(y=y_inner, X=X_inner, update_params=update_params)
        else:
            self._vectorize("update", y=y_inner, X=X_inner, update_params=update_params)

        return self

    def update_predict(
        self,
        y,
        cv=None,
        X=None,
        update_params=True,
        reset_forecaster=True,
    ):
        """Make predictions and update model iteratively over the test set.

        Shorthand to carry out chain of multiple ``update`` / ``predict``
        executions, with data playback based on temporal splitter ``cv``.

        Same as the following (if only ``y``, ``cv`` are non-default):

        1. ``self.update(y=cv.split_series(y)[0][0])``
        2. remember ``self.predict()`` (return later in single batch)
        3. ``self.update(y=cv.split_series(y)[1][0])``
        4. remember ``self.predict()`` (return later in single batch)
        5. etc
        6. return all remembered predictions

        If no estimator-specific update method has been implemented,
        default fall-back is as follows:

            * ``update_params=True``: fitting to all observed data so far
            * ``update_params=False``: updates cutoff and remembers data only

        State required:
            Requires state to be "fitted", i.e., ``self.is_fitted=True``.

        Accesses in self:

            * Fitted model attributes ending in "_".
            * ``self.cutoff``, ``self.is_fitted``

        Writes to self (unless ``reset_forecaster=True``):
            * Updates ``self.cutoff`` to latest index seen in ``y``.
            * If ``update_params=True``, updates fitted model attributes ending in "_".

        Does not update state if ``reset_forecaster=True``.

        Parameters
        ----------
        y : time series in ``sktime`` compatible data container format.
            Time series with which to update the forecaster.

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

            For further details on data format, see glossary on :term:`mtype`.
            For usage, see forecasting tutorial ``examples/01_forecasting.ipynb``

        cv : temporal cross-validation generator inheriting from BaseSplitter, optional
            for example, ``SlidingWindowSplitter`` or ``ExpandingWindowSplitter``;
            default = ExpandingWindowSplitter with ``initial_window=1`` and defaults
            = individual data points in y/X are added and forecast one-by-one,
            ``initial_window = 1``, ``step_length = 1`` and ``fh = 1``

        X : time series in sktime compatible format, optional (default=None)
            Exogeneous time series for updating and forecasting
            Should be of same scitype (``Series``, ``Panel``, or ``Hierarchical``)
            as ``y`` in ``fit``.
            If ``self.get_tag("X-y-must-have-same-index")``,
            ``X.index`` must contain ``fh`` index reference.

        update_params : bool, optional (default=True)
            whether model parameters should be updated.
            If ``False``, only the cutoff is updated, model parameters
            (e.g., coefficients) are not updated.

        reset_forecaster : bool, optional (default=True)

            * if True, will not change the state of the forecaster,
              i.e., update/predict sequence is run with a copy,
              and cutoff, model parameters, data memory of self do not change

            * if False, will update self when the update/predict sequence is run
              as if update/predict were called directly

        Returns
        -------
        y_pred : object that tabulates point forecasts from multiple split batches
            format depends on pairs (cutoff, absolute horizon) forecast overall

            * if collection of absolute horizon points is unique:
              type is time series in sktime compatible data container format
              cutoff is suppressed in output
              has same type as the y that has been passed most recently:
              Series, Panel, Hierarchical scitype, same format (see above)

            * if collection of absolute horizon points is not unique:
              type is a pandas DataFrame, with row and col index being time stamps
              row index corresponds to cutoffs that are predicted from
              column index corresponds to absolute horizons that are predicted
              entry is the point prediction of col index predicted from row index
              entry is nan if no prediction is made at that (cutoff, horizon) pair
        """
        from sktime.split import ExpandingWindowSplitter

        if cv is None:
            cv = ExpandingWindowSplitter(initial_window=1)

        self.check_is_fitted()

        # input checks and minor coercions on X, y
        X_inner, y_inner = self._check_X_y(X=X, y=y)

        cv = check_cv(cv)

        return self._predict_moving_cutoff(
            y=y_inner,
            cv=cv,
            X=X_inner,
            update_params=update_params,
            reset_forecaster=reset_forecaster,
        )

    def update_predict_single(
        self,
        y=None,
        fh=None,
        X=None,
        update_params=True,
    ):
        """Update model with new data and make forecasts.

        This method is useful for updating and making forecasts in a single step.

        If no estimator-specific update method has been implemented,
        default fall-back is first update, then predict.

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_".
            Pointers to seen data, self._y and self.X
            self.cutoff, self._is_fitted
            If update_params=True, model attributes ending in "_".

        Writes to self:
            Update self._y and self._X with ``y`` and ``X``, by appending rows.
            Updates self.cutoff and self._cutoff to last index seen in ``y``.
            If update_params=True,
                updates fitted model attributes ending in "_".

        Parameters
        ----------
        y : time series in ``sktime`` compatible data container format.
            Time series with which to update the forecaster.

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

            For further details on data format, see glossary on :term:`mtype`.
            For usage, see forecasting tutorial ``examples/01_forecasting.ipynb``

        fh : int, list, pd.Index coercible, or ``ForecastingHorizon``, default=None
            The forecasting horizon encoding the time stamps to forecast at.
            Should not be passed if has already been passed in ``fit``.
            If has not been passed in fit, must be passed, not optional

        X : time series in sktime compatible format, optional (default=None)
            Exogeneous time series for updating and forecasting
            Should be of same scitype (``Series``, ``Panel``, or ``Hierarchical``)
            as ``y`` in ``fit``.
            If ``self.get_tag("X-y-must-have-same-index")``,
            ``X.index`` must contain ``fh`` index reference.

        update_params : bool, optional (default=True)
            whether model parameters should be updated.
            If ``False``, only the cutoff is updated, model parameters
            (e.g., coefficients) are not updated.

        Returns
        -------
        y_pred : time series in sktime compatible data container format
            Point forecasts at ``fh``, with same index as ``fh``.
            ``y_pred`` has same type as the ``y`` that has been passed most recently:
            ``Series``, ``Panel``, ``Hierarchical`` scitype, same format (see above)
        """
        if y is None or (hasattr(y, "__len__") and len(y) == 0):
            warn(
                f"empty y passed to update_predict of {self}, "
                "no update was carried out",
                obj=self,
            )
            return self.predict(fh=fh, X=X)

        self.check_is_fitted()

        # input checks and minor coercions on X, y
        X_inner, y_inner = self._check_X_y(X=X, y=y)

        # update internal _X/_y with the new X/y
        # this also updates cutoff from y
        self._update_y_X(y_inner, X_inner)

        # check fh and coerce to ForecastingHorizon, if not already passed in fit
        fh = self._check_fh(fh)

        # checks and conversions complete, pass to inner update_predict_single
        if not self._is_vectorized:
            y_pred = self._update_predict_single(
                y=y_inner, X=X_inner, fh=fh, update_params=update_params
            )
        else:
            y_pred = self._vectorize(
                "update_predict_single",
                y=y_inner,
                X=X_inner,
                fh=fh,
                update_params=update_params,
            )

        # convert to output mtype, identical with last y mtype seen
        y_pred = convert_to(
            y_pred,
            self._y_metadata["mtype"],
            store=self._converter_store_y,
            store_behaviour="freeze",
        )

        return y_pred

    def predict_residuals(self, y=None, X=None):
        """Return residuals of time series forecasts.

        Residuals will be computed for forecasts at y.index.

        If fh must be passed in fit, must agree with y.index.
        If y is an np.ndarray, and no fh has been passed in fit,
        the residuals will be computed at a fh of range(len(y.shape[0]))

        State required:
            Requires state to be "fitted".
            If fh has been set, must correspond to index of y (pandas or integer)

        Accesses in self:
            Fitted model attributes ending in "_".
            self.cutoff, self._is_fitted

        Writes to self:
            Nothing.

        Parameters
        ----------
        y : time series in sktime compatible data container format
            Time series with ground truth observations, to compute residuals to.
            Must have same type, dimension, and indices as expected return of predict.

            If None, the y seen so far (self._y) are used, in particular:

            * if preceded by a single fit call, then in-sample residuals are produced
            * if fit requires ``fh``, it must have pointed to index of y in fit

        X : time series in sktime compatible format, optional (default=None)
            Exogeneous time series for updating and forecasting
            Should be of same scitype (``Series``, ``Panel``, or ``Hierarchical``)
            as ``y`` in ``fit``.
            If ``self.get_tag("X-y-must-have-same-index")``,
            ``X.index`` must contain both ``fh`` index reference and ``y.index``.

        Returns
        -------
        y_res : time series in ``sktime`` compatible data container format
            Forecast residuals at ``fh`, with same index as ``fh``.
            ``y_res`` has same type as the ``y`` that has been passed most recently:
            ``Series``, ``Panel``, ``Hierarchical`` scitype, same format (see above)
        """
        self.check_is_fitted()

        # clone self._fh to avoid any side-effects to self due to calling _check_fh
        # and predict()
        if self._fh is not None:
            fh_orig = deepcopy(self._fh)
        else:
            fh_orig = None

        # if no y is passed, the so far observed y is used
        if y is None and self.get_config()["remember_data"]:
            y = self._y

        # we want residuals, so fh must be the index of y
        # if data frame: take directly from y
        # to avoid issues with _set_fh, we convert to relative if self.fh is
        if isinstance(y, (pd.DataFrame, pd.Series)):
            fh = ForecastingHorizon(y.index, is_relative=False, freq=self._cutoff)
            if self._fh is not None and self.fh.is_relative:
                fh = fh.to_relative(self._cutoff)
            fh = self._check_fh(fh)
        # if np.ndarray, rows are not indexed
        # so will be interpreted as range(len), or existing fh if it is stored
        elif isinstance(y, np.ndarray):
            if self._fh is None:
                fh = range(y.shape[0])
            else:
                fh = self.fh
        else:
            raise TypeError("y must be a supported Series mtype")

        y_pred = self.predict(fh=fh, X=X)

        if type(y_pred) is not type(y):
            y = convert_to(y, self._y_metadata["mtype"])

        y_res = y - y_pred

        # write fh back to self that was given before calling predict_residuals to
        # avoid side-effects
        self._fh = fh_orig

        return y_res

    def score(self, y, X=None, fh=None):
        """Scores forecast against ground truth, using MAPE (non-symmetric).

        Parameters
        ----------
        y : pd.Series, pd.DataFrame, or np.ndarray (1D or 2D)
            Time series to score

        fh : int, list, pd.Index coercible, or ``ForecastingHorizon``, default=None
            The forecasting horizon encoding the time stamps to forecast at.
        X : pd.DataFrame, or 2D np.array, optional (default=None)
            Exogeneous time series to score
            if self.get_tag("X-y-must-have-same-index"), X.index must contain y.index

        Returns
        -------
        score : float
            MAPE loss of self.predict(fh, X) with respect to y_test.

        See Also
        --------
        :meth:`sktime.performance_metrics.forecasting.mean_absolute_percentage_error`
        """
        # no input checks needed here, they will be performed
        # in predict and loss function
        from sktime.performance_metrics.forecasting import (
            mean_absolute_percentage_error,
        )

        # specify non-symmetric explicitly as it changed in the past
        return mean_absolute_percentage_error(y, self.predict(fh, X), symmetric=False)

    def get_fitted_params(self, deep=True):
        """Get fitted parameters.

        State required:
            Requires state to be "fitted".

        Parameters
        ----------
        deep : bool, default=True
            Whether to return fitted parameters of components.

            * If True, will return a dict of parameter name : value for this object,
              including fitted parameters of fittable components
              (= BaseEstimator-valued parameters).
            * If False, will return a dict of parameter name : value for this object,
              but not include fitted parameters of components.

        Returns
        -------
        fitted_params : dict with str-valued keys
            Dictionary of fitted parameters, paramname : paramvalue
            keys-value pairs include:

            * always: all fitted parameters of this object, as via ``get_param_names``
              values are fitted parameter value for that key, of this object
            * if ``deep=True``, also contains keys/value pairs of component parameters
              parameters of components are indexed as ``[componentname]__[paramname]``
              all parameters of ``componentname`` appear as ``paramname`` with its value
            * if ``deep=True``, also contains arbitrary levels of component recursion,
              e.g., ``[componentname]__[componentcomponentname]__[paramname]``, etc
        """
        # if self is not vectorized, run the default get_fitted_params
        if not getattr(self, "_is_vectorized", False):
            return super().get_fitted_params(deep=deep)

        # otherwise, we delegate to the instances' get_fitted_params
        # instances' parameters are returned at dataframe-slice-like keys
        fitted_params = {}

        # forecasters contains a pd.DataFrame with the individual forecasters
        forecasters = self.forecasters_

        # return forecasters in the "forecasters" param
        fitted_params["forecasters"] = forecasters

        def _to_str(x):
            if isinstance(x, str):
                x = f"'{x}'"
            return str(x)

        # populate fitted_params with forecasters and their parameters
        for ix, col in product(forecasters.index, forecasters.columns):
            fcst = forecasters.loc[ix, col]
            fcst_key = f"forecasters.loc[{_to_str(ix)},{_to_str(col)}]"
            fitted_params[fcst_key] = fcst
            fcst_params = fcst.get_fitted_params(deep=deep)
            for key, val in fcst_params.items():
                fitted_params[f"{fcst_key}__{key}"] = val

        return fitted_params

    def _check_X_y(self, X=None, y=None):
        """Check and coerce X/y for fit/predict/update functions.

        Parameters
        ----------
        y : pd.Series, pd.DataFrame, or np.ndarray (1D or 2D), optional (default=None)
            Time series to check.
        X : pd.DataFrame, or 2D np.array, optional (default=None)
            Exogeneous time series.

        Returns
        -------
        y_inner : Series, Panel, or Hierarchical object, or VectorizedDF
                compatible with self.get_tag("y_inner_mtype") format
            Case 1: self.get_tag("y_inner_mtype") supports scitype of y, then
                converted/coerced version of y, mtype determined by "y_inner_mtype" tag
            Case 2: self.get_tag("y_inner_mtype") does not support scitype of y, then
                VectorizedDF of y, iterated as the most complex supported scitype
                    (complexity order: Hierarchical > Panel > Series)
            Case 3: None if y was None
        X_inner : Series, Panel, or Hierarchical object, or VectorizedDF
                compatible with self.get_tag("X_inner_mtype") format
            Case 1: self.get_tag("X_inner_mtype") supports scitype of X, then
                converted/coerced version of X, mtype determined by "X_inner_mtype" tag
            Case 2: self.get_tag("X_inner_mtype") does not support scitype of X, then
                VectorizedDF of X, iterated as the most complex supported scitype
            Case 3: None if X was None

        Raises
        ------
        TypeError if y or X is not one of the permissible Series mtypes
        TypeError if y is not compatible with self.get_tag("scitype:y")
            if tag value is "univariate", y must be univariate
            if tag value is "multivariate", y must be bi- or higher-variate
            if tag value is "both", y can be either
        TypeError if self.get_tag("X-y-must-have-same-index") is True
            and the index set of X is not a super-set of the index set of y

        Writes to self
        --------------
        _y_metadata : dict with str keys, metadata from checking y
        _converter_store_y : dict, metadata from conversion for back-conversion
        """
        if X is None and y is None:
            return None, None

        def _most_complex_scitype(scitypes, smaller_equal_than=None):
            """Return most complex scitype in a list of str."""
            if "Hierarchical" in scitypes and smaller_equal_than == "Hierarchical":
                return "Hierarchical"
            elif "Panel" in scitypes and smaller_equal_than != "Series":
                return "Panel"
            elif "Series" in scitypes:
                return "Series"
            else:
                raise ValueError(
                    f"Error in {type(self).__name__}, no series scitypes supported, "
                    "likely a bug in estimator: scitypes arg passed to "
                    f"_most_complex_scitype are {scitypes}"
                )

        def _check_missing(metadata, obj_name):
            """Check input metadata against self's missing capability tag."""
            if not self.get_tag("handles-missing-data"):
                msg = (
                    f"{type(self).__name__} cannot handle missing data (nans), "
                    f"but {obj_name} passed contained missing data."
                )
                if self.get_class_tag("handles-missing-data"):
                    msg = msg + (
                        f" Whether instances of {type(self).__name__} can handle "
                        "missing data depends on parameters of the instance, "
                        "e.g., estimator components."
                    )
                if metadata["has_nans"]:
                    raise ValueError(msg)

        # retrieve supported mtypes
        y_inner_mtype = _coerce_to_list(self.get_tag("y_inner_mtype"))
        X_inner_mtype = _coerce_to_list(self.get_tag("X_inner_mtype"))
        y_inner_scitype = mtype_to_scitype(y_inner_mtype, return_unique=True)
        X_inner_scitype = mtype_to_scitype(X_inner_mtype, return_unique=True)

        ALLOWED_SCITYPES = ["Series", "Panel", "Hierarchical"]
        FORBIDDEN_MTYPES = ["numpyflat", "pd-wide"]

        mtypes_messages = []
        for scitype in ALLOWED_SCITYPES:
            mtypes = set(scitype_to_mtype(scitype))
            mtypes = list(mtypes.difference(FORBIDDEN_MTYPES))
            mtypes_messages.append(f'"For {scitype} scitype: {mtypes}. ')

        # checking y
        if y is not None:
            # request only required metadata from checks
            y_metadata_required = ["n_features", "feature_names", "feature_kind"]
            if self.get_tag("scitype:y") != "both":
                y_metadata_required += ["is_univariate"]
            if not self.get_tag("handles-missing-data"):
                y_metadata_required += ["has_nans"]

            y_valid, y_msg, y_metadata = check_is_scitype(
                y,
                scitype=ALLOWED_SCITYPES,
                return_metadata=y_metadata_required,
                var_name="y",
            )

            msg_start = (
                f"Unsupported input data type in {self.__class__.__name__}, input y"
            )
            allowed_msg = (
                "Allowed scitypes for y in forecasting are "
                f"{', '.join(ALLOWED_SCITYPES)}, "
                "for instance a pandas.DataFrame with sktime compatible time indices, "
                "or with MultiIndex and last(-1) level an sktime compatible time index."
                " See the forecasting tutorial examples/01_forecasting.ipynb, or"
                " the data format tutorial examples/AA_datatypes_and_datasets.ipynb"
            )
            if not y_valid:
                check_is_error_msg(
                    y_msg,
                    var_name=msg_start,
                    allowed_msg=allowed_msg,
                    raise_exception=True,
                )

            if DtypeKind.CATEGORICAL in y_metadata["feature_kind"]:
                raise TypeError(
                    "Forecasters do not support categorical features in endogeneous y."
                )

            y_scitype = y_metadata["scitype"]
            self._y_metadata = y_metadata
            self._y_mtype_last_seen = y_metadata["mtype"]

            req_vec_because_rows = y_scitype not in y_inner_scitype
            req_vec_because_cols = (
                self.get_tag("scitype:y") == "univariate"
                and not y_metadata["is_univariate"]
            )
            requires_vectorization = req_vec_because_rows or req_vec_because_cols

            if (
                self.get_tag("scitype:y") == "multivariate"
                and y_metadata["is_univariate"]
            ):
                raise ValueError(
                    f"Unsupported input data type in {type(self).__name__}, "
                    "this forecaster accepts only strictly multivariate data. "
                    "y must have two or more variables, but found only one."
                )

            _check_missing(y_metadata, "y")

        else:
            # y_scitype is used below - set to None if y is None
            y_scitype = None
            requires_vectorization = False
        # end checking y

        # checking X
        if X is not None:
            # request only required metadata from checks
            X_metadata_required = ["feature_kind"]
            if not self.get_tag("handles-missing-data"):
                X_metadata_required += ["has_nans"]

            X_valid, X_msg, X_metadata = check_is_scitype(
                X,
                scitype=ALLOWED_SCITYPES,
                return_metadata=X_metadata_required,
                var_name="X",
            )

            msg_start = (
                f"Unsupported input data type in {self.__class__.__name__}, input X"
            )
            allowed_msg = (
                "Allowed scitypes for X in forecasting are None, "
                f"{', '.join(ALLOWED_SCITYPES)}, "
                "for instance a pandas.DataFrame with sktime compatible time indices, "
                "or with MultiIndex and last(-1) level an sktime compatible time index."
                " See the forecasting tutorial examples/01_forecasting.ipynb, or"
                " the data format tutorial examples/AA_datatypes_and_datasets.ipynb"
            )
            if not X_valid:
                check_is_error_msg(
                    X_msg,
                    var_name=msg_start,
                    allowed_msg=allowed_msg,
                    raise_exception=True,
                )

            if (
                not self.get_tag("ignores-exogeneous-X")
                and DtypeKind.CATEGORICAL in X_metadata["feature_kind"]
                and not self.get_tag("capability:categorical_in_X")
            ):
                # replace error with encoding logic in next step.
                raise TypeError(
                    f"Forecaster {self} does not support categorical features in "
                    "exogeneous X."
                )

            X_scitype = X_metadata["scitype"]
            X_requires_vectorization = X_scitype not in X_inner_scitype
            requires_vectorization = requires_vectorization or X_requires_vectorization

            _check_missing(X_metadata, "X")

        else:
            # X_scitype is used below - set to None if X is None
            X_scitype = None

        # extra check: if X is ignored by inner methods, pass None to them
        if self.get_tag("ignores-exogeneous-X"):
            X = None
            X_scitype = None
        # end checking X

        # compatibility checks between X and y
        if X is not None and y is not None:
            if self.get_tag("X-y-must-have-same-index"):
                # currently, check_equal_time_index only works for Series
                # TODO: fix this so the check is general, using get_time_index
                if not self.get_tag("ignores-exogeneous-X") and X_scitype == "Series":
                    check_equal_time_index(X, y, mode="contains")

            if y_scitype != X_scitype:
                raise TypeError("X and y must have the same scitype")
        # end compatibility checking X and y

        # TODO: add tests that :
        #   y_inner_scitype are same as X_inner_scitype
        #   y_inner_scitype always includes "less index" scitypes

        # convert X & y to supported inner type, if necessary
        #####################################################

        # convert X and y to a supported internal mtype
        #  it X/y mtype is already supported, no conversion takes place
        #  if X/y is None, then no conversion takes place (returns None)
        #  if vectorization is required, we wrap in Vect

        if not requires_vectorization:
            # converts y, skips conversion if already of right type
            y_inner = convert_to(
                y,
                to_type=y_inner_mtype,
                as_scitype=y_scitype,  # we are dealing with series
                store=self._converter_store_y,
                store_behaviour="reset",
            )

            # converts X, converts None to None if X is None
            X_inner = convert_to(
                X,
                to_type=X_inner_mtype,
                as_scitype=X_scitype,  # we are dealing with series
            )
        else:
            iterate_as = _most_complex_scitype(
                y_inner_scitype, smaller_equal_than=y_scitype
            )
            if y is not None:
                y_inner = VectorizedDF(
                    X=y,
                    iterate_as=iterate_as,
                    is_scitype=y_scitype,
                    iterate_cols=req_vec_because_cols,
                )
            else:
                y_inner = None
            if X is not None:
                X_inner = VectorizedDF(X=X, iterate_as=iterate_as, is_scitype=X_scitype)
            else:
                X_inner = None

        return X_inner, y_inner

    def _check_X(self, X=None):
        """Shorthand for _check_X_y with one argument X, see _check_X_y."""
        return self._check_X_y(X=X)[0]

    def _update_X(self, X, enforce_index_type=None):
        if X is not None and self.get_config()["remember_data"]:
            X = check_X(X, enforce_index_type=enforce_index_type)
            self._X = update_data(self._X, X)

    def _update_y_X(self, y, X=None, enforce_index_type=None):
        """Update internal memory of seen training data.

        Accesses in self:
        _y : only if exists, then assumed same type as y and same cols
        _X : only if exists, then assumed same type as X and same cols
            these assumptions should be guaranteed by calls

        Writes to self:
        _y : same type as y - new rows from y are added to current _y
            if _y does not exist, stores y as _y
        _X : same type as X - new rows from X are added to current _X
            if _X does not exist, stores X as _X
            this is only done if X is not None
        cutoff : is set to latest index seen in y

        _y and _X are guaranteed to be one of mtypes:
            pd.DataFrame, pd.Series, np.ndarray, pd-multiindex, numpy3D,
            pd_multiindex_hier

        Parameters
        ----------
        y : pd.Series, pd.DataFrame, or np.ndarray (1D or 2D)
            Endogenous time series
        X : pd.DataFrame or 2D np.ndarray, optional (default=None)
            Exogeneous time series
        """
        if y is not None and self.get_config()["remember_data"]:
            # unwrap y if VectorizedDF
            if isinstance(y, VectorizedDF):
                y = y.X_multiindex
            # if _y does not exist yet, initialize it with y
            if not hasattr(self, "_y") or self._y is None or not self.is_fitted:
                self._y = y
            else:
                self._y = update_data(self._y, y)

            # set cutoff to the end of the observation horizon
            self._set_cutoff_from_y(y)

        if X is not None and self.get_config()["remember_data"]:
            # unwrap X if VectorizedDF
            if isinstance(X, VectorizedDF):
                X = X.X_multiindex
            # if _X does not exist yet, initialize it with X
            if not hasattr(self, "_X") or self._X is None or not self.is_fitted:
                self._X = X
            else:
                self._X = update_data(self._X, X)

    @property
    def cutoff(self):
        """Cut-off = "present time" state of forecaster.

        Returns
        -------
        cutoff : pandas compatible index element, or None
            pandas compatible index element, if cutoff has been set; None otherwise
        """
        if not hasattr(self, "_cutoff"):
            return None
        else:
            return self._cutoff

    def _set_cutoff(self, cutoff):
        """Set and update cutoff.

        Parameters
        ----------
        cutoff: pandas compatible index or index element

        Notes
        -----
        Set self._cutoff to ``cutoff``, coerced to a pandas.Index.
        """
        if not isinstance(cutoff, pd.Index):
            cutoff = pd.Index([cutoff])
        self._cutoff = cutoff

    def _set_cutoff_from_y(self, y):
        """Set and update cutoff from series y.

        Parameters
        ----------
        y : sktime compatible time series data container
            must be of one of the following mtypes:
                pd.Series, pd.DataFrame, np.ndarray, of Series scitype
                pd.multiindex, numpy3D, nested_univ, df-list, of Panel scitype
                pd_multiindex_hier, of Hierarchical scitype

        Notes
        -----
        Set self._cutoff to pandas.Index containing latest index seen in ``y``.
        """
        cutoff_idx = get_cutoff(y, self.cutoff, return_index=True)
        self._cutoff = cutoff_idx

    @property
    def fh(self):
        """Forecasting horizon that was passed."""
        # raise error if some method tries to accessed it before it has been set
        if self._fh is None:
            raise ValueError(
                f"No `fh` has been set yet, in this instance of "
                f"{self.__class__.__name__}, "
                "please specify `fh` in `fit` or `predict`"
            )

        return self._fh

    def _check_fh(self, fh, pred_int=False):
        """Check, set and update the forecasting horizon.

        Called from all methods where fh can be passed:
            fit, predict-like, update-like

        Reads and writes to self._fh.
        Reads self._cutoff, self._is_fitted, self._is_vectorized.

        Therefore, requires self._check_X_y(X=X, y=y) and
        self._update_y_X(y_inner, X_inner) to have been run at least once.

        Writes fh to self._fh if does not exist.
        Checks equality of fh with self._fh if exists, raises error if not equal.
        Assigns the frequency inferred from self._y
        to the returned forecasting horizon object.

        Parameters
        ----------
        fh : int, list, pd.Index coercible, or ``ForecastingHorizon``, default=None
             If fh is not None and not of type ForecastingHorizon it is coerced to
             ForecastingHorizon (e.g. in sktime.utils.validation.forecasting.check_fh)
             In particular, if fh is of type pd.Index it is coerced via
             ForecastingHorizon(fh, is_relative=False)
        pred_int: Check pred_int:insample tag instead of insample tag.

        Returns
        -------
        self._fh : ForecastingHorizon or None
            if ForecastingHorizon, last passed fh coerced to ForecastingHorizon

        Raises
        ------
        ValueError if self._fh exists and is inconsistent with fh
        ValueError if fh is not passed (None) in a case where it must be:
            - in fit, if self has the tag "requires-fh-in-fit" (value True)
            - in predict, if it has not been passed in fit
        """
        requires_fh = self.get_tag("requires-fh-in-fit")

        msg = (
            f"This is because fitting of the "
            f"forecaster {self.__class__.__name__} "
            f"depends on `fh`. "
        )

        # below loop treats four cases from three conditions:
        #  A. forecaster is fitted yes/no - self.is_fitted
        #  B. no fh is passed yes/no - fh is None
        #  C. fh is optional in fit yes/no - optfh

        # B. no fh is passed
        if fh is None:
            # A. strategy fitted (call of predict or similar)
            if self._is_fitted:
                # in case C. fh is optional in fit:
                # if there is none from before, there is none overall - raise error
                if not requires_fh and self._fh is None:
                    raise ValueError(
                        "The forecasting horizon `fh` must be passed "
                        "either to `fit` or `predict`, but was found in neither "
                        f"call of this {self.__class__.__name__} instance's methods."
                    )
                # in case C. fh is not optional in fit: this is fine
                # any error would have already been caught in fit

            # A. strategy not fitted (call of fit)
            elif requires_fh:
                # in case fh is not optional in fit:
                # fh must be passed in fit
                raise ValueError(
                    "The forecasting horizon `fh` must be passed to "
                    f"`fit` of {self.__class__.__name__}, but none was found. " + msg
                )
                # in case C. fh is optional in fit:
                # this is fine, nothing to check/raise

        # B. fh is passed
        else:
            # If fh is passed, coerce to ForecastingHorizon and validate (all cases)

            # if vectorized only check freq against the inner loop cutoff (check each
            # fcstr) since cutoff/frequency can be different for each compared to the
            # entire panel but the same relative fh
            if getattr(self, "_is_vectorized", False):
                fh = check_fh(fh=fh)
            else:
                fh = check_fh(fh=fh, freq=self._cutoff)

            # fh is written to self if one of the following is true
            # - estimator has not been fitted yet (for safety from side effects)
            # - fh has not been seen yet
            # - fh has been seen, but was optional in fit,
            #     this means fh needs not be same and can be overwritten
            if not requires_fh or not self._fh or not self._is_fitted:
                self._fh = fh
            # there is one error condition:
            # - fh is mandatory in fit, i.e., fh in predict must be same if passed
            # - fh already passed, and estimator is fitted
            # - fh that was passed in fit is not the same as seen in predict
            # note that elif means: optfh == False, and self._is_fitted == True
            elif self._fh and not np.array_equal(fh, self._fh):
                # raise error if existing fh and new one don't match
                raise ValueError(
                    "A different forecasting horizon `fh` has been "
                    "provided from "
                    "the one seen already in `fit`, in this instance of "
                    f"{self.__class__.__name__}. "
                    "If you want to change the forecasting "
                    "horizon, please re-fit the forecaster. " + msg
                )
            # if existing one and new match, ignore new one
        in_sample_pred = (
            self.get_tag("capability:insample")
            if not pred_int
            else self.get_tag("capability:pred_int:insample")
        )
        if (
            not in_sample_pred
            and self._fh is not None
            and not self._fh.is_all_out_of_sample(self._cutoff)
        ):
            msg = (
                f"{self.__class__.__name__} "
                f"can not perform in-sample prediction. "
                f"Found fh with in sample index: "
                f"{fh}"
            )
            raise NotImplementedError(msg)

        return self._fh

    def _vectorize(self, methodname, **kwargs):
        """Vectorized/iterated loop over method of BaseForecaster.

        Uses forecasters_ attribute to store one forecaster per loop index.
        """
        FIT_METHODS = ["fit", "update"]
        PREDICT_METHODS = [
            "predict",
            "update_predict_single",
            "predict_quantiles",
            "predict_interval",
            "predict_var",
        ]

        # retrieve data arguments
        X = kwargs.pop("X", None)
        y = kwargs.get("y", None)

        # add some common arguments to kwargs
        kwargs["args_rowvec"] = {"X": X}
        kwargs["rowname_default"] = "forecasters"
        kwargs["colname_default"] = "forecasters"

        # fit-like methods: write y to self._yvec; then run method; clone first if fit
        if methodname in FIT_METHODS:
            self._yvec = y

            if methodname == "fit":
                forecasters_ = y.vectorize_est(
                    self,
                    method="clone",
                    rowname_default="forecasters",
                    colname_default="forecasters",
                    backend=self.get_config()["backend:parallel"],
                    backend_params=self.get_config()["backend:parallel:params"],
                )
            else:
                forecasters_ = self.forecasters_

            self.forecasters_ = y.vectorize_est(
                forecasters_,
                method=methodname,
                backend=self.get_config()["backend:parallel"],
                backend_params=self.get_config()["backend:parallel:params"],
                **kwargs,
            )
            return self

        # predict-like methods: return as list, then run through reconstruct
        # to obtain a pandas based container in one of the pandas mtype formats
        elif methodname in PREDICT_METHODS:
            if methodname == "update_predict_single":
                self._yvec = y

            y_preds = self._yvec.vectorize_est(
                self.forecasters_,
                method=methodname,
                return_type="list",
                backend=self.get_config()["backend:parallel"],
                backend_params=self.get_config()["backend:parallel:params"],
                **kwargs,
            )

            # if we vectorize over columns,
            #   we need to replace top column level with variable names - part 1
            m = len(self.forecasters_.columns)
            col_multiindex = "multiindex" if m > 1 else "none"
            y_pred = self._yvec.reconstruct(
                y_preds, overwrite_index=True, col_multiindex=col_multiindex
            )
            # if vectorize over columns replace top column level with variable names
            if col_multiindex == "multiindex":
                y_pred.columns = y_pred.columns.droplevel(1)
            return y_pred

    def _fit(self, y, X, fh):
        """Fit forecaster to training data.

        private _fit containing the core logic, called from fit

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        y : guaranteed to be of a type in self.get_tag("y_inner_mtype")
            Time series to which to fit the forecaster.
            if self.get_tag("scitype:y")=="univariate":
                guaranteed to have a single column/variable
            if self.get_tag("scitype:y")=="multivariate":
                guaranteed to have 2 or more columns
            if self.get_tag("scitype:y")=="both": no restrictions apply
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
            Required (non-optional) here if self.get_tag("requires-fh-in-fit")==True
            Otherwise, if not passed in _fit, guaranteed to be passed in _predict
        X : optional (default=None)
            guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Exogeneous time series to fit to.

        Returns
        -------
        self : reference to self
        """
        raise NotImplementedError("abstract method")

    def _predict(self, fh, X):
        """Forecast time series at future horizon.

        private _predict containing the core logic, called from predict

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
            If not passed in _fit, guaranteed to be passed here
        X : optional (default=None)
            guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Exogeneous time series for the forecast

        Returns
        -------
        y_pred : pd.Series
            Point predictions
        """
        raise NotImplementedError("abstract method")

    def _update(self, y, X=None, update_params=True):
        """Update time series to incremental training data.

        private _update containing the core logic, called from update

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Writes to self:
            Sets fitted model attributes ending in "_", if update_params=True.
            Does not write to self if update_params=False.

        Parameters
        ----------
        y : guaranteed to be of a type in self.get_tag("y_inner_mtype")
            Time series with which to update the forecaster.
            if self.get_tag("scitype:y")=="univariate":
                guaranteed to have a single column/variable
            if self.get_tag("scitype:y")=="multivariate":
                guaranteed to have 2 or more columns
            if self.get_tag("scitype:y")=="both": no restrictions apply
        X : optional (default=None)
            guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Exogeneous time series for the forecast
        update_params : bool, optional (default=True)
            whether model parameters should be updated

        Returns
        -------
        self : reference to self
        """
        if update_params and self.get_config()["remember_data"]:
            # default to re-fitting if update is not implemented
            warn(
                f"NotImplementedWarning: {self.__class__.__name__} "
                f"does not have a custom `update` method implemented. "
                f"{self.__class__.__name__} will be refit each time "
                f"`update` is called with update_params=True. "
                "To refit less often, use the wrappers in the "
                "forecasting.stream module, e.g., UpdateEvery.",
                obj=self,
            )
            # we need to overwrite the mtype last seen and converter store, since the _y
            #    may have been converted
            mtype_last_seen = self._y_mtype_last_seen
            y_metadata = self._y_metadata
            _converter_store_y = self._converter_store_y
            # refit with updated data, not only passed data
            self.fit(y=self._y, X=self._X, fh=self._fh)
            # todo: should probably be self._fit, not self.fit
            # but looping to self.fit for now to avoid interface break
            self._y_mtype_last_seen = mtype_last_seen
            self._y_metadata = y_metadata
            self._converter_store_y = _converter_store_y

        # if update_params=False, and there are no components, do nothing
        # if update_params=False, and there are components, we update cutoffs
        elif self.is_composite():
            # default to calling component _updates if update is not implemented
            warn(
                f"NotImplementedWarning: {self.__class__.__name__} "
                f"does not have a custom `update` method implemented. "
                f"{self.__class__.__name__} will update all component cutoffs each time"
                f" `update` is called with update_params=False.",
                obj=self,
            )
            comp_forecasters = self._components(base_class=BaseForecaster)
            for comp in comp_forecasters.values():
                comp.update(y=y, X=X, update_params=False)

        return self

    def _update_predict_single(
        self,
        y,
        fh,
        X=None,
        update_params=True,
    ):
        """Update forecaster and then make forecasts.

        Implements default behaviour of calling update and predict sequentially, but can
        be overwritten by subclasses to implement more efficient updating algorithms
        when available.
        """
        self.update(y=y, X=X, update_params=update_params)
        return self.predict(fh=fh, X=X)

    def _predict_moving_cutoff(
        self, y, cv, X=None, update_params=True, reset_forecaster=True
    ):
        """Make single-step or multi-step moving cutoff predictions.

        Parameters
        ----------
        y : time series in sktime compatible data container format
                Time series to which to fit the forecaster in the update.
            y can be in one of the following formats, must be same scitype as in fit:
            Series scitype: pd.Series, pd.DataFrame, or np.ndarray (1D or 2D)
                for vanilla forecasting, one time series
            Panel scitype: pd.DataFrame with 2-level row MultiIndex,
                3D np.ndarray, list of Series pd.DataFrame, or nested pd.DataFrame
                for global or panel forecasting
            Hierarchical scitype: pd.DataFrame with 3 or more level row MultiIndex
                for hierarchical forecasting
            Number of columns admissible depend on the "scitype:y" tag:
                if self.get_tag("scitype:y")=="univariate":
                    y must have a single column/variable
                if self.get_tag("scitype:y")=="multivariate":
                    y must have 2 or more columns
                if self.get_tag("scitype:y")=="both": no restrictions on columns apply
            For further details:
                on usage, see forecasting tutorial examples/01_forecasting.ipynb
                on specification of formats, examples/AA_datatypes_and_datasets.ipynb
        cv : temporal cross-validation generator, optional (default=None)
        X : time series in sktime compatible format, optional (default=None)
                Exogeneous time series for updating and forecasting
            Should be of same scitype (Series, Panel, or Hierarchical) as y
            if self.get_tag("X-y-must-have-same-index"),
                X.index must contain y.index and fh.index both
            there are no restrictions on number of columns (unlike for y)
        update_params : bool, optional (default=True)
            whether model parameters should be updated in each update step
        reset_forecaster : bool, optional (default=True)
            if True, will not change the state of the forecaster,
                i.e., update/predict sequence is run with a copy,
                and cutoff, model parameters, data memory of self do not change
            if False, will update self when the update/predict sequence is run
                as if update/predict were called directly

        Returns
        -------
        y_pred = pd.Series
        """
        fh = cv.get_fh()
        y_preds = []
        cutoffs = []

        # enter into a detached cutoff mode, if reset_forecaster is True
        if reset_forecaster:
            self_copy = deepcopy(self)
        # otherwise just work with a reference to self
        else:
            self_copy = self

        # set cutoff to time point before data
        y_first_index = get_cutoff(y, return_index=True, reverse_order=True)
        self_copy._set_cutoff(_shift(y_first_index, by=-1, return_index=True))

        if isinstance(y, VectorizedDF):
            y = y.X
        if isinstance(X, VectorizedDF):
            X = X.X

        # iterate over data
        for new_window, _ in cv.split(y):
            y_new = y.iloc[new_window]

            # we use `update_predict_single` here
            #  this updates the forecasting horizon
            y_pred = self_copy.update_predict_single(
                y=y_new,
                fh=fh,
                X=X,
                update_params=update_params,
            )
            y_preds.append(y_pred)
            cutoffs.append(self_copy.cutoff)

            for i in range(len(y_preds)):
                y_preds[i] = convert_to(
                    y_preds[i],
                    self._y_metadata["mtype"],
                    store=self._converter_store_y,
                    store_behaviour="freeze",
                )
        return _format_moving_cutoff_predictions(y_preds, cutoffs)

    def _get_varnames(self, y=None):
        """Return variable column for DataFrame-like returns.

        Primarily used as helper for probabilistic predict-like methods.
        Assumes that _check_X_y has been called, and self._y_metadata set.

        Parameter
        ---------
        y : ignored, present for downwards compatibility

        Returns
        -------
        varnames : iterable of integer or str variable names
            can be list or pd.Index
            variable names for DataFrame-like returns
            identical to self._y_varnames if this attribute exists
        """
        featnames = self._y_metadata["feature_names"]
        return featnames

    def _get_columns(self, method="predict", **kwargs):
        """Return column names for DataFrame-like returns.

        Primarily used as helper for probabilistic predict-like methods.
        Assumes that _check_X_y has been called, and self._y_metadata set.

        Parameter
        ---------
        method : str, optional (default="predict")
            method for which to return column names
            one of "predict", "predict_interval", "predict_quantiles", "predict_var"
        kwargs : dict
            additional keyword arguments passed to private method
            important: args to private method, e.g., _predict, _predict_interval

        Returns
        -------
        columns : pd.Index
            column names
        """
        featnames = self._get_varnames()

        if method in ["predict", "predict_var"]:
            return featnames
        else:
            assert method in ["predict_interval", "predict_quantiles"]

        if method == "predict_interval":
            coverage = kwargs.get("coverage", None)
            if coverage is None:
                raise ValueError(
                    "coverage must be passed to _get_columns for predict_interval"
                )
            return pd.MultiIndex.from_product([featnames, coverage, ["lower", "upper"]])

        if method == "predict_quantiles":
            alpha = kwargs.get("alpha", None)
            if alpha is None:
                raise ValueError(
                    "alpha must be passed to _get_columns for predict_quantiles"
                )
            return pd.MultiIndex.from_product([featnames, alpha])


# initialize dynamic docstrings
BaseForecaster._init_dynamic_doc()


class _BaseGlobalForecaster(BaseForecaster):
    """Base global forecaster template class.

    This class is a temporal solution, might be merged into BaseForecaster later.

    The base forecaster specifies the methods and method signatures that all
    global forecasters have to implement.

    Specific implementations of these methods is deferred to concrete forecasters.

    """

    _tags = {"object_type": ["global_forecaster", "forecaster"]}

    def predict(self, fh=None, X=None, y=None):
        """Forecast time series at future horizon.

        State required:
            Requires state to be "fitted", i.e., ``self.is_fitted=True``.

        Accesses in self:

            * Fitted model attributes ending in "_".
            * ``self.cutoff``, ``self.is_fitted``

        Writes to self:
            Stores ``fh`` to ``self.fh`` if ``fh`` is passed and has not been passed
            previously.

        Parameters
        ----------
        fh : int, list, pd.Index coercible, or ``ForecastingHorizon``, default=None
            The forecasting horizon encoding the time stamps to forecast at.
            Should not be passed if has already been passed in ``fit``.
            If has not been passed in fit, must be passed, not optional

        X : time series in ``sktime`` compatible format, optional (default=None)
            Exogeneous time series to use in prediction.
            Should be of same scitype (``Series``, ``Panel``, or ``Hierarchical``)
            as ``y`` in ``fit``.
            If ``self.get_tag("X-y-must-have-same-index")``,
            ``X.index`` must contain ``fh`` index reference.
            If ``y`` is not passed (not performing global forecasting), ``X`` should
            only contain the time points to be predicted.
            If ``y`` is passed (performing global forecasting), ``X`` must contain
            all historical values and the time points to be predicted.

        y : time series in ``sktime`` compatible format, optional (default=None)
            Historical values of the time series that should be predicted.
            If not None, global forecasting will be performed.
            Only pass the historical values not the time points to be predicted.

        Returns
        -------
        y_pred : time series in sktime compatible data container format
            Point forecasts at ``fh``, with same index as ``fh``.
            ``y_pred`` has same type as the ``y`` that has been passed most recently:
            ``Series``, ``Panel``, ``Hierarchical`` scitype, same format (see above)

        Notes
        -----
        If ``y`` is not None, global forecast will be performed.
        In global forecast mode,
        ``X`` should contain all historical values and the time points to be predicted,
        while ``y`` should only contain historical values
        not the time points to be predicted.

        If ``y`` is None, non global forecast will be performed.
        In non global forecast mode,
        ``X`` should only contain the time points to be predicted,
        while ``y`` should only contain historical values
        not the time points to be predicted.
        """
        # check global forecasting tag
        gf = self.get_tag(
            "capability:global_forecasting", tag_value_default=False, raise_error=False
        )
        if not gf and y is not None:
            ValueError("no global forecasting support!")

        # handle inputs
        self.check_is_fitted()
        if y is None:
            self._global_forecasting = False
        else:
            self._global_forecasting = True
        # check and convert X/y
        X_inner, y_inner = self._check_X_y(X=X, y=y)

        # this also updates cutoff from y
        # be cautious, in fit self._X and self._y is also updated but not here!
        if y_inner is not None:
            self._set_cutoff_from_y(y_inner)

        # check fh and coerce to ForecastingHorizon, if not already passed in fit
        fh = self._check_fh(fh)

        # we call the ordinary _predict if no looping/vectorization needed
        if not self._is_vectorized:
            y_pred = self._predict(fh=fh, X=X_inner, y=y_inner)
        else:
            # otherwise we call the vectorized version of predict
            y_pred = self._vectorize("predict", y=y_inner, X=X_inner, fh=fh)

        # convert to output mtype, identical with last y mtype seen
        y_out = convert_to(
            y_pred,
            self._y_metadata["mtype"],
            store=self._converter_store_y,
            store_behaviour="freeze",
        )

        return y_out

    def _predict(self, fh, X, y):
        """Forecast time series at future horizon.

        private _predict containing the core logic, called from predict

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
            If not passed in _fit, guaranteed to be passed here
        X : optional (default=None)
            guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Exogeneous time series for the forecast
        y : time series in ``sktime`` compatible format, optional (default=None)
            Historical values of the time series that should be predicted.

        Returns
        -------
        y_pred : pd.Series
            Point predictions
        """
        raise NotImplementedError("abstract method")

    def predict_quantiles(self, fh=None, X=None, alpha=None, y=None):
        """Compute/return quantile forecasts.

        If ``alpha`` is iterable, multiple quantiles will be calculated.

        State required:
            Requires state to be "fitted", i.e., ``self.is_fitted=True``.

        Accesses in self:

            * Fitted model attributes ending in "_".
            * ``self.cutoff``, ``self.is_fitted``

        Writes to self:
            Stores ``fh`` to ``self.fh`` if ``fh`` is passed and has not been passed
            previously.

        Parameters
        ----------
        fh : int, list, np.array or ``ForecastingHorizon``, optional (default=None)
            The forecasting horizon encoding the time stamps to forecast at.
            Should not be passed if has already been passed in ``fit``.
            If has not been passed in fit, must be passed, not optional

        X : time series in ``sktime`` compatible format, optional (default=None)
            Exogeneous time series to use in prediction.
            Should be of same scitype (``Series``, ``Panel``, or ``Hierarchical``)
            as ``y`` in ``fit``.
            If ``self.get_tag("X-y-must-have-same-index")``,
            ``X.index`` must contain ``fh`` index reference.
            If ``y`` is passed (performing global forecasting), ``X`` must contain
            all historical values and the time points to be predicted.

        alpha : float or list of float of unique values, optional (default=[0.05, 0.95])
            A probability or list of, at which quantile forecasts are computed.

        y : time series in ``sktime`` compatible format, optional (default=None)
            Historical values of the time series that should be predicted.
            If not None, global forecasting will be performed.
            Only pass the historical values not the time points to be predicted.

        Returns
        -------
        quantiles : pd.DataFrame
            Column has multi-index: first level is variable name from y in fit,
                second level being the values of alpha passed to the function.
            Row index is fh, with additional (upper) levels equal to instance levels,
                    from y seen in fit, if y seen in fit was Panel or Hierarchical.
            Entries are quantile forecasts, for var in col index,
                at quantile probability in second col index, for the row index.

        Notes
        -----
        If ``y`` is not None, global forecast will be performed.
        In global forecast mode,
        ``X`` should contain all historical values and the time points to be predicted,
        while ``y`` should only contain historical values
        not the time points to be predicted.

        If ``y`` is None, non global forecast will be performed.
        In non global forecast mode,
        ``X`` should only contain the time points to be predicted,
        while ``y`` should only contain historical values
        not the time points to be predicted.
        """
        if not self.get_tag("capability:pred_int"):
            raise NotImplementedError(
                f"{self.__class__.__name__} does not have the capability to return "
                "quantile predictions. If you "
                "think this estimator should have the capability, please open "
                "an issue on sktime."
            )
        # check global forecasting tag
        gf = self.get_tag(
            "capability:global_forecasting", tag_value_default=False, raise_error=False
        )
        if not gf and y is not None:
            ValueError("no global forecasting support!")

        self.check_is_fitted()
        if y is None:
            self._global_forecasting = False
        else:
            self._global_forecasting = True

        # input checks and conversions

        # check fh and coerce to ForecastingHorizon, if not already passed in fit
        fh = self._check_fh(fh)
        if y is None:
            self._global_forecasting = False
        else:
            self._global_forecasting = True
        # default alpha
        if alpha is None:
            alpha = [0.05, 0.95]
        # check alpha and coerce to list
        alpha = check_alpha(alpha, name="alpha")

        # check and convert X/y
        X_inner, y_inner = self._check_X_y(X=X, y=y)

        # this also updates cutoff from y
        # be cautious, in fit self._X and self._y is also updated but not here!
        if y_inner is not None:
            self._set_cutoff_from_y(y_inner)

        # check fh and coerce to ForecastingHorizon, if not already passed in fit
        fh = self._check_fh(fh)

        # we call the ordinary _predict_quantiles if no looping/vectorization needed
        if not self._is_vectorized:
            quantiles = self._predict_quantiles(fh=fh, X=X_inner, alpha=alpha, y=y)
        else:
            # otherwise we call the vectorized version of predict_quantiles
            quantiles = self._vectorize(
                "predict_quantiles",
                fh=fh,
                X=X_inner,
                alpha=alpha,
                y=y,
            )

        return quantiles

    def predict_interval(self, fh=None, X=None, coverage=0.90, y=None):
        """Compute/return prediction interval forecasts.

        If ``coverage`` is iterable, multiple intervals will be calculated.

        State required:
            Requires state to be "fitted", i.e., ``self.is_fitted=True``.

        Accesses in self:

            * Fitted model attributes ending in "_".
            * ``self.cutoff``, ``self.is_fitted``

        Writes to self:
            Stores ``fh`` to ``self.fh`` if ``fh`` is passed and has not been passed
            previously.

        Parameters
        ----------
        fh : int, list, np.array or ``ForecastingHorizon``, optional (default=None)
            The forecasting horizon encoding the time stamps to forecast at.
            Should not be passed if has already been passed in ``fit``.
            If has not been passed in fit, must be passed, not optional

        X : time series in ``sktime`` compatible format, optional (default=None)
            Exogeneous time series to use in prediction.
            Should be of same scitype (``Series``, ``Panel``, or ``Hierarchical``)
            as ``y`` in ``fit``.
            If ``self.get_tag("X-y-must-have-same-index")``,
            ``X.index`` must contain ``fh`` index reference.
            If ``y`` is passed (performing global forecasting), ``X`` must contain
            all historical values and the time points to be predicted.

        coverage : float or list of float of unique values, optional (default=0.90)
           nominal coverage(s) of predictive interval(s)

        y : time series in ``sktime`` compatible format, optional (default=None)
            Historical values of the time series that should be predicted.
            If not None, global forecasting will be performed.
            Only pass the historical values not the time points to be predicted.

        Returns
        -------
        pred_int : pd.DataFrame
            Column has multi-index: first level is variable name from y in fit,
                second level coverage fractions for which intervals were computed.
                    in the same order as in input ``coverage``.
                Third level is string "lower" or "upper", for lower/upper interval end.
            Row index is fh, with additional (upper) levels equal to instance levels,
                from y seen in fit, if y seen in fit was Panel or Hierarchical.
            Entries are forecasts of lower/upper interval end,
                for var in col index, at nominal coverage in second col index,
                lower/upper depending on third col index, for the row index.
                Upper/lower interval end forecasts are equivalent to
                quantile forecasts at alpha = 0.5 - c/2, 0.5 + c/2 for c in coverage.

        Notes
        -----
        If ``y`` is not None, global forecast will be performed.
        In global forecast mode,
        ``X`` should contain all historical values and the time points to be predicted,
        while ``y`` should only contain historical values
        not the time points to be predicted.

        If ``y`` is None, non global forecast will be performed.
        In non global forecast mode,
        ``X`` should only contain the time points to be predicted,
        while ``y`` should only contain historical values
        not the time points to be predicted.
        """
        if not self.get_tag("capability:pred_int"):
            raise NotImplementedError(
                f"{self.__class__.__name__} does not have the capability to return "
                "prediction intervals. If you "
                "think this estimator should have the capability, please open "
                "an issue on sktime."
            )
        # check global forecasting tag
        gf = self.get_tag(
            "capability:global_forecasting", tag_value_default=False, raise_error=False
        )
        if not gf and y is not None:
            ValueError("no global forecasting support!")

        self.check_is_fitted()
        if y is None:
            self._global_forecasting = False
        else:
            self._global_forecasting = True

        # input checks and conversions

        # check fh and coerce to ForecastingHorizon, if not already passed in fit
        fh = self._check_fh(fh)

        # check alpha and coerce to list
        coverage = check_alpha(coverage, name="coverage")

        # check and convert X/y
        X_inner, y_inner = self._check_X_y(X=X, y=y)

        # this also updates cutoff from y
        # be cautious, in fit self._X and self._y is also updated but not here!
        if y_inner is not None:
            self._set_cutoff_from_y(y_inner)

        # we call the ordinary _predict_interval if no looping/vectorization needed
        if not self._is_vectorized:
            pred_int = self._predict_interval(
                fh=fh, X=X_inner, coverage=coverage, y=y_inner
            )
        else:
            # otherwise we call the vectorized version of predict_interval
            pred_int = self._vectorize(
                "predict_interval",
                fh=fh,
                X=X_inner,
                coverage=coverage,
                y=y_inner,
            )

        return pred_int

    def predict_var(self, fh=None, X=None, cov=False, y=None):
        """Compute/return variance forecasts.

        State required:
            Requires state to be "fitted", i.e., ``self.is_fitted=True``.

        Accesses in self:

            * Fitted model attributes ending in "_".
            * ``self.cutoff``, ``self.is_fitted``

        Writes to self:
            Stores ``fh`` to ``self.fh`` if ``fh`` is passed and has not been passed
            previously.

        Parameters
        ----------
        fh : int, list, np.array or ``ForecastingHorizon``, optional (default=None)
            The forecasting horizon encoding the time stamps to forecast at.
            Should not be passed if has already been passed in ``fit``.
            If has not been passed in fit, must be passed, not optional

        X : time series in ``sktime`` compatible format, optional (default=None)
            Exogeneous time series to use in prediction.
            Should be of same scitype (``Series``, ``Panel``, or ``Hierarchical``)
            as ``y`` in ``fit``.
            If ``self.get_tag("X-y-must-have-same-index")``,
            ``X.index`` must contain ``fh`` index reference.

        cov : bool, optional (default=False)
            if True, computes covariance matrix forecast.
            if False, computes marginal variance forecasts.

        y : time series in ``sktime`` compatible format, optional (default=None)
            Historical values of the time series that should be predicted.
            If not None, global forecasting will be performed.
            Only pass the historical values not the time points to be predicted.

        Returns
        -------
        pred_var : pd.DataFrame, format dependent on ``cov`` variable
            If cov=False:
                Column names are exactly those of ``y`` passed in ``fit``/``update``.
                    For nameless formats, column index will be a RangeIndex.
                Row index is fh, with additional levels equal to instance levels,
                    from y seen in fit, if y seen in fit was Panel or Hierarchical.
                Entries are variance forecasts, for var in col index.
                A variance forecast for given variable and fh index is a predicted
                    variance for that variable and index, given observed data.
            If cov=True:
                Column index is a multiindex: 1st level is variable names (as above)
                    2nd level is fh.
                Row index is fh, with additional levels equal to instance levels,
                    from y seen in fit, if y seen in fit was Panel or Hierarchical.
                Entries are (co-)variance forecasts, for var in col index, and
                    covariance between time index in row and col.
                Note: no covariance forecasts are returned between different variables.

        Notes
        -----
        If ``y`` is not None, global forecast will be performed.
        In global forecast mode,
        ``X`` should contain all historical values and the time points to be predicted,
        while ``y`` should only contain historical values
        not the time points to be predicted.

        If ``y`` is None, non global forecast will be performed.
        In non global forecast mode,
        ``X`` should only contain the time points to be predicted,
        while ``y`` should only contain historical values
        not the time points to be predicted.
        """
        if not self.get_tag("capability:pred_int"):
            raise NotImplementedError(
                f"{self.__class__.__name__} does not have the capability to return "
                "variance predictions. If you "
                "think this estimator should have the capability, please open "
                "an issue on sktime."
            )
        gf = self.get_tag(
            "capability:global_forecasting", tag_value_default=False, raise_error=False
        )
        if not gf and y is not None:
            ValueError("no global forecasting support!")
        self.check_is_fitted()
        if y is None:
            self._global_forecasting = False
        else:
            self._global_forecasting = True

        # input checks and conversions

        # check fh and coerce to ForecastingHorizon, if not already passed in fit
        fh = self._check_fh(fh)

        # check and convert X/y
        X_inner, y_inner = self._check_X_y(X=X, y=y)

        # this also updates cutoff from y
        # be cautious, in fit self._X and self._y is also updated but not here!
        if y_inner is not None:
            self._set_cutoff_from_y(y_inner)

        # we call the ordinary _predict_interval if no looping/vectorization needed
        if not self._is_vectorized:
            pred_var = self._predict_var(fh=fh, X=X_inner, cov=cov, y=y)
        else:
            # otherwise we call the vectorized version of predict_interval
            pred_var = self._vectorize("predict_var", fh=fh, X=X_inner, cov=cov, y=y)

        return pred_var

    def predict_proba(self, fh=None, X=None, marginal=True, y=None):
        """Compute/return fully probabilistic forecasts.

        Note: currently only implemented for Series (non-panel, non-hierarchical) y.

        State required:
            Requires state to be "fitted", i.e., ``self.is_fitted=True``.

        Accesses in self:

            * Fitted model attributes ending in "_".
            * ``self.cutoff``, ``self.is_fitted``

        Writes to self:
            Stores ``fh`` to ``self.fh`` if ``fh`` is passed and has not been passed
            previously.

        Parameters
        ----------
        fh : int, list, np.array or ``ForecastingHorizon``, optional (default=None)
            The forecasting horizon encoding the time stamps to forecast at.
            Should not be passed if has already been passed in ``fit``.
            If has not been passed in fit, must be passed, not optional

        X : time series in ``sktime`` compatible format, optional (default=None)
            Exogeneous time series to use in prediction.
            Should be of same scitype (``Series``, ``Panel``, or ``Hierarchical``)
            as ``y`` in ``fit``.
            If ``self.get_tag("X-y-must-have-same-index")``,
            ``X.index`` must contain ``fh`` index reference.
            If ``y`` is passed (performing global forecasting), ``X`` must contain
            all historical values and the time points to be predicted.

        marginal : bool, optional (default=True)
            whether returned distribution is marginal by time index

        y : time series in ``sktime`` compatible format, optional (default=None)
            Historical values of the time series that should be predicted.
            If not None, global forecasting will be performed.
            Only pass the historical values not the time points to be predicted.

        Returns
        -------
        pred_dist : sktime BaseDistribution
            predictive distribution
            if marginal=True, will be marginal distribution by time point
            if marginal=False and implemented by method, will be joint

        Notes
        -----
        If ``y`` is not None, global forecast will be performed.
        In global forecast mode,
        ``X`` should contain all historical values and the time points to be predicted,
        while ``y`` should only contain historical values
        not the time points to be predicted.

        If ``y`` is None, non global forecast will be performed.
        In non global forecast mode,
        ``X`` should only contain the time points to be predicted,
        while ``y`` should only contain historical values
        not the time points to be predicted.
        """
        if not self.get_tag("capability:pred_int"):
            raise NotImplementedError(
                f"{self.__class__.__name__} does not have the capability to return "
                "fully probabilistic predictions. If you "
                "think this estimator should have the capability, please open "
                "an issue on sktime."
            )

        if hasattr(self, "_is_vectorized") and self._is_vectorized:
            raise NotImplementedError(
                "automated vectorization for predict_proba is not implemented"
            )

        # check global forecasting tag
        gf = self.get_tag(
            "capability:global_forecasting", tag_value_default=False, raise_error=False
        )
        if not gf and y is not None:
            ValueError("no global forecasting support!")

        self.check_is_fitted()
        if y is None:
            self._global_forecasting = False
        else:
            self._global_forecasting = True

        # input checks and conversions

        # check fh and coerce to ForecastingHorizon, if not already passed in fit
        fh = self._check_fh(fh)

        # check and convert X/y
        X_inner, y_inner = self._check_X_y(X=X, y=y)

        # this also updates cutoff from y
        # be cautious, in fit self._X and self._y is also updated but not here!
        if y_inner is not None:
            self._set_cutoff_from_y(y_inner)

        # pass to inner _predict_proba
        pred_dist = self._predict_proba(fh=fh, X=X_inner, marginal=marginal, y=y)

        return pred_dist

    # @classmethod
    # def _implementation_counts(cls) -> dict:
    #     """Functions need at least n overrides to be counted as implemented.

    #     A function needs to be specified only if n!=1.

    #     Returns
    #     -------
    #     dict
    #         key is function name, and the value is n.
    #     """
    #     return {
    #         "_predict_proba": 2,
    #         "_predict_var": 2,
    #         "_predict_interval": 2,
    #         "_predict_quantiles": 2,
    #     }

    # @classmethod
    # def _has_implementation_of(cls, method):
    #     """Check if method has a concrete implementation in this class.

    #     This assumes that having an implementation is equivalent to
    #         at least n overrides of `method` in the method resolution order.

    #     Parameters
    #     ----------
    #     method : str
    #         name of method to check implementation of

    #     Returns
    #     -------
    #     bool, whether method has implementation in cls
    #         True if cls.method has been overridden at least n times in
    #         the inheritance tree (according to method resolution order)
    #         n is different for each function. If a function has been overridden
    #         in _BaseGlobalForecaster and is going to be overridden in
    #         specific forecaster again, n should be 2.
    #         n should be specified in return of self._implementation_counts if n!=1.
    #     """
    #     # walk through method resolution order and inspect methods
    #     #   of classes and direct parents, "adjacent" classes in mro
    #     mro = inspect.getmro(cls)
    #     # collect all methods that are not none
    #     methods = [getattr(c, method, None) for c in mro]
    #     methods = [m for m in methods if m is not None]
    #     implementation_counts = cls._implementation_counts()
    #     if method in implementation_counts.keys():
    #         n = implementation_counts[method]
    #     else:
    #         n = 1
    #     _n = 0
    #     for i in range(len(methods) - 1):
    #         # the method has been overridden once iff
    #         #  at least two of the methods collected are not equal
    #         #  equivalently: some two adjacent methods are not equal
    #         overridden = methods[i] != methods[i + 1]
    #         if overridden:
    #             _n += 1
    #         if _n >= n:
    #             return True

    #     return False


def _format_moving_cutoff_predictions(y_preds, cutoffs):
    """Format moving-cutoff predictions.

    Parameters
    ----------
    y_preds: list of pd.Series or pd.DataFrames, of length n
            must have equal index and equal columns
    cutoffs: iterable of cutoffs, of length n

    Returns
    -------
    y_pred: pd.DataFrame, composed of entries of y_preds
        if length of elements in y_preds is 2 or larger:
            row-index = index common to the y_preds elements
            col-index = (cutoff[i], y_pred.column)
            entry is forecast at horizon given by row, from cutoff/variable at column
        if length of elements in y_preds is 1:
            row-index = forecasting horizon
            col-index = y_pred.column
    """
    # check that input format is correct
    if not isinstance(y_preds, list):
        raise ValueError(f"`y_preds` must be a list, but found: {type(y_preds)}")
    if len(y_preds) == 0:
        return pd.DataFrame(columns=cutoffs)
    if not isinstance(y_preds[0], (pd.DataFrame, pd.Series)):
        raise ValueError("y_preds must be a list of pd.Series or pd.DataFrame")
    ylen = len(y_preds[0])
    ytype = type(y_preds[0])
    if isinstance(y_preds[0], pd.DataFrame):
        ycols = y_preds[0].columns
    for i, y_pred in enumerate(y_preds):
        if not isinstance(y_pred, ytype):
            raise ValueError(
                "all elements of y_preds must be of the same type, "
                f"but y_pred[0] is {ytype} and y_pred[{i}] is {type(y_pred)}"
            )
        if not len(y_pred) == ylen:
            raise ValueError("all elements of y_preds must be of the same length")
    if isinstance(y_preds[0], pd.DataFrame):
        for y_pred in y_preds:
            if not y_pred.columns.equals(ycols):
                raise ValueError("all elements of y_preds must have the same columns")

    if len(y_preds[0]) == 1:
        # return series for single step ahead predictions
        y_pred = pd.concat(y_preds)
    else:
        cutoffs = [cutoff[0] for cutoff in cutoffs]
        y_pred = pd.concat(y_preds, axis=1, keys=cutoffs)

    if not y_pred.index.is_monotonic_increasing:
        y_pred = y_pred.sort_index()

    if hasattr(y_preds[0], "columns") and not isinstance(y_pred.columns, pd.MultiIndex):
        col_ordered = y_preds[0].columns
        y_pred = y_pred.loc[:, col_ordered]

    return y_pred
