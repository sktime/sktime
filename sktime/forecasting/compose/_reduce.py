#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Reduction approaches to forecasting."""

__author__ = [
    "mloning",
    "AyushmaanSeth",
    "danbartl",
    "kAnand77",
    "LuisZugasti",
    "Lovkush-A",
    "fkiraly",
    "benheid",
]

__all__ = [
    "make_reduction",
    "DirectTimeSeriesRegressionForecaster",
    "RecursiveTimeSeriesRegressionForecaster",
    "MultioutputTimeSeriesRegressionForecaster",
    "DirectTabularRegressionForecaster",
    "RecursiveTabularRegressionForecaster",
    "MultioutputTabularRegressionForecaster",
    "DirRecTabularRegressionForecaster",
    "DirRecTimeSeriesRegressionForecaster",
    "DirectReductionForecaster",
    "RecursiveReductionForecaster",
    "YfromX",
]

import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline

from sktime.datatypes._utilities import get_time_index
from sktime.forecasting.base import BaseForecaster, ForecastingHorizon
from sktime.forecasting.base._fh import _index_range
from sktime.forecasting.base._sktime import _BaseWindowForecaster
from sktime.registry import is_scitype, scitype
from sktime.transformations.compose import FeatureUnion
from sktime.transformations.panel.reduce import Tabularizer
from sktime.transformations.series.impute import Imputer
from sktime.transformations.series.lag import Lag, ReducerTransform
from sktime.transformations.series.summarize import WindowSummarizer
from sktime.utils.datetime import _shift
from sktime.utils.dependencies import _check_soft_dependencies
from sktime.utils.estimators.dispatch import construct_dispatch
from sktime.utils.sklearn import is_sklearn_estimator, prep_skl_df, sklearn_scitype
from sktime.utils.sklearn._tag_adapter import get_sklearn_tag
from sktime.utils.validation import check_window_length
from sktime.utils.warnings import warn


def _concat_y_X(y, X):
    """Concatenate y and X prior to sliding-window transform."""
    z = y.to_numpy()
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    if X is not None:
        z = np.column_stack([z, X.to_numpy()])
    return z


# def _check_fh(fh):
#     """Check fh prior to sliding-window transform."""
#     assert fh.is_relative
#     assert fh.is_all_out_of_sample()
#     return fh.to_indexer().to_numpy()


def _ensure_relative_oos_int_fh(fh, cutoff=None):
    """Coerce fh to a 1-D NumPy array of positive *relative* integer steps.

    - If fh is a ForecastingHorizon:
        * If relative: require strictly out-of-sample, then convert to indexer.
        * If absolute: require `cutoff` to convert to relative; then validate.
    - If fh is array-like: treat as relative steps and validate.

    Returns
    -------
    np.ndarray[int], shape (n,), strictly positive (>=1).
    """
    if isinstance(fh, ForecastingHorizon):
        if fh.is_relative:
            if not fh.is_all_out_of_sample():
                raise ValueError("fh must be strictly out-of-sample (all steps >= 1).")
            arr = np.asarray(fh.to_indexer(), dtype=int).reshape(-1)
        else:
            if cutoff is None:
                raise ValueError("Absolute fh provided but no `cutoff` to convert.")
            arr = np.asarray(fh.to_relative(cutoff), dtype=int).reshape(-1)
            if (arr < 1).any():
                raise ValueError("Converted relative steps must be >= 1.")
    else:
        # array-like -> assume relative
        arr = np.asarray(fh, dtype=int).reshape(-1)
        if (arr < 1).any():
            raise ValueError("Relative steps must be >= 1.")
    return arr


def _sliding_window_transform(
    y,
    window_length,
    fh,
    X=None,
    transformers=None,
    scitype="tabular-regressor",
    pooling="local",
    windows_identical=True,
):
    """Transform time series data using sliding window.

    See ``test_sliding_window_transform_explicit`` in test_reduce.py for explicit
    example.

    Parameters
    ----------
    y : pd.Series
        Endogenous time series
    window_length : int
        Window length for transformed feature variables
    fh : ForecastingHorizon
        Forecasting horizon for transformed target variable
    X : pd.DataFrame, optional (default=None)
        Exogenous series.
    transformers: list of transformers (default = None)
        A suitable list of transformers that allows for using an en-bloc approach with
        make_reduction. This means that instead of using the raw past observations of
        y across the window length, suitable features will be generated directly from
        the past raw observations. Currently only supports WindowSummarizer (or a list
        of WindowSummarizers) to generate features e.g. the mean of the past 7
        observations.
    pooling: str {"local", "global"}, optional
        Specifies whether separate models will be fit at the level of each instance
        (local) of if you wish to fit a single model to all instances ("global").
    scitype : str {"tabular-regressor", "time-series-regressor"}, optional
        Scitype of estimator to use with transformed data.
        - If "tabular-regressor", returns X as tabular 2d array
        - If "time-series-regressor", returns X as panel 3d array
    windows_identical: bool, (default = True)
        Direct forecasting only.
        Specifies whether all direct models use the same number of observations
        (True: Total observations + 1 - window_length - maximum forecasting horizon)
        or a different number of observations (False: Total observations + 1
        - window_length - forecasting horizon).

    Returns
    -------
    yt : np.ndarray, shape = (n_timepoints - window_length, 1)
        Transformed target variable.
    Xt : np.ndarray, shape = (n_timepoints - window_length, n_variables,
    window_length)
        Transformed lagged values of target variable and exogenous variables,
        excluding contemporaneous values.
    """
    # There are different ways to implement this transform. Pre-allocating an
    # array and filling it by iterating over the window length seems to be the most
    # efficient one.

    ts_index = get_time_index(y)
    n_timepoints = ts_index.shape[0]
    window_length = check_window_length(window_length, n_timepoints)

    kwargs = {"y": y, "window_length": window_length, "X": X}
    if pooling == "global":
        kwargs = {"transformers": transformers}
        _sliding_window_trans_f = _sliding_window_transform_global
    else:  # if pooling == "local":
        kwargs = {"fh": fh, "windows_identical": windows_identical}
        _sliding_window_trans_f = _sliding_window_transform_local

    yt, Xt = _sliding_window_trans_f(y=y, X=X, window_length=window_length, **kwargs)

    # Pre-allocate array for sliding windows.
    # If the scitype is tabular regression, we have to convert X into a 2d array.
    if scitype == "tabular-regressor" and transformers is None:
        Xt = Xt.reshape(Xt.shape[0], -1)

    assert Xt.ndim == 2 or Xt.ndim == 3
    assert yt.ndim == 2

    return yt, Xt


def _sliding_window_transform_local(y, window_length, fh, X, windows_identical):
    """Transform time series data using sliding window for local pooling."""
    z = _concat_y_X(y, X)
    n_timepoints, n_variables = z.shape

    # fh = _check_fh(fh)
    fh = _ensure_relative_oos_int_fh(fh)
    fh_max = int(np.max(fh)) if len(fh) else 0

    if window_length + fh_max >= n_timepoints:
        raise ValueError(
            "The `window_length` and `fh` are incompatible with the length of `y`"
        )

    # Get the effective window length accounting for the forecasting horizon.
    effective_window_length = window_length + fh_max
    Zt = np.zeros(
        (
            n_timepoints + effective_window_length,
            n_variables,
            effective_window_length + 1,
        )
    )

    # Transform data.
    for k in range(effective_window_length + 1):
        i = effective_window_length - k
        j = n_timepoints + effective_window_length - k
        Zt[i:j, :, k] = z

    # Truncate data, selecting only full windows, discarding incomplete ones.
    if windows_identical is True:
        Zt = Zt[effective_window_length:-effective_window_length]
    else:
        Zt = Zt[effective_window_length:-window_length]
    # Return transformed feature and target variables separately. This
    # excludes contemporaneous values of the exogenous variables. Including them
    # would lead to unequal-length data, with more time points for
    # exogenous series than the target series, which is currently not supported.
    yt = Zt[:, 0, window_length + fh]
    Xt = Zt[:, :, :window_length]

    return yt, Xt


def _sliding_window_transform_global(y, window_length, X, transformers):
    """Transform time series data using sliding window for global pooling."""
    n_cut = -window_length

    if len(transformers) == 1:
        tf_fit = transformers[0].fit(y)
    else:
        feat = [("trafo_" + str(index), i) for index, i in enumerate(transformers)]
        tf_fit = FeatureUnion(feat).fit(y)
    X_from_y = tf_fit.transform(y)

    X_from_y_cut = _cut_df(X_from_y, n_obs=n_cut)
    yt = _cut_df(y, n_obs=n_cut)

    if X is not None:
        X_cut = _cut_df(X, n_obs=n_cut)
        Xt = pd.concat([X_from_y_cut, X_cut], axis=1)
    else:
        Xt = X_from_y_cut

    return yt, Xt


class MissingExogenousDataError(RuntimeError):
    """Future X not provided but are expected.

    Raised when a forecast that requires exogenous variables is requested
    but future X rows for the forecast horizon are not provided.
    """

    pass


class _ReducerMixin:
    """Common utilities for reducers."""

    # def _get_expected_pred_idx(self, fh):
    #     """Construct DataFrame Index expected in y_pred, return of _predict.

    #     Parameters
    #     ----------
    #     fh : ForecastingHorizon, fh of self; or, iterable coercible to pd.Index

    #     Returns
    #     -------
    #     fh_idx : pd.Index, expected index of y_pred returned by _predict
    #         CAVEAT: sorted by index level -1, since reduction is applied by fh
    #     """
    #     if isinstance(fh, ForecastingHorizon):
    #         fh_idx = pd.Index(fh.to_absolute_index(self.cutoff))
    #     else:
    #         fh_idx = pd.Index(fh)
    #     y_index = self._y.index

    #     if isinstance(y_index, pd.MultiIndex):
    #         y_inst_idx = y_index.droplevel(-1).unique()
    #         if isinstance(y_inst_idx, pd.MultiIndex):
    #             tuples = [x + (y,) for x in y_inst_idx for y in fh_idx]
    #         else:
    #             tuples = [(x, y) for x in y_inst_idx for y in fh_idx]
    #         fh_idx = pd.MultiIndex.from_tuples(tuples)

    #     if hasattr(y_index, "names") and y_index.names is not None:
    #         fh_idx.names = y_index.names

    #     return fh_idx

    def _cutoff_scalar(self):
        c = self.cutoff
        # unwrap 1-elem Index to a scalar
        if isinstance(c, (pd.Index, pd.PeriodIndex, pd.DatetimeIndex)):
            c = c[-1]
        # if it's a Period without freq, attach one from training index (or infer)
        if isinstance(c, pd.Period) and getattr(c, "freq", None) is None:
            y_idx = getattr(self, "_y", None)
            freq = None
            if y_idx is not None:
                y_idx = y_idx.index
                if isinstance(y_idx, (pd.PeriodIndex, pd.DatetimeIndex)):
                    freq = getattr(y_idx, "freq", None)
                    if freq is None:
                        try:
                            freq = pd.infer_freq(y_idx)
                        except Exception:
                            freq = None
            if freq is not None:
                try:
                    # rebuild Period with explicit freq
                    c = pd.Period(str(c), freq=freq)
                except Exception:
                    # fallback via timestamp if needed
                    c = pd.Period(c.to_timestamp(), freq=freq)
        return c

    def _get_expected_pred_idx(self, fh):
        """Construct DataFrame Index expected in y_pred, return of _predict.

        Parameters
        ----------
        fh : ForecastingHorizon, or iterable coercible to pd.Index

        Returns
        -------
        fh_idx : pd.Index
            Expected index of y_pred returned by _predict.
            CAVEAT: sorted by index level -1, since reduction is applied by fh.
        """
        # normalize fh to a pandas Index of absolute time points
        # If fh not provided at predict-time, use the one remembered from fit
        if fh is None:
            fh = self.fh

        if isinstance(fh, ForecastingHorizon):
            fh_abs = pd.Index(fh.to_absolute_index(self._cutoff_scalar()))
        else:
            fh_abs = pd.Index(fh)

        y_index = self._y.index

        # MultiIndex case: replicate all outer levels and append absolute horizon
        if isinstance(y_index, pd.MultiIndex):
            left = y_index.droplevel(-1).unique()
            names = y_index.names

            if isinstance(left, pd.MultiIndex):
                # left elements are tuples of the outer levels
                tuples = [(*lvl, t) for lvl in left for t in fh_abs]
                fh_idx = pd.MultiIndex.from_tuples(tuples, names=names)
            else:
                # Use from_product so single string level not split into chars
                fh_idx = pd.MultiIndex.from_product([left, fh_abs], names=names)
            return fh_idx

        # Single-level time index: preserve dtype/name where reasonable
        fh_idx = fh_abs
        if isinstance(y_index, pd.PeriodIndex):
            try:
                fh_idx = pd.PeriodIndex(fh_abs, freq=y_index.freq)
            except Exception:
                fh_idx = pd.Index(fh_abs)
        elif isinstance(y_index, pd.DatetimeIndex):
            try:
                fh_idx = pd.DatetimeIndex(fh_abs, tz=y_index.tz)
            except Exception:
                fh_idx = pd.Index(fh_abs)

        # carry over the name of the original time index if present
        if getattr(y_index, "name", None) is not None:
            try:
                fh_idx.name = y_index.name
            except Exception:
                fh_idx.name = "y_index.name"

        return fh_idx

    def _assert_future_X_coverage(self, X_pool, fh):
        """Ensure future X rows exist for all forecast indices if exo data was used."""
        # Was this forecaster fit with exogenous data?
        uses_exog = getattr(self, "_uses_exog", None)
        if uses_exog is None:
            # set during fit; be tolerant if older pickles do not have it
            uses_exog = self._X is not None

        if not uses_exog:
            return  # no exog requirement

        if X_pool is None:
            raise MissingExogenousDataError(
                "Forecaster was fitted with exog X, but no X was supplied to predict."
            )

        # Compute the *requested* absolute times (gappy allowed)
        fh = (
            ForecastingHorizon(fh, is_relative=True)
            if not isinstance(fh, ForecastingHorizon)
            else fh
        )
        abs_times = fh.to_absolute(self._cutoff_scalar()).to_pandas()

        # Build the required index we need to see in X
        if isinstance(self._y.index, pd.MultiIndex):
            series_level = self._y.index.get_level_values(0).unique()
            required_idx = pd.MultiIndex.from_product(
                [series_level, abs_times],
                names=self._y.index.names,  # usually ['series','time']
            )
        else:
            required_idx = abs_times

        # Check presence
        missing = required_idx.difference(X_pool.index)
        if len(missing) > 0:
            # show up to a couple missing stamps for clarity
            sample = list(missing[:3])
            raise MissingExogenousDataError(
                f"Missing future X rows for forecast horizon. "
                f"Examples: {sample} (total missing: {len(missing)})."
            )


class _Reducer(_BaseWindowForecaster, _ReducerMixin):
    """Base class for reducing forecasting to regression."""

    _tags = {
        "authors": [
            "mloning",
            "AyushmaanSeth",
            "danbartl",
            "kAnand77",
            "LuisZugasti",
            "Lovkush-A",
            "fkiraly",
            "benheid",
        ],
        "capability:exogenous": True,  # reduction uses X in non-trivial way
        "capability:missing_values": True,
        "capability:insample": False,
        "capability:pred_int": True,
        "capability:pred_int:insample": False,
    }

    def __init__(
        self,
        estimator,
        window_length=10,
        transformers=None,
        pooling="local",
    ):
        super().__init__(window_length=window_length)
        self.transformers = transformers
        self.transformers_ = None
        self.estimator = estimator
        self.pooling = pooling
        self._cv = None

        # it seems that the sklearn tags are not fully reliable
        # see discussion in PR #3405 and issue #3402
        # therefore this is commented out until sktime and sklearn are better aligned
        # self.set_tags(
        #     **{"capability:missing_values": estimator._get_tags()["allow_nan"]}
        # )

        # for dealing with probabilistic regressors:
        # self._est_type encodes information what type of estimator is passed
        if hasattr(estimator, "get_tags"):
            _est_type = estimator.get_tag("object_type", "regressor", False)
        else:
            _est_type = "regressor"

        if _est_type not in ["regressor", "regressor_proba"]:
            raise TypeError(
                f"error in {type(self).__name}, "
                "estimator must be either an sklearn compatible "
                "regressor, or an skpro probabilistic regressor."
            )

        # has probabilistic mode iff the estimator is of type regressor_proba
        self.set_tags(**{"capability:pred_int": _est_type == "regressor_proba"})

        self._est_type = _est_type

    def _is_predictable(self, last_window):
        """Check if we can make predictions from last window."""
        return (
            len(last_window) == self.window_length_
            and np.sum(np.isnan(last_window)) == 0
            and np.sum(np.isinf(last_window)) == 0
        )

    def _predict_quantiles(self, fh, X, alpha):
        """Compute/return prediction quantiles for a forecast.

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon
            The forecasting horizon with the steps ahead to to predict.
        X :  sktime time series object, optional (default=None)
            guaranteed to be of an mtype in self.get_tag("X_inner_mtype")
            Exogeneous time series for the forecast
        alpha : list of float (guaranteed not None and floats in [0,1] interval)
            A list of probabilities at which quantile forecasts are computed.
        """
        kwargs = {"X": X, "alpha": alpha, "method": "predict_quantiles"}

        y_pred = self._predict_boilerplate(fh, **kwargs)

        return y_pred

    def _predict_in_sample(self, fh, X=None, **kwargs):
        # Note that we currently only support out-of-sample predictions. For the
        # direct and multioutput strategy, we need to check this already during fit,
        # as the fh is required for fitting.
        pass

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        # naming convention is as follows:
        #   reducers with Tabular take an sklearn estimator, e.g., LinearRegressor
        #   reducers with TimeSeries take an sktime supervised estimator
        #       e.g., pipeline of Tabularizer and Linear Regression
        # which of these is the case, we check by checking substring in the class name
        est = LinearRegression()
        if "TimeSeries" in cls.__name__:
            est = make_pipeline(Tabularizer(), est)

        params = [{"estimator": est, "window_length": 3}]

        PROBA_IMPLEMENTED = ["DirectTabularRegressionForecaster"]
        self_supports_proba = cls.__name__ in PROBA_IMPLEMENTED

        if _check_soft_dependencies("skpro", severity="none") and self_supports_proba:
            # local import to keep skpro optional at import time
            from skpro.regression.residual import ResidualDouble

            params_proba_local = {
                "estimator": ResidualDouble.create_test_instance(),
                "pooling": "local",
                "window_length": 3,
            }
            params_proba_global = {
                "estimator": ResidualDouble.create_test_instance(),
                "pooling": "global",
                "window_length": 4,
            }

            params = params + [params_proba_local, params_proba_global]
        return params

    def _get_shifted_window(self, shift=0, y_update=None, X_update=None):
        """Get the start and end points of a shifted window.

        In recursive forecasting, the time based features need to be recalculated for
        every time step that is forecast. This is done in an iterative fashion over
        every forecasting horizon step. Shift specifies the timestamp over which the
        iteration is done, i.e. a shift of 0 will get a window between window_length
        steps in the past and t=0, shift = 1 will be window_length - 1 steps in the past
        and t= 1 etc- up to the forecasting horizon.

        Will also apply any transformers passed to the recursive reducer to y. This en
        block approach of directly applying the transformers is more efficient than
        creating all lags first across the window and then applying the transformers
        to the lagged data.

        Please see below a graphical representation of the logic using the following
        symbols:

        ``z`` = first observation to forecast.
        Not part of the window.
        ``*`` = (other) time stamps in the window which is summarized
        ``x`` = observations, past or future, not part of the window

        For``window_length = 7`` and ``fh = [3]`` we get the following windows

        ``shift = 0``
        |--------------------------- |
        | x x x x * * * * * * * z x x|
        |----------------------------|

        ``shift = 1``
        |--------------------------- |
        | x x x x x * * * * * * * z x|
        |----------------------------|

        ``shift = 2``
        |--------------------------- |
        | x x x x x x * * * * * * * z|
        |----------------------------|

        Parameters
        ----------
        shift: int, default=0
            this will be correspond to the shift of the window_length into the future
        y_update : a pandas Series or Dataframe
            y values that were obtained in the recursive fashion.
        X_update : a pandas Series or Dataframe
            X values also need to be cut based on the into windows, see above.

        Returns
        -------
        y, X: A pandas dataframe or series
            contains the y and X data prepared for the respective windows, see above.
        """
        # Panel-aware windowing: build window per series key for global/panel pooling
        if isinstance(self._y.index, pd.MultiIndex):
            names = self._y.index.names
            series_keys = self._y.index.droplevel(-1).unique()
            time_idx = self._y.index.get_level_values(-1)
            base_time = time_idx.max()
            # try to carry/derive a frequency for the 1-elem cutoff index
            freq = getattr(time_idx, "freq", None)
            if freq is None:
                try:
                    freq = pd.infer_freq(time_idx)
                except Exception:
                    freq = None
            tz = getattr(time_idx, "tz", None)

            # period-aware construction of the 1-element "cutoff" index
            if isinstance(base_time, pd.Period):
                cutoff0 = pd.PeriodIndex([base_time], freq=base_time.freq)
            elif isinstance(time_idx, pd.PeriodIndex):
                # base_time may be a scalar period pulled from a PeriodIndex
                cutoff0 = pd.PeriodIndex([base_time], freq=time_idx.freq)
            elif freq is not None:
                cutoff0 = pd.date_range(start=base_time, periods=1, freq=freq, tz=tz)
            else:
                cutoff0 = pd.DatetimeIndex([base_time], tz=tz)

            if isinstance(cutoff0, pd.DatetimeIndex) and cutoff0.freq is None:
                # try the panel's time-level frequency we computed above
                freq1 = freq

                # if still unknown, infer from any series (robust to panel concat)
                if freq1 is None:
                    try:
                        first_key = (
                            series_keys[0]
                            if not isinstance(series_keys, pd.MultiIndex)
                            else series_keys[0]
                        )
                        ser_idx = self._y.loc[first_key].index
                        # prefer explicit freq on the single-series index, else infer
                        freq1 = getattr(ser_idx, "freq", None) or pd.infer_freq(ser_idx)
                    except Exception:
                        freq1 = None

                if freq1 is not None:
                    # rebuild 1-element index with an actual freq so _shift can work
                    cutoff0 = pd.date_range(
                        start=cutoff0[0], periods=1, freq=freq1, tz=cutoff0.tz
                    )
                else:
                    # last resort: make an index with a fixed delta from single series
                    try:
                        ser_idx = self._y.loc[first_key].index  # re-use if available
                        if len(ser_idx) >= 2:
                            delta = ser_idx[1] - ser_idx[0]
                            # normalize numeric (eg from RangeIndex) to Timedelta
                            if not isinstance(
                                delta, (pd.Timedelta, pd.DateOffset, np.timedelta64)
                            ):
                                delta = pd.Timedelta(int(delta), unit="D")
                        else:
                            delta = pd.Timedelta(days=1)
                        # ----------------------------------------------------------
                    except Exception:
                        delta = pd.Timedelta(days=1)
                    start = cutoff0[0]
                    # # create a fake 2-step range to give _shift something to work with
                    # # cutoff0=pd.DatetimeIndex([start,start+delta],tz=start.tz)[:1]
                    # build 1-step range with explicit freq so shift logic has a freq
                    cutoff0 = pd.date_range(
                        start=start, periods=1, freq=delta, tz=start.tz
                    )

            # shift the one-element cutoff index; if it has no freq, recover one
            try:
                cutoff = _shift(cutoff0, by=shift, return_index=True)
            except Exception:
                # (a) try the panel time-level freq you already computed
                freq1 = freq
                ser_idx = None

                # (b) if still unknown, infer from any one series (no duplicate times)
                if freq1 is None:
                    try:
                        first_key = (
                            series_keys[0]
                            if not isinstance(series_keys, pd.MultiIndex)
                            else series_keys[0]
                        )
                        ser_idx = self._y.loc[first_key].index
                        freq1 = getattr(ser_idx, "freq", None) or pd.infer_freq(ser_idx)
                    except Exception:
                        freq1 = None

                if freq1 is not None:
                    # rebuild a 1-elem DatetimeIndex with a real freq, then shift
                    cutoff0 = pd.date_range(
                        start=base_time, periods=1, freq=freq1, tz=tz
                    )
                    cutoff = _shift(cutoff0, by=shift, return_index=True)
                else:
                    # (c) last resort: manual shift by a constant step delta
                    try:
                        if len(ser_idx) >= 2:
                            delta = ser_idx[1] - ser_idx[0]
                            # normalize deltas (e.g., from RangeIndex) to a Timedelta
                            if not isinstance(
                                delta, (pd.Timedelta, pd.DateOffset, np.timedelta64)
                            ):
                                delta = pd.Timedelta(int(delta), unit="D")
                        else:
                            delta = pd.Timedelta(days=1)
                    except Exception:
                        delta = pd.Timedelta(days=1)
                    # produce the shifted single timestamp directly
                    cutoff = pd.DatetimeIndex([base_time + shift * delta], tz=tz)

            relative_int = pd.Index(range(-self.window_length_ + 1, 2))

            if isinstance(cutoff, pd.DatetimeIndex) and cutoff.freq is None:
                # try the panel time-level freq you computed above
                freq1 = freq

                # if still unknown, infer from any single series (no duplicates)
                if freq1 is None:
                    try:
                        first_key = (
                            series_keys[0]
                            if not isinstance(series_keys, pd.MultiIndex)
                            else series_keys[0]
                        )
                        ser_idx = self._y.loc[first_key].index
                        freq1 = getattr(ser_idx, "freq", None) or pd.infer_freq(ser_idx)
                    except Exception:
                        freq1 = None

                if freq1 is not None:
                    # rebuild 1-elem cutoff with a real freq, then use _index_range
                    cutoff = pd.date_range(
                        start=cutoff[0], periods=1, freq=freq1, tz=cutoff.tz
                    )
                    times = _index_range(relative_int, cutoff)
                else:
                    # last-resort: constant step from one series (works when regular)
                    try:
                        if len(ser_idx) >= 2:
                            delta = ser_idx[1] - ser_idx[0]
                            # normalize deltas (e.g., from RangeIndex) to a Timedelta
                            if not isinstance(
                                delta, (pd.Timedelta, pd.DateOffset, np.timedelta64)
                            ):
                                delta = pd.Timedelta(int(delta), unit="D")
                        else:
                            delta = pd.Timedelta(days=1)
                    except NameError:
                        # ser_idx may not exist if earlier try failed
                        delta = pd.Timedelta(days=1)
                    base = cutoff[0]
                    times = pd.DatetimeIndex(
                        [base + i * delta for i in relative_int], tz=base.tz
                    )
            else:
                # PeriodIndex cutoff (or DatetimeIndex with freq) → normal path
                times = _index_range(relative_int, cutoff)

            # Build forecast frame via _create_fcst_df expand left-levels by times.
            # NB: pass only time index (times), not a pre-built product MultiIndex.
            y_raw = _create_fcst_df(times, self._y)
            y_raw.update(self._y)
            if y_update is not None:
                y_raw.update(y_update)
        else:
            # --- single-series path (as before) ---
            c = self._cutoff
            if isinstance(c, pd.Period):
                cutoff_idx = pd.PeriodIndex([c], freq=c.freq)
            elif isinstance(c, pd.Timestamp):
                # Ensure the 1-length DatetimeIndex has a frequency so .shift() works
                base_freq = getattr(getattr(self, "_y", None), "index", None)
                base_freq = getattr(base_freq, "freq", None)
                if (
                    base_freq is None
                    and getattr(getattr(self, "_y", None), "index", None) is not None
                ):
                    try:
                        base_freq = pd.infer_freq(self._y.index)
                    except Exception:
                        base_freq = None
                if base_freq is not None:
                    cutoff_idx = pd.date_range(
                        start=c, periods=1, freq=base_freq, tz=c.tz
                    )
                else:
                    # fallback: single-element index with no freq
                    # (we will avoid shifting elsewhere)
                    cutoff_idx = pd.DatetimeIndex([c], tz=c.tz)
            elif isinstance(c, (pd.PeriodIndex, pd.DatetimeIndex, pd.Index)):
                cutoff_idx = c[:1]
            else:
                cutoff_idx = pd.Index([c])

            # shift to the target step; if no freq, compute by index math
            try:
                cutoff = _shift(cutoff_idx, by=shift, return_index=True)
            except Exception:
                # No frequency available — emulate shift by walking a dense range
                # using the frequency of the base index if possible
                if isinstance(cutoff_idx, pd.DatetimeIndex) and len(cutoff_idx) == 1:
                    freq = getattr(self._y.index, "freq", None) or pd.infer_freq(
                        self._y.index
                    )
                    if freq is not None:
                        cutoff = pd.date_range(
                            start=cutoff_idx[0], periods=1, freq=freq
                        )
                        if shift != 0:
                            cutoff = pd.date_range(
                                start=cutoff[0], periods=1 + abs(shift), freq=freq
                            )[shift]
                            cutoff = pd.DatetimeIndex([cutoff])
                    else:
                        # Last resort: keep same timestamp (no real shift possible)
                        cutoff = cutoff_idx
                else:
                    cutoff = cutoff_idx

            relative_int = pd.Index(range(-self.window_length_ + 1, 2))
            times = _index_range(relative_int, cutoff)
            if isinstance(cutoff, pd.DatetimeIndex) and cutoff.tz is not None:
                times = times.tz_localize(cutoff.tz)
            y_raw = _create_fcst_df(times, self._y)
            y_raw.update(self._y)
            if y_update is not None:
                y_raw.update(y_update)

        # X features at the summary point only (the last time in the window)
        last_time = times[-1]

        if self._X is not None:
            # build the single-row index for X at last_time
            if isinstance(self._y.index, pd.MultiIndex):
                # panel/global case: index is (series_key, last_time)
                if isinstance(series_keys, pd.MultiIndex):
                    X_idx = pd.MultiIndex.from_tuples(
                        [(*k, last_time) for k in series_keys], names=names
                    )
                else:
                    X_idx = pd.MultiIndex.from_product(
                        [series_keys, pd.Index([last_time])], names=names
                    )
            else:
                # single series, preserve index type if possible
                if isinstance(last_time, pd.Period):
                    X_idx = pd.PeriodIndex([last_time], freq=last_time.freq)
                elif isinstance(last_time, pd.Timestamp):
                    X_idx = pd.DatetimeIndex([last_time], tz=last_time.tz)
                else:
                    X_idx = pd.Index([last_time])

            # create the X frame at last_time and update with known/future rows
            X = _create_fcst_df(X_idx, self._X)
            X.update(self._X)
            if X_update is not None:
                X.update(X_update)
            X_cut = _cut_df(X)

            # features derived from y_raw at last_time
            if len(self.transformers_) == 1:
                X_from_y_cut = _cut_df(self.transformers_[0].fit_transform(y_raw))
            else:
                X_from_y_cut = _cut_df(
                    FeatureUnion(
                        [
                            ("trafo_" + str(i), t)
                            for i, t in enumerate(self.transformers_)
                        ]
                    ).fit_transform(y_raw)
                )
            X = pd.concat([X_from_y_cut, X_cut], axis=1)
        else:
            # no exog: only features derived from y_raw at last_time
            if len(self.transformers_) == 1:
                X = _cut_df(self.transformers_[0].fit_transform(y_raw))
            else:
                ref = self.transformers_
                feat = [("trafo_" + str(i), t) for i, t in enumerate(ref)]
                X = _cut_df(FeatureUnion(feat).fit_transform(y_raw))

        y = _cut_df(y_raw)
        return y, X


class _DirectReducer(_Reducer):
    strategy = "direct"
    _tags = {
        "requires-fh-in-fit": True,  # is the forecasting horizon required in fit?
    }

    def __init__(
        self,
        estimator,
        window_length=10,
        transformers=None,
        pooling="local",
        windows_identical=True,
    ):
        self.windows_identical = windows_identical
        super().__init__(
            estimator=estimator,
            window_length=window_length,
            transformers=transformers,
            pooling=pooling,
        )

    def _transform(self, y, X=None):
        fh = self.fh.to_relative(self._cutoff_scalar())
        return _sliding_window_transform(
            y,
            window_length=self.window_length_,
            fh=fh,
            X=X,
            transformers=self.transformers_,
            scitype=self._estimator_scitype,
            pooling=self.pooling,
            windows_identical=self.windows_identical,
        )

    def _fit(self, y, X, fh):
        """Fit to training data.

        Parameters
        ----------
        y : pd.Series
            Target time series to which to fit the forecaster.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored
        fh : int, list or np.array, optional (default=None)
             The forecasters horizon with the steps ahead to to predict.

        Returns
        -------
        self : Estimator
            An fitted instance of self.
        """
        # We currently only support out-of-sample predictions. For the direct
        # strategy, we need to check this at the beginning of fit, as the fh is
        # required for fitting.
        self._timepoints = get_time_index(y)
        n_timepoints = len(self._timepoints)

        if self.pooling is not None and self.pooling not in ["local", "global"]:
            raise ValueError(
                "pooling must be one of local, global" + f" but found {self.pooling}"
            )

        if self.window_length is not None and self.transformers is not None:
            raise ValueError(
                "Transformers provided, suggesting en-bloc approach"
                + " to derive reduction features. Window length will be"
                + " inferred, please set to None"
            )
        if self.transformers is not None and self.pooling == "local":
            raise ValueError(
                "Transformers currently cannot be provided"
                + "for models that run locally"
            )
        pd_format = isinstance(y, pd.Series) or isinstance(y, pd.DataFrame)
        if self.pooling == "local":
            if pd_format is True and isinstance(y, pd.MultiIndex):
                warn(
                    "Pooling is by default 'local', which"
                    + " means that separate models will be fit at the level of"
                    + " each instance. If you wish to fit a single model to"
                    + " all instances, please specify pooling = 'global'.",
                    obj=self,
                )
        self.window_length_ = check_window_length(
            self.window_length, n_timepoints=len(y)
        )
        if self.transformers is not None:
            self.transformers_ = clone(self.transformers)

        if self.transformers is None and self.pooling == "global":
            kwargs = {
                "lag_feature": {
                    "lag": list(range(1, self.window_length + 1)),
                }
            }
            self.transformers_ = [WindowSummarizer(**kwargs, n_jobs=1)]

        if self.window_length is None:
            trafo = self.transformers_
            fit_trafo = [i.fit(y) for i in trafo]
            ts = [i.truncate_start for i in fit_trafo if hasattr(i, "truncate_start")]

            if len(ts) > 0:
                self.window_length_ = max(ts)
            else:
                raise ValueError(
                    "Reduce must either have window length as argument"
                    + "or needs to have it passed by transformer via"
                    + "truncate_start"
                )

            if self.transformers_ is not None and n_timepoints < max(ts):
                raise ValueError(
                    "Not sufficient observations to calculate transformations"
                    + "Please reduce window length / window lagging to match"
                    + "observation size"
                )

        yt, Xt = self._transform(y, X)
        if hasattr(Xt, "columns"):
            Xt.columns = Xt.columns.astype(str)

        # Iterate over forecasting horizon, fitting a separate estimator for each step.
        self.estimators_ = []
        for i in range(len(self.fh)):
            fh_rel = fh.to_relative(self._cutoff_scalar())
            estimator = clone(self.estimator)

            if self.transformers_ is not None:
                fh_rel = fh.to_relative(self._cutoff_scalar())
                Xt_cut = _cut_df(Xt, n_timepoints - fh_rel[i] + 1, type="head")
                yt_cut = _cut_df(yt, n_timepoints - fh_rel[i] + 1)
            elif self.windows_identical is True or (fh_rel[i] - 1) == 0:
                Xt_cut = Xt
                yt_cut = yt[:, i]
            else:
                Xt_cut = Xt[: -(fh_rel[i] - 1)]
                yt_cut = yt[: -(fh_rel[i] - 1), i]

            # coercion to pandas for skpro proba regressors
            if self._est_type != "regressor" and not isinstance(Xt, pd.DataFrame):
                Xt_cut = pd.DataFrame(Xt_cut)
            if self._est_type != "regressor" and not isinstance(yt, pd.DataFrame):
                yt_cut = pd.DataFrame(yt_cut)

            estimator.fit(Xt_cut, yt_cut)
            self.estimators_.append(estimator)
        return self

    def _predict_last_window(self, fh, X=None, **kwargs):
        """.

        In recursive reduction, iteration must be done over the
        entire forecasting horizon. Specifically, when transformers are
        applied to y that generate features in X, forecasting must be done step by
        step to integrate the latest prediction of for the new set of features in
        X derived from that y.

        Parameters
        ----------
        fh : int, list, np.array or ForecastingHorizon
            Forecasting horizon
        X : pd.DataFrame, optional (default=None)
            Exogenous time series

        Returns
        -------
        y_return = pd.Series or pd.DataFrame
        """
        if "method" in kwargs:
            method = kwargs.pop("method")
        else:
            method = "predict"

        # estimator type for case branches
        est_type = self._est_type
        # "regressor" for sklearn, "regressor_proba" for skpro

        if self._X is not None and X is None:
            raise ValueError(
                "`X` must be passed to `predict` if `X` is given in `fit`."
            )

        if self.pooling == "global":
            y_last, X_last = self._get_shifted_window(X_update=X)
            ys = np.array(y_last)
            if not np.sum(np.isnan(ys)) == 0 and np.sum(np.isinf(ys)) == 0:
                return self._predict_nan(fh, method=method, **kwargs)
        else:
            y_last, X_last = self._get_last_window()
            if not self._is_predictable(y_last):
                return self._predict_nan(fh, method=method, **kwargs)
        # Get last window of available data.
        # If we cannot generate a prediction from the available data, return nan.

        if isinstance(X_last, pd.DataFrame):
            X_last = prep_skl_df(X_last)

        def pool_preds(y_preds):
            """Pool predictions from different estimators.

            Parameters
            ----------
            y_preds : list of pd.DataFrame
                List of predictions from different estimators.
            """
            y_pred = y_preds.pop(0)
            for y_pred_i in y_preds:
                y_pred = y_pred.combine_first(y_pred_i)
            return y_pred

        def _coerce_to_numpy(y_pred):
            """Coerce predictions to numpy array, assumes pd.DataFrame or numpy."""
            if isinstance(y_pred, pd.DataFrame):
                return y_pred.values
            else:
                return y_pred

        if self.pooling == "global":
            fh_abs = fh.to_absolute_index(self._cutoff_scalar())
            y_preds = []
            for i, estimator in enumerate(self.estimators_):
                y_pred_est = getattr(estimator, method)(X_last, **kwargs)

                # --- normalize and slice to the current horizon i (panel/global) ---
                y_arr = _coerce_to_numpy(y_pred_est)  # 1D or 2D -> numpy
                if y_arr.ndim == 1:
                    y_arr = y_arr.reshape(-1, 1)

                # number of series in the panel (last level is time)
                n_series = (
                    len(self._y.index.droplevel(-1).unique())
                    if isinstance(self._y.index, pd.MultiIndex)
                    else 1
                )

                # some estimators may return (#series * #fh, n_targets)
                # even in direct mode; # in that case, slice out the
                # block for the current horizon i
                if y_arr.shape[0] == n_series * len(self.estimators_):
                    y_arr = y_arr[i * n_series : (i + 1) * n_series]

                # build columns for quantiles output: (var_name, alpha) per column
                alphas = kwargs.get("alpha", [])
                # infer var names from “last window” y (single-step frame)
                varnames = list(
                    getattr(y_last, "columns", getattr(self._y, "columns", ["var_0"]))
                )

                if alphas:
                    # MultiIndex with (variable, alpha)
                    # matches width len(varnames) * len(alphas)
                    qcols = pd.MultiIndex.from_product([varnames, alphas])
                else:
                    # fallback: just use variable names (e.g., predict, not quantiles)
                    qcols = varnames

                # --- build the one-step frame for fh_abs[i] ---
                # y_pred_i = _create_fcst_df([fh_abs[i]], self._y, fill=y_arr)
                y_pred_i = _create_fcst_df(
                    [fh_abs[i]], self._y, fill=y_arr, columns=qcols
                )

                y_preds.append(y_pred_i)
            y_pred = pool_preds(y_preds)
        else:
            # Pre-allocate arrays.
            if self._X is None:
                n_columns = 1
            else:
                # X is ignored here, since we currently only look at lagged values for
                # exogenous variables and not contemporaneous ones.
                n_columns = self._X.shape[1] + 1

            # Pre-allocate arrays.
            window_length = self.window_length_
            X_pred = np.zeros((1, n_columns, window_length))

            # Fill pre-allocated arrays with available data.
            X_pred[:, 0, :] = y_last
            if self._X is not None:
                X_pred[:, 1:, :] = X_last.T

            # We need to make sure that X has the same order as used in fit.
            if self._estimator_scitype == "tabular-regressor":
                X_pred = X_pred.reshape(1, -1)

            # Allocate array for predictions.
            if est_type == "regressor":
                y_pred = np.zeros(len(fh))
            else:  # est_type == "regressor_proba"
                y_preds = []

            # Iterate over estimators/forecast horizon
            for i, estimator in enumerate(self.estimators_):
                y_pred_est = getattr(estimator, method)(X_pred, **kwargs)
                if est_type == "regressor":
                    y_pred[i] = y_pred_est[0]
                else:  # est_type == "regressor_proba"
                    y_pred_v = _coerce_to_numpy(y_pred_est)
                    y_pred_i = _create_fcst_df([fh[i]], y_pred_est, fill=y_pred_v)
                    y_preds.append(y_pred_i)

            if est_type != "regressor":
                y_pred = pool_preds(y_preds)

        # coerce index and columns to expected
        # index = fh.get_expected_pred_idx(y=self._y, cutoff=self.cutoff)
        index = self._get_expected_pred_idx(fh)
        columns = self._get_columns(method=method, **kwargs)
        if isinstance(y_pred, pd.DataFrame):
            y_pred.index = index
            y_pred.columns = columns
        else:
            y_pred = pd.DataFrame(y_pred, index=index, columns=columns)

        return y_pred


class _MultioutputReducer(_Reducer):
    strategy = "multioutput"
    _tags = {
        "requires-fh-in-fit": True,  # is the forecasting horizon required in fit?
    }

    def _transform(self, y, X=None):
        fh = self.fh.to_relative(self._cutoff_scalar())
        return _sliding_window_transform(
            y,
            window_length=self.window_length,
            fh=fh,
            X=X,
            scitype=self._estimator_scitype,
        )

    def _fit(self, y, X, fh):
        """Fit to training data.

        Parameters
        ----------
        y : pd.Series
            Target time series to which to fit the forecaster.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored
        fh : int, list or np.array, optional (default=None)
             The forecasters horizon with the steps ahead to to predict.

        Returns
        -------
        self : returns an instance of self.
        """
        # We currently only support out-of-sample predictions. For the direct
        # strategy, we need to check this at the beginning of fit, as the fh is
        # required for fitting.
        self.window_length_ = check_window_length(
            self.window_length, n_timepoints=len(y)
        )

        yt, Xt = self._transform(y, X)

        # Fit a multi-output estimator to the transformed data.
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(Xt, yt)
        return self

    def _predict_last_window(self, fh, X=None, **kwargs):
        """Predict to training data.

        Parameters
        ----------
        fh : int, list, np.array or ForecastingHorizon
            Forecasting horizon
        X : pd.DataFrame, optional (default=None)
            Exogenous time series

        Returns
        -------
        y_pred = pd.Series or pd.DataFrame
        """
        # Get last window of available data.
        y_last, X_last = self._get_last_window()

        # If we cannot generate a prediction from the available data, return nan.
        if not self._is_predictable(y_last):
            return self._predict_nan(fh)

        if self._X is None:
            n_columns = 1
        else:
            # X is ignored here, since we currently only look at lagged values for
            # exogenous variables and not contemporaneous ones.
            n_columns = self._X.shape[1] + 1

        # Pre-allocate arrays.
        window_length = self.window_length_
        X_pred = np.zeros((1, n_columns, window_length))

        # Fill pre-allocated arrays with available data.
        X_pred[:, 0, :] = y_last
        if self._X is not None:
            X_pred[:, 1:, :] = X_last.T

        # We need to make sure that X has the same order as used in fit.
        if self._estimator_scitype == "tabular-regressor":
            X_pred = X_pred.reshape(1, -1)

        # Iterate over estimators/forecast horizon
        y_pred = self.estimator_.predict(X_pred)
        return y_pred.ravel()


class _RecursiveReducer(_Reducer):
    strategy = "recursive"

    def _transform(self, y, X=None):
        # For the recursive strategy, the forecasting horizon for the sliding-window
        # transform is simply a one-step ahead horizon, regardless of the horizon
        # used during prediction.
        fh = ForecastingHorizon([1])
        return _sliding_window_transform(
            y,
            self.window_length_,
            fh,
            X=X,
            transformers=self.transformers_,
            scitype=self._estimator_scitype,
            pooling=self.pooling,
        )

    def _fit(self, y, X, fh):
        """Fit to training data.

        Parameters
        ----------
        y : pd.Series
            Target time series to which to fit the forecaster.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored
        fh : int, list or np.array, optional (default=None)
             The forecasters horizon with the steps ahead to to predict.

        Returns
        -------
        self : returns an instance of self.
        """
        if self.pooling is not None and self.pooling not in ["local", "global"]:
            raise ValueError(
                "pooling must be one of local, global" + f" but found {self.pooling}"
            )

        if self.window_length is not None and self.transformers is not None:
            raise ValueError(
                "Transformers provided, suggesting en-bloc approach"
                + " to derive reduction features. Window length will be"
                + " inferred, please set to None"
            )
        if self.transformers is not None and self.pooling == "local":
            raise ValueError(
                "Transformers currently cannot be provided"
                + "for models that run locally"
            )

        pd_format = isinstance(y, pd.Series) or isinstance(y, pd.DataFrame)

        self._timepoints = get_time_index(y)
        n_timepoints = len(self._timepoints)

        self.window_length_ = check_window_length(
            self.window_length, n_timepoints=n_timepoints
        )

        if self.pooling == "local":
            if pd_format is True and isinstance(y, pd.MultiIndex):
                warn(
                    "Pooling is by default 'local', which"
                    + " means that separate models will be fit at the level of"
                    + " each instance. If you wish to fit a single model to"
                    + " all instances, please specify pooling = 'global'.",
                    obj=self,
                )
        if self.transformers is not None:
            self.transformers_ = clone(self.transformers)

        if self.transformers is None and self.pooling == "global":
            kwargs = {
                "lag_feature": {
                    "lag": list(range(1, self.window_length + 1)),
                }
            }
            self.transformers_ = [WindowSummarizer(**kwargs, n_jobs=1)]

        if self.window_length is None:
            trafo = self.transformers_
            fit_trafo = [i.fit(y) for i in trafo]
            ts = [i.truncate_start for i in fit_trafo if hasattr(i, "truncate_start")]

            if len(ts) > 0:
                self.window_length_ = max(ts)
            else:
                raise ValueError(
                    "Reduce must either have window length as argument"
                    + "or needs to have it passed by transformer via"
                    + "truncate_start"
                )

            if self.transformers_ is not None and n_timepoints < max(ts):
                raise ValueError(
                    "Not sufficient observations to calculate transformations"
                    + "Please reduce window length / window lagging to match"
                    + "observation size"
                )

        yt, Xt = self._transform(y, X)

        # Make sure yt is 1d array to avoid DataConversion warning from scikit-learn.
        if self.transformers_ is not None:
            yt = yt.to_numpy().ravel()
        else:
            yt = yt.ravel()

        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(Xt, yt)
        return self

    def _predict_last_window(self, fh, X=None, **kwargs):
        """.

        In recursive reduction, iteration must be done over the
        entire forecasting horizon. Specifically, when transformers are
        applied to y that generate features in X, forecasting must be done step by
        step to integrate the latest prediction of for the new set of features in
        X derived from that y.

        Parameters
        ----------
        fh : int, list, np.array or ForecastingHorizon
            Forecasting horizon
        X : pd.DataFrame, optional (default=None)
            Exogenous time series
        return_pred_int : bool
        alpha : float or array-like

        Returns
        -------
        y_return = pd.Series or pd.DataFrame
        """
        if self._X is not None and X is None:
            raise ValueError(
                "`X` must be passed to `predict` if `X` is given in `fit`."
            )

        # Get last window of available data.
        # If we cannot generate a prediction from the available data, return nan.

        if self.pooling == "global":
            y_last, X_last = self._get_shifted_window(X_update=X)
            ys = np.array(y_last)
            if not np.sum(np.isnan(ys)) == 0 and np.sum(np.isinf(ys)) == 0:
                return self._predict_nan(fh)
        else:
            y_last, X_last = self._get_last_window()
            if not self._is_predictable(y_last):
                return self._predict_nan(fh)

        if self.pooling == "global":
            fh_max = fh.to_relative(self._cutoff_scalar())[-1]
            relative = pd.Index(list(map(int, range(1, fh_max + 1))))

            # Build a 1-element cutoff index that carries/inherits freq, if possible
            c = self._cutoff_scalar()
            if isinstance(c, pd.Period):
                cutoff_idx = pd.PeriodIndex([c], freq=c.freq)
            elif isinstance(c, pd.Timestamp):
                # Try to inherit or infer a real freq for the 1-elem cutoff index.
                freq = None
                y_idx = getattr(self, "_y", None)
                y_idx = getattr(y_idx, "index", None)

                if isinstance(y_idx, pd.MultiIndex):
                    # infer from one series (avoid duplicate times in pooled level)
                    try:
                        keys = y_idx.droplevel(-1).unique()
                        first_key = (
                            keys[0] if not isinstance(keys, pd.MultiIndex) else keys[0]
                        )
                        ser_idx = self._y.loc[first_key].index
                        freq = getattr(ser_idx, "freq", None) or pd.infer_freq(ser_idx)
                    except Exception:
                        freq = None
                else:
                    # single-series case: infer directly
                    if y_idx is not None:
                        freq = getattr(y_idx, "freq", None)
                        if freq is None:
                            try:
                                freq = pd.infer_freq(y_idx)
                            except Exception:
                                freq = None

                cutoff_idx = (
                    pd.date_range(start=c, periods=1, freq=freq, tz=c.tz)
                    if freq is not None
                    else pd.DatetimeIndex([c], tz=c.tz)
                )
            elif isinstance(c, (pd.PeriodIndex, pd.DatetimeIndex, pd.Index)):
                cutoff_idx = c[:1]
            else:
                cutoff_idx = pd.Index([c])

            index_range = _index_range(relative, cutoff_idx)

            y_pred = _create_fcst_df(index_range, self._y)

            for i in range(fh_max):
                # Generate predictions.
                y_pred_vector = self.estimator_.predict(X_last)
                y_pred_curr = _create_fcst_df(
                    [index_range[i]], self._y, fill=y_pred_vector
                )
                y_pred.update(y_pred_curr)

                # # Update last window with previous prediction.
                if i + 1 != fh_max:
                    y_last, X_last = self._get_shifted_window(
                        y_update=y_pred, X_update=X, shift=i + 1
                    )
        else:
            # Pre-allocate arrays.
            if X is None:
                n_columns = 1
            else:
                n_columns = X.shape[1] + 1
            window_length = self.window_length_
            fh_max = fh.to_relative(self._cutoff_scalar())[-1]

            y_pred = np.zeros(fh_max)

            # Array with input data for prediction.
            last = np.zeros((1, n_columns, window_length + fh_max))

            # Fill pre-allocated arrays with available data.
            last[:, 0, :window_length] = y_last
            if X is not None:
                X_to_use = np.concatenate(
                    [X_last.T, X.iloc[-(last.shape[2] - window_length) :, :].T], axis=1
                )
                if X_to_use.shape[1] < window_length + fh_max:
                    X_to_use = np.pad(
                        X_to_use,
                        ((0, 0), (0, window_length + fh_max - X_to_use.shape[1])),
                        "edge",
                    )
                elif X_to_use.shape[1] > window_length + fh_max:
                    X_to_use = X_to_use[:, : window_length + fh_max]
                # else X_to_use.shape[1] == window_length + fh_max
                # and there are no additional steps to take
                last[:, 1:] = X_to_use

            # Recursively generate predictions by iterating over forecasting horizon.
            for i in range(fh_max):
                # Slice prediction window.
                X_pred = last[:, :, i : window_length + i]

                # Reshape data into tabular array.
                if self._estimator_scitype == "tabular-regressor":
                    X_pred = X_pred.reshape(1, -1)

                # Generate predictions.
                y_pred[i] = self.estimator_.predict(X_pred)[0]

                # Update last window with previous prediction.
                last[:, 0, window_length + i] = y_pred[i]

        # While the recursive strategy requires to generate predictions for all steps
        # until the furthest step in the forecasting horizon, we only return the
        # requested ones.
        fh_idx = fh.to_indexer(self.cutoff)

        if isinstance(self._y.index, pd.MultiIndex):
            yi_grp = self._y.index.names[0:-1]
            y_return = y_pred.groupby(yi_grp, as_index=False).nth(fh_idx.to_list())
        elif isinstance(y_pred, pd.Series) or isinstance(y_pred, pd.DataFrame):
            y_return = y_pred.iloc[fh_idx]
            if hasattr(y_return.index, "freq"):
                if y_return.index.freq != y_pred.index.freq:
                    y_return.index.freq = None
        else:
            y_return = y_pred[fh_idx]

        return y_return


class _DirRecReducer(_Reducer):
    strategy = "dirrec"
    _tags = {
        "requires-fh-in-fit": True,  # is the forecasting horizon required in fit?
        "capability:exogenous": False,
    }

    def _transform(self, y, X=None):
        # Note that the transform for dirrec is the same as in the direct
        # strategy.
        fh = self.fh.to_relative(self._cutoff_scalar())
        return _sliding_window_transform(
            y,
            window_length=self.window_length,
            fh=fh,
            X=X,
            scitype=self._estimator_scitype,
        )

    def _fit(self, y, X, fh):
        """Fit to training data.

        Parameters
        ----------
        y : pd.Series
            Target time series to which to fit the forecaster.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored
        fh : int, list or np.array, optional (default=None)
             The forecasters horizon with the steps ahead to to predict.

        Returns
        -------
        self : Estimator
            An fitted instance of self.
        """
        # todo: logic for X below is broken. Escape X until fixed.
        if X is not None:
            X = None

        self.window_length_ = check_window_length(
            self.window_length, n_timepoints=len(y)
        )

        # Transform the data using sliding-window.
        yt, Xt = self._transform(y, X)

        # We cast the 2d tabular array into a 3d panel array to handle the data
        # consistently for the reduction to tabular and time-series regression.
        if self._estimator_scitype == "tabular-regressor":
            Xt = np.expand_dims(Xt, axis=1)

        # This only works without exogenous variables. To support exogenous
        # variables, we need additional values for X to fill the array
        # appropriately.
        X_full = np.concatenate([Xt, np.expand_dims(yt, axis=1)], axis=2)

        self.estimators_ = []
        n_timepoints = Xt.shape[2]

        for i in range(len(self.fh)):
            estimator = clone(self.estimator)

            # Slice data using expanding window.
            X_fit = X_full[:, :, : n_timepoints + i]

            # Convert to 2d tabular array for reduction to tabular regression.
            if self._estimator_scitype == "tabular-regressor":
                X_fit = X_fit.reshape(X_fit.shape[0], -1)

            estimator.fit(X_fit, yt[:, i])
            self.estimators_.append(estimator)
        return self

    def _predict_last_window(self, fh, X=None, **kwargs):
        """Fit to training data.

        Parameters
        ----------
        fh : int, list, np.array or ForecastingHorizon
            Forecasting horizon
        X : pd.DataFrame, optional (default=None)
            Exogenous time series

        Returns
        -------
        y_pred = pd.Series or pd.DataFrame
        """
        # Exogenous variables are not yet support for the dirrec strategy.
        # todo: implement this. For now, we escape.
        if X is not None:
            X = None

        # Get last window of available data.
        y_last, X_last = self._get_last_window()
        if not self._is_predictable(y_last):
            return self._predict_nan(fh)

        window_length = self.window_length_

        # Pre-allocated arrays.
        # We set `n_columns` here to 1, because exogenous variables
        # are not yet supported.
        n_columns = 1
        X_full = np.zeros((1, n_columns, window_length + len(self.fh)))
        X_full[:, 0, :window_length] = y_last

        y_pred = np.zeros(len(fh))

        for i in range(len(self.fh)):
            # Slice data using expanding window.
            X_pred = X_full[:, :, : window_length + i]

            if self._estimator_scitype == "tabular-regressor":
                X_pred = X_pred.reshape(1, -1)

            y_pred[i] = self.estimators_[i].predict(X_pred)[0]

            # Update the last window with previously predicted value.
            X_full[:, :, window_length + i] = y_pred[i]

        return y_pred


def _asfreq_per_series_safe(y, freq, how="start"):
    """Like apply_method_per_series(..., 'asfreq', ...), but robust to string keys."""
    if not isinstance(y.index, pd.MultiIndex):
        return y.asfreq(freq, how=how)

    parts = []
    names = y.index.names
    outer = y.index.droplevel(-1).unique()

    for key in outer:
        ys = y.loc[key].asfreq(freq, how=how)
        # Reattach outer level(s) without exploding strings like "s1" -> ('s','1')
        if isinstance(key, tuple):
            idx = pd.MultiIndex.from_tuples([(*key, t) for t in ys.index], names=names)
        else:
            idx = pd.MultiIndex.from_product([[key], ys.index], names=names)
        ys = ys.copy()
        ys.index = idx
        parts.append(ys)

    return pd.concat(parts).sort_index()


class DirectTabularRegressionForecaster(_DirectReducer):
    """Direct reduction from forecasting to tabular regression.

    For the direct reduction strategy, a separate forecaster is fitted
    for each step ahead of the forecasting horizon.

    Parameters
    ----------
    estimator : Estimator
        A tabular regression estimator as provided by scikit-learn.
    window_length : int, optional (default=10)
        The length of the sliding window used to transform the series
        into a tabular matrix.
    """

    _tags = {
        **_DirectReducer._tags,  # inherit parent tags if it defines any
        "capability:exogenous": True,
        "capability:pred_int": True,
    }

    def __init__(
        self,
        estimator,
        window_length=10,
        transformers=None,
        pooling="local",
        windows_identical=True,
    ):
        super().__init__(
            estimator=estimator, window_length=window_length, transformers=transformers
        )
        self.pooling = pooling
        self.windows_identical = windows_identical

        if pooling == "local":
            mtypes_y = "pd.Series"
            mtypes_x = "pd.DataFrame"
        elif pooling == "global":
            mtypes_y = ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"]
            mtypes_x = mtypes_y
        elif pooling == "panel":
            mtypes_y = ["pd.DataFrame", "pd-multiindex"]
            mtypes_x = mtypes_y
        else:
            raise ValueError(
                "pooling in DirectReductionForecaster must be one of"
                ' "local", "global", "panel", '
                f"but found {pooling}"
            )
        self.set_tags(**{"X_inner_mtype": mtypes_x})
        self.set_tags(**{"y_inner_mtype": mtypes_y})

    _estimator_scitype = "tabular-regressor"


class MultioutputTabularRegressionForecaster(_MultioutputReducer):
    """Multioutput reduction from forecasting to tabular regression.

    For the multioutput strategy, a single estimator capable of handling multioutput
    targets is fitted to all the future steps in the forecasting horizon.

    Parameters
    ----------
    estimator : Estimator
        A tabular regression estimator as provided by scikit-learn.
    window_length : int, optional (default=10)
        The length of the sliding window used to transform the series into
        a tabular matrix.
    """

    _estimator_scitype = "tabular-regressor"


class RecursiveTabularRegressionForecaster(_RecursiveReducer):
    """Recursive reduction from forecasting to tabular regression.

    For the recursive strategy, a single estimator is fit for a one-step-ahead
    forecasting horizon and then called iteratively to predict multiple steps ahead.

    Parameters
    ----------
    estimator : Estimator
        A tabular regression estimator as provided by scikit-learn.
    window_length : int, optional (default=10)
        The length of the sliding window used to transform the series into
        a tabular matrix.
    transformers: list of transformers (default = None)
        A suitable list of transformers that allows for using an en-bloc approach with
        make_reduction. This means that instead of using the raw past observations of
        y across the window length, suitable features will be generated directly from
        the past raw observations. Currently only supports WindowSummarizer (or a list
        of WindowSummarizers) to generate features e.g. the mean of the past 7
        observations.
    pooling: str {"local", "global"}, optional
        Specifies whether separate models will be fit at the level of each instance
        (local) of if you wish to fit a single model to all instances ("global").
    """

    _tags = {
        "requires-fh-in-fit": False,  # is the forecasting horizon required in fit?
    }

    def __init__(
        self,
        estimator,
        window_length=10,
        transformers=None,
        pooling="local",
    ):
        super(_RecursiveReducer, self).__init__(
            estimator=estimator, window_length=window_length, transformers=transformers
        )
        self.pooling = pooling

        if pooling == "local":
            mtypes_y = "pd.Series"
            mtypes_x = "pd.DataFrame"
        elif pooling == "global":
            mtypes_y = ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"]
            mtypes_x = mtypes_y
        elif pooling == "panel":
            mtypes_y = ["pd.DataFrame", "pd-multiindex"]
            mtypes_x = mtypes_y
        else:
            raise ValueError(
                "pooling in DirectReductionForecaster must be one of"
                ' "local", "global", "panel", '
                f"but found {pooling}"
            )
        self.set_tags(**{"X_inner_mtype": mtypes_x})
        self.set_tags(**{"y_inner_mtype": mtypes_y})

    _estimator_scitype = "tabular-regressor"


class DirRecTabularRegressionForecaster(_DirRecReducer):
    """Dir-rec reduction from forecasting to tabular regression.

    For the hybrid dir-rec strategy, a separate forecaster is fitted
    for each step ahead of the forecasting horizon and then
    the previous forecasting horizon is added as an input
    for training the next forecaster, following the recursive
    strategy.

    Parameters
    ----------
    estimator : sklearn estimator object
        Tabular regressor.
    window_length : int, optional (default=10)
        The length of the sliding window used to transform the series into
        a tabular matrix
    """

    _estimator_scitype = "tabular-regressor"


class DirectTimeSeriesRegressionForecaster(_DirectReducer):
    """Direct reduction from forecasting to time-series regression.

    For the direct reduction strategy, a separate forecaster is fitted
    for each step ahead of the forecasting horizon.

    Parameters
    ----------
    estimator : Estimator
        A time-series regression estimator as provided by sktime.
    window_length : int, optional (default=10)
        The length of the sliding window used to transform the series into
        a tabular matrix.
    """

    _estimator_scitype = "time-series-regressor"


class MultioutputTimeSeriesRegressionForecaster(_MultioutputReducer):
    """Multioutput reduction from forecasting to time series regression.

    For the multioutput strategy, a single estimator capable of handling multioutput
    targets is fitted to all the future steps in the forecasting horizon.

    Parameters
    ----------
    estimator : Estimator
        A time-series regression estimator as provided by sktime.
    window_length : int, optional (default=10)
        The length of the sliding window used to transform the series into
        a tabular matrix.
    """

    _estimator_scitype = "time-series-regressor"


class RecursiveTimeSeriesRegressionForecaster(_RecursiveReducer):
    """Recursive reduction from forecasting to time series regression.

    For the recursive strategy, a single estimator is fit for a one-step-ahead
    forecasting horizon and then called iteratively to predict multiple steps ahead.

    Parameters
    ----------
    estimator : Estimator
        A time-series regression estimator as provided by sktime.
    window_length : int, optional (default=10)
        The length of the sliding window used to transform the series into
        a tabular matrix.
    """

    _tags = {
        "requires-fh-in-fit": False,  # is the forecasting horizon required in fit?
    }

    _estimator_scitype = "time-series-regressor"


class DirRecTimeSeriesRegressionForecaster(_DirRecReducer):
    """Dir-rec reduction from forecasting to time-series regression.

    For the hybrid dir-rec strategy, a separate forecaster is fitted
    for each step ahead of the forecasting horizon and then
    the previous forecasting horizon is added as an input
    for training the next forecaster, following the recursive
    strategy.

    Parameters
    ----------
    estimator : sktime estimator object
        Time-series regressor.
    window_length : int, optional (default=10)
        The length of the sliding window used to transform the series into
        a tabular matrix
    """

    _estimator_scitype = "time-series-regressor"


def make_reduction(
    estimator,
    strategy="recursive",
    window_length=10,
    scitype="infer",
    transformers=None,
    pooling="local",
    windows_identical=True,
):
    r"""Make forecaster based on reduction to tabular or time-series regression.

    During fitting, a sliding-window approach is used to first transform the
    time series into tabular or panel data, which is then used to fit a tabular or
    time-series regression estimator. During prediction, the last available data is
    used as input to the fitted regression estimator to generate forecasts.

    Please see below a graphical representation of the make_reduction logic using the
    following symbols:

    - ``y`` = forecast target.
    - ``x`` = past values of y that are used as features (X) to forecast y
    - ``*`` = observations, past or future, neither part of window nor forecast.

    Assume we have the following training data (14 observations)::

    |----------------------------|
    | * * * * * * * * * * * * * *|
    |----------------------------|

    And want to forecast with ``window_length = 9`` and ``fh = [2, 4]``.

    By construction, a recursive reducer always targets the first data point after
    the window, irrespective of the forecasting horizons requested.
    In the example the following 5 windows are created::

    |----------------------------|
    | x x x x x x x x x y * * * *|
    | * x x x x x x x x x y * * *|
    | * * x x x x x x x x x y * *|
    | * * * x x x x x x x x x y *|
    | * * * * x x x x x x x x x y|
    |----------------------------|

    Direct Reducers will create multiple models, one for each forecasting horizon.
    With the argument ``windows_identical = True`` (default) the windows used to train
    the model are defined by the maximum forecasting horizon.
    Only two complete windows can be defined in this example
    ``fh = 4`` (maximum of ``fh = [2, 4]``)::

    |----------------------------|
    | x x x x x x x x x * * * y *|
    | * x x x x x x x x x * * * y|
    |----------------------------|

    All other forecasting horizons will also use those two (maximal) windows.
    ``fh = 2``::

    |----------------------------|
    | x x x x x x x x x * y * * *|
    | * x x x x x x x x x * y * *|
    |----------------------------|

    With ``windows_identical = False`` we drop the requirement to use the same windows
    for each of the direct models, so more windows can be created for horizons other
    than the maximum forecasting horizon.
    ``fh = 2``::

    |----------------------------|
    | x x x x x x x x x * y * * *|
    | * x x x x x x x x x * y * *|
    | * * x x x x x x x x x * y *|
    | * * * x x x x x x x x x * y|
    |----------------------------|

    ``fh = 4``::

    |----------------------------|
    | x x x x x x x x x * * * y *|
    | * x x x x x x x x x * * * y|
    |----------------------------|

    Use ``windows_identical = True`` if you want to compare the forecasting
    performance across different horizons, since all models trained will use the
    same windows. Use ``windows_identical = False`` if you want to have the highest
    forecasting accuracy for each forecasting horizon.

    Parameters
    ----------
    estimator : an estimator instance, can be:

        * scikit-learn regressor or interface compatible
        * sktime time series regressor
        * skpro tabular probabilistic supervised regressor, only for direct reduction
          this will result in a probabilistic forecaster

    strategy : str, optional (default="recursive")
        The strategy to generate forecasts. Must be one of "direct", "recursive" or
        "multioutput".

    window_length : int, optional (default=10)
        Window length used in sliding window transformation.

    scitype : str, optional (default="infer")
        Legacy argument for downwards compatibility, should not be used.
        ``make_reduction`` will automatically infer the correct type of ``estimator``.
        This internal inference can be force-overridden by the ``scitype`` argument.
        Must be one of "infer", "tabular-regressor" or "time-series-regressor".
        If the scitype cannot be inferred, this is a bug and should be reported.

    transformers: list of transformers (default = None)
        A suitable list of transformers that allows for using an en-bloc approach with
        make_reduction. This means that instead of using the raw past observations of
        y across the window length, suitable features will be generated directly from
        the past raw observations. Currently only supports WindowSummarizer (or a list
        of WindowSummarizers) to generate features e.g. the mean of the past 7
        observations. Currently only works for RecursiveTimeSeriesRegressionForecaster.

    pooling: str {"local", "global"}, optional
        Specifies whether separate models will be fit at the level of each instance
        (local) of if you wish to fit a single model to all instances ("global").
        Currently only works for RecursiveTimeSeriesRegressionForecaster.

    windows_identical: bool, (default = True)
        Direct forecasting only.
        Specifies whether all direct models use the same X windows from y (True: Number
        of windows = total observations + 1 - window_length - maximum forecasting
        horizon) or a different number of X windows depending on the forecasting horizon
        (False: Number of windows = total observations + 1 - window_length
        - forecasting horizon). See pictionary below for more information.

    Returns
    -------
    forecaster : an sktime forecaster object
        the reduction forecaster, wrapping ``estimator``
        class is determined by the ``strategy`` argument and type of ``estimator``.

    Examples
    --------
    >>> from sktime.forecasting.compose import make_reduction
    >>> from sktime.datasets import load_airline
    >>> from sklearn.ensemble import GradientBoostingRegressor
    >>> y = load_airline()
    >>> regressor = GradientBoostingRegressor()
    >>> forecaster = make_reduction(regressor, window_length=15, strategy="recursive")
    >>> forecaster.fit(y)
    RecursiveTabularRegressionForecaster(...)
    >>> y_pred = forecaster.predict(fh=[1,2,3])

    References
    ----------
    .. [1] Bontempi, Gianluca & Ben Taieb, Souhaib & Le Borgne, Yann-Aël. (2013).
        Machine Learning Strategies for Time Series Forecasting.
    """
    # We provide this function as a factory method for user convenience.
    strategy = _check_strategy(strategy)
    scitype = _check_scitype(scitype)

    if scitype == "infer":
        scitype = _infer_scitype(estimator)

    Forecaster = _get_forecaster(scitype, strategy)

    dispatch_params = {
        "estimator": estimator,
        "window_length": window_length,
        "transformers": transformers,
        "pooling": pooling,
        "windows_identical": windows_identical,
    }

    return construct_dispatch(Forecaster, dispatch_params)


def _check_scitype(scitype):
    valid_scitypes = ("infer", "tabular-regressor", "time-series-regressor")
    if scitype not in valid_scitypes:
        raise ValueError(
            f"Invalid `scitype`. `scitype` must be one of:"
            f" {valid_scitypes}, but found: {scitype}."
        )
    return scitype


def _infer_scitype(estimator):
    """Infer scitype from estimator.

    Returns
    -------
    scitype : str
        The inferred scitype of the estimator.

        * if sklearn estimator, returns tabular-regressor etc, one of the returns
          of sklearn_scitype prefixed with "tabular-".
        * if sktime/skpro or skbase estimator, returns the scitype of the estimator
          as found in the object_type tag.
        * if none of the above applies, returns "tabular-regressor" as fallback default.
    """
    if is_sklearn_estimator(estimator):
        return f"tabular-{sklearn_scitype(estimator)}"
    else:
        if is_scitype(estimator, ["object", "estimator"]):
            return "tabular-regressor"
        if is_scitype(estimator, "regressor"):
            return "time-series-regressor"
        else:
            return scitype(estimator, raise_on_unknown=False)


def _check_strategy(strategy):
    valid_strategies = ("direct", "recursive", "multioutput", "dirrec")
    if strategy not in valid_strategies:
        raise ValueError(
            f"Invalid `strategy`. `strategy` must be one of :"
            f" {valid_strategies}, but found: {strategy}."
        )
    return strategy


def _get_forecaster(scitype, strategy):
    """Select forecaster for a given scientific type and reduction strategy."""
    registry = {
        "tabular-regressor": {
            "direct": DirectTabularRegressionForecaster,
            "recursive": RecursiveTabularRegressionForecaster,
            "multioutput": MultioutputTabularRegressionForecaster,
            "dirrec": DirRecTabularRegressionForecaster,
        },
        "time-series-regressor": {
            "direct": DirectTimeSeriesRegressionForecaster,
            "recursive": RecursiveTimeSeriesRegressionForecaster,
            "multioutput": MultioutputTimeSeriesRegressionForecaster,
            "dirrec": DirRecTimeSeriesRegressionForecaster,
        },
        "regressor_proba": {"direct": DirectTabularRegressionForecaster},
    }

    if scitype not in registry:
        raise ValueError(
            "Error in make_reduction, no reduction strategies defined for "
            f"specified or inferred scitype of estimator: {scitype}. "
            f"Valid scitypes are: {list(registry.keys())}."
        )
    if strategy not in registry[scitype]:
        raise ValueError(
            f"Error in make_reduction, strategy {strategy} not defined for "
            f"specified or inferred scitype {scitype}. "
            f"Valid strategies are: {list(registry[scitype].keys())}."
        )
    return registry[scitype][strategy]


def _cut_df(X, n_obs=1, type="tail"):
    """Cut input at tail or head, supports grouping."""
    if n_obs == 0:
        return X.copy()
    if isinstance(X.index, pd.MultiIndex):
        levels = list(range(X.index.nlevels - 1))
        if type == "tail":
            X = X.groupby(level=levels, as_index=False).tail(n_obs)
        elif type == "head":
            X = X.groupby(level=levels, as_index=False).head(n_obs)
    else:
        if type == "tail":
            X = X.tail(n_obs)
        elif type == "head":
            X = X.head(n_obs)
    return X


# def _create_fcst_df(target_date, origin_df, fill=None):
#     """Create an empty multiindex dataframe from origin dataframe.

#     In recursive forecasting, a new dataframe needs to be created that collects all
#     forecasting steps (even for forecasting horizons other than those of interests).
#     For example for fh =[1,2,12] we need the whole forecasting horizons from 1 to 12.

#     Parameters
#     ----------
#     target_date : a list of dates
#         this will be correspond to the new timepoints index to be created in the
#         forecasting dataframe
#     origin_df : a pandas Series or Dataframe
#         the origin_df corresponds to the dataframe with the historic data. Useful
#         information inferred from that df is the index of the historic df
#         as well as the names of the original columns and the type of the object
#         (dataframe or series)
#     fill : a numpy.ndarray (optional)
#         Corresponds to a numpy array of values that is used to fill up the dataframe.
#         Useful when forecasts are returned from a forecasting models that discards
#         the hierarchical structure of the input pandas dataframe

#     Returns
#     -------
#     A pandas dataframe or series
#     """
#     if not isinstance(target_date, ForecastingHorizon):
#         ix = pd.Index(target_date)
#         fh = ForecastingHorizon(ix, is_relative=False)
#     else:
#         fh = target_date.to_absolute()

#     index = fh.get_expected_pred_idx(origin_df)

#     if isinstance(origin_df, pd.Series):
#         columns = [origin_df.name]
#     else:
#         columns = origin_df.columns.to_list()

#     if fill is None:
#         values = 0
#     else:
#         values = fill

#     res = pd.DataFrame(values, index=index, columns=columns, dtype="float64")

#     if isinstance(origin_df, pd.Series) and not isinstance(index, pd.MultiIndex):
#         res = res.iloc[:, 0]
#         res.name = origin_df.name

#     return res


# def _create_fcst_df(target_date, origin_df, fill=None, columns=None):
#     """Create an empty forecasting frame aligned to origin_df's index structure.

#     Parameters
#     ----------
#     target_date : iterable of dates or ForecastingHorizon (abs or already resolvable)
#         New timepoints for the forecast frame (last level of the index).
#     origin_df : pd.Series or pd.DataFrame
#         Provides the original index structure (including outer levels & names)
#         and the column names (for DataFrame) or name (for Series).
#     fill : scalar or array-like, optional
#         If provided, pre-fill the frame with this value; otherwise zeros.

#     Returns
#     -------
#     pd.Series or pd.DataFrame
#         With the same outer index levels and column structure as origin_df, and
#         the last level replaced by target_date.
#     """
#     # Normalize target_date to a pandas Index
#     if isinstance(target_date, ForecastingHorizon):
#         # Try to treat it as already absolute; fall back to a plain Index if needed
#         try:
#             td = target_date.to_absolute()
#             tgt = td.to_pandas() if hasattr(td, "to_pandas") else pd.Index(td)
#         except TypeError:
#             tgt = pd.Index(target_date)
#     else:
#         tgt = pd.Index(target_date)

#     idx0 = origin_df.index
#     # Build the forecast index to mirror origin_df's structure
#     if isinstance(idx0, pd.MultiIndex):
#         left = idx0.droplevel(-1).unique()
#         names = idx0.names

#         if isinstance(left, pd.MultiIndex):
#             tuples = [(*lvl, t) for lvl in left for t in tgt]
#             index = pd.MultiIndex.from_tuples(tuples, names=names)
#         else:
#             index = pd.MultiIndex.from_product([left, tgt], names=names)
#     else:
#         # Single-level time index: preserve dtype where possible
#         if isinstance(idx0, pd.PeriodIndex):
#             try:
#                 tgt = pd.PeriodIndex(tgt, freq=idx0.freq)
#             except Exception:
#                 tgt = pd.Index(tgt)
#         elif isinstance(idx0, pd.DatetimeIndex):
#             try:
#                 tgt = pd.DatetimeIndex(tgt, tz=idx0.tz)
#             except Exception:
#                 tgt = pd.Index(tgt)
#         index = tgt
#         # carry over name if present
#         if getattr(idx0, "name", None) is not None:
#             index.name = idx0.name

#     # Columns / values
#     if columns is None:
#         columns = (
#             [origin_df.name]
#             if isinstance(origin_df, pd.Series)
#             else list(origin_df.columns)
#         )
#     values = 0 if fill is None else fill

#     res = pd.DataFrame(values, index=index, columns=columns, dtype="float64")

#     # If the origin was a Series and the result isn't hierarchical, return a Series
#     if isinstance(origin_df, pd.Series) and not isinstance(index, pd.MultiIndex):
#         res = res.iloc[:, 0]
#         res.name = origin_df.name

#     return res


def _create_fcst_df(target_date, origin_df, fill=None, columns=None):
    """Create an empty forecasting frame aligned to origin_df's index structure.

    Parameters
    ----------
    target_date : iterable of dates, MultiIndex, or ForecastingHorizon
        New timepoints for the forecast frame. Can be:
        * time-only index (usual case), or
        * a full MultiIndex matching origin_df.index (e.g., (series, time)), or
        * a ForecastingHorizon that can be resolved to absolute times.
    origin_df : pd.Series or pd.DataFrame
        Provides the original index structure (including outer levels & names)
        and the column names (for DataFrame) or name (for Series).
    fill : scalar or array-like, optional
        If provided, pre-fill the frame with this value; otherwise zeros.
    columns : sequence, optional
        Column names to use instead of origin_df's.

    Returns
    -------
    pd.Series or pd.DataFrame
        With the same outer index levels and column structure as origin_df, and
        the last level replaced by target_date (if time-only was given) or
        exactly `target_date` (if a full MultiIndex was given).
    """
    # --- normalize target_date to a pandas Index / MultiIndex ---
    if isinstance(target_date, ForecastingHorizon):
        try:
            td_abs = target_date.to_absolute()
            tgt = (
                td_abs.to_pandas() if hasattr(td_abs, "to_pandas") else pd.Index(td_abs)
            )
        except TypeError:
            tgt = pd.Index(target_date)
    else:
        # If it's already a (Multi)Index, use as-is; otherwise wrap in Index
        if isinstance(target_date, (pd.Index, pd.MultiIndex)):
            tgt = target_date
        else:
            tgt = pd.Index(target_date)

    idx0 = origin_df.index

    # Helper: detect "Index of tuples" with tuple length == nlevels
    def _is_full_tuple_index(ix, nlevels):
        if not isinstance(ix, pd.Index) or isinstance(ix, pd.MultiIndex):
            return False
        if ix.dtype != object:
            return False
        vals = ix.tolist()
        return len(vals) > 0 and all(
            isinstance(v, tuple) and len(v) == nlevels for v in vals
        )

    # --- build the forecast index to mirror origin_df's structure ---
    if isinstance(idx0, pd.MultiIndex):
        names = idx0.names
        nlevels = idx0.nlevels

        if isinstance(tgt, pd.MultiIndex):
            # If caller supplied a full MultiIndex, trust it (but set names)
            index = tgt
            if index.names != names:
                index = index.set_names(names)
        elif _is_full_tuple_index(tgt, nlevels):
            # Caller passed an Index of tuples matching full shape -> make MultiIndex
            index = pd.MultiIndex.from_tuples(list(tgt), names=names)
        else:
            # Usual case: caller passed time-only; replicate across outer levels
            left = idx0.droplevel(-1).unique()
            if isinstance(left, pd.MultiIndex):
                tuples = [(*lvl, t) for lvl in left for t in tgt]
                index = pd.MultiIndex.from_tuples(tuples, names=names)
            else:
                index = pd.MultiIndex.from_product([left, tgt], names=names)
    else:
        # Single-level time index: preserve dtype where possible
        if isinstance(idx0, pd.PeriodIndex):
            try:
                tgt = pd.PeriodIndex(tgt, freq=idx0.freq)
            except Exception:
                tgt = pd.Index(tgt)
        elif isinstance(idx0, pd.DatetimeIndex):
            try:
                tgt = pd.DatetimeIndex(tgt, tz=idx0.tz)
            except Exception:
                tgt = pd.Index(tgt)
        index = tgt
        # carry over name if present
        if getattr(idx0, "name", None) is not None:
            try:
                index = index.set_names(idx0.name)
            except Exception:
                index.name = idx0.name

    # Columns / values
    if columns is None:
        columns = (
            [origin_df.name]
            if isinstance(origin_df, pd.Series)
            else list(origin_df.columns)
        )
    values = 0 if fill is None else fill

    res = pd.DataFrame(values, index=index, columns=columns, dtype="float64")

    # If the origin was a Series and the result isn't hierarchical, return a Series
    if isinstance(origin_df, pd.Series) and not isinstance(index, pd.MultiIndex):
        res = res.iloc[:, 0]
        res.name = origin_df.name

    return res


# def slice_at_ix(df, ix):
#     """Slice pd.DataFrame at one index value, valid for simple Index and MultiIndex.

#     Parameters
#     ----------
#     df : pd.DataFrame
#     ix : pandas compatible index value, or iterable of index values (incl pd.Index)

#     Returns
#     -------
#     pd.DataFrame, row(s) of df, sliced at last (-1 st) level of df being equal to ix
#         all index levels are retained in the return, none are dropped
#         CAVEAT: index is sorted by last (-1 st) level if ix is iterable
#     """
#     if isinstance(ix, (list, pd.Index, ForecastingHorizon)):
#         return pd.concat([slice_at_ix(df, x) for x in ix])
#     if isinstance(df.index, pd.MultiIndex):
#         return df.xs(ix, level=-1, axis=0, drop_level=False)
#     else:
#         return df.loc[[ix]]


def slice_at_ix(df, ix):
    """Return the row(s) at ix; if ix missing, return the floor (earlier) row.

    - Simple Index: choose the greatest label <= ix (if none, choose earliest).
    - MultiIndex: apply the rule on the last level and return all rows at that time.
    """
    # Normalize ForecastingHorizon to a pandas Index (absolute if already resolvable)
    try:
        _FH = ForecastingHorizon
    except Exception:  # very defensive: avoid import-time issues
        _FH = ()  # tuple so isinstance(..., _FH) is always False if import fails

    if isinstance(ix, _FH):
        # slice_at_ix doesn't know a cutoff; it must be given an **absolute** FH.
        if getattr(ix, "is_relative", False):
            raise TypeError(
                "slice_at_ix expects an absolute ForecastingHorizon; "
                "pass fh.to_absolute(cutoff) (or a pandas Index) instead."
            )
        # absolute FH to plain pandas Index, no cutoff required
        ix = ix.to_pandas() if hasattr(ix, "to_pandas") else pd.Index(ix)

    if isinstance(ix, (list, np.ndarray, pd.Index)):
        # concatenate results for each element
        parts = [slice_at_ix(df, x) for x in ix]
        return pd.concat([p for p in parts if p is not None and not p.empty])

    if isinstance(df.index, pd.MultiIndex):
        # floor on last level
        last = df.index.get_level_values(-1)
        unique_times = pd.Index(np.unique(last))
        # use pad (floor); if below earliest, fall back to earliest
        pos = unique_times.get_indexer([ix], method="pad")
        if pos[0] == -1:
            t = unique_times[0]
        else:
            t = unique_times[pos[0]]
        return df.xs(t, level=-1, axis=0, drop_level=False)
    else:
        # simple index: floor or earliest
        idxer = df.index.get_indexer([ix], method="pad")
        if idxer[0] == -1:
            idxer = [0]
        return df.iloc[[idxer[0]]]


def _combine_exog_frames(
    X_new: pd.DataFrame | None,
    X_old: pd.DataFrame | None,
    y_index: pd.Index | None = None,
) -> pd.DataFrame | None:
    """Safely combine exogenous frames.

    Needed when one index is MultiIndex (panel) and the other is
    single-level (time). If needed, broadcast X_new to the MultiIndex
    of X_old so .combine_first can align without raising.
    """
    if X_old is None:
        return X_new
    if X_new is None:
        return X_old

    if isinstance(X_old.index, pd.MultiIndex) and not isinstance(
        X_new.index, pd.MultiIndex
    ):
        # Get series keys & index names from the stored training exog
        mi_names = X_old.index.names  # typically ['series', 'time']
        series_levels = X_old.index.droplevel(-1).unique()
        times = X_new.index

        # Broadcast X_new over series_levels -> MultiIndex aligned to X_old
        # Create the target MultiIndex first
        target_mi = pd.MultiIndex.from_product([series_levels, times], names=mi_names)

        # Repeat X_new rows for each series key and set the MultiIndex
        X_rep = X_new.loc[times].copy()
        X_rep = X_rep.loc[np.repeat(X_rep.index.values, len(series_levels))]
        X_rep.index = target_mi

        # Now the indices are compatible
        return X_rep.combine_first(X_old)

    # Otherwise, both are single-level or both already compatible
    return X_new.combine_first(X_old)


def _get_notna_idx(df):
    """Get sub-index of df that contains rows without nans.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    df_notna_idx : pd.Index
        sub-set of df.index that contains rows of df without nans
        index is in same order as of df
    """
    df_notna_bool = df.notnull().all(axis=1)
    df_notna_idx = df.index[df_notna_bool]
    return df_notna_idx


class DirectReductionForecaster(BaseForecaster, _ReducerMixin):
    """Direct reduction forecaster, incl single-output, multi-output, exogeneous Dir.

    Implements direct reduction, of forecasting to tabular regression.

    For no ``X``, defaults to DirMO (direct multioutput) for ``X_treatment =
    "concurrent"``,
    and simple direct (direct single-output) for ``X_treatment = "shifted"``.

    Direct single-output with concurrent ``X`` behaviour can be configured
    by passing a single-output ``scikit-learn`` compatible transformer.

    Algorithm details:

    In ``fit``, given endogeneous time series ``y`` and possibly exogeneous ``X``:
        fits ``estimator`` to feature-label pairs as defined as follows.
    if `X_treatment = "concurrent":
        features = ``y(t)``, ``y(t-1)``, ..., ``y(t-window_size)``, if provided:
        ``X(t+h)``
        labels = ``y(t+h)`` for ``h`` in the forecasting horizon
        ranging over all ``t`` where the above have been observed (are in the index)
        for each ``h`` in the forecasting horizon (separate estimator fitted per ``h``)
    if `X_treatment = "shifted":
        features = ``y(t)``, ``y(t-1)``, ..., ``y(t-window_size)``, if provided:
        ``X(t)``
        labels = ``y(t+h_1)``, ..., ``y(t+h_k)`` for ``h_j`` in the forecasting horizon
        ranging over all ``t`` where the above have been observed (are in the index)
        estimator is fitted as a multi-output estimator (for all ``h_j``
        simultaneously)

    In ``predict``, given possibly exogeneous ``X``, at cutoff time ``c``,
    if `X_treatment = "concurrent":
        applies fitted estimators' predict to
        feature = ``y(c)``, ``y(c-1)``, ..., ``y(c-window_size)``, if provided:
        ``X(c+h)``
        to obtain a prediction for ``y(c+h)``, for each ``h`` in the forecasting horizon
    if `X_treatment = "shifted":
        applies fitted estimator's predict to
        features = ``y(c)``, ``y(c-1)``, ..., ``y(c-window_size)``, if provided:
        ``X(c)``
        to obtain prediction for ``y(c+h_1)``, ..., ``y(c+h_k)`` for ``h_j`` in forec.
        horizon

    Parameters
    ----------
    estimator : sklearn regressor, must be compatible with sklearn interface
        tabular regression algorithm used in reduction algorithm

    window_length : int, optional, default=10
        window length used in the reduction algorithm

    transformers : currently not used

    X_treatment : str, optional, one of "concurrent" (default) or "shifted"
        determines the timestamps of X from which y(t+h) is predicted, for horizon h
        "concurrent": y(t+h) is predicted from lagged y, and X(t+h), for all h in fh
            in particular, if no y-lags are specified, y(t+h) is predicted from X(t)
        "shifted": y(t+h) is predicted from lagged y, and X(t), for all h in fh
            in particular, if no y-lags are specified, y(t+h) is predicted from X(t+h)

    impute_method : str, None, or sktime transformation, optional
        Imputation method to use for missing values in the lagged data

        * default="bfill"
        * if str, admissible strings are of ``Imputer.method`` parameter, see there.
          To pass further parameters, pass the ``Imputer`` transformer directly,
          as described below.
        * if sktime transformer, this transformer is applied to the lagged data.
          This needs to be a transformer that removes missing data, and can be
          an ``Imputer``.
        * if None, no imputation is done when applying ``Lag`` transformer

    pooling : str, one of ["local", "global", "panel"], optional, default="local"
        level on which data are pooled to fit the supervised regression model
        "local" = unit/instance level, one reduced model per lowest hierarchy level
        "global" = top level, one reduced model overall, on pooled data ignoring levels
        "panel" = second lowest level, one reduced model per panel level (-2)
        if there are 2 or less levels, "global" and "panel" result in the same
        if there is only 1 level (single time series), all three settings agree

    windows_identical : bool, optional, default=False
        Specifies whether all direct models use the same number of observations
        or a different number of observations.

        * `True` : Uniform window of length (total observations - maximum
          forecasting horizon). Note: Currently, there are no missing arising
          from window length due to backwards imputation in
          `ReductionTransformer`. Without imputation, the window size
          corresponds to (total observations + 1 - window_length + maximum
          forecasting horizon).
        * `False` : Window size differs for each forecasting horizon. Window
          length corresponds to (total observations + 1 - window_length +
          forecasting horizon).
    """

    _tags = {
        "authors": "fkiraly",
        "maintainers": "hliebert",
        "requires-fh-in-fit": True,  # is the forecasting horizon required in fit?
        "capability:exogenous": True,
        "capability:multivariate": True,
        "capability:insample": True,
        "X_inner_mtype": ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"],
        "y_inner_mtype": ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"],
        # CI and test flags
        # -----------------
        "tests:core": True,  # should tests be triggered by framework changes?
        "tests:libs": ["sktime.transformations.series.lag"],
    }

    def __init__(
        self,
        estimator,
        window_length=10,
        transformers=None,
        X_treatment="concurrent",
        impute_method="bfill",
        pooling="local",
        windows_identical=False,
    ):
        self.window_length = window_length
        self.transformers = transformers
        self.transformers_ = None
        self.estimator = estimator
        self.X_treatment = X_treatment
        self.impute_method = impute_method
        self.pooling = pooling
        self.windows_identical = windows_identical
        self._lags = list(range(window_length))
        super().__init__()

        if pooling == "local":
            mtypes = "pd.DataFrame"
        elif pooling == "global":
            mtypes = ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"]
        elif pooling == "panel":
            mtypes = ["pd.DataFrame", "pd-multiindex"]
        else:
            raise ValueError(
                "pooling in DirectReductionForecaster must be one of"
                ' "local", "global", "panel", '
                f"but found {pooling}"
            )
        self.set_tags(**{"X_inner_mtype": mtypes})
        self.set_tags(**{"y_inner_mtype": mtypes})

        # it seems that the sklearn tags are not fully reliable
        # see discussion in PR #3405 and issue #3402
        # therefore this is commented out until sktime and sklearn are better aligned
        # self.set_tags(
        #     **{"capability:missing_values": estimator._get_tags()["allow_nan"]}
        # )

    def _fit(self, y, X, fh):
        """Fit dispatcher based on X_treatment and windows_identical."""
        # shifted X (future X unknown) and identical windows reduce to
        # multioutput regression, o/w fit multiple individual estimators
        if (self.X_treatment == "shifted") and (self.windows_identical is True):
            return self._fit_multioutput(y=y, X=X, fh=fh)
        else:
            return self._fit_multiple(y=y, X=X, fh=fh)

    def _predict(self, X=None, fh=None):
        """Predict dispatcher based on X_treatment and windows_identical."""
        if self.X_treatment == "shifted":
            if self.windows_identical is True:
                return self._predict_multioutput(X=X, fh=fh)
            else:
                return self._predict_multiple(
                    X=X, fh=fh
                )  # was (X=self._X, fh=fh) which is wrong
        else:
            return self._predict_multiple(X=X, fh=fh)

    def _fit_multioutput(self, y, X=None, fh=None):
        """Fit to training data."""
        impute_method = self.impute_method
        lags = self._lags
        trafos = self.transformers

        # lagger_y_to_X_ will lag y to obtain the sklearn X
        lagger_y_to_X = ReducerTransform(
            lags=lags, transformers=trafos, impute_method=impute_method
        )
        self.lagger_y_to_X_ = lagger_y_to_X

        # lagger_y_to_y_ will lag y to obtain the sklearn y
        fh_rel = fh.to_relative(self._cutoff_scalar())
        y_lags = list(fh_rel)
        y_lags = [-x for x in y_lags]
        lagger_y_to_y = Lag(lags=y_lags, index_out="original", keep_column_names=True)
        self.lagger_y_to_y_ = lagger_y_to_y

        yt = lagger_y_to_y.fit_transform(X=y)
        y_notna_idx = _get_notna_idx(yt)

        # we now check whether the set of full lags is empty
        # if yes, we set a flag, since we cannot fit the reducer
        # instead, later, we return a dummy prediction
        if len(y_notna_idx) == 0:
            self.empty_lags_ = True
            self.dummy_value_ = y.mean()
            return self
        else:
            self.empty_lags_ = False

        yt = yt.loc[y_notna_idx]

        Xt = lagger_y_to_X.fit_transform(X=y, y=X)
        Xt = Xt.loc[y_notna_idx]

        Xt = prep_skl_df(Xt)
        yt = prep_skl_df(yt)

        estimator = clone(self.estimator)
        if not get_sklearn_tag(estimator, "capability:multioutput"):
            estimator = MultiOutputRegressor(estimator)
        estimator.fit(Xt, yt)
        self.estimator_ = estimator

        return self

    def _predict_multioutput(self, fh=None, X=None):
        """Predict core logic."""
        y_cols = self._y.columns
        fh_idx = self._get_expected_pred_idx(fh=fh)

        if self.empty_lags_:
            ret = pd.DataFrame(index=fh_idx, columns=y_cols)
            for i in ret.index:
                ret.loc[i] = self.dummy_value_
            return ret

        lagger_y_to_X = self.lagger_y_to_X_

        Xt = lagger_y_to_X.transform(X=self._y, y=self._X)
        Xt_lastrow = slice_at_ix(Xt, self.cutoff)
        Xt_lastrow = prep_skl_df(Xt_lastrow)

        estimator = self.estimator_
        # 2D numpy array with col index = (fh, var) and 1 row
        y_pred = estimator.predict(Xt_lastrow)
        y_pred = y_pred.reshape((len(fh_idx), len(y_cols)))

        y_pred = pd.DataFrame(y_pred, columns=y_cols, index=fh_idx)

        if isinstance(y_pred.index, pd.MultiIndex):
            y_pred = y_pred.sort_index()

        return y_pred

    def _fit_multiple(self, y, X=None, fh=None):
        """Fit to training data."""
        impute_method = self.impute_method
        X_treatment = self.X_treatment
        windows_identical = self.windows_identical

        # lagger_y_to_X_ will lag y to obtain the sklearn X
        lags = self._lags

        # lagger_y_to_y_ will lag y to obtain the sklearn y
        fh_rel = fh.to_relative(self._cutoff_scalar())
        y_lags = list(fh_rel)
        y_lags = [-x for x in y_lags]

        # lagging behaviour is per fh, so w initialize dicts
        # copied to self.lagger_y_to_X/y_, by reference
        lagger_y_to_y = dict()
        lagger_y_to_X = dict()
        self.lagger_y_to_y_ = lagger_y_to_y
        self.lagger_y_to_X_ = lagger_y_to_X

        self.estimators_ = []

        for lag in y_lags:
            t = Lag(lags=lag, index_out="original", keep_column_names=True)
            lagger_y_to_y[lag] = t

            yt = lagger_y_to_y[lag].fit_transform(X=y)

            impute_method = self.impute_method
            lags = self._lags
            trafos = self.transformers

            # determine whether to use concurrent X (lead them) or shifted (0)
            X_lag = (
                lag if X_treatment == "concurrent" else lag + 1
            )  # had else 0 but that seems wrong for Direct

            # lagger_y_to_X_ will lag y to obtain the sklearn X
            # also updates self.lagger_y_to_X_ by reference
            lagger_y_to_X[lag] = ReducerTransform(
                lags=lags,
                shifted_vars_lag=X_lag,
                transformers=trafos,
                impute_method=impute_method,
            )

            Xtt = lagger_y_to_X[lag].fit_transform(X=y, y=X)
            Xtt_notna_idx = _get_notna_idx(Xtt)
            yt_notna_idx = _get_notna_idx(yt)
            notna_idx = Xtt_notna_idx.intersection(yt_notna_idx)

            yt = yt.loc[notna_idx]
            Xtt = Xtt.loc[notna_idx]

            if windows_identical:
                # determine offset for uniform window length
                # convert to abs values to account for in-sample prediction
                offset = np.abs(fh_rel.to_numpy()).max() - abs(lag)
                yt = yt[offset:]
                Xtt = Xtt[offset:]

            Xtt = prep_skl_df(Xtt)
            yt = prep_skl_df(yt)

            # we now check whether the set of full lags is empty
            # if yes, we set a flag, since we cannot fit the reducer
            # instead, later, we return a dummy prediction
            if len(notna_idx) == 0:
                self.estimators_.append(y.mean())
            else:
                estimator = clone(self.estimator)
                estimator.fit(Xtt, yt)
                self.estimators_.append(estimator)

        return self

    def _predict_multiple(self, X=None, fh=None):
        """Fit to training data."""
        if X is not None and self._X is not None:
            X_pool = X.combine_first(self._X)
        elif X is None and self._X is not None:
            X_pool = self._X
        else:
            X_pool = X

        fh_idx = self._get_expected_pred_idx(fh=fh)
        y_cols = self._y.columns

        lagger_y_to_X = self.lagger_y_to_X_

        fh_rel = fh.to_relative(self._cutoff_scalar())
        fh_abs = fh.to_absolute(self._cutoff_scalar())
        y_lags = list(fh_rel)
        y_abs = list(fh_abs)

        y_pred_list = []

        for i, lag in enumerate(y_lags):
            predict_idx = y_abs[i]
            Xt = lagger_y_to_X[-lag].transform(X=self._y, y=X_pool)

            if self.X_treatment == "shifted":
                # features taken at cutoff for all horizons
                base_ix = self.cutoff
            else:
                # concurrent features at the absolute target time
                base_ix = predict_idx

            Xtt_predrow = slice_at_ix(Xt, base_ix)  # robust “floor” selection
            Xtt_predrow = prep_skl_df(Xtt_predrow)

            estimator = self.estimators_[i]

            # if = no training indices in _fit, fill in y training mean
            if isinstance(estimator, pd.Series):
                y_pred_i = pd.DataFrame(index=[0], columns=y_cols)
                y_pred_i.iloc[0] = estimator
            # otherwise proceed as per direct reduction algorithm
            else:
                y_pred_i = estimator.predict(Xtt_predrow)
            # 2D numpy array with col index = (var) and 1 row
            y_pred_list.append(y_pred_i)

        y_pred = np.concatenate(y_pred_list)
        y_pred = pd.DataFrame(y_pred, columns=y_cols, index=fh_idx)

        if isinstance(y_pred.index, pd.MultiIndex):
            y_pred = y_pred.sort_index()

        return y_pred

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        est = LinearRegression()
        params1 = {
            "estimator": est,
            "window_length": 3,
            "X_treatment": "shifted",
            "pooling": "global",  # all internal mtypes are tested across scenarios
            "windows_identical": True,
        }
        params2 = {
            "estimator": est,
            "window_length": 3,
            "X_treatment": "concurrent",
            "pooling": "global",
            "windows_identical": True,
        }
        params3 = {
            "estimator": est,
            "window_length": 3,
            "X_treatment": "shifted",
            "pooling": "global",  # all internal mtypes are tested across scenarios
            "windows_identical": False,
        }
        params4 = {
            "estimator": est,
            "window_length": 3,
            "X_treatment": "concurrent",
            "pooling": "global",
            "windows_identical": False,
        }
        params5 = {"estimator": est, "window_length": 0}

        params = [params1, params2, params3, params4, params5]

        # this fails because catboost is not sklearn compatible
        # and fails set_params contracts already in sklearn;
        # so it also fails them in sktime...
        # left here for future reference, e.g., test for non-compliant estimators
        #
        # if _check_soft_dependencies("catboost", severity="none"):
        #     from catboost import CatBoostRegressor
        #
        #     est = CatBoostRegressor(learning_rate=1, depth=6, loss_function="RMSE")
        #     params6 = {"estimator": est, "window_length": 3}
        #     params.append(params6)
        return params


class OriginalRecursiveReductionForecaster(BaseForecaster, _ReducerMixin):
    """Recursive reduction forecaster, incl exogeneous Rec.

    Implements recursive reduction, of forecasting to tabular regression.

    Algorithm details:

    In ``fit``, given endogeneous time series ``y`` and possibly exogeneous ``X``:
        fits ``estimator`` to feature-label pairs as defined as follows.

        features = ``y(t)``, ``y(t-1)``, ..., ``y(t-window_size)``, if provided:
        ``X(t+1)``
        labels = ``y(t+1)``
        ranging over all ``t`` where the above have been observed (are in the index)

    In ``predict``, given possibly exogeneous ``X``, at cutoff time ``c``,
        applies fitted estimators' predict to
        feature = ``y(c)``, ``y(c-1)``, ..., ``y(c-window_size)``, if provided:
        ``X(c+1)``
        to obtain a prediction for ``y(c+1)``.
        If a given ``y(t)`` has not been observed, it is replaced by a prediction
        obtained in the same way - done repeatedly until all predictions are obtained.
        Out-of-sample, this results in the "recursive" behaviour, where predictions
        at time points c+1, c+2, etc, are obtained iteratively.
        In-sample, predictions are obtained in a single step, with potential
        missing values obtained via the ``impute`` strategy chosen.

    Parameters
    ----------
    estimator : sklearn regressor, must be compatible with sklearn interface
        tabular regression algorithm used in reduction algorithm

    window_length : int, optional, default=10
        window length used in the reduction algorithm

    impute_method : str, None, or sktime transformation, optional
        Imputation method to use for missing values in the lagged data

        * default="bfill"
        * if str, admissible strings are of ``Imputer.method`` parameter, see there.
          To pass further parameters, pass the ``Imputer`` transformer directly,
          as described below.
        * if sktime transformer, this transformer is applied to the lagged data.
          This needs to be a transformer that removes missing data, and can be
          an ``Imputer``.
        * if None, no imputation is done when applying ``Lag`` transformer

    pooling : str, one of ["local", "global", "panel"], optional, default="local"
        level on which data are pooled to fit the supervised regression model
        "local" = unit/instance level, one reduced model per lowest hierarchy level
        "global" = top level, one reduced model overall, on pooled data ignoring levels
        "panel" = second lowest level, one reduced model per panel level (-2)
        if there are 2 or less levels, "global" and "panel" result in the same
        if there is only 1 level (single time series), all three settings agree
    X_treatment : str, optional, one of "concurrent" (default) or "shifted"
        determines the timestamps of X from which y(t+h) is predicted, for horizon h
        "concurrent": y(t+h) is predicted from lagged y, and X(t+h), for all h in fh
            in particular, if no y-lags are specified, y(t+h) is predicted from X(t)
        "shifted": y(t+h) is predicted from lagged y, and X(t), for all h in fh
            in particular, if no y-lags are specified, y(t+h) is predicted from X(t+h)
    """

    _tags = {
        "authors": "fkiraly",
        "requires-fh-in-fit": False,  # is the forecasting horizon required in fit?
        "capability:exogenous": True,
        "X_inner_mtype": ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"],
        "y_inner_mtype": ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"],
        # CI and test flags
        # -----------------
        "tests:libs": ["sktime.transformations.series.lag"],
    }

    def __init__(
        self,
        estimator,
        window_length=10,
        impute_method="bfill",
        pooling="local",
        X_treatment="concurrent",
    ):
        self.window_length = window_length
        self.estimator = estimator
        self.impute_method = impute_method
        self.pooling = pooling
        self.lagger_y_to_X_ = None
        self._lags = list(range(1, window_length + 1))
        self.X_treatment = X_treatment
        super().__init__()

        warn(
            "RecursiveReductionForecaster is experimental, and interfaces may change. "
            "user feedback is appreciated in issue #3224 here: "
            "https://github.com/alan-turing-institute/sktime/issues/3224"
        )

        if pooling == "local":
            mtypes = "pd.DataFrame"
        elif pooling == "global":
            mtypes = ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"]
        elif pooling == "panel":
            mtypes = ["pd.DataFrame", "pd-multiindex"]
        else:
            raise ValueError(
                "pooling in RecursiveReductionForecaster must be one of"
                ' "local", "global", "panel", '
                f"but found {pooling}"
            )
        self.set_tags(**{"X_inner_mtype": mtypes})
        self.set_tags(**{"y_inner_mtype": mtypes})

        if isinstance(impute_method, str):
            self._impute_method = Imputer(method=impute_method)
        elif impute_method is None:
            self._impute_method = None
        elif scitype(impute_method) == "transformer":
            self._impute_method = impute_method.clone()
        else:
            raise ValueError(
                f"Error in ReducerTransform, "
                f"impute_method must be str, None, or sktime transformer, "
                f"but found {impute_method}"
            )

    def create_lagged_features(self, y):
        """Create lagged time-based features from y and shift them for alignment.

        This function applies a lag transformation to `y` to create features for
        time series forecasting. The features are then shifted forward so they
        predict the next step in recursive forecasting.

        Parameters
        ----------
        y : pd.DataFrame
            The endogenous time series used to generate lagged features.

        Returns
        -------
        X_lagged_aligned : pd.DataFrame
            A transformed dataset where lagged features are shifted to align
            with the next target value `y(t+1)`.
        """
        lags = self._lags
        lagger_y_to_X = Lag(lags=lags, index_out="extend")

        if self._impute_method is not None:
            lagger_y_to_X = lagger_y_to_X * self._impute_method.clone()
        self.lagger_y_to_X_ = lagger_y_to_X

        X_time = lagger_y_to_X.fit_transform(y)

        # lag_shifter = Lag(lags=1, index_out="extend")
        # X_time_aligned = lag_shifter.fit_transform(X_time)
        # return X_time_aligned

        return X_time

    def _fit(self, y, X, fh):
        """Fit forecaster to training data.

        private _fit containing the core logic, called from fit

        Parameters
        ----------
        y : pd.DataFrame
            mtype is pd.DataFrame, pd-multiindex, or pd_multiindex_hier
            Time series to which to fit the forecaster.
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
            Required (non-optional) here if self.get_tag("requires-fh-in-fit")==True
            Otherwise, if not passed in _fit, guaranteed to be passed in _predict
        X : pd.DataFrame optional (default=None)
            mtype is pd.DataFrame, pd-multiindex, or pd_multiindex_hier
            Exogeneous time series to fit to.

        Returns
        -------
        self : reference to self
        """
        # todo: very similar to _fit_concurrent of DirectReductionForecaster - refactor?

        # impute_method = self._impute_method
        X_treatment = self.X_treatment

        # lagger_y_to_X_ will lag y and later concat X to obtain the sklearn X
        lags = self._lags
        lagger_y_to_X = Lag(lags=lags, index_out="extend")

        # if impute_method is not None:
        #    lagger_y_to_X = lagger_y_to_X * impute_method.clone()
        self.lagger_y_to_X_ = lagger_y_to_X

        Xt = lagger_y_to_X.fit_transform(y)

        # lag is 1, since we want to do recursive forecasting with 1 step ahead
        # column names will be kept for consistency

        # define lag_plus for *exog* alignment (furhter below)
        lag_plus = Lag(lags=1, index_out="extend", keep_column_names=True)
        # Xtt = lag_plus.fit_transform(Xt)
        Xtt = Xt
        Xtt_notna_idx = _get_notna_idx(Xtt)
        notna_idx = Xtt_notna_idx.intersection(y.index)

        # yt is the target forecast value
        yt = y.loc[notna_idx]
        Xtt = Xtt.loc[notna_idx]

        if len(notna_idx) == 0:
            self.estimator_ = y.mean()
        else:
            if X is not None:
                # if X_treatment is shifted, lag X by 1 to obtain X(t+1) i.e. X_inner
                X_inner = lag_plus.fit_transform(X) if X_treatment == "shifted" else X
                Xtt = pd.concat([X_inner.loc[notna_idx], Xtt], axis=1)

            Xtt = prep_skl_df(Xtt)
            self._feature_cols_ = list(Xtt.columns)
            yt = prep_skl_df(yt)

            # store feature column names (if any) to preserve during predict
            if hasattr(Xtt, "columns"):
                self._feature_cols_ = list(Xtt.columns)
            else:
                self._feature_cols_ = None

            estimator = clone(self.estimator)
            estimator.fit(Xtt, yt)
            self.estimator_ = estimator

        return self

    def _predict(self, X=None, fh=None):
        """Forecast time series at future horizon.

        private _predict containing the core logic, called from predict

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
            If not passed in _fit, guaranteed to be passed here
        X : pd.DataFrame, optional (default=None)
            mtype is pd.DataFrame, pd-multiindex, or pd_multiindex_hier
            Exogeneous time series for the forecast

        Returns
        -------
        y_pred : pd.DataFrame, same type as y in _fit
            Point predictions
        """
        if X is not None and self._X is not None:
            # X_pool = X.combine_first(self._X)
            X_pool = _combine_exog_frames(
                X,
                self._X,
                getattr(self, "_y", None).index if hasattr(self, "_y") else None,
            )
        elif X is None and self._X is not None:
            X_pool = self._X
        else:
            X_pool = X

        fh_oos = fh.to_out_of_sample(self.cutoff)
        fh_ins = fh.to_in_sample(self.cutoff)

        if len(fh_oos) == 0:
            y_pred = self._predict_in_sample(X_pool, fh_ins)
        elif len(fh_ins) == 0:
            y_pred = self._predict_out_of_sample(X_pool, fh_oos)
        else:
            y_pred_ins = self._predict_in_sample(X_pool, fh_ins)
            y_pred_oos = self._predict_out_of_sample(X_pool, fh_oos)
            y_pred = pd.concat([y_pred_ins, y_pred_oos], axis=0)

        if isinstance(y_pred.index, pd.MultiIndex):
            y_pred = y_pred.sort_index()

        return y_pred

    # def _get_window_local(self, cutoff, window_length, y_orig):
    #     start = _shift(cutoff, by=-window_length + 1)
    #     cutoff = cutoff[0]
    #     y = y_orig.loc[start:cutoff]

    #     # check for missing values
    #     if len(y) < window_length:
    #         idx = pd.period_range(
    #             start=y.index.min(), end=y.index.max(), freq=y.index.freq
    #         )
    #         y = y.reindex(idx)
    #         if self._impute_method:
    #             y = self._impute_method.fit_transform(y)

    #     y = y.to_numpy()
    #     X = (
    #         self._X.loc[cutoff].to_frame().T if self._X is not None else None
    #     )  # exoxenous

    #     return y, X

    def _get_window_local(self, cutoff, window_length, y_orig):
        # Normalize cutoff to a scalar label
        if isinstance(cutoff, (pd.Index, pd.DatetimeIndex, pd.PeriodIndex)):
            cutoff_scalar = cutoff[0]
        else:
            cutoff_scalar = cutoff

        # --- pick the single series (local path) and get its time index ---
        if isinstance(y_orig.index, pd.MultiIndex):
            # choose the series that actually contains the cutoff
            inst_key = None
            for k in y_orig.index.droplevel(-1).unique():
                if cutoff_scalar in y_orig.xs(k, level=0).index:
                    inst_key = k
                    break
            if inst_key is None:
                # fallback: take the last series present
                inst_key = y_orig.index.get_level_values(0)[-1]
            y_s = y_orig.xs(inst_key, level=0)
        else:
            y_s = y_orig
            inst_key = None  # not used

        # --- find positional window [start_pos : cutoff_pos], no freq needed ---
        try:
            pos = y_s.index.get_loc(cutoff_scalar)
            if not isinstance(pos, (int, np.integer)):
                if isinstance(pos, slice):
                    pos = pos.stop - 1
                else:  # e.g. boolean mask / array of positions
                    pos = np.asarray(pos).max()
        except KeyError:
            # cutoff not in index: take the rightmost label <= cutoff
            pos = y_s.index.searchsorted(cutoff_scalar, side="right") - 1

        start_pos = max(0, pos - (window_length - 1))
        idx_segment = y_s.index[start_pos : pos + 1]
        y_win = y_s.loc[idx_segment]

        # coerce to 1D if we got a single-column DataFrame
        if isinstance(y_win, pd.DataFrame):
            y_win = y_win.iloc[:, 0]

        # left-pad if shorter than window_length; optional imputation
        if len(y_win) < window_length:
            pad = np.full(window_length - len(y_win), np.nan, dtype=float)
            y_vals = np.concatenate([pad, y_win.to_numpy(dtype=float, copy=False)])
            if getattr(self, "_impute_method", None) is not None:
                y_imp = self._impute_method.fit_transform(pd.Series(y_vals))
                y_vals = np.asarray(y_imp).reshape(-1)
        else:
            y_vals = y_win.to_numpy(dtype=float, copy=False)

        # best-effort X alignment at cutoff (not required in v2-local, but safe)
        X = None
        if getattr(self, "_X", None) is not None:
            try:
                if isinstance(self._X.index, pd.MultiIndex) and (inst_key is not None):
                    X = self._X.xs(inst_key, level=0).loc[[cutoff_scalar]]
                else:
                    X = self._X.loc[[cutoff_scalar]]
            except Exception as ex:
                warnings.warn(
                    "_get_window_local: could not align X at cutoff "
                    f"{cutoff_scalar!r}: {ex}"
                )
                X = None

        return y_vals, X

    def _get_window_global(self, cutoff, window_length, y_orig):
        """Return last window_length values per series up to `cutoff` (inclusive).

        Handle case of global pooling. Robust to missing freq, scalar cutoffs,
        and tupled time levels.
        """
        # ---------- ensure MultiIndex on y_orig ----------
        idx = y_orig.index
        if not isinstance(idx, pd.MultiIndex):
            # case A: it's already an Index of tuples -> rewrap as a proper MultiIndex
            if isinstance(idx, pd.Index) and all(isinstance(i, tuple) for i in idx):
                names = None
                if hasattr(self, "_y") and isinstance(self._y.index, pd.MultiIndex):
                    names = self._y.index.names
                y_orig = y_orig.copy()
                y_orig.index = pd.MultiIndex.from_tuples(idx, names=names)
            else:
                # case B: a single time index
                time_name = y_orig.index.name or getattr(self, "_time_name", "time")
                series_name = getattr(self, "_series_name", "series")

                if isinstance(y_orig, pd.DataFrame) and y_orig.shape[1] > 1:
                    # WIDE -> LONG: stack columns into a single 'y' column
                    y_long = y_orig.stack().to_frame("y")  # index: (time, series)
                    y_long.index.names = [time_name, series_name]
                    y_orig = y_long.swaplevel(
                        0, 1
                    ).sort_index()  # index: (series, time)
                else:
                    # univariate -> synthesize a single series level
                    if isinstance(y_orig, pd.Series):
                        y_orig = y_orig.to_frame("y")
                    y_orig.index = pd.MultiIndex.from_product(
                        [[series_name], y_orig.index], names=[series_name, time_name]
                    )

        if y_orig.index.has_duplicates:
            # print("\n[get_window_global] ----- DEBUG START -----")
            # print(
            #     f"[get_window_global] cutoff={repr(cutoff)}, \
            #        window_length={window_length}"
            # )
            # print(
            #     f"[get_window_global] y_orig.index.names={y_orig.index.names}, \
            #        is_multi={isinstance(y_orig.index, pd.MultiIndex)}"
            # )

            dup_idx = y_orig.index[y_orig.index.duplicated(keep=False)]
            # count unique duplicate keys (series,time)
            # dup_keys = list(map(tuple, dup_idx))
            # n_keys = len(set(dup_keys))
            # print(
            #     f"[get_window_global] DUPLICATE (series,time) rows: \
            #            total_rows={len(dup_idx)}, unique_keys={n_keys}"
            # )
            try:
                print(
                    "[get_window_global] First duplicate rows:\n",
                    y_orig.loc[dup_idx].sort_index().head(20),
                )
            except Exception as e:
                print(f"[get_window_global] printing duplicate rows failed: {e}")

        # ---------- helpers ----------
        def _as_1len_index(x):
            """Normalize cutoff to a 1-length Index matching its dtype."""
            if isinstance(x, pd.Index):
                return x[:1]
            if isinstance(x, pd.Period):
                return pd.PeriodIndex([x], freq=x.freq)
            if isinstance(x, pd.Timestamp):
                return pd.DatetimeIndex([x], tz=x.tz)
            return pd.Index([x])

        def _coerce_time_index(idx_like):
            """Ensure scalar entries; if entries are tuples, take their last element."""
            if isinstance(idx_like, pd.MultiIndex):
                idx_like = idx_like.get_level_values(-1)
            if isinstance(idx_like, pd.Index) and idx_like.dtype == object:
                vals = list(idx_like)
                if any(isinstance(v, tuple) for v in vals):
                    vals = [v[-1] if isinstance(v, tuple) else v for v in vals]
                    return pd.Index(vals)
            return pd.Index(idx_like)

        # ---------- normalize cutoff & extract/clean time axis ----------
        cutoff_idx = _as_1len_index(cutoff)
        cutoff_val = cutoff_idx[0]

        time_level_raw = y_orig.index.get_level_values(-1)
        time_level = _coerce_time_index(time_level_raw)

        # Unique times; if sorting fails (mixed types), sort by string as last resort
        try:
            unique_times = time_level.unique().sort_values()
        except TypeError:
            unique_times = pd.Index(time_level.unique()).sort_values(
                key=lambda x: x.astype(str)
            )

        # ---------- locate cutoff position on the global time axis ----------
        pos = unique_times.get_indexer([cutoff_val])[0]
        if pos == -1:
            pos = unique_times.searchsorted(cutoff_val)
            pos = max(0, min(pos, len(unique_times) - 1))

        start_pos = max(0, pos - (window_length - 1))
        observed_window = unique_times[
            start_pos : pos + 1
        ]  # may be < window_length near start

        # If a reliable freq is known, build a regular target window ending at cutoff;
        # otherwise, use observed_window and pad per-series later.
        full_idx = observed_window
        if isinstance(time_level, pd.PeriodIndex):
            end = (
                cutoff_val if isinstance(cutoff_val, pd.Period) else observed_window[-1]
            )
            full_idx = pd.period_range(
                end - (window_length - 1),
                end,
                freq=time_level.freq,
                name=time_level.name,
            )
        elif isinstance(time_level, pd.DatetimeIndex):
            freq = time_level.freq or pd.infer_freq(time_level)
            if freq is not None:
                end = (
                    cutoff_val
                    if isinstance(cutoff_val, pd.Timestamp)
                    else observed_window[-1]
                )
                full_idx = pd.date_range(
                    end=end,
                    periods=window_length,
                    freq=freq,
                    tz=time_level.tz,
                    name=time_level.name,
                )

        # ---------- build feature matrix (rows=series, cols=lags reversed) ----------
        all_series = y_orig.index.droplevel(-1).unique()
        y_time_features = np.zeros((len(all_series), window_length), dtype=float)

        for i, s in enumerate(all_series):
            # print(f"top of loop: i / s = {i} / {s}")
            y_s = y_orig.loc[s]

            # ensure y_s is a 1D Series of target values
            if isinstance(y_s, pd.DataFrame):
                if y_s.shape[1] == 1:
                    y_s = y_s.iloc[:, 0]
                else:
                    col = "y" if "y" in y_s.columns else y_s.columns[0]
                    y_s = y_s[col]

            # clean per-series time index (in case of tuples/objects)
            idx_s = _coerce_time_index(y_s.index)
            y_s = pd.Series(y_s.to_numpy(copy=False), index=idx_s)

            # print(f"y_s = {y_s}")
            # print(f"type(y_s) = {type(y_s)}")
            # print(f"y_s.shape = {y_s.shape}")

            idx_target = full_idx if len(full_idx) == window_length else observed_window

            # ---- pre-check: duplicates in y_s or idx_target ----
            if not y_s.index.is_unique:
                dups = y_s.index[y_s.index.duplicated(keep=False)]
                # print("\n[get_window_global] --- DEBUG (per-series duplicates) ---")
                # print(f"[get_window_global] series={s!r}")
                # print(
                #     f"[get_window_global] \
                #        y_s.index.is_unique={y_s.index.is_unique} (n={len(y_s.index)})"
                # )
                try:
                    vc = pd.Series(dups).value_counts().head(10)
                    print(
                        "[get_window_global] duplicated time -> counts (top 10):\n", vc
                    )
                    print(
                        "[get_window_global] first rows for duplicated times:\n",
                        y_s.loc[dups].sort_index().head(10),
                    )
                except Exception as ex:
                    print("[get_window_global] failed to print dup rows:", ex)

            # if not pd.Index(idx_target).is_unique:
            #     tgt_dup = pd.Index(idx_target)[
            #         pd.Index(idx_target).duplicated(keep=False)
            #     ]
            # print("\n[get_window_global] --- DEBUG (idx_target duplicates) ---")
            # print(
            #     f"[get_window_global] series={s!r}, \
            #         duplicate labels in idx_target \
            #         (first 20): {list(tgt_dup[:20])}"
            # )

            # align; if dtype mismatches, intersect then reindex
            try:
                y_win = y_s.reindex(idx_target)
            except Exception:
                y_win = y_s[y_s.index.isin(idx_target)].reindex(idx_target)

            # left-pad if shorter than window_length
            if len(y_win) < window_length:
                pad = np.full(window_length - len(y_win), np.nan, dtype=float)
                y_vals = np.concatenate([pad, y_win.to_numpy(dtype=float, copy=False)])
            else:
                y_vals = y_win.to_numpy(dtype=float, copy=False)

            # optional imputation
            if getattr(self, "_impute_method", None):
                y_imp = self._impute_method.fit_transform(pd.Series(y_vals))
                y_vals = np.asarray(y_imp).reshape(-1)

            # reverse for lag order
            y_time_features[i] = y_vals[::-1]

        X = None  # exogenous features are not assembled here (v2 no-X path)
        return y_time_features, X

    def _get_window(self, cutoff=None, window_length=None, y_orig=None):
        cutoff = self.cutoff if cutoff is None else cutoff
        window_length = self.window_length if window_length is None else window_length
        y_orig = self._y if y_orig is None else y_orig

        if self.pooling == "local":
            return self._get_window_local(cutoff, window_length, y_orig)
        elif self.pooling == "global":
            return self._get_window_global(cutoff, window_length, y_orig)
        elif self.pooling == "panel":
            # For <=2 index levels panel pooling degenerates to global pooling
            # (spec states global and panel identical when <=2 levels).
            # If higher hierarchy present (>=3 levels), panel pooling would
            # correspond to modelling per second-lowest level; not yet optimized.
            if isinstance(y_orig.index, pd.MultiIndex) and y_orig.index.nlevels <= 2:
                return self._get_window_global(cutoff, window_length, y_orig)
            else:
                # Fallback: return global-style window for now; future work:
                # implement per-panel window extraction (#panel-optimization)
                return self._get_window_global(cutoff, window_length, y_orig)

    def _is_predictable(self, last_window, window_length):
        """Check if we can make predictions from last window."""
        return (
            len(last_window) == window_length
            and np.sum(np.isnan(last_window)) == 0
            and np.sum(np.isinf(last_window)) == 0
        )

    # def _create_nan_df(self, fh):
    #     """Return nan predictions for horizon fh."""
    #     index = fh.to_absolute(self.cutoff).to_pandas()
    #     y_pred = pd.DataFrame(index=index, columns=self._y.columns)
    #     return y_pred

    def _create_fallback_df(self, fh):
        """Return fallback predictions (constant mean if available else NaN)."""
        # index = fh.to_absolute(self.cutoff).to_pandas()
        index = fh.to_absolute(self._cutoff_scalar()).to_pandas()
        y_pred = pd.DataFrame(index=index, columns=self._y.columns)
        est = getattr(self, "estimator_", None)
        if isinstance(est, pd.Series):
            for col in y_pred.columns:
                y_pred[col] = est[col] if col in est.index else np.nan
        return y_pred

    def _predict_out_of_sample_v2_global(
        self, X_pool, fh
    ):  # TODO: why are exogenous features not used?
        """Recursive reducer: predict out of sample (ahead of cutoff).

        Copied and hacked from _RecursiveReducer._predict_last_window.

        In recursive reduction, iteration must be done over the
        entire forecasting horizon. Specifically, when transformers are
        applied to y that generate features in X, forecasting must be done step by
        step to integrate the latest prediction of for the new set of features in
        X derived from that y.

        Parameters
        ----------
        X_pool : pd.DataFrame
            Exogenous & time based features for the forecast

        fh : int, list, np.array or ForecastingHorizon
            Forecasting horizon

        Returns
        -------
        y_return = pd.Series or pd.DataFrame
        """
        # If exogenous data are present (in-fit or provided now), fall back to
        # the legacy v1 path which already supports X for correctness.
        # This maintains performance benefit of v2 for the no-X case while
        # enabling functionality with X.
        if (self._X is not None) or (X_pool is not None):
            return self._predict_out_of_sample_v1(X_pool, fh)

        # Get last window of available data.
        # If we cannot generate a prediction from the available data, return nan.
        y_last, X_last = self._get_window()
        ys = np.array(y_last)
        if np.isnan(ys).any() or np.isinf(ys).any():
            return self._create_fallback_df(fh)

        fh_max = fh.to_relative(self._cutoff_scalar())[-1]
        relative = pd.Index(list(map(int, range(1, fh_max + 1))))
        index_range = _index_range(relative, self.cutoff)
        if isinstance(self.cutoff, pd.Timestamp) and self.cutoff.tz is not None:
            index_range = index_range.tz_localize(self.cutoff.tz)

        y_pred = _create_fcst_df(index_range, self._y)

        orig_idx = self._y.index
        if isinstance(orig_idx, pd.MultiIndex) and not isinstance(
            y_pred.index, pd.MultiIndex
        ):
            # _create_fcst_df may return Index of tuples; rewrap as MultiIndex
            if all(isinstance(ix, tuple) for ix in y_pred.index):
                y_pred.index = pd.MultiIndex.from_tuples(
                    y_pred.index, names=orig_idx.names
                )

        y_last_df = self._y.copy()
        for i in range(fh_max):
            # Generate predictions.
            if getattr(self, "_feature_cols_", None) is not None and y_last.shape[
                1
            ] == len(self._feature_cols_):
                X_last_df = pd.DataFrame(
                    y_last, columns=self._feature_cols_
                )  # <- new local name
                y_pred_vector = self.estimator_.predict(X_last_df)
            else:
                y_pred_vector = self.estimator_.predict(y_last)
            y_pred_curr = _create_fcst_df([index_range[i]], self._y, fill=y_pred_vector)

            if isinstance(orig_idx, pd.MultiIndex) and not isinstance(
                y_pred_curr.index, pd.MultiIndex
            ):
                if all(isinstance(ix, tuple) for ix in y_pred_curr.index):
                    y_pred_curr.index = pd.MultiIndex.from_tuples(
                        y_pred_curr.index, names=orig_idx.names
                    )

            y_pred.update(y_pred_curr)

            # # Update last window with previous prediction.
            if i + 1 != fh_max:
                # Append preds to previous df
                # merge on index except from last
                tmp = pd.concat([y_last_df, y_pred_curr])
                if not isinstance(tmp.index, pd.MultiIndex):
                    # handle rare mixed-object index; coerce if we can
                    if all(isinstance(ix, tuple) for ix in tmp.index):
                        tmp.index = pd.MultiIndex.from_tuples(
                            tmp.index, names=orig_idx.names
                        )
                if isinstance(tmp.index, pd.MultiIndex):
                    tmp = tmp.sort_index()
                y_last_df = tmp

                y_last, X_last = self._get_window(
                    cutoff=index_range[i : i + 1], y_orig=y_last_df
                )
        return y_pred

    def _predict_out_of_sample_v2_local(
        self, X_pool, fh
    ):  # TODO: why are exogenous features not used?
        """Recursive reducer: predict out of sample (ahead of cutoff).

        Copied and hacked from _RecursiveReducer._predict_last_window.

        In recursive reduction, iteration must be done over the
        entire forecasting horizon. Specifically, when transformers are
        applied to y that generate features in X, forecasting must be done step by
        step to integrate the latest prediction for the new set of features in
        X derived from that y.

        Parameters
        ----------
        X_pool : pd.DataFrame
            Exogenous & time based features for the forecast

        fh : int, list, np.array or ForecastingHorizon
            Forecasting horizon

        Returns
        -------
        y_return = pd.Series or pd.DataFrame
        """
        # if self._X is not None:
        #    raise ValueError(
        #        "Do not call this function if model uses exogenous variables X."
        #    )

        # Get last window of available data.
        # If we cannot generate a prediction from the available data, return nan.
        # y_last, X_last = self._get_window(self._cutoff, self.window_length, self._y)

        # Force local window extraction even if pooling=='global' (vectorized wide case)
        cutoff_idx = (
            self._cutoff
            if isinstance(self._cutoff, (pd.Index, pd.DatetimeIndex, pd.PeriodIndex))
            else pd.Index([self._cutoff])
        )
        y_last, X_last = self._get_window_local(
            cutoff=cutoff_idx, window_length=self.window_length, y_orig=self._y
        )
        if not self._is_predictable(y_last, self.window_length):
            return self._create_fallback_df(fh)

        # Pre-allocate arrays.
        n_columns = 1
        window_length = self.window_length
        fh_max = fh.to_relative(self._cutoff_scalar())[-1]

        y_pred = np.zeros(fh_max)

        # Array with input data for prediction.
        last = np.zeros((1, n_columns, window_length + fh_max))

        # Fill pre-allocated arrays with available time based features.
        last[:, 0, :window_length] = y_last.T

        if X_pool is not None:
            # fh_absolute = fh.to_absolute(self.cutoff)
            # Drive selection by absolute labels instead of positions
            # first_label = fh_absolute[0]
            # Pre-compute a dense absolute index from cutoff+1..cutoff+max(fh)
            dense_abs_fh, _ = self._generate_fh_no_gaps(fh)
            try:
                dense_abs_idx = dense_abs_fh.to_pandas()
            except Exception:
                dense_abs_idx = pd.Index(dense_abs_fh)

        # Recursively generate predictions by iterating over forecasting horizon.
        for i in range(fh_max):
            # Slice prediction window.
            X_pred = last[:, :, i : window_length + i]

            # Reshape data into tabular array.
            # if self._estimator_scitype == "tabular-regressor":
            X_pred = X_pred.reshape(1, -1)[
                :, ::-1
            ]  # reverse order of columns to match lag order

            if X_pool is not None:
                # label sequence for absolute future times (already computed above)
                label_i = dense_abs_idx[i]

                # ---- helpers to make labels scalar & fetch the row safely ----
                def _as_scalar_label(lbl):
                    # If we ever get a 1-length Index, turn it into a scalar
                    if isinstance(lbl, (pd.Index, pd.DatetimeIndex, pd.PeriodIndex)):
                        return lbl[0] if len(lbl) == 1 else lbl
                    return lbl

                def _row_from(label):
                    lab = _as_scalar_label(label)
                    if X_pool is not None and lab in X_pool.index:
                        return X_pool.loc[lab].to_numpy().reshape(1, -1)
                    if getattr(self, "_X", None) is not None and lab in self._X.index:
                        return self._X.loc[lab].to_numpy().reshape(1, -1)
                    raise MissingExogenousDataError(
                        f"Missing exogenous data for timestamp {lab!r} "
                        f"(checked future X passed to predict and training X)."
                    )

                # ---- alignment semantics you specified ----
                # step i forecasts t+i+1 from time t+i
                if getattr(self, "X_treatment", "concurrent") == "shifted":
                    # shifted uses exog(t+i):
                    # first step needs exog at cutoff (t) from training X
                    label_for_X = self.cutoff if i == 0 else dense_abs_idx[i - 1]
                else:
                    # concurrent uses exog(t+i+1): take the forecast label itself
                    label_for_X = label_i

                row = _row_from(label_for_X)
                X_pred = np.concatenate((row, X_pred), axis=1)

            # Generate predictions (ensure robust scalar extraction) with names
            if getattr(self, "_feature_cols_", None) is not None and X_pred.shape[
                1
            ] == len(self._feature_cols_):
                X_pred_df = pd.DataFrame(X_pred, columns=self._feature_cols_)
                _raw_pred = self.estimator_.predict(X_pred_df)
            else:
                _raw_pred = self.estimator_.predict(X_pred)
            # handle outputs like list/Series/ndarray; ravel then take first element
            _scalar = np.asarray(_raw_pred).ravel()[0]
            y_pred[i] = float(_scalar)

            # Update last window with previous prediction.
            last[:, 0, window_length + i] = y_pred[i]

        return y_pred

    def _filter_and_adjust_predictions(self, fh, y_pred):
        """Filter predictions to requested fh and fix freq when needed."""
        fh_idx = fh.to_indexer(self.cutoff)

        # If train index was multi-level, *may* need to group by non-time levels.
        if isinstance(self._y.index, pd.MultiIndex):
            yi_grp = [n for n in self._y.index.names[:-1] if n is not None]

            # See where those keys live in y_pred (index levels? columns? nowhere?)
            idx_names = list(getattr(getattr(y_pred, "index", None), "names", []))
            cols = (
                list(getattr(y_pred, "columns", []))
                if isinstance(y_pred, pd.DataFrame)
                else []
            )

            if yi_grp and all(k in idx_names for k in yi_grp):
                # Standard long/panel case: group by series-like levels on the index.
                y_return = y_pred.groupby(yi_grp, as_index=False).nth(fh_idx.to_list())

            elif (
                yi_grp
                and isinstance(y_pred, pd.DataFrame)
                and all(k in cols for k in yi_grp)
            ):
                # Keys are in the columns (rare); group after reset_index.
                tmp = y_pred.reset_index()
                y_return = tmp.groupby(yi_grp, as_index=False).nth(fh_idx.to_list())
                # if we still have a time column, restore it as index
                time_col = self._y.index.names[-1]
                if time_col in y_return.columns:
                    y_return = y_return.set_index(time_col)

            else:
                # Degenerate single-series case (wide vectorization):
                #     no grouping; select rows.
                if isinstance(y_pred, (pd.Series, pd.DataFrame)):
                    y_return = y_pred.iloc[fh_idx]
                else:
                    y_return = y_pred[fh_idx]

                # …and inject a series level so the vectorizer can reassemble A/B/C
                if isinstance(y_return, (pd.Series, pd.DataFrame)) and not isinstance(
                    y_return.index, pd.MultiIndex
                ):
                    ser_level_name = self._y.index.names[-2] or "series"
                    # use the sole training column as the series label
                    ser_label = (
                        self._y.columns[0]
                        if hasattr(self._y, "columns") and len(self._y.columns) == 1
                        else "series"
                    )
                    y_return = y_return.copy()
                    # make sure we have a DataFrame
                    if not isinstance(y_return, pd.DataFrame):
                        y_return = y_return.to_frame(
                            self._y.columns[0] if hasattr(self._y, "columns") else "y"
                        )
                    y_return[ser_level_name] = ser_label
                    y_return = y_return.set_index(ser_level_name, append=True)
                    # reorder to (series, time)
                    y_return.index = y_return.index.swaplevel(-1, -2)
                    y_return = y_return.sort_index()

        else:
            # Univariate / no-multiindex: just select rows.
            if isinstance(y_pred, (pd.Series, pd.DataFrame)):
                y_return = y_pred.iloc[fh_idx]
                # keep pandas from complaining about freq mismatches
                if hasattr(y_return.index, "freq") and hasattr(y_pred.index, "freq"):
                    if y_return.index.freq != y_pred.index.freq:
                        y_return.index.freq = None
            else:
                y_return = y_pred[fh_idx]

        return y_return

    # def _generate_fh_no_gaps(self, fh):
    #     """Create a forecasting horizon with no gaps for continuous indexing."""
    #     fh_rel = fh.to_relative(self.cutoff)
    #     y_lags = list(fh_rel)

    #     # Ensure all positive forecast horizons are covered
    #     y_lags_no_gaps = range(1, y_lags[-1] + 1)
    #     y_abs_no_gaps = ForecastingHorizon(
    #         list(y_lags_no_gaps), is_relative=True, freq=self._cutoff
    #     ).to_absolute_index(self._cutoff)

    #     return y_abs_no_gaps, y_lags_no_gaps

    # def _generate_fh_no_gaps(self, fh):
    #     """Return a dense (no-gaps) absolute index.

    #     From cutoff through max fh, plus its relative lags.

    #     Parameters
    #     ----------
    #     fh : ForecastingHorizon or array-like

    #     Returns
    #     -------
    #     y_abs_no_gaps : pd.Index
    #         Absolute (time) index from cutoff+1 through cutoff+max(fh) without gaps.
    #     y_lags_no_gaps : range
    #         1..max(fh) as integer relative lags.
    #     """
    #     fh_arr = _ensure_relative_oos_int_fh(fh, cutoff=self.cutoff)
    #     fh_max = int(np.max(fh_arr)) if len(fh_arr) else 0

    #     # Dense relative lags 1..fh_max
    #     y_lags_no_gaps = range(1, fh_max + 1)

    #     # Turn dense lags into absolute time index from the current cutoff
    #     dense_fh = ForecastingHorizon(list(y_lags_no_gaps), is_relative=True)
    #     y_abs_no_gaps = dense_fh.to_absolute_index(self.cutoff)

    #     return y_abs_no_gaps, y_lags_no_gaps

    # def _generate_fh_no_gaps(self, fh):
    #     """Return a gapless absolute FH and the dense relative steps [1..fh_max]."""
    #     # normalize to relative ints ahead of cutoff
    #     fh_rel = _ensure_relative_oos_int_fh(fh, cutoff=self.cutoff)
    #     fh_max = int(np.max(fh_rel)) if len(fh_rel) else 0
    #     y_lags_no_gaps = range(1, fh_max + 1)

    #     # keep time type/freq consistent with training index
    #     dense_fh_rel = ForecastingHorizon(
    #         list(y_lags_no_gaps), is_relative=True, freq=self.cutoff
    #     )
    #     dense_fh_abs = dense_fh_rel.to_absolute(self.cutoff)  # <-- FH (not Index)

    #     return dense_fh_abs, y_lags_no_gaps

    # def _generate_fh_no_gaps(self, fh):

    #     # ensure cutoff is a scalar label, not a 1-element Index
    #     cutoff_scalar = self.cutoff
    #     if isinstance(cutoff_scalar, (pd.Index, pd.PeriodIndex, pd.DatetimeIndex)):
    #         cutoff_scalar = cutoff_scalar[0]

    #     fh_rel = _ensure_relative_oos_int_fh(fh, cutoff=cutoff_scalar)
    #     fh_max = int(np.max(fh_rel)) if len(fh_rel) else 0
    #     y_lags_no_gaps = range(1, fh_max + 1)

    #     # --- derive a proper frequency for the relative FH ---
    #     freq = None
    #     idx = getattr(self, "_y", None)
    #     if idx is not None:
    #         idx = self._y.index
    #         if isinstance(idx, (pd.DatetimeIndex, pd.PeriodIndex)):
    #             freq = idx.freq
    #             if freq is None:
    #                 try:
    #                     freq = pd.infer_freq(idx)
    #                 except Exception:
    #                     freq = None
    #     if freq is None:
    #         # fall back to freq of cutoff when available
    #         #     (works for Period; Timestamp has no freq)
    #         if isinstance(cutoff_scalar, pd.Period):
    #             freq = cutoff_scalar.freq
    #         # for plain Timestamp without index freq, leave freq=None (no safe guess)

    #     # build a relative FH and carry the freq so
    #     #     .to_absolute(...) can offset correctly
    #     dense_fh_rel = ForecastingHorizon(list(y_lags_no_gaps), is_relative=True,
    #                                       freq=freq)

    #     # convert to absolute using the scalar cutoff
    #     dense_fh_abs = dense_fh_rel.to_absolute(cutoff_scalar)
    #     return dense_fh_abs, y_lags_no_gaps

    def _generate_fh_no_gaps(self, fh):
        """Return a gapless absolute FH and the dense relative steps [1..fh_max]."""
        # 1) scalar cutoff
        cutoff_scalar = self.cutoff
        if isinstance(cutoff_scalar, (pd.Index, pd.PeriodIndex, pd.DatetimeIndex)):
            cutoff_scalar = cutoff_scalar[0]

        # 2) requested FH in absolute labels (pd.Index/PeriodIndex/DatetimeIndex/int)
        if isinstance(fh, ForecastingHorizon):
            abs_fh = fh.to_absolute(cutoff_scalar).to_pandas()
        else:
            # treat as relative steps
            rel = _ensure_relative_oos_int_fh(fh, cutoff=cutoff_scalar)
            fh_rel = ForecastingHorizon(list(rel), is_relative=True)
            abs_fh = fh_rel.to_absolute(cutoff_scalar).to_pandas()

        if len(abs_fh) == 0:
            # nothing requested -> empty absolute index and empty dense lags
            return pd.Index([], dtype=getattr(self._y.index, "dtype", None)), range(
                1, 1
            )

        # 3) compute inclusive [cutoff+1 ... max(abs_fh)]
        last_abs = abs_fh.max()

        def _step_forward(label, n=1):
            # advance one step according to index type
            if isinstance(label, pd.Period):
                return label + n
            if isinstance(label, pd.Timestamp):
                # use training index freq if available; else infer; else day
                freq = getattr(getattr(self, "_y", None), "index", None)
                freq = getattr(freq, "freq", None) or (
                    pd.infer_freq(self._y.index) if hasattr(self._y, "index") else None
                )
                if freq is None:
                    # default to 1 day if nothing avail (works for tests w/ daily data)
                    return label + pd.Timedelta(days=n)
                return pd.date_range(
                    start=label, periods=n + 1, freq=freq, tz=label.tz
                )[-1]
            # integer-like
            return label + n

        first_abs = _step_forward(cutoff_scalar, 1)

        # 4) build gapless absolute index
        if isinstance(last_abs, pd.Period):
            gapless_abs = pd.period_range(
                start=first_abs, end=last_abs, freq=last_abs.freq
            )
        elif isinstance(last_abs, pd.Timestamp):
            # same tz and inferred/original frequency
            freq = getattr(getattr(self, "_y", None), "index", None)
            freq = getattr(freq, "freq", None) or (
                pd.infer_freq(self._y.index) if hasattr(self._y, "index") else None
            )
            if freq is None:
                # fallback: day
                gapless_abs = pd.date_range(
                    start=first_abs, end=last_abs, tz=last_abs.tz
                )
            else:
                gapless_abs = pd.date_range(
                    start=first_abs, end=last_abs, freq=freq, tz=last_abs.tz
                )
        else:
            # integers
            gapless_abs = pd.Index(range(int(first_abs), int(last_abs) + 1))

        # 5) dense lags have the same length as the gapless absolute index
        y_lags_no_gaps = range(1, len(gapless_abs) + 1)

        # return as absolute FH (callers use .to_pandas())
        dense_fh_abs = ForecastingHorizon(gapless_abs, is_relative=False)
        return dense_fh_abs, y_lags_no_gaps

    def _predict_out_of_sample(self, X_pool, fh):
        """Recursive reducer: predict out of sample (ahead of cutoff)."""
        # very similar to _predict_concurrent of DirectReductionForecaster - refactor?
        # Strategy selection:
        #   global  -> optimized v2 global path (fallback to v1 inside if exogenous X)
        #   local   -> optimized v2 local path
        #   panel   -> fallback to legacy v1 path for correctness (gappy fh indexing)
        #             TODO: implement optimized v2 panel path (#panel-optimization)

        if isinstance(getattr(self, "estimator_", None), pd.Series):
            # Produce a DataFrame of repeated means on the absolute fh index
            return self._create_fallback_df(fh)

        already_filtered = False
        if self.pooling == "panel":
            # v1 path already returns only the requested fh rows
            y_pred = self._predict_out_of_sample_v1(X_pool, fh)
            already_filtered = True
        elif self.pooling == "global" and isinstance(self._y.index, pd.MultiIndex):
            y_pred = self._predict_out_of_sample_v2_global(X_pool, fh)
            # v2_global falls back to v1 when X is present; v1 already returns only the
            # requested fh rows, so skip the second filtering step.
            if (self._X is not None) or (X_pool is not None):
                already_filtered = True
        elif self.pooling == "local" or not isinstance(self._y.index, pd.MultiIndex):
            y_pred = self._predict_out_of_sample_v2_local(X_pool, fh)
        else:
            raise ValueError(
                "Unsupported pooling setting for RecursiveReductionForecaster: "
                f"{self.pooling}"
            )

        # Filter to requested fh only unless already filtered in path
        if already_filtered:
            y_return = y_pred
        else:
            y_return = self._filter_and_adjust_predictions(fh, y_pred)

        # If result is a raw numpy array (local path), wrap into DataFrame with
        # the absolute fh index. This avoids attempting to coerce onto a gapless
        # index of length fh_max (which caused shape mismatches for gappy fh).
        if isinstance(y_return, np.ndarray):
            fh_abs_index = fh.to_absolute(self.cutoff).to_pandas()
            y_return = pd.DataFrame(
                y_return, columns=self._y.columns, index=fh_abs_index
            )
        elif isinstance(y_return, pd.Series):
            # ensure DataFrame with correct columns
            fh_abs_index = fh.to_absolute(self.cutoff).to_pandas()
            y_return = pd.DataFrame(
                y_return.values, columns=self._y.columns, index=fh_abs_index
            )
        # if already a DataFrame, we assume indices align with requested fh

        if isinstance(y_return.index, pd.MultiIndex):
            y_return = y_return.sort_index()

        return y_return

    def _predict_out_of_sample_v1(self, X_pool, _fh):
        """Recursive reducer: predict out of sample (ahead of cutoff) — v1 semantics.

        Use strict index typing & accumulator behaviour to match v2 outputs.
        """
        # this routine has a bug for relative fh
        # instead of tracking it down, just use absolute fh
        fh = _fh.to_absolute(self._cutoff_scalar())

        # final, possibly gappy index we must return (typed like v2)
        fh_idx = self._get_expected_pred_idx(fh=fh)
        self._assert_future_X_coverage(X_pool, fh)

        # dense horizon driver (just for loop count)
        _, y_lags_no_gaps = self._generate_fh_no_gaps(fh)

        # recursive state starts at the observed y
        y_plus_preds = self._y

        # use the *fitted* lagger from training, not a new one
        lagger_y_to_X = self.lagger_y_to_X_

        # extend-by-one helper; include impute if configured (like original v1)
        lag_plus = Lag(lags=1, index_out="extend", keep_column_names=True)
        if self._impute_method is not None:
            lag_plus = lag_plus * self._impute_method.clone()

        # exogenous pool we may extend in lock-step
        X_ext = X_pool

        # pre-allocate an accumulator exactly on requested horizons (typed)
        y_pred_full = _create_fcst_df(fh_idx, self._y)

        for _ in y_lags_no_gaps:
            # keep frequency consistent if fh carries a freq
            if getattr(self.fh, "freq", None) is not None:
                y_plus_preds = _asfreq_per_series_safe(
                    y_plus_preds, self.fh.freq, how="start"
                )

            # expose the next prediction timestamp
            y_plus_one = lag_plus.fit_transform(y_plus_preds)
            next_time_raw = (
                y_plus_one.index.get_level_values(-1)[-1]
                if isinstance(y_plus_one.index, pd.MultiIndex)
                else y_plus_one.index[-1]
            )

            # recursive design from y-lags, extended by one to include next_time_raw
            Xt = lagger_y_to_X.transform(y_plus_preds)
            Xtt = Xt.copy()
            # Xtt = lag_plus.fit_transform(Xt)

            # pick the single design row for the next timestamp
            if isinstance(Xtt.index, pd.MultiIndex):
                Xtt_row = Xtt.xs(next_time_raw, level=-1, drop_level=False)
            else:
                Xtt_row = Xtt.loc[[next_time_raw]]

            # if exog is present: extend/slice it in lock-step and concat to design
            if X_ext is not None:
                X_ext = lag_plus.fit_transform(X_ext)
                if isinstance(X_ext.index, pd.MultiIndex):
                    X_ex_row = X_ext.xs(next_time_raw, level=-1, drop_level=False)
                else:
                    X_ex_row = X_ext.loc[[next_time_raw]]
                Xtt_row = pd.concat([X_ex_row, Xtt_row], axis=1)

            Xtt_row = prep_skl_df(Xtt_row)

            # build the *typed* index for this step so it matches v2 exactly
            step_idx = self._get_expected_pred_idx(fh=[next_time_raw])
            n_rows = len(step_idx)

            # 1-step prediction for all series present at this timestamp
            est = self.estimator_
            if isinstance(est, pd.Series):
                # constant-mean fallback: repeat row-wise
                vals = np.tile(est.values, (n_rows, 1))
                y_step = pd.DataFrame(vals, index=step_idx, columns=self._y.columns)
            else:
                y_hat = est.predict(Xtt_row)
                y_step = pd.DataFrame(y_hat, index=step_idx, columns=self._y.columns)

            # write into accumulator (last value wins) and into recursive state
            y_pred_full.update(y_step)
            y_plus_preds = y_plus_preds.combine_first(y_step)

        # return exactly the requested horizons, already typed
        return y_pred_full.loc[fh_idx]

    def _predict_in_sample(self, X_pool, fh):
        """Recursive reducer: predict out of sample (in past of of cutoff)."""
        fh_idx = self._get_expected_pred_idx(fh=fh)
        y_cols = self._y.columns

        lagger_y_to_X = self.lagger_y_to_X_

        fh_abs = fh.to_absolute(self._cutoff_scalar())
        y = self._y

        Xt = lagger_y_to_X.transform(y)

        # column names will be kept for consistency
        lag_plus = Lag(lags=1, index_out="extend", keep_column_names=True)

        if self._impute_method is not None:
            lag_plus = lag_plus * self._impute_method.clone()

        Xtt = lag_plus.fit_transform(Xt)

        Xtt_predrows = slice_at_ix(Xtt, fh_abs)
        if X_pool is not None:
            Xtt_predrows = pd.concat(
                [slice_at_ix(X_pool, fh_abs), Xtt_predrows], axis=1
            )

        Xtt_predrows = prep_skl_df(Xtt_predrows)
        estimator = self.estimator_

        # if = no training indices in _fit, fill in y training mean
        if isinstance(estimator, pd.Series):
            y_pred = pd.DataFrame(index=fh_idx, columns=y_cols)
            y_pred = y_pred.fillna(self.estimator)
        # otherwise proceed as per direct reduction algorithm
        else:
            y_pred = estimator.predict(Xtt_predrows)
            # 2D numpy array with col index = (var) and 1 row
            y_pred = pd.DataFrame(y_pred, columns=y_cols, index=fh_idx)

        return y_pred

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        est = LinearRegression()
        forecaster_imputer = Imputer(
            method="forecaster", forecaster=DirectReductionForecaster(estimator=est)
        )

        params1 = {
            "estimator": est,
            "window_length": 3,
            "pooling": "global",  # all internal mtypes are tested across scenarios
        }
        params2 = {
            "estimator": est,
            "window_length": 4,
            "pooling": "local",
            "X_treatment": "shifted",
            "impute_method": None,  # None is the default
        }
        params3 = {
            "estimator": est,
            "window_length": 4,
            "pooling": "local",
            "impute_method": forecaster_imputer,  # test imputation with forecaster
        }
        params4 = {
            "estimator": est,
            "window_length": 4,
            "pooling": "global",
            "impute_method": forecaster_imputer,
            "X_treatment": "shifted",
        }
        params5 = {
            "estimator": est,
            "window_length": 4,
            "pooling": "local",
            "impute_method": "pad",
        }
        params6 = {
            "estimator": est,
            "window_length": 4,
            "pooling": "global",
            "impute_method": "pad",
        }

        return [params1, params2, params3, params4, params5, params6]


class RecursiveReductionForecaster(OriginalRecursiveReductionForecaster):
    """Public class with wide/long auto-handling for pooling='global'.

    CASE 1 (WIDE):
        time index (one level) + multiple cols -> as multiple series.
            Convert to LONG for _fit; remember flags; _predict then return WIDE.
    CASE 2 (LONG): MultiIndex (series, time) -> classic global training on LONG.
                   Normalize to single target col 'y'; _predict then return WIDE.

    For pooling!='global' or univariate inputs, defer to the original _fit/_predict.
    """

    # Tell BaseForecaster we can handle multivariate & hierarchical without vectorizing
    _tags = dict(OriginalRecursiveReductionForecaster._tags)
    _tags.update(
        {
            "capability:multivariate": True,  # do not split DataFrame columns
            # keep inner mtypes broad so our _fit/_predict see the full object
            "capability:exogenous": True,
            "capability:insample": True,
            "y_inner_mtype": ["pd-multiindex", "pd_multiindex_hier", "pd.DataFrame"],
            "X_inner_mtype": ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"],
            "tests:core": True,
        }
    )

    # flags remembered after _fit
    _was_wide_input: bool = False
    _was_long_input: bool = False
    _orig_columns = None
    _time_name: str = "time"
    _series_name: str = "series"

    def _broadcast_X_to_panel(self, X: pd.DataFrame, y_index: pd.Index) -> pd.DataFrame:
        """Alignment of indexes of y and X.

        If y is panel (MultiIndex ['series','time']) and X is 1-level over time,
        broadcast X across the series level so X and y share the same index shape/names.
        """
        if (
            X is None
            or not isinstance(y_index, pd.MultiIndex)
            or isinstance(X.index, pd.MultiIndex)
        ):
            return X

        # series + time from y
        series_levels = y_index.get_level_values(0).unique()

        # rep X for each series; creae a MultiIndex with names ['series','time']
        X_broadcast = pd.concat(
            {s: X.copy() for s in series_levels}, names=["series", "time"]
        )

        # Make sure time level matches X's index name if it had one;
        # tests usually care about names (y already has names ['series','time'])
        return X_broadcast

    def fit(self, y, X=None, fh=None):
        # remember original index/columns for roundtripping
        # (you already set these in _to_long_from_wide; keep that behavior)
        if getattr(self, "pooling", None) == "global" and self._is_wide(y):
            y = self._to_long_from_wide(y)
            self._was_wide_input = True
            self._was_long_input = False
        else:
            # keep state consistent
            self._was_wide_input = False
            self._was_long_input = isinstance(getattr(y, "index", None), pd.MultiIndex)

        X = self._broadcast_X_to_panel(X, y.index)

        # IMPORTANT:
        # call base public fit (not _fit), so BaseForecaster stores y_metadata, etc
        return super().fit(y=y, X=X, fh=fh)

    # 3) Override PUBLIC predict to roundtrip back to WIDE if we trained from WIDE.
    def predict(self, fh=None, X=None):
        y_pred = super().predict(
            fh=fh, X=X
        )  # will return LONG mtype because we trained on LONG
        if getattr(self, "_was_wide_input", False):
            # convert pooled LONG back to WIDE columns in original order
            y_pred = self._to_wide_from_long(y_pred)
        return y_pred

    # -------- helpers --------
    def _is_wide(self, y):
        return (
            isinstance(y, pd.DataFrame)
            and not isinstance(y.index, pd.MultiIndex)
            and y.shape[1] >= 2
        )

    def _is_long_multi(self, y):
        return isinstance(y.index, pd.MultiIndex) and y.index.nlevels >= 2

    def _to_long_from_wide(self, y_wide: pd.DataFrame) -> pd.DataFrame:
        self._time_name = y_wide.index.name or self._time_name
        self._orig_columns = list(y_wide.columns)
        y_long = y_wide.stack().to_frame("y")  # (time, series)
        y_long.index = y_long.index.set_names([self._time_name, self._series_name])
        y_long = y_long.swaplevel(0, 1).sort_index()  # (series, time)
        return y_long

    def _to_wide_from_long(self, y_long):
        # Accept Series or DataFrame; expect index=(series, time)
        if isinstance(y_long, pd.Series):
            y_long = y_long.to_frame("y")
        if not isinstance(y_long.index, pd.MultiIndex) or y_long.index.nlevels < 2:
            return y_long

        df = y_long.copy()
        df.index = df.index.set_names([self._series_name, self._time_name])
        wide = df.unstack(level=0)  # columns -> (val_col, series)
        if isinstance(
            wide.columns, pd.MultiIndex
        ) and "y" in wide.columns.get_level_values(0):
            wide = wide["y"]

        # restore original column order if available
        if self._orig_columns is not None:
            existing = [c for c in self._orig_columns if c in wide.columns]
            missing = [c for c in wide.columns if c not in existing]
            wide = wide[existing + missing]

        return wide

    # -------- core overrides --------
    def _fit(self, y, X=None, fh=None):
        # If not global pooling, retain legacy behavior
        if getattr(self, "pooling", None) != "global":
            self._was_wide_input = False
            self._was_long_input = False
            return OriginalRecursiveReductionForecaster._fit(self, y, X=X, fh=fh)

        self._uses_exog = X is not None

        # CASE 1: WIDE (time index + multiple columns)
        if self._is_wide(y):
            if X is not None:
                raise ValueError("Wide data plus exogenous not supported")
            y_long = self._to_long_from_wide(y)
            self._was_wide_input = True
            self._was_long_input = False
            # NOTE: if you later pass X with concurrent treatment,
            #           mirror the transform for X here.
            ret = OriginalRecursiveReductionForecaster._fit(
                self, y=y_long, X=None, fh=fh
            )
            return ret

        # CASE 2: LONG (MultiIndex >=2 levels; last level should be time per spec)
        if self._is_long_multi(y):
            # normalize to a single target column named 'y'
            if isinstance(y, pd.Series):
                y_long = y.to_frame("y")
            elif isinstance(y, pd.DataFrame):
                if y.shape[1] == 1:
                    y_long = (
                        y.rename(columns={y.columns[0]: "y"})
                        if y.columns[0] != "y"
                        else y
                    )
                else:
                    col = "y" if "y" in y.columns else y.columns[0]
                    y_long = y[[col]].rename(columns={col: "y"})
            else:
                y_long = pd.DataFrame(y, columns=["y"])

            # remember level names, if present
            if isinstance(y_long.index, pd.MultiIndex):
                names = list(y_long.index.names)
                if names and len(names) >= 2:
                    self._series_name = names[-2] or self._series_name
                    self._time_name = names[-1] or self._time_name

            ## self._was_wide_input = False - this is set correctly in fit() override
            self._was_long_input = True
            return OriginalRecursiveReductionForecaster._fit(self, y=y_long, X=X, fh=fh)

        # Otherwise (univariate or other shapes): delegate to legacy
        self._was_wide_input = False
        self._was_long_input = False
        return OriginalRecursiveReductionForecaster._fit(self, y=y, X=X, fh=fh)

    def _predict(self, fh=None, X=None):
        y_pred = OriginalRecursiveReductionForecaster._predict(self, fh=fh, X=X)
        return y_pred


class YfromX(BaseForecaster, _ReducerMixin):
    """Simple reduction predicting endogeneous from concurrent exogeneous variables.

    Tabulates all seen ``X`` and ``y`` by time index and applies
    tabular supervised regression.

    In ``fit``, given endogeneous time series ``y`` and exogeneous ``X``:
    fits ``estimator`` to feature-label pairs as defined as follows.

    features = :math:`y(t)`, labels: :math:`X(t)`
    ranging over all :math:`t` where the above have been observed (are in the index)

    In ``predict``, at a time :math:`t` in the forecasting horizon, uses ``estimator``
    to predict :math:`y(t)`, from labels: :math:`X(t)`

    If regressor is ``skpro`` probabilistic regressor, and has ``predict_interval`` etc,
    uses ``estimator`` to predict :math:`y(t)`, from labels: :math:`X(t)`,
    passing on the ``predict_interval`` etc arguments.

    If no exogeneous data is provided, will predict the mean of ``y`` seen in ``fit``.

    In order to use a fit not on the entire historical data
    and update periodically, combine this with ``UpdateRefitsEvery``.

    In order to deal with missing data, combine this with ``Imputer``.

    To construct an custom direct reducer,
    combine with ``YtoX``, ``Lag``, or ``ReducerTransform``.

    Parameters
    ----------
    estimator : sklearn regressor or skpro probabilistic regressor,
        must be compatible with sklearn or skpro interface
        tabular regression algorithm used in reduction algorithm
        if skpro regressor, resulting forecaster will have probabilistic capability
    pooling : str, one of ["local", "global", "panel"], optional, default="local"
        level on which data are pooled to fit the supervised regression model
        "local" = unit/instance level, one reduced model per lowest hierarchy level
        "global" = top level, one reduced model overall, on pooled data ignoring levels
        "panel" = second lowest level, one reduced model per panel level (-2)
        if there are 2 or less levels, "global" and "panel" result in the same
        if there is only 1 level (single time series), all three settings agree

    Examples
    --------
    >>> from sktime.datasets import load_longley
    >>> from sktime.split import temporal_train_test_split
    >>> from sktime.forecasting.compose import YfromX
    >>> from sklearn.linear_model import LinearRegression
    >>>
    >>> y, X = load_longley()
    >>> y_train, y_test, X_train, X_test = temporal_train_test_split(y, X)
    >>> fh = y_test.index
    >>>
    >>> f = YfromX(LinearRegression())
    >>> f.fit(y=y_train, X=X_train, fh=fh)
    YfromX(...)
    >>> y_pred = f.predict(X=X_test)

    YfromX can also be used with skpro probabilistic regressors,
    in this case the resulting forecaster will be capable of probabilistic forecasts:
    >>> from skpro.regression.residual import ResidualDouble  # doctest: +SKIP
    >>> reg_proba = ResidualDouble(LinearRegression())  # doctest: +SKIP
    >>> f = YfromX(reg_proba)  # doctest: +SKIP
    >>> f.fit(y=y_train, X=X_train, fh=fh)  # doctest: +SKIP
    YfromX(...)
    >>> y_pred = f.predict_interval(X=X_test)  # doctest: +SKIP
    """

    _tags = {
        "requires-fh-in-fit": False,  # is the forecasting horizon required in fit?
        "capability:exogenous": True,
        "capability:missing_values": True,
        "X_inner_mtype": ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"],
        "y_inner_mtype": ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"],
        "capability:pred_int": True,
        "capability:categorical_in_X": True,
    }

    def __init__(self, estimator, pooling="local"):
        self.estimator = estimator
        self.pooling = pooling
        super().__init__()

        # self._est_type encodes information what type of estimator is passed
        if hasattr(estimator, "get_tags"):
            _est_type = estimator.get_tag("object_type", "regressor", False)
        else:
            _est_type = "regressor"

        if _est_type not in ["regressor", "regressor_proba"]:
            raise TypeError(
                "error in YfromX, estimator must be either an sklearn compatible "
                "regressor, or an skpro probabilistic regressor."
            )

        # has probabilistic mode iff the estimator is of type regressor_proba
        self.set_tags(**{"capability:pred_int": _est_type == "regressor_proba"})

        self._est_type = _est_type

        if pooling == "local":
            mtypes = "pd.DataFrame"
        elif pooling == "global":
            mtypes = ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"]
        elif pooling == "panel":
            mtypes = ["pd.DataFrame", "pd-multiindex"]
        else:
            raise ValueError(
                "pooling in DirectReductionForecaster must be one of"
                ' "local", "global", "panel", '
                f"but found {pooling}"
            )
        self.set_tags(**{"X_inner_mtype": mtypes})
        self.set_tags(**{"y_inner_mtype": mtypes})

    def _fit(self, y, X, fh):
        """Fit forecaster to training data.

        private _fit containing the core logic, called from fit

        Parameters
        ----------
        y : pd.DataFrame
            mtype is pd.DataFrame, pd-multiindex, or pd_multiindex_hier
            Time series to which to fit the forecaster.
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
            Required (non-optional) here if self.get_tag("requires-fh-in-fit")==True
            Otherwise, if not passed in _fit, guaranteed to be passed in _predict
        X : pd.DataFrame optional (default=None)
            mtype is pd.DataFrame, pd-multiindex, or pd_multiindex_hier
            Exogeneous time series to fit to.

        Returns
        -------
        self : reference to self
        """
        _est_type = self._est_type

        if X is None:
            if _est_type == "regressor":
                estimator = DummyRegressor()
            else:  # "proba_regressor"
                if not _check_soft_dependencies("skpro", severity="none"):
                    raise ModuleNotFoundError(
                        "Probability forecasting with reduction requires optional "
                        "dependency 'skpro'. Please `pip install skpro` to use this "
                        "mode."
                    )
                from skpro.regression.residual import ResidualDouble

                dummy = DummyRegressor()
                estimator = ResidualDouble(dummy)
            X = prep_skl_df(y, copy_df=True)
        else:
            X = prep_skl_df(X, copy_df=True)
            estimator = clone(self.estimator)

        if _est_type == "regressor":
            y = prep_skl_df(y, copy_df=True)
            y = y.values.flatten()

        estimator.fit(X, y)
        self.estimator_ = estimator

        return self

    def _predict(self, X=None, fh=None):
        """Forecast time series at future horizon.

        private _predict containing the core logic, called from predict

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
            If not passed in _fit, guaranteed to be passed here
        X : pd.DataFrame, optional (default=None)
            mtype is pd.DataFrame, pd-multiindex, or pd_multiindex_hier
            Exogeneous time series for the forecast

        Returns
        -------
        y_pred : pd.DataFrame, same type as y in _fit
            Point predictions
        """
        _est_type = self._est_type

        fh_idx = self._get_expected_pred_idx(fh=fh)

        X_idx = self._get_pred_X(X=X, fh_idx=fh_idx)
        y_pred = self.estimator_.predict(X_idx)

        if _est_type == "regressor":
            y_cols = self._y.columns
            y_pred = pd.DataFrame(y_pred, index=fh_idx, columns=y_cols)

        return y_pred

    def _predict_quantiles(self, fh, X, alpha):
        """Compute/return prediction quantiles for a forecast.

        private _predict_quantiles containing the core logic,
            called from predict_quantiles and possibly predict_interval

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon
            The forecasting horizon with the steps ahead to to predict.
        X :  sktime time series object, optional (default=None)
            guaranteed to be of an mtype in self.get_tag("X_inner_mtype")
            Exogeneous time series for the forecast
        alpha : list of float (guaranteed not None and floats in [0,1] interval)
            A list of probabilities at which quantile forecasts are computed.

        Returns
        -------
        quantiles : pd.DataFrame
            Column has multi-index: first level is variable name from y in fit,
                second level being the values of alpha passed to the function.
            Row index is fh, with additional (upper) levels equal to instance levels,
                    from y seen in fit, if y_inner_mtype is Panel or Hierarchical.
            Entries are quantile forecasts, for var in col index,
                at quantile probability in second col index, for the row index.
        """
        fh_idx = self._get_expected_pred_idx(fh=fh)
        X_idx = self._get_pred_X(X=X, fh_idx=fh_idx)
        y_pred = self.estimator_.predict_quantiles(X_idx, alpha=alpha)
        return y_pred

    def _predict_interval(self, fh, X, coverage):
        """Compute/return prediction quantiles for a forecast.

        private _predict_interval containing the core logic,
            called from predict_interval and possibly predict_quantiles

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon
            The forecasting horizon with the steps ahead to to predict.
        X :  sktime time series object, optional (default=None)
            guaranteed to be of an mtype in self.get_tag("X_inner_mtype")
            Exogeneous time series for the forecast
        coverage : list of float (guaranteed not None and floats in [0,1] interval)
           nominal coverage(s) of predictive interval(s)

        Returns
        -------
        pred_int : pd.DataFrame
            Column has multi-index: first level is variable name from y in fit,
                second level coverage fractions for which intervals were computed.
                    in the same order as in input ``coverage``.
                Third level is string "lower" or "upper", for lower/upper interval end.
            Row index is fh, with additional (upper) levels equal to instance levels,
                from y seen in fit, if y_inner_mtype is Panel or Hierarchical.
            Entries are forecasts of lower/upper interval end,
                for var in col index, at nominal coverage in second col index,
                lower/upper depending on third col index, for the row index.
                Upper/lower interval end forecasts are equivalent to
                quantile forecasts at alpha = 0.5 - c/2, 0.5 + c/2 for c in coverage.
        """
        fh_idx = self._get_expected_pred_idx(fh=fh)
        X_idx = self._get_pred_X(X=X, fh_idx=fh_idx)
        y_pred = self.estimator_.predict_interval(X_idx, coverage=coverage)
        return y_pred

    def _predict_var(self, fh, X=None, cov=False):
        """Forecast variance at future horizon.

        private _predict_var containing the core logic, called from predict_var

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
            If not passed in _fit, guaranteed to be passed here
        X :  sktime time series object, optional (default=None)
            guaranteed to be of an mtype in self.get_tag("X_inner_mtype")
            Exogeneous time series for the forecast
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
                    from y seen in fit, if y_inner_mtype is Panel or Hierarchical.
                Entries are variance forecasts, for var in col index.
                A variance forecast for given variable and fh index is a predicted
                    variance for that variable and index, given observed data.
            If cov=True:
                Column index is a multiindex: 1st level is variable names (as above)
                    2nd level is fh.
                Row index is fh, with additional levels equal to instance levels,
                    from y seen in fit, if y_inner_mtype is Panel or Hierarchical.
                Entries are (co-)variance forecasts, for var in col index, and
                    covariance between time index in row and col.
                Note: no covariance forecasts are returned between different variables.
        """
        fh_idx = self._get_expected_pred_idx(fh=fh)
        X_idx = self._get_pred_X(X=X, fh_idx=fh_idx)
        y_pred = self.estimator_.predict_var(X_idx)
        return y_pred

    def _predict_proba(self, fh, X, marginal=True):
        """Compute/return fully probabilistic forecasts.

        private _predict_proba containing the core logic, called from predict_proba

        Parameters
        ----------
        fh : int, list, np.array or ForecastingHorizon (not optional)
            The forecasting horizon encoding the time stamps to forecast at.
            if has not been passed in fit, must be passed, not optional
        X : sktime time series object, optional (default=None)
                Exogeneous time series for the forecast
            Should be of same scitype (Series, Panel, or Hierarchical) as y in fit
            if self.get_tag("X-y-must-have-same-index"),
                X.index must contain fh.index and y.index both
        marginal : bool, optional (default=True)
            whether returned distribution is marginal by time index

        Returns
        -------
        pred_dist : sktime BaseDistribution
            predictive distribution
            if marginal=True, will be marginal distribution by time point
            if marginal=False and implemented by method, will be joint
        """
        fh_idx = self._get_expected_pred_idx(fh=fh)
        X_idx = self._get_pred_X(X=X, fh_idx=fh_idx)
        y_pred = self.estimator_.predict_proba(X_idx)
        return y_pred

    def _get_pred_X(self, X, fh_idx):
        y_cols = self._y.columns

        if X is not None and self._X is not None:
            X_pool = X.combine_first(self._X)
        elif X is None and self._X is not None:
            X_pool = self._X
        elif X is not None:
            X_pool = X
        else:
            X_pool = pd.DataFrame(0, index=fh_idx, columns=y_cols)

        X_pool = prep_skl_df(X_pool, copy_df=True)

        X_idx = X_pool.loc[fh_idx]
        return X_idx

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        params1 = {
            "estimator": LinearRegression(),
            "pooling": "local",
        }

        params2 = {
            "estimator": RandomForestRegressor(),
            "pooling": "global",  # all internal mtypes are tested across scenarios
        }

        params = [params1, params2]

        if _check_soft_dependencies("skpro", severity="none"):
            from skpro.regression.residual import ResidualDouble

            params3 = {
                "estimator": ResidualDouble.create_test_instance(),
                "pooling": "global",
            }

            params = params + [params3]

        return params
