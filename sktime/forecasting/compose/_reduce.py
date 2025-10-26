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
    "ericjb",
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

import os
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline as SkPipeline
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

try:
    from sktime.datatypes._vectorize import VectorizedDF
except Exception:
    VectorizedDF = ()


def _unwrap_vdf(obj):
    """Return the underlying pandas object if obj is a VectorizedDF; else obj."""
    if obj is None:
        return None
    if isinstance(obj, VectorizedDF):
        # Prefer the internal dataframe if present; else try a safe converter.
        if hasattr(obj, "_df"):
            return obj._df
        if hasattr(obj, "to_pandas"):
            try:
                return obj.to_pandas()
            except Exception:
                # If conversion fails for any reason, just fall through
                return obj
    return obj


def _rrlog(msg):
    # cheap, dependency-free conditional logger so we can see what's happening
    # if os.environ.get("SKTIME_DEBUG_RR", "1") == "1":
    dbgprint(f"[RR.unwrap] {msg}")
    pass  # remove when uncommenting


# alias used throughout helpers
def _d(msg):
    # _rrlog(msg)
    pass  # remove when uncommenting


def dbgprint(*args, **kwargs):
    if False:
        print(*args, **kwargs)


def _unwrap_vectorized_df(obj):
    if obj is None:
        return None
    try:
        from sktime.datatypes._vectorize import VectorizedDF
    except Exception:
        VectorizedDF = type("VectorizedDF", (), {})

    if isinstance(obj, VectorizedDF):
        # 1) If the wrapper already carries a MultiIndex DataFrame, just use it.
        cand = getattr(obj, "y_multiindex", None)
        if isinstance(cand, pd.DataFrame):
            _d(f"[RR.unwrap] using y_multiindex directly: shape={cand.shape}")
            return cand
        cand = getattr(obj, "X_multiindex", None)
        if isinstance(cand, pd.DataFrame):
            _d(f"[RR.unwrap] using X_multiindex directly: shape={cand.shape}")
            return cand

        # 2) Reconstruct from stored pieces (values + multiindex + columns)
        vals = getattr(obj, "Y", getattr(obj, "X", None))
        mi_idx = getattr(obj, "y_mi_index", getattr(obj, "X_mi_index", None))
        mi_cols = getattr(obj, "y_mi_columns", getattr(obj, "X_mi_columns", None))
        if (vals is not None) and (mi_idx is not None) and (mi_cols is not None):
            try:
                arr = np.asarray(vals)
                if arr.ndim == 3:
                    n_inst, n_time, n_feat = arr.shape
                    arr = arr.reshape(n_inst * n_time, n_feat)
                elif arr.ndim > 2:
                    arr = arr.reshape(arr.shape[0] * arr.shape[1], -1)
                df = pd.DataFrame(arr, index=mi_idx, columns=mi_cols)
                _d(
                    f"[RR.unwrap] reconstructed MI DataFrame: shape={df.shape}, "
                    f"index={type(df.index)}"
                )
                return df
            except Exception as e:
                _d(f"[RR.unwrap] reconstruction failed: {e!r}")

        # 3) Known converters across sktime versions
        try:
            from sktime.datatypes import convert_to

            for target, scitype in (
                ("pd_multiindex_hier", "Hierarchical"),
                ("pd-multiindex", "Panel"),
            ):
                try:
                    df = convert_to(obj, target, as_scitype=scitype)
                    _d(f"[RR.unwrap] convert_to(...,'{target}')->OK: type={type(df)}")
                    return df
                except Exception as e:
                    _d(f"[RR.unwrap] convert_to(..., '{target}') failed: {e!r}")
        except Exception as e:
            _d(f"[RR.unwrap] import convert_to failed: {e!r}")

        # 4) Very last-gasp
        try:
            df = pd.DataFrame(obj)
            _d("[RR.unwrap] pd.DataFrame(obj) -> OK")
            return df
        except Exception as e:
            _d(f"[RR.unwrap] pd.DataFrame(obj) failed: {e!r}")

        _d("[RR.unwrap] FAILED to unwrap; returning original VectorizedDF")
        return obj

    # Not a VectorizedDF: return as-is
    return obj


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


def _to_nested_from_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Convert a wide lagged tabular frame into sktime-nested format (row-wise).

    Input:  numeric DataFrame with columns like 'lag_0__0', 'lag_1__0', ...
            (scalars per cell; one row per training sample)
    Output: object-dtype DataFrame with the SAME NUMBER OF ROWS as input;
            each cell holds a pd.Series of the per-row lag sequence for that variable.

    Notes
    -----
    - If df is already nested (any object dtype), return df unchanged.
    - If columns do not have __, treat all columns as one variable group.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df

    # Already nested? keep as-is
    dtypes = getattr(df, "dtypes", None)
    if dtypes is not None and any(dtypes == "object"):
        return df

    cols = list(df.columns)
    # Group columns by suffix after '__' (variable id). If none, single group.
    has_dunder = any("__" in str(c) for c in cols)
    if not has_dunder:
        groups = {"0": cols}
    else:
        groups = {}
        for c in cols:
            c_str = str(c)
            var_id = c_str.split("__")[-1]
            groups.setdefault(var_id, []).append(c)

    import re

    lag_re = re.compile(r"lag_(\d+)")

    def _lag_key(colname: str) -> int:
        m = lag_re.search(str(colname))
        return int(m.group(1)) if m else 0

    # Build nested columns: one per var-id; each row -> Series of that row's lags
    nested_cols = {}
    for var_id, gcols in groups.items():
        gcols_sorted = sorted(gcols, key=_lag_key)
        vals = df[gcols_sorted].to_numpy()  # shape (n_rows, n_lags_for_var)
        # Crucial: keep per-ROW series, not a single long series
        series_per_row = pd.Series(
            [pd.Series(row) for row in vals], index=df.index, dtype="object"
        )
        nested_cols[f"var_{var_id}"] = series_per_row

    nested = pd.DataFrame(nested_cols, index=df.index)
    # Ensure object dtype (nested) and preserve the row count
    for c in nested.columns:
        nested[c] = nested[c].astype("object")
    return nested


def _expand_single_row_nested_to_rows(
    Xn: pd.DataFrame, target_len: int
) -> pd.DataFrame:
    """Do special processing when there is an sklearn Tabularizer.

    If Xn is a single-row nested df (1,k) and every cell holds a series/array
    of length target_len, expand it to target_len rows; each expanded row holds
    a length-1 series containing the corresponding element. Otherwise return Xn.
    """
    if isinstance(Xn, pd.DataFrame) and Xn.shape[0] == 1:
        row = Xn.iloc[0]

        # all cells must be sequence-like and same expected length
        def _len_ok(v):
            return isinstance(v, (pd.Series, np.ndarray, list)) and len(v) == target_len

        if all(_len_ok(cell) for cell in row):
            new = {}
            for col in Xn.columns:
                seq = list(row[col])
                new[col] = [pd.Series([v]) for v in seq]
            return pd.DataFrame(new, index=pd.RangeIndex(target_len))
    return Xn


def _has_tabularizer_step(est):
    """Return True if `est` is an sklearn Pipeline that includes a Tabularizer.

    (robust to import location by checking the class name).
    """
    if not isinstance(est, SkPipeline):
        return False
    for _, step in est.steps:
        cls = type(step)
        if cls.__name__ == "Tabularizer":
            return True
    return False


class MissingExogenousDataError(RuntimeError):
    """Future X not provided but are expected.

    Raised when a forecast that requires exogenous variables is requested
    but future X rows for the forecast horizon are not provided.
    """

    pass


class _ReducerMixin:
    """Common utilities for reducers."""

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

    def _cutoff_as_1elem_index_with_freq(self):
        """Return a 1-element Index at the cutoff that carries/inherits a usable freq.

        This avoids `to_offset(None)` issues when converting relative FH to absolute.
        """
        c = self._cutoff_scalar()

        if isinstance(c, pd.Period):
            return pd.PeriodIndex([c], freq=c.freq)

        if isinstance(c, pd.Timestamp):
            # Try to inherit or infer a real freq for the 1-elem cutoff index.
            freq = None
            y_idx = getattr(self, "_y", None)
            y_idx = getattr(y_idx, "index", None)

            if isinstance(y_idx, pd.MultiIndex):
                # infer from one series (avoid duplicate times in pooled level)
                try:
                    keys = y_idx.droplevel(-1).unique()
                    ser_idx = self._y.loc[keys[0]].index
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

            return (
                pd.date_range(start=c, periods=1, freq=freq, tz=c.tz)
                if freq is not None
                else pd.DatetimeIndex([c], tz=c.tz)
            )

        if isinstance(c, (pd.PeriodIndex, pd.DatetimeIndex, pd.Index)):
            return c[:1]

        # last resort: plain Index
        return pd.Index([c])

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

        # ---- DEBUG (safe) ----------------------------------------------------
        # if getattr(self, "verbose", False):
        #     try:
        #         print(
        #             "[_get_expected_pred_idx] ENTER",
        #             f"type(fh)={type(fh).__name__}",
        #             f"is_FH={isinstance(fh, ForecastingHorizon)}",
        #         )
        #         print(
        #             "[_get_expected_pred_idx] y_index:",
        #             type(y_index).__name__,
        #             "names="
        #             + (
        #                 ",".join(y_index.names)
        #                 if hasattr(y_index, "names")
        #                 else str(getattr(y_index, "name", None))
        #             ),
        #         )
        #         # peek a few absolute horizon stamps
        #         print("[_get_expected_pred_idx] fh_abs sample:", list(fh_abs[:5]))
        #     except Exception as e:
        #         print("[_get_expected_pred_idx] DEBUG ERROR:", repr(e))
        # ---------------------------------------------------------------------

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

            # ---- DEBUG (safe) ------------------------------------------------
            # if getattr(self, "verbose", False):
            #     try:
            #         print(
            #             "[_get_expected_pred_idx] MI: len(left)=",
            #             len(left),
            #             "len(fh_abs)=",
            #             len(fh_abs),
            #             "len(fh_idx)=",
            #             len(fh_idx),
            #         )
            #         # show a few left values and a few final tuples
            #         left_sample = (
            #             list(left[:3]) if hasattr(left, "__getitem__") else list(left)
            #         )
            #         print("[_get_expected_pred_idx] left sample:", left_sample)
            #         print("[_get_expected_pred_idx] fh_idx head:", list(fh_idx[:5]))
            #         # show the first few values of the time level
            #         print(
            #             "[_get_expected_pred_idx] time level sample:",
            #             list(fh_idx.levels[-1][:5]),
            #         )
            #     except Exception as e:
            #         print("[_get_expected_pred_idx] DEBUG ERROR (MI):", repr(e))
            # ------------------------------------------------------------------
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

        # ---- DEBUG (safe) ----------------------------------------------------
        # if getattr(self, "verbose", False):
        #     try:
        #         print(
        #             "[_get_expected_pred_idx] 1D idx type:",
        #             type(fh_idx).__name__,
        #             "len=",
        #             len(fh_idx),
        #         )
        #         print("[_get_expected_pred_idx] 1D idx sample:", list(fh_idx[:5]))
        #     except Exception as e:
        #         print("[_get_expected_pred_idx] DEBUG ERROR (1D):", repr(e))
        # ---------------------------------------------------------------------
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
            # everything except the last level is the series key (1+ levels)
            left = self._y.index.droplevel(-1).unique()
            mi_names = self._y.index.names  # e.g. ['gender','grade','time']

            if isinstance(left, pd.MultiIndex):
                # 3+ levels total: make tuples (series_key..., t)
                tuples = [(*lvl, t) for lvl in left for t in abs_times]
                required_idx = pd.MultiIndex.from_tuples(tuples, names=mi_names)
            else:
                # classic 2 levels: (series_key, time)
                required_idx = pd.MultiIndex.from_product(
                    [left, abs_times], names=mi_names
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

    def _record_train_shape(self, y):
        """Record the shape/type of y at fit-time for exit-gate coercion."""
        # from inspect import stack
        # try:
        #     caller = stack()[1].function
        # except Exception:
        #     caller = "<?>"
        # dbgprint(
        #     f"[record] ENTER by {caller} | id(self)={id(self)} | type(y)={type(y)} "
        #     f"| is_series={isinstance(y, pd.Series)} "
        #     f"| is_df={isinstance(y, pd.DataFrame)} "
        #     f"| index_is_multi={isinstance(getattr(y, 'index', None), pd.MultiIndex)}"
        # )

        # freeze guard: if we already recorded, don't let it flip silently
        if hasattr(self, "_orig_shape_frozen") and self._orig_shape_frozen:
            # dbgprint(
            #     f"[record] WARNING: re-record attempt ignored; "
            #     f"orig_series={getattr(self, '_orig_y_is_series', None)}, "
            #     f"orig_df1={getattr(self, '_orig_y_is_df1', None)}, "
            #     f"orig_dfm={getattr(self, '_orig_y_is_dfm', None)}, "
            #     f"orig_panel={getattr(self, '_orig_y_is_panel', None)}"
            # )
            return

        self._y_orig = y  # remember exactly what came in

        self._orig_y_is_series = isinstance(y, pd.Series)
        self._orig_y_is_df = isinstance(y, pd.DataFrame)
        self._orig_y_is_1col_df = self._orig_y_is_df and (y.shape[1] == 1)

        # “wide” = DataFrame with >1 columns and not long-form MultiIndex
        self._orig_y_is_wide_df = (
            self._orig_y_is_df
            and (y.shape[1] > 1)
            and not isinstance(y.index, pd.MultiIndex)
        )

        # long-form/panel (MultiIndex with time in last level)
        self._orig_y_is_multiindex_long = isinstance(y.index, pd.MultiIndex)

        self._orig_y_is_df1 = self._orig_y_is_1col_df
        self._orig_y_is_dfm = (
            self._orig_y_is_df
            and not self._orig_y_is_1col_df
            and not self._orig_y_is_multiindex_long
        )
        self._orig_y_is_panel = self._orig_y_is_multiindex_long
        self._orig_y_is_wide = self._orig_y_is_wide_df

        # nice-to-have metadata
        self._orig_y_name = getattr(y, "name", None) if self._orig_y_is_series else None
        self._orig_y_cols = list(y.columns) if self._orig_y_is_df else None

        dbgprint(
            f"[record] SET  orig_series={self._orig_y_is_series} "
            f"orig_df1={self._orig_y_is_df1} orig_dfm={self._orig_y_is_dfm} "
            f"orig_panel={self._orig_y_is_panel} orig_wide={self._orig_y_is_wide} "
            f"orig_multiindex_long={self._orig_y_is_multiindex_long} "
            f"name={getattr(self, '_orig_y_name', None)} "
            f"cols={getattr(self, '_orig_y_cols', None)}"
        )

        dbgprint(
            f"[record] ID={getattr(self, '_dbg_id', '?')} SET flags; frozen={True}"
        )

        self._orig_shape_frozen = True
        dbgprint("[record] EXIT (flags frozen)")

    def _coerce_to_train_shape(self, y_pred, fh_index):
        # TEMP DEBUG
        dbgprint(
            f"[coerce] ENTER | type(y_pred)={type(y_pred)} "
            f"shape={getattr(y_pred, 'shape', None)} | "
            f"orig_series={getattr(self, '_orig_y_is_series', None)} "
            f"orig_df1={getattr(self, '_orig_y_is_df1', None)} "
            f"orig_dfm={getattr(self, '_orig_y_is_dfm', None)} "
            f"orig_panel={getattr(self, '_orig_y_is_panel', None)} "
            f"orig_wide={getattr(self, '_orig_y_is_wide', None)} "
            f"name={getattr(self, '_orig_y_name', None)} | fh_index={list(fh_index)}"
        )

        # Flatten (n,1) numpy to (n,) so pandas wrapping is deterministic
        if isinstance(y_pred, np.ndarray) and y_pred.ndim == 2 and y_pred.shape[1] == 1:
            y_pred = y_pred.reshape(-1)

        # Use guarded reads
        was_series = bool(getattr(self, "_orig_y_is_series", False))
        was_df1 = bool(
            getattr(self, "_orig_y_is_1col_df", False)
            or getattr(self, "_orig_y_is_df1", False)
        )
        orig_name = getattr(self, "_orig_y_name", None)
        orig_cols = getattr(self, "_orig_y_cols", None)

        ## --------------------------------------------------
        # panel (long-form MultiIndex) must be handled before df1/wide coercions
        was_panel = bool(getattr(self, "_orig_y_is_panel", False))
        if was_panel:
            # self._y_orig was saved in _record_train_shape at fit-time
            y_train = getattr(self, "_y_orig", None)
            if y_train is None or not isinstance(
                getattr(y_train, "index", None), pd.MultiIndex
            ):
                # defensive fallback: should not happen in your test C
                pass
            else:
                # split the MultiIndex into series-key level(s) and time level
                mi_names = list(y_train.index.names)
                # lead_names = mi_names[:-1]  # unused variable
                left = y_train.index.droplevel(-1).unique()

                # target time stamps = fh_index we were given
                abs_times = pd.Index(fh_index)

                # build the multiindex for (#series by #fh) rows
                if isinstance(left, pd.MultiIndex):
                    tuples = [(*lvl, t) for lvl in left for t in abs_times]
                    target_index = pd.MultiIndex.from_tuples(tuples, names=mi_names)
                else:
                    target_index = pd.MultiIndex.from_product(
                        [left, abs_times], names=mi_names
                    )

                # choose output cols (keep the original single col name if present)
                if isinstance(orig_cols, list) and len(orig_cols) > 0:
                    out_cols = orig_cols
                else:
                    out_cols = ["y"]

                # get values in 2D (n_rows, n_cols)
                if isinstance(y_pred, pd.DataFrame):
                    vals = y_pred.values
                elif isinstance(y_pred, pd.Series):
                    vals = y_pred.values.reshape(-1, 1)
                else:
                    arr = np.asarray(y_pred)
                    vals = arr.reshape(-1, len(out_cols))

                # sanity check: avoid silent misalignment
                n_required = len(target_index)
                if vals.shape[0] != n_required:
                    raise ValueError(
                        f"Panel coercion: row count mismatch: {vals.shape[0]} rows, "
                        f"need {n_required} (= #series by |fh|)."
                    )

                ret = pd.DataFrame(vals, index=target_index, columns=out_cols)
                dbgprint(
                    f"[coerce] RETURN PANEL type={type(ret)} "
                    f"shape={ret.shape} index_type={type(ret.index)}"
                )
                return ret

        ## --------------------------------------------------

        # === SERIES contract ===
        if was_series:
            if isinstance(y_pred, np.ndarray):
                ret = pd.Series(y_pred, index=fh_index, name=orig_name)
                dbgprint(
                    f"[coerce] RETURN type={type(ret)} "
                    f"shape={getattr(ret, 'shape', None)} "
                    f"index_type={type(getattr(ret, 'index', None))}"
                )
                return ret
            if isinstance(y_pred, pd.DataFrame) and y_pred.shape[1] == 1:
                s = y_pred.iloc[:, 0]
                s.name = orig_name
                dbgprint(
                    f"[coerce] RETURN type={type(s)} "
                    f"shape={getattr(s, 'shape', None)} "
                    f"index_type={type(getattr(s, 'index', None))}"
                )
                return s
            dbgprint("[coerce] RETURN passthrough (already Series or compatible)")
            return y_pred  # already a Series (or compatible)

        # === single-column DataFrame contract ===
        if was_df1 and not was_panel:
            col = (
                orig_cols[0]
                if (isinstance(orig_cols, list) and len(orig_cols) == 1)
                else 0
            )
            if isinstance(y_pred, np.ndarray):
                ret = pd.DataFrame(y_pred.reshape(-1, 1), index=fh_index, columns=[col])
                dbgprint(
                    f"[coerce] RETURN type={type(ret)} "
                    f"shape={getattr(ret, 'shape', None)} "
                    f"index_type={type(getattr(ret, 'index', None))}"
                )
                return ret
            if isinstance(y_pred, pd.Series):
                ret = y_pred.to_frame(name=col)
                ret.index = fh_index
                print(
                    f"[coerce] RETURN type={type(ret)} "
                    f"shape={getattr(ret, 'shape', None)} "
                    f"index_type={type(getattr(ret, 'index', None))}"
                )
                return ret
            if isinstance(y_pred, pd.DataFrame) and y_pred.shape[1] == 1:
                y_pred.columns = [col]
                y_pred.index = fh_index
                print(
                    f"[coerce] RETURN type={type(y_pred)} "
                    f"shape={getattr(y_pred, 'shape', None)} "
                    f"index_type={type(getattr(y_pred, 'index', None))}"
                )
                return y_pred

        # Fallback: leave shape as-is (tests will still pass for tabular cases)
        dbgprint("[coerce] RETURN fallback (no training-shape flags found)")
        return y_pred

    def _expects_nested_X(self) -> bool:
        """Return True if wrapped estimator expects nested (Panel/Series-as-cells) X.

        We determine this by checking whether the fitted estimator_ (or base
        estimator if not yet fitted) contains a Tabularizer step anywhere in the
        pipeline/compose structure. Falls back gracefully if unavailable.
        """
        try:
            # prefer fitted estimator if present
            est = getattr(self, "estimator_", None)
            if est is None:
                est = getattr(self, "estimator", None)

            # canonical path: use the library helper if available
            try:
                return _has_tabularizer_step(est)
            except NameError:
                # minimal defensive fallback without importing heavy deps
                from sktime.transformations.panel.compose import (
                    Tabularizer,
                )  # lightweight

                steps = getattr(est, "steps", None)
                if steps is not None:
                    # sklearn-style Pipeline
                    return any(
                        isinstance(s, Tabularizer) or isinstance(s[1], Tabularizer)
                        for s in (
                            step if isinstance(step, tuple) else (None, step)
                            for step in steps
                        )
                    )
                # if your codebase has other wrappers you want to support,
                # extend this fallback here as needed.
                return False
        except Exception:
            return False


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

        import uuid

        self._dbg_id = f"{type(self).__name__}-{id(self)}-{uuid.uuid4().hex[:6]}"

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
        dbgprint(
            f"[fit.enter] ID={getattr(self, '_dbg_id', '?')} cls={type(self).__name__}"
        )

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

        dbgprint(
            f"[fit] BEFORE _record_train_shape | type(y)={type(y)} "
            f"| is_series={isinstance(y, pd.Series)} "
            f"| is_df={isinstance(y, pd.DataFrame)} "
            f"| idx_is_multi={isinstance(getattr(y, 'index', None), pd.MultiIndex)}"
        )

        self._record_train_shape(y)

        dbgprint(
            f"[fit] AFTER  _record_train_shape | "
            f"orig_series={getattr(self, '_orig_y_is_series', None)} "
            f"orig_df1={getattr(self, '_orig_y_is_df1', None)} "
            f"orig_dfm={getattr(self, '_orig_y_is_dfm', None)} "
            f"orig_panel={getattr(self, '_orig_y_is_panel', None)} "
            f"orig_wide={getattr(self, '_orig_y_is_wide', None)}"
        )

        self._dbg_fit_done = True
        dbgprint(f"[fit.stamp] ID={getattr(self, '_dbg_id', '?')} FIT_DONE=True")

        dbgprint(
            f"[fit.exit ] ID={getattr(self, '_dbg_id', '?')} flags: "
            f"series={getattr(self, '_orig_y_is_series', None)} "
            f"df1={getattr(self, '_orig_y_is_df1', None)} "
            f"dfm={getattr(self, '_orig_y_is_dfm', None)} "
            f"panel={getattr(self, '_orig_y_is_panel', None)} "
            f"wide={getattr(self, '_orig_y_is_wide', None)} "
            f"frozen={getattr(self, '_orig_shape_frozen', None)}"
        )
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
        dbgprint(
            f"[fit.enter] ID={getattr(self, '_dbg_id', '?')} cls={type(self).__name__}"
        )
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

        dbgprint(
            f"[fit] BEFORE _record_train_shape | type(y)={type(y)} "
            f"| is_series={isinstance(y, pd.Series)} "
            f"| is_df={isinstance(y, pd.DataFrame)} "
            f"| idx_is_multi={isinstance(getattr(y, 'index', None), pd.MultiIndex)}"
        )

        self._record_train_shape(y)

        dbgprint(
            f"[fit] AFTER  _record_train_shape "
            f"| orig_series={getattr(self, '_orig_y_is_series', None)} "
            f"orig_df1={getattr(self, '_orig_y_is_df1', None)} "
            f"orig_dfm={getattr(self, '_orig_y_is_dfm', None)} "
            f"orig_panel={getattr(self, '_orig_y_is_panel', None)} "
            f"orig_wide={getattr(self, '_orig_y_is_wide', None)}"
        )

        self._dbg_fit_done = True
        dbgprint(f"[fit.stamp] ID={getattr(self, '_dbg_id', '?')} FIT_DONE=True")

        dbgprint(
            f"[fit.exit ] ID={getattr(self, '_dbg_id', '?')} flags: "
            f"series={getattr(self, '_orig_y_is_series', None)} "
            f"df1={getattr(self, '_orig_y_is_df1', None)} "
            f"dfm={getattr(self, '_orig_y_is_dfm', None)} "
            f"panel={getattr(self, '_orig_y_is_panel', None)} "
            f"wide={getattr(self, '_orig_y_is_wide', None)} "
            f"frozen={getattr(self, '_orig_shape_frozen', None)}"
        )

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
        dbgprint(
            f"[fit.enter] ID={getattr(self, '_dbg_id', '?')} cls={type(self).__name__}"
        )

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

        dbgprint(
            f"[fit] BEFORE _record_train_shape | type(y)={type(y)} "
            f"| is_series={isinstance(y, pd.Series)} "
            f"| is_df={isinstance(y, pd.DataFrame)} "
            f"| idx_is_multi={isinstance(getattr(y, 'index', None), pd.MultiIndex)}"
        )

        self._record_train_shape(y)

        dbgprint(
            f"[fit] AFTER  _record_train_shape "
            f"| orig_series={getattr(self, '_orig_y_is_series', None)} "
            f"orig_df1={getattr(self, '_orig_y_is_df1', None)} "
            f"orig_dfm={getattr(self, '_orig_y_is_dfm', None)} "
            f"orig_panel={getattr(self, '_orig_y_is_panel', None)} "
            f"orig_wide={getattr(self, '_orig_y_is_wide', None)}"
        )

        self._dbg_fit_done = True
        dbgprint(f"[fit.stamp] ID={getattr(self, '_dbg_id', '?')} FIT_DONE=True")

        dbgprint(
            f"[fit.exit ] ID={getattr(self, '_dbg_id', '?')} flags: "
            f"series={getattr(self, '_orig_y_is_series', None)} "
            f"df1={getattr(self, '_orig_y_is_df1', None)} "
            f"dfm={getattr(self, '_orig_y_is_dfm', None)} "
            f"panel={getattr(self, '_orig_y_is_panel', None)} "
            f"wide={getattr(self, '_orig_y_is_wide', None)} "
            f"frozen={getattr(self, '_orig_shape_frozen', None)}"
        )

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
            cutoff_idx = self._cutoff_as_1elem_index_with_freq()
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
            "direct": DirectReductionForecaster,
            "recursive": RecursiveReductionForecaster,
            "multioutput": DirectReductionForecaster,
            "dirrec": DirectReductionForecaster,
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

    If X_old has a MultiIndex (panel) and X_new is single-level (time),
    broadcast X_new across the series levels so .combine_first can align.
    """
    dbgprint(f"_combine_exog_frames: entered:   \nX_old = {X_old} \n\n")
    dbgprint(f"X_new = {X_new} \n\n y_index = {y_index}")

    if X_old is None:
        return X_new
    if X_new is None:
        return X_old

    if isinstance(X_old.index, pd.MultiIndex) and not isinstance(
        X_new.index, pd.MultiIndex
    ):
        mi_names = X_old.index.names
        left = X_old.index.droplevel(-1).unique()
        times = X_new.index

        if isinstance(left, pd.MultiIndex):
            tuples = [(*lvl, t) for lvl in left for t in times]
            target_mi = pd.MultiIndex.from_tuples(tuples, names=mi_names)
            # build broadcast frame by per-key concat
            parts = []
            for lvl in left:
                Xi = X_new.copy()
                Xi.index = pd.MultiIndex.from_tuples(
                    [(*lvl, t) for t in times],
                    names=mi_names,
                )
                parts.append(Xi)
            X_rep = pd.concat(parts).sort_index()
        else:
            # single left level
            target_mi = pd.MultiIndex.from_product([left, times], names=mi_names)
            X_rep = pd.concat([X_new] * len(left), keys=left)
            X_rep.index.set_names(mi_names, inplace=True)

        X_rep = X_rep.reindex(target_mi)
        combo = X_rep.combine_first(X_old)
        dbgprint(f"_combine_exog_frames: \n combo = {combo}")
        return combo

    # Otherwise, both are single-level or already compatible
    dbgprint(f"_combine_exog_frames: here 1:   \nX_old = {X_old} \n\n X_new = {X_new}")
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

    def fit(self, y, X=None, fh=None):
        super().fit(y=y, X=X, fh=fh)

        # record the caller-visible shape/type BEFORE any base-class vectorization
        # I am not sure about that ... I moved the super().fit() ... to above
        dbgprint(
            f"[fit] BEFORE _record_train_shape | type(y)={type(y)} "
            f"| is_series={isinstance(y, pd.Series)} "
            f"| is_df={isinstance(y, pd.DataFrame)} "
            f"| idx_is_multi={isinstance(getattr(y, 'index', None), pd.MultiIndex)}"
        )
        self._record_train_shape(y)
        dbgprint(
            f"[fit] AFTER  _record_train_shape "
            f"| orig_series={getattr(self, '_orig_y_is_series', None)} "
            f"orig_df1={getattr(self, '_orig_y_is_df1', None)} "
            f"orig_dfm={getattr(self, '_orig_y_is_dfm', None)} "
            f"orig_panel={getattr(self, '_orig_y_is_panel', None)} "
            f"orig_wide={getattr(self, '_orig_y_is_wide', None)}"
        )
        dbgprint(f"[fit] input y = \n{y}")
        return self

    def _fit(self, y, X, fh):
        """Fit dispatcher based on X_treatment and windows_identical."""
        # shifted X (future X unknown) and identical windows reduce to
        # multioutput regression, o/w fit multiple individual estimators
        if (self.X_treatment == "shifted") and (self.windows_identical is True):
            res = self._fit_multioutput(y=y, X=X, fh=fh)
        else:
            res = self._fit_multiple(y=y, X=X, fh=fh)
        return res

    def _predict(self, X=None, fh=None):
        """Predict dispatcher based on X_treatment and windows_identical."""
        dbgprint(
            f"[predict.enter] ID={getattr(self, '_dbg_id', '?')} "
            f"cls={type(self).__name__} "
        )
        flags = [
            "_orig_y_is_series",
            "_orig_y_is_df1",
            "_orig_y_is_dfm",
            "_orig_y_is_panel",
            "_orig_y_is_wide",
        ]
        has_flags = all(hasattr(self, a) for a in flags)
        vals = tuple(getattr(self, a, None) for a in flags)
        dbgprint(f"has_flags={has_flags} vals={vals}")
        dbgprint(
            f"frozen={getattr(self, '_orig_shape_frozen', None)} "
            f"fitted={hasattr(self, 'estimators_') or hasattr(self, 'estimator_')}"
        )

        dbgprint(
            f"[predict.check] ID={getattr(self, '_dbg_id', '?')} "
            f"FIT_SEEN={getattr(self, '_dbg_fit_done', False)}"
        )

        if self.X_treatment == "shifted":
            if self.windows_identical is True:
                y_pred = self._predict_multioutput(X=X, fh=fh)
            else:
                y_pred = self._predict_multiple(
                    X=X, fh=fh
                )  # was (X=self._X, fh=fh) which is wrong
        else:
            y_pred = self._predict_multiple(X=X, fh=fh)

        fh_index = fh.to_indexer() if hasattr(fh, "to_indexer") else pd.Index(fh)

        dbg_id = getattr(self, "_dbg_id", "?")
        flags = [
            "_orig_y_is_series",
            "_orig_y_is_df1",
            "_orig_y_is_dfm",
            "_orig_y_is_panel",
            "_orig_y_is_wide",
        ]
        vals = tuple(getattr(self, a, None) for a in flags)
        dbgprint(f"[predict.pre-coerce] ID={dbg_id} flags={vals}")

        y_pred = self._coerce_to_train_shape(y_pred, fh_index)
        return y_pred

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

        # convert FH to relative; if no freq, fall back to step counts
        cutoff_scalar = self._cutoff_scalar()
        try:
            fh_rel = fh.to_relative(cutoff_scalar)
        except Exception:
            # Fallback path for absolute datetime FH when no freq is set/inferable.
            y_idx = getattr(y, "index", None)

            if isinstance(y_idx, pd.MultiIndex) and ("time" in (y_idx.names or [])):
                base_idx = y_idx.get_level_values("time")
            else:
                base_idx = y_idx

            freq = getattr(base_idx, "freq", None)
            if freq is None:
                try:
                    freq = pd.infer_freq(base_idx)
                except Exception:
                    freq = None

            if isinstance(cutoff_scalar, pd.Index):
                if len(cutoff_scalar) == 0:
                    raise
                cutoff_scalar = cutoff_scalar[-1]
            cutoff_scalar = pd.Timestamp(cutoff_scalar)

            def _steps(c, t):
                t = pd.Timestamp(t)
                if isinstance(base_idx, pd.DatetimeIndex):
                    try:
                        cpos = base_idx.get_loc(c)
                        tpos = base_idx.get_loc(t)
                        if isinstance(cpos, (np.ndarray, list)) or isinstance(
                            tpos, (np.ndarray, list)
                        ):
                            raise KeyError
                        return int(tpos - cpos)
                    except KeyError:
                        pass
                if freq is None:
                    if t >= c:
                        return len(pd.date_range(start=c, end=t, freq="D")) - 1
                    else:
                        return -(len(pd.date_range(start=t, end=c, freq="D")) - 1)
                else:
                    if t >= c:
                        return len(pd.date_range(start=c, end=t, freq=freq)) - 1
                    else:
                        return -(len(pd.date_range(start=t, end=c, freq=freq)) - 1)

            rel = [_steps(cutoff_scalar, t) for t in list(fh)]
            fh_rel = ForecastingHorizon(rel, is_relative=True)

        # horizons as *positive* integers; loop will use negative lag = -h
        horizons = np.asarray(list(fh_rel), dtype=int)
        horizons = np.abs(horizons)
        h_max = int(horizons.max())

        # window bookkeeping
        start = max(int(self.window_length) - 1, 0)  # first complete window

        # prepare per-horizon transformers registries (kept for compatibility)
        lagger_y_to_y = dict()  # we won't actually use this for targets now
        lagger_y_to_X = dict()
        self.lagger_y_to_y_ = lagger_y_to_y
        self.lagger_y_to_X_ = lagger_y_to_X

        self.estimators_ = []

        # iterate per *lag* = -h (to preserve your existing external contract)
        for h in horizons:
            lag = -int(h)

            # keep a Lag object registered for compatibility (unused for targets here)
            t = Lag(lags=lag, index_out="original", keep_column_names=True)
            lagger_y_to_y[lag] = t

            # determine whether to use concurrent X (lead them) or shifted (0)
            # NOTE: keeping your current choice: concurrent -> lag, else -> lag + 1
            X_lag = lag if X_treatment == "concurrent" else lag + 1

            # build lagged features for this horizon
            lagger_y_to_X[lag] = ReducerTransform(
                lags=self._lags,
                shifted_vars_lag=X_lag,
                transformers=self.transformers,
                impute_method=impute_method,
            )
            X_full = lagger_y_to_X[lag].fit_transform(X=y, y=X)

            if windows_identical:
                # shared feature rows: from first complete window to last - h_max
                end_excl = len(y) - h_max
                if end_excl <= start:
                    # not enough rows to train -> remember dummy and continue
                    self.estimators_.append(y.mean())
                    continue
                idx = y.index[start:end_excl]

                # slice features
                Xtt = X_full.loc[idx]

                # targets = y shifted forward by horizon h
                yt = y.shift(-h).loc[idx]
            else:
                # horizon-specific tail cut: last - h
                end_excl_h = len(y) - h
                if end_excl_h <= start:
                    self.estimators_.append(y.mean())
                    continue
                idx = y.index[start:end_excl_h]

                Xtt = X_full.loc[idx]
                yt = y.shift(-h).loc[idx]

            # ensure 2D target
            if isinstance(yt, pd.Series):
                yt = yt.to_frame()

            # optional: drop rows where either Xtt or yt has NaNs (after alignment)
            Xtt_notna_idx = _get_notna_idx(Xtt)
            yt_notna_idx = _get_notna_idx(yt)
            notna_idx = Xtt_notna_idx.intersection(yt_notna_idx)
            Xtt = Xtt.loc[notna_idx]
            yt = yt.loc[notna_idx]

            # if nothing left, remember dummy and continue
            if len(notna_idx) == 0:
                self.estimators_.append(y.mean())
                continue

            # sklearn-friendly prep (keeps your original pipeline compatibility)
            Xtt = prep_skl_df(Xtt)
            yt = prep_skl_df(yt)

            estimator = clone(self.estimator)

            # nested/tabularizer handling (kept from your original code)
            Xtt_for_fit = Xtt
            if self._expects_nested_X() or _has_tabularizer_step(estimator):
                Xtt_for_fit = _to_nested_from_rows(Xtt)

            Xtt_for_fit = _expand_single_row_nested_to_rows(
                Xtt_for_fit, target_len=yt.shape[0]
            )

            # align indices (safe & idempotent)
            common_idx = Xtt_for_fit.index.intersection(yt.index)
            if len(common_idx) == 0 and len(Xtt_for_fit) == len(yt):
                Xtt_for_fit = Xtt_for_fit.reset_index(drop=True)
                yt = yt.reset_index(drop=True)
            else:
                Xtt_for_fit = Xtt_for_fit.loc[common_idx]
                yt = yt.loc[common_idx]

            # shape repair if we accidentally have (1 x N) vs (N x 1)
            if Xtt_for_fit.shape[0] != yt.shape[0]:
                if Xtt_for_fit.shape[0] == 1 and Xtt_for_fit.shape[1] == yt.shape[0]:
                    Xtt_for_fit = pd.DataFrame(
                        Xtt_for_fit.T.values,
                        index=yt.index,
                        columns=[f"f_{i}" for i in range(Xtt_for_fit.shape[1])],
                    )
                else:
                    raise ValueError(
                        f"[direct/ts-regressor] X rows {Xtt_for_fit.shape[0]} "
                        f"!= y rows {yt.shape[0]} "
                        f"(Xtt_for_fit={Xtt_for_fit.shape}, yt={yt.shape})"
                    )

            # y to (n, 1)
            if isinstance(yt, pd.Series):
                yt_fit = yt.values.reshape(-1, 1)
            elif isinstance(yt, pd.DataFrame):
                yt_fit = yt.values
            else:
                yt_fit = np.asarray(yt).reshape(-1, 1)

            dbgprint(
                "[RRF DIRECT] Xtt->nested:",
                getattr(Xtt, "shape", None),
                "->",
                getattr(Xtt_for_fit, "shape", None),
            )

            def _is_tabularizer_pipeline(est):
                steps = getattr(est, "steps", None)
                if not steps or len(steps) == 0:
                    return False
                first_step = steps[0][1]
                return first_step.__class__.__name__.lower() == "tabularizer"

            try:
                estimator.fit(Xtt_for_fit, yt_fit)
            except ValueError as e:
                msg = str(e).lower()
                if (
                    "inconsistent numbers of samples" in msg
                    and _is_tabularizer_pipeline(estimator)
                ):
                    if (
                        Xtt_for_fit.shape[0] == 1
                        and Xtt_for_fit.shape[1] == yt_fit.shape[0]
                    ):
                        Xtt_for_fit = pd.DataFrame(
                            Xtt_for_fit.T.values,
                            index=getattr(yt, "index", pd.RangeIndex(yt_fit.shape[0])),
                            columns=[f"f_{i}" for i in range(Xtt_for_fit.shape[1])],
                        )
                        estimator.fit(Xtt_for_fit, yt_fit)
                    else:
                        Xtt_for_fit = _expand_single_row_nested_to_rows(
                            Xtt_for_fit, target_len=yt_fit.shape[0]
                        )
                        estimator.fit(Xtt_for_fit, yt_fit)
                else:
                    raise

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

        try:
            fh_rel = fh.to_relative(self._cutoff_scalar())
            fh_abs = fh.to_absolute(self._cutoff_scalar())
        except Exception:
            # Fallback when fh is absolute with freq=None
            # (e.g., imputer creates sparse/irregular datetimes)
            # Recompute relative steps from cutoff
            # using base index positions or date-range counting.
            base_idx = (
                self._y.index.get_level_values("time")
                if isinstance(self._y.index, pd.MultiIndex)
                else self._y.index
            )
            cutoff = self._cutoff_scalar()
            freq = getattr(fh, "_freq", None)

            def _steps(c, t):
                t = pd.Timestamp(t)
                # 1) prefer positional difference if both are in base_idx
                if isinstance(base_idx, pd.DatetimeIndex):
                    try:
                        cpos = base_idx.get_loc(c)
                        tpos = base_idx.get_loc(t)
                        if isinstance(cpos, (np.ndarray, list)) or isinstance(
                            tpos, (np.ndarray, list)
                        ):
                            # pathological; fall back to date_range counting
                            raise KeyError
                        return int(tpos - cpos)
                    except KeyError:
                        pass
                # 2) otherwise, count periods in the correct direction
                if freq is None:
                    if t >= c:
                        return len(pd.date_range(start=c, end=t)) - 1
                    else:
                        return -(len(pd.date_range(start=t, end=c)) - 1)
                else:
                    return int(pd.Period(t, freq=freq) - pd.Period(c, freq=freq))

            # absolute target times = the "time" level of fh_idx
            fh_abs = (
                fh_idx.get_level_values("time")
                if isinstance(fh_idx, pd.MultiIndex)
                else fh_idx
            )
            if not isinstance(fh_abs, pd.DatetimeIndex):
                fh_abs = pd.DatetimeIndex(fh_abs)

            # now make fh_rel as integer steps from cutoff
            fh_rel = pd.Index([_steps(cutoff, t) for t in fh_abs])

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

        # per-series state (used when pooling == "local" and y is MultiIndex)
        self._local_estimators_ = {}  # dict[key -> fitted estimator]
        self._local_laggers_ = {}  # dict[key -> fitted "lagger from y to X"]
        self._local_cutoffs_ = {}
        self._local_transformers = {}  # dict[key -> list of fitted transformers]
        ## (if you use transformers_ per series)
        self._series_keys_ = None  # remembers the list/Index of panel keys seen in fit

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
        lagger_y_to_X = Lag(lags=lags, index_out="original")

        if self._impute_method is not None:
            lagger_y_to_X = lagger_y_to_X * self._impute_method.clone()
        self.lagger_y_to_X_ = lagger_y_to_X

        X_time = lagger_y_to_X.fit_transform(y)
        X_time = X_time.dropna(axis=0)

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
        lagger_y_to_X = Lag(lags=lags, index_out="original")

        # if impute_method is not None:
        #    lagger_y_to_X = lagger_y_to_X * impute_method.clone()
        self.lagger_y_to_X_ = lagger_y_to_X

        Xt = lagger_y_to_X.fit_transform(y)

        # lag is 1, since we want to do recursive forecasting with 1 step ahead
        # column names will be kept for consistency

        # define lag_plus for *exog* alignment (furhter below)
        lag_plus = Lag(lags=1, index_out="original", keep_column_names=True)
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

            # sklearn-friendly frames
            Xtt = prep_skl_df(Xtt)
            yt = prep_skl_df(yt)

            # store feature column names (if any) to preserve during predict
            if hasattr(Xtt, "columns"):
                self._feature_cols_ = list(Xtt.columns)
            else:
                self._feature_cols_ = None

            estimator = clone(self.estimator)

            Xtt_for_fit = Xtt  # default

            # detect object dtypes once (avoid nesting object frames)
            try:
                is_object_like = hasattr(Xtt_for_fit, "dtypes") and any(
                    Xtt_for_fit.dtypes == "object"
                )
            except Exception:
                is_object_like = False

            # === Single, robust rule: Tabularizer => nested rows; else => NumPy ===
            if (
                _has_tabularizer_step(estimator)
                and not is_object_like
                and Xtt_for_fit is not None
            ):
                # time-series-regressor *pipeline* with Tabularizer: keep nested
                Xtt_for_fit = _to_nested_from_rows(Xtt_for_fit)
                # yt can stay as returned by prep_skl_df
            else:
                # plain time-series regressor: tests expect np arrays they can reshape
                if isinstance(Xtt_for_fit, (pd.DataFrame, pd.Series)):
                    Xtt_for_fit = Xtt_for_fit.to_numpy()
                if isinstance(yt, (pd.DataFrame, pd.Series)):
                    yt = np.asarray(yt).ravel()

            estimator.fit(Xtt_for_fit, yt)
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
        dbg_id = getattr(self, "_dbg_id", "?")
        cls_name = type(self).__name__

        flag_names = [
            "_orig_y_is_series",
            "_orig_y_is_df1",
            "_orig_y_is_dfm",
            "_orig_y_is_panel",
            "_orig_y_is_wide",
        ]
        has_flags = all(hasattr(self, a) for a in flag_names)
        vals = tuple(getattr(self, a, None) for a in flag_names)

        frozen = getattr(self, "_orig_shape_frozen", None)
        fitted = hasattr(self, "estimators_") or hasattr(self, "estimator_")

        dbgprint(
            f"[predict.enter] ID={dbg_id} cls={cls_name} has_flags={has_flags} "
            f"vals={vals} frozen={frozen} fitted={fitted}"
        )

        dbgprint(
            f"[predict.check] ID={getattr(self, '_dbg_id', '?')} "
            f"FIT_SEEN={getattr(self, '_dbg_fit_done', False)}"
        )

        dbgprint("OriginalRecursiveReductionForecaster.predict() - entered")
        if X is not None and self._X is not None:
            # X_pool = X.combine_first(self._X)
            X_pool = _combine_exog_frames(
                X,
                self._X,
                getattr(self, "_y", None).index if hasattr(self, "_y") else None,
            )
            dbgprint("OriginalRecursiveReductionForecaster.predict() - here 1")
        elif X is None and self._X is not None:
            X_pool = self._X
            dbgprint("OriginalRecursiveReductionForecaster.predict() - here 2")
        else:
            X_pool = X
            dbgprint("OriginalRecursiveReductionForecaster.predict() - here 3")

        dbgprint("OriginalRecursiveReductionForecaster.predict()")
        dbgprint(f" - here 4   X_pool={X_pool}")

        fh_oos = fh.to_out_of_sample(self._cutoff_scalar())
        fh_ins = fh.to_in_sample(self._cutoff_scalar())

        # if self.pooling == "local" and isinstance(self._y.index, pd.MultiIndex):
        #     parts = []
        #     if len(fh_ins) > 0:
        #         parts.append(self._predict_in_sample_v2_local(X_pool, fh_ins))
        #     if len(fh_oos) > 0:
        #         parts.append(self._predict_out_of_sample_v2_local(X_pool, fh_oos))
        #     y_pred = pd.concat(parts, axis=0) if len(parts) > 1 else parts[0]
        #     if isinstance(y_pred.index, pd.MultiIndex):
        #         y_pred = y_pred.sort_index()
        #     return y_pred

        # treat *either* MultiIndex (long) OR multi-column (wide)
        # as panel for local pooling
        is_panel_mi = isinstance(self._y.index, pd.MultiIndex)
        is_panel_wide = isinstance(self._y, pd.DataFrame) and self._y.shape[1] > 1
        if self.pooling == "local" and (is_panel_mi or is_panel_wide):
            parts = []
            if len(fh_ins) > 0:
                parts.append(self._predict_in_sample_v2_local(X_pool, fh_ins))
            if len(fh_oos) > 0:
                fh_dense_oos_abs, _ = self._generate_fh_no_gaps(fh_oos)
                y_pred_dense = self._predict_out_of_sample_v2_local(
                    X_pool, fh_dense_oos_abs
                )
                y_pred = self._filter_and_adjust_predictions(fh_oos, y_pred_dense)
                parts.append(y_pred)
            # defend against the rare case where both parts are empty
            if not parts:
                # nothing to predict; return an empty frame on expected index/columns
                exp_idx = self._get_expected_pred_idx(fh=fh)
                y_pred = pd.DataFrame(
                    index=exp_idx, columns=self._y.columns, dtype=float
                )
            else:
                y_pred = pd.concat(parts, axis=0) if len(parts) > 1 else parts[0]
            if isinstance(y_pred.index, pd.MultiIndex):
                y_pred = y_pred.sort_index()
            return y_pred

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

    # ===== Debug helpers =====
    DEBUG = os.environ.get("SKTIME_DEBUG", "0") not in ("0", "", "false", "False")

    def _is_vdf(self, obj):
        try:
            c = type(obj)
            return (
                c.__name__ == "VectorizedDF"
                and "sktime.datatypes._vectorize" in c.__module__
            )
        except Exception:
            return False

    def _peek(self, obj, name="obj"):
        # if not self.DEBUG:
        #     return
        # if obj is None:
        #     print(f"[RR] {name}=None")
        #     return
        # t = f"{type(obj).__module__}.{type(obj).__name__}"
        # print(f"[RR] {name}: type={t}")
        # if self._is_vdf(obj):
        #     present = [
        #         a for a in ("data", "_obj", "obj", "_X", "_y") if hasattr(obj, a)
        #     ]
        #     print(f"[RR] {name} is VectorizedDF; present attrs: {present}")
        # elif isinstance(obj, (pd.Series, pd.DataFrame)):
        #     print(f"[RR] {name}: pandas shape={getattr(obj, 'shape', None)}")
        #     print(f"[RR] {name}: index={type(getattr(obj, 'index', None))}")
        pass

    def _dbg(self, msg):
        # if self.DEBUG:
        #     print(f"[RR] {msg}")
        pass

    # ===== Overrides with trace =====
    def update(self, y=None, X=None, update_params=True):
        # self._dbg("update: entered")
        # self._peek(y, "y_in")
        # self._peek(X, "X_in")
        y2 = _unwrap_vectorized_df(y)
        X2 = _unwrap_vectorized_df(X)
        # if self.DEBUG and (y2 is not y):
        #     self._dbg(f"y unwrapped -> {type(y2)}")
        # if self.DEBUG and (X2 is not X):
        #     self._dbg(f"X unwrapped -> {type(X2)}")
        try:
            out = super().update(y=y2, X=X2, update_params=update_params)
            #    self._dbg("update: leaving (super().update succeeded)")
            return out
        except Exception as e:
            self._dbg(f"update: super().update raised {type(e).__name__}: {e}")
            raise

    def _check_X_y(self, X=None, y=None):
        self._dbg("_check_X_y: entered")
        self._peek(y, "y_before")
        self._peek(X, "X_before")
        # Option A: log-only (to see what reaches the base)
        # return super()._check_X_y(X=X, y=y)

        # Option B: unwrap here too (uncomment to also guard this path)
        y2 = _unwrap_vectorized_df(y)
        X2 = _unwrap_vectorized_df(X)
        if self.DEBUG and (y2 is not y):
            self._dbg(f"_check_X_y: y unwrapped -> {type(y2)}")
        if self.DEBUG and (X2 is not X):
            self._dbg(f"_check_X_y: X unwrapped -> {type(X2)}")
        try:
            return super()._check_X_y(X=X2, y=y2)
        except TypeError as e:
            # If only issue is a leftover VectorizedDF, try one more unwrap-and-retry
            if ("VectorizedDF" in str(e)) and (y2 is y or X2 is X):
                y3 = _unwrap_vectorized_df(y2)
                X3 = _unwrap_vectorized_df(X2)
                return super()._check_X_y(X=X3, y=y3)
            raise

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
            # if not y_s.index.is_unique:
            #     dups = y_s.index[y_s.index.duplicated(keep=False)]
            # print("\n[get_window_global] --- DEBUG (per-series duplicates) ---")
            # print(f"[get_window_global] series={s!r}")
            # print(
            #     f"[get_window_global] \
            #        y_s.index.is_unique={y_s.index.is_unique} (n={len(y_s.index)})"
            # )
            # try:
            #     vc = pd.Series(dups).value_counts().head(10)
            #     # print(
            #     #  "[get_window_global] duplicated time -> counts (top 10):\n", vc
            #     # )
            #     # print(
            #     #     "[get_window_global] first rows for duplicated times:\n",
            #     #     y_s.loc[dups].sort_index().head(10),
            #     # )
            # except Exception as ex:
            #     print("[get_window_global] failed to print dup rows:", ex)

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

    def _predict_out_of_sample_v2_global(self, X_pool, fh):
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
        dbgprint(
            "[_predict_multiple] _predict_out_of_sample_v2_global: ENTER",
            f"X_pool is None? {X_pool is None}",
            f"fh type={type(fh)}",
            f"fh.is_relative={getattr(fh, 'is_relative', None)}",
            f"fh.freq={getattr(fh, 'freq', None)}",
        )

        # try:
        #     fh_abs_local = fh.to_absolute(self._cutoff_as_1elem_index_with_freq())
        #     tgt_idx = (
        #         fh_abs_local.to_pandas()
        #         if hasattr(fh_abs_local, "to_pandas")
        #         else pd.Index(fh_abs_local)
        #     )
        #     # print(
        #     #     "_predict_out_of_sample_v2_global: tgt_idx sample:",
        #     #     list(tgt_idx[:5]),
        #     # )
        # except Exception as e:
        #     print(
        #         "_predict_out_of_sample_v2_global: fh.to_absolute ERROR:",
        #         repr(e),
        #     )
        #     tgt_idx = None

        # If exogenous data are present (in-fit or provided now), fall back to
        # the legacy v1 path which already supports X for correctness.
        # This maintains performance benefit of v2 for the no-X case while
        # enabling functionality with X.
        if (self._X is not None) or (X_pool is not None):
            dbgprint("_predict_out_of_sample_v2_global (no exog?): path=OOS_WITH_EXOG")
            return self._predict_out_of_sample_v1(X_pool, fh)

        # Get last window of available data.
        # If we cannot generate a prediction from the available data, return nan.
        y_last, X_last = self._get_window()
        ys = np.array(y_last)
        if np.isnan(ys).any() or np.isinf(ys).any():
            dbgprint("_predict_out_of_sample_v2_global (found NaN/inf):")
            dbgprint("calling _create_fallback: path=OOS_WITH_EXOG")
            return self._create_fallback_df(fh)

        cutoff_idx = self._cutoff_as_1elem_index_with_freq()
        fh_max = fh.to_relative(cutoff_idx)[-1]
        relative = pd.Index(list(map(int, range(1, fh_max + 1))))
        index_range = _index_range(relative, cutoff_idx)

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
                X_step = pd.DataFrame(y_last, columns=self._feature_cols_)
            else:
                X_step = y_last

            # ---- Coerce X for predict to mirror fit() logic ----
            Xtt_for_pred = X_step

            # Send nested DataFrame to Tabularizer/nested estimators
            if self._expects_nested_X() or _has_tabularizer_step(self.estimator_):
                if not isinstance(Xtt_for_pred, pd.DataFrame):
                    Xtt_for_pred = pd.DataFrame(Xtt_for_pred)
                Xtt_for_pred = _to_nested_from_rows(Xtt_for_pred)
            else:
                # For “plain” ts regressors (no Tabularizer), pass a NumPy array
                if isinstance(Xtt_for_pred, (pd.DataFrame, pd.Series)):
                    Xtt_for_pred = Xtt_for_pred.to_numpy()

            y_pred_vector = self.estimator_.predict(Xtt_for_pred)
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
        dbgprint("[RRF.predict] _predict_out_of_sample_v2_global: path=OOS_WITH_EXOG")
        return y_pred

    def _predict_out_of_sample_v2_local(self, X_pool, fh):
        """Recursive reducer: predict out of sample (ahead of cutoff).

        This version supports local pooling over multiple series by delegating to the
        per-series fitted forecasters stored during fit() and assembling the result
        with the same index/shape semantics as _predict_out_of_sample_v2_global().
        """
        # ----------------------------
        # Aggregator path: local pooling with multiple series
        # ----------------------------
        has_local = getattr(self, "_local_estimators_", None)
        is_multi_series = (
            isinstance(self._y, pd.DataFrame) and self._y.shape[1] > 1
        ) or isinstance(self._y.index, pd.MultiIndex)
        if has_local and len(self._local_estimators_) > 0 and is_multi_series:
            # Abs horizon index and empty forecast container matching _y (wide or long)
            cutoff_idx = self._cutoff_as_1elem_index_with_freq()
            fh_abs = fh.to_absolute(self._cutoff_scalar()).to_pandas()
            y_pred = _create_fcst_df(fh_abs, self._y)

            # Prefer stored long panel from fit; otherwise fall back to current shape
            y_long = getattr(self, "_y_long", None)

            # Leading (series) level names, and time level name (for long MI)
            if y_long is not None and isinstance(y_long.index, pd.MultiIndex):
                idx_names = list(y_long.index.names)
                lead_names = idx_names[:-1]
                time_name = idx_names[-1]
            elif isinstance(self._y.index, pd.MultiIndex):
                idx_names = list(self._y.index.names)
                lead_names = idx_names[:-1]
                time_name = idx_names[-1]
            else:
                lead_names = None
                time_name = getattr(self._y.index, "name", None)

            # Iterate per-series local models
            for key, est in self._local_estimators_.items():
                lagger = self._local_laggers_.get(key)
                if lagger is None:
                    raise KeyError(f"No lagger stored for key={key!r}")

                # get the per-series target as a single-index frame with column name "y"
                if y_long is not None and isinstance(y_long.index, pd.MultiIndex):
                    _key = (
                        key
                        if isinstance(lead_names, (str, int))
                        else (key if isinstance(key, tuple) else (key,))
                    )
                    y_key = y_long.xs(_key, level=lead_names, drop_level=True)

                    if isinstance(y_key, pd.Series):
                        y_key = y_key.to_frame("y")
                    elif list(y_key.columns) != ["y"]:
                        y_key = y_key.copy()
                        y_key.columns = ["y"]
                elif isinstance(self._y.index, pd.MultiIndex):
                    dbgprint(
                        "[RRF DEBUG predict] self._y.index.names:",
                        getattr(self._y.index, "names", None),
                    )
                    dbgprint(
                        "[RRF DEBUG predict] self._y.index.nlevels:",
                        getattr(self._y.index, "nlevels", None),
                    )
                    dbgprint(
                        "[RRF DEBUG predict] head idx:", self._y.index[:5].tolist()
                    )
                    dbgprint(
                        "[RRF DEBUG predict] lead_names:",
                        lead_names,
                        type(lead_names),
                        "len:",
                        (len(lead_names) if hasattr(lead_names, "__len__") else None),
                    )
                    dbgprint(
                        "[RRF DEBUG predict] key:",
                        key,
                        type(key),
                        "len:",
                        (
                            len(key)
                            if hasattr(key, "__len__")
                            and not isinstance(key, (str, bytes))
                            else None
                        ),
                    )
                    # normalize level/key shapes for pandas.xs
                    _lv = (
                        lead_names[0]
                        if isinstance(lead_names, (list, tuple))
                        and len(lead_names) == 1
                        else lead_names
                    )
                    _kk = key[0] if isinstance(key, tuple) and len(key) == 1 else key
                    y_key = self._y.xs(_kk, level=_lv, drop_level=True)
                    if isinstance(y_key, pd.Series):
                        y_key = y_key.to_frame("y")
                    elif list(y_key.columns) != ["y"]:
                        y_key = y_key.copy()
                        y_key.columns = ["y"]
                else:
                    # wide panel; single column DataFrame then rename to "y"
                    y_key = self._y[[key]].copy()
                    y_key.columns = ["y"]

                # mutable rolling copy to enable recursive feedback
                y_roll = y_key.copy()

                # --- walk the horizon recursively, appending each prediction ---
                yhat_vec = []
                for t_abs in fh_abs:
                    # Add a placeholder future row so lagger emits lags at t_abs
                    if t_abs not in y_roll.index:
                        y_roll = pd.concat(
                            [
                                y_roll,
                                pd.DataFrame(
                                    [np.nan],
                                    index=pd.Index([t_abs], name=y_roll.index.name),
                                    columns=y_roll.columns,
                                ),
                            ]
                        )

                    X_full_k = lagger.transform(y_roll)

                    # Take the lag row at t_abs
                    if t_abs in X_full_k.index:
                        X_row = X_full_k.loc[[t_abs]]
                    else:
                        if len(X_full_k) == 0:
                            yhat_i = np.nan
                            yhat_vec.append(yhat_i)
                            y_roll.loc[t_abs, "y"] = yhat_i
                            continue
                        X_row = X_full_k.iloc[[-1]]

                    X_row = prep_skl_df(X_row)

                    # ---- Coerce predict input to mirror fit() contract ----
                    Xtt_for_pred = X_row
                    if self._expects_nested_X() or _has_tabularizer_step(est):
                        # nested expected -> ensure DataFrame, then nest rows
                        if not isinstance(Xtt_for_pred, pd.DataFrame):
                            Xtt_for_pred = pd.DataFrame(Xtt_for_pred)
                        Xtt_for_pred = _to_nested_from_rows(Xtt_for_pred)
                    else:
                        # plain ts-regressor -> NumPy array
                        if isinstance(Xtt_for_pred, (pd.DataFrame, pd.Series)):
                            Xtt_for_pred = Xtt_for_pred.to_numpy()

                    yhat = est.predict(Xtt_for_pred)
                    yhat_i = float(np.asarray(yhat).ravel()[0])
                    yhat_vec.append(yhat_i)

                    # feedback: write pred at t_abs so next step can use it in lags
                    y_roll.loc[t_abs, "y"] = yhat_i

                # --- write this series' vector back into the combined y_pred ---
                if isinstance(y_pred.index, pd.MultiIndex):
                    # MultiIndex long: construct rows (key..., time) and update
                    key_tpl = key if isinstance(key, tuple) else (key,)
                    mi_rows = pd.MultiIndex.from_tuples(
                        [(*key_tpl, t) for t in fh_abs],
                        names=(lead_names + [time_name]) if lead_names else [time_name],
                    )
                    df_key = pd.DataFrame(
                        yhat_vec, index=mi_rows, columns=self._y.columns
                    )
                else:
                    # wide: fill only this series' column
                    df_key = _create_fcst_df(fh_abs, self._y, fill=np.nan)
                    if isinstance(df_key, pd.DataFrame) and (key in df_key.columns):
                        df_key.loc[:, key] = yhat_vec

                y_pred.update(df_key)

            if isinstance(y_pred.index, pd.MultiIndex):
                y_pred = y_pred.sort_index()
            return y_pred

        # ----------------------------
        # Single-series path (inner local model)
        # ----------------------------
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

        n_columns = 1
        window_length = self.window_length
        fh_max = fh.to_relative(self._cutoff_scalar())[-1]

        y_pred = np.zeros(fh_max)
        last = np.zeros((1, n_columns, window_length + fh_max))
        last[:, 0, :window_length] = y_last.T

        if X_pool is not None:
            dense_abs_fh, _ = self._generate_fh_no_gaps(fh)
            try:
                dense_abs_idx = dense_abs_fh.to_pandas()
            except Exception:
                dense_abs_idx = pd.Index(dense_abs_fh)

        inst_key = None
        if isinstance(self._y.index, pd.MultiIndex):
            cutoff_scalar = self._cutoff_scalar()
            for k in self._y.index.droplevel(-1).unique():
                if cutoff_scalar in self._y.xs(k, level=0).index:
                    inst_key = k
                    break
            if inst_key is None:
                inst_key = self._y.index.get_level_values(0)[-1]
        elif isinstance(self._y, pd.DataFrame) and self._y.shape[1] == 1:
            inst_key = self._y.columns[0]

        if getattr(self, "_local_estimators_", None):
            est = self._local_estimators_.get(inst_key)
            if est is None:
                raise KeyError(f"No local estimator stored for key={inst_key!r}")
        else:
            est = getattr(self, "estimator_", None)
            if est is None:
                raise AttributeError(
                    "No estimator_ on forecaster and no local estimators found."
                )

        for i in range(fh_max):
            X_pred = last[:, :, i : window_length + i].reshape(1, -1)[:, ::-1]

            if X_pool is not None:
                label_i = dense_abs_idx[i]

                def _as_scalar_label(lbl):
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

                if getattr(self, "X_treatment", "concurrent") == "shifted":
                    label_for_X = self.cutoff if i == 0 else dense_abs_idx[i - 1]
                else:
                    label_for_X = label_i

                row = _row_from(label_for_X)
                X_pred = np.concatenate((row, X_pred), axis=1)

            # ----- build X_step (DataFrame if we have feature names), then coerce -----
            if getattr(self, "_feature_cols_", None) is not None and X_pred.shape[
                1
            ] == len(self._feature_cols_):
                X_step = pd.DataFrame(X_pred, columns=self._feature_cols_)
            else:
                X_step = X_pred

            Xtt_for_pred = X_step
            if self._expects_nested_X() or _has_tabularizer_step(est):
                if not isinstance(Xtt_for_pred, pd.DataFrame):
                    Xtt_for_pred = pd.DataFrame(
                        Xtt_for_pred,
                        columns=(
                            self._feature_cols_
                            if getattr(self, "_feature_cols_", None) is not None
                            else None
                        ),
                    )
                Xtt_for_pred = _to_nested_from_rows(Xtt_for_pred)
            else:
                if isinstance(Xtt_for_pred, (pd.DataFrame, pd.Series)):
                    Xtt_for_pred = Xtt_for_pred.to_numpy()

            _raw = est.predict(Xtt_for_pred)
            y_pred[i] = float(np.asarray(_raw).ravel()[0])
            last[:, 0, window_length + i] = y_pred[i]

        return y_pred

    def _predict_in_sample_v2_local(self, X_pool, fh):
        """In-sample predictions for MultiIndex (panel) with pooling='local'."""
        #  0) Preconditions:
        #        we expect local + MultiIndex, and per-series artifacts from fit ---
        if not (self.pooling == "local" and isinstance(self._y.index, pd.MultiIndex)):
            # fall back to original implementation if ever called outside local+panel
            return self._predict_in_sample(X_pool, fh)

        #  1) Ensure a relative FH has a frequency so it can be made absolute ---
        if isinstance(fh, ForecastingHorizon) and fh.is_relative and fh.freq is None:
            time_idx = self._y.index.get_level_values(-1)
            freq = pd.infer_freq(time_idx) or getattr(time_idx, "freqstr", None) or "D"
            fh = ForecastingHorizon(fh.to_numpy(), is_relative=True, freq=freq)

        #  2) Names and keys for the MultiIndex ---
        idx_names = list(self._y.index.names)  # e.g., ["series", "time"]
        lead_names = idx_names[:-1]  # e.g., ["series"]
        time_name = idx_names[-1]  # e.g., "time"

        # Prefer keys saved during fit; otherwise infer from current data
        series_keys = getattr(self, "_series_keys_", None)
        if series_keys is None:
            series_keys = self._y.index.droplevel(-1).unique()

        #  3) Prepare output container on the expected prediction index ---
        out_idx = self._get_expected_pred_idx(fh=fh)  # MultiIndex: (keys..., time)
        y_pred = _create_fcst_df(out_idx, self._y)  # replaces following line
        # y_pred = pd.DataFrame(index=out_idx, columns=self._y.columns, dtype=float)

        #  4) Per-series prediction (in-sample) ---
        parts = []  # collect per-key frames, then update y_pred

        for key in series_keys:
            # Normalize key to tuple for consistent MultiIndex construction
            key_tpl = key if isinstance(key, tuple) else (key,)

            est = self._local_estimators_.get(key)
            lagger = self._local_laggers_.get(key)
            if est is None or lagger is None:
                raise KeyError(f"No local estimator/lagger stored for key={key!r}")

            # Slice this series as single-index (DatetimeIndex) y_k
            if len(lead_names) == 1:
                # level needs to be a scalar when key is scalar
                y_k = self._y.xs(key, level=lead_names[0], drop_level=True)
            else:
                # multiple leading levels: use tuple key and list level
                key_tpl = key if isinstance(key, tuple) else (key,)
                y_k = self._y.xs(key_tpl, level=lead_names, drop_level=True)

            # Guarantee a usable freq for absolute conversion and lagger alignment
            if y_k.index.freq is None:
                inferred = (
                    self._local_freqs_.get(key)
                    if hasattr(self, "_local_freqs_")
                    else None
                )
                inferred = inferred or pd.infer_freq(y_k.index)
                if inferred is not None:
                    y_k = y_k.asfreq(inferred)

            # Convert fh to absolute timestamps relative to this key's cutoff
            cutoff_k = getattr(self, "_local_cutoffs_", {}).get(key, y_k.index.max())
            fh_abs_k = fh.to_absolute_index(cutoff_k)
            try:
                abs_idx_k = fh_abs_k.to_pandas()
            except Exception:
                abs_idx_k = pd.Index(fh_abs_k)

            # Build the full in-sample design with the SAME lagger used in fit
            # (Lag.transform drops the first 'max(lags)' rows;
            # index aligns to usable rows)
            X_full_k = lagger.transform(y_k)

            if X_pool is not None:
                # NOTE: same exog-join you use in _predict_out_of_sample_v2_local
                # Example placeholder:
                # X_exog_k = X_pool.xs(key, level=lead_names) \
                #    if isinstance(X_pool.index, pd.MultiIndex) else X_pool
                # X_exog_k = X_exog_k.reindex(X_full_k.index)  # align by time
                # X_full_k = pd.concat([X_exog_k, X_full_k], axis=1)
                raise ValueError("predict in-sample with exogenous TBD")

            # Only timestamps with enough history have rows in X_full_k
            target_rows_k = X_full_k.index.intersection(abs_idx_k)
            if len(target_rows_k) == 0:
                continue

            # Predict for this key at the required in-sample timestamps
            y_hat_k = est.predict(X_full_k.loc[target_rows_k])

            # Build a (key,time)-indexed DataFrame to merge into y_pred
            df_k = pd.DataFrame(y_hat_k, index=target_rows_k, columns=self._y.columns)
            # Construct the MultiIndex: (lead levels..., time)
            mi_k = pd.MultiIndex.from_tuples(
                [(*key_tpl, t) for t in target_rows_k],
                names=lead_names + [time_name],
            )
            df_k.index = mi_k
            parts.append(df_k)

        # 5) Assemble final frame ---
        if parts:
            stacked = pd.concat(parts, axis=0)
            # Update pre-allocated y_pred (so it has exactly expected index order)
            y_pred.update(stacked)

        if isinstance(y_pred.index, pd.MultiIndex):
            y_pred = y_pred.sort_index()

        return y_pred

    def _get_local_estimator(self, key):
        # Case 1: you stored bare estimators
        if hasattr(self, "estimators_") and key in getattr(self, "estimators_", {}):
            return self.estimators_[key]
        # Case 2: you stored child forecasters
        if hasattr(self, "forecasters_") and key in getattr(self, "forecasters_", {}):
            child = self.forecasters_[key]
            return getattr(child, "estimator_", child)
        raise KeyError(f"No local estimator found for key={key!r}")

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
                # prefer stored freq; else infer from y.index
                idx = getattr(getattr(self, "_y", None), "index", None)
                if isinstance(idx, pd.MultiIndex):
                    # use the time level for frequency inference
                    idx = idx.get_level_values(-1)

                freq = getattr(idx, "freq", None)
                if freq is None and idx is not None:
                    try:
                        freq = pd.infer_freq(idx)
                    except Exception:
                        freq = None

                if freq is None:
                    # default to daily step if nothing is available
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
            # Get a 1D time index to infer frequency from (handles MultiIndex panels)
            _idx = getattr(getattr(self, "_y", None), "index", None)
            if isinstance(_idx, pd.MultiIndex):
                _time_idx = _idx.get_level_values(-1)
            else:
                _time_idx = _idx

            # Try explicit .freq first, then infer; if unavailable, leave as None
            freq = getattr(_time_idx, "freq", None)
            if freq is None and _time_idx is not None:
                try:
                    freq = pd.infer_freq(_time_idx)
                except Exception:
                    freq = None

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

        # 5) steps has the same length as the gapless absolute index
        steps_no_gaps = range(1, len(gapless_abs) + 1)

        # return as absolute FH (callers use .to_pandas())
        dense_fh_abs = ForecastingHorizon(gapless_abs, is_relative=False)
        return dense_fh_abs, steps_no_gaps

    def _predict_out_of_sample(self, X_pool, fh):
        """Recursive reducer: predict out of sample (ahead of cutoff)."""
        # very similar to _predict_concurrent of DirectReductionForecaster - refactor?
        # Strategy selection:
        #   global  -> optimized v2 global path (fallback to v1 inside if exogenous X)
        #   local   -> optimized v2 local path
        #   panel   -> fallback to legacy v1 path for correctness (gappy fh indexing)
        #             TODO: implement optimized v2 panel path (#panel-optimization)

        exog_present = self._X is not None or X_pool is not None

        dbgprint(f"[RRF.predict] exog_present={exog_present}")
        dbgprint(f"[RRF.predict] fh_in={fh}")  # raw object
        try:
            fh_abs_idx = fh.to_absolute_index(self._cutoff_scalar())
            to_pd = getattr(fh_abs_idx, "to_pandas", None)
            fh_abs_list = list(to_pd() if callable(to_pd) else fh_abs_idx)
            dbgprint(f"[RRF.predict] fh_abs={fh_abs_list}")
        except Exception as e:
            dbgprint(f"[RRF.predict] fh_abs=ERROR: {e}")

        if isinstance(getattr(self, "estimator_", None), pd.Series):
            # Produce a DataFrame of repeated means on the absolute fh index
            return self._create_fallback_df(fh)

        already_filtered = False
        if self.pooling == "panel":
            # v1 path already returns only the requested fh rows
            y_pred = self._predict_out_of_sample_v1(X_pool, fh)
            already_filtered = True
        elif self.pooling == "global" and isinstance(self._y.index, pd.MultiIndex):
            dbgprint(
                "OriginalRecursiveReductionForecaster._predict_out_of_sample() - here"
            )
            y_pred = self._predict_out_of_sample_v2_global(X_pool, fh)
            # v2_global falls back to v1 when X is present; v1 already returns only the
            # requested fh rows, so skip the second filtering step.
            if (self._X is not None) or (X_pool is not None):
                already_filtered = True
        elif self.pooling == "local" or not isinstance(self._y.index, pd.MultiIndex):
            fh_dense_abs, _ = self._generate_fh_no_gaps(fh)
            y_pred = self._predict_out_of_sample_v2_local(X_pool, fh_dense_abs)
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
        """Recursive reducer: predict out of sample (ahead of cutoff) — v1 semantics."""
        dbgprint(
            "[_predict_multiple] _predict_out_of_sample_v1: ENTER",
            f"X_pool is None? {X_pool is None}",
            f"fh type={type(_fh)}",
            f"fh.is_relative={getattr(_fh, 'is_relative', None)}",
            f"fh.freq={getattr(_fh, 'freq', None)}",
        )

        # absolute horizon & pandas index of targets
        fh_abs_local = _fh.to_absolute(self._cutoff_as_1elem_index_with_freq())
        tgt_idx = (
            fh_abs_local.to_pandas()
            if hasattr(fh_abs_local, "to_pandas")
            else pd.Index(fh_abs_local)
        )
        dbgprint(
            "[_predict_multiple] _predict_out_of_sample_v1: tgt_idx sample:",
            list(tgt_idx[:5]),
        )

        fh = fh_abs_local  # keep absolute FH
        fh_idx = self._get_expected_pred_idx(fh=fh)
        self._assert_future_X_coverage(X_pool, fh)

        y_plus_preds = self._y  # recursive state
        lagger_y_to_X = self.lagger_y_to_X_  # fitted lagger (training)

        X_ext = X_pool
        if X_ext is not None and self.X_treatment == "shifted":
            X_ext = X_ext.shift(1)  # keep your convention

        y_pred_full = _create_fcst_df(fh_idx, self._y)

        # --- drive exactly over the requested absolute stamps ---
        for t in tgt_idx:
            if getattr(self.fh, "freq", None) is not None:
                y_plus_preds = _asfreq_per_series_safe(
                    y_plus_preds, self.fh.freq, how="start"
                )
            ##--------------------------------------------------------------------------------
            # 1) extend the *target y* by one step so the new timestamp exists
            #    (this gives us a frame with the extra time row we want to predict for)
            y_extend = Lag(lags=1, index_out="extend", keep_column_names=True)
            if self._impute_method is not None:
                y_extend = y_extend * self._impute_method.clone()
            y_plus_one = y_extend.fit_transform(y_plus_preds)

            # 2) build the lagged design from the *extended y*
            Xtt = lagger_y_to_X.transform(y_plus_one)

            # the new row's timestamp (t) must now be present
            next_time_raw = (
                y_plus_one.index.get_level_values(-1)[-1]
                if isinstance(y_plus_one.index, pd.MultiIndex)
                else y_plus_one.index[-1]
            )
            if next_time_raw != t:
                print("[OOSv1] WARN: next_time_raw != t", next_time_raw, t)

            # 3) pick the single design row at t
            if isinstance(Xtt.index, pd.MultiIndex):
                Xtt_row = Xtt.xs(t, level=-1, drop_level=False)
            else:
                Xtt_row = Xtt.loc[[t]]
            ##------------------------------------------------------------------

            # 3) exogenous row(s) for t, concat with lag design
            if X_ext is not None:
                if isinstance(X_ext.index, pd.MultiIndex):
                    X_ex_row = X_ext.xs(t, level=-1, drop_level=False)
                else:
                    X_ex_row = X_ext.loc[[t]]
                Xtt_row = pd.concat([X_ex_row, Xtt_row], axis=1)

            ##------------------------------------------------------------------
            # 3.5)
            # overwrite lag_k__y cols at time t from recursive state y_plus_preds
            try:
                # identify lag columns for the target (works for any # of lags)
                lag_cols = [
                    c
                    for c in Xtt_row.columns
                    if isinstance(c, str) and c.startswith("lag_") and c.endswith("__y")
                ]
                if lag_cols:
                    # map column -> k (from "lag_k__y")
                    lag_k = {c: int(c.split("_")[1]) for c in lag_cols}
                    # target column name (assumed single target column)
                    target_col = (
                        self._y.columns[0] if hasattr(self._y, "columns") else "y"
                    )

                    if isinstance(Xtt_row.index, pd.MultiIndex):
                        # per series
                        series_keys = Xtt_row.index.get_level_values(0).unique()
                        for key in series_keys:
                            # last observed/predicted values for this series
                            yk = y_plus_preds.xs(key, level=0)[target_col]
                            # the row we are writing for (key, t)
                            row_idx = (
                                Xtt_row.xs(key, level=0, drop_level=False)
                                .xs(t, level=-1, drop_level=False)
                                .index
                            )
                            # compute lag values from y_plus_preds tail
                            vals = {
                                c: float(yk.iloc[-k])
                                for c, k in lag_k.items()
                                if len(yk) >= k
                            }
                            if vals:
                                Xtt_row.loc[row_idx, list(vals.keys())] = list(
                                    vals.values()
                                )
                    else:
                        # single series
                        yk = y_plus_preds[target_col]
                        vals = {
                            c: float(yk.iloc[-k])
                            for c, k in lag_k.items()
                            if len(yk) >= k
                        }
                        if vals:
                            Xtt_row.loc[:, list(vals.keys())] = list(vals.values())
            except Exception as e:
                print("[OOSv1] WARN: lag overwrite skipped due to:", repr(e))
            # --- END FIX ---

            # 4) sklearn-friendly
            Xtt_row = prep_skl_df(Xtt_row)

            # 5) typed step index for writing predictions at t
            step_idx = self._get_expected_pred_idx(fh=[t])
            n_rows = len(step_idx)

            # 6) predict 1-step for all series at t
            est = self.estimator_
            if isinstance(est, pd.Series):
                vals = np.tile(est.values, (n_rows, 1))
                y_step = pd.DataFrame(vals, index=step_idx, columns=self._y.columns)
            else:
                dbgprint(
                    "[OOSv1] ITER: model=",
                    type(est).__name__,
                    " rows=",
                    Xtt_row.shape[0],
                    " time head=",
                    (
                        list(Xtt_row.index.get_level_values(-1)[:1])
                        if isinstance(Xtt_row.index, pd.MultiIndex)
                        else list(Xtt_row.index[:1])
                    ),
                )
                y_hat = est.predict(Xtt_row)
                y_step = pd.DataFrame(y_hat, index=step_idx, columns=self._y.columns)

            # 7) write & feed back
            y_pred_full.update(y_step)
            y_plus_preds = y_plus_preds.combine_first(
                y_step
            )  # enables recursion for next t

        dbgprint("[RRF.predict] _predict_out_of_sample_v1: path=OOS_WITH_EXOG")
        # visualize and return exactly requested horizons
        try:
            dbgprint(
                "[OOSv1] DONE: fh_idx time:",
                list(fh_idx.levels[-1])
                if isinstance(fh_idx, pd.MultiIndex)
                else list(fh_idx),
            )
            dbgprint("[OOSv1] DONE: y_pred_full head:\n", y_pred_full.head(10))
            dbgprint(
                "[OOSv1] DONE: y_pred_full missing:",
                int(y_pred_full.isna().sum().sum()),
            )
        except Exception:
            dbgprint("")
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
        lag_plus = Lag(lags=1, index_out="original", keep_column_names=True)

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
            "capability:multivariate": True,  # handle wide DataFrames natively
            "capability:hierarchical": True,  # handle MultiIndex/panels natively
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
        """Handle y when MultiIndex (..., time) and X is single-level (time).

        Broadcast X across the leading levels so X and y share the same index.
        Works for any number of leading levels.
        """
        # nothing to do if: no X, y is not MultiIndex, or X already MultiIndex
        if (
            X is None
            or not isinstance(y_index, pd.MultiIndex)
            or isinstance(X.index, pd.MultiIndex)
        ):
            return X

        # all leading levels form the "instance" key; last level is time
        # e.g. names ["gender","grade","time"] -> keys are tuples (gender, grade)
        leading_names = list(y_index.names[:-1])
        full_names = list(y_index.names)
        if not leading_names:
            # edge case: y has only one level (shouldn't come here), just return X
            return X

        # unique keys in the same order as they appear in y
        leading_keys = y_index.droplevel(
            -1
        ).unique()  # MultiIndex of tuples (or Index if 1 level)

        # Broadcast: a dict with tuple keys is expanded into multiple outer index levels
        X_broadcast = pd.concat(dict.fromkeys(leading_keys, X), names=full_names)
        return X_broadcast

    def fit(self, y, X=None, fh=None):
        y = _unwrap_vdf(y)
        X = _unwrap_vdf(X)

        # remember original index/columns for roundtripping
        # (you already set these in _to_long_from_wide; keep that behavior)
        # dump_obj("RecursiveReductionForecaster.fit() - entered", "y", y)
        # dump_obj("RecursiveReductionForecaster.fit()", "X", X)
        dbgprint(f"self.pooling = {self.pooling}")
        dbgprint(f"self._is_wide(y) = {self._is_wide(y)}")

        if self._is_wide(y):
            y_long = self._to_long_from_wide(y)
        else:
            y_long = y  # n.b. y_long may be a single series

        X = self._broadcast_X_to_panel(X, y_long.index)

        # IMPORTANT:
        # call base public fit (not _fit), so BaseForecaster stores y_metadata, etc
        dbgprint(
            "[fit.debug] BEFORE super().fit | has_frozen=",
            getattr(self, "_orig_shape_frozen", None),
            " flags=",
            getattr(self, "_orig_y_is_series", None),
            getattr(self, "_orig_y_is_df1", None),
            getattr(self, "_orig_y_is_dfm", None),
            getattr(self, "_orig_y_is_panel", None),
            getattr(self, "_orig_y_is_wide", None),
        )

        super().fit(y=y_long, X=X, fh=fh)

        if self._is_wide(y):
            self._was_wide_input = True
            self._was_long_input = False
        else:
            # keep state consistent
            self._was_wide_input = False
            self._was_long_input = isinstance(getattr(y, "index", None), pd.MultiIndex)

        self._y_orig = y.copy()  # to make it availabe in predict
        if getattr(self, "_was_wide_input", False):
            self._y_long = y_long  # save so it can be recalled in predict

        dbgprint(
            "[fit.debug] AFTER  super().fit  | has_frozen=",
            getattr(self, "_orig_shape_frozen", None),
            " flags=",
            getattr(self, "_orig_y_is_series", None),
            getattr(self, "_orig_y_is_df1", None),
            getattr(self, "_orig_y_is_dfm", None),
            getattr(self, "_orig_y_is_panel", None),
            getattr(self, "_orig_y_is_wide", None),
        )

        dbgprint(
            f"[fit] BEFORE _record_train_shape | type(y)={type(y)} "
            f"| is_series={isinstance(y, pd.Series)} "
            f"| is_df={isinstance(y, pd.DataFrame)} "
            f"| idx_is_multi={isinstance(getattr(y, 'index', None), pd.MultiIndex)}"
        )

        self._record_train_shape(y)

        dbgprint(
            f"[fit] AFTER  _record_train_shape | "
            f"orig_series={getattr(self, '_orig_y_is_series', None)} "
            f"orig_df1={getattr(self, '_orig_y_is_df1', None)} "
            f"orig_dfm={getattr(self, '_orig_y_is_dfm', None)} "
            f"orig_panel={getattr(self, '_orig_y_is_panel', None)} "
            f"orig_wide={getattr(self, '_orig_y_is_wide', None)}"
        )

        if self.pooling == "local" and isinstance(y_long.index, pd.MultiIndex):
            # names like ["series", "time"]; last level is time in your traces
            # time_name = y_long.index.names[-1]
            lead_names = y_long.index.names[:-1]

            # iterate leading keys in the order they appear
            # normalize level: if it's a list/tuple of length 1, use a scalar
            _lv = (
                lead_names[0]
                if isinstance(lead_names, (list, tuple)) and len(lead_names) == 1
                else lead_names
            )
            for key, y_part in y_long.groupby(level=_lv, sort=False):
                # drop leading levels; keep only the time index for this sub-series
                y_key = y_part.droplevel(lead_names)

                # remember per-series cutoff (last observed stamp)
                self._local_cutoffs_[key] = y_key.index.max()

                # OPTIONAL but nice to have:
                freq_k = y_key.index.freqstr or pd.infer_freq(y_key.index)
                if not hasattr(self, "_local_freqs_"):
                    self._local_freqs_ = {}
                self._local_freqs_[key] = freq_k  # may be None if cannot infer

                # --- build lag features for training (same as single-series) ---
                lagger = Lag(
                    lags=self._lags, index_out="original"
                )  # force original index
                X_key = lagger.fit_transform(y_key)  #  like lag_1__y, lag_2__y, ...
                X_key = X_key.dropna(axis=0)  # drop rows made NaN by lags

                # align y to X (drop NA rows introduced by lags)
                y_key_aligned = y_key.loc[X_key.index]

                # fit estimator clone on this key
                est_key = clone(self.estimator)
                # flatten y for regressors if needed
                y_fit = y_key_aligned.values.reshape(-1, 1)
                if hasattr(est_key, "fit"):
                    est_key.fit(X_key, y_fit)

                # store artifacts
                self._local_estimators_[key] = est_key
                self._local_laggers_[key] = lagger

            # keep a small flag so predict() knows we trained locally
            self._trained_local_multiindex_ = True
        else:
            self._trained_local_multiindex_ = False

        if getattr(self, "_was_wide_input", False):
            self._y = y  # needed to pass CI tests (which is a pain)

        return

    # 3) Override PUBLIC predict to roundtrip back to WIDE if we trained from WIDE.
    def predict(self, fh=None, X=None):
        dbg_id = getattr(self, "_dbg_id", "?")
        cls_name = type(self).__name__

        flag_names = [
            "_orig_y_is_series",
            "_orig_y_is_df1",
            "_orig_y_is_dfm",
            "_orig_y_is_panel",
            "_orig_y_is_wide",
        ]
        has_flags = all(hasattr(self, a) for a in flag_names)
        vals = tuple(getattr(self, a, None) for a in flag_names)

        frozen = getattr(self, "_orig_shape_frozen", None)
        fitted = hasattr(self, "estimators_") or hasattr(self, "estimator_")

        dbgprint(
            f"[predict.enter] ID={dbg_id} cls={cls_name} has_flags={has_flags} "
            f"vals={vals} frozen={frozen} fitted={fitted}"
        )

        dbgprint(
            f"[predict.check] ID={getattr(self, '_dbg_id', '?')} "
            f"FIT_SEEN={getattr(self, '_dbg_fit_done', False)}"
        )

        self.check_is_fitted()
        # fallback to the horizon provided at fit time (sktime contract)
        if fh is None:
            fh = self.fh
        if fh is None:
            raise ValueError(
                "No `fh` has been set yet, in this instance of "
                "RecursiveReductionForecaster, "
                "please specify `fh` in `fit` or `predict`"
            )

        dbgprint(
            "[RRF.predict] ENTER",
            f"is_fh_FH={hasattr(fh, 'is_relative')}",
            f"is_relative={getattr(fh, 'is_relative', None)}",
            f"fh.freq={getattr(fh, 'freq', None)}",
        )

        cutoff_ix = self._cutoff_as_1elem_index_with_freq()
        dbgprint(
            "[RRF.predict] cutoff_ix:",
            type(cutoff_ix),
            getattr(cutoff_ix, "freq", None),
            getattr(cutoff_ix, "tz", None),
            list(cutoff_ix[:1]),
        )

        # try:
        #     fh_abs_dbg = fh.to_absolute(cutoff_ix) \
        #         if hasattr(fh, "to_absolute") else fh
        #     # Make a tiny peek (at most 5)
        #     fh_abs_dbg_idx = (
        #         fh_abs_dbg.to_pandas()
        #         if hasattr(fh_abs_dbg, "to_pandas")
        #         else pd.Index(fh_abs_dbg)
        #     )
        #     # print("[RRF.predict] fh_abs peek:", list(fh_abs_dbg_idx[:5]))
        # except Exception as e:
        #     print("[RRF.predict] fh_abs=ERROR:", repr(e))

        # If we trained from WIDE, temporarily put _y back to the LONG form we fitted on
        was_wide = getattr(self, "_was_wide_input", False)
        if was_wide:
            self._y = self._y_long

        # Ensure fh is validated and stored (mirrors BaseForecaster behavior lightly)
        if fh is not None:
            self._check_fh(fh)  # sets self.fh internally

        # Call the parent ALGO directly to avoid BaseForecaster's output coercion
        y_pred = OriginalRecursiveReductionForecaster._predict(self, fh=self.fh, X=X)

        # If started WIDE and we're not in the moving-cutoff update path, return WIDE
        in_update = getattr(self, "_predict_for_update", False)
        if was_wide and not in_update:
            y_pred = self._to_wide_from_long(y_pred)

        # Restore caller-facing _y to the original format
        if was_wide:
            self._y = self._y_orig

        # --- Preserve caller-facing shape based on training y type ---
        # === Canonicalize output shape to match the training target ===
        fh_index = self.fh.to_pandas() if hasattr(self.fh, "to_pandas") else self.fh

        dbgprint(
            f"[predict.pre-coerce] ID={getattr(self, '_dbg_id', '?')} flags="
            # f"{getattr(self, '_orig_y_is_series', None),
            # getattr(self, '_orig_y_is_df1', None),
            # getattr(self, '_orig_y_is_dfm', None),
            # getattr(self, '_orig_y_is_panel', None),
            # getattr(self, '_orig_y_is_wide', None)}"
        )

        dbgprint(
            f"[predict] pre-coerce | type(y_pred)={type(y_pred)} "
            f"shape={getattr(y_pred, 'shape', None)} "
            f"| orig_series={getattr(self, '_orig_y_is_series', None)} "
            f"orig_df1={getattr(self, '_orig_y_is_df1', None)} "
            f"orig_dfm={getattr(self, '_orig_y_is_dfm', None)} "
            f"orig_panel={getattr(self, '_orig_y_is_panel', None)} "
            f"orig_wide={getattr(self, '_orig_y_is_wide', None)}"
        )

        y_pred = self._coerce_to_train_shape(y_pred, fh_index)
        return y_pred

    # 4) Override PUBLIC update
    def update(self, y=None, X=None, update_params=True):
        # Intercept to normalize inputs before any parent/base validation.
        y = _unwrap_vdf(y)
        X = _unwrap_vdf(X)

        # keep internal representation consistent with fit()
        if (
            getattr(self, "_was_wide_input", False)
            and y is not None
            and self._is_wide(y)
        ):
            y = self._to_long_from_wide(y)

        return super().update(y=y, X=X, update_params=update_params)

    # 5) Override internal
    def _check_X_y(self, X=None, y=None):
        # Last line of defense: ensure pandas hits the validator.
        X = _unwrap_vdf(X)
        y = _unwrap_vdf(y)
        return super()._check_X_y(X=X, y=y)

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


#    def _predict(self, fh=None, X=None):
#        y_pred = OriginalRecursiveReductionForecaster._predict(self, fh=fh, X=X)
#        return y_pred


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
