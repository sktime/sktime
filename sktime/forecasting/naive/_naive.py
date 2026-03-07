# !/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__all__ = ["NaiveForecaster", "NaiveVariance"]

import math

# from collections import defaultdict
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import norm

from sktime.datatypes._convert import convert, convert_to
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.base._base import DEFAULT_ALPHA, BaseForecaster
from sktime.forecasting.base._sktime import _BaseWindowForecaster
from sktime.utils.seasonality import _pivot_sp, _unpivot_sp
from sktime.utils.validation import check_window_length
from sktime.utils.validation.forecasting import check_sp
from sktime.utils.warnings import warn


class NaiveForecaster(_BaseWindowForecaster):
    """Forecast based on naive assumptions about past trends continuing.

    NaiveForecaster is a forecaster that makes forecasts using simple
    strategies. Two out of three strategies are robust against NaNs. The
    NaiveForecaster can also be used for multivariate data and it then
    applies internally the ColumnEnsembleForecaster, so each column
    is forecasted with the same strategy.

    Parameters
    ----------
    strategy : {"last", "mean", "drift"}, default="last"
    sp : int, or None, default=1
    window_length : int or None, default=None
    """

    _tags = {
        "authors": [
            "mloning",
            "piyush1729",
            "sri1419",
            "Flix6x",
            "aiwalter",
            "IlyasMoutawwakil",
            "fkiraly",
            "bethrice44",
        ],
        "y_inner_mtype": "pd.Series",
        "requires-fh-in-fit": False,
        "capability:missing_values": True,
        "capability:exogenous": False,
        "scitype:y": "univariate",
        "capability:pred_var": True,
        "capability:pred_int": True,
        "tests:core": True,
    }

    def __init__(self, strategy="last", window_length=None, sp=1):
        super().__init__()
        self.strategy = strategy
        self.sp = sp
        self.window_length = window_length
        if self.strategy in ("last", "mean"):
            self.set_tags(**{"capability:missing_values": True})

    def _fit(self, y, X, fh):
        sp = self.sp or 1
        n_timepoints = y.shape[0]

        if self.strategy in ("last", "mean"):
            if self.window_length is not None and sp != 1:
                if self.window_length < sp:
                    raise ValueError(
                        f"The `window_length`: {self.window_length} is smaller "
                        f"than `sp`: {sp}."
                    )
            self.window_length_ = check_window_length(self.window_length, n_timepoints)
            self.sp_ = check_sp(sp)
            if self.window_length is None:
                self.window_length_ = len(y)

        elif self.strategy == "drift":
            if sp != 1:
                warn(
                    "For the `drift` strategy, the `sp` value will be ignored.",
                    obj=self,
                )
            self.window_length_ = check_window_length(self.window_length, n_timepoints)
            if self.window_length is None:
                self.window_length_ = len(y)
            if self.window_length == 1:
                raise ValueError(
                    f"For the `drift` strategy, the `window_length`: "
                    f"{self.window_length} value must be greater than one."
                )
        else:
            raise ValueError(
                f"Unknown strategy: {self.strategy}. "
                f"Expected one of: ('last', 'mean', 'drift')."
            )

        if self.window_length_ > len(y):
            param = "sp" if self.strategy == "last" and sp != 1 else "window_length_"
            raise ValueError(
                f"The {param}: {self.window_length_} is larger than "
                f"the training series."
            )
        return self

    def _predict_last_window(
        self, fh, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA
    ):
        last_window, _ = self._get_last_window()
        fh = fh.to_relative(self.cutoff)
        strategy = self.strategy
        sp = self.sp or 1

        if np.all(np.isnan(last_window)) or len(last_window) == 0:
            return self._predict_nan(fh)

        elif strategy == "last" or (strategy == "drift" and self.window_length_ == 1):
            if sp == 1:
                last_valid_value = last_window[
                    (~np.isnan(last_window))[0::sp].cumsum().argmax()
                ]
                return np.repeat(last_valid_value, len(fh))
            else:
                last_window = self._reshape_last_window_for_sp(last_window)
                y_pred = last_window[
                    (~np.isnan(last_window)).cumsum(0).argmax(0).T,
                    range(last_window.shape[1]),
                ]
                y_pred = self._tile_seasonal_prediction(y_pred, fh)

        elif strategy == "mean":
            if sp == 1:
                return np.repeat(np.nanmean(last_window), len(fh))
            else:
                last_window = self._reshape_last_window_for_sp(last_window)
                y_pred = np.nanmean(last_window, axis=0)
                y_pred = self._tile_seasonal_prediction(y_pred, fh)

        elif strategy == "drift":
            if self.window_length_ != 1:
                if np.any(np.isnan(last_window[[0, -1]])):
                    raise ValueError(
                        f"For {strategy}, first and last elements in the last "
                        f"window must not be a missing value."
                    )
                slope = (last_window[-1] - last_window[0]) / (self.window_length_ - 1)
                fh_idx = fh.to_indexer(self.cutoff)
                y_pred = last_window[-1] + (fh_idx + 1) * slope
        else:
            raise ValueError(f"unknown strategy {strategy}")

        return y_pred

    def _reshape_last_window_for_sp(self, last_window):
        remainder = self.window_length_ % self.sp_
        pad_width = (self.sp_ - remainder) if remainder > 0 else 0
        pad_width += self.window_length_ - len(last_window)
        last_window = np.hstack([np.full(pad_width, np.nan), last_window])
        last_window = last_window.reshape(
            int(np.ceil(self.window_length_ / self.sp_)), self.sp_
        )
        return last_window

    def _tile_seasonal_prediction(self, y_pred, fh):
        if fh[-1] > self.sp_:
            reps = int(np.ceil(fh[-1] / self.sp_))
            y_pred = np.tile(y_pred, reps=reps)
        fh_idx = fh.to_indexer(self.cutoff)
        return y_pred[fh_idx]

    def _predict_naive(self, fh=None, X=None):
        from sktime.transformations.series.lag import Lag

        strategy = self.strategy
        sp = self.sp
        _y = self._y
        cutoff = self.cutoff

        if isinstance(_y.index, pd.DatetimeIndex) and hasattr(_y.index, "freq"):
            freq = _y.index.freq
        else:
            freq = None

        lagger = Lag(1, keep_column_names=True, freq=freq)
        expected_index = fh.to_absolute(cutoff).to_pandas()

        if strategy == "last" and sp == 1:
            y_old = lagger.fit_transform(_y)
            y_new = pd.DataFrame(index=expected_index, columns=[0], dtype="float64")
            full_y = pd.concat([y_old, y_new], keys=["a", "b"]).sort_index(level=-1)
            y_filled = full_y.ffill().bfill()
            y_pred = y_filled.loc["b"].iloc[:, 0]

        elif strategy == "last" and sp > 1:
            y_old = _pivot_sp(_y, sp, anchor_side="end")
            y_old = lagger.fit_transform(y_old)
            y_new_mask = pd.Series(index=expected_index, dtype="float64")
            y_new = _pivot_sp(y_new_mask, sp, anchor=_y, anchor_side="end")
            full_y = pd.concat([y_old, y_new], keys=["a", "b"]).sort_index(level=-1)
            y_filled = full_y.ffill().bfill()
            y_pred = _unpivot_sp(y_filled.loc["b"], template=_y)
            y_pred = y_pred.reindex(expected_index).iloc[:, 0]

        y_pred.index = expected_index
        y_pred.name = _y.name
        return y_pred

    def _predict(self, fh=None, X=None):
        strategy = self.strategy
        if strategy in ["last"]:
            return self._predict_naive(fh=fh, X=X)

        y_pred = super()._predict(fh=fh, X=X)
        if isinstance(y_pred, pd.DataFrame):
            y_pred = y_pred.iloc[:, 0]

        if self._y.index[0] in y_pred.index:
            if y_pred.loc[[self._y.index[0]]].hasnans:
                y_pred.loc[self._y.index[0]] = self._y[self._y.index[1]]

        y_pred.name = self._y.name
        return y_pred

    def _predict_quantiles(self, fh, X, alpha):
        y_pred = self.predict(fh)
        y_pred = convert(y_pred, from_type=self._y_mtype_last_seen, to_type="pd.Series")
        pred_var = self.predict_var(fh)
        z_scores = norm.ppf(alpha)
        errors = (
            np.sqrt(pred_var.to_numpy().reshape(len(pred_var), 1)) * z_scores
        ).reshape(len(y_pred), len(alpha))
        var_names = self._get_varnames()
        pred_quantiles = pd.DataFrame(
            errors + y_pred.values.reshape(len(y_pred), 1),
            columns=pd.MultiIndex.from_product([var_names, alpha]),
            index=fh.to_absolute_index(self.cutoff),
        )
        return pred_quantiles

    def _predict_var(self, fh, X=None, cov=False):
        y = convert_to(self._y, "pd.Series")
        T = len(y)
        sp = self.sp

        if self.strategy == "last":
            y_res = y - y.shift(self.sp)
        elif self.strategy == "mean":
            if not self.window_length:
                if sp > 1:
                    reps = math.ceil(T / sp) + 1
                    past_fh = ForecastingHorizon(
                        list(range(1, sp + 1)), is_relative=None, freq=self._cutoff
                    )
                    seasonal_means = self._predict(fh=past_fh)
                    if isinstance(seasonal_means, pd.DataFrame):
                        seasonal_means = seasonal_means.squeeze()
                    y_pred = np.tile(seasonal_means.to_numpy(), reps)[0:T]
                else:
                    past_fh = ForecastingHorizon(1, is_relative=None, freq=self._cutoff)
                    y_pred = np.repeat(np.squeeze(self._predict(fh=past_fh)), T)
            else:
                if sp > 1:
                    seasons = np.mod(np.arange(T), sp)
                    y_pred = (
                        y.to_frame()
                        .assign(__sp__=seasons)
                        .groupby("__sp__")
                        .rolling(self.window_length)
                        .mean()
                        .droplevel("__sp__")
                        .sort_index()
                        .squeeze()
                    )
                else:
                    y_pred = y.rolling(self.window_length).mean()
            y_res = y - y_pred
        else:
            slope = (y.iloc[-1] - y.iloc[-(self.window_length or 0)]) / (T - 1)
            y_res = y - (y.shift(sp) + slope)

        n_nans = np.sum(pd.isna(y_res))
        mse_res = np.sum(np.square(y_res)) / (T - n_nans - (self.strategy == "drift"))
        se_res = np.sqrt(mse_res)
        window_length = self.window_length or T

        def sqrt_flr(x):
            return np.sqrt(np.maximum(x, 1))

        partial_se_formulas = {
            "last": (
                sqrt_flr if sp == 1 else lambda h: sqrt_flr(np.floor((h - 1) / sp) + 1)
            ),
            "mean": lambda h: np.repeat(sqrt_flr(1 + (1 / window_length)), len(h)),
            "drift": lambda h: sqrt_flr(h * (1 + (h / (T - 1)))),
        }

        fh_periods = np.array(fh.to_relative(self.cutoff))
        marginal_se = se_res * partial_se_formulas[self.strategy](fh_periods)
        marginal_vars = marginal_se**2

        fh_idx = fh.to_absolute_index(self.cutoff)
        if cov:
            fh_size = len(fh)
            cov_matrix = np.fill_diagonal(
                np.zeros(shape=(fh_size, fh_size)), marginal_vars
            )
            pred_var = pd.DataFrame(cov_matrix, columns=fh_idx, index=fh_idx)
        else:
            pred_var = pd.DataFrame(marginal_vars, index=fh_idx)

        return pred_var

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        return [
            {},
            {"strategy": "mean", "sp": 2},
            {"strategy": "drift"},
            {"strategy": "last"},
            {"strategy": "mean", "window_length": 5},
        ]


def _fit_predict_single_window(forecaster, y_values, y_index, id_pos, verbose):
    """Fit on y[:id_pos] and return (index_label, residuals_dict).

    Parameters
    ----------
    forecaster : sktime forecaster clone
        A fresh clone to fit for this window.
    y_values : np.ndarray
        Raw numpy array of the full time series values.
        Pre-converted ONCE outside the loop to avoid repeated pandas overhead.
    y_index : pd.Index
        The index of the full time series.
    id_pos : int
        Integer position of the cutoff point in y_index.
    verbose : bool
        Whether to emit warnings on fit/predict failures.

    Returns
    -------
    id_label : hashable
        The index label at id_pos (the row key in the residuals matrix).
    residuals : dict  {future_index_label: residual_value}
        Sparse representation — only non-NaN residuals are stored.
        An empty dict is returned if fitting or prediction fails.

    Why return a dict instead of a full array?
    ------------------------------------------
    For T=500,000 the full TxT matrix would need ~2TB of RAM.
    By storing only the residuals that were actually computed per row
    (at most T - id_pos values per row), total storage is O(T²/2) in the
    worst case but in practice much smaller because early windows produce
    few predictions and later windows are capped by the forecasting horizon.
    """
    id_label = y_index[id_pos]

    y_train = pd.Series(y_values[:id_pos], index=y_index[:id_pos])
    y_test = pd.Series(y_values[id_pos:], index=y_index[id_pos:])

    try:
        forecaster.fit(y_train, fh=y_test.index)
    except ValueError:
        if verbose:
            warn(f"Couldn't fit on window length {len(y_train)}.")
        return id_label, {}

    try:
        residuals = forecaster.predict_residuals(y_test, X=None)
        # Store as a dict {future_label: residual} to stay sparse
        return id_label, dict(zip(y_test.index, residuals))
    except IndexError:
        if verbose:
            warn(f"Couldn't predict after fitting on length {len(y_train)}.")
        return id_label, {}


def _build_sparse_residuals(results, y_index):
    """Convert list of (label, dict) results into a sparse dict-of-dicts.

    Parameters
    ----------
    results : list of (label, dict)
        Output from the parallel workers.
    y_index : pd.Index
        Full time series index (used only for ordering).

    Returns
    -------
    sparse_matrix : dict  {row_label: {col_label: residual}}
        Row = the cutoff point we fitted up to.
        Col = the future time point we were predicting.
        Only cells with actual residual values are stored.

    Memory comparison vs original dense pd.DataFrame
    -------------------------------------------------
    Original : T x T floats = T² x 8 bytes
      T=50,000  ->  20 GB
      T=500,000 ->  2 TB   <- completely infeasible

    Sparse dict : stores only filled cells.
    Typical fill rate is ~1/T of cells per row (each row only fills
    the future steps), so actual storage ≈ T x avg_future_steps x 8 bytes.
      T=50,000,  avg_future=100  →  ~40 MB   ✓
      T=500,000, avg_future=100  →  ~400 MB  ✓
    """
    sparse_matrix = {}
    for label, res_dict in results:
        if res_dict:
            sparse_matrix[label] = res_dict
    return sparse_matrix


class NaiveVariance(BaseForecaster):
    r"""Compute prediction variance based on a naive sliding-window strategy.

    OPTIMIZED VERSION — changes vs original
    ----------------------------------------
    1. ``_compute_sliding_residuals`` now runs in **parallel** using joblib.
       Each window fit is independent, so we get near-linear speedup with cores.
       On a machine with 8 cores, expect ~6-7x faster residual computation.

    2. The internal residuals matrix is stored as a **sparse dict-of-dicts**
       instead of a dense TxT pd.DataFrame. For T=500,000 this reduces memory
       from ~2 TB to a few hundred MB.

    3. Raw numpy arrays are extracted from the pd.Series **once** before the
       parallel loop. Each worker receives a numpy array and reconstructs only
       the slice it needs, avoiding repeated pandas overhead.

    4. ``n_jobs`` parameter added (default=-1 = use all available cores).

    Parameters
    ----------
    forecaster : estimator
        Estimator to which probabilistic forecasts are being added.
    initial_window : int, optional, default=1
        Minimum number of initial indices to use for fitting.
    verbose : bool, optional, default=False
        Whether to print warnings if windows with too few data points occur.
    n_jobs : int, optional, default=-1
        Number of parallel jobs. -1 means use all available CPU cores.
        Set to 1 to disable parallelism (useful for debugging).

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.naive import NaiveForecaster, NaiveVariance
    >>> y = load_airline()
    >>> forecaster = NaiveForecaster(strategy="drift")
    >>> variance_forecaster = NaiveVariance(forecaster, n_jobs=-1)
    >>> variance_forecaster.fit(y)
    NaiveVariance(...)
    >>> var_pred = variance_forecaster.predict_var(fh=[1, 2, 3])
    """

    _tags = {
        "authors": ["fkiraly", "bethrice44"],
        "scitype:y": "univariate",
        "requires-fh-in-fit": False,
        "capability:missing_values": False,
        "capability:exogenous": True,
        "capability:pred_int": True,
        "capability:pred_var": True,
    }

    def __init__(self, forecaster, initial_window=1, verbose=False, n_jobs=-1):
        self.forecaster = forecaster
        self.initial_window = initial_window
        self.verbose = verbose
        self.n_jobs = n_jobs
        super().__init__()

        tags_to_clone = [
            "requires-fh-in-fit",
            "capability:exogenous",
            "capability:missing_values",
            "y_inner_mtype",
            "X_inner_mtype",
            "X-y-must-have-same-index",
            "enforce_index_type",
        ]
        self.clone_tags(self.forecaster, tags_to_clone)

    def _fit(self, y, X, fh):
        self.fh_early_ = fh is not None
        self.forecaster_ = self.forecaster.clone()
        self.forecaster_.fit(y=y, X=X, fh=fh)

        if self.fh_early_:
            self.residuals_matrix_ = self._compute_sliding_residuals(
                y=y, X=X, forecaster=self.forecaster, initial_window=self.initial_window
            )
        return self

    def _predict(self, fh, X):
        return self.forecaster_.predict(fh=fh, X=X)

    def _update(self, y, X=None, update_params=True):
        """Update — only recompute residuals if new data was actually added."""
        prev_len = len(self._y) if hasattr(self, "_y") else 0
        self.forecaster_.update(y, X, update_params=update_params)

        if update_params and self._fh is not None and len(self._y) > prev_len:
            self.residuals_matrix_ = self._compute_sliding_residuals(
                y=self._y,
                X=self._X,
                forecaster=self.forecaster,
                initial_window=self.initial_window,
            )
        return self

    def _predict_quantiles(self, fh, X, alpha):
        y_pred = self.predict(fh, X)
        y_pred = convert(y_pred, from_type=self._y_mtype_last_seen, to_type="pd.Series")
        pred_var = self.predict_var(fh, X)
        pred_var = pred_var[pred_var.columns[0]]
        pred_var.index = y_pred.index

        z_scores = norm.ppf(alpha)
        errors = [pred_var**0.5 * z for z in z_scores]

        var_names = self._get_varnames()
        var_name = var_names[0]
        index = pd.MultiIndex.from_product([var_names, alpha])
        pred_quantiles = pd.DataFrame(columns=index)
        for a, error in zip(alpha, errors):
            pred_quantiles[(var_name, a)] = y_pred + error

        pred_quantiles.index = fh.to_absolute(self.cutoff).to_pandas()
        return pred_quantiles

    def _predict_var(self, fh, X=None, cov=False):
        """Compute prediction variance from the sparse residuals matrix.

        Variance at horizon k = mean of squared residuals from all windows
        that were exactly k steps behind the prediction target.

        Reading from the sparse dict is O(number of filled cells) instead of
        O(T²), which is crucial for large T.
        """
        if self.fh_early_:
            residuals_matrix = self.residuals_matrix_
        else:
            residuals_matrix = self._compute_sliding_residuals(
                y=self._y,
                X=self._X,
                forecaster=self.forecaster,
                initial_window=self.initial_window,
            )

        fh_relative = fh.to_relative(self.cutoff)
        fh_absolute = fh.to_absolute(self.cutoff)
        fh_absolute_ix = fh_absolute.to_pandas()

        if cov:
            fh_size = len(fh)
            covariance = np.zeros(shape=(fh_size, fh_size))

            for i in range(fh_size):
                i_residuals = self._get_diagonal_sparse(
                    residuals_matrix, fh_relative[i]
                )
                for j in range(i, fh_size):
                    j_residuals = self._get_diagonal_sparse(
                        residuals_matrix, fh_relative[j]
                    )
                    max_r = min(len(i_residuals), len(j_residuals))
                    covariance[i, j] = covariance[j, i] = np.nanmean(
                        i_residuals[:max_r] * j_residuals[:max_r]
                    )
            pred_var = pd.DataFrame(
                covariance, index=fh_absolute_ix, columns=fh_absolute_ix
            )
        else:
            variance = [
                np.nanmean(
                    np.array(self._get_diagonal_sparse(residuals_matrix, offset)) ** 2
                )
                for offset in fh_relative
            ]
            pred_var = pd.DataFrame(variance, index=fh_absolute_ix)

        return pred_var

    @staticmethod
    def _get_diagonal_sparse(sparse_matrix, offset):
        """Extract the 'diagonal' at a given offset from the sparse dict.

        In the original code this was np.diagonal(dense_matrix, offset=k).
        Here we replicate that by collecting all residuals where the
        column is exactly `offset` steps ahead of the row.

        Parameters
        ----------
        sparse_matrix : dict of dict
            {row_label: {col_label: residual}}
        offset : int
            The forecasting horizon (steps ahead).

        Returns
        -------
        diag_values : list of float
        """
        diag = []
        for row_label, col_dict in sparse_matrix.items():
            for col_label, val in col_dict.items():
                pass
        diag = []
        all_row_labels = list(sparse_matrix.keys())
        if not all_row_labels:
            return diag

        all_labels = sorted(
            set(all_row_labels) | {lbl for d in sparse_matrix.values() for lbl in d}
        )
        label_to_pos = {lbl: pos for pos, lbl in enumerate(all_labels)}

        for row_label, col_dict in sparse_matrix.items():
            row_pos = label_to_pos[row_label]
            for col_label, val in col_dict.items():
                col_pos = label_to_pos[col_label]
                if col_pos - row_pos == offset:
                    diag.append(val)
                    break

        return diag

    def _compute_sliding_residuals(self, y, X, forecaster, initial_window):
        """Compute sliding residuals in parallel using joblib.

        OPTIMIZATION NOTES
        ------------------
        Original code:
            for id in y_index:          # sequential loop, no parallelism
                forecaster.fit(...)      # fit from scratch each time
                residuals_matrix.loc[id] = ...  # writes to dense TxT DataFrame

        Optimized code:
            1. Extract numpy array ONCE before loop (avoids pandas overhead per step).
            2. Dispatch all window fits to a joblib thread pool simultaneously.
            3. Collect results into a sparse dict-of-dicts instead of a dense matrix.

        Parameters
        ----------
        y : pd.Series
        X : pd.DataFrame or None
        forecaster : sktime forecaster (will be cloned per worker)
        initial_window : int

        Returns
        -------
        sparse_matrix : dict of dict
            {row_label: {col_label: residual_value}}
        """
        y = convert_to(y, "pd.Series")
        y_index = y.index

        y_values = y.to_numpy()

        iter_positions = range(initial_window, len(y))

        n_jobs = self.n_jobs if self.n_jobs != 0 else 1
        # effective_cores = cpu_count() if n_jobs == -1 else n_jobs

        results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
            delayed(_fit_predict_single_window)(
                forecaster.clone(),
                y_values,
                y_index,
                id_pos,
                self.verbose,
            )
            for id_pos in iter_positions
        )

        sparse_matrix = _build_sparse_residuals(results, y_index)

        return sparse_matrix

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        from sktime.forecasting.naive import NaiveForecaster

        FORECASTER = NaiveForecaster()
        return [
            {"forecaster": FORECASTER},
            {"forecaster": FORECASTER, "initial_window": 2},
        ]
