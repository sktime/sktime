"""Pretrain-aware reduction forecaster."""

__author__ = ["felipeangelimvieira"]
__all__ = [
    "BaseWindowNormalizer",
    "MeanWindowNormalizer",
    "SubtractMeanNormalizer",
    "ZScoreWindowNormalizer",
    "MinMaxWindowNormalizer",
    "ReductionForecaster",
]

import numpy as np
import pandas as pd
from sklearn.base import clone

from sktime.base import BaseObject
from sktime.forecasting.base._base import BaseForecaster


class BaseWindowNormalizer(BaseObject):
    """Base contract for row-wise reduction normalization strategies.

    A normalizer uses the current lag window as context. It transforms lag
    features and, optionally, the supervised target using the same context, and
    can invert a normalized prediction back to the original scale. Subclasses
    should implement private methods. Public methods handle input coercion and
    delegate to the private implementations.
    """

    _tags = {
        "object_type": "window-normalizer",
        "capability:missing_values": True,
    }

    def __init__(self):
        super().__init__()

    def transform(self, lags, target=None):
        """Transform a lag window and optional target."""
        lags = self._coerce_lag_vector(lags)
        if target is not None:
            target = float(target)
        return self._transform(lags, target=target)

    def _transform(self, lags, target=None):
        """Transform a lag window and optional target.

        private _transform containing core logic, called from transform.
        """
        lags = np.asarray(lags, dtype=float)
        loc, scale = self._loc_scale(lags)
        lags_t = (lags - loc) / scale
        if target is None:
            return lags_t, None
        return lags_t, (float(target) - loc) / scale

    def inverse_transform(self, y, lags):
        """Invert a transformed prediction using lag-window context."""
        lags = self._coerce_lag_vector(lags)
        y = float(y)
        return self._inverse_transform(y, lags)

    def _inverse_transform(self, y, lags):
        """Invert a transformed prediction using lag-window context.

        private _inverse_transform containing core logic, called from
        inverse_transform.
        """
        lags = np.asarray(lags, dtype=float)
        loc, scale = self._loc_scale(lags)
        return float(y) * scale + loc

    def batch_transform(self, lags, target=None):
        """Transform lag-window rows and optional target values.

        Parameters
        ----------
        lags : array-like of shape (n_rows, window_length)
            Lag windows to transform row-wise.
        target : array-like of shape (n_rows,), optional
            Target values to transform with the corresponding lag-window context.

        Returns
        -------
        lags_t : np.ndarray of shape (n_rows, window_length)
            Transformed lag windows.
        target_t : np.ndarray of shape (n_rows,) or None
            Transformed targets, or ``None`` if no target was passed.
        """
        lags = self._coerce_lag_matrix(lags)
        if target is not None:
            target = self._coerce_target_vector(target, n_rows=lags.shape[0])
        return self._batch_transform(lags, target=target)

    def _batch_transform(self, lags, target=None):
        """Transform lag-window rows and optional target values.

        private _batch_transform containing core logic, called from
        batch_transform.
        """
        lags_t = np.empty_like(lags, dtype=float)

        if target is None:
            for i, lag_row in enumerate(lags):
                lag_row_t, _ = self._transform(lag_row, None)
                lags_t[i] = lag_row_t
            return lags_t, None

        target_t = np.empty(lags.shape[0], dtype=float)

        for i, (lag_row, target_value) in enumerate(zip(lags, target)):
            lag_row_t, target_value_t = self._transform(lag_row, target_value)
            lags_t[i] = lag_row_t
            target_t[i] = target_value_t

        return lags_t, target_t

    def batch_inverse_transform(self, y, lags):
        """Invert transformed prediction rows using lag-window context."""
        lags = self._coerce_lag_matrix(lags)
        y = self._coerce_target_vector(y, n_rows=lags.shape[0])
        return self._batch_inverse_transform(y, lags)

    def _batch_inverse_transform(self, y, lags):
        """Invert transformed prediction rows using lag-window context.

        private _batch_inverse_transform containing core logic, called from
        batch_inverse_transform.
        """
        return np.asarray(
            [
                self._inverse_transform(y_value, lag_row)
                for y_value, lag_row in zip(y, lags)
            ],
            dtype=float,
        )

    def _batch_transform_from_loc_scale(self, lags, target=None):
        """Vectorized batch transform for location-scale normalizers."""
        lags = self._coerce_lag_matrix(lags)
        loc, scale = self._batch_loc_scale(lags)
        lags_t = (lags - loc[:, None]) / scale[:, None]

        if target is None:
            return lags_t, None

        target = self._coerce_target_vector(target, n_rows=lags.shape[0])
        return lags_t, (target - loc) / scale

    def _batch_inverse_from_loc_scale(self, y, lags):
        """Vectorized batch inverse for location-scale normalizers."""
        lags = self._coerce_lag_matrix(lags)
        y = self._coerce_target_vector(y, n_rows=lags.shape[0])
        loc, scale = self._batch_loc_scale(lags)
        return y * scale + loc

    def _batch_loc_scale(self, lags):
        """Return vectorized location and scale for lag-window rows."""
        loc = np.empty(lags.shape[0], dtype=float)
        scale = np.empty(lags.shape[0], dtype=float)
        for i, lag_row in enumerate(lags):
            loc[i], scale[i] = self._loc_scale(lag_row)
        return loc, scale

    def _loc_scale(self, lags):
        """Return location and scale for a lag window."""
        return 0.0, 1.0

    @staticmethod
    def _finite_or(value, fallback):
        """Return value if finite, otherwise fallback."""
        value = float(value)
        return value if np.isfinite(value) else fallback

    @staticmethod
    def _coerce_lag_vector(lags):
        """Coerce lag input to a 1D float array."""
        lags = np.asarray(lags, dtype=float)
        if lags.ndim != 1:
            raise ValueError("lags must be a 1D array-like.")
        return lags

    @staticmethod
    def _clean_scale(scale):
        """Return finite, non-zero scale array."""
        scale = np.asarray(scale, dtype=float)
        scale = np.where(np.isfinite(scale), scale, 1.0)
        scale = np.where(np.abs(scale) < 1e-12, 1.0, scale)
        return scale

    @staticmethod
    def _coerce_lag_matrix(lags):
        """Coerce lag input to a 2D float array."""
        lags = np.asarray(lags, dtype=float)
        if lags.ndim == 1:
            lags = lags.reshape(1, -1)
        if lags.ndim != 2:
            raise ValueError("lags must be a 1D or 2D array-like.")
        return lags

    @staticmethod
    def _coerce_target_vector(target, n_rows):
        """Coerce target input to a 1D float array matching lag rows."""
        target = np.asarray(target, dtype=float)
        if target.ndim == 0:
            target = target.reshape(1)
        if target.ndim != 1:
            raise ValueError("target must be a 1D array-like.")
        if target.shape[0] != n_rows:
            raise ValueError(
                "target must have the same number of rows as lags; "
                f"found {target.shape[0]} and {n_rows}."
            )
        return target


class _VectorizedLocScaleWindowNormalizer(BaseWindowNormalizer):
    """Base class for normalizers with vectorized location-scale batches."""

    def _batch_transform(self, lags, target=None):
        """Transform lag-window rows and optional target values."""
        return self._batch_transform_from_loc_scale(lags, target=target)

    def _batch_inverse_transform(self, y, lags):
        """Invert transformed prediction rows using lag-window context."""
        return self._batch_inverse_from_loc_scale(y, lags)


class MeanWindowNormalizer(_VectorizedLocScaleWindowNormalizer):
    """Scale values by the lag-window mean."""

    def _loc_scale(self, lags):
        mean = np.nanmean(lags) if lags.size else 1.0
        scale = self._finite_or(mean, 1.0)
        if abs(scale) < 1e-12:
            scale = 1.0
        return 0.0, scale

    def _batch_loc_scale(self, lags):
        """Return vectorized location and scale for lag-window rows."""
        n_rows = lags.shape[0]
        loc = np.zeros(n_rows, dtype=float)
        scale = np.nanmean(lags, axis=1) if lags.shape[1] else np.ones(n_rows)
        scale = self._clean_scale(scale)
        return loc, scale


class SubtractMeanNormalizer(_VectorizedLocScaleWindowNormalizer):
    """Center values by subtracting the lag-window mean."""

    def _loc_scale(self, lags):
        mean = np.nanmean(lags) if lags.size else 0.0
        loc = self._finite_or(mean, 0.0)
        return loc, 1.0

    def _batch_loc_scale(self, lags):
        """Return vectorized location and scale for lag-window rows."""
        n_rows = lags.shape[0]
        loc = np.nanmean(lags, axis=1) if lags.shape[1] else np.zeros(n_rows)
        loc = np.where(np.isfinite(loc), loc, 0.0)
        scale = np.ones(n_rows, dtype=float)
        return loc, scale


class ZScoreWindowNormalizer(_VectorizedLocScaleWindowNormalizer):
    """Standardize values by lag-window mean and standard deviation."""

    def _loc_scale(self, lags):
        mean = np.nanmean(lags) if lags.size else 0.0
        std = np.nanstd(lags) if lags.size else 1.0
        loc = self._finite_or(mean, 0.0)
        scale = self._finite_or(std, 1.0)
        if abs(scale) < 1e-12:
            scale = 1.0
        return loc, scale

    def _batch_loc_scale(self, lags):
        """Return vectorized location and scale for lag-window rows."""
        n_rows = lags.shape[0]
        loc = np.nanmean(lags, axis=1) if lags.shape[1] else np.zeros(n_rows)
        loc = np.where(np.isfinite(loc), loc, 0.0)
        scale = np.nanstd(lags, axis=1) if lags.shape[1] else np.ones(n_rows)
        scale = self._clean_scale(scale)
        return loc, scale


class MinMaxWindowNormalizer(_VectorizedLocScaleWindowNormalizer):
    """Scale values by lag-window minimum and range."""

    def _loc_scale(self, lags):
        if lags.size:
            loc = self._finite_or(np.nanmin(lags), 0.0)
            high = self._finite_or(np.nanmax(lags), loc + 1.0)
        else:
            loc = 0.0
            high = 1.0
        scale = high - loc
        if not np.isfinite(scale) or abs(scale) < 1e-12:
            scale = 1.0
        return loc, scale

    def _batch_loc_scale(self, lags):
        """Return vectorized location and scale for lag-window rows."""
        n_rows = lags.shape[0]
        if lags.shape[1]:
            loc_raw = np.nanmin(lags, axis=1)
            loc = np.where(np.isfinite(loc_raw), loc_raw, 0.0)
            high_raw = np.nanmax(lags, axis=1)
            high = np.where(np.isfinite(high_raw), high_raw, loc + 1.0)
        else:
            loc = np.zeros(n_rows, dtype=float)
            high = np.ones(n_rows, dtype=float)
        scale = self._clean_scale(high - loc)
        return loc, scale


_NORMALIZER_ALIASES = {
    "mean": MeanWindowNormalizer,
    "divide_mean": MeanWindowNormalizer,
    "subtract_mean": SubtractMeanNormalizer,
    "zscore": ZScoreWindowNormalizer,
    "normalize": ZScoreWindowNormalizer,
    "minmax": MinMaxWindowNormalizer,
}

_FIT_CONTEXT_ATTRS = (
    "last_window_",
    "train_index_",
    "y_was_dataframe_",
    "y_name_",
)


def _resolve_normalizer(normalization_strategy):
    """Resolve a normalizer alias or object to a BaseWindowNormalizer."""
    if normalization_strategy is None:
        return None

    if isinstance(normalization_strategy, str):
        key = normalization_strategy.lower()
        if key not in _NORMALIZER_ALIASES:
            valid = sorted(_NORMALIZER_ALIASES)
            raise ValueError(
                "Unknown normalization_strategy. "
                f"Expected one of {valid}, but found {normalization_strategy!r}."
            )
        return _NORMALIZER_ALIASES[key]()

    if isinstance(normalization_strategy, BaseWindowNormalizer):
        return normalization_strategy.clone()

    if callable(normalization_strategy):
        resolved = normalization_strategy()
        if isinstance(resolved, BaseWindowNormalizer):
            return resolved

    raise TypeError(
        "normalization_strategy must be None, a string alias, a "
        "BaseWindowNormalizer, or a callable returning a BaseWindowNormalizer."
    )


def _check_regressor(estimator):
    """Check that estimator follows the sklearn regressor protocol."""
    if not hasattr(estimator, "fit") or not hasattr(estimator, "predict"):
        raise TypeError("estimator must implement fit and predict.")


def _coerce_univariate_y(y):
    """Return y as a Series plus output metadata."""
    if isinstance(y, pd.DataFrame):
        if y.shape[1] != 1:
            raise ValueError("ReductionForecaster supports univariate y only.")
        name = y.columns[0]
        return y.iloc[:, 0].rename(name), True, name
    if isinstance(y, pd.Series):
        return y.copy(), False, y.name
    raise TypeError("ReductionForecaster expects y as a pandas Series or DataFrame.")


def _coerce_pretrain_y(y):
    """Return y as pooled univariate series for global pretraining."""
    if isinstance(y, pd.Series):
        return y.copy()
    if not isinstance(y, pd.DataFrame):
        raise TypeError(
            "ReductionForecaster expects y as a pandas Series or DataFrame."
        )
    if y.shape[1] == 1:
        return y.iloc[:, 0].rename(y.columns[0])

    pieces = []
    if isinstance(y.index, pd.MultiIndex):
        id_arrays = [
            y.index.get_level_values(level) for level in range(y.index.nlevels - 1)
        ]
        id_names = list(y.index.names[:-1])
        time_values = y.index.get_level_values(-1)
        time_name = y.index.names[-1]
    else:
        id_arrays = []
        id_names = []
        time_values = y.index
        time_name = y.index.name

    for column in y.columns:
        y_col = y.loc[:, column].copy()
        column_values = pd.Index(
            np.repeat(str(column), len(y_col)),
            name="__variable__",
        )
        y_col.index = pd.MultiIndex.from_arrays(
            [*id_arrays, column_values, time_values],
            names=[*id_names, "__variable__", time_name],
        )
        pieces.append(y_col)

    return pd.concat(pieces)


def _sort_time_index(y):
    """Sort a single series by time if needed."""
    if not y.index.is_monotonic_increasing:
        return y.sort_index()
    return y


def _iter_series_groups(y):
    """Yield (id_tuple, series_with_time_index) from single or MultiIndex y."""
    if isinstance(y.index, pd.MultiIndex):
        id_levels = list(range(y.index.nlevels - 1))
        group_level = id_levels[0] if len(id_levels) == 1 else id_levels
        for ids, y_group in y.groupby(level=group_level, sort=False):
            if not isinstance(ids, tuple):
                ids = (ids,)
            yield ids, _sort_time_index(y_group.droplevel(id_levels))
    else:
        yield ("__local__",), _sort_time_index(y)


def _n_instances(y):
    """Return number of series instances in y."""
    if isinstance(y.index, pd.MultiIndex):
        return len(y.index.droplevel(-1).unique())
    return 1


def _coerce_group_X(X, ids):
    """Return X slice for one group, or None."""
    if X is None:
        return None
    if isinstance(X.index, pd.MultiIndex):
        id_levels = list(range(X.index.nlevels - 1))
        x_ids = ids[: len(id_levels)]
        x_levels = id_levels[: len(x_ids)]
        if len(x_ids) == 1:
            key = x_ids[0]
            level = x_levels[0]
        else:
            key = x_ids
            level = x_levels
        X = X.xs(key, level=level)
    if not X.index.is_monotonic_increasing:
        X = X.sort_index()
    return X


def _prepare_supervised_group_cache(y, X, window_length):
    """Prepare per-series NumPy data reused by all horizon heads."""
    group_cache = []

    for ids, y_group in _iter_series_groups(y):
        values = y_group.to_numpy(dtype=float)
        if values.ndim == 2:
            values = values.ravel()
        X_group = _coerce_group_X(X, ids)

        if len(values) <= window_length:
            continue

        windows = np.lib.stride_tricks.sliding_window_view(values, window_length)[:-1]
        cache = {
            "values": values,
            "windows": windows,
            "X_targets": None,
        }

        if X_group is not None:
            X_group = X_group.loc[~X_group.index.duplicated(keep="first")]
            target_index = y_group.index[window_length:]
            indexer = X_group.index.get_indexer(target_index)
            if np.any(indexer == -1):
                raise ValueError(
                    "X must contain rows at all supervised target timestamps."
                )
            cache["X_targets"] = X_group.to_numpy(dtype=float)[indexer]

        group_cache.append(cache)

    return group_cache


def _build_supervised_table_from_cache(
    group_cache, window_length, steps_ahead, normalizer
):
    """Build one horizon's supervised table from reusable group cache."""
    X_blocks = []
    y_blocks = []

    for cache in group_cache:
        values = cache["values"]
        n_rows = len(values) - window_length - steps_ahead + 1

        if n_rows <= 0:
            continue

        lags = cache["windows"][:n_rows]
        target = values[window_length + steps_ahead - 1 :]

        if normalizer is None:
            lags_features = lags
        else:
            lags_features, target = normalizer.batch_transform(lags, target)

        X_targets = cache["X_targets"]
        if X_targets is not None:
            start = steps_ahead - 1
            stop = start + n_rows
            X_target_block = X_targets[start:stop]
            lags_features = np.hstack([lags_features, X_target_block])

        X_blocks.append(lags_features)
        y_blocks.append(np.asarray(target, dtype=float))

    if not X_blocks:
        raise ValueError(
            "Not enough observations to build reduction rows. Need at least "
            "window_length + steps_ahead observations in one series."
        )

    return np.concatenate(X_blocks, axis=0), np.concatenate(y_blocks, axis=0)


def _build_supervised_table(y, X, window_length, steps_ahead, normalizer):
    """Build pooled lag-to-target table for one horizon."""
    group_cache = _prepare_supervised_group_cache(
        y=y,
        X=X,
        window_length=window_length,
    )
    return _build_supervised_table_from_cache(
        group_cache=group_cache,
        window_length=window_length,
        steps_ahead=steps_ahead,
        normalizer=normalizer,
    )


class ReductionForecaster(BaseForecaster):
    """Global reduction forecaster with pretrainable direct heads.

    The forecaster trains one sklearn-style regressor per horizon step up to
    ``steps_ahead``. Forecasts beyond that range are produced recursively in
    blocks of ``steps_ahead``, reusing the direct heads for each block.
    """

    _tags = {
        "capability:pretrain": True,
        "capability:exogenous": True,
        "capability:insample": False,
        "capability:pred_int": False,
        "capability:missing_values": False,
        "capability:non_contiguous_X": False,
        "capability:multivariate": False,
        "enforce_index_type": None,
        "requires-fh-in-fit": False,
        "y_inner_mtype": [
            "pd.Series",
            "pd.DataFrame",
            "pd-multiindex",
            "pd_multiindex_hier",
        ],
        "X_inner_mtype": ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"],
        "X-y-must-have-same-index": True,
        "python_dependencies": "scikit-learn",
    }

    def __init__(
        self,
        estimator,
        window_length=10,
        steps_ahead=1,
        normalization_strategy=None,
    ):
        self.estimator = estimator
        self.window_length = window_length
        self.steps_ahead = steps_ahead
        self.normalization_strategy = normalization_strategy
        super().__init__()

    def _fit_heads(self, y, X=None):
        """Fit direct heads on y and optional X."""
        _check_regressor(self.estimator)
        if not isinstance(self.window_length, int) or self.window_length < 1:
            raise ValueError("window_length must be a positive integer.")
        if not isinstance(self.steps_ahead, int) or self.steps_ahead < 1:
            raise ValueError("steps_ahead must be a positive integer.")

        normalizer = _resolve_normalizer(self.normalization_strategy)
        group_cache = _prepare_supervised_group_cache(
            y=y,
            X=X,
            window_length=self.window_length,
        )
        estimators = []
        for step in range(1, self.steps_ahead + 1):
            Xt, yt = _build_supervised_table_from_cache(
                group_cache=group_cache,
                window_length=self.window_length,
                steps_ahead=step,
                normalizer=normalizer,
            )
            estimator = clone(self.estimator)
            estimator.fit(Xt, yt)
            estimators.append(estimator)

        self.normalizer_ = normalizer
        self.direct_estimators_ = estimators
        self.one_step_estimator_ = estimators[0]
        self.x_columns_ = list(X.columns) if X is not None else None
        return self

    def _pretrain(self, y, X=None, fh=None):
        """Pretrain direct heads on panel or hierarchical data."""
        y = _coerce_pretrain_y(y)
        self._fit_heads(y, X=X)
        self.heads_source_ = "pretrain"
        self.n_pretrain_instances_ = _n_instances(y)
        self.n_pretrain_timepoints_ = len(y)
        return self

    def _pretrain_update(self, y, X=None, fh=None):
        """Update pretrained direct heads with a new global batch."""
        for attr in _FIT_CONTEXT_ATTRS:
            if hasattr(self, attr):
                delattr(self, attr)
        return self._pretrain(y=y, X=X, fh=fh)

    def _fit(self, y, X=None, fh=None):
        """Fit local forecasting context, preserving pretrained heads."""
        y, y_was_dataframe, y_name = _coerce_univariate_y(y)
        y = _sort_time_index(y)

        if hasattr(self, "direct_estimators_"):
            if self.x_columns_ is None and X is not None:
                raise ValueError(
                    "ReductionForecaster was pretrained without X and cannot "
                    "add exogenous features in fit."
                )
        else:
            self._fit_heads(y, X=X)
            self.heads_source_ = "fit"

        if isinstance(y.index, pd.MultiIndex):
            self._store_panel_fit_context(y)
        else:
            if len(y) < self.window_length:
                raise ValueError(
                    "Need at least window_length observations to store local context."
                )
            self.is_panel_ = False
            self.last_window_ = y.iloc[-self.window_length :].to_numpy(dtype=float)

        self.train_index_ = y.index
        self.y_was_dataframe_ = y_was_dataframe
        self.y_name_ = y_name
        return self

    def _store_panel_fit_context(self, y):
        """Store per-instance last windows for panel/hierarchical prediction."""
        self.is_panel_ = True
        self.group_last_windows_ = {}
        self.group_cutoffs_ = {}
        self.group_ids_ = []
        self.y_id_names_ = list(y.index.names[:-1])
        self.y_time_name_ = y.index.names[-1]

        for ids, y_group in _iter_series_groups(y):
            if len(y_group) < self.window_length:
                raise ValueError(
                    "Need at least window_length observations in every series "
                    "to store local context."
                )
            self.group_ids_.append(ids)
            self.group_last_windows_[ids] = y_group.iloc[
                -self.window_length :
            ].to_numpy(dtype=float)
            self.group_cutoffs_[ids] = y_group.index[-1:]

        first_id = self.group_ids_[0]
        self.last_window_ = self.group_last_windows_[first_id].copy()

    def _make_prediction_row(self, lags, X_row=None):
        """Make a single regressor row from lags and optional exogenous values."""
        normalizer = getattr(self, "normalizer_", None)
        if normalizer is not None:
            lags = np.asarray(lags, dtype=float).reshape(1, -1)
            lags_t, _ = normalizer.batch_transform(lags)
            lags_t = lags_t[0]
        else:
            lags_t = np.asarray(lags, dtype=float)

        if X_row is None:
            return lags_t.reshape(1, -1)
        return np.concatenate([lags_t, np.asarray(X_row, dtype=float)]).reshape(1, -1)

    def _invert_prediction(self, y_pred, lags):
        """Invert normalized prediction if a normalizer is active."""
        normalizer = getattr(self, "normalizer_", None)
        if normalizer is None:
            return float(y_pred)
        y_pred = np.asarray([y_pred], dtype=float)
        lags = np.asarray(lags, dtype=float).reshape(1, -1)
        return normalizer.batch_inverse_transform(y_pred, lags)[0]

    def _predict_all_steps(self, horizon, X=None, last_window=None):
        """Predict all relative steps from 1 to horizon in direct-recursive blocks."""
        direct_estimators = self.direct_estimators_
        if last_window is None:
            rolling_window = self.last_window_.copy()
        else:
            rolling_window = np.asarray(last_window, dtype=float).copy()
        preds = np.zeros(horizon, dtype=float)
        X_block = X

        for block_start in range(0, horizon, self.steps_ahead):
            block_window = rolling_window.copy()
            block_size = min(self.steps_ahead, horizon - block_start)

            for block_offset in range(block_size):
                step = block_start + block_offset
                X_row = None if X_block is None else X_block[step]
                row = self._make_prediction_row(block_window, X_row=X_row)
                yhat = direct_estimators[block_offset].predict(row)[0]
                preds[step] = self._invert_prediction(yhat, block_window)

            for yhat in preds[block_start : block_start + block_size]:
                rolling_window = np.roll(rolling_window, -1)
                rolling_window[-1] = yhat

        return preds

    def _get_future_X_block(self, X, full_index):
        """Return future X rows aligned to full out-of-sample index."""
        if self.x_columns_ is None:
            return None
        if X is None:
            raise ValueError(
                "This ReductionForecaster was trained with X; pass future X to predict."
            )

        missing_columns = [col for col in self.x_columns_ if col not in X.columns]
        if missing_columns:
            raise ValueError(
                f"X is missing columns seen in training: {missing_columns}"
            )

        X_future = X.loc[:, self.x_columns_].reindex(full_index)
        if X_future.isna().to_numpy().any():
            missing_index = X_future.index[X_future.isna().any(axis=1)]
            raise ValueError(
                "X must contain rows for all required future timestamps. "
                f"Missing examples: {list(missing_index[:3])}"
            )
        return X_future.to_numpy(dtype=float)

    def _predict_values_for_window(self, fh, cutoff, last_window, X=None):
        """Predict one fitted series context and return values plus index."""
        rel = fh.to_relative(cutoff).to_pandas()
        rel_values = np.asarray(rel, dtype=int)

        if np.any(rel_values < 1):
            raise NotImplementedError(
                "ReductionForecaster currently supports out-of-sample fh only."
            )

        horizon = int(np.max(rel_values))
        full_fh = fh.__class__(np.arange(1, horizon + 1), is_relative=True)
        full_index = full_fh.to_absolute_index(cutoff)
        X_block = self._get_future_X_block(X, full_index)
        all_preds = self._predict_all_steps(horizon, X=X_block, last_window=last_window)
        values = np.asarray([all_preds[step - 1] for step in rel_values], dtype=float)
        index = fh.to_absolute_index(cutoff)
        return values, index

    def _predict_panel(self, fh, X=None):
        """Predict all panel/hierarchical instances from stored local contexts."""
        all_values = []
        all_index = []

        for ids in self.group_ids_:
            X_group = _coerce_group_X(X, ids)
            values, index = self._predict_values_for_window(
                fh=fh,
                cutoff=self.group_cutoffs_[ids],
                last_window=self.group_last_windows_[ids],
                X=X_group,
            )
            all_values.extend(values)
            all_index.extend((*ids, time) for time in index)

        index = pd.MultiIndex.from_tuples(
            all_index,
            names=[*self.y_id_names_, self.y_time_name_],
        )
        values = np.asarray(all_values, dtype=float)

        if getattr(self, "y_was_dataframe_", False):
            return pd.DataFrame(values, index=index, columns=[self.y_name_])
        return pd.Series(values, index=index, name=self.y_name_)

    def _predict(self, fh, X=None):
        """Predict relative out-of-sample horizons."""
        if getattr(self, "is_panel_", False):
            return self._predict_panel(fh=fh, X=X)

        values, index = self._predict_values_for_window(
            fh=fh,
            cutoff=self.cutoff,
            last_window=self.last_window_,
            X=X,
        )

        if getattr(self, "y_was_dataframe_", False):
            return pd.DataFrame(values, index=index, columns=[self.y_name_])
        return pd.Series(values, index=index, name=self.y_name_)

    def _get_fitted_params(self):
        """Return fitted context and model parameters."""
        params = {
            "last_window": self.last_window_.copy(),
        }
        if getattr(self, "heads_source_", None) == "fit":
            params["direct_estimators"] = self.direct_estimators_
            params["one_step_estimator"] = self.one_step_estimator_
            params["normalizer"] = self.normalizer_
        return params

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return test parameters."""
        from sklearn.linear_model import LinearRegression, Ridge

        params = {
            "estimator": LinearRegression(),
            "window_length": 3,
            "steps_ahead": 2,
            "normalization_strategy": "mean",
        }
        params_alt = {
            "estimator": Ridge(alpha=0.1),
            "window_length": 4,
            "steps_ahead": 1,
            "normalization_strategy": None,
        }
        return [params, params_alt]
