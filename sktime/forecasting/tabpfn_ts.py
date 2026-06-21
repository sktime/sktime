# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements TabPFN-TS forecaster."""

__author__ = ["keshavnanda"]
__all__ = ["TabPFNTSForecaster"]

import numpy as np
import pandas as pd

from sktime.forecasting.base import BaseForecaster
from sktime.utils.singleton import _multiton


class TabPFNTSForecaster(BaseForecaster):
    """Interface to TabPFN-TS by Prior Labs.

    TabPFN-TS is a zero-shot forecasting model that frames time series forecasting
    as tabular regression with TabPFN and temporal feature engineering.

    Parameters
    ----------
    max_context_length : int, default=4096
        Maximum number of historical time points used as context.
    temporal_features : list or None, default=None
        Temporal feature generators passed to ``tabpfn_time_series.TabPFNTSPipeline``.
        If None, TabPFN-TS uses its default feature generators.
    tabpfn_mode : {"client", "local"}, default="client"
        TabPFN inference mode. ``"client"`` uses the Prior Labs API client,
        ``"local"`` uses a local TabPFN regressor.
    tabpfn_output_selection : {"mean", "median", "mode"}, default="median"
        Aggregation used by TabPFN-TS for point predictions.
    tabpfn_model_config : dict or None, default=None
        Configuration dictionary for the underlying TabPFN model.
    quantiles : list of float or None, default=None
        Quantiles requested from TabPFN-TS. These are used by probabilistic methods
        where available and are also passed during point prediction.
    ignore_deps : bool, default=False
        If True, dependency checks are skipped.

    Attributes
    ----------
    pipeline_ : tabpfn_time_series.TabPFNTSPipeline
        Underlying TabPFN-TS forecasting pipeline.

    References
    ----------
    .. [1] https://github.com/PriorLabs/tabpfn-time-series
    .. [2] Hoo, S. B., and others. Zero-shot Time Series Forecasting with TabPFN.

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.tabpfn_ts import TabPFNTSForecaster
    >>> y = load_airline()
    >>> forecaster = TabPFNTSForecaster()  # doctest: +SKIP
    >>> forecaster.fit(y)  # doctest: +SKIP
    TabPFNTSForecaster(...)
    >>> y_pred = forecaster.predict(fh=[1, 2, 3])  # doctest: +SKIP
    """

    _tags = {
        "authors": ["keshavnanda"],
        "maintainers": ["keshavnanda"],
        "python_dependencies": "tabpfn-time-series",
        "y_inner_mtype": ["pd.Series", "pd.DataFrame"],
        "X_inner_mtype": ["pd.Series", "pd.DataFrame", "None"],
        "capability:exogenous": True,
        "capability:insample": False,
        "capability:missing_values": True,
        "capability:multivariate": True,
        "capability:pred_int": False,
        "requires-fh-in-fit": False,
        "X-y-must-have-same-index": True,
        "tests:vm": True,
        "tests:skip_by_name": [
            "test_persistence_via_pickle",
            "test_save_estimators_to_file",
        ],
    }

    _DEFAULT_QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    def __init__(
        self,
        max_context_length=4096,
        temporal_features=None,
        tabpfn_mode="client",
        tabpfn_output_selection="median",
        tabpfn_model_config=None,
        quantiles=None,
        ignore_deps=False,
    ):
        self.max_context_length = max_context_length
        self.temporal_features = temporal_features
        self.tabpfn_mode = tabpfn_mode
        self.tabpfn_output_selection = tabpfn_output_selection
        self.tabpfn_model_config = tabpfn_model_config
        self.quantiles = quantiles
        self.ignore_deps = ignore_deps

        if ignore_deps:
            self.set_tags(python_dependencies=[])

        super().__init__()

    def _get_pipeline_class(self):
        """Return TabPFN-TS pipeline class."""
        from tabpfn_time_series import TabPFNTSPipeline

        return TabPFNTSPipeline

    def _get_tabpfn_mode(self):
        """Return TabPFN-TS mode enum value."""
        from tabpfn_time_series import TabPFNMode

        mode = self.tabpfn_mode
        if isinstance(mode, str):
            mode = mode.lower()
            if mode == "client":
                return TabPFNMode.CLIENT
            if mode == "local":
                return TabPFNMode.LOCAL
        return mode

    def _get_quantiles(self):
        """Return quantiles passed to TabPFN-TS."""
        if self.quantiles is None:
            return self._DEFAULT_QUANTILES
        return self.quantiles

    def _make_pipeline(self):
        """Construct TabPFN-TS pipeline."""
        Pipeline = self._get_pipeline_class()

        pipeline_kwargs = {
            "max_context_length": self.max_context_length,
            "tabpfn_mode": self._get_tabpfn_mode(),
            "tabpfn_output_selection": self.tabpfn_output_selection,
        }
        if self.temporal_features is not None:
            pipeline_kwargs["temporal_features"] = self.temporal_features
        if self.tabpfn_model_config is not None:
            pipeline_kwargs["tabpfn_model_config"] = self.tabpfn_model_config

        key = str(sorted((key, repr(value)) for key, value in pipeline_kwargs.items()))
        return _CachedTabPFNTSPipeline(
            key=key,
            pipeline_cls=Pipeline,
            pipeline_kwargs=pipeline_kwargs,
        ).load()

    def _fit(self, y, X=None, fh=None):
        """Fit forecaster to training data."""
        self.pipeline_ = self._make_pipeline()
        self._context_df = self._to_context_df(y, X)
        self._is_univariate = not isinstance(y, pd.DataFrame) or y.shape[1] == 1
        if isinstance(y, pd.DataFrame):
            self._y_columns = y.columns
        else:
            self._y_columns = pd.Index([y.name if y.name is not None else "target"])
        return self

    def _predict(self, fh, X=None):
        """Forecast time series at future horizon."""
        fh_rel = fh.to_relative(self.cutoff).to_pandas().astype(int)
        prediction_length = int(fh_rel.max())
        fh_abs = fh.to_absolute(self.cutoff).to_pandas()

        future_df = self._to_future_df(prediction_length, X)
        pred = self.pipeline_.predict_df(
            self._context_df,
            future_df=future_df,
            quantiles=self._get_quantiles(),
        )

        values = self._format_predictions(pred, fh_rel)
        if self._is_univariate:
            return pd.Series(values[:, 0], index=fh_abs, name=self._y_columns[0])
        return pd.DataFrame(values, index=fh_abs, columns=self._y_columns)

    def _to_context_df(self, y, X=None):
        """Convert sktime y/X to TabPFN-TS context DataFrame."""
        y_df = self._coerce_to_frame(y, name="target")
        timestamps = self._timestamps_from_index(y_df.index)
        X_df = self._coerce_optional_X(X)

        frames = []
        for column in y_df.columns:
            frame = pd.DataFrame(
                {
                    "item_id": str(column),
                    "timestamp": timestamps,
                    "target": y_df[column].to_numpy(),
                }
            )
            if X_df is not None:
                frame = pd.concat(
                    [frame.reset_index(drop=True), X_df.reset_index(drop=True)],
                    axis=1,
                )
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)

    def _to_future_df(self, prediction_length, X=None):
        """Convert future X, if any, to TabPFN-TS future DataFrame."""
        timestamps = self._future_timestamps(prediction_length)
        X_df = None
        if X is not None:
            X_df = self._coerce_optional_X(X)
            X_df = X_df.iloc[:prediction_length].reset_index(drop=True)

        frames = []
        for column in self._y_columns:
            frame = pd.DataFrame({"item_id": str(column), "timestamp": timestamps})
            if X_df is not None:
                frame = pd.concat([frame, X_df], axis=1)
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)

    @staticmethod
    def _coerce_to_frame(obj, name):
        """Coerce pandas Series or DataFrame to DataFrame."""
        if isinstance(obj, pd.Series):
            return obj.to_frame(name=obj.name if obj.name is not None else name)
        return obj

    def _coerce_optional_X(self, X):
        """Coerce optional exogenous data to a DataFrame."""
        if X is None:
            return None
        X_df = self._coerce_to_frame(X, name="X")
        X_df = X_df.copy()
        X_df.columns = [str(col) for col in X_df.columns]
        return X_df

    def _timestamps_from_index(self, index):
        """Return TabPFN-TS compatible timestamps for an sktime index."""
        self._timestamp_freq = None
        self._uses_artificial_timestamps = False

        if isinstance(index, pd.PeriodIndex):
            timestamps = index.to_timestamp()
        elif isinstance(index, pd.DatetimeIndex):
            timestamps = index
        else:
            self._uses_artificial_timestamps = True
            timestamps = pd.date_range("2000-01-01", periods=len(index), freq="D")

        freq = getattr(timestamps, "freq", None)
        if freq is None:
            freq = pd.infer_freq(timestamps)
        self._timestamp_freq = freq if freq is not None else "D"
        self._last_timestamp = timestamps[-1]
        return timestamps

    def _future_timestamps(self, prediction_length):
        """Return future timestamps for TabPFN-TS prediction."""
        offset = pd.tseries.frequencies.to_offset(self._timestamp_freq)
        start = self._last_timestamp + offset
        return pd.date_range(start=start, periods=prediction_length, freq=offset)

    def _format_predictions(self, pred, fh_rel):
        """Format TabPFN-TS predictions as a numpy array."""
        pred = pred.reset_index()
        value_col = "target"
        positions = fh_rel.to_numpy() - 1

        values = []
        for column in self._y_columns:
            item_pred = pred[pred["item_id"].astype(str) == str(column)]
            values.append(item_pred[value_col].to_numpy()[positions])
        return np.asarray(values).T

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        return {"ignore_deps": True}


@_multiton
class _CachedTabPFNTSPipeline:
    """Cached TabPFN-TS pipeline, one per unique model configuration."""

    def __init__(self, key, pipeline_cls, pipeline_kwargs):
        self.key = key
        self.pipeline = pipeline_cls(**pipeline_kwargs)

    def load(self):
        """Return cached TabPFN-TS pipeline."""
        return self.pipeline
