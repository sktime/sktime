# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements TabPFN forecaster."""

__author__ = ["priyanshuharshbodhi1"]
__all__ = ["TabPFNForecaster"]

import pandas as pd

from sktime.forecasting.base import BaseForecaster


class TabPFNForecaster(BaseForecaster):
    """Interface to TabPFN-TS for zero-shot time series forecasting.

    Wraps ``tabpfn_time_series.TabPFNTSPipeline`` as an sktime forecaster.

    Parameters
    ----------
    max_context_length : int, default=4096
        Maximum number of context observations passed to the pipeline.
    tabpfn_mode : {"local", "client"}, default="local"
        ``"local"`` runs inference locally, ``"client"`` uses the
        hosted TabPFN cloud API.
    tabpfn_output_selection : {"mean", "median", "mode"}, default="median"
        Aggregation for the TabPFN ensemble output.

    References
    ----------
    .. [1] https://github.com/PriorLabs/tabpfn-time-series

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.tabpfn import TabPFNForecaster
    >>> y = load_airline()
    >>> forecaster = TabPFNForecaster(tabpfn_mode="local")  # doctest: +SKIP
    >>> forecaster.fit(y)  # doctest: +SKIP
    TabPFNForecaster(...)
    >>> y_pred = forecaster.predict(fh=[1, 2, 3])  # doctest: +SKIP
    """

    _tags = {
        "authors": ["priyanshuharshbodhi1"],
        "maintainers": ["priyanshuharshbodhi1"],
        "python_dependencies": ["tabpfn-time-series"],
        "y_inner_mtype": "pd.Series",
        "X_inner_mtype": "pd.DataFrame",
        "scitype:y": "univariate",
        "capability:exogenous": True,
        "requires-fh-in-fit": False,
        "X-y-must-have-same-index": True,
        "enforce_index_type": None,
        "capability:missing_values": True,
        "capability:insample": False,
    }

    def __init__(
        self,
        max_context_length=4096,
        tabpfn_mode="local",
        tabpfn_output_selection="median",
    ):
        self.max_context_length = max_context_length
        self.tabpfn_mode = tabpfn_mode
        self.tabpfn_output_selection = tabpfn_output_selection

        super().__init__()

    @staticmethod
    def _is_integer_index(index):
        return isinstance(index, pd.Index) and pd.api.types.is_integer_dtype(index)

    @classmethod
    def _require_supported_index(cls, index, var_name):
        is_ok = isinstance(
            index, (pd.DatetimeIndex, pd.PeriodIndex, pd.RangeIndex)
        ) or cls._is_integer_index(index)

        if not is_ok:
            raise NotImplementedError(
                f"{var_name} must have DatetimeIndex, PeriodIndex, RangeIndex, "
                f"or integer Index, but found {type(index)}."
            )

    @classmethod
    def _to_timestamp_index(cls, index, var_name):
        """Convert a supported index to DatetimeIndex for TabPFN-TS."""
        cls._require_supported_index(index, var_name=var_name)

        if isinstance(index, pd.PeriodIndex):
            return index.to_timestamp()
        if isinstance(index, pd.DatetimeIndex):
            return index

        # RangeIndex / plain integer -> synthetic daily timestamps
        return pd.DatetimeIndex(
            pd.to_datetime(index.to_numpy(), unit="D", origin="unix"),
            name=index.name,
        )

    def _make_context_df(self):
        """Build the flat context DataFrame expected by predict_df."""
        context_df = self._y_context.rename("target").to_frame()
        if self._X_context is not None:
            context_df = context_df.join(self._X_context, how="left")

        context_df.index = self._to_timestamp_index(context_df.index, "y")
        context_df.index.name = "timestamp"
        context_df = context_df.reset_index()
        context_df.insert(0, "item_id", self._item_id)

        return context_df

    def _make_future_df(self, fh_abs, X):
        """Build the flat future DataFrame expected by predict_df."""
        fh_datetime = self._to_timestamp_index(fh_abs, "fh")
        future_df = pd.DataFrame({"item_id": self._item_id, "timestamp": fh_datetime})

        if X is not None:
            self._require_supported_index(X.index, "X")
            X_future = X.copy()
            X_future.index = self._to_timestamp_index(X_future.index, "X")
            X_future = X_future.reindex(fh_datetime)
            future_df = pd.concat(
                [future_df.reset_index(drop=True), X_future.reset_index(drop=True)],
                axis=1,
            )

        return future_df, fh_datetime

    def _fit(self, y, X=None, fh=None):
        """Fit forecaster to training data."""
        from tabpfn_time_series import (
            TABPFN_DEFAULT_CONFIG,
            TabPFNMode,
            TabPFNTSPipeline,
        )

        self._require_supported_index(y.index, "y")
        if X is not None:
            self._require_supported_index(X.index, "X")

        mode_map = {
            "local": TabPFNMode.LOCAL,
            "client": TabPFNMode.CLIENT,
        }

        self._pipeline = TabPFNTSPipeline(
            max_context_length=self.max_context_length,
            tabpfn_mode=mode_map[self.tabpfn_mode],
            tabpfn_output_selection=self.tabpfn_output_selection,
            tabpfn_model_config=TABPFN_DEFAULT_CONFIG,
        )

        self._y_context = y.copy()
        self._X_context = X.copy() if X is not None else None
        self._X_columns = [] if X is None else list(X.columns)
        self._item_id = "__sktime_series_0"

        return self

    def _predict(self, fh, X=None):
        """Forecast time series at future horizon."""
        fh_abs = fh.to_absolute_index(self.cutoff)
        self._require_supported_index(fh_abs, "fh")

        context_df = self._make_context_df()
        future_df, fh_datetime = self._make_future_df(fh_abs, X)

        preds_df = self._pipeline.predict_df(
            context_df=context_df,
            future_df=future_df,
        )

        # predict_df returns MultiIndex (item_id, timestamp)
        if isinstance(preds_df.index, pd.MultiIndex):
            if "item_id" in preds_df.index.names:
                preds_df = preds_df.xs(self._item_id, level="item_id", drop_level=True)
            else:
                preds_df = preds_df.droplevel(0)

        preds_df = preds_df.reindex(fh_datetime)
        preds_df.index = fh_abs

        y_pred = preds_df["target"].copy()
        y_pred.name = self._y_context.name

        return y_pred

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        return [{"max_context_length": 512, "tabpfn_mode": "local"}]
