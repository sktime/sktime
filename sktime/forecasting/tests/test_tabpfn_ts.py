"""Tests for TabPFNTSForecaster."""

import numpy as np
import pandas as pd

from sktime.forecasting.tabpfn_ts import TabPFNTSForecaster


class _DummyTabPFNTSPipeline:
    """Small stand-in for tabpfn_time_series.TabPFNTSPipeline."""

    def __init__(
        self,
        max_context_length=4096,
        temporal_features=None,
        tabpfn_mode="client",
        tabpfn_output_selection="median",
        tabpfn_model_config=None,
    ):
        self.max_context_length = max_context_length
        self.temporal_features = temporal_features
        self.tabpfn_mode = tabpfn_mode
        self.tabpfn_output_selection = tabpfn_output_selection
        self.tabpfn_model_config = tabpfn_model_config

    def predict_df(
        self, context_df, future_df=None, prediction_length=None, quantiles=None
    ):
        """Predict simple constant forecasts."""
        if prediction_length is not None:
            raise ValueError("test expects future_df path")

        values = []
        means = context_df.groupby("item_id")["target"].mean()
        for _, row in future_df.iterrows():
            values.append(means.loc[row["item_id"]])

        pred = future_df[["item_id", "timestamp"]].copy()
        pred["target"] = values
        return pred.set_index(["item_id", "timestamp"])


def test_tabpfn_ts_forecaster_uses_pipeline_predict_df(monkeypatch):
    """TabPFNTSForecaster should adapt y and delegate to TabPFN-TS pipeline."""

    monkeypatch.setattr(
        TabPFNTSForecaster,
        "_get_pipeline_class",
        lambda self: _DummyTabPFNTSPipeline,
    )
    monkeypatch.setattr(TabPFNTSForecaster, "_get_tabpfn_mode", lambda self: "client")

    y = pd.Series(
        np.arange(10, dtype=float),
        index=pd.date_range("2024-01-01", periods=10, freq="D"),
        name="series",
    )
    forecaster = TabPFNTSForecaster(max_context_length=5, ignore_deps=True)

    forecaster.fit(y)
    y_pred = forecaster.predict(fh=[1, 3])

    assert len(y_pred) == 2
    assert y_pred.name == "series"
    assert forecaster.pipeline_.max_context_length == 5


def test_tabpfn_ts_forecaster_formats_multivariate_predictions(monkeypatch):
    """TabPFNTSForecaster should map multivariate y columns to item ids."""

    monkeypatch.setattr(
        TabPFNTSForecaster,
        "_get_pipeline_class",
        lambda self: _DummyTabPFNTSPipeline,
    )
    monkeypatch.setattr(TabPFNTSForecaster, "_get_tabpfn_mode", lambda self: "client")

    y = pd.DataFrame(
        {"a": np.arange(10, dtype=float), "b": np.arange(10, 20, dtype=float)},
        index=pd.date_range("2024-01-01", periods=10, freq="D"),
    )
    forecaster = TabPFNTSForecaster(ignore_deps=True)

    forecaster.fit(y)
    y_pred = forecaster.predict(fh=[1, 2])

    assert list(y_pred.columns) == ["a", "b"]
    assert y_pred.shape == (2, 2)
