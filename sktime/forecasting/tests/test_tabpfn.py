# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for TabPFNForecaster."""

import sys
import types
import uuid

import numpy as np
import pandas as pd
import pytest

from sktime.forecasting.tabpfn import TabPFNForecaster


def _install_fake_tabpfn_module(monkeypatch):
    """Inject a minimal mock of tabpfn_time_series and return the pipeline class."""
    monkeypatch.setattr(
        "sktime.forecasting.base._base._check_estimator_deps",
        lambda *args, **kwargs: True,
    )

    fake_module = types.ModuleType("tabpfn_time_series")

    class TabPFNMode:
        LOCAL = "local"
        CLIENT = "client"

    class TabPFNTSPipeline:
        init_calls = 0

        def __init__(self, **kwargs):
            type(self).init_calls += 1
            self.kwargs = kwargs

        def predict_df(self, context_df, future_df, quantiles):
            assert "item_id" in context_df.columns
            assert "timestamp" in context_df.columns
            assert "target" in context_df.columns
            assert "item_id" in future_df.columns
            assert "timestamp" in future_df.columns

            idx = pd.MultiIndex.from_arrays(
                [
                    future_df["item_id"].to_numpy(),
                    pd.to_datetime(future_df["timestamp"]).to_numpy(),
                ],
                names=["item_id", "timestamp"],
            )

            values = np.arange(len(future_df), dtype=float)
            pred = pd.DataFrame({"target": values}, index=idx)

            for q in quantiles:
                pred[float(q)] = values + float(q)

            return pred

    fake_module.TabPFNMode = TabPFNMode
    fake_module.TabPFNTSPipeline = TabPFNTSPipeline
    fake_module.TABPFN_DEFAULT_CONFIG = {"model_path": "fake-model.ckpt"}

    monkeypatch.setitem(sys.modules, "tabpfn_time_series", fake_module)

    return TabPFNTSPipeline


def test_tabpfn_predict_and_quantiles_period_index(monkeypatch):
    """Fit/predict/predict_quantiles with PeriodIndex and exogenous data."""
    _install_fake_tabpfn_module(monkeypatch)

    y_index = pd.period_range("2020-01", periods=8, freq="M")
    y = pd.Series(np.arange(8.0), index=y_index, name="y")
    X_train = pd.DataFrame({"temp": np.linspace(0, 1, 8)}, index=y_index)

    fh = [1, 2, 3]
    X_future_index = pd.period_range(y_index[-1] + 1, periods=3, freq="M")
    X_future = pd.DataFrame({"temp": [1.1, 1.2, 1.3]}, index=X_future_index)

    forecaster = TabPFNForecaster(max_context_length=64, tabpfn_mode="local")
    forecaster.fit(y, X=X_train)

    y_pred = forecaster.predict(fh=fh, X=X_future)

    assert isinstance(y_pred, pd.Series)
    assert isinstance(y_pred.index, pd.PeriodIndex)
    assert y_pred.index.equals(X_future_index)
    assert y_pred.name == "y"

    q_pred = forecaster.predict_quantiles(fh=fh, X=X_future, alpha=[0.1, 0.9])
    assert ("y", 0.1) in q_pred.columns
    assert ("y", 0.9) in q_pred.columns
    assert q_pred.index.equals(X_future_index)


def test_tabpfn_multiton_reuses_pipeline(monkeypatch):
    """Identical config creates only one underlying TabPFNTSPipeline."""
    pipeline_cls = _install_fake_tabpfn_module(monkeypatch)
    model_id = f"fake-{uuid.uuid4().hex}.ckpt"

    y_index = pd.date_range("2023-01-01", periods=10, freq="D")
    y = pd.Series(np.arange(10.0), index=y_index)

    f1 = TabPFNForecaster(
        max_context_length=32,
        tabpfn_mode="local",
        tabpfn_model_config={"model_path": model_id},
    )
    f2 = TabPFNForecaster(
        max_context_length=32,
        tabpfn_mode="local",
        tabpfn_model_config={"model_path": model_id},
    )

    before = pipeline_cls.init_calls
    f1.fit(y)
    f2.fit(y)
    after = pipeline_cls.init_calls

    assert after - before == 1, "identical configs should share one pipeline"


def test_tabpfn_future_covariates_required_when_used_in_fit(monkeypatch):
    """Predict without X raises when fit received exogenous data."""
    _install_fake_tabpfn_module(monkeypatch)

    y_index = pd.date_range("2023-01-01", periods=6, freq="D")
    y = pd.Series(np.arange(6.0), index=y_index)
    X_train = pd.DataFrame({"x1": np.arange(6.0)}, index=y_index)

    forecaster = TabPFNForecaster(
        max_context_length=16,
        tabpfn_mode="local",
        ignore_future_covariates_if_missing=False,
    )
    forecaster.fit(y, X=X_train)

    with pytest.raises(ValueError, match="Future covariates are required"):
        forecaster.predict(fh=[1, 2], X=None)


def test_tabpfn_supports_range_index(monkeypatch):
    """RangeIndex input is handled and the output index is correct."""
    _install_fake_tabpfn_module(monkeypatch)

    y = pd.Series(np.arange(8.0), index=pd.RangeIndex(8))
    forecaster = TabPFNForecaster(max_context_length=32, tabpfn_mode="local")
    forecaster.fit(y)

    y_pred = forecaster.predict(fh=[1, 2, 3])

    assert list(y_pred.index) == [8, 9, 10]


def test_tabpfn_quantile_columns_for_unnamed_series(monkeypatch):
    """Quantile columns use 0 as variable name for unnamed series."""
    _install_fake_tabpfn_module(monkeypatch)

    y = pd.Series(np.arange(6.0), index=pd.RangeIndex(6))
    forecaster = TabPFNForecaster(max_context_length=32, tabpfn_mode="local")
    forecaster.fit(y)

    pred_q = forecaster.predict_quantiles(fh=[1, 2], alpha=[0.1])
    assert (0, 0.1) in pred_q.columns


def test_tabpfn_datetime_index_no_exog(monkeypatch):
    """Basic fit/predict with DatetimeIndex and no exogenous variables."""
    _install_fake_tabpfn_module(monkeypatch)

    y_index = pd.date_range("2024-01-01", periods=12, freq="D")
    y = pd.Series(np.random.default_rng(42).random(12), index=y_index, name="val")
    forecaster = TabPFNForecaster(max_context_length=64, tabpfn_mode="local")
    forecaster.fit(y)

    y_pred = forecaster.predict(fh=[1, 2, 3, 4])

    assert isinstance(y_pred, pd.Series)
    assert isinstance(y_pred.index, pd.DatetimeIndex)
    assert len(y_pred) == 4
    assert y_pred.name == "val"


def test_tabpfn_ignore_missing_covariates_flag(monkeypatch):
    """ignore_future_covariates_if_missing=True suppresses the error."""
    _install_fake_tabpfn_module(monkeypatch)

    y_index = pd.date_range("2023-06-01", periods=5, freq="D")
    y = pd.Series(np.arange(5.0), index=y_index)
    X_train = pd.DataFrame({"feat": np.ones(5)}, index=y_index)

    forecaster = TabPFNForecaster(
        max_context_length=16,
        tabpfn_mode="local",
        ignore_future_covariates_if_missing=True,
    )
    forecaster.fit(y, X=X_train)

    # should not raise even though X is missing at predict time
    y_pred = forecaster.predict(fh=[1, 2])
    assert len(y_pred) == 2
