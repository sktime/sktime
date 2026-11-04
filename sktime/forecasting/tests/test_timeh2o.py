"""Offline contract tests for the TimeH2O adapter; no gated weights required."""

import os
import sys
import types

import numpy as np
import pandas as pd
import pytest

from sktime.forecasting.timeh2o import TimeH2OForecaster


class _OfflineTimeH2OForecaster(TimeH2OForecaster):
    __doc__ = TimeH2OForecaster.__doc__

    # A separate class keeps cached production tags intact during test discovery.
    _tags = {"python_dependencies": [], "python_version": None}


class _Model:
    """Deterministic substitute for the gated, optional inference boundary."""

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        return cls()

    def forecast(self, context, horizon, future_covariates=None, **kwargs):
        # Distinguish targets, horizon positions, and requested quantiles.
        point = context[:, -1, None] + np.arange(1, horizon + 1)
        if future_covariates is not None:
            assert len(future_covariates[0]) == context.shape[1] + horizon
            point = point + np.asarray(future_covariates[0][-horizon:]).astype(float)
        alpha = kwargs.get("quantile_levels", [0.1, 0.5, 0.9])
        return types.SimpleNamespace(
            point=point,
            quantiles=point[..., None] + np.asarray(alpha),
        )


@pytest.fixture
def forecaster_class(monkeypatch):
    """Replace only the optional upstream model and its network loading."""
    monkeypatch.setitem(sys.modules, "h2ofm", types.SimpleNamespace(TimeH2O=_Model))
    return _OfflineTimeH2OForecaster


@pytest.mark.parametrize(
    "index",
    [
        pd.RangeIndex(4),
        pd.period_range("2024-01", periods=4, freq="M"),
        pd.date_range("2024-01-01", periods=4),
    ],
)
def test_point_sparse_and_absolute_horizon(forecaster_class, index):
    """Select horizon positions correctly and retain target/index metadata."""
    from sktime.forecasting.base import ForecastingHorizon

    y = pd.DataFrame(
        {"a": [1.0, 2.0, 3.0, 4.0], "b": [10.0, 20.0, 30.0, 40.0]},
        index=index.rename("time"),
    )
    f = forecaster_class().fit(y)
    pred = f.predict([1, 3])
    np.testing.assert_array_equal(pred, [[5, 41], [7, 43]])
    assert list(pred.columns) == ["a", "b"]
    assert pred.index.name == "time"
    pd.testing.assert_frame_equal(
        pred, f.predict(ForecastingHorizon(pred.index, is_relative=False))
    )


def test_quantiles_and_intervals(forecaster_class):
    """Quantile values and columns follow sktime's variable-major convention."""
    f = forecaster_class().fit(pd.DataFrame({"a": [1.0, 4.0], "b": [10.0, 40.0]}))
    q = f.predict_quantiles([1, 3], alpha=[0.9, 0.05, 0.5])
    assert list(q.columns) == [(v, a) for v in ["a", "b"] for a in [0.9, 0.05, 0.5]]
    np.testing.assert_allclose(
        q, [[5.9, 5.05, 5.5, 41.9, 41.05, 41.5], [7.9, 7.05, 7.5, 43.9, 43.05, 43.5]]
    )
    interval = f.predict_interval([1], coverage=0.8)
    np.testing.assert_allclose(interval, [[5.1, 5.9, 41.1, 41.9]])


def test_covariates_alignment_and_mixed_dtypes(forecaster_class):
    """Preserve numeric rows when other covariates are categorical."""
    y = pd.Series([1.0, 2.0, 3.0, 4.0], name="y")
    X = pd.DataFrame({"price": [1.0, 2.0, 3.0, 4.0], "holiday": ["a", "b", "a", "b"]})
    f = forecaster_class().fit(y, X=X)
    future = pd.DataFrame(
        {"holiday": ["c", "b", "c"], "price": [10.0, 20.0, 30.0]}, index=[4, 5, 6]
    )
    np.testing.assert_allclose(f.predict([1, 3], X=future), [15, 37])


@pytest.mark.parametrize(
    "future,match",
    [
        (None, "X"),
        (pd.DataFrame({"x": [1.0, 2.0]}, index=[4, 6]), "every|all|missing"),
        (pd.DataFrame({"other": [1.0, 2.0, 3.0]}, index=[4, 5, 6]), "columns"),
    ],
)
def test_invalid_future_covariates(forecaster_class, future, match):
    """Reject missing intermediate rows and incompatible covariate columns."""
    f = forecaster_class().fit(
        pd.Series([1.0, 2.0, 3.0, 4.0]), X=pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    )
    with pytest.raises(ValueError, match=match):
        f.predict([1, 3], X=future)


def test_update_uses_latest_history(forecaster_class):
    """Zero-shot updates change the context even without parameter fitting."""
    f = forecaster_class().fit(pd.Series([1.0, 2.0, 3.0, 4.0]))
    f.update(pd.Series([100.0], index=[4]), update_params=False)
    np.testing.assert_array_equal(f.predict([1, 2]), [101, 102])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model_kwargs": {"device": "cpu"}},
        {"forecast_kwargs": {"horizon": 2}},
        {"forecast_kwargs": {"quantile_levels": [0.5]}},
    ],
)
def test_reserved_options_rejected(forecaster_class, kwargs):
    """Options cannot silently override adapter-controlled arguments."""
    with pytest.raises(ValueError, match="reserved|Reserved"):
        forecaster_class(**kwargs).fit(pd.Series([1.0, 2.0, 3.0]))


def test_offline_estimator_contract(forecaster_class):
    """Run sktime interface checks without accepting or downloading weights."""
    from sktime.utils.estimator_checks import check_estimator

    results = check_estimator(forecaster_class, verbose=0)
    failures = {name: result for name, result in results.items() if result != "PASSED"}
    assert results
    assert not failures, "\n".join(
        f"{name}: {error!r}" for name, error in failures.items()
    )


def test_nullable_numeric_inputs(forecaster_class, monkeypatch):
    """Nullable numeric arrays must not become upstream categorical labels."""

    class NumericModel(_Model):
        def forecast(self, context, *args, future_covariates=None, **kwargs):
            assert context.dtype.kind == "f"
            assert np.isnan(context[0, 1])
            assert pd.api.types.is_numeric_dtype(future_covariates[0])
            assert pd.isna(future_covariates[0].iloc[1])
            return super().forecast(
                context, *args, future_covariates=future_covariates, **kwargs
            )

    monkeypatch.setattr(sys.modules["h2ofm"], "TimeH2O", NumericModel)
    y = pd.DataFrame(
        {
            "a": pd.Series([1, None, 3, 4], dtype="Int64"),
            "b": pd.Series([10, None, 30, 40], dtype="Float64"),
        }
    )
    X = pd.DataFrame({"x": pd.Series([1, None, 3, 4], dtype="Int64")})
    f = forecaster_class().fit(y, X=X)
    future = pd.DataFrame({"x": pd.Series([10], index=[4], dtype="Int64")})
    np.testing.assert_array_equal(f.predict([1], X=future), [[15, 51]])


def test_numeric_categorical_covariates(forecaster_class, monkeypatch):
    """Numeric category labels must retain their categorical meaning upstream."""

    class CategoricalModel(_Model):
        def forecast(self, context, horizon, future_covariates=None, **kwargs):
            assert isinstance(future_covariates[0].dtype, pd.CategoricalDtype)
            return super().forecast(context, horizon, **kwargs)

    monkeypatch.setattr(sys.modules["h2ofm"], "TimeH2O", CategoricalModel)
    dtype = pd.CategoricalDtype([1, 2, 3])
    X = pd.DataFrame({"label": pd.Series([1, 2, 1, 2], dtype=dtype)})
    future = pd.DataFrame({"label": pd.Series([3], dtype=dtype, index=[4])})
    f = forecaster_class().fit(pd.Series([1.0, 2.0, 3.0, 4.0]), X=X)
    np.testing.assert_array_equal(f.predict([1], X=future), [5])


def test_missing_values_and_input_ownership(forecaster_class):
    """Passing missing observations must not modify caller-owned data."""
    y = pd.Series([1.0, np.nan, 3.0, 4.0], name="target")
    before = y.copy()
    f = forecaster_class().fit(y)
    np.testing.assert_array_equal(f.predict([1, 3]), [5, 7])
    pd.testing.assert_series_equal(y, before)


@pytest.mark.parametrize("update_params", [True, False])
def test_covariates_after_update(forecaster_class, update_params):
    """Updates append matching historical covariates and keep their alignment."""
    f = forecaster_class().fit(
        pd.Series([1.0, 2.0, 3.0, 4.0]), X=pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    )
    f.update(
        pd.Series([100.0], index=[4]),
        X=pd.DataFrame({"x": [5.0]}, index=[4]),
        update_params=update_params,
    )
    np.testing.assert_array_equal(
        f.predict([1], X=pd.DataFrame({"x": [10.0]}, index=[5])), [111]
    )


def test_options_forwarding_and_ownership(forecaster_class, monkeypatch):
    """Use load/inference options without adding keys to the caller's dicts."""

    class ConfiguredModel(_Model):
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            assert path == "local-checkpoint"
            assert kwargs == {
                "device": "cpu",
                "batch_size": 8,
                "local_files_only": True,
            }
            return cls()

        def forecast(self, *args, **kwargs):
            assert kwargs["context_length"] == 16
            return super().forecast(*args, **kwargs)

    monkeypatch.setattr(sys.modules["h2ofm"], "TimeH2O", ConfiguredModel)
    model_kwargs = {"local_files_only": True}
    forecast_kwargs = {"context_length": 16}
    f = forecaster_class(
        model_path="local-checkpoint",
        device="cpu",
        batch_size=8,
        model_kwargs=model_kwargs,
        forecast_kwargs=forecast_kwargs,
    ).fit(pd.Series([1.0, 4.0]))
    np.testing.assert_array_equal(f.predict([1]), [5])
    f.predict_quantiles([1], alpha=[0.1, 0.9])
    assert model_kwargs == {"local_files_only": True}
    assert forecast_kwargs == {"context_length": 16}


@pytest.mark.skipif(
    os.environ.get("SKTIME_TIMEH2O_INTEGRATION") != "1",
    reason="Opt in after accepting HF model terms and installing h2ofm",
)
def test_timeh2o_real_checkpoint():
    """Exercise the actual gated checkpoint only when explicitly requested."""
    pytest.importorskip("h2ofm")

    y = pd.Series(np.sin(np.arange(64) / 10), name="target")
    f = TimeH2OForecaster(device="cpu").fit(y)
    pred = f.predict([1, 3])
    quantiles = f.predict_quantiles([1, 3], alpha=[0.1, 0.5, 0.9])
    assert np.isfinite(pred).all()
    np.testing.assert_allclose(pred, quantiles[("target", 0.5)])
