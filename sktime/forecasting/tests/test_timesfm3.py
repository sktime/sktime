"""Tests for the TimesFM 3 forecaster."""

from __future__ import annotations

import pickle
from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from sktime.datasets import load_airline
from sktime.forecasting.timesfm3 import TimesFM3Forecaster
from sktime.tests.test_switch import run_test_for_class

_DEFAULT_QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


@dataclass
class _FakeForecastOutput:
    forecast: np.ndarray
    quantiles: np.ndarray | None = None


class _FakeUpstreamForecaster:
    """Minimal upstream stand-in that records ``predict`` calls."""

    def __init__(self, global_context=128):
        self.calls = []
        self.global_context = global_context
        self.config = SimpleNamespace(
            quantiles=_DEFAULT_QUANTILES,
            median_quantile_index=4,
        )
        self.model = SimpleNamespace(
            transformer_config=SimpleNamespace(
                transformer=SimpleNamespace(max_variates=32)
            )
        )

    def predict(
        self,
        context,
        horizon,
        past_only_covariates=None,
        past_future_covariates=None,
        ts_id=None,
        return_quantiles=False,
        use_symmetric_averaging=False,
        make_positive=False,
        sort_quantiles=True,
        use_znorm=False,
        padding_mode="none",
    ):
        self.calls.append(
            {
                "context": np.array(context, copy=True),
                "horizon": horizon,
                "past_only_covariates": None
                if past_only_covariates is None
                else np.array(past_only_covariates, copy=True),
                "past_future_covariates": None
                if past_future_covariates is None
                else np.array(past_future_covariates, copy=True),
                "return_quantiles": return_quantiles,
                "use_symmetric_averaging": use_symmetric_averaging,
                "make_positive": make_positive,
                "sort_quantiles": sort_quantiles,
                "use_znorm": use_znorm,
                "padding_mode": padding_mode,
            }
        )

        context_arr = np.atleast_2d(np.asarray(context, dtype=np.float32))
        n_targets = context_arr.shape[0]
        forecast = context_arr.mean(axis=1, keepdims=True) + np.arange(
            1, horizon + 1, dtype=np.float32
        )
        if n_targets == 1:
            point = forecast[0]
        else:
            point = forecast

        quantiles = None
        if return_quantiles:
            base = np.atleast_2d(point)
            quantiles = np.stack(
                [base + (q - 0.5) for q in _DEFAULT_QUANTILES], axis=-1
            )
            if n_targets == 1:
                quantiles = quantiles[0]

        return _FakeForecastOutput(forecast=point, quantiles=quantiles)


def _make_forecaster(**kwargs):
    params = {
        "license_accepted": True,
        "ignore_deps": True,
        **kwargs,
    }
    return TimesFM3Forecaster(**params)


@contextmanager
def _clear_class_dependencies():
    """Temporarily clear the class-level ``python_dependencies`` tag."""
    tags = dict(TimesFM3Forecaster._tags)
    tags["python_dependencies"] = []
    with patch.object(TimesFM3Forecaster, "_tags", tags):
        yield


@contextmanager
def _patch_upstream(fake):
    with (
        _clear_class_dependencies(),
        patch.object(
            TimesFM3Forecaster,
            "_load_model",
            autospec=True,
            side_effect=lambda self: setattr(self, "forecaster_", fake) or fake,
        ),
        patch(
            "sktime.forecasting.base._base._check_estimator_deps",
            return_value=True,
        ),
    ):
        yield


def test_patch_upstream_clears_class_dependency_tag():
    """Mocked tests must not require the class-level timesfm dependency."""
    original = TimesFM3Forecaster.get_class_tag("python_dependencies")
    with _patch_upstream(_FakeUpstreamForecaster()):
        deps = TimesFM3Forecaster.get_class_tag("python_dependencies")
        assert deps in ([], None), (
            "mocked TimesFM3 tests must clear the class-level "
            f"python_dependencies tag, but get_class_tag returns {deps!r}"
        )
    assert TimesFM3Forecaster.get_class_tag("python_dependencies") == original


def test_license_rejected_before_model_load():
    """Fit raises when license has not been accepted."""
    forecaster = TimesFM3Forecaster(ignore_deps=True)
    y = pd.Series([1.0, 2.0, 3.0, 4.0])

    with (
        _clear_class_dependencies(),
        patch(
            "sktime.forecasting.base._base._check_estimator_deps",
            return_value=True,
        ),
    ):
        with pytest.raises(ValueError, match="license_accepted"):
            forecaster.fit(y)


def test_univariate_series_point_forecast_index():
    """Univariate Series predictions use absolute future index."""
    fake = _FakeUpstreamForecaster()
    index = pd.date_range("2020-01-01", periods=6, freq="D")
    y = pd.Series(np.arange(6, dtype=float), index=index, name="y")
    fh = [1, 3]

    with _patch_upstream(fake):
        forecaster = _make_forecaster()
        y_pred = forecaster.fit(y).predict(fh=fh)

    assert list(y_pred.index) == [
        index[-1] + pd.Timedelta(days=1),
        index[-1] + pd.Timedelta(days=3),
    ]


def test_multivariate_dataframe_point_forecast():
    """Multivariate DataFrame predictions retain column names."""
    fake = _FakeUpstreamForecaster()
    y = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [3.0, 2.0, 1.0]})

    with _patch_upstream(fake):
        forecaster = _make_forecaster()
        y_pred = forecaster.fit(y).predict(fh=[1, 2])

    assert list(y_pred.columns) == ["a", "b"]
    assert fake.calls[0]["context"].shape == (2, 3)


def test_predict_quantiles_available_levels():
    """Quantile output uses sktime MultiIndex columns for supported levels."""
    fake = _FakeUpstreamForecaster()
    y = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [3.0, 2.0, 1.0]})

    with _patch_upstream(fake):
        forecaster = _make_forecaster()
        y_quant = forecaster.fit(y).predict_quantiles(fh=[1, 2], alpha=[0.1, 0.9])

    assert isinstance(y_quant.columns, pd.MultiIndex)
    assert set(y_quant.columns.get_level_values(0)) == {"a", "b"}
    assert set(y_quant.columns.get_level_values(1)) == {0.1, 0.9}


def test_predict_quantiles_interpolates_intermediate_and_clamps_tails():
    """Off-grid quantile levels interpolate; outer levels clamp to the native grid."""
    fake = _FakeUpstreamForecaster()
    y = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [3.0, 2.0, 1.0]})

    with _patch_upstream(fake):
        forecaster = _make_forecaster()
        fitted = forecaster.fit(y)
        native = fitted.predict_quantiles(fh=[1, 2], alpha=[0.1, 0.2, 0.3, 0.9])
        requested = fitted.predict_quantiles(fh=[1, 2], alpha=[0.05, 0.25, 0.9])

    np.testing.assert_allclose(
        requested[("a", 0.05)].to_numpy(),
        native[("a", 0.1)].to_numpy(),
    )
    np.testing.assert_allclose(
        requested[("b", 0.05)].to_numpy(),
        native[("b", 0.1)].to_numpy(),
    )
    np.testing.assert_allclose(
        requested[("a", 0.25)].to_numpy(),
        0.5 * (native[("a", 0.2)].to_numpy() + native[("a", 0.3)].to_numpy()),
    )
    np.testing.assert_allclose(
        requested[("a", 0.9)].to_numpy(),
        native[("a", 0.9)].to_numpy(),
    )


def test_empty_past_covariates_fits_without_x():
    """An empty ``past_covariates`` list is equivalent to no covariates."""
    fake = _FakeUpstreamForecaster()
    y = pd.Series([1.0, 2.0, 3.0, 4.0])

    with _patch_upstream(fake):
        forecaster = _make_forecaster(past_covariates=[])
        y_pred = forecaster.fit(y).predict(fh=[1])

    assert len(y_pred) == 1
    assert fake.calls[0]["past_only_covariates"] is None
    assert fake.calls[0]["past_future_covariates"] is None


def test_nonempty_past_covariates_requires_x():
    """A non-empty ``past_covariates`` list still requires fit-time ``X``."""
    fake = _FakeUpstreamForecaster()
    y = pd.Series([1.0, 2.0, 3.0, 4.0])

    with _patch_upstream(fake):
        forecaster = _make_forecaster(past_covariates=["po"])
        with pytest.raises(ValueError, match="past_covariates"):
            forecaster.fit(y)


def test_upstream_arrays_without_exogenous():
    """Adapter forwards target context without covariate arrays."""
    fake = _FakeUpstreamForecaster()
    y = pd.Series([1.0, 2.0, 3.0, 4.0])

    with _patch_upstream(fake):
        forecaster = _make_forecaster()
        forecaster.fit(y).predict(fh=[1, 2])

    call = fake.calls[0]
    np.testing.assert_allclose(call["context"], [[1.0, 2.0, 3.0, 4.0]])
    assert call["past_only_covariates"] is None
    assert call["past_future_covariates"] is None


def test_upstream_arrays_past_only_covariates():
    """Past-only covariates are forwarded with context length only."""
    fake = _FakeUpstreamForecaster()
    y = pd.Series([1.0, 2.0, 3.0, 4.0])
    X = pd.DataFrame({"po": [10.0, 20.0, 30.0, 40.0]})

    with _patch_upstream(fake):
        forecaster = _make_forecaster(past_covariates=["po"])
        forecaster.fit(y, X=X).predict(fh=[1, 2])

    call = fake.calls[0]
    np.testing.assert_allclose(call["past_only_covariates"], [[10.0, 20.0, 30.0, 40.0]])
    assert call["past_future_covariates"] is None


def test_upstream_arrays_past_future_covariates():
    """Past-and-future covariates span context plus horizon."""
    fake = _FakeUpstreamForecaster()
    y = pd.Series([1.0, 2.0, 3.0, 4.0])
    X = pd.DataFrame({"pf": [10.0, 20.0, 30.0, 40.0]})
    X_future = pd.DataFrame({"pf": [50.0, 60.0]})

    with _patch_upstream(fake):
        forecaster = _make_forecaster()
        forecaster.fit(y, X=X).predict(fh=[1, 2], X=X_future)

    call = fake.calls[0]
    np.testing.assert_allclose(
        call["past_future_covariates"], [[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]]
    )


def test_upstream_arrays_mixed_covariates():
    """Mixed past-only and past-and-future covariates are partitioned correctly."""
    fake = _FakeUpstreamForecaster()
    y = pd.Series([1.0, 2.0, 3.0, 4.0])
    X = pd.DataFrame({"po": [1.0, 2.0, 3.0, 4.0], "pf": [10.0, 20.0, 30.0, 40.0]})
    X_future = pd.DataFrame({"pf": [50.0, 60.0]})

    with _patch_upstream(fake):
        forecaster = _make_forecaster(past_covariates=["po"])
        forecaster.fit(y, X=X).predict(fh=[1, 2], X=X_future)

    call = fake.calls[0]
    np.testing.assert_allclose(call["past_only_covariates"], [[1.0, 2.0, 3.0, 4.0]])
    np.testing.assert_allclose(
        call["past_future_covariates"], [[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]]
    )


def test_predict_without_fit_exog_when_future_covariates_used():
    """Predict requires fit-time history when future covariates are used."""
    fake = _FakeUpstreamForecaster()
    y = pd.Series([1.0, 2.0, 3.0, 4.0])
    X = pd.DataFrame({"pf": [10.0, 20.0, 30.0, 40.0]})

    with _patch_upstream(fake):
        forecaster = _make_forecaster()
        forecaster.fit(y, X=X)
        with pytest.raises(ValueError, match="must also be provided in predict"):
            forecaster.predict(fh=[1, 2])


def test_predict_missing_future_covariate_column():
    """Missing future covariate columns in predict raise an error."""
    fake = _FakeUpstreamForecaster()
    y = pd.Series([1.0, 2.0, 3.0, 4.0])
    X = pd.DataFrame({"pf": [10.0, 20.0, 30.0, 40.0]})
    X_future = pd.DataFrame({"other": [1.0, 2.0]})

    with _patch_upstream(fake):
        forecaster = _make_forecaster()
        forecaster.fit(y, X=X)
        with pytest.raises(ValueError, match="missing past-and-future"):
            forecaster.predict(fh=[1, 2], X=X_future)


def test_predict_extra_future_covariate_column():
    """Extra future covariate columns in predict raise an error."""
    fake = _FakeUpstreamForecaster()
    y = pd.Series([1.0, 2.0, 3.0, 4.0])
    X = pd.DataFrame({"pf": [10.0, 20.0, 30.0, 40.0]})
    X_future = pd.DataFrame({"pf": [50.0, 60.0], "extra": [1.0, 2.0]})

    with _patch_upstream(fake):
        forecaster = _make_forecaster()
        forecaster.fit(y, X=X)
        with pytest.raises(ValueError, match="contains columns that were not declared"):
            forecaster.predict(fh=[1, 2], X=X_future)


def test_predict_incomplete_future_coverage():
    """Incomplete future covariate coverage raises an error."""
    fake = _FakeUpstreamForecaster()
    y = pd.Series([1.0, 2.0, 3.0, 4.0])
    X = pd.DataFrame({"pf": [10.0, 20.0, 30.0, 40.0]})
    X_future = pd.DataFrame({"pf": [50.0]})

    with _patch_upstream(fake):
        forecaster = _make_forecaster()
        forecaster.fit(y, X=X)
        with pytest.raises(ValueError, match="must cover at least"):
            forecaster.predict(fh=[1, 2], X=X_future)


def test_unknown_past_covariate_column():
    """Unknown ``past_covariates`` entries raise at fit time."""
    fake = _FakeUpstreamForecaster()
    y = pd.Series([1.0, 2.0, 3.0, 4.0])
    X = pd.DataFrame({"pf": [10.0, 20.0, 30.0, 40.0]})

    with _patch_upstream(fake):
        forecaster = _make_forecaster(past_covariates=["missing"])
        with pytest.raises(ValueError, match="not present in fit-time `X`"):
            forecaster.fit(y, X=X)


def test_duplicate_past_covariates():
    """Duplicate ``past_covariates`` names raise at fit time."""
    fake = _FakeUpstreamForecaster()
    y = pd.Series([1.0, 2.0, 3.0, 4.0])
    X = pd.DataFrame({"po": [10.0, 20.0, 30.0, 40.0]})

    with _patch_upstream(fake):
        forecaster = _make_forecaster(past_covariates=["po", "po"])
        with pytest.raises(ValueError, match="unique column names"):
            forecaster.fit(y, X=X)


def test_non_numeric_exog():
    """Non-numeric exogenous columns raise at fit time."""
    fake = _FakeUpstreamForecaster()
    y = pd.Series([1.0, 2.0, 3.0, 4.0])
    X = pd.DataFrame({"cat": ["a", "b", "c", "d"]})

    with _patch_upstream(fake):
        forecaster = _make_forecaster()
        with pytest.raises(TypeError, match="categorical features"):
            forecaster.fit(y, X=X)


def test_variate_limit_exceeded():
    """More than 32 total variates raise at fit time."""
    fake = _FakeUpstreamForecaster()
    target_cols = {f"y{i}": np.ones(4) for i in range(20)}
    exog_cols = {f"x{i}": np.ones(4) for i in range(13)}
    y = pd.DataFrame(target_cols)
    X = pd.DataFrame(exog_cols)

    with _patch_upstream(fake):
        forecaster = _make_forecaster()
        with pytest.raises(ValueError, match="model limit"):
            forecaster.fit(y, X=X)


def test_missing_values_are_forwarded():
    """NaN values in context are forwarded to upstream unchanged."""
    fake = _FakeUpstreamForecaster()
    y = pd.Series([1.0, np.nan, 3.0, 4.0])

    with _patch_upstream(fake):
        forecaster = _make_forecaster()
        forecaster.fit(y).predict(fh=[1])

    call = fake.calls[0]
    assert np.isnan(call["context"][0, 1])


def test_pickle_excludes_model_and_reloads():
    """Pickle clears upstream model; predict reloads through cache."""
    fake = _FakeUpstreamForecaster()
    y = pd.Series([1.0, 2.0, 3.0, 4.0])

    with _patch_upstream(fake):
        forecaster = _make_forecaster()
        forecaster.fit(y)
        assert forecaster.forecaster_ is fake

        dumped = pickle.dumps(forecaster)
        restored = pickle.loads(dumped)
        assert restored.forecaster_ is None

    with _patch_upstream(fake):
        restored.predict(fh=[1])

    assert len(fake.calls) == 1


@pytest.mark.skipif(
    not run_test_for_class(TimesFM3Forecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_timesfm3_airline_predictions_match_upstream():
    """Adapter univariate output matches direct upstream inference."""
    y = load_airline()
    y_train = y.iloc[:-12]
    fh = np.arange(1, 4)

    forecaster = TimesFM3Forecaster(device="cpu", license_accepted=True)
    forecaster.fit(y_train)
    y_pred = forecaster.predict(fh=fh)

    upstream = forecaster.forecaster_
    context = y_train.values.astype(np.float32)
    direct = upstream.predict(context=context, horizon=3, return_quantiles=False)

    np.testing.assert_allclose(
        y_pred.iloc[:3].to_numpy(),
        direct.forecast[:3],
        rtol=1e-5,
        atol=1e-4,
    )


@pytest.mark.skipif(
    not run_test_for_class(TimesFM3Forecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_timesfm3_multivariate_covariate_parity_with_upstream():
    """Mixed covariate multivariate output matches direct upstream inference."""
    rng = np.random.default_rng(0)
    n = 32
    horizon = 4
    y = pd.DataFrame(
        {
            "target_a": rng.normal(size=n),
            "target_b": rng.normal(size=n),
        }
    )
    X = pd.DataFrame(
        {
            "past_only": rng.normal(size=n),
            "future_known": rng.normal(size=n),
        }
    )
    X_future = pd.DataFrame({"future_known": rng.normal(size=horizon)})

    forecaster = TimesFM3Forecaster(
        device="cpu",
        license_accepted=True,
        past_covariates=["past_only"],
    )
    y_pred = forecaster.fit(y, X=X).predict(fh=list(range(1, horizon + 1)), X=X_future)
    y_quant = forecaster.predict_quantiles(
        fh=list(range(1, horizon + 1)), X=X_future, alpha=[0.1, 0.5, 0.9]
    )

    upstream = forecaster.forecaster_
    target = y.values.T.astype(np.float32)
    past_only = X[["past_only"]].values.T.astype(np.float32)
    past = X[["future_known"]].values.T.astype(np.float32)
    future = X_future[["future_known"]].values.T.astype(np.float32)
    past_future = np.concatenate([past, future], axis=1)

    direct = upstream.predict(
        context=target,
        horizon=horizon,
        past_only_covariates=past_only,
        past_future_covariates=past_future,
        return_quantiles=True,
    )

    np.testing.assert_allclose(
        y_pred.to_numpy(), direct.forecast.T, rtol=1e-5, atol=1e-4
    )
    for alpha in [0.1, 0.5, 0.9]:
        idx = _DEFAULT_QUANTILES.index(alpha)
        expected = direct.quantiles[:, :, idx]
        actual = y_quant.xs(alpha, level=1, axis=1).to_numpy()
        np.testing.assert_allclose(actual, expected.T, rtol=1e-5, atol=1e-4)


@pytest.mark.skipif(
    not run_test_for_class(TimesFM3Forecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_timesfm3_missing_values_match_upstream():
    """Interior NaNs are handled consistently with upstream interpolation."""
    y = pd.Series([1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0] * 4)
    fh = [1, 2]

    forecaster = TimesFM3Forecaster(device="cpu", license_accepted=True)
    y_pred = forecaster.fit(y).predict(fh=fh)

    upstream = forecaster.forecaster_
    direct = upstream.predict(
        context=y.values.astype(np.float32),
        horizon=2,
        return_quantiles=False,
    )

    np.testing.assert_allclose(
        y_pred.to_numpy(),
        direct.forecast[:2],
        rtol=1e-5,
        atol=1e-4,
    )
