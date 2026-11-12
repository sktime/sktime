"""Tests for the ci-proto forecasting adapter without downloading weights."""

import numpy as np
import pandas as pd
import pytest

from sktime.forecasting.ci_proto import CIProtoForecaster


class _FakeModel:
    quantiles = np.arange(0.1, 1.0, 0.1)

    def forecast(self, history, horizon, **kwargs):
        self.history = history
        self.horizon = horizon
        self.kwargs = kwargs
        n_variables = history.shape[0]
        return np.broadcast_to(
            np.arange(horizon)[None, None, :] + self.quantiles[None, :, None],
            (n_variables, len(self.quantiles), horizon),
        ).copy()


@pytest.fixture
def fake_model(monkeypatch):
    model = _FakeModel()
    monkeypatch.setattr(
        "sktime.forecasting.base._base._check_estimator_deps",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        "sktime.forecasting.ci_proto._CachedCIProto",
        lambda **kwargs: type("Loader", (), {"load": lambda self: model})(),
    )
    return model


def test_sparse_horizon_multivariate_and_quantiles(fake_model):
    """The adapter selects requested steps and the median in target column order."""
    y = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [4.0, 5.0, 6.0]})
    forecaster = CIProtoForecaster(device="cpu").fit(y)
    prediction = forecaster.predict(fh=[1, 3])
    assert list(prediction.columns) == ["a", "b"]
    np.testing.assert_allclose(prediction.to_numpy(), [[0.5, 0.5], [2.5, 2.5]])
    assert fake_model.horizon == 3
    assert fake_model.history.shape == (2, 3)
    assert np.isnan(fake_model.history[0, 1].item())


def test_covariates_are_split_by_future_availability(fake_model):
    """Known future columns include their history and future values."""
    y = pd.DataFrame({"target": [1.0, 2.0, 3.0]})
    X = pd.DataFrame({"past": [5.0, 6.0, 7.0], "known": [8.0, 9.0, 10.0]})
    X_future = pd.DataFrame({"known": [11.0, 12.0]}, index=[3, 4])
    forecaster = CIProtoForecaster(device="cpu")
    forecaster.fit(y, X=X).predict([1, 2], X=X_future)
    np.testing.assert_allclose(fake_model.kwargs["past_covariates"], [[5, 6, 7]])
    np.testing.assert_allclose(
        fake_model.kwargs["future_covariates"], [[8, 9, 10, 11, 12]]
    )


def test_predict_quantiles_matches_native_grid(fake_model):
    """Probabilistic results preserve the upstream quantile and time axes."""
    y = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    forecaster = CIProtoForecaster(device="cpu").fit(y)
    quantiles = forecaster.predict_quantiles(fh=[1, 3], alpha=[0.1, 0.5, 0.9])
    assert quantiles.shape == (2, 6)
    np.testing.assert_allclose(quantiles[("a", 0.1)], [0.1, 2.1])
    np.testing.assert_allclose(quantiles[("b", 0.5)], [0.5, 2.5])

    distribution = forecaster.predict_proba(fh=[1, 3])
    np.testing.assert_allclose(
        distribution.quantile([0.1, 0.5, 0.9]).to_numpy(),
        quantiles.to_numpy(),
    )
    intervals = forecaster.predict_interval(fh=[1, 3], coverage=0.8)
    assert intervals.shape == (2, 4)


def test_rejects_gap_in_future_covariates(fake_model):
    """Known future values must cover unrequested intermediate steps."""
    y = pd.DataFrame({"target": [1.0, 2.0, 3.0]})
    X = pd.DataFrame({"known": [8.0, 9.0, 10.0]})
    X_future = pd.DataFrame({"known": [11.0, 13.0]}, index=[3, 5])
    forecaster = CIProtoForecaster(device="cpu").fit(y, X=X)
    with pytest.raises(ValueError, match="full forecast horizon|consecutive"):
        forecaster.predict([1, 3], X=X_future)
