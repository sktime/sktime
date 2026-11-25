"""Tests for the Simons forecaster adapter."""

import numpy as np
import pandas as pd
import pytest

from sktime.forecasting.simons import SimonsForecaster
from sktime.libs.simons import QUANTILES
from sktime.tests.test_switch import run_test_for_class

pytestmark = pytest.mark.skipif(
    not run_test_for_class(SimonsForecaster),
    reason="run test only if Simons soft dependencies are installed",
)

# Generate using the upstream package:
# pip install "git+https://github.com/Abel-ai-lab/Simons@0d0042f8da45140172bd36980be08654c0017d8b"
# import numpy as np
# import torch
# from simons import QUANTILES, Simons
# history = torch.as_tensor(
#     np.sin(np.arange(200, dtype=np.float32) / 10)[None, None, :],
#     dtype=torch.float32,
# )
# raw = Simons.from_pretrained("abel-lab/Simons-25M", device="cpu").forecast(
#     history, horizon=3
# )
# point = raw.detach().cpu().numpy()[0, 0, :, 4]
# quantiles = np.apply_along_axis(
#     lambda row: np.interp([0.1, 0.25, 0.5, 0.9], QUANTILES, row),
#     axis=-1,
#     arr=raw.detach().cpu().numpy()[0, 0],
# )
# print(point)
# print(quantiles)
# input series: np.sin(np.arange(200, dtype=np.float32) / 10), horizon=3,
# checkpoint=abel-lab/Simons-25M,
# revision=main 0d0042f8da45140172bd36980be08654c0017d8b,
# torch=2.11.0+cpu
_GOLDEN_POINT = np.array([0.9178922, 0.95235765, 0.979503], dtype=np.float32)
_GOLDEN_QUANTILES = np.array(
    [
        [0.90627027, 0.91169557, 0.91789222, 0.92707527],
        [0.93742347, 0.94417527, 0.95235765, 0.96073419],
        [0.96606874, 0.97444379, 0.97950298, 0.98200995],
    ],
    dtype=np.float32,
)


def _make_y():
    """Create a deterministic univariate series."""
    return pd.Series(
        np.sin(np.arange(200, dtype=np.float32) / 10),
        index=pd.RangeIndex(200),
        name="y",
    )


def _make_x(index):
    """Create varying known-future covariates."""
    values = np.asarray(index, dtype=np.float32)
    return pd.DataFrame(
        {"x1": values, "x2": np.cos(values / 7)},
        index=index,
    )


def test_simons_non_contiguous_fh_index():
    """Non-contiguous horizons preserve the requested absolute index."""
    forecaster = SimonsForecaster(model_path="abel-lab/Simons-25M").fit(_make_y())

    y_pred = forecaster.predict(fh=[1, 5, 10])

    assert y_pred.index.equals(pd.Index([200, 204, 209]))


def test_simons_golden_outputs():
    """The 25M CPU checkpoint remains stable on a fixed series."""
    forecaster = SimonsForecaster(model_path="abel-lab/Simons-25M", device="cpu").fit(
        _make_y()
    )

    point = forecaster.predict(fh=[1, 2, 3])
    quantiles = forecaster.predict_quantiles(fh=[1, 2, 3], alpha=[0.1, 0.25, 0.5, 0.9])

    np.testing.assert_allclose(
        point.to_numpy().ravel(), _GOLDEN_POINT, rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(
        quantiles.to_numpy(), _GOLDEN_QUANTILES, rtol=1e-5, atol=1e-6
    )

    import torch

    from sktime.libs.simons import Simons

    history = np.asarray(_make_y(), dtype=np.float32)[None, None, :]
    raw = Simons.from_pretrained(
        "abel-lab/Simons-25M",
        device="cpu",
    ).forecast(torch.as_tensor(history, dtype=torch.float32), horizon=3)
    np.testing.assert_allclose(
        raw.detach().cpu().numpy()[0, 0, :, 4],
        point.to_numpy(),
        rtol=1e-6,
        atol=1e-7,
    )
    native = raw.detach().cpu().numpy()[0, 0]
    upstream_quantiles = np.apply_along_axis(
        lambda row: np.interp([0.1, 0.25, 0.5, 0.9], QUANTILES, row),
        axis=-1,
        arr=native,
    )
    np.testing.assert_allclose(
        quantiles.to_numpy(),
        upstream_quantiles,
        rtol=1e-6,
        atol=1e-7,
    )


def test_simons_update_matches_full_fit():
    """Updating with new observations uses the full context without refitting."""
    y = _make_y()
    updated = SimonsForecaster(model_path="abel-lab/Simons-25M").fit(y.iloc[:150])
    updated.update(y.iloc[150:], update_params=False)
    expected = SimonsForecaster(model_path="abel-lab/Simons-25M").fit(y)

    np.testing.assert_allclose(
        updated.predict(fh=[1, 2, 3]).to_numpy(),
        expected.predict(fh=[1, 2, 3]).to_numpy(),
        rtol=1e-5,
        atol=1e-6,
    )


def test_simons_predict_proba_matches_native_quantiles():
    """The histogram distribution preserves Simons' native quantiles."""
    forecaster = SimonsForecaster(model_path="abel-lab/Simons-25M").fit(_make_y())
    distribution = forecaster.predict_proba(fh=[1, 2, 3])
    quantiles = forecaster.predict_quantiles(fh=[1, 2, 3], alpha=QUANTILES)

    assert distribution.index.equals(quantiles.index)
    for level in QUANTILES:
        probabilities = pd.DataFrame(level, index=distribution.index, columns=["y"])
        actual = distribution.ppf(probabilities)
        expected = quantiles[("y", level)].to_numpy().reshape(-1, 1)
        np.testing.assert_allclose(actual.to_numpy(), expected, rtol=1e-5, atol=1e-6)


def test_simons_accepts_categorical_covariates():
    """Categorical covariates with differing category sets work end to end."""
    y = _make_y()
    x_history = pd.DataFrame(
        {"category": pd.Categorical(["a", "b"] * 100, categories=["a", "b"])},
        index=y.index,
    )
    x_future = pd.DataFrame(
        {"category": pd.Categorical(["b", "c"] * 5, categories=["b", "c"])},
        index=pd.RangeIndex(200, 210),
    )
    forecaster = SimonsForecaster(model_path="abel-lab/Simons-25M").fit(y, X=x_history)

    prediction = forecaster.predict(fh=[1, 5, 10], X=x_future)
    assert prediction.shape == (3,)
    assert prediction.index.equals(pd.Index([200, 204, 209]))


def test_simons_numeric_categorical_covariates_share_codes(monkeypatch):
    """Numeric labels use shared codes across fit and predict."""

    class CovariateEchoModel:
        def forecast(self, history, horizon, covariates):
            future_codes = covariates[:, :1, -horizon:]
            return future_codes.unsqueeze(-1).expand(-1, -1, -1, len(QUANTILES))

    monkeypatch.setattr(
        SimonsForecaster, "_load_model", lambda self: CovariateEchoModel()
    )
    y = _make_y()
    x_history = pd.DataFrame(
        {"category": pd.Categorical([1, 2] * 100)},
        index=y.index,
    )
    x_future = pd.DataFrame(
        {"category": pd.Categorical([2, 3])},
        index=pd.RangeIndex(200, 202),
    )
    forecaster = SimonsForecaster().fit(y, X=x_history)

    prediction = forecaster.predict(fh=[1, 2], X=x_future)

    assert prediction.shape == (2,)
    assert prediction.index.equals(pd.Index([200, 201]))
    np.testing.assert_array_equal(prediction.to_numpy(), np.array([1.0, 2.0]))


def test_simons_accepts_18_covariates_with_constants():
    """Constant covariates are dropped and the remaining 15 are accepted."""
    y = _make_y()
    x_history = pd.DataFrame(
        {f"x{i}": np.arange(200, dtype=np.float32) + i for i in range(15)}
        | {f"constant{i}": np.ones(200) for i in range(3)},
        index=y.index,
    )
    x_future = pd.DataFrame(
        {f"x{i}": np.arange(200, 210, dtype=np.float32) + i for i in range(15)}
        | {f"constant{i}": np.ones(10) for i in range(3)},
        index=pd.RangeIndex(200, 210),
    )
    forecaster = SimonsForecaster(model_path="abel-lab/Simons-25M").fit(y, X=x_history)

    prediction = forecaster.predict(fh=[1, 5, 10], X=x_future)
    assert prediction.shape == (3,)


def test_simons_alpha_interpolation_and_clamping():
    """Quantiles interpolate in-range and clamp out-of-range levels."""
    forecaster = SimonsForecaster(model_path="abel-lab/Simons-25M").fit(_make_y())

    quantiles = forecaster.predict_quantiles(fh=[1], alpha=[0.2, 0.25, 0.3])
    np.testing.assert_allclose(
        quantiles[("y", 0.25)].to_numpy(),
        (quantiles[("y", 0.2)] + quantiles[("y", 0.3)]).to_numpy() / 2,
    )
    with pytest.warns(UserWarning, match="clamped"):
        clamped = forecaster.predict_quantiles(fh=[1], alpha=0.025)
    native = forecaster.predict_quantiles(fh=[1], alpha=0.1)
    np.testing.assert_allclose(clamped.to_numpy(), native.to_numpy())


def test_simons_missing_future_x_raises():
    """Future X must contain every row through the maximum horizon."""
    y = _make_y()
    x_history = _make_x(y.index)
    x_future = _make_x(pd.RangeIndex(200, 210)).drop(index=205)
    forecaster = SimonsForecaster(model_path="abel-lab/Simons-25M").fit(y, X=x_history)

    with pytest.raises(ValueError, match="every future row"):
        forecaster.predict(fh=[1, 5, 10], X=x_future)


def test_simons_forecast_with_x_differs():
    """Known-future covariates affect the Simons forecast."""
    y = _make_y()
    x_history = _make_x(y.index)
    x_future = _make_x(pd.RangeIndex(200, 210))
    with_x = SimonsForecaster(model_path="abel-lab/Simons-25M").fit(y, X=x_history)
    without_x = SimonsForecaster(model_path="abel-lab/Simons-25M").fit(y)

    with_x_pred = with_x.predict(fh=[1, 5, 10], X=x_future)
    without_x_pred = without_x.predict(fh=[1, 5, 10])

    assert not np.allclose(with_x_pred.to_numpy(), without_x_pred.to_numpy())
