"""Tests for the Simons forecaster adapter."""

import numpy as np
import pandas as pd
import pytest

from sktime.forecasting.simons import SimonsForecaster
from sktime.tests.test_switch import run_test_for_class

pytestmark = pytest.mark.skipif(
    not run_test_for_class(SimonsForecaster),
    reason="run test only if Simons soft dependencies are installed",
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
