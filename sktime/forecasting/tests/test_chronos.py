"""Tests for probabilistic forecasting in ChronosForecaster."""

import pytest
import pandas as pd
import numpy as np
from skbase.utils.dependencies import _check_soft_dependencies
from sktime.forecasting.base import ForecastingHorizon


def _make_datetime_series():
    """Create a simple datetime-indexed series to avoid PeriodIndex freq issues."""
    idx = pd.date_range(start="2000-01", periods=60, freq="MS")
    vals = np.random.default_rng(42).normal(100, 10, size=60)
    return pd.Series(vals, index=idx, name="y")


@pytest.mark.skipif(
    not _check_soft_dependencies("chronos-forecasting", severity="none"),
    reason="requires chronos-forecasting",
)
def test_chronos_predict_quantiles_shape():
    from sktime.forecasting.chronos import ChronosForecaster

    y = _make_datetime_series()
    fh = [1, 2, 3]
    alpha = [0.1, 0.5, 0.9]

    fc = ChronosForecaster("amazon/chronos-t5-tiny")
    fc.fit(y)
    result = fc.predict_quantiles(fh=fh, alpha=alpha)

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (3, 3)
    assert isinstance(result.columns, pd.MultiIndex)
    assert list(result.columns.get_level_values("quantile")) == alpha


@pytest.mark.skipif(
    not _check_soft_dependencies("chronos-forecasting", severity="none"),
    reason="requires chronos-forecasting",
)
def test_chronos_predict_interval_shape_and_ordering():
    from sktime.forecasting.chronos import ChronosForecaster

    y = _make_datetime_series()
    fh = [1, 2, 3]
    coverage = [0.8, 0.9]

    fc = ChronosForecaster("amazon/chronos-t5-tiny")
    fc.fit(y)
    result = fc.predict_interval(fh=fh, coverage=coverage)

    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == 3
    assert isinstance(result.columns, pd.MultiIndex)
    var = result.columns.get_level_values("variable")[0]
    for c in coverage:
        lower = result[(var, c, "lower")].values
        upper = result[(var, c, "upper")].values
        assert (lower <= upper).all(), f"lower > upper for coverage={c}"


@pytest.mark.skipif(
    not _check_soft_dependencies("chronos-forecasting", severity="none"),
    reason="requires chronos-forecasting",
)
def test_chronos_bolt_predict_quantiles():
    """Same checks but with Chronos-Bolt variant."""
    from sktime.forecasting.chronos import ChronosForecaster

    y = _make_datetime_series()
    fc = ChronosForecaster("amazon/chronos-bolt-tiny")
    fc.fit(y)
    result = fc.predict_quantiles(fh=[1, 2], alpha=[0.1, 0.9])

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (2, 2)
    assert isinstance(result.columns, pd.MultiIndex)
