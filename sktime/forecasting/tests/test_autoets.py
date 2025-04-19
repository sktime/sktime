"""Tests for AutoETS predict_quantiles with different forecasting horizons."""

__author__ = ["nahcol10"]

import gc

import numpy as np
import pandas as pd
import pytest

from sktime.datasets import load_airline
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.ets import AutoETS
from sktime.utils.dependencies import _check_soft_dependencies


@pytest.fixture
@pytest.mark.skipif(
    not _check_soft_dependencies("statsmodels", severity="none"),
    reason="statsmodels is not available",
)
def airline_data():
    """Fixture to load airline data once for all tests."""
    return load_airline().iloc[:60]


@pytest.fixture
@pytest.mark.skipif(
    not _check_soft_dependencies("statsmodels", severity="none"),
    reason="statsmodels is not available",
)
def fitted_forecaster(airline_data):
    """Fixture to create and fit an AutoETS forecaster."""
    forecaster = AutoETS(auto=True, n_jobs=1, sp=12)
    forecaster.fit(airline_data)
    yield forecaster
    # Clean up
    del forecaster
    gc.collect()


@pytest.fixture
@pytest.mark.skipif(
    not _check_soft_dependencies("statsmodels", severity="none"),
    reason="statsmodels is not available",
)
def fitted_auto_false_forecaster(airline_data):
    """Fixture for AutoETS with auto=False."""
    forecaster = AutoETS(auto=False, error="add", trend="add", seasonal="add", sp=12)
    forecaster.fit(airline_data)
    yield forecaster
    # Clean up
    del forecaster
    gc.collect()


# Parameterized test for different horizon types
@pytest.mark.parametrize(
    "fh",
    [
        [1, 2, 3],
        [3],
        [1, 5, 10],
        1,
        5,
        [-3, -2, -1],
        [-2, -1, 1, 2],
        np.array([1, 2, 3]),
        pd.Index([1, 2, 3]),
    ],
)
@pytest.mark.skipif(
    not _check_soft_dependencies("statsmodels", severity="none"),
    reason="statsmodels is not available",
)
def test_predict_quantiles_horizon_types(fitted_forecaster, fh):
    """Test predict_quantiles with various horizon types."""
    result = fitted_forecaster.predict_quantiles(fh=fh)

    assert isinstance(result, pd.DataFrame)
    expected_length = len(fh) if hasattr(fh, "__len__") else 1
    assert len(result) == expected_length
    assert len(result.columns) >= 2  # At least 2 quantiles


@pytest.mark.skipif(
    not _check_soft_dependencies("statsmodels", severity="none"),
    reason="statsmodels is not available",
)
def test_predict_quantiles_forecasting_horizon(fitted_forecaster):
    """Test with explicit ForecastingHorizon object."""
    fh = ForecastingHorizon([1, 2, 3], is_relative=True)
    result = fitted_forecaster.predict_quantiles(fh=fh)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(fh)


@pytest.mark.parametrize("alpha", [[0.1, 0.5, 0.9], [0.5], [0.025, 0.975]])
@pytest.mark.skipif(
    not _check_soft_dependencies("statsmodels", severity="none"),
    reason="statsmodels is not available",
)
def test_predict_quantiles_alphas(fitted_forecaster, alpha):
    """Test predict_quantiles with different alpha values."""
    fh = [1, 2]
    result = fitted_forecaster.predict_quantiles(fh=fh, alpha=alpha)

    assert isinstance(result, pd.DataFrame)
    assert len(result.columns.get_level_values(1).unique()) == len(alpha)

    var_name = result.columns.get_level_values(0)[0]
    for i in range(len(result)):
        values = [result.iloc[i][(var_name, a)] for a in sorted(alpha)]
        assert all(values[j] <= values[j + 1] for j in range(len(values) - 1))


@pytest.mark.skipif(
    not _check_soft_dependencies("statsmodels", severity="none"),
    reason="statsmodels is not available",
)
def test_predict_quantiles_auto_false(fitted_auto_false_forecaster):
    """Test with auto=False model."""
    result = fitted_auto_false_forecaster.predict_quantiles(fh=[1, 2])
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2


@pytest.mark.skipif(
    not _check_soft_dependencies("statsmodels", severity="none"),
    reason="statsmodels is not available",
)
def test_fh_immutability(fitted_forecaster):
    """Verify original FH isn't modified."""
    original_fh = ForecastingHorizon([1, 2], is_relative=True)
    original_values = original_fh.to_pandas().copy()

    fitted_forecaster.predict_quantiles(fh=original_fh)
    pd.testing.assert_index_equal(original_values, original_fh.to_pandas())


@pytest.mark.parametrize("fh", [[1, 5], [1, 3, 5]])
@pytest.mark.skipif(
    not _check_soft_dependencies("statsmodels", severity="none"),
    reason="statsmodels is not available",
)
def test_discontinuous_horizons(fitted_forecaster, fh):
    """Test with discontinuous horizons."""
    result = fitted_forecaster.predict_quantiles(fh=fh)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(fh)


@pytest.mark.skipif(
    not _check_soft_dependencies("statsmodels", severity="none"),
    reason="statsmodels is not available",
)
def test_long_horizon(fitted_forecaster):
    """Test with longer horizons."""
    long_fh = [10, 20]
    result = fitted_forecaster.predict_quantiles(fh=long_fh)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(long_fh)
    assert len(result.columns) >= 2
