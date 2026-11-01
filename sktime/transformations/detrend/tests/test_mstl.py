# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests MSTL functionality."""

__author__ = ["krishna-t"]

import pytest

from sktime.datasets import load_airline
from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.detrend.mstl import MSTL


@pytest.mark.skipif(
    not run_test_for_class([MSTL]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_transform_returns_correct_components():
    """Tests if expected components are returned when return_components=True."""
    # Load our default test dataset
    series = load_airline()
    series.index = series.index.to_timestamp()

    # Initialize the MSTL transformer with specific parameters
    transformer = MSTL(periods=[3, 12], return_components=True)

    # Fit the transformer to the data
    transformer.fit(series)

    # Transform the data
    transformed = transformer.transform(series)

    # Check if the transformed data has the expected components
    assert "trend" in transformed.columns, (
        "Test of MSTL.transform failed with return_components=True, "
        "returned DataFrame columns are missing 'trend' variable."
    )
    assert "resid" in transformed.columns, (
        "Test of MSTL.transform failed with return_components=True, "
        "returned DataFrame columns are missing 'resid' variable."
    )
    assert "seasonal_3" in transformed.columns, (
        "Test of MSTL.transform failed with return_components=True, "
        "returned DataFrame columns are missing 'seasonal_3 "
        "variable."
    )
    assert "seasonal_12" in transformed.columns, (
        "Test of MSTL.transform failed with return_components=True, "
        "returned DataFrame columns are missing 'seasonal_12' "
        "variable."
    )


@pytest.mark.skipif(
    not run_test_for_class(MSTL),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_mstl_deseason_pipeline():
    """Test simple MSTL multiple deseasonalization pipeline."""
    from sktime.forecasting.trend import TrendForecaster

    mstl_trafo = MSTL(periods=[2, 12])
    mstl_deseason_fcst = mstl_trafo * TrendForecaster()
    y = load_airline()
    mstl_deseason_fcst.fit(y, fh=[1, 2, 3])
    y_pred = mstl_deseason_fcst.predict()

    # check expected output shape
    assert y_pred.shape == (3,)
    # assert there are no nans
    assert not y_pred.isna().any()


@pytest.mark.skipif(
    not run_test_for_class(MSTL),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_mstl_component_pipeline():
    """Test simple MSTL component forecasting pipeline."""
    from sktime.forecasting.compose import ColumnEnsembleForecaster
    from sktime.forecasting.naive import NaiveForecaster
    from sktime.forecasting.trend import TrendForecaster

    mstl_trafo_comp = MSTL(periods=[2, 12], return_components=True)
    mstl_component_fcst = mstl_trafo_comp * ColumnEnsembleForecaster(
        [
            ("trend", TrendForecaster(), "trend"),
            ("sp2", NaiveForecaster(strategy="last", sp=2), "seasonal_2"),
            ("sp12", NaiveForecaster(strategy="last", sp=12), "seasonal_12"),
            ("residual", NaiveForecaster(strategy="last"), "resid"),
        ]
    )
    y = load_airline()
    mstl_component_fcst.fit(y, fh=[1, 2, 3])
    y_pred = mstl_component_fcst.predict()
    # check expected output shape
    assert y_pred.shape == (3,)
    # assert there are no nans
    assert not y_pred.isna().any()


@pytest.mark.skipif(
    not run_test_for_class([MSTL]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("periods", [3, [3], [3, 12], (3, 12)])
def test_inverse_transform_round_trip(periods):
    """Test that inverse_transform recovers the series for any form of periods.

    ``_inverse_transform`` iterated ``self.periods`` directly, which is the raw
    user parameter and may be a single int, and looked up a
    ``seasonal_<period>`` column, which statsmodels only produces when there is
    more than one period. A single period therefore raised ``TypeError`` when
    given as an int and ``KeyError`` when given as a one element sequence.
    """
    import numpy as np
    import pandas as pd

    index = pd.date_range("2020-01-01", periods=40, freq="D")
    X = pd.DataFrame({"a": np.linspace(1.0, 20.0, 40)}, index=index)

    transformer = MSTL(periods=periods, return_components=False)
    Xt = transformer.fit_transform(X)
    Xi = transformer.inverse_transform(Xt)

    np.testing.assert_allclose(
        np.asarray(Xi, dtype=float).ravel(),
        np.asarray(X, dtype=float).ravel(),
        atol=1e-8,
    )
