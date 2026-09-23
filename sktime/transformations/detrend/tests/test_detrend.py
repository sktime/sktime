"""Test detrenders."""

import numpy as np
import pandas as pd
import pytest

from sktime.datasets import load_airline
from sktime.tests.test_switch import run_test_for_class, run_test_module_changed
from sktime.transformations.detrend import Detrender

__author__ = ["mloning", "KishManani"]
__all__ = []


@pytest.fixture()
def y_series():
    return load_airline()


@pytest.fixture()
def y_dataframe():
    return load_airline().to_frame()


@pytest.mark.skipif(
    not run_test_for_class([Detrender])
    and not run_test_module_changed("sktime.transformations.detrend"),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_polynomial_detrending():
    """Test that transformer results agree with manual detrending."""
    from sktime.forecasting.trend import PolynomialTrendForecaster
    from sktime.forecasting.trend.tests.test_trend import get_expected_polynomial_coefs

    y = pd.Series(np.arange(20) * 0.5) + np.random.normal(0, 1, size=20)
    forecaster = PolynomialTrendForecaster(degree=1, with_intercept=True)
    transformer = Detrender(forecaster)
    transformer.fit(y)

    # check coefficients
    actual_coefs = transformer.forecaster_.regressor_.steps[-1][-1].coef_
    expected_coefs = get_expected_polynomial_coefs(y, degree=1, with_intercept=True)[
        ::-1
    ]
    np.testing.assert_array_almost_equal(actual_coefs, expected_coefs)

    # check trend
    n = len(y)
    expected_trend = expected_coefs[0] + np.arange(n) * expected_coefs[1]
    expected_trend_2D = np.reshape(expected_trend, (n, 1))
    actual_trend = transformer.forecaster_.predict(-np.arange(n))
    np.testing.assert_array_almost_equal(actual_trend, expected_trend_2D)

    # check residuals
    actual = transformer.transform(y)
    expected = y - expected_trend
    np.testing.assert_array_almost_equal(actual, expected)


@pytest.mark.skipif(
    not run_test_for_class([Detrender])
    and not run_test_module_changed("sktime.transformations.detrend"),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_multiplicative_detrending_series(y_series):
    """Tests we get the expected result when setting `model=multiplicative`."""
    from sktime.forecasting.trend import PolynomialTrendForecaster

    # Load test dataset
    y = y_series

    # Get the trend
    forecaster = PolynomialTrendForecaster(degree=1, with_intercept=True)
    trend = forecaster.fit_predict(y, fh=y.index)

    # De-trend the time series
    detrender = Detrender(forecaster, model="multiplicative")
    y_transformed = detrender.fit_transform(y)

    # Compute the expected de-trended time series
    expected = y / trend

    pd.testing.assert_series_equal(y_transformed, expected)


@pytest.mark.skipif(
    not run_test_for_class([Detrender])
    and not run_test_module_changed("sktime.transformations.detrend"),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_multiplicative_detrending_dataframe(y_dataframe):
    """Tests we get the expected result when setting `model=multiplicative`."""
    from sktime.forecasting.trend import PolynomialTrendForecaster

    # Load test dataset
    y = y_dataframe

    # Get the trend
    forecaster = PolynomialTrendForecaster(degree=1, with_intercept=True)
    trend = forecaster.fit_predict(y, fh=y.index)

    # De-trend the time series
    detrender = Detrender(forecaster, model="multiplicative")
    y_transformed = detrender.fit_transform(y)

    # Compute the expected de-trended time series
    expected = y / trend

    pd.testing.assert_frame_equal(y_transformed, expected)


@pytest.mark.skipif(
    not run_test_for_class([Detrender])
    and not run_test_module_changed("sktime.transformations.detrend"),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_additive_detrending_series(y_series):
    """Tests we get the expected result when setting `model=additive`."""
    from sktime.forecasting.trend import PolynomialTrendForecaster

    # Load test dataset
    y = y_series

    # Get the trend
    forecaster = PolynomialTrendForecaster(degree=1, with_intercept=True)
    trend = forecaster.fit_predict(y, fh=y.index)

    # De-trend the time series
    detrender = Detrender(forecaster, model="additive")
    y_transformed = detrender.fit_transform(y)

    # Compute the expected de-trended time series
    expected = y - trend

    pd.testing.assert_series_equal(y_transformed, expected)


@pytest.mark.skipif(
    not run_test_for_class([Detrender])
    and not run_test_module_changed("sktime.transformations.detrend"),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_additive_detrending_dataframe(y_dataframe):
    """Tests we get the expected result when setting `model=additive`."""
    from sktime.forecasting.trend import PolynomialTrendForecaster

    # Load test dataset
    y = y_dataframe

    # Get the trend
    forecaster = PolynomialTrendForecaster(degree=1, with_intercept=True)
    trend = forecaster.fit_predict(y, fh=y.index)

    # De-trend the time series
    detrender = Detrender(forecaster, model="additive")
    y_transformed = detrender.fit_transform(y)

    # Compute the expected de-trended time series
    expected = y - trend

    pd.testing.assert_frame_equal(y_transformed, expected)


@pytest.mark.skipif(
    not run_test_for_class(Detrender),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_detrend_panel_unequal_lengths():
    """Test that Detrender on an unbalanced panel does not produce NaN rows."""
    idx1 = pd.MultiIndex.from_product([[0], range(4)], names=["instance", "time"])
    idx2 = pd.MultiIndex.from_product([[1], range(5)], names=["instance", "time"])
    s1 = pd.Series([10.0, 20.0, 30.0, 40.0], index=idx1)
    s2 = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx2)
    panel = pd.concat([s1, s2]).to_frame(name="value")

    from sktime.forecasting.trend import TrendForecaster

    out = Detrender(forecaster=TrendForecaster()).fit_transform(panel)

    assert out.shape == panel.shape
    assert panel.index.equals(out.index)
    assert not out.isna().any().any()


@pytest.mark.skipif(
    not run_test_for_class(Detrender),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_detrend_panel_unequal_start():
    """Test that Detrender on a panel with staggered start times does not produce NaN."""
    idx_a = pd.MultiIndex.from_product([[0], range(5, 10)], names=["instance", "time"])
    idx_b = pd.MultiIndex.from_product([[1], range(10)], names=["instance", "time"])
    sa = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0], index=idx_a)
    sb = pd.Series([float(i) for i in range(10)], index=idx_b)
    panel = pd.concat([sa, sb]).to_frame(name="value")

    from sktime.forecasting.trend import TrendForecaster

    out = Detrender(forecaster=TrendForecaster()).fit_transform(panel)

    assert out.shape == panel.shape
    assert panel.index.equals(out.index)
    assert not out.isna().any().any()


@pytest.mark.skipif(
    not run_test_for_class(Detrender),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_detrend_panel_multiindex_hier():
    """Test that Detrender on a hierarchical panel with unequal lengths preserves the index."""
    import numpy as np

    idx = pd.MultiIndex.from_tuples(
        [
            ("A", "a", 0),
            ("A", "a", 1),
            ("A", "a", 2),
            ("A", "b", 0),
            ("A", "b", 1),
            ("A", "b", 2),
            ("A", "b", 3),
            ("B", "a", 0),
            ("B", "a", 1),
            ("B", "b", 0),
            ("B", "b", 1),
            ("B", "b", 2),
        ],
        names=["top", "mid", "time"],
    )
    vals = np.arange(12, dtype=float)
    panel = pd.DataFrame({"value": vals}, index=idx)

    from sktime.forecasting.trend import TrendForecaster

    out = Detrender(forecaster=TrendForecaster()).fit_transform(panel)

    assert out.shape == panel.shape
    assert panel.index.equals(out.index)
    assert not out.isna().any().any()


@pytest.mark.skipif(
    not run_test_for_class(Detrender),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_detrend_panel_inverse_roundtrip():
    """Test that inverse_transform recovers an unbalanced panel exactly."""
    idx1 = pd.MultiIndex.from_product([[0], range(4)], names=["instance", "time"])
    idx2 = pd.MultiIndex.from_product([[1], range(5)], names=["instance", "time"])
    s1 = pd.Series([10.0, 20.0, 30.0, 40.0], index=idx1)
    s2 = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx2)
    panel = pd.concat([s1, s2]).to_frame(name="value")

    from sktime.forecasting.trend import TrendForecaster

    tr = Detrender(forecaster=TrendForecaster())
    tr.fit(panel)
    detrended = tr.transform(panel)
    restored = tr.inverse_transform(detrended)

    assert restored.shape == panel.shape
    assert panel.index.equals(restored.index)
    assert not restored.isna().any().any()
    pd.testing.assert_frame_equal(restored, panel, check_exact=False, atol=1e-10)

