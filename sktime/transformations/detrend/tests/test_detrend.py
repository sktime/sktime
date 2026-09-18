"""Test detrenders."""

import numpy as np
import pandas as pd
import pytest

from sktime.datasets import load_airline
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.forecasting.trend.tests.test_trend import get_expected_polynomial_coefs
from sktime.tests.test_switch import run_test_for_class
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
    not run_test_for_class([Detrender, PolynomialTrendForecaster]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_polynomial_detrending():
    """Test that transformer results agree with manual detrending."""
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
    not run_test_for_class([Detrender, PolynomialTrendForecaster]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_multiplicative_detrending_series(y_series):
    """Tests we get the expected result when setting `model=multiplicative`."""
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
    not run_test_for_class([Detrender, PolynomialTrendForecaster]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_multiplicative_detrending_dataframe(y_dataframe):
    """Tests we get the expected result when setting `model=multiplicative`."""
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
    not run_test_for_class([Detrender, PolynomialTrendForecaster]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_additive_detrending_series(y_series):
    """Tests we get the expected result when setting `model=additive`."""
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
    not run_test_for_class([Detrender, PolynomialTrendForecaster]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_additive_detrending_dataframe(y_dataframe):
    """Tests we get the expected result when setting `model=additive`."""
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
    """Detrender on unbalanced panel must not produce extra rows or NaN.

    Regression test for #11196: _get_fh_from_X used the union of timestamps
    across all instances as fh, causing X - X_pred to align on the union and
    introduce NaN rows for short instances.
    """
    idx1 = pd.MultiIndex.from_product([[0], range(4)], names=["instance", "time"])
    idx2 = pd.MultiIndex.from_product([[1], range(5)], names=["instance", "time"])
    s1 = pd.Series([10.0, 20.0, 30.0, 40.0], index=idx1)
    s2 = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx2)
    panel = pd.concat([s1, s2]).to_frame(name="value")

    from sktime.forecasting.trend import TrendForecaster

    out = Detrender(forecaster=TrendForecaster()).fit_transform(panel)

    assert out.shape == panel.shape, f"shape mismatch: {out.shape} != {panel.shape}"
    assert panel.index.equals(out.index), "output index != input index"
    assert not out.isna().any().any(), "unexpected NaN in output"


@pytest.mark.skipif(
    not run_test_for_class(Detrender),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_detrend_panel_unequal_start():
    """Detrender on panel with different start timestamps must not produce NaN.

    Regression test for #11196: instance 0 covers time 5-9, instance 1 covers
    time 0-9, so the union timestamp set includes in-sample timestamps for
    instance 1 that are out-of-range for instance 0.
    """
    idx_a = pd.MultiIndex.from_product([[0], range(5, 10)], names=["instance", "time"])
    idx_b = pd.MultiIndex.from_product([[1], range(10)], names=["instance", "time"])
    sa = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0], index=idx_a)
    sb = pd.Series([float(i) for i in range(10)], index=idx_b)
    panel = pd.concat([sa, sb]).to_frame(name="value")

    from sktime.forecasting.trend import TrendForecaster

    out = Detrender(forecaster=TrendForecaster()).fit_transform(panel)

    assert out.shape == panel.shape, f"shape mismatch: {out.shape} != {panel.shape}"
    assert panel.index.equals(out.index), "output index != input index"
    assert not out.isna().any().any(), "unexpected NaN in output"


@pytest.mark.skipif(
    not run_test_for_class(Detrender),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_detrend_panel_multiindex_hier():
    """Detrender on hierarchical panel must preserve index and produce no NaN.

    Regression test for #11196: same union-fh issue applies to pd_multiindex_hier.
    Uses two top-level groups with two sub-instances each of unequal length.
    """
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

    assert out.shape == panel.shape, f"shape mismatch: {out.shape} != {panel.shape}"
    assert panel.index.equals(out.index), "output index != input index"
    assert not out.isna().any().any(), "unexpected NaN in output"


@pytest.mark.skipif(
    not run_test_for_class(Detrender),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_detrend_panel_inverse_roundtrip():
    """inverse_transform(transform(X)) must recover X exactly on unbalanced panel.

    Regression test for #11196: checks that both _transform and _inverse_transform
    carry the reindex fix and that the round-trip produces no extra rows, no NaN,
    and zero numerical error.
    """
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

    assert restored.shape == panel.shape, (
        f"round-trip shape mismatch: {restored.shape} != {panel.shape}"
    )
    assert panel.index.equals(restored.index), "round-trip index != input index"
    assert not restored.isna().any().any(), "unexpected NaN after round-trip"
    pd.testing.assert_frame_equal(restored, panel, check_exact=False, atol=1e-10)

