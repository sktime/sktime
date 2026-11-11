"""Tests for TabularToSeriesAdaptor."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["mloning"]

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import FunctionTransformer, PowerTransformer, StandardScaler

from sktime.datasets import load_airline
from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.adapt import TabularToSeriesAdaptor


@pytest.mark.skipif(
    not run_test_for_class([TabularToSeriesAdaptor, PowerTransformer]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_boxcox_transform():
    """Test whether adaptor based transformer behaves like the raw wrapped method."""
    from scipy.stats import boxcox

    y = load_airline()
    t = TabularToSeriesAdaptor(PowerTransformer(method="box-cox", standardize=False))
    actual = t.fit_transform(y)

    expected, _ = boxcox(np.asarray(y))  # returns fitted lambda as second output
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("input_type", ["numpy", "pandas"])
@pytest.mark.parametrize("pooling", ["local", "global"])
@pytest.mark.parametrize("index_kind", ["integer", "datetime", "period"])
@pytest.mark.parametrize("nlevels", [1, 2, 3])
def test_inverse_transform_preserves_index(input_type, pooling, index_kind, nlevels):
    """Inverse scaling preserves the index of the current batch, including panels."""
    if index_kind == "integer":
        time = pd.RangeIndex(10, 16, name="time")
    elif index_kind == "datetime":
        time = pd.date_range("2026-01-01", periods=6, freq="D", name="time")
    else:
        time = pd.period_range("2026-01", periods=6, freq="M", name="time")
    if nlevels == 1:
        index = time
    elif nlevels == 2:
        index = pd.MultiIndex.from_product([["a", "b"], time], names=["store", "time"])
    else:
        index = pd.MultiIndex.from_product(
            [["north", "south"], ["a", "b"], time],
            names=["region", "store", "time"],
        )
    X = pd.DataFrame(
        np.arange(len(index) * 2, dtype=float).reshape(-1, 2) + 10,
        index=index,
        columns=["sales", "cost"],
    )
    transformer = TabularToSeriesAdaptor(
        StandardScaler(), pooling=pooling, input_type=input_type
    ).fit(X)
    batch = X.iloc[::2]
    transformed = transformer.transform(batch)
    pd.testing.assert_index_equal(transformed.index, batch.index)

    actual = transformer.inverse_transform(transformed)

    pd.testing.assert_index_equal(actual.index, batch.index)
    np.testing.assert_allclose(actual, batch)


@pytest.mark.parametrize("pooling", ["local", "global"])
@pytest.mark.parametrize("panel", [False, True])
def test_inverse_transform_dataframe_output(pooling, panel):
    """A wrapped transform returning a DataFrame also keeps the original dates."""
    index = pd.date_range("2026-01-01", periods=6, freq="D", name="time")
    if panel:
        index = pd.MultiIndex.from_product([["a", "b"], index], names=["store", "time"])
    X = pd.DataFrame({"sales": np.arange(len(index), dtype=float) + 10}, index=index)
    transformer = TabularToSeriesAdaptor(
        FunctionTransformer(func=np.log1p, inverse_func=np.expm1),
        pooling=pooling,
        input_type="pandas",
    )

    actual = transformer.inverse_transform(transformer.fit_transform(X))

    pd.testing.assert_frame_equal(actual, X)


@pytest.mark.parametrize("pooling", ["local", "global"])
@pytest.mark.parametrize("index_kind", ["integer", "datetime", "panel"])
def test_inverse_transform_series_output(pooling, index_kind):
    """Relabel a Series backend result by position without aligning its values."""
    index = pd.Index([1, 2, 3], name="time")
    if index_kind == "datetime":
        index = pd.date_range("2026-01-01", periods=3, name="time")
    elif index_kind == "panel":
        index = pd.MultiIndex.from_product([["a", "b"], index], names=["store", "time"])
    X = pd.DataFrame({"sales": np.arange(len(index), dtype=float) + 10}, index=index)
    transformer = TabularToSeriesAdaptor(
        FunctionTransformer(
            func=np.log1p,
            inverse_func=lambda X: np.expm1(X).squeeze(axis="columns"),
            check_inverse=False,
        ),
        pooling=pooling,
        input_type="pandas",
    )

    actual = transformer.inverse_transform(transformer.fit_transform(X))

    pd.testing.assert_frame_equal(actual, X)


@pytest.mark.parametrize("pooling", ["local", "global"])
def test_inverse_transform_result_index_is_independent(pooling):
    """Changing result index metadata must not modify the inverse input."""
    index = pd.date_range("2026-01-01", periods=3, name="time")
    X = pd.DataFrame({"sales": [10.0, 11.0, 12.0]}, index=index)
    transformer = TabularToSeriesAdaptor(
        FunctionTransformer(), pooling=pooling, input_type="pandas"
    )
    transformed = transformer.fit_transform(X)
    original = transformed.copy(deep=True)

    actual = transformer.inverse_transform(transformed)
    actual.index.name = "result_time"

    pd.testing.assert_frame_equal(transformed, original)


@pytest.mark.parametrize("input_type", ["numpy", "pandas"])
@pytest.mark.parametrize("pooling", ["local", "global"])
@pytest.mark.parametrize("panel", [False, True])
def test_scaled_forecast_preserves_horizon(input_type, pooling, panel):
    """Inverse scaling keeps sparse forecast dates and panel instance labels."""
    from sktime.forecasting.compose import TransformedTargetForecaster
    from sktime.forecasting.naive import NaiveForecaster

    time = pd.date_range("2026-01-01", periods=6, freq="D", name="time")
    future = pd.DatetimeIndex(["2026-01-07", "2026-01-09"])
    index = time
    expected_index = future
    expected_values = [15.0, 15.0]
    if panel:
        index = pd.MultiIndex.from_product([["a", "b"], time], names=["store", "time"])
        expected_index = pd.MultiIndex.from_product(
            [["a", "b"], future], names=["store", "time"]
        )
        expected_values = [15.0, 15.0, 21.0, 21.0]
    y = pd.DataFrame({"sales": np.arange(len(index), dtype=float) + 10}, index=index)
    forecaster = TransformedTargetForecaster(
        [
            TabularToSeriesAdaptor(
                StandardScaler(), pooling=pooling, input_type=input_type
            ),
            NaiveForecaster(),
        ]
    )

    actual = forecaster.fit(y, fh=[1, 3]).predict()

    pd.testing.assert_index_equal(actual.index, expected_index)
    np.testing.assert_allclose(actual.iloc[:, 0], expected_values)
