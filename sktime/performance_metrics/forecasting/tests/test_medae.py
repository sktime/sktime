# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Regression tests for MedianAbsoluteError output aggregation."""

import numpy as np
import pandas as pd
import pytest

from sktime.performance_metrics.forecasting import MedianAbsoluteError
from sktime.tests.test_switch import run_test_for_class

pytestmark = pytest.mark.skipif(
    not run_test_for_class(MedianAbsoluteError),
    reason="run test only if softdeps are present and incrementally (if requested)",
)


@pytest.mark.parametrize("as_numpy", [False, True])
@pytest.mark.parametrize("n_timepoints", [1, 3, 4])
@pytest.mark.parametrize(
    "multioutput, expected_by_index, expected_aggregate",
    [
        pytest.param(
            "uniform_average",
            [4.0, 4.0, 4.0, 5.0],
            [4.0, 8 / 3, 10 / 3],
            id="uniform",
        ),
        pytest.param(
            "raw_values",
            [[0.0, 2.0, 10.0], [9.0, 0.0, 3.0], [3.0, 9.0, 0.0], [12.0, 1.0, 2.0]],
            [[0.0, 2.0, 10.0], [3.0, 2.0, 3.0], [6.0, 1.5, 2.5]],
            id="raw",
        ),
        pytest.param(
            [1 / 3, 1 / 3, 1 / 3],
            [4.0, 4.0, 4.0, 5.0],
            [4.0, 8 / 3, 10 / 3],
            id="equal-weights",
        ),
        pytest.param(
            [0.2, 0.3, 0.5],
            [5.6, 3.3, 3.3, 3.7],
            [5.6, 2.7, 2.9],
            id="list-weights",
        ),
        pytest.param(
            np.array([0.2, 0.3, 0.5]),
            [5.6, 3.3, 3.3, 3.7],
            [5.6, 2.7, 2.9],
            id="array-weights",
        ),
    ],
)
def test_medae_multioutput(
    multioutput, expected_by_index, expected_aggregate, n_timepoints, as_numpy
):
    """Average variables by index, but take each variable's median over time."""
    index = pd.date_range("2024-01-01", periods=n_timepoints, name="time")
    columns = ["c", "a", "b"]
    y_true = pd.DataFrame(1.0, index=index, columns=columns)
    signed_errors = np.array(
        [[0.0, -2.0, 10.0], [-9.0, 0.0, 3.0], [3.0, -9.0, 0.0], [-12.0, 1.0, -2.0]]
    )[:n_timepoints]
    y_pred = y_true + signed_errors
    if as_numpy:
        y_true, y_pred = y_true.to_numpy(), y_pred.to_numpy()
        index, columns = pd.RangeIndex(n_timepoints), pd.RangeIndex(3)

    metric = MedianAbsoluteError(multioutput=multioutput)
    # Hand-computed column medians distinguish median-over-time from row averaging.
    aggregate = expected_aggregate[[1, 3, 4].index(n_timepoints)]
    np.testing.assert_allclose(metric.evaluate(y_true, y_pred), aggregate)
    np.testing.assert_allclose(metric(y_true, y_pred), aggregate)

    expected = expected_by_index[:n_timepoints]
    if isinstance(multioutput, str) and multioutput == "raw_values":
        expected = pd.DataFrame(expected, index=index, columns=columns)
        assert_equal = pd.testing.assert_frame_equal
    else:
        expected = pd.Series(expected, index=index)
        assert_equal = pd.testing.assert_series_equal

    assert_equal(metric.evaluate_by_index(y_true, y_pred), expected)
    metric.set_params(by_index=True)
    assert_equal(metric(y_true, y_pred), expected)
    # by_index only changes __call__, not the explicit evaluate method.
    np.testing.assert_allclose(metric.evaluate(y_true, y_pred), aggregate)


@pytest.mark.parametrize("input_type", ["series", "dataframe", "numpy"])
@pytest.mark.parametrize("multioutput", ["uniform_average", "raw_values"])
def test_medae_univariate(input_type, multioutput):
    """Univariate absolute errors and their median are unchanged."""
    y_true = pd.Series([1.0, 1.0, 1.0], name="a")
    y_pred = pd.Series([1.0, -1.0, 11.0], name="a")
    if input_type == "dataframe":
        y_true, y_pred = y_true.to_frame(), y_pred.to_frame()
    elif input_type == "numpy":
        y_true, y_pred = y_true.to_numpy(), y_pred.to_numpy()

    metric = MedianAbsoluteError(multioutput=multioutput)
    np.testing.assert_allclose(metric(y_true, y_pred), 2.0)
    expected = [0.0, 2.0, 10.0]
    if multioutput == "raw_values":
        expected = np.array(expected).reshape(-1, 1)
    np.testing.assert_allclose(metric.evaluate_by_index(y_true, y_pred), expected)
    metric.set_params(by_index=True)
    np.testing.assert_allclose(metric(y_true, y_pred), expected)
