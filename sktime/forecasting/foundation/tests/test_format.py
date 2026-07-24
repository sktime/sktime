"""Tests for foundation forecast result formatting."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

import numpy as np
import pandas as pd
import pytest

from sktime.forecasting.foundation import ForecastRequest, ForecastResult
from sktime.forecasting.foundation._format import (
    format_point_result,
    format_quantile_result,
)


@pytest.fixture
def forecast_request():
    """Sparse future request used to test full-horizon row selection."""
    return ForecastRequest(
        relative_fh=(1, 3),
        absolute_index=pd.Index([10, 12]),
        alpha=None,
    )


@pytest.fixture
def y():
    """Target metadata used for output column names."""
    return pd.DataFrame({"a": [1.0], "b": [2.0]})


@pytest.mark.parametrize(
    "result",
    [
        ForecastResult(mean=[[1, 2], [3, 4], [5, 6]], median=np.zeros((3, 2))),
        ForecastResult(median=[[1, 2], [3, 4], [5, 6]]),
        ForecastResult(quantiles={0.5: [[1, 2], [3, 4], [5, 6]]}),
    ],
)
def test_format_point_result_uses_available_point_summary(result, forecast_request, y):
    """Mean, median, and the median quantile are valid point summaries."""
    actual = format_point_result(result=result, request=forecast_request, y=y)
    expected = pd.DataFrame(
        [[1, 2], [5, 6]], index=pd.Index([10, 12]), columns=["a", "b"]
    )

    pd.testing.assert_frame_equal(actual, expected)


def test_format_point_result_accepts_requested_rows_directly(forecast_request, y):
    """Backends may return either a full horizon or only requested rows."""
    result = ForecastResult(mean=[[1, 2], [5, 6]])

    actual = format_point_result(result=result, request=forecast_request, y=y)

    expected = pd.DataFrame(
        [[1, 2], [5, 6]], index=pd.Index([10, 12]), columns=["a", "b"]
    )
    pd.testing.assert_frame_equal(actual, expected)


def test_format_quantile_result_tolerates_float_key_rounding(forecast_request, y):
    """Numerically equivalent alpha keys are matched after rounding."""
    result = ForecastResult(
        quantiles={
            0.10000000000001: [[1, 2], [3, 4], [5, 6]],
            0.9: [[7, 8], [9, 10], [11, 12]],
        }
    )

    actual = format_quantile_result(
        result=result,
        request=forecast_request,
        y=y,
        alpha=[0.1, 0.9],
    )
    expected = pd.DataFrame(
        [[1, 7, 2, 8], [5, 11, 6, 12]],
        index=pd.Index([10, 12]),
        columns=pd.MultiIndex.from_product([["a", "b"], [0.1, 0.9]]),
    )

    pd.testing.assert_frame_equal(actual, expected)


def test_format_point_result_requires_point_values(forecast_request, y):
    """Missing point summaries produce an actionable error."""
    with pytest.raises(ValueError, match="does not contain a point forecast"):
        format_point_result(ForecastResult(), request=forecast_request, y=y)


def test_format_quantile_result_requires_requested_alpha(forecast_request, y):
    """A missing requested quantile reports the unavailable alpha."""
    result = ForecastResult(quantiles={0.5: [[1, 2], [3, 4], [5, 6]]})

    with pytest.raises(ValueError, match="alpha=0.1"):
        format_quantile_result(result, request=forecast_request, y=y, alpha=[0.1])
