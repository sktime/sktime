"""Tests for get intervals in time series forests."""

from typing import Optional

import numpy as np
import pytest
from numpy.random import RandomState

from sktime.base._panel.forest._tsf import _get_intervals
from sktime.tests.test_switch import run_test_module_changed


@pytest.mark.skipif(
    not run_test_module_changed("sktime.base._panel.forest"),
    reason="skip test if required soft dependency not available",
)
@pytest.mark.parametrize(
    "number_of_intervals, min_interval, "
    "number_of_series, inner_series_length, expected_intervals",
    [(4, 3, 4, 6, np.array([[12, 17], [0, 3], [3, 10], [9, 12]]))],
)
def test_get_intervals(
    number_of_intervals: int,
    min_interval: int,
    number_of_series: int,
    inner_series_length: int,
    expected_intervals: np.ndarray,
):
    """Test get intervals."""
    # given
    given_n_intervals = number_of_intervals
    given_min_interval = min_interval
    given_series_length = inner_series_length * number_of_series
    given_rng: RandomState = RandomState(0)

    # When
    intervals = _get_intervals(
        n_intervals=given_n_intervals,
        min_interval=given_min_interval,
        series_length=given_series_length,
        rng=given_rng,
    )

    # Then
    assert np.array_equal(intervals, expected_intervals)


@pytest.mark.skipif(
    not run_test_module_changed("sktime.base._panel.forest"),
    reason="skip test if required soft dependency not available",
)
@pytest.mark.parametrize("number_of_intervals", [2, 5])
@pytest.mark.parametrize("min_interval", [3, 10, 30])
@pytest.mark.parametrize("inner_series_length", [10, 30, 100])
@pytest.mark.parametrize("number_of_series", [3, 4, 10])
def test_get_intervals_should_produce_as_much_interval_as_given(
    number_of_intervals: int,
    min_interval: int,
    inner_series_length: int,
    number_of_series: int,
):
    """Test get_intervals should produce as much interval as given."""
    # given
    given_n_intervals = number_of_intervals
    given_min_interval = min_interval
    given_series_length = inner_series_length * number_of_series
    given_rng: RandomState = RandomState(42)
    given_inner_series_length: Optional[int] = inner_series_length

    # When
    intervals = _get_intervals(
        n_intervals=given_n_intervals,
        min_interval=given_min_interval,
        series_length=given_series_length,
        rng=given_rng,
        inner_series_length=given_inner_series_length,
    )

    # Then
    assert len(intervals) == given_n_intervals


@pytest.mark.skipif(
    not run_test_module_changed("sktime.base._panel.forest"),
    reason="skip test if required soft dependency not available",
)
@pytest.mark.parametrize("number_of_intervals", [2, 5])
@pytest.mark.parametrize("min_interval", [3, 10, 30])
@pytest.mark.parametrize("inner_series_length", [100])
@pytest.mark.parametrize("number_of_series", [3, 4, 10])
def test_get_intervals_at_least_greater_than_min_interval_given(
    number_of_intervals: int,
    min_interval: int,
    inner_series_length: int,
    number_of_series: int,
):
    """Test get_intervals should at least greater than min interval given."""
    # given
    given_n_intervals = number_of_intervals
    given_min_interval = min_interval
    given_series_length = inner_series_length * number_of_series
    given_rng: RandomState = RandomState(42)
    given_inner_series_length: Optional[int] = inner_series_length

    # When
    intervals = _get_intervals(
        n_intervals=given_n_intervals,
        min_interval=given_min_interval,
        series_length=given_series_length,
        rng=given_rng,
        inner_series_length=given_inner_series_length,
    )

    # Then
    assert all((intervals[:, 1] - intervals[:, 0]) >= given_min_interval)


@pytest.mark.skipif(
    not run_test_module_changed("sktime.base._panel.forest"),
    reason="skip test if required soft dependency not available",
)
@pytest.mark.parametrize("number_of_intervals", [2, 5])
@pytest.mark.parametrize("min_interval", [30, 50])
@pytest.mark.parametrize("inner_series_length", [10, 20])
@pytest.mark.parametrize("number_of_series", [3, 4, 10])
def test_get_intervals_equals_to_inner_series_length_given_too_high_min_interval(
    number_of_intervals: int,
    min_interval: int,
    inner_series_length: int,
    number_of_series: int,
):
    """Test get_intervals equals to inner series length given."""
    # given
    given_n_intervals = number_of_intervals
    given_min_interval = min_interval
    given_series_length = inner_series_length * number_of_series
    given_rng: RandomState = RandomState(42)
    given_inner_series_length: Optional[int] = inner_series_length

    # When
    intervals = _get_intervals(
        n_intervals=given_n_intervals,
        min_interval=given_min_interval,
        series_length=given_series_length,
        rng=given_rng,
        inner_series_length=given_inner_series_length,
    )

    # Then
    assert all((intervals[:, 1] - intervals[:, 0]) == given_inner_series_length)


@pytest.mark.skipif(
    not run_test_module_changed("sktime.base._panel.forest"),
    reason="skip test if required soft dependency not available",
)
@pytest.mark.parametrize("number_of_intervals", [2, 5])
@pytest.mark.parametrize("min_interval", [3, 10, 30])
@pytest.mark.parametrize("inner_series_length", [10, 30, 100])
@pytest.mark.parametrize("number_of_series", [3, 4, 10])
def test_get_intervals_should_produce_valid_intervals(
    number_of_intervals: int,
    min_interval: int,
    inner_series_length: int,
    number_of_series: int,
):
    """Tests get_intervals should produce valid intervals."""
    # given
    given_n_intervals = number_of_intervals
    given_min_interval = min_interval
    given_series_length = inner_series_length * number_of_series
    given_rng: RandomState = RandomState(42)
    given_inner_series_length: Optional[int] = inner_series_length

    # When
    intervals = _get_intervals(
        n_intervals=given_n_intervals,
        min_interval=given_min_interval,
        series_length=given_series_length,
        rng=given_rng,
        inner_series_length=given_inner_series_length,
    )

    # Then
    assert np.min(intervals) >= 0
    assert np.max(intervals) <= given_series_length


@pytest.mark.skipif(
    not run_test_module_changed("sktime.base._panel.forest"),
    reason="skip test if required soft dependency not available",
)
@pytest.mark.parametrize("number_of_intervals", [2, 5])
@pytest.mark.parametrize("min_interval", [3, 10, 30])
@pytest.mark.parametrize("inner_series_length", [10, 30, 100])
@pytest.mark.parametrize("number_of_series", [3, 4, 10])
def test_get_intervals_should_produce_intervals_contained_in_inner_series_bins(
    number_of_intervals: int,
    min_interval: int,
    inner_series_length: int,
    number_of_series: int,
):
    """Tests get_intervals should produce intervals contained in inner series bins."""
    # given
    given_n_intervals = number_of_intervals
    given_min_interval = min_interval
    given_series_length = inner_series_length * number_of_series
    given_rng: RandomState = RandomState(42)
    given_inner_series_length: Optional[int] = inner_series_length

    # When
    intervals = _get_intervals(
        n_intervals=given_n_intervals,
        min_interval=given_min_interval,
        series_length=given_series_length,
        rng=given_rng,
        inner_series_length=given_inner_series_length,
    )

    # Then
    assert all(
        intervals[:, 0]
        <= (intervals[:, 0] // given_inner_series_length + 1)
        * given_inner_series_length
    )
