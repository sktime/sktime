"""Tests for get intervals in time series forests."""

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
    given_inner_series_length: int | None = inner_series_length

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
    given_inner_series_length: int | None = inner_series_length

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
    given_inner_series_length: int | None = inner_series_length

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
    given_inner_series_length: int | None = inner_series_length

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
    given_inner_series_length: int | None = inner_series_length

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


@pytest.mark.skipif(
    not run_test_module_changed("sktime.base._panel.forest"),
    reason="skip test if required soft dependency not available",
)
def test_transform_multivariate_output_shape():
    """Test _transform_multivariate returns correct shape."""
    from sktime.base._panel.forest._tsf import _transform_multivariate

    n_samples, n_channels, series_length = 10, 3, 50
    n_intervals = 5
    X = np.random.rand(n_samples, n_channels, series_length)
    intervals = np.array([[i * 8, i * 8 + 8] for i in range(n_intervals)])
    result = _transform_multivariate(X, intervals)
    assert result.shape == (n_samples, 3 * n_intervals * n_channels)


@pytest.mark.skipif(
    not run_test_module_changed("sktime.base._panel.forest"),
    reason="skip test if required soft dependency not available",
)
def test_tsf_multivariate_fit_predict():
    """Test TimeSeriesForestClassifier fits and predicts on multivariate data."""
    from sktime.classification.interval_based import TimeSeriesForestClassifier

    n_samples, n_channels, series_length = 20, 3, 30
    X = np.random.rand(n_samples, n_channels, series_length)
    y = np.array(["a"] * 10 + ["b"] * 10)

    clf = TimeSeriesForestClassifier(n_estimators=3, use_multivariate="yes")
    clf.fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == (n_samples,)
    assert set(preds).issubset({"a", "b"})


@pytest.mark.skipif(
    not run_test_module_changed("sktime.base._panel.forest"),
    reason="skip test if required soft dependency not available",
)
def test_tsf_invalid_use_multivariate():
    """Test TimeSeriesForestClassifier raises on invalid use_multivariate."""
    from sktime.classification.interval_based import TimeSeriesForestClassifier

    with pytest.raises(ValueError, match="use_multivariate must be one of"):
        TimeSeriesForestClassifier(use_multivariate="invalid")


@pytest.mark.skipif(
    not run_test_module_changed("sktime.base._panel.forest"),
    reason="skip test if required soft dependency not available",
)
def test_tsf_univariate_unchanged():
    """Test that univariate behavior is unchanged with use_multivariate=no."""
    from sktime.classification.interval_based import TimeSeriesForestClassifier
    from sktime.datasets import load_unit_test

    X_train, y_train = load_unit_test(split="train", return_X_y=True)
    X_test, _ = load_unit_test(split="test", return_X_y=True)

    clf = TimeSeriesForestClassifier(n_estimators=3, use_multivariate="no")
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    assert preds.shape == (X_test.shape[0],)
