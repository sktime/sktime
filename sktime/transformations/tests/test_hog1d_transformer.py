"""HOG1D transformer test code."""

import numbers

import numpy as np
import pandas as pd
import pytest

from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.hog1d import HOG1DTransformer
from sktime.utils._testing.panel import _make_nested_from_array


# Check that exception is raised for bad num intervals.
# input types - string, float, negative int, negative float, empty dict
# and an int that is larger than the time series length.
# correct input is meant to be a positive integer of 1 or more.
@pytest.mark.skipif(
    not run_test_for_class(HOG1DTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("bad_num_intervals", ["str", 1.2, -1.2, -1, {}, 11, 0])
def test_bad_num_intervals(bad_num_intervals):
    """Test that exception is raised for bad num intervals."""
    X = _make_nested_from_array(np.ones(10), n_instances=10, n_columns=1)

    if not isinstance(bad_num_intervals, int):
        with pytest.raises(TypeError):
            HOG1DTransformer(num_intervals=bad_num_intervals).fit(X).transform(X)
    else:
        with pytest.raises(ValueError):
            HOG1DTransformer(num_intervals=bad_num_intervals).fit(X).transform(X)


# Check that exception is raised for bad num bins.
# input types - string, float, negative float,
# negative int, empty dict and zero.
# correct input is meant to be a positive integer of 1 or more.
@pytest.mark.skipif(
    not run_test_for_class(HOG1DTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("bad_num_bins", ["str", 1.2, -1.2, -1, {}, 0])
def test_bad_num_bins(bad_num_bins):
    """Test that exception is raised for bad num bins."""
    X = _make_nested_from_array(np.ones(10), n_instances=10, n_columns=1)

    if not isinstance(bad_num_bins, int):
        with pytest.raises(TypeError):
            HOG1DTransformer(num_bins=bad_num_bins).fit(X).transform(X)
    else:
        with pytest.raises(ValueError):
            HOG1DTransformer(num_bins=bad_num_bins).fit(X).transform(X)


# Check that exception is raised for bad scaling factor.
# input types - string, float, negative float, negative int,
# empty dict and zero.
# correct input is meant to be any number (so the floats and
# ints shouldn't raise an error).
@pytest.mark.skipif(
    not run_test_for_class(HOG1DTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("bad_scaling_factor", ["str", 1.2, -1.2, -1, {}, 0])
def test_bad_scaling_factor(bad_scaling_factor):
    """Test that exception is raised for bad scaling factor."""
    X = _make_nested_from_array(np.ones(10), n_instances=10, n_columns=1)

    if not isinstance(bad_scaling_factor, numbers.Number):
        with pytest.raises(TypeError):
            HOG1DTransformer(scaling_factor=bad_scaling_factor).fit(X).transform(X)
    else:
        HOG1DTransformer(scaling_factor=bad_scaling_factor).fit(X).transform(X)


@pytest.mark.skipif(
    not run_test_for_class(HOG1DTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_output_of_transformer():
    """Test that the transformer has changed the data correctly."""
    X = _make_nested_from_array(
        np.array([4, 6, 10, 12, 8, 6, 5, 5]), n_instances=1, n_columns=1
    )

    h = HOG1DTransformer().fit(X)
    res = h.transform(X)
    orig = convert_list_to_dataframe([[0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0]])
    orig.columns = X.columns
    assert check_if_dataframes_are_equal(res, orig)

    X = _make_nested_from_array(
        np.array([-5, 2.5, 1, 3, 10, -1.5, 6, 12, -3, 0.2]), n_instances=1, n_columns=1
    )
    h = h.fit(X)
    res = h.transform(X)
    orig = convert_list_to_dataframe([[0, 0, 0, 0, 4, 1, 0, 0, 0, 0, 2, 0, 2, 1, 0, 0]])
    orig.columns = X.columns
    assert check_if_dataframes_are_equal(res, orig)


# (num_intervals is 2 by default)
@pytest.mark.skipif(
    not run_test_for_class(HOG1DTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("num_bins,corr_series_length", [(4, 8), (8, 16), (12, 24)])
def test_output_dimensions(num_bins, corr_series_length):
    """Test that the time series length should always be num_bins*num_intervals."""
    X = _make_nested_from_array(np.ones(13), n_instances=10, n_columns=1)

    h = HOG1DTransformer(num_bins=num_bins).fit(X)
    res = h.transform(X)

    # get the dimension of the generated dataframe.
    act_time_series_length = res.iloc[0, 0].shape[0]
    num_rows = res.shape[0]
    num_cols = res.shape[1]

    assert act_time_series_length == corr_series_length
    assert num_rows == 10
    assert num_cols == 1


@pytest.mark.skipif(
    not run_test_for_class(HOG1DTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_hog1d_performs_correctly_along_each_dim():
    """Test that HOG1D produces the same result along each dimension."""
    X = _make_nested_from_array(
        np.array([4, 6, 10, 12, 8, 6, 5, 5]), n_instances=1, n_columns=2
    )

    h = HOG1DTransformer().fit(X)
    res = h.transform(X)
    orig = convert_list_to_dataframe(
        [
            [0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0],
            [0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0],
        ]
    )
    orig.columns = X.columns
    assert check_if_dataframes_are_equal(res, orig)


def _hog1d_reference(X, num_bins, scaling_factor):
    """HOG1D of one interval, element by element, as in the original algorithm."""
    import math

    X = np.pad(np.asarray(X, dtype=float), 1, mode="edge")
    gradients = [
        scaling_factor * 0.5 * (X[i + 1] - X[i - 1]) for i in range(1, len(X) - 1)
    ]
    orients = [math.degrees(math.atan(g)) for g in gradients]
    boundaries = [
        -90 + (180 / num_bins) + ((180 / num_bins) * x) for x in range(num_bins)
    ]
    histogram = [0.0] * num_bins
    for orient in orients:
        for y, boundary in enumerate(boundaries):
            if orient <= boundary:
                histogram[y] += 1.0
                break
    return histogram


@pytest.mark.skipif(
    not run_test_for_class(HOG1DTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("num_bins", [2, 8, 12])
@pytest.mark.parametrize("scaling_factor", [0.1, 1.0, 10.0])
def test_get_hog1d_matches_reference(num_bins, scaling_factor):
    """Vectorized histogram of gradients matches the element-wise reference."""
    rng = np.random.default_rng(0)
    transformer = HOG1DTransformer(num_bins=num_bins, scaling_factor=scaling_factor)

    series = [
        rng.standard_normal(50),
        rng.standard_normal(7) * 100,
        np.zeros(10),  # all gradients zero, orientation exactly zero
        np.arange(10.0),  # constant gradient, orientation on a bin boundary for 1.0
        np.array([5.0]),  # single element
        np.array([1e12, -1e12] * 5),  # orientations at +-90 degrees
    ]
    for X in series:
        expected = _hog1d_reference(X, num_bins, scaling_factor)
        assert transformer._get_hog1d(X) == expected
        assert sum(expected) == len(X)


def convert_list_to_dataframe(list_to_convert):
    """Convert a Python list to a Pandas dataframe."""
    df = pd.DataFrame()
    for i in range(len(list_to_convert)):
        inst = list_to_convert[i]
        data = []
        data.append(pd.Series(inst))
        df[i] = data

    return df


def check_if_dataframes_are_equal(df1, df2):
    """Check that pandas DataFrames are equal."""
    from pandas.testing import assert_frame_equal

    try:
        assert_frame_equal(df1, df2)
        return True
    except AssertionError:
        return False
