"""HOG1D transformer test code."""

import numbers

import numpy as np
import pandas as pd
import pytest

from sktime.transformations.panel.hog1d import HOG1DTransformer
from sktime.utils._testing.panel import _make_nested_from_array


# Check that exception is raised for bad num intervals.
# input types - string, float, negative int, negative float, empty dict
# and an int that is larger than the time series length.
# correct input is meant to be a positive integer of 1 or more.
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
@pytest.mark.parametrize("bad_scaling_factor", ["str", 1.2, -1.2, -1, {}, 0])
def test_bad_scaling_factor(bad_scaling_factor):
    """Test that exception is raised for bad scaling factor."""
    X = _make_nested_from_array(np.ones(10), n_instances=10, n_columns=1)

    if not isinstance(bad_scaling_factor, numbers.Number):
        with pytest.raises(TypeError):
            HOG1DTransformer(scaling_factor=bad_scaling_factor).fit(X).transform(X)
    else:
        HOG1DTransformer(scaling_factor=bad_scaling_factor).fit(X).transform(X)


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
