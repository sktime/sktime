"""Tests for SlidingWindowSegmenter."""

__author__ = ["mloning"]


import numpy as np
import pandas as pd
import pytest

from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.panel.segment import SlidingWindowSegmenter
from sktime.utils._testing.panel import _make_nested_from_array


# Check that exception is raised for bad window length.
# input types - string, float, negative int, negative float and empty dict
# correct input is meant to be a positive integer of 1 or more.
@pytest.mark.skipif(
    not run_test_for_class(SlidingWindowSegmenter),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("bad_window_length", ["str", 1.2, -1.2, -1, {}])
def test_bad_input_args(bad_window_length):
    """Test that bad inputs raise value error."""
    X = _make_nested_from_array(np.ones(10), n_instances=10, n_columns=1)

    if not isinstance(bad_window_length, int):
        with pytest.raises(TypeError):
            SlidingWindowSegmenter(window_length=bad_window_length).fit(X).transform(X)
    else:
        with pytest.raises(ValueError):
            SlidingWindowSegmenter(window_length=bad_window_length).fit(X).transform(X)


# Check the transformer has changed the data correctly.
@pytest.mark.skipif(
    not run_test_for_class(SlidingWindowSegmenter),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_output_of_transformer():
    """Test correct output of SlidingWindowSegmenter."""
    X = _make_nested_from_array(
        np.array([1, 2, 3, 4, 5, 6]), n_instances=1, n_columns=1
    )

    st = SlidingWindowSegmenter(window_length=1).fit(X)
    res = st.transform(X)
    orig = convert_list_to_dataframe([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
    assert check_if_dataframes_are_equal(res, orig)

    st = SlidingWindowSegmenter(window_length=5).fit(X)
    res = st.transform(X)
    orig = convert_list_to_dataframe(
        [
            [1.0, 1.0, 1.0, 2.0, 3.0],
            [1.0, 1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [2.0, 3.0, 4.0, 5.0, 6.0],
            [3.0, 4.0, 5.0, 6.0, 6.0],
            [4.0, 5.0, 6.0, 6.0, 6.0],
        ]
    )

    assert check_if_dataframes_are_equal(res, orig)

    st = SlidingWindowSegmenter(window_length=10).fit(X)
    res = st.transform(X)
    orig = convert_list_to_dataframe(
        [
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 6.0],
            [1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 6.0, 6.0],
            [1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 6.0, 6.0, 6.0],
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 6.0, 6.0, 6.0, 6.0],
        ]
    )
    assert check_if_dataframes_are_equal(res, orig)


@pytest.mark.skipif(
    not run_test_for_class(SlidingWindowSegmenter),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "time_series_length,window_length", [(5, 1), (10, 5), (15, 9), (20, 13), (25, 19)]
)
def test_output_dimensions(time_series_length, window_length):
    """Test output dimensions of SlidingWindowSegmenter."""
    X = _make_nested_from_array(
        np.ones(time_series_length), n_instances=10, n_columns=1
    )

    st = SlidingWindowSegmenter(window_length=window_length).fit(X)
    res = st.transform(X)

    # get the dimension of the generated dataframe.
    corr_time_series_length = res.iloc[0, 0].shape[0]
    num_rows = res.shape[0]
    num_cols = res.shape[1]

    assert corr_time_series_length == window_length
    assert num_rows == 10
    assert num_cols == time_series_length


def convert_list_to_dataframe(list_to_convert):
    """Convert list into a pandas data frame."""
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
