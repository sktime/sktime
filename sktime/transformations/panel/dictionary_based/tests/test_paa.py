# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pytest
from sktime.transformations.panel.dictionary_based._paa import PAA
from sktime.utils._testing.panel import _make_nested_from_array


# Check that exception is raised for bad num intervals.
# input types - string, float, negative int, negative float, empty dict
# and an int that is larger than the time series length.
# correct input is meant to be a positive integer of 1 or more.
@pytest.mark.parametrize("bad_num_intervals", ["str", 1.2, -1.2, -1, {}, 11, 0])
def test_bad_input_args(bad_num_intervals):
    X = _make_nested_from_array(np.ones(10), n_instances=10, n_columns=1)

    if not isinstance(bad_num_intervals, int):
        with pytest.raises(TypeError):
            PAA(num_intervals=bad_num_intervals).fit(X).transform(X)
    else:
        with pytest.raises(ValueError):
            PAA(num_intervals=bad_num_intervals).fit(X).transform(X)


# Check the transformer has changed the data correctly.
def test_output_of_transformer():
    X = _make_nested_from_array(
        np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), n_instances=1, n_columns=1
    )

    p = PAA(num_intervals=3).fit(X)
    res = p.transform(X)
    orig = convert_list_to_dataframe([[2.2, 5.5, 8.8]])
    orig.columns = X.columns
    assert check_if_dataframes_are_equal(res, orig)


def test_output_dimensions():
    # test with univariate
    X = _make_nested_from_array(np.ones(12), n_instances=10, n_columns=1)

    p = PAA(num_intervals=5).fit(X)
    res = p.transform(X)

    # get the dimension of the generated dataframe.
    corr_time_series_length = res.iloc[0, 0].shape[0]
    num_rows = res.shape[0]
    num_cols = res.shape[1]

    assert corr_time_series_length == 5
    assert num_rows == 10
    assert num_cols == 1

    # test with multivariate
    X = _make_nested_from_array(np.ones(12), n_instances=10, n_columns=5)

    p = PAA(num_intervals=5).fit(X)
    res = p.transform(X)

    # get the dimension of the generated dataframe.
    corr_time_series_length = res.iloc[0, 0].shape[0]
    num_rows = res.shape[0]
    num_cols = res.shape[1]

    assert corr_time_series_length == 5
    assert num_rows == 10
    assert num_cols == 5


# This is to check that PAA produces the same result along each dimension
def test_paa_performs_correcly_along_each_dim():
    X = _make_nested_from_array(
        np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), n_instances=1, n_columns=2
    )

    p = PAA(num_intervals=3).fit(X)
    res = p.transform(X)
    orig = convert_list_to_dataframe([[2.2, 5.5, 8.8], [2.2, 5.5, 8.8]])
    orig.columns = X.columns
    assert check_if_dataframes_are_equal(res, orig)


def convert_list_to_dataframe(list_to_convert):
    # Convert this into a panda's data frame
    df = pd.DataFrame()
    for i in range(len(list_to_convert)):
        inst = list_to_convert[i]
        data = []
        data.append(pd.Series(inst))
        df[i] = data
    return df


def check_if_dataframes_are_equal(df1, df2):
    """
    for some reason, this is how you check that two dataframes are equal.
    """
    from pandas.testing import assert_frame_equal

    try:
        assert_frame_equal(df1, df2)
        return True
    except AssertionError:
        return False
