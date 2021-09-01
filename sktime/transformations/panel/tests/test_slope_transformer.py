# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pytest
import math
from sktime.transformations.panel.slope import SlopeTransformer
from sktime.utils._testing.panel import _make_nested_from_array


# Check that exception is raised for bad num levels.
# input types - string, float, negative int, negative float, empty dict.
# correct input is meant to be a positive integer of 1 or more.
@pytest.mark.parametrize("bad_num_intervals", ["str", 1.2, -1.2, -1, {}, 0])
def test_bad_input_args(bad_num_intervals):
    X = _make_nested_from_array(np.ones(10), n_instances=10, n_columns=1)

    if not isinstance(bad_num_intervals, int):
        with pytest.raises(TypeError):
            SlopeTransformer(num_intervals=bad_num_intervals).fit(X).transform(X)
    else:
        with pytest.raises(ValueError):
            SlopeTransformer(num_intervals=bad_num_intervals).fit(X).transform(X)


# Check the transformer has changed the data correctly.
def test_output_of_transformer():

    X = _make_nested_from_array(
        np.array([4, 6, 10, 12, 8, 6, 5, 5]), n_instances=1, n_columns=1
    )

    s = SlopeTransformer(num_intervals=2).fit(X)
    res = s.transform(X)
    orig = convert_list_to_dataframe(
        [[(5 + math.sqrt(41)) / 4, (1 + math.sqrt(101)) / -10]]
    )
    orig.columns = X.columns
    assert check_if_dataframes_are_equal(res, orig)

    X = _make_nested_from_array(
        np.array([-5, 2.5, 1, 3, 10, -1.5, 6, 12, -3, 0.2]), n_instances=1, n_columns=1
    )
    s = s.fit(X)
    res = s.transform(X)
    orig = convert_list_to_dataframe(
        [
            [
                (104.8 + math.sqrt(14704.04)) / 61,
                (143.752 + math.sqrt(20790.0775)) / -11.2,
            ]
        ]
    )
    orig.columns = X.columns
    assert check_if_dataframes_are_equal(res, orig)


@pytest.mark.parametrize("num_intervals,corr_series_length", [(2, 2), (5, 5), (8, 8)])
def test_output_dimensions(num_intervals, corr_series_length):

    X = _make_nested_from_array(np.ones(13), n_instances=10, n_columns=1)

    s = SlopeTransformer(num_intervals=num_intervals).fit(X)
    res = s.transform(X)

    # get the dimension of the generated dataframe.
    act_time_series_length = res.iloc[0, 0].shape[0]
    num_rows = res.shape[0]
    num_cols = res.shape[1]

    assert act_time_series_length == corr_series_length
    assert num_rows == 10
    assert num_cols == 1


# This is to check that Slope produces the same result along each dimension
def test_slope_performs_correcly_along_each_dim():

    X = _make_nested_from_array(
        np.array([4, 6, 10, 12, 8, 6, 5, 5]), n_instances=1, n_columns=2
    )

    s = SlopeTransformer(num_intervals=2).fit(X)
    res = s.transform(X)
    orig = convert_list_to_dataframe(
        [
            [(5 + math.sqrt(41)) / 4, (1 + math.sqrt(101)) / -10],
            [(5 + math.sqrt(41)) / 4, (1 + math.sqrt(101)) / -10],
        ]
    )
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
