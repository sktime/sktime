# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pytest
import math
from sktime.transformations.panel.dwt import DWTTransformer
from sktime.utils._testing.panel import _make_nested_from_array


# Check that exception is raised for bad num levels.
# input types - string, float, negative int, negative float, empty dict.
# correct input is meant to be a positive integer of 0 or more.
@pytest.mark.parametrize("bad_num_levels", ["str", 1.2, -1.2, -1, {}])
def test_bad_input_args(bad_num_levels):
    X = _make_nested_from_array(np.ones(10), n_instances=10, n_columns=1)

    if not isinstance(bad_num_levels, int):
        with pytest.raises(TypeError):
            DWTTransformer(num_levels=bad_num_levels).fit(X).transform(X)
    else:
        with pytest.raises(ValueError):
            DWTTransformer(num_levels=bad_num_levels).fit(X).transform(X)


# Check the transformer has changed the data correctly.
def test_output_of_transformer():

    X = _make_nested_from_array(
        np.array([4, 6, 10, 12, 8, 6, 5, 5]), n_instances=1, n_columns=1
    )

    d = DWTTransformer(num_levels=2).fit(X)
    res = d.transform(X)
    orig = convert_list_to_dataframe(
        [[16, 12, -6, 2, -math.sqrt(2), -math.sqrt(2), math.sqrt(2), 0]]
    )
    orig.columns = X.columns
    assert check_if_dataframes_are_equal(res, orig)

    X = _make_nested_from_array(
        np.array([-5, 2.5, 1, 3, 10, -1.5, 6, 12, -3]), n_instances=1, n_columns=1
    )
    d = d.fit(X)
    res = d.transform(X)
    orig = convert_list_to_dataframe(
        [
            [
                0.75000,
                13.25000,
                -3.25000,
                -4.75000,
                -5.303301,
                -1.414214,
                8.131728,
                -4.242641,
            ]
        ]
    )
    # These are equivalent but cannot exactly test if two floats are equal
    # res.iloc[0,0]
    # orig.iloc[0,0]
    # assert check_if_dataframes_are_equal(res,orig)


# This is to test that if num_levels = 0 then no change occurs.
def test_no_levels_does_no_change():

    X = _make_nested_from_array(
        np.array([1, 2, 3, 4, 5, 56]), n_instances=1, n_columns=1
    )
    d = DWTTransformer(num_levels=0).fit(X)
    res = d.transform(X)
    assert check_if_dataframes_are_equal(res, X)


@pytest.mark.parametrize("num_levels,corr_series_length", [(2, 12), (3, 11), (4, 12)])
def test_output_dimensions(num_levels, corr_series_length):

    X = _make_nested_from_array(np.ones(13), n_instances=10, n_columns=1)

    d = DWTTransformer(num_levels=num_levels).fit(X)
    res = d.transform(X)

    # get the dimension of the generated dataframe.
    act_time_series_length = res.iloc[0, 0].shape[0]
    num_rows = res.shape[0]
    num_cols = res.shape[1]

    assert act_time_series_length == corr_series_length
    assert num_rows == 10
    assert num_cols == 1


# This is to check that DWT produces the same result along each dimension
def test_dwt_performs_correcly_along_each_dim():

    X = _make_nested_from_array(
        np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), n_instances=1, n_columns=2
    )

    d = DWTTransformer(num_levels=3).fit(X)
    res = d.transform(X)
    orig = convert_list_to_dataframe(
        [
            [
                9 * math.sqrt(2),
                -4 * math.sqrt(2),
                -2,
                -2,
                -math.sqrt(2) / 2,
                -math.sqrt(2) / 2,
                -math.sqrt(2) / 2,
                -math.sqrt(2) / 2,
                -math.sqrt(2) / 2,
            ],
            [
                9 * math.sqrt(2),
                -4 * math.sqrt(2),
                -2,
                -2,
                -math.sqrt(2) / 2,
                -math.sqrt(2) / 2,
                -math.sqrt(2) / 2,
                -math.sqrt(2) / 2,
                -math.sqrt(2) / 2,
            ],
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
