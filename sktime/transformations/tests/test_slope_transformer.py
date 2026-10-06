"""Slope transformer test code."""

import math
import statistics

import numpy as np
import pandas as pd
import pytest

from sktime.datatypes import convert
from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.slope import SlopeTransformer
from sktime.utils._testing.panel import _make_nested_from_array


# Check that exception is raised for bad num levels.
# input types - string, float, negative int, negative float, empty dict.
# correct input is meant to be a positive integer of 1 or more.
@pytest.mark.skipif(
    not run_test_for_class(SlopeTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("bad_num_intervals", ["str", 1.2, -1.2, -1, {}, 0])
def test_bad_input_args(bad_num_intervals):
    """Test that exception is raised for bad num levels."""
    X = _make_nested_from_array(np.ones(10), n_instances=10, n_columns=1)

    if not isinstance(bad_num_intervals, int):
        with pytest.raises(TypeError):
            SlopeTransformer(num_intervals=bad_num_intervals).fit(X).transform(X)
    else:
        with pytest.raises(ValueError):
            SlopeTransformer(num_intervals=bad_num_intervals).fit(X).transform(X)


@pytest.mark.skipif(
    not run_test_for_class(SlopeTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_output_of_transformer():
    """Test that the transformer has changed the data correctly."""
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


@pytest.mark.skipif(
    not run_test_for_class(SlopeTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("num_intervals,corr_series_length", [(2, 2), (5, 5), (8, 8)])
def test_output_dimensions(num_intervals, corr_series_length):
    """Test output dimensions."""
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


@pytest.mark.skipif(
    not run_test_for_class(SlopeTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_slope_performs_correctly_along_each_dim():
    """Test that Slope produces the same result along each dimension."""
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


def _loop_reference(Y):
    """Reference implementation of TLS slope before vectorization."""
    X = [(i + 1) for i in range(len(Y))]
    meanX = statistics.mean(X)
    meanY = statistics.mean(Y)
    yminYbar = [(y - meanY) ** 2 for y in Y]
    xminXbar = [(x - meanX) ** 2 for x in X]
    w = sum(yminYbar) - sum(xminXbar)
    temp = []
    for x in range(len(X)):
        temp.append((X[x] - meanX) * (Y[x] - meanY))
    r = 2 * sum(temp)
    if r == 0:
        return 0.0
    return (w + math.sqrt(w**2 + r**2)) / r


def _loop_transform_reference(arr, num_intervals):
    """Reference loop transform on a 2D numpy array [n_instances, n_timepoints]."""
    avg = arr.shape[1] / float(num_intervals)
    output = []
    for row in arr:
        row_gradients = []
        beginning = 0.0
        while beginning < len(row):
            seg = row[int(beginning) : int(beginning + avg)]
            row_gradients.append(_loop_reference(seg))
            beginning += avg
        output.append(row_gradients)
    return np.asarray(output)


@pytest.mark.skipif(
    not run_test_for_class(SlopeTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("n_instances", [1, 5, 20])
@pytest.mark.parametrize("series_length", [8, 13, 25, 100])
@pytest.mark.parametrize("num_intervals", [2, 3, 5, 8])
def test_matches_loop_reference(n_instances, series_length, num_intervals):
    """Test vectorized SlopeTransformer matches the reference loop implementation."""
    rng = np.random.default_rng(42)
    arr = rng.normal(size=(n_instances, series_length))
    X_3d = arr[:, np.newaxis, :]
    X = convert(X_3d, from_type="numpy3D", to_type="nested_univ")

    transformer = SlopeTransformer(num_intervals=num_intervals)
    res = transformer.fit_transform(X)

    actual = np.asarray([res.iloc[i, 0].to_numpy() for i in range(n_instances)])
    expected = _loop_transform_reference(arr, num_intervals)

    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-7)


@pytest.mark.skipif(
    not run_test_for_class(SlopeTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_constant_series_zero_gradient():
    """Test constant series where r == 0 produces zero gradients without NaN."""
    arr = np.ones((5, 20))
    X_3d = arr[:, np.newaxis, :]
    X = convert(X_3d, from_type="numpy3D", to_type="nested_univ")

    transformer = SlopeTransformer(num_intervals=4)
    res = transformer.fit_transform(X)

    actual = np.asarray([res.iloc[i, 0].to_numpy() for i in range(5)])
    np.testing.assert_allclose(actual, 0.0, atol=1e-9)
