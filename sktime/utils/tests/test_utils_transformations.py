import numpy as np
import pandas as pd
import pytest

from sktime.utils.testing import generate_df_from_array
from sktime.utils.transformations import remove_trend, add_trend
from sktime.utils.transformations import tabularize


def test_tabularize():
    n_obs_X = 20
    n_cols_X = 3
    X = generate_df_from_array(np.random.normal(size=n_obs_X), n_rows=10, n_cols=n_cols_X)

    # Test single series input.
    Xt = tabularize(X.iloc[:, 0], return_array=True)
    assert Xt.shape[0] == X.shape[0]
    assert Xt.shape[1] == n_obs_X

    Xt = tabularize(X.iloc[:, 0])
    assert Xt.index.equals(X.index)

    # Test dataframe input with columns having series of different length.
    n_obs_Y = 13
    n_cols_Y = 2
    Y = generate_df_from_array(np.random.normal(size=n_obs_Y), n_rows=10, n_cols=n_cols_Y)
    X = pd.concat([X, Y], axis=1)

    Xt = tabularize(X, return_array=True)
    assert Xt.shape[0] == X.shape[0]
    assert Xt.shape[1] == (n_cols_X * n_obs_X) + (n_cols_Y * n_obs_Y)

    Xt = tabularize(X)
    assert Xt.index.equals(X.index)


def test_remove_trend_mean():
    # generate random data
    x = np.random.normal(size=(10, 5))
    m = x.mean(axis=1).reshape(-1, 1)
    expected = x - m

    # transform, remove trend
    actual, theta = remove_trend(x, order=0, axis=1)

    # check results
    np.testing.assert_array_equal(theta, m)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("order", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("axis", [0, 1])
def test_add_remove_trend(order, axis):
    # generate random data
    x = np.random.normal(size=(10, 5))

    # transform, remove trend
    xt, theta = remove_trend(x, order=order, axis=axis)

    # inverse transform, add trend
    xit = add_trend(xt, theta, axis=axis)

    # check if original data and inverse transform data are the same
    np.testing.assert_array_almost_equal(x, xit)
