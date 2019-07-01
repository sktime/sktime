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


@pytest.mark.parametrize("order", [0, 1, 2])
def test_remove_add_trend(order):

    # 2d input, multiple columns
    x = np.random.normal(size=(10, 5))

    axis = 0
    xt, theta = remove_trend(x, order=order, axis=axis)
    assert xt.shape == x.shape
    assert theta.shape == (xt.shape[1], order + 1)
    xit = add_trend(xt, theta=theta, axis=axis)
    np.testing.assert_array_almost_equal(x, xit)

    axis = 1
    xt, theta = remove_trend(x, order=order, axis=axis)
    assert xt.shape == x.shape
    assert theta.shape == (xt.shape[0], order + 1)
    xit = add_trend(xt, theta=theta, axis=axis)
    np.testing.assert_array_almost_equal(x, xit)

    # 2d input, single column
    x = np.random.normal(size=(10, 1))

    axis = 0
    xt, theta = remove_trend(x, order=order, axis=axis)
    assert xt.shape == x.shape
    assert theta.shape == (x.shape[1], order + 1)
    xit = add_trend(xt, theta=theta, axis=axis)
    np.testing.assert_array_almost_equal(x, xit)

    axis = 1
    xt, theta = remove_trend(x, order=order, axis=axis)
    assert xt.shape == x.shape
    assert theta.shape == (x.shape[0], order + 1)
    assert np.allclose(xt, 0)  # should be zero when demeaning single column array along columns
    xit = add_trend(xt, theta=theta, axis=axis)
    np.testing.assert_array_almost_equal(x, xit)

    # 1d input
    x = np.random.normal(size=(10,))

    axis = 0
    xt, theta = remove_trend(x, order=order, axis=axis)
    assert xt.shape == x.shape
    assert theta.shape == (1, order + 1)
    xit = add_trend(xt, theta=theta, axis=axis)
    np.testing.assert_array_almost_equal(x, xit)

    # using axis 1 on 1d array should raise error
    axis = 1
    with pytest.raises(IndexError):
        xt, theta = remove_trend(x, order=order, axis=axis)

    # axis >= 2 should raise error, only 1d and 2d arrays are supported
    axis = 2
    with pytest.raises(IndexError):
        xt, theta = remove_trend(x, order=order, axis=axis)
