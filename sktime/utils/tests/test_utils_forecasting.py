import numpy as np
import pandas as pd
import pytest
from sktime.utils.data_container import tabularize
from sktime.utils._testing import generate_df_from_array
from sktime.utils._testing.forecasting import generate_polynomial_series
from sktime.utils.time_series import add_trend
from sktime.utils.time_series import fit_trend
from sktime.utils.time_series import remove_trend


def test_tabularize():
    n_obs_X = 20
    n_cols_X = 3
    X = generate_df_from_array(np.random.normal(size=n_obs_X), n_rows=10,
                               n_cols=n_cols_X)

    # Test single series input.
    Xt = tabularize(X.iloc[:, 0], return_array=True)
    assert Xt.shape[0] == X.shape[0]
    assert Xt.shape[1] == n_obs_X

    Xt = tabularize(X.iloc[:, 0])
    assert Xt.index.equals(X.index)

    # Test dataframe input with columns having series of different length.
    n_obs_Y = 13
    n_cols_Y = 2
    Y = generate_df_from_array(np.random.normal(size=n_obs_Y), n_rows=10,
                               n_cols=n_cols_Y)
    X = pd.concat([X, Y], axis=1)

    Xt = tabularize(X, return_array=True)
    assert Xt.shape[0] == X.shape[0]
    assert Xt.shape[1] == (n_cols_X * n_obs_X) + (n_cols_Y * n_obs_Y)

    Xt = tabularize(X)
    assert Xt.index.equals(X.index)


@pytest.mark.parametrize("order", [0, 1, 2])  # polynomial order
@pytest.mark.parametrize("n_obs",
                         [1, 10, 20])  # number of time series observations
@pytest.mark.parametrize("n_samples", [1, 10, 20])  # number of samples
def test_fit_remove_add_trend(order, n_samples, n_obs):
    # generate random polynomial series data
    coefs = np.random.normal(size=order + 1).reshape(-1, 1)
    x = np.column_stack([generate_polynomial_series(n_obs, order, coefs=coefs)
                         for _ in range(n_samples)]).T
    # assert x.shape == (n_samples, n_obs)

    # check shape of fitted coefficients
    coefs = fit_trend(x, order=order)
    assert coefs.shape == (n_samples, order + 1)

    # test if trend if properly remove when given true order
    xt = remove_trend(x, coefs)
    np.testing.assert_array_almost_equal(xt, np.zeros(x.shape))

    # test inverse transform restores original series
    xit = add_trend(xt, coefs=coefs)
    np.testing.assert_array_almost_equal(x, xit)
