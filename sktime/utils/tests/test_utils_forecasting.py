import numpy as np
import pandas as pd
import pytest

from sktime.utils._testing.forecasting import generate_polynomial_series
from sktime.utils.time_series import add_trend
from sktime.utils.time_series import fit_trend
from sktime.utils.time_series import remove_trend


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
