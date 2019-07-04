import pytest
import numpy as np
import pandas as pd

from sktime.utils.transformations import tabularise, select_times
from sktime.utils.testing import generate_time_series_data_with_trend
from sktime.datasets import load_gunpoint
from sktime.transformers.series_to_series import Detrender


@pytest.mark.parametrize("order", [0, 1, 2])
def test_transform_inverse_transform_equivalence(order):
    X, _ = load_gunpoint(return_X_y=True)
    X = X.sample(10)
    tran = Detrender(order=order)
    Xt = tran.fit_transform(X)
    assert X.shape == Xt.shape

    Xit = tran.inverse_transform(Xt)
    np.testing.assert_array_almost_equal(tabularise(X), tabularise(Xit))


@pytest.mark.parametrize("order", [0, 1, 2])
@pytest.mark.parametrize("n_samples", [1, 10])
def test_transform_inverse_transform_equivalence(n_samples, order):
    # generate data
    n_obs = 100
    X = generate_time_series_data_with_trend(n_samples=n_samples, n_obs=n_obs, order=order)

    # split data for testing
    cutoff = n_obs - (n_obs // 4)
    a_times = np.arange(n_obs)[:cutoff]
    b_times = np.arange(n_obs)[cutoff:]

    A = select_times(X, a_times)
    B = select_times(X, b_times)

    # test successful de-trending when true order of trend is given
    tran = Detrender(order=order)
    At = tran.fit_transform(A)
    np.testing.assert_array_almost_equal(At.iloc[0, 0].values, np.zeros(cutoff))

    # test inverse transform restores original series
    Ait = tran.inverse_transform(At)
    np.testing.assert_array_almost_equal(Ait.iloc[0, 0].values, A.iloc[0, 0].values)

    # test correct inverse transform on new data with a different time index
    # e.g. necessary for inverse transforms after predicting/forecasting
    c = pd.Series(np.zeros(n_obs - cutoff), index=b_times)
    C = pd.DataFrame(pd.Series([c]))
    Cit = tran.inverse_transform(C)
    np.testing.assert_array_almost_equal(B.iloc[0, 0].values, Cit.iloc[0, 0].values)

