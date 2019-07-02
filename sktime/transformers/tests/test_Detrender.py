import pytest
import numpy as np
import pandas as pd

from sktime.utils.transformations import tabularise
from sktime.utils.testing import generate_polynomial_series
from sktime.datasets import load_gunpoint
from sktime.datasets import load_shampoo_sales
from sktime.transformers.series_to_series import Detrender

gunpoint, _ = load_gunpoint(return_X_y=True)
shampoo = load_shampoo_sales(return_y_as_dataframe=True)


@pytest.mark.parametrize("data", [gunpoint])
@pytest.mark.parametrize("order", [0, 1, 2])
def test_transform_inverse_transform_equivalence(data, order):
    X = data
    tran = Detrender(order=order)
    Xt = tran.fit_transform(X)
    assert X.shape == Xt.shape

    Xit = tran.inverse_transform(Xt)
    np.testing.assert_array_almost_equal(tabularise(X), tabularise(Xit))


@pytest.mark.parametrize("order", [0, 1, 2])
def test_inverse_transform_on_new_index(order):
    # generate data
    n_obs = 20
    coefs = np.random.normal(size=order + 1).reshape(-1, 1)
    x = generate_polynomial_series(n_obs, order, coefs=coefs).ravel()
    s = pd.Series(x)

    # split data for testing
    cutoff = n_obs - (n_obs // 2)
    a = s.iloc[:cutoff]
    b = s.iloc[cutoff:]

    A = pd.DataFrame(pd.Series([a]))
    B = pd.DataFrame(pd.Series([b]))

    # test successful de-trending when true order of trend is given
    tran = Detrender(order=order)
    At = tran.fit_transform(A)
    np.testing.assert_array_almost_equal(At.iloc[0, 0].values, np.zeros(cutoff))

    # test inverse transform restores original series
    Ait = tran.inverse_transform(At)
    np.testing.assert_array_almost_equal(Ait.iloc[0, 0].values, A.iloc[0, 0].values)

    # test correct inverse transform on new data with a different time index
    # e.g. necessary for inverse transforms after predicting/forecasting
    c = pd.Series(np.zeros(n_obs - cutoff), index=b.index)
    C = pd.DataFrame(pd.Series([c]))
    Cit = tran.inverse_transform(C)
    np.testing.assert_array_almost_equal(B.iloc[0, 0].values, Cit.iloc[0, 0].values)

