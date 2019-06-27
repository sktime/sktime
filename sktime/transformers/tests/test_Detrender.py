
import pytest
import numpy as np

from sktime.utils.transformations import remove_trend, add_trend, tabularise
from sktime.datasets import load_gunpoint
from sktime.datasets import load_shampoo_sales
from sktime.transformers.series_to_series import Detrender

gunpoint, _ = load_gunpoint(return_X_y=True)
shampoo = load_shampoo_sales(return_y_as_dataframe=True)


@pytest.mark.parametrize("data", [gunpoint, shampoo])
@pytest.mark.parametrize("order", [0, 1, 2, 3, 4])
def test_Detrender(data, order):

    X = data
    tran = Detrender(order=order)
    Xt = tran.fit_transform(X)
    Xit = tran.inverse_transform(Xt)

    np.testing.assert_array_almost_equal(tabularise(X), tabularise(Xit), decimal=10)
