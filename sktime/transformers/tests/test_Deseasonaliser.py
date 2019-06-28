import numpy as np
from sktime.datasets import load_gunpoint
from sktime.datasets import load_shampoo_sales
from sktime.utils.transformations import tabularize, detabularize
from sktime.transformers.series_to_series import Deseasonaliser
import pytest

gunpoint, _ = load_gunpoint(return_X_y=True)
shampoo = load_shampoo_sales(return_y_as_dataframe=True)


@pytest.mark.parametrize("X", [gunpoint, shampoo])
@pytest.mark.parametrize("freq", [1, 4, 7, 12, 24])
def test_Deseasonaliser(X, freq):

    # ensure only positive time series values
    X = detabularize(tabularize(X) + 100)

    t = Deseasonaliser(freq=freq)
    Xt = t.fit_transform(X)
    assert Xt.shape == X.shape

    Xit = t.inverse_transform(Xt)
    np.testing.assert_array_almost_equal(tabularize(X), tabularize(Xit))
