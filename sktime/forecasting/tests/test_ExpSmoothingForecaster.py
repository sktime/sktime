import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sktime.datasets import load_shampoo_sales
from sktime.forecasting import ExpSmoothingForecaster

__author__ = ["Markus Löning", "@big-o"]

# forecast horizons
FHS = ([1], [1, 3], np.array([1]), np.array([1, 3]), np.arange(5) + 1)

# load test data
y = load_shampoo_sales()


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_set_params():
    params = {"trend": "additive"}

    m = ExpSmoothingForecaster(**params)
    m.fit(y, fh=1)
    expected = m.predict()

    m = ExpSmoothingForecaster()
    m.set_params(**params)
    m.fit(y, fh=1)
    y_pred = m.predict()

    assert_array_equal(y_pred, expected)
