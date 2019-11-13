import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal

from sktime.forecasters import DummyForecaster
from sktime.datasets import load_shampoo_sales

__author__ = ["Markus Löning"]

# forecast horizons
FHS = ([1], [1, 3], np.array([1]), np.array([1, 3]), np.arange(1, 5))

# load test data
y = load_shampoo_sales()


# test strategies of dummy forecaster
# TODO add test for linear strategy
@pytest.mark.filterwarnings('ignore::FutureWarning')
@pytest.mark.parametrize("fh", FHS)
@pytest.mark.parametrize("strategy, expected", [("mean", y.mean()), ("last", y.iloc[-1])])
def test_DummyForecaster_strategies(fh, strategy, expected):
    m = DummyForecaster(strategy=strategy)
    m.fit(y, fh=fh)
    y_pred = m.predict()
    assert y_pred.shape == (len(fh),)
    assert_array_almost_equal(y_pred, expected, decimal=4)
