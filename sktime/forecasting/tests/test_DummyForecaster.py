import numpy as np
import pytest
from sktime.forecasting.forecasters import DummyForecaster
from sktime.datasets import load_shampoo_sales

y = load_shampoo_sales()


# TODO add test for linear strategy
@pytest.mark.parametrize(
    "strategy, expected", [
        ("mean", y.iloc[0].mean()),
        ("last", y.iloc[0].iloc[-1])
    ])
def test_DummyForecaster_strategies(strategy, expected):
    fh = np.array([1, 2])
    m = DummyForecaster(strategy=strategy)
    m.fit(y)
    y_pred = m.predict(fh=fh)
    np.testing.assert_array_almost_equal(y_pred, expected, decimal=4)

