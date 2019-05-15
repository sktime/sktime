import pytest
import numpy as np
from numpy.testing import assert_array_equal

from sktime.forecasters import DummyForecaster
from sktime.forecasters import ExpSmoothingForecaster
from sktime.forecasters import ARIMAForecaster
from sktime.forecasters import EnsembleForecaster
from sktime.datasets import load_shampoo_sales

__author__ = "Markus Löning"


# forecasters
FORECASTERS = (DummyForecaster, ExpSmoothingForecaster, ARIMAForecaster)

# forecast horizons
FHS = (None, [1], [1, 3], np.array([1]), np.array([1, 3]), np.arange(5))

# load test data
y = load_shampoo_sales()


# test default forecasters output for different forecasting horizons
@pytest.mark.parametrize("fh", FHS)
def test_EnsembleForecaster_fhs(fh):
    estimators = [
        ('ses', ExpSmoothingForecaster()),
        ('last', DummyForecaster(strategy='last'))
    ]
    m = EnsembleForecaster(estimators=estimators)
    m.fit(y)
    y_pred = m.predict(fh=fh)

    # adjust for default value
    if fh is None:
        fh = np.array([1])
    if isinstance(fh, list):
        fh = np.asarray(fh)

    # test length of output
    assert len(y_pred) == len(fh)

    # test index
    assert_array_equal(y_pred.index.values, y.iloc[0].index[-1] + fh)


