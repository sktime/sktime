import pytest
import numpy as np
from numpy.testing import assert_array_equal

from sktime.forecasters import DummyForecaster, EnsembleForecaster
from sktime.forecasters import ExpSmoothingForecaster
from sktime.forecasters import ARIMAForecaster
from sktime.datasets import load_shampoo_sales
from sktime.utils.validation.forecasting import validate_fh

__author__ = "Markus Löning"


# forecasters
FORECASTERS = (DummyForecaster, ExpSmoothingForecaster, ARIMAForecaster)

# forecast horizons
FHS = (None, [1], [1, 3], np.array([1]), np.array([1, 3]), np.arange(5))

# load test data
y = load_shampoo_sales()

# test default forecasters output for different forecasters horizons
@pytest.mark.parametrize("fh", FHS)
def test_EnsembleForecaster_fhs(fh):
    estimators = [
        ('ses', ExpSmoothingForecaster()),
        ('last', DummyForecaster(strategy='last'))
    ]
    m = EnsembleForecaster(estimators=estimators)

    if fh is None:
        # Fit and predict with default fh
        m.fit(y)
        y_pred = m.predict()

        # for further checks, set fh to default
        fh = validate_fh(1)
    else:
        # Validate fh and then fit/predict
        fh = validate_fh(fh)
        m.fit(y, fh=fh)
        y_pred = m.predict(fh=fh)

    # test length of output
    assert len(y_pred) == len(fh)

    # test index
    assert_array_equal(y_pred.index.values, y.iloc[0].index[-1] + fh)


