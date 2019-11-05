__author__ = "Markus Löning"

import pytest
import numpy as np
from numpy.testing import assert_array_equal
from sklearn.metrics import mean_squared_error

from sktime.forecasters import DummyForecaster
from sktime.forecasters import ExpSmoothingForecaster
from sktime.forecasters import ARIMAForecaster
from sktime.datasets import load_shampoo_sales
from sktime.utils.validation.forecasting import validate_fh
from sktime.forecasters.model_selection import temporal_train_test_split


# forecasters
FORECASTERS = (DummyForecaster, ExpSmoothingForecaster, ARIMAForecaster)
FORECASTER_PARAMS = {DummyForecaster: {"strategy": "last"},
                     ExpSmoothingForecaster: {"trend": "additive"},
                     ARIMAForecaster: {"order": (1, 1, 0)}}

# forecast horizons
FHS = (np.array([1]), np.array([1, 3]), np.arange(1, 4))

# load test data
y = load_shampoo_sales()


# test default forecasters output for different forecasters horizons
@pytest.mark.parametrize("forecaster", FORECASTERS)
@pytest.mark.parametrize("fh", FHS)
def test_fhs(forecaster, fh):
    m = forecaster()

    m.fit(y, fh=fh)
    y_pred = m.predict(fh=fh)

    # adjust for default value
    fh = validate_fh(fh)

    # test length of output
    assert len(y_pred) == len(fh)

    # test index
    assert_array_equal(y_pred.index.values, y.index[-1] + fh)


@pytest.mark.parametrize("forecaster, params", FORECASTER_PARAMS.items())
def test_set_params(forecaster, params):
    m = forecaster(**params)
    m.fit(y, fh=1)
    expected = m.predict()

    m = forecaster()
    m.set_params(**params)
    m.fit(y, fh=1)
    y_pred = m.predict()

    assert_array_equal(y_pred, expected)


@pytest.mark.parametrize("forecaster", FORECASTERS)
@pytest.mark.parametrize("fh", FHS)
def test_score(forecaster, fh):
    m = forecaster()
    y_train, y_test = temporal_train_test_split(y, fh)

    m.fit(y_train, fh=fh)
    y_pred = m.predict()
    expected = np.sqrt(mean_squared_error(y_pred.values, y_test.values))
    assert m.score(y_test, fh=fh) == expected
