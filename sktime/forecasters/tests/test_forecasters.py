import pytest
import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal
from sklearn.metrics import mean_squared_error

from sktime.forecasters.forecasters import DummyForecaster
from sktime.forecasters.forecasters import ExpSmoothingForecaster
from sktime.forecasters.forecasters import ARIMAForecaster
from sktime.datasets import load_shampoo_sales

__author__ = "Markus Löning"


# forecasters
FORECASTERS = (DummyForecaster, ExpSmoothingForecaster, ARIMAForecaster)
FORECASTER_PARAMS = {DummyForecaster: {"strategy": "last"},
                     ExpSmoothingForecaster: {"trend": "additive"},
                     ARIMAForecaster: {"order": (1, 1, 0)}}

# forecast horizons
FHS = (None, [1], [1, 3], np.array([1]), np.array([1, 3]), np.arange(5))

# load test data
y = load_shampoo_sales()


# test default forecasters output for different forecasting horizons
@pytest.mark.filterwarnings('ignore::FutureWarning')
@pytest.mark.parametrize("forecaster", FORECASTERS)
@pytest.mark.parametrize("fh", FHS)
def test_fhs(forecaster, fh):
    m = forecaster()

    m.fit(y)
    y_pred = m.predict(fh=fh)

    # adjust for default value
    if fh is None:
        fh = np.array([1])
    elif isinstance(fh, list):
        fh = np.asarray(fh)

    # test length of output
    assert len(y_pred) == len(fh)

    # test index
    assert_array_equal(y_pred.index.values, y.iloc[0].index[-1] + fh)


@pytest.mark.filterwarnings('ignore::FutureWarning')
@pytest.mark.parametrize("forecaster, params", FORECASTER_PARAMS.items())
def test_set_params(forecaster, params):
    m = forecaster(**params)
    m.fit(y)
    expected = m.predict()

    m = forecaster()
    m.set_params(**params)
    m.fit(y)
    y_pred = m.predict()

    assert_array_equal(y_pred, expected)


@pytest.mark.filterwarnings('ignore::FutureWarning')
@pytest.mark.parametrize("forecaster", FORECASTERS)
@pytest.mark.parametrize("fh", FHS)
def test_score(forecaster, fh):
    m = forecaster()
    train = pd.Series([y.iloc[0].iloc[:30]])
    test = pd.Series([y.iloc[0].iloc[30:]])
    fh = np.arange(len(test.iloc[0])) + 1
    m.fit(train)
    y_pred = m.predict(fh=fh)
    expected = np.sqrt(mean_squared_error(y_pred.values, test.iloc[0].values))
    assert m.score(test, fh=fh) == expected
