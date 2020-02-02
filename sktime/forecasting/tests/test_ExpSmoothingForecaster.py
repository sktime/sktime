import pytest
import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal

from sktime.forecasting import ExpSmoothingForecaster
from sktime.datasets import load_shampoo_sales
from sktime.performance_metrics.forecasting import mase_loss
from sktime.utils.validation.forecasting import validate_fh

__author__ = ["Markus Löning", "@big-o"]


# forecast horizons
FHS = ([1], [1, 3], np.array([1]), np.array([1, 3]), np.arange(5) + 1)

# load test data
y = load_shampoo_sales()


# test default forecasters output for different forecasters horizons
@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.parametrize("fh", FHS)
def test_fhs(fh):
    m = ExpSmoothingForecaster()

    m.fit(y, fh=fh)
    y_pred = m.predict(fh=fh)

    # adjust for default value
    fh = validate_fh(fh)

    # test length of output
    assert len(y_pred) == len(fh)

    # test index
    assert_array_equal(y_pred.index.values, y.index[-1] + fh)


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


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_score():
    m = ExpSmoothingForecaster()
    y_train = y.iloc[:30]
    y_test = y.iloc[30:]
    fh = np.arange(len(y_test)) + 1

    m.fit(y_train, fh=fh)
    y_pred = m.predict(fh=fh)

    expected = -mase_loss(y_pred, y_test, y_train)
    assert m.score(y_test, y_train, fh=fh) == expected
