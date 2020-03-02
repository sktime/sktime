__author__ = ["Markus Löning", "@big-o"]
__all__ = ["test_set_params"]

import pytest
from numpy.testing import assert_array_equal
from sktime.forecasting import ExponentialSmoothingForecaster
from sktime.utils.testing.forecasting import make_forecasting_problem

# load test data
y_train, y_test = make_forecasting_problem()


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_set_params():
    params = {"trend": "additive"}

    f = ExponentialSmoothingForecaster(**params)
    f.fit(y_train, fh=1)
    expected = f.predict()

    f = ExponentialSmoothingForecaster()
    f.set_params(**params)
    f.fit(y_train, fh=1)
    y_pred = f.predict()

    assert_array_equal(y_pred, expected)
