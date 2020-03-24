__author__ = ["@big-o"]

import numpy as np
import pandas as pd
import pytest
from sktime.datasets import load_airline
from sktime.forecasting import ThetaForecaster
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.tests import TEST_OOS_FHS, TEST_ALPHAS
from sktime.utils.testing.forecasting import make_forecasting_problem
from sktime.utils.validation.forecasting import check_fh


def test_predictive_performance_on_airline():
    y = np.log1p(load_airline())
    y_train, y_test = temporal_train_test_split(y)
    fh = np.arange(len(y_test)) + 1

    f = ThetaForecaster(sp=12)
    f.fit(y_train)
    y_pred = f.predict(fh=fh)

    # Performance on this particular dataset should be reasonably good.
    np.testing.assert_allclose(y_pred, y_test, rtol=0.05)


@pytest.mark.parametrize("fh", TEST_OOS_FHS)
def test_pred_errors_against_y_test(fh):
    y_train, y_test = make_forecasting_problem()
    f = ThetaForecaster()
    f.fit(y_train, fh)
    y_pred = f.predict(return_pred_int=False)
    errors = f._compute_pred_errors(alpha=0.1)
    if isinstance(errors, pd.Series):
        errors = [errors]
    y_test = y_test.iloc[check_fh(fh) - 1]
    for error in errors:
        assert np.all(y_pred - error < y_test)
        assert np.all(y_test < y_pred + error)
