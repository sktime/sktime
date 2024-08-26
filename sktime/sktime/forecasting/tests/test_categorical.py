"""Tests for checking with categorical inputs

Cases where error must be raised are tested in test_all_forecasters in
- test_categorical_y_raises_error
- test_categorical_X_raises_error
"""

__author__ = ["Abhay-Lejith"]


import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

from sktime.forecasting.compose import SkforecastAutoreg
from sktime.forecasting.compose._reduce import YfromX
from sktime.forecasting.dummy import ForecastKnownValues
from sktime.tests.test_switch import run_test_for_class


def test_dummy_est_with_categorical_capability():
    """Test that categorical data works when native support is available.

    This test uses the dummy forecaster with modified tags to imitate a forecaster
    which supports categorical natively in exogeneous X for checking whether
    categorical data passes through the boilerplate checks without error.
    """
    y = pd.DataFrame(range(9))

    est = ForecastKnownValues(y)
    modified_tags = {
        "ignores-exogeneous-X": False,
        "capability:categorical_in_X": True,
    }
    est.set_tags(**modified_tags)

    yt = y[:6]
    X = pd.DataFrame({"col_0": ["a", "b", "c", "a", "b", "c", "a", "b", "c"]})
    Xt = X[:6]

    est.fit(yt, Xt, fh=[1, 2, 3])
    est.predict(X=X[6:])
    est.update(y[6:], X[6:])


def create_mixed_dtype_df():
    random_generator = np.random.default_rng(seed=0)
    sample_data = {
        "target_col": random_generator.standard_normal(size=50),
        "categorical": random_generator.choice(["a", "b", "c", "d"], size=50),
        "numeric_col": random_generator.standard_exponential(size=50),
    }
    df = pd.DataFrame(data=sample_data)
    y_train = df["target_col"][:40]
    X_train = df[["categorical", "numeric_col"]][:40]
    X_test = df[["categorical", "numeric_col"]][40:]

    return y_train, X_train, X_test


@pytest.mark.skipif(
    not run_test_for_class(SkforecastAutoreg),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_skforecast_with_categorical():
    y_train, X_train, X_test = create_mixed_dtype_df()

    regressor = HistGradientBoostingRegressor(categorical_features=["categorical"])
    forecaster = SkforecastAutoreg(regressor, 3)

    forecaster.fit(y_train, X_train)
    forecaster.predict(10, X_test)


def test_YfromX_with_categorical():
    y_train, X_train, X_test = create_mixed_dtype_df()

    regressor = HistGradientBoostingRegressor(categorical_features=["categorical"])
    forecaster = YfromX(regressor)

    forecaster.fit(y_train, X_train)
    forecaster.predict(10, X_test)
