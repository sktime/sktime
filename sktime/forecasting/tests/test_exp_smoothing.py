"""Test exponential smoothing forecasters."""

__author__ = ["mloning", "big-o", "ciaran-g"]
__all__ = ["test_set_params"]

import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from sktime.datasets import load_airline
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.split import temporal_train_test_split
from sktime.tests.test_switch import run_test_for_class
from sktime.utils._testing.forecasting import make_forecasting_problem

# load test data
y = make_forecasting_problem()
y_train, y_test = temporal_train_test_split(y, train_size=0.75)


@pytest.mark.skipif(
    not run_test_for_class(ExponentialSmoothing),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_set_params():
    """Test set_params."""
    params = {"trend": "additive"}

    f = ExponentialSmoothing(**params)
    f.fit(y_train, fh=1)
    expected = f.predict()

    f = ExponentialSmoothing()
    f.set_params(**params)
    f.fit(y_train, fh=1)
    y_pred = f.predict()

    assert_array_equal(y_pred, expected)


@pytest.mark.skipif(
    not run_test_for_class(ExponentialSmoothing),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def check_panel_expsmooth():
    """Test exponential smoothing on panel data with datetime index."""
    # make panel with hour of day panel and datetime index
    y = load_airline()
    y.index = pd.date_range(start="1960-01-01", periods=len(y.index), freq="H")
    y.index.names = ["datetime"]
    y.name = "passengers"
    y = y.to_frame()
    y["hour_of_day"] = y.index.hour
    y = y.reset_index().set_index(["hour_of_day", "datetime"]).sort_index()

    forecaster = ExponentialSmoothing(trend="add", sp=1)
    forecaster.fit(y)
    forecaster.predict(fh=[1, 3])


@pytest.mark.skipif(
    not run_test_for_class(ExponentialSmoothing),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def check_panel_with_freq():
    """Test exponential smoothing on panel datam, datetime index with frequency."""
    # make panel with hour of day panel and datetime index
    y = load_airline()
    ind = pd.date_range(
        start="1960-01-01", periods=len(y.index), freq="H", name="datetime"
    )
    y = pd.DataFrame(y.values, index=ind, columns=["passengers"])
    y = y.set_index([y.index.hour.rename("hour"), y.index]).sort_index()
    assert y.loc[0].index.freq == pd.Timedelta("24H"), "Expected 24H frequency"

    fh = [1, 2]
    y_train, y_test = temporal_train_test_split(y, test_size=len(fh))
    forecaster = ExponentialSmoothing(trend="add", sp=1)

    # fit update predict
    forecaster.fit(y_train)
    forecaster.update(y_test)
    y_pred_update = forecaster.predict(fh=fh)

    # fit no update
    forecaster.fit(y)
    y_pred = forecaster.predict(fh=fh)

    assert y_pred.equals(y_pred_update), "Expected same predictions after update"
