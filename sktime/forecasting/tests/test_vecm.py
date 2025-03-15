"""Tests the VAR model."""

__author__ = ["thayeylolu", "AurumnPegasus"]
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.vecm import VECM
from sktime.split import temporal_train_test_split
from sktime.tests.test_switch import run_test_for_class
from sktime.utils.dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not run_test_for_class(VECM),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_VECM_against_statsmodels():
    """Compares Sktime's and Statsmodel's VECM."""
    from statsmodels.tsa.api import VECM as _VECM

    pandas2 = _check_soft_dependencies("pandas>=2.0.0", severity="none")
    if pandas2:
        freq = "ME"
    else:
        freq = "M"

    index = pd.date_range(start="2005", end="2006-12", freq=freq)
    df = pd.DataFrame(
        np.random.randint(0, 100, size=(23, 2)),
        columns=list("AB"),
        index=pd.PeriodIndex(index),
    )
    train, test = temporal_train_test_split(df)
    sktime_model = VECM()
    fh = ForecastingHorizon([1, 3, 4, 5, 7, 9])
    _ = sktime_model.fit(train)
    y_pred = sktime_model.predict(fh=fh)

    stats = _VECM(train)
    stats_fit = stats.fit()
    fh_int = fh.to_relative(train.index[-1])
    # lagged = stats_fit.k_ar
    y_pred_stats = stats_fit.predict(steps=fh_int[-1])
    new_arr = []
    for i in fh_int:
        new_arr.append(y_pred_stats[i - 1])
    # print("predicted: \n")
    # print(y_pred)
    # print("actual: \n")
    # print(new_arr)
    assert_allclose(y_pred, new_arr)
