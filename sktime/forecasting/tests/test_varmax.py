"""Tests the VARMAX model."""

__author__ = ["KatieBuc"]

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.varmax import VARMAX
from sktime.split import temporal_train_test_split
from sktime.tests.test_switch import run_test_for_class

np.random.seed(13455)
index = pd.date_range(start="2020-01", end="2021-12", freq="M")
df = pd.DataFrame(
    np.random.randint(0, 100, size=(23, 3)),
    columns=list("ABC"),
    index=pd.PeriodIndex(index),
)


@pytest.mark.skip(reason="undiagnosed failure, see #6260")
@pytest.mark.skipif(
    not run_test_for_class(VARMAX),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_VARMAX_against_statsmodels():
    """Compares Sktime's and Statsmodel's VARMAX.

    with default variables.
    """
    from statsmodels.tsa.api import VARMAX as _VARMAX

    train, _ = temporal_train_test_split(df.astype("float64"))
    y = train[["A", "B"]]

    sktime_model = VARMAX()
    fh = ForecastingHorizon([1, 3, 4, 5, 7, 9])
    sktime_model.fit(y)
    y_pred = sktime_model.predict(fh=fh)

    stats = _VARMAX(y)
    stats_fit = stats.fit()
    start, end = len(train) + fh[0] - 1, len(train) + fh[-1] - 1
    y_pred_stats = stats_fit.predict(start=start, end=end)
    y_pred_stats = y_pred_stats.loc[fh.to_absolute_index(train.index[-1])]

    assert_allclose(y_pred, y_pred_stats)


@pytest.mark.skip(reason="undiagnosed failure, see #6260")
@pytest.mark.skipif(
    not run_test_for_class(VARMAX),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_VARMAX_against_statsmodels_with_exog():
    """Compares Sktime's and Statsmodel's VARMAX.

    with exogenous input.
    """
    from statsmodels.tsa.api import VARMAX as _VARMAX

    train, test = temporal_train_test_split(df.astype("float64"))
    y_train, X_train = train[["A", "B"]], train[["C"]]
    _, X_test = test[["A", "B"]], test[["C"]]
    fh = ForecastingHorizon([1, 2, 3, 4, 5, 6])
    assert len(fh) == len(X_test)

    sktime_model = VARMAX()
    sktime_model.fit(y_train, X=X_train)
    y_pred = sktime_model.predict(fh=fh, X=X_test)

    stats = _VARMAX(y_train, exog=X_train)
    stats_fit = stats.fit()
    start, end = len(train) + fh[0] - 1, len(train) + fh[-1] - 1
    y_pred_stats = stats_fit.predict(start=start, end=end, exog=X_test)
    y_pred_stats = y_pred_stats.loc[fh.to_absolute_index(train.index[-1])]

    assert_allclose(y_pred, y_pred_stats)
