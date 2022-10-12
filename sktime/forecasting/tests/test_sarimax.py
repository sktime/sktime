# -*- coding: utf-8 -*-
"""Tests the SARIMAX model."""
__author__ = ["TNTran92"]

import pytest
from numpy.testing import assert_allclose

from sktime.forecasting.sarimax import SARIMAX
from sktime.utils._testing.forecasting import make_forecasting_problem
from sktime.utils.validation._dependencies import _check_soft_dependencies

df = make_forecasting_problem()


@pytest.mark.skipif(
    not _check_soft_dependencies("statsmodels", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_SARIMAX_against_statsmodels():
    """Compares Sktime's and Statsmodel's SARIMAX."""
    from statsmodels.tsa.api import SARIMAX as _SARIMAX

    sktime_model = SARIMAX(order=(1, 0, 0), trend="t", seasonal_order=(1, 0, 0, 6))
    sktime_model.fit(df)
    y_pred = sktime_model.predict(df.index)

    stats = _SARIMAX(endog=df, order=(1, 0, 0), trend="t", seasonal_order=(1, 0, 0, 6))
    stats_fit = stats.fit()
    stats_pred = stats_fit.predict(df.index[0])
    assert_allclose(y_pred.tolist(), stats_pred.tolist())
