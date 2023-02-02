# -*- coding: utf-8 -*-
"""Tests the AutoReg model."""
__author__ = ["pranavvp16"]

import pytest
from numpy.testing import assert_allclose

from sktime.forecasting.autoreg import AutoReg
from sktime.utils._testing.forecasting import make_forecasting_problem
from sktime.utils.validation._dependencies import _check_soft_dependencies

df = make_forecasting_problem()


@pytest.mark.skipif(
    not _check_soft_dependencies("statsmodels", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_AutoReg_against_statsmodel():
    """Compare Sktime's and Statsmodel's AutoReg."""
    from statsmodels.tsa.api import AutoReg as _AutoReg

    sktime_model = AutoReg(lags=4)
    sktime_model.fit(df)
    sktime_pred = sktime_model.predict(df.index)

    stats_model = _AutoReg(endog=df, lags=4)
    stats_fit = stats_model.fit()
    stats_pred = stats_fit.predict(df.index[0])
    assert_allclose(sktime_pred.tolist(), stats_pred.tolist())
