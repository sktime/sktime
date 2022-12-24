# -*- coding: utf-8 -*-
"""Tests for Prophet.

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""

__author__ = ["fkiraly"]

import pandas as pd
import pytest

from sktime.forecasting.fbprophet import Prophet
from sktime.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies("prophet", severity="none"),
    reason="skip test if required soft dependency not available",
)
@pytest.mark.parametrize("indextype", ["range", "period"])
def test_prophet_nonnative_index(indextype):
    """Check prophet with RangeIndex and PeriodIndex."""
    y = pd.DataFrame({"a": [1, 2, 3, 4]})
    X = pd.DataFrame({"b": [1, 5, 3, 3, 5, 6], "c": [5, 5, 3, 3, 4, 2]})

    if indextype == "period":
        y.index = pd.period_range('2000-01-01', periods=4)
        X.index = pd.period_range('2000-01-01', periods=6)

    X_train = X.iloc[:4]
    X_test = X.iloc[4:]

    fh = [1, 2]

    f = Prophet()
    f.fit(y, X=X_train)
    y_pred = f.predict(fh=fh, X=X_test)

    if indextype == "range":
        assert isinstance(y_pred.index, pd.RangeIndex)
    if indextype == "period":
        assert isinstance(y_pred.index, pd.PeriodIndex)
