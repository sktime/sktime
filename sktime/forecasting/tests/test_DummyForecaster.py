#!/usr/bin/env python3 -u
# coding: utf-8

__author__ = "Markus Löning"

import numpy as np
import pandas as pd
import pytest

from sktime.forecasting.dummy import DummyForecaster
from sktime.forecasting.model_selection import RollingWindowSplit

n_timepoints = 30
n_train = 20
s = pd.Series(np.arange(n_timepoints))
y_train = s.iloc[:n_train]
y_test = s.iloc[n_train:]


@pytest.mark.parametrize("fh", [1, 3, np.arange(1, 5)])
@pytest.mark.parametrize("strategy", ["last"])
def test_predict_values(fh, strategy):
    f = DummyForecaster(strategy=strategy)
    f.fit(y_train)
    y_pred = f.predict(fh)

    if strategy == "last":
        expected = np.repeat(y_train.iloc[-1], len(f.fh))

    np.testing.assert_array_equal(y_pred, expected)
