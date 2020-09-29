#!/usr/bin/env python3 -u
# coding: utf-8
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Markus Löning", "Ayushmaan Seth"]

import numpy as np
import pandas as pd
import pytest
from sktime.forecasting.compose._reduce import ReducedRegressionForecaster
from sktime.forecasting.tests import TEST_OOS_FHS
from sktime.forecasting.tests import TEST_SPS
from sktime.forecasting.tests import TEST_WINDOW_LENGTHS
from sktime.utils.validation.forecasting import check_fh


n_timepoints = 30
n_train = 20
s = pd.Series(np.arange(n_timepoints))
y_train = s.iloc[:n_train]
y_test = s.iloc[n_train:]


@pytest.mark.parametrize("fh", TEST_OOS_FHS)
def test_strategy_direct(fh):
    pass


@pytest.mark.parametrize("fh", TEST_OOS_FHS)
def test_strategy_recursive(fh):
    pass


@pytest.mark.parametrize("fh", TEST_OOS_FHS)
def test_strategy_dirrec(fh):
    pass
