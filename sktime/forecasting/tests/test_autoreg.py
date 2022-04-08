"""Tests the AutoReg Model."""
__author__ = ["ryali1"]

import numpy as np
from numpy.testing import assert_allclose
from statsmodels.tsa.ar_model import AutoReg as _AutoReg
from sktime.forecasting.autoreg import AutoReg
from sktime.utils._testing.forecasting import make_forecasting_problem

y = make_forecasting_problem()

def test_AutoReg_against_statsmodels():
  """ Compares Sktime's and Statsmodels AutoReg Implementations."""
  sktimeAutoReg = AutoReg(lags = 10)
  sktimeAutoReg.fit(y)
  y_pred = sktimeAutoReg.predict(fh = np.arange(1,10))

  statsmodelAutoReg = _AutoReg(y, lags = 10 ).fit()
  statsmodel_pred = statsmodelAutoReg.predict(start=len(y), end=len(y)+8)
  assert_allclose(y_pred.tolist(), statsmodel_pred.tolist())