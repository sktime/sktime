"""Tests the VARReduce model against statsmodels VAR."""

import numpy as np
import pytest

from sktime.datasets import load_longley
from sktime.forecasting.var import VAR
from sktime.forecasting.var_reduce import VARReduce
from sktime.tests.test_switch import run_test_for_class

__author__ = ["meraldoantonio"]


@pytest.mark.skipif(
    not run_test_for_class([VAR, VARReduce]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_VAR_against_statsmodels():
    """Compares default VARReduce to statmodels VAR."""
    y = load_longley()[1]

    y_var = VAR().fit_predict(y, fh=[1, 2, 3])
    y_varr = VARReduce().fit_predict(y, fh=[1, 2, 3])
    np.testing.assert_allclose(y_var.values, y_varr.values)
