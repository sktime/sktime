"""Tests the VARReduce model against statsmodels VAR."""

import numpy as np
import pytest

from sktime.datasets import load_longley
from sktime.forecasting.var import VAR
from sktime.forecasting.var_reduce import VARReduce
from sktime.tests.test_switch import run_test_for_class

__author__ = ["meraldoantonio", "Akanksha Trehun"]


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


@pytest.mark.skipif(
    not run_test_for_class([VARReduce]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_varreduce_invalid_regressor_no_fit_raises():
    """Regressor without 'fit' must raise ValueError at construction.

    Regression test for bug #10005: bare assert statements were stripped under
    python -O, so an incompatible regressor was silently accepted.
    """

    class NoFitRegressor:
        def predict(self, X):
            pass

    with pytest.raises(ValueError, match="fit"):
        VARReduce(regressor=NoFitRegressor())


@pytest.mark.skipif(
    not run_test_for_class([VARReduce]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_varreduce_invalid_regressor_no_predict_raises():
    """Regressor without 'predict' must raise ValueError at construction."""

    class NoPredict:
        def fit(self, X, y):
            pass

    with pytest.raises(ValueError, match="predict"):
        VARReduce(regressor=NoPredict())
