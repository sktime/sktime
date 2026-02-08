# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for johansen parameter estimator."""

__author__ = ["OldPatrick"]

import numpy as np
import pandas as pd
import pytest

from sktime.datasets import load_airline
from sktime.param_est.impulse import ImpulseResponseFunction
from sktime.utils.dependencies import _check_estimator_deps

@pytest.mark.skipif(
    not _check_estimator_deps(ImpulseResponseFunction, severity="none"),
    reason="skip test if required soft dependencies not available",
)
def test_irf_on_varmax():
    """Test ImpulseResponseFunctions on airline data with VARMAX."""
    pass

def test_irf_on_dynamicfactor():
    """Test ImpulseResponseFunctions on airline data with DynamicFactor."""
    X = load_airline()
    X2 = X.shift(1).bfill()
    df = pd.DataFrame({"X": X, "X2": X2})
    irf_est = ImpulseResponseFunction()
    irf_est.fit(df)

    #assert list(coint_est.get_fitted_params()["ind"]) == [0, 1]
    # actual = coint_est.get_fitted_params()["ind"]
    # expected = [0, 1]
    # np.testing.assert_array_equal(actual, expected)

def test_irf_on_VECM():
    """Test ImpulseResponseFunctions on airline data with VECM."""
    pass