"""Tests for CFFilter wrapper annotation estimator."""

__author__ = ["ken-maeda"]

import pandas as pd
import pytest
from numpy import array_equal

from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.series.cffilter import CFFilter


@pytest.mark.skipif(
    not run_test_for_class(CFFilter),
    reason="skip test only if softdeps are present and incrementally (if requested)",
)
def test_CFFilter_wrapper():
    """Verify that the wrapped CFFilter estimator agrees with statsmodel."""
    # moved all potential soft dependency import inside the test:
    import statsmodels.api as sm

    dta = sm.datasets.macrodata.load_pandas().data
    index = pd.date_range(start="1959Q1", end="2009Q4", freq="Q")
    dta.set_index(index, inplace=True)
    sm_cycles = sm.tsa.filters.cffilter(dta[["realinv"]], 6, 32, True)[0]
    cf = CFFilter(6, 32, True)
    sk_cycles = cf.fit_transform(X=dta[["realinv"]]).squeeze("columns")
    assert array_equal(sm_cycles, sk_cycles)
