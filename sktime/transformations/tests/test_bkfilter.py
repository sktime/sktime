"""Tests for BKfilter wrapper annotation estimator."""

__author__ = ["klam-data", "pyyim"]

import pandas as pd
import pytest
from numpy import array_equal

from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.series.bkfilter import BKFilter


@pytest.mark.skipif(
    not run_test_for_class(BKFilter),
    reason="skip test only if softdeps are present and incrementally (if requested)",
)
def test_BKFilter_wrapper():
    """Verify that the wrapped BKFilter estimator agrees with statsmodel."""
    # moved all potential soft dependency import inside the test:
    import statsmodels.api as sm

    dta = sm.datasets.macrodata.load_pandas().data
    index = pd.date_range(start="1959Q1", end="2009Q4", freq="Q")
    dta.set_index(index, inplace=True)
    sm_cycles = sm.tsa.filters.bkfilter(dta[["realinv"]], 6, 24, 12)
    bk = BKFilter(6, 24, 12)
    sk_cycles = bk.fit_transform(X=dta[["realinv"]])
    assert array_equal(sm_cycles, sk_cycles)
