"""Tests for HPfilter wrapper annotation estimator."""

__author__ = ["ken-maeda"]

import pandas as pd
import pytest
from numpy import array_equal

from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.series.hpfilter import HPFilter


@pytest.mark.skipif(
    not run_test_for_class(HPFilter),
    reason="skip test only if softdeps are present and incrementally (if requested)",
)
def test_HPFilter_wrapper():
    """Verify that the wrapped HPFilter estimator agrees with statsmodel."""
    # moved all potential soft dependency import inside the test:
    import statsmodels.api as sm

    dta = sm.datasets.macrodata.load_pandas().data
    index = pd.date_range(start="1959Q1", end="2009Q4", freq="Q")
    dta.set_index(index, inplace=True)
    sm_cycle = sm.tsa.filters.hpfilter(dta[["realinv"]], 1600)[0]
    sm_cycle = pd.DataFrame(sm_cycle)
    hp = HPFilter(lamb=1600)
    sk_cycle = hp.fit_transform(X=dta[["realinv"]])
    assert array_equal(sm_cycle, sk_cycle)
