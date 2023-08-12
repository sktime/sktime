"""Tests for CFFilter wrapper annotation estimator."""

__author__ = ["ken-maeda"]

import pandas as pd
import pytest
from numpy import array_equal

from sktime.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies("statsmodels.api", severity="none"),
    reason="skip test if required soft dependency for statsmodels.api not available",
)
def test_CFFilter_wrapper():
    """Verify that the wrapped CFFilter estimator agrees with statsmodel."""
    # moved all potential soft dependency import inside the test:
    import statsmodels.api as sm

    from sktime.transformations.series.cffilter import CFFilter as _CFFilter

    dta = sm.datasets.macrodata.load_pandas().data
    index = pd.date_range(start="1959Q1", end="2009Q4", freq="Q")
    dta.set_index(index, inplace=True)
    sm_cycles = sm.tsa.filters.cffilter(dta[["realinv"]], 6, 32, True)[0]
    cf = _CFFilter(6, 32, True)
    sk_cycles = cf.fit_transform(X=dta[["realinv"]]).squeeze("columns")
    assert array_equal(sm_cycles, sk_cycles)
