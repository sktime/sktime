"""Tests for HPfilter wrapper annotation estimator."""

__author__ = ["ken-maeda"]

import pandas as pd
import pytest
from numpy import array_equal

from sktime.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies("statsmodels.api", severity="none"),
    reason="skip test if required soft dependency for statsmodels.api not available",
)
def test_HPFilter_wrapper():
    """Verify that the wrapped HPFilter estimator agrees with statsmodel."""
    # moved all potential soft dependency import inside the test:

    import statsmodels.api as sm

    from sktime.transformations.series.hpfilter import HPFilter as _HPFilter

    dta = sm.datasets.macrodata.load_pandas().data
    index = pd.date_range(start="1959Q1", end="2009Q4", freq="Q")
    dta.set_index(index, inplace=True)
    sm_cycle = sm.tsa.filters.hpfilter(dta[["realinv"]], 1600)[0]
    sm_cycle = pd.DataFrame(sm_cycle)
    hp = _HPFilter(lamb=1600)
    sk_cycle = hp.fit_transform(X=dta[["realinv"]])
    assert array_equal(sm_cycle, sk_cycle)
