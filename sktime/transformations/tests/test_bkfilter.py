# -*- coding: utf-8 -*-
"""Tests for BKfilter wrapper annotation estimator."""

__author__ = ["klam-data", "pyyim"]

import pandas as pd
import pytest
from numpy import array_equal

from sktime.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies("statsmodels.api", severity="none"),
    reason="skip test if required soft dependency for statsmodels.api not available",
)
def test_BKFilter_wrapper():
    """Verify that the wrapped BKFilter estimator agrees with statsmodel."""
    # moved all potential soft dependency import inside the test:
    import statsmodels.api as sm

    from sktime.transformations.series.bkfilter import BKFilter as _BKFilter

    dta = sm.datasets.macrodata.load_pandas().data
    index = pd.date_range(start="1959Q1", end="2009Q4", freq="Q")
    dta.set_index(index, inplace=True)
    sm_cycles = sm.tsa.filters.bkfilter(dta[["realinv"]], 6, 24, 12)
    bk = _BKFilter(6, 24, 12)
    sk_cycles = bk.fit_transform(X=dta[["realinv"]])
    assert array_equal(sm_cycles, sk_cycles)
