"""Tests for lag parameter estimators."""

__author__ = ["satvshr"]

import pytest
from numpy.testing import assert_allclose

from sktime.datasets import load_airline
from sktime.param_est.lag import ARLagOrderSelector
from sktime.utils.dependencies import _check_estimator_deps


@pytest.mark.skipif(
    not _check_estimator_deps(ARLagOrderSelector, severity="none"),
    reason="skip test if required soft dependencies not available",
)
def test_against_statsmodels():
    """Compare sktime's ARLagOrderSelector with statsmodels ar_select_order."""
    from statsmodels.tsa.ar_model import ar_select_order

    y = load_airline()
    sm_result = ar_select_order(
        endog=y,
        maxlag=5,
        ic="bic",
        glob=False,
        trend="c",
        seasonal=False,
    )

    selector = ARLagOrderSelector(
        maxlag=5,
        ic="bic",
        glob=False,
        trend="c",
        seasonal=False,
    )
    selector.fit(y)

    assert sm_result.ar_lags == selector.selected_model_
    assert_allclose(min(sm_result.bic.values()), selector.ic_value_, rtol=1e-5)
