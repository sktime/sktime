"""Test the ComposableTimeSeriesForestRegressor."""

import numpy as np
import pytest
from sklearn.metrics import r2_score

from sktime.regression.compose import ComposableTimeSeriesForestRegressor
from sktime.tests.test_switch import run_test_for_class
from sktime.utils._testing.panel import make_regression_problem


@pytest.mark.skipif(
    not run_test_for_class(ComposableTimeSeriesForestRegressor),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_tsf_oob_score():
    """Test composable TSF regressor with out-of-bag score.

    Failure case of bug #11409, where fit failed with ``oob_score=True``.
    """
    X, y = make_regression_problem(random_state=0)

    reg = ComposableTimeSeriesForestRegressor(
        n_estimators=10, bootstrap=True, oob_score=True, random_state=0
    )
    reg.fit(X, y)

    assert reg.oob_prediction_.shape == (X.shape[0],)
    assert np.isclose(reg.oob_score_, r2_score(y, reg.oob_prediction_))
