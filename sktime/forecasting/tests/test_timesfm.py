"""Tests for TimesFMForecaster.

TimesFmForecaster has a very restrictive dependency set,
which may cancel out with the matrix testing strategy.

Therefore, we call a check_estimator separately.
"""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

import pytest

from sktime.forecasting.timesfm_forecaster import TimesFMForecaster
from sktime.tests.test_switch import run_test_for_class
from sktime.utils import check_estimator


@pytest.mark.skipif(
    not run_test_for_class(TimesFMForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_timesfmforecaster():
    """Run standard test suite for TimesFMForecaster.

    TimesFmForecaster has a very restrictive dependency set,
    which may cancel out with the matrix testing strategy.

    Therefore, we call a check_estimator separately.
    """
    check_estimator(TimesFMForecaster, raise_exceptions=True)
