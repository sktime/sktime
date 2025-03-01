"""Tests for TimeMoEForecaster.

TimeMoEForecaster has a very restrictive dependency set,
which may cancel with a matrix testing strategy.

Hence, we call a check_estimator separately on TimeMoEForecaster
"""

import pytest

from sktime.forecasting.timemoe import TimeMoEForecaster
from sktime.tests.test_switch import run_test_for_class
from sktime.utils import check_estimator


@pytest.mark.skipif(
    not run_test_for_class(TimeMoEForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_timemoeforecaster():
    """Run standard test suite for the TimeMoEForecaster.

    TimeMoE has a very restrictive dependency set,
    which may cancel out due to a matrix testing strategy.

    Therefore, we call a check_estimator separately.
    """

    check_estimator(TimeMoEForecaster, raise_exceptions=True)
