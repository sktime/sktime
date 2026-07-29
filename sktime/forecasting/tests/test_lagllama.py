"""Tests for LagLlamaForecaster.

LagLlamaForecaster has a very restrictive dependency set,
which may cancel out with the matrix testing strategy.

Therefore, we call a check_estimator separately.
"""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

import pytest

from sktime.forecasting.lagllama import LagLlamaForecaster
from sktime.tests.test_switch import run_test_for_class
from sktime.utils import check_estimator


@pytest.mark.skipif(
    not run_test_for_class(LagLlamaForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_lagllamaforecaster():
    """Run standard test suite for LagLlamaForecaster.

    LagLlamaForecaster has a very restrictive dependency set,
    which may cancel out with the matrix testing strategy.

    Therefore, we call a check_estimator separately.
    """
    check_estimator(LagLlamaForecaster, raise_exceptions=True)
