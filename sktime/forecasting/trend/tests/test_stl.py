"""Test STLForecaster."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["ericjb"]

import pytest

from sktime.datasets import load_airline
from sktime.forecasting.trend import STLForecaster
from sktime.tests.test_switch import run_test_for_class


# zero trend does not work without intercept
@pytest.mark.skipif(
    not run_test_for_class([STLForecaster]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_plot_components():
    """Test plot_components method"""
    y = load_airline()
    fc = STLForecaster(sp=12)
    fc.fit(y)
    _, _ = fc.plot_components(title="My Decomposition Plot")
