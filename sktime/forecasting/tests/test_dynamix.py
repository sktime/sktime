"""Tests for DynaMixForecaster.

DynaMix has a restrictive, non-PyPI, GPL-licensed dependency set, which may
cancel out with the matrix testing strategy. Therefore, we call a
``check_estimator`` separately. The license-gate test does not require the
soft dependencies and is always run.
"""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

import pytest

from sktime.forecasting.dynamix import DynaMixForecaster
from sktime.tests.test_switch import run_test_for_class
from sktime.utils import check_estimator


@pytest.mark.skipif(
    not run_test_for_class(DynaMixForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_dynamixforecaster():
    """Run standard test suite for DynaMixForecaster.

    DynaMixForecaster has a restrictive dependency set (the GPL-licensed
    ``dynamix`` package is not available on PyPI), which may cancel out with the
    matrix testing strategy. Therefore, we call a check_estimator separately.
    """
    check_estimator(DynaMixForecaster, raise_exceptions=True)


def test_dynamix_license_gate():
    """Test that the GPL license gate is enforced in the constructor.

    This test does not require ``torch`` or the ``dynamix`` package, as the
    license check is performed before any model loading.
    """
    # without accepting the license, construction must raise a ValueError
    with pytest.raises(ValueError, match="license"):
        DynaMixForecaster()

    # accepting the license must allow construction without raising the gate
    forecaster = DynaMixForecaster(license_accepted=True)
    assert forecaster.license_accepted is True
