# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for IgnoreX."""

from unittest.mock import MagicMock

import pytest

from sktime.datasets import load_longley
from sktime.forecasting.compose import IgnoreX
from sktime.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(IgnoreX),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("ignore_x", [True, False, None])
def test_ignoreX(ignore_x):
    """Test that indeed X=None is passed iff the args of IgnoreX claim to do so."""
    y, X = load_longley()

    fcst = MagicMock()

    if ignore_x is None:
        igx = IgnoreX(forecaster=fcst)
    else:
        igx = IgnoreX(forecaster=fcst, ignore_x=ignore_x)

    igx.fit(y, fh=[1, 2, 3], X=X)

    mock_fitted = igx.forecaster_.fit
    call_args_list = mock_fitted.call_args_list
    all_calls_X_none = all(call_args[1]["X"] is None for call_args in call_args_list)
    any_calls_X_none = any(call_args[1]["X"] is None for call_args in call_args_list)

    if ignore_x in [True, None]:
        assert all_calls_X_none
    else:
        assert not any_calls_X_none
