#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for SquaringResiduals parameter validation."""

__author__ = ["Akanksha Trehun"]

import pytest

from sktime.forecasting.squaring_residuals import SquaringResiduals
from sktime.utils.dependencies import _check_estimator_deps


@pytest.mark.skipif(
    not _check_estimator_deps(SquaringResiduals, severity="none"),
    reason="skip test if required soft dependencies not available",
)
@pytest.mark.parametrize("bad_distr", ["gaussian", "uniform", 42, None])
def test_squaring_residuals_invalid_distr_raises(bad_distr):
    """Invalid distr must raise ValueError at construction, not silently pass.

    Regression test for bug #10004: bare assert statements were stripped under
    python -O, so invalid arguments were silently accepted.
    """
    with pytest.raises(ValueError, match="distr"):
        SquaringResiduals(distr=bad_distr)


@pytest.mark.skipif(
    not _check_estimator_deps(SquaringResiduals, severity="none"),
    reason="skip test if required soft dependencies not available",
)
@pytest.mark.parametrize("bad_strategy", ["cube", "log", 1, None])
def test_squaring_residuals_invalid_strategy_raises(bad_strategy):
    """Invalid strategy must raise ValueError at construction."""
    with pytest.raises(ValueError, match="strategy"):
        SquaringResiduals(strategy=bad_strategy)


@pytest.mark.skipif(
    not _check_estimator_deps(SquaringResiduals, severity="none"),
    reason="skip test if required soft dependencies not available",
)
@pytest.mark.parametrize("bad_window", [0, -1, -100])
def test_squaring_residuals_invalid_initial_window_raises(bad_window):
    """initial_window < 1 must raise ValueError at construction."""
    with pytest.raises(ValueError, match="initial_window"):
        SquaringResiduals(initial_window=bad_window)
