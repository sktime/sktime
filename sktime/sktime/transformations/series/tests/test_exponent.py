#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for the power transformers."""

__author__ = ["RNKuhns"]
__all__ = []

import pytest

from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.series.exponent import ExponentTransformer, SqrtTransformer
from sktime.utils._testing.series import _make_series

power_transformers = [ExponentTransformer, SqrtTransformer]


@pytest.mark.skipif(
    not run_test_for_class(power_transformers),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("power_transformer", power_transformers)
@pytest.mark.parametrize("_offset", ["a", [1, 2.3]])
def test_wrong_offset_type_raises_error(power_transformer, _offset):
    y = _make_series(n_timepoints=75)

    # Test input types
    match = f"Expected `offset` to be int or float, but found {type(_offset)}."
    with pytest.raises(ValueError, match=match):
        transformer = power_transformer(offset=_offset)
        transformer.fit(y)


# Test only applies to PowerTransformer b/c SqrtTransformer doesn't have power
# hyperparameter
@pytest.mark.skipif(
    not run_test_for_class(power_transformers),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("power_transformer", power_transformers[:1])
@pytest.mark.parametrize("_power", ["a", [1, 2.3]])
def test_wrong_power_type_raises_error(power_transformer, _power):
    y = _make_series(n_timepoints=75)

    # Test input types
    match = f"Expected `power` to be int or float, but found {type(_power)}."
    with pytest.raises(ValueError, match=match):
        transformer = power_transformer(power=_power)
        transformer.fit(y)
