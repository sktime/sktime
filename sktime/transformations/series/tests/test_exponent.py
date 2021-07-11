#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Ryan Kuhns"]
__all__ = []

import pytest

from sktime.utils._testing.series import _make_series
from sktime.transformations.series.exponent import ExponentTransformer, SqrtTransformer

power_transformers = [ExponentTransformer, SqrtTransformer]


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
@pytest.mark.parametrize("power_transformer", power_transformers[:1])
@pytest.mark.parametrize("_power", ["a", [1, 2.3]])
def test_wrong_power_type_raises_error(power_transformer, _power):
    y = _make_series(n_timepoints=75)

    # Test input types
    match = f"Expected `power` to be int or float, but found {type(_power)}."
    with pytest.raises(ValueError, match=match):
        transformer = power_transformer(power=_power)
        transformer.fit(y)
