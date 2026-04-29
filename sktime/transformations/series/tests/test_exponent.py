#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for the power transformers."""

__author__ = ["RNKuhns", "Akanksha Trehun"]
__all__ = []

import numpy as np
import pandas as pd
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


@pytest.mark.skipif(
    not run_test_for_class([ExponentTransformer]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("power", [0.5, 2.0, 3.0])
def test_inverse_transform_roundtrip_with_negatives(power):
    """Regression test for inverse_transform with offset='auto' and negative data.

    Failure case in bug #10001.
    """
    X = pd.DataFrame({"col": [-2.0, -0.5, 0.0, 1.0, 3.0]})
    t = ExponentTransformer(power=power, offset="auto")
    t.fit(X)
    Xt = t.transform(X)
    X_recovered = t.inverse_transform(Xt)
    np.testing.assert_allclose(X.to_numpy(), X_recovered.to_numpy(), atol=1e-10)
