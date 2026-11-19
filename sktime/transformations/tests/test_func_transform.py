# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Regression tests for FunctionTransformer inverse checks with keyword arguments."""

import numpy as np
import pandas as pd
import pytest

from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.func_transform import FunctionTransformer

pytestmark = pytest.mark.skipif(
    not run_test_for_class(FunctionTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)


def _affine_required(X, *, scale, offset):
    return scale * X + offset


def _undo_affine_required(X, *, scale, offset):
    return (X - offset) / scale


def _affine(X, *, scale=1.0, offset=0.0):
    return scale * X + offset


def _undo_affine(X, *, scale=1.0, offset=0.0):
    return (X - offset) / scale


def _input(kind, with_nan=False):
    values = np.array([1.0, np.nan if with_nan else 2.0, 3.0])
    if kind == "series":
        return pd.Series(values, name="signal")
    if kind == "frame":
        return pd.DataFrame({"first": values, "second": values + 1})
    return values


@pytest.mark.parametrize("kind", ["array", "series", "frame"])
@pytest.mark.parametrize("with_nan", [False, True])
def test_inverse_check_passes_required_keyword_arguments(kind, with_nan):
    """A valid configured inverse pair must accept required keyword parameters."""
    X = _input(kind, with_nan)
    forward_args = {"scale": 2.0, "offset": 3.0}
    inverse_args = {"scale": 2.0, "offset": 3.0}
    transformer = FunctionTransformer(
        func=_affine_required,
        inverse_func=_undo_affine_required,
        kw_args=forward_args,
        inv_kw_args=inverse_args,
    )

    result = transformer.fit_transform(X)

    np.testing.assert_allclose(result, 2 * X + 3, equal_nan=True)
    np.testing.assert_allclose(transformer.inverse_transform(result), X, equal_nan=True)
    assert forward_args == inverse_args == {"scale": 2.0, "offset": 3.0}


@pytest.mark.parametrize("kind", ["array", "series", "frame"])
def test_inverse_check_uses_configured_functions(kind):
    """Default functions can be inverse while the configured pair is not."""
    transformer = FunctionTransformer(
        func=_affine,
        inverse_func=_undo_affine,
        kw_args={"scale": 2.0, "offset": 3.0},
        inv_kw_args={"scale": 4.0, "offset": 3.0},
    )

    with pytest.raises(UserWarning, match="not strictly inverse"):
        transformer.fit_transform(_input(kind))


@pytest.mark.parametrize("kind", ["array", "series", "frame"])
def test_inverse_check_can_be_disabled_for_configured_functions(kind):
    """Disabling validation still uses the keyword parameters for each operation."""
    X = _input(kind)
    transformer = FunctionTransformer(
        func=_affine_required,
        inverse_func=_undo_affine_required,
        kw_args={"scale": 2.0, "offset": 3.0},
        inv_kw_args={"scale": 4.0, "offset": 3.0},
        check_inverse=False,
    )

    result = transformer.fit_transform(X)

    np.testing.assert_allclose(result, 2 * X + 3)
    np.testing.assert_allclose(transformer.inverse_transform(result), X / 2)
