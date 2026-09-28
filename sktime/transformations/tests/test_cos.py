"""Tests for CosineTransformer."""

__author__ = ["aminmiral"]

from math import pi

import numpy as np
import pandas as pd
import pytest

from sktime.tests.test_switch import run_test_module_changed
from sktime.transformations.cos import CosineTransformer


@pytest.mark.skipif(
    not run_test_module_changed("sktime.transformations"),
    reason="run test only if anything in sktime.transformations module has changed",
)
def test_inverse_transform_range_tag_is_the_true_domain():
    """Test the declared domain of invertibility is one the inverse actually has.

    ``capability:inverse_transform:range`` is documented as the domain of
    invertibility, and ``test_transform_inverse_transform_equivalent`` asserts the
    round trip on exactly that interval. ``np.arccos`` returns values in
    ``[0, pi]``, so the identity holds there and nowhere else.
    """
    transformer = CosineTransformer()
    inv_range = transformer.get_tag("capability:inverse_transform:range")

    X = pd.DataFrame({"a": np.linspace(inv_range[0], inv_range[1], 25)})
    Xt = transformer.fit_transform(X)
    Xit = transformer.inverse_transform(Xt)

    np.testing.assert_allclose(Xit.to_numpy(), X.to_numpy(), atol=1e-12)


@pytest.mark.skipif(
    not run_test_module_changed("sktime.transformations"),
    reason="run test only if anything in sktime.transformations module has changed",
)
def test_inverse_transform_outside_range_is_the_arccos_branch():
    """Test the documented behaviour outside the domain of invertibility.

    Outside ``[0, pi]`` the inverse returns the value in ``[0, pi]`` with the same
    cosine, rather than the input. Pinning it keeps the docstring honest.
    """
    X = pd.DataFrame({"a": [-2.0, -1.0, 4.0, 5.0]})

    transformer = CosineTransformer()
    Xit = transformer.inverse_transform(transformer.fit_transform(X))

    expected = np.arccos(np.cos(X.to_numpy()))
    np.testing.assert_allclose(Xit.to_numpy(), expected, atol=1e-12)
    assert not np.allclose(Xit.to_numpy(), X.to_numpy())


@pytest.mark.skipif(
    not run_test_module_changed("sktime.transformations"),
    reason="run test only if anything in sktime.transformations module has changed",
)
def test_transform_is_numpy_cos():
    """Test the forward transform is plain elementwise cosine."""
    X = pd.DataFrame({"a": np.linspace(-2 * pi, 2 * pi, 17)})

    Xt = CosineTransformer().fit_transform(X)

    np.testing.assert_allclose(Xt.to_numpy(), np.cos(X.to_numpy()), atol=1e-12)
