# -*- coding: utf-8 -*-
import numpy as np
import pytest
from sktime.transformations.panel.signature_based import SignatureTransformer
from sktime.utils.validation._dependencies import _check_soft_dependencies

_check_soft_dependencies("esig")
import esig  # noqa: E402


def test_generalised_signature_method():
    # Build an array X, note that this is [n_sample, n_channels, length] shape.
    n_channels = 3
    depth = 4
    X = np.random.randn(5, n_channels, 10)

    # Check the global dimension comes out correctly
    method = SignatureTransformer(depth=depth, window_name="global")
    assert method.fit_transform(X).shape[1] == esig.sigdim(n_channels + 1, depth) - 1

    # Check dyadic dim
    method = SignatureTransformer(depth=depth, window_name="dyadic", window_depth=3)
    assert (
        method.fit_transform(X).shape[1]
        == (esig.sigdim(n_channels + 1, depth) - 1) * 15
    )

    # Ensure an example
    X = np.array([[0, 1], [2, 3], [1, 1]]).reshape(-1, 2, 3)
    method = SignatureTransformer(depth=2, window_name="global")
    true_arr = np.array(
        [[1.0, 2.0, 1.0, 0.5, 1.33333333, -0.5, 0.66666667, 2.0, -1.0, 1.5, 3.0, 0.5]]
    )
    assert np.allclose(method.fit_transform(X), true_arr)


def test_window_error():
    X = np.random.randn(5, 2, 3)

    # Check dyadic gives a value error
    method = SignatureTransformer(window_name="dyadic", window_depth=10)
    with pytest.raises(ValueError):
        method.fit_transform(X)

    # Expanding and sliding errors
    method = SignatureTransformer(
        window_name="expanding", window_length=10, window_step=5
    )
    with pytest.raises(ValueError):
        method.fit_transform(X)
    method = SignatureTransformer(
        window_name="sliding", window_length=10, window_step=5
    )
    with pytest.raises(ValueError):
        method.fit_transform(X)
