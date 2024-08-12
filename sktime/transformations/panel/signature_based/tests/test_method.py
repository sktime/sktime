"""Tests for signature method."""

import numpy as np
import pytest

from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.panel.signature_based import SignatureTransformer


@pytest.mark.skipif(
    not run_test_for_class(SignatureTransformer),
    reason="skip test if python environment requirements for estimator are not met",
)
def test_generalised_signature_method():
    """Check that dimension and dim of output are correct."""
    # Build an array X, note that this is [n_sample, n_channels, length] shape.
    import esig

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


@pytest.mark.skipif(
    not run_test_for_class(SignatureTransformer),
    reason="skip test if python environment requirements for estimator are not met",
)
def test_window_error():
    """Test that wrong window parameters raise error."""
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
