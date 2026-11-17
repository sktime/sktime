"""Testing weighted statistics utilities."""

import numpy as np
import pytest

from sktime.tests.test_switch import run_test_module_changed
from sktime.utils.stats import _weighted_percentile


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.utils"]),
    reason="Run if utils module has changed.",
)
@pytest.mark.parametrize("n", [1, 2, 7, 8])
def test_weighted_percentile_median(n):
    """Test that the weighted median agrees with np.median.

    With equal weights, the result must equal ``np.median``, also for even ``n``.
    With integer weights, the result must equal ``np.median`` of the values
    repeated as often as their weight.
    """
    rng = np.random.default_rng(42)
    x = rng.normal(size=n)
    k = rng.integers(1, 4, size=n)

    assert np.isclose(_weighted_percentile(x, np.ones(n)), np.median(x))
    assert np.isclose(_weighted_percentile(x, k), np.median(np.repeat(x, k)))

    X = rng.normal(size=(n, 3))
    assert np.allclose(_weighted_percentile(X, np.ones(n)), np.median(X, axis=0))

    assert np.isclose(_weighted_percentile(x, np.ones(n), 0), x.min())
    assert np.isclose(_weighted_percentile(x, np.ones(n), 100), x.max())
