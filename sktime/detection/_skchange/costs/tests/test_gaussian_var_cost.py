import numpy as np
import pytest

from sktime.detection._skchange.costs import GaussianCost


@pytest.mark.parametrize(
    "mean, var",
    [
        (0.0, np.array([1.0, 2.0, 3.0])),  # Too many entries compared to data
        (0.0, -1.0),  # Negative variance
    ],
)
def test_invalid_fixed_covariance(mean, var):
    """Test that invalid fixed covariance matrix raises errors."""
    X = np.random.randn(100, 2)
    cost = GaussianCost(param=(mean, var))
    with pytest.raises(ValueError):
        cost.fit(X)
