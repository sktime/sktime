import numpy as np
import pytest

from sktime.detection._skchange.costs import L2Cost


def test_l2_cost_check_fixed_param():
    cost = L2Cost()
    X = np.array([[1, 2], [3, 4]])
    mean = cost._check_fixed_param(1.0, X)
    np.testing.assert_array_equal(mean, np.array([1.0]))

    mean = cost._check_fixed_param([1.0, 2.0], X)
    np.testing.assert_array_equal(mean, np.array([1.0, 2.0]))

    with pytest.raises(ValueError):
        cost._check_fixed_param([1.0, 2.0, 3.0], X)
