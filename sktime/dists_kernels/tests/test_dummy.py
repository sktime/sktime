import numpy as np
import pytest

from sktime.dists_kernels.dummy import ConstantPwTrafoPanel
from sktime.utils._testing.panel import _make_panel_X

def test_constant_pw_trafo_panel():
    """Test ConstantPwTrafoPanel returns constant distance matrix."""
    X1 = _make_panel_X(n_instances=3, n_columns=1, n_timepoints=5)
    X2 = _make_panel_X(n_instances=4, n_columns=1, n_timepoints=5)

    # Test with default constant (0)
    dist0 = ConstantPwTrafoPanel()
    res0 = dist0.transform(X1, X2)
    assert res0.shape == (3, 4)
    np.testing.assert_array_equal(res0, np.zeros((3, 4)))

    # Test with custom constant
    dist42 = ConstantPwTrafoPanel(constant=42.0)
    res42 = dist42.transform(X1, X2)
    assert res42.shape == (3, 4)
    np.testing.assert_array_equal(res42, np.ones((3, 4)) * 42.0)

def test_constant_pw_trafo_panel_self_transform():
    """Test ConstantPwTrafoPanel transform with only one argument."""
    X = _make_panel_X(n_instances=5, n_columns=1, n_timepoints=3)
    
    dist = ConstantPwTrafoPanel(constant=7.5)
    res = dist.transform(X)
    assert res.shape == (5, 5)
    np.testing.assert_array_equal(res, np.ones((5, 5)) * 7.5)

