import numpy as np

from sktime.dists_kernels.algebra import CombinedDistance
from sktime.dists_kernels.dummy import ConstantPwTrafoPanel
from sktime.utils._testing.panel import _make_panel_X


def test_combined_distance_add():
    """Test CombinedDistance with addition."""
    X1 = _make_panel_X(n_instances=3, n_columns=1, n_timepoints=5)
    X2 = _make_panel_X(n_instances=2, n_columns=1, n_timepoints=5)

    t1 = ConstantPwTrafoPanel(constant=3.0)
    t2 = ConstantPwTrafoPanel(constant=4.0)

    combined = CombinedDistance([("t1", t1), ("t2", t2)], operation="+")

    distmat = combined.transform(X1, X2)

    assert distmat.shape == (3, 2)
    np.testing.assert_array_equal(distmat, np.ones((3, 2)) * 7.0)


def test_combined_distance_multiply():
    """Test CombinedDistance with multiplication."""
    X1 = _make_panel_X(n_instances=4, n_columns=1, n_timepoints=3)

    t1 = ConstantPwTrafoPanel(constant=2.0)
    t2 = ConstantPwTrafoPanel(constant=5.0)

    combined = CombinedDistance([("t1", t1), ("t2", t2)], operation="*")

    distmat = combined.transform(X1)

    assert distmat.shape == (4, 4)
    np.testing.assert_array_equal(distmat, np.ones((4, 4)) * 10.0)


def test_combined_distance_default_mean():
    """Test CombinedDistance defaults to mean."""
    X1 = _make_panel_X(n_instances=2, n_columns=1, n_timepoints=4)

    t1 = ConstantPwTrafoPanel(constant=2.0)
    t2 = ConstantPwTrafoPanel(constant=8.0)

    combined = CombinedDistance([("t1", t1), ("t2", t2)])

    distmat = combined.transform(X1)

    np.testing.assert_array_equal(distmat, np.ones((2, 2)) * 5.0)
