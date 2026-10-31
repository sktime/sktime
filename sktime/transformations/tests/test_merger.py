import numpy as np
import pytest

from sktime.transformations.merger import Merger
from sktime.utils._testing.panel import _make_panel


def test_merger_stride_0_mean():
    """Test Merger aggregation with stride=0 and method='mean'."""
    # Create input panel data of shape (2 instances, 1 variable, 3 timepoints)
    X = np.array([
        [[1.0, 2.0, 3.0]],
        [[3.0, 4.0, 5.0]]
    ])
    merger = Merger(method="mean", stride=0)
    res = merger.fit_transform(X)
    
    # In sktime, panel transform output is cast back to a pandas DataFrame/Series
    # We compare the underlying values
    assert res.shape == (3, 1)
    np.testing.assert_array_equal(res.values, np.array([[2.0], [3.0], [4.0]]))


def test_merger_stride_1_mean():
    """Test Merger aggregation with stride=1 and method='mean'."""
    X = np.array([
        [[1.0, 2.0, 3.0]],
        [[3.0, 4.0, 5.0]]
    ])
    merger = Merger(method="mean", stride=1)
    res = merger.fit_transform(X)
    
    assert res.shape == (4, 1)
    np.testing.assert_array_equal(res.values, np.array([[1.0], [2.5], [3.5], [5.0]]))


def test_merger_stride_1_median():
    """Test Merger aggregation with stride=1 and method='median'."""
    X = np.array([
        [[1.0, 2.0, 3.0]],
        [[3.0, 10.0, 5.0]],
        [[5.0, 6.0, 7.0]]
    ])
    merger = Merger(method="median", stride=1)
    res = merger.fit_transform(X)
    
    # instance 0: 1  2  3
    # instance 1:    3 10  5
    # instance 2:       5  6  7
    # Column medians:
    # 0: median([1]) = 1.0
    # 1: median([2, 3]) = 2.5
    # 2: median([3, 10, 5]) = 5.0
    # 3: median([5, 6]) = 5.5
    # 4: median([7]) = 7.0
    
    assert res.shape == (5, 1)
    np.testing.assert_array_equal(res.values, np.array([[1.0], [2.5], [5.0], [5.5], [7.0]]))

def test_merger_invalid_method():
    """Test Merger raises ValueError on invalid method."""
    with pytest.raises(ValueError, match="must be 'mean' or 'median'"):
        Merger(method="invalid_method")

