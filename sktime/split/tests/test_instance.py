import numpy as np
import pytest
from sklearn.model_selection import KFold
import pandas as pd

from sktime.split import InstanceSplitter
from sktime.utils._testing.hierarchical import _make_hierarchical


def test_instance_splitter_kfold():
    """Test InstanceSplitter with sklearn KFold."""
    y = _make_hierarchical(hierarchy_levels=(4,), min_timepoints=3, max_timepoints=3)
    
    cv = KFold(n_splits=2)
    splitter = InstanceSplitter(cv)
    
    splits = list(splitter.split(y))
    
    assert len(splits) == 2
    assert splitter.get_n_splits(y) == 2
    
    train_0, test_0 = splits[0]
    
    assert len(train_0) == 6
    assert len(test_0) == 6
    assert len(np.intersect1d(train_0, test_0)) == 0
    np.testing.assert_array_equal(np.sort(np.concatenate([train_0, test_0])), np.arange(12))

def test_instance_splitter_single_index():
    """Test InstanceSplitter when passing a flat non-hierarchical series."""
    y = pd.Series([1, 2, 3, 4, 5])
    
    cv = KFold(n_splits=2)
    splitter = InstanceSplitter(cv)
    
    with pytest.raises(ValueError):
        list(splitter.split(y))
