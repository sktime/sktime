import numpy as np

from sktime.clustering.dbscan import TimeSeriesDBSCAN
from sktime.dists_kernels.dummy import ConstantPwTrafoPanel
from sktime.utils._testing.panel import _make_panel_X

def test_dbscan_dummy_distance():
    """Test TimeSeriesDBSCAN with a constant 0.0 distance matrix."""
    X = _make_panel_X(n_instances=4, n_columns=1, n_timepoints=5)
    
    dist = ConstantPwTrafoPanel(constant=0.0)
    dbscan = TimeSeriesDBSCAN(distance=dist, eps=0.1, min_samples=2)
    
    dbscan.fit(X)
    labels = dbscan.predict(X)
    
    assert labels.shape == (4,)
    np.testing.assert_array_equal(labels, np.zeros(4))
    
def test_dbscan_dummy_distance_large():
    """Test TimeSeriesDBSCAN with a constant 1.0 distance matrix."""
    X = _make_panel_X(n_instances=4, n_columns=1, n_timepoints=5)
    
    dist = ConstantPwTrafoPanel(constant=1.0)
    dbscan = TimeSeriesDBSCAN(distance=dist, eps=0.1, min_samples=2)
    
    dbscan.fit(X)
    labels = dbscan.predict(X)
    
    assert labels.shape == (4,)
    # Noise points in DBSCAN are labeled as -1
    np.testing.assert_array_equal(labels, -np.ones(4))
