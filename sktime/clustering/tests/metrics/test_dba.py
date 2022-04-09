
import numpy as np

from sktime.clustering.metrics.medoids import medoids
from sktime.clustering.metrics.averaging._dba import dba
from sktime.distances.tests._utils import create_test_distance_numpy


def test_dba():
    """Test medoids."""
    X = create_test_distance_numpy(10, 3, 3, random_state=2)
    test_dba = dba(X)
    joe = ''