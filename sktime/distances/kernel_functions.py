from elastic_cython import dtw_distance
import numpy as np
from tslearn.utils import to_time_series
from  scipy.spatial.distance import cdist


def GDS_pairs(s1,s2):
    s1 = to_time_series(s1)
    s2 = to_time_series(s2)
    dist = dtw_distance(s1, s2, 20)
    return np.exp(-dist)

def GDS_matrix(X,Y):
    M=cdist(X,Y,metric=GDS_pairs)
    return M

