import numpy as np
from tslearn.utils import to_time_series
from  scipy.spatial.distance import cdist
from elastic_cython import (
    ddtw_distance, dtw_distance, erp_distance, lcss_distance, msm_distance, wddtw_distance, wdtw_distance,
    )

def GDS_pairs(s1,s2):
    s1 = to_time_series(s1)
    s2 = to_time_series(s2)
    dist = dtw_distance(s1, s2, 0)
    return np.exp(-dist)

def GDS_matrix(X,Y):
    M=cdist(X,Y,metric=GDS_pairs)
    return M

def distance_matrix(distance_measure, **kwargs):

    def distance(a, b):
        a = to_time_series(a)
        b = to_time_series(b)
        dist = distance_measure(a, b, **kwargs)
        return np.exp(-dist)

    def build_matrix(X, Y):
        matrix = cdist(X, Y, metric=distance)
        return matrix

    return build_matrix

