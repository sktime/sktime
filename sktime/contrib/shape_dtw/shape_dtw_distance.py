import numpy as np
from sktime.distances.elastic_cython import dtw_distance

def shape_dtw_distance(first, second, **kwargs):
    
    print(first)
    
    dist = dtw_distance(first,second)

    return dist