import numpy as np
from tslearn.utils import to_time_series
from  scipy.spatial.distance import cdist
from elastic_cython import (
    ddtw_distance, dtw_distance, erp_distance, lcss_distance, msm_distance, wddtw_distance, wdtw_distance,
    )



#Kernels for dtw distance
def GDS_dtw_pairs(s1,s2,sigma,w):
    s1 = to_time_series(s1)
    s2 = to_time_series(s2)
    dist = dtw_distance(s1, s2, w)
    return np.exp(-(dist**2) / (sigma**2))


def GDS_dtw_matrix(X,Y,sigma,w):
    M=cdist(X,Y,metric=GDS_dtw_pairs,sigma=sigma,w=w)
    return M


#Kernels for wdtw distance
def GDS_wdtw_pairs(s1,s2,sigma,g):
    s1 = to_time_series(s1)
    s2 = to_time_series(s2)
    dist = wdtw_distance(s1, s2, g)
    return np.exp(-(dist**2) / (sigma**2))


def GDS_wdtw_matrix(X,Y,sigma,g):
    M=cdist(X,Y,metric=GDS_wdtw_pairs,sigma=sigma,g=g)
    return M





#Kernels for ddtw distance
def GDS_ddtw_pairs(s1,s2,sigma,w):
    s1 = to_time_series(s1)
    s2 = to_time_series(s2)
    dist = ddtw_distance(s1, s2, w)
    return np.exp(-(dist**2) / (sigma**2))


def GDS_ddtw_matrix(X,Y,sigma,w):
    M=cdist(X,Y,metric=GDS_ddtw_pairs,sigma=sigma,w=w)
    return M




#Kernels for wddtw distance
def GDS_wddtw_pairs(s1,s2,sigma,g):
    s1 = to_time_series(s1)
    s2 = to_time_series(s2)
    dist = wddtw_distance(s1, s2, g)
    return np.exp(-(dist**2) / (sigma**2))


def GDS_wddtw_matrix(X,Y,sigma,g):
    M=cdist(X,Y,metric=GDS_wddtw_pairs,sigma=sigma,g=g)
    return M




#Kernels for msm distance
def GDS_msm_pairs(s1,s2,sigma,c):
    s1 = to_time_series(s1)
    s2 = to_time_series(s2)
    dist = msm_distance(s1, s2,c)
    return np.exp(-(dist**2) / (sigma**2))


def GDS_msm_matrix(X,Y,sigma,c):
    M=cdist(X,Y,metric=GDS_msm_pairs,sigma=sigma,c=c)
    return M




#Kernels for lcss distance
def GDS_lcss_pairs(s1,s2,sigma, delta, epsilon):
    s1 = to_time_series(s1)
    s2 = to_time_series(s2)
    dist = lcss_distance(s1, s2,delta, epsilon)
    return np.exp(-(dist**2) / (sigma**2))


def GDS_lcss_matrix(X,Y,sigma,delta, epsilon):
    M=cdist(X,Y,metric=GDS_lcss_pairs,sigma=sigma, delta=delta, epsilon=epsilon)
    return M






#Kernels for erp distance
def GDS_erp_pairs(s1,s2,sigma, band_size, g):
    s1 = to_time_series(s1)
    s2 = to_time_series(s2)
    dist = erp_distance(s1, s2,band_size, g)
    return np.exp(-(dist**2) / (sigma**2))


def GDS_erp_matrix(X,Y,sigma, band_size, g):
    M=cdist(X,Y,metric=GDS_erp_pairs,sigma=sigma,band_size=band_size, g=g)
    return M



















def distance_matrix(distance_measure, **kwargs):
    sigma = kwargs['sigma']


    def distance(a, b, **kwargs):
        a = to_time_series(a)
        b = to_time_series(b)
        dist = distance_measure(a, b, **kwargs)
        return np.exp(-(dist**2) / sigma**2)


    def build_matrix(X, Y):
        matrix = cdist(X, Y, metric=distance)
        return matrix

    return build_matrix

