# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, nonecheck=False,
# TODO remove in v0.10.0
# the functionality in this file is depreciated and to be replaced with a version
# based on numba.
# believe it or not, the below variable is required for cython to compile properly. A
# global python variable hooks into a c global variable. Without this functions do
# not compile properly!
# the functionality in this file is depreciated and to be replaced with a version
# based on numba.
STUFF = "Hi"  # https://stackoverflow.com/questions/8024805/cython-compiled-c-extension-importerror-dynamic-module-does-not-define-init-fu

import numpy as np

cimport numpy as np

np.import_array()

from libc.float cimport DBL_MAX
from libc.math cimport exp, fabs, sqrt

from warnings import warn


cdef inline double min_c(double a, double b):
    """min c docstring."""
    return a if a <= b else b
cdef inline int max_c_int(int a, int b):
    """max c int docstring."""
    return a if a >= b else b
cdef inline int min_c_int(int a, int b):
    """min c int docstring."""
    return a if a <= b else b

# TO-DO: convert DDTW and WDDTW to use slope-based derivatives rather than np.diff

# Adapted version of the DTW algorithm taken from https://github.com/hfawaz/aaltd18/tree/master/distances/dtw
#
# @InProceedings{IsmailFawaz2018,
#   Title                    = {Data augmentation using synthetic data for time series classification with deep residual networks},
#   Author                   = {Ismail Fawaz, Hassan and Forestier, Germain and Weber, Jonathan and Idoumghar, Lhassane and Muller, Pierre-Alain},
#   Booktitle                = {International Workshop on Advanced Analytics and Learning on Temporal Data, {ECML} {PKDD}},
#   Year                     = {2018}
# }
#
# Thanks in particular to GitHub @hfawaz!
#
# it takes as argument two time series with shape (l,m) where l is the length
# of the time series and m is the number of dimensions
# for multivariate time series
# even if we have univariate time series, we should have a shape equal to (l,1)
# the w argument corresponds to the length of the warping window in percentage of
# the smallest length of the time series min(x,y) - if negative then no warping window
# this function assumes that x is shorter than y
def dtw_distance(
        np.ndarray[double, ndim=2] x,
        np.ndarray[double, ndim=2] y,
        double w = -1):
    """Cython version of DTW distance.

    Arguments
    ---------
    x : np.array
    y : np.array
    w : weight

    Returns
    -------
    float
    """
    warn("Cython DTW is deprecated from V0.10")
    # make sure x is shorter than y
    # if not permute
    cdef np.ndarray[double, ndim=2] X = x
    cdef np.ndarray[double, ndim=2] Y = y
    cdef np.ndarray[double, ndim=2] t

    if len(X)>len(Y):
        t = X
        X = Y
        Y = t

    cdef int r,c, im,jm,lx, jstart, jstop, idx_inf_left, ly, band
    cdef Py_ssize_t i, j
    cdef double curr

    lx = len(X)
    ly = len(Y)
    r = lx + 1
    c = ly +1

    if w < 0:
        band = max_c_int(lx,ly)
    else:
        band = int(w*max_c_int(lx,ly))
    cdef np.ndarray[double, ndim=2] D = np.zeros((r,c), dtype=np.float64)

    D[0,1:] = DBL_MAX
    D[1:,0] = DBL_MAX

    # inspired by https://stackoverflow.com/a/27948463/9234713
    D[1:,1:] = np.square(X[:,np.newaxis]-Y).sum(axis=2).astype(np.float64)

    for i in range(1,r):
        jstart = max_c_int(1 , i-band)
        jstop = min_c_int(c , i+band+1)
        idx_inf_left = i-band-1

        if idx_inf_left >= 0 :
            D[i,idx_inf_left] = DBL_MAX

        for j in range(jstart,jstop):
            im = i-1
            jm = j-1
            D[i,j] = D[i,j] + min_c(min_c(D[im,j],D[i,jm]),D[im,jm])

        if jstop < c:
            D[i][jstop] = DBL_MAX

    return D[lx,ly]

def wdtw_distance(np.ndarray[double, ndim=2] x, np.ndarray[double, ndim=2] y , double g = 0.05):

    warn("Cython WDTW is deprecated from V0.10")
    # make sure x is shorter than y
    # if not permute
    cdef np.ndarray[double, ndim=2] X = x
    cdef np.ndarray[double, ndim=2] Y = y
    cdef np.ndarray[double, ndim=2] t

    if len(X)>len(Y):
        t = X
        X = Y
        Y = t

    cdef int r,c, im,jm,lx, jstart, jstop, idx_inf_left, ly, band
    cdef Py_ssize_t i, j
    cdef double curr

    lx = len(X)
    ly = len(Y)
    r = lx + 1
    c = ly +1

    # get weights with cdef helper function
    cdef np.ndarray[double, ndim=1] weight_vector = _wdtw_calc_weights(lx,g)
    cdef np.ndarray[double, ndim=2] D = np.zeros((r,c), dtype=np.float64)

    D[0,1:] = DBL_MAX
    D[1:,0] = DBL_MAX

    D[1:,1:] = np.square(X[:,np.newaxis]-Y).sum(axis=2).astype(np.float64) # inspired by https://stackoverflow.com/a/27948463/9234713

    for row in range(1,r):
        for column in range(1,c):
            D[row,column] *= weight_vector[<int>fabs(row-column)]

    for i in range(1,r):
        jstart = max_c_int(1 , i-ly)
        jstop = min_c_int(c , i+ly+1)
        idx_inf_left = i-ly-1

        if idx_inf_left >= 0 :
            D[i,idx_inf_left] = DBL_MAX

        for j in range(jstart,jstop):
            im = i-1
            jm = j-1
            # D[i,j] = min_c(min_c(D[im,j],D[i,jm]),D[im,jm]) + weight_vector[j-i]*D[i,j]
            D[i,j] = min_c(min_c(D[im,j],D[i,jm]),D[im,jm]) + D[i,j]

        if jstop < c:
            D[i][jstop] = DBL_MAX

    return D[lx,ly]

# note - this implementation is more convenient for general use but it is more efficient
# for standalone use to transform the data once, then use DTW on the transformed data
def ddtw_distance(np.ndarray[double, ndim=2] x, np.ndarray[double, ndim=2] y , double w = -1):
    warn("Cython DDTW is deprecated from V0.10")
    return dtw_distance(np.diff(x.T).T,np.diff(y.T).T,w)


# note - this implementation is more convenient for use in ensembles, etc., but it is more efficient
# for standalone use to transform the data once, then use WDTW on the transformed data
def wddtw_distance(np.ndarray[double, ndim=2] x, np.ndarray[double, ndim=2] y , double g = 0):
    warn("Cython WDDTW is deprecated from V0.10")
    return wdtw_distance(np.diff(x.T).T,np.diff(y.T).T,g)

def msm_distance(np.ndarray[double, ndim=2] x, np.ndarray[double, ndim=2] y, double c = 1, int dim_to_use = 0):
    warn("Cython MSM is deprecated from V0.10")
    cdef np.ndarray[double, ndim=2] first = x
    cdef np.ndarray[double, ndim=2] second = y
    cdef np.ndarray[double, ndim=2] temp
    if len(first) > len(second):
        temp = first
        first = second
        second = temp

    cdef Py_ssize_t i, j
    cdef int rows,columns, im, jm, lx, jstart, jstop, idx_inf_left, ly, m, n
    cdef double curr, d1, d2, d3, t

    m = len(first)
    n = len(second)

    cdef np.ndarray[double, ndim=2] cost = np.zeros((m,n),dtype=np.float64)

    # Initialization
    cost[0, 0] = fabs(first[0,dim_to_use] - second[0,dim_to_use])
    for i in range(1,m):
        # cost[i, 0] = cost[i - 1, 0] + _msm_calc_cost(first[i], first[i - 1], second[0], c)
        t = _msm_calc_cost(first[i,dim_to_use], first[i - 1,dim_to_use], second[0,dim_to_use], c)
        cost[i, 0] = cost[i - 1, 0] + t

    for i in range(1, n):
        # cost[0, i] = cost[0, i - 1] + _msm_calc_cost(second[i], first[0], second[i - 1], c)
        t = _msm_calc_cost(second[i,dim_to_use], first[0,dim_to_use], second[i - 1,dim_to_use], c)
        cost[0, i] = cost[0, i - 1] + t

     # Main Loop
    for i in range(1, m):
        for j in range(1, n):
            d1 = cost[i - 1, j - 1] + fabs(first[i,dim_to_use] - second[j,dim_to_use])
            d2 = cost[i - 1, j] + _msm_calc_cost(first[i,dim_to_use], first[i - 1,dim_to_use], second[j,dim_to_use], c)
            d3 = cost[i, j - 1] + _msm_calc_cost(second[j,dim_to_use], first[i,dim_to_use], second[j - 1,dim_to_use], c)
            cost[i, j] = min_c(min_c(d1,d2),d3)

    return cost[m - 1, n - 1]

#
cdef _msm_calc_cost(double new_point, double x, double y, double c):
    cdef double dist = 0

    if ((x <= new_point) and (new_point <= y)) or ((y <= new_point) and (new_point <= x)):
        return c
    else:
        return c + min_c(fabs(new_point - x), fabs(new_point - y))

def lcss_distance(np.ndarray[double, ndim=2] x, np.ndarray[double, ndim=2] y, int delta
= 3, double epsilon = 0.05,
                  int dim_to_use = 0):
    warn("Cython LCSS is deprecated from V0.10")

    cdef np.ndarray[double, ndim=2] first = x
    cdef np.ndarray[double, ndim=2] second = y
    cdef np.ndarray[double, ndim=2] temp

    cdef int m, n, max_val
    cdef Py_ssize_t i, j

    if len(first) > len(second):
        temp = first
        first = second
        second = temp

    m = len(first)
    n = len(second)

    cdef np.ndarray[int, ndim=2] lcss = np.zeros([m + 1, n + 1], dtype=np.int32)

    for i in range(m):
        for j in range(i - delta, i + delta + 1):
            if j < 0:
                j = -1
            elif j >= n:
                j = i + delta
            elif second[j, dim_to_use] + epsilon >= first[i, dim_to_use] >= second[j, dim_to_use] - epsilon:
                lcss[i + 1, j + 1] = lcss[i,j] + 1
            elif lcss[i,j + 1] > lcss[i + 1,j]:
                lcss[i + 1,j + 1] = lcss[i,j + 1]
            else:
                lcss[i + 1,j + 1] = lcss[i + 1, j]

    max_val = -1
    for i in range(1, len(lcss[len(lcss) - 1])):
        if lcss[len(lcss) - 1, i] > max_val:
            max_val = lcss[len(lcss) - 1, i]

    return 1 - (max_val / m)

#cython: boundscheck=False, wraparound=False, nonecheck=False
def twe_distance(np.ndarray[double, ndim=2] ta, np.ndarray[double, ndim=2] tb, double penalty = 1,
                 double stiffness = 1):
    warn("Cython TWE is deprecated from V0.10")
    cdef int dim = ta.shape[1] - 1
    cdef double dist, disti1, distj1
    cdef np.ndarray[double, ndim=1] tsa = np.zeros([len(ta) + 1], dtype=np.double)
    cdef np.ndarray[double, ndim=1] tsb = np.zeros([len(tb) + 1], dtype=np.double)

    cdef int r = len(ta)
    cdef int c = len(tb)
    cdef int i, j, k

    cdef np.ndarray[double, ndim=2] D = np.zeros([r + 1, c + 1], dtype=np.double)
    cdef np.ndarray[double, ndim=1] Di1 = np.zeros([r + 1], dtype=np.double)
    cdef np.ndarray[double, ndim=1] Dj1 = np.zeros([c + 1], dtype=np.double)

    for i in range(0, len(tsa)):
        tsa[i] = (i + 1)
    for i in range(0, len(tsb)):
        tsb[i] = (i + 1)

    # local costs initializations
    for j in range(1, c + 1):
        distj1 = 0
        for k in range(0, dim + 1):
            if j > 1:
                #CHANGE AJB 8/1/16: Only use power of 2 for speed up,
                distj1 += (tb[j - 2][k] - tb[j - 1][k]) * (tb[j - 2][k] - tb[j - 1][k])
            # OLD VERSION                    distj1+=Math.pow(Math.abs(tb[j-2][k]-tb[j-1][k]),degree)
            # in c:               distj1+=pow(fabs(tb[j-2][k]-tb[j-1][k]),degree)
            else:
                distj1 += tb[j - 1][k] * tb[j - 1][k]
        # OLD              		distj1+=Math.pow(Math.abs(tb[j-1][k]),degree)
        Dj1[j] = (distj1)

    for i in range(1, r + 1):
        disti1 = 0
        for k in range(0, dim + 1):
            if i > 1:
                disti1 += (ta[i - 2][k] - ta[i - 1][k]) * (ta[i - 2][k] - ta[i - 1][k])
            # OLD                 disti1+=Math.pow(Math.abs(ta[i-2][k]-ta[i-1][k]),degree)
            else:
                disti1 += (ta[i - 1][k]) * (ta[i - 1][k])
        # OLD                  disti1+=Math.pow(Math.abs(ta[i-1][k]),degree)
        Di1[i] = (disti1)

        for j in range(1, c + 1):
            dist = 0
            for k in range(0, dim + 1):
                dist += (ta[i - 1][k] - tb[j - 1][k]) * (ta[i - 1][k] - tb[j - 1][k])
                #                  dist+=Math.pow(Math.abs(ta[i-1][k]-tb[j-1][k]),degree)
                if i > 1 and j > 1:
                    dist += (ta[i - 2][k] - tb[j - 2][k]) * (ta[i - 2][k] - tb[j - 2][k])
            #                    dist+=Math.pow(Math.abs(ta[i-2][k]-tb[j-2][k]),degree)
            D[i][j] = dist

    # border of the cost matrix initialization
    D[0][0] = 0
    for i in range(1, r + 1):
        D[i][0] = D[i - 1][0] + Di1[i]
    for j in range(1, c + 1):
        D[0][j] = D[0][j - 1] + Dj1[j]

    cdef double dmin, htrans, dist0
    cdef int iback

    for i in range(1, r + 1):
        for j in range(1, c + 1):
            htrans = np.abs((tsa[i - 1] - tsb[j - 1]))
            if j > 1 and i > 1:
                htrans += np.abs((tsa[i - 2] - tsb[j - 2]))
            dist0 = D[i - 1][j - 1] + stiffness * htrans + D[i][j]
            dmin = dist0
            if i > 1:
                htrans = ((tsa[i - 1] - tsa[i - 2]))
            else:
                htrans = tsa[i - 1]
            dist = Di1[i] + D[i - 1][j] + penalty + stiffness * htrans
            if dmin > dist:
                dmin = dist
            if j > 1:
                htrans = (tsb[j - 1] - tsb[j - 2])
            else:
                htrans = tsb[j - 1]
            dist = Dj1[j] + D[i][j - 1] + penalty + stiffness * htrans
            if dmin > dist:
                dmin = dist
            D[i][j] = dmin

    dist = D[r][c]
    return dist

def erp_distance(np.ndarray[double, ndim=2] x, np.ndarray[double, ndim=2] y, double band_size = 5, double g = 0,
                 int dim_to_use = 0):
    """
    Adapted from:
        This file is part of ELKI:
        Environment for Developing KDD-Applications Supported by Index-Structures

        Copyright (C) 2011
        Ludwig-Maximilians-UniversitÃ¤t MÃ¼nchen
        Lehr- und Forschungseinheit fÃ¼r Datenbanksystemethe
        ELKI Development Team

        This program is free software: you can redistribute it and/or modify
        it under the terms of the GNU Affero General Public License as published by
        the Free Software Foundation, either version 3 of the License, or
        (at your option) any later version.

        This program is distributed in the hope that it will be useful,
        but WITHOUT ANY WARRANTY; without even the implied warranty of
        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
        GNU Affero General Public License for more details.

        You should have received a copy of the GNU Affero General Public License
        along with this program.  If not, see <http://www.gnu.org/licenses/>.
    """
    warn("Cython ERP is deprecated from V0.10")
    cdef np.ndarray[double, ndim=2] first = x
    cdef np.ndarray[double, ndim=2] second = y
    cdef np.ndarray[double, ndim=2] t

    cdef Py_ssize_t i, j
    cdef int m, n, band, left, right
    cdef double val1, val2, diff, d1, d2, d12, dist1, dist2, dist12, cost

    if len(first) > len(second):
        t = first
        first = second
        second = t

    m = len(first)
    n = len(second)

    cdef np.ndarray[double, ndim=1] curr = np.zeros(m)
    cdef np.ndarray[double, ndim=1] prev = np.zeros(m)
    cdef np.ndarray[double, ndim=1] temp = np.zeros(m)

    band = np.ceil(band_size * m)
    for i in range(m):
        temp = prev
        prev = curr
        curr = temp

        left = i-(band+1)
        if left < 0:
            left = 0

        right = i + band + 1
        if right > m-1:
            right = m-1
        for j in range(left,right+1):
            if fabs(i-j) <= band:
                d1 = sqrt((first[i,dim_to_use]-g)*(first[i,dim_to_use]-g))
                d2 = sqrt((second[j,dim_to_use]-g)*(second[j,dim_to_use]-g))
                d3 = sqrt((first[i,dim_to_use]-second[j,dim_to_use])*(first[i,dim_to_use]-second[j,dim_to_use]))
                d1*=d1
                d2*=d2
                d3*=d3
                cost = 0
                if i+j!=0:
                    # print("here")
                    if i == 0 or ((j != 0) and (((prev[j - 1] + d3) > (curr[j - 1] + d2)) and ((curr[j - 1] + d2) < (prev[j] + d1)))):
                        # # del
                        cost = curr[j - 1] + d2
                    elif (j == 0) or ((i != 0) and (((prev[j - 1] + d3) > (prev[j] + d1)) and ((prev[j] + d1) < (curr[j - 1] + d2)))):
                        # # ins
                        cost = prev[j] + d1
                    else:
                        # # match
                        cost = prev[j - 1] + d3
                    # print(cost)
                curr[j] = cost
            else:
                curr[j] = np.inf

    return sqrt(curr[m-1])

cdef _get_der(np.ndarray[double, ndim=2] x):
    cdef np.ndarray[double, ndim=2] der_x = np.empty((len(x),len(x[0])-1))
    cdef int i
    for i in range(len(x)):
        der_x[i] = np.diff(x[i])
    return der_x

cdef _wdtw_calc_weights(int len_x, double g):
    cdef np.ndarray[double, ndim=1] weight_vector = np.zeros(len_x)
    cdef int i
    for i in range(len_x):
        # weight_vector[i] = 1/(1+np.exp(-g*(i-len_x/2)))
        weight_vector[i] = 1/(1+exp(-g*(i-len_x/2)))
    return weight_vector
