# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

# This file is part of wildboar
#
# wildboar is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# wildboar is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

# Authors: Isak Karlsson

# This implementation is heaviliy inspired by the UCRSuite.
#
# References
#
#  - Rakthanmanon, et al., Searching and Mining Trillions of Time
#    Series Subsequences under Dynamic Time Warping (2012)
#  - http://www.cs.ucr.edu/~eamonn/UCRsuite.html

from libc.stdlib cimport malloc
from libc.stdlib cimport free
from libc.string cimport memset

from libc.math cimport INFINITY
from libc.math cimport sqrt
from libc.math cimport fabs
from libc.math cimport floor

from sktime.transformers.distances._distance cimport TSDatabase

from sktime.transformers.distances._distance cimport ShapeletInfo
from sktime.transformers.distances._distance cimport ScaledDistanceMeasure
from sktime.transformers.distances._distance cimport Shapelet

cdef void deque_init(Deque* c, size_t capacity) nogil:
    c[0].capacity = capacity
    c[0].size = 0
    c[0].queue = <size_t*>malloc(sizeof(size_t) * capacity)
    c[0].front = 0
    c[0].back = capacity - 1


cdef void deque_reset(Deque* c) nogil:
    c[0].size = 0
    c[0].front = 0
    c[0].back = c[0].capacity - 1


cdef void deque_destroy(Deque* c) nogil:
    free(c[0].queue)


cdef void deque_push_back(Deque* c, size_t v) nogil:
    c[0].queue[c[0].back] = v
    c[0].back -= 1
    if c[0].back < 0:
        c[0].back = c[0].capacity - 1

    c[0].size += 1


cdef void deque_pop_front(Deque* c) nogil:
    c[0].front -= 1
    if c[0].front < 0:
        c[0].front = c[0].capacity - 1
    c[0].size -= 1


cdef void deque_pop_back(Deque* c) nogil:
    c[0].back = (c[0].back + 1) % c[0].capacity
    c[0].size -= 1


cdef size_t deque_front(Deque* c) nogil:
    cdef int tmp = c[0].front - 1
    if tmp < 0:
        tmp = c[0].capacity - 1
    return c[0].queue[tmp]


cdef size_t deque_back(Deque* c) nogil:
    cdef int tmp = (c[0].back + 1) % c[0].capacity
    return c[0].queue[tmp]


cdef bint deque_empty(Deque* c) nogil:
    return c[0].size == 0


cdef size_t deque_size(Deque* c) nogil:
    return c[0].size


cdef void find_min_max(size_t offset, size_t stride, size_t length,
                       double* T, size_t r, double* lower, double* upper,
                       Deque* dl, Deque* du) nogil:
    cdef size_t i
    cdef size_t k

    cdef double current, prev

    deque_reset(du)
    deque_reset(dl)

    deque_push_back(du, 0)
    deque_push_back(dl, 0)

    for i in range(1, length):
        if i > r:
            k = i - r - 1
            upper[k] = T[offset + stride * deque_front(du)]
            lower[k] = T[offset + stride * deque_front(dl)]

        current = T[offset + stride * i]
        prev = T[offset + stride * (i - 1)]
        if current > prev:
            deque_pop_back(du)
            while (not deque_empty(du) and
                   current > T[offset + stride * deque_back(du)]):
                deque_pop_back(du)
        else:
            deque_pop_back(dl)
            while (not deque_empty(dl) and
                   current < T[offset + stride * deque_back(dl)]):
                deque_pop_back(dl)

        deque_push_back(du, i)
        deque_push_back(dl, i)

        if i == 2 * r + 1 + deque_front(du):
            deque_pop_front(du)
        elif i == 2 * r + 1 + deque_front(dl):
            deque_pop_front(dl)

    for i in range(length, length + r + 1):
        upper[i - r - 1] = T[offset + stride * deque_front(du)]
        lower[i - r - 1] = T[offset + stride * deque_front(dl)]

        if i - deque_front(du) >= 2 * r + 1:
            deque_pop_front(du)
        if i - deque_front(dl) >= 2 * r + 1:
            deque_pop_front(dl)


cdef inline double dist(double x, double y) nogil:
    cdef double s = x - y
    return s * s


cdef double constant_lower_bound(size_t s_offset, size_t s_stride, double* S,
                                 double s_mean, double s_std, size_t t_offset,
                                 size_t t_stride, double* T, double t_mean,
                                 double t_std, size_t length,
                                 double best_dist) nogil:
    cdef double t_x0, t_y0, s_x0, s_y0
    cdef double t_x1, ty1, s_x1, s_y1
    cdef double t_x2, t_y2, s_x2, s_y2
    cdef double distance, min_dist

    # first and last in T
    t_x0 = (T[t_offset] - t_mean) / t_std
    t_y0 = (T[t_offset + t_stride * (length - 1)] - t_mean) / t_std

    # first and last in S
    s_x0 = (S[s_offset] - s_mean) / s_std
    s_y0 = (S[s_offset + s_stride * (length - 1)] - s_mean) / s_std

    min_dist = dist(t_x0, s_x0) + dist(t_y0, s_y0)
    if min_dist >= best_dist:
        return min_dist

    t_x1 = (T[t_offset + t_stride * 1] - t_mean) / t_std
    s_x1 = (S[s_offset + s_stride * 1] - s_mean) / s_std
    min_dist += min(
        min(dist(t_x1, s_x0), dist(t_x0, s_x1)),
        dist(t_x1, s_x1))

    if min_dist >= best_dist:
        return min_dist

    t_y1 = (T[t_offset + t_stride * (length - 2)] - t_mean) / t_std
    s_y1 = (S[s_offset + s_stride * (length - 2)] - s_mean) / s_std
    min_dist += min(
        min(dist(t_y1, s_y1), dist(t_y0, s_y1)),
        dist(t_y1, s_y1))

    if min_dist >= best_dist:
        return min_dist

    t_x2 = (T[t_offset + t_stride * 2] - t_mean) / t_std
    s_x2 = (S[s_offset + s_stride * 2] - s_mean) / s_std
    min_dist += min(min(dist(t_x0, s_x2),
                        min(dist(t_x1, s_x2),
                            dist(t_x2, s_x2)),
                        dist(t_x2, s_x1)),
                    dist(t_x2, s_x0))

    if min_dist >= best_dist:
        return min_dist

    t_y2 = (T[t_offset + t_stride * (length - 3)] - t_mean) / t_std
    s_y2 = (S[s_offset + s_stride * (length - 3)] - s_mean) / s_std

    min_dist += min(min(dist(t_y0, s_y2),
                        min(dist(t_y1, s_y2),
                            dist(t_y2, s_y2)),
                        dist(t_y2, s_y1)),
                    dist(t_y2, s_y0))

    return min_dist

cdef double cumulative_bound(size_t offset, size_t stride, size_t length,
                             double mean, double std, double* T,
                             double lu_mean, double lu_std, double* lower,
                             double* upper, double* cb, double best_so_far) nogil:
    cdef double min_dist = 0
    cdef double x, d, us, ls
    cdef size_t i

    for i in range(0, length):
        if min_dist >= best_so_far:
            break

        x = (T[offset + stride * i] - mean) / std
        us = (upper[i] - lu_mean) / lu_std
        ls = (lower[i] - lu_mean) / lu_std
        if x > us:
            d = dist(x, us)
        elif x < ls:
            d = dist(x, ls)
        else:
            d = 0

        min_dist += d
        cb[i] = d
    return min_dist


cdef inline double inner_dtw(size_t s_offset, size_t s_stride, int s_length,
                             double s_mean, double s_std, double* S,
                             double mean, double std, size_t x_offset,
                             double* X_buffer, int r, double* cb,
                             double* cost, double* cost_prev,
                             double min_dist) nogil:
    cdef int i = 0
    cdef int j = 0
    cdef int k = 0

    cdef double x
    cdef double y
    cdef double z
    cdef double min_cost, distance

    cdef double* cost_tmp
    for i in range(0, 2 * r + 1):
        cost[i] = INFINITY
        cost_prev[i] = INFINITY

    for i in range(0, s_length):
        k = max(0, r - i)
        min_cost = INFINITY
        for j in range(max(0, i - r), min(s_length, i + r + 1)):
            if i == 0 and j == 0:
                min_cost = dist((S[s_offset] - s_mean) / s_std,
                                (X_buffer[x_offset] - mean)  / std)
                cost[k] = min_cost
            else:
                if j - 1 < 0 or k - 1 < 0:
                    y = INFINITY
                else:
                    y = cost[k - 1]

                if i - 1 < 0 or k + 1 > 2 * r:
                    x = INFINITY
                else:
                    x = cost_prev[k + 1]

                if i - 1 < 0 or j - 1 < 0:
                    z = INFINITY
                else:
                    z = cost_prev[k]

                distance = dist((S[s_offset + s_stride * i] - s_mean) / s_std,
                                (X_buffer[x_offset + j]- mean) / std)
                cost[k] = min(min(x, y), z) + distance
                if cost[k] < min_cost:
                    min_cost = cost[k]

            k += 1

        if i + r < s_length - 1 and min_cost + cb[i + r + 1] >= min_dist:
            return min_cost + cb[i + r + 1]

        cost_tmp = cost
        cost = cost_prev
        cost_prev = cost_tmp
    return cost_prev[k - 1]


cdef double scaled_dtw_distance(size_t s_offset,
                                size_t s_stride,
                                size_t s_length,
                                double s_mean,
                                double s_std,
                                double* S,
                                size_t t_offset,
                                size_t t_stride,
                                size_t t_length,
                                double* T,
                                size_t r,
                                double* X_buffer,
                                double* cost,
                                double* cost_prev,
                                double* s_lower,
                                double* s_upper,
                                double* t_lower,
                                double* t_upper,
                                double* cb,
                                double* cb_1,
                                double* cb_2,
                                size_t* index) nogil:
    cdef double current_value = 0
    cdef double mean = 0
    cdef double std = 0
    cdef double dist = 0
    cdef double min_dist = INFINITY

    cdef double lb_kim
    cdef double lb_k
    cdef double lb_k2

    cdef double ex = 0
    cdef double ex2 = 0

    cdef size_t i
    cdef size_t j
    cdef size_t k
    cdef size_t I
    cdef size_t buffer_pos

    for i in range(t_length):
        current_value = T[t_offset + t_stride * i]
        ex += current_value
        ex2 += current_value * current_value

        buffer_pos = i % s_length
        X_buffer[buffer_pos] = current_value
        X_buffer[buffer_pos + s_length] = current_value

        if i >= s_length - 1:
            j = (i + 1) % s_length
            I = i - (s_length - 1)
            mean = ex / s_length
            std = sqrt(ex2 / s_length - mean * mean)
            lb_kim = constant_lower_bound(s_offset, s_stride, S,
                                          s_mean, s_std, j, 1, X_buffer,
                                          mean, std, s_length, min_dist)

            if lb_kim < min_dist:
                lb_k = cumulative_bound(j, 1, s_length, mean, std, X_buffer,
                                        s_mean, s_std, s_lower, s_upper,
                                        cb_1, min_dist)
                if lb_k < min_dist:
                    lb_k2 = cumulative_bound(
                        s_offset, s_stride, s_length, s_mean, s_std, S,
                        mean, std, t_lower + I, t_upper + I, cb_2, min_dist)

                    if lb_k2 < min_dist:
                        if lb_k > lb_k2:
                            cb[s_length - 1] = cb_1[s_length - 1]
                            for k in range(s_length - 2, -1, -1):
                                cb[k] = cb[k + 1] + cb_1[k]
                        else:
                            cb[s_length - 1] = cb_2[s_length - 1]
                            for k in range(s_length - 2, -1, -1):
                                cb[k] = cb[k + 1] + cb_2[k]
                        dist = inner_dtw(
                            s_offset, s_stride, s_length, s_mean,
                            s_std, S, mean, std, j, X_buffer, r, cb,
                            cost, cost_prev, min_dist)

                        if dist < min_dist:
                            if index != NULL:
                                index[0] = (i + 1) - s_length
                            min_dist = dist

            current_value = X_buffer[j]
            ex -= current_value
            ex2 -= current_value * current_value

    return sqrt(min_dist)


cdef inline size_t compute_warp_width_(size_t length, double r) nogil:
    if r == 1:
        return length - 1
    if r < 1:
        return <size_t> floor(length * r)
    else:
        return <size_t> floor(r)
        

cdef class ScaledDtwDistance(ScaledDistanceMeasure):
    cdef double* X_buffer
    cdef double* lower
    cdef double* upper
    cdef double* cost
    cdef double* cost_prev
    cdef double* cb
    cdef double* cb_1
    cdef double* cb_2

    cdef Deque du
    cdef Deque dl

    cdef size_t max_warp_width
    cdef double r

    def __cinit__(self, size_t n_timestep, double r = 0):
        if r < 0:
            raise ValueError("illegal warp width")
        self.r = r
        self.max_warp_width = compute_warp_width_(n_timestep, self.r)
        self.X_buffer = <double*> malloc(sizeof(double) * n_timestep * 2)
        self.lower = <double*> malloc(sizeof(double) * n_timestep)
        self.upper = <double*> malloc(sizeof(double) * n_timestep)
        self.cost = <double*> malloc(sizeof(double) * 2 * self.max_warp_width + 1)
        self.cost_prev = <double*> malloc(sizeof(double) * 2 * self.max_warp_width + 1)
        self.cb = <double*> malloc(sizeof(double) * n_timestep)
        self.cb_1 = <double*> malloc(sizeof(double) * n_timestep)
        self.cb_2 = <double*> malloc(sizeof(double) * n_timestep)

        if(self.X_buffer == NULL or
           self.lower == NULL or
           self.upper == NULL or
           self.cost == NULL or
           self.cost_prev == NULL or
           self.cb == NULL or
           self.cb_1 == NULL or
           self.cb_2 == NULL):
            raise MemoryError()

        deque_init(&self.dl, 2 * self.max_warp_width + 2)
        deque_init(&self.du, 2 * self.max_warp_width + 2)

    def __dealloc__(self):
        free(self.X_buffer)
        free(self.lower)
        free(self.upper)
        free(self.cost)
        free(self.cost_prev)
        free(self.cb)
        free(self.cb_1)
        free(self.cb_2)

    cdef ShapeletInfo new_shapelet_info(self,
                                        TSDatabase td,
                                        size_t index,
                                        size_t start,
                                        size_t length,
                                        size_t dim) nogil:
        cdef ShapeletInfo shapelet_info
        cdef DtwExtra* dtw_extra
        cdef size_t shapelet_offset
        
        shapelet_info = ScaledDistanceMeasure.new_shapelet_info(self,
                                                                td, index,
                                                                start,
                                                                length, dim)
        
        dtw_extra = <DtwExtra*> malloc(sizeof(DtwExtra))
        dtw_extra[0].lower = <double*> malloc(sizeof(double) * length)
        dtw_extra[0].upper = <double*> malloc(sizeof(double) * length)

        shapelet_offset = (index * td.sample_stride +
                           start * td.timestep_stride +
                           dim * td.dim_stride)
        cdef size_t warp_width = compute_warp_width_(length, self.r)
        find_min_max(shapelet_offset, td.timestep_stride, length, td.data,
                     warp_width, dtw_extra[0].lower, dtw_extra[0].upper,
                     &self.dl, &self.du)
        
        shapelet_info.extra = dtw_extra
        return shapelet_info

    cdef double shapelet_distance(self,
                                  Shapelet s,
                                  TSDatabase td,
                                  size_t t_index,
                                  size_t* return_index=NULL) nogil:
        cdef size_t sample_offset = (t_index * td.sample_stride +
                                     s.dim * td.dim_stride)
        
        cdef double* s_lower
        cdef double* s_upper
        cdef DtwExtra* extra
        cdef size_t warp_width = compute_warp_width_(s.length, self.r)

        if s.extra != NULL:
            extra = <DtwExtra*> s.extra
            s_lower = extra[0].lower
            s_upper = extra[0].upper
        else:
            s_lower = <double*> malloc(sizeof(double) * s.length)
            s_upper = <double*> malloc(sizeof(double) * s.length)
            
            find_min_max(0, 1, s.length, s.data, warp_width, s_lower, s_upper,
                         &self.dl, &self.du)
            find_min_max(sample_offset, td.timestep_stride, td.n_timestep,
                         td.data, warp_width, self.lower, self.upper,
                         &self.dl, &self.du)

        cdef double distance = scaled_dtw_distance(0,
                                                   1,
                                                   s.length,
                                                   s.mean,
                                                   s.std,
                                                   s.data,
                                                   sample_offset,
                                                   td.timestep_stride,
                                                   td.n_timestep,
                                                   td.data,
                                                   warp_width,
                                                   self.X_buffer,
                                                   self.cost,
                                                   self.cost_prev,
                                                   s_lower,
                                                   s_upper,
                                                   self.lower,
                                                   self.upper,
                                                   self.cb,
                                                   self.cb_1,
                                                   self.cb_2,
                                                   return_index)
        if s.extra == NULL:
            free(s_lower)
            free(s_upper)

        return distance

    cdef double shapelet_info_distance(self,
                                       ShapeletInfo s,
                                       TSDatabase td,
                                       size_t t_index) nogil:
        cdef size_t sample_offset = (t_index * td.sample_stride +
                                     s.dim * td.dim_stride)
        cdef size_t shapelet_offset = (s.index * td.sample_stride +
                                       s.dim * td.dim_stride +
                                       s.start * td.timestep_stride)

        cdef size_t warp_width = compute_warp_width_(s.length, self.r)
        
        cdef DtwExtra* dtw_extra = <DtwExtra*> s.extra
        find_min_max(sample_offset, td.timestep_stride, td.n_timestep,
                     td.data, warp_width, self.lower, self.upper, &self.dl,
                     &self.du)
        return scaled_dtw_distance(shapelet_offset,
                                   td.timestep_stride,
                                   s.length,
                                   s.mean,
                                   s.std,
                                   td.data,
                                   sample_offset,
                                   td.timestep_stride,
                                   td.n_timestep,
                                   td.data,
                                   warp_width,
                                   self.X_buffer,
                                   self.cost,
                                   self.cost_prev,
                                   dtw_extra[0].lower,
                                   dtw_extra[0].upper,
                                   self.lower,
                                   self.upper,
                                   self.cb,
                                   self.cb_1,
                                   self.cb_2,
                                   NULL)
