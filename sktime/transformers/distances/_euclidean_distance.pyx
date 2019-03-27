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
# wildboar is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
# License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see
# <http://www.gnu.org/licenses/>.

# Authors: Isak Karlsson

import numpy as np
cimport numpy as np

from libc.stdlib cimport realloc
from libc.stdlib cimport malloc
from libc.stdlib cimport free

from libc.math cimport sqrt
from libc.math cimport INFINITY, NAN

from sktime.transformers.distances._distance cimport ScaledDistanceMeasure
from sktime.transformers.distances._distance cimport DistanceMeasure

from sktime.transformers.distances._distance cimport TSDatabase

from sktime.transformers.distances._distance cimport Shapelet
from sktime.transformers.distances._distance cimport ShapeletInfo


cdef class ScaledEuclideanDistance(ScaledDistanceMeasure):
    
    def __cinit__(self, size_t n_timestep, *args, **kwargs):
        self.X_buffer = <double*> malloc(
            sizeof(double) * n_timestep * 2)

    def __dealloc__(self):
        free(self.X_buffer)

    cdef double shapelet_distance(self,
                                  Shapelet s,
                                  TSDatabase td,
                                  size_t t_index,
                                  size_t* return_index = NULL) nogil:
        cdef size_t sample_offset = (t_index * td.sample_stride +
                                     s.dim * td.dim_stride)

        return scaled_euclidean_distance(
            0,
            1,
            s.length,
            s.mean,
            s.std,
            s.data,
            sample_offset,
            td.timestep_stride,
            td.n_timestep,
            td.data,
            self.X_buffer,
            return_index)
        
    cdef double shapelet_info_distance(self,
                                       ShapeletInfo s,
                                       TSDatabase td,
                                       size_t t_index) nogil:

        cdef size_t sample_offset = (t_index * td.sample_stride +
                                     s.dim * td.dim_stride)
        cdef size_t shapelet_offset = (s.index * td.sample_stride +
                                       s.dim * td.dim_stride +
                                       s.start * td.timestep_stride)
        return scaled_euclidean_distance(
            shapelet_offset,
            td.timestep_stride,
            s.length,
            s.mean,
            s.std,
            td.data,
            sample_offset,
            td.timestep_stride,
            td.n_timestep,
            td.data,
            self.X_buffer,
            NULL)

    cdef int shapelet_matches(
            self, Shapelet s, TSDatabase td, size_t t_index,
            double threshold, size_t** matches,  double** distances,
            size_t* n_matches) nogil except -1:
        cdef size_t sample_offset = (t_index * td.sample_stride +
                                     s.dim * td.dim_stride)

        return scaled_euclidean_distance_matches(
            0,
            1,
            s.length,
            s.mean,
            s.std,
            s.data,
            sample_offset,
            td.timestep_stride,
            td.n_timestep,
            td.data,
            self.X_buffer,
            threshold,
            distances,
            matches,
            n_matches)
            


cdef class EuclideanDistance(DistanceMeasure):

    cdef double shapelet_distance(
            self, Shapelet s, TSDatabase td, size_t t_index,
            size_t* return_index = NULL) nogil:
        cdef size_t sample_offset = (t_index * td.sample_stride +
                                     s.dim * td.dim_stride)

        return euclidean_distance(
            0,
            1,
            s.length,
            s.data,
            sample_offset,
            td.timestep_stride,
            td.n_timestep,
            td.data,
            return_index)
    
    cdef double shapelet_info_distance(
        self, ShapeletInfo s, TSDatabase td, size_t t_index) nogil:

        cdef size_t sample_offset = (t_index * td.sample_stride +
                                     s.dim * td.dim_stride)
        cdef size_t shapelet_offset = (s.index * td.sample_stride +
                                       s.dim * td.dim_stride +
                                       s.start * td.timestep_stride)
        return euclidean_distance(
            shapelet_offset,
            td.timestep_stride,
            s.length,
            td.data,
            sample_offset,
            td.timestep_stride,
            td.n_timestep,
            td.data,
            NULL)

    cdef int shapelet_matches(
            self, Shapelet s, TSDatabase td, size_t t_index,
            double threshold, size_t** matches,  double** distances,
            size_t* n_matches) nogil except -1:
        cdef size_t sample_offset = (t_index * td.sample_stride +
                                     s.dim * td.dim_stride)

        return euclidean_distance_matches(
            0,
            1,
            s.length,
            s.data,
            sample_offset,
            td.timestep_stride,
            td.n_timestep,
            td.data,
            threshold,
            distances,
            matches,
            n_matches)

cdef double scaled_euclidean_distance(size_t s_offset,
                                      size_t s_stride,
                                      size_t s_length,
                                      double s_mean,
                                      double s_std,
                                      double*S,
                                      size_t t_offset,
                                      size_t t_stride,
                                      size_t t_length,
                                      double*T,
                                      double*X_buffer,
                                      size_t*index) nogil:
    cdef double current_value = 0
    cdef double mean = 0
    cdef double std = 0
    cdef double dist = 0
    cdef double min_dist = INFINITY

    cdef double ex = 0
    cdef double ex2 = 0

    cdef size_t i
    cdef size_t j
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
            mean = ex / s_length
            std = sqrt(ex2 / s_length - mean * mean)
            dist = inner_scaled_euclidean_distance(s_offset, s_length, s_mean, s_std,
                                                   j, mean, std, S, s_stride,
                                                   X_buffer, min_dist)

            if dist < min_dist:
                min_dist = dist
                if index != NULL:
                    index[0] = (i + 1) - s_length

            current_value = X_buffer[j]
            ex -= current_value
            ex2 -= current_value * current_value

    return sqrt(min_dist)


cdef inline double inner_scaled_euclidean_distance(size_t offset,
                                                   size_t length,
                                                   double s_mean,
                                                   double s_std,
                                                   size_t j,
                                                   double mean,
                                                   double std,
                                                   double*X,
                                                   size_t timestep_stride,
                                                   double*X_buffer,
                                                   double min_dist) nogil:
    # Compute the distance between the shapelet (starting at `offset`
    # and ending at `offset + length` normalized with `s_mean` and
    # `s_std` with the shapelet in `X_buffer` starting at `0` and
    # ending at `length` normalized with `mean` and `std`
    cdef double dist = 0
    cdef double x
    cdef size_t i
    cdef bint std_zero = std == 0
    cdef bint s_std_zero = s_std == 0

    # distance is zero
    if s_std_zero and std_zero:
        return 0

    for i in range(length):
        if dist >= min_dist:
            break

        x = (X[offset + timestep_stride * i] - s_mean) / s_std
        if not std_zero:
            x -= (X_buffer[i + j] - mean) / std
        dist += x * x

    return dist


cdef double euclidean_distance(size_t s_offset,
                               size_t s_stride,
                               size_t s_length,
                               double*S,
                               size_t t_offset,
                               size_t t_stride,
                               size_t t_length,
                               double*T,
                               size_t*index) nogil:
    cdef double dist = 0
    cdef double min_dist = INFINITY

    cdef size_t i
    cdef size_t j
    cdef double x
    for i in range(t_length - s_length + 1):
        dist = 0
        for j in range(s_length):
            if dist >= min_dist:
                break

            x = T[t_offset + t_stride * i + j]
            x -= S[s_offset + s_stride * j]
            dist += x * x

        if dist < min_dist:
            min_dist = dist
            if index != NULL:
                index[0] = i

    return sqrt(min_dist)


cdef int euclidean_distance_matches(size_t s_offset,
                                    size_t s_stride,
                                    size_t s_length,
                                    double* S,
                                    size_t t_offset,
                                    size_t t_stride,
                                    size_t t_length,
                                    double* T,
                                    double threshold,
                                    double** distances,
                                    size_t** matches,
                                    size_t* n_matches) nogil except -1:
    cdef double dist = 0
    cdef size_t capacity = 1
    cdef size_t tmp_capacity
    cdef size_t i
    cdef size_t j
    cdef double x

    matches[0] = <size_t*> malloc(sizeof(size_t) * capacity)
    distances[0] = <double*> malloc(sizeof(double) * capacity)
    n_matches[0] = 0

    threshold = threshold * threshold
    for i in range(t_length - s_length + 1):
        dist = 0
        for j in range(s_length):
            if dist > threshold:
                break

            x = T[t_offset + t_stride * i + j]
            x -= S[s_offset + s_stride * j]
            dist += x * x
        if dist <= threshold:
            tmp_capacity = capacity
            realloc_array(<void**> matches, n_matches[0], sizeof(size_t), &tmp_capacity)
            realloc_array(<void**> distances, n_matches[0], sizeof(double), &capacity)
            matches[0][n_matches[0]] = i
            distances[0][n_matches[0]] = sqrt(dist)
            n_matches[0] += 1

    return 0


cdef int scaled_euclidean_distance_matches(size_t s_offset,
                                           size_t s_stride,
                                           size_t s_length,
                                           double s_mean,
                                           double s_std,
                                           double* S,
                                           size_t t_offset,
                                           size_t t_stride,
                                           size_t t_length,
                                           double* T,
                                           double* X_buffer,
                                           double threshold,
                                           double** distances,
                                           size_t** matches,                                              
                                           size_t* n_matches) nogil except -1:
    cdef double current_value = 0
    cdef double mean = 0
    cdef double std = 0
    cdef double dist = 0

    cdef double ex = 0
    cdef double ex2 = 0

    cdef size_t i
    cdef size_t j
    cdef size_t buffer_pos
    cdef size_t capacity = 1
    cdef size_t tmp_capacity

    matches[0] = <size_t*> malloc(sizeof(size_t) * capacity)
    distances[0] = <double*> malloc(sizeof(double) * capacity)
    n_matches[0] = 0

    threshold = threshold * threshold

    for i in range(t_length):
        current_value = T[t_offset + t_stride * i]
        ex += current_value
        ex2 += current_value * current_value

        buffer_pos = i % s_length
        X_buffer[buffer_pos] = current_value
        X_buffer[buffer_pos + s_length] = current_value
        if i >= s_length - 1:
            j = (i + 1) % s_length
            mean = ex / s_length
            std = sqrt(ex2 / s_length - mean * mean)
            dist = inner_scaled_euclidean_distance(
                s_offset, s_length, s_mean, s_std, j, mean, std, S, s_stride,
                X_buffer, threshold)
            
            if dist <= threshold:
                tmp_capacity = capacity
                realloc_array(
                    <void**> matches, n_matches[0], sizeof(size_t), &tmp_capacity)
                realloc_array(
                    <void**> distances, n_matches[0], sizeof(double),  &capacity)

                matches[0][n_matches[0]] = (i + 1) - s_length
                distances[0][n_matches[0]] = sqrt(dist)

                n_matches[0] += 1

            current_value = X_buffer[j]
            ex -= current_value
            ex2 -= current_value * current_value

    return 0


cdef int realloc_array(void** a, size_t p, size_t size, size_t* cap)  nogil except -1:
    cdef void* tmp = a[0]
    if p >= cap[0]:
        cap[0] *= 2
        tmp = realloc(a[0], size*32 * cap[0])
        if tmp == NULL:
            with gil:
                raise MemoryError()
    a[0] = tmp
