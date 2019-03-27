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

cimport numpy as np

cdef struct ShapeletInfo:
    size_t index  # the index of the shapelet sample
    size_t start  # the start position
    size_t length # the length of the shapelet
    size_t dim    # the dimension of the shapelet
    double mean   # the mean of the shapelet
    double std    # the stanard devision
    void* extra


cdef struct TSDatabase:
    size_t n_samples       # the number of samples
    size_t n_timestep      # the number of timesteps
    size_t n_dims

    double* data           # the data
    size_t sample_stride   # the stride for samples
    size_t timestep_stride # the `feature` stride
    size_t dim_stride      # the dimension stride


cdef class Shapelet:
    cdef readonly size_t length
    cdef readonly size_t dim
    cdef readonly double mean
    cdef readonly double std
    cdef double* data
    cdef void* extra


cdef class DistanceMeasure:

    cdef ShapeletInfo new_shapelet_info(
            self, TSDatabase td, size_t index, size_t start,
            size_t length, size_t dim) nogil

    cdef Shapelet get_shapelet(self, ShapeletInfo s, TSDatabase td)

    cdef Shapelet new_shapelet(self, np.ndarray t, size_t dim)

    cdef int shapelet_matches(
        self, Shapelet s, TSDatabase td, size_t t_index,
        double threhold, size_t** matches,  double** distances,
        size_t* n_matches) nogil except -1

    cdef double shapelet_distance(
            self, Shapelet s, TSDatabase td, size_t t_index,
            size_t* return_index=*) nogil
        
    cdef double shapelet_info_distance(
            self, ShapeletInfo s, TSDatabase td, size_t t_index) nogil

    cdef void shapelet_info_distances(
            self, ShapeletInfo s, TSDatabase td, size_t* samples,
            double* distances, size_t n_samples) nogil

cdef class ScaledDistanceMeasure(DistanceMeasure):
    pass

cdef TSDatabase ts_database_new(np.ndarray X)

cdef void shapelet_info_init(ShapeletInfo* s) nogil

cdef void shapelet_info_free(ShapeletInfo shapelet_info) nogil
