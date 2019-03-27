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

from sktime.transformers.distances._distance cimport DistanceMeasure
from sktime.transformers.distances._distance cimport ScaledDistanceMeasure

cdef class EuclideanDistance(DistanceMeasure):
    pass

cdef class ScaledEuclideanDistance(ScaledDistanceMeasure):
    cdef double* X_buffer

cdef double scaled_euclidean_distance(size_t s_offset,
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
                                      size_t* index) nogil


cdef double euclidean_distance(size_t s_offset,
                               size_t s_stride,
                               size_t s_length,
                               double* S,
                               size_t t_offset,
                               size_t t_stride,
                               size_t t_length,
                               double* T,
                               size_t* index) nogil
