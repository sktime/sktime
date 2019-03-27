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

cdef enum:
    RAND_R_MAX = 2147483647

cdef void print_c_array_d(object name, double* arr, size_t length)

cdef void print_c_array_i(object name, size_t* arr, size_t length)

cdef size_t label_distribution(const size_t* samples, double*
                               sample_weights, size_t start, size_t
                               end, const size_t* labels, size_t
                               labels_stride, size_t n_classes,
                               double* n_weighted_samples, double* dist) nogil


cdef void argsort(double* values, size_t* order, size_t length) nogil


cdef size_t rand_r(size_t* seed) nogil


cdef size_t rand_int(size_t min_val, size_t max_val, size_t* seed) nogil
