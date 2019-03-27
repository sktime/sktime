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

from sktime.transformers.distances._distance cimport ShapeletInfo, Shapelet

cdef struct SplitPoint:
   size_t split_point
   double threshold
   ShapeletInfo shapelet_info


cdef class Node:
    cdef readonly bint is_leaf

    # if node_type == BRANCH
    cdef readonly double threshold
    cdef readonly Shapelet shapelet

    cdef readonly Node left
    cdef readonly Node right

    # if node_type == LEAF
    cdef double* distribution
    cdef size_t n_labels

