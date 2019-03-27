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

import numpy as np
cimport numpy as np

from libc.stdlib cimport free
from libc.stdlib cimport malloc

from sktime.transformers.distances._distance cimport TSDatabase
from sktime.transformers.distances._distance cimport ts_database_new
from sktime.transformers.distances._distance cimport DistanceMeasure
from sktime.transformers.distances._distance cimport Shapelet
from sktime.transformers.distances._distance cimport ScaledDistanceMeasure

from sklearn.utils import check_array

import sktime.transformers.distances._euclidean_distance
import sktime.transformers.distances._dtw_distance

DISTANCE_MEASURE = {
    'euclidean': sktime.transformers.distances._euclidean_distance.EuclideanDistance,
    'scaled_euclidean': sktime.transformers.distances._euclidean_distance.ScaledEuclideanDistance,
    'scaled_dtw': sktime.transformers.distances._dtw_distance.ScaledDtwDistance,
}

# validate and convert shapelet to sutable format
def validate_shapelet_(shapelet):
    cdef np.ndarray s = check_array(
        shapelet, ensure_2d=False, dtype=np.float64, order="c")
    if s.ndim > 1:
        raise ValueError("only 1d shapelets allowed")

    if not s.flags.contiguous:
        s = np.ascontiguousarray(s, dtype=np.float64)
    return s


# validate and convert time series data to suitable
def validate_data_(data):
    cdef np.ndarray x = check_array(
        data, ensure_2d=False, allow_nd=True, dtype=np.float64, order="c")
    if x.ndim == 1:
        x = x.reshape(-1, x.shape[0])

    if not x.flags.contiguous:
        x = np.ascontiguousarray(x, dtype=np.float64)
    return x

def check_sample_(sample, n_samples):
    if sample < 0 or sample >= n_samples:
        raise ValueError("illegal sample {}".format(sample))

def check_dim_(dim, ndims):
    if dim < 0 or dim >= ndims:
        raise ValueError("illegal dimension {}".format(dim))

cdef np.ndarray make_match_array_(
        size_t* matches,  size_t n_matches):
    if n_matches > 0:
        match_array = np.empty(n_matches, dtype=np.intp)
        for i in range(n_matches):
            match_array[i] = matches[i]
        return match_array
    else:
        return None

cdef np.ndarray make_distance_array_(
        double* distances,  size_t n_matches):
    if n_matches > 0:
        dist_array = np.empty(n_matches, dtype=np.float64)
        for i in range(n_matches):
            dist_array[i] = distances[i]
        return dist_array
    else:
        return None


def distance(
        shapelet,
        data,
        dim=0,
        sample=None,
        metric="euclidean",
        metric_params=None,
        return_index=False):
    """Computes the minimum distance between `s` and the samples in `x`

    :param shapelet: the subsequence `array_like`

    :param data: the samples `[n_samples, n_timesteps]` or
                 `[n_samples, n_dims, n_timesteps]`

    :param dim: the time series dimension to search (default: 0)

    :param sample: the samples to compare to `int` or `array_like` or
                   `None`; if `sample` is `None`, return the distance
                   to all samples in data. Note that if, `n_samples`
                   is 1 a scalar is returned; otherwise a array is
                   returned.

    :param metric: the distance measure

    :param return_index: if `true` return the index of the best
                         match. If there are many equally good
                         best matches, the first is returned.

    :returns: `float`, `(float, int)`, `float [n_samples]` or `(float
              [n_samples], int [n_samples]` depending on input

    """
    cdef np.ndarray s = validate_shapelet_(shapelet)
    cdef np.ndarray x = validate_data_(data)
    if sample is None:
        if x.shape[0] == 1:
            sample = 0
        else:
            sample = np.arange(x.shape[0])


    cdef TSDatabase sd = ts_database_new(x)

    check_dim_(dim, sd.n_dims)
    cdef double min_dist
    cdef size_t min_index

    cdef double mean = 0
    cdef double std = 0

    if metric_params is None:
        metric_params = {}

    cdef DistanceMeasure distance_measure = DISTANCE_MEASURE[metric](
        sd.n_timestep, **metric_params)

    cdef Shapelet shape = distance_measure.new_shapelet(s, dim)
    if isinstance(sample, int):
        min_dist = distance_measure.shapelet_distance(
            shape, sd, sample, &min_index)

        if return_index:
            return min_dist, min_index
        else:
            return min_dist
    else:  # assume an `array_like` object for `samples`
        samples = check_array(sample, ensure_2d=False, dtype=np.int)
        dist = []
        ind = []
        for i in samples:
            min_dist = distance_measure.shapelet_distance(
                shape, sd, i, &min_index)

            dist.append(min_dist)
            ind.append(min_index)

        if return_index:
            return np.array(dist), np.array(ind)
        else:
            return np.array(dist)


def matches(
        shapelet,
        data,
        threshold,
        dim=0,
        sample=None,
        metric="euclidean",
        metric_params=None,
        return_distance=False):
    """Return the positions in data (one array per `sample`) where
    `shapelet` is closer than `threshold`.

    :param s: the subsequence `array_like`

    :param x: the samples `[n_samples, n_timestep]` or `[n_sample,
              n_dim, n_timestep]`

    :param threshold: the maximum threshold for match

    :param dim: the time series dimension to search (default: 0)

    :param sample: the samples to compare to `int` or `array_like` or
                   `None`.  If `None` compare to all. (default:
                   `None`)

    :param metric: the distance measure

    :param metric_params: additional parameters to the metric function
                          (optional, dict, default: None)

    :returns: `[n_matches]`, or `[[n_matches], ... n_samples]`

    """
    cdef np.ndarray s = validate_shapelet_(shapelet)
    cdef np.ndarray x = validate_data_(data)
    check_dim_(dim, x.ndim)
    if sample is None:
        if x.shape[0] == 1:
            sample = 0
        else:
            sample = np.arange(x.shape[0])

    cdef TSDatabase sd = ts_database_new(x)

    cdef size_t* matches
    cdef double* distances
    cdef size_t n_matches

    if metric_params is None:
        metric_params = {}

    cdef DistanceMeasure distance_measure = DISTANCE_MEASURE[metric](
        sd.n_timestep, **metric_params)
    cdef Shapelet shape = distance_measure.new_shapelet(s, dim)

    cdef size_t i
    if isinstance(sample, int):
        check_sample_(sample, sd.n_samples)
        distance_measure.shapelet_matches(
            shape, sd, sample, threshold, &matches, &distances, &n_matches)

        match_array = make_match_array_(matches, n_matches)
        distance_array =  make_distance_array_(distances, n_matches)
        free(distances)
        free(matches)

        if return_distance:
            return distance_array, match_array
        else:
            return match_array
    else:
        samples = check_array(sample, ensure_2d=False, dtype=np.int)
        match_list = []
        distance_list = []
        for i in samples:
            check_sample_(i, sd.n_samples)
            distance_measure.shapelet_matches(
                shape, sd, i, threshold, &matches, &distances, &n_matches)
            match_array = make_match_array_(matches, n_matches)
            distance_array = make_distance_array_(distances, n_matches)
            match_list.append(match_array)
            distance_list.append(distance_array)
            free(matches)
            free(distances)
                
        if return_distance:
            return distance_list, match_list
        else:
            return match_list
