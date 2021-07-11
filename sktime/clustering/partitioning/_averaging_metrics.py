# -*- coding: utf-8 -*-
"""Cluster averaging metrics"""

__author__ = ["Christopher Holder", "Tony Bagnall"]
__all__ = ["MeanAveraging", "BarycenterAveraging"]

import numpy as np

from functools import reduce
from sktime.clustering.base import BaseClusterAverage
from sktime.clustering.base._typing import NumpyArray


class MeanAveraging(BaseClusterAverage):
    """Mean Averaging algorithm to average a time series

    Parameters
    ----------
    series: Numpy_Array
        Series to get the mean of

    n_iterations: int
        Number of iterations to refine the average
    """

    def __init__(self, series: NumpyArray, n_iterations: int = 10):

        super(MeanAveraging, self).__init__(series, n_iterations)

    def average(self):
        """
        Method used to get the mean average

        Returns
        -------
        Numpy_Array:
            Array containing the estimated average
        """
        return self.series.mean(axis=0)


class BarycenterAveraging(BaseClusterAverage):
    """DTW Barycenter averaging algorithm to average a time series
    taking adavtange of dtw allignment

    Parameters
    ----------
    series: Numpy_Array
        The series to find the DBA average of

    n_iterations: int
        Number of iterations to refine the average

    /*******************************************************************************
     * Copyright (C) 2018 Francois Petitjean
     *
     * This program is free software: you can redistribute it and/or modify
     * it under the terms of the GNU General Public License as published by
     * the Free Software Foundation, version 3 of the License.
     *
     * This program is distributed in the hope that it will be useful,
     * but WITHOUT ANY WARRANTY; without even the implied warranty of
     * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
     * GNU General Public License for more details.
     *
     * You should have received a copy of the GNU General Public License
     * along with this program.  If not, see <http://www.gnu.org/licenses/>.
     ******************************************************************************/
    TODO: This is basically copied from
    https://github.com/fpetitjean/DBA/blob/master/DBA.py
    This need massive optimisations and likely needs to be done in
    cython. This is basically doing its own implementation of dtw
    because it needs the cost matrix from the dtw calculation
    and the current sktime one doesnt return that so need to figure
    that out
    """

    def __init__(self, series: NumpyArray, n_iterations: int = 10):
        super(BarycenterAveraging, self).__init__(series, n_iterations)

    def average(self):
        """
        Method used to get the barycenter average

        Returns
        -------
        Numpy_Array:
            Array containing the estimated average
        """
        max_length = reduce(max, map(len, self.series))

        cost_mat = np.zeros((max_length, max_length))
        delta_mat = np.zeros((max_length, max_length))
        path_mat = np.zeros((max_length, max_length), dtype=np.int8)

        medoid_ind = self._approximate_medoid_index(cost_mat, delta_mat)
        center = self.series[medoid_ind]

        for _ in range(self.n_iterations):
            center = self._DBA_update(center, cost_mat, path_mat, delta_mat)

        return center

    def _approximate_medoid_index(self, cost_mat, delta_mat):
        if len(self.series) <= 50:
            indices = range(0, len(self.series))
        else:
            indices = np.random.choice(range(0, len(self.series)), 50, replace=False)

        medoid_ind = -1
        best_ss = 1e20
        for index_candidate in indices:
            candidate = self.series[index_candidate]
            ss = self._sum_of_squares(candidate, cost_mat, delta_mat)
            if medoid_ind == -1 or ss < best_ss:
                best_ss = ss
                medoid_ind = index_candidate
        return medoid_ind

    def _sum_of_squares(self, s, cost_mat, delta_mat):
        return sum(
            map(lambda t: self._squared_DTW(s, t, cost_mat, delta_mat), self.series)
        )

    def _DTW(self, s, t, cost_mat, delta_mat):
        return np.sqrt(self._squared_DTW(s, t, cost_mat, delta_mat))

    def _squared_DTW(self, s, t, cost_mat, delta_mat):
        s_len = len(s)
        t_len = len(t)
        self._fill_delta_mat_dtw(s, t, delta_mat)
        cost_mat[0, 0] = delta_mat[0, 0]
        for i in range(1, s_len):
            cost_mat[i, 0] = cost_mat[i - 1, 0] + delta_mat[i, 0]

        for j in range(1, t_len):
            cost_mat[0, j] = cost_mat[0, j - 1] + delta_mat[0, j]

        for i in range(1, s_len):
            for j in range(1, t_len):
                diag, left, top = (
                    cost_mat[i - 1, j - 1],
                    cost_mat[i, j - 1],
                    cost_mat[i - 1, j],
                )
                if diag <= left:
                    if diag <= top:
                        res = diag
                    else:
                        res = top
                else:
                    if left <= top:
                        res = left
                    else:
                        res = top
                cost_mat[i, j] = res + delta_mat[i, j]
        return cost_mat[s_len - 1, t_len - 1]

    def _fill_delta_mat_dtw(self, center, s, delta_mat):
        slim = delta_mat[: len(center), : len(s)]
        np.subtract.outer(center, s, out=slim)
        np.square(slim, out=slim)

    def _DBA_update(self, center, cost_mat, path_mat, delta_mat):
        options_argmin = [(-1, -1), (0, -1), (-1, 0)]
        updated_center = np.zeros(center.shape)
        n_elements = np.array(np.zeros(center.shape), dtype=int)
        center_length = len(center)
        for s in self.series:
            s_len = len(s)
            self._fill_delta_mat_dtw(center, s, delta_mat)
            cost_mat[0, 0] = delta_mat[0, 0]
            path_mat[0, 0] = -1

            for i in range(1, center_length):
                cost_mat[i, 0] = cost_mat[i - 1, 0] + delta_mat[i, 0]
                path_mat[i, 0] = 2

            for j in range(1, s_len):
                cost_mat[0, j] = cost_mat[0, j - 1] + delta_mat[0, j]
                path_mat[0, j] = 1

            for i in range(1, center_length):
                for j in range(1, s_len):
                    diag, left, top = (
                        cost_mat[i - 1, j - 1],
                        cost_mat[i, j - 1],
                        cost_mat[i - 1, j],
                    )
                    if diag <= left:
                        if diag <= top:
                            res = diag
                            path_mat[i, j] = 0
                        else:
                            res = top
                            path_mat[i, j] = 2
                    else:
                        if left <= top:
                            res = left
                            path_mat[i, j] = 1
                        else:
                            res = top
                            path_mat[i, j] = 2

                    cost_mat[i, j] = res + delta_mat[i, j]

            i = center_length - 1
            j = s_len - 1

            while path_mat[i, j] != -1:
                updated_center[i] += s[j]
                n_elements[i] += 1
                move = options_argmin[path_mat[i, j]]
                i += move[0]
                j += move[1]
            assert i == 0 and j == 0
            updated_center[i] += s[j]
            n_elements[i] += 1

        return np.divide(updated_center, n_elements)
