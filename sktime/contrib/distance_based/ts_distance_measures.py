import numpy as np

__author__ = "Jason Lines"
"""
python versions of distance functions. These are fearful slow, and the cython ones should be used
these are in sktime.distances.elastic

        """


def dtw_distance(first, second, **kwargs):

    def dtw_single_channel(first, second, **kwargs):
        cutoff = np.inf
        try:
            window = kwargs["window"]
        except:
            window = 1.0
        # print("window: "+str(window))

        n = len(first)
        m = len(second)

        warp_matrix = np.full([n, m], np.inf)
        if n > m:
            window_size = n * window
        else:
            window_size = m * window
        window_size = int(window_size)

        dist = lambda x1, x2: ((x1 - x2) ** 2)

        pairwise_distances = np.asarray([[dist(x1, x2) for x2 in second] for x1 in first])

        # initialise edges of the warping matrix
        warp_matrix[0][0] = pairwise_distances[0][0]
        for i in range(1, window_size):
            warp_matrix[0][i] = pairwise_distances[0][i] + warp_matrix[0][i - 1]
            warp_matrix[i][0] = pairwise_distances[i][0] + warp_matrix[i - 1][0]

        # now visit all allowed cells, calculate the value as the distance in this cell + min(top, left, or top-left)
        # traverse each row,
        for row in range(1, n):
            cutoff_beaten = False
            # traverse left and right by the allowed window
            for column in range(row - window_size, row + 1 + window_size):
                if column < 1 or column >= m:
                    continue

                # find smallest entry in the warping matrix, either above, to the left, or diagonally left and up
                above = warp_matrix[row - 1][column]
                left = warp_matrix[row][column - 1]
                diag = warp_matrix[row - 1][column - 1]

                # add the pairwise distance for [row][column] to the minimum of the three possible potential cells
                warp_matrix[row][column] = pairwise_distances[row][column] + min([above, left, diag])

                # check for evidence that cutoff has been beaten on this row (if using)
                if cutoff is not None and warp_matrix[row][column] < cutoff:
                    cutoff_beaten = True

            # if using a cutoff, at least one calculated value on this row MUST be less than the cutoff otherwise the
            # final distance is guaranteed not to be less. Therefore, if the cutoff has not been beaten, early-abandon
            if cutoff is not None and cutoff_beaten is False:
                return float("inf")

        return warp_matrix[n - 1][m - 1]

    if isinstance(first, np.ndarray) and isinstance(first[0], float) is True:
        return dtw_single_channel(first, second, **kwargs)

    dist = 0
    for dim in range(0, len(first)):
        dist += dtw_single_channel(first[dim], second[dim], **kwargs)

    return dist


def derivative_dtw_distance(first, second, **kwargs):

    if isinstance(first, np.ndarray) and isinstance(first[0], float) is True:
        return dtw_distance(np.diff(first), np.diff(second), **kwargs)

    dist = 0
    for dim in range(0, len(first)):
        dist += dtw_distance([first[dim].diff()[1:]], [second[dim].diff()[1:]], **kwargs)
    return dist


def weighted_dtw_distance(first, second, **kwargs):

    def wdtw_single_channel(first, second, **kwargs):
        try:
            g = kwargs["g"]
        except:
            g = 0.0

        m = len(first);
        n = len(second);

        weight_vector = [1/(1+np.exp(-g*(i-m/2))) for i in range(0,m)]

        dist = lambda x1, x2: ((x1 - x2) ** 2)
        pairwise_distances = np.asarray([[dist(x1, x2) for x2 in second] for x1 in first])

        # initialise edges of the warping matrix
        distances = np.full([n, m], np.inf)
        distances[0][0] = weight_vector[0]*pairwise_distances[0][0]

        # top row
        for i in range(1,n):
            distances[0][i] = distances[0][i - 1] + weight_vector[i] * pairwise_distances[0][i]

        # first column
        for i in range(1, m):
            distances[i][0] = distances[i - 1][0] + weight_vector[i] * pairwise_distances[i][0]

        # warp rest
        for i in range(1,m):
            for j in range(1,n):
                min_dist = np.min([distances[i][j - 1], distances[i - 1][j], distances[i - 1][j - 1]])
                # print(min_dist)
                distances[i][j] = min_dist+weight_vector[np.abs(i-j)] * pairwise_distances[i][j]
        return distances[m-1][n-1]

    if isinstance(first, np.ndarray) and isinstance(first[0], float) is True:
        return wdtw_single_channel(first, second, **kwargs)
    dist = 0
    for dim in range(0, len(first)):
        dist += wdtw_single_channel(first[dim], second[dim], **kwargs)

    return dist


def weighted_derivative_dtw_distance(first, second, **kwargs):

    if isinstance(first, np.ndarray) and isinstance(first[0], float) is True:
        return weighted_dtw_distance(np.diff(first), np.diff(second), **kwargs)

    dist = 0
    for dim in range(0, len(first)):
        dist += weighted_dtw_distance([first[dim].diff()[1:]], [second[dim].diff()[1:]], **kwargs)
    return dist


def lcss_distance(first, second, **kwargs):

    def lcss_single_channel(first, second, **kwargs) -> float:
        try:
            delta = kwargs["delta"]
        except KeyError:
            delta = 3

        try:
            epsilon = kwargs["epsilon"]
        except KeyError:
            epsilon = 1

        m = len(first)
        n = len(second)

        lcss = np.zeros([m + 1, n + 1])

        for i in range(0, m):
            # for (int j = i-delta; j <= i+delta; j++){
            for j in range(i - delta, i + delta + 1):
                if j < 0:
                    j = -1
                elif j >= n:
                    j = i + delta
                elif second[j] + epsilon >= first[i] and second[j] - epsilon <= first[i]:
                    lcss[i + 1][j + 1] = lcss[i][j] + 1
                elif lcss[i][j + 1] > lcss[i + 1][j]:
                    lcss[i + 1][j + 1] = lcss[i][j + 1]
                else:
                    lcss[i + 1][j + 1] = lcss[i + 1][j]

        max_val = -1;
        for i in range(1, len(lcss[len(lcss) - 1])):
            if lcss[len(lcss) - 1][i] > max_val:
                max_val = lcss[len(lcss) - 1][i];

        return 1 - (max_val / m);

    if isinstance(first, np.ndarray) and isinstance(first[0], float) is True:
        return lcss_single_channel(first, second, **kwargs)

    dist = 0
    for dim in range(0,len(first)):
        dist += lcss_single_channel(first[dim],second[dim],**kwargs)
    return dist


def msm_distance(first, second, **kwargs):

    def msm_single_channel(first, second, **kwargs) -> float:
        try:
            c = kwargs["c"]
        except KeyError:
            c = 0.1

        m = len(first)
        n = len(second)

        cost = np.zeros([m, n])

        def calc_cost(new_point, x, y):
            dist = 0

            if ((x <= new_point) and (new_point <= y)) or ((y <= new_point) and (new_point <= x)):
                return c
            else:
                return c + min(np.abs(new_point - x), np.abs(new_point - y))

        # Initialization
        cost[0][0] = np.abs(first[0] - second[0])
        for i in range(1, m):
            cost[i][0] = cost[i - 1][0] + calc_cost(first[i], first[i - 1], second[0])

        for i in range(1, n):
            cost[0][i] = cost[0][i - 1] + calc_cost(second[i], first[0], second[i - 1])

        # Main Loop
        for i in range(1, m):
            for j in range(1, n):
                d1 = cost[i - 1][j - 1] + np.abs(first[i] - second[j])
                d2 = cost[i - 1][j] + calc_cost(first[i], first[i - 1], second[j])
                d3 = cost[i][j - 1] + calc_cost(second[j], first[i], second[j - 1])
                cost[i][j] = min(d1, d2, d3)

        return cost[m - 1][n - 1];

    if isinstance(first, np.ndarray) and isinstance(first[0], float) is True:
        return msm_single_channel(first, second, **kwargs)

    dist = 0
    for dim in range(0, len(first)):
        dist += msm_single_channel(first[dim], second[dim], **kwargs)
    return dist


def erp_distance(first, second, **kwargs):

    def erp_single_channel(first, second, **kwargs):
        """
        Adapted from:
            This file is part of ELKI:
            Environment for Developing KDD-Applications Supported by Index-Structures

            Copyright (C) 2011
            Ludwig-Maximilians-UniversitÃ¤t MÃ¼nchen
            Lehr- und Forschungseinheit fÃ¼r Datenbanksysteme
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

        try:
            band_size = kwargs["band_size"]
        except KeyError:
            band_size = 5
        try:
            g = kwargs["g"]
        except KeyError:
            g = 0.5

        m = len(first)
        n = len(second)
        band = np.ceil(band_size * m)
        curr = np.empty(m)
        prev = np.empty(m)

        for i in range(0, m):
            temp = prev
            prev = curr
            curr = temp
            l = i - (band + 1)

            if l < 0:
                l = 0

            r = i + (band + 1);
            if r > m - 1:
                r = (m - 1)

            for j in range(l, r + 1):
                if np.abs(i - j) <= band:
                    val1 = first[i]
                    val2 = g
                    diff = (val1 - val2)
                    d1 = np.sqrt(diff * diff)

                    val1 = g
                    val2 = second[j]
                    diff = (val1 - val2)
                    d2 = np.sqrt(diff * diff)

                    val1 = first[i]
                    val2 = second[j]
                    diff = (val1 - val2)
                    d12 = np.sqrt(diff * diff)

                    dist1 = d1 * d1
                    dist2 = d2 * d2
                    dist12 = d12 * d12

                    if i + j != 0:
                        if i == 0 or (j != 0 and (((prev[j - 1] + dist12) > (curr[j - 1] + dist2)) and ((curr[j - 1] + dist2) < (prev[j] + dist1)))):
                            # del
                            cost = curr[j - 1] + dist2
                        elif (j == 0) or ((i != 0) and (((prev[j - 1] + dist12) > (prev[j] + dist1)) and ((prev[j] + dist1) < (curr[j - 1] + dist2)))):
                            # ins
                            cost = prev[j] + dist1;
                        else:
                            # match
                            cost = prev[j - 1] + dist12
                    else:
                        cost = 0

                    curr[j] = cost

        return np.sqrt(curr[m - 1]);

    if isinstance(first, np.ndarray) and isinstance(first[0], float) is True:
        return erp_single_channel(first, second, **kwargs)

    dist = 0
    for dim in range(0, len(first)):
        dist += erp_single_channel(first[dim], second[dim], **kwargs)
    return dist
