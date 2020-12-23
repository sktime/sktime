__author__ = "Ansgar Asseburg"
__email__ = "s2092795@stud.uni-frankfurt.de"

import numpy as np


def agdtw_distance(first, second, **kwargs):
    # idea and algorithm was taken from:
    # @inproceedings{XueZTWL17,
    #   author    = {Yangtao Xue and
    #                Li Zhang and
    #                Zhiwei Tao and
    #                Bangjun Wang and
    #                Fanzhang Li},
    #   editor    = {Derong Liu and
    #                Shengli Xie and
    #                Yuanqing Li and
    #                Dongbin Zhao and
    #                El{-}Sayed M. El{-}Alfy},
    #   title     = {An Altered Kernel Transformation for Time Series
    #   Classification},
    #   booktitle = {Neural Information Processing - 24th International
    #   Conference, {ICONIP}
    #                2017, Guangzhou, China, November 14-18, 2017,
    #                Proceedings, Part {V}},
    #   series    = {Lecture Notes in Computer Science},
    #   volume    = {10638},
    #   pages     = {455--465},
    #   publisher = {Springer},
    #   year      = {2017},
    #   url       = {https://doi.org/10.1007/978-3-319-70139-4\_46},
    #   doi       = {10.1007/978-3-319-70139-4\_46},
    #   timestamp = {Tue, 14 May 2019 10:00:42 +0200},
    #   biburl    = {https://dblp.org/rec/conf/iconip/XueZTWL17.bib},
    #   bibsource = {dblp computer science bibliography, https://dblp.org}
    # }

    def warping_matrix(series_1, series_2):
        """
        Creates the warping matrix
        @series_1: numpy array containing the first series
        @series_2: numpy array containing the second series
        @return: 2D numpy array containing the minimum squared distances
        """
        row_dim = len(series_1)
        col_dim = len(series_2)
        warp_matrix = np.full([row_dim, col_dim], np.inf)
        warp_matrix[0, 0] = 0
        first_values = series_1.iloc[:, 0]
        second_values = series_2.iloc[:, 0]
        for row in range(row_dim):
            for col in range(col_dim):
                min_index = index_of_section_min(warp_matrix, (row, col))
                min_dist_to_here = warp_matrix[min_index]
                warp_matrix[row, col] = \
                    (first_values[row] - second_values[col]) ** 2 \
                    + min_dist_to_here
        return warp_matrix

    def warping_path(matrix):
        """
        Creates the warping path along the minimum total distance between the
        two series in the matrix
        @matrix: the warping matrix
        @return: numpy array containing the warping path
        """

        matrix_dim = matrix.shape
        warp_path = np.full((sum(matrix_dim), 3),
                            np.NAN)  # initialize to max length warping path
        wp_index = len(warp_path) - 1  # we start at the end

        wm_index = (matrix_dim[0] - 1, matrix_dim[1] - 1)
        while True:
            # store the distance, index of Ys, index of Xs
            warp_path[wp_index] = matrix[wm_index], wm_index[0], wm_index[1]
            wp_index -= 1

            if wm_index == (0, 0):
                break  # we're finished

            # point to min element
            wm_index = index_of_section_min(matrix, wm_index)

        # remove the NANs
        return warp_path[np.logical_not(np.isnan(warp_path).any(axis=1))]

    def index_of_section_min(matrix, current_index=(0, 0)):
        """
        Reads the values around the given source cell and returns an index to
        the minimum of these values. Respecting the boundaries of the
        original matrix the function evaluates the following values:
        ... __________________
        ... | value | value  | ...
        ... | value | source | ...
        ... | value | value  | ...
            __________________
        ...
        @param matrix: a numpy array
        @param current_index: a tuple with the 2D index pointing to current
        cell
        @return: a tuple containing the index for the matrix pointing to the
                    minimum value within the given section
        """

        section_org_row = max(current_index[0] - 1, 0)
        section_org_col = max(current_index[1] - 1, 0)
        section_end_row = min(current_index[0] + 1, matrix.shape[0])
        section_end_col = current_index[1]
        # copy section
        section = matrix[
                  section_org_row:section_end_row + 1,
                  section_org_col:section_end_col + 1
                  ].copy()

        section_current_row = min(current_index[0], 1)
        section_current_col = min(current_index[1], 1)
        # prevent current cell from being found as minimum
        section[(section_current_row, section_current_col)] = np.inf
        # find minimum cell in section
        min_section_index = np.unravel_index(np.argmin(section, axis=None),
                                             section.shape)
        # find minimum index
        min_row = section_org_row + min_section_index[0]
        min_col = section_org_col + min_section_index[1]
        return min_row, min_col  # point to minimum element from section

    def variance_of_warping_path(path):
        # ToDo: calculate the variance
        return None

    def distance_along_warping_path(path):
        # ToDo: calculate the agdtw distance
        distance = path[-1, 0]
        return distance

    warp_matrix = warping_matrix(first, second)
    warp_path = warping_path(warp_matrix)

    return distance_along_warping_path(warp_path)
