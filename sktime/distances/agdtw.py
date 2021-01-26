__author__ = "Ansgar Asseburg"
__email__ = "devaa@donnerluetjen.de"

import numpy as np


def agdtw_distance(first, second, window=1, sigma=1.0):
    """
    idea and algorithm was taken from:

    @inproceedings{XueZTWL17,
      author    = {Yangtao Xue and
                   Li Zhang and
                   Zhiwei Tao and
                   Bangjun Wang and
                   Fanzhang Li},
      editor    = {Derong Liu and
                   Shengli Xie and
                   Yuanqing Li and
                   Dongbin Zhao and
                   El{-}Sayed M. El{-}Alfy},
      title     = {An Altered Kernel Transformation for Time Series
      Classification},
      booktitle = {Neural Information Processing - 24th International
      Conference, {ICONIP}
                   2017, Guangzhou, China, November 14-18, 2017,
                   Proceedings, Part {V}},
      series    = {Lecture Notes in Computer Science},
      volume    = {10638},
      pages     = {455--465},
      publisher = {Springer},
      year      = {2017},
      url       = {https://doi.org/10.1007/978-3-319-70139-4\_46},
      doi       = {10.1007/978-3-319-70139-4\_46},
      timestamp = {Tue, 14 May 2019 10:00:42 +0200},
      biburl    = {https://dblp.org/rec/conf/iconip/XueZTWL17.bib},
      bibsource = {dblp computer science bibliography, https://dblp.org}
    }

    the method accepts two univariate time series, eg. 2D single row arrays
    @param first: numpy array containing the first time series
    @param second: numpy array containing the second time series
    @param window: float representing the window width as ratio of the window
    and the longer series
    @param sigma: float representing the kernel parameter
    @return: a float containing the kernel distance
    """

    # make sure time series are univariate
    if first.shape[0] * second.shape[0] != 1:
        raise ValueError("time series must be univariate!")

    # reduce series to 1D arrays
    first = first.squeeze()
    second = second.squeeze()
    pairwise_distances = get_pairwise_distances(first, second)
    warp_matrix = warping_matrix(pairwise_distances, window)
    warp_path = squared_euclidean_along_warp_path(warp_matrix,
                                                  pairwise_distances)

    return kernel_distance(warp_path, sigma)


def get_pairwise_distances(first, second):
    """
    calculates the pairwise squared euclidean distances for the two series
    @param first: np.array containing the first series
    @param second: np.array containing the second series
    @return: np.array containing a matrix with pairwise squared euclidean
    distances
    """
    
    return np.power(np.subtract.outer(first, second) ** 2, 2)


def warping_matrix(pairwise_distances, window=1.0):
    """
    Creates the warping matrix while respecting a given window
    *** part of the code was adopted from elastic.py by Jason Lines ***
    @param series_1: numpy array containing the first series
    @param series_2: numpy array containing the second series
    @param window: float representing the window width as ratio of the
    window and the longer series
    @return: 2D numpy array containing the minimum squared distances
    """

    row_dim = pairwise_distances.shape[0]
    col_dim = pairwise_distances.shape[1]

    # 1 >= window >= 0
    window = min(1.0, abs(window))
    # working on indices requires absolute_window_size to be integer
    absolute_window_size = int(min(row_dim, col_dim) * window)

    # initialise matrix
    warp_matrix = np.full([row_dim, col_dim], np.inf)

    warp_matrix[0][0] = pairwise_distances[0][0]
    # then initialise edges of the warping matrix with accumulated distances
    for i in range(1, absolute_window_size):
        warp_matrix[0][i] = pairwise_distances[0][i] + warp_matrix[0][
            i - 1]
        warp_matrix[i][0] = pairwise_distances[i][0] + warp_matrix[i - 1][
            0]

    # now visit all allowed cells, calculate the value as the distance
    # in this cell + min(top, left, or top-left)
    # traverse each row,
    for row in range(1, row_dim):
        # traverse left and right by the allowed window
        window_range_start = max(1, row - absolute_window_size)
        window_range_end = min(col_dim, row + absolute_window_size + 1)
        for column in range(window_range_start, window_range_end):
            # if not 1 <= column < col_dim:
            #     continue

            # find smallest entry in the warping matrix, either above,
            # to the left, or diagonally left and up
            above = warp_matrix[row - 1][column]
            left = warp_matrix[row][column - 1]
            diag = warp_matrix[row - 1][column - 1]

            # add the pairwise distance for [row][column] to the minimum
            # of the three possible potential cells
            warp_matrix[row][column] = \
                pairwise_distances[row][column] + min(above, left, diag)

    return warp_matrix


def euclidean_distance(x, y):
    """
    Calculates the euclidean distance between two elements of two
    univariate time series
    @param x: element from first series
    @param y: element from second series
    @return euclidean distance
    """
    return abs(x - y)


def squared_euclidean_along_warp_path(matrix, pairwise_distances):
    """
    Creates the warping path along the minimum total distance between the
    two series in the matrix
    @matrix: the warping matrix
    @first: numpy array containing the first time series
    @second: numpy array containing the second time series
    @return: numpy array containing the euclidean distance along the warp path
    between the relevant elements in the form [dist_eu, i_s, j_s
    """
    matrix_dim = matrix.shape
    warp_path = np.full((sum(matrix_dim), 1), np.NAN)
    # initialize to max length warping path
    wp_index = len(warp_path) - 1  # we start at the end

    wm_index = (matrix_dim[0] - 1, matrix_dim[1] - 1)
    while True:
        # store the distances, index of Ys, index of Xs
        warp_path[wp_index] = [pairwise_distances[wm_index]]
        wp_index -= 1

        if wm_index == (0, 0):
            break  # we're finished

        # point to min element
        wm_index = index_of_section_min_around(matrix, wm_index)

    # remove the remaining NANs
    return warp_path[~ (np.isnan(warp_path).any(axis=1))]


def dynamic_section(matrix, current_index=(0, 0)):
    """
    returns a section of the given matrix neighboring the current cell to
    the left and above and below
    @param matrix: a numpy array containing the original matrix
    @param current_index: a tuple of integers representing the index of the
    current cell into the matrix
    @return: a numpy array containing the desired section
    """
    section_org_row = max(current_index[0] - 1, 0)
    section_org_col = max(current_index[1] - 1, 0)
    section_end_row = min(current_index[0] + 1, matrix.shape[0])
    section_end_col = current_index[1]
    # copy section and make its elements float to for allow np.inf
    section = matrix[
              section_org_row:section_end_row + 1,
              section_org_col:section_end_col + 1
              ].copy().astype(float)
    return section


def index_of_section_min_around(matrix, current_index=(0, 0)):
    """
    Reads the values around the given current cell and returns an index to
    the minimum of these values. Respecting the boundaries of the
    original matrix the function evaluates the following values:
    ... ___________________
    ... | value | value   | ...
    ... | value | current | ...
    ... | value | value   | ...
        ___________________
    ...
    @param matrix: numpy array containing the warping matrix
    @param current_index: tuple with the 2D index pointing to current cell
    @return: a tuple containing the index for the matrix pointing to the
             minimum value within the given section
    """

    # if at origin of matrix return (0, 0)
    if current_index == (0, 0):
        return current_index
    section = dynamic_section(matrix, current_index)

    section_current_row = min(current_index[0], 1)
    section_current_col = min(current_index[1], 1)
    # prevent current cell from being found as minimum
    section[(section_current_row, section_current_col)] = np.inf
    # find minimum cell in section
    min_section_index = np.unravel_index(np.argmin(section, axis=None),
                                         section.shape)
    # find minimum index
    section_org_row = max(current_index[0] - 1, 0)
    section_org_col = max(current_index[1] - 1, 0)
    min_row = section_org_row + min_section_index[0]
    min_col = section_org_col + min_section_index[1]
    return min_row, min_col  # point to minimum element from section


def kernel_distance(squared_euclidean_distances, sigma):
    """
    calculates the kernel distance by processing each individual distance
    along the warping squared_euclidean_distances
    @param squared_euclidean_distances: numpy array containing the warping
    squared_euclidean_distances with euclidean
    distance between the relevant elements in the form [dist_eu, i_s, j_s
    @param sigma: float representing the kernel parameter
    @return: float containing the kernel distance
    """
    if sigma == 0:
        raise ZeroDivisionError(
            "The kernel parameter <sigma> must not be zero, "
            "since we're using it for a division.")
    normalized_distances = squared_euclidean_distances / (sigma ** 2)
    kernel_distances = np.exp(-normalized_distances).squeeze()
    return sum(kernel_distances)


if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    from sktime.datasets import load_UCR_UEA_dataset
    from sktime.classification.distance_based import \
        KNeighborsTimeSeriesClassifier
    X, y = load_UCR_UEA_dataset("SwedishLeaf", return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    knn = KNeighborsTimeSeriesClassifier(n_neighbors=1, metric="agdtw",
                                         metric_params={'window': 1,
                                                        'sigma': 1})
    knn.fit(X_train, y_train)
    knn.score(X_test, y_test)
    # from numpy import random as rd
    #
    # d = agdtw_distance(rd.uniform(50, 100, (1, rd.randint(50, 100))),
    #                    rd.uniform(50, 100, (1, rd.randint(50, 100))))
    # print(d)
