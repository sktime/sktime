__author__ = "Ansgar Asseburg"
__email__ = "devaa@donnerluetjen.de"

import pytest
from sktime.distances.tests._config import TEST_YS


"""
run on commandline from root with:
    ptw --runner "pytest sktime/distances/tests/test_agdtw.py" 
        -- --last-failed --new-first
"""


def pytest_assertrepr_compare(op, left, right):
    import numpy as np
    if op == '==' and (
            isinstance(left, np.array) and isinstance(right, np.array)
    ) or (
            isinstance(left, tuple) and isinstance(right, tuple)
    ):
        return [f'{left} in {right}']


def test_euclidean_distance():
    import sktime.distances.agdtw as agdtw

    assert agdtw.euclidean_distance(0, 0) == 0
    assert agdtw.euclidean_distance(1, 1) == 0
    assert agdtw.euclidean_distance(3, 4) == 1
    assert agdtw.euclidean_distance(4, 3) == 1


def test_dynamic_section():
    import numpy as np
    import sktime.distances.agdtw as agdtw

    test_matrix = np.array([
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8]
    ])
    test_range = np.array([
        [[[0], [3]], (0, 0)],
        [[[0, 1], [3, 4]], (0, 1)],
        [[[1, 2], [4, 5]], (0, 2)],
        [[[0], [3], [6]], (1, 0)],
        [[[0, 1], [3, 4], [6, 7]], (1, 1)],
        [[[1, 2], [4, 5], [7, 8]], (1, 2)],
        [[[3], [6]], (2, 0)],
        [[[3, 4], [6, 7]], (2, 1)],
        [[[4, 5], [7, 8]], (2, 2)]
    ])
    for expected_section, source_indices in test_range:
        actual_section = agdtw.dynamic_section(test_matrix, source_indices)
        assert (expected_section == actual_section).all(), \
            f'source_indices: {tuple(source_indices)}'


def test_index_of_section_min():
    import numpy as np
    import sktime.distances.agdtw as agdtw

    test_matrix = np.array([
        [0, 1, 2],
        [3, 4, 5],
        [1, 7, 0]
    ])
    test_range = np.array([
        [(0, 0), (0, 0)],
        [(0, 0), (0, 1)],
        [(0, 1), (0, 2)],
        [(0, 0), (1, 0)],
        [(0, 0), (1, 1)],
        [(2, 2), (1, 2)],
        [(1, 0), (2, 0)],
        [(2, 0), (2, 1)],
        [(1, 1), (2, 2)]
    ])
    for expected_indices, source_indices in test_range:
        actual_indices = agdtw.index_of_section_min_around(test_matrix,
                                                           tuple(source_indices))
        assert (expected_indices == actual_indices).all(), \
            f'at source indices: {tuple(source_indices)}'


def test_warping_matrix_with_window_1():
    import numpy as np
    import sktime.distances.agdtw as agdtw

    series_1 = np.array([1, 2, 3, 4, 5])
    series_2 = np.array([1, 2, 3, 4, 5])
    expected_result = np.array([[0, 1, 5, 14, 30],
                                [1, 0, 1, 5, 14],
                                [5, 1, 0, 1, 5],
                                [14, 5, 1, 0, 1],
                                [30, 14, 5, 1, 0]])
    actual_result = agdtw.warping_matrix(series_1, series_2)

    assert actual_result.shape == expected_result.shape
    assert (actual_result == expected_result).all()


def test_warping_matrix_with_window_a_quarter():
    import numpy as np
    import sktime.distances.agdtw as agdtw

    series_1 = np.array([1, 2, 3, 4, 5])
    series_2 = np.array([1, 2, 3, 4, 5])
    expected_result = np.array([[0, 1, np.inf, np.inf, np.inf],
                                [1, 0, 1, np.inf, np.inf],
                                [np.inf, 1, 0, 1, np.inf],
                                [np.inf, np.inf, 1, 0, 1],
                                [np.inf, np.inf, np.inf, 1, 0]])
    actual_result = agdtw.warping_matrix(series_1, series_2, window=.25)

    assert actual_result.shape == expected_result.shape
    assert (actual_result == expected_result).all()


def test_warping_matrix_with_window_as_zero():
    import numpy as np
    import sktime.distances.agdtw as agdtw

    series_1 = np.array([1, 2, 3, 4, 5])
    series_2 = np.array([1, 2, 3, 4, 5])
    expected_result = np.array([[0, np.inf, np.inf, np.inf, np.inf],
                                [np.inf, 0, np.inf, np.inf, np.inf],
                                [np.inf, np.inf, 0, np.inf, np.inf],
                                [np.inf, np.inf, np.inf, 0, np.inf],
                                [np.inf, np.inf, np.inf, np.inf, 0]])
    actual_result = agdtw.warping_matrix(series_1, series_2, window=0)

    assert actual_result.shape == expected_result.shape
    assert (actual_result == expected_result).all()


def test_warping_path_with_symmetric_series():
    import numpy as np
    import sktime.distances.agdtw as agdtw

    series_1 = np.array([1, 2, 3, 4, 5])
    series_2 = np.array([1, 2, 3, 4, 5])
    expected_result = np.array([
        [0, 0, 0],
        [0, 1, 1],
        [0, 2, 2],
        [0, 3, 3],
        [0, 4, 4]
    ])
    matrix = agdtw.warping_matrix(series_1, series_2)
    actual_result = agdtw.warping_path(matrix, series_1, series_2)
    assert (expected_result == actual_result).all()


def test_warping_path_with_longer_second_series():
    import numpy as np
    import sktime.distances.agdtw as agdtw

    series_1 = np.array([1, 2, 3, 2, 2])
    series_2 = np.array([5, 7, 4, 4, 3, 2])
    expected_result = np.array([
        [4, 0, 0],
        [5, 1, 1],
        [1, 2, 2],
        [1, 2, 3],
        [0, 2, 4],
        [0, 3, 5],
        [0, 4, 5]
    ])
    matrix = agdtw.warping_matrix(series_1, series_2)
    actual_result = agdtw.warping_path(matrix, series_1, series_2)
    assert (expected_result == actual_result).all()


def test_warping_path_with_longer_first_series():
    import numpy as np
    import sktime.distances.agdtw as agdtw

    series_2 = np.array([1, 2, 3, 2, 2])
    series_1 = np.array([5, 7, 4, 4, 3, 2])
    expected_result = np.array([
        [4, 0, 0],
        [5, 1, 1],
        [1, 2, 2],
        [1, 3, 2],
        [0, 4, 2],
        [0, 5, 3],
        [0, 5, 4]
    ])
    matrix = agdtw.warping_matrix(series_1, series_2)
    actual_result = agdtw.warping_path(matrix, series_1, series_2)
    assert (expected_result == actual_result).all()


def test_kernel_distance_with_sigma_1():
    import numpy as np
    import sktime.distances.agdtw as agdtw
    sample_path = np.array([
        [1, 0, 0],
        [3, 1, 1],
        [3, 2, 2],
        [5, 2, 3]
    ])
    sigma = 1.0
    expected_result = \
        np.exp(-(abs(1 / sigma) ** 2)) + \
        np.exp(-(abs(3 / sigma) ** 2)) + \
        np.exp(-(abs(3 / sigma) ** 2)) + \
        np.exp(-(abs(5 / sigma) ** 2))
    actual_result = agdtw.kernel_distance(sample_path, sigma)
    assert expected_result == pytest.approx(actual_result)


def test_kernel_distance_with_sigma_one_half():
    import numpy as np
    import sktime.distances.agdtw as agdtw
    sample_path = np.array([
        [1, 0, 0],
        [3, 1, 1],
        [3, 2, 2],
        [5, 2, 3]
    ])
    sigma = .5
    expected_result = \
        np.exp(-(abs(1 / sigma) ** 2)) + \
        np.exp(-(abs(3 / sigma) ** 2)) + \
        np.exp(-(abs(3 / sigma) ** 2)) + \
        np.exp(-(abs(5 / sigma) ** 2))
    actual_result = agdtw.kernel_distance(sample_path, sigma)
    assert expected_result == pytest.approx(actual_result)


def test_kernel_distance_with_sigma_zero():
    import numpy as np
    import sktime.distances.agdtw as agdtw
    sample_path = np.array([
        [1, 0, 0],
        [3, 1, 1],
        [3, 2, 2],
        [5, 2, 3]
    ])
    sigma = 0.0
    with pytest.raises(ZeroDivisionError):
        agdtw.kernel_distance(sample_path, sigma)


def test_kernel_distance_with_sigma_thirtythree():
    import numpy as np
    import sktime.distances.agdtw as agdtw
    sample_path = np.array([
        [1, 0, 0],
        [3, 1, 1],
        [3, 2, 2],
        [5, 2, 3]
    ])
    sigma = 33
    expected_result = \
        np.exp(-(abs(1 / sigma) ** 2)) + \
        np.exp(-(abs(3 / sigma) ** 2)) + \
        np.exp(-(abs(3 / sigma) ** 2)) + \
        np.exp(-(abs(5 / sigma) ** 2))
    actual_result = agdtw.kernel_distance(sample_path, sigma)
    assert expected_result == pytest.approx(actual_result)

@pytest.mark.parametrize("series_1, series_2", TEST_YS)
def test_agdtw_distance_returns_single_value(series_1, series_2):
    import numpy as np
    from numbers import Number
    import sktime.distances.agdtw as agdtw

    actual_result = agdtw.agdtw_distance(series_1, series_2)
    assert isinstance(actual_result, Number)
