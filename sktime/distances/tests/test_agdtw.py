__author__ = "Ansgar Asseburg"
__email__ = "devaa@donnerluetjen.de"

import pytest
import sktime.distances.tests._config as cfg


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
    pairwise_distances = agdtw.get_pairwise_distances(series_1, series_2)
    expected_result = np.array([[0, 1, 5, 14, 30],
                                [1, 0, 1, 5, 14],
                                [5, 1, 0, 1, 5],
                                [14, 5, 1, 0, 1],
                                [30, 14, 5, 1, 0]])
    actual_result = agdtw.warping_matrix(pairwise_distances)

    assert actual_result.shape == expected_result.shape
    assert (actual_result == expected_result).all()


def test_warping_matrix_with_window_a_quarter():
    import numpy as np
    import sktime.distances.agdtw as agdtw

    series_1 = np.array([1, 2, 3, 4, 5])
    series_2 = np.array([1, 2, 3, 4, 5])
    pairwise_distances = agdtw.get_pairwise_distances(series_1, series_2)
    expected_result = np.array([[0, np.inf, np.inf, np.inf, np.inf],
                                [np.inf, 0, 1, np.inf, np.inf],
                                [np.inf, 1, 0, 1, np.inf],
                                [np.inf, np.inf, 1, 0, 1],
                                [np.inf, np.inf, np.inf, 1, 0]])
    actual_result = agdtw.warping_matrix(pairwise_distances, window=.25)

    assert actual_result.shape == expected_result.shape
    assert (actual_result == expected_result).all()


def test_warping_matrix_with_window_as_zero():
    import numpy as np
    import sktime.distances.agdtw as agdtw

    series_1 = np.array([1, 2, 3, 4, 5])
    series_2 = np.array([1, 2, 3, 4, 5])
    pairwise_distances = agdtw.get_pairwise_distances(series_1, series_2)
    expected_result = np.array([[0, np.inf, np.inf, np.inf, np.inf],
                                [np.inf, 0, np.inf, np.inf, np.inf],
                                [np.inf, np.inf, 0, np.inf, np.inf],
                                [np.inf, np.inf, np.inf, 0, np.inf],
                                [np.inf, np.inf, np.inf, np.inf, 0]])
    actual_result = agdtw.warping_matrix(pairwise_distances, window=0)

    assert actual_result.shape == expected_result.shape
    assert (actual_result == expected_result).all()


def test_warping_path_with_symmetric_series():
    import numpy as np
    import sktime.distances.agdtw as agdtw

    series_1 = np.array([1, 2, 3, 4, 5])
    series_2 = np.array([1, 2, 3, 4, 5])
    pairwise_distances = agdtw.get_pairwise_distances(series_1, series_2)
    expected_result = np.array([
        [0],
        [0],
        [0],
        [0],
        [0]
    ])
    matrix = agdtw.warping_matrix(pairwise_distances)
    actual_result = agdtw.squared_euclidean_along_warp_path(matrix,
                                                            pairwise_distances)
    assert (expected_result == actual_result).all()


def test_warping_path_with_longer_second_series():
    import numpy as np
    import sktime.distances.agdtw as agdtw

    series_1 = np.array([1, 2, 3, 2, 2])
    series_2 = np.array([5, 7, 4, 4, 3, 2])
    pairwise_distances = agdtw.get_pairwise_distances(series_1, series_2)
    expected_result = np.array([
        [16],
        [25],
        [1],
        [1],
        [0],
        [0],
        [0]
    ])
    matrix = agdtw.warping_matrix(pairwise_distances)
    actual_result = agdtw.squared_euclidean_along_warp_path(matrix,
                                                            pairwise_distances)
    assert (expected_result == actual_result).all()


def test_warping_path_with_longer_first_series():
    import numpy as np
    import sktime.distances.agdtw as agdtw

    series_2 = np.array([1, 2, 3, 2, 2])
    series_1 = np.array([5, 7, 4, 4, 3, 2])
    pairwise_distances = agdtw.get_pairwise_distances(series_1, series_2)
    expected_result = np.array([
        [16],
        [25],
        [1],
        [1],
        [0],
        [0],
        [0]
    ])
    matrix = agdtw.warping_matrix(pairwise_distances)
    actual_result = agdtw.squared_euclidean_along_warp_path(matrix,
                                                            pairwise_distances)
    assert (expected_result == actual_result).all()


def test_kernel_distance_with_sigma_1():
    import numpy as np
    import sktime.distances.agdtw as agdtw
    sample_path = np.array([
        [1],
        [4],
        [4],
        [25]
    ])
    sigma = 1.0
    expected_result = \
        np.exp(-1 / (sigma ** 2)) + \
        np.exp(-4 / (sigma ** 2)) + \
        np.exp(-4 / (sigma ** 2)) + \
        np.exp(-25 / (sigma ** 2))
    actual_result = agdtw.kernel_distance(sample_path, sigma)
    assert expected_result == pytest.approx(actual_result)


def test_kernel_distance_with_sigma_one_half():
    import numpy as np
    import sktime.distances.agdtw as agdtw
    sample_path = np.array([
        [1],
        [4],
        [4],
        [25]
    ])
    sigma = .5
    expected_result = \
        np.exp(-1 / (sigma ** 2)) + \
        np.exp(-4 / (sigma ** 2)) + \
        np.exp(-4 / (sigma ** 2)) + \
        np.exp(-25 / (sigma ** 2))
    actual_result = agdtw.kernel_distance(sample_path, sigma)
    assert expected_result == pytest.approx(actual_result)


def test_kernel_distance_throws_with_sigma_zero():
    import numpy as np
    import sktime.distances.agdtw as agdtw
    sample_path = np.array([
        [1],
        [4],
        [4],
        [25]
    ])
    sigma = 0.0
    with pytest.raises(ZeroDivisionError):
        agdtw.kernel_distance(sample_path, sigma)


def test_kernel_distance_with_sigma_thirtythree():
    import numpy as np
    import sktime.distances.agdtw as agdtw
    sample_path = np.array([
        [1],
        [4],
        [4],
        [25]
    ])
    sigma = 33
    expected_result = \
        np.exp(-1 / (sigma ** 2)) + \
        np.exp(-4 / (sigma ** 2)) + \
        np.exp(-4 / (sigma ** 2)) + \
        np.exp(-25 / (sigma ** 2))
    actual_result = agdtw.kernel_distance(sample_path, sigma)
    assert expected_result == pytest.approx(actual_result)


@pytest.mark.parametrize("series_1, series_2", cfg.MULTIVARIATES)
def test_agdtw_distance_throws_for_multivariates(series_1, series_2):
    import numpy as np
    import sktime.distances.agdtw as agdtw
    with pytest.raises(ValueError) as e_info:
        agdtw.agdtw_distance(series_1, series_2)
    assert "univariate" in str(e_info.value)


@pytest.mark.parametrize("series_1, series_2", cfg.UNIVARIATES)
def test_agdtw_distance_returns_single_value(series_1, series_2):
    import numpy as np
    from numbers import Number
    import sktime.distances.agdtw as agdtw

    actual_result = agdtw.agdtw_distance(series_1, series_2)
    assert isinstance(actual_result, Number)


@pytest.mark.parametrize("series_1, series_2, correct_result", cfg.SAMPLE)
def test_agdtw_distance_returns_correct_result(series_1, series_2,
                                               correct_result):
    import numpy as np
    from numbers import Number
    import sktime.distances.agdtw as agdtw

    actual_result = agdtw.agdtw_distance(series_1, series_2)
    assert correct_result == pytest.approx(actual_result)
