import numpy as np

from sktime.distances.distance_rework.tests.redo import BaseDistance
from sktime.distances.distance_rework.tests.redo._euclidean import _EuclideanDistance

def _distance_tests(
        dist: BaseDistance,
        x: np.ndarray,
        y: np.ndarray,
        expected_independent: float,
        expected_dependent: float
):
    """Test a BaseDistance object.

    Parameters
    ----------
    dist : BaseDistance
        The distance object to test.
    x : np.ndarray
        The first time series.
    y : np.ndarray
        The second time series.
    expected_independent : float
        The expected result for the independent strategy.
    expected_dependent : float
        The expected result for the dependent strategy.
    """
    independent_result = dist.distance(x, y, strategy="independent")
    dependent_result = dist.distance(x, y, strategy="dependent")
    assert independent_result == expected_independent
    assert dependent_result == expected_dependent

    assert independent_result == dist.independent_distance(x, y)
    assert dependent_result == dist.dependent_distance(x, y)

    independent_distance_factory = dist.distance_factory(x, y, strategy="independent")
    dependent_distance_factory = dist.distance_factory(x, y, strategy="independent")
    assert independent_result == independent_distance_factory(x, y)
    assert dependent_result == dependent_distance_factory(x, y)

    # Not all distances will have a cost matrix so we disable them.
    if dist._has_cost_matrix == False:
        return

    independent_cost_matrix, independent_result = dist.distance(
        x, y, strategy="independent", return_cost_matrix=True
    )
    dependent_cost_matrix, dependent_result = dist.distance(
        x, y, strategy="dependent", return_cost_matrix=True
    )
    assert isinstance(independent_cost_matrix, np.ndarray)
    assert independent_cost_matrix[-1, -1] == independent_result
    assert isinstance(dependent_cost_matrix, np.ndarray)
    assert dependent_cost_matrix[-1, -1] == dependent_result

    independent_path, independent_path_result = dist.distance_alignment_path(
        x, y, strategy="independent"
    )
    dependent_path, dependent_path_result = dist.distance_alignment_path(
        x, y, strategy="dependent"
    )
    assert independent_path_result == independent_result
    assert dependent_path_result == dependent_result
    assert isinstance(independent_path, list)
    assert isinstance(dependent_path, list)

x = np.array(
    [[2, 35, 14, 5, 68, 7.5, 68, 7, 11, 13], [5, 68, 7.5, 68, 7, 11, 13, 5, 68, 7]]
)
y = np.array(
    [[8, 19, 10, 12, 68, 7.5, 60, 7, 10, 14], [15, 12, 4, 62, 17, 10, 3, 15, 48, 7]]
)

def _get_test_result(dist: BaseDistance):
    """Utility method to get the results of a distance test quickly."""
    print("\n")
    independent_result = dist.distance(x, y, strategy="independent")
    dependent_result = dist.distance(x, y, strategy="dependent")
    obj_type = str(type(dist)).split('.')[-1].split("'")[0]
    print(f'_distance_tests({obj_type}(), x, y, {independent_result}, {dependent_result})')

def test_euclidean_distance():
    dist = _EuclideanDistance()
    x = np.array([2, 35, 14, 5, 68, 7.5, 68, 7, 11, 13])
    y = np.array([5, 68, 7.5, 68, 7, 11, 13, 5, 68, 7])
    independent_result = dist.distance(x, y, strategy="independent")
    dependent_result = dist.distance(x, y, strategy="dependent")
    joe = ''
    assert independent_result == 66.39465339920075
    assert dependent_result == 66.39465339920075

def test_dtw():
    pass