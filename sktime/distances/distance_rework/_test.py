import numpy as np
from numba import njit
import numba as nb

from sktime.distances.tests._utils import create_test_distance_numpy


TIME_SERIES_1D = 'float64[:]'
TIME_SERIES_2D = 'float64[:, :]'
NUMBA_COST_MATRIX = 'float64[:, :]'
NUMBA_PATH = 'int16[:, :]'
DISTANCE = 'float64'

INDEPENDENT_COST_MATRIX_DISTANCE_SIGNATURE = 'Tuple((float64[:, :], float64))(float64[:], float64[:])'
DEPENDENT_SIGNATURE = 'Tuple((float64[:, :], float64))(float64[:, :], float64[:, :])'


# @njit('Tuple((float64[:, :], float64))(float64[:], float64[:])', cache=True)
@njit(f'Tuple(({NUMBA_COST_MATRIX}, {DISTANCE}))({TIME_SERIES_1D}, {TIME_SERIES_1D})', cache=True)
def independent_dtw(x: np.ndarray, y: np.ndarray):
    print(x.ndim)
    print(y.ndim)
    return np.array([[1.0,  2.0], [2.3, 2.3]]), 3.0

# @njit('Tuple((float64[:, :], float64))(float64[:, :], float64[:, :])', cache=True)
@njit(f'Tuple(({NUMBA_COST_MATRIX}, {DISTANCE}))({TIME_SERIES_2D}, {TIME_SERIES_2D})', cache=True)
def dependent_dtw(x: np.ndarray, y: np.ndarray):
    print(x.ndim)
    print(y.ndim)
    return np.array([[1.0,  2.0], [2.3, 2.3]]), 3.0

@njit()
def test_call(ind_x, ind_y, dep_x, dep_y):
    ind_cm, ind_dist = independent_dtw(ind_x, ind_y)
    dep_cm, dep_dist = dependent_dtw(dep_x, dep_y)
    return ind_dist


def test_other():
    print("''")
    print(f'Tuple(({NUMBA_COST_MATRIX}, {DISTANCE}))({TIME_SERIES_1D}, {TIME_SERIES_1D})')
    print('Tuple((float64[:, :], float64))(float64[:, :], float64[:, :])')

def test_dtw():
    distances_ts = create_test_distance_numpy(2, 5, 5)
    x_ind = distances_ts[0][0]
    y_ind = distances_ts[1][0]
    x_dep = distances_ts[0]
    y_dep = distances_ts[1]

    test = test_call(x_ind, y_ind, x_dep, y_dep)

    joe = ''

    pass

