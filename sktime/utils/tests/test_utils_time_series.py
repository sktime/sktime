from sktime.utils.time_series import time_series_slope
from sktime.transformers.tests.test_RandomIntervalFeatureExtractor import generate_df_from_array
import numpy as np
from scipy.stats import linregress

N_ITER = 100


def test_time_series_slope():
    Y = np.array(generate_df_from_array(np.random.normal(size=10), n_rows=100).iloc[:, 0].tolist())
    y = Y[0, :]

    # Compare with scipy's linear regression function
    x = np.arange(y.size) + 1
    a = linregress(x, y).slope
    b = time_series_slope(y)
    np.testing.assert_almost_equal(a, b, decimal=10)

    # Check computations over axis
    a = np.apply_along_axis(time_series_slope, 1, Y)
    b = time_series_slope(Y, axis=1)
    np.testing.assert_equal(a, b)

    a = time_series_slope(Y, axis=1)[0]
    b = time_series_slope(y)
    np.testing.assert_equal(a, b)

    # Check linear and constant cases
    for step in [-1, 0, 1]:
        y = np.arange(1, 4) * step
        np.testing.assert_almost_equal(time_series_slope(y), step, decimal=10)




