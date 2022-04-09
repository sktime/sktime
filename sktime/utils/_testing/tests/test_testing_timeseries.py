import numpy as np
import pandas as pd
from typing import Union, List

from sktime.datatypes import convert_to, Mtype, get_examples, mtype
from sktime.utils._testing.testing_timeseries import TestingTimeseries
from sktime.utils._testing.series import _make_series
from sktime.utils._testing.panel import _make_panel_X
from sktime.utils.validation.panel import from_nested_to_3d_numpy


def example_ts_func(x: Union[np.ndarray, pd.Series, pd.DataFrame], convert_type: Mtype):
    converted_x = convert_to(x, convert_type)
    return np.sum(converted_x)


def test_series_timeseries():
    # Univariate example
    univariate = TestingTimeseries.create_univariate_series(10)
    expected_result = np.sum(univariate.as_type(Mtype.NP_NDARRAY))
    for test_type in univariate.iter_timeseries_variations():
        assert example_ts_func(test_type, Mtype.NP_NDARRAY) == expected_result

    # Multivariate example
    multivariate = TestingTimeseries.create_multivariate_series(10, 10)
    expected_result = np.sum(multivariate.as_type(Mtype.NP_NDARRAY))
    for test_type in multivariate.iter_timeseries_variations():
        assert example_ts_func(test_type, Mtype.NP_NDARRAY) == expected_result


def test_panel_timeseries():
    multivariate_panel = TestingTimeseries.create_multivariate_panel(10, 10, 10)
    expected_result = np.sum(multivariate_panel.as_type(Mtype.NUMPY_3D))
    for ts, ts_type in multivariate_panel.iter_timeseries_variations_with_type():
        assert example_ts_func(ts, Mtype.NUMPY_3D) == expected_result


def test_example():
    # univariate_series = _make_series(10, 1)
    # c_uni_series = convert_to(univariate_series, Mtype.NP_NDARRAY)
    #
    # multivariate_series = _make_series(10, 10)
    # c_multi_series = convert_to(multivariate_series, Mtype.NP_NDARRAY)
    #
    # univariate_panel = _make_panel_X(n_instances=10, n_columns=1, n_timepoints=10)
    # c_uni_panel = convert_to(univariate_panel, Mtype.NUMPY_3D)

    #  nested_univ

    multivariate_panel = _make_panel_X(n_instances=10, n_columns=10, n_timepoints=10)
    c_multi_panel = convert_to(multivariate_panel, Mtype.NUMPY_3D)
    test = mtype(multivariate_panel, as_scitype='Panel')
    pass
