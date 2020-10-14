# -*- coding: utf-8 -*-
import numpy as np
import pytest

from sktime.series_as_features.tests._config import N_COLUMNS
from sktime.series_as_features.tests._config import N_INSTANCES
from sktime.series_as_features.tests._config import N_TIMEPOINTS
from sktime.utils._testing import make_classification_problem
from sktime.utils.data_container import from_3d_numpy_to_nested
from sktime.utils.data_container import from_nested_to_2d_array
from sktime.utils.data_container import from_nested_to_3d_numpy
from sktime.utils.data_container import is_nested_dataframe


@pytest.mark.parametrize("n_instances", N_INSTANCES)
@pytest.mark.parametrize("n_columns", N_COLUMNS)
@pytest.mark.parametrize("n_timepoints", N_TIMEPOINTS)
def test_from_nested_to_3d_numpy(n_instances, n_columns, n_timepoints):
    nested, _ = make_classification_problem(n_instances, n_columns, n_timepoints)
    array = from_nested_to_3d_numpy(nested)

    # check types and shapes
    assert isinstance(array, np.ndarray)
    assert array.shape == (n_instances, n_columns, n_timepoints)

    # check values of random series
    np.testing.assert_array_equal(nested.iloc[1, 0], array[1, 0, :])


@pytest.mark.parametrize("n_instances", N_INSTANCES)
@pytest.mark.parametrize("n_columns", N_COLUMNS)
@pytest.mark.parametrize("n_timepoints", N_TIMEPOINTS)
def test_from_3d_numpy_to_nested(n_instances, n_columns, n_timepoints):
    array = np.random.normal(size=(n_instances, n_columns, n_timepoints))
    nested = from_3d_numpy_to_nested(array)

    # check types and shapes
    assert is_nested_dataframe(nested)
    assert nested.shape == (n_instances, n_columns)
    assert nested.iloc[0, 0].shape[0] == n_timepoints

    # check values of random series
    np.testing.assert_array_equal(nested.iloc[1, 0], array[1, 0, :])


@pytest.mark.parametrize("n_instances", N_INSTANCES)
@pytest.mark.parametrize("n_columns", N_COLUMNS)
@pytest.mark.parametrize("n_timepoints", N_TIMEPOINTS)
def test_from_nested_to_2d_array(n_instances, n_columns, n_timepoints):
    nested, _ = make_classification_problem(n_instances, n_columns, n_timepoints)

    array = from_nested_to_2d_array(nested)
    assert array.shape == (n_instances, n_columns * n_timepoints)
    assert array.index.equals(nested.index)
