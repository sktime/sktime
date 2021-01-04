# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pytest
from sktime.series_as_features.tests._config import N_COLUMNS
from sktime.series_as_features.tests._config import N_INSTANCES
from sktime.series_as_features.tests._config import N_TIMEPOINTS
from sktime.utils._testing.panel import make_classification_problem
from sktime.utils.data_io import make_multi_index_dataframe
from sktime.utils.data_processing import from_3d_numpy_to_nested
from sktime.utils.data_processing import from_nested_to_2d_array
from sktime.utils.data_processing import from_nested_to_3d_numpy
from sktime.utils.data_processing import from_3d_numpy_to_2d_array
from sktime.utils.data_processing import from_multi_index_to_3d_numpy
from sktime.utils.data_processing import from_3d_numpy_to_multi_index
from sktime.utils.data_processing import from_multi_index_to_nested
from sktime.utils.data_processing import from_nested_to_multi_index
from sktime.utils.data_processing import is_nested_dataframe
from sktime.utils.data_processing import are_columns_nested
from sktime.utils.data_processing import nested_dataframes_equal


@pytest.mark.parametrize("n_instances", N_INSTANCES)
@pytest.mark.parametrize("n_columns", N_COLUMNS)
@pytest.mark.parametrize("n_timepoints", N_TIMEPOINTS)
def test_are_columns_nested(n_instances, n_columns, n_timepoints):
    nested, _ = make_classification_problem(n_instances, n_columns, n_timepoints)
    zero_df = pd.DataFrame(np.zeros_like(nested))
    nested_heterogenous1 = pd.concat([zero_df, nested], axis=1)
    nested_heterogenous2 = nested.copy()
    nested_heterogenous2["primitive_col"] = 1.0

    assert [*are_columns_nested(nested)] == [True] * n_columns
    assert [*are_columns_nested(nested_heterogenous1)] == [False] * n_columns + [
        True
    ] * n_columns
    assert [*are_columns_nested(nested_heterogenous2)] == [True] * n_columns + [False]


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


@pytest.mark.parametrize("n_instances", N_INSTANCES)
@pytest.mark.parametrize("n_columns", N_COLUMNS)
@pytest.mark.parametrize("n_timepoints", N_TIMEPOINTS)
def test_from_3d_numpy_to_2d_array(n_instances, n_columns, n_timepoints):
    array = np.random.normal(size=(n_instances, n_columns, n_timepoints))
    array_2d = from_3d_numpy_to_2d_array(array)

    assert array_2d.shape == (n_instances, n_columns * n_timepoints)


@pytest.mark.parametrize("n_instances", N_INSTANCES)
@pytest.mark.parametrize("n_columns", N_COLUMNS)
@pytest.mark.parametrize("n_timepoints", N_TIMEPOINTS)
def test_from_multi_index_to_3d_numpy(n_instances, n_columns, n_timepoints):
    mi_df = make_multi_index_dataframe(
        n_instances=n_instances, n_timepoints=n_timepoints, n_columns=n_columns
    )

    array = from_multi_index_to_3d_numpy(
        mi_df, instance_index="case_id", time_index="reading_id"
    )

    assert isinstance(array, np.ndarray)
    assert array.shape == (n_instances, n_columns, n_timepoints)


@pytest.mark.parametrize("n_instances", N_INSTANCES)
@pytest.mark.parametrize("n_columns", N_COLUMNS)
@pytest.mark.parametrize("n_timepoints", N_TIMEPOINTS)
def test_from_3d_numpy_to_multi_index(n_instances, n_columns, n_timepoints):
    array = np.random.normal(size=(n_instances, n_columns, n_timepoints))

    mi_df = from_3d_numpy_to_multi_index(
        array, instance_index=None, time_index=None, column_names=None
    )

    col_names = ["column_" + str(i) for i in range(n_columns)]
    mi_df_named = from_3d_numpy_to_multi_index(
        array, instance_index="case_id", time_index="reading_id", column_names=col_names
    )

    assert isinstance(mi_df, pd.DataFrame)
    assert mi_df.index.names == ["instances", "timepoints"]
    assert (mi_df.columns == ["var_" + str(i) for i in range(n_columns)]).all()

    assert isinstance(mi_df_named, pd.DataFrame)
    assert mi_df_named.index.names == ["case_id", "reading_id"]
    assert (mi_df_named.columns == col_names).all()


@pytest.mark.parametrize("n_instances", N_INSTANCES)
@pytest.mark.parametrize("n_columns", N_COLUMNS)
@pytest.mark.parametrize("n_timepoints", N_TIMEPOINTS)
def test_from_multi_index_to_nested(n_instances, n_columns, n_timepoints):
    mi_df = make_multi_index_dataframe(
        n_instances=n_instances, n_timepoints=n_timepoints, n_columns=n_columns
    )
    nested_df = from_multi_index_to_nested(
        mi_df, instance_index="case_id", cells_as_numpy=False
    )

    assert is_nested_dataframe(nested_df)
    assert nested_df.shape == (n_instances, n_columns)
    assert (nested_df.columns == mi_df.columns).all()


@pytest.mark.parametrize("n_instances", N_INSTANCES)
@pytest.mark.parametrize("n_columns", N_COLUMNS)
@pytest.mark.parametrize("n_timepoints", N_TIMEPOINTS)
def test_from_nested_to_multi_index(n_instances, n_columns, n_timepoints):
    nested, _ = make_classification_problem(n_instances, n_columns, n_timepoints)
    mi_df = from_nested_to_multi_index(
        nested, instance_index="case_id", time_index="reading_id"
    )

    # n_timepoints_max = nested.applymap(_nested_cell_timepoints).sum().max()

    assert isinstance(mi_df, pd.DataFrame)
    assert mi_df.shape == (n_instances * n_timepoints, n_columns)
    assert mi_df.index.names == ["case_id", "reading_id"]


@pytest.mark.parametrize("n_instances", N_INSTANCES)
@pytest.mark.parametrize("n_columns", N_COLUMNS)
@pytest.mark.parametrize("n_timepoints", N_TIMEPOINTS)
def test_is_nested_dataframe(n_instances, n_columns, n_timepoints):
    array = np.random.normal(size=(n_instances, n_columns, n_timepoints))
    nested, _ = make_classification_problem(n_instances, n_columns, n_timepoints)
    zero_df = pd.DataFrame(np.zeros_like(nested))
    nested_heterogenous = pd.concat([zero_df, nested], axis=1)

    mi_df = make_multi_index_dataframe(
        n_instances=n_instances, n_timepoints=n_timepoints, n_columns=n_columns
    )

    assert not is_nested_dataframe(array)
    assert not is_nested_dataframe(mi_df)
    assert is_nested_dataframe(nested)
    assert is_nested_dataframe(nested_heterogenous)


@pytest.mark.parametrize("n_instances", N_INSTANCES)
@pytest.mark.parametrize("n_columns", N_COLUMNS)
@pytest.mark.parametrize("n_timepoints", N_TIMEPOINTS)
def test_nested_dataframe_equal(n_instances, n_columns, n_timepoints):
    nested, _ = make_classification_problem(n_instances, n_columns, n_timepoints)
    nested_copy = nested.copy()

    # Replace nested value with primitive
    nested_unequal1 = nested.copy()
    nested_unequal1.iat[0, 0] = 17.0

    # Replace nested value with random nested array values
    nested_unequal2 = nested.copy()
    nested_unequal2.iloc[0, 0] = np.random.normal(size=(23))

    # Replace nested value with random nested pd.Series
    nested_unequal3 = nested.copy()
    nested_unequal3.iat[0, 0] = pd.Series(np.random.normal(size=(23)))

    assert nested_dataframes_equal(nested, nested_copy)
    assert not nested_dataframes_equal(nested, nested_unequal1)
    assert not nested_dataframes_equal(nested, nested_unequal2)
    assert not nested_dataframes_equal(nested, nested_unequal3)
