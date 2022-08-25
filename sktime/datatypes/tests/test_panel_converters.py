# -*- coding: utf-8 -*-
"""Testing panel converters - internal functions and more extensive fixtures."""
import numpy as np
import pandas as pd
import pytest

from sktime.datasets import generate_example_long_table, make_multi_index_dataframe
from sktime.datatypes._adapter import convert_from_multiindex_to_listdataset
from sktime.datatypes._panel._check import are_columns_nested, is_nested_dataframe
from sktime.datatypes._panel._convert import (
    from_2d_array_to_nested,
    from_3d_numpy_to_2d_array,
    from_3d_numpy_to_multi_index,
    from_3d_numpy_to_nested,
    from_long_to_nested,
    from_multi_index_to_3d_numpy,
    from_multi_index_to_nested,
    from_nested_to_2d_array,
    from_nested_to_3d_numpy,
    from_nested_to_long,
    from_nested_to_multi_index,
)
from sktime.utils._testing.panel import make_classification_problem
from sktime.utils.validation._dependencies import _check_soft_dependencies

N_INSTANCES = [10, 15]
N_COLUMNS = [3, 5]
N_TIMEPOINTS = [3, 5]
N_CLASSES = [2, 5]


@pytest.mark.parametrize("n_instances", N_INSTANCES)
@pytest.mark.parametrize("n_columns", N_COLUMNS)
@pytest.mark.parametrize("n_timepoints", N_TIMEPOINTS)
def test_are_columns_nested(n_instances, n_columns, n_timepoints):
    """Test are_columns_nested for correctness."""
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
    """Test from_nested_to_3d_numpy for correctness."""
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
    """Test from_3d_numpy_to_nested for correctness."""
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
    """Test from_nested_to_2d_array for correctness."""
    nested, _ = make_classification_problem(n_instances, n_columns, n_timepoints)

    array = from_nested_to_2d_array(nested)
    assert array.shape == (n_instances, n_columns * n_timepoints)
    assert array.index.equals(nested.index)


@pytest.mark.parametrize("n_instances", N_INSTANCES)
@pytest.mark.parametrize("n_columns", N_COLUMNS)
@pytest.mark.parametrize("n_timepoints", N_TIMEPOINTS)
def test_from_3d_numpy_to_2d_array(n_instances, n_columns, n_timepoints):
    """Test from_3d_numpy_to_2d_array for correctness."""
    array = np.random.normal(size=(n_instances, n_columns, n_timepoints))
    array_2d = from_3d_numpy_to_2d_array(array)

    assert array_2d.shape == (n_instances, n_columns * n_timepoints)


@pytest.mark.parametrize("n_instances", N_INSTANCES)
@pytest.mark.parametrize("n_columns", N_COLUMNS)
@pytest.mark.parametrize("n_timepoints", N_TIMEPOINTS)
def test_from_multi_index_to_3d_numpy(n_instances, n_columns, n_timepoints):
    """Test from_multi_index_to_3d_numpy for correctness."""
    mi_df = make_multi_index_dataframe(
        n_instances=n_instances, n_timepoints=n_timepoints, n_columns=n_columns
    )

    array = from_multi_index_to_3d_numpy(mi_df)

    assert isinstance(array, np.ndarray)
    assert array.shape == (n_instances, n_columns, n_timepoints)


@pytest.mark.parametrize("n_instances", N_INSTANCES)
@pytest.mark.parametrize("n_columns", N_COLUMNS)
@pytest.mark.parametrize("n_timepoints", N_TIMEPOINTS)
def test_from_3d_numpy_to_multi_index(n_instances, n_columns, n_timepoints):
    """Test from_3d_numpy_to_multi_index for correctness."""
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
    """Test from_multi_index_to_nested for correctness."""
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
    """Test from_nested_to_multi_index for correctness."""
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
    """Test is_nested_dataframe for correctness."""
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
def test_from_2d_array_to_nested(n_instances, n_columns, n_timepoints):
    """Test from_2d_array_to_nested for correctness."""
    rng = np.random.default_rng()
    X_2d = rng.standard_normal((n_instances, n_timepoints))
    nested_df = from_2d_array_to_nested(X_2d)

    assert is_nested_dataframe(nested_df)
    assert nested_df.shape == (n_instances, 1)


@pytest.mark.parametrize("n_instances", N_INSTANCES)
@pytest.mark.parametrize("n_columns", N_COLUMNS)
@pytest.mark.parametrize("n_timepoints", N_TIMEPOINTS)
def test_from_long_to_nested(n_instances, n_columns, n_timepoints):
    """Test from_long_to_nested for correctness."""
    X_long = generate_example_long_table(
        num_cases=n_instances, series_len=n_timepoints, num_dims=n_columns
    )
    nested_df = from_long_to_nested(X_long)

    assert is_nested_dataframe(nested_df)
    assert nested_df.shape == (n_instances, n_columns)


@pytest.mark.parametrize("n_instances", N_INSTANCES)
@pytest.mark.parametrize("n_columns", N_COLUMNS)
@pytest.mark.parametrize("n_timepoints", N_TIMEPOINTS)
def test_from_nested_to_long(n_instances, n_columns, n_timepoints):
    """Test from_nested_to_long for correctness."""
    nested, _ = make_classification_problem(n_instances, n_columns, n_timepoints)
    X_long = from_nested_to_long(
        nested,
        instance_column_name="case_id",
        time_column_name="reading_id",
        dimension_column_name="dim_id",
    )

    assert isinstance(X_long, pd.DataFrame)
    assert X_long.shape == (n_instances * n_timepoints * n_columns, 4)
    assert (X_long.columns == ["case_id", "reading_id", "dim_id", "value"]).all()


@pytest.mark.skipif(
    not _check_soft_dependencies("gluonts", severity="none"),
    reason="requires gluonts package in the example",
)
@pytest.mark.parametrize("n_instances", N_INSTANCES)
@pytest.mark.parametrize("n_columns", N_COLUMNS)
@pytest.mark.parametrize("n_timepoints", N_TIMEPOINTS)
def test_from_multiindex_to_listdataset(n_instances, n_columns, n_timepoints):
    """Test from multiindex DF to listdataset for gluonts."""
    import numpy as np
    import pandas as pd

    from sktime.datatypes import convert_to

    # from sktime.datatypes._adapters import convert_from_multiindex_to_listdataset

    def random_datetimes_or_dates(
        start, end, out_format="datetime", n=10, random_seed=42
    ):
        """Generate random pd Datetime in the start to end range.

        unix timestamp is in ns by default.
        Divide the unix time value by 10**9 to make it seconds
        (or 24*60*60*10**9 to make it days).
        The corresponding unit variable is passed to the pd.to_datetime function.
        Values for the (divide_by, unit) pair to select is defined by the out_format
        parameter.
        for 1 -> out_format='datetime'
        for 2 -> out_format=anything else.
        """
        np.random.seed(random_seed)
        (divide_by, unit) = (
            (10**9, "s")
            if out_format == "datetime"
            else (24 * 60 * 60 * 10**9, "D")
        )

        start_u = start.value // divide_by
        end_u = end.value // divide_by

        return pd.to_datetime(np.random.randint(start_u, end_u, n), unit=unit)

    def _make_example_multiindex(
        n_instances, n_columns, n_timepoints, random_seed=42
    ) -> pd.DataFrame:

        import numpy as np

        start = pd.to_datetime("1750-01-01")
        end = pd.to_datetime("2022-07-01")
        inputDF = np.random.randint(1, 99, size=(n_instances * n_timepoints, n_columns))
        n_instances = n_instances
        column_name = []
        for i in range(n_columns):
            column_name.append("dim_" + str(i))

        random_start_date = random_datetimes_or_dates(
            start, end, out_format="out datetime", n=n_instances, random_seed=42
        )

        level0_idx = [
            list(np.full(n_timepoints, instance)) for instance in range(n_instances)
        ]
        level0_idx = np.ravel(level0_idx)

        level1_idx = [
            list(
                pd.date_range(
                    random_start_date[instance], periods=n_timepoints, freq="H"
                )
            )
            for instance in range(n_instances)
        ]
        level1_idx = np.ravel(level1_idx)

        multi_idx = pd.MultiIndex.from_arrays(
            [level0_idx, level1_idx], names=("instance", "datetime")
        )

        inputDF_return = pd.DataFrame(inputDF, columns=column_name, index=multi_idx)

        return inputDF_return

    MULTIINDEX_DF = _make_example_multiindex(n_instances, n_columns, n_timepoints)
    # Result from the converter
    listdataset_result = convert_from_multiindex_to_listdataset(MULTIINDEX_DF)
    listdataset_result_list = list(listdataset_result)
    # Result from raw data
    dimension_name = MULTIINDEX_DF.columns
    # Convert MULTIINDEX_DF to nested_univ format to compare with listdataset
    control_result = convert_to(MULTIINDEX_DF, to_type="nested_univ")
    control_result = control_result.reset_index()
    control_result = control_result[dimension_name]

    # Perform the test
    for instance, _dim_name in control_result.iterrows():
        for dim_no, dim in enumerate(dimension_name):
            np.testing.assert_array_equal(
                control_result.loc[instance, dim].to_numpy(),
                listdataset_result_list[instance]["target"][dim_no],
            )
