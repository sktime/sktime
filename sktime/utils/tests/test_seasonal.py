# -*- coding: utf-8 -*-
"""Testing seasonal utilities."""

from math import ceil

import numpy as np
import pandas as pd
import pytest

from sktime.utils._testing.series import _make_series
from sktime.utils.seasonality import _pivot_sp, _unpivot_sp


@pytest.mark.parametrize("n_timepoints", [50, 1])
@pytest.mark.parametrize("index_type", ["period", "datetime", "range", "int"])
@pytest.mark.parametrize("sp", [2, 10])
def test_pivot_sp(sp, index_type, n_timepoints):
    """Test _pivot_sp contract."""
    df = _make_series(
        n_timepoints=n_timepoints,
        n_columns=1,
        index_type=index_type,
        return_mtype="pd.DataFrame",
    )

    df_pivot = _pivot_sp(df, sp)

    if index_type != "range":
        assert isinstance(df_pivot.index, type(df.index))
    else:
        pd.api.types.is_integer_dtype(df_pivot.index)

    assert len(df_pivot.columns) == min(sp, len(df))

    assert len(df_pivot) == ceil(len(df) / sp)

    # compare values in pivot and plain frame,
    # check these are the same if read in left-right-then-top-down order
    pivot_values = df_pivot.values.flatten()
    pivot_values = pivot_values[~np.isnan(pivot_values)]
    df_values = df.values.flatten()
    assert np.all(df_values == pivot_values)


@pytest.mark.parametrize("n_timepoints", [50, 1])
@pytest.mark.parametrize("index_type", ["period", "datetime", "range", "int"])
@pytest.mark.parametrize("sp", [2, 10])
def test_unpivot_sp(sp, index_type, n_timepoints):
    """Test _unpivot_sp contract."""
    df = _make_series(
        n_timepoints=n_timepoints,
        n_columns=1,
        index_type=index_type,
        return_mtype="pd.DataFrame",
    )
    df.columns = ["foo"]

    df_pivot = _pivot_sp(df, sp)

    df_unpivot = _unpivot_sp(df=df_pivot, template=df)

    if index_type != "range":
        assert isinstance(df_pivot.index, type(df.index))
    else:
        pd.api.types.is_integer_dtype(df_pivot.index)

    assert len(df_unpivot) == len(df)
    assert np.all(df_unpivot.index == df.index)
    assert np.all(df_unpivot == df)


@pytest.mark.parametrize("n_timepoints", [50, 2])
@pytest.mark.parametrize("index_type", ["period", "datetime", "range", "int"])
@pytest.mark.parametrize("sp", [3, 10])
def test_pivot_sp_consistent(sp, index_type, n_timepoints):
    """Test _pivot_sp consistency between offsets."""
    df = _make_series(
        n_timepoints=n_timepoints,
        n_columns=1,
        index_type=index_type,
        return_mtype="pd.DataFrame",
    )
    df2 = df.iloc[1:]

    df_pivot = _pivot_sp(df, sp)
    df2_pivot = _pivot_sp(df2, sp, anchor=df)

    assert np.all(df_pivot.index == df2_pivot.index)

    df_pivot_values = df_pivot.values.flatten()
    df_pivot_values = df_pivot_values[~np.isnan(df_pivot_values)]
    df2_pivot_values = df2_pivot.values.flatten()
    df2_pivot_values = df2_pivot_values[~np.isnan(df2_pivot_values)]
    assert np.all(df_pivot_values[1:] == df2_pivot_values)
