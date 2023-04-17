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
    """Test that random_partition returns a disjoint partition."""
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

    if index_type in ["int", "range"]:
        assert len(df_pivot) == ceil(len(df) / sp)
    else:
        assert len(df_pivot) == ceil((len(df) + 1) / sp)

    # compare values in pivot and plain frame,
    # check these are the same if read in left-right-then-top-down order
    pivot_values = df_pivot.values.flatten()
    pivot_values = pivot_values[~np.isnan(pivot_values)]
    df_values = df.values.flatten()
    assert(np.all(df_values == pivot_values))


@pytest.mark.parametrize("n_timepoints", [50, 1])
@pytest.mark.parametrize("index_type", ["period", "datetime", "range", "int"])
@pytest.mark.parametrize("sp", [2, 10])
def test_unpivot_sp(sp, index_type, n_timepoints):
    """Test that random_partition returns a disjoint partition."""
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
