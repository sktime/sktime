"""Testing seasonal utilities."""

from math import ceil

import numpy as np
import pandas as pd
import pytest

from sktime.tests.test_switch import run_test_module_changed
from sktime.utils._testing.series import _make_series
from sktime.utils.seasonality import _pivot_sp, _unpivot_sp


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.utils"]),
    reason="Run if utils module has changed.",
)
@pytest.mark.parametrize("n_timepoints", [49, 1])
@pytest.mark.parametrize("index_type", ["period", "datetime", "range", "int"])
@pytest.mark.parametrize("sp", [2, 10])
@pytest.mark.parametrize("anchor_side", ["start", "end"])
def test_pivot_sp(sp, index_type, n_timepoints, anchor_side):
    """Test _pivot_sp contract."""
    df = _make_series(
        n_timepoints=n_timepoints,
        n_columns=1,
        index_type=index_type,
        return_mtype="pd.DataFrame",
    )

    df_pivot = _pivot_sp(df, sp, anchor_side=anchor_side)

    if index_type != "range":
        assert isinstance(df_pivot.index, type(df.index))
    else:
        pd.api.types.is_integer_dtype(df_pivot.index)

    # ensure expected size of the resulting DataFrame
    assert len(df_pivot.columns) == min(sp, len(df))
    assert len(df_pivot) == ceil(len(df) / sp)

    # compare values in pivot and plain frame,
    # check these are the same if read in left-right-then-top-down order
    pivot_values = df_pivot.values.flatten()
    pivot_values = pivot_values[~np.isnan(pivot_values)]
    df_values = df.values.flatten()
    assert np.all(df_values == pivot_values)

    # if anchor_side is "start", top left should be non-nan and bottom right nan
    # if anchor_side is "end", top left should be nan and bottom right non-nan
    # only if there is more than one timepoint, otherwise the result will be 1-element
    if n_timepoints > 1:
        if anchor_side == "start":
            assert not np.isnan(df_pivot.iloc[0, 0])
            assert np.isnan(df_pivot.iloc[-1, -1])
        else:
            assert np.isnan(df_pivot.iloc[0, 0])
            assert not np.isnan(df_pivot.iloc[-1, -1])


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.utils"]),
    reason="Run if utils module has changed.",
)
@pytest.mark.parametrize("n_timepoints", [49, 1])
@pytest.mark.parametrize("index_type", ["period", "datetime", "range", "int"])
@pytest.mark.parametrize("sp", [2, 10])
@pytest.mark.parametrize("anchor_side", ["start", "end"])
def test_unpivot_sp(sp, index_type, n_timepoints, anchor_side):
    """Test _unpivot_sp contract."""
    df = _make_series(
        n_timepoints=n_timepoints,
        n_columns=1,
        index_type=index_type,
        return_mtype="pd.DataFrame",
    )
    df.columns = ["foo"]

    df_pivot = _pivot_sp(df, sp, anchor_side=anchor_side)

    df_unpivot = _unpivot_sp(df=df_pivot, template=df)

    if index_type != "range":
        assert isinstance(df_pivot.index, type(df.index))
    else:
        pd.api.types.is_integer_dtype(df_pivot.index)

    assert len(df_unpivot) == len(df)
    assert np.all(df_unpivot.index == df.index)
    assert np.all(df_unpivot == df)


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.utils"]),
    reason="Run if utils module has changed.",
)
@pytest.mark.parametrize("n_timepoints", [50, 2])
@pytest.mark.parametrize("index_type", ["period", "datetime", "range", "int"])
@pytest.mark.parametrize("sp", [3, 10])
@pytest.mark.parametrize("anchor_side", ["start", "end"])
def test_pivot_sp_consistent(sp, index_type, n_timepoints, anchor_side):
    """Test _pivot_sp consistency between offsets."""
    df = _make_series(
        n_timepoints=n_timepoints,
        n_columns=1,
        index_type=index_type,
        return_mtype="pd.DataFrame",
    )
    df2 = df.iloc[1:]

    df_pivot = _pivot_sp(df, sp, anchor_side=anchor_side)
    df2_pivot = _pivot_sp(df2, sp, anchor=df, anchor_side=anchor_side)

    assert np.all(df_pivot.index == df2_pivot.index)

    df_pivot_values = df_pivot.values.flatten()
    df_pivot_values = df_pivot_values[~np.isnan(df_pivot_values)]
    df2_pivot_values = df2_pivot.values.flatten()
    df2_pivot_values = df2_pivot_values[~np.isnan(df2_pivot_values)]
    assert np.all(df_pivot_values[1:] == df2_pivot_values)
