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


# --- New multivariate tests (hand-computed) for PR #10313 ---


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.utils"]),
    reason="Run if utils module has changed.",
)
def test_pivot_sp_univariate_backward_compat():
    """Test _pivot_sp: univariate output still has flat (non-MultiIndex) columns.

    Regression guard: the multivariate fix must not break the univariate API.
    Columns must remain a flat integer index [0, 1, 2], not a MultiIndex.

    Hand-computed:
      Input: values [10, 20, 30, 40, 50, 60], sp=3, RangeIndex(6)
      Epochs:  row 0 = [10, 20, 30]
               row 1 = [40, 50, 60]
      Expected columns: [0, 1, 2]  (flat, no variable-name level)
    """
    df = pd.DataFrame({"A": [10, 20, 30, 40, 50, 60]}, index=range(6))
    result = _pivot_sp(df, sp=3)

    # columns must be flat — NOT a MultiIndex
    assert not isinstance(result.columns, pd.MultiIndex), (
        "Univariate _pivot_sp should return flat column index, got MultiIndex"
    )
    assert result.columns.tolist() == [0, 1, 2]

    # values must match hand-computed layout
    expected = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
    np.testing.assert_array_equal(result.values, expected)


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.utils"]),
    reason="Run if utils module has changed.",
)
def test_pivot_sp_multivariate_correctness():
    """Test _pivot_sp: multivariate input produces correct MultiIndex columns.

    Hand-computed (sp=3, 2 variables A and B, 6 timepoints):

      Input df:
        index | A  | B
            0 | 10 | 1
            1 | 20 | 2
            2 | 30 | 3
            3 | 40 | 4
            4 | 50 | 5
            5 | 60 | 6

      Epoch 0 (indices 0,1,2): A->[10,20,30], B->[1,2,3]
      Epoch 1 (indices 3,4,5): A->[40,50,60], B->[4,5,6]

      Expected columns (MultiIndex):
        ('A',0), ('A',1), ('A',2), ('B',0), ('B',1), ('B',2)
      Expected row 0: [10, 20, 30, 1, 2, 3]
      Expected row 1: [40, 50, 60, 4, 5, 6]

    Key property: no column collision between A offsets and B offsets.
    """
    df = pd.DataFrame(
        {"A": [10, 20, 30, 40, 50, 60], "B": [1, 2, 3, 4, 5, 6]},
        index=range(6),
    )
    result = _pivot_sp(df, sp=3)

    # columns must be a MultiIndex (variable x offset)
    assert isinstance(result.columns, pd.MultiIndex), (
        "Multivariate _pivot_sp should return MultiIndex columns"
    )

    # hand-computed values per variable and offset
    np.testing.assert_array_equal(result[("A", 0)].values, [10, 40])
    np.testing.assert_array_equal(result[("A", 1)].values, [20, 50])
    np.testing.assert_array_equal(result[("A", 2)].values, [30, 60])
    np.testing.assert_array_equal(result[("B", 0)].values, [1, 4])
    np.testing.assert_array_equal(result[("B", 1)].values, [2, 5])
    np.testing.assert_array_equal(result[("B", 2)].values, [3, 6])

    # shape: 2 epochs x (2 vars * 3 offsets) = 2 x 6
    assert result.shape == (2, 6)


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.utils"]),
    reason="Run if utils module has changed.",
)
def test_unpivot_sp_multivariate_correctness():
    """Test _unpivot_sp: correctly inverts _pivot_sp for multivariate input.

    Hand-computed (sp=3, 2 variables A and B, 6 timepoints):

      After _pivot_sp, result has MultiIndex columns (A,0..2) and (B,0..2).
      After _unpivot_sp with the original df as template, must recover exactly:

        index | A  | B
            0 | 10 | 1
            1 | 20 | 2
            2 | 30 | 3
            3 | 40 | 4
            4 | 50 | 5
            5 | 60 | 6
    """
    df = pd.DataFrame(
        {"A": [10, 20, 30, 40, 50, 60], "B": [1, 2, 3, 4, 5, 6]},
        index=range(6),
    )
    pivoted = _pivot_sp(df, sp=3)
    recovered = _unpivot_sp(df=pivoted, template=df)

    # must recover original shape
    assert recovered.shape == df.shape

    # must recover original column names
    assert recovered.columns.tolist() == ["A", "B"]

    # must recover original index
    np.testing.assert_array_equal(recovered.index, df.index)

    # must recover original values (hand-computed)
    np.testing.assert_array_equal(recovered[["A", "B"]].values, df.values)


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.utils"]),
    reason="Run if utils module has changed.",
)
@pytest.mark.parametrize("sp", [2, 4])
def test_pivot_sp_multivariate_no_collision(sp):
    """Test _pivot_sp: no column name collision for multivariate input.

    For any sp, variables A and B must occupy distinct columns.
    Previously (before the fix), droplevel(0) would produce duplicate
    column names like [0, 1, 0, 1] for sp=2 with two variables.

    Hand-computed (sp=2, n_timepoints=4):
      A offsets: ('A',0), ('A',1)
      B offsets: ('B',0), ('B',1)
      Total 4 distinct columns — no collision.
    """
    df = pd.DataFrame(
        {"A": [1.0, 2.0, 3.0, 4.0], "B": [5.0, 6.0, 7.0, 8.0]},
        index=range(4),
    )
    result = _pivot_sp(df, sp=sp)

    # All column names must be unique (no collision)
    assert len(result.columns) == len(set(result.columns)), (
        f"Column collision detected in _pivot_sp with sp={sp} and 2 variables"
    )

    # Must be a MultiIndex for multivariate input
    assert isinstance(result.columns, pd.MultiIndex)
