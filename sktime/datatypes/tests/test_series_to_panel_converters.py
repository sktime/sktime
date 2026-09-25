"""Tests for cross-scitype conversions between Series, Panel, and Hierarchical."""

import pandas as pd
import pytest

from sktime.datatypes import check_is_mtype
from sktime.datatypes._series_as_panel._convert import (
    convert_Hierarchical_to_Panel,
    convert_Hierarchical_to_Series,
    convert_Panel_to_Hierarchical,
    convert_Panel_to_Series,
    convert_Series_to_Hierarchical,
    convert_Series_to_Panel,
)
from sktime.utils.dependencies import _check_soft_dependencies


def test_convert_series_panel_unsupported_type_raises():
    """Test TypeError on unsupported types."""
    with pytest.raises(TypeError, match="supported Series mtype"):
        convert_Series_to_Panel("invalid_input")

    with pytest.raises(TypeError, match="supported Panel mtype"):
        convert_Panel_to_Series("invalid_input")


def test_convert_panel_to_series_pandas_immutability():
    """Test that input pandas multiindex DataFrame is not mutated in-place."""
    idx = pd.MultiIndex.from_tuples([(0, 0), (0, 1)], names=["instances", "timepoints"])
    df_panel = pd.DataFrame({"a": [1, 2]}, index=idx)
    original_nlevels = df_panel.index.nlevels

    res = convert_Panel_to_Series(df_panel)
    assert df_panel.index.nlevels == original_nlevels
    assert res.index.nlevels == 1


@pytest.mark.skipif(
    not _check_soft_dependencies("polars", severity="none"),
    reason="skip test if polars is not installed",
)
def test_convert_polars_series_to_panel_roundtrip():
    """Test Polars Series to Panel and back, with and without pre-existing index."""
    import polars as pl

    # Case 1: Index-less Polars DataFrame
    df_no_index = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    panel, mtype = convert_Series_to_Panel(df_no_index, return_to_mtype=True)
    assert mtype == "polars_panel"
    assert check_is_mtype(panel, "polars_panel", return_metadata=False)

    recovered, s_mtype = convert_Panel_to_Series(panel, return_to_mtype=True)
    assert s_mtype == "pl.DataFrame"
    assert check_is_mtype(recovered, "pl.DataFrame", return_metadata=False)
    assert recovered["a"].to_list() == df_no_index["a"].to_list()

    # Case 2: Polars DataFrame with existing __index__time
    df_with_index = pl.DataFrame(
        {
            "__index__time": [10, 20, 30],
            "a": [1, 2, 3],
            "b": [4, 5, 6],
        }
    )
    panel2, mtype2 = convert_Series_to_Panel(df_with_index, return_to_mtype=True)
    assert mtype2 == "polars_panel"
    assert check_is_mtype(panel2, "polars_panel", return_metadata=False)

    recovered2, s_mtype2 = convert_Panel_to_Series(panel2, return_to_mtype=True)
    assert s_mtype2 == "pl.DataFrame"
    assert check_is_mtype(recovered2, "pl.DataFrame", return_metadata=False)
    assert "__index__time" in recovered2.columns
    assert "__index__instances" not in recovered2.columns


@pytest.mark.skipif(
    not _check_soft_dependencies("polars", severity="none"),
    reason="skip test if polars is not installed",
)
def test_convert_polars_hierarchical_roundtrips():
    """Test Polars Hierarchical conversions roundtrips."""
    import polars as pl

    df_series = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    # Series to Hierarchical and back
    hier, h_mtype = convert_Series_to_Hierarchical(df_series, return_to_mtype=True)
    assert h_mtype == "polars_hierarchical"
    assert check_is_mtype(hier, "polars_hierarchical", return_metadata=False)

    rec_series, s_mtype = convert_Hierarchical_to_Series(hier, return_to_mtype=True)
    assert s_mtype == "pl.DataFrame"
    assert check_is_mtype(rec_series, "pl.DataFrame", return_metadata=False)
    assert rec_series["a"].to_list() == df_series["a"].to_list()

    # Panel to Hierarchical and back
    panel, _ = convert_Series_to_Panel(df_series, return_to_mtype=True)
    hier_from_panel, hp_mtype = convert_Panel_to_Hierarchical(
        panel, return_to_mtype=True
    )
    assert hp_mtype == "polars_hierarchical"
    assert check_is_mtype(hier_from_panel, "polars_hierarchical", return_metadata=False)

    rec_panel, p_mtype = convert_Hierarchical_to_Panel(
        hier_from_panel, return_to_mtype=True
    )
    assert p_mtype == "polars_panel"
    assert check_is_mtype(rec_panel, "polars_panel", return_metadata=False)
