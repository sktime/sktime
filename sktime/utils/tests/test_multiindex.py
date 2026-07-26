"""Testing multiindex utilities."""

import numpy as np
import pandas as pd
import pytest

from sktime.tests.test_switch import run_test_module_changed
from sktime.utils.multiindex import (
    apply_split,
    flatten_multiindex,
    is_hierarchical,
    rename_multiindex,
)


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.utils.multiindex"]),
    reason="Run if multiindex module has changed.",
)
def test_flatten_multiindex():
    """Test flatten_multiindex contract."""
    mi = pd.MultiIndex.from_product([["a", "b"], [0, 42]])

    flat = flatten_multiindex(mi)

    expected = pd.Index(["a__0", "a__42", "b__0", "b__42"])

    assert isinstance(flat, pd.Index)
    assert (expected == flat).all()


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.utils.multiindex"]),
    reason="Run if multiindex module has changed.",
)
def test_rename_multiindex():
    """Test rename_multiindex contract."""
    mi = pd.MultiIndex.from_tuples([("a", 1), ("a", 42), ("b", 1), ("c", 0)])

    idx_name = "foobar"

    flat = rename_multiindex(mi, feature_names_out="flat", idx_name=idx_name)
    expected_flat = pd.Index(["a__1", "a__42", "b__1", "c__0"])

    assert isinstance(flat, pd.Index)
    assert (expected_flat == flat).all()

    multi = rename_multiindex(mi, feature_names_out="multiindex", idx_name=idx_name)
    expected_multi = mi

    assert isinstance(multi, pd.MultiIndex)
    assert (expected_multi == multi).all()

    auto = rename_multiindex(mi, feature_names_out="auto", idx_name=idx_name)
    expected_auto = pd.Index(["a__1", "42", "b__1", "0"])

    assert isinstance(auto, pd.Index)
    assert (expected_auto == auto).all()

    with pytest.raises(ValueError, match=idx_name):
        rename_multiindex(mi, feature_names_out="original", idx_name=idx_name)

    mi2 = pd.MultiIndex.from_tuples([("a", 1), ("a", 42), ("b", 2), ("c", 0)])

    flat = rename_multiindex(mi2, feature_names_out="flat", idx_name=idx_name)
    expected_flat = pd.Index(["a__1", "a__42", "b__2", "c__0"])

    assert isinstance(flat, pd.Index)
    assert (expected_flat == flat).all()

    multi = rename_multiindex(mi2, feature_names_out="multiindex", idx_name=idx_name)
    expected_multi = mi2

    assert isinstance(multi, pd.MultiIndex)
    assert (expected_multi == multi).all()

    auto = rename_multiindex(mi2, feature_names_out="auto", idx_name=idx_name)
    expected_auto = pd.Index(["1", "42", "2", "0"])

    assert isinstance(auto, pd.Index)
    assert (expected_auto == auto).all()

    orig = rename_multiindex(mi2, feature_names_out="auto", idx_name=idx_name)
    expected_orig = pd.Index(["1", "42", "2", "0"])

    assert isinstance(orig, pd.Index)
    assert (expected_orig == orig).all()


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.utils.multiindex"]),
    reason="Run if multiindex module has changed.",
)
def test_apply_split():
    """Test apply_split() and check shapes"""
    mi = pd.MultiIndex.from_product([["a", "b"], [0, 42]])

    iloc_ix = [1]

    result = apply_split(mi, iloc_ix)

    expected = np.array([2, 3])

    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, expected)

    # another test case from the docstring
    y = pd.MultiIndex.from_tuples([(0, 0), (0, 1), (1, 0), (1, 1)])
    iloc_ix = np.array([1, 0])
    result2 = apply_split(y, iloc_ix)
    expected2 = np.array([2, 3, 0, 1])

    assert isinstance(result2, np.ndarray)
    assert np.array_equal(result2, expected2)


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.utils.multiindex"]),
    reason="Run if multiindex module has changed.",
)
def test_is_hierarchical():
    """Test is_hierarchical()"""
    data_hier = {
        ("Z1", "R1", "A"): 1,
        ("Z1", "R1", "B"): 2,
        ("Z1", "R1", "C"): 3,
        ("Z1", "R2", "D"): 4,
        ("Z1", "R2", "E"): 5,
        ("Z1", "R2", "F"): 6,
        ("Z2", "R3", "G"): 7,
        ("Z2", "R3", "H"): 8,
        ("Z2", "R3", "I"): 9,
        ("Z2", "R4", "J"): 10,
        ("Z2", "R4", "K"): 11,
        ("Z2", "R4", "L"): 12,
    }

    data_not_hier = {
        ("Z1", "R1", "A"): 1,
        ("Z1", "R1", "B"): 2,
        ("Z1", "R1", "C"): 3,
        ("Z1", "R2", "D"): 4,
        ("Z1", "R2", "E"): 5,
        ("Z1", "R2", "F"): 6,
        ("Z2", "R3", "G"): 7,
        ("Z2", "R1", "H"): 8,  # <- breaks the hierarchy (R1 under both Z1 and Z2)
        ("Z2", "R3", "I"): 9,
        ("Z2", "R4", "J"): 10,
        ("Z2", "R4", "K"): 11,
        ("Z2", "R4", "L"): 12,
    }

    # Create MultiIndex DataFrames
    df_hier = pd.DataFrame.from_dict(data_hier, orient="index", columns=["Value"])
    df_hier.index = pd.MultiIndex.from_tuples(df_hier.index)

    df_not_hier = pd.DataFrame.from_dict(
        data_not_hier, orient="index", columns=["Value"]
    )
    df_not_hier.index = pd.MultiIndex.from_tuples(df_not_hier.index)

    result_true = is_hierarchical(df_hier.index)
    result_false = is_hierarchical(df_not_hier.index)

    assert result_true
    assert not result_false
