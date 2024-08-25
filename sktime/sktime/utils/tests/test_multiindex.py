"""Testing multiindex utilities."""

import pandas as pd
import pytest

from sktime.tests.test_switch import run_test_module_changed
from sktime.utils.multiindex import flatten_multiindex, rename_multiindex


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
