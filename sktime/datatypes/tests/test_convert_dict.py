"""Tests for the converter registry, get_convert_dict and BaseConverter entries."""

__author__ = ["PragnyaKhandelwal"]

import pytest

from sktime.datatypes._base import BaseConverter
from sktime.datatypes._convert import get_convert_dict
from sktime.tests.test_switch import run_test_module_changed

# scitypes whose conversions are fully migrated to BaseConverter classes
MIGRATED_SCITYPES = ["Hierarchical", "Proba"]


@pytest.mark.skipif(
    not run_test_module_changed("sktime.datatypes"),
    reason="Test only if sktime.datatypes or utils.parallel has been changed",
)
def test_get_convert_dict_returns_copy():
    """Test that get_convert_dict returns a copy, not the cached dict."""
    convert_dict = get_convert_dict()
    convert_dict.clear()
    assert len(get_convert_dict()) > 0


@pytest.mark.skipif(
    not run_test_module_changed("sktime.datatypes"),
    reason="Test only if sktime.datatypes or utils.parallel has been changed",
)
@pytest.mark.parametrize("soft_deps", ["present", "all"])
def test_convert_dict_keys_and_values(soft_deps):
    """Test that keys are (mtype_from, mtype_to, scitype) and values callable."""
    convert_dict = get_convert_dict(soft_deps=soft_deps)
    for key, converter in convert_dict.items():
        assert isinstance(key, tuple) and len(key) == 3
        assert callable(converter)


@pytest.mark.skipif(
    not run_test_module_changed("sktime.datatypes"),
    reason="Test only if sktime.datatypes or utils.parallel has been changed",
)
@pytest.mark.parametrize("scitype", MIGRATED_SCITYPES)
def test_migrated_conversions_are_baseconverters(scitype):
    """Test that migrated conversions are BaseConverter matching their key."""
    convert_dict = get_convert_dict(soft_deps="all")
    keys = [k for k in convert_dict if k[2] == scitype]
    assert len(keys) > 0

    for key in keys:
        converter = convert_dict[key]
        assert isinstance(converter, BaseConverter)
        assert converter._get_key() == key
        mtype_from, mtype_to, _ = key
        assert converter.get_tag("mtype_from") == mtype_from
        assert converter.get_tag("mtype_to") == mtype_to


@pytest.mark.skipif(
    not run_test_module_changed("sktime.datatypes"),
    reason="Test only if sktime.datatypes or utils.parallel has been changed",
)
def test_soft_dep_conversions_included_with_all():
    """Test that soft dependency conversions are included with soft_deps="all"."""
    convert_dict = get_convert_dict(soft_deps="all")
    expected = [
        ("dask_hierarchical", "pd_multiindex_hier", "Hierarchical"),
        ("pd_multiindex_hier", "dask_hierarchical", "Hierarchical"),
        ("polars_hierarchical", "pd_multiindex_hier", "Hierarchical"),
        ("pd_multiindex_hier", "polars_hierarchical", "Hierarchical"),
        ("dask_hierarchical", "polars_hierarchical", "Hierarchical"),
        ("polars_hierarchical", "dask_hierarchical", "Hierarchical"),
    ]
    for key in expected:
        assert key in convert_dict
