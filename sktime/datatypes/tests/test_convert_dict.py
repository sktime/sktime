"""Tests for the converter registry, get_convert_dict and BaseConverter entries."""

__author__ = ["PragnyaKhandelwal"]

import pytest

from sktime.datatypes._base import BaseConverter
from sktime.datatypes._convert import get_convert_dict
from sktime.tests.test_switch import run_test_module_changed

SCITYPES = ["Series", "Panel", "Hierarchical", "Table", "Proba"]

# mtypes without a datatype class, conversions from/to these stay function based
MTYPES_WITHOUT_CLASS = ["pd-long", "pd-wide", "pl.LazyFrame"]


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
@pytest.mark.parametrize("scitype", SCITYPES)
def test_conversions_are_baseconverters(scitype):
    """Test that conversions are BaseConverter matching their key.

    Exceptions are conversions obtained by chaining via ``_extend_conversions``,
    and conversions from or to mtypes without a datatype class.
    """
    convert_dict = get_convert_dict(soft_deps="all")
    keys = [k for k in convert_dict if k[2] == scitype]
    assert len(keys) > 0

    n_converters = 0
    for key in keys:
        converter = convert_dict[key]
        mtype_from, mtype_to, _ = key
        if isinstance(converter, BaseConverter):
            n_converters += 1
            assert converter._get_key() == key
            assert converter.get_tag("mtype_from") == mtype_from
            assert converter.get_tag("mtype_to") == mtype_to
        else:
            no_class = mtype_from in MTYPES_WITHOUT_CLASS
            no_class = no_class or mtype_to in MTYPES_WITHOUT_CLASS
            chained = converter.__name__ == "concat_fun"
            assert no_class or chained, f"{key} is not a BaseConverter"

    assert n_converters > 0


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
