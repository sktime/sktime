# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for chaining of mtype conversions."""

__author__ = ["adity1raut"]

import numpy as np
import pandas as pd
import pytest

from sktime.datatypes._base import BaseConverter, ConverterChain
from sktime.datatypes._base._chain import CHAIN_STORE_KEY
from sktime.datatypes._convert import convert, convert_dict
from sktime.datatypes._convert_utils._chain import (
    get_conversion_path,
    get_converter_chain,
)
from sktime.tests.test_switch import run_test_module_changed
from sktime.utils.dependencies import _check_soft_dependencies

# mtype pairs used by the mock converters below, see _MockConverter
_MOCK_PAIRS = [
    ("A", "B"),
    ("B", "C"),
    ("C", "B"),
    ("B", "A"),
    ("A", "D"),
    ("D", "E"),
    ("E", "C"),
]

# all tests in this module are run only if datatypes have changed
pytestmark = pytest.mark.skipif(
    not run_test_module_changed("sktime.datatypes"),
    reason="Test only if sktime.datatypes has been changed",
)


class _MockConverter(BaseConverter):
    """Mock converter between mock mtypes, for testing chaining.

    Appends the mtype converted to, to the object converted, which is a list.
    Records the conversion in the store, under the key ``"converted_by"``, and
    what it found in the store before, under the key ``"saw"``.
    """

    _tags = {"lossy": False}

    @classmethod
    def get_conversions(cls):
        """Get all conversions - the mock pairs."""
        return _MOCK_PAIRS

    def __init__(self, mtype_from=None, mtype_to=None, lossy=False):
        self.lossy = lossy
        super().__init__(mtype_from=mtype_from, mtype_to=mtype_to)
        self.set_tags(**{"lossy": lossy})

    def _convert(self, obj, store=None):
        if isinstance(store, dict):
            store["saw"] = store.get("converted_by", None)
            store["converted_by"] = f"{self.mtype_from}->{self.mtype_to}"
        return list(obj) + [self.mtype_to]


def _mock_convert_dict(pairs, lossy=()):
    """Return a mock conversion dict, with conversions for ``pairs``.

    Parameters
    ----------
    pairs : iterable of pairs of str
        mtype pairs to define conversions for
    lossy : iterable of pairs of str, optional
        subset of ``pairs``, conversions to declare as lossy
    """
    return {
        (mtype_from, mtype_to, "Mock"): _MockConverter(
            mtype_from=mtype_from,
            mtype_to=mtype_to,
            lossy=(mtype_from, mtype_to) in lossy,
        )
        for mtype_from, mtype_to in pairs
    }


def test_get_conversion_path():
    """Test that a shortest conversion path is found, if one exists."""
    mock_dict = _mock_convert_dict([("A", "B"), ("B", "C")])

    assert get_conversion_path("A", "C", "Mock", mock_dict) == ["A", "B", "C"]
    # paths of defined conversions are of length one
    assert get_conversion_path("A", "B", "Mock", mock_dict) == ["A", "B"]


def test_get_conversion_path_no_path():
    """Test that no path is found if the target is not reachable."""
    mock_dict = _mock_convert_dict([("A", "B"), ("D", "E")])

    assert get_conversion_path("A", "E", "Mock", mock_dict) is None
    # conversions to self are not chained
    assert get_conversion_path("A", "A", "Mock", mock_dict) is None
    # conversions of a different scitype are not chained
    assert get_conversion_path("A", "B", "OtherMock", mock_dict) is None
    assert get_converter_chain("A", "E", "Mock", mock_dict) is None


def test_get_conversion_path_prefers_non_lossy():
    """Test that a non-lossy path is preferred over a shorter lossy path."""
    pairs = [("A", "B"), ("B", "C"), ("A", "D"), ("D", "E"), ("E", "C")]
    mock_dict = _mock_convert_dict(pairs, lossy=[("A", "B")])

    # the path via B is shorter, but lossy, so the path via D, E is used
    assert get_conversion_path("A", "C", "Mock", mock_dict) == ["A", "D", "E", "C"]

    # if the only path is lossy, it is used
    mock_dict = _mock_convert_dict([("A", "B"), ("B", "C")], lossy=[("A", "B")])
    assert get_conversion_path("A", "C", "Mock", mock_dict) == ["A", "B", "C"]


def test_get_converter_chain_cache_invalidation():
    """Test that the conversion path cache is invalidated if conversions change."""
    mock_dict = _mock_convert_dict([("A", "B")])
    assert get_converter_chain("A", "C", "Mock", mock_dict) is None

    mock_dict.update(_mock_convert_dict([("B", "C")]))
    chain = get_converter_chain("A", "C", "Mock", mock_dict)
    assert isinstance(chain, ConverterChain)


def test_converter_chain_applies_elements_in_sequence():
    """Test that a chain applies its elements in the order given."""
    mock_dict = _mock_convert_dict([("A", "B"), ("B", "C")])
    chain = get_converter_chain("A", "C", "Mock", mock_dict)

    assert chain.get_tag("mtype_from") == "A"
    assert chain.get_tag("mtype_to") == "C"
    assert chain(["A"]) == ["A", "B", "C"]


def test_converter_chain_infers_mtypes():
    """Test that a chain infers its mtypes from its elements."""
    converters = [
        _MockConverter(mtype_from="A", mtype_to="B"),
        _MockConverter(mtype_from="B", mtype_to="C"),
    ]
    chain = ConverterChain(converters=converters)

    assert chain.get_tag("mtype_from") == "A"
    assert chain.get_tag("mtype_to") == "C"


def test_converter_chain_lossy_tag():
    """Test that a chain is lossy if and only if an element is lossy."""
    non_lossy = _MockConverter(mtype_from="A", mtype_to="B")
    lossy = _MockConverter(mtype_from="B", mtype_to="C", lossy=True)

    assert not ConverterChain([non_lossy]).get_tag("lossy")
    assert ConverterChain([non_lossy, lossy]).get_tag("lossy")
    assert ConverterChain([lossy]).get_tag("lossy")


def test_converter_chain_invalid_converters():
    """Test that a chain raises if converters are not a sequence of callables."""
    with pytest.raises(ValueError, match="at least one converter"):
        ConverterChain(converters=[], mtype_from="A", mtype_to="C")

    with pytest.raises(TypeError, match="must be callable"):
        ConverterChain(converters=["not a converter"], mtype_from="A", mtype_to="C")

    # mtypes can neither be inferred from a function, nor are they passed
    with pytest.raises(ValueError, match="must be passed to the constructor"):
        ConverterChain(converters=[lambda obj, store=None: obj])


def test_converter_chain_element_stores_are_separate():
    """Test that elements of a chain do not overwrite each other's store."""
    mock_dict = _mock_convert_dict([("A", "B"), ("B", "C")])
    chain = get_converter_chain("A", "C", "Mock", mock_dict)

    store = {}
    chain(["A"], store=store)

    element_stores = store[CHAIN_STORE_KEY]["stores"]
    assert len(element_stores) == 2
    # both elements record under the same key, in their own store
    assert element_stores[0]["converted_by"] == "A->B"
    assert element_stores[1]["converted_by"] == "B->C"


def test_converter_chain_reverse_uses_stores_in_reverse():
    """Test that the reverse of a chain picks up the element stores, reversed."""
    mock_dict = _mock_convert_dict([("A", "B"), ("B", "C"), ("C", "B"), ("B", "A")])
    store = {}

    chain = get_converter_chain("A", "C", "Mock", mock_dict)
    chain(["A"], store=store)

    reverse_chain = get_converter_chain("C", "A", "Mock", mock_dict)
    reverse_chain(["C"], store=store)

    element_stores = store[CHAIN_STORE_KEY]["stores"]
    # the first element of the reverse chain, C->B, must have been passed the
    # store of the last element of the forward chain, B->C
    assert element_stores[0]["saw"] == "B->C"
    assert element_stores[1]["saw"] == "A->B"
    assert store[CHAIN_STORE_KEY]["mtype_from"] == "C"
    assert store[CHAIN_STORE_KEY]["mtype_to"] == "A"


def test_convert_chains_undefined_conversion():
    """Test that convert chains conversions, if none is defined directly."""
    assert ("pd-multiindex", "pd-wide", "Panel") not in convert_dict

    index = pd.MultiIndex.from_product(
        [[0, 1, 2], [0, 1, 2, 3]], names=["instances", "timepoints"]
    )
    obj = pd.DataFrame({"var_0": np.arange(12, dtype="float64")}, index=index)

    converted = convert(obj, "pd-multiindex", "pd-wide", "Panel")

    # pd-wide has instances as rows and time points as columns
    assert isinstance(converted, pd.DataFrame)
    assert converted.shape == (3, 4)

    # converting back results in the values and index of the original
    # the variable name is not preserved, as pd-wide does not store it
    back = convert(converted, "pd-wide", "pd-multiindex", "Panel")
    np.testing.assert_array_equal(back.to_numpy(), obj.to_numpy())
    assert back.index.equals(obj.index)


def test_convert_raises_if_no_chain(monkeypatch):
    """Test that convert raises if no conversion can be obtained by chaining.

    Conversions are patched to contain the identity only, since all mtypes of a
    scitype are connected by the conversions that sktime defines.
    """
    import sktime.datatypes._convert as _convert_module

    identity = {("pd.DataFrame", "pd.DataFrame", "Series"): lambda obj, store=None: obj}
    monkeypatch.setattr(_convert_module, "convert_dict", identity)

    obj = pd.DataFrame({"a": [1, 2, 3]})

    with pytest.raises(NotImplementedError, match="no conversion defined"):
        _convert_module.convert(obj, "pd.DataFrame", "np.ndarray", "Series")


@pytest.mark.skipif(
    not _check_soft_dependencies("polars", severity="none"),
    reason="skip test if polars is not available",
)
def test_convert_chains_polars_conversion():
    """Test chaining of conversions that are only defined via pandas.

    Conversions between polars and non-pandas mtypes are not defined directly,
    only via pandas, so they are obtained by chaining.
    """
    from sktime.datatypes._examples import get_examples

    assert ("pl.DataFrame", "np.ndarray", "Series") not in convert_dict

    obj = get_examples(mtype="pl.DataFrame", as_scitype="Series")[0]
    expected = get_examples(mtype="pd.DataFrame", as_scitype="Series")[0]

    converted = convert(obj, "pl.DataFrame", "np.ndarray", "Series")

    assert isinstance(converted, np.ndarray)
    np.testing.assert_array_equal(converted, expected.to_numpy())
