# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for pandas version compatibility utilities."""

import pandas as pd
import pytest

from sktime.utils.dependencies import _check_soft_dependencies
from sktime.utils.pandas import (
    decode_freq_alias,
    encode_freq_alias,
    freq_equal,
    hash_pandas_index,
    index_sort_values,
    is_pandas_ge_2_1,
    is_pandas_ge_3,
    to_offset_compat,
)


@pytest.mark.parametrize(
    "legacy, modern",
    [
        ("M", "ME"),
        ("2M", "2ME"),
        ("Q", "QE"),
        ("A", "YE"),
        ("BM", "BME"),
        ("D", "D"),
        ("MS", "MS"),
    ],
)
def test_encode_decode_freq_alias_roundtrip(legacy, modern):
    """Legacy and modern aliases round-trip on pandas >= 2.1."""
    if is_pandas_ge_2_1():
        assert encode_freq_alias(legacy) == modern
        assert decode_freq_alias(modern) == legacy
    else:
        assert encode_freq_alias(legacy) == legacy
        assert decode_freq_alias(modern) == modern


def test_freq_equal_legacy_and_modern():
    """M and ME represent the same frequency when compared."""
    assert freq_equal("M", "ME")


def test_to_offset_compat_monthly():
    """Monthly legacy alias parses on all supported pandas versions."""
    offset = to_offset_compat("M")
    assert offset is not None
    if is_pandas_ge_2_1():
        assert offset.freqstr in {"ME", "M"}
    else:
        assert offset.freqstr == "M"


@pytest.mark.parametrize("index", [pd.Index([3, 1, 2]), pd.Index(["c", "a", "b"])])
def test_index_sort_values(index):
    """index_sort_values returns sorted Index."""
    result = index_sort_values(index)
    assert list(result) == sorted(index.tolist())


def test_hash_pandas_index_stable():
    """hash_pandas_index is stable for repeated calls."""
    index = pd.Index([1, 2, 3])
    assert hash_pandas_index(index) == hash_pandas_index(index)


@pytest.mark.skipif(
    not _check_soft_dependencies("pandas>=2.1.0", severity="none"),
    reason="test requires pandas >= 2.1",
)
def test_decode_freq_alias_month_end():
    """ME decodes to sktime canonical M on pandas >= 2.1."""
    assert decode_freq_alias("ME") == "M"
