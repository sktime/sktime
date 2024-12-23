# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Converters between polars and pandas, with multiindex convention."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from sktime.datatypes._adapter.polars import (
    convert_pandas_to_polars,
    convert_polars_to_pandas,
)
from sktime.utils.dependencies import _check_soft_dependencies

# simple pd.DataFrame fixture
pd_fixture_simple = pd.DataFrame({"foo": [2, 3, 4], "bar": [3, 4, 5]})

# multiindex pd.DataFrame fixture
pd_fixture_multiindex = pd.DataFrame(
    {"a": [1, 2, 3], "b": [1, 2, 3], "foo": [2, 3, 4], "bar": [3, 4, 5]}
)

pd_fixture_multiindex = pd_fixture_multiindex.set_index(["a", "b"], append=True)
pd_fixture_multiindex.index.names = ["a", None, "foo"]
pd_fixture_multiindex.columns = ["a", "foo"]

PANDAS_FIXTURES = [pd_fixture_simple, pd_fixture_multiindex]


@pytest.mark.skipif(
    not _check_soft_dependencies("polars", severity="none"),
    reason="skip test if required soft dependency for polars not available",
)
def test_convert_polars_to_pandas_multiindex():
    """Tests conversion from polars to pandas."""
    from polars import from_pandas

    pd_fixture = pd.DataFrame(
        {"__index__a": [1, 2, 3], "__index__1": [1, 2, 3], "a": [2, 3, 4]}
    )
    pd_fixture_orig = deepcopy(pd_fixture)
    polars_fixture = from_pandas(pd_fixture)

    result = convert_polars_to_pandas(polars_fixture)

    # tests absence of side effects
    assert pd_fixture.equals(pd_fixture_orig)

    # test expected output format
    assert result.index.names == ["a", None]
    assert result.columns == ["a"]
    assert (result.values == pd_fixture[["a"]].values).all()


@pytest.mark.skipif(
    not _check_soft_dependencies("polars", severity="none"),
    reason="skip test if required soft dependency for polars not available",
)
def test_convert_pd_multiindex_to_polars():
    """Tests conversion from pandas to polars is correct for MultiIndex."""
    pd_fixture = pd_fixture_multiindex.copy()
    pd_fixture_orig = deepcopy(pd_fixture)
    result = convert_pandas_to_polars(pd_fixture)

    # tests absence of side effects
    assert pd_fixture.equals(pd_fixture_orig)

    # test expected output format
    expected_columns = ["__index__a", "__index__1", "__index__foo", "a", "foo"]
    assert result.columns == expected_columns
    result_values = result.to_pandas()[["a", "foo"]].values
    pd_fixture_values = pd_fixture[["a", "foo"]].values
    assert (result_values == pd_fixture_values).all()


@pytest.mark.skipif(
    not _check_soft_dependencies("polars", severity="none"),
    reason="skip test if required soft dependency for polars not available",
)
@pytest.mark.parametrize("pd_fixture", PANDAS_FIXTURES)
def test_convert_pd_polars_inverse(pd_fixture):
    """Tests conversions from pandas from/to polars are inverses."""
    polars_result = convert_pandas_to_polars(pd_fixture)
    pd_result = convert_polars_to_pandas(polars_result)

    assert pd_result.equals(pd_fixture)


@pytest.mark.skipif(
    not _check_soft_dependencies("polars", severity="none"),
    reason="skip test if required soft dependency for polars not available",
)
@pytest.mark.parametrize("pd_fixture", PANDAS_FIXTURES)
def test_convert_pd_polars_inverse_lazy(pd_fixture):
    """Tests conversions from pandas from/to polars are inverses."""
    polars_result = convert_pandas_to_polars(pd_fixture, lazy=True)
    pd_result = convert_polars_to_pandas(polars_result)

    assert pd_result.equals(pd_fixture)


@pytest.mark.skipif(
    not _check_soft_dependencies("polars", severity="none"),
    reason="skip test if required soft dependency for polars not available",
)
def test_pandas_to_polars_with_pandas_period_index_and_inverse():
    """Tests conversion from pandas to polars with pandas PeriodIndex."""
    df = pd.DataFrame(
        {"a": [1, 2, 3], "b": [1, 2, 3], "foo": [2, 3, 4], "bar": [3, 4, 5]}
    )
    df = df.set_index(pd.period_range("2021-01-01", periods=3))

    pl_result = convert_pandas_to_polars(df)
    pd_result = convert_polars_to_pandas(pl_result, infer_freq=True)

    assert pl_result.columns == ["__index__0", "a", "b", "foo", "bar"]
    assert list(pd_result.columns) == list(df.columns)
    assert pd_result.index.freq == df.index.freq


@pytest.mark.skipif(
    not _check_soft_dependencies("polars", severity="none"),
    reason="skip test if required soft dependency for polars not available",
)
def test_pd_multiindex_to_polars_with_pandas_period_index_and_inverse():
    """Tests conversion from pandas to polars with pandas PeriodIndex for Multiindex."""
    data = np.random.randn(6, 3)

    arrays = [
        ["A", "A", "B", "B", "C", "C"],
        [1, 2, 1, 2, 1, 2],
        pd.period_range("2023-01", periods=6, freq="M"),
    ]

    index = pd.MultiIndex.from_arrays(arrays, names=["1", "2", "period"])
    pd_multiindex_period_index = pd.DataFrame(
        data, index=index, columns=["foo", "bar", "a"]
    )

    pl_result = convert_pandas_to_polars(pd_multiindex_period_index)
    pd_result = convert_polars_to_pandas(pl_result, infer_freq=True)

    assert pl_result.columns == [
        "__index__1",
        "__index__2",
        "__index__period",
        "foo",
        "bar",
        "a",
    ]
    assert list(pd_result.columns) == list(pd_multiindex_period_index.columns)
    assert pd_result.index.names == pd_multiindex_period_index.index.names
    assert (
        pd_result.index.levels[-1].freq
        == pd_multiindex_period_index.index.levels[-1].freq
    )
