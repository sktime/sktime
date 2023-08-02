# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for converter utilities between dask and pandas, with multiindex convention."""

from copy import deepcopy

import pandas as pd
import pytest

from sktime.datatypes._adapter.dask_to_pd import (
    convert_dask_to_pandas,
    convert_pandas_to_dask,
)
from sktime.utils.validation._dependencies import _check_soft_dependencies

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
    not _check_soft_dependencies("dask", severity="none"),
    reason="skip test if required soft dependency for dask not available",
)
def test_convert_dask_to_pd_multiindex():
    """Tests back-conversion from dask to pandas is correct for MultiIndex."""
    from dask.dataframe import from_pandas

    pd_fixture = pd.DataFrame(
        {"__index__a": [1, 2, 3], "__index__1": [1, 2, 3], "a": [2, 3, 4]}
    )
    pd_fixture_orig = deepcopy(pd_fixture)
    dask_fixture = from_pandas(pd_fixture, npartitions=1)

    result = convert_dask_to_pandas(dask_fixture)

    # tests absence of side effects
    assert pd_fixture.equals(pd_fixture_orig)

    # test expected output format
    assert result.index.names == ["a", None]
    assert result.columns == ["a"]
    assert (result.values == pd_fixture[["a"]].values).all()


@pytest.mark.skipif(
    not _check_soft_dependencies("dask", severity="none"),
    reason="skip test if required soft dependency for dask not available",
)
def test_convert_pd_multiindex_to_dask():
    """Tests conversion from pandas to dask is correct for MultiIndex."""
    pd_fixture = pd_fixture_multiindex.copy()
    pd_fixture_orig = deepcopy(pd_fixture)
    result = convert_pandas_to_dask(pd_fixture)

    # tests absence of side effects
    assert pd_fixture.equals(pd_fixture_orig)

    # test expected output format
    expected_cols = ["__index__a", "__index__1", "__index__foo", "a", "foo"]
    assert list(result.columns) == expected_cols
    result_values = result.loc[:, ["a", "foo"]].compute().values
    assert (result_values == pd_fixture.values).all()


@pytest.mark.skipif(
    not _check_soft_dependencies("dask", severity="none"),
    reason="skip test if required soft dependency for dask not available",
)
@pytest.mark.parametrize("pd_fixture", PANDAS_FIXTURES)
def test_convert_pd_dask_inverse(pd_fixture):
    """Tests conversions from pandas from/to dask are inverses."""
    dask_result = convert_pandas_to_dask(pd_fixture)
    back_result = convert_dask_to_pandas(dask_result)

    assert pd_fixture.equals(back_result)
