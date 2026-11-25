"""Tests for specific bugfixes to conversion logic."""

__author__ = ["fkiraly", "ericjb"]

import numpy as np
import pandas as pd
import pytest

from sktime.datasets import load_airline
from sktime.datatypes import check_is_mtype, convert_to
from sktime.datatypes._series._convert import convert_MvS_to_UvS_as_Series
from sktime.tests.test_switch import run_test_module_changed


@pytest.mark.skipif(
    not run_test_module_changed("sktime.datatypes"),
    reason="Test only if sktime.datatypes or utils.parallel has been changed",
)
def test_multiindex_to_df_list_large_level_values():
    """Tests for failure condition in bug #4668.

    Conversion from pd-multiindex to df-list would fail if the
    first MultiIndex level (level index 0) had strictly more levels
    than unique values in it, this can occur post subsetting.
    """
    from sktime.datasets import load_osuleaf
    from sktime.datatypes import convert_to

    X, _ = load_osuleaf(return_type="pd-multiindex")
    X1 = X.loc[:3]

    convert_to(X1, "df-list")


@pytest.mark.xfail(reason="Failing test for bug #7928, to be fixed")
@pytest.mark.skipif(
    not run_test_module_changed("sktime.datatypes"),
    reason="Test only if sktime.datatypes or utils.parallel has been changed",
)
def test_convert_MvS_to_UvS_as_Series():
    """Checks that column name in MvS is preserved as attr name in UvS"""
    y = load_airline()
    z = y.to_frame()
    w = convert_MvS_to_UvS_as_Series(z)

    assert y.name == w.name


@pytest.mark.skipif(
    not run_test_module_changed("sktime.datatypes"),
    reason="Test only if sktime.datatypes has been changed",
)
@pytest.mark.parametrize(
    "columns",
    [
        pd.Index(["temperature", "pressure", "humidity"]),
        pd.Index([7, 2, 5]),
        pd.MultiIndex.from_tuples([("sensor", "z"), ("sensor", "a"), ("other", "m")]),
        pd.Index([False, True]),
        pd.Index([True, False]),
    ],
    ids=["string", "integer", "multiindex", "boolean", "boolean-reversed"],
)
def test_dflist_to_numpy3d_aligns_variables(columns):
    """Array channels follow the first frame's variable order, see #11799."""
    n_columns = len(columns)
    expected = np.arange(9 * n_columns, dtype=float).reshape(3, n_columns, 3)
    permutations = [np.roll(np.arange(n_columns), i) for i in range(3)]
    panel = [
        pd.DataFrame(values.T, columns=columns).iloc[:, permutation]
        for values, permutation in zip(expected, permutations)
    ]
    before = [frame.copy(deep=True) for frame in panel]
    assert check_is_mtype(panel, "df-list")

    actual = convert_to(panel, "numpy3D", as_scitype="Panel")
    via_pandas = convert_to(
        convert_to(panel, "pd-multiindex", as_scitype="Panel"),
        "numpy3D",
        as_scitype="Panel",
    )

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual, via_pandas)
    for frame, original in zip(panel, before):
        pd.testing.assert_frame_equal(frame, original)


@pytest.mark.skipif(
    not run_test_module_changed("sktime.datatypes"),
    reason="Test only if sktime.datatypes has been changed",
)
@pytest.mark.parametrize("mismatch", ["missing_column", "different_column", "length"])
def test_dflist_to_numpy3d_rejects_incompatible_frames(mismatch):
    """Column alignment must not fill missing variables or unequal lengths."""
    first = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    second = first.copy()
    if mismatch == "missing_column":
        second = second.drop(columns="b")
    elif mismatch == "different_column":
        second = second.rename(columns={"b": "c"})
    else:
        second = second.iloc[:1]

    with pytest.raises(ValueError, match="same length and columns"):
        convert_to([first, second], "numpy3D", as_scitype="Panel")
