"""Tests for specific bugfixes to conversion logic."""

__author__ = ["fkiraly", "ericjb"]

import pytest

from sktime.datasets import load_airline
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
