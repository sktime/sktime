"""Tests for specific bugfixes to conversion logic."""

__author__ = ["fkiraly", "ericjb"]

import pytest
from skbase.utils.dependencies import _check_soft_dependencies

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


@pytest.mark.skipif(
    not run_test_module_changed("sktime.datatypes")
    or not _check_soft_dependencies("polars", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_convert_polars_numpy_as_series():
    """Tests for failure condition in bug #11271.

    Conversion between the pl.DataFrame and np.ndarray mtypes of scitype Series
    was not defined in either direction, so convert raised NotImplementedError.
    """
    import numpy as np
    import pandas as pd
    import polars as pl

    from sktime.datatypes import check_is_mtype, convert
    from sktime.utils.deep_equals import deep_equals

    X_pl = pl.DataFrame(
        {
            "__index__": pd.date_range("2020-01-01", periods=5, freq="D"),
            "y": np.arange(5.0),
        }
    )
    store = {}

    X_np = convert(
        obj=X_pl,
        from_type="pl.DataFrame",
        to_type="np.ndarray",
        as_scitype="Series",
        store=store,
    )

    assert check_is_mtype(X_np, "np.ndarray", return_metadata=False)
    assert np.array_equal(X_np, np.arange(5.0).reshape(-1, 1))

    X_pl_back = convert(
        obj=X_np,
        from_type="np.ndarray",
        to_type="pl.DataFrame",
        as_scitype="Series",
        store=store,
    )

    assert check_is_mtype(X_pl_back, "pl.DataFrame", return_metadata=False)
    assert deep_equals(X_pl_back, X_pl)

    arr = np.arange(10.0).reshape(5, 2)

    arr_pl = convert(
        obj=arr, from_type="np.ndarray", to_type="pl.DataFrame", as_scitype="Series"
    )

    assert check_is_mtype(arr_pl, "pl.DataFrame", return_metadata=False)

    arr_back = convert(
        obj=arr_pl, from_type="pl.DataFrame", to_type="np.ndarray", as_scitype="Series"
    )

    assert np.array_equal(arr_back, arr)
