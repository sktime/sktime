"""Tests for specific bugfixes to conversion logic."""

__author__ = ["fkiraly"]

import pytest

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


@pytest.mark.skipif(
    not run_test_module_changed("sktime.datatypes"),
    reason="Test only if sktime.datatypes or utils.parallel has been changed",
)
@pytest.mark.parametrize("name", ["0", 0, None])
def test_pdseries_round_trips(name):
    """Test consistency of round trips between pd.Series and pd.DataFrame mtypes."""
    import pandas as pd

    from sktime.datatypes import convert_to

    # series -> df -> series round trip
    y = pd.Series([1, 2, 3], name=name)

    store = {}

    y_df = convert_to(y, "pd.DataFrame", store=store)

    y_round_trip = convert_to(y_df, "pd.Series", store=store)

    assert y_round_trip.name == name

    # df -> series -> df round trip
    y = pd.DataFrame([1, 2, 3], columns=[name])

    store = {}

    y_series = convert_to(y, "pd.Series", store=store)

    y_round_trip = convert_to(y_series, "pd.DataFrame", store=store)

    assert y_round_trip.columns[0] == name
