"""Testing utilities in the datatype module."""

__author__ = ["fkiraly"]

import numpy as np
import pandas as pd
import pytest

from sktime.datatypes import update_data
from sktime.datatypes._check import check_is_mtype
from sktime.datatypes._examples import get_examples
from sktime.datatypes._utilities import (
    _get_cutoff_from_index,
    get_cutoff,
    get_slice,
    get_time_index,
    get_window,
)
from sktime.tests.test_switch import run_test_module_changed
from sktime.utils._testing.hierarchical import _make_hierarchical

SCITYPE_MTYPE_PAIRS = [
    ("Series", "pd.Series"),
    ("Series", "pd.DataFrame"),
    ("Series", "np.ndarray"),
    ("Panel", "pd-multiindex"),
    ("Panel", "numpy3D"),
    ("Panel", "nested_univ"),
    ("Panel", "df-list"),
    ("Hierarchical", "pd_multiindex_hier"),
]


@pytest.mark.skipif(
    not run_test_module_changed("sktime.datatypes"),
    reason="Test only if sktime.datatypes or utils.parallel has been changed",
)
@pytest.mark.parametrize("scitype,mtype", SCITYPE_MTYPE_PAIRS)
def test_get_time_index(scitype, mtype):
    """Tests that get_time_index returns the expected output.

    Note: this is tested only for fixtures with equal time index across instances,
    as get_time_index assumes that.

    Parameters
    ----------
    scitype : str - scitype of input
    mtype : str - mtype of input

    Raises
    ------
    AssertionError if get_time_index does not return the expected return
        for any fixture example of given scitype, mtype
    """
    # get_time_index currently does not work for df-list type, skip
    if mtype == "df-list":
        return None

    # retrieve example fixture
    fixtures = get_examples(mtype=mtype, as_scitype=scitype, return_metadata=True)

    for fixture_tuple in fixtures.values():
        fixture = fixture_tuple[0]
        fixture_metadata = fixture_tuple[2]

        if fixture is None:
            continue
        if not fixture_metadata.get("is_equal_index", True):
            continue

        idx = get_time_index(fixture)

        msg = f"get_time_index should return pd.Index, but found {type(idx)}"
        assert isinstance(idx, pd.Index), msg

        if mtype in ["pd.Series", "pd.DataFrame"]:
            assert (idx == fixture.index).all()

        if mtype in ["np.ndarray", "numpy3D"]:
            assert isinstance(idx, pd.RangeIndex)
            if mtype == "np.ndarray":
                assert len(idx) == fixture.shape[0]
            else:
                assert len(idx) == fixture.shape[-1]

        if mtype in ["pd-multiindex", "pd_multiindex_hier"]:
            exp_idx = fixture.index.get_level_values(-1).unique()
            assert (idx == exp_idx).all()


@pytest.mark.skipif(
    not run_test_module_changed("sktime.datatypes"),
    reason="Test only if sktime.datatypes or utils.parallel has been changed",
)
@pytest.mark.parametrize("convert_input", [True, False])
@pytest.mark.parametrize("reverse_order", [True, False])
@pytest.mark.parametrize("return_index", [True, False])
@pytest.mark.parametrize("scitype,mtype", SCITYPE_MTYPE_PAIRS)
def test_get_cutoff(scitype, mtype, return_index, reverse_order, convert_input):
    """Tests that get_cutoff has correct output.

    Parameters
    ----------
    scitype : str - scitype of input
    mtype : str - mtype of input
    return_index : bool - whether index (True) or index element is returned (False)
    reverse_order : bool - whether first (True) or last index (False) is retrieved
    convert_input : bool - whether input is converted (True) or passed through (False)

    Raises
    ------
    AssertionError if get_cutoff does not return a length 1 pandas.index
        for any fixture example of given scitype, mtype
    """
    # retrieve example fixture
    fixtures = get_examples(mtype=mtype, as_scitype=scitype, return_metadata=True)

    for fixture_tuple in fixtures.values():
        fixture = fixture_tuple[0]
        fixture_metadata = fixture_tuple[2]
        fixture_equally_spaced = fixture_metadata.get("is_equally_spaced", True)
        fixture_equal_index = fixture_metadata.get("is_equal_index", True)

        if fixture is None:
            continue

        cutoff = get_cutoff(
            fixture,
            return_index=return_index,
            reverse_order=reverse_order,
            convert_input=convert_input,
        )

        if return_index:
            expected_types = pd.Index
            cutoff_val = cutoff[0]
        else:
            expected_types = (int, float, np.int64, pd.Timestamp)
            cutoff_val = cutoff

        msg = (
            f"incorrect return type of get_cutoff"
            f"expected {expected_types}, found {type(cutoff)}"
        )

        assert isinstance(cutoff, expected_types), msg

        if return_index:
            assert len(cutoff) == 1
            if isinstance(cutoff_val, (pd.Period, pd.Timestamp)):
                assert hasattr(cutoff, "freq")
                if fixture_equally_spaced and fixture_equal_index:
                    assert cutoff.freq is not None

        if isinstance(fixture, np.ndarray):
            if reverse_order:
                assert cutoff_val == 0
            else:
                assert cutoff_val > 0

        if mtype in ["pd.Series", "pd.DataFrame"]:
            if reverse_order:
                assert cutoff_val == fixture.index[0]
            else:
                assert cutoff_val == fixture.index[-1]

        if mtype in ["pd-multiindex", "pd_multiindex_hier"]:
            time_idx = fixture.index.get_level_values(-1)
            if reverse_order:
                assert cutoff_val == time_idx.min()
            else:
                assert cutoff_val == time_idx.max()


@pytest.mark.skipif(
    not run_test_module_changed("sktime.datatypes"),
    reason="Test only if sktime.datatypes or utils.parallel has been changed",
)
@pytest.mark.parametrize("reverse_order", [True, False])
def test_get_cutoff_from_index(reverse_order):
    """Tests that _get_cutoff_from_index has correct output.

    Parameters
    ----------
    return_index : bool - whether index (True) or index element is returned (False)
    reverse_order : bool - whether first (True) or last index (False) is retrieved

    Raises
    ------
    AssertionError if _get_cutoff_from_index does not return a length 1 pandas.index
    AssertionError if _get_cutoff_from_index does not return the correct cutoff value
    """
    hier_fixture = _make_hierarchical()
    hier_idx = hier_fixture.index

    cutoff = _get_cutoff_from_index(
        hier_idx, return_index=True, reverse_order=reverse_order
    )
    idx = _get_cutoff_from_index(
        hier_idx, return_index=False, reverse_order=reverse_order
    )

    assert isinstance(cutoff, pd.DatetimeIndex) and len(cutoff) == 1
    assert cutoff.freq == "D"
    assert idx == cutoff[0]

    if reverse_order:
        assert idx == pd.Timestamp("2000-01-01")
    else:
        assert idx == pd.Timestamp("2000-01-12")

    series_fixture = get_examples("pd.Series")[0]
    series_idx = series_fixture.index

    cutoff = _get_cutoff_from_index(
        series_idx, return_index=True, reverse_order=reverse_order
    )
    idx = _get_cutoff_from_index(
        series_idx, return_index=False, reverse_order=reverse_order
    )

    assert isinstance(cutoff, pd.Index) and len(cutoff) == 1
    assert pd.api.types.is_integer_dtype(cutoff)
    assert idx == cutoff[0]

    if reverse_order:
        assert idx == 0
    else:
        assert idx == 3


@pytest.mark.skipif(
    not run_test_module_changed("sktime.datatypes"),
    reason="Test only if sktime.datatypes or utils.parallel has been changed",
)
@pytest.mark.parametrize("bad_inputs", ["foo", 12345, [[[]]]])
def test_get_cutoff_wrong_input(bad_inputs):
    """Tests that get_cutoff raises error on bad input when input checks are enabled.

    Parameters
    ----------
    bad_inputs : inputs that should set off the input checks

    Raises
    ------
    Exception (from pytest) if the error is not raised as expected
    """
    with pytest.raises(Exception, match="must be of Series, Panel, or Hierarchical"):
        get_cutoff(bad_inputs, check_input=True)


@pytest.mark.skipif(
    not run_test_module_changed("sktime.datatypes"),
    reason="Test only if sktime.datatypes or utils.parallel has been changed",
)
def test_get_cutoff_inferred_freq():
    """Tests that get_cutoff infers the freq in a case where it is not directly set.

    Ensures that the bug in #4405 does not occur, combined with the forecaster contract.
    """
    np.random.seed(seed=0)

    past_data = pd.DataFrame(
        {
            "time_identifier": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-05",
                    "2024-01-06",
                    "2024-01-07",
                    "2024-01-08",
                ],
                format="%Y-%m-%d",
            ),
            "series_data": np.random.random(size=8),
        }
    )
    past_data = past_data.set_index(["time_identifier"])
    cutoff = get_cutoff(past_data, return_index=True)
    assert cutoff.freq == "D"


@pytest.mark.skipif(
    not run_test_module_changed("sktime.datatypes"),
    reason="Test only if sktime.datatypes or utils.parallel has been changed",
)
def test_get_cutoff_inferred_freq_small_series():
    """Tests that get_cutoff does not fail on series smaller than three elements.

    The purpose of this test is to check that the ValueError raised by pd.infer_freq is
    not propagated to the user, but rather caught and handled by falling back to None.

    See https://github.com/sktime/sktime/issues/5853
    and https://github.com/sktime/sktime/pull/6097 for more details.
    """
    y = pd.DataFrame(
        data={"y": [1, 2]},
        index=pd.to_datetime(["2020-01-01", "2020-01-02"]),
    )
    cutoff = get_cutoff(y, return_index=True)
    assert cutoff.freq is None

    # Check that it also works for multi-indexed DataFrames
    y = _make_hierarchical(hierarchy_levels=(2,), min_timepoints=2, max_timepoints=3)
    cutoff = get_cutoff(y, return_index=True)
    assert cutoff.freq is None


@pytest.mark.skipif(
    not run_test_module_changed("sktime.datatypes"),
    reason="Test only if sktime.datatypes or utils.parallel has been changed",
)
@pytest.mark.parametrize("window_length, lag", [(2, 0), (None, 0), (4, 1)])
@pytest.mark.parametrize("scitype,mtype", SCITYPE_MTYPE_PAIRS)
def test_get_window_output_type(scitype, mtype, window_length, lag):
    """Tests that get_window runs for all mtypes, and returns output of same mtype.

    Parameters
    ----------
    scitype : str - scitype of input
    mtype : str - mtype of input
    window_length : int, passed to get_window
    lag : int, passed to get_window

    Raises
    ------
    Exception if get_window raises one
    """
    # retrieve example fixture
    fixture = get_examples(mtype=mtype, as_scitype=scitype, return_lossy=False)[0]
    X = get_window(fixture, window_length=window_length, lag=lag)
    valid, err, _ = check_is_mtype(
        X, mtype=mtype, return_metadata=True, msg_return_dict="list"
    )

    msg = (
        f"get_window should return an output of mtype {mtype} for that type of input, "
        f"but it returns an output not conformant with that mtype."
        f"Error from mtype check: {err}"
    )

    assert valid, msg


@pytest.mark.skipif(
    not run_test_module_changed("sktime.datatypes"),
    reason="Test only if sktime.datatypes or utils.parallel has been changed",
)
def test_get_window_expected_result():
    """Tests that get_window produces return of the right length.

    Raises
    ------
    Exception if get_window raises one
    AssertionError if get_window output shape is not as expected
    """
    X_df = get_examples(mtype="pd.DataFrame")[0]
    assert len(get_window(X_df, 2, 1)) == 2
    assert len(get_window(X_df, 3, 1)) == 3
    assert len(get_window(X_df, 1, 2)) == 1
    assert len(get_window(X_df, 3, 4)) == 0
    assert len(get_window(X_df, 3, None)) == 3
    assert len(get_window(X_df, None, 2)) == 2
    assert len(get_window(X_df, None, None)) == 4

    X_mi = get_examples(mtype="pd-multiindex")[0]
    assert len(get_window(X_mi, 3, 1)) == 6
    assert len(get_window(X_mi, 2, 0)) == 6
    assert len(get_window(X_mi, 2, 4)) == 0
    assert len(get_window(X_mi, 1, 2)) == 3
    assert len(get_window(X_mi, 2, None)) == 6
    assert len(get_window(X_mi, None, 2)) == 3
    assert len(get_window(X_mi, None, None)) == 9

    X_hi = get_examples(mtype="pd_multiindex_hier")[0]
    assert len(get_window(X_hi, 3, 1)) == 12
    assert len(get_window(X_hi, 2, 0)) == 12
    assert len(get_window(X_hi, 2, 4)) == 0
    assert len(get_window(X_hi, 1, 2)) == 6
    assert len(get_window(X_hi, 2, None)) == 12
    assert len(get_window(X_hi, None, 2)) == 6
    assert len(get_window(X_hi, None, None)) == 18

    X_np3d = get_examples(mtype="numpy3D")[0]
    assert get_window(X_np3d, 3, 1).shape == (2, 2, 3)
    assert get_window(X_np3d, 2, 0).shape == (2, 2, 3)
    assert get_window(X_np3d, 2, 4).shape == (0, 2, 3)
    assert get_window(X_np3d, 1, 2).shape == (1, 2, 3)
    assert get_window(X_np3d, 2, None).shape == (2, 2, 3)
    assert get_window(X_np3d, None, 2).shape == (1, 2, 3)
    assert get_window(X_np3d, None, None).shape == (3, 2, 3)


@pytest.mark.skipif(
    not run_test_module_changed("sktime.datatypes"),
    reason="Test only if sktime.datatypes or utils.parallel has been changed",
)
@pytest.mark.parametrize("scitype,mtype", SCITYPE_MTYPE_PAIRS)
def test_get_slice_output_type(scitype, mtype):
    """Tests that get_slice runs for all mtypes, and returns output of same mtype.

    Parameters
    ----------
    scitype : str - scitype of input
    mtype : str - mtype of input

    Raises
    ------
    Exception if get_slice raises one
    """
    # retrieve example fixture
    fixture = get_examples(mtype=mtype, as_scitype=scitype, return_lossy=False)[0]
    X = get_slice(fixture)
    valid, err, _ = check_is_mtype(
        X, mtype=mtype, return_metadata=True, msg_return_dict="list"
    )

    msg = (
        f"get_slice should return an output of mtype {mtype} for that type of input, "
        f"but it returns an output not conformant with that mtype."
        f"Error from mtype check: {err}"
    )

    assert valid, msg


@pytest.mark.skipif(
    not run_test_module_changed("sktime.datatypes"),
    reason="Test only if sktime.datatypes or utils.parallel has been changed",
)
def test_get_slice_expected_result():
    """Tests that get_slice produces return of the right length.

    Raises
    ------
    Exception if get_slice raises one
    """
    X_df = get_examples(mtype="pd.DataFrame")[0]
    assert len(get_slice(X_df, start=1, end=3)) == 2

    X_s = get_examples(mtype="pd.Series")[0]
    assert len(get_slice(X_s, start=1, end=3)) == 2

    X_np = get_examples(mtype="numpy3D")[0]
    assert get_slice(X_np, start=1, end=3).shape == (2, 2, 3)


@pytest.mark.skipif(
    not run_test_module_changed("sktime.datatypes"),
    reason="Test only if sktime.datatypes or utils.parallel has been changed",
)
def test_retain_series_freq_on_update():
    """Tests that the frequency of a series is retained after updating it"""
    from sktime.datasets import load_airline
    from sktime.split import temporal_train_test_split

    y = load_airline()

    # create dummy index with hourly timestamps and panel data by hour of day
    ind = pd.date_range(
        start="1960-01-01 10:00:00", periods=len(y.index), freq="24H", name="datetime"
    )
    y = pd.Series(y.values, index=ind, name="passengers")
    y_train, y_test = temporal_train_test_split(y, test_size=2)

    # update the series with the test data
    y_new = update_data(y_train, y_test)

    assert y_new.equals(y)
    assert y_new.index.equals(y.index)
    assert y_new.index.freq == y.index.freq
    assert y_new.index.freqstr == y.index.freqstr
    assert y.index.equals(y_new.index.to_period().to_timestamp())
