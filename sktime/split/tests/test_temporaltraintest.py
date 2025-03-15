# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for temporal train test splitter."""

import numpy as np
import pandas as pd
import pytest

from sktime.datasets import load_airline
from sktime.datatypes._utilities import get_cutoff
from sktime.forecasting.tests._config import TEST_OOS_FHS, VALID_INDEX_FH_COMBINATIONS
from sktime.split import TemporalTrainTestSplitter, temporal_train_test_split
from sktime.tests.test_switch import run_test_for_class
from sktime.utils._testing.forecasting import _make_fh
from sktime.utils._testing.series import _make_series


def _check_train_test_split_y(fh, split):
    assert len(split) == 2

    train, test = split
    assert isinstance(train, pd.Series)
    assert isinstance(test, pd.Series)
    assert set(train.index).isdisjoint(test.index)
    for test_timepoint in test.index:
        assert np.all(train.index < test_timepoint)
    assert len(test) == len(fh)
    assert len(train) > 0

    cutoff = train.index[-1]
    np.testing.assert_array_equal(test.index, fh.to_absolute(cutoff).to_numpy())


@pytest.mark.skipif(
    not run_test_for_class([TemporalTrainTestSplitter, temporal_train_test_split]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "index_type, fh_type, is_relative", VALID_INDEX_FH_COMBINATIONS
)
@pytest.mark.parametrize("values", TEST_OOS_FHS)
def test_split_by_fh(index_type, fh_type, is_relative, values):
    """Test temporal_train_test_split."""
    if fh_type == "timedelta":
        pytest.skip(
            "ForecastingHorizon with timedelta values "
            "is currently experimental and not supported everywhere"
        )
    y = _make_series(20, index_type=index_type)
    cutoff = get_cutoff(y.iloc[:10], return_index=True)
    fh = _make_fh(cutoff, values, fh_type, is_relative)
    split = temporal_train_test_split(y, fh=fh)
    _check_train_test_split_y(fh, split)


@pytest.mark.skipif(
    not run_test_for_class([TemporalTrainTestSplitter, temporal_train_test_split]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_temporal_train_test_split_float_only_y():
    """Test temporal_train_test_split expected output on float size inputs."""
    y = load_airline()

    # only test_size
    y_train, y_test = temporal_train_test_split(y, test_size=0.2)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)
    assert len(y_train) == 115
    assert len(y_test) == 29
    assert (y[:115] == y_train).all()
    assert (y[-29:] == y_test).all()

    # only train_size
    y_train, y_test = temporal_train_test_split(y, train_size=0.8)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)
    assert len(y_train) == 115
    assert len(y_test) == 29
    assert (y[:115] == y_train).all()
    assert (y[-29:] == y_test).all()

    # both train and test size
    y_train, y_test = temporal_train_test_split(y, train_size=0.3, test_size=0.2)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)
    assert len(y_train) == 43
    assert len(y_test) == 29
    assert (y[:43] == y_train).all()
    assert (y[43 : (43 + 29)] == y_test).all()


@pytest.mark.skipif(
    not run_test_for_class([TemporalTrainTestSplitter, temporal_train_test_split]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_temporal_train_test_split_int_only_y():
    """Test temporal_train_test_split expected output on float size inputs."""
    y = load_airline()

    # only test_size
    y_train, y_test = temporal_train_test_split(y, test_size=29)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)
    assert len(y_train) == 115
    assert len(y_test) == 29
    assert (y[:115] == y_train).all()
    assert (y[-29:] == y_test).all()

    # only train_size
    y_train, y_test = temporal_train_test_split(y, train_size=115)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)
    assert len(y_train) == 115
    assert len(y_test) == 29
    assert (y[:115] == y_train).all()
    assert (y[-29:] == y_test).all()

    # both train and test size
    y_train, y_test = temporal_train_test_split(y, train_size=43, test_size=29)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)
    assert len(y_train) == 43
    assert len(y_test) == 29
    assert (y[:43] == y_train).all()
    assert (y[43 : (43 + 29)] == y_test).all()


@pytest.mark.skipif(
    not run_test_for_class([TemporalTrainTestSplitter, temporal_train_test_split]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_temporal_train_test_split_hierarchical():
    """Test correctness of temporal_train_test_split for hierarchical data.

    Failure case is bug #5107.
    """
    from sktime.utils._testing.hierarchical import _make_hierarchical

    y = _make_hierarchical()
    y_train, y_test = temporal_train_test_split(y, test_size=0.2)

    # y has 12 time indices per series, the same indices for all series
    # so 9 indices should be in train and 3 in test
    # the failure case in 5107 would ignore the hierarchical structure and just
    # put the last 20 indices of the hierarchical data frame in test
    assert len(y_train.index.get_level_values(-1).unique()) == 9
    assert len(y_test.index.get_level_values(-1).unique()) == 3
