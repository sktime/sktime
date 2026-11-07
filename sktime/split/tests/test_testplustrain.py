# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for TestPlusTrain splitter."""

import numpy as np
import pytest

from sktime.split import SlidingWindowSplitter, TestPlusTrainSplitter
from sktime.tests.test_switch import run_test_for_class
from sktime.utils._testing.series import _make_series


@pytest.mark.skipif(
    not run_test_for_class([SlidingWindowSplitter, TestPlusTrainSplitter]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_test_plus_train_splitter_fh():
    """Test that TestPlusTrainSplitter takes fh from cv.

    Failure case of bug #10945: fh silently fell back to the
    ``BaseSplitter`` defaults, 1 and 10, instead of those of the wrapped splitter.
    """
    # both values must differ from the BaseSplitter defaults, 10 and 1,
    # otherwise the assertions below pass on the unfixed fallback
    cv = SlidingWindowSplitter(window_length=5, fh=[1, 2, 3])
    splitter = TestPlusTrainSplitter(cv)

    assert splitter.fh == [1, 2, 3]


@pytest.mark.skipif(
    not run_test_for_class([SlidingWindowSplitter, TestPlusTrainSplitter]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_test_plus_train_splitter():
    """Test that TestPlusTrainSplitter adds the train set to the test set."""
    y = _make_series()
    cv = SlidingWindowSplitter(window_length=5, fh=[1, 2, 3])
    splitter = TestPlusTrainSplitter(cv)

    # iloc references, via _split
    splits_iloc = list(splitter.split(y))
    splits_cv_iloc = list(cv.split(y))
    assert len(splits_iloc) == len(splits_cv_iloc) > 0

    for (train, test), (train_inner, test_inner) in zip(splits_iloc, splits_cv_iloc):
        assert np.all(train == train_inner)
        assert np.all(test == np.union1d(train_inner, test_inner))

    # loc references, via _split_loc, a separate implementation
    splits_loc = list(splitter.split_loc(y))
    splits_cv_loc = list(cv.split_loc(y))
    assert len(splits_loc) == len(splits_cv_loc) > 0

    for (train, test), (train_inner, test_inner) in zip(splits_loc, splits_cv_loc):
        assert np.all(train == train_inner)
        assert np.all(test == train_inner.union(test_inner))
