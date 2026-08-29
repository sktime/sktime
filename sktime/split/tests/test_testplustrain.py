# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for TestPlusTrainSplitter."""

from sktime.split import SlidingWindowSplitter, TestPlusTrainSplitter


def test_test_plus_train_forwards_fh_and_window_length():
    """Test that TestPlusTrainSplitter forwards fh and window_length from cv.

    Regression test for issue #10945.
    """
    cv = SlidingWindowSplitter(window_length=5, fh=[1, 2, 3])
    splitter = TestPlusTrainSplitter(cv)

    assert splitter.fh == cv.fh
    assert splitter.window_length == cv.window_length
