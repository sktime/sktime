# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for sliding greedy splitter."""

import numpy as np
import pandas as pd
import pytest

from sktime.split import SlidingGreedySplitter
from sktime.tests.test_switch import run_test_for_class
from sktime.utils._testing.hierarchical import _make_hierarchical
from sktime.utils._testing.series import _make_series


@pytest.mark.skipif(
    not run_test_for_class(SlidingGreedySplitter),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_sliding_greedy_splitter_lengths():
    """Test that SlidingGreedySplitter returns the correct lengths."""
    y = np.arange(10)
    cv = SlidingGreedySplitter(train_size=4, test_size=2, folds=3)

    lengths = [(len(trn), len(tst)) for trn, tst in cv.split_series(y)]
    assert lengths == [(4, 2), (4, 2), (4, 2)]


@pytest.mark.skipif(
    not run_test_for_class(SlidingGreedySplitter),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_sliding_greedy_splitter_indices():
    """Test that SlidingGreedySplitter returns the correct indices."""
    y = np.arange(10)
    cv = SlidingGreedySplitter(train_size=4, test_size=2, folds=2)

    # Convert to list of tuples for easy comparison
    splits = [(list(trn), list(tst)) for trn, tst in cv.split(y)]
    
    # The expected indices for a sliding greedy splitter with train_size=4, test_size=2, folds=2
    expected = [([2, 3, 4, 5], [6, 7]), ([4, 5, 6, 7], [8, 9])]
    
    # Compare actual and expected splits
    assert len(splits) == len(expected)
    for i, (actual, exp) in enumerate(zip(splits, expected)):
        assert actual[0] == exp[0], f"Train indices mismatch in fold {i}"
        assert actual[1] == exp[1], f"Test indices mismatch in fold {i}"


@pytest.mark.skipif(
    not run_test_for_class(SlidingGreedySplitter),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_sliding_greedy_splitter_step_length():
    """Test SlidingGreedySplitter with custom step_length."""
    y = np.arange(10)
    cv = SlidingGreedySplitter(train_size=4, test_size=2, folds=2, step_length=1)

    splits = [(list(trn), list(tst)) for trn, tst in cv.split(y)]
    expected = [([3, 4, 5, 6], [7, 8]), ([4, 5, 6, 7], [8, 9])]
    
    assert len(splits) == len(expected)
    for i, (actual, exp) in enumerate(zip(splits, expected)):
        assert actual[0] == exp[0], f"Train indices mismatch in fold {i}"
        assert actual[1] == exp[1], f"Test indices mismatch in fold {i}" 