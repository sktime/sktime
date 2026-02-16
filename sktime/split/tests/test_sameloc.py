# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for SameLoc splitter."""

import numpy as np
import pytest

from sktime.split import ExpandingWindowSplitter, SameLocSplitter
from sktime.tests.test_switch import run_test_for_class
from sktime.utils._testing.hierarchical import _make_hierarchical


@pytest.mark.skipif(
    not run_test_for_class([ExpandingWindowSplitter, SameLocSplitter]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_same_loc_splitter():
    """Test that SameLocSplitter works as intended."""
    from sktime.datasets import load_airline

    y = load_airline()
    y_template = y[:60]
    cv_tpl = ExpandingWindowSplitter(fh=[2, 4], initial_window=24, step_length=12)

    splitter = SameLocSplitter(cv_tpl, y_template)

    # these should be the same
    # not in general, but only because y is longer only at the end
    split_template_iloc = list(cv_tpl.split(y_template))
    split_templated_iloc = list(splitter.split(y))

    for (t1, tt1), (t2, tt2) in zip(split_template_iloc, split_templated_iloc):
        assert np.all(t1 == t2)
        assert np.all(tt1 == tt2)

    # these should be in general the same
    split_template_loc = list(cv_tpl.split_loc(y_template))
    split_templated_loc = list(splitter.split_loc(y))

    for (t1, tt1), (t2, tt2) in zip(split_template_loc, split_templated_loc):
        assert np.all(t1 == t2)
        assert np.all(tt1 == tt2)


@pytest.mark.skipif(
    not run_test_for_class([ExpandingWindowSplitter, SameLocSplitter]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_same_loc_splitter_hierarchical():
    """Test that SameLocSplitter works as intended for hierarchical data."""
    hierarchy_levels1 = (2, 2)
    hierarchy_levels2 = (3, 4)
    n1 = 7
    n2 = 2 * n1
    y_template = _make_hierarchical(
        hierarchy_levels=hierarchy_levels1, max_timepoints=n1, min_timepoints=n1
    )

    y = _make_hierarchical(
        hierarchy_levels=hierarchy_levels2, max_timepoints=n2, min_timepoints=n2
    )

    cv_tpl = ExpandingWindowSplitter(fh=[1, 2], initial_window=1, step_length=2)

    splitter = SameLocSplitter(cv_tpl, y_template)

    # these should be in general the same
    split_template_loc = list(cv_tpl.split_loc(y_template))
    split_templated_loc = list(splitter.split_loc(y))

    for (t1, tt1), (t2, tt2) in zip(split_template_loc, split_templated_loc):
        assert np.all(t1 == t2)
        assert np.all(tt1 == tt2)
