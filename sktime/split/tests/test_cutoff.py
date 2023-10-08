# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for cutoff splitter."""

import numpy as np
import pytest

from sktime.forecasting.tests._config import (
    TEST_CUTOFFS,
    TEST_FHS,
    TEST_FHS_TIMEDELTA,
    TEST_WINDOW_LENGTHS,
    TEST_YS,
)
from sktime.split import CutoffSplitter
from sktime.split.base._common import _inputs_are_supported
from sktime.split.tests.test_split import _check_cv


@pytest.mark.parametrize("y", TEST_YS)
@pytest.mark.parametrize("cutoffs", TEST_CUTOFFS)
@pytest.mark.parametrize("fh", [*TEST_FHS, *TEST_FHS_TIMEDELTA])
@pytest.mark.parametrize("window_length", TEST_WINDOW_LENGTHS)
def test_cutoff_window_splitter(y, cutoffs, fh, window_length):
    """Test CutoffSplitter."""
    if _inputs_are_supported([cutoffs, fh, window_length]):
        cv = CutoffSplitter(cutoffs, fh=fh, window_length=window_length)
        train_windows, test_windows, cutoffs, n_splits = _check_cv(cv, y)
        np.testing.assert_array_equal(cutoffs, cv.get_cutoffs(y))
    else:
        match = "Unsupported combination of types"
        with pytest.raises(TypeError, match=match):
            CutoffSplitter(cutoffs, fh=fh, window_length=window_length)
