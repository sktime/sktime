"""Tests for synthetic hierarchical data generation."""

import numpy as np
import pytest
from pandas.testing import assert_frame_equal

from sktime.utils._testing.hierarchical import _make_hierarchical


@pytest.mark.filterwarnings("error::RuntimeWarning")
@pytest.mark.parametrize("all_positive", [True, False])
@pytest.mark.parametrize("n_columns", [1, 3])
@pytest.mark.parametrize(
    "hierarchy_levels, min_timepoints, max_timepoints, same_cutoff, index_type",
    [
        ((1,), 1, 1, True, "range"),
        ((1,), 2, 2, True, "range"),
        ((1,), 3, 3, True, "range"),
        ((2,), 8, 8, True, "period"),
        ((2, 3), 8, 8, True, "datetime"),
        ((2, 3), 4, 8, True, "range"),
        ((2, 3), 4, 8, False, "int"),
    ],
)
def test_make_hierarchical_add_nan_preserves_other_values(
    all_positive,
    n_columns,
    hierarchy_levels,
    min_timepoints,
    max_timepoints,
    same_cutoff,
    index_type,
):
    kwargs = {
        "hierarchy_levels": hierarchy_levels,
        "min_timepoints": min_timepoints,
        "max_timepoints": max_timepoints,
        "same_cutoff": same_cutoff,
        "index_type": index_type,
        "n_columns": n_columns,
        "all_positive": all_positive,
        "random_state": 42,
    }
    expected = _make_hierarchical(**kwargs)
    actual = _make_hierarchical(add_nan=True, **kwargs)
    expected.iloc[[0, len(expected) // 2, -1]] = np.nan

    assert_frame_equal(actual, expected)
