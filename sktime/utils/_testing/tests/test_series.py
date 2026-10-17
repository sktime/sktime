"""Tests for synthetic Series data generation."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal, assert_series_equal

from sktime.utils._testing.series import _make_series


@pytest.mark.filterwarnings("error::RuntimeWarning")
@pytest.mark.parametrize("all_positive", [True, False])
@pytest.mark.parametrize("n_timepoints", [1, 2, 3, 8])
@pytest.mark.parametrize(
    "n_columns, return_params",
    [
        (1, {}),
        (3, {}),
        (1, {"return_mtype": "pd.Series"}),
        (1, {"return_mtype": "pd.DataFrame"}),
        (3, {"return_mtype": "pd.DataFrame"}),
        (1, {"return_mtype": "np.ndarray"}),
        (3, {"return_mtype": "np.ndarray"}),
        (1, {"return_numpy": True}),
        (3, {"return_numpy": True}),
    ],
)
def test_make_series_add_nan_preserves_other_values(
    all_positive, n_timepoints, n_columns, return_params
):
    kwargs = {
        "n_timepoints": n_timepoints,
        "n_columns": n_columns,
        "all_positive": all_positive,
        "random_state": 42,
        **return_params,
    }
    expected = _make_series(**kwargs)
    actual = _make_series(add_nan=True, **kwargs)
    missing_rows = [0, n_timepoints // 2, -1]

    if isinstance(expected, np.ndarray):
        expected[missing_rows] = np.nan
        assert_array_equal(actual, expected)
    else:
        expected.iloc[missing_rows] = np.nan
        if isinstance(expected, pd.Series):
            assert_series_equal(actual, expected)
        else:
            assert_frame_equal(actual, expected)
