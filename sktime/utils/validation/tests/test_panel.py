#!/usr/bin/env python3 -u
"""Unit tests for sktime.utils.validation.panel check functions."""

__author__ = ["mloning", "TonyBagnall"]
__all__ = ["test_check_X_bad_input_args"]

import numpy as np
import pandas as pd
import pytest

from sktime.utils._testing.panel import make_classification_problem
from sktime.utils.validation.panel import check_X, check_X_y, check_y

BAD_INPUT_ARGS = [
    [0, 1, 2],  # list
    np.empty(2),  # 1d np.array
    np.empty((3, 2, 3, 2)),  # 4d np.array
    pd.DataFrame(np.empty((2, 3))),  # non-nested pd.DataFrame
]
y = pd.Series(dtype=int)


@pytest.mark.parametrize("X", BAD_INPUT_ARGS)
def test_check_X_bad_input_args(X):
    """Test for the correct reaction for bad input in check_X."""
    with pytest.raises(ValueError):
        check_X(X)

    with pytest.raises(ValueError):
        check_X_y(X, y)


def test_check_enforce_min_instances():
    """Test minimum instances enforced in check_X."""
    X, y = make_classification_problem(n_instances=3)
    msg = r"instance"
    with pytest.raises(ValueError, match=msg):
        check_X(X, enforce_min_instances=4)

    with pytest.raises(ValueError, match=msg):
        check_X_y(X, y, enforce_min_instances=4)

    with pytest.raises(ValueError, match=msg):
        check_y(y, enforce_min_instances=4)


def test_check_X_enforce_univariate():
    """Test univariate enforced in check_X."""
    X, y = make_classification_problem(n_columns=2)
    msg = r"univariate"
    with pytest.raises(ValueError, match=msg):
        check_X(X, enforce_univariate=True)

    with pytest.raises(ValueError, match=msg):
        check_X_y(X, y, enforce_univariate=True)


def test_check_X_enforce_min_columns():
    """Test minimum columns enforced in check_X."""
    X, y = make_classification_problem(n_columns=2)
    msg = r"columns"
    with pytest.raises(ValueError, match=msg):
        check_X(X, enforce_min_columns=3)

    with pytest.raises(ValueError, match=msg):
        check_X_y(X, y, enforce_min_columns=3)
