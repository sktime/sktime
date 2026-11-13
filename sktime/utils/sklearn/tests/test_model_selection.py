"""Tests for model selection utilities in utils.sklearn."""

__author__ = ["yash-sangwan"]

import numpy as np
import pytest

from sktime.utils.sklearn._model_selection import _check_param_grid


@pytest.mark.parametrize(
    "param_grid, match",
    [
        ({"strategy": "prior"}, "needs to be a list"),
        ({"strategy": []}, "non-empty sequence"),
        ({"strategy": np.zeros((2, 2))}, "one-dimensional"),
    ],
)
def test_check_param_grid_raises(param_grid, match):
    """Malformed parameter grids raise informative errors."""
    with pytest.raises(ValueError, match=match):
        _check_param_grid(param_grid)


@pytest.mark.parametrize(
    "param_grid",
    [
        {"strategy": ["a", "b"]},
        {"n_neighbors": np.array([1, 3])},
        [{"strategy": ["a"]}, {"n_neighbors": (1, 3)}],
    ],
)
def test_check_param_grid_accepts_valid(param_grid):
    """Valid parameter grids pass validation, in dict and list of dict form."""
    _check_param_grid(param_grid)
