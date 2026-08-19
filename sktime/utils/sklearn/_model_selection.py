"""Fork of sklearn model selection utilities that were removed upstream."""
# copyright/attribution scikit-learn developers
# _check_param_grid is a copy of the function of the same name from sklearn 1.0.2,
# which was removed in later versions of scikit-learn

__all__ = ["_check_param_grid"]

from collections.abc import Sequence

import numpy as np


def _check_param_grid(param_grid):
    """Validate param_grid, from sklearn 1.0.2, before it was removed.

    Parameters
    ----------
    param_grid : dict or list of dict
        parameter grid to validate, as passed to a grid search tuner

    Raises
    ------
    ValueError
        if a parameter grid value is not a non-empty, one-dimensional sequence
    """
    if hasattr(param_grid, "items"):
        param_grid = [param_grid]

    for p in param_grid:
        for name, v in p.items():
            if isinstance(v, np.ndarray) and v.ndim > 1:
                raise ValueError("Parameter array should be one-dimensional.")

            if isinstance(v, str) or not isinstance(v, (np.ndarray, Sequence)):
                raise ValueError(
                    f"Parameter grid for parameter ({name}) needs to"
                    f" be a list or numpy array, but got ({type(v)})."
                    " Single values need to be wrapped in a list"
                    " with one element."
                )

            if len(v) == 0:
                raise ValueError(
                    f"Parameter values for parameter ({name}) need "
                    "to be a non-empty sequence."
                )
