"""Utility functions for interval scorers."""

import numpy as np


def check_cuts_array(
    cuts: np.ndarray,
    n_samples: int,
    min_size: int | None = None,
    last_dim_size: int = 2,
) -> np.ndarray:
    """Check array type cuts.

    Parameters
    ----------
    cuts : np.ndarray
        Array of cuts to check.
    n_samples : int
        Number of samples in the data.
    min_size : int, optional (default=1)
        Minimum size of the intervals obtained by the cuts.
    last_dim_size : int, optional (default=2)
        Size of the last dimension.

    Returns
    -------
    cuts : np.ndarray
        The unmodified input cuts array.

    Raises
    ------
    ValueError
        If the cuts does not meet the requirements.
    """
    if min_size is None:
        min_size = 1

    if cuts.ndim != 2:
        raise ValueError("The cuts must be a 2D array.")

    if not np.issubdtype(cuts.dtype, np.integer):
        raise ValueError("The cuts must be of integer type.")

    if cuts.shape[-1] != last_dim_size:
        raise ValueError(
            "The cuts must be specified as an array with length "
            f"{last_dim_size} in the last dimension."
        )

    if not np.all(cuts >= 0) or not np.all(cuts <= n_samples):
        raise ValueError(
            "All cuts must be non-negative, and less than "
            f"or equal to the number of samples=({n_samples})."
        )

    interval_sizes = np.diff(cuts, axis=1)
    if not np.all(interval_sizes >= min_size):
        min_interval_size = np.min(interval_sizes)
        raise ValueError(
            "All rows in `cuts` must be strictly increasing and each entry must"
            f" be more than min_size={min_size} apart."
            f" Found a minimum interval size of {min_interval_size}."
        )
    return cuts
