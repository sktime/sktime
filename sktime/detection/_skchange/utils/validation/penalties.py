"""Validation functions for penalties."""

import copy

import numpy as np


def check_penalty(
    penalty: np.ndarray | float | None,
    arg_name: str,
    caller_name: str,
    require_constant_penalty: bool = False,
    allow_none: bool = True,
    allow_non_decreasing: bool = False,
) -> None:
    """Check if the given penalty is valid.

    Parameters
    ----------
    penalty : np.ndarray | float | None
        The penalty to check.
    arg_name : str
        The name of the argument. Used for error messages.
    caller_name : str
        The name of the caller. Used for error messages.
    require_constant_penalty : bool, default = False
        If True, the penalty must be a single value (constant penalty).
    allow_none : bool, default = True
        If True, the penalty can be None. If False, the penalty cannot be None.
    allow_non_decreasing : bool, default = False
        If True, the penalty can be non-decreasing.
    """
    penalty = copy.deepcopy(penalty)  # Avoid modifying the original penalty.

    if not allow_none and penalty is None:
        raise ValueError(f"`{arg_name}` cannot be None in {caller_name}")
    if penalty is None:
        return None

    penalty = np.atleast_1d(np.asarray([penalty]).squeeze())
    if penalty.ndim != 1:
        raise ValueError(
            f"`{arg_name}` must be a 1D array in {caller_name}."
            f" Got {penalty.ndim}D array."
        )
    if penalty.size < 1:
        raise ValueError(
            f"`{arg_name}` must have at least one element in {caller_name}"
        )
    if require_constant_penalty and penalty.size != 1:
        raise ValueError(
            f"`{arg_name}` must be a single penalty value in {caller_name}."
            f" Got {penalty.size} values."
        )
    if not np.all(penalty >= 0.0):
        raise ValueError(f"`{arg_name}` must be non-negative in {caller_name}")
    if not allow_non_decreasing and not np.all(np.diff(penalty) >= 0):
        raise ValueError(f"`{arg_name}` must be non-decreasing in {caller_name}")


def check_penalty_against_data(
    penalty: np.ndarray | float,
    X: np.ndarray,
    caller_name: str,
) -> None:
    """Check if the given penalty is valid against the data.

    Parameters
    ----------
    penalty : np.ndarray | float | None
        The penalty to check.
    X : np.ndarray
        The data to check against.
    arg_name : str
        The name of the argument. Used for error messages.
    caller_name : str
        The name of the caller. Used for error messages.
    """
    penalty = copy.deepcopy(penalty)  # Avoid modifying the original penalty.
    penalty = np.asarray([penalty]).flatten()

    if penalty.size != 1 and penalty.size != X.shape[1]:
        raise ValueError(
            f"`penalty` must be a single value or an array of size"
            f" `X.shape[1]={X.shape[1]}` in {caller_name}."
            f" Got `penalty.size={penalty.size}`."
        )
