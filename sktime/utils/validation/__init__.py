# -*- coding: utf-8 -*-
__all__ = ["is_int", "is_float", "check_n_jobs", "check_window_length"]
__author__ = ["Markus Löning", "Taiwo Owoseni"]

import os

import numpy as np


def is_int(x):
    """Check if x is of integer type, but not boolean"""
    # boolean are subclasses of integers in Python, so explicitly exclude them
    return isinstance(x, (int, np.integer)) and not isinstance(x, bool)


def is_float(x):
    """Check if x is of float type"""
    return isinstance(x, (float, np.floating))


def check_n_jobs(n_jobs):
    """Check `n_jobs` parameter according to the scikit-learn convention.

    Parameters
    ----------
    n_jobs : int, positive or -1
        The number of jobs for parallelization.

    Returns
    -------
    n_jobs : int
        Checked number of jobs.
    """
    # scikit-learn convention
    # https://scikit-learn.org/stable/glossary.html#term-n-jobs
    if n_jobs is None:
        return 1
    elif not is_int(n_jobs):
        raise ValueError(f"`n_jobs` must be None or an integer, but found: {n_jobs}")
    elif n_jobs < 0:
        return os.cpu_count() - n_jobs + 1
    else:
        return n_jobs


def check_window_length(window_length, n_timepoints=None, name="window_length"):
    """Validate window length"""
    """
    Parameters
    ----------
    window_length: positive int, positive float in (0, 1), or None
        The window length:
        - If int, the total number of time points.
        - If float, the fraction of time points relative to `n_timepoints`.
    n_timepoints: positive int, optional (default=None)
        The number of time points to which to apply `window_length` when
        passed as a float (fraction). Will be ignored if `window_length` is
        an integer.
    name: str
        Name of argument for error messages.

    Returns
    -------
    window_length: int
    """
    if window_length is None:
        return window_length

    elif is_int(window_length) and window_length >= 1:
        return window_length

    elif is_float(window_length) and 0 < window_length < 1:
        # Check `n_timepoints`.
        if not is_int(n_timepoints) or n_timepoints < 2:
            raise ValueError(
                f"`n_timepoints` must be a positive integer, but found:"
                f" {n_timepoints}."
            )

        # Compute fraction relative to `n_timepoints`.
        return int(np.ceil(window_length * n_timepoints))

    else:
        raise ValueError(
            f"`{name}` must be a positive integer >= 1, or"
            f"float in (0, 1) or None, but found: {window_length}."
        )
