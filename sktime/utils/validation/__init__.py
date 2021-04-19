# -*- coding: utf-8 -*-
__all__ = ["is_int", "check_n_jobs"]
__author__ = ["Markus Löning"]

import os

import numpy as np


def is_int(x):
    """Check if x is of integer type, but not boolean"""
    # boolean are subclasses of integers in Python, so explicitly exclude them
    return isinstance(x, (int, np.integer)) and not isinstance(x, bool)


def is_float(x):
    """Check if x is of integer type, but not boolean"""
    # boolean are subclasses of integers in Python, so explicitly exclude them
    return isinstance(x, (float, np.floating)) and not isinstance(x, bool)


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
    window_length: positive int or positive float
        The number of training set used for splitting
        Window length for transformed feature variables

    Returns
    -------
    window_length: int
    """
    if window_length is not None:

        if n_timepoints is None:
            n_timepoints = 1
        elif not is_int(n_timepoints) or n_timepoints is None:
            raise ValueError("n_timepoints must be a positive integer >= 1")

        valid = False
        if is_int(window_length) and window_length >= 1:
            valid = True

        elif not valid and (is_float(window_length) and 0 < window_length < 1):
            valid = True
            window_length = int(np.ceil(window_length * n_timepoints))

        elif is_float(window_length) and 0 < window_length < 1:
            raise Exception(
                f"if window_lenght : {window_length} is a float, "
                f" n_timepoints cannot be None. input n_timepoints"
            )

        elif not valid:
            raise ValueError(
                f"`{name}` must be a positive integer >= 1, "
                f"float between 0 and 1, or None "
                f"but found: {window_length}"
            )
    return window_length
