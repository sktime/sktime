# -*- coding: utf-8 -*-
__all__ = ["is_int", "check_n_jobs"]
__author__ = ["Markus Löning"]

import os

import numpy as np


def is_int(x):
    """Check if x is of integer type, but not boolean"""
    # boolean are subclasses of integers in Python, so explicitly exclude them
    return isinstance(x, (int, np.integer)) and not isinstance(x, bool)


def check_n_jobs(n_jobs):
    """Check n_jobs parameter according to scikit-learn convention.

    Parameters
    ----------
    n_jobs : int, positive or -1

    Returns
    -------
    n_jobs : int
    """
    # scikit-learn convention
    # https://scikit-learn.org/stable/glossary.html#term-n-jobs
    if n_jobs is None:
        return 1
    if not is_int(n_jobs):
        raise ValueError(f"`n_jobs` must be None or an integer, but found: {n_jobs}")
    if n_jobs < 0:
        return os.cpu_count() - n_jobs + 1
    else:
        return n_jobs


def check_window_length(window_length, name="`window_length`"):
    """Validate window length"""
    if window_length is not None:
        if not is_int(window_length) or window_length < 1:
            raise ValueError(
                f"{name} must be a positive integer >= 1 or None, "
                f"but found: {window_length}"
            )
    return window_length
