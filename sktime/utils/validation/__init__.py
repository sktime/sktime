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
    if n_jobs == -1:
        return os.cpu_count()
    else:
        return n_jobs
