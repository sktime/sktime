#!/usr/bin/env python3 -u
# coding: utf-8

__author__ = ["Markus Löning"]
__all__ = []

from typing import Iterable

import numpy as np
import pandas as pd


def _subtract_time(a, b):
    """Helper function to subtract time points"""
    difference = a - b

    if isinstance(difference, Iterable):
        if isinstance(difference[0], (pd.Period, pd.DateOffset)):
            return np.array([value.n for value in difference])
        else:
            return difference

    elif isinstance(difference, (pd.Period, pd.DateOffset)):
        return difference.n

    else:
        return difference
