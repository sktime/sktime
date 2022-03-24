# -*- coding: utf-8 -*-
"""Perform benchmarking timing experiments to help assess impact of code changes.

These are not formal tests. Instead, they form a means of manual sanity checks to
ensure that changes to algorithms do not result in significant slow down,
or conversely to assess whether changes give significant speed up.
"""
__author__ = ["TonyBagnall"]

import numpy as np


def distance_timing():
    """Time distance functions for increased series length."""
    for i in range(100):
        first = np.ndarray()
