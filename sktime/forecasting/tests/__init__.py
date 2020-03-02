#!/usr/bin/env python3 -u
# coding: utf-8

__author__ = ["Markus Löning"]

import numpy as np

# default parameter testing grid
DEFAULT_WINDOW_LENGTHS = [1, 5]
DEFAULT_STEP_LENGTHS = [1, 5]
DEFAULT_FHS = [1, np.array([2, 5])]
DEFAULT_INSAMPLE_FHS = [
    -3,  # single in-sample
    np.array([-2, -5]),  # multiple in-sample
    0,
    np.array([-3, 2])  # mixed in-sample and out-of-sample
]
DEFAULT_SPS = [3, 7, 12]
DEFAULT_ALPHAS = [0.95, 0.9]
