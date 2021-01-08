#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-

__author__ = ["Ansgar Asseburg"]
__all__ = [
    "UNIVARIATES",
    "MULTIVARIATES",
    "SAMPLE"
]

import pandas as pd
from sktime.utils._testing.series import _make_series
from numpy import random as rd
import numpy as np

RANDOM_SEED = 42

UNIVARIATES = [
    [rd.uniform(50, 100, (1, rd.randint(50, 100))),
     rd.uniform(50, 100, (1, rd.randint(50, 100)))],
    [rd.uniform(-50, 100, (1, rd.randint(50, 100))),
     rd.uniform(-50, 100, (1, rd.randint(50, 100)))]
]

MULTIVARIATES = [
    [rd.uniform(50, 100, (rd.randint(2, 10), rd.randint(50, 100))),
     rd.uniform(50, 100, (rd.randint(2, 10), rd.randint(50, 100)))],
    [rd.uniform(-50, 100, (rd.randint(2, 10), rd.randint(50, 100))),
     rd.uniform(-50, 100, (rd.randint(2, 10), rd.randint(50, 100)))]
]

SAMPLE = [
    [np.array([[5, 7, 4, 4, 3, 2]]),
     np.array([[1, 2, 3, 2, 2]]),
     3.735758994891947]
]
