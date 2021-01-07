#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-

__author__ = ["Ansgar Asseburg"]
__all__ = [
    "UNIVARIATES",
    "MULTIVARIATES"
]

import pandas as pd
from sktime.utils._testing.series import _make_series
from numpy import random as rd

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
