# -*- coding: utf-8 -*-
"""All time series classifiers and some classification data sets."""

__author__ = ["mloning", "fkiraly"]

import numpy as np
import pandas as pd

from sktime.datasets import (
    load_arrow_head,
    load_basic_motions,
    load_gunpoint,
    load_osuleaf,
)
from sktime.registry import all_estimators

est_tuples = all_estimators(estimator_types="classifier", return_names=True)
est_names, ests = zip(*est_tuples)

for i, x in enumerate(est_tuples):
    exec(f"{x[0]} = ests[{i}]")

__all__ = list(est_names) + [
    "pd",
    "np",
    "load_gunpoint",
    "load_osuleaf",
    "load_basic_motions",
    "load_arrow_head",
]
