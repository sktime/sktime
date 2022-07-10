# -*- coding: utf-8 -*-
"""All transformers in sktime."""

__author__ = ["fkiraly"]

import numpy as np
import pandas as pd

from sktime.registry import all_estimators

est_tuples = all_estimators(estimator_types="transformer", return_names=True)
est_names, ests = zip(*est_tuples)

for i, x in enumerate(est_tuples):
    exec(f"{x[0]} = ests[{i}]")

__all__ = list(est_names) + ["pd", "np"]
