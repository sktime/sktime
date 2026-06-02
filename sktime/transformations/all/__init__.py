"""All transformers in sktime."""

__author__ = ["fkiraly"]

import numpy as np
import pandas as pd

from sktime.registry import all_estimators

est_tuples = all_estimators(estimator_types="transformer", return_names=True)
est_names, ests = zip(*est_tuples)

safe_est_names = []
for i, x in enumerate(est_tuples):
    name = x[0]
    if not name.isidentifier():
        continue
    globals()[name] = ests[i]
    safe_est_names.append(name)

__all__ = safe_est_names + ["pd", "np"]
