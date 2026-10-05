# !/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Import all time series forecasting functionality available in sktime."""

from sktime.registry import all_estimators

est_tuples = all_estimators(estimator_types="forecaster", return_names=True)
est_names, ests = zip(*est_tuples)

for i, x in enumerate(est_tuples):
    exec(f"{x[0]} = ests[{i}]")

__all__ = list(est_names)
