"""All time series detectors."""

from sktime.registry import all_estimators

est_tuples = all_estimators(estimator_types="detector", return_names=True)
est_names, ests = zip(*est_tuples)

for i, x in enumerate(est_tuples):
    exec(f"{x[0]} = ests[{i}]")

__all__ = list(est_names)
