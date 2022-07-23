# -*- coding: utf-8 -*-
"""Utilities to profile computational and memory performance of estimators."""

from inspect import isclass
from itertools import product
from timeit import default_timer as timer

import pandas as pd

from sktime.utils._testing.panel import make_classification_problem


def profile_classifier(
    est,
    n_instance_grid=None,
    n_timepoints_grid=None,
    n_replicates=20,
    return_replicates=False
):
    """Profile runtime of a classifier.

    Parameters
    ----------
    est : sktime classifier, BaseClassifier descendant, object or class
    n_instance_grid : list of int, default = None = [20, 40, 60, 80]
        list of instance sizes (n_instances) in classification X/y to test
    n_timepoints_grid : list of int, default = None = [20, 40, 60, 80]
        list of time series lengths (n_instances) in classification X/y to test
    n_replicates : int, optional, default = 20
        number of fit/predict replicates per individual size
    return_replicates : bool, optional, default = False
    """
    if isclass(est):
        est = est.create_test_instance()
    if n_instance_grid is None:
        n_instance_grid = [20, 40, 60, 80]
    if n_timepoints_grid is None:
        n_timepoints_grid = [20, 40, 60, 80]
    grid = list(product(n_instance_grid, n_timepoints_grid, range(n_replicates)))

    time_fit_list = []
    time_pred_list = []
    for n_inst, n_tp, rep in grid:
        X, y = make_classification_problem(n_instances=n_inst, n_timepoints=n_tp)

        est_i = est.clone()

        start_fit = timer()
        est_i.fit(X, y)
        end_fit = timer()
        time_fit_list += [end_fit - start_fit]

        start_predict = timer()
        _ = est_i.predict(X)
        end_predict = timer()
        time_pred_list += [end_predict - start_predict]

    results = pd.DataFrame(
        {"time_fit": time_fit_list, "time_pred": time_pred_list},
        index=pd.MultiIndex.from_tuples(grid)
    )

    return results
