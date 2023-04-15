# -*- coding: utf-8 -*-
"""Utilities to profile computational and memory performance of estimators."""

from inspect import isclass
from itertools import product
from timeit import default_timer as timer

import pandas as pd

from sktime.utils._testing.panel import make_classification_problem


def profile_classifier(
    est,
    n_instances_grid=None,
    n_timepoints_grid=None,
    n_replicates=20,
    return_replicates=False,
):
    """Profile runtime of a classifier.

    Carries out a single fit and in-sample predict of est, on data
    generated via make_classification_problem with n_instances, n_timepoints
    as in the grid defined by the lists n_instances_grid, n_timepoints_grid.
    Each experiment is repeated n_replicates times.

    Of each experiment, time spent in fit and time spent in predict is measured.


    Parameters
    ----------
    est : sktime classifier, BaseClassifier descendant, object or class
    n_instances_grid : list of int, default = None = [20, 40, 60, 80]
        list of instance sizes (n_instances) in classification X/y to test
    n_timepoints_grid : list of int, default = None = [20, 40, 60, 80]
        list of time series lengths (n_instances) in classification X/y to test
    n_replicates : int, optional, default = 20
        number of fit/predict replicates per individual size
    return_replicates : bool, optional, default = False
        Whether times are returned for each replicate (True) or summarized (False)

    Returns
    -------
    pd.DataFrame with results of experiment
    If return_replicates=False:
        row index = (n_instances, n_timepoints)
        col index = ("time_fit", mean or std) or ("time_pred", mean or std)
        entries are mean/std of times spent in fit/predict, over replicate time sample
    If return_replicates=True:
        row index = (n_instances, n_timepoints, replicate_id)
        col index = "time_fit" or "time_pred"
        entries are individual times spent in fit/predict, for replicate w replicate_id

    Examples
    --------
    >>> from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
    >>> from sktime.utils.profiling import profile_classifier
    >>>
    >>> n_instances_grid = [20, 30]
    >>> n_timepoints_grid = [20, 30]
    >>> results = profile_classifier(
    ...     KNeighborsTimeSeriesClassifier,
    ...     n_instances_grid=n_instances_grid,
    ...     n_timepoints_grid=n_timepoints_grid,
    ... )
    """
    if isclass(est):
        est = est.create_test_instance()
    if n_instances_grid is None:
        n_instances_grid = [20, 40, 60, 80]
    if n_timepoints_grid is None:
        n_timepoints_grid = [20, 40, 60, 80]
    grid = list(product(n_instances_grid, n_timepoints_grid, range(n_replicates)))

    time_fit_list = []
    time_pred_list = []
    for n_inst, n_tp, _rep in grid:
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
        index=pd.MultiIndex.from_tuples(
            grid, names=["n_instances", "n_timepoints", "replicate_id"]
        ),
    )

    if return_replicates:
        return results

    else:
        res = results
        names = ["n_instances", "n_timepoints"]
        means = res.groupby(res.index.droplevel(-1)).mean()
        means.index = pd.MultiIndex.from_tuples(means.index, names=names)
        std = res.groupby(res.index.droplevel(-1)).std()
        std.index = pd.MultiIndex.from_tuples(std.index, names=names)

        res_summary = pd.concat([means, std], axis=1, keys=["mean", "std"])
        res_summary.columns = res_summary.columns.swaplevel().sort_values()

        return res_summary
