#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Markus Löning"]
__all__ = [
    "make_classification_problem",
    "make_regression_problem",
]

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_random_state


def _make_series_as_features_X(
    y, n_columns, n_timepoints, return_numpy=False, random_state=None
):
    n_instances = len(y)
    rng = check_random_state(random_state)

    # 3d numpy array
    if return_numpy:
        X = np.empty((n_instances, n_columns, n_timepoints))
        for i in range(n_instances):
            for j in range(n_columns):
                X[i, j, :] = rng.normal(loc=y[i] * 100, scale=0.5, size=n_timepoints)
        return X

    # nested pandas DataFrame
    else:
        columns = []
        for _ in range(n_columns):
            rows = []
            for j in range(n_instances):
                # we use the y value for the mean of the generated time series
                row = pd.Series(rng.normal(loc=y[j] * 20, scale=0.5, size=n_timepoints))
                rows.append(row)
            column = pd.Series(rows)
            columns.append(column)
        return pd.DataFrame(columns).T


def _make_regression_y(n_instances=10, return_numpy=True, random_state=None):
    rng = check_random_state(random_state)
    y = rng.normal(size=n_instances)
    if return_numpy:
        return y
    else:
        return pd.Series(y)


def _make_classification_y(
    n_instances=10, n_classes=2, return_numpy=True, random_state=None
):
    if not n_instances > n_classes:
        raise ValueError("n_instances must be bigger than n_classes")
    rng = check_random_state(random_state)
    n_repeats = int(np.ceil(n_instances / n_classes))
    y = np.tile(np.arange(n_classes), n_repeats)[:n_instances]
    rng.shuffle(y)
    if return_numpy:
        return y
    else:
        return pd.Series(y)


def make_classification_problem(
    n_instances=15,
    n_columns=1,
    n_timepoints=20,
    n_classes=2,
    return_numpy=False,
    random_state=None,
):
    y = _make_classification_y(
        n_instances, n_classes, return_numpy=return_numpy, random_state=random_state
    )
    X = _make_series_as_features_X(
        y, n_columns, n_timepoints, return_numpy=return_numpy, random_state=random_state
    )

    return X, y


def make_regression_problem(
    n_instances=5, n_columns=1, n_timepoints=20, return_numpy=False, random_state=None
):
    y = _make_regression_y(
        n_instances, random_state=random_state, return_numpy=return_numpy
    )
    X = _make_series_as_features_X(
        y, n_columns, n_timepoints, return_numpy=return_numpy, random_state=random_state
    )
    return X, y
