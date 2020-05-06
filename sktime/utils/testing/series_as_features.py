#!/usr/bin/env python3 -u
# coding: utf-8

__author__ = ["Markus Löning"]
__all__ = [
    "make_classification_problem",
    "make_regression_problem",
]

import pandas as pd
from sklearn.utils.validation import check_random_state


def _make_series_as_features_X(n_instances, n_columns, n_timepoints,
                               random_state=None):
    rng = check_random_state(random_state)
    columns = []
    for i in range(n_columns):
        rows = []
        for j in range(n_instances):
            row = pd.Series(rng.normal(size=n_timepoints))
            rows.append(row)
        column = pd.Series(rows)
        columns.append(column)
    return pd.DataFrame(columns).T


def make_classification_problem(n_instances=20, n_columns=1,
                                n_timepoints=20, random_state=None):
    X = _make_series_as_features_X(n_instances, n_columns, n_timepoints,
                                   random_state=random_state)

    rng = check_random_state(random_state)
    y = pd.Series(rng.randint(0, 1, size=n_instances))
    return X, y


def make_regression_problem(n_instances=20, n_columns=1, n_timepoints=20,
                            random_state=None):
    X = _make_series_as_features_X(n_instances, n_columns, n_timepoints,
                                   random_state=random_state)
    rng = check_random_state(random_state)
    y = pd.Series(rng.normal(size=n_instances))
    return X, y
