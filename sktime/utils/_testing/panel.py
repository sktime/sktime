#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Markus Löning"]
__all__ = [
    "make_classification_problem",
    "make_regression_problem",
    "make_transformer_problem",
]

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_random_state

from sktime.utils.data_processing import from_3d_numpy_to_nested
from sktime.clustering.tests._clustering_tests import generate_univaritate_series


def _make_panel_X(
    n_instances=20,
    n_columns=1,
    n_timepoints=20,
    y=None,
    return_numpy=False,
    random_state=None,
):
    # If target variable y is given, we ignore n_instances and instead generate as
    # many instances as in the target variable
    if y is not None:
        y = np.asarray(y)
        n_instances = len(y)
    rng = check_random_state(random_state)

    # Generate data as 3d numpy array
    X = rng.normal(scale=0.5, size=(n_instances, n_columns, n_timepoints))

    # Generate association between data and target variable
    if y is not None:
        X = X + (y * 100).reshape(-1, 1, 1)

    if return_numpy:
        return X
    else:
        return from_3d_numpy_to_nested(X)


def _make_regression_y(n_instances=20, return_numpy=True, random_state=None):
    rng = check_random_state(random_state)
    y = rng.normal(size=n_instances)
    if return_numpy:
        return y
    else:
        return pd.Series(y)


def _make_classification_y(
    n_instances=20, n_classes=2, return_numpy=True, random_state=None
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
    n_instances=20,
    n_columns=1,
    n_timepoints=20,
    n_classes=2,
    return_numpy=False,
    random_state=None,
):
    y = _make_classification_y(
        n_instances, n_classes, return_numpy=return_numpy, random_state=random_state
    )
    X = _make_panel_X(
        n_columns=n_columns,
        n_timepoints=n_timepoints,
        return_numpy=return_numpy,
        random_state=random_state,
        y=y,
    )

    return X, y


def make_regression_problem(
    n_instances=20, n_columns=1, n_timepoints=20, return_numpy=False, random_state=None
):
    y = _make_regression_y(
        n_instances, random_state=random_state, return_numpy=return_numpy
    )
    X = _make_panel_X(
        n_columns=n_columns,
        n_timepoints=n_timepoints,
        return_numpy=return_numpy,
        random_state=random_state,
        y=y,
    )
    return X, y


def make_clustering_problem(
    n_instances=20, series_size=20, return_numpy=True, random_state=None
):
    # Can only currently support univariate so converting
    # to univaritate for the time being
    X = generate_univaritate_series(n_instances, series_size, random_state)

    if return_numpy:
        return X
    else:
        return pd.Series(X)


def make_transformer_problem(
    n_instances=20,
    n_columns=1,
    n_timepoints=20,
    return_numpy=True,
    random_state=None,
    panel=True,
):
    if not panel:
        X = make_transformer_problem(
            n_instances=n_instances,
            n_columns=n_columns,
            n_timepoints=n_timepoints,
            return_numpy=True,
            random_state=random_state,
            panel=True,
        )
        if return_numpy:
            X = X[0]
        else:
            X = pd.DataFrame(X[0])
    else:
        X = _make_panel_X(
            n_instances=n_instances,
            n_columns=n_columns,
            n_timepoints=n_timepoints,
            return_numpy=True,
            random_state=random_state,
        )
        if not return_numpy:
            arr = []
            for data in X:
                arr.append(pd.DataFrame(data))
            X = arr

    return X


def _make_nested_from_array(array, n_instances=20, n_columns=1):
    return pd.DataFrame(
        [[pd.Series(array) for _ in range(n_columns)] for _ in range(n_instances)],
        columns=[f"col{c}" for c in range(n_columns)],
    )
