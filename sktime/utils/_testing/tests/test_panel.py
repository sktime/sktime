#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-

__author__ = ["Markus Löning"]
__all__ = []

import numpy as np
import pandas as pd
import pytest

from sktime.series_as_features.tests._config import N_CLASSES
from sktime.series_as_features.tests._config import N_COLUMNS
from sktime.series_as_features.tests._config import N_INSTANCES
from sktime.series_as_features.tests._config import N_TIMEPOINTS
from sktime.utils._testing.panel import make_classification_problem
from sktime.utils._testing.panel import make_regression_problem


def _check_X_y_pandas(X, y, n_instances, n_columns, n_timepoints):
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert X.shape[0] == y.shape[0] == n_instances
    assert X.shape[1] == n_columns
    assert X.iloc[0, 0].shape == (n_timepoints,)


def _check_X_y_numpy(X, y, n_instances, n_columns, n_timepoints):
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape == (n_instances, n_columns, n_timepoints)
    assert y.shape == (n_instances,)


def _check_X_y(X, y, n_instances, n_columns, n_timepoints, check_numpy=False):
    if check_numpy:
        _check_X_y_numpy(X, y, n_instances, n_columns, n_timepoints)
    else:
        _check_X_y_pandas(X, y, n_instances, n_columns, n_timepoints)


@pytest.mark.parametrize("n_instances", N_INSTANCES)
@pytest.mark.parametrize("n_columns", N_COLUMNS)
@pytest.mark.parametrize("n_timepoints", N_TIMEPOINTS)
@pytest.mark.parametrize("n_classes", N_CLASSES)
@pytest.mark.parametrize("return_numpy", [True, False])
def test_make_classification_problem(
    n_instances, n_columns, n_timepoints, n_classes, return_numpy
):
    X, y = make_classification_problem(
        n_instances=n_instances,
        n_classes=n_classes,
        n_columns=n_columns,
        n_timepoints=n_timepoints,
        return_numpy=return_numpy,
    )

    # check dimensions of generated data
    _check_X_y(X, y, n_instances, n_columns, n_timepoints, check_numpy=return_numpy)

    # check number of classes
    assert len(np.unique(y)) == n_classes


@pytest.mark.parametrize("n_instances", N_INSTANCES)
@pytest.mark.parametrize("n_columns", N_COLUMNS)
@pytest.mark.parametrize("n_timepoints", N_TIMEPOINTS)
@pytest.mark.parametrize("return_numpy", [True, False])
def test_make_regression_problem(n_instances, n_columns, n_timepoints, return_numpy):
    X, y = make_regression_problem(
        n_instances=n_instances,
        n_columns=n_columns,
        n_timepoints=n_timepoints,
        return_numpy=return_numpy,
    )

    # check dimensions of generated data
    _check_X_y(X, y, n_instances, n_columns, n_timepoints, check_numpy=return_numpy)
