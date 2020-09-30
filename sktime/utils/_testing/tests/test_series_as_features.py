#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-

__author__ = ["Markus Löning"]
__all__ = []

import pandas as pd
import pytest

from sktime.series_as_features.tests._config import N_CLASSES
from sktime.series_as_features.tests._config import N_COLUMNS
from sktime.series_as_features.tests._config import N_INSTANCES
from sktime.series_as_features.tests._config import N_TIMEPOINTS
from sktime.utils._testing import make_classification_problem
from sktime.utils._testing import make_regression_problem


def _check_X_y(X, y, n_instances, n_columns, n_timepoints):
    assert X.shape[0] == y.shape[0] == n_instances
    assert X.shape[1] == n_columns
    assert X.iloc[0, 0].shape == (n_timepoints,)
    assert isinstance(y, pd.Series)
    assert isinstance(X, pd.DataFrame)


@pytest.mark.parametrize("n_instances", N_INSTANCES)
@pytest.mark.parametrize("n_columns", N_COLUMNS)
@pytest.mark.parametrize("n_timepoints", N_TIMEPOINTS)
@pytest.mark.parametrize("n_classes", N_CLASSES)
def test_make_classification_problem(n_instances, n_columns, n_timepoints, n_classes):
    X, y = make_classification_problem(
        n_instances=n_instances,
        n_classes=n_classes,
        n_columns=n_columns,
        n_timepoints=n_timepoints,
    )

    # check dimensions of generated data
    _check_X_y(X, y, n_instances, n_columns, n_timepoints)

    # check number of classes
    assert y.nunique() == n_classes


@pytest.mark.parametrize("n_instances", N_INSTANCES)
@pytest.mark.parametrize("n_columns", N_COLUMNS)
@pytest.mark.parametrize("n_timepoints", N_TIMEPOINTS)
def test_make_regression_problem(n_instances, n_columns, n_timepoints):
    X, y = make_regression_problem(
        n_instances=n_instances, n_columns=n_columns, n_timepoints=n_timepoints
    )

    # check dimensions of generated data
    _check_X_y(X, y, n_instances, n_columns, n_timepoints)
