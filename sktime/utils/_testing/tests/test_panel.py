# -*- coding: utf-8 -*-
"""Tests for data and scenario generators in _testing.panel module."""

__author__ = ["mloning", "fkiraly"]
__all__ = []

import numpy as np
import pandas as pd
import pytest

from sktime.datatypes import check_is_mtype
from sktime.utils._testing.panel import (
    _make_panel,
    make_classification_problem,
    make_regression_problem,
)

N_INSTANCES = [10, 15]
N_COLUMNS = [3, 5]
N_TIMEPOINTS = [3, 5]
N_CLASSES = [2, 5]


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


MTYPES = ["pd-multiindex", "numpy3D", "nested_univ", "df-list"]


@pytest.mark.parametrize("n_instances", N_INSTANCES)
@pytest.mark.parametrize("n_columns", N_COLUMNS)
@pytest.mark.parametrize("n_timepoints", N_TIMEPOINTS)
@pytest.mark.parametrize("return_mtype", MTYPES)
def test_make_panel(n_instances, n_columns, n_timepoints, return_mtype):
    """Test that _make_panel utility returns panel data of right format."""
    X = _make_panel(
        n_instances=n_instances,
        n_columns=n_columns,
        n_timepoints=n_timepoints,
        return_mtype=return_mtype,
    )

    valid, _, metadata = check_is_mtype(X, mtype=return_mtype, return_metadata=True)
    msg = f"_make_panel_X generated data does not comply with mtype {return_mtype}"
    assert valid, msg
    assert metadata["n_instances"] == n_instances
    assert metadata["is_univariate"] == (n_columns == 1)


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
