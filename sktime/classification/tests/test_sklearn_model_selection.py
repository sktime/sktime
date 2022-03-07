# -*- coding: utf-8 -*-
"""Unit tests for classifier compatability with sklearn model selection."""

__author__ = ["MatthewMiddlehurst"]

import numpy as np
import pytest
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    GroupShuffleSplit,
    HalvingGridSearchCV,
    HalvingRandomSearchCV,
    KFold,
    LeaveOneOut,
    LeavePGroupsOut,
    LeavePOut,
    RandomizedSearchCV,
    RepeatedKFold,
    ShuffleSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
    StratifiedShuffleSplit,
    TimeSeriesSplit,
    cross_val_score,
)

from sktime.classification.interval_based import CanonicalIntervalForest
from sktime.utils._testing.estimator_checks import _make_args

DATA_ARGS = [
    {"return_numpy": True, "n_columns": 2},
    {"return_numpy": False, "n_columns": 2},
]
CROSS_VALIDATION_METHODS = [
    KFold(n_splits=2),
    RepeatedKFold(n_splits=2, n_repeats=2),
    LeaveOneOut(),
    LeavePOut(p=5),
    ShuffleSplit(n_splits=2, test_size=0.25),
    StratifiedKFold(n_splits=2),
    StratifiedShuffleSplit(n_splits=2, test_size=0.25),
    GroupKFold(n_splits=2),
    StratifiedGroupKFold(n_splits=2),
    LeavePGroupsOut(n_groups=5),
    GroupShuffleSplit(n_splits=2, test_size=0.25),
    TimeSeriesSplit(n_splits=2),
]
PARAMETER_TUNING_METHODS = [
    GridSearchCV,
    RandomizedSearchCV,
    HalvingGridSearchCV,
    HalvingRandomSearchCV,
]


@pytest.mark.parametrize("data_args", DATA_ARGS)
def test_sklearn_cross_validation(data_args):
    """Test sklearn cross-validation works with sktime panel data and classifiers."""
    clf = CanonicalIntervalForest(n_estimators=3, n_intervals=2, att_subsample_size=2)
    fit_args = _make_args(clf, "fit", **data_args)

    scores = cross_val_score(clf, *fit_args, cv=KFold(n_splits=3))
    assert isinstance(scores, np.ndarray)


@pytest.mark.parametrize("data_args", DATA_ARGS)
@pytest.mark.parametrize("cross_validation_method", CROSS_VALIDATION_METHODS)
def test_sklearn_cross_validation_iterators(data_args, cross_validation_method):
    """Test if sklearn cross-validation iterators can handle sktime panel data."""
    clf = CanonicalIntervalForest(n_estimators=3, n_intervals=2, att_subsample_size=2)
    fit_args = _make_args(clf, "fit", **data_args)
    groups = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10]

    for train, test in cross_validation_method.split(*fit_args, groups=groups):
        assert isinstance(train, np.ndarray) and isinstance(test, np.ndarray)


@pytest.mark.parametrize("data_args", DATA_ARGS)
@pytest.mark.parametrize("parameter_tuning_method", PARAMETER_TUNING_METHODS)
def test_sklearn_parameter_tuning(data_args, parameter_tuning_method):
    """Test if sklearn parameter tuners can handle sktime panel data and classifiers."""
    clf = CanonicalIntervalForest(n_estimators=3)
    param_grid = {"n_intervals": [2, 3], "att_subsample_size": [2, 3]}
    fit_args = _make_args(clf, "fit", **data_args)

    parameter_tuning_method = parameter_tuning_method(
        clf, param_grid, cv=KFold(n_splits=3)
    )
    parameter_tuning_method.fit(*fit_args)
    assert isinstance(parameter_tuning_method.best_estimator_, CanonicalIntervalForest)
