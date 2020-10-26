#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-

__author__ = ["Markus Löning"]
__all__ = []

import pytest

from sktime.tests._config import EXCLUDE_ESTIMATORS
from sktime.tests._config import VALID_ESTIMATOR_BASE_TYPE_LOOKUP
from sktime.tests._config import VALID_ESTIMATOR_TYPES
from sktime.utils import all_estimators


@pytest.mark.parametrize("return_names", [True, False])
def test_all_estimators_return_names(return_names):
    estimators = all_estimators(return_names=return_names)
    assert isinstance(estimators, list)
    assert len(estimators) > 0

    if return_names:
        assert all([isinstance(estimator, tuple) for estimator in estimators])
        names, estimators = list(zip(*estimators))
        assert all([isinstance(name, str) for name in names])
        assert all(
            [name == estimator.__name__ for name, estimator in zip(names, estimators)]
        )

    assert all([isinstance(estimator, type) for estimator in estimators])


@pytest.mark.parametrize("exclude_estimators", ["NaiveForecaster", EXCLUDE_ESTIMATORS])
def test_all_estimators_exclude_estimators(exclude_estimators):
    estimators = all_estimators(
        return_names=True, exclude_estimators=exclude_estimators
    )
    assert isinstance(estimators, list)
    assert len(estimators) > 0
    names, estimators = list(zip(*estimators))

    if not isinstance(exclude_estimators, list):
        exclude_estimators = [exclude_estimators]
    for estimator in exclude_estimators:
        assert estimator not in names


@pytest.mark.parametrize(
    "estimator_type", [*VALID_ESTIMATOR_BASE_TYPE_LOOKUP.keys(), *VALID_ESTIMATOR_TYPES]
)
def test_all_estimators_filter_estimator_types(estimator_type):
    estimators = all_estimators(estimator_types=estimator_type, return_names=False)
    assert isinstance(estimators, list)
    # assert len(estimators) > 0

    if isinstance(estimator_type, str):
        estimator_type = VALID_ESTIMATOR_BASE_TYPE_LOOKUP[estimator_type]

    assert all([issubclass(estimator, estimator_type) for estimator in estimators])
