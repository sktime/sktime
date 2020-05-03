#!/usr/bin/env python3 -u
# coding: utf-8

__author__ = ["Markus Löning"]
__all__ = []

import pytest
from sklearn.base import clone
from sktime.utils import all_estimators
from sktime.utils.testing.construct import _construct_instance

ALL_ESTIMATORS = [e[1] for e in all_estimators()]


@pytest.mark.parametrize("Estimator", ALL_ESTIMATORS)
def test_basic_interface(Estimator):
    estimator = _construct_instance(Estimator)
    assert hasattr(estimator, "fit")
    assert hasattr(estimator, "set_params")
    assert hasattr(estimator, "get_params")


@pytest.mark.parametrize("Estimator", ALL_ESTIMATORS)
def test_params_set_get(Estimator):
    estimator = _construct_instance(Estimator)

    # check get params
    params = estimator.get_params()
    assert isinstance(params, dict)

    # check set params returns self
    assert estimator.set_params(**params) is estimator


@pytest.mark.parametrize("Estimator", ALL_ESTIMATORS)
def test_clone(Estimator):
    estimator = _construct_instance(Estimator)
    clone(estimator)
