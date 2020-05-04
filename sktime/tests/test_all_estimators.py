#!/usr/bin/env python3 -u
# coding: utf-8

__author__ = ["Markus Löning"]
__all__ = [
]

import pytest
from sktime.utils import all_estimators
from sktime.utils.testing.estimator_checks import check_estimator

ALL_ESTIMATORS = [e[1] for e in all_estimators()]


@pytest.mark.parametrize("Estimator", ALL_ESTIMATORS)
def test_estimator(Estimator):
    check_estimator(Estimator)

