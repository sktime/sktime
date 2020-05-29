#!/usr/bin/env python3 -u
# coding: utf-8

__author__ = ["Markus Löning"]
__all__ = [
    "test_estimator"
]

import pytest
from sktime.tests.config import EXCLUDED
from sktime.utils import all_estimators
from sktime.utils.testing.estimator_checks import check_estimator

ALL_ESTIMATORS = [e[1] for e in all_estimators() if
                  e[0] not in EXCLUDED]


@pytest.mark.parametrize("Estimator", ALL_ESTIMATORS)
def test_estimator(Estimator):
    # We run a number of basic checks on all estimators to ensure correct
    # implementation of our framework and compatibility with scikit-learn
    check_estimator(Estimator)
