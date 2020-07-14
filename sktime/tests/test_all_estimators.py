#!/usr/bin/env python3 -u
# coding: utf-8
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Markus Löning"]
__all__ = [
    "test_estimator"
]

import pytest
from sktime.tests._config import EXCLUDED_ESTIMATORS
from sktime.utils import all_estimators
from sktime.utils._testing.estimator_checks import check_estimator
from sktime.tests._config import EXCLUDED_TESTS

ALL_ESTIMATORS = [e[1] for e in all_estimators() if
                  e[0] not in EXCLUDED_ESTIMATORS]


@pytest.mark.parametrize("Estimator", ALL_ESTIMATORS)
def test_estimator(Estimator):
    # We run a number of basic checks on all estimators to ensure correct
    # implementation of our framework and compatibility with scikit-learn
    check_estimator(Estimator, EXCLUDED_TESTS.get(Estimator.__name__, []))
