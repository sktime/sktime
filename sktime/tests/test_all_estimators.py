#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Markus Löning"]
__all__ = ["test_estimator"]

import pytest

from sktime.tests._config import EXCLUDE_ESTIMATORS
from sktime.tests._config import EXCLUDED_TESTS
from sktime.utils import all_estimators
from sktime.utils._testing.estimator_checks import check_estimator

ALL_ESTIMATORS = all_estimators(
    return_names=False, exclude_estimators=EXCLUDE_ESTIMATORS
)


@pytest.mark.parametrize("Estimator", ALL_ESTIMATORS)
def test_estimator(Estimator):
    # We run a number of basic checks on all estimators to ensure correct
    # implementation of our framework and compatibility with scikit-learn.
    check_estimator(Estimator, EXCLUDED_TESTS.get(Estimator.__name__, []))
