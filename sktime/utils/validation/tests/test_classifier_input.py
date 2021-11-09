# -*- coding: utf-8 -*-
"""Tests for base classsifier class."""
__author__ = ["TonyBagnall"]

import numpy as np
import pandas as pd
import pytest

from sktime.utils.validation.panel import check_classifier_input


def test_check_classifier_input():
    """Test for valid estimator format.

    1. Test correct: X: np.array of 2 and 3 dimensions vs y:np.array and np.Series
    2. Test correct: X: pd.DataFrame with 1 and 3 cols vs y:np.array and np.Series
    3. Test incorrect: X with fewer cases than y
    4. Test incorrect: y as a list
    5. Test incorrect: too few cases or too short a series
    """
    # 1. Test correct: X: np.array of 2 and 3 dimensions vs y:np.array and np.Series
    test_X1 = np.random.uniform(-1, 1, size=(5, 10))
    test_X2 = np.random.uniform(-1, 1, size=(5, 2, 10))
    test_y1 = np.random.randint(0, 1, size=5)
    test_y2 = pd.Series(np.random.randn(5))
    check_classifier_input(test_X1)
    check_classifier_input(test_X2)
    check_classifier_input(test_X1, test_y1)
    check_classifier_input(test_X2, test_y1)
    check_classifier_input(test_X1, test_y2)
    check_classifier_input(test_X2, test_y2)
    # 2. Test correct: X: pd.DataFrame with 1 (univariate) and 3 cols(multivariate) vs
    # y:np.array and np.Series
    instance_list = []
    for _ in range(0, 5):
        instance_list.append(pd.Series(np.random.randn(10)))
    test_X3 = pd.DataFrame(dtype=np.float32)
    test_X3["dimension_1"] = instance_list
    test_X4 = pd.DataFrame(dtype=np.float32)
    for i in range(0, 3):
        instance_list = []
        for _ in range(0, 5):
            instance_list.append(pd.Series(np.random.randn(10)))
        test_X4["dimension_" + str(i)] = instance_list
    check_classifier_input(test_X3, test_y1)
    check_classifier_input(test_X4, test_y1)
    check_classifier_input(test_X3, test_y2)
    check_classifier_input(test_X4, test_y2)
    # 3. Test incorrect: X with fewer cases than y
    test_X5 = np.random.uniform(-1, 1, size=(3, 4, 10))
    with pytest.raises(ValueError, match=r".*Mismatch in number of cases*."):
        check_classifier_input(test_X5, test_y1)
    # 4. Test incorrect data type: y is a List
    test_y3 = [1, 2, 3, 4, 5]
    with pytest.raises(ValueError, match=r".*y must be a np.array or a pd.Series*."):
        check_classifier_input(test_X1, test_y3)
    # 5. Test incorrect: too few cases or too short a series
    with pytest.raises(ValueError, match=r".*Minimum number of cases required*."):
        check_classifier_input(test_X1, test_y1, enforce_min_instances=6)
    with pytest.raises(ValueError, match=r".*Series length below the minimum*."):
        check_classifier_input(test_X1, test_y1, enforce_min_series_length=11)
