# -*- coding: utf-8 -*-
"""Tests for base classsifier class."""
__author__ = ["TonyBagnall"]

import numpy as np
import pandas as pd
import pytest

from sktime.utils.validation.panel import (
    _has_nans,
    _nested_dataframe_has_nans,
    _nested_dataframe_has_unequal,
    check_classifier_input,
    get_data_characteristics,
)


def test_check_classifier_input():
    """Test for valid estimator format.

    1. Test correct: X: np.array of 2 and 3 dimensions vs y:np.array and np.Series
    2. Test correct: X: pd.DataFrame with 1 and 3 cols vs y:np.array and np.Series
    3. Test incorrect: X with fewer cases than y
    4. Test incorrect: y as a list
    5. Test incorrect: too few cases or too short a series
    6. todo: test y input.
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
    test_X3 = _create_nested_dataframe(5, 1, 10)
    test_X4 = _create_nested_dataframe(5, 3, 10)
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


def test_get_data_characteristics():
    """Test for correct data query.

    get_data_characteristics tests to see if the data has missing, is multivariate
    and/or is unequal length. There are eight combinations to test for both
    np.ndarray and pd.DataFrame input. Missing detection is also tested in
    test_has_nans and unequal length is tested in test_has_unequal.
    """
    # 1.1 False, False, False. Test no missing, univariate, equal length with array
    test_X1 = np.random.uniform(-1, 1, size=(5, 10))
    missing, multivariate, unequal = get_data_characteristics(test_X1)
    assert missing is False and multivariate is False and unequal is False
    # 1.1 Test no missing, univariate, equal length with DataFrame
    test_X2 = _create_nested_dataframe(5, 1, 10)
    missing, multivariate, unequal = get_data_characteristics(test_X2)
    assert missing is False and multivariate is False and unequal is False
    # 2.1 True, False, False. Test with missing, univariate, equal length with array
    test_X1[0][0] = np.nan
    missing, multivariate, unequal = get_data_characteristics(test_X1)
    assert missing is True and multivariate is False and unequal is False
    # 2.2 Test with missing, univariate, equal length with DataFrame
    test_X2.iloc[0, 0][0] = np.nan
    missing, multivariate, unequal = get_data_characteristics(test_X2)
    assert missing is True and multivariate is False and unequal is False
    # 3.1 False, True, False. Test no missing, multivariate, equal length with array
    test_X1 = np.random.uniform(-1, 1, size=(5, 2, 10))
    missing, multivariate, unequal = get_data_characteristics(test_X1)
    assert missing is False and multivariate is True and unequal is False
    # 3.2 Test no missing, multivariate, equal length with DataFrame
    test_X2 = _create_nested_dataframe(5, 5, 10)
    missing, multivariate, unequal = get_data_characteristics(test_X2)
    assert missing is False and multivariate is True and unequal is False
    # 4.1 True, True, False. Test missing, multivariate, equal length with array
    test_X1[0][0] = np.nan
    missing, multivariate, unequal = get_data_characteristics(test_X1)
    assert missing is True and multivariate is True and unequal is False
    # 4.2 Test missing, multivariate, equal length with DataFrame
    test_X2.iloc[0, 0][0] = np.nan
    missing, multivariate, unequal = get_data_characteristics(test_X2)
    assert missing is True and multivariate is True and unequal is False
    # 5 False, False, True. Test no missing, univariate, unequal length
    test_X3 = _create_unequal_length_nested_dataframe(5, 1, 10)
    missing, multivariate, unequal = get_data_characteristics(test_X3)
    assert missing is False and multivariate is False and unequal is True
    # 6.1 True, False, True. Test no missing, multivariate, unequal length with
    test_X3.iloc[0, 0][0] = np.nan
    missing, multivariate, unequal = get_data_characteristics(test_X3)
    assert missing is True and multivariate is False and unequal is True
    # 7.1 False, True, True. Test no missing, multivariate, unequal with Dataframe
    test_X3 = _create_unequal_length_nested_dataframe(5, 4, 10)
    missing, multivariate, unequal = get_data_characteristics(test_X3)
    assert missing is False and multivariate is True and unequal is True
    # 8.1 Test multivariate, unequal length, missing with Dataframe
    test_X3.iloc[0, 0][0] = np.nan
    missing, multivariate, unequal = get_data_characteristics(test_X3)
    assert missing is True and multivariate is True and unequal is True


def test_has_nans():
    """Test nan checking in arrays and DataFrames."""
    # 1.1 missing with array
    test_X1 = np.random.uniform(-1, 1, size=(5, 10))
    missing = _has_nans(test_X1)
    assert missing is False
    test_X1[4][9] = np.nan
    missing = _has_nans(test_X1)
    assert missing is True
    test_X1[0][0] = np.nan
    missing = _has_nans(test_X1)
    assert missing is True
    test_X2 = _create_nested_dataframe(5, 1, 10)
    test_X3 = _create_nested_dataframe(5, 3, 10)
    missing = _nested_dataframe_has_nans(test_X2)
    assert missing is False
    missing = _nested_dataframe_has_nans(test_X3)
    assert missing is False
    test_X2.iloc[0, 0][0] = np.nan
    missing = _nested_dataframe_has_nans(test_X2)
    assert missing is True
    test_X3.iloc[4, 2][9] = np.nan
    missing = _nested_dataframe_has_nans(test_X3)
    assert missing is True


def test_has_unequal():
    """Test whether unequal length series in a DataFrame are correcty detected."""
    # Equal length.
    test_X1 = _create_nested_dataframe(5, 1, 10)
    test_X2 = _create_nested_dataframe(5, 3, 10)
    unequal = _nested_dataframe_has_unequal(test_X1)
    assert unequal is False
    unequal = _nested_dataframe_has_unequal(test_X2)
    assert unequal is False
    test_X3 = _create_unequal_length_nested_dataframe(5, 1, 10)
    unequal = _nested_dataframe_has_unequal(test_X3)
    assert unequal is True


def _create_nested_dataframe(cases=5, dimensions=1, length=10):
    testy = pd.DataFrame(dtype=np.float32)
    for i in range(0, dimensions):
        instance_list = []
        for _ in range(0, cases):
            instance_list.append(pd.Series(np.random.randn(length)))
        testy["dimension_" + str(i + 1)] = instance_list
    return testy


def _create_unequal_length_nested_dataframe(cases=5, dimensions=1, length=10):
    testy = pd.DataFrame(dtype=np.float32)
    for i in range(0, dimensions):
        instance_list = []
        for _ in range(0, cases - 1):
            instance_list.append(pd.Series(np.random.randn(length)))
        instance_list.append(pd.Series(np.random.randn(length - 1)))
        testy["dimension_" + str(i + 1)] = instance_list

    return testy
