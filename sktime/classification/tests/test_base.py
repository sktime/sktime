# -*- coding: utf-8 -*-
"""Unit tests for classifier base class functionality."""

__author__ = ["mloning", "fkiraly", "TonyBagnall", "MatthewMiddlehurst"]

import numpy as np
import pandas as pd
import pytest

from sktime.classification.base import (
    BaseClassifier,
    _check_classifier_input,
    _get_data_characteristics,
    _has_nans,
    _internal_convert,
    _nested_dataframe_has_nans,
    _nested_dataframe_has_unequal,
)


class _DummyClassifier(BaseClassifier):
    """Dummy classifier for testing base class fit/predict/predict_proba."""

    def _fit(self, X, y):
        """Fit dummy."""
        return self

    def _predict(self, X):
        """Predict dummy."""
        return self

    def _predict_proba(self):
        """Predict proba dummy."""
        return self


class _DummyHandlesAllInput(BaseClassifier):
    """Dummy classifier for testing base class fit/predict/predict_proba."""

    _tags = {
        "capability:multivariate": True,
        "capability:unequal_length": True,
        "capability:missing_values": True,
    }

    def _fit(self, X, y):
        """Fit dummy."""
        return self

    def _predict(self, X):
        """Predict dummy."""
        return self

    def _predict_proba(self):
        """Predict proba dummy."""
        return self


class _DummyConvertPandas(BaseClassifier):
    """Dummy classifier for testing base class fit/predict/predict_proba."""

    _tags = {
        "X_inner_mtype": "nested_univ",  # which type do _fit/_predict, support for X?
    }

    def _fit(self, X, y):
        """Fit dummy."""
        return self

    def _predict(self, X):
        """Predict dummy."""
        return self

    def _predict_proba(self):
        """Predict proba dummy."""
        return self


multivariate_message = r"X must be univariate, this classifier cannot deal with"
missing_message = r"The data has missing values"
unequal_message = r"The data has unequal length series"
incorrect_X_data_structure = r"must be a np.array or a pd.Series"
incorrect_y_data_structure = r"must be 1-dimensional"


def test_base_classifier_fit():
    """Test function for the BaseClassifier class fit.

    Test fit. It should:
    1. Work with 2D, 3D and DataFrame for X and nparray for y.
    2. Calculate the number of classes and record the fit time.
    3. have self.n_jobs set or throw  an exception if the classifier can
    multithread.
    4. Set the class dictionary correctly.
    5. Set is_fitted after a call to _fit.
    6. Return self.
    """
    dummy = _DummyClassifier()
    cases = 5
    length = 10
    test_X1 = np.random.uniform(-1, 1, size=(cases, length))
    test_X2 = np.random.uniform(-1, 1, size=(cases, 2, length))
    test_X3 = _create_example_dataframe(cases=cases, dimensions=1, length=length)
    test_X4 = _create_example_dataframe(cases=cases, dimensions=3, length=length)
    test_y1 = np.random.randint(0, 2, size=(cases))
    result = dummy.fit(test_X1, test_y1)
    assert result is dummy
    with pytest.raises(ValueError, match=multivariate_message):
        result = dummy.fit(test_X2, test_y1)
    assert result is dummy
    result = dummy.fit(test_X3, test_y1)
    assert result is dummy
    with pytest.raises(ValueError, match=multivariate_message):
        result = dummy.fit(test_X4, test_y1)
    assert result is dummy
    # Raise a specific error if y is in a 2D matrix (1,cases)?
    test_y2 = np.array([test_y1])
    # What if y is in a 2D matrix (cases,1)?
    test_y2 = np.array([test_y1]).transpose()
    with pytest.raises(ValueError, match=incorrect_y_data_structure):
        result = dummy.fit(test_X1, test_y2)
    # Pass a data fram
    with pytest.raises(ValueError, match=incorrect_X_data_structure):
        result = dummy.fit(test_X1, test_X3)


def test_check_capabilities():
    """Test the checking of capabilities.

    There are eight different combinations to be tested with a classifier that can
    handle it and that cannot. I need to rewrite this to stop setting the tags
    directly.
    """
    handles_none = _DummyClassifier()

    handles_none._check_capabilities(False, False, False)
    with pytest.raises(ValueError, match=missing_message):
        handles_none._check_capabilities(True, True, True)
        handles_none._check_capabilities(True, True, False)
        handles_none._check_capabilities(True, False, False)
        handles_none._check_capabilities(True, False, True)
    with pytest.raises(ValueError, match=multivariate_message):
        handles_none._check_capabilities(False, True, True)
        handles_none._check_capabilities(False, True, False)
        handles_none._check_capabilities(False, False, True)
    with pytest.raises(ValueError, match=unequal_message):
        handles_none._check_capabilities(False, False, True)

    handles_all = _DummyHandlesAllInput()
    handles_all._check_capabilities(False, False, False)
    handles_all._check_capabilities(False, False, False)
    handles_all._check_capabilities(True, True, True)
    handles_all._check_capabilities(True, True, False)
    handles_all._check_capabilities(True, False, True)
    handles_all._check_capabilities(False, True, True)
    handles_all._check_capabilities(True, False, False)
    handles_all._check_capabilities(False, True, False)
    handles_all._check_capabilities(False, False, True)
    handles_all._check_capabilities(False, False, False)


def test_convert_input():
    """Test the conversions from dataframe to numpy.

    "convert_X_to_numpy": True,
    "convert_X_to_dataframe": False,
    1. Pass a 2D numpy X, get a 3D numpy X
    2. Pass a 3D numpy X, get a 3D numpy X
    3. Pass a pandas numpy X, equal length, get a 3D numpy X
    4. Pass a pd.Series y, get a pd.Series back
    5. Pass a np.ndarray y, get a pd.Series back
    """
    cases = 5
    length = 10
    test_X1 = np.random.uniform(-1, 1, size=(cases, length))
    test_X2 = np.random.uniform(-1, 1, size=(cases, 2, length))
    tester = _DummyClassifier()
    tempX = tester._convert_X(test_X2)
    assert tempX.shape[0] == cases and tempX.shape[1] == 2 and tempX.shape[2] == length
    instance_list = []
    for _ in range(0, cases):
        instance_list.append(pd.Series(np.random.randn(10)))
    test_X3 = _create_example_dataframe(cases=cases, dimensions=1, length=length)
    test_X4 = _create_example_dataframe(cases=cases, dimensions=3, length=length)
    tempX = tester._convert_X(test_X3)
    assert tempX.shape[0] == cases and tempX.shape[1] == 1 and tempX.shape[2] == length
    tempX = tester._convert_X(test_X4)
    assert tempX.shape[0] == cases and tempX.shape[1] == 3 and tempX.shape[2] == length
    tester = _DummyConvertPandas()
    tempX = tester._convert_X(test_X2)
    assert isinstance(tempX, pd.DataFrame)
    assert tempX.shape[0] == cases
    assert tempX.shape[1] == 2
    test_y1 = np.random.randint(0, 1, size=(cases))
    test_y1 = pd.Series(test_y1)
    tempX, tempY = _internal_convert(test_X1, test_y1)
    assert isinstance(tempY, np.ndarray)
    assert isinstance(tempX, np.ndarray)
    assert tempX.ndim == 3


def _create_example_dataframe(cases=5, dimensions=1, length=10):
    """Create a simple data frame set of time series (X) for testing."""
    test_X = pd.DataFrame(dtype=np.float32)
    for i in range(0, dimensions):
        instance_list = []
        for _ in range(0, cases):
            instance_list.append(pd.Series(np.random.randn(length)))
        test_X["dimension_" + str(i)] = instance_list
    return test_X


def test__check_classifier_input():
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
    _check_classifier_input(test_X1)
    _check_classifier_input(test_X2)
    _check_classifier_input(test_X1, test_y1)
    _check_classifier_input(test_X2, test_y1)
    _check_classifier_input(test_X1, test_y2)
    _check_classifier_input(test_X2, test_y2)
    # 2. Test correct: X: pd.DataFrame with 1 (univariate) and 3 cols(multivariate) vs
    # y:np.array and np.Series
    test_X3 = _create_nested_dataframe(5, 1, 10)
    test_X4 = _create_nested_dataframe(5, 3, 10)
    _check_classifier_input(test_X3, test_y1)
    _check_classifier_input(test_X4, test_y1)
    _check_classifier_input(test_X3, test_y2)
    _check_classifier_input(test_X4, test_y2)
    # 3. Test incorrect: X with fewer cases than y
    test_X5 = np.random.uniform(-1, 1, size=(3, 4, 10))
    with pytest.raises(ValueError, match=r".*Mismatch in number of cases*."):
        _check_classifier_input(test_X5, test_y1)
    # 4. Test incorrect data type: y is a List
    test_y3 = [1, 2, 3, 4, 5]
    with pytest.raises(ValueError, match=r".*y must be a np.array or a pd.Series*."):
        _check_classifier_input(test_X1, test_y3)
    # 5. Test incorrect: too few cases or too short a series
    with pytest.raises(ValueError, match=r".*Minimum number of cases required*."):
        _check_classifier_input(test_X1, test_y1, enforce_min_instances=6)
    with pytest.raises(ValueError, match=r".*Series length below the minimum*."):
        _check_classifier_input(test_X1, test_y1, enforce_min_series_length=11)


def test_get_data_characteristics():
    """Test for correct data query.

    _get_data_characteristics tests to see if the data has missing, is multivariate
    and/or is unequal length. There are eight combinations to test for both
    np.ndarray and pd.DataFrame input. Missing detection is also tested in
    test_has_nans and unequal length is tested in test_has_unequal.
    """
    # 1.1 False, False, False. Test no missing, univariate, equal length with array
    test_X1 = np.random.uniform(-1, 1, size=(5, 10))
    missing, multivariate, unequal = _get_data_characteristics(test_X1)
    assert missing is False and multivariate is False and unequal is False
    # 1.1 Test no missing, univariate, equal length with DataFrame
    test_X2 = _create_nested_dataframe(5, 1, 10)
    missing, multivariate, unequal = _get_data_characteristics(test_X2)
    assert missing is False and multivariate is False and unequal is False
    # 2.1 True, False, False. Test with missing, univariate, equal length with array
    test_X1[0][0] = np.nan
    missing, multivariate, unequal = _get_data_characteristics(test_X1)
    assert missing is True and multivariate is False and unequal is False
    # 2.2 Test with missing, univariate, equal length with DataFrame
    test_X2.iloc[0, 0][0] = np.nan
    missing, multivariate, unequal = _get_data_characteristics(test_X2)
    assert missing is True and multivariate is False and unequal is False
    # 3.1 False, True, False. Test no missing, multivariate, equal length with array
    test_X1 = np.random.uniform(-1, 1, size=(5, 2, 10))
    missing, multivariate, unequal = _get_data_characteristics(test_X1)
    assert missing is False and multivariate is True and unequal is False
    # 3.2 Test no missing, multivariate, equal length with DataFrame
    test_X2 = _create_nested_dataframe(5, 5, 10)
    missing, multivariate, unequal = _get_data_characteristics(test_X2)
    assert missing is False and multivariate is True and unequal is False
    # 4.1 True, True, False. Test missing, multivariate, equal length with array
    test_X1[0][0] = np.nan
    missing, multivariate, unequal = _get_data_characteristics(test_X1)
    assert missing is True and multivariate is True and unequal is False
    # 4.2 Test missing, multivariate, equal length with DataFrame
    test_X2.iloc[0, 0][0] = np.nan
    missing, multivariate, unequal = _get_data_characteristics(test_X2)
    assert missing is True and multivariate is True and unequal is False
    # 5 False, False, True. Test no missing, univariate, unequal length
    test_X3 = _create_unequal_length_nested_dataframe(5, 1, 10)
    missing, multivariate, unequal = _get_data_characteristics(test_X3)
    assert missing is False and multivariate is False and unequal is True
    # 6.1 True, False, True. Test no missing, multivariate, unequal length with
    test_X3.iloc[0, 0][0] = np.nan
    missing, multivariate, unequal = _get_data_characteristics(test_X3)
    assert missing is True and multivariate is False and unequal is True
    # 7.1 False, True, True. Test no missing, multivariate, unequal with Dataframe
    test_X3 = _create_unequal_length_nested_dataframe(5, 4, 10)
    missing, multivariate, unequal = _get_data_characteristics(test_X3)
    assert missing is False and multivariate is True and unequal is True
    # 8.1 Test multivariate, unequal length, missing with Dataframe
    test_X3.iloc[0, 0][0] = np.nan
    missing, multivariate, unequal = _get_data_characteristics(test_X3)
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
