# -*- coding: utf-8 -*-
"""Unit tests for classifier base class functionality."""

__author__ = ["mloning", "fkiraly", "TonyBagnall", "MatthewMiddlehurst"]

import numpy as np
import pandas as pd
import pytest

from sktime.classification.base import BaseClassifier


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

    handles_none.check_capabilities(False, False, False)
    with pytest.raises(ValueError, match=missing_message):
        handles_none.check_capabilities(True, True, True)
        handles_none.check_capabilities(True, True, False)
        handles_none.check_capabilities(True, False, False)
        handles_none.check_capabilities(True, False, True)
    with pytest.raises(ValueError, match=multivariate_message):
        handles_none.check_capabilities(False, True, True)
        handles_none.check_capabilities(False, True, False)
        handles_none.check_capabilities(False, False, True)
    with pytest.raises(ValueError, match=unequal_message):
        handles_none.check_capabilities(False, False, True)

    handles_all = _DummyClassifier()
    handles_all._tags["capability:multivariate"] = True
    handles_all._tags["capability:unequal_length"] = True
    handles_all._tags["capability:missing_values"] = True
    handles_all.check_capabilities(False, False, False)
    handles_all.check_capabilities(False, False, False)
    handles_all.check_capabilities(True, True, True)
    handles_all.check_capabilities(True, True, False)
    handles_all.check_capabilities(True, False, True)
    handles_all.check_capabilities(False, True, True)
    handles_all.check_capabilities(True, False, False)
    handles_all.check_capabilities(False, True, False)
    handles_all.check_capabilities(False, False, True)
    handles_all.check_capabilities(False, False, False)
    handles_all._tags["capability:multivariate"] = False
    handles_all._tags["capability:unequal_length"] = False
    handles_all._tags["capability:missing_values"] = False


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
    tempX = tester.convert_X(test_X1)
    assert tempX.shape[0] == cases and tempX.shape[1] == 1 and tempX.shape[2] == length
    tempX = tester.convert_X(test_X2)
    assert tempX.shape[0] == cases and tempX.shape[1] == 2 and tempX.shape[2] == length
    instance_list = []
    for _ in range(0, cases):
        instance_list.append(pd.Series(np.random.randn(10)))
    test_X3 = _create_example_dataframe(cases=cases, dimensions=1, length=length)
    test_X4 = _create_example_dataframe(cases=cases, dimensions=3, length=length)
    tempX = tester.convert_X(test_X3)
    assert tempX.shape[0] == cases and tempX.shape[1] == 1 and tempX.shape[2] == length
    tempX = tester.convert_X(test_X4)
    assert tempX.shape[0] == cases and tempX.shape[1] == 3 and tempX.shape[2] == length
    tester._tags["convert_X_to_numpy"] = False
    tester._tags["convert_X_to_dataframe"] = True
    tempX = tester.convert_X(test_X1)
    assert isinstance(tempX, pd.DataFrame)
    assert isinstance(tempX, pd.DataFrame)
    assert tempX.shape[0] == cases
    assert tempX.shape[1] == 1
    tempX = tester.convert_X(test_X2)
    assert isinstance(tempX, pd.DataFrame)
    assert tempX.shape[0] == cases
    assert tempX.shape[1] == 2
    test_y1 = np.random.randint(0, 1, size=(cases))
    test_y2 = pd.Series(test_y1)
    tempY = tester.convert_y(test_y1)
    assert isinstance(tempY, np.ndarray)
    tempY = tester.convert_y(test_y2)
    assert isinstance(tempY, np.ndarray)
    tester._tags["convert_X_to_numpy"] = True
    tester._tags["convert_X_to_dataframe"] = False


def _create_example_dataframe(cases=5, dimensions=1, length=10):
    """Create a simple data frame set of time series (X) for testing."""
    test_X = pd.DataFrame(dtype=np.float32)
    for i in range(0, dimensions):
        instance_list = []
        for _ in range(0, cases):
            instance_list.append(pd.Series(np.random.randn(length)))
        test_X["dimension_" + str(i)] = instance_list
    return test_X
