# -*- coding: utf-8 -*-
"""Unit tests for classifier base class functionality."""

__author__ = ["mloning", "fkiraly", "TonyBagnall", "MatthewMiddlehurst"]

import numpy as np
import pandas as pd
import pytest

from sktime.classification.base import BaseClassifier


def test_check_capabilities():
    """Test the checking of capabilities.

    There are eight different combinations to be tested with a classifier that can
    handle it and that cannot.
    """
    handles_none = BaseClassifier()

    handles_none.check_capabilities(False, False, False)
    with pytest.raises(ValueError, match=r"The data has missing values"):
        handles_none.check_capabilities(True, True, True)
        handles_none.check_capabilities(True, True, False)
        handles_none.check_capabilities(True, False, False)
        handles_none.check_capabilities(True, False, True)
    with pytest.raises(ValueError, match=r"The data is multivariate"):
        handles_none.check_capabilities(False, True, True)
        handles_none.check_capabilities(False, True, False)
        handles_none.check_capabilities(False, False, True)
    with pytest.raises(ValueError, match=r"The data has unequal length series"):
        handles_none.check_capabilities(False, False, True)

    handles_all = BaseClassifier()
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


def test_convert_input():
    """Test the conversions from dataframe to numpy.

    "coerce-X-to-numpy": True,
    "coerce-X-to-pandas": False,
    1. Pass a 2D numpy X, get a 3D numpy X
    2. Pass a 3D numpy X, get a 3D numpy X
    3. Pass a pandas numpy X, equal length, get a 3D numpy X
    4. Pass a pd.Series y, get a 1S numpy y
        "coerce-X-to-numpy": False,
        "coerce-X-to-pandas": True,
    """
    test_X1 = np.random.uniform(-1, 1, size=(5, 10))
    test_X2 = np.random.uniform(-1, 1, size=(5, 2, 10))
    test_y1 = np.random.randint(0, 1, size=5)
    tester = BaseClassifier()
    tempX, tempy = tester.convert_input(test_X1, test_y1)
    assert tempX.shape[0] == 5 and tempX.shape[1] == 1 and tempX.shape[2] == 10
    tempX, tempy = tester.convert_input(test_X2, test_y1)
    assert tempX.shape[0] == 5 and tempX.shape[1] == 2 and tempX.shape[2] == 10
    instance_list = []
    for i in range(0, 5):
        instance_list.append(pd.Series(np.random.randn(10)))
    test_X3 = pd.DataFrame(dtype=np.float32)
    test_X3["dimension_1"] = instance_list
    test_X4 = pd.DataFrame(dtype=np.float32)
    for i in range(0, 3):
        instance_list = []
        for j in range(0, 5):
            instance_list.append(pd.Series(np.random.randn(10)))
        test_X4["dimension_" + str(i)] = instance_list
    test_y2 = pd.Series(np.random.randn(5))
    tempX, tempy = tester.convert_input(test_X3, test_y1)
    assert tempX.shape[0] == 5 and tempX.shape[1] == 1 and tempX.shape[2] == 10
    tempX, tempy = tester.convert_input(test_X4, test_y1)
    assert tempX.shape[0] == 5 and tempX.shape[1] == 3 and tempX.shape[2] == 10
    tester._tags["coerce-X-to-numpy"] = False
    tester._tags["coerce-X-to-pandas"] = True
    tempX, tempy = tester.convert_input(test_X1, test_y1)
    assert isinstance(tempX, pd.DataFrame)
    assert tempX.shape[0] == 5
    assert tempX.shape[1] == 1
    tempX, tempy = tester.convert_input(test_X2, test_y1)
    assert isinstance(tempX, pd.DataFrame)
    assert tempX.shape[0] == 5
    assert tempX.shape[1] == 2
