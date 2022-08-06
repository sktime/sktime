# -*- coding: utf-8 -*-
"""Tests for contrib data_io."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from pandas._testing import assert_frame_equal

import sktime
from sktime._contrib.data_io import load_from_tsfile
from sktime.datasets._data_io import MODULE


def test_load_from_tsfile():
    """Test function for loading TS formats.

    Test
    1. Univariate equal length (UnitTest) returns 2D numpy X, 1D numpy y
    2. Multivariate equal length (BasicMotions) returns 3D numpy X, 1D numpy y
    3. Univariate and multivariate unequal length (PLAID) return X as DataFrame
    """
    data_path = MODULE + "/data/UnitTest/UnitTest_TRAIN.ts"
    # Test 1.1: load univariate equal length (UnitTest), should return 2D array and 1D
    # array, test first and last data
    # Test 1.2: Load a problem without y values (UnitTest),  test first and last data.
    X, y = load_from_tsfile(data_path)
    assert isinstance(X, np.ndarray) and isinstance(y, np.ndarray)
    assert X.ndim == 2
    assert X.shape == (20, 24) and y.shape == (20,)
    assert X[0][0] == 573.0
    X2 = load_from_tsfile(data_path, return_y=False)
    assert isinstance(X2, np.ndarray)
    assert X2.ndim == 2

    # Test 2: load multivare equal length (BasicMotions), should return 3D array and 1D
    # array, test first and last data.
    data_path = MODULE + "/data/BasicMotions/BasicMotions_TRAIN.ts"
    X, y = load_from_tsfile(data_path)
    assert isinstance(X, np.ndarray) and isinstance(y, np.ndarray)
    assert X.shape == (40, 6, 100) and y.shape == (40,)
    assert X[1][2][3] == -1.898794
    # Test 3.1: load univariate unequal length (PLAID), should return a one column
    # dataframe,
    data_path = MODULE + "/data/PLAID/PLAID_TRAIN.ts"
    X, y = load_from_tsfile(data_path)
    assert isinstance(X, pd.DataFrame) and isinstance(y, np.ndarray)
    assert X.shape == (537, 1) and y.shape == (537,)
    # Test 3.2: load multivariate unequal length (JapaneseVowels), should return a X
    # columns dataframe,
    data_path = MODULE + "/data/JapaneseVowels/JapaneseVowels_TRAIN.ts"
    X, y = load_from_tsfile(data_path)
    assert isinstance(X, pd.DataFrame) and isinstance(y, np.ndarray)
    assert X.shape == (270, 12) and y.shape == (270,)
