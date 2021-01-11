#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Sebastiaan Koel", "Emilia Rose"]
__all__ = []

import os

import numpy as np
import pandas as pd
import pytest

from sktime.datasets import load_UCR_UEA_dataset
from sktime.datasets import load_arrow_head
from sktime.datasets import load_uschange
from sktime.utils._testing.estimator_checks import _assert_array_almost_equal

_CHECKS = {
    "uschange": {
        "columns": ["Income", "Production", "Savings", "Unemployment"],
        "len_y": 187,
        "len_X": 187,
        "data_types_X": {
            "Income": "float64",
            "Production": "float64",
            "Savings": "float64",
            "Unemployment": "float64",
        },
        "data_type_y": "float64",
        "data": load_uschange(),
    },
}


@pytest.mark.parametrize("dataset", sorted(_CHECKS.keys()))
def test_data_loaders(dataset):
    """
    asserts if datasets are loaded correctly
    ----------
    dataset: dictionary with values to assert against should contain:
        'columns' : list with column names in correct order,
        'len_y'   : lenght of the y series (int),
        'len_X'   : lenght of the X series/dataframe (int),
        'data_types_X' : dictionary with column name keys and dtype as value,
        'data_type_y'  : dtype if y column (string)
        'data'    : tuple with y series and X series/dataframe if one is not
                    applicable fill with None value,
    """
    checks = _CHECKS[dataset]
    y = checks["data"][0]
    X = checks["data"][1]

    if y is not None:
        assert isinstance(y, pd.Series)
        assert len(y) == checks["len_y"]
        assert y.dtype == checks["data_type_y"]

    if X is not None:
        if len(checks["data_types_X"]) > 1:
            assert isinstance(X, pd.DataFrame)
        else:
            assert isinstance(X, pd.Series)

        assert X.columns.values.tolist() == checks["columns"]

        for col, dt in checks["data_types_X"].items():
            assert X[col].dtype == dt

        assert len(X) == checks["len_X"]


def test_load_UCR_UEA_dataset_invalid_dataset():
    with pytest.raises(ValueError, match=r"Invalid dataset name"):
        load_UCR_UEA_dataset("invalid-name")


def test_load_UCR_UEA_dataset_download(tmpdir):
    # tmpdir is a pytest fixture
    extract_path = tmpdir.mkdtemp()
    name = "ArrowHead"
    actual_X, actual_y = load_UCR_UEA_dataset(
        name, return_X_y=True, extract_path=extract_path
    )
    data_path = os.path.join(extract_path, name)
    assert os.path.exists(data_path)

    # check files
    files = [
        f"{name}.txt",
        f"{name}_TEST.arff",
        f"{name}_TEST.ts",
        f"{name}_TEST.txt",
        f"{name}_TRAIN.arff",
        f"{name}_TRAIN.ts",
        f"{name}_TRAIN.txt",
        # "README.md",
    ]

    for file in os.listdir(data_path):
        assert file in files
        files.remove(file)
    assert len(files) == 0

    # check data
    expected_X, expected_y = load_arrow_head(return_X_y=True)
    _assert_array_almost_equal(actual_X, expected_X, decimal=4)
    np.testing.assert_array_equal(expected_y, actual_y)
