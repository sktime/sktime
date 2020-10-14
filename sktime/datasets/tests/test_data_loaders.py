#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Sebastiaan Koel", "Emilia Rose"]
__all__ = []

import pytest
import pandas as pd
import shutil
import tempfile
import os
from sktime.datasets import load_uschange, load_UCR_UEA_dataset

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


def test_dataset_downloading():
    """
    Asserts if datasets are downloaded correctly and load correctly

    Returns
    -------
    None.

    """
    with pytest.raises(ValueError):
        load_UCR_UEA_dataset("Chinatown1")

    test_dir = tempfile.mkdtemp()
    chinatown = load_UCR_UEA_dataset(
        "Chinatown", extract_path=test_dir, return_X_y=(True)
    )

    assert os.path.exists(os.path.join(test_dir, "Chinatown"))

    subfile_names = [
        "Chinatown.txt",
        "Chinatown_TEST.arff",
        "Chinatown_TEST.ts",
        "Chinatown_TEST.txt",
        "Chinatown_TRAIN.arff",
        "Chinatown_TRAIN.ts",
        "Chinatown_TRAIN.txt",
        "README.md",
    ]

    for path in os.listdir(os.path.join(test_dir, "Chinatown")):
        assert path in subfile_names
        subfile_names.remove(path)
    assert len(subfile_names) == 0

    assert (
        len(open(os.path.join(test_dir, "Chinatown", "Chinatown_TEST.txt")).readlines())
        == 343
    )

    assert len(chinatown[0]) == 363
    assert len(chinatown[1]) == 363

    shutil.rmtree(test_dir)
