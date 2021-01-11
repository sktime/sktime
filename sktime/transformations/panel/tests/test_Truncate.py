# -*- coding: utf-8 -*-
# from sklearn.pipeline import Pipeline
from sktime.datasets.base import _load_dataset
from sktime.transformations.panel.truncation import TruncationTransformer

# from sklearn.ensemble import RandomForestClassifier
from sktime.utils.data_processing import from_nested_to_2d_array

# import pandas as pd


def test_truncation_transformer():
    # load data
    name = "JapaneseVowels"
    X_train, y_train = _load_dataset(name, split="train", return_X_y=True)
    X_test, y_test = _load_dataset(name, split="test", return_X_y=True)

    # print(X_train)

    truncated_transformer = TruncationTransformer(5)
    Xt = truncated_transformer.fit_transform(X_train)

    # when we tabulrize the data it has 12 dimensions
    # and we've truncated them all to 5 long.
    data = from_nested_to_2d_array(Xt)
    assert len(data.columns) == 5 * 12


def test_truncation_paramterised_transformer():
    # load data
    name = "JapaneseVowels"
    X_train, y_train = _load_dataset(name, split="train", return_X_y=True)
    X_test, y_test = _load_dataset(name, split="test", return_X_y=True)

    # print(X_train)

    truncated_transformer = TruncationTransformer(2, 10)
    Xt = truncated_transformer.fit_transform(X_train)

    # when we tabulrize the data it has 12 dimensions
    # and we've truncated them all to (10-2) long.
    data = from_nested_to_2d_array(Xt)
    assert len(data.columns) == 8 * 12
