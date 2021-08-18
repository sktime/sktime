# -*- coding: utf-8 -*-
# from sklearn.pipeline import Pipeline
"""Test Truncator transformer."""

from sktime.datasets import load_japanese_vowels
from sktime.transformations.panel.truncation import TruncationTransformer

# from sklearn.ensemble import RandomForestClassifier
from sktime.datatypes._panel._convert import from_nested_to_2d_array

# import pandas as pd


def test_truncation_transformer():
    """Test truncation to the shortest series length."""
    # load data
    X_train, y_train = load_japanese_vowels(split="train", return_X_y=True)
    X_test, y_test = load_japanese_vowels(split="test", return_X_y=True)

    # print(X_train)

    truncated_transformer = TruncationTransformer(5)
    Xt = truncated_transformer.fit_transform(X_train)

    # when we tabulrize the data it has 12 dimensions
    # and we've truncated them all to 5 long.
    data = from_nested_to_2d_array(Xt)
    assert len(data.columns) == 5 * 12


def test_truncation_paramterised_transformer():
    """Test truncation to the a user defined length."""
    # load data
    X_train, y_train = load_japanese_vowels(split="train", return_X_y=True)
    X_test, y_test = load_japanese_vowels(split="test", return_X_y=True)

    # print(X_train)

    truncated_transformer = TruncationTransformer(2, 10)
    Xt = truncated_transformer.fit_transform(X_train)

    # when we tabulrize the data it has 12 dimensions
    # and we've truncated them all to (10-2) long.
    data = from_nested_to_2d_array(Xt)
    assert len(data.columns) == 8 * 12
