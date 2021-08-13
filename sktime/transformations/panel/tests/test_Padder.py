# -*- coding: utf-8 -*-
"""Test the Padder transformer."""

from sktime.datasets import load_japanese_vowels
from sktime.transformations.panel.padder import PaddingTransformer

# from sklearn.ensemble import RandomForestClassifier
from sktime.datatypes._panel._convert import from_nested_to_2d_array

# import pandas as pd


def test_padding_transformer():
    """Test the dimensions after padding."""
    # load data
    X_train, y_train = load_japanese_vowels(split="train", return_X_y=True)
    X_test, y_test = load_japanese_vowels(split="test", return_X_y=True)

    padding_transformer = PaddingTransformer()
    Xt = padding_transformer.fit_transform(X_train)

    # when we tabulrize the data it has 12 dimensions
    # and we've padded them to there normal length of 29
    data = from_nested_to_2d_array(Xt)
    assert len(data.columns) == 29 * 12


def test_padding_parameterised_transformer():
    """Test padding to user determined length."""
    # load data
    X_train, y_train = load_japanese_vowels(split="train", return_X_y=True)
    X_test, y_test = load_japanese_vowels(split="test", return_X_y=True)

    # print(X_train)

    padding_transformer = PaddingTransformer(pad_length=40)
    Xt = padding_transformer.fit_transform(X_train)

    # when we tabulrize the data it has 12 dimensions
    # and we've truncated them all to (10-2) long.
    data = from_nested_to_2d_array(Xt)
    assert len(data.columns) == 40 * 12


def test_padding_fill_value_transformer():
    """Test full fill padding."""
    # load data
    X_train, y_train = load_japanese_vowels(split="train", return_X_y=True)
    X_test, y_test = load_japanese_vowels(split="test", return_X_y=True)

    # print(X_train)

    padding_transformer = PaddingTransformer(pad_length=40, fill_value=1)
    Xt = padding_transformer.fit_transform(X_train)

    # when we tabulrize the data it has 12 dimensions
    # and we've truncated them all to (10-2) long.
    data = from_nested_to_2d_array(Xt)
    assert len(data.columns) == 40 * 12
