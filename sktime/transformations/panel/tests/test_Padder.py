# -*- coding: utf-8 -*-
"""Test the Padder transformer."""

from sktime.datasets import load_basic_motions
from sktime.transformations.panel.padder import PaddingTransformer

from sktime.datatypes._panel._convert import from_nested_to_2d_array


def test_padding_transformer():
    """Test the dimensions after padding."""
    # load data
    X_train, y_train = load_basic_motions(split="train", return_X_y=True)

    padding_transformer = PaddingTransformer()
    Xt = padding_transformer.fit_transform(X_train)

    # when we tabularize the data it has 6 dimensions
    # and we've padded them to there normal length of 100
    data = from_nested_to_2d_array(Xt)
    assert len(data.columns) == 100 * 6


def test_padding_parameterised_transformer():
    """Test padding to user determined length."""
    # load data
    X_train, y_train = load_basic_motions(split="train", return_X_y=True)

    padding_transformer = PaddingTransformer(pad_length=120)
    Xt = padding_transformer.fit_transform(X_train)

    # when we tabularize the data it has 6 dimensions
    # and we've padded them all to 120 long.
    data = from_nested_to_2d_array(Xt)
    assert len(data.columns) == 120 * 6


def test_padding_fill_value_transformer():
    """Test full fill padding."""
    # load data
    X_train, y_train = load_basic_motions(split="train", return_X_y=True)

    padding_transformer = PaddingTransformer(pad_length=120, fill_value=1)
    Xt = padding_transformer.fit_transform(X_train)

    # when we tabularize the data it has 6 dimensions
    # and we've padded them all to 120 long.
    data = from_nested_to_2d_array(Xt)
    assert len(data.columns) == 120 * 6
