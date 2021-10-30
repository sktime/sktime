# -*- coding: utf-8 -*-
"""Test Truncator transformer."""

from sktime.datasets import load_basic_motions
from sktime.transformations.panel.truncation import TruncationTransformer

from sktime.datatypes._panel._convert import from_nested_to_2d_array


def test_truncation_transformer():
    """Test truncation to the shortest series length."""
    # load data
    X_train, y_train = load_basic_motions(split="train", return_X_y=True)

    truncated_transformer = TruncationTransformer(5)
    Xt = truncated_transformer.fit_transform(X_train)

    # when we tabularize the data it has 6 dimensions
    # and we've truncated them all to 5 long.
    data = from_nested_to_2d_array(Xt)
    assert len(data.columns) == 5 * 6


def test_truncation_paramterised_transformer():
    """Test truncation to the a user defined length."""
    # load data
    X_train, y_train = load_basic_motions(split="train", return_X_y=True)

    truncated_transformer = TruncationTransformer(2, 10)
    Xt = truncated_transformer.fit_transform(X_train)

    # when we tabularize the data it has 6 dimensions
    # and we've truncated them all to (10-2) long.
    data = from_nested_to_2d_array(Xt)
    assert len(data.columns) == 8 * 6
