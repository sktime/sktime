#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for MultiplexTransformer and associated dunders."""

__author__ = ["miraep8"]

from sklearn.base import clone

from sktime.datasets import load_shampoo_sales
from sktime.transformations.multiplexer import MultiplexTransformer
from sktime.transformations.series.impute import Imputer


def test_multiplex_transformer_alone():
    """Test behavior of MultiplexTransformer.

    Because MultiplexTransformer is in many ways a wrapper for an underlying
    transformer - we can confirm that if the selected_transformer is set that the
    MultiplexTransformer delegates all its transformation responsibilities as expected.
    """
    from numpy.testing import assert_array_equal

    y = load_shampoo_sales()
    # Note - we select two forecasters which are deterministic.
    transformer_tuples = [
        ("mean", Imputer(method="mean")),
        ("nearest", Imputer(method="nearest")),
    ]
    transformer_names = [name for name, _ in transformer_tuples]
    transformers = [transformer for _, transformer in transformer_tuples]
    multiplex_transformer = MultiplexTransformer(transformers=transformer_tuples)
    # for each of the forecasters - check that the wrapped forecaster predictions
    # agree with the unwrapped forecaster predictions!
    for ind, name in enumerate(transformer_names):
        # make a copy to ensure we don't reference the same objectL
        test_transformer = clone(transformers[ind])
        y_transform_indiv = test_transformer.fit_transform(y)
        multiplex_transformer.selected_transformer = name
        # Note- MultiplexForecaster will make a copy of the forecaster before fitting.
        y_transform_multi = multiplex_transformer.fit_transform(y)
        assert_array_equal(y_transform_indiv, y_transform_multi)
