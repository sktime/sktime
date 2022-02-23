#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Unit tests for Bootrstapping transformers."""

__author__ = ["ltsaprounis"]

import pandas as pd
import pytest

from sktime.datasets import load_airline
from sktime.transformations.bootstrap import BootsrappingTransformer

y = load_airline()
y_index = y.index


def test_bootstrapping_transformer_no_seasonal_period():
    """Tests that an exception is raised if sp<2."""
    with pytest.raises(NotImplementedError) as ex:
        transformer = BootsrappingTransformer(sp=1)
        transformer.fit(y)

        assert "BootstrappingTransformer does not support non-seasonal data" == ex.value


def test_bootstrapping_transformer_series_shorter_than_sp():
    """Tests that an exception is raised if sp>len(y)."""
    with pytest.raises(ValueError) as ex:
        transformer = BootsrappingTransformer(sp=12)
        transformer.fit(y.iloc[1:9])

        msg = (
            "BootstrappingTransformer requires that sp is greater than the length of X"
        )

        assert msg == ex.value


@pytest.mark.parametrize(
    "series_name, return_actual, expected_index",
    [
        (
            None,
            True,
            pd.MultiIndex.from_product(
                [["actual", "synthetic_0", "synthetic_1"], y_index]
            ),
        ),
        (
            None,
            False,
            pd.MultiIndex.from_product([["synthetic_0", "synthetic_1"], y_index]),
        ),
        (
            "test",
            True,
            pd.MultiIndex.from_product(
                [["test_actual", "test_synthetic_0", "test_synthetic_1"], y_index]
            ),
        ),
    ],
)
def test_bootstrapping_transformer_panel_format(
    series_name, return_actual, expected_index
):
    """Tests that the final panel has the right index."""
    transformer = BootsrappingTransformer(
        n_series=2, sp=12, return_actual=return_actual, series_name=series_name
    )
    y_hat = transformer.fit_transform(y)
    assert expected_index.equals(y_hat.index)


@pytest.mark.parametrize("block_length", [1, 5])
def test_moving_block_bootstrap(block_length):
    """Tests for the BootsrappingTransformer._moving_block_bootstrap.

    1. the output series has the same index as the input
    2. basic checks for the distribution of the bootstrapped values
       i.e. actual min/max >= bootstapped min/max
    """
    y_hat = BootsrappingTransformer._moving_block_bootstrap(
        y, block_length=block_length
    )
    assert (
        y_hat.index.equals(y_index)
        & (y_hat.max() <= y.max())
        & (y_hat.min() >= y.min())
    )
