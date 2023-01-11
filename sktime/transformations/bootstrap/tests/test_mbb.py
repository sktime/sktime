#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Unit tests for Bootrstapping transformers."""

__author__ = ["ltsaprounis"]

import pandas as pd
import pytest

from sktime.datasets import load_airline
from sktime.transformations.bootstrap import (
    MovingBlockBootstrapTransformer,
    STLBootstrapTransformer,
)
from sktime.transformations.bootstrap._mbb import (
    _get_series_name,
    _moving_block_bootstrap,
)
from sktime.utils.validation._dependencies import _check_soft_dependencies

y = load_airline()
y_index = y.index


@pytest.mark.skipif(
    not _check_soft_dependencies("statsmodels", severity="none"),
    reason="skip test if required soft dependency for hmmlearn not available",
)
def test_bootstrapping_transformer_no_seasonal_period():
    """Tests that an exception is raised if sp<2."""
    with pytest.raises(NotImplementedError) as ex:
        transformer = STLBootstrapTransformer(sp=1)
        transformer.fit(y)

        assert "STLBootstrapTransformer does not support non-seasonal data" == ex.value


@pytest.mark.skipif(
    not _check_soft_dependencies("statsmodels", severity="none"),
    reason="skip test if required soft dependency for hmmlearn not available",
)
def test_bootstrapping_transformer_series_shorter_than_sp():
    """Tests that an exception is raised if sp>len(y)."""
    with pytest.raises(ValueError) as ex:
        transformer = STLBootstrapTransformer(sp=12)
        transformer.fit(y.iloc[1:9])

        msg = "STLBootstrapTransformer requires that sp is greater than the length of X"

        assert msg == ex.value


@pytest.mark.skipif(
    not _check_soft_dependencies("statsmodels", severity="none"),
    reason="skip test if required soft dependency for hmmlearn not available",
)
@pytest.mark.parametrize(
    "transformer_class", [STLBootstrapTransformer, MovingBlockBootstrapTransformer]
)
def test_block_length_exception(transformer_class):
    """Tests that a Value error is raised when block_length is smaller than len(X)."""
    msg = (
        f"{transformer_class.__name__} requires that block_length"
        " is greater than the length of X"
    )
    with pytest.raises(ValueError) as ex:
        transformer = transformer_class(block_length=12)
        transformer.fit_transform(y.iloc[1:9])

        assert msg == ex.value


index_return_actual_true = pd.MultiIndex.from_product(
    [["actual", "synthetic_0", "synthetic_1"], y_index]
)
index_return_actual_false = pd.MultiIndex.from_product(
    [["synthetic_0", "synthetic_1"], y_index]
)


@pytest.mark.skipif(
    not _check_soft_dependencies("statsmodels", severity="none"),
    reason="skip test if required soft dependency for hmmlearn not available",
)
@pytest.mark.parametrize(
    "transformer_class, return_actual, expected_index",
    [
        (
            MovingBlockBootstrapTransformer,
            True,
            index_return_actual_true,
        ),
        (
            STLBootstrapTransformer,
            True,
            index_return_actual_true,
        ),
        (
            MovingBlockBootstrapTransformer,
            False,
            index_return_actual_false,
        ),
        (
            STLBootstrapTransformer,
            False,
            index_return_actual_false,
        ),
    ],
)
def test_bootstrap_transformers_panel_format(
    transformer_class, return_actual, expected_index
):
    """Tests that the final panel has the right index."""
    transformer = transformer_class(n_series=2, return_actual=return_actual)
    y_hat = transformer.fit_transform(y)
    assert expected_index.equals(y_hat.index) and (y_hat.columns[0] == y.name)


@pytest.mark.parametrize(
    "block_length, replacement", [(1, True), (5, True), (1, False), (5, False)]
)
def test_moving_block_bootstrap(block_length, replacement):
    """Tests for the _moving_block_bootstrap.

    1. the output series has the same index as the input
    2. basic checks for the distribution of the bootstrapped values
       i.e. actual min/max >= bootstapped min/max
    """
    y_hat = _moving_block_bootstrap(
        y, block_length=block_length, replacement=replacement
    )
    assert (
        y_hat.index.equals(y_index)
        & (y_hat.max() <= y.max())
        & (y_hat.min() >= y.min())
    )


@pytest.mark.parametrize("ts", [y, y.to_frame()])
def test_get_series_name(ts):
    """Test _get_series_name returns the right string."""
    assert _get_series_name(ts) == "Number of airline passengers"
