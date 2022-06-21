#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for Bagging Forecasters."""

import pandas as pd
import pytest

from sktime.datasets import load_airline
from sktime.forecasting.compose import BaggingForecaster
from sktime.forecasting.compose._bagging import _calculate_data_quantiles
from sktime.forecasting.naive import NaiveForecaster
from sktime.transformations.bootstrap import STLBootstrapTransformer
from sktime.transformations.series.boxcox import LogTransformer

y = load_airline()


@pytest.mark.parametrize("transformer", [LogTransformer, NaiveForecaster])
def test_bagging_forecaster_transformer_type_error(transformer):
    """Test that the right exception is raised for invalid transformer."""
    with pytest.raises(TypeError) as ex:
        f = BaggingForecaster(
            bootstrap_transformer=transformer, forecaster=NaiveForecaster(sp=12)
        )
        f.fit(y)
        msg = (
            "bootstrap_transformer in BaggingForecaster should be a Transformer "
            "that take as input a Series and output a Panel."
        )
        assert msg == ex.value


@pytest.mark.parametrize("forecaster", [LogTransformer])
def test_bagging_forecaster_forecaster_type_error(forecaster):
    """Test that the right exception is raised for invalid forecaster."""
    with pytest.raises(TypeError) as ex:
        f = BaggingForecaster(
            bootstrap_transformer=STLBootstrapTransformer(sp=12),
            forecaster=forecaster,
        )
        f.fit(y)
        msg = "forecaster in BaggingForecaster should be an sktime Forecaster"
        assert msg == ex.value


def test_calculate_data_quantiles():
    """Test that we calculate quantiles correctly."""
    series_names = ["s1", "s2", "s3"]
    fh = [1, 2]
    alpha = [0, 0.5, 1]
    quantiles_column_index = pd.MultiIndex.from_product([["Quantiles"], alpha])

    index = pd.MultiIndex.from_product(
        [series_names, fh], names=["time_series", "time"]
    )
    df = pd.DataFrame(data=[1, 10, 2, 11, 3, 12], index=index)

    output_df = pd.DataFrame(
        data=[[1.0, 2.0, 3.0], [10.0, 11.0, 12.0]],
        columns=quantiles_column_index,
        index=pd.Index(data=fh, name="time"),
    )

    pd.testing.assert_frame_equal(_calculate_data_quantiles(df, alpha), output_df)
