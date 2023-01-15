#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

"""Test functionality of summary transformer."""
__author__ = ["RNKuhns"]
import re

import numpy as np
import pandas as pd
import pytest

from sktime.transformations.series.summarize import (
    ALLOWED_SUM_FUNCS,
    SummaryTransformer,
)
from sktime.utils._testing.series import _make_series

# Test individual summary functions + lists and tuples of all summary functions
sum_funcs_to_test = [ALLOWED_SUM_FUNCS[0]] + [ALLOWED_SUM_FUNCS]
sum_funcs_to_test.append(tuple(ALLOWED_SUM_FUNCS))
quantiles_to_test = [0.7, [0.5, 0.25, 0.1], (0.3, 0.0002, 0.99), None]

# Incorrect inputs to test to ensure they raise expected errors
incorrect_sum_funcs_to_test = [
    "meen",
    "md",
    0.25,
    ["mean", "medain"],
    np.array(["mean", "max"]),
]
incorrect_quantiles_to_test = [25, "0.25", "median", [0.25, 1.25], [0.25, "median"]]

# Test functionality on pd.Series and pd.DataFrame (uni- and multi-variate) input
y1 = _make_series(n_timepoints=75)
y2 = _make_series(n_timepoints=75)
y1.name, y2.name = "y1", "y2"
y_df_uni = pd.DataFrame(y1)
y_df_multi = pd.concat([y1, y2], axis=1)
data_to_test = [y1, y_df_uni, y_df_multi]


@pytest.mark.parametrize("y", data_to_test)
@pytest.mark.parametrize("summary_arg", sum_funcs_to_test)
@pytest.mark.parametrize("quantile_arg", quantiles_to_test)
def test_summary_transformer_output_type(y, summary_arg, quantile_arg):
    """Test whether output is DataFrame of correct dimensions."""
    transformer = SummaryTransformer(
        summary_function=summary_arg, quantiles=quantile_arg
    )
    transformer.fit(y)
    yt = transformer.transform(y)

    output_is_dataframe = isinstance(yt, pd.DataFrame)
    assert output_is_dataframe

    # compute number of expected rows and columns

    # all test cases are single series, so single row
    expected_instances = 1

    # expected number of feature types = quantiles plus summaries
    expected_sum_features = 1 if isinstance(summary_arg, str) else len(summary_arg)
    if quantile_arg is None:
        expected_q_features = 0
    elif isinstance(quantile_arg, (int, float)):
        expected_q_features = 1
    else:
        expected_q_features = len(quantile_arg)
    expected_features = expected_sum_features + expected_q_features

    # for multivariate series, columns = no variables * no feature types
    if isinstance(y, pd.DataFrame):
        expected_features = len(y.columns) * expected_features

    assert yt.shape == (expected_instances, expected_features)


@pytest.mark.parametrize("summary_arg", incorrect_sum_funcs_to_test)
def test_summary_transformer_incorrect_summary_function_raises_error(summary_arg):
    """Test if correct errors are raised for invalid summary_function input."""
    msg = rf"""`summary_function` must be None, or str or a list or tuple made up of
          {ALLOWED_SUM_FUNCS}.
          """
    with pytest.raises(ValueError, match=re.escape(msg)):
        transformer = SummaryTransformer(summary_function=summary_arg, quantiles=None)
        transformer.fit_transform(data_to_test[0])


@pytest.mark.parametrize("quantile_arg", incorrect_quantiles_to_test)
def test_summary_transformer_incorrect_quantile_raises_error(quantile_arg):
    """Test if correct errors are raised for invalid quantiles input."""
    msg = """`quantiles` must be None, int, float or a list or tuple made up of
          int and float values that are between 0 and 1.
          """
    with pytest.raises(ValueError, match=msg):
        transformer = SummaryTransformer(
            summary_function="mean", quantiles=quantile_arg
        )
        transformer.fit_transform(data_to_test[0])
