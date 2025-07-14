#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Test functionality of summary transformer."""

__author__ = ["RNKuhns"]
import re

import numpy as np
import pandas as pd
import pytest

from sktime.tests.test_switch import run_test_for_class
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


def _make_test_data(i):
    # Test functionality on pd.Series and pd.DataFrame (uni- and multi-variate) input
    y1 = _make_series(n_timepoints=75)
    y2 = _make_series(n_timepoints=75)
    y1.name, y2.name = "y1", "y2"
    y_df_uni = pd.DataFrame(y1)
    y_df_multi = pd.concat([y1, y2], axis=1)
    data_to_test = [y1, y_df_uni, y_df_multi]
    return data_to_test[i]


@pytest.mark.skipif(
    not run_test_for_class(SummaryTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("data_id", [0, 1, 2])
@pytest.mark.parametrize("summary_arg", sum_funcs_to_test)
@pytest.mark.parametrize("quantile_arg", quantiles_to_test)
def test_summary_transformer_output_type(data_id, summary_arg, quantile_arg):
    """Test whether output is DataFrame of correct dimensions."""
    y = _make_test_data(data_id)

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


@pytest.mark.skipif(
    not run_test_for_class(SummaryTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("summary_arg", incorrect_sum_funcs_to_test)
def test_summary_transformer_incorrect_summary_function_raises_error(summary_arg):
    """Test if correct errors are raised for invalid summary_function input."""
    msg = rf"""summary_function must be None, or str or a list or tuple made up of
          {ALLOWED_SUM_FUNCS}.
          """
    with pytest.raises(ValueError, match=re.escape(msg)):
        transformer = SummaryTransformer(summary_function=summary_arg, quantiles=None)
        transformer.fit_transform(_make_test_data(0))


@pytest.mark.skipif(
    not run_test_for_class(SummaryTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("quantile_arg", incorrect_quantiles_to_test)
def test_summary_transformer_incorrect_quantile_raises_error(quantile_arg):
    """Test if correct errors are raised for invalid quantiles input."""
    msg = """quantiles must be None, int, float or a list or tuple made up of
          int and float values that are between 0 and 1.
          """
    with pytest.raises(ValueError, match=msg):
        transformer = SummaryTransformer(
            summary_function="mean", quantiles=quantile_arg
        )
        transformer.fit_transform(_make_test_data(0))


@pytest.mark.skipif(
    not run_test_for_class(SummaryTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_summarize_no_lossy_setitem():
    """Test that SummaryTransformer.fit does not raise LossySetitemError."""
    data = pd.Series(range(12, 0, -1))
    transformer = SummaryTransformer()
    transformer.fit(data)
    assert True  # Reaching here means no LossySetitemError was raised
