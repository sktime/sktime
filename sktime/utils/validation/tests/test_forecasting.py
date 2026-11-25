#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Test forecasting module."""

__author__ = ["mloning"]

import numpy as np
import pandas as pd
import pytest
from pytest import raises

from sktime.tests.test_switch import run_test_for_class
from sktime.utils.validation.forecasting import check_fh, check_scoring

empty_input = (np.array([], dtype=int), [], pd.Index([], dtype=int))


@pytest.mark.skipif(
    not run_test_for_class(check_fh),
    reason="Run if tested function has changed.",
)
@pytest.mark.parametrize("arg", empty_input)
def test_check_fh_empty_input(arg):
    """Test that fh validation throws an error with empty container."""
    with raises(ValueError, match="`fh` must not be empty"):
        check_fh(arg)


@pytest.mark.skipif(
    not run_test_for_class(check_scoring),
    reason="Run if tested function has changed.",
)
@pytest.mark.parametrize("scoring", ["MeanAbsoluteError", "MeanAbsoluteError()"])
def test_check_scoring_metric_name_string(scoring):
    """Test that metric name strings, with or without brackets, give an instance.

    Failure case of bug #11858: "MeanAbsoluteError" resolved to the metric class,
    not an instance, and crashed downstream in ``evaluate`` and the tuners.
    """
    from sktime.performance_metrics.forecasting import MeanAbsoluteError

    assert isinstance(check_scoring(scoring), MeanAbsoluteError)


@pytest.mark.skipif(
    not run_test_for_class(check_scoring),
    reason="Run if tested function has changed.",
)
def test_evaluate_with_metric_class_name_string():
    """Test that evaluate works with a metric class name string, see #11858."""
    from sktime.datasets import load_airline
    from sktime.forecasting.model_evaluation import evaluate
    from sktime.forecasting.naive import NaiveForecaster
    from sktime.split import ExpandingWindowSplitter

    y = load_airline()
    cv = ExpandingWindowSplitter(initial_window=100, step_length=12, fh=[1, 2, 3])

    results = evaluate(NaiveForecaster(), cv, y, scoring="MeanAbsoluteError")

    assert "test_MeanAbsoluteError" in results.columns
