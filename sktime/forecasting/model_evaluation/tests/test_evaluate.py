#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Martin Walter"]
__all__ = ["test_evaluate"]

from sktime.datasets import load_airline
from sktime.forecasting.model_evaluation import evaluate
from sktime.forecasting.model_selection import ExpandingWindowSplitter
from sktime.forecasting.naive import NaiveForecaster
import pandas as pd


def test_evaluate():
    y = load_airline()
    forecaster = NaiveForecaster(strategy="drift", sp=12)
    cv = ExpandingWindowSplitter(
        initial_window=24,
        step_length=24,
        fh=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        window_length=10,
    )
    df = evaluate(forecaster=forecaster, y=y, cv=cv, strategy="update")
    # just making sure the function is running
    assert isinstance(df, pd.DataFrame)
