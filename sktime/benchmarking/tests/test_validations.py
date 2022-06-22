# -*- coding: utf-8 -*-
"""Tests for benchmarking validation functions."""

import numpy as np
import pandas as pd

from sktime.benchmarking import validations
from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.model_selection._split import BaseSplitter
from sktime.performance_metrics.forecasting import MeanAbsoluteError, MeanSquaredError


class FakeSplitter(BaseSplitter):
    """A fake splitter that returns the whole series (for train and test) each split."""

    def __init__(self, n_splits, fh):
        self.n_splits = n_splits
        self.fh = fh
        super().__init__()

    def _split(self, y):
        for _ in range(self.n_splits):
            yield y, y


class LastValueEstimator(BaseForecaster):
    """An estimator that repeats the last value of the train data as the forecast."""

    _tags = {
        "requires-fh-in-fit": False,
        # "scitype:y": "univariate",
    }

    def __init__(self):
        self.last_value = None
        super().__init__()

    def _fit(self, y, X=None, fh=None):
        self.last_value = y.iloc[-1]

    def _predict(self, fh=None, X=None):
        return pd.Series(np.full(fill_value=self.last_value, shape=len(fh)))


def test_forecasting_validation():
    """Test that output dict of the validation function is as expected.

    We want to avoid testing sktime functionality outside of the validator, so we use
    fake splitters and estimators and just check the results dict.
    """

    def dataset_loader():
        return pd.Series([1, 2, 3, 4])

    # Fake splitter will give the whole series as train and test data for each split
    cv_splitter = FakeSplitter(n_splits=2, fh=4)
    # Our estimator predicts the final value of the train data, so all 4s for each split
    estimator = LastValueEstimator()

    results = validations.forecasting_validation(
        dataset_loader=dataset_loader,
        cv_splitter=cv_splitter,
        scorers=[MeanAbsoluteError(), MeanSquaredError()],
        estimator=estimator,
    )

    # Just check that we have the results keys that we expect
    assert len(results) == 8
    assert results["MeanAbsoluteError_fold_0_test"] == 1.5
    assert results["MeanAbsoluteError_fold_1_test"] == 1.5
    assert results["MeanSquaredError_fold_0_test"] == 3.5
    assert results["MeanSquaredError_fold_1_test"] == 3.5
    assert results["MeanAbsoluteError_mean"] == 1.5
    assert results["MeanSquaredError_mean"] == 3.5
    # All the splits are the same, so the std will be 0
    assert results["MeanAbsoluteError_std"] == 0
    assert results["MeanSquaredError_std"] == 0


def test_forecasting_validation_deterministic():
    """Test validation gives deterministic results using a deterministic splitter."""

    def dataset_loader():
        return pd.Series([1, 2, 3, 4])

    # Fake splitter will give the whole series as train and test data for each split
    # And it's clearly deterministic
    cv_splitter = FakeSplitter(n_splits=2, fh=4)
    # Our estimator predicts the final value of the train data, so all 4s for each split
    estimator = LastValueEstimator()

    results = validations.forecasting_validation(
        dataset_loader=dataset_loader,
        cv_splitter=cv_splitter,
        scorers=[MeanAbsoluteError(), MeanSquaredError()],
        estimator=estimator,
    )
    repeated_results = validations.forecasting_validation(
        dataset_loader=dataset_loader,
        cv_splitter=cv_splitter,
        scorers=[MeanAbsoluteError(), MeanSquaredError()],
        estimator=estimator,
    )

    assert results == repeated_results
