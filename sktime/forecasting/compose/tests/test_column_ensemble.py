#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file).
"""Unit tests of ColumnEnsembleForecaster functionality."""

__author__ = ["GuzalBulatova", "canbooo", "fkiraly"]

import numpy as np
import pandas as pd
import pytest

from sktime.forecasting.compose import ColumnEnsembleForecaster
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.parametrize(
    "forecasters",
    [
        [
            ("trend", PolynomialTrendForecaster(), 0),
            ("naive", NaiveForecaster(), 1),
            ("ses", NaiveForecaster(strategy="mean"), 2),
        ]
    ],
)
@pytest.mark.parametrize(
    "fh", [(np.arange(1, 11)), (np.arange(1, 33)), (np.arange(1, 3))]
)
def test_column_ensemble_shape(forecasters, fh):
    """Check the shape of the returned prediction."""
    y = pd.DataFrame(np.random.randint(0, 100, size=(100, 3)), columns=list("ABC"))
    forecaster = ColumnEnsembleForecaster(forecasters)
    forecaster.fit(y, fh=fh)
    actual = forecaster.predict()
    assert actual.shape == (len(fh), y.shape[1])


@pytest.mark.parametrize(
    "forecasters",
    [
        [
            ("trend", PolynomialTrendForecaster(), 0),
            ("naive1", NaiveForecaster(), 1),
            ("naive", NaiveForecaster(), 0),
        ],
        [("trend", PolynomialTrendForecaster(), 0), ("naive", NaiveForecaster(), 1)],
    ],
)
def test_invalid_forecasters_indices(forecasters):
    """Check if invalid forecaster indices return correct Error."""
    y = pd.DataFrame(np.random.randint(0, 100, size=(100, 3)), columns=list("ABC"))
    forecaster = ColumnEnsembleForecaster(forecasters=forecasters)
    with pytest.raises(ValueError, match=r"column"):
        forecaster.fit(y, fh=[1, 2])


def test_column_ensemble_string_cols():
    """Check that ColumnEnsembleForecaster works with string columns."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    fc = ColumnEnsembleForecaster(
        [(f"trans_{col}", NaiveForecaster(), col) for col in "ab"]
    )
    fc.fit(df, fh=[1, 42])
    fc.predict()


def test_column_ensemble_multivariate_and_int():
    """Check that ColumnEnsembleForecaster works with string columns."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    fc = ColumnEnsembleForecaster(
        [("ab", NaiveForecaster(), ["a", 1]), ("c", NaiveForecaster(), np.int64(2))]
    )
    fc.fit(df, fh=[1, 42])
    fc.predict()


@pytest.mark.skipif(
    not _check_soft_dependencies("statsmodels", severity="none"),
    reason="skip test if required soft dependencies not available",
)
def test_column_ensemble_hierarchical():
    """Tests column ensemble with hierarchical reconciliation, see bug #3784."""
    from sktime.datatypes import get_examples
    from sktime.datatypes._utilities import get_window
    from sktime.forecasting.sarimax import SARIMAX
    from sktime.transformations.hierarchical.aggregate import Aggregator
    from sktime.transformations.hierarchical.reconcile import Reconciler

    X = get_examples("pd_multiindex_hier")[0]
    y = get_examples("pd_multiindex_hier")[1]

    X_train = get_window(X, lag=1)
    y_train = get_window(y, lag=1)

    f = Aggregator() * (Aggregator() ** SARIMAX()) * Reconciler()

    f_hat = ColumnEnsembleForecaster(f)
    f_hat.fit(y=y_train, X=X_train, fh=1)
