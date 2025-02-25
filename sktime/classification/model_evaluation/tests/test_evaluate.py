#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for model classification module.

In particular, function `evaluate`, that performs time series cross-validation, is
tested with various configurations for correct output.
"""

__author__ = ["ksharma6"]

__all__ = [
    "test_evaluate_common_configs",
]

import numpy as np
import pandas as pd
import pytest
from sklearn import metrics
from sklearn.model_selection import KFold

# from sktime.datasets import load_airline, load_longley
from sktime.classification.base import BaseClassifier
from sktime.classification.dummy import DummyClassifier
from sktime.classification.model_evaluation import evaluate
from sktime.classification.model_evaluation._functions import (
    _check_scores,
    _get_column_order_and_datatype,
)
from sktime.tests.test_switch import run_test_for_class
from sktime.utils._testing.panel import make_classification_problem
from sktime.utils.dependencies import _check_soft_dependencies
from sktime.utils.parallel import _get_parallel_test_fixtures

METRICS = [
    metrics.accuracy_score,
]

# list of parallelization backends to test
BACKENDS = _get_parallel_test_fixtures("estimator")


def _check_evaluate_output(out, cv, y, scoring, return_data, return_model):
    assert isinstance(out, pd.DataFrame)
    # Check column names.
    scoring = _check_scores(scoring)
    columns = _get_column_order_and_datatype(
        metric_types=scoring, return_data=return_data, return_model=return_model
    )
    assert set(out.columns) == columns.keys(), "Columns are not identical"

    # Check number of rows against number of splits.
    n_splits = cv.get_n_splits(y)
    assert out.shape[0] == n_splits

    # Check if all timings are positive.
    assert np.all(out.filter(like="_time") >= 0)

    # Check cutoffs.
    np.testing.assert_array_equal(
        out["cutoff"].to_numpy(), y.iloc[cv.get_cutoffs(y)].index.to_numpy()
    )

    # Check fitted models
    if return_model:
        assert "fitted_classifier" in out.columns
        assert all(
            isinstance(f, (BaseClassifier, type(None)))
            for f in out["fitted_classifier"].values
        )


@pytest.mark.skipif(
    not run_test_for_class(evaluate),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("scoring", METRICS)
@pytest.mark.parametrize("backend", BACKENDS)
def test_evaluate_common_configs(scoring, backend):
    """Test evaluate common configs."""
    # skip test for dask backend if dask is not installed
    if backend == "dask" and not _check_soft_dependencies("dask", severity="none"):
        return None

    X, y = make_classification_problem(n_timepoints=30)
    classifier = DummyClassifier()
    classifier.fit(X)
    y_pred = classifier.predict()

    cv = KFold(n_splits=3, shuffle=False),

    out = evaluate(
        classifier=classifier,
        cv=cv,
        X=X,
        y=y,
        scoring=scoring,
        **backend,
    )
    _check_evaluate_output(
        out=out,
        cv=cv,
        y=y,
        scoring=scoring(y_true=y, y_pred=y_pred),
        return_data=False,
        return_model=False,
    )

    # check scoring
    actual = out.loc[:, f"test_{scoring.name}"]

    n_splits = cv.get_n_splits(y)
    expected = np.empty(n_splits)
    for i, (train, test) in enumerate(cv.split(y)):
        c = classifier.clone()
        c.fit(y.iloc[train])
        expected[i] = scoring(y.iloc[test], c.predict(), y_train=y.iloc[train])

    np.testing.assert_array_equal(actual, expected)
