#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for model classification module.

In particular, function `evaluate`, that performs time series cross-validation, is
tested with various configurations for correct output.
"""

__author__ = ["ksharma6"]

__all__ = []

import numpy as np
import pandas as pd
from sklearn import metrics

from sktime.classification.model_evaluation._functions import (
    _check_scores,
    _get_column_order_and_datatype,
)
from sktime.utils.parallel import _get_parallel_test_fixtures

METRICS = [
    metrics.accuracy_score(),
]

# list of parallelization backends to test
BACKENDS = _get_parallel_test_fixtures("estimator")


def _check_evaluate_output(out, cv, y, scoring, return_data):
    assert isinstance(out, pd.DataFrame)
    # Check column names
    scoring = _check_scores(scoring)
    columns = _get_column_order_and_datatype(scoring, return_data)
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
