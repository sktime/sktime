"""Tests for ColumnEnsembleClassifier."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

import numpy as np
import pandas as pd
import pytest

from sktime.classification.compose import ColumnEnsembleClassifier
from sktime.classification.dummy import DummyClassifier
from sktime.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(ColumnEnsembleClassifier),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_column_ensemble_pd_multiindex_inner_mtype():
    """Regression: X_inner_mtype is pd-multiindex only; nested_univ dropped."""
    clf = ColumnEnsembleClassifier(
        estimators=[
            ("d0", DummyClassifier(strategy="most_frequent"), 0),
            ("d1", DummyClassifier(strategy="prior"), 1),
        ]
    )
    assert clf.get_tag("X_inner_mtype") == ["pd-multiindex"]

    # equal-length multivariate panel as pd-multiindex
    index = pd.MultiIndex.from_product(
        [[0, 1, 2, 3], [0, 1, 2, 3, 4]], names=["instances", "timepoints"]
    )
    X = pd.DataFrame(
        {
            "var_0": np.arange(20, dtype=float),
            "var_1": np.arange(20, dtype=float)[::-1],
        },
        index=index,
    )
    y = np.array(["a", "a", "b", "b"])

    clf.fit(X, y)
    y_pred = clf.predict(X)
    assert y_pred.shape == (4,)
    y_proba = clf.predict_proba(X)
    assert y_proba.shape == (4, 2)
