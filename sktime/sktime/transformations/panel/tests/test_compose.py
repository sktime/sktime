"""Tests for panel compositors."""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from sktime.datasets import load_basic_motions
from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.panel.compose import ColumnTransformer
from sktime.transformations.panel.reduce import Tabularizer


@pytest.mark.skipif(
    not run_test_for_class(ColumnTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_ColumnTransformer_pipeline():
    """Test pipeline with ColumnTransformer."""
    X_train, y_train = load_basic_motions(split="train", return_X_y=True)
    X_test, y_test = load_basic_motions(split="test", return_X_y=True)

    # using Identity function transformations (transform series to series)
    def id_func(X):
        return X

    column_transformer = ColumnTransformer(
        [
            ("id0", FunctionTransformer(func=id_func, validate=False), ["dim_0"]),
            ("id1", FunctionTransformer(func=id_func, validate=False), ["dim_1"]),
        ]
    )
    steps = [
        ("extract", column_transformer),
        ("tabularise", Tabularizer()),
        ("classify", RandomForestClassifier(n_estimators=2, random_state=1)),
    ]
    model = Pipeline(steps=steps)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    assert y_pred.shape[0] == y_test.shape[0]
    np.testing.assert_array_equal(np.unique(y_pred), np.unique(y_test))
