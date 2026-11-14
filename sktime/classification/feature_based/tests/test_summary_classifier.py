"""Tests for SummaryClassifier."""

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from sktime.classification.feature_based import SummaryClassifier
from sktime.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(SummaryClassifier),
    reason="run test only if soft dependencies are present and incrementally",
)
def test_summary_classifier_dflist_column_order():
    """Reordering named variables within instances preserves predictions, #11799."""
    panel = [
        pd.DataFrame({"temperature": [v] * 4, "pressure": [100.0 - v] * 4})
        for v in [0.0, 0.0, 100.0, 100.0]
    ]
    reordered = [
        frame.iloc[:, ::-1] if i in [1, 3] else frame.copy()
        for i, frame in enumerate(panel)
    ]
    y = np.array([0, 0, 1, 1])
    classifier = SummaryClassifier(
        summary_functions=("mean",),
        summary_quantiles=None,
        estimator=DecisionTreeClassifier(random_state=0),
    ).fit(panel, y)

    np.testing.assert_array_equal(classifier.predict(panel), y)
    np.testing.assert_array_equal(classifier.predict(reordered), y)
    np.testing.assert_array_equal(
        classifier.predict_proba(reordered), classifier.predict_proba(panel)
    )
