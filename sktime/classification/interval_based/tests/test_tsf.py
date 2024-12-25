"""Tests for feature importance in time series forests."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from sktime.classification.interval_based import TimeSeriesForestClassifier
from sktime.tests.test_switch import run_test_for_class
from sktime.utils._testing.panel import make_classification_problem

TESTED_MODULE = "sktime.classification.interval_based._tsf"

X_train, y_train = make_classification_problem()


@patch(
    f"{TESTED_MODULE}.TimeSeriesForestClassifier."
    f"_extract_feature_importance_by_feature_type_per_tree"
)
@pytest.mark.skipif(
    not run_test_for_class(TimeSeriesForestClassifier),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("n_estimators", [2, 5])
def test_time_series_forest_classifier_feature_importance(
    extract_feature_importance_of_feature_mock: MagicMock,
    n_estimators: int,
):
    """Test TimeSeriesForestClassifier feature importance."""
    # Given
    given_n_intervals = 2
    given_n_estimators = n_estimators
    given_time_series_forest_classifier = TimeSeriesForestClassifier(
        n_estimators=given_n_estimators
    )

    given_time_series_forest_classifier.series_length = 20
    given_time_series_forest_classifier.estimators_ = MagicMock()
    given_time_series_forest_classifier.intervals_ = np.array(
        [[[0, 9], [15, 20]]] * given_n_estimators
    )
    given_time_series_forest_classifier.n_intervals = given_n_intervals

    extract_feature_importance_of_feature_mock.return_value = np.ones(
        given_time_series_forest_classifier.n_intervals
    )

    # When
    fi = given_time_series_forest_classifier.feature_importances_

    # Then
    expected_fi = pd.DataFrame(
        {
            "mean": [
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0,
                0,
                0,
                0,
                0,
                0,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
            ],
            "std": [
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0,
                0,
                0,
                0,
                0,
                0,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
            ],
            "slope": [
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0,
                0,
                0,
                0,
                0,
                0,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
            ],
        }
    )
    pd.testing.assert_frame_equal(expected_fi, fi)


@pytest.mark.skipif(
    not run_test_for_class(TimeSeriesForestClassifier),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("number_of_intervals", [2, 5])
@pytest.mark.parametrize("feature_type", ["mean", "std", "slope"])
def test__extract_feature_importance_by_feature_type_per_tree(
    number_of_intervals: int, feature_type: str
):
    """Test TimeSeriesForestClassifier feature type feature importance."""
    # given
    given_number_of_intervals = number_of_intervals
    given_tree_feature_importance = np.array([1, 2, 3] * given_number_of_intervals)
    given_feature_type = feature_type
    given_tsf_classifier = TimeSeriesForestClassifier()

    # When
    feature_importance_of_feature_type_from_tree_feature_importance = (
        given_tsf_classifier._extract_feature_importance_by_feature_type_per_tree(
            given_tree_feature_importance, given_feature_type
        )
    )

    # Then
    expected_feature_importance_of_feature_type_from_tree_feature_importance = {
        "mean": np.ones(given_number_of_intervals) * 1,
        "std": np.ones(given_number_of_intervals) * 2,
        "slope": np.ones(given_number_of_intervals) * 3,
    }
    np.testing.assert_array_equal(
        expected_feature_importance_of_feature_type_from_tree_feature_importance[
            given_feature_type
        ],
        feature_importance_of_feature_type_from_tree_feature_importance,
    )
