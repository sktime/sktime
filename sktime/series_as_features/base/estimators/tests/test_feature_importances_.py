# -*- coding: utf-8 -*-
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeClassifier

from sktime.classification.compose._ensemble import ComposableTimeSeriesForestClassifier
from sktime.transformations.panel.compose import (
    SeriesToPrimitivesRowTransformer,
)
from sktime.transformations.panel.segment import IntervalSegmenter
from sktime.transformations.panel.summarize._extract import (
    RandomIntervalFeatureExtractor,
)
from sktime.utils._testing.panel import make_classification_problem

X_train, y_train = make_classification_problem()


# Check results of a simple case of single estimator, single feature and
# single interval from different but equivalent implementations
def test_feature_importances_single_feature_interval_and_estimator():
    random_state = 1234

    # Compute using default method
    features = [np.mean]
    steps = [
        (
            "transform",
            RandomIntervalFeatureExtractor(
                n_intervals=1, features=features, random_state=random_state
            ),
        ),
        ("clf", DecisionTreeClassifier()),
    ]
    base_estimator = Pipeline(steps)
    clf1 = ComposableTimeSeriesForestClassifier(
        estimator=base_estimator, random_state=random_state, n_estimators=1
    )
    clf1.fit(X_train, y_train)

    # Extract the interval and the estimator, and compute using pipelines
    intervals = clf1.estimators_[0].steps[0][1].intervals_
    steps = [
        ("segment", IntervalSegmenter(intervals)),
        (
            "transform",
            FeatureUnion(
                [
                    (
                        "mean",
                        SeriesToPrimitivesRowTransformer(
                            FunctionTransformer(func=np.mean, validate=False),
                            check_transformer=False,
                        ),
                    )
                ]
            ),
        ),
        ("clf", clone(clf1.estimators_[0].steps[-1][1])),
    ]
    clf2 = Pipeline(steps)
    clf2.fit(X_train, y_train)

    # Check for feature importances obtained from the estimators
    fi_expected = clf1.estimators_[0].steps[-1][1].feature_importances_
    fi_actual = clf2.steps[-1][1].feature_importances_
    np.testing.assert_array_equal(fi_actual, fi_expected)


# Check for 4 more complex cases with 3 features, with both numbers of
# intervals and estimators varied from 1 to 2.
# Feature importances from each estimator on each interval, and
# normalised feature values of the time series are checked using
# different but equivalent implementations
@pytest.mark.parametrize("n_intervals", [1])
@pytest.mark.parametrize("n_estimators", [1, 2])
def test_feature_importances_multi_intervals_estimators(n_intervals, n_estimators):
    random_state = 1234
    n_features = 2

    # Compute feature importances using the default method
    features = [np.mean, np.std]
    steps = [
        (
            "transform",
            RandomIntervalFeatureExtractor(
                n_intervals=n_intervals, features=features, random_state=random_state
            ),
        ),
        ("clf", DecisionTreeClassifier()),
    ]
    base_estimator = Pipeline(steps)
    clf1 = ComposableTimeSeriesForestClassifier(
        estimator=base_estimator, random_state=random_state, n_estimators=n_estimators
    )
    clf1.fit(X_train, y_train)

    fi_expected = np.zeros([n_estimators, n_intervals * n_features])
    fi_actual = np.zeros([n_estimators, n_intervals * n_features])

    # Obtain intervals and decision trees from fitted classifier
    for i in range(n_estimators):
        intervals = clf1.estimators_[i].steps[0][1].intervals_
        steps = [
            ("segment", IntervalSegmenter(intervals)),
            (
                "transform",
                FeatureUnion(
                    [
                        (
                            "mean",
                            SeriesToPrimitivesRowTransformer(
                                FunctionTransformer(func=np.mean, validate=False),
                                check_transformer=False,
                            ),
                        ),
                        (
                            "std",
                            SeriesToPrimitivesRowTransformer(
                                FunctionTransformer(func=np.std, validate=False),
                                check_transformer=False,
                            ),
                        ),
                    ]
                ),
            ),
            ("clf", clone(clf1.estimators_[i].steps[-1][1])),
        ]
        clf2 = Pipeline(steps)
        clf2.fit(X_train, y_train)

        # Compute and check for individual feature importances
        fi_expected[i, :] = clf1.estimators_[i].steps[-1][1].feature_importances_
        fi_actual[i, :] = clf2.steps[-1][1].feature_importances_
        np.testing.assert_array_equal(fi_actual[i, :], fi_expected[i, :])

    # Compute normalised feature values of the time series using the
    # default property
    fis_expacted = clf1.feature_importances_

    # Compute normalised feature values of the time series from the pipeline
    # implementation
    n_timepoints = len(clf1.estimators_[0].steps[0][1]._time_index)
    fis_actual = np.zeros((n_timepoints, n_features))

    for i in range(n_estimators):
        intervals = clf1.estimators_[i].steps[0][1].intervals_
        for j in range(n_features):
            for k in range(n_intervals):
                start, end = intervals[k]
                fis_actual[start:end, j] += fi_actual[i, (j * n_intervals) + k]
    fis_actual = fis_actual / n_estimators / n_intervals
    np.testing.assert_array_equal(fis_actual, fis_expacted)
