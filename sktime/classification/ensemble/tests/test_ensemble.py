"""Test the ComposableTimeSeriesForestClassifier."""

__author__ = ["mloning"]

import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeClassifier

from sktime.classification.ensemble import ComposableTimeSeriesForestClassifier
from sktime.datasets import load_unit_test
from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.compose import FeatureUnion
from sktime.transformations.panel.segment import RandomIntervalSegmenter
from sktime.transformations.panel.summarize import RandomIntervalFeatureExtractor
from sktime.transformations.series.adapt import TabularToSeriesAdaptor
from sktime.utils._testing.panel import make_classification_problem
from sktime.utils.slope_and_trend import _slope

X, y = make_classification_problem()
n_classes = len(np.unique(y))

mean_transformer = TabularToSeriesAdaptor(
    FunctionTransformer(func=np.mean, validate=False, kw_args={"axis": 0})
)
std_transformer = TabularToSeriesAdaptor(
    FunctionTransformer(func=np.std, validate=False, kw_args={"axis": 0})
)


# Check simple cases.
@pytest.mark.skipif(
    not run_test_for_class(ComposableTimeSeriesForestClassifier),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_tsf_predict_proba():
    """Test composable TSF predict proba."""
    clf = ComposableTimeSeriesForestClassifier(n_estimators=2)
    clf.fit(X, y)
    proba = clf.predict_proba(X)

    assert proba.shape == (X.shape[0], n_classes)
    np.testing.assert_array_equal(np.ones(X.shape[0]), np.sum(proba, axis=1))

    # test single row input
    y_proba = clf.predict_proba(X.iloc[[0], :])
    assert y_proba.shape == (1, n_classes)

    y_pred = clf.predict(X.iloc[[0], :])
    assert y_pred.shape == (1,)


# Compare results from different but equivalent implementations
# @pytest.mark.parametrize("n_intervals", ["log", 1, 3])
@pytest.mark.xfail(reason="SeriesToPrimitivesTransformer will be deprecated, see 2179")
@pytest.mark.parametrize("n_intervals", [1])
@pytest.mark.parametrize("n_estimators", [1, 3])
def test_equivalent_model_specifications(n_intervals, n_estimators):
    """Test composable TSF vs an equivalent model."""
    random_state = 1234
    X_train, y_train = load_unit_test(split="train")
    X_test, y_test = load_unit_test(split="test")

    # Due to tie-breaking/floating point rounding in the final decision tree
    # classifier, the results depend on the
    # exact column order of the input data

    # Compare pipeline predictions outside of ensemble.
    steps = [
        (
            "segment",
            RandomIntervalSegmenter(n_intervals=n_intervals, random_state=random_state),
        ),
        (
            "transform",
            FeatureUnion([("mean", mean_transformer), ("std", std_transformer)]),
        ),
        ("clf", DecisionTreeClassifier(random_state=random_state, max_depth=2)),
    ]
    clf1 = Pipeline(steps)
    clf1.fit(X_train, y_train)
    a = clf1.predict(X_test)

    steps = [
        (
            "transform",
            RandomIntervalFeatureExtractor(
                n_intervals=n_intervals,
                features=[np.mean, np.std],
                random_state=random_state,
            ),
        ),
        ("clf", DecisionTreeClassifier(random_state=random_state, max_depth=2)),
    ]
    clf2 = Pipeline(steps)
    clf2.fit(X_train, y_train)
    b = clf2.predict(X_test)
    np.array_equal(a, b)


# Compare TimeSeriesForest ensemble predictions using pipeline as estimator
@pytest.mark.skipif(
    not run_test_for_class(ComposableTimeSeriesForestClassifier),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("n_intervals", ["sqrt", 1])
@pytest.mark.parametrize("n_estimators", [1, 3])
def test_tsf_predictions(n_estimators, n_intervals):
    """Test TSF predictions."""
    random_state = 1234
    X_train, y_train = load_unit_test(split="train")
    X_test, y_test = load_unit_test(split="test")

    features = [np.mean, np.std, _slope]
    steps = [
        (
            "transform",
            RandomIntervalFeatureExtractor(
                random_state=random_state, features=features
            ),
        ),
        ("clf", DecisionTreeClassifier(random_state=random_state, max_depth=2)),
    ]
    estimator = Pipeline(steps)

    clf1 = ComposableTimeSeriesForestClassifier(
        estimator=estimator, random_state=random_state, n_estimators=n_estimators
    )
    clf1.fit(X_train, y_train)
    a = clf1.predict_proba(X_test)

    # default, semi-modular implementation using
    # RandomIntervalFeatureExtractor internally
    clf2 = ComposableTimeSeriesForestClassifier(
        random_state=random_state, n_estimators=n_estimators
    )
    clf2.fit(X_train, y_train)
    b = clf2.predict_proba(X_test)

    np.testing.assert_array_equal(a, b)
