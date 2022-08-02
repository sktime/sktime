# -*- coding: utf-8 -*-
"""TEASER test code."""
import numpy as np
import pytest
from numpy import testing
from sklearn.ensemble import IsolationForest

from sktime.classification.early_classification._teaser import TEASER
from sktime.classification.early_classification.tests.test_all_early_classifiers import (  # noqa: E501
    load_unit_data,
)
from sktime.classification.interval_based import TimeSeriesForestClassifier
from sktime.datatypes._panel._convert import from_nested_to_3d_numpy


def test_teaser_with_different_decision_maker():
    """Test of TEASER with different One-Class-Classifier."""
    X_train, y_train, X_test, y_test, indices = load_unit_data()

    # train probability threshold
    teaser = TEASER(
        random_state=0,
        classification_points=[6, 10, 16, 24],
        estimator=TimeSeriesForestClassifier(n_estimators=10, random_state=0),
        one_class_classifier=IsolationForest(n_estimators=5),
        one_class_param_grid={"bootstrap": [True, False]},
    )
    teaser.fit(X_train, y_train)

    full_probas, _ = teaser.predict_proba(X_test.iloc[indices])
    testing.assert_array_almost_equal(
        full_probas, teaser_if_unit_test_probas, decimal=2
    )

    # make sure update ends up with the same probas
    teaser.reset_state_info()

    X_test = from_nested_to_3d_numpy(X_test)[indices]
    final_probas = np.zeros((10, 2))
    open_idx = np.arange(0, 10)

    for i in teaser.classification_points:
        probas, decisions = teaser.update_predict_proba(X_test[:, :, :i])
        X_test, open_idx, final_idx = teaser.split_indices_and_filter(
            X_test, open_idx, decisions
        )
        final_probas[final_idx] = probas[decisions]

        if len(X_test) == 0:
            break

    testing.assert_array_almost_equal(
        final_probas, teaser_if_unit_test_probas, decimal=2
    )


def test_teaser_near_classification_points():
    """Test of TEASER with incremental time stamps outside defined class points."""
    X_train, y_train, X_test, y_test, indices = load_unit_data()

    # train probability threshold
    teaser = TEASER(
        random_state=0,
        classification_points=[6, 10, 14, 18, 24],
        estimator=TimeSeriesForestClassifier(n_estimators=5, random_state=0),
    )
    teaser.fit(X_train, y_train)

    # use test_points that are not within list above
    test_points = [7, 11, 19, 20]

    X_test = from_nested_to_3d_numpy(X_test)
    X_test = X_test[indices]

    decisions = np.zeros(len(X_test), dtype=bool)
    for i in test_points:
        X_test = X_test[np.invert(decisions)]
        X = X_test[:, :, :i]

        if i == 20:
            with pytest.raises(ValueError):
                teaser.update_predict_proba(X)
        else:
            _, decisions = teaser.update_predict(X)


def test_teaser_default():
    """Test of TEASER on the full data with the default estimator."""
    X_train, y_train, X_test, y_test, indices = load_unit_data()

    # train probability threshold
    teaser = TEASER(
        random_state=0,
        classification_points=[6, 10, 16, 24],
    )
    teaser.fit(X_train, y_train)

    _, acc, earl = teaser.score(X_test.iloc[indices], y_test)
    testing.assert_allclose(acc, 0.6, rtol=0.01)
    testing.assert_allclose(earl, 0.766667, rtol=0.01)

    testing.assert_allclose(teaser._train_accuracy, 0.9, rtol=0.01)
    testing.assert_allclose(teaser._train_earliness, 0.7333, rtol=0.01)


teaser_if_unit_test_probas = np.array(
    [
        [0.0, 1.0],
        [0.7, 0.3],
        [0.0, 1.0],
        [1.0, 0.0],
        [0.7, 0.3],
        [1.0, 0.0],
        [1.0, 0.0],
        [0.2, 0.8],
        [0.9, 0.1],
        [1.0, 0.0],
    ]
)
