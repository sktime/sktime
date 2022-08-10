# -*- coding: utf-8 -*-
"""Tests for evaluate_classification module."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import get_scorer
from sklearn.model_selection import ShuffleSplit, cross_validate

from sktime.classification.kernel_based import RocketClassifier
from sktime.classification.model_evaluation._function import evaluate_classification
from sktime.datasets import load_arrow_head


def _split(X, y, train, test):
    """Split y and X for given train and test set indices.

    X: nest_univ format of input data
    y: numpy array of label as int
    train: list of index of train data, obtained from cross-validation method
    test: list of index of test data, obtained from cross-validation method
    """
    y_train = y[train]
    y_test = y[test]

    X_train = X.loc[train, :]
    X_test = X.loc[test, :]

    return X_train, y_train, X_test, y_test


@pytest.mark.parametrize("fold_no", [5])
@pytest.mark.parametrize("random_seed", [21])
def test_evaluate_classification_metrics(fold_no, random_seed):
    """Test evaluate for basic classification problems."""
    # Merge train and test into one dataset
    arrow_train_X, arrow_train_y = load_arrow_head(
        split="train", return_type="nested_univ"
    )
    arrow_test_X, arrow_test_y = load_arrow_head(
        split="test", return_type="nested_univ"
    )
    # Merge train and test set for cv
    arrow_X = pd.concat([arrow_train_X, arrow_test_X], axis=0)
    arrow_X = arrow_X.reset_index().drop(columns=["index"])
    arrow_y = np.concatenate([arrow_train_y, arrow_test_y], axis=0)

    classifier = RocketClassifier()
    cv = ShuffleSplit(n_splits=fold_no, test_size=0.2, random_state=random_seed)

    actual = evaluate_classification(classifier=classifier, X=arrow_X, y=arrow_y, cv=cv)

    classifier = RocketClassifier().clone()
    scoring = "accuracy"
    scorer = get_scorer(scoring)
    expected = cross_validate(
        classifier, X=arrow_X, y=arrow_y, cv=cv, n_jobs=-1, scoring=scorer
    )
    expected = pd.DataFrame(expected)

    np.testing.assert_allclose(
        expected["test_score"].mean(), actual["score"].mean(), rtol=1e-2
    )
