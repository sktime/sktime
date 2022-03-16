# -*- coding: utf-8 -*-
"""Unit tests for early classifier input output."""

__author__ = ["mloning", "TonyBagnall", "MatthewMiddlehurst"]
__all__ = [
    "test_classifier_output",
    "test_multivariate_input",
    "test_3d_numpy_input",
]

import numpy as np
import pytest

import sktime.classification.tests.test_all_classifiers as ct
from sktime.registry import all_estimators
from sktime.tests._config import EXCLUDE_ESTIMATORS
from sktime.utils._testing.estimator_checks import _make_args

EARLY_CLASSIFIERS = all_estimators(
    "early_classifier", return_names=False, exclude_estimators=EXCLUDE_ESTIMATORS
)
n_classes = 3


@pytest.mark.parametrize("estimator", EARLY_CLASSIFIERS)
def test_3d_numpy_input(estimator):
    """Test early classifiers handle 3D numpy input correctly."""
    ct.test_3d_numpy_input(estimator)


@pytest.mark.parametrize("estimator", EARLY_CLASSIFIERS)
def test_multivariate_input(estimator):
    """Test early classifiers handle multivariate pd.DataFrame input correctly."""
    ct.test_multivariate_input(estimator)


@pytest.mark.parametrize("estimator", EARLY_CLASSIFIERS)
def test_classifier_output(estimator):
    """Test classifier outputs the correct data types and values.

    Test predict produces a np.array or pd.Series with only values seen in the train
    data, and that predict_proba probability estimates add up to one.
    """
    estimator = estimator.create_test_instance()
    X_train, y_train = _make_args(estimator, "fit", n_classes=n_classes)
    estimator.fit(X_train, y_train)

    X_new = _make_args(estimator, "predict")[0]

    # check predict
    y_pred = estimator.predict(X_new)
    assert isinstance(y_pred, np.ndarray)
    assert y_pred.shape == (X_new.shape[0],)
    assert np.all(np.isin(np.unique(y_pred), np.unique(y_train)))

    # check predict proba
    if hasattr(estimator, "predict_proba"):
        y_proba = estimator.predict_proba(X_new)
        assert isinstance(y_proba, np.ndarray)
        assert y_proba.shape == (X_new.shape[0], n_classes)
        np.testing.assert_allclose(y_proba.sum(axis=1), 1)
